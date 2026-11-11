"""Joint multi-medium training with clean-scene EMA prediction and PCGrad-norm."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import DOMAINS, PairedDataset
from data.sampling import BalancedExposureSampler
from models import SmoothAlpha
from models.latent import EMATargetEncoder, LatentPredictor, latent_prediction_loss
from models.losses import RestorationLoss


def pcgrad_step(model, predictor, target, criterion, optimizer, batch,
                epoch, step, config, norm_ema):
    """Normalize each present medium's gradient, project conflicts and update once."""
    parameters = list(model.parameters()) + list(predictor.parameters())
    gradients, present, losses = [], [], []
    latent_weight = (config["loss"]["latent"]
                     if epoch > config["loss"]["latent_warmup_epochs"] else 0.0)
    for domain_id in range(len(DOMAINS)):
        mask = batch["domain_id"] == domain_id
        if not mask.any():
            continue
        observed, reference, depth = (batch[key][mask] for key in ("obs", "gt", "depth"))
        output = model(observed, depth)
        loss, _ = criterion(output, observed, reference, epoch)
        latent_loss = latent_prediction_loss(predictor(output["z"]), target.encode(reference, depth))
        if latent_weight > 0:
            loss = loss + latent_weight * latent_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.cat([parameter.grad.flatten() if parameter.grad is not None
                              else parameter.new_zeros(parameter.numel()) for parameter in parameters])
        magnitude = float(gradient.norm())
        if not np.isfinite(magnitude):
            raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, step {step}")
        momentum = config["gradient_norm_momentum"]
        norm_ema[domain_id] = (momentum * norm_ema[domain_id] + (1 - momentum) * magnitude
                               if domain_id in norm_ema else magnitude)
        gradients.append(gradient / (norm_ema[domain_id] + 1e-12))
        present.append(domain_id)
        losses.append(float(loss.detach()))

    if not gradients:
        raise ValueError("The batch contains no recognized medium")
    generator = torch.Generator().manual_seed(config["seed"] * 1000003 + step)
    projected = [gradient.clone() for gradient in gradients]
    for i in range(len(gradients)):
        for j in torch.randperm(len(gradients), generator=generator).tolist():
            if i == j:
                continue
            product = torch.dot(projected[i], gradients[j])
            if product < 0:
                projected[i] -= product / (gradients[j].norm().square() + 1e-12) * gradients[j]
    scale = sum(norm_ema[domain] for domain in present) / len(present)
    combined = torch.stack(projected).mean(0) * scale
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = combined[offset:offset + count].view_as(parameter).clone()
        offset += count
    torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
    optimizer.step()
    return float(np.mean(losses))


@torch.inference_mode()
def validation_psnr(model, loader, device):
    """Average per-image PSNR over the complete validation partition."""
    model.eval()
    scores = []
    for batch in loader:
        output = model(batch["obs"].to(device), batch["depth"].to(device))["x_pred"].clamp(0, 1)
        prediction = output.cpu().numpy()
        reference = batch["gt"].numpy()
        for pred, truth in zip(prediction, reference):
            mse = float(np.mean((pred - truth) ** 2))
            scores.append(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/smooth_alpha_joint.yaml")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/smooth_alpha_joint_seed42"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, help="Override the training seed, e.g. 43 or 44")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if tuple(config["domains"]) != DOMAINS:
        raise ValueError("The joint configuration must use water, fog, smoke in that order")
    if config["epoch_samples"] % config["batch_size"]:
        raise ValueError("The exposure budget must be divisible by batch size")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(args.device)
    datasets = [PairedDataset(args.data_root, domain, "train", config["image_size"]) for domain in DOMAINS]
    sampler = BalancedExposureSampler(datasets, config["epoch_samples"], config["seed"])
    loader = DataLoader(ConcatDataset(datasets), batch_size=config["batch_size"], sampler=sampler,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda", drop_last=True)
    validation = {
        domain: DataLoader(PairedDataset(args.data_root, domain, "val", config["image_size"]),
                           batch_size=config["batch_size"], num_workers=0, shuffle=False)
        for domain in DOMAINS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory to preserve existing checkpoints: {args.output}")
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model = SmoothAlpha().to(device)
    predictor = LatentPredictor().to(device)
    target = EMATargetEncoder(model.detail_path, config["ema"]["start"], config["ema"]["end"]).to(device)
    criterion = RestorationLoss(config["loss"]).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()),
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["minimum_learning_rate"])
    norm_ema, best_score, step = {}, float("-inf"), 0
    total_steps = config["epochs"] * len(loader)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        predictor.train()
        sampler.set_epoch(epoch)
        losses = []
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            losses.append(pcgrad_step(model, predictor, target, criterion, optimizer, batch,
                                      epoch, step, config, norm_ema))
            step += 1
            target.update(model.detail_path, step, total_steps)
        scheduler.step()
        record = {"epoch": epoch, "optimizer_steps": step, "loss": float(np.mean(losses)),
                  "learning_rate": scheduler.get_last_lr()[0]}
        if epoch % config["validation_every"] == 0 or epoch == config["epochs"]:
            scores = {domain: validation_psnr(model, validation[domain], device) for domain in DOMAINS}
            score = float(np.mean(list(scores.values())))
            record.update(validation_psnr=scores, macro_validation_psnr=score)
            if score > best_score:
                best_score = score
                # Only the restoration state is distributed for inference.
                checkpoint = {
                    "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "epoch": epoch, "seed": config["seed"], "config": config,
                    "validation_psnr": scores,
                }
                torch.save(checkpoint, args.output / f"smooth_alpha_joint_seed{config['seed']}.pth")
        print(json.dumps(record), flush=True)
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
