"""Evaluate the released checkpoint on the fixed paired benchmark partitions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import DOMAINS, PairedDataset
from metrics import paired_metrics, structural_metrics, uciqe, uiqm
from models import SmoothAlpha


@torch.inference_mode()
def evaluate(model, dataset, device, perceptual, image_directory=None):
    """Evaluate every listed test image; missing inputs and unavailable metrics fail."""
    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        observed = batch["obs"].to(device)
        reference = batch["gt"].to(device)
        output = model(observed, batch["depth"].to(device))
        prediction = output["x_pred"].clamp(0, 1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite prediction for {batch['fname'][0]}")
        pred = prediction[0].cpu().numpy().transpose(1, 2, 0)
        truth = reference[0].cpu().numpy().transpose(1, 2, 0)
        quantized = (pred * 255).astype(np.uint8)
        truth_uint8 = (truth * 255).astype(np.uint8)
        row = {"filename": batch["fname"][0], **paired_metrics(pred, truth)}
        row["lpips"] = float(perceptual(prediction * 2 - 1, reference * 2 - 1).item())
        row.update(uiqm=float(uiqm(quantized)), uciqe=float(uciqe(quantized)))
        row.update(structural_metrics(quantized, truth_uint8))
        if not all(np.isfinite(value) for key, value in row.items() if key != "filename"):
            raise FloatingPointError(f"Non-finite metric for {row['filename']}")
        rows.append(row)
        if image_directory is not None:
            image_directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(quantized).save(image_directory / (Path(row["filename"]).stem + ".png"))
    if not rows:
        raise ValueError("Cannot evaluate an empty partition")
    summary = {"n": len(rows)}
    for name in rows[0]:
        if name != "filename":
            values = [row[name] for row in rows]
            summary[name] = {"mean": float(np.mean(values)), "image_std": float(np.std(values))}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--domains", choices=DOMAINS, nargs="+", default=list(DOMAINS))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device(args.device)
    datasets = {domain: PairedDataset(args.data_root, domain, args.split) for domain in args.domains}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SmoothAlpha()
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()
    import lpips
    perceptual = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "image_size": 256, "domains": {}}
    for domain, dataset in datasets.items():
        image_directory = args.output / "images" / domain if args.save_images else None
        summary, rows = evaluate(model, dataset, device, perceptual, image_directory)
        report["domains"][domain] = summary
        with (args.output / f"{domain}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"{domain}: n={summary['n']}, PSNR={summary['psnr']['mean']:.4f}, "
              f"SSIM={summary['ssim']['mean']:.6f}, LPIPS={summary['lpips']['mean']:.6f}", flush=True)
    report["macro_psnr"] = float(np.mean([value["psnr"]["mean"] for value in report["domains"].values()]))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Macro PSNR: {report['macro_psnr']:.4f} dB")


if __name__ == "__main__":
    main()
