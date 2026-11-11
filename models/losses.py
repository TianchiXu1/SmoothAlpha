"""The manuscript's restoration and physical regularization objective."""
import torch
import torch.nn.functional as F
from torch import nn


def total_variation(value):
    """Anisotropic mean absolute horizontal and vertical differences."""
    return ((value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean()
            + (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean())


def perceptual_weight(epoch, config):
    """Ramp LPIPS from 0.1 to 0.5 over epochs 20--24."""
    fraction = max(0.0, min(1.0, (epoch - config["perceptual_start_epoch"] + 1)
                                 / config["perceptual_ramp_epochs"]))
    return config["perceptual"] * fraction


class RestorationLoss(nn.Module):
    """L1, MS-SSIM, LPIPS, stopped-gradient observation loss and field TV."""

    def __init__(self, config):
        super().__init__()
        from pytorch_msssim import MS_SSIM
        import lpips

        self.config = config
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3, win_size=7)
        # LPIPS initialization must not change the seeded optimization stream.
        with torch.random.fork_rng():
            self.perceptual = lpips.LPIPS(net="alex").eval().requires_grad_(False)

    def forward(self, output, observed, reference, epoch):
        weight = self.config
        restored = output["x_pred"]
        transmission, veil = output["T"], output["veil"]
        reobserved = restored.detach() * transmission + veil * (1.0 - transmission)
        terms = {
            "reconstruction": F.l1_loss(restored, reference),
            "observation_consistency": F.l1_loss(reobserved, observed),
            "extinction_tv": total_variation(output["delta_lowres"]),
            "extinction_amplitude": output["delta_lowres"].abs().mean(),
            "veiling_tv": total_variation(veil),
            "output_tv": total_variation(restored),
        }
        loss = (weight["reconstruction"] * terms["reconstruction"]
                + weight["observation_consistency"] * terms["observation_consistency"]
                + weight["extinction_tv"] * terms["extinction_tv"]
                + weight["output_tv"] * terms["output_tv"]
                + weight["extinction_amplitude"] * terms["extinction_amplitude"]
                + weight["veiling_tv"] * terms["veiling_tv"])
        ramp = perceptual_weight(epoch, weight)
        if ramp > 0:
            terms["perceptual"] = self.perceptual(
                restored.clamp(0, 1) * 2 - 1, reference.clamp(0, 1) * 2 - 1).mean()
            loss = loss + ramp * terms["perceptual"]
        terms["ms_ssim"] = 1.0 - self.ms_ssim(restored.clamp(0, 1), reference.clamp(0, 1))
        loss = loss + weight["ms_ssim"] * terms["ms_ssim"]
        return loss, {name: float(value.detach()) for name, value in terms.items()}
