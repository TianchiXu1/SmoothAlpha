"""Training-only clean-scene latent prediction and EMA target encoder."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class LatentPredictor(nn.Module):
    """Convolutional predictor P(z) of the paired clean bottleneck representation z^+."""

    def __init__(self, channels=224, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, hidden)
        self.out = nn.Conv2d(hidden, channels, 1)

    def forward(self, z):
        value = F.silu(self.gn1(self.conv1(z)))
        value = F.silu(self.gn2(self.conv2(value)))
        return self.out(value)


class EMATargetEncoder(nn.Module):
    """Frozen EMA target encoder \bar{E} using the encoder half of the restoration backbone."""

    def __init__(self, detail_path, m_start=0.996, m_end=1.0):
        super().__init__()
        self.shadow = copy.deepcopy(detail_path)
        self.shadow.requires_grad_(False)
        self.shadow.eval()
        self.m_start = float(m_start)
        self.m_end = float(m_end)

    def momentum(self, step, total_steps):
        fraction = min(max(step / max(total_steps, 1), 0.0), 1.0)
        return self.m_end - (self.m_end - self.m_start) * (math.cos(math.pi * fraction) + 1.0) / 2.0

    @torch.no_grad()
    def update(self, detail_path, step, total_steps):
        momentum = self.momentum(step, total_steps)
        online = detail_path.state_dict()
        for name, value in self.shadow.state_dict().items():
            source = online[name]
            if value.is_floating_point():
                value.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)
            else:
                value.copy_(source)
        return momentum

    @torch.no_grad()
    def encode(self, clean_image, depth):
        return self.shadow.encode(clean_image, depth)


def latent_prediction_loss(predicted, target):
    """Smooth L1 distance (L_pred) between affine-free single-group normalized feature maps."""
    return F.smooth_l1_loss(F.group_norm(predicted, 1), F.group_norm(target.detach(), 1))


# Backward-compatible aliases
JEPAPredictor = LatentPredictor
jepa_loss = latent_prediction_loss
