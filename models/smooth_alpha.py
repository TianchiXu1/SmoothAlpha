"""SmoothAlpha restoration model with spatial physics conditioning."""

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """Residual block with eight-group normalization."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SkipGate(nn.Module):
    """Identity-initialized depth gate for a skip feature map."""

    def __init__(self, channels, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, skip, depth):
        depth = F.interpolate(depth, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.tanh(self.net(torch.cat([skip, depth], dim=1)))
        return skip * (1.0 + 0.25 * gate)


def _spatial_film(channels):
    module = nn.Sequential(
        nn.Conv2d(9, 32, 1), nn.SiLU(), nn.Conv2d(32, 2 * channels, 1),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


class RestorationBackbone(nn.Module):
    """Four-level encoder-decoder with spatial FiLM and gated skips."""

    def __init__(self, base_ch=56):
        super().__init__()
        c = base_ch
        self.enc1 = nn.Sequential(nn.Conv2d(4, c, 3, padding=1), nn.SiLU(), ResBlock(c))
        self.down1 = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.SiLU(), ResBlock(c), nn.Conv2d(c, 2 * c, 3, padding=1),
            nn.SiLU(), ResBlock(2 * c),
        )
        self.down2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        self.enc3 = nn.Sequential(
            nn.SiLU(), ResBlock(2 * c), nn.Conv2d(2 * c, 4 * c, 3, padding=1),
            nn.SiLU(), ResBlock(4 * c),
        )
        self.down3 = nn.Conv2d(4 * c, 4 * c, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(nn.SiLU(), ResBlock(4 * c), ResBlock(4 * c))
        self.gate1 = SkipGate(c)
        self.gate2 = SkipGate(2 * c)
        self.gate3 = SkipGate(4 * c)
        self.up3 = nn.ConvTranspose2d(4 * c, 4 * c, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.SiLU(), nn.Conv2d(8 * c, 4 * c, 1), ResBlock(4 * c),
            nn.Conv2d(4 * c, 2 * c, 1), ResBlock(2 * c),
        )
        self.up2 = nn.ConvTranspose2d(2 * c, 2 * c, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.SiLU(), ResBlock(4 * c), nn.Conv2d(4 * c, c, 1), ResBlock(c),
        )
        self.up1 = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.SiLU(), ResBlock(2 * c), nn.Conv2d(2 * c, c, 1), ResBlock(c),
        )
        self.out_mean = nn.Conv2d(c, 3, 1)
        self.out_logvar = nn.Sequential(nn.Conv2d(c, 3, 1))
        self.out_ext = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 3, 1),
        )
        self.sfilm_e3 = _spatial_film(4 * c)
        self.sfilm_b = _spatial_film(4 * c)
        self.sfilm_d3 = _spatial_film(2 * c)

    def encode(self, image, depth):
        """Encode an image without physical modulation for the EMA target."""
        e1 = self.enc1(torch.cat([image, depth], dim=1))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        return self.bottleneck(self.down3(e3))

    def forward(self, image, depth, condition):
        e1 = self.enc1(torch.cat([image, depth], dim=1))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        cm3 = F.interpolate(condition, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.sfilm_e3(cm3).chunk(2, dim=1)
        e3 = e3 * (1.0 + gamma) + beta
        z = self.bottleneck(self.down3(e3))
        cm_b = F.interpolate(condition, size=z.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.sfilm_b(cm_b).chunk(2, dim=1)
        b = z * (1.0 + gamma) + beta
        d3 = F.interpolate(self.up3(b), size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, self.gate3(e3, depth)], dim=1))
        gamma, beta = self.sfilm_d3(cm3).chunk(2, dim=1)
        d3 = d3 * (1.0 + gamma) + beta
        d2 = F.interpolate(self.up2(d3), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, self.gate2(e2, depth)], dim=1))
        d1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, self.gate1(e1, depth)], dim=1))
        proposal = F.softplus(self.out_mean(d1))
        extinction = F.softplus(self.out_ext(d1))
        logvar = self.out_logvar(d1).clamp(-10, 10)
        size = image.shape[-2:]
        proposal = F.interpolate(proposal, size=size, mode="bilinear", align_corners=False)
        extinction = F.interpolate(extinction, size=size, mode="bilinear", align_corners=False)
        logvar = F.interpolate(logvar, size=size, mode="bilinear", align_corners=False)
        return proposal, extinction, logvar, z


class SmoothAlphaField(nn.Module):
    """Low-resolution residual and global scene parameterization of the medium."""

    def __init__(self):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.delta_head = nn.Conv2d(64, 3, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64),
            nn.SiLU(), nn.Linear(64, 6),
        )
        self.veil_low_head = nn.Conv2d(64, 3, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.veil_low_head.weight)
        nn.init.zeros_(self.veil_low_head.bias)
        nn.init.zeros_(self.global_head[-1].weight)
        with torch.no_grad():
            self.global_head[-1].bias.copy_(torch.tensor([-1.5] * 3 + [-2.0] * 3))

    def forward(self, image, depth):
        features = self.feat_net(torch.cat([image, depth], dim=1))
        delta_low = self.delta_head(features)
        delta = F.interpolate(delta_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.global_head(features)
        scene_raw = raw[:, :3]
        alpha = F.softplus(scene_raw[:, :, None, None] + delta).clamp(max=8.0)
        transmission = torch.exp(-depth.clamp(min=1e-3) * alpha).clamp(min=1e-4, max=1.0)
        veil_low = F.interpolate(
            self.veil_low_head(features), size=image.shape[-2:], mode="bilinear", align_corners=False,
        )
        veil = torch.sigmoid(raw[:, 3:, None, None] + veil_low)
        return {
            "alpha": alpha, "T": transmission, "veil": veil,
            "a_scene": F.softplus(scene_raw), "delta_lowres": delta_low,
        }


class SmoothAlpha(nn.Module):
    """Shared RGB restoration model for water, fog, and smoke.

    Inputs are degraded RGB images and normalized degraded-input depth cues.
    The output follows x_pred = proposal * T + veil * (1 - T).
    """

    def __init__(self, base_ch=56):
        super().__init__()
        if base_ch <= 0 or base_ch % 8:
            raise ValueError("base_ch must be a positive multiple of eight")
        self.detail_path = RestorationBackbone(base_ch)
        self.alpha_field = SmoothAlphaField()
        self.register_buffer("blend_threshold", torch.tensor(0.1))
        self.register_buffer("blend_temperature", torch.tensor(0.05))
        self.regime_unc = nn.Sequential(
            nn.Conv2d(18, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 3, 1), nn.Softplus(),
        )

    def forward(self, image, depth):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape B x 3 x H x W")
        if depth.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError("depth must have shape B x 1 x H x W matching image")
        fields = self.alpha_field(image, depth)
        height, width = image.shape[-2:]
        condition_size = (max(1, height // 4), max(1, width // 4))
        delta = F.interpolate(fields["delta_lowres"], size=condition_size, mode="bilinear", align_corners=False)
        veil = F.interpolate(fields["veil"], size=condition_size, mode="bilinear", align_corners=False)
        scene = fields["a_scene"][:, :, None, None].expand(-1, -1, *condition_size)
        condition = torch.cat([delta, veil, scene], dim=1)
        proposal, _, _, z = self.detail_path(image, depth, condition)
        output = proposal * fields["T"] + fields["veil"] * (1.0 - fields["T"])
        return {
            **fields, "x_pred": output, "proposal": proposal, "z": z, "latent": z,
        }
