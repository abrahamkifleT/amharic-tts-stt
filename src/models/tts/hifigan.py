"""
hifigan.py — HiFi-GAN Neural Vocoder
======================================
Implements HiFi-GAN V1 from:
  "HiFi-GAN: Generative Adversarial Networks for Efficient and
   High Fidelity Speech Synthesis" (Kong et al., 2020).

Converts Mel-spectrograms → high-quality 22050Hz audio waveforms.

Architecture:
  Generator:
    Transposed Conv (upsample) × 4
    + Multi-Receptive Field Fusion (MRF) residual blocks

  Discriminators (for training):
    Multi-Period Discriminator (MPD)
    Multi-Scale Discriminator (MSD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# HiFi-GAN V1 hyperparameters
HIFIGAN_CONFIG = {
    "upsample_rates":         [8, 8, 2, 2],     # total ×256 → 22050Hz / 256 hop ≈ 86 frames/s
    "upsample_kernel_sizes":  [16, 16, 4, 4],
    "resblock_kernel_sizes":  [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_initial_channel": 128,
    "n_mels":                 80,
}


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block with dilated convolutions
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Multi-dilated residual block used in the HiFi-GAN generator.
    Multiple dilation rates give the generator multi-scale context.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            p = (kernel_size - 1) * d // 2
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p)
            )
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2)
            )

        # Weight init
        for c in list(self.convs1) + list(self.convs2):
            nn.init.normal_(c.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt   # residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
# HiFi-GAN Generator
# ─────────────────────────────────────────────────────────────────────────────

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator:
      Mel → [TransposeConv × upsample + MRF residual blocks] → Waveform

    Input:  [B, n_mels, T_mel]
    Output: [B, 1, T_audio]  where T_audio = T_mel × product(upsample_rates)
    """

    def __init__(self, cfg: dict = HIFIGAN_CONFIG):
        super().__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])

        # Input convolution
        self.conv_pre = nn.Conv1d(
            cfg["n_mels"],
            cfg["upsample_initial_channel"],
            kernel_size=7, stride=1, padding=3
        )

        # Upsampling transposed convolutions
        self.ups = nn.ModuleList()
        ch = cfg["upsample_initial_channel"]
        for i, (r, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])):
            self.ups.append(nn.ConvTranspose1d(
                ch, ch // 2,
                kernel_size=k, stride=r,
                padding=(k - r) // 2
            ))
            ch //= 2

        # Multi-Receptive Field Fusion blocks
        self.resblocks = nn.ModuleList()
        ch = cfg["upsample_initial_channel"]
        for i in range(self.num_upsamples):
            ch //= 2
            for k, d in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock(ch, kernel_size=k, dilations=d))

        # Output convolution
        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, stride=1, padding=3)
        nn.init.normal_(self.conv_post.weight, std=0.01)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T_mel] Mel-spectrogram

        Returns:
            [B, 1, T_audio] audio waveform in [-1, 1]
        """
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # Sum MRF outputs
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = rb(x)
                else:
                    xs += rb(x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)   # output in [-1, 1]
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminators (for adversarial training)
# ─────────────────────────────────────────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """
    Single period discriminator operating on every p-th sample.
    Captures periodicity at different time scales.
    """

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        p = (kernel_size - 1) // 2

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32,  (kernel_size, 1), (stride, 1), (p, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (p, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (p, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (p, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (p, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] waveform

        Returns:
            score: [B, 1, T', 1]
            fmaps: list of intermediate feature maps
        """
        x = x.squeeze(1)   # [B, T]
        T = x.shape[-1]

        # Pad to be divisible by period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            T = x.shape[-1]

        x = x.view(x.shape[0], 1, T // self.period, self.period)  # [B, 1, T/p, p]

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmaps


class MultiPeriodDiscriminator(nn.Module):
    """MPD: uses periods [2, 3, 5, 7, 11] to capture multi-scale structure."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator for average-pooled waveforms."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128, 15, 1, 7)),
            norm(nn.Conv1d(128,  128, 41, 2, 20, groups=4)),
            norm(nn.Conv1d(128,  256, 41, 2, 20, groups=16)),
            norm(nn.Conv1d(256,  512, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(1024,1024, 41, 1, 20, groups=16)),
            norm(nn.Conv1d(1024,1024,  5, 1,  2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmaps


class MultiScaleDiscriminator(nn.Module):
    """MSD: operates at 3 audio scales (original, ×2 avg pool, ×4 avg pool)."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y     = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def feature_matching_loss(fmap_r: list, fmap_g: list) -> torch.Tensor:
    """L1 distance between real and generated feature maps (from discriminators)."""
    loss = 0.0
    for r, g in zip(fmap_r, fmap_g):
        loss += F.l1_loss(r, g.detach())
    return loss * 2.0


def discriminator_loss(disc_real_outputs: list, disc_gen_outputs: list) -> torch.Tensor:
    """Standard GAN discriminator loss (least-squares)."""
    loss = 0.0
    for r, g in zip(disc_real_outputs, disc_gen_outputs):
        r_loss = torch.mean((1 - r) ** 2)
        g_loss = torch.mean(g ** 2)
        loss += r_loss + g_loss
    return loss


def generator_loss(disc_gen_outputs: list) -> torch.Tensor:
    """GAN generator loss (fool the discriminator)."""
    loss = 0.0
    for g in disc_gen_outputs:
        loss += torch.mean((1 - g) ** 2)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = HiFiGANGenerator()
    print(f"HiFi-GAN Generator parameters: {gen.count_parameters():,}")

    mel = torch.randn(2, 80, 100)
    audio = gen(mel)
    print(f"Input mel: {mel.shape}")
    print(f"Output audio: {audio.shape}")   # should be [2, 1, ~25600]
    print("✅ HiFi-GAN Generator forward pass OK")
