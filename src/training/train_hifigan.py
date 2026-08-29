"""
train_hifigan.py — HiFi-GAN Vocoder Adversarial Training
==========================================================
Trains the HiFi-GAN vocoder to convert Mel-spectrograms → waveforms.

Loss components:
  D_loss = Discriminator (GAN) loss
  G_loss = Generator loss + Feature Matching loss + Mel Spectrogram loss

Usage:
    python src/training/train_hifigan.py
    # Or use: notebooks/03_train_tts_colab.ipynb (Phase 2)
"""

import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.tts.hifigan import (
    HiFiGANGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    feature_matching_loss, discriminator_loss, generator_loss, HIFIGAN_CONFIG
)
from src.data.dataset import get_hifigan_dataloader
from src.training.metrics import TrainingTracker

DEFAULT_CONFIG = {
    "data": {
        "train_manifest": "data/processed/tts/train/manifest.csv",
        "val_manifest":   "data/processed/tts/val/manifest.csv",
        "batch_size":     8,
        "num_workers":    0,
        "segment_size":   8192,
    },
    "training": {
        "max_steps":      300_000,
        "lr_G":           2e-4,
        "lr_D":           2e-4,
        "betas":          [0.8, 0.99],
        "lr_decay":       0.999,
        "grad_clip":      1000.0,
        "log_interval":   100,
        "save_interval":  5000,
        "checkpoint_dir": "checkpoints",
        "log_dir":        "runs/hifigan",
        "sample_rate":    22050,
        "n_mels":         80,
        "n_fft":          1024,
        "hop_length":     256,
    },
}


def compute_mel_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    n_mels: int = 80,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Mel-spectrogram reconstruction loss (L1) between real and generated audio.
    Ensures the generated waveform matches the spectral content of real audio.
    """
    mel_fn = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, f_min=0.0, f_max=8000.0, power=1.0, normalized=False,
        center=False,
    ).to(device)

    y_mel     = torch.log(torch.clamp(mel_fn(y.squeeze(1)),     min=1e-5))
    y_hat_mel = torch.log(torch.clamp(mel_fn(y_hat.squeeze(1)), min=1e-5))

    return torch.nn.functional.l1_loss(y_hat_mel, y_mel)


def train_hifigan(config: dict):
    """Main HiFi-GAN adversarial training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Amharic HiFi-GAN Vocoder Training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Models
    G   = HiFiGANGenerator(HIFIGAN_CONFIG).to(device)
    MPD = MultiPeriodDiscriminator().to(device)
    MSD = MultiScaleDiscriminator().to(device)

    print(f"  Generator parameters:      {G.count_parameters():,}")

    # Optimizers
    opt_G = optim.AdamW(
        G.parameters(),
        lr=config["training"]["lr_G"],
        betas=tuple(config["training"]["betas"]),
    )
    opt_D = optim.AdamW(
        list(MPD.parameters()) + list(MSD.parameters()),
        lr=config["training"]["lr_D"],
        betas=tuple(config["training"]["betas"]),
    )

    # LR schedulers
    sch_G = optim.lr_scheduler.ExponentialLR(opt_G, gamma=config["training"]["lr_decay"])
    sch_D = optim.lr_scheduler.ExponentialLR(opt_D, gamma=config["training"]["lr_decay"])

    # Data
    train_loader = get_hifigan_dataloader(
        config["data"]["train_manifest"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        shuffle=True,
    )

    tracker = TrainingTracker(
        log_dir=config["training"]["log_dir"],
        experiment_name="AmharicHiFiGAN",
    )

    # Resume
    ckpt_dir  = Path(config["training"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "hifigan_latest.pt"
    start_step = 0
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        G.load_state_dict(ckpt.get("G_state", {}))
        MPD.load_state_dict(ckpt.get("MPD_state", {}))
        MSD.load_state_dict(ckpt.get("MSD_state", {}))
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from step {start_step}")

    G.train()
    MPD.train()
    MSD.train()
    train_iter = iter(train_loader)

    print(f"  Training for {config['training']['max_steps']:,} steps...")
    start_time = time.time()

    for step in range(start_step, config["training"]["max_steps"]):
        try:
            y, mel = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            y, mel = next(train_iter)

        y   = y.unsqueeze(1).to(device)    # [B, 1, T]
        mel = mel.to(device)               # [B, n_mels, T_seg]

        # Generate audio from mel
        y_hat = G(mel)

        # ── Discriminator update ──────────────────────────────────────────
        opt_D.zero_grad()

        y_d_rs_mpd, y_d_gs_mpd, _, _ = MPD(y, y_hat.detach())
        y_d_rs_msd, y_d_gs_msd, _, _ = MSD(y, y_hat.detach())

        loss_D = (
            discriminator_loss(y_d_rs_mpd, y_d_gs_mpd) +
            discriminator_loss(y_d_rs_msd, y_d_gs_msd)
        )
        loss_D.backward()
        opt_D.step()

        # ── Generator update ──────────────────────────────────────────────
        opt_G.zero_grad()

        _, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = MPD(y, y_hat)
        _, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = MSD(y, y_hat)

        loss_fm  = (
            feature_matching_loss(fmap_rs_mpd, fmap_gs_mpd) +
            feature_matching_loss(fmap_rs_msd, fmap_gs_msd)
        )
        loss_gen = generator_loss(y_d_gs_mpd) + generator_loss(y_d_gs_msd)
        loss_mel = compute_mel_loss(
            y_hat, y,
            n_mels=config["training"]["n_mels"],
            sample_rate=config["training"]["sample_rate"],
            n_fft=config["training"]["n_fft"],
            hop_length=config["training"]["hop_length"],
            device=device,
        ) * 45.0   # weight mel loss higher

        loss_G = loss_gen + loss_fm + loss_mel
        loss_G.backward()
        opt_G.step()

        # ── Logging ───────────────────────────────────────────────────────
        if step % config["training"]["log_interval"] == 0:
            elapsed = time.time() - start_time
            tracker.log_scalars({
                "G/total": loss_G.item(),
                "G/mel":   loss_mel.item(),
                "G/fm":    loss_fm.item(),
                "D/total": loss_D.item(),
            }, step)
            print(f"  Step {step:6d} | G: {loss_G.item():.3f} "
                  f"(mel: {loss_mel.item():.3f} | fm: {loss_fm.item():.3f}) "
                  f"| D: {loss_D.item():.3f} | {elapsed:.0f}s")

        # Decay LR every epoch
        if step % len(train_loader) == 0 and step > 0:
            sch_G.step()
            sch_D.step()

        # Checkpoint
        if step % config["training"]["save_interval"] == 0 and step > 0:
            ckpt_data = {
                "step":      step,
                "G_state":   G.state_dict(),
                "MPD_state": MPD.state_dict(),
                "MSD_state": MSD.state_dict(),
            }
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_data, str(ckpt_path))
            torch.save({"step": step, "G_state": G.state_dict()},
                       str(ckpt_dir / "hifigan_best.pt"))
            print(f"  [Checkpoint] Saved step {step}")

    tracker.close()
    print("\n✅ HiFi-GAN Training complete!")


if __name__ == "__main__":
    train_hifigan(DEFAULT_CONFIG)
