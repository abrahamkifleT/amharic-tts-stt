"""
train_tts.py — Tacotron2 TTS Training Loop
============================================
Full training script for the Tacotron2 + PostNet TTS model.

Loss function:
  L_total = L_mel_before + L_mel_after + L_gate
  where:
    L_mel_before = MSE(mel_out, mel_target)
    L_mel_after  = MSE(mel_refined, mel_target)
    L_gate       = BCEWithLogits(gate_out, stop_targets)

Usage:
    python src/training/train_tts.py
    # Or use notebooks/03_train_tts_colab.ipynb for GPU training
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.tts.tacotron2 import Tacotron2
from src.data.dataset import get_tts_dataloader
from src.training.metrics import TrainingTracker, mel_cepstral_distortion


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": {
        "embed_dim":         512,
        "encoder_dim":       512,
        "decoder_dim":       1024,
        "attention_dim":     128,
        "prenet_dim":        256,
        "n_mels":            80,
        "max_decoder_steps": 1000,
        "gate_threshold":    0.5,
        "dropout":           0.1,
    },
    "data": {
        "train_manifest": "data/processed/tts/train/manifest.csv",
        "val_manifest":   "data/processed/tts/val/manifest.csv",
        "batch_size":     4,    # Reduce to 2 for CPU
        "num_workers":    0,
    },
    "training": {
        "max_steps":      200_000,
        "learning_rate":  1e-3,
        "weight_decay":   1e-6,
        "grad_clip":      1.0,
        "log_interval":   50,
        "eval_interval":  500,
        "save_interval":  1000,
        "checkpoint_dir": "checkpoints",
        "log_dir":        "runs/tts",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────────────────────

class Tacotron2Loss(nn.Module):
    """
    Combined Tacotron2 training loss:
      - Mel MSE loss (before PostNet)
      - Mel MSE loss (after PostNet)
      - Gate BCE loss (stop token prediction)
    """

    def __init__(self, mel_weight: float = 1.0, gate_weight: float = 1.0):
        super().__init__()
        self.mel_weight  = mel_weight
        self.gate_weight = gate_weight
        self.mse         = nn.MSELoss()
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(
        self,
        mel_out:      torch.Tensor,   # [B, n_mels, T] raw decoder output
        mel_refined:  torch.Tensor,   # [B, n_mels, T] PostNet output
        gate_out:     torch.Tensor,   # [B, T] stop token logits
        mel_targets:  torch.Tensor,   # [B, n_mels, T]
        stop_targets: torch.Tensor,   # [B, T] binary stop tokens
        mel_lens:     torch.Tensor,   # [B] actual lengths
    ):
        # Mask padded frames for accurate loss computation
        B, n_mels, T = mel_targets.shape
        mask = torch.arange(T, device=mel_lens.device).unsqueeze(0) < mel_lens.unsqueeze(1)

        mel_mask = mask.unsqueeze(1).expand_as(mel_targets)   # [B, n_mels, T]

        # Mel losses (only on real frames)
        l_mel_before = self.mse(mel_out[mel_mask],     mel_targets[mel_mask])
        l_mel_after  = self.mse(mel_refined[mel_mask], mel_targets[mel_mask])

        # Gate loss
        l_gate = self.bce(gate_out[mask], stop_targets[mask])

        total = (
            self.mel_weight * (l_mel_before + l_mel_after) +
            self.gate_weight * l_gate
        )

        return total, {
            "mel_before": l_mel_before.item(),
            "mel_after":  l_mel_after.item(),
            "gate":       l_gate.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validation Step
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_tts(model, val_loader, loss_fn, device, max_batches: int = 20):
    """Run TTS validation and compute MCD."""
    model.eval()
    total_loss = 0.0
    mcd_scores = []
    n_batches  = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break

        tokens       = batch["tokens"].to(device)
        token_lens   = batch["token_lens"].to(device)
        mel_targets  = batch["mel_targets"].to(device)
        mel_lens     = batch["mel_lens"].to(device)
        stop_targets = batch["stop_targets"].to(device)

        mel_out, mel_refined, gate_out = model(tokens, token_lens, mel_targets, mel_lens)

        loss, _ = loss_fn(mel_out, mel_refined, gate_out, mel_targets, stop_targets, mel_lens)
        total_loss += loss.item()

        # MCD for first sample in batch
        mcd = mel_cepstral_distortion(mel_targets[0], mel_refined[0])
        if mcd < 50.0:  # sanity filter
            mcd_scores.append(mcd)

        n_batches += 1

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "mcd":      sum(mcd_scores) / max(len(mcd_scores), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_tts(config: dict):
    """
    Main Tacotron2 TTS training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Amharic TTS Training — Tacotron2")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Build model
    model = Tacotron2(**config["model"]).to(device)
    print(f"  Model parameters: {model.count_parameters():,}")

    # Data loaders
    train_loader = get_tts_dataloader(
        config["data"]["train_manifest"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        shuffle=True,
    )
    val_loader = get_tts_dataloader(
        config["data"]["val_manifest"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        shuffle=False,
    )

    # Loss + optimizer
    loss_fn = Tacotron2Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5
    )

    # Tracker
    tracker = TrainingTracker(
        log_dir=config["training"]["log_dir"],
        experiment_name="AmharicTTS-Tacotron2",
    )

    # Resume if available
    ckpt_dir  = Path(config["training"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "tts_latest.pt"
    start_step = 0
    if ckpt_path.exists():
        start_step = tracker.load_checkpoint(model, optimizer, str(ckpt_path))

    model.train()
    train_iter = iter(train_loader)
    step = start_step
    best_loss = float("inf")

    print(f"  Training for {config['training']['max_steps']:,} steps...")
    start_time = time.time()

    for step in range(start_step, config["training"]["max_steps"]):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens       = batch["tokens"].to(device)
        token_lens   = batch["token_lens"].to(device)
        mel_targets  = batch["mel_targets"].to(device)
        mel_lens     = batch["mel_lens"].to(device)
        stop_targets = batch["stop_targets"].to(device)

        optimizer.zero_grad()
        mel_out, mel_refined, gate_out = model(tokens, token_lens, mel_targets, mel_lens)
        loss, sub_losses = loss_fn(
            mel_out, mel_refined, gate_out, mel_targets, stop_targets, mel_lens
        )

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

        # Logging
        if step % config["training"]["log_interval"] == 0:
            elapsed = time.time() - start_time
            tracker.log_scalars(
                {"loss/train": loss.item(), **{f"loss/{k}": v for k, v in sub_losses.items()}},
                step
            )
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Step {step:6d} | total: {loss.item():.4f} | "
                  f"mel: {sub_losses['mel_after']:.4f} | "
                  f"gate: {sub_losses['gate']:.4f} | "
                  f"lr: {lr:.1e} | {elapsed:.0f}s")

        # Validation
        if step % config["training"]["eval_interval"] == 0 and step > 0:
            val_metrics = validate_tts(model, val_loader, loss_fn, device)
            tracker.log_scalars(val_metrics, step)
            scheduler.step(val_metrics["val_loss"])
            print(f"\n  [Val @ {step}] loss: {val_metrics['val_loss']:.4f} | "
                  f"MCD: {val_metrics['mcd']:.2f} dB\n")

            if val_metrics["val_loss"] < best_loss:
                best_loss = val_metrics["val_loss"]
                tracker.save_checkpoint(
                    model, optimizer, step,
                    str(ckpt_dir / "tts_best.pt"),
                    extra={"val_loss": best_loss, "mcd": val_metrics["mcd"]},
                )
                print(f"  ✅ New best val_loss: {best_loss:.4f}")

        # Periodic save
        if step % config["training"]["save_interval"] == 0 and step > 0:
            tracker.save_checkpoint(model, optimizer, step, str(ckpt_path))

    # Final save
    tracker.save_checkpoint(model, optimizer, step, str(ckpt_path))
    tracker.print_summary()
    tracker.close()
    print(f"\n✅ TTS Training complete! Best loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Amharic TTS (Tacotron2)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps

    train_tts(config)


if __name__ == "__main__":
    main()
