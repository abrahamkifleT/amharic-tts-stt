"""
train_asr.py — Amharic ASR Training Loop (Conformer-CTC)
=========================================================
Full training script for the Amharic ASR model.

Usage:
    # CPU training (slow but works):
    python src/training/train_asr.py --config configs/asr_small.yaml

    # Google Colab (recommended):
    # Use notebooks/02_train_asr_colab.ipynb

Features:
    - CTC loss with SpecAugment data augmentation
    - Learning rate warm-up + cosine annealing
    - WER evaluation on validation set every N steps
    - TensorBoard logging + model checkpointing
    - Resume from checkpoint
"""

import os
import sys
import math
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.asr.asr_model import AmharicASRModel, build_asr_model
from src.data.dataset import get_asr_dataloader
from src.training.metrics import (
    TrainingTracker, word_error_rate, character_error_rate, decode_predictions
)
from src.data.amharic_g2p import tokens_to_text


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model": {
        "n_mels":       80,
        "d_model":      256,
        "n_heads":      4,
        "n_layers":     6,
        "ff_expansion": 4,
        "kernel_size":  31,
        "dropout":      0.1,
    },
    # Data
    "data": {
        "train_manifest": "data/processed/asr/train/manifest.csv",
        "val_manifest":   "data/processed/asr/val/manifest.csv",
        "batch_size":     8,    # Reduce to 4 for CPU
        "num_workers":    0,    # 0 for Windows CPU
        "augment":        True,
    },
    # Training
    "training": {
        "max_steps":       100_000,
        "warmup_steps":    4_000,
        "learning_rate":   1e-3,
        "grad_clip":       5.0,
        "log_interval":    100,
        "eval_interval":   1_000,
        "save_interval":   2_000,
        "checkpoint_dir":  "checkpoints",
        "log_dir":         "runs/asr",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Schedule (Warm-up + Cosine Decay)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Transformer learning rate schedule (Vaswani et al., 2017):
      lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


# ─────────────────────────────────────────────────────────────────────────────
# Validation Step
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, max_batches: int = 50):
    """
    Run validation and compute WER/CER on the validation set.

    Args:
        model:      ASR model
        val_loader: Validation DataLoader
        device:     torch device
        max_batches: Maximum batches to evaluate (for speed)

    Returns:
        dict with val_loss, wer, cer
    """
    model.eval()
    total_loss = 0.0
    all_refs   = []
    all_hyps   = []
    n_batches  = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break

        mels       = batch["mels"].to(device)
        tokens     = batch["tokens"].to(device)
        mel_lens   = batch["mel_lens"].to(device)
        token_lens = batch["token_lens"].to(device)

        log_probs, out_lens = model(mels, mel_lens)

        loss = model.compute_ctc_loss(log_probs, tokens, out_lens, token_lens)
        if not torch.isnan(loss):
            total_loss += loss.item()

        # Decode
        hyps = decode_predictions(log_probs)
        refs = [tokens_to_text(tokens[i, :token_lens[i]].tolist()) for i in range(len(hyps))]

        all_refs.extend(refs)
        all_hyps.extend(hyps)
        n_batches += 1

    model.train()

    wer = word_error_rate(all_refs, all_hyps)
    cer = character_error_rate(all_refs, all_hyps)

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "wer":      wer,
        "cer":      cer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    """
    Main ASR training loop.

    Args:
        config: Nested configuration dict (see DEFAULT_CONFIG)
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Amharic ASR Training — Conformer-CTC")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Build model
    model = build_asr_model(config["model"]).to(device)
    print(f"  Model parameters: {model.count_parameters():,}")

    # Data loaders
    train_loader = get_asr_dataloader(
        config["data"]["train_manifest"],
        batch_size=config["data"]["batch_size"],
        augment=config["data"]["augment"],
        num_workers=config["data"]["num_workers"],
        shuffle=True,
    )
    val_loader = get_asr_dataloader(
        config["data"]["val_manifest"],
        batch_size=config["data"]["batch_size"],
        augment=False,
        num_workers=config["data"]["num_workers"],
        shuffle=False,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Training tracker
    tracker = TrainingTracker(
        log_dir=config["training"]["log_dir"],
        experiment_name="AmharicASR-Conformer-CTC",
    )

    # Resume from checkpoint if exists
    ckpt_dir  = Path(config["training"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "asr_latest.pt"
    start_step = 0
    if ckpt_path.exists():
        start_step = tracker.load_checkpoint(model, optimizer, str(ckpt_path))
        print(f"  Resuming from step {start_step}")

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    step = start_step
    best_wer = float("inf")

    print(f"  Training for {config['training']['max_steps']:,} steps...\n")
    start_time = time.time()

    for step in range(start_step, config["training"]["max_steps"]):
        # Get batch (cycle through dataset)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        mels       = batch["mels"].to(device)
        tokens     = batch["tokens"].to(device)
        mel_lens   = batch["mel_lens"].to(device)
        token_lens = batch["token_lens"].to(device)

        # Learning rate schedule
        lr = get_lr(step + 1, config["model"]["d_model"], config["training"]["warmup_steps"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass
        optimizer.zero_grad()
        log_probs, out_lens = model(mels, mel_lens)
        loss = model.compute_ctc_loss(log_probs, tokens, out_lens, token_lens)

        # Backward pass
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

        # Logging
        if step % config["training"]["log_interval"] == 0:
            elapsed = time.time() - start_time
            tracker.log_scalars({"loss/train": loss.item(), "lr": lr}, step)
            print(f"  Step {step:6d} | loss: {loss.item():.4f} | lr: {lr:.2e} | {elapsed:.0f}s")

        # Validation
        if step % config["training"]["eval_interval"] == 0 and step > 0:
            val_metrics = validate(model, val_loader, device)
            tracker.log_scalars(val_metrics, step)
            print(f"\n  [Val @ {step}] loss: {val_metrics['val_loss']:.4f} | "
                  f"WER: {val_metrics['wer']:.3f} | CER: {val_metrics['cer']:.3f}")

            # Save best model
            if val_metrics["wer"] < best_wer:
                best_wer = val_metrics["wer"]
                tracker.save_checkpoint(
                    model, optimizer, step,
                    str(ckpt_dir / "asr_best.pt"),
                    extra={"wer": best_wer, "config": config["model"]},
                )
                print(f"  ✅ New best WER: {best_wer:.3f}")

        # Periodic checkpoint
        if step % config["training"]["save_interval"] == 0 and step > 0:
            tracker.save_checkpoint(model, optimizer, step, str(ckpt_path))

    # Final checkpoint
    tracker.save_checkpoint(model, optimizer, step, str(ckpt_path))
    tracker.print_summary()
    tracker.close()
    print(f"\n✅ Training complete! Best WER: {best_wer:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Amharic ASR (Conformer-CTC)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (optional, uses defaults if not given)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.config:
        with open(args.config) as f:
            user_config = yaml.safe_load(f)
        # Deep merge
        for section, values in user_config.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values

    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps

    train(config)


if __name__ == "__main__":
    main()
