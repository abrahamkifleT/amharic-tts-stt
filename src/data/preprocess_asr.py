"""
preprocess_asr.py — Audio Preprocessing for ASR Training
=========================================================
Prepares raw Amharic audio data for training the Conformer-CTC model:
  1. Resamples all audio to 16kHz mono
  2. Normalizes RMS volume
  3. Computes log-Mel spectrograms (80 bands)
  4. Tokenizes transcripts using the Amharic vocabulary
  5. Splits dataset into train / val / test

Usage:
    python src/data/preprocess_asr.py \
        --input  data/raw/all_metadata.csv \
        --output data/processed/asr \
        --split  0.8 0.1 0.1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.amharic_g2p import normalize_text, text_to_tokens, VOCAB_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# ASR Audio Configuration
# ─────────────────────────────────────────────────────────────────────────────

ASR_CONFIG = {
    "sample_rate":   16000,   # Hz — standard for ASR
    "n_mels":        80,      # Mel filter banks
    "n_fft":         400,     # FFT window size (25ms at 16kHz)
    "hop_length":    160,     # Hop size (10ms at 16kHz)
    "win_length":    400,     # Window length
    "f_min":         0.0,
    "f_max":         8000.0,
    "max_duration":  20.0,    # seconds — discard longer clips
    "min_duration":  0.1,     # seconds — discard shorter clips
    "target_rms":    0.05,    # RMS normalization target
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resample(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file (WAV/MP3/FLAC)
        target_sr:  Target sample rate in Hz

    Returns:
        1-D float32 tensor (mono waveform)
    """
    waveform, orig_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if orig_sr != target_sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # [T]


def normalize_volume(waveform: torch.Tensor, target_rms: float = 0.05) -> torch.Tensor:
    """
    Normalize waveform to a target RMS level.

    Args:
        waveform:   1-D audio tensor
        target_rms: Desired RMS amplitude

    Returns:
        Volume-normalized waveform
    """
    rms = waveform.pow(2).mean().sqrt()
    if rms > 1e-8:
        waveform = waveform * (target_rms / rms)
    return torch.clamp(waveform, -1.0, 1.0)


def compute_log_mel(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """
    Compute log-Mel spectrogram from a waveform.

    Args:
        waveform:    1-D audio tensor [T]
        sample_rate: Audio sample rate
        n_mels:      Number of Mel filter banks
        n_fft:       FFT size
        hop_length:  Hop size in samples
        win_length:  Window length in samples
        f_min, f_max: Frequency range for Mel filters

    Returns:
        Log-Mel spectrogram [n_mels, T_frames]
    """
    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
        normalized=False,
    )
    mel = transform(waveform)                   # [n_mels, T]
    log_mel = torch.log(mel + 1e-9)             # log compression
    return log_mel


# ─────────────────────────────────────────────────────────────────────────────
# Main Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_asr(
    metadata_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float   = 0.1,
    test_ratio: float  = 0.1,
    max_workers: int   = 4,
):
    """
    Full ASR preprocessing pipeline.

    For each audio file:
    - Load & resample to 16kHz
    - Normalize volume
    - Compute log-Mel spectrogram → save as .pt tensor
    - Tokenize transcript → save alongside

    Produces:
        output_dir/
          train/
            <id>.pt   (dict with keys: mel, tokens, text, duration)
          val/
          test/
          config.json  (ASR config used)
    """
    import json
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"[ASR Preprocess] Loaded {len(df)} utterances from {metadata_csv}")

    # Filter by duration
    if "duration" not in df.columns:
        # Estimate if not present
        df["duration"] = 5.0   # placeholder

    df = df[
        (df["duration"] >= ASR_CONFIG["min_duration"]) &
        (df["duration"] <= ASR_CONFIG["max_duration"])
    ].reset_index(drop=True)
    print(f"  After duration filter: {len(df)} utterances")

    # Normalize text and filter empty
    df["text_norm"] = df["text"].apply(normalize_text)
    df = df[df["text_norm"].str.len() > 0].reset_index(drop=True)
    print(f"  After text filter: {len(df)} utterances")

    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": df.iloc[:n_train],
        "val":   df.iloc[n_train : n_train + n_val],
        "test":  df.iloc[n_train + n_val :],
    }

    stats = {}
    for split_name, split_df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        print(f"\n  Processing {split_name} ({len(split_df)} utterances)...")
        skipped  = 0
        saved    = []

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split_name}"):
            try:
                # Load audio
                waveform = load_and_resample(
                    row["file"], target_sr=ASR_CONFIG["sample_rate"]
                )
                waveform = normalize_volume(waveform, target_rms=ASR_CONFIG["target_rms"])

                # Compute log-Mel
                log_mel = compute_log_mel(
                    waveform,
                    sample_rate=ASR_CONFIG["sample_rate"],
                    n_mels=ASR_CONFIG["n_mels"],
                    n_fft=ASR_CONFIG["n_fft"],
                    hop_length=ASR_CONFIG["hop_length"],
                    win_length=ASR_CONFIG["win_length"],
                    f_min=ASR_CONFIG["f_min"],
                    f_max=ASR_CONFIG["f_max"],
                )  # [n_mels, T]

                # Tokenize
                tokens = text_to_tokens(row["text_norm"], add_bos=False, add_eos=False)

                # Save processed sample
                sample_id = f"{split_name}_{idx:06d}"
                out_path  = split_dir / f"{sample_id}.pt"

                torch.save({
                    "id":       sample_id,
                    "mel":      log_mel,                      # [n_mels, T]
                    "tokens":   torch.tensor(tokens, dtype=torch.long),
                    "text":     row["text_norm"],
                    "duration": waveform.shape[0] / ASR_CONFIG["sample_rate"],
                    "dataset":  row.get("dataset", "unknown"),
                }, out_path)

                saved.append({
                    "id":       sample_id,
                    "file":     str(out_path),
                    "text":     row["text_norm"],
                    "n_frames": log_mel.shape[1],
                    "n_tokens": len(tokens),
                })

            except Exception as e:
                skipped += 1
                continue

        # Save split manifest
        manifest = pd.DataFrame(saved)
        manifest.to_csv(split_dir / "manifest.csv", index=False)
        stats[split_name] = {"total": len(split_df), "saved": len(saved), "skipped": skipped}
        print(f"    Saved: {len(saved)}, Skipped: {skipped}")

    # Save config
    config_path = output_dir / "asr_config.json"
    with open(config_path, "w") as f:
        json.dump({**ASR_CONFIG, "vocab_size": VOCAB_SIZE}, f, indent=2)

    print(f"\n[ASR Preprocess] Complete!")
    print(f"  Output: {output_dir}")
    print(f"  Config: {config_path}")
    for split, s in stats.items():
        print(f"  {split:6s}: {s['saved']:5d} / {s['total']:5d}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess Amharic audio data for ASR")
    parser.add_argument("--input",  type=Path, required=True,
                        help="Path to all_metadata.csv or a single dataset CSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for processed .pt files")
    parser.add_argument("--split",  type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratios (default: 0.8 0.1 0.1)")
    args = parser.parse_args()

    train_r, val_r, test_r = args.split
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    preprocess_asr(
        metadata_csv=args.input,
        output_dir=args.output,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
    )


if __name__ == "__main__":
    main()
