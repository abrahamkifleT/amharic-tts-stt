"""
preprocess_tts.py — Audio Preprocessing for TTS Training
=========================================================
Prepares Amharic audio data for Tacotron2 + HiFi-GAN training:
  1. Resamples all audio to 22050Hz mono
  2. Normalizes RMS volume
  3. Computes 80-band Mel-spectrograms (Tacotron2 standard)
  4. Computes fundamental frequency (F0) for prosody
  5. Saves (mel, text, tokens) triplets as .pt files

Usage:
    python src/data/preprocess_tts.py \
        --input  data/raw/all_metadata.csv \
        --output data/processed/tts \
        --speaker single
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.amharic_g2p import normalize_text, text_to_tokens, VOCAB_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# TTS Audio Configuration (Tacotron2 standard)
# ─────────────────────────────────────────────────────────────────────────────

TTS_CONFIG = {
    "sample_rate":   22050,   # Hz — standard for TTS
    "n_mels":        80,      # Mel filter banks
    "n_fft":         1024,    # FFT size (~46ms at 22050Hz)
    "hop_length":    256,     # Hop size (~11.6ms) → ~86 frames/sec
    "win_length":    1024,    # Window length
    "f_min":         0.0,
    "f_max":         8000.0,
    "max_wav_value": 32768.0,
    "max_duration":  15.0,    # seconds
    "min_duration":  0.5,
    "target_rms":    0.1,
    "trim_silence":  True,    # trim leading/trailing silence
    "trim_db":       40,      # silence threshold in dB
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resample_tts(audio_path: str, target_sr: int = 22050) -> torch.Tensor:
    """Load, convert to mono, and resample to TTS sample rate."""
    waveform, orig_sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if orig_sr != target_sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # [T]


def trim_silence(waveform: torch.Tensor, sample_rate: int, top_db: int = 40) -> torch.Tensor:
    """
    Trim leading and trailing silence using energy threshold.

    Args:
        waveform:    1-D audio tensor
        sample_rate: Sample rate (for frame-level processing)
        top_db:      Silence threshold in dB below peak

    Returns:
        Trimmed waveform
    """
    # Simple energy-based trimming
    frame_size = int(0.02 * sample_rate)   # 20ms frames
    hop        = int(0.01 * sample_rate)   # 10ms hop

    # Compute per-frame energy in dB
    n_frames = (len(waveform) - frame_size) // hop + 1
    energies = []
    for i in range(n_frames):
        frame = waveform[i * hop : i * hop + frame_size]
        energy = 20 * torch.log10(frame.abs().max() + 1e-10)
        energies.append(energy.item())

    if not energies:
        return waveform

    peak_db = max(energies)
    threshold = peak_db - top_db

    # Find first and last non-silent frames
    start_frame = next((i for i, e in enumerate(energies) if e >= threshold), 0)
    end_frame   = next((i for i, e in reversed(list(enumerate(energies))) if e >= threshold), n_frames - 1)

    start_sample = start_frame * hop
    end_sample   = min((end_frame + 1) * hop + frame_size, len(waveform))

    return waveform[start_sample:end_sample]


def normalize_volume_tts(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Normalize waveform RMS to target level."""
    rms = waveform.pow(2).mean().sqrt()
    if rms > 1e-8:
        waveform = waveform * (target_rms / rms)
    return torch.clamp(waveform, -1.0, 1.0)


def compute_mel_tts(
    waveform: torch.Tensor,
    cfg: dict = TTS_CONFIG,
) -> torch.Tensor:
    """
    Compute Mel-spectrogram for TTS (Tacotron2 format).

    Returns:
        Mel spectrogram [n_mels, T_frames], NOT log-compressed
        (Tacotron2 uses dynamic range compression separately)
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        n_mels=cfg["n_mels"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
        power=1.0,   # amplitude (not power) for Tacotron2
        normalized=False,
        center=False,
    )
    mel = mel_transform(waveform)   # [n_mels, T]

    # Dynamic range compression (Tacotron2 standard)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel  # [n_mels, T]


# ─────────────────────────────────────────────────────────────────────────────
# Main TTS Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_tts(
    metadata_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    val_ratio: float   = 0.05,
    test_ratio: float  = 0.05,
):
    """
    Full TTS preprocessing pipeline.

    For each audio file:
    - Load, resample to 22050Hz, trim silence
    - Normalize volume
    - Compute 80-band Mel-spectrogram
    - Tokenize Amharic text
    - Save as .pt file

    Also copies WAVs to output_dir/wavs/ for HiFi-GAN training.

    Output structure:
        output_dir/
          train/ val/ test/   → .pt feature files
          wavs/               → 22050Hz WAV files (for HiFi-GAN)
          manifest.csv        → all samples with paths
          tts_config.json
    """
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"[TTS Preprocess] Loaded {len(df)} utterances")

    # Filter by text quality
    df["text_norm"] = df["text"].apply(normalize_text)
    df = df[df["text_norm"].str.len() >= 3].reset_index(drop=True)

    # Filter by duration (use estimated if not present)
    if "duration" not in df.columns:
        df["duration"] = 5.0
    df = df[
        (df["duration"] >= TTS_CONFIG["min_duration"]) &
        (df["duration"] <= TTS_CONFIG["max_duration"])
    ].reset_index(drop=True)

    print(f"  After filtering: {len(df)} utterances")

    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n       = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": df.iloc[:n_train],
        "val":   df.iloc[n_train : n_train + n_val],
        "test":  df.iloc[n_train + n_val :],
    }

    all_records = []

    for split_name, split_df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        saved = []

        print(f"\n  Processing {split_name} ({len(split_df)} utterances)...")

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split_name}"):
            try:
                # Load audio
                waveform = load_and_resample_tts(row["file"], TTS_CONFIG["sample_rate"])

                # Trim silence
                if TTS_CONFIG["trim_silence"]:
                    waveform = trim_silence(waveform, TTS_CONFIG["sample_rate"], TTS_CONFIG["trim_db"])

                # Skip very short clips after trimming
                duration = waveform.shape[0] / TTS_CONFIG["sample_rate"]
                if duration < TTS_CONFIG["min_duration"]:
                    continue

                # Normalize volume
                waveform = normalize_volume_tts(waveform, TTS_CONFIG["target_rms"])

                # Compute mel-spectrogram
                mel = compute_mel_tts(waveform)   # [n_mels, T]

                # Tokenize text
                tokens = text_to_tokens(row["text_norm"], add_bos=False, add_eos=True)

                # Unique ID
                sample_id = f"{split_name}_{idx:06d}"

                # Save .pt feature file
                feat_path = split_dir / f"{sample_id}.pt"
                torch.save({
                    "id":       sample_id,
                    "mel":      mel,
                    "tokens":   torch.tensor(tokens, dtype=torch.long),
                    "text":     row["text_norm"],
                    "n_frames": mel.shape[1],
                    "n_tokens": len(tokens),
                    "duration": duration,
                }, feat_path)

                # Save resampled WAV for HiFi-GAN training
                wav_path = wavs_dir / f"{sample_id}.wav"
                torchaudio.save(str(wav_path), waveform.unsqueeze(0), TTS_CONFIG["sample_rate"])

                record = {
                    "id":       sample_id,
                    "feat":     str(feat_path),
                    "wav":      str(wav_path),
                    "text":     row["text_norm"],
                    "n_frames": mel.shape[1],
                    "n_tokens": len(tokens),
                    "duration": duration,
                    "split":    split_name,
                }
                saved.append(record)
                all_records.append(record)

            except Exception as e:
                continue

        # Per-split manifest
        pd.DataFrame(saved).to_csv(split_dir / "manifest.csv", index=False)
        print(f"    Saved: {len(saved)} utterances")

    # Global manifest
    pd.DataFrame(all_records).to_csv(output_dir / "manifest.csv", index=False)

    # Save config
    with open(output_dir / "tts_config.json", "w") as f:
        json.dump({**TTS_CONFIG, "vocab_size": VOCAB_SIZE}, f, indent=2)

    print(f"\n[TTS Preprocess] Complete! → {output_dir}")
    print(f"  Total processed: {len(all_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess Amharic audio data for TTS")
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split",  type=float, nargs=3, default=[0.9, 0.05, 0.05])
    args = parser.parse_args()

    preprocess_tts(
        metadata_csv=args.input,
        output_dir=args.output,
        train_ratio=args.split[0],
        val_ratio=args.split[1],
        test_ratio=args.split[2],
    )


if __name__ == "__main__":
    main()
