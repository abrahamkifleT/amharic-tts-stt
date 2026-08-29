"""
dataset.py — PyTorch Dataset Classes for Amharic ASR & TTS
===========================================================
Provides two Dataset classes:
  - AmharicASRDataset: loads preprocessed .pt files for Conformer-CTC training
  - AmharicTTSDataset: loads preprocessed .pt files for Tacotron2 training
  - AmharicHiFiGANDataset: loads WAV segments for HiFi-GAN vocoder training
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ═════════════════════════════════════════════════════════════════════════════
# 1.  ASR Dataset
# ═════════════════════════════════════════════════════════════════════════════

class AmharicASRDataset(Dataset):
    """
    Dataset for Conformer-CTC ASR training.

    Each sample is a pre-processed .pt file containing:
      - mel:    [n_mels, T_frames] log-Mel spectrogram
      - tokens: [N] integer token IDs
      - text:   raw Amharic string
    """

    def __init__(
        self,
        manifest_csv: str,
        max_token_len: int = 512,
        max_frame_len: int = 2000,
        augment: bool      = False,
    ):
        """
        Args:
            manifest_csv:  Path to split manifest CSV (id, file, text, n_frames, n_tokens)
            max_token_len: Discard samples with more tokens than this
            max_frame_len: Discard samples with more mel frames than this
            augment:       Whether to apply SpecAugment during training
        """
        self.manifest = pd.read_csv(manifest_csv)
        self.augment  = augment

        # Filter by length
        self.manifest = self.manifest[
            (self.manifest["n_tokens"] <= max_token_len) &
            (self.manifest["n_frames"] <= max_frame_len)
        ].reset_index(drop=True)

        print(f"[ASRDataset] {len(self.manifest)} samples from {manifest_csv}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row  = self.manifest.iloc[idx]
        data = torch.load(row["file"], weights_only=True)

        mel    = data["mel"]     # [n_mels, T]
        tokens = data["tokens"]  # [N]

        if self.augment:
            mel = self._spec_augment(mel)

        return {
            "id":     row["id"],
            "mel":    mel,
            "tokens": tokens,
            "mel_len":    torch.tensor(mel.shape[1], dtype=torch.long),
            "token_len":  torch.tensor(len(tokens),  dtype=torch.long),
        }

    def _spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment: random time and frequency masking.
        Significantly improves ASR robustness.

        Args:
            mel: [n_mels, T]

        Returns:
            Augmented mel spectrogram
        """
        mel = mel.clone()
        n_mels, T = mel.shape

        # Frequency masking (up to 27 bins)
        f_mask_width = random.randint(0, min(27, n_mels))
        f_start      = random.randint(0, n_mels - f_mask_width)
        mel[f_start : f_start + f_mask_width, :] = 0.0

        # Time masking (up to 100 frames or 10% of total)
        t_mask_width = random.randint(0, min(100, int(0.1 * T)))
        t_start      = random.randint(0, max(0, T - t_mask_width))
        mel[:, t_start : t_start + t_mask_width] = 0.0

        return mel


def asr_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate ASR samples into a padded batch.

    Pads mel spectrograms along the time axis and tokens with 0 (PAD).

    Returns dict with:
      mels:      [B, n_mels, T_max]
      tokens:    [B, N_max]
      mel_lens:  [B]
      token_lens:[B]
    """
    mels       = [item["mel"].T for item in batch]       # list of [T, n_mels]
    tokens     = [item["tokens"] for item in batch]
    mel_lens   = torch.stack([item["mel_len"]   for item in batch])
    token_lens = torch.stack([item["token_len"] for item in batch])

    mels_padded   = pad_sequence(mels, batch_first=True).transpose(1, 2)   # [B, n_mels, T]
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0) # [B, N]

    return {
        "mels":       mels_padded,
        "tokens":     tokens_padded,
        "mel_lens":   mel_lens,
        "token_lens": token_lens,
    }


def get_asr_dataloader(
    manifest_csv: str,
    batch_size: int  = 16,
    augment: bool    = False,
    num_workers: int = 2,
    shuffle: bool    = True,
) -> DataLoader:
    """Create a DataLoader for ASR training."""
    dataset = AmharicASRDataset(manifest_csv, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=asr_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TTS Dataset
# ═════════════════════════════════════════════════════════════════════════════

class AmharicTTSDataset(Dataset):
    """
    Dataset for Tacotron2 TTS training.

    Each sample is a pre-processed .pt file containing:
      - mel:    [n_mels, T_frames] Mel-spectrogram
      - tokens: [N] character token IDs (input to encoder)
      - text:   raw Amharic string
    """

    def __init__(
        self,
        manifest_csv: str,
        max_token_len: int = 200,
        max_mel_len:   int = 1500,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[
            (self.manifest["n_tokens"] <= max_token_len) &
            (self.manifest["n_frames"] <= max_mel_len)
        ].reset_index(drop=True)

        print(f"[TTSDataset] {len(self.manifest)} samples from {manifest_csv}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row  = self.manifest.iloc[idx]
        data = torch.load(row["feat"], weights_only=True)

        return {
            "id":       row["id"],
            "tokens":   data["tokens"],      # [N]
            "mel":      data["mel"],          # [n_mels, T]
            "text":     data["text"],
            "n_tokens": len(data["tokens"]),
            "n_frames": data["mel"].shape[1],
        }


def tts_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate TTS samples into a padded batch.

    Tacotron2 needs:
      - token sequences padded to max length
      - mel targets padded to max length
      - stop token targets (0=continue, 1=stop)
    """
    batch = sorted(batch, key=lambda x: x["n_tokens"], reverse=True)

    tokens     = [item["tokens"] for item in batch]
    mels       = [item["mel"].T  for item in batch]   # → [T, n_mels]
    token_lens = torch.tensor([item["n_tokens"] for item in batch], dtype=torch.long)
    mel_lens   = torch.tensor([item["n_frames"] for item in batch], dtype=torch.long)

    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)  # [B, N]
    mels_padded   = pad_sequence(mels,   batch_first=True, padding_value=0.0) # [B, T, n_mels]
    mels_padded   = mels_padded.transpose(1, 2)   # [B, n_mels, T]

    # Build stop token targets: 0 everywhere except last real frame
    B, _, T_max = mels_padded.shape
    stop_targets = torch.zeros(B, T_max)
    for i, mel_len in enumerate(mel_lens):
        if mel_len > 0:
            stop_targets[i, mel_len - 1] = 1.0  # stop at last real frame

    return {
        "tokens":      tokens_padded,   # [B, N_max]
        "token_lens":  token_lens,      # [B]
        "mel_targets": mels_padded,     # [B, n_mels, T_max]
        "mel_lens":    mel_lens,        # [B]
        "stop_targets":stop_targets,    # [B, T_max]
    }


def get_tts_dataloader(
    manifest_csv: str,
    batch_size: int  = 8,
    num_workers: int = 2,
    shuffle: bool    = True,
) -> DataLoader:
    """Create a DataLoader for TTS training."""
    dataset = AmharicTTSDataset(manifest_csv)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tts_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  HiFi-GAN Dataset
# ═════════════════════════════════════════════════════════════════════════════

class AmharicHiFiGANDataset(Dataset):
    """
    Dataset for HiFi-GAN vocoder training.
    Uses raw WAV files + their Mel-spectrograms.
    Segments are randomly cropped to a fixed length.
    """

    SEGMENT_SIZE = 8192   # audio samples per training segment (~0.37s at 22050Hz)

    def __init__(
        self,
        manifest_csv: str,
        segment_size: int = SEGMENT_SIZE,
        fine_tuning: bool = False,
    ):
        """
        Args:
            manifest_csv:  Path to TTS manifest CSV (contains 'wav' and 'feat' columns)
            segment_size:  Audio samples per segment for adversarial training
            fine_tuning:   If True, use ground-truth mel (for fine-tuning on GT)
        """
        self.manifest     = pd.read_csv(manifest_csv)
        self.segment_size = segment_size
        self.fine_tuning  = fine_tuning
        print(f"[HiFiGANDataset] {len(self.manifest)} samples")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        # Load raw WAV
        waveform, sr = torchaudio.load(row["wav"])
        waveform = waveform.squeeze(0)   # [T]

        # Random segment crop for adversarial training
        if waveform.shape[0] >= self.segment_size:
            start = random.randint(0, waveform.shape[0] - self.segment_size)
            waveform = waveform[start : start + self.segment_size]
        else:
            # Pad short clips
            pad_len  = self.segment_size - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Load pre-computed mel (from TTS preprocessor)
        data = torch.load(row["feat"], weights_only=True)
        mel  = data["mel"]   # [n_mels, T_mel]

        # Align mel to segment
        mel_hop  = 256   # must match TTS_CONFIG["hop_length"]
        mel_start = start // mel_hop if waveform.shape[0] == self.segment_size else 0
        mel_end   = mel_start + self.segment_size // mel_hop

        if mel_end > mel.shape[1]:
            mel_end   = mel.shape[1]
            mel_start = max(0, mel_end - self.segment_size // mel_hop)

        mel_segment = mel[:, mel_start:mel_end]

        return waveform, mel_segment   # ([segment_size], [n_mels, T_seg])


def get_hifigan_dataloader(
    manifest_csv: str,
    batch_size: int  = 16,
    num_workers: int = 2,
    shuffle: bool    = True,
) -> DataLoader:
    """Create a DataLoader for HiFi-GAN training."""
    dataset = AmharicHiFiGANDataset(manifest_csv)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
