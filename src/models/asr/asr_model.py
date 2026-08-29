"""
asr_model.py — Full Conformer-CTC ASR Model
=============================================
Combines ConvSubsampling + Conformer Encoder + CTC head into a
complete end-to-end speech recognition model for Amharic.

Architecture:
  Input: Log-Mel spectrogram [B, n_mels, T]
    ↓ ConvSubsampling (×4 time reduction)
    ↓ Linear projection + positional encoding
    ↓ Conformer Blocks × n_layers
    ↓ Linear → CTC softmax over Amharic character vocab
  Output: Log-probabilities [T/4, B, vocab_size]
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.models.asr.conformer import ConformerBlock, ConvSubsampling, RelativePositionalEncoding
from src.data.amharic_g2p import VOCAB_SIZE, PAD_TOKEN, AMHARIC_VOCAB


# ─────────────────────────────────────────────────────────────────────────────
# CTC Head
# ─────────────────────────────────────────────────────────────────────────────

class CTCHead(nn.Module):
    """
    Linear projection from encoder hidden dim to vocab logits for CTC.
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            log_probs: [T, B, vocab_size]  (CTC expects time-first)
        """
        x = self.dropout(x)
        x = self.linear(x)            # [B, T, vocab_size]
        x = x.transpose(0, 1)        # [T, B, vocab_size]
        return F.log_softmax(x, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Full ASR Model
# ─────────────────────────────────────────────────────────────────────────────

class AmharicASRModel(nn.Module):
    """
    End-to-end Conformer-CTC Amharic ASR model.

    Hyperparameters (default: small model suitable for CPU training):
      n_mels:       80    (log-Mel feature dimension)
      d_model:      256   (encoder hidden dimension)
      n_heads:      4     (attention heads)
      n_layers:     6     (Conformer blocks)
      ff_expansion: 4     (feed-forward expansion ratio)
      kernel_size:  31    (depthwise conv kernel)
      dropout:      0.1

    Larger configuration for GPU training:
      d_model=512, n_heads=8, n_layers=12
    """

    def __init__(
        self,
        n_mels:       int   = 80,
        vocab_size:   int   = VOCAB_SIZE,
        d_model:      int   = 256,
        n_heads:      int   = 4,
        n_layers:     int   = 6,
        ff_expansion: int   = 4,
        kernel_size:  int   = 31,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size

        # 1. Convolutional subsampling: mel → [B, T/4, d_model]
        self.subsampling = ConvSubsampling(n_mels=n_mels, d_model=d_model)

        # 2. Input projection + dropout
        self.input_proj    = nn.Linear(d_model, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 3. Positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model, dropout=dropout)

        # 4. Conformer encoder blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # 5. CTC output head
        self.ctc_head = CTCHead(d_model, vocab_size, dropout=dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(
        self,
        mels: torch.Tensor,
        mel_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            mels:     [B, n_mels, T] log-Mel spectrograms
            mel_lens: [B] actual lengths before padding (optional)

        Returns:
            log_probs: [T/4, B, vocab_size] — for CTC loss
            out_lens:  [B]                  — output lengths after subsampling
        """
        B, n_mels, T = mels.shape

        # 1. Subsampling: [B, n_mels, T] → [B, T/4, d_model]
        x = self.subsampling(mels)
        T_sub = x.shape[1]

        # Compute output lengths after ×4 subsampling
        if mel_lens is not None:
            out_lens = torch.ceil(mel_lens.float() / 4).long().clamp(max=T_sub)
        else:
            out_lens = torch.full((B,), T_sub, dtype=torch.long)

        # 2. Input projection
        x = self.input_proj(x)
        x = self.input_dropout(x)

        # 3. Positional encoding
        x, _ = self.pos_enc(x)

        # 4. Build padding mask [B, T_sub]: True where padded
        key_padding_mask = None
        if mel_lens is not None:
            mask = torch.arange(T_sub, device=x.device).unsqueeze(0) >= out_lens.unsqueeze(1)
            key_padding_mask = mask   # [B, T_sub]

        # 5. Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        # 6. CTC head: [B, T, d_model] → [T, B, vocab_size]
        log_probs = self.ctc_head(x)

        return log_probs, out_lens

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def greedy_decode(log_probs: torch.Tensor, blank_id: int = 0) -> list:
        """
        CTC greedy decoding: argmax + collapse repeated + remove blank.

        Args:
            log_probs: [T, B, vocab_size]
            blank_id:  Index of the CTC blank token (default 0 = PAD)

        Returns:
            List of B decoded token sequences
        """
        # [T, B] argmax
        best = log_probs.argmax(dim=-1).transpose(0, 1)  # [B, T]
        results = []

        for b in range(best.shape[0]):
            seq = best[b].tolist()
            # Collapse repeats
            collapsed = [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
            # Remove blank (PAD)
            decoded = [t for t in collapsed if t != blank_id]
            results.append(decoded)

        return results

    def compute_ctc_loss(
        self,
        log_probs:   torch.Tensor,
        targets:     torch.Tensor,
        out_lens:    torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            log_probs:   [T, B, vocab_size]
            targets:     [B, N_max] padded token sequences
            out_lens:    [B] encoder output lengths
            target_lens: [B] target sequence lengths

        Returns:
            Scalar CTC loss
        """
        return F.ctc_loss(
            log_probs   = log_probs,
            targets     = targets,
            input_lengths  = out_lens,
            target_lengths = target_lens,
            blank          = AMHARIC_VOCAB[PAD_TOKEN],
            reduction      = "mean",
            zero_infinity  = True,   # stability: ignore inf losses
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_asr_model(config: Dict) -> AmharicASRModel:
    """
    Build ASR model from a config dict.

    Example config:
    {
        "n_mels": 80,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 6,
        "ff_expansion": 4,
        "kernel_size": 31,
        "dropout": 0.1,
    }
    """
    return AmharicASRModel(
        n_mels=config.get("n_mels", 80),
        vocab_size=VOCAB_SIZE,
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 6),
        ff_expansion=config.get("ff_expansion", 4),
        kernel_size=config.get("kernel_size", 31),
        dropout=config.get("dropout", 0.1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = AmharicASRModel(n_mels=80, d_model=256, n_heads=4, n_layers=6)
    print(f"Model parameters: {model.count_parameters():,}")

    # Fake batch: 2 samples, 80 mel bins, 200 frames
    dummy_mel     = torch.randn(2, 80, 200)
    dummy_lengths = torch.tensor([200, 160])

    log_probs, out_lens = model(dummy_mel, dummy_lengths)
    print(f"log_probs shape: {log_probs.shape}")   # [T/4, B, vocab]
    print(f"out_lens:        {out_lens}")

    decoded = AmharicASRModel.greedy_decode(log_probs)
    print(f"Decoded lengths: {[len(d) for d in decoded]}")
    print("✅ ASR model forward pass OK")
