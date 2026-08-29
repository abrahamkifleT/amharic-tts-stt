"""
conformer.py — Conformer Block Architecture
============================================
Implements the Conformer (Convolution-augmented Transformer) encoder,
as described in: "Conformer: Convolution-augmented Transformer for
Speech Recognition" (Gulati et al., 2020).

Architecture of each Conformer block:
  1. Feed-Forward Module (half-step)
  2. Multi-Head Self-Attention Module
  3. Convolution Module
  4. Feed-Forward Module (half-step)
  5. Layer Normalization

This combination of attention (global context) + convolution (local
patterns) is particularly effective for speech recognition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x). Better than ReLU for speech."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit along the last dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Conformer (Transformer-XL style).
    Allows the model to attend to relative positions rather than absolute.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Sinusoidal positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            x with positional encoding added,
            positional encoding tensor (for relative attention)
        """
        seq_len = x.size(1)
        pos_enc = self.pe[:, :seq_len, :]   # [1, T, d_model]
        x = x + pos_enc
        return self.dropout(x), self.dropout(pos_enc)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Module
# ─────────────────────────────────────────────────────────────────────────────

class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module:
      LayerNorm → Linear(d→4d) → Swish → Dropout → Linear(4d→d) → Dropout
    Applied twice in each Conformer block with 1/2 residual weight.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm     = nn.LayerNorm(d_model)
        self.fc1      = nn.Linear(d_model, d_model * expansion)
        self.act      = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2      = nn.Linear(d_model * expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x   # half-step residual


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention Module
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttentionModule(nn.Module):
    """
    Conformer Multi-Head Self-Attention with relative positional encoding.
    LayerNorm → MHSA → Dropout → Residual
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"

        self.norm      = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, T, d_model]
            key_padding_mask: [B, T] bool mask (True = ignore)

        Returns:
            [B, T, d_model]
        """
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        return residual + x


# ─────────────────────────────────────────────────────────────────────────────
# Convolution Module
# ─────────────────────────────────────────────────────────────────────────────

class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module:
      LayerNorm → PointwiseConv → GLU → DepthwiseConv → BN → Swish
               → PointwiseConv → Dropout → Residual

    The depthwise separable convolution captures local acoustic patterns
    that attention misses.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2

        self.norm         = nn.LayerNorm(d_model)
        self.pointwise1   = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu          = GLU()
        self.depthwise    = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model   # depthwise
        )
        self.batch_norm   = nn.BatchNorm1d(d_model)
        self.activation   = Swish()
        self.pointwise2   = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        residual = x
        x = self.norm(x)

        # [B, T, d_model] → [B, d_model, T] for Conv1d
        x = x.transpose(1, 2)
        x = self.pointwise1(x)          # [B, 2*d_model, T]
        x = self.glu(x.transpose(1, 2)).transpose(1, 2)  # GLU → [B, d_model, T]
        x = self.depthwise(x)           # [B, d_model, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)          # [B, d_model, T]
        x = self.dropout(x)

        # Back to [B, T, d_model]
        x = x.transpose(1, 2)
        return residual + x


# ─────────────────────────────────────────────────────────────────────────────
# Conformer Block
# ─────────────────────────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    Single Conformer Block:
      FF(1/2) → MHSA → Conv → FF(1/2) → LayerNorm
    """

    def __init__(
        self,
        d_model:     int = 256,
        n_heads:     int = 4,
        ff_expansion:int = 4,
        kernel_size: int = 31,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.ff1   = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn  = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv  = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2   = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm  = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, T, d_model]
            key_padding_mask: [B, T] bool

        Returns:
            [B, T, d_model]
        """
        x = self.ff1(x)
        x = self.attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Convolutional Subsampling (×4 time reduction)
# ─────────────────────────────────────────────────────────────────────────────

class ConvSubsampling(nn.Module):
    """
    2-layer Conv2d subsampling that reduces time by factor of 4.
    Converts log-Mel [B, 1, n_mels, T] → [B, T/4, d_model].

    This is standard for Conformer-based ASR to reduce sequence length
    before the attention layers.
    """

    def __init__(self, n_mels: int = 80, d_model: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model // 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model // 4, d_model // 4, kernel_size=3, stride=2, padding=1)
        self.act   = nn.ReLU()

        # Compute output feature size
        out_mels = math.ceil(n_mels / 4)
        self.proj = nn.Linear(d_model // 4 * out_mels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_mels, T]

        Returns:
            [B, T/4, d_model]
        """
        x = x.unsqueeze(1)            # [B, 1, n_mels, T]
        x = self.act(self.conv1(x))   # [B, C, n_mels/2, T/2]
        x = self.act(self.conv2(x))   # [B, C, n_mels/4, T/4]

        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2)    # [B, T, C, M]
        x = x.contiguous().view(B, T, C * M)   # [B, T, C*M]
        x = self.proj(x)              # [B, T, d_model]
        return x
