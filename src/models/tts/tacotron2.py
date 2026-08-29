"""
tacotron2.py — Full Tacotron2 TTS Model
========================================
Implements the complete Tacotron2 architecture from:
  "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram
   Predictions" (Shen et al., 2018).

Pipeline:
  Amharic characters → Encoder → Location-Sensitive Attention
  → Decoder → Mel spectrogram → PostNet → Refined Mel

This file assembles the Encoder, Attention, Decoder, and PostNet
submodules into the complete end-to-end model.
"""

import sys
import math
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.data.amharic_g2p import VOCAB_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Tacotron2Encoder(nn.Module):
    """
    Tacotron2 encoder:
      Character Embedding (512) → 3×Conv1d + BN + ReLU → BiLSTM(256)
    Maps a sequence of character IDs to encoder hidden states.
    """

    def __init__(
        self,
        vocab_size:    int = VOCAB_SIZE,
        embed_dim:     int = 512,
        n_conv_layers: int = 3,
        kernel_size:   int = 5,
        dropout:       float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Convolutional stack
        convs = []
        for i in range(n_conv_layers):
            in_ch  = embed_dim
            out_ch = embed_dim
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.conv_stack = nn.Sequential(*convs)

        # Bidirectional LSTM (→ 512 total, 256 per direction)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens:     [B, N] integer character IDs
            token_lens: [B] actual lengths (for packed LSTM)

        Returns:
            encoder_out: [B, N, 512] encoder hidden states
        """
        # Embedding: [B, N] → [B, N, embed_dim]
        x = self.embedding(tokens)

        # Conv stack expects [B, C, T]
        x = x.transpose(1, 2)       # [B, embed_dim, N]
        x = self.conv_stack(x)
        x = x.transpose(1, 2)       # [B, N, embed_dim]

        # BiLSTM
        if token_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, token_lens.cpu(), batch_first=True, enforce_sorted=True
            )
            out, _ = self.bilstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.bilstm(x)

        return out   # [B, N, 512]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Location-Sensitive Attention
# ─────────────────────────────────────────────────────────────────────────────

class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention from Tacotron2:
      Combines content-based attention (query + key) with
      location-based attention (accumulated attention weights).
    This prevents the model from getting "stuck" attending to the same
    encoder position and ensures monotonic left-to-right alignment.
    """

    def __init__(
        self,
        attention_dim:       int = 128,
        encoder_dim:         int = 512,
        decoder_dim:         int = 1024,
        n_location_filters:  int = 32,
        location_kernel_size:int = 31,
    ):
        super().__init__()
        # Project encoder outputs to attention dim
        self.W = nn.Linear(encoder_dim, attention_dim, bias=False)
        # Project decoder state to attention dim
        self.V = nn.Linear(decoder_dim, attention_dim, bias=False)
        # Location convolutional feature
        self.F = nn.Conv1d(
            1, n_location_filters,
            kernel_size=location_kernel_size,
            padding=(location_kernel_size - 1) // 2,
            bias=False,
        )
        self.U = nn.Linear(n_location_filters, attention_dim, bias=False)
        # Score projection
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        query:          torch.Tensor,    # [B, decoder_dim]
        encoder_out:    torch.Tensor,    # [B, N, encoder_dim]
        prev_attention: torch.Tensor,    # [B, N] cumulative attention weights
        mask:           Optional[torch.Tensor] = None,  # [B, N] True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:          Decoder LSTM hidden state [B, decoder_dim]
            encoder_out:    Encoder hidden states [B, N, 512]
            prev_attention: Accumulated attention from previous steps [B, N]
            mask:           Encoder padding mask [B, N]

        Returns:
            context:    Context vector [B, encoder_dim]
            attention:  Attention weights [B, N]
        """
        # Content features
        key = self.W(encoder_out)                          # [B, N, attn_dim]
        query_feat = self.V(query).unsqueeze(1)            # [B, 1, attn_dim]

        # Location features from cumulative attention
        loc_feat = self.F(prev_attention.unsqueeze(1))     # [B, n_filters, N]
        loc_feat = loc_feat.transpose(1, 2)                # [B, N, n_filters]
        loc_feat = self.U(loc_feat)                        # [B, N, attn_dim]

        # Energy scores
        energy = self.v(torch.tanh(key + query_feat + loc_feat)).squeeze(-1)  # [B, N]

        # Mask padding positions with -inf
        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        attention = F.softmax(energy, dim=-1)  # [B, N]

        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attention.unsqueeze(1), encoder_out).squeeze(1)  # [B, enc_dim]

        return context, attention


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decoder Prenet
# ─────────────────────────────────────────────────────────────────────────────

class Prenet(nn.Module):
    """
    Tacotron2 decoder prenet:
      2 × (Linear → ReLU → Dropout[0.5])
    Applied to the previous mel frame before feeding to decoder LSTM.
    Dropout is always active (even at inference) to help prevent
    attention collapse.
    """

    def __init__(self, in_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_mels] — previous mel frame

        Returns:
            [B, hidden_dim]
        """
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=True)  # always on!
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=True)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. PostNet
# ─────────────────────────────────────────────────────────────────────────────

class PostNet(nn.Module):
    """
    Tacotron2 PostNet: 5×Conv1d that refines the mel prediction.
    Operates on the full mel sequence to add fine details.
    """

    def __init__(
        self,
        n_mels:      int   = 80,
        n_layers:    int   = 5,
        n_channels:  int   = 512,
        kernel_size: int   = 5,
        dropout:     float = 0.5,
    ):
        super().__init__()
        layers = []
        padding = (kernel_size - 1) // 2

        for i in range(n_layers):
            in_ch  = n_mels    if i == 0          else n_channels
            out_ch = n_mels    if i == n_layers-1 else n_channels
            act    = nn.Tanh() if i < n_layers-1  else nn.Identity()

            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                act,
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T]

        Returns:
            residual: [B, n_mels, T] — added to raw mel prediction
        """
        return self.net(mel)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Tacotron2 Model
# ─────────────────────────────────────────────────────────────────────────────

class Tacotron2(nn.Module):
    """
    Complete Tacotron2 model for Amharic Text-to-Speech.

    Encoder maps character sequences → encoder states.
    Decoder autoregressively generates mel frames one at a time,
    attending to encoder states via location-sensitive attention.
    PostNet refines the complete mel sequence.
    """

    def __init__(
        self,
        vocab_size:       int   = VOCAB_SIZE,
        embed_dim:        int   = 512,
        encoder_dim:      int   = 512,
        decoder_dim:      int   = 1024,
        attention_dim:    int   = 128,
        prenet_dim:       int   = 256,
        n_mels:           int   = 80,
        max_decoder_steps:int   = 2000,
        gate_threshold:   float = 0.5,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.n_mels            = n_mels
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold    = gate_threshold
        self.encoder_dim       = encoder_dim
        self.decoder_dim       = decoder_dim

        # Encoder
        self.encoder = Tacotron2Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Prenet
        self.prenet = Prenet(in_dim=n_mels, hidden_dim=prenet_dim)

        # Attention
        self.attention = LocationSensitiveAttention(
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )

        # Decoder LSTM (2 layers)
        # Input: prenet_out + context_vector
        self.decoder_lstm1 = nn.LSTMCell(prenet_dim + encoder_dim, decoder_dim)
        self.decoder_lstm2 = nn.LSTMCell(decoder_dim, decoder_dim)

        # Mel projection: decoder_dim + context → n_mels
        self.mel_linear = nn.Linear(decoder_dim + encoder_dim, n_mels)

        # Stop token gate
        self.gate_linear = nn.Linear(decoder_dim + encoder_dim, 1)

        # PostNet
        self.postnet = PostNet(n_mels=n_mels)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        tokens:      torch.Tensor,
        token_lens:  torch.Tensor,
        mel_targets: torch.Tensor,
        mel_lens:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass (training mode).

        Args:
            tokens:      [B, N] character IDs
            token_lens:  [B] encoder input lengths
            mel_targets: [B, n_mels, T] target mel spectrogram
            mel_lens:    [B] target mel lengths

        Returns:
            mel_out:     [B, n_mels, T] raw decoder mel output
            mel_refined: [B, n_mels, T] PostNet-refined mel
            gate_out:    [B, T] stop token logits
        """
        B, _, T = mel_targets.shape

        # 1. Encode text
        encoder_out = self.encoder(tokens, token_lens)   # [B, N, 512]
        N = encoder_out.shape[1]

        # 2. Encoder mask (True = pad)
        enc_mask = (
            torch.arange(N, device=tokens.device).unsqueeze(0) >= token_lens.unsqueeze(1)
        )

        # 3. Initialize decoder state
        h1 = torch.zeros(B, self.decoder_dim, device=tokens.device)
        c1 = torch.zeros(B, self.decoder_dim, device=tokens.device)
        h2 = torch.zeros(B, self.decoder_dim, device=tokens.device)
        c2 = torch.zeros(B, self.decoder_dim, device=tokens.device)

        prev_mel      = torch.zeros(B, self.n_mels, device=tokens.device)
        prev_attention = torch.zeros(B, N, device=tokens.device)
        context        = torch.zeros(B, self.encoder_dim, device=tokens.device)

        mel_outputs  = []
        gate_outputs = []

        # 4. Autoregressive decoding with teacher forcing
        # Teacher forcing: feed ground-truth mel frame (shifted by 1)
        mel_inputs = torch.cat([
            torch.zeros(B, self.n_mels, 1, device=tokens.device),
            mel_targets[:, :, :-1]
        ], dim=2)  # [B, n_mels, T]

        for t in range(T):
            prenet_out = self.prenet(mel_inputs[:, :, t])   # [B, prenet_dim]
            lstm1_input = torch.cat([prenet_out, context], dim=-1)  # [B, prenet+enc]

            h1, c1 = self.decoder_lstm1(lstm1_input, (h1, c1))
            h2, c2 = self.decoder_lstm2(h1, (h2, c2))

            context, prev_attention = self.attention(
                query=h2,
                encoder_out=encoder_out,
                prev_attention=prev_attention,
                mask=enc_mask,
            )

            decoder_ctx = torch.cat([h2, context], dim=-1)   # [B, 1024+512]
            mel_frame   = self.mel_linear(decoder_ctx)        # [B, n_mels]
            gate_frame  = self.gate_linear(decoder_ctx)       # [B, 1]

            mel_outputs.append(mel_frame)
            gate_outputs.append(gate_frame)

        mel_out  = torch.stack(mel_outputs,  dim=-1)  # [B, n_mels, T]
        gate_out = torch.cat(gate_outputs, dim=-1)    # [B, T]

        # 5. PostNet refinement
        mel_refined = mel_out + self.postnet(mel_out)   # [B, n_mels, T]

        return mel_out, mel_refined, gate_out

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, token_lens: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive inference (no teacher forcing).

        Args:
            tokens:     [B, N] character IDs
            token_lens: [B] lengths

        Returns:
            mel_refined: [B, n_mels, T] generated mel spectrogram
        """
        B = tokens.shape[0]
        encoder_out = self.encoder(tokens, token_lens)
        N = encoder_out.shape[1]

        enc_mask = (
            torch.arange(N, device=tokens.device).unsqueeze(0) >= token_lens.unsqueeze(1)
        )

        h1 = torch.zeros(B, self.decoder_dim, device=tokens.device)
        c1 = torch.zeros(B, self.decoder_dim, device=tokens.device)
        h2 = torch.zeros(B, self.decoder_dim, device=tokens.device)
        c2 = torch.zeros(B, self.decoder_dim, device=tokens.device)

        prev_mel       = torch.zeros(B, self.n_mels, device=tokens.device)
        prev_attention = torch.zeros(B, N, device=tokens.device)
        context        = torch.zeros(B, self.encoder_dim, device=tokens.device)

        mel_outputs = []

        for step in range(self.max_decoder_steps):
            prenet_out  = self.prenet(prev_mel)
            lstm1_input = torch.cat([prenet_out, context], dim=-1)

            h1, c1 = self.decoder_lstm1(lstm1_input, (h1, c1))
            h2, c2 = self.decoder_lstm2(h1, (h2, c2))

            context, prev_attention = self.attention(
                query=h2,
                encoder_out=encoder_out,
                prev_attention=prev_attention,
                mask=enc_mask,
            )

            decoder_ctx = torch.cat([h2, context], dim=-1)
            mel_frame   = self.mel_linear(decoder_ctx)           # [B, n_mels]
            gate_logit  = self.gate_linear(decoder_ctx).squeeze(-1)  # [B]

            mel_outputs.append(mel_frame)
            prev_mel = mel_frame   # autoregressive: use own output

            # Stop when gate fires for all samples in batch
            if torch.sigmoid(gate_logit).min() > self.gate_threshold:
                break

        mel_out = torch.stack(mel_outputs, dim=-1)    # [B, n_mels, T]
        mel_refined = mel_out + self.postnet(mel_out)
        return mel_refined

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = Tacotron2()
    print(f"Tacotron2 parameters: {model.count_parameters():,}")

    B, N, T = 2, 20, 100
    tokens      = torch.randint(1, 100, (B, N))
    token_lens  = torch.tensor([N, N - 5])
    mel_targets = torch.randn(B, 80, T)
    mel_lens    = torch.tensor([T, T - 10])

    mel_out, mel_refined, gate_out = model(tokens, token_lens, mel_targets, mel_lens)
    print(f"mel_out shape:     {mel_out.shape}")
    print(f"mel_refined shape: {mel_refined.shape}")
    print(f"gate_out shape:    {gate_out.shape}")
    print("✅ Tacotron2 forward pass OK")
