"""
metrics.py — Evaluation Metrics for ASR and TTS
================================================
Implements:
  - Word Error Rate (WER) for ASR evaluation
  - Character Error Rate (CER) for ASR evaluation
  - Mel Cepstral Distortion (MCD) for TTS evaluation
  - Training progress tracker with TensorBoard logging
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.amharic_g2p import tokens_to_text, AMHARIC_VOCAB_INV


# ─────────────────────────────────────────────────────────────────────────────
# 1. Word Error Rate (WER)
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(ref: list, hyp: list) -> int:
    """
    Compute Levenshtein edit distance between two sequences.

    Args:
        ref: Reference sequence
        hyp: Hypothesis sequence

    Returns:
        Minimum edit distance (insertions + deletions + substitutions)
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1]  # substitution
                )
    return dp[m][n]


def word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute batch Word Error Rate (WER).

    WER = (substitutions + deletions + insertions) / total_reference_words

    Args:
        references:  List of reference Amharic transcripts
        hypotheses:  List of model-predicted transcripts

    Returns:
        WER as a float in [0, 1] (lower is better)
    """
    total_words  = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_words  += max(len(ref_words), 1)
        total_errors += _edit_distance(ref_words, hyp_words)

    return total_errors / max(total_words, 1)


def character_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute batch Character Error Rate (CER).
    More appropriate for morphologically rich languages like Amharic.

    Args:
        references:  List of reference strings
        hypotheses:  List of hypothesis strings

    Returns:
        CER as a float in [0, 1]
    """
    total_chars  = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        total_chars  += max(len(ref_chars), 1)
        total_errors += _edit_distance(ref_chars, hyp_chars)

    return total_errors / max(total_chars, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mel Cepstral Distortion (MCD)
# ─────────────────────────────────────────────────────────────────────────────

def mel_cepstral_distortion(
    mel_true: torch.Tensor,
    mel_pred: torch.Tensor,
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD) between reference and predicted mels.

    MCD measures the spectral difference between synthesized and natural speech.
    Lower is better. Typical good TTS: MCD < 8 dB.

    Args:
        mel_true: [n_mels, T] ground-truth Mel spectrogram
        mel_pred: [n_mels, T] predicted Mel spectrogram (same length)

    Returns:
        MCD value in dB
    """
    # Align lengths (truncate to shorter)
    T = min(mel_true.shape[-1], mel_pred.shape[-1])
    mel_true = mel_true[:, :T].detach().cpu().numpy()
    mel_pred = mel_pred[:, :T].detach().cpu().numpy()

    # Convert log-mel to linear, then compute MCD
    # MCD = (10 / ln(10)) * sqrt(2 * sum((mc_true - mc_pred)^2))
    diff = mel_true - mel_pred
    mcd  = (10.0 / math.log(10.0)) * math.sqrt(2.0) * np.sqrt(
        np.mean(np.sum(diff ** 2, axis=0))
    )
    return float(mcd)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ASR Decode Helper
# ─────────────────────────────────────────────────────────────────────────────

def decode_predictions(
    log_probs: torch.Tensor,
    blank_id: int = 0,
) -> List[str]:
    """
    CTC greedy decode → list of Amharic text strings.

    Args:
        log_probs: [T, B, vocab_size]
        blank_id:  CTC blank token index

    Returns:
        List of decoded Amharic strings
    """
    best = log_probs.argmax(dim=-1).transpose(0, 1)   # [B, T]
    texts = []
    for b in range(best.shape[0]):
        seq = best[b].tolist()
        # Collapse repeats
        collapsed = [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
        # Remove blank
        decoded_ids = [t for t in collapsed if t != blank_id]
        text = tokens_to_text(decoded_ids)
        texts.append(text)
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training Progress Tracker
# ─────────────────────────────────────────────────────────────────────────────

class TrainingTracker:
    """
    Tracks training metrics and logs to TensorBoard.

    Usage:
        tracker = TrainingTracker(log_dir="runs/asr_experiment")
        tracker.log_scalar("loss/train", 1.23, step=100)
        tracker.log_scalars({"wer": 0.45, "cer": 0.30}, step=200)
        tracker.save_checkpoint(model, optimizer, step, "checkpoints/asr_best.pt")
    """

    def __init__(self, log_dir: str, experiment_name: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.step = 0

        # History for in-memory tracking
        self.history: Dict[str, List[Tuple[int, float]]] = {}

        # TensorBoard writer (optional)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"[TensorBoard] Logging to: {self.log_dir}")
            print(f"  Run: tensorboard --logdir={self.log_dir.parent}")
        except ImportError:
            print("[TrainingTracker] TensorBoard not available. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if tag not in self.history:
            self.history[tag] = []
        self.history[tag].append((step, value))

        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log multiple scalars at once."""
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)

    def log_audio(self, tag: str, audio: torch.Tensor, sample_rate: int, step: int):
        """Log audio sample to TensorBoard."""
        if self.writer:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        path: str,
        extra: Optional[dict] = None,
    ):
        """
        Save model checkpoint to disk.

        Args:
            model:     PyTorch model to save
            optimizer: Optimizer state
            step:      Current training step
            path:      Checkpoint file path
            extra:     Additional metadata to save (e.g., WER, config)
        """
        ckpt = {
            "step":           step,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "history":        self.history,
        }
        if extra:
            ckpt.update(extra)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(path))
        print(f"[Checkpoint] Saved → {path} (step={step})")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: str,
    ) -> int:
        """
        Load checkpoint from disk.

        Returns:
            step: The training step at which checkpoint was saved
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        step = ckpt.get("step", 0)
        print(f"[Checkpoint] Loaded from {path} (step={step})")
        return step

    def get_best_metric(self, tag: str, mode: str = "min") -> Tuple[int, float]:
        """
        Get the step and value of the best recorded metric.

        Args:
            tag:  Metric name
            mode: 'min' for loss/WER, 'max' for accuracy

        Returns:
            (best_step, best_value)
        """
        if tag not in self.history or not self.history[tag]:
            return 0, float("inf") if mode == "min" else 0.0

        steps_vals = self.history[tag]
        if mode == "min":
            return min(steps_vals, key=lambda sv: sv[1])
        else:
            return max(steps_vals, key=lambda sv: sv[1])

    def print_summary(self):
        """Print a summary of tracked metrics."""
        print("\n" + "=" * 50)
        print(f"Training Summary — {self.experiment_name}")
        print("=" * 50)
        for tag, history in self.history.items():
            if history:
                best_step, best_val = self.get_best_metric(tag)
                last_step, last_val = history[-1]
                print(f"  {tag:30s} | last: {last_val:.4f} (step {last_step}) | best: {best_val:.4f} (step {best_step})")

    def close(self):
        if self.writer:
            self.writer.close()
