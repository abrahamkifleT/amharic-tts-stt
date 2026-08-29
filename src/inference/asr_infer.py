"""
asr_infer.py — Run Amharic Speech Recognition Inference
========================================================
Transcribes an Amharic audio file into Ethiopic text.

Usage:
    python src/inference/asr_infer.py --audio path/to/audio.wav --checkpoint checkpoints/asr_best.pt
"""

import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.asr.asr_model import AmharicASRModel
from src.data.preprocess_asr import load_and_resample, normalize_volume, compute_log_mel, ASR_CONFIG
from src.training.metrics import decode_predictions


def transcribe_audio(
    audio_path: str,
    checkpoint_path: str = "checkpoints/asr_best.pt",
    device: str = "cpu",
) -> str:
    """
    Transcribe an audio file into Amharic text.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # Initialize model
    model = AmharicASRModel(
        n_mels=ASR_CONFIG["n_mels"],
        d_model=256,
        n_heads=4,
        n_layers=6,
    ).to(dev)

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"[ASR] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[ASR Warning] Checkpoint {checkpoint_path} not found. Running with initialized weights (demo/untrained).")

    model.eval()

    # Audio preprocessing
    waveform = load_and_resample(audio_path, target_sr=ASR_CONFIG["sample_rate"])
    waveform = normalize_volume(waveform, target_rms=ASR_CONFIG["target_rms"])
    log_mel = compute_log_mel(
        waveform,
        sample_rate=ASR_CONFIG["sample_rate"],
        n_mels=ASR_CONFIG["n_mels"],
        n_fft=ASR_CONFIG["n_fft"],
        hop_length=ASR_CONFIG["hop_length"],
        win_length=ASR_CONFIG["win_length"],
    )  # [n_mels, T]

    log_mel_batch = log_mel.unsqueeze(0).to(dev)  # [1, n_mels, T]
    mel_len = torch.tensor([log_mel.shape[1]], dtype=torch.long, device=dev)

    with torch.no_grad():
        log_probs, out_lens = model(log_mel_batch, mel_len)
        texts = decode_predictions(log_probs)

    transcription = texts[0] if texts else ""
    return transcription


def main():
    parser = argparse.ArgumentParser(description="Amharic Speech-to-Text Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/asr_best.pt", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    result = transcribe_audio(args.audio, args.checkpoint, args.device)
    print("\n" + "=" * 40)
    print("Transcription (አማርኛ):")
    print(result if result else "[No speech recognized / Untrained model]")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
