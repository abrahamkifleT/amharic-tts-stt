"""
tts_infer.py — Run Amharic Text-to-Speech Inference
===================================================
Synthesizes Amharic text into audio waveform using Tacotron2 + HiFi-GAN.

Usage:
    python src/inference/tts_infer.py --text "ሰላም ዓለም" --output output.wav
"""

import sys
import argparse
from pathlib import Path
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.tts.tacotron2 import Tacotron2
from src.models.tts.hifigan import HiFiGANGenerator, HIFIGAN_CONFIG
from src.data.amharic_g2p import normalize_text, text_to_tokens


def synthesize_speech(
    text: str,
    output_path: str = "output.wav",
    tts_checkpoint: str = "checkpoints/tts_best.pt",
    vocoder_checkpoint: str = "checkpoints/hifigan_best.pt",
    device: str = "cpu",
) -> str:
    """
    Synthesize Amharic text into speech audio.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # 1. Normalize and Tokenize text
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Text is empty after normalization.")

    tokens = text_to_tokens(normalized_text, add_bos=False, add_eos=True)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=dev)
    token_lens = torch.tensor([len(tokens)], dtype=torch.long, device=dev)

    # 2. Load Tacotron2
    tacotron = Tacotron2().to(dev)
    tts_ckpt = Path(tts_checkpoint)
    if tts_ckpt.exists():
        ckpt = torch.load(str(tts_ckpt), map_location=dev, weights_only=False)
        tacotron.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"[TTS] Loaded Tacotron2 from {tts_ckpt}")
    else:
        print(f"[TTS Warning] Tacotron2 checkpoint not found at {tts_checkpoint}. Using initialized weights.")

    tacotron.eval()

    # 3. Load HiFi-GAN Vocoder
    vocoder = HiFiGANGenerator(HIFIGAN_CONFIG).to(dev)
    voc_ckpt = Path(vocoder_checkpoint)
    if voc_ckpt.exists():
        ckpt = torch.load(str(voc_ckpt), map_location=dev, weights_only=False)
        vocoder.load_state_dict(ckpt.get("G_state", ckpt.get("model_state", ckpt)))
        print(f"[Vocoder] Loaded HiFi-GAN from {voc_ckpt}")
    else:
        print(f"[Vocoder Warning] HiFi-GAN checkpoint not found at {vocoder_checkpoint}. Using initialized weights.")

    vocoder.eval()

    # 4. Generate Mel Spectrogram & Waveform
    with torch.no_grad():
        mel_refined = tacotron.infer(tokens_tensor, token_lens)  # [1, n_mels, T]
        audio_tensor = vocoder(mel_refined).squeeze(0)          # [1, T_audio]

    # 5. Save WAV
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_p), audio_tensor.cpu(), 22050)
    print(f"[TTS] Audio saved to: {out_p}")
    return str(out_p)


def main():
    parser = argparse.ArgumentParser(description="Amharic Text-to-Speech Inference")
    parser.add_argument("--text", type=str, required=True, help="Amharic text to speak")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--tts_checkpoint", type=str, default="checkpoints/tts_best.pt")
    parser.add_argument("--vocoder_checkpoint", type=str, default="checkpoints/hifigan_best.pt")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    synthesize_speech(
        text=args.text,
        output_path=args.output,
        tts_checkpoint=args.tts_checkpoint,
        vocoder_checkpoint=args.vocoder_checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
