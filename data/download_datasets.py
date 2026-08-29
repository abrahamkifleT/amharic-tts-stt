"""
download_datasets.py — Download Amharic Speech Datasets
========================================================
Downloads and organizes three open Amharic speech datasets:
  1. FLEURS Amharic  (via HuggingFace datasets)
  2. Mozilla Common Voice Amharic  (via HuggingFace datasets)
  3. OpenSLR ALFFA SLR25  (direct HTTP download)

Usage:
    python data/download_datasets.py --datasets fleurs common_voice alffa
    python data/download_datasets.py --datasets fleurs   # single dataset
"""

import os
import sys
import argparse
import tarfile
import zipfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# 1. FLEURS Amharic (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def download_fleurs(output_dir: Path):
    """
    Download the FLEURS Amharic split from HuggingFace.
    Saves audio as WAV + metadata CSV.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] Install the 'datasets' library: pip install datasets")
        return

    out = output_dir / "fleurs"
    out.mkdir(parents=True, exist_ok=True)

    print("\n[FLEURS] Downloading Amharic FLEURS dataset (am_et)...")
    ds = load_dataset("google/fleurs", "am_et", trust_remote_code=True)

    import soundfile as sf
    import pandas as pd

    for split in ["train", "validation", "test"]:
        split_dir = out / split
        split_dir.mkdir(exist_ok=True)
        records = []

        print(f"  Processing FLEURS {split} split ({len(ds[split])} examples)...")
        for i, example in enumerate(tqdm(ds[split], desc=f"  FLEURS {split}")):
            # Save audio
            audio_array = example["audio"]["array"]
            sample_rate = example["audio"]["sampling_rate"]
            fname = f"fleurs_{split}_{i:05d}.wav"
            fpath = split_dir / fname
            sf.write(str(fpath), audio_array, sample_rate)

            records.append({
                "file":      str(fpath.relative_to(out)),
                "text":      example["transcription"],
                "duration":  len(audio_array) / sample_rate,
                "split":     split,
            })

        # Save metadata
        pd.DataFrame(records).to_csv(out / f"{split}_metadata.csv", index=False)
        print(f"  Saved {len(records)} clips to {split_dir}")

    print(f"[FLEURS] Done → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mozilla Common Voice Amharic (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def download_common_voice(output_dir: Path, version: str = "cv-corpus-15.0-2023-09-08"):
    """
    Download Mozilla Common Voice Amharic via HuggingFace datasets.
    Note: Mozilla Common Voice requires accepting the license on HuggingFace Hub.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] Install the 'datasets' library: pip install datasets")
        return

    out = output_dir / "common_voice"
    out.mkdir(parents=True, exist_ok=True)

    print("\n[CommonVoice] Downloading Mozilla Common Voice Amharic...")
    print("  Note: You may need to be logged in to HuggingFace Hub:")
    print("  Run: huggingface-cli login")

    try:
        import soundfile as sf
        import pandas as pd

        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "am",
            trust_remote_code=True,
        )

        for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            split_dir = out / split_name
            split_dir.mkdir(exist_ok=True)
            records = []

            if hf_split not in ds:
                print(f"  Split '{hf_split}' not found, skipping.")
                continue

            split_data = ds[hf_split]
            print(f"  Processing CommonVoice {split_name} ({len(split_data)} examples)...")

            for i, example in enumerate(tqdm(split_data, desc=f"  CV {split_name}")):
                audio_array = example["audio"]["array"]
                sample_rate = example["audio"]["sampling_rate"]
                fname = f"cv_{split_name}_{i:05d}.wav"
                fpath = split_dir / fname
                sf.write(str(fpath), audio_array, sample_rate)

                records.append({
                    "file":     str(fpath.relative_to(out)),
                    "text":     example["sentence"],
                    "duration": len(audio_array) / sample_rate,
                    "split":    split_name,
                })

            pd.DataFrame(records).to_csv(out / f"{split_name}_metadata.csv", index=False)
            print(f"  Saved {len(records)} clips.")

    except Exception as e:
        print(f"[CommonVoice] Error: {e}")
        print("  Tip: Try: huggingface-cli login  (requires HuggingFace account)")

    print(f"[CommonVoice] Done → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenSLR ALFFA SLR25 (Direct download)
# ─────────────────────────────────────────────────────────────────────────────

ALFFA_URL = "https://www.openslr.org/resources/25/amharic.zip"


def _download_file(url: str, dest: Path):
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"  {dest.name}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_alffa(output_dir: Path):
    """
    Download OpenSLR ALFFA Amharic dataset (SLR25).
    Extracts WAV files and converts Kaldi-format transcripts to CSV.
    """
    out = output_dir / "alffa"
    out.mkdir(parents=True, exist_ok=True)

    zip_path = out / "amharic.zip"

    print("\n[ALFFA] Downloading OpenSLR ALFFA Amharic (SLR25)...")
    if not zip_path.exists():
        try:
            _download_file(ALFFA_URL, zip_path)
        except Exception as e:
            print(f"[ALFFA] Download failed: {e}")
            print(f"  Manual download: {ALFFA_URL}")
            print(f"  Place the zip at: {zip_path}")
            return
    else:
        print(f"  Found existing zip: {zip_path}")

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)

    # Convert Kaldi text file to CSV
    _convert_alffa_to_csv(out)

    print(f"[ALFFA] Done → {out}")


def _convert_alffa_to_csv(alffa_dir: Path):
    """Convert Kaldi-format text transcript to a simple CSV."""
    import pandas as pd

    text_file = alffa_dir / "amharic" / "data" / "train" / "text"
    wav_dir   = alffa_dir / "amharic" / "data" / "train" / "wav"

    if not text_file.exists():
        print(f"  [ALFFA] Could not find text file at {text_file}. Skipping CSV conversion.")
        return

    records = []
    with open(text_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, transcript = parts
                wav_path = wav_dir / f"{utt_id}.wav"
                if wav_path.exists():
                    records.append({
                        "file":  str(wav_path.relative_to(alffa_dir)),
                        "text":  transcript,
                        "split": "train",
                    })

    if records:
        out_csv = alffa_dir / "train_metadata.csv"
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"  Converted {len(records)} ALFFA utterances → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge All Datasets into Unified Format
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets(data_dir: Path):
    """
    Merge all downloaded datasets into a single combined CSV.
    Output: data/raw/all_metadata.csv
    """
    import pandas as pd

    all_records = []
    for ds_name in ["fleurs", "common_voice", "alffa"]:
        ds_dir = data_dir / ds_name
        for csv_file in ds_dir.glob("*_metadata.csv"):
            df = pd.read_csv(csv_file)
            df["dataset"] = ds_name
            # Make file paths absolute
            df["file"] = df["file"].apply(lambda p: str(ds_dir / p))
            all_records.append(df)

    if not all_records:
        print("[Merge] No datasets found to merge.")
        return

    combined = pd.concat(all_records, ignore_index=True)
    out_path  = data_dir / "all_metadata.csv"
    combined.to_csv(out_path, index=False)

    print(f"\n[Merge] Combined dataset: {len(combined)} total utterances")
    print(combined.groupby("dataset")["file"].count().to_string())
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Amharic speech datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["fleurs", "common_voice", "alffa", "all"],
        default=["fleurs"],
        help="Which datasets to download (default: fleurs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all downloaded datasets into a single CSV after download",
    )
    args = parser.parse_args()

    if "all" in args.datasets:
        args.datasets = ["fleurs", "common_voice", "alffa"]

    args.output.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        if ds == "fleurs":
            download_fleurs(args.output)
        elif ds == "common_voice":
            download_common_voice(args.output)
        elif ds == "alffa":
            download_alffa(args.output)

    if args.merge:
        merge_datasets(args.output)

    print("\n✅ Download complete!")
    print(f"   Data saved to: {args.output}")
    print("\nNext steps:")
    print("  python src/data/preprocess_asr.py --input data/raw --output data/processed/asr")
    print("  python src/data/preprocess_tts.py --input data/raw --output data/processed/tts")


if __name__ == "__main__":
    main()
