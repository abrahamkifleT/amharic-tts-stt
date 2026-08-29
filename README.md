# Amharic Text-to-Speech & Speech-to-Text (From Scratch)

A research-grade, fully open-source pipeline for training and serving **Amharic (አማርኛ)** speech AI models — built from scratch using PyTorch, no external APIs.

---

## Features

- 🧠 **ASR (Speech → Text):** Conformer + CTC decoder trained on Amharic audio
- 🔊 **TTS (Text → Speech):** Tacotron2 acoustic model + HiFi-GAN vocoder
- 📝 **Amharic G2P:** Custom Grapheme-to-Phoneme for all Ethiopic Unicode blocks
- 🎙️ **Recording Studio:** Built-in web UI to record & contribute your own voice data
- 📊 **Training Dashboard:** Live loss/WER visualizations in the browser
- 🚀 **Colab Notebooks:** Ready-to-run GPU training notebooks (free GPU)

---

## Project Structure

```
├── data/                    # Datasets & processed data
│   ├── raw/                 # Downloaded raw speech datasets
│   ├── processed/           # Preprocessed features (mel, tokens)
│   └── recordings/          # Your own recorded clips
├── src/
│   ├── data/                # G2P, preprocessing, PyTorch datasets
│   ├── models/
│   │   ├── asr/             # Conformer-CTC ASR model
│   │   └── tts/             # Tacotron2 + HiFi-GAN TTS model
│   ├── training/            # Training loops & metrics
│   └── inference/           # Inference scripts
├── notebooks/               # Google Colab training notebooks
├── checkpoints/             # Saved model weights
├── app/                     # Flask web inference app
│   ├── server.py
│   ├── templates/
│   └── static/
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Download Datasets
```bash
python data/download_datasets.py --datasets fleurs common_voice alffa
```

### 3. Preprocess Data
```bash
# For ASR
python src/data/preprocess_asr.py --input data/raw --output data/processed/asr

# For TTS
python src/data/preprocess_tts.py --input data/raw --output data/processed/tts
```

### 4. Train Models (Recommended: Google Colab)
Open the notebooks in Google Colab for free GPU training:
- `notebooks/02_train_asr_colab.ipynb` — Train ASR (Conformer-CTC)
- `notebooks/03_train_tts_colab.ipynb` — Train TTS (Tacotron2 + HiFi-GAN)

Then download the `.pt` checkpoint files to `checkpoints/`.

### 5. Run the Web App
```bash
python app/server.py
# Open: http://localhost:5000
```

---

## Datasets Used

| Dataset | Source | Size | License |
|---|---|---|---|
| Mozilla Common Voice (am) | [commonvoice.mozilla.org](https://commonvoice.mozilla.org) | ~10K clips | CC-0 |
| FLEURS Amharic | [HuggingFace](https://huggingface.co/datasets/google/fleurs) | ~3K clips | CC-BY-4.0 |
| OpenSLR ALFFA (SLR25) | [openslr.org/25](https://openslr.org/25) | ~5 hours | Apache-2.0 |

---

## Model Architectures

### ASR — Conformer-CTC
```
Audio (16kHz) → Log-Mel (80 bands) → Conv Subsampling
→ Conformer Blocks ×6 → Linear → CTC Softmax → Amharic characters
```

### TTS — Tacotron2 + HiFi-GAN
```
Amharic text → Char Embeddings → Encoder (Conv+BiLSTM)
→ Location-Sensitive Attention → Decoder (LSTM+Prenet)
→ Mel Spectrogram → PostNet → HiFi-GAN → Audio (22050Hz)
```

---

## Evaluation Metrics

| Task | Metric | Target |
|---|---|---|
| ASR | Word Error Rate (WER) | < 30% |
| TTS | Mel Cepstral Distortion (MCD) | < 8 dB |
| TTS | Naturalness | Intelligible |

---

## Requirements

- Python 3.9+
- PyTorch 2.1+
- 8GB+ RAM (for local inference)
- GPU recommended for training (Google Colab works great)

---

## License

MIT License — See [LICENSE](LICENSE)
