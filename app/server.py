"""
server.py — Flask Web Inference & Recording Studio Server
==========================================================
Provides web interface and REST API for:
  1. Speech-to-Text (STT) inference
  2. Text-to-Speech (TTS) inference
  3. Recording Studio for local dataset augmentation
  4. Model & system status
"""

import os
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import uuid
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Add root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.inference.asr_infer import transcribe_audio
from src.inference.tts_infer import synthesize_speech
from src.data.amharic_g2p import normalize_text, is_amharic

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_DIR = ROOT_DIR / "data" / "recordings"
OUTPUT_AUDIO_DIR = ROOT_DIR / "app" / "static" / "audio_out"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return model checkpoint availability."""
    asr_ckpt = (ROOT_DIR / "checkpoints" / "asr_best.pt").exists()
    tts_ckpt = (ROOT_DIR / "checkpoints" / "tts_best.pt").exists()
    voc_ckpt = (ROOT_DIR / "checkpoints" / "hifigan_best.pt").exists()
    
    return jsonify({
        "status": "online",
        "asr_checkpoint": asr_ckpt,
        "tts_checkpoint": tts_ckpt,
        "vocoder_checkpoint": voc_ckpt,
        "mode": "Trained Models" if (asr_ckpt and tts_ckpt) else "Model Architecture / Demo Mode"
    })


@app.route("/api/stt", methods=["POST"])
def run_stt():
    """Speech-to-Text inference endpoint."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex[:8]}.wav"
    audio_file.save(str(temp_path))

    try:
        start_t = time.time()
        transcript = transcribe_audio(str(temp_path))
        latency = round(time.time() - start_t, 3)
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

        return jsonify({
            "success": True,
            "transcript": transcript,
            "latency_sec": latency
        })
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts", methods=["POST"])
def run_tts():
    """Text-to-Speech inference endpoint."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    out_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
    out_path = OUTPUT_AUDIO_DIR / out_filename

    try:
        start_t = time.time()
        synthesize_speech(text, output_path=str(out_path))
        latency = round(time.time() - start_t, 3)

        return jsonify({
            "success": True,
            "audio_url": f"/static/audio_out/{out_filename}",
            "latency_sec": latency,
            "normalized_text": normalize_text(text)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/record", methods=["POST"])
def save_recording():
    """Save user recordings for custom Amharic dataset augmentation."""
    if "audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing audio or text parameter"}), 400

    audio_file = request.files["audio"]
    text = request.form["text"].strip()
    rec_id = f"custom_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    wav_path = UPLOAD_DIR / f"{rec_id}.wav"
    audio_file.save(str(wav_path))

    # Append to metadata CSV
    meta_csv = UPLOAD_DIR / "metadata.csv"
    is_new = not meta_csv.exists()
    with open(meta_csv, "a", encoding="utf-8") as f:
        if is_new:
            f.write("id,file,text\n")
        f.write(f'{rec_id},"{wav_path}","{text}"\n')

    return jsonify({
        "success": True,
        "id": rec_id,
        "message": "Recording saved successfully to data/recordings/"
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Amharic Speech AI Studio Web Server")
    print("Server starting at http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
