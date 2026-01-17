import os
import re
import requests
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from together import Together
import base64
import subprocess
import uuid
import time
import math
from PIL import Image

# -------------------------------------------------
# ENV + APP SETUP
# -------------------------------------------------

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "temporary-secret")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# AZURE + TOGETHER CONFIG (UNCHANGED)
# -------------------------------------------------

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")

client = Together()

# -------------------------------------------------
# BASIC PAGE ROUTES (PHASE 1 – VISIBILITY ONLY)
# -------------------------------------------------

@app.route("/")
@app.route("/home")
def home():
    """
    Home / landing page
    """
    return render_template("home.html")


@app.route("/input")
def input_page():
    return render_template("input.html")


@app.route("/process-text", methods=["POST"])
def process_text():
    # later: receive text, store in session
    return redirect(url_for("result_page"))


@app.route("/process-audio", methods=["POST"])
def process_audio():
    # later: handle file upload + STT
    return redirect(url_for("result_page"))


@app.route("/process-mic", methods=["POST"])
def process_mic():
    # later: handle base64 mic audio
    return redirect(url_for("result_page"))



@app.route("/result")
def result_page():
    """
    Result page (dummy content for now)
    """
    return render_template("result.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Temporary bridge route.
    Later this will:
    - accept text/audio
    - call STT
    - call AI
    For now: just redirect to result page.
    """
    return redirect(url_for("result_page"))


# -------------------------------------------------
# EXISTING TRANSCRIPTION ROUTE (UNCHANGED)
# -------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe_audio_base64():
    try:
        data = request.get_json(force=True)
        print("📥 Raw incoming JSON:", data)

        audio_base64 = data.get("audio")
        language_code = data.get("language", "en-IN")

        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400

        # Step 1: Decode & save raw audio
        audio_bytes = base64.b64decode(audio_base64)
        raw_audio_path = f"{UPLOAD_FOLDER}/raw_{uuid.uuid4().hex}.webm"
        with open(raw_audio_path, "wb") as f:
            f.write(audio_bytes)

        # Step 2: Convert to WAV
        wav_audio_path = raw_audio_path.replace(".webm", ".wav")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", raw_audio_path,
            "-ac", "1", "-ar", "16000", wav_audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Step 3: Azure Speech-to-Text
        stt_url = f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "audio/wav",
            "Accept": "application/json"
        }
        params = {"language": language_code}

        with open(wav_audio_path, "rb") as audio_file:
            stt_response = requests.post(
                stt_url, headers=headers, params=params, data=audio_file
            )

        stt_data = stt_response.json()
        original_text = stt_data.get("DisplayText", "")

        if not original_text:
            return jsonify({"error": "Speech recognition returned empty text"}), 500

        # Step 4: Translation (if needed)
        if language_code.startswith("en"):
            translated_text = original_text
        else:
            trans_url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&to=en"
            trans_headers = {
                "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
                "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
                "Content-Type": "application/json"
            }
            trans_body = [{"Text": original_text}]
            trans_response = requests.post(
                trans_url, headers=trans_headers, json=trans_body
            )
            trans_data = trans_response.json()
            translated_text = trans_data[0]["translations"][0]["text"]

        return jsonify({
            "original_text": original_text,
            "translated_text": translated_text,
            "language_code": language_code
        })

    except subprocess.CalledProcessError:
        return jsonify({"error": "Audio conversion failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# HEALTH CHECK (OPTIONAL, SAFE)
# -------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# -------------------------------------------------
# LOCAL ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
