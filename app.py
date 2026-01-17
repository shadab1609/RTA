import os
import re
import requests
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from together import Together
from flask import jsonify
import base64
import subprocess
import uuid
import time
import math
from PIL import Image
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "temporary-secret")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")

client = Together()

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_base64():
    try:
        data = request.get_json(force=True)
        print("📥 Raw incoming JSON:", data)

        audio_base64 = data.get("audio")
        language_code = data.get("language", "en-IN")

        if not audio_base64:
            print("❗ No audio data found in request.")
            return jsonify({"error": "No audio data provided"}), 400

        # Step 1: Decode & save raw audio (input)
        audio_bytes = base64.b64decode(audio_base64)
        raw_audio_path = f"static/uploads/raw_{uuid.uuid4().hex}.webm"
        with open(raw_audio_path, "wb") as f:
            f.write(audio_bytes)
        print(f"💾 Raw audio saved at {raw_audio_path}")

        # Step 2: Convert to WAV using ffmpeg
        wav_audio_path = raw_audio_path.replace(".webm", ".wav")
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", raw_audio_path, "-ac", "1", "-ar", "16000", wav_audio_path]
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"🎧 Converted WAV saved at {wav_audio_path}")

        # Step 3: Azure STT
        stt_url = f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "audio/wav",
            "Accept": "application/json"
        }
        params = {
            "language": language_code
        }

        with open(wav_audio_path, 'rb') as audio_file:
            stt_response = requests.post(stt_url, headers=headers, params=params, data=audio_file)

        print("🔁 Azure STT Status:", stt_response.status_code)
        print("🔊 Azure STT Raw Response:", stt_response.text)

        stt_data = stt_response.json()
        original_text = stt_data.get("DisplayText", "")
        if not original_text:
            return jsonify({"error": "Speech recognition returned empty text"}), 500

        print("🗣️ Transcribed:", original_text)

        # Step 4: Translate if needed
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
            trans_response = requests.post(trans_url, headers=trans_headers, json=trans_body)

            print("🌍 Translator Status:", trans_response.status_code)
            print("🌍 Translator Raw Response:", trans_response.text)

            trans_data = trans_response.json()
            translated_text = trans_data[0]["translations"][0]["text"]

        print("🌐 Final Output:", translated_text)

        return jsonify({
            "original_text": original_text,
            "translated_text": translated_text,
            "language_code": language_code
        })

    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg conversion failed:", e)
        return jsonify({"error": "Audio conversion failed"}), 500

    except Exception as e:
        print("❌ Error in transcription/translation:", str(e))
        return jsonify({"error": str(e)}), 500


