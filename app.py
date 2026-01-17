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
import traceback

# -------------------------------------------------
# ENV + APP SETUP
# -------------------------------------------------

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "temporary-secret")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# AZURE + TOGETHER CONFIG
# -------------------------------------------------

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")


client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

print("✅ App started")
print("🔑 Together API key loaded:", bool(os.getenv("TOGETHER_API_KEY")))

def get_translation_code(form_language_code):
    """Convert form language code to Azure Translator language code"""
    language_mapping = {
        'en-IN': 'en',      # English -> English (no translation)
        'hi-IN': 'hi',      # Hindi -> Hindi
        'or-IN': 'or'       # Odia -> Odia
    }
    return language_mapping.get(form_language_code, 'en')

SUPPORTED_LANGUAGES = {
    'en-IN': 'English',
    'hi-IN': 'Hindi', 
    'or-IN': 'Odia'
}

# -------------------------------------------------
# BASIC PAGE ROUTES
# -------------------------------------------------

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/input")
def input_page():
    return render_template("input.html")


# -------------------------------------------------
# AI HELPERS WITH DEBUGGING
# -------------------------------------------------

def call_together(prompt):
    try:
        print("🤖 Sending prompt to Together.ai...")
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        print("✅ Together.ai response received")
        return output

    except Exception as e:
        print("❌ Together.ai call failed")
        traceback.print_exc()
        return "⚠️ AI generation failed. Please try again."


def generate_key_notes(text):
    print("📝 Generating key notes...")
    prompt = f"""
You are an AI assistant that extracts key notes from classroom discussions.

Instructions:
- Extract 5 to 8 concise key points.
- Each point must be short and factual.
- Do NOT explain.
- Do NOT repeat ideas.

Return the result as a plain numbered list.

Discussion:
{text}
"""
    return call_together(prompt)


def generate_detailed_points(text):
    print("📘 Generating detailed discussion...")
    prompt = f"""
You are an AI assistant that explains discussions in detail.

Instructions:
- Analyze the discussion text below.
- Describe the main arguments, counterarguments, and themes.
- Write in clear, academic but simple language.
- Do NOT use bullet points.
- Do NOT add conclusions not present in the text.

Return 2–4 structured paragraphs.

Discussion:
{text}
"""
    return call_together(prompt)


import json

def generate_memory_map(text):
    print("🧠 Generating memory map...")
    prompt = f"""
You are an AI assistant that converts discussions into structured knowledge graphs.

Instructions:
- Analyze the discussion text below.
- Extract key concepts, arguments, concerns, and outcomes.
- Represent them as a knowledge graph in JSON format.
- Use short, clear labels.

Node rules:
- Each node must have:
  - id (short unique string)
  - label (human-readable)
  - type (one of: concept, argument, concern, outcome)

Edge rules:
- Each edge must have:
  - from (node id)
  - to (node id)
  - relation (one of: supports, challenges, leads_to)

Constraints:
- Output ONLY valid JSON.
- Do NOT include explanations or extra text.
- Do NOT invent information not present in the discussion.
- Limit to a maximum of 15 nodes.

JSON format:
{{
  "nodes": [...],
  "edges": [...]
}}

Discussion:
{text}
"""
    raw_output = call_together(prompt)

    try:
        graph = json.loads(raw_output)
        print("✅ Memory map JSON parsed successfully")
        return graph
    except Exception:
        print("❌ Failed to parse memory map JSON")
        print(raw_output)
        return {
            "nodes": [],
            "edges": []
        }


@app.route('/translate_text', methods=['POST'])
def translate_text():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        from_lang = data.get("from", None)          # e.g., "or-IN" or "hi-IN"
        to_lang = data.get("to", "en")             # default English

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Azure Translator endpoint (same as /transcribe uses)
        trans_url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&to={to_lang}"

        # If from_lang is provided, include it — it's optional for Azure
        if from_lang:
            trans_url += f"&from={from_lang}"

        trans_headers = {
            "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
            "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
            "Content-Type": "application/json"
        }

        trans_body = [{"Text": text}]

        response = requests.post(trans_url, headers=trans_headers, json=trans_body)

        print("🌐 Typed Translation Status:", response.status_code)
        print("🌐 Typed Translation Raw:", response.text)

        if response.status_code != 200:
            return jsonify({
                "error": "Azure translation failed",
                "details": response.text
            }), response.status_code

        response_json = response.json()

        translated_text = response_json[0]["translations"][0]["text"]

        return jsonify({"translated_text": translated_text})

    except Exception as e:
        print("❌ Error in /translate_text:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ Transcribe audio from base64 and auto-translate to English
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

# -------------------------------------------------
# PROCESS TEXT INPUT
# -------------------------------------------------

@app.route("/process-text", methods=["POST"])
def process_text():
    try:
        input_text = request.form.get("discussion_text", "").strip()
        print("📥 Received text input")

        if not input_text:
            print("⚠️ Empty input received")
            return redirect(url_for("input_page"))

        key_notes = generate_key_notes(input_text)
        detailed_points = generate_detailed_points(input_text)
        memory_map = generate_memory_map(input_text)
        
        session["key_notes"] = key_notes
        session["detailed_points"] = detailed_points
        session["memory_map"] = memory_map

        print("✅ AI outputs stored in session")
        return redirect(url_for("result_page"))

    except Exception as e:
        print("❌ Error in /process-text")
        traceback.print_exc()
        return "Internal error during processing", 500


@app.route("/process-audio", methods=["POST"])
def process_audio():
    try:
        print("🎧 Audio file received")

        if "audio_file" not in request.files:
            print("❌ No audio_file field in request")
            return redirect(url_for("input_page"))

        audio_file = request.files["audio_file"]

        if audio_file.filename == "":
            print("❌ Empty filename")
            return redirect(url_for("input_page"))

        raw_path = os.path.join(
            UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}_{audio_file.filename}"
        )
        audio_file.save(raw_path)
        print(f"💾 Saved raw audio: {raw_path}")

        wav_path = raw_path.rsplit(".", 1)[0] + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", wav_path],
            check=True
        )
        print(f"🎼 Converted to WAV: {wav_path}")

        # Convert WAV to base64
        with open(wav_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        language_code = request.form.get("language", "en-IN")

        # Call existing /transcribe
        with app.test_client() as client:
            response = client.post(
                "/transcribe",
                json={"audio": audio_base64, "language": language_code}
            )

        if response.status_code != 200:
            print("❌ Transcription failed")
            return redirect(url_for("input_page"))

        data = response.get_json()
        original_text = data.get("original_text", "")
        translated_text = data.get("translated_text", "")

        if not translated_text:
            print("❌ Empty transcription")
            return redirect(url_for("input_page"))

        session["original_transcript"] = original_text
        session["translated_transcript"] = translated_text

        key_notes = generate_key_notes(translated_text)
        detailed_points = generate_detailed_points(translated_text)
        memory_map = generate_memory_map(translated_text)

        session["key_notes"] = key_notes
        session["detailed_points"] = detailed_points
        session["memory_map"] = memory_map

        print("✅ Audio pipeline complete")
        return redirect(url_for("result_page"))

    except Exception:
        print("❌ Error in /process-audio")
        traceback.print_exc()
        return "Audio processing error", 500



# -------------------------------------------------
# RESULT PAGE
# -------------------------------------------------

@app.route("/result")
def result_page():
    return render_template(
        "result.html",
        key_notes=session.get("key_notes", ""),
        detailed_points=session.get("detailed_points", ""),
        memory_map=session.get("memory_map", {})
    )




# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

