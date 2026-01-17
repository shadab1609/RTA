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

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

print("✅ App started")
print("🔑 Together API key loaded:", bool(os.getenv("TOGETHER_API_KEY")))

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
# EXISTING TRANSCRIPTION ROUTE (UNCHANGED)
# -------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe_audio_base64():
    try:
        data = request.get_json(force=True)
        print("🎙️ Incoming audio transcription request")

        audio_base64 = data.get("audio")
        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400

        audio_bytes = base64.b64decode(audio_base64)
        raw_audio_path = f"{UPLOAD_FOLDER}/raw_{uuid.uuid4().hex}.webm"
        with open(raw_audio_path, "wb") as f:
            f.write(audio_bytes)

        wav_audio_path = raw_audio_path.replace(".webm", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_audio_path, "-ac", "1", "-ar", "16000", wav_audio_path],
            check=True
        )

        return jsonify({"status": "Audio processed (STT later)"}), 200

    except Exception as e:
        print("❌ Error in transcription route")
        traceback.print_exc()
        return jsonify({"error": "Transcription failed"}), 500


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

