import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def analyze_call_transcript(audio_file_path: str, acoustic_data: dict) -> dict:
    """Sends audio directly to Gemini for transcription + triage in one call."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    system_instruction = f"""
    You are an expert emergency medical dispatcher AI. First transcribe the audio,
    then analyze it alongside the provided acoustic stress data.
    Acoustic Data: {acoustic_data}

    Return strictly a valid JSON object matching this schema:
    {{
      "transcript": "Full transcription of the audio",
      "location": "Extracted address or location description (or 'Unknown')",
      "incident_type": "Short summary e.g. Cardiac Arrest, Vehicle Collision, Non-Emergency",
      "symptoms": ["List of extracted medical symptoms"],
      "urgency_level": "Use MPDS categories: OMEGA, ALPHA, BRAVO, CHARLIE, DELTA, or ECHO
                        (OMEGA lowest priority, ECHO highest/imminent death)",
      "confidence_score": Integer from 0 to 100,
      "recommended_action": "Suggested dispatch action",
      "caller_emotional_state": "Based on transcript AND acoustic stress level",
      "needs_human_verification": Boolean (true if confidence below 85 or ambiguous)
    }}
    Do not output markdown code blocks or prose. Output raw JSON only.
    """

    try:
        audio_bytes = open(audio_file_path, "rb").read()
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                "Analyze this emergency call audio."
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "transcript": "",
            "location": "Unknown",
            "incident_type": "System Error - Manual Review Required",
            "symptoms": [],
            "urgency_level": "DELTA",
            "confidence_score": 0,
            "recommended_action": "Review call audio immediately",
            "caller_emotional_state": "Unknown",
            "needs_human_verification": True
        }