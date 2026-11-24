from pynput import mouse
import sounddevice as sd
import scipy.io.wavfile as wav
import json
import os
from datetime import datetime
from acoustic_engine import analyze_acoustic_stress
from triage_engine import analyze_call_transcript

QUEUE_FILE = "live_calls.json"
RECORD_SECONDS = 18
SAMPLE_RATE = 44100

def load_queue():
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            return json.load(f)
    return []

def save_ticket(ticket):
    queue = load_queue()
    queue.append(ticket)
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

def record_and_process():
    print("\n🔴 TRIGGER DETECTED — Recording live emergency call...")
    print(f"Recording for {RECORD_SECONDS} seconds. Speak now...")
    recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    filename = f"live_call_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    wav.write(filename, SAMPLE_RATE, recording)
    print(f"Saved recording: {filename}")

    print("Analyzing acoustic stress...")
    acoustic_data = analyze_acoustic_stress(filename)

    print("Sending to Gemini for transcription + triage...")
    triage_data = analyze_call_transcript(filename, acoustic_data)
    triage_data["file"] = filename
    triage_data["timestamp"] = datetime.now().isoformat()

    save_ticket(triage_data)

    print(f"\n✅ RESULT: {triage_data.get('urgency_level')} priority — {triage_data.get('incident_type')}")
    print(f"Confidence: {triage_data.get('confidence_score')}% | Needs review: {triage_data.get('needs_human_verification')}")
    print("\nReady for next trigger. Right-click the mouse to simulate another call.\n")

def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.right:
        record_and_process()

if __name__ == "__main__":
    print("=" * 50)
    print("VoiceTriage AI — Physical Trigger Unit ACTIVE")
    print(f"RIGHT-CLICK the mouse to simulate an incoming emergency call ({RECORD_SECONDS}s recording window).")
    print("Press Ctrl+C in this terminal to stop.")
    print("=" * 50)
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()