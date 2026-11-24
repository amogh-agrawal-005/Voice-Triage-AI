# VoiceTriage AI: Intelligent Emergency Call Pre-Screening & Physical Triage Prototype

An autonomous, AI-driven emergency voice pre-screening system built for **Mercer | Mettl AI Arena 3.0 — Physical Prototype Challenge**.

VoiceTriage AI is triggered by a physical input device, captures live emergency-call audio, transcribes and triages it in a single multimodal AI call, and feeds the resulting structured ticket into a live, sorted dispatcher dashboard — all without requiring manual human data entry.

---

## 1. Event & Theme Alignment

* **Hackathon:** Mercer | Mettl AI Arena 3.0 (2026)
* **Challenge Format:** Physical Prototype Challenge
* **Primary Theme:** **AI Voice** (Voice Chatbots / Automated Phone Systems)
* **Secondary Theme:** **AI Biology** (Healthcare / Emergency Triage / Symptom Screening)
* **Core Objective:** Deploy AI/ML technology to automate high-friction processes, significantly reducing manual human involvement during critical bottlenecks — demonstrated here through a physical trigger-to-dashboard prototype, not a software simulation alone.

---

## 2. Evaluation Matrix Compliance Strategy

| Evaluation Dimension | Project Execution Strategy |
| :--- | :--- |
| **Novelty** | Extends standard voice input beyond dictation by performing real-time multi-attribute medical triage, acoustic stress extraction, and priority scoring — all in a single multimodal Gemini call that transcribes and reasons over audio directly. |
| **Usability** | A physical right-click trigger initiates live audio capture with zero manual data entry; results flow automatically into a live, auto-refreshing dispatcher dashboard. |
| **Innovation** | Solves a real public safety bottleneck — overloaded emergency call centers, operator fatigue, and delayed response during mass casualty events or disasters — while demonstrating a genuine physical sense-to-action loop under a zero-capital hardware constraint. |
| **Documentation Quality** | Complete process flow, tech stack justification, prompt engineering, architecture diagram, physical prototype design, setup guide, tested output matrix, and ethics section — plus two supporting documents covering global deployment feasibility and scalability (see Section 9). |

---

## 3. Project Overview: The Problem & Solution

### The Problem
Emergency call centers globally face immense operational strain. Call takers handle massive volumes, leading to high burnout rates and increased hold times for critical callers. During public disasters, call centers are frequently flooded with non-emergency queries or pocket dials, delaying dispatch to life-threatening emergencies.

### The Solution: VoiceTriage AI
VoiceTriage AI acts as a **Level-0 Autonomous First Responder**, built around a genuine physical trigger rather than a simulated one:

1. **Physical Trigger:** A right-click on a dedicated USB mouse — repurposed as a single-purpose panic-button input device — initiates a live 15-second audio recording through the system microphone.
2. **Listens & Analyzes Acoustics:** Extracts pitch-variance stress markers directly from the raw audio waveform using `librosa`.
3. **Transcribes & Triages in One Call:** Sends the audio directly to Google Gemini, which transcribes it and reasons over both the transcript and the acoustic stress data to extract structured JSON (location, symptoms, incident type, and priority level) in a single request.
4. **Multi-Call Queue Management:** Populates a color-coded, sorted, auto-refreshing **Dispatcher Dashboard** where high-priority cases bypass hold queues and low-confidence/ambiguous calls are explicitly flagged for human review — with the original audio made available for direct playback.

---

## 4. Technical Architecture & Tech Stack

```text
   [ Physical Trigger ]
   (Right-click on USB mouse,
    intercepted via pynput)
              │
              ▼
   [ Live Audio Capture ]
   (sounddevice, 18s window)
              │
   ┌──────────┴──────────┐
   ▼                     ▼
[ librosa ]     [ Gemini API — audio input ]
(Acoustics)     (Transcription + Triage in one call)
   │                     │
   └──────────┬──────────┘
              ▼
   [ Streamlit UI / Dashboard ]
   (Color-coded, sorted queue +
    audio playback for flagged calls)
```

### Tech Stack Justification
* **Physical Trigger:** `pynput` — intercepts a right-click from a standard USB mouse, decoupled from its pointing function, turning an already-owned device into a dedicated panic button at zero material cost.
* **Live Audio Capture:** `sounddevice` + `scipy` — records a 18-second window from the system microphone the moment the trigger fires.
* **Frontend / Dashboard Framework:** `Streamlit` (Python) — rapid reactive rendering for a live, sortable dispatch queue, with `streamlit-autorefresh` for automatic queue polling.
* **Acoustic Biomarkers:** `librosa` — lightweight local pitch-variance analysis, no API cost.
* **Triage & Reasoning Engine:** `Google Gemini` (`google-genai` SDK, model: `gemini-3.5-flash-lite`) — accepts audio directly as input, so one API call handles both transcription and structured triage reasoning. No separate speech-to-text service or API key required.
* **Programming Language:** Python 3.10+

**Single-API design note:** this project intentionally runs on Gemini alone — no OpenAI/Whisper dependency — so it works entirely on Google's free API tier with only one key to manage.

**Zero-capital hardware note:** the physical layer was deliberately built from an already-owned USB mouse rather than purchased electronics, given a zero-budget constraint during the prototype phase. See Section 6 for the full engineering rationale and the production hardware path this is designed to scale into.

---

## 5. Development Steps & Implementation Guide

### Step 1: Get a free Gemini API key
Go to **aistudio.google.com** → "Get API key" → generate one. Free tier is sufficient for this project; no billing setup required.

### Step 2: Repository Setup & Environment
```bash
git clone https://github.com/your-username/VoiceTriage-AI.git
cd VoiceTriage-AI

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Dependencies (`requirements.txt`)
```text
streamlit
streamlit-autorefresh
google-genai
pandas
python-dotenv
librosa
soundfile
pynput
sounddevice
scipy
```
```bash
pip install -r requirements.txt
```

### Step 4: Environment Variables (`.env`)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> ⚠️ **Never commit this file.** Ensure `.env` is listed in `.gitignore`. Anyone running this project needs their own free key from aistudio.google.com — this repo does not include one.

### Step 5: Core Implementation Components

#### A. Physical Trigger Engine (`trigger_engine.py`)
Runs as a standalone background process. Right-clicking the mouse triggers a live 15-second recording, runs it through the same acoustic + Gemini pipeline used for batch testing, and writes the result to `live_calls.json` for the dashboard to pick up.

```python
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

    acoustic_data = analyze_acoustic_stress(filename)
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
```

#### B. Acoustic Biomarker Engine (`acoustic_engine.py`)
```python
import librosa
import numpy as np

def analyze_acoustic_stress(audio_file_path: str) -> dict:
    """Analyzes audio for stress markers like pitch variance."""
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        pitch_variance = np.var(valid_f0) if len(valid_f0) > 0 else 0

        # Thresholds are illustrative placeholders for demo, not clinically calibrated
        if pitch_variance > 5000:
            stress_level = "High (Potential Panic)"
        elif pitch_variance > 1000:
            stress_level = "Elevated"
        else:
            stress_level = "Normal/Calm"

        return {"pitch_variance": float(pitch_variance), "acoustic_stress": stress_level}
    except Exception as e:
        return {"pitch_variance": 0, "acoustic_stress": "Unknown"}
```

#### C. Gemini Multimodal Triage Engine (`triage_engine.py`)
```python
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
```

#### D. Interactive Dashboard (`app.py`)
```python
import streamlit as st
import pandas as pd
import json
import os
from streamlit_autorefresh import st_autorefresh
from triage_engine import analyze_call_transcript
from acoustic_engine import analyze_acoustic_stress

st.set_page_config(page_title="VoiceTriage AI - Dispatcher Console", layout="wide")
st.title("🚨 VoiceTriage AI | Emergency Pre-Screening Console")

if 'call_queue' not in st.session_state:
    st.session_state.call_queue = []

def load_live_calls():
    """Loads live_calls.json into the queue, avoiding duplicate entries by filename."""
    if os.path.exists("live_calls.json"):
        with open("live_calls.json", "r") as f:
            live_tickets = json.load(f)
        existing_files = {t.get('file') for t in st.session_state.call_queue}
        new_tickets = [t for t in live_tickets if t.get('file') not in existing_files]
        st.session_state.call_queue.extend(new_tickets)
        return len(new_tickets)
    return 0

# --- Sidebar controls ---
st.sidebar.header("Control Panel")
simulate_calls = st.sidebar.button("🎙️ Simulate Batch Incoming Calls", use_container_width=True)
load_live = st.sidebar.button("📥 Load Live Triggered Calls", use_container_width=True)
refresh_now = st.sidebar.button("🔄 Refresh Queue", use_container_width=True)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s", value=False)
if auto_refresh:
    st_autorefresh(interval=60000, key="auto_refresh_timer")

if simulate_calls:
    with st.spinner("Processing batch incoming audio calls..."):
        st.session_state.call_queue = []
        sample_files = [
            "calm_call.wav", "panic_call.wav", "pocket_dial.wav",
            "moderate_call.wav", "ambiguous_call.wav", "drunk_lost.wav"
        ]

        for file in sample_files:
            try:
                acoustic_data = analyze_acoustic_stress(file)
                triage_data = analyze_call_transcript(file, acoustic_data)
                triage_data['file'] = file
                st.session_state.call_queue.append(triage_data)
            except Exception as e:
                st.session_state.call_queue.append({
                    "file": file, "location": "Unknown",
                    "incident_type": "Processing Failure", "urgency_level": "DELTA",
                    "confidence_score": 0, "needs_human_verification": True,
                    "symptoms": [], "recommended_action": "Review source audio file immediately"
                })

if load_live or refresh_now or auto_refresh:
    new_count = load_live_calls()
    if load_live or refresh_now:
        if new_count > 0:
            st.success(f"Loaded {new_count} new live-triggered call(s) into the queue.")
        elif os.path.exists("live_calls.json"):
            st.info("No new live calls since last refresh.")
        else:
            st.warning("No live calls recorded yet. Run trigger_engine.py and right-click the mouse first.")

if st.session_state.call_queue:
    st.subheader("Active Call Queue")
    urgency_map = {"ECHO": 6, "DELTA": 5, "CHARLIE": 4, "BRAVO": 3, "ALPHA": 2, "OMEGA": 1}
    sorted_queue = sorted(st.session_state.call_queue,
                          key=lambda x: urgency_map.get(x.get('urgency_level', 'OMEGA'), 0),
                          reverse=True)
    df = pd.DataFrame(sorted_queue)

    if 'file' not in df.columns:
        df['file'] = "N/A"

    st.dataframe(
        df[['urgency_level', 'incident_type', 'location', 'confidence_score', 'needs_human_verification', 'file']],
        use_container_width=True
    )

    # --- Flagged calls: let the dispatcher actually listen to them ---
    flagged = [t for t in sorted_queue if t.get('needs_human_verification')]
    if flagged:
        st.markdown("---")
        st.subheader("🔎 Calls Requiring Human Verification")
        for ticket in flagged:
            file_path = ticket.get('file')
            with st.expander(f"{ticket.get('urgency_level', 'N/A')} — {ticket.get('incident_type', 'Unknown')} ({file_path})"):
                st.write(f"**Transcript:** {ticket.get('transcript', 'N/A')}")
                st.write(f"**Confidence:** {ticket.get('confidence_score')}%")
                if file_path and os.path.exists(file_path):
                    st.audio(file_path)
                else:
                    st.caption("Audio file not found on disk.")

    st.markdown("---")
    st.subheader("Detailed Ticket View (Top Priority)")
    top_ticket = sorted_queue[0]
    urgency = top_ticket.get("urgency_level", "OMEGA")

    if urgency in ["ECHO", "DELTA"]:
        st.error(f"⚠️ {urgency} PRIORITY DETECTED")
    elif urgency in ["CHARLIE", "BRAVO"]:
        st.warning(f"⚡ {urgency} PRIORITY DETECTED")
    else:
        st.success(f"ℹ️ {urgency} PRIORITY DETECTED")

    if top_ticket.get('needs_human_verification'):
        st.error("FLAGGED: Requires Immediate Human Verification")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Transcript:** {top_ticket.get('transcript', 'N/A')}")
        st.write(f"**Location:** {top_ticket.get('location')}")
        st.write(f"**Confidence:** {top_ticket.get('confidence_score')}%")
    with col2:
        st.write(f"**Symptoms:** {', '.join(top_ticket.get('symptoms', []))}")
        st.write(f"**Action:** {top_ticket.get('recommended_action')}")
        st.write(f"**Caller State:** {top_ticket.get('caller_emotional_state', 'Unknown')}")
```

### Step 6: Run the Physical Trigger + Dashboard

Open **two terminals**:

**Terminal 1 — start the physical trigger listener:**
```bash
python trigger_engine.py
```
Leave this running. Right-click the mouse at any time to record and triage a live call.

**Terminal 2 — start the dashboard:**
```bash
streamlit run app.py
```
Click **"Load Live Triggered Calls"** (or enable auto-refresh) to bring right-click-triggered results into the queue, or click **"Simulate Batch Incoming Calls"** to run the six pre-recorded test scenarios below.

---

## 6. Physical Prototype Design

### High-Level Design
This prototype phase operated under a zero-capital constraint. Rather than treat this as a blocker, the physical-input requirement was met by repurposing an already-owned USB mouse as a dedicated trigger device — its right-click signal is intercepted at the software level via `pynput` and decoupled from its normal pointing function, making it function as a single-purpose panic button. This is an honest, minimal-but-real physical input mechanism, not a simulated one.

### Production Path
The trigger mechanism is fully decoupled from the processing pipeline (acoustic analysis + Gemini triage), so it can be replaced with a dedicated tactile push-button wired to a microcontroller (Arduino/ESP32) — paired with a status LED and small speaker for on-device feedback — with no change required to `acoustic_engine.py` or `triage_engine.py`.

### Safety Consideration
The right-click trigger is deliberately distinct from a left-click or touchpad tap to reduce accidental activation. During live demonstration, disabling the laptop touchpad is recommended so only the dedicated mouse can trigger a recording.

---

## 7. Tested Output Matrix

The system was validated against six recorded scenarios spanning the full MPDS urgency spectrum, plus live right-click-triggered recordings captured directly through the physical prototype:

| Urgency Level | Test Scenario | Confidence | Human Verification Flagged |
| :--- | :--- | :--- | :--- |
| **ECHO** | Cardiac arrest, unconscious patient | 98% | No |
| **BRAVO** | Fall with injury, conscious patient | 92% | No |
| **ALPHA** | Ambiguous / unclear situation | 40% | **Yes** |
| **ALPHA** | Confused / disoriented caller | 40% | **Yes** |
| **OMEGA** | Animal rescue (non-emergency) | 95% | No |
| **OMEGA** | Non-emergency inquiry | 99% | No |

This confirms the system does not force artificially high confidence on ambiguous input — when the caller's situation is genuinely unclear, the model both lowers its confidence score and explicitly flags the call for human review, rather than guessing. Flagged calls surface directly in the dashboard with their original audio available for playback (see Section 5, Step D).

---

## 8. Insights, Justifications & Ethical Considerations

* **Triage Methodology Grounding:** Urgency scoring maps to the Medical Priority Dispatch System (MPDS) tiers (OMEGA → ECHO). *Note: This is a simplified educational adaptation for demonstration purposes and is not a certified dispatch protocol.*
* **Multimodal Analysis:** VoiceTriage AI extracts acoustic stress markers (pitch variance via `librosa`) directly from the raw waveform, letting the model evaluate caller distress independent of transcription accuracy alone.
* **Data Privacy & Ethics:** Emergency calls inherently contain PHI/PII. This project operates as a transient pre-processor. In production, API calls must be routed through enterprise-tier, zero-data-retention endpoints, with a PII redaction layer between audio ingestion and LLM analysis.
* **Human-in-the-Loop Philosophy:** VoiceTriage AI is a **pre-screening decision-support tool**, not a dispatcher replacement. Confidence scores, explicit low-confidence flagging, and direct audio playback for review (Section 7) ensure ambiguous calls always route to a human.
* **Low-Latency Fail-safes:** API failures or malformed responses default to a `DELTA` urgency with mandatory human review, ensuring zero false-negative safety breaches.

---

## 9. Supporting Documentation

Two additional documents accompany this README as part of the full submission package:

* **Design Documentation** — full System Design (High-Level + Low-Level), MVP description, Testing & Validation, and Iteration & Execution log.
* **Scalability & Global Deployment Snapshot** — real-world statistics on emergency call volume, existing AI-triage deployment precedent, computed per-call API cost, and governance/regulatory considerations (including the EU AI Act's explicit high-risk classification of this exact use case) across multiple countries.

---

## 10. Future Roadmap

1. **Dedicated Hardware Trigger:** Replace the mouse-click trigger with a tactile push-button wired to a microcontroller (Arduino/ESP32), with an on-device status LED and speaker — no change required to the existing software pipeline.
2. **Multi-Turn Clarification (State Machine):** Structured follow-up questions (e.g., "Is the patient breathing?") for calls lacking clear severity markers.
3. **Direct Computer-Aided Dispatch (CAD) Integration:** REST API connection to real emergency CAD systems for automated unit dispatching.
