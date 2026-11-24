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
        sample_files = ["calm_call.wav", "panic_call.wav", "pocket_dial.wav", "moderate_call.wav", "ambiguous_call.wav", "drunk_lost.wav"]

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

    # Ensure 'file' column always exists even if some tickets lack it
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