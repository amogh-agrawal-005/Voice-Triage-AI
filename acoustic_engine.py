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