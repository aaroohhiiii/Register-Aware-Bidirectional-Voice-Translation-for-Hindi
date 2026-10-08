"""Evaluation script for TTS with 8 Hindi test sentences across registers.

Generates WAV files and inspects acoustic properties (duration, RMS amplitude, silence).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
from tts import speak

TEST_SENTENCES = [
    ("sentence_01_greeting", "नमस्ते, आप सब कैसे हैं?"),
    ("sentence_02_formal_warm", "दादी जी, आपकी तबीयत अब कैसी है?"),
    ("sentence_03_informal_question", "तू आज शाम को कहां जा रहा है?"),
    ("sentence_04_respectful_casual", "तुम यह काम कल तक पूरा कर लेना।"),
    ("sentence_05_reverse_formal_service", "जी सर, आपका ऑर्डर तुरंत तैयार हो जाएगा।"),
    ("sentence_06_medical_formal", "कृपया बैठिए और अपनी पुरानी दवाइयों की रसीद दिखाइए।"),
    ("sentence_07_compound_pause", "अगर आप चाहें, तो हम कल सुबह दोबारा बात कर सकते हैं।"),
    ("sentence_08_short_exclamation", "बहुत बहुत धन्यवाद! आपका दिन शुभ हो।"),
]

OUT_DIR = Path("demo/tts_test_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"Running TTS Evaluation on {len(TEST_SENTENCES)} Hindi Sentences")
print("=" * 70)

results = []

for tag, text in TEST_SENTENCES:
    wav_path = OUT_DIR / f"{tag}.wav"
    generated_path = speak(text, wav_path)
    
    # Read back audio and analyze
    audio, sr = sf.read(generated_path)
    duration = len(audio) / sr
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    
    # Check silence (energy < 0.01)
    silence_frames = np.sum(np.abs(audio) < 0.005)
    silence_ratio = silence_frames / len(audio)
    
    status = "OK"
    issues = []
    if peak > 0.99:
        issues.append("Possible clipping")
    if rms < 0.01:
        issues.append("Too quiet / near silence")
    if duration < 0.5:
        issues.append("Abnormally short")
    if silence_ratio > 0.6:
        issues.append("High silence ratio")
        
    status_str = ", ".join(issues) if issues else "Clean (No issues)"
    
    print(f"\n[ID: {tag}]")
    print(f"  Text: {text}")
    print(f"  File: {wav_path.name}")
    print(f"  Duration: {duration:.2f}s | Sample Rate: {sr} Hz | Peak: {peak:.3f} | RMS: {rms:.3f}")
    print(f"  Acoustic Check: {status_str}")
    
    results.append({
        "id": tag,
        "text": text,
        "file": str(wav_path),
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "status": status_str,
    })

print("\n" + "=" * 70)
print("TTS Batch Evaluation Completed Successfully!")
print("=" * 70)
