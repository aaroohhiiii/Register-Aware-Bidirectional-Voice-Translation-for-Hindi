"""tools/test_mic.py — Quick diagnostic tool to test microphone signal."""

import sys
import numpy as np
import sounddevice as sd

print("=" * 60)
print(" 🎙️  MICROPHONE DIAGNOSTIC TEST")
print("=" * 60)

default_input = sd.default.device[0]
print(f"Default Input Device Index: {default_input}")
try:
    dev_info = sd.query_devices(default_input)
    print(f"Device Name: {dev_info['name']}")
    print(f"Host API: {sd.query_hostapis(dev_info['hostapi'])['name']}")
    print(f"Default Sample Rate: {dev_info['default_samplerate']} Hz")
except Exception as e:
    print(f"Error querying default device: {e}")

print("\nListing all available input devices:")
input_devs = []
for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] > 0:
        api_name = sd.query_hostapis(d['hostapi'])['name']
        input_devs.append(i)
        print(f"  [{i:2d}] {d['name']} ({api_name})")

print("\n🔴 Recording 3 seconds of audio... Please speak into your mic now!")
try:
    sr = 16000
    rec = sd.rec(int(3 * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    peak = float(np.max(np.abs(rec)))
    rms = float(np.sqrt(np.mean(rec**2)))
    print(f"\n📊 Results:")
    print(f"  Peak Amplitude: {peak:.6f}")
    print(f"  RMS Level:      {rms:.6f}")
    if peak < 0.001:
        print("\n⚠️  WARNING: Audio level is near zero (silence recorded).")
        print("Possible causes:")
        print("  1. Windows Microphone Privacy: Settings > Privacy & security > Microphone -> Ensure 'Let desktop apps access your microphone' is ON.")
        print("  2. Mic is hardware-muted (mic button on laptop/headset) or volume slider is at 0.")
        print("  3. An external headset / earbuds (AirPods, Galaxy Buds, etc.) is connected as default mic.")
    else:
        print("\n✅ Microphone is WORKING and capturing clear audio!")
except Exception as exc:
    print(f"❌ Recording error: {exc}")
