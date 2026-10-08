"""tools/record_clips.py — Interactive audio recording tool for 18 test prompts.

Features:
- Native samplerate capture + high quality resampling to 16 kHz mono.
- Instant volume/peak level check to alert if mic is muted.
- Easy re-record ('r') or skip ('s').
"""

from __future__ import annotations

import csv
import io
import sys
import threading
import time
from pathlib import Path

# Force UTF-8 encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_CSV = _PROJECT_ROOT / "tools" / "clip_script.csv"
CLIPS_DIR = _PROJECT_ROOT / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16_000
MAX_DURATION_S = 20.0


def record_clip_interactive(prompt_id: str, condition: str, prompt: str, device: int | None = None) -> None:
    out_wav = CLIPS_DIR / f"{prompt_id}.wav"

    dev_info = sd.query_devices(device if device is not None else sd.default.device[0])
    native_sr = int(dev_info.get("default_samplerate", 44100))
    channels = min(1, dev_info.get("max_input_channels", 1))

    while True:
        print("\n" + "─" * 70)
        print(f"🎙️  Clip: [{prompt_id}] ({condition.upper()})")
        print(f"📜  Prompt: \"{prompt}\"")
        if condition.lower() == "noisy":
            print("🔊  [Play background noise for noisy condition]")
        print("─" * 70)
        print("Press [ENTER] to record (or 's' + [ENTER] to skip): ", end="", flush=True)

        user_input = input().strip().lower()
        if user_input == "s":
            print(f"⏭️  Skipped {prompt_id}.")
            return

        print(f"\n🔴 RECORDING... Speak now! (Press [ENTER] to finish, max {MAX_DURATION_S:.0f}s)")

        frames: list[np.ndarray] = []
        stop_event = threading.Event()

        def audio_callback(indata, frame_count, time_info, status):  # noqa: ARG001
            if status:
                print(status, file=sys.stderr)
            frames.append(indata.copy())

        # Open stream with native rate for maximum Windows soundcard compatibility
        stream = sd.InputStream(
            samplerate=native_sr,
            channels=channels,
            dtype="float32",
            device=device,
            callback=audio_callback,
        )

        def wait_for_enter():
            input()
            stop_event.set()

        t_input = threading.Thread(target=wait_for_enter, daemon=True)
        t_input.start()

        start_time = time.time()
        with stream:
            while not stop_event.is_set():
                if time.time() - start_time >= MAX_DURATION_S:
                    print("\n⏱️  Reached max duration (20s). Stopping.")
                    break
                sd.sleep(50)

        if not frames:
            print("⚠️  No audio frames recorded. Retrying...")
            continue

        raw_audio = np.concatenate(frames, axis=0).squeeze()
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)

        # Resample to 16 kHz if needed
        if native_sr != TARGET_SR:
            audio_16k = librosa.resample(raw_audio, orig_sr=native_sr, target_sr=TARGET_SR).astype(np.float32)
        else:
            audio_16k = raw_audio.astype(np.float32)

        peak = float(np.max(np.abs(audio_16k)))
        duration_s = len(audio_16k) / TARGET_SR

        # Save 16 kHz WAV
        sf.write(str(out_wav), audio_16k, TARGET_SR, subtype="PCM_16")

        if peak < 0.005:
            print(f"\n⚠️  [SILENCE WARNING] Peak amplitude is {peak:.6f}. The mic might be muted or blocked by Windows privacy settings!")
        else:
            print(f"\n✅  [OK] Saved {out_wav.name} ({duration_s:.2f}s, Peak: {peak:.3f})")

        print("Options: Press [ENTER] to accept, 'r' + [ENTER] to re-record: ", end="", flush=True)
        choice = input().strip().lower()
        if choice == "r":
            print("🔄  Re-recording clip...")
            continue
        break


def main():
    if not SCRIPT_CSV.exists():
        print(f"❌ Script file not found: {SCRIPT_CSV}")
        sys.exit(1)

    with open(SCRIPT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("\n" + "=" * 70)
    print(f" 🎙️  HINDI ASR RECORDING SUITE ({len(rows)} prompts)")
    print("=" * 70)

    for row in rows:
        record_clip_interactive(
            prompt_id=row["clip_id"],
            condition=row["condition"],
            prompt=row["prompt_text"],
        )

    print("\n" + "=" * 70)
    print(f"✅ All recordings saved to: {CLIPS_DIR.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
