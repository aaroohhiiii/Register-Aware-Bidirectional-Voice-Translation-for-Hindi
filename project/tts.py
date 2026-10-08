"""Hindi Text-to-Speech (TTS) wrapper module for the project package.

Exports `speak(text, out_path)` and `synthesize_speech(text, output_path)`
using the neural Hindi VITS voice model.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tts import speak


def synthesize_speech(text: str, output_path: str | Path | None = None) -> bytes:
    """Synthesize Hindi text and return audio bytes (and save if output_path given)."""
    target = output_path or (ROOT_DIR / "sample_audio" / "temp_tts.wav")
    saved_path = speak(text, target)
    return Path(saved_path).read_bytes()


__all__ = ["speak", "synthesize_speech"]
