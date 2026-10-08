"""ui_logic.py — Testable logic layer for the Gradio UI (no Gradio imports here).

Public API
----------
redact(text) -> str
build_status_line(result: dict) -> str
friendly_error(result_or_exception) -> str
process_turn(audio_path, direction_label, scenario, asr_backend, tts_backend,
             voice, language_mode, target_override, cloud_on, session_log) -> dict
format_log(session_log: list) -> str

Constants
---------
ASR_CHOICES, TTS_CHOICES, UI_TIMBRE_VOICES
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import config
import language_hint
import script_check
from pipeline import run_pipeline

logger = logging.getLogger(__name__)

# ── Option constants ──────────────────────────────────────────────────────────
ASR_CHOICES: List[str] = ["whisper", "prisma"]
TTS_CHOICES: List[str] = ["timbre", "parler", "mms"]
UI_TIMBRE_VOICES: List[str] = list(getattr(config, "UI_TIMBRE_VOICES", [config.TIMBRE_VOICE]))

# Language dropdown: auto + the two Timbre language codes from config
LANGUAGE_CHOICES: List[str] = [
    "Auto (recommended)",
    config.TIMBRE_LANGUAGE,           # "hi-IN"
    config.TIMBRE_MIXED_LANGUAGE,     # "hi-en"
]

# Direction labels shown in the UI
DIRECTION_CHOICES: List[str] = [
    "A: User -> Other",
    "B: Other -> User",
]

# Target override choices
TARGET_CHOICES: List[str] = ["Auto (from direction)", "formal", "informal"]

# Build scenario dropdown labels from config
SCENARIO_LABELS: Dict[str, str] = {
    key: data["label"]
    for key, data in config.SCENARIOS.items()
}
# Reverse map: label -> key
_LABEL_TO_SCENARIO: Dict[str, str] = {v: k for k, v in SCENARIO_LABELS.items()}

DEFAULT_SCENARIO_LABEL: str = SCENARIO_LABELS.get(
    config.DEFAULT_SCENARIO, config.DEFAULT_SCENARIO
)


# ── Secret redaction ─────────────────────────────────────────────────────────

def redact(text: str) -> str:
    """Replace secret values in *text* with '[hidden]'.

    Patterns removed:
    - The actual GNANI_API_KEY value (from env)
    - Any 'X-API-Key-ID' header value pattern
    - Any token matching hf_[A-Za-z0-9]{20,} (HuggingFace tokens)
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Actual API key value from env
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if api_key and api_key in text:
        text = text.replace(api_key, "[hidden]")

    # 2. X-API-Key-ID header values (key: value or key=value forms)
    text = re.sub(
        r"(X-API-Key-ID['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9_\-\.]{8,})",
        r"\1[hidden]",
        text,
        flags=re.IGNORECASE,
    )

    # 3. HuggingFace tokens: hf_ followed by 20+ alphanum chars
    text = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "[hidden]", text)

    return text


# ── Status line ───────────────────────────────────────────────────────────────

def build_status_line(result: dict, cloud_on: bool = True) -> str:
    """Build a single human-readable status line from a pipeline result dict.

    Example:
        ASR: prisma (412 ms) | Clean+Detect: 3 ms | Transform: 5 ms |
        TTS: timbre (1840 ms) | Total: 2.3 s | Fallback: none
    """
    asr_backend = result.get("asr_backend", "?")
    asr_ms = result.get("asr_ms", 0.0)

    clean_ms = result.get("clean_ms", 0.0) + result.get("detect_ms", 0.0)
    transform_ms = result.get("transform_ms", 0.0)
    normalize_ms = result.get("normalize_ms", 0.0)

    tts_backend = result.get("tts_backend") or "—"
    tts_ms = result.get("tts_ms", 0.0)

    total_ms = result.get("total_ms", 0.0)
    total_s = round(total_ms / 1000, 2)

    # Fallback description
    fallback_parts = []
    if result.get("fallback_used") or result.get("asr_fallback_reason"):
        reason = redact(result.get("asr_fallback_reason") or "")
        fallback_parts.append(f"ASR→whisper ({reason})" if reason else "ASR fallback")
    if result.get("tts_fallback_used"):
        fallback_parts.append(f"TTS→{tts_backend}")
    fallback_str = ", ".join(fallback_parts) if fallback_parts else "none"

    parts = [
        f"ASR: {asr_backend} ({asr_ms:.0f} ms)",
        f"Clean+Detect: {clean_ms:.0f} ms",
        f"Transform: {transform_ms:.0f} ms",
        f"Normalize: {normalize_ms:.0f} ms",
        f"TTS: {tts_backend} ({tts_ms:.0f} ms)",
        f"Total: {total_s} s",
        f"Fallback: {fallback_str}",
    ]

    if not cloud_on:
        parts.append("Cloud: OFF (fully local)")

    return " | ".join(parts)


# ── Friendly error messages ───────────────────────────────────────────────────

def friendly_error(result_or_exception: Any) -> str:
    """Map known error cases to plain user-readable messages.

    All output passes through redact().
    """
    # Extract error string
    if isinstance(result_or_exception, dict):
        err = result_or_exception.get("error") or ""
    elif isinstance(result_or_exception, Exception):
        err = str(result_or_exception)
    else:
        err = str(result_or_exception)

    err_lower = err.lower()

    # Ordered from most specific to most generic
    if not err:
        return ""

    if "no audio" in err_lower or err_lower in ("none", ""):
        return "No audio provided. Please record or upload a clip first."

    if "file not found" in err_lower or "audio file not found" in err_lower:
        return "Audio file could not be found. Please try recording again."

    if "could not prepare audio" in err_lower or "unreadable" in err_lower:
        return "The audio file could not be read. Try a different format (WAV is safest)."

    if "too_long_for_prisma" in err_lower or "too long" in err_lower:
        return "Clip is over 60 seconds — Prisma supports up to 60 s. Trim the audio or switch to Whisper."

    if "empty_asr_transcript" in err_lower or "empty transcript" in err_lower:
        return "No speech detected, please try again."

    if "empty_after_cleanup" in err_lower:
        return "The transcript was empty after cleanup. Try speaking more clearly."

    if "missing_api_key" in err_lower or "gnani_api_key is not set" in err_lower or "gnani key not set" in err_lower:
        return "Gnani key not set; use Cloud OFF mode or add your key to .env and restart."

    if "timed out" in err_lower or "timeout" in err_lower:
        return "The cloud API request timed out. Check your connection or switch to Cloud OFF mode."

    # HTTP status codes — show only the code, not response body
    http_match = re.search(r"http\s*(\d{3})", err_lower)
    if http_match:
        code = http_match.group(1)
        return f"Cloud API returned HTTP {code}. Check your API key or try again later."

    if "timbre" in err_lower and "failed" in err_lower:
        return "Timbre TTS failed. The app will try the next backend automatically."

    if "parler" in err_lower and "failed" in err_lower:
        return "Parler TTS failed. The app will try MMS as fallback."

    if "mms" in err_lower and "failed" in err_lower:
        return "MMS TTS failed. All backends exhausted — check your environment."

    if "transform_failed" in err_lower:
        return "Register transformation failed; the cleaned text was used instead."

    if "detect_failed" in err_lower:
        return "Register detection failed; transformation was skipped."

    if "asr_failed" in err_lower:
        return "Speech recognition failed unexpectedly. Please try again."

    # Generic fallback — redact secrets before showing
    return redact(f"An error occurred: {err[:200]}")


# ── Session log ───────────────────────────────────────────────────────────────

def format_log(session_log: List[Dict]) -> str:
    """Format session_log as a readable markdown string."""
    if not session_log:
        return "_No turns yet._"

    lines = ["### Conversation Log\n"]
    for entry in session_log:
        turn_n = entry.get("turn", "?")
        direction = entry.get("direction", "?")
        cleaned = redact(entry.get("cleaned", "") or "")
        transformed = redact(entry.get("transformed", "") or "")
        asr_b = entry.get("asr_backend", "?")
        tts_b = entry.get("tts_backend", "?")
        ms = entry.get("total_ms", 0)
        err = entry.get("error")

        lines.append(f"**Turn {turn_n}** [{direction}]")
        if err:
            lines.append(f"- ⚠️ Error: {redact(str(err)[:120])}")
        else:
            lines.append(f"- **Heard:** {cleaned}")
            lines.append(f"- **Said:** {transformed}")
            lines.append(f"- Backends: ASR={asr_b}, TTS={tts_b} | {ms:.0f} ms")
        lines.append("")

    return "\n".join(lines)


# ── Core process_turn ─────────────────────────────────────────────────────────

def process_turn(
    audio_path: Optional[str],
    direction_label: str,
    scenario_label: str,
    asr_backend: str,
    tts_backend: str,
    voice: str,
    language_mode: str,
    target_override: str,
    cloud_on: bool,
    session_log: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Process one conversation turn and return UI-ready values.

    Parameters
    ----------
    audio_path      : Path to recorded/uploaded WAV file, or None.
    direction_label : "A: User -> Other" or "B: Other -> User".
    scenario_label  : Human-readable scenario label (from SCENARIO_LABELS).
    asr_backend     : "whisper" or "prisma".
    tts_backend     : "timbre", "parler", or "mms".
    voice           : Timbre voice name (ignored for local backends).
    language_mode   : "Auto (recommended)", "hi-IN", or "hi-en".
    target_override : "Auto (from direction)", "formal", or "informal".
    cloud_on        : If False, force local-only mode (whisper ASR, no timbre TTS).
    session_log     : Mutable list of past turn dicts; a new entry is appended.

    Returns
    -------
    dict with keys:
        raw, cleaned, detected, target, transformed, normalized,
        output_audio, status_line, script_warning, error_message,
        session_log (the updated list), log_text (formatted markdown)
    """
    if session_log is None:
        session_log = []

    turn_number = len(session_log) + 1

    # ── Defaults for the return dict ─────────────────────────────────────────
    ui = {
        "raw": "",
        "cleaned": "",
        "detected": "",
        "target": "",
        "transformed": "",
        "normalized": "",
        "output_audio": None,
        "status_line": "",
        "script_warning": "",
        "error_message": "",
        "session_log": session_log,
        "log_text": "",
    }

    # ── Guard: no audio ───────────────────────────────────────────────────────
    if not audio_path:
        ui["error_message"] = friendly_error({"error": "no audio"})
        ui["log_text"] = format_log(session_log)
        # Still log this failed turn
        session_log.append({
            "turn": turn_number,
            "direction": _direction_letter(direction_label),
            "cleaned": "",
            "transformed": "",
            "asr_backend": asr_backend,
            "tts_backend": None,
            "total_ms": 0.0,
            "error": "no_audio",
        })
        ui["log_text"] = format_log(session_log)
        return ui

    # ── Map direction label → "A"/"B" ────────────────────────────────────────
    direction = _direction_letter(direction_label)

    # ── Map scenario label → config key ──────────────────────────────────────
    scenario_key = _LABEL_TO_SCENARIO.get(scenario_label, config.DEFAULT_SCENARIO)

    # ── Target override ───────────────────────────────────────────────────────
    if target_override == "Auto (from direction)":
        target_register = None   # let direction/scenario logic decide
    else:
        target_register = target_override.lower()  # "formal" or "informal"

    # ── Cloud-off adjustments ─────────────────────────────────────────────────
    effective_asr = asr_backend
    effective_tts = tts_backend
    cloud_note = ""

    if not cloud_on:
        effective_asr = "whisper"
        if effective_tts == "timbre":
            # Timbre is unavailable offline; switch to Parler
            effective_tts = "parler"
            cloud_note = " (Timbre disabled in offline mode; switched to Parler)"
        # MMS and Parler stay as requested

    # ── Language / voice ─────────────────────────────────────────────────────
    if language_mode == "Auto (recommended)":
        # Will be computed from normalized text inside run_pipeline via language_hint
        explicit_language = None
    else:
        explicit_language = language_mode  # "hi-IN" or "hi-en"

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_pipeline(
            audio_path=audio_path,
            target_register=target_register,
            direction=direction,
            scenario=scenario_key,
            asr_backend=effective_asr,
            tts_backend=effective_tts,
            allow_asr_fallback=True,
            synthesize=True,
            out_dir="ui_out",
            allow_cloud=cloud_on,
        )
    except Exception as exc:
        err_msg = friendly_error(exc)
        ui["error_message"] = redact(err_msg) + cloud_note
        session_log.append({
            "turn": turn_number,
            "direction": direction,
            "cleaned": "",
            "transformed": "",
            "asr_backend": effective_asr,
            "tts_backend": None,
            "total_ms": 0.0,
            "error": str(exc)[:200],
        })
        ui["session_log"] = session_log
        ui["log_text"] = format_log(session_log)
        return ui

    # ── Populate UI fields ────────────────────────────────────────────────────
    ui["raw"] = redact(result.get("raw_text", ""))
    ui["cleaned"] = redact(result.get("cleaned_text", ""))
    ui["detected"] = result.get("detected_register", "")
    ui["target"] = result.get("target_register", "")
    ui["transformed"] = redact(result.get("transformed_text", ""))
    ui["normalized"] = redact(result.get("normalized_text", ""))
    ui["output_audio"] = result.get("output_audio_path")

    # Status line
    ui["status_line"] = redact(build_status_line(result, cloud_on=cloud_on)) + cloud_note

    # Script warning
    script_status = result.get("script_status", "ok")
    if script_status in ("mixed", "latin"):
        ui["script_warning"] = (
            "⚠️ Transcript contains Latin letters; "
            "the register rules may not handle these words correctly."
        )

    # Error / warning from pipeline
    if result.get("error"):
        ui["error_message"] = redact(friendly_error(result))

    # ── Append to session log ─────────────────────────────────────────────────
    session_log.append({
        "turn": turn_number,
        "direction": direction,
        "cleaned": result.get("cleaned_text", ""),
        "transformed": result.get("transformed_text", ""),
        "asr_backend": result.get("asr_backend", effective_asr),
        "tts_backend": result.get("tts_backend"),
        "total_ms": result.get("total_ms", 0.0),
        "error": result.get("error"),
    })
    ui["session_log"] = session_log
    ui["log_text"] = format_log(session_log)

    return ui


# ── Helpers ───────────────────────────────────────────────────────────────────

def _direction_letter(direction_label: str) -> str:
    """Extract 'A' or 'B' from the direction dropdown label."""
    label = (direction_label or "A").strip().upper()
    if label.startswith("A"):
        return "A"
    if label.startswith("B"):
        return "B"
    # Fallback: try first character
    return label[0] if label else "A"


__all__ = [
    "redact",
    "build_status_line",
    "friendly_error",
    "process_turn",
    "format_log",
    "ASR_CHOICES",
    "TTS_CHOICES",
    "UI_TIMBRE_VOICES",
    "LANGUAGE_CHOICES",
    "DIRECTION_CHOICES",
    "TARGET_CHOICES",
    "SCENARIO_LABELS",
    "DEFAULT_SCENARIO_LABEL",
]
