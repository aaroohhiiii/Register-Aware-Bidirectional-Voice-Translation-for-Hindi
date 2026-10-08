"""asr.py — ASR dispatcher for Whisper (local) and Gnani Prisma (cloud STT).

Public API
----------
transcribe(audio_path, backend=None, allow_fallback=True, allow_cloud=True) -> dict:
    Returns {
        "text": str,
        "asr_backend_requested": str,
        "asr_backend_used": str,
        "asr_ms": float,
        "fallback_used": bool,
        "fallback_reason": Optional[str],
        "error": Optional[str],
    }
    When allow_cloud=False, Prisma is never called; Whisper is always used.

preload(backend=None) -> None:
    Preload and cache local models (Whisper) once at startup.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from dotenv import load_dotenv

import config

load_dotenv()

logger = logging.getLogger(__name__)

_PRISMA_URL = "https://api.vachana.ai/stt/v3"
_WHISPER_MODEL_ID = "vasista22/whisper-hindi-small"
_TARGET_SR = 16_000
_MAX_PRISMA_DURATION_S = 60.0

# ─── Whisper Cache ───────────────────────────────────────────────────────────
_whisper_processor: Optional[Any] = None
_whisper_model: Optional[Any] = None
_whisper_device: Optional[torch.device] = None

# Flag to print top-level Prisma response keys on the very first call
_prisma_first_call_done = False


def _get_device() -> torch.device:
    """Select best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preload(backend: Optional[str] = None) -> None:
    """Preload and cache the Whisper processor & model once at startup."""
    global _whisper_processor, _whisper_model, _whisper_device

    backend_clean = (backend or "whisper").lower()
    if backend_clean not in ("whisper", "all"):
        return

    if _whisper_model is not None:
        return

    device = _get_device()
    logger.info("Preloading Whisper model %s on %s...", _WHISPER_MODEL_ID, device)

    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(_WHISPER_MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(_WHISPER_MODEL_ID)

        # Fix compatibility with older 4.27 checkpoint in Transformers 4.40+
        model.generation_config.suppress_tokens = None
        model.to(device)
        model.eval()

        _whisper_processor = processor
        _whisper_model = model
        _whisper_device = device
        logger.info("Whisper model preloaded successfully on %s.", device)
    except Exception as exc:
        safe_exc = _sanitize_error(str(exc))
        raise RuntimeError(f"Failed to load Whisper model {_WHISPER_MODEL_ID}: {safe_exc}") from exc


def _sanitize_error(msg: str) -> str:
    """Ensure GNANI_API_KEY is never leaked in error messages or logs."""
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "***")
    return msg


def prepare_audio(audio_path: str | Path) -> Tuple[str, float]:
    """Read audio, convert to mono 16 kHz WAV, write to temp file.

    Returns
    -------
    tuple : (temp_wav_path: str, duration_s: float)

    Raises
    ------
    FileNotFoundError or RuntimeError if file cannot be read.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        try:
            audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1, dtype=np.float32)
        except Exception:
            mono, orig_sr = librosa.load(str(path), sr=None, mono=True)
            mono = mono.astype(np.float32)

        if orig_sr != _TARGET_SR:
            mono = librosa.resample(mono, orig_sr=orig_sr, target_sr=_TARGET_SR).astype(np.float32, copy=False)

        duration_s = round(len(mono) / _TARGET_SR, 3)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        sf.write(temp_path, mono, _TARGET_SR, format="WAV", subtype="PCM_16")
        return temp_path, duration_s
    except Exception as exc:
        raise RuntimeError(f"Could not prepare audio file {path.name}: {_sanitize_error(str(exc))}") from exc


def _transcribe_whisper(temp_wav_path: str) -> str:
    """Run cached Whisper inference on standardized 16 kHz mono WAV file."""
    preload("whisper")
    assert _whisper_processor is not None and _whisper_model is not None and _whisper_device is not None

    waveform, sr = sf.read(temp_wav_path, dtype="float32")
    inputs = _whisper_processor(
        waveform,
        sampling_rate=_TARGET_SR,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(_whisper_device)
    forced_ids = _whisper_processor.get_decoder_prompt_ids(language="hi", task="transcribe")

    with torch.inference_mode():
        predicted_ids = _whisper_model.generate(
            input_features,
            forced_decoder_ids=forced_ids,
        )
    text = _whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text


def _transcribe_prisma(temp_wav_path: str, duration_s: float) -> str:
    """Call Gnani Prisma v3 REST endpoint with 16 kHz mono WAV file."""
    global _prisma_first_call_done

    if duration_s > _MAX_PRISMA_DURATION_S:
        raise ValueError("too_long_for_prisma")

    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing_api_key")

    headers = {
        "X-API-Key-ID": api_key,
    }

    with open(temp_wav_path, "rb") as f:
        wav_bytes = f.read()

    files = {
        "audio_file": ("audio.wav", wav_bytes, "audio/wav"),
    }
    data = {
        "language_code": getattr(config, "PRISMA_LANGUAGE", "hi-IN"),
    }

    timeout_s = getattr(config, "ASR_TIMEOUT_S", 15)
    try:
        resp = requests.post(_PRISMA_URL, headers=headers, files=files, data=data, timeout=timeout_s)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Prisma request timed out after {timeout_s}s.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Prisma network error: {_sanitize_error(str(exc))}") from exc

    if resp.status_code != 200:
        trunc_body = resp.text[:200]
        raise RuntimeError(f"Prisma HTTP {resp.status_code}: {_sanitize_error(trunc_body)}")

    try:
        resp_json = resp.json()
    except Exception as exc:
        trunc_body = resp.text[:200]
        raise RuntimeError(f"Prisma returned invalid JSON: {_sanitize_error(trunc_body)}") from exc

    if not _prisma_first_call_done:
        print(f"\n[Prisma] First call response top-level keys: {list(resp_json.keys())}\n")
        _prisma_first_call_done = True

    # Parse transcript field
    raw_transcript: Optional[str] = None
    if "transcript" in resp_json:
        raw_transcript = str(resp_json["transcript"]).strip()
    elif "text" in resp_json:
        raw_transcript = str(resp_json["text"]).strip()
    elif "data" in resp_json and isinstance(resp_json["data"], dict) and "transcript" in resp_json["data"]:
        raw_transcript = str(resp_json["data"]["transcript"]).strip()

    if raw_transcript is None:
        raise RuntimeError(f"Could not find 'transcript' field in Prisma response keys: {list(resp_json.keys())}")

    if not raw_transcript:
        raise RuntimeError("empty_transcript")

    return raw_transcript


def transcribe(
    audio_path: str | Path,
    backend: Optional[str] = None,
    allow_fallback: bool = True,
    allow_cloud: bool = True,
) -> dict[str, Any]:
    """Transcribe audio file using Whisper or Gnani Prisma with optional fallback.

    Parameters
    ----------
    audio_path : str | Path
        Path to input audio file.
    backend : str | None, optional
        "whisper" or "prisma". Defaults to config.ASR_BACKEND.
    allow_fallback : bool, optional
        If True and requested backend fails, fall back to Whisper (if requested was Prisma).
    allow_cloud : bool, optional
        If False, forces Whisper backend regardless of `backend` argument;
        Prisma is never called and no network request is made.

    Returns
    -------
    dict
        {
            "text": str,
            "asr_backend_requested": str,
            "asr_backend_used": str,
            "asr_ms": float,
            "fallback_used": bool,
            "fallback_reason": Optional[str],
            "error": Optional[str]
        }
    """
    requested_backend = (backend or getattr(config, "ASR_BACKEND", "whisper")).lower()

    # ── Cloud-off guard: never call Prisma if allow_cloud is False ───────────
    if not allow_cloud:
        requested_backend = "whisper"

    if requested_backend not in ("whisper", "prisma"):
        return {
            "text": "",
            "asr_backend_requested": requested_backend,
            "asr_backend_used": requested_backend,
            "asr_ms": 0.0,
            "fallback_used": False,
            "fallback_reason": None,
            "error": f"Unknown backend: {requested_backend}. Choose 'whisper' or 'prisma'.",
        }

    t0 = time.perf_counter()
    temp_wav_path: Optional[str] = None
    try:
        temp_wav_path, duration_s = prepare_audio(audio_path)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        safe_msg = _sanitize_error(str(exc))
        return {
            "text": "",
            "asr_backend_requested": requested_backend,
            "asr_backend_used": requested_backend,
            "asr_ms": latency_ms,
            "fallback_used": False,
            "fallback_reason": None,
            "error": safe_msg,
        }

    try:
        if requested_backend == "whisper":
            text = _transcribe_whisper(temp_wav_path)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            res = {
                "text": text,
                "asr_backend_requested": "whisper",
                "asr_backend_used": "whisper",
                "asr_ms": latency_ms,
                "fallback_used": False,
                "fallback_reason": None,
                "error": None,
            }
        else:
            # Prisma requested
            fallback_reason: Optional[str] = None
            try:
                text = _transcribe_prisma(temp_wav_path, duration_s)
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                res = {
                    "text": text,
                    "asr_backend_requested": "prisma",
                    "asr_backend_used": "prisma",
                    "asr_ms": latency_ms,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "error": None,
                }
            except Exception as prisma_exc:
                safe_prisma_msg = _sanitize_error(str(prisma_exc))
                fallback_reason = safe_prisma_msg
                logger.warning("Prisma ASR failed: %s", safe_prisma_msg)

                if not allow_fallback:
                    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                    res = {
                        "text": "",
                        "asr_backend_requested": "prisma",
                        "asr_backend_used": "prisma",
                        "asr_ms": latency_ms,
                        "fallback_used": False,
                        "fallback_reason": safe_prisma_msg,
                        "error": safe_prisma_msg,
                    }
                else:
                    # Fall back to Whisper
                    logger.info("Falling back to Whisper for ASR...")
                    w_text = _transcribe_whisper(temp_wav_path)
                    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                    res = {
                        "text": w_text,
                        "asr_backend_requested": "prisma",
                        "asr_backend_used": "whisper",
                        "asr_ms": latency_ms,
                        "fallback_used": True,
                        "fallback_reason": fallback_reason,
                        "error": None,
                    }
        return res
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass


__all__ = ["transcribe", "prepare_audio", "preload"]
