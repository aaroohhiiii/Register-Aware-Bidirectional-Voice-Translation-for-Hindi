"""tts.py — Hindi Text-to-Speech dispatcher.

Public API
----------
speak(text, out_path, backend=None, speed=None, voice=None, allow_cloud=True) -> str
    Synthesise Hindi text and write a WAV file to *out_path*.
    Returns the absolute path of the written file.
    When allow_cloud=False, the Timbre cloud backend is excluded from the
    trial order; only Parler and MMS (local) are attempted.

last_backend_used : str | None
    Updated after every call to speak(); holds the name of the backend
    ("timbre", "parler", or "mms") that actually produced the audio.

Three backends are supported
-----------------------------
1. timbre  — Gnani AI cloud (primary, requires GNANI_API_KEY in .env)
2. parler  — ai4bharat/indic-parler-tts, local CPU inference
3. mms     — facebook/mms-tts-hin, local CPU inference (VITS)

If the chosen backend raises any exception the module tries the next one in
config.TTS_FALLBACK_ORDER, logging a warning without ever printing the API key.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Optional

# Load environment variables from .env (if present) before importing config,
# so that os.getenv("GNANI_API_KEY") works anywhere in this file.
from dotenv import load_dotenv

load_dotenv()

import config  # noqa: E402 — must follow load_dotenv

# ── Logging setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

# ── Module-level state ────────────────────────────────────────────────────────
last_backend_used: Optional[str] = None  # updated after every speak() call

# Lazy-loaded local model caches (populated on first use, never again).
_parler_model = None
_parler_tokenizer = None
_parler_description_tokenizer = None
_parler_device: Optional[str] = None
_parler_load_failed: bool = False   # set True after first load failure — skip retries

_mms_model = None
_mms_tokenizer = None
_mms_device: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal runs of spaces."""
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)
    return text


def _prepare_output(out_path: str | Path) -> Path:
    """Resolve *out_path* and create any missing parent directories."""
    dest = Path(out_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Timbre (Gnani AI cloud)
# ─────────────────────────────────────────────────────────────────────────────

_TIMBRE_URL = "https://api.vachana.ai/api/v1/tts/inference"


def _speak_timbre(
    text: str, dest: Path, speed: float, voice: str, language: Optional[str] = None
) -> None:
    """Call the Gnani Timbre v2.5 REST endpoint and write the audio to *dest*.

    Raises
    ------
    RuntimeError
        On any network error, HTTP non-2xx status, or missing API key.
        The error message NEVER contains the key value.
    """
    import requests  # already installed; import here so MMS/Parler-only use stays lightweight

    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GNANI_API_KEY is not set. Add it to your .env file and restart."
        )

    headers = {
        "X-API-Key-ID": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "voice": voice,
        "model": "timbre-v2.5",
        "language": language or config.TIMBRE_LANGUAGE,
        "speed": speed,
        "audio_config": {
            "sample_rate": 24000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }

    try:
        resp = requests.post(
            _TIMBRE_URL,
            json=payload,
            headers=headers,
            timeout=config.CLOUD_TIMEOUT_S,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timbre request timed out after {config.CLOUD_TIMEOUT_S}s."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Timbre network error: {exc}") from exc

    if resp.status_code != 200:
        # Log status + body for debugging, never the key.
        logger.error(
            "Timbre HTTP %d — response body: %s",
            resp.status_code,
            resp.text[:500],   # truncate very long error bodies
        )
        raise RuntimeError(
            f"Timbre returned HTTP {resp.status_code}. "
            "See the log above for the response body. "
            "Do NOT guess new field names — check the docs or fix the error shown."
        )

    dest.write_bytes(resp.content)
    logger.info("Timbre → %s (%.0f bytes)", dest.name, len(resp.content))


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Parler TTS (local)
# ─────────────────────────────────────────────────────────────────────────────

_PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"


def _load_parler():
    """Lazy-load and cache the Parler model + both tokenizers.

    After any load failure (e.g. gated-repo 401, import error) the flag
    `_parler_load_failed` is set so subsequent calls immediately raise
    without hitting the network again.
    """
    global _parler_model, _parler_tokenizer, _parler_description_tokenizer, _parler_device, _parler_load_failed

    if _parler_model is not None:
        return  # already loaded

    if _parler_load_failed:
        raise RuntimeError(
            "Parler model failed to load earlier in this process "
            "(gated repo / auth error). "
            "Run `huggingface-cli login` or set HF_TOKEN in .env, then restart."
        )
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device_name = config.PARLER_DEVICE  # "cpu" by default
    try:
        _parler_device = device_name
        logger.info("Loading Parler TTS model on %s (first call — takes ~30 s)…", device_name)

        _parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
            _PARLER_MODEL_ID
        ).to(device_name)
        _parler_model.eval()

        # The model uses two separate tokenizers: one for the prompt text, one
        # for the natural-language voice description.
        _parler_tokenizer = AutoTokenizer.from_pretrained(_PARLER_MODEL_ID)
        _parler_description_tokenizer = AutoTokenizer.from_pretrained(_PARLER_MODEL_ID)

        logger.info("Parler model loaded successfully.")
    except Exception as exc:
        # If MPS was configured, fall back silently to CPU.
        if device_name != "cpu":
            logger.warning(
                "Parler failed on %s (%s). Retrying on CPU.", device_name, exc
            )
            try:
                _parler_device = "cpu"
                _parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
                    _PARLER_MODEL_ID
                ).to("cpu")
                _parler_model.eval()
                _parler_tokenizer = AutoTokenizer.from_pretrained(_PARLER_MODEL_ID)
                _parler_description_tokenizer = AutoTokenizer.from_pretrained(_PARLER_MODEL_ID)
            except Exception as cpu_exc:
                _parler_load_failed = True
                raise RuntimeError(
                    f"Parler failed on CPU fallback too: {cpu_exc}"
                ) from cpu_exc
        else:
            # CPU load failed (e.g. gated 401) — cache the failure so we don't
            # hammer the network on every subsequent sentence.
            _parler_load_failed = True
            raise


def _pace_phrase(speed: float) -> str:
    """Return the pace wording that replaces 'moderate in pace' in the description."""
    if speed < 0.95:
        return "slightly slow pace"
    if speed > 1.05:
        return "slightly fast pace"
    return "moderate in pace"


def _speak_parler(
    text: str,
    dest: Path,
    speed: float,
    voice: str,
    language: Optional[str] = None,  # noqa: ARG001
) -> None:
    """Run local Parler-TTS inference and write a WAV to *dest*.

    The *voice* argument is ignored; voice character is encoded in the
    description string (PARLER_DESCRIPTION) set in config.py.
    Speed is approximated by adjusting the pace wording in that description.
    """
    import torch
    import soundfile as sf

    _load_parler()

    # Build a description that encodes the requested pace.
    description = config.PARLER_DESCRIPTION.replace(
        "moderate in pace", _pace_phrase(speed)
    )

    device = _parler_device

    desc_inputs = _parler_description_tokenizer(
        description, return_tensors="pt"
    ).to(device)
    text_inputs = _parler_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = _parler_model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=text_inputs.input_ids,
            prompt_mask=text_inputs.attention_mask if hasattr(text_inputs, "attention_mask") else None,
        ) if not hasattr(_parler_model, "generate") else _parler_model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=text_inputs.input_ids,
            prompt_attention_mask=text_inputs.attention_mask,
        )

    # generation shape: (1, T) — squeeze to 1-D numpy float32
    audio_array = generation.cpu().float().numpy().squeeze()
    sr = _parler_model.config.sampling_rate

    sf.write(str(dest), audio_array, sr, subtype="PCM_16")
    logger.info("Parler → %s  (sr=%d, samples=%d)", dest.name, sr, len(audio_array))


# ─────────────────────────────────────────────────────────────────────────────
# Backend: MMS (Meta Massively Multilingual Speech — VITS)
# ─────────────────────────────────────────────────────────────────────────────

def _load_mms():
    """Lazy-load and cache the MMS VITS model + tokenizer."""
    global _mms_model, _mms_tokenizer, _mms_device

    if _mms_model is not None:
        return

    import torch
    from transformers import AutoTokenizer, VitsModel

    device_name = "cpu"  # MMS on CPU; CUDA/MPS would need additional testing
    logger.info("Loading MMS-TTS-Hindi model on %s (first call — takes ~20 s)…", device_name)

    try:
        _mms_tokenizer = AutoTokenizer.from_pretrained(config.MMS_MODEL_ID)
        _mms_model = VitsModel.from_pretrained(config.MMS_MODEL_ID).to(device_name)
        _mms_model.eval()
        _mms_device = device_name
    except Exception as exc:
        raise RuntimeError(f"Failed to load MMS model: {exc}") from exc

    logger.info("MMS model loaded successfully.")


def _speak_mms(
    text: str,
    dest: Path,
    speed: float,
    voice: str,
    language: Optional[str] = None,  # noqa: ARG001
) -> None:
    """Run local MMS VITS inference and write a WAV to *dest*.

    MMS has no speed or voice control; those arguments are silently ignored.
    """
    import torch
    import soundfile as sf

    _load_mms()

    inputs = _mms_tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(_mms_device) for k, v in inputs.items()}

    with torch.no_grad():
        output = _mms_model(**inputs).waveform

    audio_array = output.cpu().numpy().squeeze()
    sr = getattr(_mms_model.config, "sampling_rate", 16000)

    sf.write(str(dest), audio_array, sr, subtype="PCM_16")
    logger.info("MMS → %s  (sr=%d, samples=%d)", dest.name, sr, len(audio_array))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher map
# ─────────────────────────────────────────────────────────────────────────────

_BACKEND_FN = {
    "timbre": _speak_timbre,
    "parler": _speak_parler,
    "mms":    _speak_mms,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def speak(
    text: str,
    out_path: str | Path,
    *,
    backend: Optional[str] = None,
    speed: Optional[float] = None,
    voice: Optional[str] = None,
    language: Optional[str] = None,
    allow_cloud: bool = True,
) -> str:
    """Synthesise Hindi *text* and write audio to *out_path*.

    Parameters
    ----------
    text      : Hindi sentence(s) to synthesise.
    out_path  : Destination .wav file path (parent dirs are created automatically).
    backend   : One of "timbre", "parler", "mms".  Defaults to config.TTS_BACKEND.
    speed     : Playback speed multiplier (0.5–2.0).  Defaults to config.TIMBRE_SPEED.
    voice     : Voice name (Timbre only).  Defaults to config.TIMBRE_VOICE.
    language  : Language code (Timbre only, e.g. "hi-IN" or "hi-en").
    allow_cloud : bool, optional
        If False, removes "timbre" from the trial order so no HTTP request
        is ever made. Falls through to Parler then MMS per the existing order.

    Returns
    -------
    str : Absolute path of the written WAV file.

    Raises
    ------
    ValueError
        If *text* is empty after stripping.
    RuntimeError
        If every backend in the fallback chain fails.
    """
    global last_backend_used

    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(text, str) or not _clean_text(text):
        raise ValueError(
            "speak(): 'text' must be a non-empty string. "
            f"Got: {text!r}"
        )

    clean = _clean_text(text)
    dest  = _prepare_output(out_path)

    # ── Resolve defaults ──────────────────────────────────────────────────────
    chosen_backend = (backend or config.TTS_BACKEND).lower()
    chosen_speed   = speed if speed is not None else config.TIMBRE_SPEED
    chosen_voice   = voice or config.TIMBRE_VOICE

    if chosen_backend not in _BACKEND_FN:
        raise ValueError(
            f"Unknown backend {chosen_backend!r}. "
            f"Choose from: {list(_BACKEND_FN)}"
        )

    # ── Build ordered trial list ──────────────────────────────────────────────
    # Try the requested backend first, then fall through TTS_FALLBACK_ORDER
    # (skipping the primary to avoid double-trying it).
    trial_order = [chosen_backend] + [
        b for b in config.TTS_FALLBACK_ORDER if b != chosen_backend
    ]

    # ── Cloud-off guard: strip Timbre so no HTTP call is ever made ─────────
    if not allow_cloud:
        trial_order = [b for b in trial_order if b != "timbre"]

    last_exc: Optional[Exception] = None

    for attempt, b in enumerate(trial_order):
        fn = _BACKEND_FN.get(b)
        if fn is None:
            logger.warning("Unknown backend %r in fallback order — skipping.", b)
            continue

        try:
            fn(clean, dest, chosen_speed, chosen_voice, language)
            last_backend_used = b
            if attempt > 0:
                logger.info(
                    "speak(): used fallback backend %r after primary %r failed.",
                    b,
                    chosen_backend,
                )
            return str(dest)

        except Exception as exc:
            # Log without including any secret values.
            safe_msg = str(exc)
            # Paranoid check: never log a string that looks like our API key.
            api_key_env = os.getenv("GNANI_API_KEY", "")
            if api_key_env and api_key_env in safe_msg:
                safe_msg = safe_msg.replace(api_key_env, "***")

            logger.warning(
                "Backend %r failed (attempt %d/%d): %s — trying next backend.",
                b,
                attempt + 1,
                len(trial_order),
                safe_msg,
            )
            last_exc = exc

    raise RuntimeError(
        f"All backends failed for text {clean[:40]!r}. "
        f"Last error: {last_exc}"
    ) from last_exc


# ── Convenience accessor ──────────────────────────────────────────────────────

def get_last_backend() -> Optional[str]:
    """Return the name of the backend that produced the most recent audio."""
    return last_backend_used


__all__ = ["speak", "get_last_backend", "last_backend_used"]


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sample_text   = sys.argv[1] if len(sys.argv) > 1 else "नमस्ते, आप कैसे हैं?"
    target_out    = sys.argv[2] if len(sys.argv) > 2 else "demo_output.wav"
    chosen        = sys.argv[3] if len(sys.argv) > 3 else None

    out_file = speak(sample_text, target_out, backend=chosen)
    print(f"Generated: {out_file}  (backend used: {last_backend_used})")
