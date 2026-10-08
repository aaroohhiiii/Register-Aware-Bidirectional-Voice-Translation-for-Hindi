"""pipeline_text.py — End-to-end text transformation and TTS pipeline for Hindi.

Wires together:
1. Register detection (Aarohi's register_detector.py)
2. Register transformation (Aarohi's register_transformer.py)
3. TTS-ready text normalization (normalize.py)
4. Language hint selection (language_hint.py)
5. Voice synthesis (tts.py)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import tts
from language_hint import pick_tts_language
from normalize import normalize_for_tts
from register_detector import detect_register, detect_register_details
from register_transformer import transform


@dataclass
class PipelineResult:
    source_text: str
    source_register: str
    target_register: str
    transformed_text: str
    normalized_text: str
    language_hint: str
    audio_path: str
    backend_used: Optional[str]
    elapsed_total_s: float
    elapsed_transform_s: float
    elapsed_normalize_s: float
    elapsed_tts_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pipeline(
    text: str,
    target_register: str,
    out_path: str | Path,
    *,
    gender: Optional[str] = None,
    backend: Optional[str] = None,
    speed: Optional[float] = None,
    voice: Optional[str] = None,
    language: Optional[str] = None,
) -> PipelineResult:
    """Transform Hindi text into target_register, normalize, and synthesise to audio.

    Parameters
    ----------
    text : str
        Input Hindi text.
    target_register : str
        Target register tier (e.g. "Formal", "Respectful Casual", "Full Informal").
    out_path : str | Path
        Destination .wav file path.
    gender : str | None, optional
        Gender override ("masculine", "feminine", or None).
    backend : str | None, optional
        TTS backend override ("timbre", "parler", "mms").
    speed : float | None, optional
        Speech speed multiplier override.
    voice : str | None, optional
        Timbre voice override.
    language : str | None, optional
        Timbre language override. If None, automatically detected via pick_tts_language.

    Returns
    -------
    PipelineResult
        Structured result containing text stages, metrics, and output path.
    """
    t_start = time.perf_counter()

    # 1. Register Detection
    source_register = detect_register(text)

    # 2. Register Transformation
    t_tr_start = time.perf_counter()
    transformed_text = transform(text, target_tier=target_register, gender=gender)
    t_tr_end = time.perf_counter()

    # 3. Normalization for TTS
    t_norm_start = time.perf_counter()
    normalized_text = normalize_for_tts(transformed_text)
    t_norm_end = time.perf_counter()

    # 4. Language Selection / Hint
    lang_to_use = language or pick_tts_language(normalized_text)

    # 5. TTS Synthesis
    t_tts_start = time.perf_counter()
    saved_path = tts.speak(
        normalized_text,
        out_path=out_path,
        backend=backend,
        speed=speed,
        voice=voice,
        language=lang_to_use,
    )
    t_tts_end = time.perf_counter()
    t_end = time.perf_counter()

    return PipelineResult(
        source_text=text,
        source_register=source_register,
        target_register=target_register,
        transformed_text=transformed_text,
        normalized_text=normalized_text,
        language_hint=lang_to_use,
        audio_path=saved_path,
        backend_used=tts.get_last_backend(),
        elapsed_total_s=round(t_end - t_start, 4),
        elapsed_transform_s=round(t_tr_end - t_tr_start, 4),
        elapsed_normalize_s=round(t_norm_end - t_norm_start, 4),
        elapsed_tts_s=round(t_tts_end - t_tts_start, 4),
    )


__all__ = ["run_pipeline", "PipelineResult"]
