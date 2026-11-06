"""pipeline.py — First full end-to-end speech transformation pipeline for Hindi.

Flow:
 1. ASR (Whisper / Gnani Prisma with fallback)
 2. Script inspection (check_script; log_flagged if non-ok, continue)
 3. Cleanup (clean)
 4. Register detection (detect_register)
 5. Target register resolution (formal <-> informal opposite rule if None)
 6. Register transformation (transform)
 7. TTS synthesis (normalize_for_tts -> pick_tts_language -> speak)

CLI Usage:
    python pipeline.py --audio clips/c01.wav [--target informal|formal] [--asr whisper|prisma] [--no-tts]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Force UTF-8 stdout/stderr on Windows to prevent UnicodeEncodeError with emojis/Devanagari
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asr
import cleanup
import config
import directions
import language_hint
import normalize
import register_detector
import register_transformer
import script_check
import tts

logger = logging.getLogger(__name__)

# Standard register tier groupings
_FORMAL_TIERS = {"Formal", "Full Formal", "Formal-Warm", "Reverse Formal", "formal"}
_INFORMAL_TIERS = {"Full Informal", "Respectful Casual", "informal"}
# Exact tier names (case-sensitive). Lowercase "formal"/"informal" are the
# coarse groups; "Formal" is the single tier.
_EXACT_TIERS = {"Full Informal", "Respectful Casual", "Formal", "Full Formal", "Formal-Warm", "Reverse Formal"}


def _resolve_target_register(detected: str, target: Optional[str]) -> tuple[str, bool]:
    """Resolve target register tier and determine if transformation should be skipped.

    Returns
    -------
    tuple : (target_register_str, transform_skipped_bool)
    """
    if target is not None and target.strip() in _EXACT_TIERS:
        # An exact tier is only skipped when the speaker already used it.
        exact = target.strip()
        return exact, detected == exact

    if target is None:
        if detected in _FORMAL_TIERS:
            target_reg = "informal"
        elif detected in _INFORMAL_TIERS:
            target_reg = "formal"
        else:
            return "unknown", True
    else:
        target_reg = target.strip().lower()

    # Determine if detected matches target category
    det_lower = detected.strip().lower()
    if det_lower == target_reg:
        return target_reg, True
    if target_reg == "formal" and detected in _FORMAL_TIERS:
        return target_reg, True
    if target_reg == "informal" and detected in _INFORMAL_TIERS:
        return target_reg, True

    return target_reg, False


def _to_transform_tier(target_reg: str) -> str:
    """Map target_reg string to exact tier string expected by register_transformer.py."""
    if target_reg.strip() in _EXACT_TIERS:
        return target_reg.strip()
    tr = target_reg.strip().lower()
    if tr in ("informal", "full informal"):
        return "Full Informal"
    if tr == "respectful casual":
        return "Respectful Casual"
    if tr == "full formal":
        return "Full Formal"
    if tr == "formal":
        return "Formal"
    if tr in ("formal-warm", "reverse formal"):
        # Match case
        for tier in ("Formal-Warm", "Reverse Formal"):
            if tier.lower() == tr:
                return tier
    return target_reg


def run_pipeline(
    audio_path: str | Path,
    target_register: Optional[str] = None,
    direction: Optional[str] = None,
    scenario: Optional[str] = None,
    asr_backend: Optional[str] = None,
    tts_backend: Optional[str] = None,
    allow_asr_fallback: bool = True,
    synthesize: bool = True,
    out_dir: str | Path = "e2e_out",
    allow_cloud: bool = True,
) -> dict[str, Any]:
    """Run full end-to-end voice translation pipeline.

    Returns dict containing all stages, metrics, and output paths.
    """
    t_start = time.perf_counter()

    audio_path_str = str(audio_path)
    base_name = Path(audio_path).stem

    # Resolve target register source
    if target_register is not None:
        resolved_target = target_register
        target_source = "explicit"
    elif direction is not None:
        resolved_target = directions.direction_to_target(direction, scenario=scenario)
        target_source = "direction"
    else:
        resolved_target = None
        target_source = "legacy_opposite"

    # Defaults & metrics initialization
    result: dict[str, Any] = {
        "audio_path": audio_path_str,
        "direction": direction,
        "scenario": scenario,
        "target_source": target_source,
        "raw_text": "",
        "script_status": "empty",
        "latin_words": [],
        "cleaned_text": "",
        "detected_register": "unknown",
        "target_register": resolved_target,  # None in legacy_opposite mode; resolved after detect
        "transform_skipped": False,
        "skip_reason": None,
        "transform_failed": False,
        "transformed_text": "",
        "normalized_text": "",
        "tts_language": "",
        "output_audio_path": None,
        "asr_backend": (asr_backend or getattr(config, "ASR_BACKEND", "whisper")).lower(),
        "tts_backend": None,
        "asr_ms": 0.0,
        "clean_ms": 0.0,
        "detect_ms": 0.0,
        "transform_ms": 0.0,
        "normalize_ms": 0.0,
        "tts_ms": 0.0,
        "total_ms": 0.0,
        "fallback_used": False,
        "asr_fallback_reason": None,
        "tts_fallback_used": False,
        "error": None,
    }

    # ── Stage 1: ASR Transcription ───────────────────────────────────────────
    try:
        asr_res = asr.transcribe(
            audio_path,
            backend=asr_backend,
            allow_fallback=allow_asr_fallback,
            allow_cloud=allow_cloud,
        )
        result["raw_text"] = asr_res.get("text", "")
        result["asr_backend"] = asr_res.get("asr_backend_used", result["asr_backend"])
        result["asr_ms"] = asr_res.get("asr_ms", 0.0)
        asr_fallback_used = asr_res.get("fallback_used", False)
        result["asr_fallback_reason"] = asr_res.get("fallback_reason")

        if asr_res.get("error"):
            result["error"] = asr_res["error"]
            if not result["raw_text"]:
                result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                return result
    except Exception as exc:
        result["error"] = f"asr_failed: {exc}"
        result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    raw_text = result["raw_text"]
    if not raw_text.strip():
        result["error"] = result["error"] or "empty_asr_transcript"
        result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    # ── Stage 2: Script Inspection ───────────────────────────────────────────
    try:
        s_res = script_check.check_script(raw_text)
        result["script_status"] = s_res.get("status", "ok")
        result["latin_words"] = s_res.get("latin_words", [])

        if result["script_status"] != "ok":
            script_check.log_flagged({
                "clip_id": base_name,
                "backend": result["asr_backend"],
                "raw_text": raw_text,
                "cleaned_text": "",
                "status": result["script_status"],
                "latin_words": result["latin_words"],
                "note": "script_check_flagged",
            })
    except Exception as exc:
        logger.warning("script_check error: %s", exc)

    # ── Stage 3: Clean Transcript ───────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        cleaned_text = cleanup.clean(raw_text, backend=result["asr_backend"])
        result["clean_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["cleaned_text"] = cleaned_text
    except Exception as exc:
        result["clean_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["error"] = result["error"] or f"clean_failed: {exc}"
        result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    if not cleaned_text.strip():
        result["error"] = result["error"] or "empty_after_cleanup"
        result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    # ── Stage 4: Detect Register ─────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        det_reg = register_detector.detect_register(cleaned_text)
        result["detect_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["detected_register"] = det_reg
    except Exception as exc:
        result["detect_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["error"] = result["error"] or f"detect_failed: {exc}"
        det_reg = "unknown"
        result["detected_register"] = det_reg

    # ── Stage 5 & 6: Resolve Target Register & Transform ───────────────────
    # Pass `resolved_target` (may be None in legacy_opposite mode) so the
    # opposite-register logic inside _resolve_target_register fires correctly.
    tgt_reg, skip_tr = _resolve_target_register(det_reg, resolved_target)
    result["target_register"] = tgt_reg

    # With an exact tier we still transform when detection is unknown
    # (e.g. no pronoun), so the persona's markers such as जी are applied.
    if det_reg == "unknown" and tgt_reg not in _EXACT_TIERS:
        result["transform_skipped"] = True
        result["skip_reason"] = "register_unknown"
        result["transformed_text"] = cleaned_text
        result["transform_failed"] = False
        result["transform_ms"] = 0.0
    elif skip_tr or tgt_reg == "unknown":
        result["transform_skipped"] = True
        result["skip_reason"] = "target_equals_detected" if skip_tr else "target_unknown"
        result["transformed_text"] = cleaned_text
        result["transform_failed"] = False
        result["transform_ms"] = 0.0
    else:
        result["transform_skipped"] = False
        result["skip_reason"] = None
        t0 = time.perf_counter()
        try:
            transform_tier = _to_transform_tier(tgt_reg)
            trans_text = register_transformer.transform(cleaned_text, target_tier=transform_tier)
            result["transform_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            if not trans_text.strip():
                result["transformed_text"] = cleaned_text
                result["transform_failed"] = True
                result["error"] = result["error"] or "transform_returned_empty"
            else:
                result["transformed_text"] = trans_text
                result["transform_failed"] = False
        except Exception as exc:
            result["transform_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["transformed_text"] = cleaned_text
            result["transform_failed"] = True
            result["error"] = result["error"] or f"transform_failed: {exc}"

    transformed_text = result["transformed_text"]

    # ── Stage 7: Normalization & TTS Synthesis ────────────────────────────────
    t0 = time.perf_counter()
    try:
        norm_text = normalize.normalize_for_tts(transformed_text)
        result["normalize_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["normalized_text"] = norm_text
    except Exception as exc:
        result["normalize_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["normalized_text"] = transformed_text
        result["error"] = result["error"] or f"normalize_failed: {exc}"

    norm_text = result["normalized_text"]
    result["tts_language"] = language_hint.pick_tts_language(norm_text) if norm_text else "hi-IN"

    if synthesize and norm_text.strip():
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        tgt_tag = tgt_reg.replace(" ", "_")
        out_wav_name = f"{base_name}_{result['asr_backend']}_{tgt_tag}.wav"
        out_wav_path = out_dir_path / out_wav_name

        req_tts_backend = (tts_backend or getattr(config, "TTS_BACKEND", "timbre")).lower()

        t0 = time.perf_counter()
        try:
            saved_file = tts.speak(
                norm_text,
                out_path=out_wav_path,
                backend=req_tts_backend,
                language=result["tts_language"],
                allow_cloud=allow_cloud,
            )
            result["tts_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["output_audio_path"] = str(Path(saved_file).resolve())
            used_tts = (tts.get_last_backend() or req_tts_backend).lower()
            result["tts_backend"] = used_tts
            result["tts_fallback_used"] = (req_tts_backend != used_tts)
        except Exception as exc:
            result["tts_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["error"] = result["error"] or f"tts_failed: {exc}"

    # Overall fallback status calculation
    result["fallback_used"] = bool(asr_fallback_used or result["tts_fallback_used"])

    result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Hindi register conversion pipeline on an audio clip.")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--target", choices=["informal", "formal"], default=None, help="Target register")
    parser.add_argument("--direction", choices=["A", "B", "a", "b"], default=None, help="Direction A or B")
    parser.add_argument("--scenario", default=None, help="Persona (friend, auto_driver, receptionist, doctor)")
    parser.add_argument("--asr", choices=["whisper", "prisma"], default=None, help="ASR backend choice")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS synthesis")
    parser.add_argument("--out-dir", default="e2e_out", help="Output directory for audio and logs")

    args = parser.parse_args()

    # Preload models once at start
    logger.info("Preloading models...")
    asr.preload(args.asr or getattr(config, "ASR_BACKEND", "whisper"))

    req_tts_backend = getattr(config, "TTS_BACKEND", "timbre").lower()
    if not args.no_tts and req_tts_backend in ("parler", "mms"):
        if req_tts_backend == "parler":
            try:
                tts._load_parler()
            except Exception as e:
                logger.warning("Parler preload warning: %s", e)
        elif req_tts_backend == "mms":
            try:
                tts._load_mms()
            except Exception as e:
                logger.warning("MMS preload warning: %s", e)

    res = run_pipeline(
        audio_path=args.audio,
        target_register=args.target,
        direction=args.direction,
        scenario=args.scenario,
        asr_backend=args.asr,
        synthesize=not args.no_tts,
        out_dir=args.out_dir,
    )

    print("\n" + "=" * 60)
    print(" 🎙️ PIPELINE STAGE SUMMARY")
    print("=" * 60)
    print(f"Input Audio    : {res['audio_path']}")
    print(f"ASR Backend    : {res['asr_backend']} ({res['asr_ms']} ms)")
    print(f"Raw Text       : {res['raw_text']}")
    print(f"Script Check   : {res['script_status']} (latin: {res['latin_words']})")
    print(f"Cleaned Text   : {res['cleaned_text']} ({res['clean_ms']} ms)")
    print(f"Detected Reg   : {res['detected_register']} ({res['detect_ms']} ms)")
    print(f"Target Reg     : {res['target_register']} (skipped: {res['transform_skipped']}, failed: {res['transform_failed']})")
    print(f"Transformed    : {res['transformed_text']} ({res['transform_ms']} ms)")
    print(f"Normalized     : {res['normalized_text']} ({res['normalize_ms']} ms)")
    print(f"TTS Language   : {res['tts_language']}")
    print(f"TTS Backend    : {res['tts_backend']} ({res['tts_ms']} ms, fallback: {res['tts_fallback_used']})")
    print(f"Output Audio   : {res['output_audio_path']}")
    print(f"Total Time     : {res['total_ms']} ms")
    if res['error']:
        print(f"⚠️ Error        : {res['error']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
