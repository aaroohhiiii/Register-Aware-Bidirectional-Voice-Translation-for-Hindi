"""conversation.py — Conversation turn-by-turn dialogue runner.

Public API
----------
run_conversation(turns, scenario=None, asr_backend=None, tts_backend=None,
                 allow_asr_fallback=True, synthesize=True, out_dir="e2e_out/convo") -> dict

CLI Usage
---------
    python conversation.py --scenario auto_driver --turns A:clips/c01.wav B:clips/c02.wav [--no-tts]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 stdout/stderr on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asr
import config
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


def run_conversation(
    turns: List[Dict[str, str]],
    scenario: Optional[str] = None,
    asr_backend: Optional[str] = None,
    tts_backend: Optional[str] = None,
    allow_asr_fallback: bool = True,
    synthesize: bool = True,
    out_dir: str | Path = "e2e_out/convo",
) -> Dict[str, Any]:
    """Run turn-by-turn dialogue through the pipeline.

    Parameters
    ----------
    turns : list[dict]
        List of turn dicts: [{"direction": "A"|"B", "audio_path": "..."}, ...]
    scenario : str | None, optional
        Scenario key in config.SCENARIOS. Defaults to config.DEFAULT_SCENARIO.
    asr_backend : str | None, optional
        "whisper" or "prisma".
    tts_backend : str | None, optional
        TTS backend override.
    allow_asr_fallback : bool, optional
        Allow Prisma -> Whisper fallback.
    synthesize : bool, optional
        Synthesize audio outputs.
    out_dir : str | Path, optional
        Directory to save dialogue audio files.

    Returns
    -------
    dict
        {"scenario": str, "turns": list[dict], "transcript": list[str]}
    """
    scen_used = scenario or getattr(config, "DEFAULT_SCENARIO", "friend")
    turn_results: List[Dict[str, Any]] = []
    transcript_lines: List[str] = []

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for idx, turn in enumerate(turns, 1):
        dir_val = str(turn.get("direction", "A")).strip().upper()
        audio_p = turn.get("audio_path", "")

        try:
            res = run_pipeline(
                audio_path=audio_p,
                direction=dir_val,
                scenario=scen_used,
                asr_backend=asr_backend,
                tts_backend=tts_backend,
                allow_asr_fallback=allow_asr_fallback,
                synthesize=synthesize,
                out_dir=out_dir_path,
            )
        except Exception as exc:
            # Capture error without failing the whole conversation
            res = {
                "audio_path": audio_p,
                "direction": dir_val,
                "scenario": scen_used,
                "raw_text": "",
                "cleaned_text": "",
                "transformed_text": "",
                "error": f"turn_failed: {exc}",
            }

        turn_results.append(res)

        cleaned = res.get("cleaned_text", "")
        transformed = res.get("transformed_text", "")

        if dir_val == "A":
            line = f"[A] user said: {cleaned} -> interlocutor hears: {transformed}"
        else:
            line = f"[B] interlocutor said: {cleaned} -> user hears: {transformed}"

        transcript_lines.append(line)

    return {
        "scenario": scen_used,
        "turns": turn_results,
        "transcript": transcript_lines,
    }


def main():
    parser = argparse.ArgumentParser(description="Run dialogue turn sequence through the pipeline.")
    parser.add_argument("--scenario", default=None, help="Persona (friend, auto_driver, receptionist, doctor)")
    parser.add_argument("--turns", nargs="+", required=True, help="Turn specs in format DIR:path/to/clip.wav (e.g. A:c01.wav B:c02.wav)")
    parser.add_argument("--asr", choices=["whisper", "prisma"], default=None, help="ASR backend choice")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS synthesis")
    parser.add_argument("--out-dir", default="e2e_out/convo", help="Output directory")

    args = parser.parse_args()

    parsed_turns = []
    for turn_spec in args.turns:
        parts = turn_spec.split(":", 1)
        if len(parts) != 2:
            print(f"❌ Invalid turn format: {turn_spec!r}. Expected DIR:PATH (e.g. A:clips/c01.wav)")
            sys.exit(1)
        parsed_turns.append({"direction": parts[0], "audio_path": parts[1]})

    # Preload ASR model
    asr.preload(args.asr or getattr(config, "ASR_BACKEND", "whisper"))

    convo_res = run_conversation(
        turns=parsed_turns,
        scenario=args.scenario,
        asr_backend=args.asr,
        synthesize=not args.no_tts,
        out_dir=args.out_dir,
    )

    print("\n" + "=" * 60)
    print(f" 🗣️ CONVERSATION TRANSCRIPT ({convo_res['scenario'].upper()})")
    print("=" * 60)
    for line in convo_res["transcript"]:
        print(line)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
