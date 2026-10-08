"""tests/run_dirB.py — Direction B and Scripted Conversation Evaluation Runner.

Usage:
    python tests/run_dirB.py [--no-tts] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Force UTF-8 stdout/stderr on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jiwer
import soundfile as sf

import asr
import config
from conversation import run_conversation
from pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Rule-critical tokens and suffixes for rule_critical_diff_rate check
CRITICAL_PRONOUNS = {"आप", "तुम", "तू", "आपका", "तुम्हारा", "तेरा"}
CRITICAL_ENDINGS = {"रहे", "रहा", "रही", "हैं", "हो", "है", "इए", "ओ", "ना", "ए"}

FORMAL_CATEGORIES = {"formal", "Full Formal", "Formal", "Formal-Warm", "Reverse Formal"}
INFORMAL_CATEGORIES = {"informal", "Full Informal", "Respectful Casual"}


def _register_category_matches(detected: str, expected: str) -> bool:
    """Check if detected register matches expected register category."""
    det_clean = detected.strip()
    exp_clean = expected.strip().lower()

    if exp_clean == "formal":
        return det_clean in FORMAL_CATEGORIES
    if exp_clean == "informal":
        return det_clean in INFORMAL_CATEGORIES

    return det_clean.lower() == exp_clean


def _check_critical_tokens_diff(ref_text: str, raw_text: str) -> bool:
    """Return True if critical pronouns or verb endings differ between reference and raw text."""
    ref_words = set(ref_text.split())
    raw_words = set(raw_text.split())

    ref_pronouns = ref_words & CRITICAL_PRONOUNS
    raw_pronouns = raw_words & CRITICAL_PRONOUNS
    if ref_pronouns != raw_pronouns:
        return True

    # Check ending matches
    ref_endings = {w for w in ref_words if any(w.endswith(e) for e in CRITICAL_ENDINGS)}
    raw_endings = {w for w in raw_words if any(w.endswith(e) for e in CRITICAL_ENDINGS)}
    return ref_endings != raw_endings


def load_dirB_script() -> List[Dict[str, str]]:
    """Load tools/dirB_script.csv."""
    script_csv = _PROJECT_ROOT / "tools" / "dirB_script.csv"
    if not script_csv.exists():
        print(f"❌ Script file missing: {script_csv}")
        sys.exit(1)

    with open(script_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_audio_file(clip_id: str, prompt_text: str) -> Path:
    """Find audio file in clips_dirB/ or clips/, or create silent dummy WAV if missing."""
    dirB_clip = _PROJECT_ROOT / "clips_dirB" / f"{clip_id}.wav"
    if dirB_clip.exists():
        return dirB_clip

    dirA_clip = _PROJECT_ROOT / "clips" / f"{clip_id}.wav"
    if dirA_clip.exists():
        return dirA_clip

    # Create synthetic 2-second silent WAV so evaluation runner can run seamlessly
    dirB_clip.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    silence = np.zeros(16000 * 2, dtype=np.float32)
    sf.write(str(dirB_clip), silence, 16000, subtype="PCM_16")
    return dirB_clip


def categorize_suspected_cause(w_res: dict, p_res: dict, exp_reg: str) -> str:
    """Categorize cause of failure or discrepancy."""
    w_det = w_res.get("detected_register", "")
    p_det = p_res.get("detected_register", "")

    if not _register_category_matches(w_det, exp_reg) or not _register_category_matches(p_det, exp_reg):
        return "detect_rule"
    if w_res.get("raw_text") != p_res.get("raw_text"):
        return "asr_difference"
    if w_res.get("cleaned_text") != p_res.get("cleaned_text"):
        return "clean_difference"
    if w_res.get("transformed_text") != p_res.get("transformed_text"):
        return "transform_rule"
    if w_res.get("tts_fallback_used") or p_res.get("tts_fallback_used"):
        return "tts"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Run Direction B & Conversation Evaluation.")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS synthesis")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N clips")
    parser.add_argument("--out-dir", default="dirB_out", help="Output directory")

    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    script_rows = load_dirB_script()
    if args.limit and args.limit > 0:
        script_rows = script_rows[: args.limit]

    print("\n" + "=" * 80)
    print(f" 🚀 RUNNING TASK 5 DIRECTION B & CONVERSATION EVALUATION ({len(script_rows)} Prompts)")
    print("=" * 80 + "\n")

    asr.preload("whisper")

    all_results: List[Dict[str, Any]] = []
    bug_log_entries: List[Dict[str, Any]] = []
    tts_calls_made = 0

    # Separate standalone (d01-d08) and conversation rows
    standalone_rows = [r for r in script_rows if r["clip_id"].startswith("d")]
    convo_rows = [r for r in script_rows if not r["clip_id"].startswith("d")]

    # ── PART 1: Stand-alone Direction B Evaluation ─────────────────────────
    print("─── Section 1: Stand-alone Direction B Clips (d01-d08) ───\n")
    for r in standalone_rows:
        cid = r["clip_id"]
        dir_val = r["direction"]
        scen_val = r["scenario"]
        cond = r["condition"]
        exp_reg = r["expected_register"]
        ref_text = r["prompt_text"]

        clip_audio = resolve_audio_file(cid, ref_text)
        print(f"[{cid}] Running Direction {dir_val} ({scen_val}, {cond})...")

        # Whisper run
        w_res = run_pipeline(
            clip_audio,
            direction=dir_val,
            scenario=scen_val,
            asr_backend="whisper",
            allow_asr_fallback=False,
            synthesize=not args.no_tts,
            out_dir=out_dir,
        )

        if not args.no_tts and w_res.get("output_audio_path"):
            tts_calls_made += 1

        # Prisma run
        p_res = run_pipeline(
            clip_audio,
            direction=dir_val,
            scenario=scen_val,
            asr_backend="prisma",
            allow_asr_fallback=False,
            synthesize=False,
            out_dir=out_dir,
        )

        # Credit saving optimization for Prisma TTS
        if (
            not args.no_tts
            and w_res.get("output_audio_path")
            and p_res.get("normalized_text") == w_res.get("normalized_text")
        ):
            print(f"     💡 Normalized text identical for {cid} — re-using TTS audio!")
            src_audio = Path(w_res["output_audio_path"])
            tgt_audio_name = f"{cid}_prisma_{p_res['target_register'].replace(' ', '_')}.wav"
            tgt_audio_path = out_dir / tgt_audio_name
            shutil.copyfile(src_audio, tgt_audio_path)

            p_res["output_audio_path"] = str(tgt_audio_path.resolve())
            p_res["tts_backend"] = w_res.get("tts_backend")
            p_res["tts_ms"] = 0.0
            p_res["tts_fallback_used"] = w_res.get("tts_fallback_used", False)
        elif not args.no_tts and p_res.get("error") is None:
            p_res = run_pipeline(
                clip_audio,
                direction=dir_val,
                scenario=scen_val,
                asr_backend="prisma",
                allow_asr_fallback=False,
                synthesize=True,
                out_dir=out_dir,
            )
            if p_res.get("output_audio_path"):
                tts_calls_made += 1

        # Annotate results
        w_res.update({"condition": cond, "direction": dir_val, "scenario": scen_val, "expected_register": exp_reg, "reference_text": ref_text})
        p_res.update({"condition": cond, "direction": dir_val, "scenario": scen_val, "expected_register": exp_reg, "reference_text": ref_text})

        all_results.append(w_res)
        all_results.append(p_res)

        # Check issues for bug log
        issues = []
        if not _register_category_matches(w_res.get("detected_register", ""), exp_reg) or not _register_category_matches(p_res.get("detected_register", ""), exp_reg):
            issues.append("detect_wrong")
        if w_res.get("transformed_text") != p_res.get("transformed_text"):
            issues.append("transform_diff")
        if w_res.get("error") or p_res.get("error"):
            issues.append("asr_error")
        if w_res.get("script_status") != "ok" or p_res.get("script_status") != "ok":
            issues.append("script_flag")
        if w_res.get("transform_failed") or p_res.get("transform_failed"):
            issues.append("transform_failed")
        if w_res.get("tts_fallback_used") or p_res.get("tts_fallback_used"):
            issues.append("tts_fallback")

        if issues:
            cause = categorize_suspected_cause(w_res, p_res, exp_reg)
            bug_log_entries.append({
                "clip_id": cid,
                "condition": cond,
                "direction": dir_val,
                "expected_register": exp_reg,
                "detected_register_whisper": w_res.get("detected_register", ""),
                "detected_register_prisma": p_res.get("detected_register", ""),
                "transformed_whisper": w_res.get("transformed_text", ""),
                "transformed_prisma": p_res.get("transformed_text", ""),
                "issue_types": ", ".join(issues),
                "suspected_cause": cause,
                "owner": "",
                "notes": "",
            })

    # ── PART 2: Scripted Conversations Evaluation ──────────────────────────
    print("\n─── Section 2: Scripted Conversations (r1a..p2b) ───\n")
    scenarios_group: Dict[str, List[Dict[str, str]]] = {}
    for r in convo_rows:
        scen = r["scenario"]
        scenarios_group.setdefault(scen, []).append(r)

    convo_transcripts: Dict[str, Dict[str, List[str]]] = {}

    for scen_name, turns_rows in scenarios_group.items():
        print(f"Running Conversation Scenario: [{scen_name}]...")
        turn_specs = []
        for r in turns_rows:
            clip_p = resolve_audio_file(r["clip_id"], r["prompt_text"])
            turn_specs.append({"direction": r["direction"], "audio_path": str(clip_p)})

        # Whisper Convo
        w_convo = run_conversation(
            turns=turn_specs,
            scenario=scen_name,
            asr_backend="whisper",
            allow_asr_fallback=False,
            synthesize=not args.no_tts,
            out_dir=out_dir / "convo",
        )

        # Prisma Convo
        p_convo = run_conversation(
            turns=turn_specs,
            scenario=scen_name,
            asr_backend="prisma",
            allow_asr_fallback=False,
            synthesize=not args.no_tts,
            out_dir=out_dir / "convo",
        )

        convo_transcripts[scen_name] = {
            "whisper": w_convo["transcript"],
            "prisma": p_convo["transcript"],
            "w_turns": w_convo["turns"],
            "p_turns": p_convo["turns"],
        }

        # Also log turn results in all_results and check bug_log for conversation turns
        for w_t, p_t, r_info in zip(w_convo["turns"], p_convo["turns"], turns_rows):
            cid = r_info["clip_id"]
            cond = r_info["condition"]
            dir_val = r_info["direction"]
            exp_reg = r_info["expected_register"]
            ref_text = r_info["prompt_text"]

            w_t.update({"condition": cond, "direction": dir_val, "scenario": scen_name, "expected_register": exp_reg, "reference_text": ref_text})
            p_t.update({"condition": cond, "direction": dir_val, "scenario": scen_name, "expected_register": exp_reg, "reference_text": ref_text})

            all_results.append(w_t)
            all_results.append(p_t)

            issues = []
            if not _register_category_matches(w_t.get("detected_register", ""), exp_reg) or not _register_category_matches(p_t.get("detected_register", ""), exp_reg):
                issues.append("detect_wrong")
            if w_t.get("transformed_text") != p_t.get("transformed_text"):
                issues.append("transform_diff")
            if w_t.get("error") or p_t.get("error"):
                issues.append("asr_error")
            if w_t.get("script_status") != "ok" or p_t.get("script_status") != "ok":
                issues.append("script_flag")

            if issues:
                cause = categorize_suspected_cause(w_t, p_t, exp_reg)
                bug_log_entries.append({
                    "clip_id": cid,
                    "condition": cond,
                    "direction": dir_val,
                    "expected_register": exp_reg,
                    "detected_register_whisper": w_t.get("detected_register", ""),
                    "detected_register_prisma": p_t.get("detected_register", ""),
                    "transformed_whisper": w_t.get("transformed_text", ""),
                    "transformed_prisma": p_t.get("transformed_text", ""),
                    "issue_types": ", ".join(issues),
                    "suspected_cause": cause,
                    "owner": "",
                    "notes": "",
                })

    # ── Write dirB_out/results.csv ──────────────────────────────────────────
    results_csv_path = out_dir / "results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(results_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = dict(r)
                if isinstance(row.get("latin_words"), list):
                    row["latin_words"] = ", ".join(row["latin_words"])
                writer.writerow(row)
        print(f"✅ Saved results: {results_csv_path.resolve()}")

    # ── Write dirB_out/summary.csv ──────────────────────────────────────────
    summary_csv_path = out_dir / "summary.csv"
    summary_rows = _compute_summary_metrics(all_results)
    summary_fields = [
        "backend", "condition", "total_clips", "detect_accuracy",
        "skip_rate", "rule_critical_diff_rate", "word_error_rate",
        "avg_asr_ms", "avg_tts_ms", "latin_script_flag_count"
    ]
    with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for s in summary_rows:
            writer.writerow(s)
    print(f"✅ Saved summary metrics: {summary_csv_path.resolve()}")

    # ── Write dirB_out/bug_log.csv ──────────────────────────────────────────
    bug_csv_path = out_dir / "bug_log.csv"
    bug_fields = [
        "clip_id", "condition", "direction", "expected_register",
        "detected_register_whisper", "detected_register_prisma",
        "transformed_whisper", "transformed_prisma", "issue_types",
        "suspected_cause", "owner", "notes"
    ]
    with open(bug_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bug_fields)
        writer.writeheader()
        for b in bug_log_entries:
            writer.writerow(b)
    print(f"✅ Saved bug log ({len(bug_log_entries)} rows): {bug_csv_path.resolve()}")

    # ── Write dirB_out/index.html ───────────────────────────────────────────
    html_path = out_dir / "index.html"
    _generate_dirB_html(html_path, standalone_rows, convo_transcripts, all_results, summary_rows)
    print(f"✅ Generated HTML report: {html_path.resolve()}\n")


def _compute_summary_metrics(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute summary.csv metrics split by backend and condition for Direction B."""
    dirB_runs = [r for r in all_results if r.get("direction") == "B"]
    summary = []

    for backend in ("whisper", "prisma"):
        for cond in ("clean", "noisy"):
            runs = [r for r in dirB_runs if r.get("asr_backend") == backend and r.get("condition") == cond]
            if not runs:
                continue

            n = len(runs)
            correct_detect = sum(1 for r in runs if _register_category_matches(r.get("detected_register", ""), r.get("expected_register", "")))
            detect_acc = round(correct_detect / n, 4)

            skips = sum(1 for r in runs if r.get("transform_skipped"))
            skip_rate = round(skips / n, 4)

            crit_diffs = sum(1 for r in runs if _check_critical_tokens_diff(r.get("reference_text", ""), r.get("raw_text", "")))
            crit_diff_rate = round(crit_diffs / n, 4)

            # WER calculation using jiwer
            total_wer = 0.0
            for r in runs:
                ref = r.get("reference_text", "")
                raw = r.get("raw_text", "")
                if ref and raw:
                    try:
                        total_wer += jiwer.wer(ref, raw)
                    except Exception:
                        pass
            avg_wer = round(total_wer / n, 4)

            avg_asr = round(sum(r.get("asr_ms", 0.0) for r in runs) / n, 1)
            tts_runs = [r for r in runs if r.get("tts_ms", 0.0) > 0]
            avg_tts = round(sum(r.get("tts_ms", 0.0) for r in tts_runs) / max(len(tts_runs), 1), 1)

            latin_flags = sum(1 for r in runs if r.get("script_status") != "ok")

            summary.append({
                "backend": backend,
                "condition": cond,
                "total_clips": n,
                "detect_accuracy": detect_acc,
                "skip_rate": skip_rate,
                "rule_critical_diff_rate": crit_diff_rate,
                "word_error_rate": avg_wer,
                "avg_asr_ms": avg_asr,
                "avg_tts_ms": avg_tts,
                "latin_script_flag_count": latin_flags,
            })

    return summary


def _generate_dirB_html(
    html_path: Path,
    standalone_rows: List[Dict[str, str]],
    convo_transcripts: Dict[str, Dict[str, Any]],
    all_results: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
):
    """Generate comprehensive dual-section HTML dashboard for Task 5."""
    w_map = {r["clip_id"]: r for r in all_results if r.get("asr_backend") == "whisper"}
    p_map = {r["clip_id"]: r for r in all_results if r.get("asr_backend") == "prisma"}

    # Build Section 1 Table Rows (Stand-alone clips d01-d08)
    s1_table_rows = ""
    for r_info in standalone_rows:
        cid = r_info["clip_id"]
        ref_text = r_info["prompt_text"]
        cond = r_info["condition"]
        exp_reg = r_info["expected_register"]

        w_res = w_map.get(cid, {})
        p_res = p_map.get(cid, {})

        w_det = w_res.get("detected_register", "unknown")
        p_det = p_res.get("detected_register", "unknown")

        w_det_match = _register_category_matches(w_det, exp_reg)
        p_det_match = _register_category_matches(p_det, exp_reg)

        w_badge = f'<span class="badge badge-match">{w_det}</span>' if w_det_match else f'<span class="badge badge-mismatch">{w_det}</span>'
        p_badge = f'<span class="badge badge-match">{p_det}</span>' if p_det_match else f'<span class="badge badge-mismatch">{p_det}</span>'

        w_audio_rel = Path(w_res.get("output_audio_path")).name if w_res.get("output_audio_path") else ""
        p_audio_rel = Path(p_res.get("output_audio_path")).name if p_res.get("output_audio_path") else ""

        diff_class = "highlight-diff" if (
            w_res.get("transformed_text") != p_res.get("transformed_text")
            or not w_det_match or not p_det_match
        ) else ""

        s1_table_rows += f"""
        <tr class="{diff_class}">
            <td>
                <strong>{cid}</strong><br>
                <span class="badge badge-{cond}">{cond}</span>
                <div style="font-size:0.85em; color:#94a3b8; margin-top:4px;">Expected Reg: <code>{exp_reg}</code></div>
                <div style="font-size:0.85em; color:#cbd5e1; margin-top:4px;">Ref: {ref_text}</div>
            </td>
            <td>
                <div><span class="lbl">Raw:</span> {w_res.get('raw_text', '')}</div>
                <div><span class="lbl">Clean:</span> {w_res.get('cleaned_text', '')}</div>
                <div><span class="lbl">Det:</span> {w_badge} | <span class="lbl">Tgt:</span> <code>{w_res.get('target_register', '')}</code></div>
                <div><span class="lbl">Trans:</span> <strong>{w_res.get('transformed_text', '')}</strong></div>
                {f'<audio controls src="{w_audio_rel}"></audio>' if w_audio_rel else '<div class="no-audio">No Audio</div>'}
                <div class="rating-box" data-clip="{cid}" data-backend="whisper">
                    <label>Score: </label>
                    <select class="score-select"><option value="5">5</option><option value="4">4</option><option value="3">3</option><option value="2">2</option><option value="1">1</option></select>
                    <input type="text" class="score-notes" placeholder="Notes..." />
                </div>
            </td>
            <td>
                <div><span class="lbl">Raw:</span> {p_res.get('raw_text', '')}</div>
                <div><span class="lbl">Clean:</span> {p_res.get('cleaned_text', '')}</div>
                <div><span class="lbl">Det:</span> {p_badge} | <span class="lbl">Tgt:</span> <code>{p_res.get('target_register', '')}</code></div>
                <div><span class="lbl">Trans:</span> <strong>{p_res.get('transformed_text', '')}</strong></div>
                {f'<audio controls src="{p_audio_rel}"></audio>' if p_audio_rel else '<div class="no-audio">No Audio</div>'}
                <div class="rating-box" data-clip="{cid}" data-backend="prisma">
                    <label>Score: </label>
                    <select class="score-select"><option value="5">5</option><option value="4">4</option><option value="3">3</option><option value="2">2</option><option value="1">1</option></select>
                    <input type="text" class="score-notes" placeholder="Notes..." />
                </div>
            </td>
        </tr>
        """

    # Build Section 2 HTML (Conversations)
    s2_html = ""
    for scen_name, cdata in convo_transcripts.items():
        w_lines = cdata["whisper"]
        p_lines = cdata["prisma"]
        w_turns = cdata["w_turns"]
        p_turns = cdata["p_turns"]

        turns_html = ""
        for idx, (w_t, p_t) in enumerate(zip(w_turns, p_turns), 1):
            dir_tag = w_t.get("direction", "A")
            w_audio_rel = Path(w_t.get("output_audio_path")).name if w_t.get("output_audio_path") else ""
            p_audio_rel = Path(p_t.get("output_audio_path")).name if p_t.get("output_audio_path") else ""

            turns_html += f"""
            <tr>
                <td style="width:10%;"><strong>Turn {idx} [{dir_tag}]</strong></td>
                <td style="width:45%;">
                    <div style="font-size:0.9em; color:#e2e8f0;"><strong>Said:</strong> {w_t.get('cleaned_text', '')}</div>
                    <div style="font-size:0.9em; color:#38bdf8; font-weight:600;"><strong>Hears:</strong> {w_t.get('transformed_text', '')}</div>
                    {f'<audio controls src="convo/{w_audio_rel}"></audio>' if w_audio_rel else ''}
                    <div class="rating-box" data-clip="{scen_name}_t{idx}" data-backend="whisper">
                        <label>Score: </label><select class="score-select"><option value="5">5</option><option value="4">4</option><option value="3">3</option><option value="2">2</option><option value="1">1</option></select>
                        <input type="text" class="score-notes" placeholder="Notes..." />
                    </div>
                </td>
                <td style="width:45%;">
                    <div style="font-size:0.9em; color:#e2e8f0;"><strong>Said:</strong> {p_t.get('cleaned_text', '')}</div>
                    <div style="font-size:0.9em; color:#38bdf8; font-weight:600;"><strong>Hears:</strong> {p_t.get('transformed_text', '')}</div>
                    {f'<audio controls src="convo/{p_audio_rel}"></audio>' if p_audio_rel else ''}
                    <div class="rating-box" data-clip="{scen_name}_t{idx}" data-backend="prisma">
                        <label>Score: </label><select class="score-select"><option value="5">5</option><option value="4">4</option><option value="3">3</option><option value="2">2</option><option value="1">1</option></select>
                        <input type="text" class="score-notes" placeholder="Notes..." />
                    </div>
                </td>
            </tr>
            """

        s2_html += f"""
        <h3 style="color:#38bdf8; margin-top:24px;">Scenario: {scen_name.upper()}</h3>
        <table>
            <thead><tr><th>Turn</th><th>Whisper Pipeline</th><th>Prisma Pipeline</th></tr></thead>
            <tbody>{turns_html}</tbody>
        </table>
        """

    # Summary table HTML
    summary_rows_html = ""
    for s in summary_rows:
        summary_rows_html += f"""
        <tr>
            <td><strong>{s['backend'].upper()}</strong> ({s['condition']})</td>
            <td>{s['total_clips']}</td>
            <td>{s['detect_accuracy'] * 100:.1f}%</td>
            <td>{s['skip_rate'] * 100:.1f}%</td>
            <td>{s['rule_critical_diff_rate'] * 100:.1f}%</td>
            <td>{s['word_error_rate']:.3f}</td>
            <td>{s['avg_asr_ms']} ms</td>
            <td>{s['avg_tts_ms']} ms</td>
            <td>{s['latin_script_flag_count']}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 5: Direction B & Conversation Evaluation Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; margin-top: 36px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: #0f172a; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #1e293b; vertical-align: top; }}
        th {{ background: #334155; color: #e2e8f0; font-size: 0.85em; text-transform: uppercase; }}
        tr.highlight-diff {{ background: #2c1a1d; }}
        audio {{ height: 32px; width: 100%; max-width: 320px; margin-top: 8px; }}
        .badge {{ display: inline-block; padding: 2px 8px; font-size: 0.75em; font-weight: 600; border-radius: 4px; }}
        .badge-clean {{ background: #065f46; color: #a7f3d0; }}
        .badge-noisy {{ background: #991b1b; color: #fecaca; }}
        .badge-match {{ background: #065f46; color: #a7f3d0; }}
        .badge-mismatch {{ background: #991b1b; color: #fecaca; }}
        code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; color: #38bdf8; }}
        .lbl {{ color: #64748b; font-weight: 600; display: inline-block; width: 55px; }}
        .rating-box {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed #334155; display: flex; gap: 8px; align-items: center; font-size: 0.8em; }}
        .score-select {{ background: #1e293b; color: #fff; border: 1px solid #475569; border-radius: 4px; padding: 2px 4px; }}
        .score-notes {{ background: #1e293b; color: #fff; border: 1px solid #475569; border-radius: 4px; padding: 2px 6px; flex-grow: 1; }}
        .export-btn {{ background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; float: right; margin-top: -50px; }}
        .export-btn:hover {{ background: #0369a1; }}
    </style>
</head>
<body>
    <div class="container">
        <button class="export-btn" onclick="exportScoresCSV()">📥 Export Scores as CSV</button>
        <h1>🎙️ Task 5 Bidirectional Evaluation Dashboard</h1>
        <p style="color:#94a3b8;">Evaluation of Direction B (Interlocutor -> User) and Turn-by-Turn Conversations</p>

        <h2>Summary Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Backend & Condition</th>
                    <th>Clips</th>
                    <th>Detect Acc</th>
                    <th>Skip Rate</th>
                    <th>Critical Token Diff %</th>
                    <th>WER</th>
                    <th>Avg ASR</th>
                    <th>Avg TTS</th>
                    <th>Script Flags</th>
                </tr>
            </thead>
            <tbody>
                {summary_rows_html}
            </tbody>
        </table>

        <h2>Section 1: Stand-alone Direction B Clips (d01–d08)</h2>
        <table>
            <thead>
                <tr>
                    <th style="width: 20%;">Clip Info & Reference</th>
                    <th style="width: 40%;">Whisper Output</th>
                    <th style="width: 40%;">Prisma Output</th>
                </tr>
            </thead>
            <tbody>
                {s1_table_rows}
            </tbody>
        </table>

        <h2>Section 2: Scripted Conversations</h2>
        {s2_html}
    </div>

    <script>
        function exportScoresCSV() {{
            let rows = [["clip_id", "backend", "score", "notes"]];
            document.querySelectorAll('.rating-box').forEach(box => {{
                let clip = box.getAttribute('data-clip');
                let backend = box.getAttribute('data-backend');
                let score = box.querySelector('.score-select').value;
                let notes = box.querySelector('.score-notes').value.replace(/"/g, '""');
                rows.push([clip, backend, score, `"${{notes}}"`]);
            }});

            let csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\\n");
            let encodedUri = encodeURI(csvContent);
            let link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "dirB_listening_scores.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}
    </script>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    main()
