"""tests/run_e2e.py — End-to-end evaluation runner comparing Whisper and Gnani Prisma ASR.

Usage:
    python tests/run_e2e.py [--no-tts] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 stdout/stderr on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import asr
import config
from pipeline import _resolve_target_register, run_pipeline

logger = logging.getLogger(__name__)


def load_clip_metadata() -> tuple[dict[str, str], dict[str, str]]:
    """Load clip condition and reference text mapping from tools/."""
    conditions: dict[str, str] = {}
    references: dict[str, str] = {}

    clip_script_path = _PROJECT_ROOT / "tools" / "clip_script.csv"
    if clip_script_path.exists():
        with open(clip_script_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("clip_id", "").strip()
                if cid:
                    conditions[cid] = row.get("condition", "clean").strip()

    ref_path = _PROJECT_ROOT / "tools" / "reference.csv"
    if ref_path.exists():
        with open(ref_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("clip_id", "").strip()
                if cid:
                    references[cid] = row.get("reference_text", "").strip()

    return conditions, references


def categorize_suspected_cause(w_res: dict, p_res: dict) -> str:
    """Classify the suspected root cause for differences/bugs between backends."""
    if w_res.get("raw_text") != p_res.get("raw_text"):
        return "asr_difference"
    if w_res.get("cleaned_text") != p_res.get("cleaned_text"):
        return "clean_difference"
    if w_res.get("detected_register") != p_res.get("detected_register"):
        return "detect_rule"
    if w_res.get("transformed_text") != p_res.get("transformed_text"):
        return "transform_rule"
    if w_res.get("tts_fallback_used") or p_res.get("tts_fallback_used"):
        return "tts"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation comparing Whisper and Prisma.")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS synthesis to save credits & run fast")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N clips")
    parser.add_argument("--out-dir", default="e2e_out", help="Output directory for reports and audio")

    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions, references = load_clip_metadata()

    clips_dir = _PROJECT_ROOT / "clips"
    clip_files = sorted(list(clips_dir.glob("c*.wav")))
    if not clip_files:
        print("❌ No clip files found in clips/ directory.")
        sys.exit(1)

    if args.limit and args.limit > 0:
        clip_files = clip_files[: args.limit]

    print("\n" + "=" * 80)
    print(f" 🚀 RUNNING TASK 4 E2E EVALUATION ON {len(clip_files)} CLIPS (TTS Synthesize: {not args.no_tts})")
    print("=" * 80 + "\n")

    # Preload Whisper ASR model once
    asr.preload("whisper")

    all_results: list[dict] = []
    bug_log_entries: list[dict] = []
    tts_calls_made = 0

    for idx, clip_path in enumerate(clip_files, 1):
        clip_id = clip_path.stem
        cond = conditions.get(clip_id, "clean")
        ref_text = references.get(clip_id, "")

        print(f"[{idx}/{len(clip_files)}] Processing {clip_id}.wav ({cond})...")

        # 1. Run Whisper ASR first (clean comparison, allow_asr_fallback=False)
        w_res = run_pipeline(
            clip_path,
            target_register=None,
            asr_backend="whisper",
            allow_asr_fallback=False,
            synthesize=False,  # We will handle TTS after target synchronization
            out_dir=out_dir,
        )

        # 2. Run Prisma ASR (clean comparison, allow_asr_fallback=False)
        p_res = run_pipeline(
            clip_path,
            target_register=None,
            asr_backend="prisma",
            allow_asr_fallback=False,
            synthesize=False,
            out_dir=out_dir,
        )

        # Synchronize target register between backends using Whisper's detected register (or Prisma's fallback)
        w_det = w_res.get("detected_register", "unknown")
        p_det = p_res.get("detected_register", "unknown")
        common_det = w_det if w_det != "unknown" else p_det
        common_target, _ = _resolve_target_register(common_det, None)

        # Re-run pipeline for both backends with the EXACT SAME target register
        w_res = run_pipeline(
            clip_path,
            target_register=common_target,
            asr_backend="whisper",
            allow_asr_fallback=False,
            synthesize=not args.no_tts,
            out_dir=out_dir,
        )

        # Check credit-saving optimization for Prisma TTS
        if not args.no_tts and w_res.get("output_audio_path") and w_res.get("normalized_text"):
            tts_calls_made += 1  # count Whisper call

        if (
            not args.no_tts
            and w_res.get("output_audio_path")
            and w_res.get("normalized_text")
            and p_res.get("error") is None
        ):
            # Check Prisma normalized text before calling TTS
            p_res_text = run_pipeline(
                clip_path,
                target_register=common_target,
                asr_backend="prisma",
                allow_asr_fallback=False,
                synthesize=False,
                out_dir=out_dir,
            )

            if p_res_text.get("normalized_text") == w_res.get("normalized_text"):
                # REUSE TTS AUDIO TO SAVE GNANI CREDITS!
                print(f"     💡 Normalized text identical for {clip_id} — re-using TTS audio!")
                p_res = p_res_text
                src_audio = Path(w_res["output_audio_path"])
                tgt_audio_name = f"{clip_id}_prisma_{common_target.replace(' ', '_')}.wav"
                tgt_audio_path = out_dir / tgt_audio_name
                shutil.copyfile(src_audio, tgt_audio_path)

                p_res["output_audio_path"] = str(tgt_audio_path.resolve())
                p_res["tts_backend"] = w_res.get("tts_backend")
                p_res["tts_ms"] = 0.0
                p_res["tts_fallback_used"] = w_res.get("tts_fallback_used", False)
            else:
                # Call Prisma TTS separately
                p_res = run_pipeline(
                    clip_path,
                    target_register=common_target,
                    asr_backend="prisma",
                    allow_asr_fallback=False,
                    synthesize=True,
                    out_dir=out_dir,
                )
                if p_res.get("output_audio_path"):
                    tts_calls_made += 1
        else:
            p_res = run_pipeline(
                clip_path,
                target_register=common_target,
                asr_backend="prisma",
                allow_asr_fallback=False,
                synthesize=not args.no_tts,
                out_dir=out_dir,
            )
            if not args.no_tts and p_res.get("output_audio_path"):
                tts_calls_made += 1

        w_res["condition"] = cond
        w_res["reference_text"] = ref_text
        p_res["condition"] = cond
        p_res["reference_text"] = ref_text

        all_results.append(w_res)
        all_results.append(p_res)

        # ── Detect Bug Log Issues ───────────────────────────────────────────
        issues = []
        if w_res.get("transformed_text") != p_res.get("transformed_text"):
            issues.append("transform_diff")
        if w_res.get("detected_register") != p_res.get("detected_register"):
            issues.append("detect_diff")
        if w_res.get("error") or p_res.get("error"):
            issues.append("asr_error")
        if not w_res.get("cleaned_text") or not p_res.get("cleaned_text"):
            issues.append("empty")
        if w_res.get("script_status") != "ok" or p_res.get("script_status") != "ok":
            issues.append("script_flag")
        if w_res.get("transform_failed") or p_res.get("transform_failed"):
            issues.append("transform_failed")
        if w_res.get("tts_fallback_used") or p_res.get("tts_fallback_used"):
            issues.append("tts_fallback")

        if issues:
            cause = categorize_suspected_cause(w_res, p_res)
            bug_log_entries.append({
                "clip_id": clip_id,
                "condition": cond,
                "issue_types": ", ".join(issues),
                "reference_text": ref_text,
                "whisper_raw": w_res.get("raw_text", ""),
                "prisma_raw": p_res.get("raw_text", ""),
                "whisper_cleaned": w_res.get("cleaned_text", ""),
                "prisma_cleaned": p_res.get("cleaned_text", ""),
                "whisper_detected": w_res.get("detected_register", ""),
                "prisma_detected": p_res.get("detected_register", ""),
                "whisper_transformed": w_res.get("transformed_text", ""),
                "prisma_transformed": p_res.get("transformed_text", ""),
                "target_register": common_target,
                "suspected_cause": cause,
                "owner": "",
                "notes": "",
            })

    # ── Write e2e_out/results.csv ─────────────────────────────────────────────
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

    # ── Write e2e_out/bug_log.csv ─────────────────────────────────────────────
    bug_csv_path = out_dir / "bug_log.csv"
    bug_fields = [
        "clip_id", "condition", "issue_types", "reference_text",
        "whisper_raw", "prisma_raw", "whisper_cleaned", "prisma_cleaned",
        "whisper_detected", "prisma_detected", "whisper_transformed",
        "prisma_transformed", "target_register", "suspected_cause",
        "owner", "notes"
    ]
    with open(bug_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bug_fields)
        writer.writeheader()
        for b in bug_log_entries:
            writer.writerow(b)
    print(f"✅ Saved bug log ({len(bug_log_entries)} rows): {bug_csv_path.resolve()}")

    # ── Write e2e_out/index.html ──────────────────────────────────────────────
    html_path = out_dir / "index.html"
    _generate_index_html(html_path, clip_files, all_results, bug_log_entries, tts_calls_made)
    print(f"✅ Generated HTML report: {html_path.resolve()}\n")

    # Summary metrics
    w_runs = [r for r in all_results if r["asr_backend"] == "whisper"]
    p_runs = [r for r in all_results if r["asr_backend"] == "prisma"]

    w_avg_asr = sum(r["asr_ms"] for r in w_runs) / max(len(w_runs), 1)
    p_avg_asr = sum(r["asr_ms"] for r in p_runs) / max(len(p_runs), 1)

    w_avg_tts = sum(r["tts_ms"] for r in w_runs if r["tts_ms"] > 0) / max(len([r for r in w_runs if r["tts_ms"] > 0]), 1)
    p_avg_tts = sum(r["tts_ms"] for r in p_runs if r["tts_ms"] > 0) / max(len([r for r in p_runs if r["tts_ms"] > 0]), 1)

    print("=" * 60)
    print(" 📊 EVALUATION SUMMARY METRICS")
    print("=" * 60)
    print(f"Total Clips Evaluated  : {len(clip_files)}")
    print(f"Total Bug Log Rows     : {len(bug_log_entries)}")
    print(f"Total TTS Calls Made   : {tts_calls_made}")
    print(f"Whisper Avg ASR Latency: {w_avg_asr:.1f} ms")
    print(f"Prisma Avg ASR Latency : {p_avg_asr:.1f} ms")
    print(f"Whisper Avg TTS Latency: {w_avg_tts:.1f} ms")
    print(f"Prisma Avg TTS Latency : {p_avg_tts:.1f} ms")
    print("=" * 60 + "\n")


def _generate_index_html(
    html_path: Path,
    clip_files: list[Path],
    all_results: list[dict],
    bug_log: list[dict],
    tts_calls_made: int,
):
    """Generate rich, interactive HTML dashboard comparing Whisper and Prisma."""
    w_map = {r["audio_path"]: r for r in all_results if r["asr_backend"] == "whisper"}
    p_map = {r["audio_path"]: r for r in all_results if r["asr_backend"] == "prisma"}

    total_clips = len(clip_files)
    transform_diffs = sum(1 for b in bug_log if "transform_diff" in b["issue_types"])
    w_errors = sum(1 for r in all_results if r["asr_backend"] == "whisper" and r["error"])
    p_errors = sum(1 for r in all_results if r["asr_backend"] == "prisma" and r["error"])

    w_runs = [r for r in all_results if r["asr_backend"] == "whisper"]
    p_runs = [r for r in all_results if r["asr_backend"] == "prisma"]

    w_avg_asr = sum(r["asr_ms"] for r in w_runs) / max(len(w_runs), 1)
    p_avg_asr = sum(r["asr_ms"] for r in p_runs) / max(len(p_runs), 1)

    w_avg_tts = sum(r["tts_ms"] for r in w_runs if r["tts_ms"] > 0) / max(len([r for r in w_runs if r["tts_ms"] > 0]), 1)
    p_avg_tts = sum(r["tts_ms"] for r in p_runs if r["tts_ms"] > 0) / max(len([r for r in p_runs if r["tts_ms"] > 0]), 1)

    table_rows = ""
    for clip_path in clip_files:
        cpath_str = str(clip_path)
        cid = clip_path.stem

        w_res = w_map.get(cpath_str, {})
        p_res = p_map.get(cpath_str, {})

        cond = w_res.get("condition", "clean")
        ref_text = w_res.get("reference_text", "")

        diff_class = "highlight-diff" if (
            w_res.get("transformed_text") != p_res.get("transformed_text")
            or w_res.get("detected_register") != p_res.get("detected_register")
            or w_res.get("error") or p_res.get("error")
        ) else ""

        w_audio_rel = Path(w_res.get("output_audio_path")).name if w_res.get("output_audio_path") else ""
        p_audio_rel = Path(p_res.get("output_audio_path")).name if p_res.get("output_audio_path") else ""

        table_rows += f"""
        <tr class="{diff_class}">
            <td>
                <strong>{cid}</strong><br>
                <span class="badge badge-{cond}">{cond}</span>
                <div style="font-size:0.8em; color:#94a3b8; margin-top:4px;">Ref: {ref_text}</div>
            </td>
            <td>
                <div class="stage-block">
                    <div><span class="lbl">Raw:</span> {w_res.get('raw_text', '')}</div>
                    <div><span class="lbl">Clean:</span> {w_res.get('cleaned_text', '')}</div>
                    <div><span class="lbl">Det:</span> <code>{w_res.get('detected_register', '')}</code> | <span class="lbl">Tgt:</span> <code>{w_res.get('target_register', '')}</code></div>
                    <div><span class="lbl">Trans:</span> <strong>{w_res.get('transformed_text', '')}</strong></div>
                    <div><span class="lbl">Norm:</span> {w_res.get('normalized_text', '')}</div>
                </div>
                {f'<audio controls src="{w_audio_rel}"></audio>' if w_audio_rel else '<div class="no-audio">No Audio</div>'}
                <div style="font-size:0.75em; color:#64748b; margin-top:4px;">
                    ASR: {w_res.get('asr_ms', 0):.0f}ms | TTS: {w_res.get('tts_ms', 0):.0f}ms ({w_res.get('tts_backend', 'none')})
                    {f' | <span class="badge badge-fallback">Fallback: {w_res.get("asr_fallback_reason")}</span>' if w_res.get('fallback_used') else ''}
                </div>
                <div class="rating-box" data-clip="{cid}" data-backend="whisper">
                    <label>Score: </label>
                    <select class="score-select">
                        <option value="5">5 - Excellent</option>
                        <option value="4">4 - Good</option>
                        <option value="3">3 - Fair</option>
                        <option value="2">2 - Poor</option>
                        <option value="1">1 - Bad</option>
                    </select>
                    <input type="text" class="score-notes" placeholder="Notes..." />
                </div>
            </td>
            <td>
                <div class="stage-block">
                    <div><span class="lbl">Raw:</span> {p_res.get('raw_text', '')}</div>
                    <div><span class="lbl">Clean:</span> {p_res.get('cleaned_text', '')}</div>
                    <div><span class="lbl">Det:</span> <code>{p_res.get('detected_register', '')}</code> | <span class="lbl">Tgt:</span> <code>{p_res.get('target_register', '')}</code></div>
                    <div><span class="lbl">Trans:</span> <strong>{p_res.get('transformed_text', '')}</strong></div>
                    <div><span class="lbl">Norm:</span> {p_res.get('normalized_text', '')}</div>
                </div>
                {f'<audio controls src="{p_audio_rel}"></audio>' if p_audio_rel else '<div class="no-audio">No Audio</div>'}
                <div style="font-size:0.75em; color:#64748b; margin-top:4px;">
                    ASR: {p_res.get('asr_ms', 0):.0f}ms | TTS: {p_res.get('tts_ms', 0):.0f}ms ({p_res.get('tts_backend', 'none')})
                    {f' | <span class="badge badge-fallback">Fallback: {p_res.get("asr_fallback_reason")}</span>' if p_res.get('fallback_used') else ''}
                </div>
                <div class="rating-box" data-clip="{cid}" data-backend="prisma">
                    <label>Score: </label>
                    <select class="score-select">
                        <option value="5">5 - Excellent</option>
                        <option value="4">4 - Good</option>
                        <option value="3">3 - Fair</option>
                        <option value="2">2 - Poor</option>
                        <option value="1">1 - Bad</option>
                    </select>
                    <input type="text" class="score-notes" placeholder="Notes..." />
                </div>
            </td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 4: Speech Register Pipeline Evaluation (Whisper vs Prisma)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #1e293b;
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        h1 {{ margin-top: 0; color: #38bdf8; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }}
        .card {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .card .val {{ font-size: 1.8em; font-weight: bold; color: #38bdf8; margin-top: 4px; }}
        .card .lbl {{ font-size: 0.85em; color: #94a3b8; text-transform: uppercase; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 24px;
            background: #0f172a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #1e293b;
            vertical-align: top;
        }}
        th {{ background: #334155; color: #e2e8f0; font-size: 0.9em; text-transform: uppercase; }}
        tr.highlight-diff {{ background: #2c1a1d; }}
        tr:hover {{ background: #1a2436; }}
        audio {{ height: 32px; width: 100%; max-width: 320px; margin-top: 8px; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            font-size: 0.75em;
            font-weight: 600;
            border-radius: 4px;
        }}
        .badge-clean {{ background: #065f46; color: #a7f3d0; }}
        .badge-noisy {{ background: #991b1b; color: #fecaca; }}
        .badge-fallback {{ background: #d97706; color: #fef3c7; }}
        code {{
            background: #334155;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            color: #38bdf8;
        }}
        .stage-block {{ font-size: 0.9em; line-height: 1.5; }}
        .stage-block .lbl {{ color: #64748b; font-weight: 600; display: inline-block; width: 55px; }}
        .rating-box {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed #334155; display: flex; gap: 8px; align-items: center; font-size: 0.8em; }}
        .score-select {{ background: #1e293b; color: #fff; border: 1px solid #475569; border-radius: 4px; padding: 2px 4px; }}
        .score-notes {{ background: #1e293b; color: #fff; border: 1px solid #475569; border-radius: 4px; padding: 2px 6px; flex-grow: 1; }}
        .export-btn {{
            background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; float: right; margin-top: -50px;
        }}
        .export-btn:hover {{ background: #0369a1; }}
    </style>
</head>
<body>
    <div class="container">
        <button class="export-btn" onclick="exportScoresCSV()">📥 Export Scores as CSV</button>
        <h1>🎙️ Task 4 Evaluation Dashboard</h1>
        <p style="color:#94a3b8;">Side-by-side comparison of local Whisper vs Gnani Prisma cloud STT on Hindi Register Translation</p>

        <div class="metrics-grid">
            <div class="card"><div class="lbl">Total Clips</div><div class="val">{total_clips}</div></div>
            <div class="card"><div class="lbl">Transform Diffs</div><div class="val" style="color:#f43f5e;">{transform_diffs}</div></div>
            <div class="card"><div class="lbl">Errors (W / P)</div><div class="val">{w_errors} / {p_errors}</div></div>
            <div class="card"><div class="lbl">Avg ASR (W / P)</div><div class="val">{w_avg_asr:.0f} / {p_avg_asr:.0f} ms</div></div>
            <div class="card"><div class="lbl">Avg TTS (W / P)</div><div class="val">{w_avg_tts:.0f} / {p_avg_tts:.0f} ms</div></div>
            <div class="card"><div class="lbl">TTS Calls Made</div><div class="val">{tts_calls_made}</div></div>
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width: 15%;">Clip & Condition</th>
                    <th style="width: 425%;">Whisper Pipeline Stage Output</th>
                    <th style="width: 42.5%;">Prisma Pipeline Stage Output</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
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
            link.setAttribute("download", "e2e_listening_scores.csv");
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
