"""tests/run_asr_compare.py — Compare Whisper vs Prisma ASR on recorded test clips.

Usage:
    .\\tts-env\\Scripts\\python.exe tests/run_asr_compare.py
"""

from __future__ import annotations

import csv
import html
import io
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Force UTF-8 encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jiwer
from asr import transcribe
from cleanup import clean
from script_check import check_script, log_flagged

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

CLIPS_DIR = _PROJECT_ROOT / "clips"
TOOLS_DIR = _PROJECT_ROOT / "tools"
ASR_OUT_DIR = _PROJECT_ROOT / "asr_out"
ASR_OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_CSV = TOOLS_DIR / "reference.csv"
SCRIPT_CSV = TOOLS_DIR / "clip_script.csv"

# Pronouns and verb endings critical for register evaluation
REGISTER_CRITICAL_TOKENS = {
    # Pronouns
    "आप", "तुम", "तू", "आपका", "तुम्हारा", "तेरा", "मैं", "हम", "आपकी", "तुम्हारी", "तेरी", "आपके", "तुम्हारे", "तेरे",
    # Verb endings / Auxiliaries
    "रहे", "रहा", "रही", "हैं", "हो", "है", "ए", "इए", "ओ", "सकते", "सकता", "सकती", "कीजिए", "करो", "कर", "दीजिए", "दो", "दे",
}


def _has_rule_critical_diff(raw_text: str, reference: str) -> bool:
    """Check if any register-critical pronoun/verb token differs from reference."""
    def extract_crit(text: str) -> set[str]:
        words = set(re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", text))
        return words & REGISTER_CRITICAL_TOKENS

    raw_crit = extract_crit(raw_text)
    ref_crit = extract_crit(reference)
    return raw_crit != ref_crit


def load_references() -> dict[str, dict[str, str]]:
    ref_map: dict[str, dict[str, str]] = {}

    # Load condition from clip_script.csv
    if SCRIPT_CSV.exists():
        with open(SCRIPT_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref_map[row["clip_id"]] = {
                    "condition": row.get("condition", "clean"),
                    "reference": row.get("prompt_text", ""),
                }

    # Load ground truth text from reference.csv
    if REF_CSV.exists():
        with open(REF_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row["clip_id"]
                if cid not in ref_map:
                    ref_map[cid] = {"condition": "clean", "reference": row["reference_text"]}
                else:
                    ref_map[cid]["reference"] = row["reference_text"]

    return ref_map


def find_filler_candidates(raw_transcripts: list[str], references: list[str]) -> list[tuple[str, int]]:
    """Identify candidate filler tokens appearing in raw transcripts but not in reference."""
    ref_words = set()
    for ref in references:
        ref_words.update(re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", ref))

    extra_tokens: Counter[str] = Counter()
    for raw in raw_transcripts:
        tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", raw)
        for tok in tokens:
            if tok not in ref_words and len(tok) <= 6:
                extra_tokens[tok] += 1

    # Return tokens appearing in 2+ clips
    return [(tok, count) for tok, count in extra_tokens.most_common() if count >= 2]


def run_asr_comparison():
    ref_map = load_references()

    # Find available audio clips (.wav, .mp3, .m4a, .ogg, .flac)
    audio_extensions = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac")
    audio_files_dict: dict[str, Path] = {}
    for ext in audio_extensions:
        for p in CLIPS_DIR.glob(ext):
            cid = p.stem.lower()
            if cid not in audio_files_dict:
                audio_files_dict[cid] = p

    wav_files = [audio_files_dict[k] for k in sorted(audio_files_dict.keys())]
    if not wav_files:
        print(f"\n⚠️  No audio clips found in {CLIPS_DIR.resolve()}.")
        print("💡 You can record on your phone/voice recorder and drop c01.wav ... c18.wav (or .mp3 / .m4a) into the clips/ folder.\n")
        return

    print("\n" + "=" * 80)
    print(f" 🎙️  RUNNING ASR BENCHMARK ({len(wav_files)} clips: Whisper vs Prisma)")
    print("=" * 80 + "\n")

    results: list[dict[str, Any]] = []
    whisper_raw_list: list[str] = []
    prisma_raw_list: list[str] = []
    all_refs: list[str] = []

    for wav_path in wav_files:
        cid = wav_path.stem
        meta = ref_map.get(cid, {"condition": "clean", "reference": ""})
        reference = meta["reference"]
        condition = meta["condition"]
        all_refs.append(reference)

        print(f"[{cid}] ({condition.upper()}) Ref: {reference}")

        # ── 1. Whisper ASR ───────────────────────────────────────────────────
        whisper_res = transcribe(wav_path, backend="whisper")
        w_raw = whisper_res["text"]
        w_lat = whisper_res["latency_ms"]
        w_err = whisper_res["error"]
        whisper_raw_list.append(w_raw)

        w_raw_script = check_script(w_raw)
        w_clean = clean(w_raw, backend="whisper")
        w_clean_script = check_script(w_clean)

        w_crit_diff = _has_rule_critical_diff(w_raw, reference) if reference else False
        w_wer = round(jiwer.wer(reference, w_raw), 3) if reference and w_raw else 1.0

        if w_raw_script["status"] != "ok":
            log_flagged({
                "clip_id": cid,
                "backend": "whisper",
                "raw_text": w_raw,
                "cleaned_text": w_clean,
                "status": w_raw_script["status"],
                "latin_words": w_raw_script["latin_words"],
                "note": f"Raw script {w_raw_script['status']}",
            })

        print(f"  Whisper ({w_lat}ms) | Raw:   \"{w_raw}\" (Script: {w_raw_script['status']})")
        print(f"          | Clean: \"{w_clean}\" (WER: {w_wer:.2f}, CritDiff: {w_crit_diff})")

        # ── 2. Prisma ASR ────────────────────────────────────────────────────
        prisma_res = transcribe(wav_path, backend="prisma")
        p_raw = prisma_res["text"]
        p_lat = prisma_res["latency_ms"]
        p_err = prisma_res["error"]
        prisma_raw_list.append(p_raw)

        p_raw_script = check_script(p_raw)
        p_clean = clean(p_raw, backend="prisma")
        p_clean_script = check_script(p_clean)

        p_crit_diff = _has_rule_critical_diff(p_raw, reference) if reference else False
        p_wer = round(jiwer.wer(reference, p_raw), 3) if reference and p_raw else 1.0

        if p_raw_script["status"] != "ok" and not p_err:
            log_flagged({
                "clip_id": cid,
                "backend": "prisma",
                "raw_text": p_raw,
                "cleaned_text": p_clean,
                "status": p_raw_script["status"],
                "latin_words": p_raw_script["latin_words"],
                "note": f"Raw script {p_raw_script['status']}",
            })

        if p_err:
            print(f"  Prisma  ({p_lat}ms) | Error: {p_err}")
        else:
            print(f"  Prisma  ({p_lat}ms) | Raw:   \"{p_raw}\" (Script: {p_raw_script['status']})")
            print(f"          | Clean: \"{p_clean}\" (WER: {p_wer:.2f}, CritDiff: {p_crit_diff})")
        print()

        results.append({
            "clip_id": cid,
            "condition": condition,
            "reference": reference,
            "whisper_raw": w_raw,
            "whisper_clean": w_clean,
            "whisper_script": w_raw_script["status"],
            "whisper_latin": w_raw_script["latin_words"],
            "whisper_lat": w_lat,
            "whisper_err": w_err,
            "whisper_wer": w_wer,
            "whisper_crit_diff": w_crit_diff,
            "prisma_raw": p_raw,
            "prisma_clean": p_clean,
            "prisma_script": p_raw_script["status"],
            "prisma_latin": p_raw_script["latin_words"],
            "prisma_lat": p_lat,
            "prisma_err": p_err,
            "prisma_wer": p_wer,
            "prisma_crit_diff": p_crit_diff,
        })

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = ASR_OUT_DIR / "results.csv"
    _write_results_csv(csv_path, results)
    print(f"📄 Results saved to: {csv_path.resolve()}")

    # ── Write Filler Candidates ──────────────────────────────────────────────
    cand_path = ASR_OUT_DIR / "filler_candidates.txt"
    _write_filler_candidates(cand_path, whisper_raw_list, prisma_raw_list, all_refs)
    print(f"📝 Filler candidates written to: {cand_path.resolve()}")

    # ── Write HTML Report ────────────────────────────────────────────────────
    html_path = ASR_OUT_DIR / "index.html"
    _write_html_report(html_path, results)
    print(f"📊 HTML visual report generated: {html_path.resolve()}\n")


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    headers = [
        "clip_id", "condition", "reference_text", "backend",
        "raw_text", "cleaned_text", "script_status", "latin_words",
        "asr_latency_ms", "rule_critical_diff", "error"
    ]
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "clip_id": r["clip_id"],
                "condition": r["condition"],
                "reference_text": r["reference"],
                "backend": "whisper",
                "raw_text": r["whisper_raw"],
                "cleaned_text": r["whisper_clean"],
                "script_status": r["whisper_script"],
                "latin_words": ", ".join(r["whisper_latin"]),
                "asr_latency_ms": r["whisper_lat"],
                "rule_critical_diff": r["whisper_crit_diff"],
                "error": r["whisper_err"] or "",
            })
            writer.writerow({
                "clip_id": r["clip_id"],
                "condition": r["condition"],
                "reference_text": r["reference"],
                "backend": "prisma",
                "raw_text": r["prisma_raw"],
                "cleaned_text": r["prisma_clean"],
                "script_status": r["prisma_script"],
                "latin_words": ", ".join(r["prisma_latin"]),
                "asr_latency_ms": r["prisma_lat"],
                "rule_critical_diff": r["prisma_crit_diff"],
                "error": r["prisma_err"] or "",
            })


def _write_filler_candidates(
    path: Path,
    whisper_raws: list[str],
    prisma_raws: list[str],
    references: list[str],
) -> None:
    w_cands = find_filler_candidates(whisper_raws, references)
    p_cands = find_filler_candidates(prisma_raws, references)

    lines = [
        "==================================================",
        " ASR FILLER SOUND CANDIDATES (Frequency >= 2)",
        "==================================================",
        "",
        "── Whisper Candidates ──",
    ]
    if w_cands:
        for tok, cnt in w_cands:
            lines.append(f"  • {tok} (count: {cnt})")
    else:
        lines.append("  (None detected)")

    lines.extend([
        "",
        "── Prisma Candidates ──",
    ])
    if p_cands:
        for tok, cnt in p_cands:
            lines.append(f"  • {tok} (count: {cnt})")
    else:
        lines.append("  (None detected)")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_report(path: Path, results: list[dict[str, Any]]) -> None:
    # Compute summary stats
    w_ok = sum(1 for r in results if r["whisper_script"] == "ok")
    w_mixed = sum(1 for r in results if r["whisper_script"] == "mixed")
    w_latin = sum(1 for r in results if r["whisper_script"] == "latin")
    w_avg_lat = round(sum(r["whisper_lat"] for r in results) / max(1, len(results)), 1)

    p_valid = [r for r in results if not r["prisma_err"]]
    p_ok = sum(1 for r in p_valid if r["prisma_script"] == "ok")
    p_mixed = sum(1 for r in p_valid if r["prisma_script"] == "mixed")
    p_latin = sum(1 for r in p_valid if r["prisma_script"] == "latin")
    p_avg_lat = round(sum(r["prisma_lat"] for r in p_valid) / max(1, len(p_valid)), 1) if p_valid else 0.0

    # Clean vs Noisy WER
    clean_items = [r for r in results if r["condition"] == "clean"]
    noisy_items = [r for r in results if r["condition"] == "noisy"]

    w_wer_clean = round(sum(r["whisper_wer"] for r in clean_items) / max(1, len(clean_items)), 3)
    w_wer_noisy = round(sum(r["whisper_wer"] for r in noisy_items) / max(1, len(noisy_items)), 3)

    p_clean_valid = [r for r in clean_items if not r["prisma_err"]]
    p_noisy_valid = [r for r in noisy_items if not r["prisma_err"]]
    p_wer_clean = round(sum(r["prisma_wer"] for r in p_clean_valid) / max(1, len(p_clean_valid)), 3) if p_clean_valid else 0.0
    p_wer_noisy = round(sum(r["prisma_wer"] for r in p_noisy_valid) / max(1, len(p_noisy_valid)), 3) if p_noisy_valid else 0.0

    def badge(status: str) -> str:
        color = "#10b981" if status == "ok" else "#f59e0b" if status == "mixed" else "#ef4444"
        return f'<span style="background:{color}; color:#fff; padding:2px 6px; border-radius:4px; font-size:0.75em; text-transform:uppercase;">{status}</span>'

    table_rows = ""
    for r in results:
        p_display = f'<div style="color:#ef4444;">Error: {html.escape(r["prisma_err"])}</div>' if r["prisma_err"] else f"""
            <div><strong>Raw:</strong> {html.escape(r['prisma_raw'])} {badge(r['prisma_script'])}</div>
            <div style="color:#38bdf8;"><strong>Clean:</strong> {html.escape(r['prisma_clean'])}</div>
            <div style="font-size:0.8em; color:#64748b;">{r['prisma_lat']}ms | WER: {r['prisma_wer']:.2f} {'⚠️ CritDiff' if r['prisma_crit_diff'] else ''}</div>
        """

        table_rows += f"""
        <tr>
            <td><strong>{r['clip_id']}</strong><br><span style="font-size:0.8em; color:#94a3b8;">{r['condition']}</span></td>
            <td><strong>{html.escape(r['reference'])}</strong></td>
            <td>
                <div><strong>Raw:</strong> {html.escape(r['whisper_raw'])} {badge(r['whisper_script'])}</div>
                <div style="color:#38bdf8;"><strong>Clean:</strong> {html.escape(r['whisper_clean'])}</div>
                <div style="font-size:0.8em; color:#64748b;">{r['whisper_lat']}ms | WER: {r['whisper_wer']:.2f} {'⚠️ CritDiff' if r['whisper_crit_diff'] else ''}</div>
            </td>
            <td>{p_display}</td>
        </tr>
        """

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Task 3: ASR Comparison Report (Whisper vs Prisma)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1300px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 24px 0; }}
        .card {{ background: #0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
        .card h3 {{ margin-top: 0; color: #94a3b8; font-size: 1.1em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 24px; background: #0f172a; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #1e293b; }}
        th {{ background: #334155; color: #e2e8f0; font-size: 0.85em; text-transform: uppercase; }}
        tr:hover {{ background: #1a2436; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Task 3: ASR Comparison & Cleanup Benchmark</h1>
        <p style="color:#94a3b8;">Side-by-side evaluation of Whisper (local) and Gnani Prisma v3 (cloud) on Hindi audio.</p>

        <div class="grid">
            <div class="card">
                <h3>🤖 Whisper (vasista22/whisper-hindi-small)</h3>
                <div><strong>Avg Latency:</strong> {w_avg_lat} ms</div>
                <div><strong>Script Distribution:</strong> OK: {w_ok} | Mixed: {w_mixed} | Latin: {w_latin}</div>
                <div><strong>WER:</strong> Clean: {w_wer_clean:.2f} | Noisy: {w_wer_noisy:.2f}</div>
            </div>
            <div class="card">
                <h3>☁️ Gnani Prisma STT v3</h3>
                <div><strong>Avg Latency:</strong> {p_avg_lat} ms</div>
                <div><strong>Script Distribution:</strong> OK: {p_ok} | Mixed: {p_mixed} | Latin: {p_latin}</div>
                <div><strong>WER:</strong> Clean: {p_wer_clean:.2f} | Noisy: {p_wer_noisy:.2f}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Clip</th>
                    <th>Reference Ground Truth</th>
                    <th>Whisper (Raw ➔ Cleaned)</th>
                    <th>Prisma (Raw ➔ Cleaned)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    path.write_text(html_code, encoding="utf-8")


if __name__ == "__main__":
    run_asr_comparison()
