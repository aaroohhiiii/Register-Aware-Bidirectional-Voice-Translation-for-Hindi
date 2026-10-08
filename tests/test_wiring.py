"""tests/test_wiring.py — Task 2 End-to-End Pipeline & Mixed-Language Test Runner.

Usage:
    .\\tts-env\\Scripts\\python.exe tests/test_wiring.py

What it does:
1. Part A: Runs 12 paired sentences across registers (Formal, Respectful Casual, Full Informal)
   through the full pipeline: Detection -> Transformation -> Normalization -> TTS.
2. Part B: Runs mixed Hindi-English sentences comparing hi-IN vs hi-en language tags.
3. Generates audio files in tts_out/task2/
4. Writes a comprehensive HTML report at tts_out/task2_report.html with audio players.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import soundfile as sf
import tts
from language_hint import pick_tts_language
from normalize import normalize_for_tts
from pipeline_text import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

OUT_DIR = _PROJECT_ROOT / "tts_out" / "task2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Part A Corpus: 12 Sentences across Register Tiers ───────────────────────

PART_A_CORPUS = [
    {
        "id": "A1",
        "input": "आप कब आ रहे हैं?",
        "target": "Respectful Casual",
        "description": "Formal → Respectful Casual",
    },
    {
        "id": "A2",
        "input": "तुम कब आ रहे हो?",
        "target": "Full Informal",
        "description": "Respectful Casual → Full Informal",
    },
    {
        "id": "A3",
        "input": "तू कब आ रहा है?",
        "target": "Formal",
        "description": "Full Informal → Formal",
    },
    {
        "id": "A4",
        "input": "कृपया यहाँ 5:30 बजे बैठिए।",
        "target": "Respectful Casual",
        "description": "Time + Formal → Casual",
    },
    {
        "id": "A5",
        "input": "यहाँ बैठो, मैं ₹500 लेकर आता हूँ।",
        "target": "Formal",
        "description": "Currency + Casual → Formal",
    },
    {
        "id": "A6",
        "input": "तू 1,500 रुपये दे दे।",
        "target": "Respectful Casual",
        "description": "Grouped digits + Tu → Tum",
    },
    {
        "id": "A7",
        "input": "क्या आप मेरी मदद कर सकते हैं?",
        "target": "Full Informal",
        "description": "Polite request → Intimate",
    },
    {
        "id": "A8",
        "input": "क्या तू मेरा काम करेगा?",
        "target": "Formal",
        "description": "Intimate → Formal polite",
    },
    {
        "id": "A9",
        "input": "आपका नाम क्या है?",
        "target": "Respectful Casual",
        "description": "Possessive Formal → Casual",
    },
    {
        "id": "A10",
        "input": "तेरा घर कहाँ है?",
        "target": "Formal",
        "description": "Possessive Tu → Aap",
    },
    {
        "id": "A11",
        "input": "आप चाय लेंगे या कॉफ़ी?",
        "target": "Respectful Casual",
        "description": "Offer Formal → Casual",
    },
    {
        "id": "A12",
        "input": "काम 50% पूरा हो गया है, आप देख लीजिए।",
        "target": "Respectful Casual",
        "description": "Percentage + Formal → Casual",
    },
]


# ─── Part B Corpus: Mixed Hindi-English Sentences ────────────────────────────

PART_B_CORPUS = [
    {
        "id": "B1",
        "text": "कृपया meeting room में wait कीजिए।",
        "target": "Respectful Casual",
        "description": "English loanwords: meeting room, wait",
    },
    {
        "id": "B2",
        "text": "कल का flight ticket book हो गया है।",
        "target": "Formal",
        "description": "English loanwords: flight ticket book",
    },
    {
        "id": "B3",
        "text": "Doctor से appointment ले लो।",
        "target": "Formal",
        "description": "English loanwords: Doctor, appointment",
    },
    {
        "id": "B4",
        "text": "यह mobile phone बहुत expensive है।",
        "target": "Respectful Casual",
        "description": "English loanwords: mobile phone, expensive",
    },
    {
        "id": "B5",
        "text": "अपना laptop table पर रख दो।",
        "target": "Formal",
        "description": "English loanwords: laptop, table",
    },
    {
        "id": "B6",
        "text": "हमारा project report submit हो गया है।",
        "target": "Respectful Casual",
        "description": "English loanwords: project report submit",
    },
]


def _wav_duration(path: Path) -> float:
    try:
        data, sr = sf.read(str(path))
        return round(len(data) / sr, 2)
    except Exception:
        return 0.0


def run_all_tests():
    print("\n" + "=" * 80)
    print(" 🚀 RUNNING TASK 2 WIRING & MIXED-LANGUAGE TESTS")
    print("=" * 80 + "\n")

    part_a_results = []
    part_b_results = []

    # ── PART A ───────────────────────────────────────────────────────────────
    print("─── Part A: Register Transformation -> Normalization -> TTS ───\n")
    for item in PART_A_CORPUS:
        out_wav = OUT_DIR / f"{item['id']}_transformed.wav"
        print(f"[{item['id']}] Input:  {item['input']}")
        print(f"     Target: {item['target']}")

        res = run_pipeline(
            text=item["input"],
            target_register=item["target"],
            out_path=out_wav,
        )

        duration = _wav_duration(out_wav)
        print(f"     Detected:   {res.source_register}")
        print(f"     Transformed:{res.transformed_text}")
        print(f"     Normalized: {res.normalized_text}")
        print(f"     Lang Hint:  {res.language_hint} | Backend: {res.backend_used} | Audio: {duration}s | Latency: {res.elapsed_total_s}s\n")

        part_a_results.append({
            "id": item["id"],
            "desc": item["description"],
            "input": item["input"],
            "src_reg": res.source_register,
            "tgt_reg": item["target"],
            "transformed": res.transformed_text,
            "normalized": res.normalized_text,
            "lang_hint": res.language_hint,
            "backend": res.backend_used,
            "duration": duration,
            "latency": res.elapsed_total_s,
            "wav_rel": f"task2/{out_wav.name}",
        })

    # ── PART B ───────────────────────────────────────────────────────────────
    print("\n─── Part B: Mixed Hindi-English Language Hint Testing ───\n")
    for item in PART_B_CORPUS:
        norm_text = normalize_for_tts(item["text"])
        detected_hint = pick_tts_language(norm_text)
        print(f"[{item['id']}] Text:       {item['text']}")
        print(f"     Normalized: {norm_text}")
        print(f"     Detected Language Hint: {detected_hint}")

        # Synthesize with detected hint (e.g. hi-en)
        wav_detected = OUT_DIR / f"{item['id']}_detected_{detected_hint}.wav"
        t0 = time.perf_counter()
        tts.speak(norm_text, wav_detected, language=detected_hint)
        lat_detected = round(time.perf_counter() - t0, 3)
        dur_detected = _wav_duration(wav_detected)

        # Synthesize with standard hi-IN for comparison
        wav_hi_in = OUT_DIR / f"{item['id']}_standard_hi_IN.wav"
        t0 = time.perf_counter()
        tts.speak(norm_text, wav_hi_in, language="hi-IN")
        lat_hi_in = round(time.perf_counter() - t0, 3)
        dur_hi_in = _wav_duration(wav_hi_in)

        print(f"     Audio ({detected_hint}): {dur_detected}s ({lat_detected}s) | Audio (hi-IN): {dur_hi_in}s ({lat_hi_in}s)\n")

        part_b_results.append({
            "id": item["id"],
            "desc": item["description"],
            "text": item["text"],
            "normalized": norm_text,
            "detected_hint": detected_hint,
            "dur_detected": dur_detected,
            "lat_detected": lat_detected,
            "wav_detected_rel": f"task2/{wav_detected.name}",
            "dur_hi_in": dur_hi_in,
            "lat_hi_in": lat_hi_in,
            "wav_hi_in_rel": f"task2/{wav_hi_in.name}",
        })

    # ── Generate HTML Report ─────────────────────────────────────────────────
    html_path = _PROJECT_ROOT / "tts_out" / "task2_report.html"
    _generate_html_report(html_path, part_a_results, part_b_results)
    print(f"\n✅ HTML Report generated: {html_path.resolve()}\n")


def _generate_html_report(path: Path, part_a: list[dict], part_b: list[dict]):
    a_rows = ""
    for r in part_a:
        a_rows += f"""
        <tr>
            <td><strong>{r['id']}</strong><br><small style="color:#64748b;">{r['desc']}</small></td>
            <td><code>{r['src_reg']}</code> → <code>{r['tgt_reg']}</code></td>
            <td>
                <div style="font-size:0.9em; color:#475569;"><strong>Raw:</strong> {r['input']}</div>
                <div style="font-size:0.9em; color:#0284c7;"><strong>Trans:</strong> {r['transformed']}</div>
                <div style="font-size:0.95em; color:#0f172a; font-weight:600;"><strong>Norm:</strong> {r['normalized']}</div>
            </td>
            <td><span class="badge">{r['lang_hint']}</span></td>
            <td>
                <audio controls preload="none" src="{r['wav_rel']}"></audio>
                <div style="font-size:0.8em; color:#64748b;">{r['duration']}s | {r['latency']}s lat ({r['backend']})</div>
            </td>
        </tr>
        """

    b_rows = ""
    for r in part_b:
        b_rows += f"""
        <tr>
            <td><strong>{r['id']}</strong><br><small style="color:#64748b;">{r['desc']}</small></td>
            <td>
                <div style="font-size:0.9em; color:#475569;"><strong>Input:</strong> {r['text']}</div>
                <div style="font-size:0.95em; color:#0f172a; font-weight:600;"><strong>Normalized:</strong> {r['normalized']}</div>
                <span class="badge" style="background:#e0f2fe; color:#0369a1;">Auto-hint: {r['detected_hint']}</span>
            </td>
            <td>
                <audio controls preload="none" src="{r['wav_detected_rel']}"></audio>
                <div style="font-size:0.8em; color:#64748b;">{r['dur_detected']}s | {r['lat_detected']}s</div>
            </td>
            <td>
                <audio controls preload="none" src="{r['wav_hi_in_rel']}"></audio>
                <div style="font-size:0.8em; color:#64748b;">{r['dur_hi_in']}s | {r['lat_hi_in']}s</div>
            </td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 2: Transformation & TTS Integration Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #1e293b;
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        h1 {{ margin-top: 0; color: #38bdf8; }}
        h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; margin-top: 36px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            background: #0f172a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #1e293b;
        }}
        th {{ background: #334155; color: #e2e8f0; font-size: 0.9em; text-transform: uppercase; }}
        tr:hover {{ background: #1a2436; }}
        audio {{ height: 32px; width: 220px; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            font-size: 0.8em;
            font-weight: 600;
            border-radius: 4px;
            background: #0369a1;
            color: #f0f9ff;
        }}
        code {{
            background: #334155;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            color: #38bdf8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Task 2 Evaluation Report</h1>
        <p style="color:#94a3b8;">Full end-to-end integration: Register Transformation ➔ Normalization ➔ Language Selection ➔ TTS</p>

        <h2>Part A: Register Transformation & TTS (12 Sentences)</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Tier Change</th>
                    <th>Transformation & Normalization</th>
                    <th>Lang</th>
                    <th>Output Audio</th>
                </tr>
            </thead>
            <tbody>
                {a_rows}
            </tbody>
        </table>

        <h2>Part B: Mixed Hindi-English Language Hint Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Mixed Text & Normalization</th>
                    <th>Auto Hint Audio (hi-en)</th>
                    <th>Standard Audio (hi-IN)</th>
                </tr>
            </thead>
            <tbody>
                {b_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    path.write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    run_all_tests()
