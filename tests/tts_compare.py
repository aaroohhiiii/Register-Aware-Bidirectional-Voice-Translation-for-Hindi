"""tests/tts_compare.py — Multi-backend TTS comparison runner.

Run from the project root:
    tts-env\\Scripts\\python.exe tests/tts_compare.py

What it does
------------
1. Runs all 10 Hindi test sentences through every backend at speed 1.0.
2. For Timbre, additionally runs them at speeds 0.85 and 1.15.
3. Saves WAV files to  tts_out/<backend>/s<NN>_speed<X>.wav
4. Writes tts_out/results.csv  (backend, sentence_no, speed, latency_ms,
                                 duration_s, status)
5. Writes tts_out/index.html   (one row per sentence, audio players for every
                                 backend/speed combination, so you can listen
                                 side-by-side).
"""

from __future__ import annotations

# Force UTF-8 output on Windows so Hindi text and special chars print cleanly.
import io
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import csv
import logging
import time
from pathlib import Path

# Make sure the project root is on sys.path so we can import tts / config.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import soundfile as sf  # noqa: E402
import tts as tts_module  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Test corpus — 10 sentences covering आप / तुम / तू registers
# ─────────────────────────────────────────────────────────────────────────────
SENTENCES = [
    (1,  "आप कहाँ जा रहे हैं?"),
    (2,  "तुम कहाँ जा रहे हो?"),
    (3,  "तू कहाँ जा रहा है?"),
    (4,  "क्या आप मुझे रास्ता बता सकते हैं?"),
    (5,  "क्या तुम मुझे रास्ता बता सकते हो?"),
    (6,  "आपका नाम क्या है?"),
    (7,  "तुम्हारा नाम क्या है?"),
    (8,  "कृपया यहाँ रुकिए।"),
    (9,  "यहीं रुक जाओ, मैं आता हूँ।"),
    (10, "किराया पाँच सौ रुपये होगा, आप बैठिए।"),
]

# Backends to try.  Timbre also gets extra speed variants.
BACKENDS = ["timbre", "parler", "mms"]

# Speeds: all backends at 1.0; Timbre also at 0.85 and 1.15.
BASE_SPEED    = 1.0
TIMBRE_SPEEDS = [0.85, 1.0, 1.15]

OUT_DIR = _PROJECT_ROOT / "tts_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: audio duration from a WAV file
# ─────────────────────────────────────────────────────────────────────────────

def _wav_duration(path: Path) -> float:
    """Return duration in seconds, or -1.0 on read error."""
    try:
        data, sr = sf.read(str(path))
        return len(data) / sr
    except Exception:
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

rows: list[dict] = []   # accumulated for CSV

# Track per-backend totals for the summary printed at the end.
backend_stats: dict[str, dict] = {
    b: {"ok": 0, "fail": 0, "latencies": []} for b in BACKENDS
}
# Entire-backend failures (set when every sentence fails).
backend_errors: dict[str, str] = {}

for backend in BACKENDS:
    speeds = TIMBRE_SPEEDS if backend == "timbre" else [BASE_SPEED]

    backend_dir = OUT_DIR / backend
    backend_dir.mkdir(parents=True, exist_ok=True)

    sentence_fail_count = 0
    backend_skipped = False   # set True if the model fails to load (skip remaining sentences)
    skip_reason = ""

    for speed in speeds:
        if backend_skipped:
            break

        speed_tag = f"{speed:.2f}".replace(".", "")   # e.g. "100", "085", "115"

        for sentence_no, text in SENTENCES:
            wav_name = f"s{sentence_no:02d}_speed{speed_tag}.wav"
            wav_path = backend_dir / wav_name

            # If the model failed to load, record all remaining rows as errors instantly.
            if backend_skipped:
                rows.append({
                    "backend":     backend,
                    "sentence_no": sentence_no,
                    "speed":       speed,
                    "latency_ms":  0,
                    "duration_s":  -1.0,
                    "status":      "skipped",
                    "error":       skip_reason[:300],
                    "wav_path":    "",
                })
                backend_stats[backend]["fail"] += 1
                sentence_fail_count += 1
                print(f"  - [{backend}] speed={speed:.2f} s{sentence_no:02d}  SKIPPED (model unavailable)")
                continue

            t0 = time.perf_counter()
            error_msg = ""
            status = "ok"

            try:
                tts_module.speak(
                    text,
                    wav_path,
                    backend=backend,
                    speed=speed,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                duration_s = _wav_duration(wav_path)
                backend_stats[backend]["ok"] += 1
                backend_stats[backend]["latencies"].append(latency_ms)

            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                duration_s = -1.0
                error_msg  = str(exc)
                status     = "error"
                sentence_fail_count += 1
                backend_stats[backend]["fail"] += 1
                logger.error(
                    "[%s | speed=%.2f | s%02d] FAILED: %s",
                    backend, speed, sentence_no, error_msg[:200],
                )
                # If sentence 1 fails with a load-level error, skip the rest.
                if sentence_no == 1:
                    backend_skipped = True
                    skip_reason = error_msg
                    logger.warning(
                        "Backend '%s' failed on sentence 1 — skipping remaining sentences.",
                        backend,
                    )

            rows.append({
                "backend":     backend,
                "sentence_no": sentence_no,
                "speed":       speed,
                "latency_ms":  round(latency_ms, 1),
                "duration_s":  round(duration_s, 3),
                "status":      status,
                "error":       error_msg[:300] if error_msg else "",
                "wav_path":    str(wav_path) if status == "ok" else "",
            })

            label = "✓" if status == "ok" else "✗"
            print(
                f"  {label} [{backend}] speed={speed:.2f} s{sentence_no:02d} "
                f"{latency_ms:7.0f} ms  {duration_s:.2f}s  {text[:40]}"
            )

    # If every single sentence for this backend failed/skipped, note it.
    if sentence_fail_count == len(SENTENCES) * len(speeds):
        backend_errors[backend] = rows[-1]["error"] or skip_reason


# ─────────────────────────────────────────────────────────────────────────────
# Write results.csv
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = ["backend", "sentence_no", "speed", "latency_ms", "duration_s", "status", "error"]
csv_path = OUT_DIR / "results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

logger.info("CSV written → %s", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Write index.html
# ─────────────────────────────────────────────────────────────────────────────

# Build a nested lookup:
#   cells[(sentence_no, backend, speed)] = relative WAV path (or "")
from collections import defaultdict  # noqa: E402 — stdlib, just moving inside script

Cell = dict  # type alias for clarity
cell_map: dict[tuple, str] = {}
for r in rows:
    key = (r["sentence_no"], r["backend"], r["speed"])
    cell_map[key] = r["wav_path"]

# Column headers: one column per (backend, speed) combination, ordered naturally.
ColKey = tuple  # (backend, speed)
col_keys: list[ColKey] = []
for b in BACKENDS:
    speeds_for_b = TIMBRE_SPEEDS if b == "timbre" else [BASE_SPEED]
    for s in speeds_for_b:
        col_keys.append((b, s))


def _rel(wav_path: str) -> str:
    """Make path relative to OUT_DIR for embedding in HTML."""
    if not wav_path:
        return ""
    try:
        return str(Path(wav_path).relative_to(OUT_DIR)).replace("\\", "/")
    except ValueError:
        return wav_path  # already relative or on a different drive


def _col_label(backend: str, speed: float) -> str:
    label = backend.capitalize()
    if backend == "timbre":
        label += f"<br><small>speed={speed:.2f}</small>"
    return label


# Assemble HTML rows.
table_rows_html = []
for sentence_no, text in SENTENCES:
    cells_html = [f"<td class='text-cell'>{sentence_no}. {text}</td>"]
    for backend, speed in col_keys:
        wav = cell_map.get((sentence_no, backend, speed), "")
        rel = _rel(wav)
        if rel:
            player = (
                f'<audio controls preload="none">'
                f'<source src="{rel}" type="audio/wav"></audio>'
            )
        else:
            player = '<span class="missing">—</span>'
        cells_html.append(f"<td class='audio-cell'>{player}</td>")
    table_rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

col_headers_html = "<th>Sentence</th>" + "".join(
    f"<th>{_col_label(b, s)}</th>" for b, s in col_keys
)

html = f"""<!DOCTYPE html>
<html lang="hi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hindi TTS Comparison</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #0f1117;
      color: #e8eaf0;
      margin: 0;
      padding: 20px;
    }}
    h1 {{
      color: #a78bfa;
      margin-bottom: 4px;
    }}
    p.subtitle {{
      color: #8892a4;
      margin-top: 0;
      font-size: 0.9em;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 16px;
    }}
    th {{
      background: #1e2130;
      color: #c4b5fd;
      padding: 10px 14px;
      border: 1px solid #2d3148;
      font-size: 0.85em;
      text-align: center;
    }}
    td {{
      border: 1px solid #2d3148;
      padding: 8px 12px;
      vertical-align: middle;
    }}
    tr:nth-child(even) td {{
      background: #141824;
    }}
    td.text-cell {{
      font-family: 'Noto Sans Devanagari', serif;
      font-size: 1.05em;
      min-width: 280px;
      color: #e8eaf0;
    }}
    td.audio-cell {{
      text-align: center;
      min-width: 220px;
    }}
    audio {{
      width: 200px;
      height: 30px;
    }}
    .missing {{
      color: #4b5563;
      font-size: 1.2em;
    }}
    .legend {{
      margin-top: 24px;
      font-size: 0.8em;
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <h1>Hindi TTS — Backend Comparison</h1>
  <p class="subtitle">
    3 backends · 10 sentences · آप/तुम/तू register pairs
    &nbsp;|&nbsp; Timbre also evaluated at speed 0.85 and 1.15
  </p>
  <table>
    <thead><tr>{col_headers_html}</tr></thead>
    <tbody>
      {"".join(table_rows_html)}
    </tbody>
  </table>
  <div class="legend">
    Generated by tests/tts_compare.py &nbsp;|&nbsp;
    Results CSV: results.csv
  </div>
</body>
</html>
"""

html_path = OUT_DIR / "index.html"
html_path.write_text(html, encoding="utf-8")
logger.info("HTML written → %s", html_path)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("TTS COMPARISON SUMMARY")
print("=" * 70)

for b in BACKENDS:
    st = backend_stats[b]
    ok   = st["ok"]
    fail = st["fail"]
    lats = st["latencies"]
    avg  = sum(lats) / len(lats) if lats else 0.0
    med  = sorted(lats)[len(lats) // 2] if lats else 0.0

    if b in backend_errors and ok == 0:
        status_line = f"FAILED ENTIRELY  — {backend_errors[b][:120]}"
    else:
        status_line = (
            f"OK={ok:3d}  FAIL={fail:3d}  "
            f"avg latency={avg:6.0f} ms  median={med:6.0f} ms"
        )
    print(f"  {b.upper():<8}  {status_line}")

print()
print(f"Audio files → {OUT_DIR}")
print(f"CSV         → {csv_path}")
print(f"Listener    → Open in browser: {html_path}")
print("=" * 70)
