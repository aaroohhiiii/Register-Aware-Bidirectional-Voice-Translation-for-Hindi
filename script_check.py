"""script_check.py — Devanagari and Latin script inspector for Hindi ASR transcripts.

Public API
----------
check_script(text: str) -> dict
    Inspects text script composition without modifying or transliterating.

log_flagged(entry: dict, csv_path: str = "flagged_for_aarohi.csv") -> None
    Appends non-"ok" script cases to CSV for downstream review.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any

_DEVA_CHAR = re.compile(r"[\u0900-\u097F]")
_LATIN_CHAR = re.compile(r"[a-zA-Z]")
_LATIN_WORD = re.compile(r"\b[a-zA-Z]+\b")


def check_script(text: str) -> dict[str, Any]:
    """Analyze script composition of Hindi text.

    Parameters
    ----------
    text : str
        Input transcript string.

    Returns
    -------
    dict
        {
            "status": "ok" | "mixed" | "latin" | "empty",
            "deva_count": int,
            "latin_count": int,
            "latin_ratio": float,
            "latin_words": list[str],
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "empty",
            "deva_count": 0,
            "latin_count": 0,
            "latin_ratio": 0.0,
            "latin_words": [],
        }

    norm = unicodedata.normalize("NFC", text)
    deva_letters = _DEVA_CHAR.findall(norm)
    latin_letters = _LATIN_CHAR.findall(norm)
    latin_words = _LATIN_WORD.findall(norm)

    deva_count = len(deva_letters)
    latin_count = len(latin_letters)
    total_letters = deva_count + latin_count

    if total_letters == 0:
        status = "empty"
        latin_ratio = 0.0
    elif latin_count == 0:
        status = "ok"
        latin_ratio = 0.0
    elif deva_count == 0:
        status = "latin"
        latin_ratio = 1.0
    else:
        latin_ratio = round(latin_count / total_letters, 4)
        if latin_ratio > 0.6:
            status = "latin"
        else:
            status = "mixed"

    return {
        "status": status,
        "deva_count": deva_count,
        "latin_count": latin_count,
        "latin_ratio": latin_ratio,
        "latin_words": latin_words,
    }


def log_flagged(entry: dict[str, Any], csv_path: str | Path = "flagged_for_aarohi.csv") -> None:
    """Append non-'ok' cases to flagged_for_aarohi.csv."""
    path = Path(csv_path)
    file_exists = path.exists()

    headers = [
        "clip_id",
        "backend",
        "raw_text",
        "cleaned_text",
        "status",
        "latin_words",
        "note",
    ]

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        latin_words_str = ", ".join(entry.get("latin_words", [])) if isinstance(entry.get("latin_words"), list) else str(entry.get("latin_words", ""))
        writer.writerow({
            "clip_id": entry.get("clip_id", ""),
            "backend": entry.get("backend", ""),
            "raw_text": entry.get("raw_text", ""),
            "cleaned_text": entry.get("cleaned_text", ""),
            "status": entry.get("status", ""),
            "latin_words": latin_words_str,
            "note": entry.get("note", ""),
        })


__all__ = ["check_script", "log_flagged"]
