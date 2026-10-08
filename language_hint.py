"""language_hint.py — Script-ratio classifier for Hindi TTS language selection.

Public API
----------
pick_tts_language(text: str) -> str
    Returns a Timbre language code based on the script composition of *text*.

    - Pure Devanagari (no meaningful Latin words)  → config.TIMBRE_LANGUAGE  ("hi-IN")
    - Contains at least one Latin word ≥ 2 letters → config.TIMBRE_MIXED_LANGUAGE ("hi-en")

A "meaningful Latin word" means a run of 2+ ASCII letters (a-z/A-Z), excluding
single-letter abbreviations that could be Roman numerals or common loanword connectors.
"""

from __future__ import annotations

import re

import config

# Matches a run of 2+ Latin letters (ASCII a-z/A-Z), word-boundary anchored.
# Intentionally strict: single letters (I, a, A) do not count.
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")


def pick_tts_language(text: str) -> str:
    """Return the appropriate Timbre language code for *text*.

    Parameters
    ----------
    text : str
        The normalized Hindi string (output of normalize_for_tts is fine,
        but raw text works too — only script composition matters here).

    Returns
    -------
    str
        ``config.TIMBRE_LANGUAGE``  if purely Devanagari,
        ``config.TIMBRE_MIXED_LANGUAGE``  if any Latin word ≥ 2 letters is found.
    """
    if _LATIN_WORD_RE.search(text):
        return config.TIMBRE_MIXED_LANGUAGE
    return config.TIMBRE_LANGUAGE


__all__ = ["pick_tts_language"]


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    _cases = [
        ("आप कहाँ जा रहे हैं?",          "should be hi-IN"),
        ("मुझे airport जाना है।",           "should be hi-en"),
        ("मेरा phone नहीं चल रहा है।",     "should be hi-en"),
        ("किराया पाँच सौ रुपये होगा।",     "should be hi-IN"),
        ("यहाँ A भी है।",                   "should be hi-IN (single letter)"),
        ("WhatsApp पर message भेजो।",       "should be hi-en"),
    ]
    for text, note in _cases:
        result = pick_tts_language(text)
        print(f"{result:8s}  ({note})  →  {text}")
