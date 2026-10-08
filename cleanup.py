"""cleanup.py — Pre-processing and cleanup layer for ASR transcripts in Hindi.

Public API
----------
clean(text: str, backend: str = "common") -> str
    Deterministic, idempotent transcript cleaner.
"""

from __future__ import annotations

import re
import unicodedata

# ─── Filler Lists ────────────────────────────────────────────────────────────

FILLERS_COMMON = [
    "उम्म", "उम", "अम्म", "अं", "आँ", "ऊँ", "हम्म", "आह", "एह", "ईई",
    "um", "uh", "umm", "uhh", "er", "ah", "hmm", "hm",
]
FILLERS_WHISPER: list[str] = []   # can be extended after observing Whisper transcripts
FILLERS_PRISMA: list[str] = []    # can be extended after observing Prisma transcripts


# ─── Discourse Words ─────────────────────────────────────────────────────────

DISCOURSE_WORDS = {"मतलब", "यानी", "तो", "वो", "ना", "अच्छा", "वैसे"}


# Ensure all whitelist entries are NFC normalized
REDUPLICATION_WHITELIST = {
    unicodedata.normalize("NFC", w) for w in [
        "धीरे धीरे",
        "बार बार",
        "जल्दी जल्दी",
        "कभी कभी",
        "थोड़ा थोड़ा",
        "अलग अलग",
        "साथ साथ",
        "पास पास",
        "ठीक ठीक",
        "सीधा सीधा",
        "और और",
        "हाँ हाँ",
        "नहीं नहीं",
    ]
}


# ─── Helper Regex Patterns ───────────────────────────────────────────────────

_MULTISPACE = re.compile(r"[ \t\r\f\v]+")
_MULTILINE = re.compile(r"\n+")
_ELLIPSIS = re.compile(r"\.{2,}")
_MULTICOMMA = re.compile(r",\s*,+")


def _get_fillers(backend: str) -> list[str]:
    fillers = list(FILLERS_COMMON)
    b = backend.lower()
    if b == "whisper":
        fillers.extend(FILLERS_WHISPER)
    elif b == "prisma":
        fillers.extend(FILLERS_PRISMA)
    return sorted(fillers, key=len, reverse=True)


def _remove_fillers(text: str, fillers: list[str]) -> str:
    """Remove filler sounds as standalone tokens or with trailing ellipses/commas."""
    for filler in fillers:
        # Match filler surrounded by word boundaries / punctuation / whitespace
        pattern = rf"(?i)(?<![\w\u0900-\u097F]){re.escape(filler)}(?![\w\u0900-\u097F])[,\s\.\…]*"
        text = re.sub(pattern, " ", text)
    return text


def _remove_hesitation_discourse(text: str) -> str:
    """Remove discourse words ONLY when they act as hesitations.

    - At the start of string followed by comma, ellipsis, or colon/pause mark.
    - Stacked introductory discourse words at the start (e.g. "अच्छा, तो ").
    - Isolated between commas (e.g. ", मतलब, ").
    """
    words_pattern = "|".join(re.escape(w) for w in DISCOURSE_WORDS)

    # 1. Stacked leading hesitation words: e.g. "अच्छा, तो " -> ""
    stacked_lead_pattern = rf"^\s*(?:{words_pattern})\s*[,;\.\…\-–—]+\s*(?:{words_pattern})\s+"
    text = re.sub(stacked_lead_pattern, "", text)

    # 2. Leading hesitation words followed strictly by comma, ellipsis, or dash
    # e.g. "मतलब, ", "अच्छा, ", "वो, " (NOT "तो आप")
    lead_comma_pattern = rf"^\s*(?:{words_pattern})\s*[,;\.\…\-–—]+\s*"
    while True:
        new_text = re.sub(lead_comma_pattern, "", text)
        if new_text == text:
            break
        text = new_text

    # 3. Medial isolated discourse words between commas
    # e.g. "आप, मतलब, यहाँ बैठिए" -> "आप, यहाँ बैठिए"
    medial_pattern = rf",\s*(?:{words_pattern})\s*,"
    while True:
        new_text = re.sub(medial_pattern, ",", text)
        if new_text == text:
            break
        text = new_text

    return text


def _collapse_stutter(text: str) -> str:
    """Collapse immediate duplicate words (stuttering) unless whitelisted."""
    tokens = text.split()
    if len(tokens) < 2:
        return text

    cleaned_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        curr_token = tokens[i]
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]

            # Strip punctuation for comparison
            curr_clean = unicodedata.normalize("NFC", re.sub(r"[^\w\u0900-\u097F]", "", curr_token))
            next_clean = unicodedata.normalize("NFC", re.sub(r"[^\w\u0900-\u097F]", "", next_token))

            pair_clean = f"{curr_clean} {next_clean}"
            if curr_clean and curr_clean == next_clean:
                if pair_clean not in REDUPLICATION_WHITELIST:
                    # Stutter detected: skip curr_token, advance to next
                    i += 1
                    continue

        cleaned_tokens.append(curr_token)
        i += 1

    return " ".join(cleaned_tokens)


def _remove_truncated_fragments(text: str) -> str:
    """Remove single character dangling fragments at start/end of string or standalone."""
    tokens = text.split()
    if not tokens:
        return ""

    # Check lone consonant / single character at start or end if detached
    filtered = []
    for idx, tok in enumerate(tokens):
        clean_tok = re.sub(r"[^\w]", "", tok)
        # Keep single-letter words that are legitimate in Hindi: "न", "व" if context, but strip detached Latin consonants or lone punctuation
        if len(clean_tok) == 1 and clean_tok in "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ":
            continue
        filtered.append(tok)

    return " ".join(filtered)


def _normalize_punctuation(text: str) -> str:
    """Normalize terminal punctuation: '.' at the end -> '।', keep '?' and '!'."""
    text = _MULTISPACE.sub(" ", text).strip()
    if not text:
        return ""

    # Replace lone or multiple trailing commas / ellipses
    text = re.sub(r"[,;\s\.\…]+$", "", text).strip()
    if not text:
        return ""

    # If ends with '.', replace with '।'
    if text.endswith("."):
        text = text[:-1].rstrip() + "।"
    elif not text.endswith(("।", "?", "!")):
        text += "।"

    return text


def clean(text: str, backend: str = "common") -> str:
    """Clean ASR transcript deterministically and idempotently.

    Parameters
    ----------
    text : str
        Raw transcript from ASR.
    backend : str, optional
        ASR backend ("common", "whisper", "prisma").

    Returns
    -------
    str
        Cleaned, hesitation-free, normalized text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode NFC & basic whitespace cleanup
    text = unicodedata.normalize("NFC", text)
    text = _MULTILINE.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    if not text:
        return ""

    # 2. Remove filler sounds
    fillers = _get_fillers(backend)
    text = _remove_fillers(text, fillers)
    text = _MULTISPACE.sub(" ", text).strip()

    # 3. Remove hesitation discourse words (start/between commas)
    text = _remove_hesitation_discourse(text)
    text = _MULTISPACE.sub(" ", text).strip()

    # 4. Collapse stuttering repetitions (respecting REDUPLICATION_WHITELIST)
    text = _collapse_stutter(text)

    # 5. Remove dangling cut-off fragments
    text = _remove_truncated_fragments(text)

    # 6. Normalize punctuation
    text = _normalize_punctuation(text)

    # Clean leftover empty commas / spaces
    text = re.sub(r"^\s*,\s*", "", text)
    text = _MULTISPACE.sub(" ", text).strip()

    return text


__all__ = ["clean", "FILLERS_COMMON", "FILLERS_WHISPER", "FILLERS_PRISMA", "REDUPLICATION_WHITELIST", "DISCOURSE_WORDS"]
