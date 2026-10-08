"""Filler word and disfluency cleanup module for transcribed Hindi speech.

Cleans up common hesitation markers and filler particles before register transformation.
"""

from __future__ import annotations

import re

# Common Hindi and Hinglish speech fillers and hesitation sounds
DEFAULT_FILLERS = [
    r"\bअम्म+\b",
    r"\bउम्म+\b",
    r"\bआह+\b",
    r"\bऊह+\b",
    r"\buh+\b",
    r"\bum+\b",
    r"\bah+\b",
    r"\ber+\b",
    r"\bमतलब\b(?=\s*,|\s+कि|\s+मतलब)",  # repeated or hesitative "मतलब"
    r"\bयानी\b(?=\s*,|\s+यानी)",
    r"\bजैसे\s+कि\b(?=\s*,)",
    r"\bलाइक\b",
    r"\blike\b",
]

COMPILED_FILLERS = [re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_FILLERS]


def remove_fillers(text: str, custom_fillers: list[str] | None = None) -> str:
    """Remove filler words and repeated hesitation tokens from transcribed text.

    Args:
        text (str): Raw transcribed input text.
        custom_fillers (list[str] | None): Optional extra regex patterns.

    Returns:
        str: Cleaned text without filler particles.
    """
    if not text or not text.strip():
        return ""

    cleaned = text
    patterns = COMPILED_FILLERS
    if custom_fillers:
        patterns = patterns + [re.compile(p, re.IGNORECASE) for p in custom_fillers]

    for pattern in patterns:
        cleaned = pattern.sub("", cleaned)

    # Collapse multiple spaces and clean up loose punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?।])", r"\1", cleaned)
    return cleaned


__all__ = ["remove_fillers"]
