"""normalize.py — Pre-TTS text normalizer for Hindi.

Public API
----------
normalize_for_tts(text: str) -> str
    Returns a clean, TTS-ready Hindi string. The function is deterministic
    and idempotent: normalize_for_tts(normalize_for_tts(x)) == normalize_for_tts(x).

Processing pipeline (in order)
-------------------------------
a. Unicode NFC; strip; collapse repeated whitespace / newlines → single space.
b. Convert numbers (Latin & Devanagari digits) to Hindi spoken words:
   - Integers:        500      → "पाँच सौ"
   - Decimals:        2.5      → "दो दशमलव पाँच"
   - Comma-grouped:   1,500    → treated as 1500 → "एक हज़ार पाँच सौ"
   - Devanagari:      ५०० → 500 → same path
c. Currency:
   - ₹ / Rs. / Rs  before or after a number → "<number> रुपये"
   - bare "<number> रुपये" already in text → left alone
   - % → "प्रतिशत"
d. Times: HH:MM → "<H> बजकर <M> मिनट"  (handles 5:30, 12:05, etc.)
e. Punctuation cleanup:
   - Remove stray: * # _ ~ ^ | and emoji / non-print characters
   - Collapse repeated punctuation: ??→?, !!→!, ...→।
   - Ensure sentence ends with terminal punctuation (? ! or ।)
   - Devanagari danda "।" is preferred terminal mark.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Match


# ─── Devanagari digit map ────────────────────────────────────────────────────

_DEVA_DIGIT = str.maketrans("०१२३४५६७८९", "0123456789")


def _deva_to_latin(text: str) -> str:
    """Replace Devanagari digit characters with ASCII equivalents."""
    return text.translate(_DEVA_DIGIT)


# ─── Native Hindi Number Words (0–99) ────────────────────────────────────────

_HINDI_NUMBERS_0_TO_99 = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच",
    6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस",
    11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह",
    16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस",
    21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस", 30: "तीस",
    31: "इकत्तीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस",
    36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस", 40: "चालीस",
    41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चौवालीस", 45: "पैंतालीस",
    46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास",
    51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन",
    56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 60: "साठ",
    61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ",
    66: "छियासठ", 67: "सरसठ", 68: "अड़सठ", 69: "उनहत्तर", 70: "सत्तर",
    71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी", 80: "अस्सी",
    81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 90: "नब्बे",
    91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पंचानवे",
    96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे",
}


@lru_cache(maxsize=1024)
def _int_to_hindi(n: int) -> str:
    """Convert a non-negative integer to Hindi spoken words."""
    if n < 0:
        return f"ऋण {_int_to_hindi(abs(n))}"
    if n in _HINDI_NUMBERS_0_TO_99:
        return _HINDI_NUMBERS_0_TO_99[n]

    parts: list[str] = []

    # Crores (1,00,00,000)
    if n >= 10000000:
        crore_val = n // 10000000
        n %= 10000000
        parts.append(f"{_int_to_hindi(crore_val)} करोड़")

    # Lakhs (1,00,000)
    if n >= 100000:
        lakh_val = n // 100000
        n %= 100000
        parts.append(f"{_int_to_hindi(lakh_val)} लाख")

    # Thousands (1,000)
    if n >= 1000:
        thousand_val = n // 1000
        n %= 1000
        parts.append(f"{_int_to_hindi(thousand_val)} हज़ार")

    # Hundreds (100)
    if n >= 100:
        hundred_val = n // 100
        n %= 100
        parts.append(f"{_int_to_hindi(hundred_val)} सौ")

    # Remaining 1-99
    if n > 0:
        parts.append(_HINDI_NUMBERS_0_TO_99[n])

    return " ".join(parts)


@lru_cache(maxsize=256)
def _float_to_hindi(whole: int, frac_str: str) -> str:
    """Convert decimal numbers to '<whole> दशमलव <digit digit…>'."""
    whole_words = _int_to_hindi(whole)
    digit_words = " ".join(_int_to_hindi(int(d)) for d in frac_str)
    return f"{whole_words} दशमलव {digit_words}"


def _number_to_hindi(match: Match) -> str:
    """Regex substitution callback for numbers."""
    raw = match.group(0).replace(",", "")
    if "." in raw:
        parts = raw.split(".", 1)
        try:
            whole = int(parts[0]) if parts[0] else 0
            frac = parts[1]
            return _float_to_hindi(whole, frac)
        except ValueError:
            return match.group(0)
    else:
        try:
            return _int_to_hindi(int(raw))
        except ValueError:
            return match.group(0)


_NUMBER_RE = re.compile(
    r"\b"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"   # integer (with or without commas)
    r"(?:\.\d+)?"                    # optional decimal part
    r"\b"
)


# ─── Currency Expansion ──────────────────────────────────────────────────────

_CURRENCY_PREFIX_RE = re.compile(
    r"(?:₹|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)"
)
_CURRENCY_SUFFIX_RE = re.compile(
    r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:₹|Rs\.?|INR)(?!\s*रुपये)"
)


def _expand_currency(text: str) -> str:
    """Convert ₹500 / Rs 500 / 500 Rs into '<words> रुपये'."""
    def _replace_prefix(m: Match) -> str:
        num_str = m.group(1).replace(",", "")
        if "." in num_str:
            whole, frac = num_str.split(".", 1)
            words = _float_to_hindi(int(whole or 0), frac)
        else:
            words = _int_to_hindi(int(num_str))
        return f"{words} रुपये"

    def _replace_suffix(m: Match) -> str:
        num_str = m.group(1).replace(",", "")
        if "." in num_str:
            whole, frac = num_str.split(".", 1)
            words = _float_to_hindi(int(whole or 0), frac)
        else:
            words = _int_to_hindi(int(num_str))
        return f"{words} रुपये"

    text = _CURRENCY_PREFIX_RE.sub(_replace_prefix, text)
    text = _CURRENCY_SUFFIX_RE.sub(_replace_suffix, text)
    return text


# ─── Percentage Expansion ────────────────────────────────────────────────────

_PERCENT_RE = re.compile(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*%")


def _expand_percent(text: str) -> str:
    """Convert '50%' → '50 प्रतिशत' (digits expanded by subsequent pass)."""
    return _PERCENT_RE.sub(r"\1 प्रतिशत", text)


# ─── Time Expansion ──────────────────────────────────────────────────────────

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _expand_time(text: str) -> str:
    """Convert '5:30' → 'पाँच बजकर तीस मिनट'."""
    def _replace(m: Match) -> str:
        hour = int(m.group(1))
        minute = int(m.group(2))
        h_words = _int_to_hindi(hour)
        if minute == 0:
            return f"{h_words} बजे"
        m_words = _int_to_hindi(minute)
        return f"{h_words} बजकर {m_words} मिनट"

    return _TIME_RE.sub(_replace, text)


# ─── Punctuation & Whitespace Cleanup ────────────────────────────────────────

_STRAY_SYMBOLS = re.compile(r"[*#_~^|`$@&<>+=/\\]")
_REPEATED_QUESTION = re.compile(r"\?+")
_REPEATED_EXCLAMATION = re.compile(r"!+")
_REPEATED_DANDA = re.compile(r"।+")
_REPEATED_DOTS = re.compile(r"\.{2,}")
_MULTISPACE = re.compile(r"[ \t\r\f\v]+")
_MULTILINE = re.compile(r"\n+")


def _clean_punctuation(text: str) -> str:
    """Strip stray symbols, collapse repeated punctuation, ensure terminal mark."""
    text = _STRAY_SYMBOLS.sub(" ", text)
    text = _REPEATED_QUESTION.sub("?", text)
    text = _REPEATED_EXCLAMATION.sub("!", text)
    text = _REPEATED_DOTS.sub("।", text)
    text = _REPEATED_DANDA.sub("।", text)

    text = _MULTISPACE.sub(" ", text).strip()

    # Ensure sentence ends with valid terminal punctuation
    if text and not text.endswith(("।", "?", "!")):
        text += "।"

    return text


# ─── Public API ──────────────────────────────────────────────────────────────

def normalize_for_tts(text: str) -> str:
    """Convert input Hindi text into a normalized, TTS-ready string.

    Deterministic and idempotent: normalize_for_tts(f(x)) == f(x).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode NFC normalisation & collapse whitespace
    text = unicodedata.normalize("NFC", text)
    text = _MULTILINE.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()

    # 2. Devanagari digits to ASCII for consistent regex matching
    text = _deva_to_latin(text)

    # 3. Time patterns (before bare numbers)
    text = _expand_time(text)

    # 4. Currency patterns
    text = _expand_currency(text)

    # 5. Percentage patterns
    text = _expand_percent(text)

    # 6. Remaining bare numbers to Hindi words
    text = _NUMBER_RE.sub(_number_to_hindi, text)

    # 7. Punctuation cleanup and terminal mark formatting
    text = _clean_punctuation(text)

    return text


__all__ = ["normalize_for_tts"]
