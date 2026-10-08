"""test_normalize.py — Unit tests for normalize_for_tts."""

import pytest
from normalize import normalize_for_tts


def test_integer_conversion():
    assert "पाँच सौ" in normalize_for_tts("मेरे पास 500 रुपये हैं")
    assert "एक हज़ार" in normalize_for_tts("1000")


def test_devanagari_digits():
    res = normalize_for_tts("५००")
    assert "पाँच सौ" in res


def test_comma_grouped_digits():
    res = normalize_for_tts("1,500")
    assert "एक हज़ार पाँच सौ" in res


def test_decimals():
    res = normalize_for_tts("2.5")
    assert "दो दशमलव पाँच" in res


def test_currency_rupees_symbol():
    res = normalize_for_tts("₹500")
    assert "रुपये" in res
    assert "पाँच सौ" in res


def test_currency_rs_prefix():
    res = normalize_for_tts("Rs. 100")
    assert "सौ रुपये" in res


def test_percentage():
    res = normalize_for_tts("50%")
    assert "पचास प्रतिशत" in res


def test_time_conversion():
    res = normalize_for_tts("मीटिंग 5:30 पर है")
    assert "पाँच बजकर तीस मिनट" in res


def test_terminal_punctuation_added():
    res = normalize_for_tts("आप कैसे हैं")
    assert res.endswith("।") or res.endswith("?")


def test_repeated_punctuation_collapsed():
    res = normalize_for_tts("क्या हाल है???")
    assert "???" not in res
    assert res.endswith("?")


def test_stray_symbols_removed():
    res = normalize_for_tts("नमस्ते * # _ ~ दुनिया")
    assert "*" not in res
    assert "#" not in res
    assert "_" not in res


def test_idempotence():
    sample = "मेरे पास ₹500 हैं और 5:30 बजे मीटिंग है!"
    first_pass = normalize_for_tts(sample)
    second_pass = normalize_for_tts(first_pass)
    assert first_pass == second_pass


def test_mixed_english_words_preserved():
    res = normalize_for_tts("मैं office जा रहा हूँ")
    assert "office" in res
    assert "जा रहा हूँ" in res
