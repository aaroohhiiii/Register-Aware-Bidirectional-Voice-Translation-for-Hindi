"""tests/test_cleanup.py — Unit tests for cleanup.py."""

from __future__ import annotations

import pytest
from cleanup import clean, REDUPLICATION_WHITELIST


# ─── 1. Fillers Removal (12 cases) ───────────────────────────────────────────

def test_filler_umm_hindi():
    assert "मैं घर जा रहा हूँ" in clean("उम्म मैं घर जा रहा हूँ")


def test_filler_um_hindi():
    assert "आप बैठिए" in clean("उम आप बैठिए")


def test_filler_amm():
    assert "यहाँ रुकिए" in clean("अम्म, यहाँ रुकिए")


def test_filler_an():
    assert "क्या बात है" in clean("अं क्या बात है")


def test_filler_aan():
    assert "मुझे जाने दो" in clean("आँ मुझे जाने दो")


def test_filler_oon():
    assert "यह ठीक है" in clean("ऊँ यह ठीक है")


def test_filler_hmm():
    assert "मैं समझ गया" in clean("हम्म मैं समझ गया")


def test_filler_aah():
    assert "दर्द हो रहा है" in clean("आह दर्द हो रहा है")


def test_filler_eh():
    assert "यह क्या है" in clean("एह यह क्या है")


def test_filler_ee():
    assert "चलते हैं" in clean("ईई चलते हैं")


def test_filler_um_english():
    assert "I will go" in clean("um I will go")


def test_filler_uh_english():
    assert "please wait" in clean("uh, please wait")


# ─── 2. Discourse Words in Normal Clauses (Kept) ─────────────────────────────

def test_discourse_matlab_clause():
    res = clean("इसका मतलब क्या है?")
    assert "मतलब" in res
    assert res == "इसका मतलब क्या है?"


def test_discourse_to_clause():
    res = clean("तो आप कहाँ जा रहे हैं?")
    assert "तो" in res
    assert res == "तो आप कहाँ जा रहे हैं?"


def test_discourse_wo_clause():
    res = clean("मुझे वो किताब चाहिए।")
    assert "वो" in res


def test_discourse_achha_clause():
    res = clean("यह खाना बहुत अच्छा है।")
    assert "अच्छा" in res


def test_discourse_na_clause():
    res = clean("आप ना मत बोलिए।")
    assert "ना" in res


# ─── 3. Discourse Words as Hesitations (Removed) ─────────────────────────────

def test_discourse_matlab_hesitation_start():
    res = clean("मतलब, आप यहाँ रुकिए।")
    assert "मतलब" not in res
    assert res == "आप यहाँ रुकिए।"


def test_discourse_achha_to_stacked_hesitation():
    res = clean("अच्छा, तो आपका नाम क्या है?")
    assert res == "आपका नाम क्या है?"


def test_discourse_wo_hesitation():
    res = clean("वो, तुम धीरे धीरे चलो।")
    assert "वो" not in res
    assert "तुम धीरे धीरे चलो।" in res


def test_discourse_medial_hesitation():
    res = clean("आप, मतलब, यहाँ बैठिए।")
    assert "मतलब" not in res
    assert "आप" in res and "यहाँ बैठिए" in res


# ─── 4. Stutter Collapsing ───────────────────────────────────────────────────

def test_stutter_main_main():
    res = clean("मैं मैं बस स्टैंड जाना चाहता हूँ।")
    assert "मैं मैं" not in res
    assert "मैं बस स्टैंड जाना चाहता हूँ।" in res


def test_stutter_aap_aap():
    res = clean("आप आप कहाँ जा रहे हैं?")
    assert "आप आप" not in res
    assert "आप कहाँ जा रहे हैं?" in res


# ─── 5. Whitelisted Reduplications Preserved ─────────────────────────────────

@pytest.mark.parametrize("phrase", list(REDUPLICATION_WHITELIST))
def test_whitelisted_reduplications(phrase: str):
    res = clean(f"वह {phrase} करता है।")
    assert phrase in res


# ─── 6. Whitespace and Punctuation ───────────────────────────────────────────

def test_whitespace_and_newlines():
    res = clean("   नमस्ते \n\n   दुनिया   ")
    assert res == "नमस्ते दुनिया।"


def test_dot_to_danda():
    res = clean("हम आ रहे हैं.")
    assert res.endswith("।")
    assert not res.endswith(".")


def test_question_and_exclamation_preserved():
    assert clean("क्या आप तैयार हैं?").endswith("?")
    assert clean("शाबाश!").endswith("!")


def test_empty_after_clean():
    assert clean("") == ""
    assert clean("   \n\t  ") == ""
    assert clean("उम्म... uh... um...") == ""


# ─── 7. Idempotence ──────────────────────────────────────────────────────────

def test_idempotence():
    sample = "उम्म... मतलब, मैं मैं धीरे धीरे जा रहा हूँ."
    pass1 = clean(sample)
    pass2 = clean(pass1)
    assert pass1 == pass2
    assert "धीरे धीरे" in pass1
    assert "उम्म" not in pass1
    assert "मतलब" not in pass1
    assert "मैं मैं" not in pass1


# ─── 8. Hard Rules: Register Pronouns & Verb Endings Untouched ───────────────

def test_hard_rules_formal():
    text = "आप कहाँ जा रहे हैं?"
    assert clean(text) == text


def test_hard_rules_casual():
    text = "तुम कहाँ जा रहे हो?"
    assert clean(text) == text


def test_hard_rules_informal():
    text = "तू कहाँ जा रहा है?"
    assert clean(text) == text
