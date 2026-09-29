"""Tests for Indic tokenization and Hindi POS/role tagging."""

from __future__ import annotations

import unittest

from hindi_tokenizer import tokenize_and_tag


class HindiTokenizerTests(unittest.TestCase):
    def test_formal_sentence_flags_all_target_roles(self) -> None:
        tokens = tokenize_and_tag("आप कहाँ जा रहे हैं जी?")
        by_text = {token.text: token for token in tokens}

        self.assertEqual(
            [token.text for token in tokens],
            ["आप", "कहाँ", "जा", "रहे", "हैं", "जी", "?"],
        )
        self.assertEqual(by_text["आप"].role, "pronoun")
        self.assertTrue(by_text["आप"].is_pronoun)
        self.assertEqual(by_text["जा"].role, "verb")
        self.assertTrue(by_text["जा"].is_verb)
        self.assertEqual(by_text["रहे"].role, "verb")
        self.assertTrue(by_text["रहे"].is_auxiliary)
        self.assertEqual(by_text["हैं"].role, "verb")
        self.assertEqual(by_text["जी"].role, "honorific_particle")
        self.assertTrue(by_text["जी"].is_honorific_particle)
        self.assertFalse(by_text["जी"].is_verb)
        self.assertEqual(by_text["?"].role, "other")

    def test_informal_pronoun_and_auxiliary(self) -> None:
        tokens = tokenize_and_tag("तुम खाना खा रहे हो।")
        by_text = {token.text: token for token in tokens}

        self.assertEqual(by_text["तुम"].role, "pronoun")
        self.assertEqual(by_text["खा"].upos, "VERB")
        self.assertEqual(by_text["रहे"].upos, "AUX")
        self.assertEqual(by_text["हो"].upos, "AUX")
        self.assertEqual(by_text["खाना"].role, "other")

    def test_honorific_lexicon_overrides_statistical_pos(self) -> None:
        tokens = tokenize_and_tag("कृपया बैठिए सर जी।")
        by_text = {token.text: token for token in tokens}

        for marker in ("कृपया", "सर", "जी"):
            self.assertEqual(by_text[marker].role, "honorific_particle")
            self.assertTrue(by_text[marker].is_honorific_particle)
        self.assertEqual(by_text["बैठिए"].role, "verb")

    def test_unknown_tokens_are_preserved_as_other(self) -> None:
        tokens = tokenize_and_tag("राम दिल्ली गया।")

        self.assertEqual([token.text for token in tokens], ["राम", "दिल्ली", "गया", "।"])
        self.assertEqual(tokens[0].role, "other")
        self.assertEqual(tokens[-1].role, "other")

    def test_empty_sentence_is_rejected(self) -> None:
        for invalid in ("", "   "):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    tokenize_and_tag(invalid)


if __name__ == "__main__":
    unittest.main()
