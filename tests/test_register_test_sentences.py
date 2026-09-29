"""Structural validation for the locked 48-sentence register test set."""

from __future__ import annotations

from collections import Counter
import json
import unittest
from pathlib import Path

from indicnlp.tokenize import indic_tokenize


ROOT = Path(__file__).parents[1]


class RegisterTestSentenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(
            (ROOT / "register_test_sentences.json").read_text(encoding="utf-8")
        )
        cls.lookup = json.loads(
            (ROOT / "scenario_lookup.json").read_text(encoding="utf-8")
        )
        cls.records = cls.corpus["sentences"]

    def test_has_48_sequential_unique_records(self) -> None:
        self.assertEqual(len(self.records), 48)
        self.assertEqual([record["id"] for record in self.records], list(range(1, 49)))
        self.assertEqual(len({record["sentence"] for record in self.records}), 48)

    def test_has_exactly_eight_sentences_per_locked_tier(self) -> None:
        locked_tiers = {scenario["tier"] for scenario in self.lookup["scenarios"]}
        counts = Counter(record["tier"] for record in self.records)

        self.assertEqual(set(counts), locked_tiers)
        self.assertEqual(counts, Counter({tier: 8 for tier in locked_tiers}))

    def test_every_record_has_complete_ground_truth_tags(self) -> None:
        allowed_genders = {"masculine", "feminine", None}
        allowed_targets = {"addressee", "speaker", "object", None}
        for record in self.records:
            with self.subTest(record_id=record["id"]):
                self.assertTrue(record["sentence"].strip())
                self.assertIn(record["gender"]["value"], allowed_genders)
                self.assertIn(record["gender"]["applies_to"], allowed_targets)
                self.assertTrue(record["tense_mood"]["key"])
                self.assertTrue(record["tense_mood"]["label"])
                self.assertTrue(record["feature_tested"])
                self.assertTrue(record["feature_tags"])

    def test_every_sentence_is_tokenizable_by_indic_nlp(self) -> None:
        for record in self.records:
            with self.subTest(record_id=record["id"]):
                tokens = indic_tokenize.trivial_tokenize(record["sentence"], lang="hi")
                self.assertGreater(len(tokens), 1)
                self.assertEqual("".join(tokens).replace(" ", ""), record["sentence"].replace(" ", ""))

    def test_representative_register_features_are_present(self) -> None:
        by_id = {record["id"]: record for record in self.records}

        self.assertIn("तू", by_id[1]["sentence"])
        self.assertIn("भैया", by_id[10]["sentence"])
        self.assertIn("आपको", by_id[21]["sentence"])
        self.assertIn("डॉक्टर साहब", by_id[27]["sentence"])
        self.assertIn("दादी जी", by_id[33]["sentence"])
        self.assertIn("जी मैडम", by_id[41]["sentence"])


if __name__ == "__main__":
    unittest.main()

