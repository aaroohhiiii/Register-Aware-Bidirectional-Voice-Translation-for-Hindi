"""Coverage and agreement checks for the Day 4 data tables."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class Day4RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lookup = json.loads((ROOT / "scenario_lookup.json").read_text(encoding="utf-8"))
        cls.verbs = json.loads((ROOT / "verb_rules.json").read_text(encoding="utf-8"))
        cls.addresses = json.loads(
            (ROOT / "address_term_rules.json").read_text(encoding="utf-8")
        )
        cls.verb_by_tier = {entry["tier"]: entry for entry in cls.verbs["tiers"]}
        cls.address_by_tier = {
            entry["tier"]: entry for entry in cls.addresses["tier_rules"]
        }

    def test_both_tables_cover_exactly_the_six_locked_tiers(self) -> None:
        expected = {scenario["tier"] for scenario in self.lookup["scenarios"]}

        self.assertEqual(set(self.verb_by_tier), expected)
        self.assertEqual(set(self.address_by_tier), expected)
        self.assertEqual(len(self.verbs["tiers"]), 6)
        self.assertEqual(len(self.addresses["tier_rules"]), 6)

    def test_each_tier_has_both_genders_and_all_four_categories(self) -> None:
        categories = set(self.verbs["categories"])
        self.assertEqual(
            categories,
            {"present_habitual", "present_progressive", "past_completive", "modal_potential"},
        )
        for tier, entry in self.verb_by_tier.items():
            with self.subTest(tier=tier):
                self.assertEqual(set(entry["gender_patterns"]), {"masculine", "feminine"})
                for patterns in entry["gender_patterns"].values():
                    self.assertEqual(set(patterns), categories)

    def test_tu_tum_and_aap_endings(self) -> None:
        full_informal = self.verb_by_tier["Full Informal"]
        respectful = self.verb_by_tier["Respectful Casual"]
        formal = self.verb_by_tier["Formal"]

        self.assertEqual(
            full_informal["gender_patterns"]["masculine"]["present_habitual"],
            "{stem}ता है",
        )
        self.assertEqual(
            full_informal["gender_patterns"]["feminine"]["past_completive"],
            "{stem} चुकी है",
        )
        self.assertEqual(
            respectful["gender_patterns"]["masculine"]["modal_potential"],
            "{stem} सकते हो",
        )
        self.assertEqual(
            respectful["gender_patterns"]["feminine"]["present_progressive"],
            "{stem} रही हो",
        )
        self.assertEqual(
            formal["gender_patterns"]["masculine"]["past_completive"],
            "{stem} चुके हैं",
        )
        self.assertEqual(
            formal["gender_patterns"]["feminine"]["modal_potential"],
            "{stem} सकती हैं",
        )

    def test_all_aap_tiers_share_respectful_plural_morphology(self) -> None:
        aap_tiers = ["Formal", "Full Formal", "Formal-Warm", "Reverse Formal"]
        baseline = self.verb_by_tier["Formal"]["gender_patterns"]
        for tier in aap_tiers:
            with self.subTest(tier=tier):
                self.assertEqual(self.verb_by_tier[tier]["canonical_subject"], "आप")
                self.assertEqual(self.verb_by_tier[tier]["gender_patterns"], baseline)

    def test_full_informal_records_tum_variant(self) -> None:
        variant = self.verb_by_tier["Full Informal"]["accepted_casual_plural_variant"]

        self.assertEqual(variant["subject"], "तुम")
        self.assertEqual(
            variant["gender_patterns"],
            self.verb_by_tier["Respectful Casual"]["gender_patterns"],
        )

    def test_address_presence_policies_and_special_overrides(self) -> None:
        self.assertEqual(self.address_by_tier["Full Informal"]["presence_policy"], "optional")
        self.assertEqual(self.address_by_tier["Full Formal"]["presence_policy"], "required")
        self.assertEqual(
            self.address_by_tier["Formal-Warm"]["presence_policy"],
            "required_scenario_specific",
        )
        self.assertEqual(self.address_by_tier["Formal-Warm"]["generic_terms"], [])
        self.assertIn("दादी जी", self.addresses["scenario_overrides"]["elder_relative"]["terms"])
        self.assertEqual(
            self.addresses["scenario_overrides"]["customer_facing"]["presence_policy"],
            "required",
        )

    def test_asr_reliability_warning_requires_normalization(self) -> None:
        reliability = self.addresses["asr_reliability"]

        self.assertEqual(reliability["level"], "low")
        self.assertTrue(reliability["normalization_layer_required"])


if __name__ == "__main__":
    unittest.main()

