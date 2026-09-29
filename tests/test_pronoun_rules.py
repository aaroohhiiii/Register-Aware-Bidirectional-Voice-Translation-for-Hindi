"""Completeness and morphology checks for the Step 3.3 pronoun table."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


LOOKUP_PATH = Path(__file__).parents[1] / "scenario_lookup.json"
RULES_PATH = Path(__file__).parents[1] / "pronoun_rules.json"


class PronounRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        cls.lookup = json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))
        cls.by_tier = {entry["tier"]: entry for entry in cls.rules["tiers"]}

    def test_all_six_scenario_tiers_are_defined_once(self) -> None:
        scenario_tiers = {scenario["tier"] for scenario in self.lookup["scenarios"]}

        self.assertEqual(set(self.by_tier), scenario_tiers)
        self.assertEqual(len(self.rules["tiers"]), 6)

    def test_every_tier_has_all_required_cases(self) -> None:
        possessive_cases = {
            "masculine_singular_direct",
            "feminine_any_number",
            "masculine_plural_or_oblique",
        }
        for tier, entry in self.by_tier.items():
            with self.subTest(tier=tier):
                forms = entry["forms"]
                self.assertEqual(
                    set(forms),
                    {"subject", "possessive", "dative", "ergative", "postpositional"},
                )
                self.assertEqual(set(forms["possessive"]), possessive_cases)
                self.assertEqual(
                    set(forms["dative"]), {"canonical", "accepted_variant"}
                )
                self.assertEqual(
                    set(forms["postpositional"]),
                    {"instrumental_ablative", "locative_in", "locative_on", "terminative"},
                )
                for specification in forms["postpositional"].values():
                    self.assertEqual(
                        set(specification), {"canonical", "accepted_variants"}
                    )
                    self.assertTrue(specification["canonical"])

    def test_full_informal_uses_tu_paradigm(self) -> None:
        forms = self.by_tier["Full Informal"]["forms"]

        self.assertEqual(forms["subject"], "तू")
        self.assertEqual(set(forms["possessive"].values()), {"तेरा", "तेरी", "तेरे"})
        self.assertEqual(forms["dative"]["canonical"], "तुझे")
        self.assertEqual(forms["ergative"], "तूने")
        self.assertEqual(
            forms["postpositional"]["instrumental_ablative"]["canonical"],
            "तुझसे",
        )

    def test_respectful_casual_uses_tum_paradigm(self) -> None:
        forms = self.by_tier["Respectful Casual"]["forms"]

        self.assertEqual(forms["subject"], "तुम")
        self.assertEqual(
            set(forms["possessive"].values()), {"तुम्हारा", "तुम्हारी", "तुम्हारे"}
        )
        self.assertEqual(forms["dative"]["canonical"], "तुम्हें")
        self.assertEqual(forms["ergative"], "तुमने")
        self.assertEqual(
            forms["postpositional"]["instrumental_ablative"]["canonical"],
            "तुमसे",
        )

    def test_all_four_formal_tiers_use_aap_paradigm(self) -> None:
        expected_tiers = {"Formal", "Full Formal", "Formal-Warm", "Reverse Formal"}
        for tier in expected_tiers:
            with self.subTest(tier=tier):
                entry = self.by_tier[tier]
                forms = entry["forms"]
                self.assertEqual(entry["paradigm"], "aap")
                self.assertEqual(forms["subject"], "आप")
                self.assertEqual(
                    set(forms["possessive"].values()), {"आपका", "आपकी", "आपके"}
                )
                self.assertEqual(forms["dative"]["canonical"], "आपको")
                self.assertEqual(forms["ergative"], "आपने")
                self.assertEqual(
                    forms["postpositional"]["instrumental_ablative"]["canonical"],
                    "आपसे",
                )

    def test_case_audit_status_is_closed(self) -> None:
        self.assertEqual(self.rules["schema_version"], 2)
        self.assertEqual(self.rules["status"], "closed_step_6_2")


if __name__ == "__main__":
    unittest.main()
