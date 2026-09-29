"""Focused coverage for the two tag-based register tiers."""

import unittest

from register_transformer import transform
from scenario_dial import scenario_to_dial


class SpecialCaseBranchTests(unittest.TestCase):
    def test_scenario_lookup_selects_both_special_tiers(self) -> None:
        self.assertEqual(scenario_to_dial("elder_relative"), "Formal-Warm")
        self.assertEqual(scenario_to_dial("customer_facing"), "Reverse Formal")

    def test_formal_warm_uses_validated_kinship_override(self) -> None:
        self.assertEqual(
            transform(
                "तुम कैसी हो?",
                "Formal-Warm",
                "F",
                scenario="elder_relative",
                address_term="नानी जी",
            ),
            "नानी जी, आप कैसी हैं?",
        )

    def test_formal_warm_preserves_existing_relation_without_duplication(self) -> None:
        self.assertEqual(
            transform(
                "दादी, तुम कैसी हो?",
                "Formal-Warm",
                "F",
                scenario="elder_relative",
            ),
            "दादी जी, आप कैसी हैं?",
        )

    def test_reverse_formal_uses_customer_facing_override(self) -> None:
        self.assertEqual(
            transform(
                "तुम्हारा ऑर्डर तैयार है",
                "Reverse Formal",
                "F",
                scenario="customer_facing",
                address_term="मैडम",
            ),
            "मैडम, आपका ऑर्डर तैयार है",
        )

    def test_reverse_formal_preserves_masculine_self_reference(self) -> None:
        self.assertEqual(
            transform(
                "क्या मैं तेरी कोई सहायता कर सकता हूं?",
                "Reverse Formal",
                "F",
                scenario="customer_facing",
                address_term="मैडम",
            ),
            "मैडम, क्या मैं आपकी कोई सहायता कर सकता हूं?",
        )

    def test_reverse_formal_preserves_self_reference_in_mixed_clause(self) -> None:
        self.assertEqual(
            transform(
                "आप थोड़ा रुकिए, मैं अभी लाता हूं",
                "Reverse Formal",
                "M",
                scenario="customer_facing",
                address_term="सर",
            ),
            "सर, आप थोड़ा रुकिए, मैं अभी लाता हूं",
        )

    def test_reverse_formal_preserves_feminine_self_reference(self) -> None:
        self.assertEqual(
            transform(
                "जी मैडम, मैं अभी लाती हूं",
                "Reverse Formal",
                "F",
                scenario="customer_facing",
                address_term="मैडम",
            ),
            "मैडम, मैं अभी लाती हूं",
        )

    def test_invalid_special_case_context_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires scenario"):
            transform(
                "तुम ठीक हो?",
                "Formal-Warm",
                scenario="customer_facing",
            )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            transform(
                "तुम ठीक हो?",
                "Formal-Warm",
                scenario="elder_relative",
                address_term="मैडम",
            )


if __name__ == "__main__":
    unittest.main()
