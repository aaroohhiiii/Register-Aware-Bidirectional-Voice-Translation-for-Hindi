"""Tests for the scenario-to-register lookup."""

from __future__ import annotations

import unittest

from scenario_dial import scenario_to_dial


class ScenarioToDialTests(unittest.TestCase):
    def test_all_locked_scenarios(self) -> None:
        expected = {
            "friend": "Full Informal",
            "younger_sibling": "Full Informal",
            "spouse_partner": "Full Informal",
            "classmate": "Full Informal",
            "auto_driver": "Respectful Casual",
            "shopkeeper": "Respectful Casual",
            "delivery_person": "Respectful Casual",
            "colleague_peer": "Respectful Casual",
            "stranger_directions": "Respectful Casual",
            "elder_relative": "Formal-Warm",
            "landlord": "Formal",
            "bank_employee": "Formal",
            "receptionist": "Formal",
            "hotel_staff": "Formal",
            "boss_manager": "Formal",
            "teacher": "Full Formal",
            "doctor": "Full Formal",
            "job_interviewer": "Full Formal",
            "police_officer": "Full Formal",
            "customer_facing": "Reverse Formal",
        }

        actual = {key: scenario_to_dial(key) for key in expected}

        self.assertEqual(actual, expected)

    def test_unknown_scenario_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "^Unknown scenario: tourist$"):
            scenario_to_dial("tourist")

    def test_keys_are_explicit_and_case_sensitive(self) -> None:
        with self.assertRaisesRegex(ValueError, "^Unknown scenario: Friend$"):
            scenario_to_dial("Friend")


if __name__ == "__main__":
    unittest.main()

