import json
from pathlib import Path
import unittest

from register_detection_baseline import evaluate_detection, run_round_trip_smoke_test
from register_detector import TIERS, detect_register, detect_register_details


PROJECT_DIR = Path(__file__).resolve().parents[1]


class RegisterDetectorTests(unittest.TestCase):
    def test_detect_register_returns_one_locked_tier(self) -> None:
        examples = {
            "तू क्या कर रहा है?": "Full Informal",
            "तुम कितने बजे आते हो?": "Respectful Casual",
            "आप कहाँ रहते हैं?": "Formal",
            "सर, क्या मैं अंदर आ सकता हूं?": "Full Formal",
            "दादी जी, आप कैसी हैं?": "Formal-Warm",
            "सर, आपका ऑर्डर तैयार है": "Reverse Formal",
        }
        for sentence, expected in examples.items():
            with self.subTest(sentence=sentence):
                self.assertEqual(detect_register(sentence), expected)
                self.assertIn(detect_register(sentence), TIERS)

    def test_details_explain_scores_and_confidence(self) -> None:
        result = detect_register_details("दादी जी, आप कैसी हैं?")
        self.assertEqual(result.tier, "Formal-Warm")
        self.assertGreater(result.scores["Formal-Warm"], result.scores["Formal"])
        self.assertTrue(result.evidence)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            detect_register("")

    def test_locked_48_sentence_baseline(self) -> None:
        baseline = evaluate_detection()
        self.assertEqual(baseline["total"], 48)
        self.assertEqual(baseline["correct"], 36)
        self.assertEqual(baseline["accuracy_percent"], 75.0)
        failures = {row["sentence_id"] for row in baseline["rows"] if not row["correct"]}
        self.assertEqual(failures, {5, 6, 8, 26, 30, 31, 34, 35, 37, 38, 39, 40})
        self.assertEqual(baseline["per_tier"]["Respectful Casual"]["correct"], 8)
        self.assertEqual(baseline["per_tier"]["Formal"]["correct"], 8)
        self.assertEqual(baseline["per_tier"]["Reverse Formal"]["correct"], 8)
        self.assertEqual(
            baseline["confusion_pairs"],
            {
                "Full Informal -> Respectful Casual": 3,
                "Full Formal -> Formal": 3,
                "Formal-Warm -> Formal": 4,
                "Formal-Warm -> Full Formal": 2,
            },
        )

    def test_detection_to_transform_smoke_test(self) -> None:
        smoke = run_round_trip_smoke_test()
        self.assertEqual(smoke["total"], 48)
        self.assertEqual(smoke["passed"], 48)
        failures = [row for row in smoke["rows"] if not row["sensible"]]
        self.assertEqual(failures, [])
        for row in smoke["rows"]:
            self.assertNotEqual(row["detected_source_tier"], row["target_tier"])

    def test_transform_failure_log_is_component_traceable(self) -> None:
        with (PROJECT_DIR / "transformation_failure_log.json").open(encoding="utf-8") as source:
            log = json.load(source)
        self.assertEqual(log["summary"]["total_cases"], 144)
        self.assertEqual(log["summary"]["failed"], 0)
        self.assertEqual(log["summary"]["historical_failures"], len(log["failures"]))
        self.assertEqual(log["summary"]["resolved_in_step_6_1"], len(log["failures"]))
        self.assertEqual({failure["component"] for failure in log["failures"]}, {"pronoun_table"})
        for failure in log["failures"]:
            self.assertEqual(failure["status"], "resolved_step_6_1")
            self.assertTrue(failure["resolution"])
            self.assertTrue(failure["rule_source"])
            self.assertTrue(failure["cause"])
            self.assertTrue(failure["expected"])
            self.assertTrue(failure["actual"])


if __name__ == "__main__":
    unittest.main()
