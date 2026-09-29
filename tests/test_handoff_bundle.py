"""Integrity check for the self-contained Harshita handoff manifest."""

import json
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class HandoffBundleTests(unittest.TestCase):
    def test_manifest_files_metrics_and_ui_contract(self) -> None:
        manifest = json.loads(
            (PROJECT_DIR / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8")
        )
        for category, files in manifest["files"].items():
            for relative_path in files:
                with self.subTest(category=category, file=relative_path):
                    self.assertTrue((PROJECT_DIR / relative_path).is_file())

        scenarios = json.loads(
            (PROJECT_DIR / "scenario_lookup.json").read_text(encoding="utf-8")
        )["scenarios"]
        sentences = json.loads(
            (PROJECT_DIR / "register_test_sentences.json").read_text(encoding="utf-8")
        )["sentences"]
        metrics = manifest["validated_metrics"]
        self.assertEqual(metrics["scenario_count"], len(scenarios))
        self.assertEqual(metrics["tagged_sentence_count"], len(sentences))
        self.assertEqual(
            (metrics["cross_tier_transformations_passed"], metrics["cross_tier_transformations_total"]),
            (144, 144),
        )
        self.assertEqual((metrics["smoke_passed"], metrics["smoke_total"]), (48, 48))
        self.assertEqual((metrics["detection_correct"], metrics["detection_total"]), (36, 48))

        architecture = (PROJECT_DIR / "REGISTER_SELECTION_ARCHITECTURE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Require the user to select a scenario", architecture)
        self.assertIn("must not silently alter", architecture)


if __name__ == "__main__":
    unittest.main()
