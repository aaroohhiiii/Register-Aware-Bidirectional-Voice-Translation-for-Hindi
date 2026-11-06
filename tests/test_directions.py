"""tests/test_directions.py — Unit tests for Task 5 direction logic and conversation runner.

Run via:
    python -m unittest discover -s tests -p "test_directions.py"
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from conversation import run_conversation
from directions import direction_to_target
from pipeline import run_pipeline


class TestDirections(unittest.TestCase):
    def test_direction_to_target_valid(self):
        self.assertEqual(direction_to_target("A", "friend"), "Full Informal")
        self.assertEqual(direction_to_target("a", "auto_driver"), "Respectful Casual")
        self.assertEqual(direction_to_target("A", "receptionist"), "Formal")
        self.assertEqual(direction_to_target("A", "doctor"), "Full Formal")
        self.assertEqual(direction_to_target("b", "friend"), "Full Informal")
        self.assertEqual(direction_to_target("b", "auto_driver"), "Respectful Casual")
        self.assertEqual(direction_to_target("B", "receptionist"), "Formal")
        self.assertEqual(direction_to_target("B", "doctor"), "Full Formal")

    def test_persona_tiers_match_scenario_lookup(self):
        import config
        from scenario_dial import scenario_to_dial

        self.assertEqual(
            list(config.SCENARIOS), ["friend", "auto_driver", "receptionist", "doctor"]
        )
        for key, data in config.SCENARIOS.items():
            self.assertEqual(data["A_target"], scenario_to_dial(key), key)
            self.assertEqual(data["B_target"], data["A_target"], key)

    def test_direction_to_target_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            direction_to_target("C", "friend")

        with self.assertRaises(ValueError):
            direction_to_target("A", "unknown_scenario_xyz")

    @patch("pipeline.tts.speak")
    @patch("pipeline.language_hint.pick_tts_language")
    @patch("pipeline.normalize.normalize_for_tts")
    @patch("pipeline.register_transformer.transform")
    @patch("pipeline.register_detector.detect_register")
    @patch("pipeline.cleanup.clean")
    @patch("pipeline.script_check.check_script")
    @patch("pipeline.asr.transcribe")
    def test_run_pipeline_direction_target_resolution(
        self, mock_asr, mock_script, mock_clean, mock_detect, mock_transform,
        mock_norm, mock_hint, mock_speak
    ):
        mock_asr.return_value = {
            "text": "आप कहाँ जा रहे हैं",
            "asr_backend_requested": "whisper",
            "asr_backend_used": "whisper",
            "asr_ms": 10.0,
            "fallback_used": False,
            "fallback_reason": None,
            "error": None,
        }
        mock_script.return_value = {"status": "ok", "latin_words": []}
        mock_clean.return_value = "आप कहाँ जा रहे हैं।"
        mock_detect.return_value = "Formal"
        mock_transform.return_value = "तू कहाँ जा रहा है।"
        mock_norm.return_value = "तू कहाँ जा रहा है।"
        mock_hint.return_value = "hi-IN"
        mock_speak.return_value = "e2e_out/test.wav"

        # Direction B -> persona auto_driver B_target = Respectful Casual
        res = run_pipeline("clips/c01.wav", direction="B", scenario="auto_driver", synthesize=False)
        self.assertEqual(res["target_register"], "Respectful Casual")
        self.assertEqual(res["target_source"], "direction")

        # Explicit target_register overrides direction
        res_exp = run_pipeline("clips/c01.wav", target_register="formal", direction="B", scenario="auto_driver", synthesize=False)
        self.assertEqual(res_exp["target_register"], "formal")
        self.assertEqual(res_exp["target_source"], "explicit")

        # Legacy mode (direction=None, target_register=None)
        res_leg = run_pipeline("clips/c01.wav", synthesize=False)
        self.assertEqual(res_leg["target_source"], "legacy_opposite")

    @patch("pipeline.register_transformer.transform")
    @patch("pipeline.register_detector.detect_register")
    @patch("pipeline.cleanup.clean")
    @patch("pipeline.script_check.check_script")
    @patch("pipeline.asr.transcribe")
    def test_unknown_detected_register_skips_transform(
        self, mock_asr, mock_script, mock_clean, mock_detect, mock_transform
    ):
        mock_asr.return_value = {
            "text": "कुछ अजीब टेक्स्ट",
            "asr_backend_requested": "whisper",
            "asr_backend_used": "whisper",
            "asr_ms": 10.0,
            "fallback_used": False,
            "fallback_reason": None,
            "error": None,
        }
        mock_script.return_value = {"status": "ok", "latin_words": []}
        mock_clean.return_value = "कुछ अजीब टेक्स्ट।"
        mock_detect.return_value = "unknown"

        res = run_pipeline("clips/c01.wav", synthesize=False)
        self.assertTrue(res["transform_skipped"])
        self.assertEqual(res["skip_reason"], "register_unknown")
        self.assertEqual(res["transformed_text"], "कुछ अजीब टेक्स्ट।")
        mock_transform.assert_not_called()

    @patch("pipeline.register_transformer.transform")
    @patch("pipeline.register_detector.detect_register")
    @patch("pipeline.cleanup.clean")
    @patch("pipeline.script_check.check_script")
    @patch("pipeline.asr.transcribe")
    def test_each_persona_transforms_to_its_exact_tier(
        self, mock_asr, mock_script, mock_clean, mock_detect, mock_transform
    ):
        mock_asr.return_value = {
            "text": "तुम कहाँ जा रहे हो",
            "asr_backend_used": "whisper",
            "asr_ms": 10.0,
            "fallback_used": False,
            "fallback_reason": None,
            "error": None,
        }
        mock_script.return_value = {"status": "ok", "latin_words": []}
        mock_clean.return_value = "तुम कहाँ जा रहे हो?"
        mock_detect.return_value = "Respectful Casual"
        mock_transform.side_effect = lambda text, target_tier: f"{target_tier}: {text}"

        cases = {
            "friend": "Full Informal",
            "receptionist": "Formal",
            "doctor": "Full Formal",
        }
        for persona, tier in cases.items():
            res = run_pipeline("clips/c01.wav", direction="A", scenario=persona, synthesize=False)
            self.assertEqual(res["target_register"], tier, persona)
            self.assertFalse(res["transform_skipped"], persona)
            self.assertEqual(mock_transform.call_args[1]["target_tier"], tier, persona)

        # Auto driver: speaker already used Respectful Casual, so nothing changes.
        mock_transform.reset_mock()
        res = run_pipeline("clips/c01.wav", direction="A", scenario="auto_driver", synthesize=False)
        self.assertEqual(res["target_register"], "Respectful Casual")
        self.assertTrue(res["transform_skipped"])
        self.assertEqual(res["skip_reason"], "target_equals_detected")
        mock_transform.assert_not_called()

        # Formal speech to the receptionist is not skipped as "already formal"
        # when it is actually Full Formal.
        mock_detect.return_value = "Full Formal"
        res = run_pipeline("clips/c01.wav", direction="A", scenario="receptionist", synthesize=False)
        self.assertFalse(res["transform_skipped"])
        self.assertEqual(mock_transform.call_args[1]["target_tier"], "Formal")

        # Unknown register still gets the persona's tier applied.
        mock_detect.return_value = "unknown"
        res = run_pipeline("clips/c01.wav", direction="A", scenario="doctor", synthesize=False)
        self.assertFalse(res["transform_skipped"])
        self.assertEqual(mock_transform.call_args[1]["target_tier"], "Full Formal")

    @patch("conversation.run_pipeline")
    def test_run_conversation_continues_on_failing_turn(self, mock_run_pipeline):
        mock_run_pipeline.side_effect = [
            {
                "cleaned_text": "तू मुझे स्टेशन छोड़ देगा क्या?",
                "transformed_text": "क्या आप मुझे स्टेशन छोड़ देंगे?",
                "error": None,
            },
            RuntimeError("ASR failure on turn 2"),
            {
                "cleaned_text": "थोड़ा कम कर दो",
                "transformed_text": "कृपया थोड़ा कम कर दीजिए",
                "error": None,
            },
        ]

        turns = [
            {"direction": "A", "audio_path": "clip1.wav"},
            {"direction": "B", "audio_path": "clip2.wav"},
            {"direction": "A", "audio_path": "clip3.wav"},
        ]

        res = run_conversation(turns, scenario="auto_driver", synthesize=False)

        self.assertEqual(len(res["turns"]), 3)
        self.assertIsNone(res["turns"][0].get("error"))
        self.assertIsNotNone(res["turns"][1].get("error"))
        self.assertIn("turn_failed", res["turns"][1]["error"])
        self.assertEqual(len(res["transcript"]), 3)
        self.assertTrue(res["transcript"][0].startswith("[A] user said:"))


if __name__ == "__main__":
    unittest.main()
