"""tests/test_pipeline_stub.py — Unit tests for run_pipeline with mocked stages.

Run via:
    python -m unittest tests/test_pipeline_stub.py
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline import run_pipeline


class TestPipelineStub(unittest.TestCase):
    REQUIRED_KEYS = {
        "audio_path", "raw_text", "script_status", "latin_words",
        "cleaned_text", "detected_register", "target_register",
        "transform_skipped", "transform_failed", "transformed_text",
        "normalized_text", "tts_language", "output_audio_path",
        "asr_backend", "tts_backend", "asr_ms", "clean_ms",
        "detect_ms", "transform_ms", "normalize_ms", "tts_ms",
        "total_ms", "fallback_used", "asr_fallback_reason",
        "tts_fallback_used", "error",
        # Task 5 bidirectional keys
        "direction", "scenario", "target_source", "skip_reason",
    }

    @patch("pipeline.tts.speak")
    @patch("pipeline.language_hint.pick_tts_language")
    @patch("pipeline.normalize.normalize_for_tts")
    @patch("pipeline.register_transformer.transform")
    @patch("pipeline.register_detector.detect_register")
    @patch("pipeline.cleanup.clean")
    @patch("pipeline.script_check.check_script")
    @patch("pipeline.asr.transcribe")
    def test_pipeline_dict_schema_and_success_flow(
        self, mock_asr, mock_script, mock_clean, mock_detect, mock_transform,
        mock_norm, mock_hint, mock_speak
    ):
        mock_asr.return_value = {
            "text": "आप कहाँ जा रहे हैं",
            "asr_backend_requested": "whisper",
            "asr_backend_used": "whisper",
            "asr_ms": 100.0,
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

        res = run_pipeline("clips/c01.wav", synthesize=True)

        self.assertEqual(set(res.keys()), self.REQUIRED_KEYS)
        self.assertEqual(res["raw_text"], "आप कहाँ जा रहे हैं")
        self.assertEqual(res["cleaned_text"], "आप कहाँ जा रहे हैं।")
        self.assertEqual(res["detected_register"], "Formal")
        self.assertEqual(res["target_register"], "informal")
        self.assertEqual(res["transformed_text"], "तू कहाँ जा रहा है।")
        self.assertFalse(res["transform_failed"])
        self.assertIsNone(res["error"])

    @patch("pipeline.register_transformer.transform")
    @patch("pipeline.register_detector.detect_register")
    @patch("pipeline.cleanup.clean")
    @patch("pipeline.script_check.check_script")
    @patch("pipeline.asr.transcribe")
    def test_pipeline_stage_exception_captured_in_error(
        self, mock_asr, mock_script, mock_clean, mock_detect, mock_transform
    ):
        mock_asr.return_value = {
            "text": "आप कहाँ जा रहे हैं",
            "asr_backend_requested": "whisper",
            "asr_backend_used": "whisper",
            "asr_ms": 100.0,
            "fallback_used": False,
            "fallback_reason": None,
            "error": None,
        }
        mock_script.return_value = {"status": "ok", "latin_words": []}
        mock_clean.return_value = "आप कहाँ जा रहे हैं।"
        mock_detect.return_value = "Formal"
        mock_transform.side_effect = RuntimeError("Transformation rule engine error")

        # Exception should NOT crash run_pipeline; it should capture error in dict
        res = run_pipeline("clips/c01.wav", synthesize=False)

        self.assertEqual(set(res.keys()), self.REQUIRED_KEYS)
        self.assertTrue(res["transform_failed"])
        self.assertEqual(res["transformed_text"], "आप कहाँ जा रहे हैं।")
        self.assertIsNotNone(res["error"])
        self.assertIn("transform_failed", res["error"])

    @patch("pipeline.asr.transcribe")
    def test_pipeline_asr_failure_captured(self, mock_asr):
        mock_asr.return_value = {
            "text": "",
            "asr_backend_requested": "prisma",
            "asr_backend_used": "prisma",
            "asr_ms": 50.0,
            "fallback_used": False,
            "fallback_reason": "Prisma HTTP 500",
            "error": "Prisma HTTP 500",
        }

        res = run_pipeline("clips/c01.wav", synthesize=False)

        self.assertEqual(set(res.keys()), self.REQUIRED_KEYS)
        self.assertIsNotNone(res["error"])
        self.assertIn("Prisma HTTP 500", res["error"])


if __name__ == "__main__":
    unittest.main()
