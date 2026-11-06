"""tests/test_ui_logic.py — Unit tests for ui_logic.py (no Gradio, no real models).

Run via:
    python -m pytest tests/test_ui_logic.py -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch, MagicMock

# ── Helper: a typical successful pipeline result dict ─────────────────────────
def _make_result(**overrides):
    base = {
        "audio_path": "clips/c01.wav",
        "direction": "A",
        "scenario": "friend",
        "target_source": "direction",
        "raw_text": "आप कहाँ जा रहे हैं",
        "script_status": "ok",
        "latin_words": [],
        "cleaned_text": "आप कहाँ जा रहे हैं।",
        "detected_register": "Formal",
        "target_register": "informal",
        "transform_skipped": False,
        "skip_reason": None,
        "transform_failed": False,
        "transformed_text": "तू कहाँ जा रहा है।",
        "normalized_text": "तू कहाँ जा रहा है।",
        "tts_language": "hi-IN",
        "output_audio_path": "ui_out/c01_whisper_informal.wav",
        "asr_backend": "whisper",
        "tts_backend": "timbre",
        "asr_ms": 210.0,
        "clean_ms": 2.0,
        "detect_ms": 1.5,
        "transform_ms": 4.0,
        "normalize_ms": 0.5,
        "tts_ms": 1840.0,
        "total_ms": 2060.0,
        "fallback_used": False,
        "asr_fallback_reason": None,
        "tts_fallback_used": False,
        "error": None,
    }
    base.update(overrides)
    return base


class TestRedact(unittest.TestCase):
    """redact() must strip secrets from strings."""

    def setUp(self):
        # Import inside test so env is patched first
        import ui_logic
        self.redact = ui_logic.redact

    def test_redact_gnani_key(self):
        fake_key = "ABCDE12345FGHIJ67890"
        with patch.dict(os.environ, {"GNANI_API_KEY": fake_key}):
            import importlib, ui_logic
            importlib.reload(ui_logic)
            result = ui_logic.redact(f"Error: key={fake_key} was rejected")
        self.assertNotIn(fake_key, result)
        self.assertIn("[hidden]", result)

    def test_redact_hf_token(self):
        from ui_logic import redact
        text = "token=hf_ABCDEFGHIJKLMNOPQRSTUVabcdefghij in request"
        result = redact(text)
        self.assertNotIn("hf_ABCDEFGHIJKLMNOPQRSTUV", result)
        self.assertIn("[hidden]", result)

    def test_redact_api_key_header(self):
        from ui_logic import redact
        text = "X-API-Key-ID: SECRETKEY123456789"
        result = redact(text)
        self.assertNotIn("SECRETKEY123456789", result)
        self.assertIn("[hidden]", result)

    def test_redact_leaves_normal_text_alone(self):
        from ui_logic import redact
        text = "आप कहाँ जा रहे हैं।"
        self.assertEqual(redact(text), text)

    def test_redact_non_string_input(self):
        from ui_logic import redact
        # Should not crash on non-string
        result = redact(12345)
        self.assertIsInstance(result, str)


class TestBuildStatusLine(unittest.TestCase):
    """build_status_line() should produce the right segments."""

    def setUp(self):
        from ui_logic import build_status_line
        self.build = build_status_line

    def test_no_fallback(self):
        line = self.build(_make_result(), cloud_on=True)
        self.assertIn("ASR: whisper", line)
        self.assertIn("TTS: timbre", line)
        self.assertIn("Fallback: none", line)
        self.assertNotIn("Cloud: OFF", line)

    def test_asr_fallback_shown(self):
        result = _make_result(
            fallback_used=True,
            asr_backend="whisper",
            asr_fallback_reason="Prisma HTTP 500",
        )
        line = self.build(result, cloud_on=True)
        self.assertIn("ASR fell back to whisper", line)
        self.assertIn("Prisma HTTP 500", line)

    def test_tts_fallback_shown(self):
        result = _make_result(tts_fallback_used=True, tts_backend="parler")
        line = self.build(result, cloud_on=True)
        self.assertIn("TTS fell back to parler", line)

    def test_cloud_off_shown(self):
        line = self.build(_make_result(), cloud_on=False)
        self.assertIn("Cloud: OFF", line)

    def test_total_time_shown(self):
        line = self.build(_make_result(total_ms=2060.0), cloud_on=True)
        self.assertIn("2.06 s", line)


class TestFriendlyError(unittest.TestCase):
    """friendly_error() maps known cases to plain messages."""

    def setUp(self):
        from ui_logic import friendly_error
        self.friendly = friendly_error

    def test_no_audio(self):
        msg = self.friendly({"error": "no audio"})
        self.assertIn("No audio", msg)

    def test_file_not_found(self):
        msg = self.friendly({"error": "Audio file not found: clips/missing.wav"})
        self.assertIn("could not be found", msg)

    def test_too_long_for_prisma(self):
        msg = self.friendly({"error": "too_long_for_prisma"})
        self.assertIn("60", msg)

    def test_empty_transcript(self):
        msg = self.friendly({"error": "empty_asr_transcript"})
        self.assertIn("No speech detected", msg)

    def test_empty_after_cleanup(self):
        msg = self.friendly({"error": "empty_after_cleanup"})
        self.assertIn("empty after cleanup", msg)

    def test_missing_api_key(self):
        msg = self.friendly({"error": "missing_api_key"})
        self.assertIn("Gnani key not set", msg)

    def test_api_timeout(self):
        msg = self.friendly({"error": "Prisma request timed out after 15s."})
        self.assertIn("timed out", msg)

    def test_http_error_shows_only_code(self):
        msg = self.friendly({"error": "Prisma HTTP 503: service unavailable"})
        self.assertIn("503", msg)
        self.assertNotIn("service unavailable", msg.lower())

    def test_transform_failed(self):
        msg = self.friendly({"error": "transform_failed: RuntimeError..."})
        self.assertIn("transformation failed", msg)

    def test_exception_input(self):
        msg = self.friendly(ValueError("Prisma HTTP 401 blah"))
        self.assertIn("401", msg)

    def test_no_raw_traceback(self):
        # Generic errors: show max 200 chars, no raw stack traces
        msg = self.friendly({"error": "Traceback (most recent call last):\n  File..."})
        # Should be caught as generic
        self.assertIsInstance(msg, str)
        self.assertLess(len(msg), 300)

    def test_no_secret_in_output(self):
        fake_key = "MYSECRETKEY123456789ABCDE"
        with patch.dict(os.environ, {"GNANI_API_KEY": fake_key}):
            import importlib, ui_logic
            importlib.reload(ui_logic)
            msg = ui_logic.friendly_error({"error": f"X-API-Key-ID: {fake_key}"})
        self.assertNotIn(fake_key, msg)


class TestProcessTurn(unittest.TestCase):
    """process_turn() wires direction, target, cloud, session log."""

    def _mock_pipeline_result(self, **overrides):
        return _make_result(**overrides)

    @patch("ui_logic.run_pipeline")
    def test_direction_a_maps_correctly(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result()
        from ui_logic import process_turn
        ui = process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["direction"], "A")
        self.assertIsNone(ui["error_message"] or None)

    @patch("ui_logic.run_pipeline")
    def test_direction_b_maps_correctly(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result()
        from ui_logic import process_turn
        process_turn(
            audio_path="fake.wav",
            direction_label="B: Other -> User",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["direction"], "B")

    @patch("ui_logic.run_pipeline")
    def test_explicit_target_override_passed_through(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result()
        from ui_logic import process_turn
        process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Full Formal",
            cloud_on=True,
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["target_register"], "Full Formal")
        self.assertEqual(call_kwargs["scenario"], "friend")

    @patch("ui_logic.run_pipeline")
    def test_auto_target_passes_none(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result()
        from ui_logic import process_turn
        process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        self.assertIsNone(call_kwargs["target_register"])

    @patch("ui_logic.run_pipeline")
    def test_cloud_off_forces_whisper_asr(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result()
        from ui_logic import process_turn
        process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="prisma",   # requested prisma
            tts_backend="parler",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=False,         # cloud off
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["asr_backend"], "whisper")
        self.assertFalse(call_kwargs["allow_cloud"])

    @patch("ui_logic.run_pipeline")
    def test_cloud_off_switches_timbre_to_parler(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result(tts_backend="parler")
        from ui_logic import process_turn
        ui = process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",   # requested timbre
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=False,         # cloud off
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        # Should have switched to parler
        self.assertEqual(call_kwargs["tts_backend"], "parler")
        # Status line should mention cloud note
        self.assertIn("Timbre disabled", ui["status_line"])

    @patch("ui_logic.run_pipeline")
    def test_cloud_off_mms_stays_mms(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result(tts_backend="mms")
        from ui_logic import process_turn
        process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="mms",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=False,
            session_log=[],
        )
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["tts_backend"], "mms")

    def test_no_audio_returns_friendly_message(self):
        from ui_logic import process_turn
        ui = process_turn(
            audio_path=None,
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        self.assertIn("No audio", ui["error_message"])
        self.assertIsNone(ui["output_audio"])

    @patch("ui_logic.run_pipeline")
    def test_empty_transcript_produces_friendly_message(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result(
            raw_text="",
            cleaned_text="",
            error="empty_asr_transcript",
        )
        from ui_logic import process_turn
        ui = process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        self.assertIn("No speech detected", ui["error_message"])

    @patch("ui_logic.run_pipeline")
    def test_session_log_grows_per_turn(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result()
        from ui_logic import process_turn
        log = []
        for _ in range(3):
            ui = process_turn(
                audio_path="fake.wav",
                direction_label="A: User -> Other",
                scenario_label="Friend",
                asr_backend="whisper",
                tts_backend="timbre",
                voice="Nalini",
                language_mode="Auto (recommended)",
                target_override="Auto (from direction)",
                cloud_on=True,
                session_log=log,
            )
            log = ui["session_log"]

        self.assertEqual(len(log), 3)
        self.assertEqual(log[0]["turn"], 1)
        self.assertEqual(log[2]["turn"], 3)

    @patch("ui_logic.run_pipeline")
    def test_failed_turn_still_logged(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result(
            error="asr_failed: something",
        )
        from ui_logic import process_turn
        log = []
        ui = process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=log,
        )
        self.assertEqual(len(ui["session_log"]), 1)

    @patch("ui_logic.run_pipeline")
    def test_exception_in_pipeline_does_not_escape(self, mock_pipeline):
        """A crashing pipeline must not raise — process_turn catches it and sets error_message."""
        mock_pipeline.side_effect = RuntimeError("completely unexpected crash")
        from ui_logic import process_turn
        ui = process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        # Must have captured the error and returned gracefully — not re-raised
        self.assertTrue(bool(ui["error_message"]), "error_message should be set")
        # Must NOT leak the API key (generic message may include the exception text)
        api_key = os.getenv("GNANI_API_KEY", "")
        if api_key:
            self.assertNotIn(api_key, ui["error_message"])

    @patch("ui_logic.run_pipeline")
    def test_script_warning_shown_for_mixed_script(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_result(
            script_status="mixed",
            latin_words=["hello"],
        )
        from ui_logic import process_turn
        ui = process_turn(
            audio_path="fake.wav",
            direction_label="A: User -> Other",
            scenario_label="Friend",
            asr_backend="whisper",
            tts_backend="timbre",
            voice="Nalini",
            language_mode="Auto (recommended)",
            target_override="Auto (from direction)",
            cloud_on=True,
            session_log=[],
        )
        self.assertIn("Latin letters", ui["script_warning"])


class TestFormatLog(unittest.TestCase):
    """format_log() returns readable markdown."""

    def setUp(self):
        from ui_logic import format_log
        self.format_log = format_log

    def test_empty_log_returns_placeholder(self):
        result = self.format_log([])
        self.assertIn("No turns yet", result)

    def test_log_entries_numbered(self):
        log = [
            {"turn": 1, "direction": "A", "cleaned": "hello", "transformed": "world",
             "asr_backend": "whisper", "tts_backend": "timbre", "total_ms": 100, "error": None},
            {"turn": 2, "direction": "B", "cleaned": "bye", "transformed": "goodbye",
             "asr_backend": "whisper", "tts_backend": "parler", "total_ms": 200, "error": None},
        ]
        result = self.format_log(log)
        self.assertIn("Turn 1", result)
        self.assertIn("Turn 2", result)

    def test_error_turns_show_warning(self):
        log = [
            {"turn": 1, "direction": "A", "cleaned": "", "transformed": "",
             "asr_backend": "whisper", "tts_backend": None, "total_ms": 0, "error": "asr_failed"},
        ]
        result = self.format_log(log)
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
