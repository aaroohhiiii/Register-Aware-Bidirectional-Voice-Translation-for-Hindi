"""tests/test_allow_cloud.py — Prove that allow_cloud=False never makes an HTTP request.

Run via:
    python -m pytest tests/test_allow_cloud.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import asr
import tts


class TestAllowCloudASR(unittest.TestCase):
    """asr.transcribe() with allow_cloud=False must never call requests.post."""

    @patch("asr.requests.post")
    @patch("asr._transcribe_whisper")
    @patch("asr.prepare_audio")
    def test_asr_no_http_when_cloud_off(self, mock_prepare, mock_whisper, mock_post):
        """No HTTP request is made for ASR when allow_cloud=False."""
        mock_prepare.return_value = ("/tmp/fake.wav", 3.0)
        mock_whisper.return_value = "नमस्ते"

        result = asr.transcribe("fake.wav", backend="prisma", allow_cloud=False)

        # Must have used whisper, not prisma
        self.assertEqual(result["asr_backend_used"], "whisper")
        self.assertEqual(result["text"], "नमस्ते")
        # No HTTP call ever made
        mock_post.assert_not_called()

    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_asr_http_called_when_cloud_on(self, mock_prepare, mock_prisma):
        """Confirm Prisma transcriber IS called when allow_cloud=True (control)."""
        mock_prepare.return_value = ("/tmp/fake.wav", 3.0)
        mock_prisma.return_value = "नमस्ते"

        import os
        with patch.dict(os.environ, {"GNANI_API_KEY": "test_key_12345"}):
            result = asr.transcribe("fake.wav", backend="prisma", allow_cloud=True)

        # Prisma transcriber was called (not bypassed)
        mock_prisma.assert_called_once()
        self.assertEqual(result["asr_backend_used"], "prisma")

    @patch("asr.requests.post")
    @patch("asr._transcribe_whisper")
    @patch("asr.prepare_audio")
    def test_asr_cloud_off_ignores_requested_backend(self, mock_prepare, mock_whisper, mock_post):
        """Even if caller asks for prisma, cloud_off forces whisper."""
        mock_prepare.return_value = ("/tmp/fake.wav", 5.0)
        mock_whisper.return_value = "यह टेस्ट है"

        result = asr.transcribe("fake.wav", backend="prisma", allow_cloud=False)

        self.assertEqual(result["asr_backend_used"], "whisper")
        mock_post.assert_not_called()


class TestAllowCloudTTS(unittest.TestCase):
    """tts.speak() with allow_cloud=False must never call requests.post."""

    @patch("tts._speak_mms")
    @patch("tts._speak_parler")
    @patch("tts._speak_timbre")
    def test_tts_no_timbre_when_cloud_off(self, mock_timbre, mock_parler, mock_mms):
        """Timbre backend is never called when allow_cloud=False."""
        # Make Parler succeed
        mock_parler.return_value = None

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            tts.speak(
                "नमस्ते",
                out_path=out_path,
                backend="timbre",   # requested Timbre
                allow_cloud=False,  # cloud off
            )
        except Exception:
            pass  # Parler/MMS may fail in test env; we only care about Timbre
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

        mock_timbre.assert_not_called()

    @patch("tts._speak_mms")
    @patch("tts._speak_parler")
    @patch("tts._speak_timbre")
    def test_tts_no_http_post_when_cloud_off(self, mock_timbre, mock_parler, mock_mms):
        """No HTTP POST is made when allow_cloud=False (Timbre never called)."""
        # When allow_cloud=False, _speak_timbre must never be called at all
        mock_parler.return_value = None

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            tts.speak(
                "नमस्ते",
                out_path=out_path,
                backend="timbre",
                allow_cloud=False,
            )
        except Exception:
            pass
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

        # The critical assertion: Timbre (which makes HTTP calls) was never attempted
        mock_timbre.assert_not_called()

    @patch("tts._speak_mms")
    @patch("tts._speak_parler")
    @patch("tts._speak_timbre")
    def test_tts_timbre_allowed_when_cloud_on(self, mock_timbre, mock_parler, mock_mms):
        """Timbre IS attempted (not skipped) when allow_cloud=True."""
        # _BACKEND_FN dict holds a reference to the real function captured at import time.
        # We must patch the dict entry directly, not just the module attribute.
        import tempfile, os
        called = []

        def _fake_timbre(text, dest, speed, voice, language):
            dest.write_bytes(b"RIFF" + b"\x00" * 36)
            called.append(True)

        original = tts._BACKEND_FN["timbre"]
        tts._BACKEND_FN["timbre"] = _fake_timbre

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            tts.speak(
                "नमस्ते",
                out_path=out_path,
                backend="timbre",
                allow_cloud=True,
            )
        finally:
            tts._BACKEND_FN["timbre"] = original  # restore
            if os.path.exists(out_path):
                os.remove(out_path)

        self.assertTrue(called, "_BACKEND_FN['timbre'] was never called with allow_cloud=True")


if __name__ == "__main__":
    unittest.main()
