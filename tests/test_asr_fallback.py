"""tests/test_asr_fallback.py — Unit tests for ASR fallback logic with mocked network/models.

Run via:
    python -m unittest tests/test_asr_fallback.py
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import asr


class TestASRFallback(unittest.TestCase):
    def setUp(self):
        # Set a dummy key for testing
        os.environ["GNANI_API_KEY"] = "SECRET_TEST_KEY_12345"

    def tearDown(self):
        pass

    @patch("asr._transcribe_whisper")
    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_prisma_timeout_fallback(self, mock_prep, mock_prisma, mock_whisper):
        mock_prep.return_value = ("dummy.wav", 5.0)
        mock_prisma.side_effect = RuntimeError("Prisma request timed out after 15s.")
        mock_whisper.return_value = "Whisper fallback transcript"

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=True)

        self.assertEqual(res["asr_backend_requested"], "prisma")
        self.assertEqual(res["asr_backend_used"], "whisper")
        self.assertTrue(res["fallback_used"])
        self.assertIn("timed out", res["fallback_reason"])
        self.assertEqual(res["text"], "Whisper fallback transcript")
        self.assertNotIn("SECRET_TEST_KEY_12345", str(res))

    @patch("asr._transcribe_whisper")
    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_prisma_http_500_fallback(self, mock_prep, mock_prisma, mock_whisper):
        mock_prep.return_value = ("dummy.wav", 5.0)
        mock_prisma.side_effect = RuntimeError("Prisma HTTP 500: Internal Error")
        mock_whisper.return_value = "Whisper fallback transcript"

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=True)

        self.assertEqual(res["asr_backend_used"], "whisper")
        self.assertTrue(res["fallback_used"])
        self.assertIn("HTTP 500", res["fallback_reason"])
        self.assertEqual(res["text"], "Whisper fallback transcript")
        self.assertNotIn("SECRET_TEST_KEY_12345", str(res))

    @patch("asr._transcribe_whisper")
    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_prisma_missing_key_fallback(self, mock_prep, mock_prisma, mock_whisper):
        mock_prep.return_value = ("dummy.wav", 5.0)
        mock_prisma.side_effect = RuntimeError("missing_api_key")
        mock_whisper.return_value = "Whisper fallback transcript"

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=True)

        self.assertEqual(res["asr_backend_used"], "whisper")
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["fallback_reason"], "missing_api_key")
        self.assertEqual(res["text"], "Whisper fallback transcript")

    @patch("asr._transcribe_whisper")
    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_prisma_empty_transcript_fallback(self, mock_prep, mock_prisma, mock_whisper):
        mock_prep.return_value = ("dummy.wav", 5.0)
        mock_prisma.side_effect = RuntimeError("empty_transcript")
        mock_whisper.return_value = "Whisper fallback transcript"

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=True)

        self.assertEqual(res["asr_backend_used"], "whisper")
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["fallback_reason"], "empty_transcript")
        self.assertEqual(res["text"], "Whisper fallback transcript")

    @patch("asr._transcribe_whisper")
    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_clip_too_long_fallback(self, mock_prep, mock_prisma, mock_whisper):
        mock_prep.return_value = ("dummy.wav", 65.0)
        mock_prisma.side_effect = ValueError("too_long_for_prisma")
        mock_whisper.return_value = "Whisper fallback transcript"

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=True)

        self.assertEqual(res["asr_backend_used"], "whisper")
        self.assertTrue(res["fallback_used"])
        self.assertIn("too_long_for_prisma", res["fallback_reason"])
        self.assertEqual(res["text"], "Whisper fallback transcript")

    @patch("asr._transcribe_whisper")
    @patch("asr._transcribe_prisma")
    @patch("asr.prepare_audio")
    def test_allow_fallback_false_returns_error(self, mock_prep, mock_prisma, mock_whisper):
        mock_prep.return_value = ("dummy.wav", 5.0)
        mock_prisma.side_effect = RuntimeError("Prisma HTTP 500: Server Error")

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=False)

        self.assertEqual(res["asr_backend_requested"], "prisma")
        self.assertEqual(res["asr_backend_used"], "prisma")
        self.assertFalse(res["fallback_used"])
        self.assertEqual(res["text"], "")
        self.assertIsNotNone(res["error"])
        self.assertIn("HTTP 500", res["error"])
        mock_whisper.assert_not_called()

    @patch("builtins.open")
    @patch("asr.requests.post")
    @patch("asr.prepare_audio")
    def test_secret_key_never_leaked_in_error(self, mock_prep, mock_post, mock_open):
        secret = "SUPER_SECRET_GNANI_KEY_999"
        os.environ["GNANI_API_KEY"] = secret
        mock_prep.return_value = ("dummy.wav", 5.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = f"Unauthorized access with key {secret}"
        mock_post.return_value = mock_resp

        res = asr.transcribe("dummy.wav", backend="prisma", allow_fallback=False)

        self.assertNotIn(secret, str(res))
        self.assertIn("***", res["error"])


if __name__ == "__main__":
    unittest.main()
