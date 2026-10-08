"""tests/test_app_builds.py — Verify the Gradio Blocks object builds without launching.

Run via:
    python -m pytest tests/test_app_builds.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

# Must be set before importing gradio or app
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Add project root to path (needed when running from tests/ directly)
_PROJECT_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestAppBuilds(unittest.TestCase):
    """Verify that build_demo() returns a valid gr.Blocks object."""

    def test_build_demo_returns_blocks(self):
        """build_demo() must return a Gradio Blocks instance."""
        import gradio as gr
        from app import build_demo

        demo = build_demo()
        self.assertIsInstance(demo, gr.Blocks)

    def test_blocks_has_expected_elem_ids(self):
        """Key UI components must be present (identified by elem_id)."""
        from app import build_demo

        demo = build_demo()

        # Collect all elem_ids from the built demo
        elem_ids = set()

        def _collect(block):
            eid = getattr(block, "elem_id", None)
            if eid:
                elem_ids.add(eid)
            for child in getattr(block, "children", []) or []:
                _collect(child)

        _collect(demo)

        expected_ids = {
            "scenario_dropdown",
            "direction_dropdown",
            "target_dropdown",
            "asr_dropdown",
            "tts_dropdown",
            "voice_dropdown",
            "language_dropdown",
            "cloud_toggle",
            "audio_input",
            "transform_btn",
            "clear_btn",
            "raw_box",
            "cleaned_box",
            "detected_box",
            "target_box",
            "transformed_box",
            "normalized_box",
            "output_audio",
            "error_box",
            "script_warn_box",
            "log_md",
            "clear_log_btn",
        }

        missing = expected_ids - elem_ids
        self.assertSetEqual(
            missing,
            set(),
            msg=f"Missing elem_ids in built Blocks: {missing}",
        )

    def test_no_share_true(self):
        """app.py must never set share=True in its launch() call."""
        import ast
        from pathlib import Path

        app_source = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
        # Parse AST and look for launch(share=True)
        tree = ast.parse(app_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "launch":
                    for kw in node.keywords:
                        if kw.arg == "share":
                            val = kw.value
                            # share=True is forbidden
                            if isinstance(val, ast.Constant) and val.value is True:
                                self.fail("app.py contains launch(share=True) which is forbidden!")

    def test_analytics_disabled(self):
        """GRADIO_ANALYTICS_ENABLED must be set to False before import."""
        # When this module is imported, os.environ is already set at module level
        self.assertEqual(os.environ.get("GRADIO_ANALYTICS_ENABLED"), "False")

    def test_key_status_message_does_not_contain_key(self):
        """The _KEY_STATUS_MSG constant must never contain the API key value."""
        import app
        import os

        key = os.getenv("GNANI_API_KEY", "")
        if key:
            self.assertNotIn(key, app._KEY_STATUS_MSG)
        # The message should only say "loaded" or "not set"
        self.assertTrue(
            "loaded" in app._KEY_STATUS_MSG or "not set" in app._KEY_STATUS_MSG,
            msg=f"Unexpected key status message: {app._KEY_STATUS_MSG!r}",
        )


if __name__ == "__main__":
    unittest.main()
