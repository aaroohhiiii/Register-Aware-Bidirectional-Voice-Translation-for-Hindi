"""app.py — Gradio UI for the Register-Aware Hindi Voice Translation project.

Usage:
    python app.py              # binds to 127.0.0.1 (localhost only)
    python app.py --lan        # binds to 0.0.0.0 (LAN access — uses your Gnani credits!)

Outputs saved to: ui_out/
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys

# ── Must be set BEFORE importing gradio ──────────────────────────────────────
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Force UTF-8 stdout/stderr on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

import asr
import config
from ui_logic import (
    ASR_CHOICES,
    DEFAULT_SCENARIO_LABEL,
    DIRECTION_CHOICES,
    LANGUAGE_CHOICES,
    SCENARIO_LABELS,
    TARGET_CHOICES,
    TTS_CHOICES,
    UI_TIMBRE_VOICES,
    format_log,
    process_turn,
    redact,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Key presence check (shown in UI; never show the key itself) ───────────────
_GNANI_KEY_PRESENT: bool = bool(os.getenv("GNANI_API_KEY", "").strip())
_KEY_STATUS_MSG: str = (
    "🔑 Gnani key: loaded" if _GNANI_KEY_PRESENT else "⚠️ Gnani key: not set"
)
_CLOUD_DEFAULT: bool = _GNANI_KEY_PRESENT


# ── Build the Gradio Blocks UI ────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks object (does NOT launch)."""

    with gr.Blocks(
        title="Register-Aware Hindi Voice Translation",
    ) as demo:

        # ── Session state ─────────────────────────────────────────────────────
        # Persists across calls within a single browser session.
        session_log_state = gr.State([])

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🎙️ Register-Aware Hindi Voice Translation\n"
            "Speak in Hindi — the system detects your register and converts it "
            "for the target social context.\n\n"
            "> **Note:** First local TTS call (Parler / MMS) may take 20–30 s "
            "to load the model."
        )

        # ── Settings row ──────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                scenario_dd = gr.Dropdown(
                    choices=list(SCENARIO_LABELS.values()),
                    value=DEFAULT_SCENARIO_LABEL,
                    label="Scenario",
                    elem_id="scenario_dropdown",
                )
                direction_dd = gr.Dropdown(
                    choices=DIRECTION_CHOICES,
                    value=DIRECTION_CHOICES[0],
                    label="Direction",
                    elem_id="direction_dropdown",
                )
                target_dd = gr.Dropdown(
                    choices=TARGET_CHOICES,
                    value=TARGET_CHOICES[0],
                    label="Target register override",
                    elem_id="target_dropdown",
                )

            with gr.Column(scale=3):
                asr_dd = gr.Dropdown(
                    choices=ASR_CHOICES,
                    value=config.ASR_BACKEND,
                    label="ASR backend",
                    elem_id="asr_dropdown",
                )
                tts_dd = gr.Dropdown(
                    choices=TTS_CHOICES,
                    value=config.TTS_BACKEND,
                    label="TTS backend",
                    elem_id="tts_dropdown",
                )
                voice_dd = gr.Dropdown(
                    choices=UI_TIMBRE_VOICES,
                    value=UI_TIMBRE_VOICES[0],
                    label="Timbre voice (cloud only)",
                    elem_id="voice_dropdown",
                )
                lang_dd = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value=LANGUAGE_CHOICES[0],
                    label="Language mode",
                    elem_id="language_dropdown",
                )

            with gr.Column(scale=2):
                cloud_toggle = gr.Checkbox(
                    value=_CLOUD_DEFAULT,
                    label="Cloud ON",
                    elem_id="cloud_toggle",
                )
                cloud_note_md = gr.Markdown(
                    _cloud_note_text(_CLOUD_DEFAULT),
                    elem_id="cloud_note",
                )
                key_status_md = gr.Markdown(
                    _KEY_STATUS_MSG,
                    elem_id="key_status",
                )

        # Update cloud note when toggle changes
        cloud_toggle.change(
            fn=lambda on: _cloud_note_text(on),
            inputs=[cloud_toggle],
            outputs=[cloud_note_md],
        )

        gr.Markdown("---")

        # ── Main area ─────────────────────────────────────────────────────────
        with gr.Row():
            # Left column: input + controls
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Your audio (record or upload)",
                    elem_id="audio_input",
                )
                with gr.Row():
                    transform_btn = gr.Button(
                        "▶ Transform",
                        variant="primary",
                        elem_id="transform_btn",
                    )
                    clear_btn = gr.Button(
                        "🗑 Clear",
                        variant="secondary",
                        elem_id="clear_btn",
                    )

                error_box = gr.Textbox(
                    label="⚠️ Warnings / Errors",
                    lines=2,
                    interactive=False,
                    elem_id="error_box",
                )
                script_warn_box = gr.Textbox(
                    label="Script warning",
                    lines=1,
                    interactive=False,
                    elem_id="script_warn_box",
                )

            # Right column: pipeline stage outputs
            with gr.Column(scale=3):
                raw_box = gr.Textbox(
                    label="Raw ASR text",
                    lines=2,
                    interactive=False,
                    elem_id="raw_box",
                )
                cleaned_box = gr.Textbox(
                    label="Cleaned text",
                    lines=2,
                    interactive=False,
                    elem_id="cleaned_box",
                )
                with gr.Row():
                    detected_box = gr.Textbox(
                        label="Detected register",
                        lines=1,
                        interactive=False,
                        elem_id="detected_box",
                    )
                    target_box = gr.Textbox(
                        label="Target register",
                        lines=1,
                        interactive=False,
                        elem_id="target_box",
                    )
                transformed_box = gr.Textbox(
                    label="Transformed text",
                    lines=3,
                    interactive=False,
                    elem_id="transformed_box",
                )
                normalized_box = gr.Textbox(
                    label="Text sent to TTS",
                    lines=2,
                    interactive=False,
                    elem_id="normalized_box",
                )
                output_audio = gr.Audio(
                    label="Output audio",
                    interactive=False,
                    autoplay=True,
                    elem_id="output_audio",
                )
                status_md = gr.Markdown(
                    "_No run yet._",
                    elem_id="status_md",
                )

        gr.Markdown("---")

        # ── Conversation log ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Conversation Log")
                log_md = gr.Markdown(
                    "_No turns yet._",
                    elem_id="log_md",
                )
                clear_log_btn = gr.Button(
                    "🗑 Clear log",
                    elem_id="clear_log_btn",
                )

        # ── Wire up the Transform button ──────────────────────────────────────
        def _run(
            audio_path,
            direction_label,
            scenario_label,
            asr_backend,
            tts_backend,
            voice,
            language_mode,
            target_override,
            cloud_on,
            session_log,
        ):
            ui = process_turn(
                audio_path=audio_path,
                direction_label=direction_label,
                scenario_label=scenario_label,
                asr_backend=asr_backend,
                tts_backend=tts_backend,
                voice=voice,
                language_mode=language_mode,
                target_override=target_override,
                cloud_on=cloud_on,
                session_log=session_log,
            )
            return (
                ui["raw"],
                ui["cleaned"],
                ui["detected"],
                ui["target"],
                ui["transformed"],
                ui["normalized"],
                ui["output_audio"],
                ui["status_line"],
                ui["script_warning"],
                ui["error_message"],
                ui["session_log"],
                ui["log_text"],
            )

        transform_btn.click(
            fn=_run,
            inputs=[
                audio_input,
                direction_dd,
                scenario_dd,
                asr_dd,
                tts_dd,
                voice_dd,
                lang_dd,
                target_dd,
                cloud_toggle,
                session_log_state,
            ],
            outputs=[
                raw_box,
                cleaned_box,
                detected_box,
                target_box,
                transformed_box,
                normalized_box,
                output_audio,
                status_md,
                script_warn_box,
                error_box,
                session_log_state,
                log_md,
            ],
            show_progress="full",
        )

        # ── Clear button: reset stage boxes only (keep log) ───────────────────
        def _clear():
            return (
                None,   # audio_input
                "", "", "", "", "", "",  # raw, cleaned, detected, target, transformed, normalized
                None,   # output_audio
                "_No run yet._",  # status_md
                "", "",  # script_warn, error
            )

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[
                audio_input,
                raw_box, cleaned_box, detected_box, target_box,
                transformed_box, normalized_box,
                output_audio,
                status_md,
                script_warn_box, error_box,
            ],
        )

        # ── Clear log button ──────────────────────────────────────────────────
        def _clear_log():
            return [], "_No turns yet._"

        clear_log_btn.click(
            fn=_clear_log,
            inputs=[],
            outputs=[session_log_state, log_md],
        )

    return demo


def _cloud_note_text(cloud_on: bool) -> str:
    """Small markdown note shown below the Cloud toggle."""
    if cloud_on:
        return "_☁️ Your audio is sent to Gnani AI's servers._"
    return "_🔒 Fully offline mode — no data leaves your machine._"


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Hindi Register Conversion Gradio UI."
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        default=False,
        help=(
            "Bind to 0.0.0.0 so anyone on the same LAN can access the app. "
            "WARNING: this exposes your Gnani API credits to the network!"
        ),
    )
    args = parser.parse_args()

    if args.lan:
        server_name = "0.0.0.0"
        print(
            "\n⚠️  LAN mode enabled — anyone on your local network can use this app "
            "and your Gnani AI credits. Ctrl+C to stop.\n"
        )
    else:
        server_name = "127.0.0.1"

    # Preload Whisper once at startup (blocking; ~3 s on first run)
    print("Loading Whisper ASR model — please wait...")
    try:
        asr.preload("whisper")
        print("Models ready. Parler/MMS will load on first use (20–30 s).")
    except Exception as exc:
        print(f"⚠️  Whisper preload failed: {redact(str(exc))}")

    demo = build_demo()
    # concurrency_limit=1 prevents double-clicks and concurrent model calls
    demo.queue(max_size=5).launch(
        server_name=server_name,
        share=False,
        show_error=False,          # don't expose raw Python tracebacks in browser
    )


if __name__ == "__main__":
    main()
