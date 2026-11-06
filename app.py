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
    "Gnani key loaded." if _GNANI_KEY_PRESENT else "Gnani key not set."
)
_CLOUD_DEFAULT: bool = _GNANI_KEY_PRESENT


# ── Look & feel ───────────────────────────────────────────────────────────────
# Plain light green on white: solid colours, thin borders, no gradients.

_GREEN = gr.themes.Color(
    name="green_formal",
    c50="#f1f7f0", c100="#e4f0e3", c200="#cfe3cd", c300="#a9cda6",
    c400="#7bb078", c500="#4f9150", c600="#3b7f4b", c700="#2e6b47",
    c800="#25583a", c900="#1e4630", c950="#12291c",
)
_GREEN_NEUTRAL = gr.themes.Color(
    name="green_neutral",
    c50="#f7faf6", c100="#eef4ed", c200="#dde8dc", c300="#c5d6c4",
    c400="#9fb59e", c500="#7a937b", c600="#5b7262", c700="#46594b",
    c800="#34443a", c900="#24322a", c950="#16201a",
)

THEME = gr.themes.Base(
    primary_hue=_GREEN,
    secondary_hue=_GREEN,
    neutral_hue=_GREEN_NEUTRAL,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Source Sans 3"), "Noto Sans Devanagari", "sans-serif"],
).set(
    body_background_fill="#f3f7f2",
    body_text_color="#1e3527",
    body_text_color_subdued="#5b7262",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f7faf6",
    block_background_fill="#ffffff",
    block_border_color="#d9e6d8",
    block_border_width="1px",
    block_label_text_color="#2e6b47",
    block_label_background_fill="#ffffff",
    block_title_text_color="#1e3527",
    block_shadow="none",
    block_radius="8px",
    input_background_fill="#ffffff",
    input_border_color="#cddfcc",
    input_border_width="1px",
    input_border_color_focus="#4f9150",
    input_shadow_focus="none",
    button_primary_background_fill="#2e6b47",
    button_primary_background_fill_hover="#25583a",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#2e6b47",
    button_primary_shadow="none",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#eef4ed",
    button_secondary_text_color="#2e6b47",
    button_secondary_border_color="#a9cda6",
    checkbox_background_color_selected="#2e6b47",
    checkbox_border_color_selected="#2e6b47",
    color_accent_soft="#e4f0e3",
    link_text_color="#2e6b47",
    loader_color="#2e6b47",
    border_color_primary="#d9e6d8",
    border_color_accent="#4f9150",
)

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600'
    '&family=Noto+Sans+Devanagari:wght@400;600&display=swap" rel="stylesheet">'
    # Keep the light look even when the OS prefers dark mode.
    "<script>(function(){var u=new URL(location.href);"
    "if(u.searchParams.get('__theme')!=='light'){u.searchParams.set('__theme','light');"
    "location.replace(u.toString());}})();</script>"
)

CSS = """
.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  background: #f3f7f2 !important;
}
footer { display: none !important; }

/* Header */
.header { padding: 28px 0 18px; border-bottom: 1px solid #d9e6d8; }
.header h1 { font-size: 28px; font-weight: 600; color: #1e3527; margin: 0 0 6px; }
.header p { font-size: 16px; color: #5b7262; margin: 0; }

/* Headings line up with the boxes below them */
.heading .html-container { padding: 0 !important; }

/* Section headings */
.section-title {
  font-size: 18px; font-weight: 600; color: #1e3527;
  padding: 14px 0 8px; border-bottom: 1px solid #d9e6d8;
}

/* Option buttons in an even grid */
#scenario_dropdown .wrap { display: grid !important; grid-template-columns: repeat(2, 1fr); gap: 8px; }
#direction_dropdown .wrap { display: grid !important; grid-template-columns: repeat(2, 1fr); gap: 8px; }
#scenario_dropdown .wrap label, #direction_dropdown .wrap label {
  justify-content: center; margin: 0 !important; width: auto !important;
}

/* The converted sentence is the main result */
#transformed_box textarea {
  font-size: 22px !important; line-height: 1.6 !important;
  color: #1e3527 !important; background: #eef6ed !important;
  border: 1px solid #b9d6b7 !important;
}
#raw_box textarea, #cleaned_box textarea, #normalized_box textarea { font-size: 17px !important; }
#detected_box textarea, #target_box textarea { font-weight: 600 !important; color: #2e6b47 !important; }

#transform_btn, #clear_btn { min-height: 44px; font-size: 16px !important; font-weight: 600 !important; }
#clear_log_btn { max-width: 200px; }

#error_box textarea { background: #fdf6e7 !important; color: #6b4a0f !important; }

#status_md, #cloud_note, #key_status { color: #5b7262; font-size: 14px; }
#log_md { font-size: 16px; }
#log_md h3 { display: none; }
"""


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
        gr.HTML(
            '<div class="header">'
            '<h1>Register-Aware Hindi Voice Translation</h1>'
            '<p>Speak a sentence in Hindi and hear it in the tone that suits the listener.</p>'
            '</div>',
            padding=False,
            elem_classes=["heading"],
        )

        with gr.Row(equal_height=False):
            # ── Left: listener, speaker, audio ────────────────────────────────
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">Input</div>', padding=False, elem_classes=["heading"])
                with gr.Group():
                    scenario_dd = gr.Radio(
                        choices=list(SCENARIO_LABELS.values()),
                        value=DEFAULT_SCENARIO_LABEL,
                        label="Listener",
                        elem_id="scenario_dropdown",
                    )
                    direction_dd = gr.Radio(
                        choices=DIRECTION_CHOICES,
                        value=DIRECTION_CHOICES[0],
                        label="Speaker",
                        elem_id="direction_dropdown",
                    )
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                    elem_id="audio_input",
                )
                with gr.Row():
                    transform_btn = gr.Button(
                        "Convert",
                        variant="primary",
                        elem_id="transform_btn",
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_id="clear_btn",
                    )
                error_box = gr.Textbox(
                    label="Message",
                    lines=2,
                    interactive=False,
                    visible=False,
                    elem_id="error_box",
                )
                script_warn_box = gr.Textbox(
                    label="Script check",
                    lines=1,
                    interactive=False,
                    visible=False,
                    elem_id="script_warn_box",
                )

            # ── Right: output of each stage ───────────────────────────────────
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">Result</div>', padding=False, elem_classes=["heading"])
                with gr.Group():
                    with gr.Row():
                        raw_box = gr.Textbox(
                            label="Recognised text",
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
                            label="Detected tone",
                            lines=1,
                            interactive=False,
                            elem_id="detected_box",
                        )
                        target_box = gr.Textbox(
                            label="Target tone",
                            lines=1,
                            interactive=False,
                            elem_id="target_box",
                        )
                with gr.Group():
                    transformed_box = gr.Textbox(
                        label="Converted sentence",
                        lines=2,
                        interactive=False,
                        elem_id="transformed_box",
                    )
                    normalized_box = gr.Textbox(
                        label="Text sent to speech",
                        lines=1,
                        interactive=False,
                        elem_id="normalized_box",
                    )
                output_audio = gr.Audio(
                    label="Audio output",
                    interactive=False,
                    autoplay=True,
                    elem_id="output_audio",
                )
                status_md = gr.Markdown(
                    "_No run yet._",
                    elem_id="status_md",
                )

        # ── Settings ──────────────────────────────────────────────────────────
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                target_dd = gr.Dropdown(
                    choices=TARGET_CHOICES,
                    value=TARGET_CHOICES[0],
                    label="Tone override",
                    elem_id="target_dropdown",
                )
                asr_dd = gr.Dropdown(
                    choices=ASR_CHOICES,
                    value=config.ASR_BACKEND,
                    label="Speech recognition",
                    elem_id="asr_dropdown",
                )
                tts_dd = gr.Dropdown(
                    choices=TTS_CHOICES,
                    value=config.TTS_BACKEND,
                    label="Speech synthesis",
                    elem_id="tts_dropdown",
                )
            with gr.Row():
                voice_dd = gr.Dropdown(
                    choices=UI_TIMBRE_VOICES,
                    value=UI_TIMBRE_VOICES[0],
                    label="Voice (cloud only)",
                    elem_id="voice_dropdown",
                )
                lang_dd = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value=LANGUAGE_CHOICES[0],
                    label="Language",
                    elem_id="language_dropdown",
                )
                with gr.Column():
                    cloud_toggle = gr.Checkbox(
                        value=_CLOUD_DEFAULT,
                        label="Use Gnani AI cloud",
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

        # ── Conversation log ──────────────────────────────────────────────────
        with gr.Column():
            gr.HTML('<div class="section-title">Conversation log</div>', padding=False, elem_classes=["heading"])
            log_md = gr.Markdown(
                "_No turns yet._",
                elem_id="log_md",
            )
            clear_log_btn = gr.Button(
                "Clear log",
                variant="secondary",
                size="sm",
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
                gr.update(value=ui["script_warning"], visible=bool(ui["script_warning"])),
                gr.update(value=ui["error_message"], visible=bool(ui["error_message"])),
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
                gr.update(value="", visible=False),  # script_warn
                gr.update(value="", visible=False),  # error
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
        return "Cloud is on. Audio is sent to Gnani AI."
    return "Offline. Audio stays on this computer."


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

    # Hosted mode (Render and similar set $PORT): bind publicly on that port.
    port_env = os.getenv("PORT", "").strip()
    server_port = int(port_env) if port_env else None
    if server_port is not None:
        server_name = "0.0.0.0"

    # Optional login gate so strangers can't spend the Gnani credits.
    app_user = os.getenv("APP_USERNAME", "").strip()
    app_pass = os.getenv("APP_PASSWORD", "").strip()
    auth = (app_user, app_pass) if app_user and app_pass else None

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
        server_port=server_port,
        auth=auth,
        share=False,
        show_error=False,          # don't expose raw Python tracebacks in browser
        theme=THEME,
        css=CSS,
        head=HEAD,
    )


if __name__ == "__main__":
    main()
