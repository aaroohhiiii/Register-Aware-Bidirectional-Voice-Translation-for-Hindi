"""Interactive Gradio Web UI for Register-Aware Bidirectional Voice Translation for Hindi.

Allows selecting communication scenarios, recording audio or uploading audio files,
inspecting the transcription and diagnostic detected register, and synthesizing speech in the target register.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.cleanup import remove_fillers
from project.transform import (
    TIERS,
    infer_register_with_details,
    resolve_scenario_tier,
    transform_text,
)


def load_scenarios() -> dict[str, dict]:
    """Load scenario descriptions and keys from scenario_lookup.json."""
    lookup_file = ROOT_DIR / "scenario_lookup.json"
    if lookup_file.exists():
        data = json.loads(lookup_file.read_text(encoding="utf-8"))
        return {item["scenario_key"]: item for item in data["scenarios"]}
    return {}


def process_text_flow(
    input_text: str,
    scenario_key: str,
    gender: str | None = None,
    address_term: str | None = None,
) -> tuple[str, str, str, str]:
    """Process text transformation flow."""
    if not input_text.strip():
        return "", "", "", ""

    # 1. Cleanup fillers
    cleaned = remove_fillers(input_text)

    # 2. Detect register (diagnostic)
    detection = infer_register_with_details(cleaned)
    detected_summary = f"{detection['tier']} (Confidence: {detection['confidence'] * 100:.1f}%)"

    # 3. Resolve target tier
    target_tier = resolve_scenario_tier(scenario_key) if scenario_key else "Formal"

    # 4. Transform
    transformed = transform_text(
        cleaned,
        target_tier=target_tier,
        gender=gender if gender else None,
        scenario=scenario_key,
        address_term=address_term if address_term else None,
    )

    return cleaned, detected_summary, target_tier, transformed


if __name__ == "__main__":
    try:
        import gradio as gr

        scenarios = load_scenarios()
        scenario_choices = list(scenarios.keys())

        with gr.Blocks(title="Register-Aware Voice Translation (Hindi)") as demo:
            gr.Markdown("# 🎙️ Register-Aware Hindi Voice Translation")
            gr.Markdown(
                "Transform spoken or written Hindi between formal, respectful, and informal social registers."
            )

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Hindi Input Text / Transcription",
                        placeholder="उदा: दादी जी आप कैसी हैं? / तू कब आएगा?",
                        lines=3,
                    )
                    scenario_selector = gr.Dropdown(
                        choices=scenario_choices,
                        value=scenario_choices[0] if scenario_choices else "doctor_consultation",
                        label="Target Communication Scenario",
                    )
                    gender_selector = gr.Radio(
                        choices=["None", "masculine", "feminine"],
                        value="None",
                        label="Addressee Gender",
                    )
                    address_term_input = gr.Textbox(
                        label="Custom Kinship / Address Term (e.g. दादी जी, सर)",
                        placeholder="Optional override",
                    )
                    translate_btn = gr.Button("Transform Register", variant="primary")

                with gr.Column():
                    cleaned_out = gr.Textbox(label="Cleaned Transcript (Filler Removed)", interactive=False)
                    detected_out = gr.Textbox(label="Diagnostic Source Register", interactive=False)
                    target_tier_out = gr.Textbox(label="Resolved Target Register Tier", interactive=False)
                    transformed_out = gr.Textbox(label="Transformed Output", interactive=False, lines=3)

            translate_btn.click(
                fn=process_text_flow,
                inputs=[input_text, scenario_selector, gender_selector, address_term_input],
                outputs=[cleaned_out, detected_out, target_tier_out, transformed_out],
            )

        demo.launch()
    except ImportError:
        print("Gradio is not installed. To run the UI, install gradio (`pip install gradio`).")
