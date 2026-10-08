"""End-to-end Pipeline: Audio -> ASR -> Cleanup -> Register Transform -> TTS -> Audio.

Orchestrates the components into a cohesive pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project.asr import transcribe
from project.cleanup import remove_fillers
from project.transform import infer_register_with_details, resolve_scenario_tier, transform_text
from project.tts import synthesize_speech


@dataclass
class PipelineResult:
    raw_transcript: str
    cleaned_transcript: str
    detected_register: str
    detected_details: dict[str, Any]
    target_tier: str
    transformed_text: str
    output_audio_path: str | None


def run_pipeline(
    audio_input: str | Path,
    target_scenario_or_tier: str,
    output_audio_path: str | Path | None = None,
    gender: str | None = None,
    address_term: str | None = None,
    is_scenario_key: bool = True,
) -> PipelineResult:
    """Run full register-aware voice translation pipeline on input audio.

    Args:
        audio_input: Path to raw input audio file.
        target_scenario_or_tier: Scenario key (e.g. "doctor_consultation") or explicit tier name.
        output_audio_path: Where to save output audio.
        gender: Optional gender hint ("masculine", "feminine").
        address_term: Real address/kinship term if applicable.
        is_scenario_key: If True, resolve scenario key via scenario_dial; if False, use as tier.

    Returns:
        PipelineResult with all intermediate and final outputs.
    """
    # 1. ASR
    raw_text = transcribe(audio_input)

    # 2. Filler cleanup
    cleaned_text = remove_fillers(raw_text)

    # 3. Source Register Detection (diagnostic)
    detection_info = infer_register_with_details(cleaned_text)

    # 4. Resolve Target Register Tier
    if is_scenario_key:
        target_tier = resolve_scenario_tier(target_scenario_or_tier)
        scenario_key = target_scenario_or_tier
    else:
        target_tier = target_scenario_or_tier
        scenario_key = None

    # 5. Register Transformation
    transformed = transform_text(
        cleaned_text,
        target_tier=target_tier,
        gender=gender,
        scenario=scenario_key,
        address_term=address_term,
    )

    # 6. TTS Voice Synthesis
    if output_audio_path:
        synthesize_speech(transformed, output_path=output_audio_path)
        out_path_str = str(output_audio_path)
    else:
        out_path_str = None

    return PipelineResult(
        raw_transcript=raw_text,
        cleaned_transcript=cleaned_text,
        detected_register=detection_info["tier"],
        detected_details=detection_info,
        target_tier=target_tier,
        transformed_text=transformed,
        output_audio_path=out_path_str,
    )


__all__ = ["PipelineResult", "run_pipeline"]
