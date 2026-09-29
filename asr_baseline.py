"""Run the Hindi Whisper ASR preprocessing and MPS inference baseline.

Each input is loaded with soundfile, mixed down to mono, resampled to the
model's 16 kHz rate with librosa, saved as a standardized WAV, and then
transcribed on Apple's MPS device. The script deliberately fails instead of
silently falling back to CPU when MPS is unavailable.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_ID = "vasista22/whisper-hindi-small"
TARGET_SAMPLE_RATE = 16_000


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Read sample metadata and resolve paths relative to the manifest."""
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in records:
        record["path"] = str(manifest_path.parent / record["file"])
    return records


def preprocess_audio(audio_path: Path, output_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a WAV, convert it to mono, resample to 16 kHz, and save it."""
    audio, native_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    native_channels = audio.shape[1]
    mono_audio = audio.mean(axis=1, dtype=np.float32)

    if native_rate != TARGET_SAMPLE_RATE:
        mono_audio = librosa.resample(
            mono_audio,
            orig_sr=native_rate,
            target_sr=TARGET_SAMPLE_RATE,
        ).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.wav"
    sf.write(output_path, mono_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

    written_info = sf.info(output_path)
    if written_info.samplerate != TARGET_SAMPLE_RATE or written_info.channels != 1:
        raise RuntimeError(f"Standardized audio verification failed: {written_info}")

    details = {
        "input_path": str(audio_path),
        "standardized_path": str(output_path),
        "native_sample_rate_hz": native_rate,
        "native_channels": native_channels,
        "output_sample_rate_hz": written_info.samplerate,
        "output_channels": written_info.channels,
        "duration_seconds": round(len(mono_audio) / TARGET_SAMPLE_RATE, 3),
    }
    return mono_audio, details


def normalize_words(text: str) -> list[str]:
    """Apply light normalization suitable for a transparent baseline check."""
    normalized = "".join(
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in unicodedata.normalize("NFC", text.casefold())
    )
    return normalized.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word-level Levenshtein distance divided by reference length."""
    expected = normalize_words(reference)
    actual = normalize_words(hypothesis)
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / max(1, len(expected))


def load_model() -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Download/load the selected model and move every parameter to MPS."""
    if not torch.backends.mps.is_built():
        raise RuntimeError("Installed PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing silent CPU fallback")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # This 2023 checkpoint ships a Transformers 4.27-era generation config.
    # Transformers 5 expects language/task maps instead of forced_decoder_ids,
    # so derive the same Hindi transcription prompt from the tokenizer.
    prompt_tokens = dict(
        processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    )
    expected_positions = {1, 2, 3}
    if set(prompt_tokens) != expected_positions:
        raise RuntimeError(f"Unexpected Whisper decoder prompt: {prompt_tokens}")
    model.generation_config.is_multilingual = True
    model.generation_config.lang_to_id = {"<|hi|>": prompt_tokens[1]}
    model.generation_config.task_to_id = {"transcribe": prompt_tokens[2]}
    model.generation_config.no_timestamps_token_id = prompt_tokens[3]

    model.to(torch.device("mps"))
    model.eval()

    parameter_device = next(model.parameters()).device
    if parameter_device.type != "mps":
        raise RuntimeError(f"Model loaded on {parameter_device}, not MPS")
    return processor, model


def transcribe(
    audio: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
) -> tuple[str, str]:
    """Transcribe an already standardized waveform and return text/device."""
    model_inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = model_inputs.input_features.to("mps")
    if input_features.device.type != "mps":
        raise RuntimeError("Whisper input features are not on MPS")

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            language="hi",
            task="transcribe",
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text, str(input_features.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("sample_audio/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_audio/processed"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("asr_baseline_results.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N manifest entries (useful for smoke tests).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Resume at this zero-based manifest index, preserving earlier results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if args.start_index < 0 or args.start_index > len(records):
        raise ValueError("--start-index is outside the manifest")
    if args.limit is not None:
        records = records[: args.limit]

    print(f"model={MODEL_ID}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    load_started = time.perf_counter()
    processor, model = load_model()
    model_load_seconds = time.perf_counter() - load_started
    print(f"model_device={next(model.parameters()).device}")
    print(f"model_load_seconds={model_load_seconds:.3f}")

    results: list[dict[str, Any]] = []
    if args.start_index:
        if not args.results.exists():
            raise FileNotFoundError(
                f"Cannot resume because results do not exist: {args.results}"
            )
        previous_report = json.loads(args.results.read_text(encoding="utf-8"))
        results = previous_report["clips"][: args.start_index]
        if len(results) != args.start_index:
            raise ValueError("Existing results do not reach --start-index")

    for record in records[args.start_index :]:
        audio_path = Path(record["path"])
        started = time.perf_counter()
        audio, details = preprocess_audio(audio_path, args.output_dir)
        hypothesis, inference_device = transcribe(audio, processor, model)
        elapsed = time.perf_counter() - started

        reference = record.get("reference")
        result = {
            **details,
            "source_dataset": record["source_dataset"],
            "source_file": record["source_file"],
            "reference": reference,
            "transcription": hypothesis,
            "word_error_rate": (
                round(word_error_rate(reference, hypothesis), 4)
                if reference
                else None
            ),
            "manual_assessment": record.get("manual_assessment"),
            "audio_to_text_seconds": round(elapsed, 3),
            "inference_device": inference_device,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Persist after every clip so an intermittent native MPS shutdown does
        # not discard completed measurements.
        partial_report = {
            "model_id": MODEL_ID,
            "model_device": str(next(model.parameters()).device),
            "target_sample_rate_hz": TARGET_SAMPLE_RATE,
            "target_channels": 1,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "model_load_seconds": round(model_load_seconds, 3),
            "clips": results,
        }
        args.results.write_text(
            json.dumps(partial_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    report = {
        "model_id": MODEL_ID,
        "model_device": str(next(model.parameters()).device),
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "target_channels": 1,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "model_load_seconds": round(model_load_seconds, 3),
        "clips": results,
    }
    args.results.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"results={args.results}")


if __name__ == "__main__":
    main()
