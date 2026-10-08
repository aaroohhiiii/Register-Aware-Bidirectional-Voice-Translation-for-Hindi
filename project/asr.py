"""ASR wrapper module for Hindi speech recognition using Whisper.

Wraps Aarohi's Whisper ASR pipeline with automatic device selection
(CUDA / MPS / CPU) for cross-platform compatibility.
"""

from __future__ import annotations

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


def get_default_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_audio(
    audio_input: str | Path | np.ndarray,
    orig_sr: int | None = None,
    output_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, convert to mono float32, and resample audio to 16 kHz.

    Args:
        audio_input: Path to audio file or raw NumPy waveform array.
        orig_sr: Sampling rate if audio_input is a NumPy array.
        output_path: Optional path to save the standardized 16 kHz WAV.

    Returns:
        tuple: (mono_16k_waveform, details_dict)
    """
    if isinstance(audio_input, (str, Path)):
        audio_path = Path(audio_input)
        audio, native_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        native_channels = audio.shape[1]
        mono_audio = audio.mean(axis=1, dtype=np.float32)
    elif isinstance(audio_input, np.ndarray):
        audio_path = None
        native_channels = 1 if audio_input.ndim == 1 else audio_input.shape[1]
        mono_audio = audio_input.mean(axis=-1, dtype=np.float32) if audio_input.ndim > 1 else audio_input.astype(np.float32)
        native_rate = orig_sr or TARGET_SAMPLE_RATE
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    if native_rate != TARGET_SAMPLE_RATE:
        mono_audio = librosa.resample(
            mono_audio,
            orig_sr=native_rate,
            target_sr=TARGET_SAMPLE_RATE,
        ).astype(np.float32, copy=False)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, mono_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

    details = {
        "input_source": str(audio_path) if audio_path else "memory_array",
        "native_sample_rate_hz": native_rate,
        "native_channels": native_channels,
        "output_sample_rate_hz": TARGET_SAMPLE_RATE,
        "duration_seconds": round(len(mono_audio) / TARGET_SAMPLE_RATE, 3),
    }
    return mono_audio, details


class HindiASR:
    """Stateful ASR wrapper holding processor and model."""

    def __init__(self, model_id: str = MODEL_ID, device: torch.device | str | None = None):
        self.model_id = model_id
        self.device = torch.device(device) if device else get_default_device()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)

        prompt_tokens = dict(
            self.processor.get_decoder_prompt_ids(language="hi", task="transcribe")
        )
        self.model.generation_config.is_multilingual = True
        self.model.generation_config.lang_to_id = {"<|hi|>": prompt_tokens[1]}
        self.model.generation_config.task_to_id = {"transcribe": prompt_tokens[2]}
        self.model.generation_config.no_timestamps_token_id = prompt_tokens[3]

        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_input: str | Path | np.ndarray, orig_sr: int | None = None) -> str:
        """Transcribe an audio file or waveform to Hindi text.

        Signature:
            transcribe(audio_input) -> str
        """
        waveform, _ = preprocess_audio(audio_input, orig_sr=orig_sr)
        model_inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = model_inputs.input_features.to(self.device)

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                language="hi",
                task="transcribe",
            )
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return unicodedata.normalize("NFC", text)


_cached_asr: HindiASR | None = None


def get_asr_instance(device: str | None = None) -> HindiASR:
    """Singleton getter for ASR instance."""
    global _cached_asr
    if _cached_asr is None:
        _cached_asr = HindiASR(device=device)
    return _cached_asr


def transcribe(audio_path_or_array: str | Path | np.ndarray, orig_sr: int | None = None) -> str:
    """Convenience wrapper for single transcription calls."""
    asr = get_asr_instance()
    return asr.transcribe(audio_path_or_array, orig_sr=orig_sr)


__all__ = ["HindiASR", "get_asr_instance", "preprocess_audio", "transcribe"]
