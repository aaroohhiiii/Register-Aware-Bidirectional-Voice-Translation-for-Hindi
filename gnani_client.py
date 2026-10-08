import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.vachana.ai"
TTS_MODEL = "timbre-v2.5"


def _headers(content_type: str | None = None) -> dict[str, str]:
    api_key = os.environ.get("GNANI_API_KEY")
    if not api_key:
        raise RuntimeError("GNANI_API_KEY is not set. Add it to .env first.")

    headers = {"X-API-Key-ID": api_key}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _finalize_wav_header(audio: bytes) -> bytes:
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return audio

    data_idx = audio.find(b"data")
    if data_idx == -1 or data_idx + 8 > len(audio):
        return audio

    riff_size = int.from_bytes(audio[4:8], "little")
    data_size = int.from_bytes(audio[data_idx + 4 : data_idx + 8], "little")
    if riff_size != 0xFFFFFFFF and data_size != 0xFFFFFFFF:
        return audio

    fixed = bytearray(audio)
    fixed[4:8] = (len(audio) - 8).to_bytes(4, "little")
    fixed[data_idx + 4 : data_idx + 8] = (len(audio) - data_idx - 8).to_bytes(
        4, "little"
    )
    return bytes(fixed)


def prisma_transcribe(
    wav_path: str | os.PathLike[str],
    lang: str = "hi-IN",
    timeout: int = 15,
    output_format: str = "verbatim",
) -> str:
    with Path(wav_path).open("rb") as audio:
        response = requests.post(
            f"{BASE}/stt/v3",
            headers=_headers(),
            files={"audio_file": audio},
            data={"language_code": lang, "format": output_format},
            timeout=timeout,
        )

    response.raise_for_status()
    return response.json()["transcript"]


def timbre_speak(
    text: str,
    out_path: str | os.PathLike[str],
    voice: str = "Nalini",
    lang: str = "hi-IN",
    speed: float = 1.0,
    timeout: int = 20,
    sample_rate: int = 48000,
) -> Path:
    body: dict[str, Any] = {
        "text": text,
        "voice": voice,
        "model": TTS_MODEL,
        "language": lang,
        "speed": speed,
        "audio_config": {
            "sample_rate": sample_rate,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }

    response = requests.post(
        f"{BASE}/api/v1/tts/inference",
        headers=_headers("application/json"),
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_finalize_wav_header(response.content))
    return output
