# Hindi ASR setup notes

Verified on 25 September 2026 on an Apple M4 Mac running macOS 15.6.

## Environment

- Python: `3.12.13`
- Virtual environment: `.venv`
- PyTorch: `2.7.1`
- Transformers: `5.17.0`
- SoundFile: `0.14.0`
- librosa: `1.0.0`
- `pip check`: no broken requirements

Activate the environment from the project directory:

```bash
source .venv/bin/activate
```

The four direct dependencies and exact versions are also recorded in
`requirements.txt`.

## Confirmed ASR baseline

- Model: `vasista22/whisper-hindi-small`
- Model revision downloaded: `fb4a24afc20c42906deed31c3689e4be31da41fe`
- Model cache: `.cache/huggingface` (excluded from Git)
- Input expected by the model: mono audio at 16,000 Hz
- Standardized file format used by this project: mono WAV, 16,000 Hz,
  signed 16-bit PCM
- Device in the successful baseline: `mps:0`
- MPS checks: `torch.backends.mps.is_built() == True` and
  `torch.backends.mps.is_available() == True`
- Cached model load time in the recorded run: 1.420 seconds

The model weights downloaded successfully and the model and input feature
tensors were both explicitly checked to be on `mps`. The runner raises an
error if MPS is unavailable, so it cannot silently fall back to CPU.

## Audio preprocessing

The real sample clips in `sample_audio/raw` came from the public
`speechriv/audio_hindi` dataset. The three accepted inputs are mono 48 kHz
WAVs. `asr_baseline.py` performs the complete preprocessing chain:

1. Read the WAV with `soundfile` as a float32 NumPy array.
2. Average channels if an input is not mono.
3. Resample to 16,000 Hz with `librosa.resample()` when needed.
4. Write and verify a mono 16 kHz PCM WAV in `sample_audio/processed`.
5. Pass the in-memory 16 kHz waveform to `WhisperProcessor` and run
   `WhisperForConditionalGeneration` on MPS.

Source filenames and reference transcripts are recorded in
`sample_audio/manifest.json`.

## Transcription baseline

Timing starts immediately before SoundFile reads each raw WAV and ends after
the decoded text is available. It therefore includes file loading, mono
conversion, resampling, feature extraction, MPS inference, and decoding, but
not the one-time model load. WER uses punctuation-insensitive Unicode NFC text.

| Clip | Audio duration | Audio-to-text | WER | MPS transcription |
| --- | ---: | ---: | ---: | --- |
| `hindi_04.wav` | 7.732 s | 17.373 s | 0.0625 | इस प्रकार राजा संसार चंद ने मुस्लिम आक्रांताओं द्वारा तोड़े गए मंदिर का जीर्णोद्धार करवाया होगा |
| `hindi_02.wav` | 9.764 s | 16.262 s | 0.0000 | अभिनंदन को डॉक्टरों ने चार हफ्तों की मेडिकल लीव दी थी लेकिन आराम करने की बजाय वह श्रीनगर स्थित अपने स्क्वाड्रन में चले गए |
| `hindi_03.wav` | 8.700 s | 10.634 s | 0.1000 | वे अब तक जनमानस की विश्वासों पर खरी उतरी है और जर्मनी की नियंत्रण पहचान को और पुख्ता किया है |

All three outputs are readable Hindi and preserve the meaning of the reference
speech. Clip 2 is exact after punctuation normalization. Clip 4 differs only
on the spelling variant `चन्द`/`चंद`; clip 3 has two small agreement/case
errors (`के`/`की` and `हैं`/`है`). The machine-readable record is
`asr_baseline_results.json`.

Run the baseline after activating the environment:

```bash
HF_HOME=.cache/huggingface .venv/bin/python -u asr_baseline.py
```

After the model has been downloaded, an offline reproducibility run is:

```bash
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 \
  .venv/bin/python -u asr_baseline.py
```

## MPS and compatibility notes

- The project sandbox hid the Metal device and reported
  `mps.is_available() == False`; the same check in a normal, unsandboxed
  terminal returned `True`. Run MPS inference from a normal terminal session.
- No PyTorch operation fell back to CPU during the successful recorded run.
  The runner does not enable `PYTORCH_ENABLE_MPS_FALLBACK`.
- This model was saved with a Transformers 4.27-era generation config. With
  Transformers 5.17, the model card's old `forced_decoder_ids` call is rejected.
  `asr_baseline.py` derives the Hindi transcription prompt IDs from the bundled
  tokenizer and populates the current language/task generation maps.
- Transformers emits a deprecation warning about the checkpoint's legacy
  `suppress_tokens`/`begin_suppress_tokens` settings. It did not affect output.
- Repeated MPS runs occasionally ended with Python's `resource_tracker`
  semaphore-cleanup warning before producing results. A fresh one-clip test and
  the final three-clip run both completed successfully; restart the Python
  process if this intermittent native-runtime cleanup issue appears.

