# Day 2 Hindi ASR baseline

Run date: 29 September 2026

## Configuration

- Model: `vasista22/whisper-hindi-small`
- Framework: Transformers `5.17.0`, PyTorch `2.7.1`
- Device: Apple MPS (`mps:0`); both model parameters and input features were
  checked on MPS, with no CPU-fallback setting enabled
- Input batch: 10 user recordings selected evenly across the 19 uploaded files
- Raw input: mono MP3, 48,000 Hz
- ASR input: mono PCM WAV, 16,000 Hz
- Preprocessing: `soundfile` load, mono mix-down, `librosa.resample`, then
  `soundfile` write/format verification
- Model load time for the first batch process: 2.105 seconds from local cache

The selected files are listed in `sample_audio/day2_manifest.json`. Standardized
WAVs are in `sample_audio/day2_standardized`, and machine-readable results are
in `day2_asr_results.json`.

## Results

Timing begins before the raw MP3 is loaded and ends after text decoding. It
therefore covers file I/O, resampling, feature extraction, MPS inference, and
decoding, but excludes model loading.

| # | Uploaded file suffix | Duration | Audio-to-text | RTF | Transcription | Manual result |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | unnumbered | 2.328 s | 5.061 s | 2.17 | तो आज क्या कर रहे हैं | Pending reference |
| 2 | `(2)` | 3.576 s | 4.491 s | 1.26 | यार फिल्म लेकर चलता है | Pending reference |
| 3 | `(4)` | 3.168 s | 1.890 s | 0.60 | यह समान कितने का | Pending reference |
| 4 | `(6)` | 3.792 s | 5.631 s | 1.48 | मुझे अपना खाता खोलने के लिए क्या करना होगा | Pending reference |
| 5 | `(8)` | 3.576 s | 6.567 s | 1.84 | क्या आप मुझे किराया का रिसीट दे सकता है | Pending reference |
| 6 | `(10)` | 3.312 s | 3.847 s | 1.16 | क्या मैं अपना प्रश्न पूछ सकता हूं | Pending reference |
| 7 | `(12)` | 3.792 s | 4.031 s | 1.06 | डॉक्टर साहब मुझे कल रात से बुखार | Pending reference |
| 8 | `(14)` | 2.472 s | 4.695 s | 1.90 | अपने खाना खा लिया क्या | Pending reference |
| 9 | `(16)` | 3.096 s | 3.668 s | 1.18 | जी आपके लिए और कुछ चाहिए | Pending reference |
| 10 | `(18)` | 2.760 s | 2.193 s | 0.79 | आपको तैयार है | Pending reference |

`RTF` is elapsed processing time divided by audio duration; below 1.0 is faster
than real time.

## Timing observations

- Total speech duration: 31.872 seconds
- Mean clip duration: 3.187 seconds
- Mean audio-to-text time: 4.207 seconds
- Median audio-to-text time: 4.261 seconds
- Range: 1.890–6.567 seconds
- Aggregate real-time factor: 1.320
- Pearson correlation between clip duration and processing time: 0.237

These short clips show only a weak duration/timing relationship. Fixed feature
extraction and generation overhead, output-token count, and MPS warm-up effects
matter more than sub-second duration differences in this batch. Three clips ran
faster than or close to real time; the batch as a whole was slower than real
time.

## Register-critical audit

The output batch contains useful register cues:

- `आप`: clips 5 and 9; clip 10 contains the related form `आपको`
- `जी`: clip 9
- `साहब`: clip 7
- Casual marker `यार`: clip 2
- Potentially confused pronoun form `अपने`: clip 8

Several outputs deserve close comparison with the spoken references because
they are grammatically incomplete or contain agreement that may be an ASR
error:

- Clip 3: `यह समान कितने का` may have lost a final copula and may confuse
  `सामान` with `समान`.
- Clip 5: `आप ... दे सकता है` has formal-pronoun/verb disagreement.
- Clip 7 ends at `बुखार`, which may indicate a dropped final `है`.
- Clip 8's `अपने` may be a misrecognition of a pronoun such as `आपने` or
  `तुमने`; this is particularly important for the later register rules.
- Clip 10, `आपको तैयार है`, is semantically incomplete and may have dropped a
  noun or misheard a possessive form.

These are **candidate weaknesses**, not confirmed errors. Fluency or
grammaticality cannot establish what was actually spoken.

## Accuracy status and required completion input

No reference transcript was embedded in the uploaded MP3 files, and no
separate reference-text file was present. Therefore the required manual labels
(`correct`, `partially correct`, or `wrong`) and a defensible accuracy rate
cannot yet be assigned. The results intentionally store `null` for
`manual_assessment` and `word_error_rate` rather than treating model output as
ground truth.

Once the 10 spoken references are supplied in manifest order, update
`sample_audio/day2_manifest.json` and rerun the comparison/report step. The
actual MPS transcription and timing work is complete and does not need to be
repeated.

## Runtime quirk

The existing intermittent MPS/Python `resource_tracker` shutdown occurred
between clips. Per-clip checkpointing and `--start-index` resume support were
added to `asr_baseline.py`; all 10 results were recovered without CPU fallback
or repeated completed clips.

