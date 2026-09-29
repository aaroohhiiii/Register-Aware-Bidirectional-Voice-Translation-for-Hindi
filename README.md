# Register-Aware Bidirectional Voice Translation for Hindi

## 1. Project overview

This project is a proof-of-concept Hindi voice translation system designed to
preserve the social register appropriate to the person being addressed. Hindi
changes pronouns, verb agreement, imperatives, honorifics, and address terms
depending on relationships such as friend, teacher, elder relative, customer,
or auto driver. The system converts Hindi speech to text, maps a user-selected
scenario to one of six register tiers, and deterministically rewrites the text
for that relationship. The current repository contains a working MPS-based ASR
baseline and the tested register-selection, tokenization, transformation, and
detection layers. **Text cleanup, TTS, the bidirectional conversation
orchestrator, and a one-command end-to-end entry point have not been
implemented yet**, so audio-to-audio execution is not currently available from
this checkout.

## 2. Architecture diagram

The intended end-to-end architecture is:

```text
Hindi audio in
      |
      v
ASR: vasista22/whisper-hindi-small             [implemented: asr_baseline.py]
      |
      v
Raw Hindi transcript
      |
      v
Text cleanup                                      [not yet implemented]
      |
      +---------------------> detect_register()   [diagnostic only]
      |                              |
      |                              +--> evidence/confidence display or logs
      |                                   NEVER overrides the selected scenario
      v
User selects who they are speaking to in the UI  [required primary control]
      |
      v
scenario_to_dial(selected_scenario_key)
      |
      v
Authoritative target tier
      |
      v
transform(sentence, target_tier, gender, ...)
      |
      v
Register-adjusted Hindi text
      |
      v
Hindi TTS                                        [not yet implemented]
      |
      v
Hindi audio out
```

The planned bidirectional application applies this same sequence independently
to each conversation direction. No bidirectional state/orchestration module is
present yet.

## 3. Prerequisites

### Platform and Python

- Python **3.12.13** is the verified version in `SETUP_NOTES.md`.
- Python 3.9 or newer is required by the project plan; Python 3.12 is the
  reproducible choice for this repository.
- The supplied ASR runner requires an Apple Silicon Mac with a working PyTorch
  MPS backend. It intentionally raises an error instead of falling back to CPU.
- The text-only register modules and tests do not require MPS.

Check Python and Apple MPS:

```bash
python3 --version
python3 -c "import torch; print('built=', torch.backends.mps.is_built()); print('available=', torch.backends.mps.is_available())"
```

Both MPS values must be `True` before running `asr_baseline.py`.

### Python dependencies actually used

The pinned direct dependencies are in `requirements.txt`:

| Package | Version | Used for |
|---|---:|---|
| `torch` | 2.7.1 | MPS inference |
| `transformers` | 5.17.0 | Whisper processor and ASR model |
| `soundfile` | 0.14.0 | Reading and writing audio |
| `librosa` | 1.0.0 | Resampling to 16 kHz |
| `indic-nlp-library` | 0.92 | Hindi token boundaries |
| `stanza` | 1.14.0 | Hindi POS tagging |

`numpy` is also imported directly by `asr_baseline.py`; it is installed as a
dependency of the packages above (the verified environment currently has
NumPy 2.5.3).

Install the recorded dependency set with:

```bash
python -m pip install -r requirements.txt
```

### TTS dependency status

**No TTS module, model identifier, or TTS library is currently present in the
repository or `requirements.txt`.** There is therefore no truthful TTS install
or model-load command to provide yet. A TTS library must be selected and added
before the final audio-output stage can run.

## 4. Setup instructions

### 4.1 Obtain the files

No remote Git URL is recorded in this repository, so use the team-provided
clone URL if one exists. To use the supplied handoff archive:

```bash
unzip harshita_handoff_bundle.zip -d hindi-register-handoff
cd hindi-register-handoff
```

The handoff archive intentionally excludes model caches and audio files. When
working in the original project checkout, run the remaining commands from the
directory containing this `README.md`.

### 4.2 Create and activate an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 4.3 Download the Hindi Stanza resources

`hindi_tokenizer.py` loads resources only from `.cache/stanza` and does not
download them at runtime. Download the exact tokenizer/POS packages it expects:

```bash
python -c "import stanza; stanza.download('hi', model_dir='.cache/stanza', processors={'tokenize': 'hdtb', 'pos': 'hdtb_charlm'})"
```

Confirm tokenization and POS tagging:

```bash
python -c "from hindi_tokenizer import tokenize_and_tag; print(tokenize_and_tag('तुम कहाँ रहते हो?'))"
```

### 4.4 Confirm the ASR model loads on MPS

The ASR script automatically downloads
`vasista22/whisper-hindi-small` through Transformers on its first run. The
original checkout includes WAV inputs referenced by `sample_audio/manifest.json`.
Run a one-clip smoke test without overwriting the recorded baseline:

```bash
HF_HOME=.cache/huggingface python -u asr_baseline.py \
  --limit 1 \
  --output-dir /tmp/hindi-asr-smoke-audio \
  --results /tmp/hindi-asr-smoke-results.json
```

A successful run prints values including:

```text
model=vasista22/whisper-hindi-small
mps_built=True
mps_available=True
model_device=mps:0
```

It also preprocesses the input to mono 16 kHz PCM WAV and writes the temporary
result JSON. If using the handoff archive alone, supply the omitted audio files
and update a compatible manifest before running this smoke test.

### 4.5 TTS model check

There is no TTS code or selected model to check. This setup step remains
blocked until a TTS module and dependency are added.

## 5. How to run

### Full audio-to-audio pipeline

**There is currently no full-pipeline command.** The repository has no TTS
module, text-cleanup module, microphone/UI entry point, or pipeline
orchestrator. `asr_baseline.py` runs only the manifest-driven audio-to-text
baseline; it does not call `scenario_to_dial()`, `transform()`, or TTS.

The existing ASR batch can be run with:

```bash
HF_HOME=.cache/huggingface python -u asr_baseline.py
```

This uses `sample_audio/manifest.json`, writes standardized audio under
`sample_audio/processed`, and writes `asr_baseline_results.json`. It requires
MPS and the audio files from the original checkout.

### Run the text transformation layer

There is no dedicated transformation CLI yet. Invoke the tested Python API
directly. This example selects the `auto_driver` scenario, which resolves to
Respectful Casual, and rewrites a masculine addressee sentence:

```bash
python -c 'from scenario_dial import scenario_to_dial; from register_transformer import transform; scenario="auto_driver"; tier=scenario_to_dial(scenario); text="तू क्या कर रहा है?"; print("tier:", tier); print("output:", transform(text, tier, "M"))'
```

Expected output:

```text
tier: Respectful Casual
output: तुम क्या कर रहे हो?
```

For the two tag-based tiers, pass scenario context and a validated address term.
For example:

```bash
python -c 'from scenario_dial import scenario_to_dial; from register_transformer import transform; scenario="elder_relative"; tier=scenario_to_dial(scenario); print(transform("तुम कैसी हो?", tier, "F", scenario=scenario, address_term="नानी जी"))'
```

Expected output:

```text
नानी जी, आप कैसी हैं?
```

Run diagnostic register detection separately:

```bash
python -c 'from register_detector import detect_register, detect_register_details; text="आप कहाँ रहते हैं?"; print(detect_register(text)); print(detect_register_details(text))'
```

Do not use the detected result to override an explicit user-selected scenario.

## 6. Running the test suite

From the project root with the virtual environment active:

```bash
python -m unittest discover -s tests -v
```

The current expected summary is:

```text
Ran 53 tests

OK
```

The suite covers scenario lookup, tokenization/POS flags, pronoun and verb rule
tables, all 48 tagged sentences, 144 cross-tier transformations, register
detection, postpositional cases, and Formal-Warm/Reverse Formal branches.

Additional evaluation output can be reproduced with:

```bash
python register_detection_baseline.py
python cross_tier_review.py
```

The current recorded results are 144/144 passing cross-tier transformations,
48/48 passing detection-to-transformation smoke cases, and 36/48 (75%) correct
standalone register detections.

## 7. Known limitations

- `detect_register()` has a documented 75% text-only accuracy ceiling on the
  locked set. Four tiers—Formal, Full Formal, Formal-Warm, and Reverse
  Formal—share the `आप` pronoun paradigm and respectful-plural verb morphology.
  Some lower-tier sentences also use `तुम` in both Full Informal and Respectful
  Casual. Relationship and participant-role context is not always present in a
  sentence, so the user-selected scenario is the authoritative target and
  detection is diagnostic only.
- Hindi ASR has register-critical weak spots: short `तू` may be recognized as
  `तो`; respectful or gendered verb constructions can flatten (for example,
  `सकते हैं` to `सकता है`); and short honorific/address terms may be dropped.
- The ten-clip Day 2 ASR batch does not yet have supplied reference transcripts,
  so its manual correctness labels and WER remain pending. See
  `ASR_BASELINE.md` for the recorded outputs and timing data.
- The ASR runner is hard-coded to require Apple MPS and refuses CPU fallback.
- No text-cleanup/normalization stage has been implemented yet, despite the ASR
  weaknesses above.
- No TTS module or dependency has been selected or implemented.
- No microphone/UI runner, full bidirectional orchestrator, or single-command
  audio-to-audio entry point exists yet.
- The transformation layer is intentionally rule-based and degrades gracefully
  by preserving unsupported words or constructions.

## 8. Project structure

```text
.
├── README.md                         # This run/setup guide
├── requirements.txt                 # Pinned Python dependencies
├── asr_baseline.py                  # Manifest-driven Whisper ASR/MPS runner
├── hindi_tokenizer.py               # Indic NLP tokenization + Stanza POS tags
├── scenario_dial.py                 # scenario_to_dial() public API
├── scenario_lookup.json             # 20 scenarios, P/D/C values, thresholds
├── register_transformer.py          # transform() public API
├── register_detector.py             # detect_register() diagnostic API
├── register_detection_baseline.py   # 48-case detection/smoke evaluator
├── cross_tier_review.py             # Generates the 144 transformation cases
├── pronoun_rules.json               # Pronoun/postpositional forms by tier
├── verb_rules.json                  # Tier × gender × tense verb patterns
├── address_term_rules.json          # Honorific/address and scenario overrides
├── register_test_sentences.json     # 48 tagged Hindi evaluation sentences
├── asr_baseline_results.json        # Three-clip ASR baseline results
├── day2_asr_results.json            # Ten-clip Day 2 ASR results
├── transformation_failure_log.json  # Resolved transformation failure history
├── day6_bug_list.json               # Consolidated bug/limitation status
├── sample_audio/
│   ├── manifest.json                # Three-clip ASR input/reference manifest
│   ├── day2_manifest.json           # Ten-clip Day 2 manifest
│   ├── raw/                         # Source WAV files (original checkout)
│   ├── processed/                   # Standardized/generated audio
│   └── day2_standardized/           # Standardized Day 2 audio
├── tests/                            # 53 unittest cases across nine modules
├── SETUP_NOTES.md                   # Verified versions, MPS and model notes
├── ASR_BASELINE.md                  # ASR timing, outputs, and weak spots
├── SCENARIO_REFERENCE.md            # Full P/D/C framework and scenario table
├── PRONOUN_RULES.md                 # Pronoun-case audit documentation
├── VERB_ADDRESS_RULES.md            # Verb and address policy documentation
├── REGISTER_DETECTION_BASELINE.md   # Confusion matrix and 75% limitation
├── REGISTER_SELECTION_ARCHITECTURE.md # Required scenario-first UI contract
└── REPORT_READY_METHOD_EVALUATION.md  # Report-ready methods/evaluation prose
```

Generated/cache directories such as `.venv`, `.cache`, and `__pycache__` are
not source files and are excluded from version control and the handoff archive.
