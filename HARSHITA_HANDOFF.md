# Aarohi → Harshita handoff bundle

Bundle date: 30 September 2026  
Scope: validated scenario selection, Hindi register rules, transformation,
detection, tagged evaluation set, and ASR baseline evidence.

## Start here

The Day 10 interface must make scenario selection explicit. Populate the UI
from `scenario_lookup.json`, retain the selected stable `key`, resolve it with
`scenario_to_dial()`, and use that result as the transformation target.
`detect_register()` may be displayed as diagnostic source-register evidence,
but must never silently override the selected scenario. The complete UI
contract is in `REGISTER_SELECTION_ARCHITECTURE.md`.

## Canonical pipeline calls

```python
from scenario_dial import scenario_to_dial
from register_detector import detect_register
from register_transformer import transform

source_tier_hint = detect_register(sentence)  # diagnostic only
target_tier = scenario_to_dial(selected_scenario_key)  # authoritative

output = transform(
    sentence,
    target_tier,
    gender,
    scenario=(selected_scenario_key if target_tier in {"Formal-Warm", "Reverse Formal"} else None),
    address_term=selected_address_term,
)
```

For `elder_relative`, `selected_address_term` must be the real relation, such
as `दादी जी`; never infer it from gender. For `customer_facing`, use `सर`,
`मैडम`, or `जी` when known.

## Component map

| Area | Canonical artifacts | Purpose |
|---|---|---|
| Scenario framework | `scenario_dial.py`, `scenario_lookup.json` | P/D/C averaging, thresholds, and tag-based tiers |
| Tokenizer/POS | `hindi_tokenizer.py` | Indic NLP tokenization plus Stanza POS and rule-facing flags |
| Pronouns | `pronoun_rules.json`, `PRONOUN_RULES.md` | Six-tier paradigms, including instrumental and audited postpositional cases |
| Verbs/addresses | `verb_rules.json`, `address_term_rules.json`, `VERB_ADDRESS_RULES.md` | Tier × gender × tense agreement and scenario overrides |
| Transformation | `register_transformer.py` | Deterministic pronoun, verb, particle, and address rewriting |
| Detection | `register_detector.py`, `register_detection_baseline.py` | Secondary register inference with evidence and confidence |
| Evaluation corpus | `register_test_sentences.json`, `cross_tier_review.py` | 48 tagged sentences and 144 cross-tier cases |
| Evaluation reports | `REGISTER_DETECTION_BASELINE.md`, `STEP_5_TRANSFORM_REVIEW.md`, `DAY6_FINAL_REPORT.md` | Accuracy, confusion matrix, smoke test, and resolved bugs |
| ASR | `ASR_BASELINE.md`, `SETUP_NOTES.md`, result JSON files | MPS setup, timings, outputs, and known weaknesses |
| Report text | `REPORT_READY_METHOD_EVALUATION.md` | Ready-to-adapt Methodology, Evaluation, and Limitations content |

## Validated results

- Scenario lookup: all 20 scenarios pass.
- Register transformations: 144/144 (100%).
- Detection → transformation smoke test: 48/48 (100%).
- Standalone text-only detection: 36/48 (75%).
- Current automated suite: 53 tests pass at packaging time.

The detection score is not a UI blocker because target selection comes from
the explicit scenario. The 12 misses are formally explained in the report and
must not be “fixed” through sentence memorization.

## ASR baseline and weak spots

The locked model is `vasista22/whisper-hindi-small`, running on Apple MPS with
16 kHz mono WAV input. Day 2's ten-clip batch averaged 4.207 seconds from audio
input to text and had an aggregate real-time factor of 1.320. Manual accuracy
labels for that batch remain pending because reference transcripts were not
embedded with the uploaded recordings.

Register-critical weaknesses to design around:

- short pronouns can be confused, notably `तू` surfacing as `तो`;
- respectful verb agreement can flatten, for example `सकते हैं` → `सकता है`
  and `चुकी हैं` → a simpler completion such as `लिया`;
- short address and honorific terms are the least reliably preserved and may
  be dropped;
- other observed candidates include `आपने` → `अपने`, dropped copulas, and
  formal-pronoun/verb disagreement.

Therefore the UI should not treat a missing honorific in raw ASR as proof of an
informal scenario. Scenario selection remains authoritative, and a later text
normalization layer should handle these short-token ASR weaknesses.

## Run and verify

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python register_detection_baseline.py
python cross_tier_review.py
```

The Stanza Hindi resources live in `.cache/stanza` in the working project and
are intentionally not duplicated in the handoff archive. Audio/model caches
and WAV/MP3 files are also excluded; manifests, results, code, and setup notes
are included.

## Handoff boundaries

- Keep JSON tables separate from code and treat them as canonical.
- Do not replace deterministic transformation with an LLM.
- Do not make text-only detection the primary scenario selector.
- Preserve first-person service-worker verbs in Reverse Formal.
- Preserve or explicitly request the real kinship term in Formal-Warm.
