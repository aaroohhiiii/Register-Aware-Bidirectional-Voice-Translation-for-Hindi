# Register detection baseline — Steps 5.3–5.6

## Result

`detect_register(sentence)` correctly classifies **36 of 48 sentences
(75.0%)**. It accepts untagged Hindi text, invokes the Step 3.2 tokenizer/POS
tagger, and scores observable pronoun, verb-agreement, honorific, address-term,
and special-case service cues. `detect_register_details()` exposes the score,
confidence margin, and evidence for debugging.

| Ground-truth tier | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Full Informal | 5 | 8 | 62.5% |
| Respectful Casual | 8 | 8 | 100.0% |
| Formal | 8 | 8 | 100.0% |
| Full Formal | 5 | 8 | 62.5% |
| Formal-Warm | 2 | 8 | 25.0% |
| Reverse Formal | 8 | 8 | 100.0% |
| **Overall** | **36** | **48** | **75.0%** |

## Confusion matrix

Rows are ground truth and columns are detected tiers.

| Ground truth ↓ / Detected → | Full Informal | Respectful Casual | Formal | Full Formal | Formal-Warm | Reverse Formal |
|---|---:|---:|---:|---:|---:|---:|
| Full Informal | 5 | 3 | 0 | 0 | 0 | 0 |
| Respectful Casual | 0 | 8 | 0 | 0 | 0 | 0 |
| Formal | 0 | 0 | 8 | 0 | 0 | 0 |
| Full Formal | 0 | 0 | 3 | 5 | 0 | 0 |
| Formal-Warm | 0 | 0 | 4 | 2 | 2 | 0 |
| Reverse Formal | 0 | 0 | 0 | 0 | 0 | 8 |

The 12 misses therefore consist of exactly four confusion directions:

- Full Informal → Respectful Casual: 3
- Full Formal → Formal: 3
- Formal-Warm → Formal: 4
- Formal-Warm → Full Formal: 2

## Full 48-sentence baseline

| ID | Ground truth | Detected | Result |
|---:|---|---|---|
| 1 | Full Informal | Full Informal | Correct |
| 2 | Full Informal | Full Informal | Correct |
| 3 | Full Informal | Full Informal | Correct |
| 4 | Full Informal | Full Informal | Correct |
| 5 | Full Informal | Respectful Casual | **Wrong** |
| 6 | Full Informal | Respectful Casual | **Wrong** |
| 7 | Full Informal | Full Informal | Correct |
| 8 | Full Informal | Respectful Casual | **Wrong** |
| 9 | Respectful Casual | Respectful Casual | Correct |
| 10 | Respectful Casual | Respectful Casual | Correct |
| 11 | Respectful Casual | Respectful Casual | Correct |
| 12 | Respectful Casual | Respectful Casual | Correct |
| 13 | Respectful Casual | Respectful Casual | Correct |
| 14 | Respectful Casual | Respectful Casual | Correct |
| 15 | Respectful Casual | Respectful Casual | Correct |
| 16 | Respectful Casual | Respectful Casual | Correct |
| 17 | Formal | Formal | Correct |
| 18 | Formal | Formal | Correct |
| 19 | Formal | Formal | Correct |
| 20 | Formal | Formal | Correct |
| 21 | Formal | Formal | Correct |
| 22 | Formal | Formal | Correct |
| 23 | Formal | Formal | Correct |
| 24 | Formal | Formal | Correct |
| 25 | Full Formal | Full Formal | Correct |
| 26 | Full Formal | Formal | **Wrong** |
| 27 | Full Formal | Full Formal | Correct |
| 28 | Full Formal | Full Formal | Correct |
| 29 | Full Formal | Full Formal | Correct |
| 30 | Full Formal | Formal | **Wrong** |
| 31 | Full Formal | Formal | **Wrong** |
| 32 | Full Formal | Full Formal | Correct |
| 33 | Formal-Warm | Formal-Warm | Correct |
| 34 | Formal-Warm | Formal | **Wrong** |
| 35 | Formal-Warm | Full Formal | **Wrong** |
| 36 | Formal-Warm | Formal-Warm | Correct |
| 37 | Formal-Warm | Full Formal | **Wrong** |
| 38 | Formal-Warm | Formal | **Wrong** |
| 39 | Formal-Warm | Formal | **Wrong** |
| 40 | Formal-Warm | Formal | **Wrong** |
| 41 | Reverse Formal | Reverse Formal | Correct |
| 42 | Reverse Formal | Reverse Formal | Correct |
| 43 | Reverse Formal | Reverse Formal | Correct |
| 44 | Reverse Formal | Reverse Formal | Correct |
| 45 | Reverse Formal | Reverse Formal | Correct |
| 46 | Reverse Formal | Reverse Formal | Correct |
| 47 | Reverse Formal | Reverse Formal | Correct |
| 48 | Reverse Formal | Reverse Formal | Correct |

## Detection failure log

Every detection miss is assigned to the component whose available cue was
ambiguous or absent.

| IDs | Actual → detected | Component | Traceable cause |
|---|---|---|---|
| 5, 6, 8 | Full Informal → Respectful Casual | Pronoun/verb evidence | These Full Informal examples use the accepted `तुम` paradigm, which is identical to the Respectful Casual paradigm without scenario context. |
| 26, 30, 31 | Full Formal → Formal | Honorific/address logic | These sentences contain respectful `आप` morphology but no strict address or politeness marker; grammar alone matches Formal. |
| 34, 38, 39, 40 | Formal-Warm → Formal | Honorific/address logic | No kinship term is present, so the elder-relative special-case tag is not observable in the sentence. |
| 35, 37 | Formal-Warm → Full Formal | Honorific/address logic | Formal imperative `कीजिए` is observable, but the missing kinship term makes warmth unobservable and the strict-formality cue wins. |

These are not silently patched with sentence-specific memorization. The
special tiers encode relationship/context that morphology alone cannot always
recover. In particular, `तुम` legitimately occurs in both Full Informal and
Respectful Casual; all four upper tiers share `आप` and respectful-plural verb
agreement; warmth is a relationship property rather than a verb class; and
Reverse Formal depends on who occupies the service role. Scenario and
conversation state are therefore required for principled disambiguation.

## Architecture decision: detection is secondary

The **primary, load-bearing path** is user-selected scenario →
`scenario_to_dial()` → target tier → `transform()`. An explicit scenario always
wins, including the tag-based `elder_relative` and `customer_facing` cases.

`detect_register()` is a secondary signal for observing the incoming
sentence's current register, avoiding unnecessary conversion, diagnostics, or
fallback behavior when no scenario metadata is available. Its 75% standalone
accuracy does **not** select or override the main pipeline's target tier. This
is why the 12 context-identifiability misses are reportable limitations rather
than load-bearing pipeline failures.

## Transformation failure log (Step 5.3)

Manual review originally found **2 failures**, both for ID 30 when converting
instrumental `आपसे` into a low tier. Step 6.1 resolved both:

| Target | Actual | Expected | Responsible component |
|---|---|---|---|
| Full Informal | आपसे मिलकर खुशी हुई | तुझसे मिलकर खुशी हुई | Pronoun table — resolved |
| Respectful Casual | आपसे मिलकर खुशी हुई | तुमसे मिलकर खुशी हुई | Pronoun table — resolved |

The current transformation result is **144/144 (100%)**. The verb table and
honorific/address logic produced no failures in the reviewed matrix.
Machine-readable history and resolution details are in
`transformation_failure_log.json`.

## Detection → transformation smoke test (Step 5.6)

For every sentence, the detected tier selected a different target tier and the
sentence was transformed. After the Step 6.1 instrumental fix, **48/48 outputs
pass (100%)** the target-pronoun, particle, non-empty-output, and punctuation
checks. ID 30 now correctly produces `तुझसे मिलकर खुशी हुई` for its smoke-test
target.

Reproduce both baselines as JSON with:

```bash
.venv/bin/python register_detection_baseline.py
```
