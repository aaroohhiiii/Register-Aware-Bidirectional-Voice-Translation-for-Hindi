# Register-selection architecture decision

Status: **confirmed in Step 6.4**

## Decision

The target register is selected from explicit conversation context:

```text
user-selected scenario
        ↓
scenario_to_dial(scenario)
        ↓
authoritative target tier
        ↓
transform(sentence, target tier, ...)
```

`detect_register(sentence)` observes the source sentence. It is a secondary
or fallback signal and must never override a valid explicit scenario.

## Why

Register is partly grammatical but partly relational. Surface Hindi cannot
always reveal power, distance, institutional context, family relationship, or
which participant is the service worker. This is confirmed by the locked
baseline: 36/48 sentences are detectable from text alone, while 12 share the
same morphology with another tier.

The two tag-based tiers make the distinction especially clear:

- `elder_relative` selects Formal-Warm and supplies the actual kinship address
  term when known.
- `customer_facing` selects Reverse Formal and identifies the direction of
  deference from service worker to customer.

## Permitted uses of detection

- record the apparent source register;
- skip a rewrite when source and target are already aligned;
- provide diagnostics and confidence/evidence in evaluation;
- act as a best-effort fallback only when scenario context is unavailable.

## Prohibited use

A detected tier—especially a low-confidence one—must not replace the tier
returned by `scenario_to_dial()` for an explicit user-selected scenario.

## Day 10 interface contract for Harshita

1. Present the 20 scenario choices from `scenario_lookup.json`; do not duplicate
   their labels or tier assignments in UI code.
2. Require the user to select a scenario before register transformation. Store
   and pass its stable snake-case `key`, not its display label.
3. Resolve the target only with `scenario_to_dial(selected_key)`.
4. The interface may show `detect_register()` as an informational “detected
   source register” value, but it must not silently alter the selected scenario
   or resolved target tier.
5. For `elder_relative`, request or preserve the real kinship address term
   (`दादी जी`, `नानी जी`, etc.) and pass it to `transform()`; never infer the
   relationship from gender.
6. For `customer_facing`, pass the customer-facing scenario and, when known,
   the chosen address term (`सर`, `मैडम`, or `जी`).
7. If no scenario has been selected, detection may be offered as a clearly
   labelled fallback suggestion. The user must confirm it before it becomes a
   transformation target.

Minimal interface-side call pattern:

```python
target_tier = scenario_to_dial(selected_scenario_key)
output = transform(
    sentence,
    target_tier,
    gender,
    scenario=(selected_scenario_key if target_tier in {"Formal-Warm", "Reverse Formal"} else None),
    address_term=selected_address_term,
)
```
