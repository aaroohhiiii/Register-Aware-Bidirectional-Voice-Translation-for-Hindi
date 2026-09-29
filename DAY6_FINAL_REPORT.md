# Day 6 final report

## Final measured results

| Check | Result |
|---|---:|
| Cross-tier transformations | 144/144 (100%) |
| Detection → transformation smoke test | 48/48 (100%) |
| Standalone register detection | 36/48 (75%) |
| Transformation failures still open | 0 |

## Detection limitation

The 12 detection misses fall into four observed directions: Full Informal →
Respectful Casual (3), Full Formal → Formal (3), Formal-Warm → Formal (4), and
Formal-Warm → Full Formal (2). This is expected from the linguistic design:
`तुम` spans two lower tiers, the four upper tiers share `आप` agreement, and
Formal-Warm encodes a family relationship that may not be named in every
sentence. The full confusion matrix and per-sentence log are in
`REGISTER_DETECTION_BASELINE.md`.

## Architecture conclusion

Detection is not load-bearing. The user-selected scenario is resolved through
`scenario_to_dial()` and remains the authoritative target tier. Detection is
used only to observe the source register or as a fallback when scenario
metadata is absent. See `REGISTER_SELECTION_ARCHITECTURE.md`.

## Special-case branches

- Formal-Warm validates the `elder_relative` override and uses a supplied or
  existing true kinship term plus `जी`; it never guesses a relationship from
  gender.
- Reverse Formal validates the `customer_facing` override and can force an
  appropriate customer address such as `सर`, `मैडम`, or `जी`.
- Service-worker first-person agreement (`मैं ... लाता/लाती हूं`, `मैं ... कर
  सकता हूं`) is explicitly protected from second-person agreement rewrites.

## Consolidated bug status

The two observed transformation failures are resolved. The broader pronoun
case audit is closed. There are no open failures in the locked transformation
or smoke suites. The remaining detection items are confirmed report
limitations, not unresolved transformation bugs. Machine-readable status is
in `day6_bug_list.json`.
