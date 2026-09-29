# Register scenario reference

Status: **Locked after Steps 2.11–2.15.** Days 3–7 should consume
`scenario_lookup.json` rather than redefine or duplicate these values in code.

## Dimension definitions

- **Power (P):** Does the other person hold authority, status, or seniority over
  the speaker, such that deference is owed regardless of familiarity?
  `0 = no authority`, `1 = high authority`.
- **Distance (D):** How socially unfamiliar or unrelated are the two people,
  independent of power? `0 = very close`, `1 = complete stranger/institutional`.
- **Context (C):** How institutional, transactional, or high-stakes is the
  interaction itself, independent of who the person is? `0 = casual/personal`,
  `1 = highly institutional/high-stakes`.

For ordinary scenarios, the dial is `round((P + D + C) / 3, 2)`.

## Thresholds

The ranges are encoded as non-overlapping intervals for deterministic lookup.
An exact boundary belongs to the tier beginning at that boundary; the final
upper endpoint, `1.00`, is inclusive. This machine-use convention does not
change any of the 20 current assignments.

| Dial interval | Tier |
| --- | --- |
| `0.00 <= dial < 0.25` | Full Informal |
| `0.25 <= dial < 0.45` | Respectful Casual |
| `0.45 <= dial < 0.65` | Formal |
| `0.65 <= dial <= 1.00` | Full Formal |
| Tag only | Formal-Warm |
| Tag only | Reverse Formal |

## Final scenario table

| # | Scenario | P | D | C | Dial | Assignment | Tier | Justification |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | Friend | 0.0 | 0.1 | 0.0 | 0.03 | Threshold | Full Informal | No authority, close, purely casual. |
| 2 | Younger sibling | 0.0 | 0.0 | 0.0 | 0.00 | Threshold | Full Informal | No authority, closest possible, no stakes. |
| 3 | Spouse/partner | 0.0 | 0.0 | 0.0 | 0.00 | Threshold | Full Informal | No authority, maximally close, personal. |
| 4 | Classmate | 0.1 | 0.3 | 0.2 | 0.20 | Threshold | Full Informal | Peer, slight unfamiliarity if not close friends, low stakes. |
| 5 | Auto driver | 0.1 | 0.6 | 0.3 | 0.33 | Threshold | Respectful Casual | No real authority, stranger, brief transaction. |
| 6 | Shopkeeper | 0.1 | 0.6 | 0.4 | 0.37 | Threshold | Respectful Casual | No authority, stranger, transactional but routine. |
| 7 | Delivery person | 0.1 | 0.6 | 0.3 | 0.33 | Threshold | Respectful Casual | No authority, stranger, brief and routine. |
| 8 | Colleague (peer) | 0.2 | 0.4 | 0.5 | 0.37 | Threshold | Respectful Casual | No hierarchy, moderately familiar, workplace context raises stakes slightly. |
| 9 | Stranger asking directions | 0.1 | 0.8 | 0.2 | 0.37 | Threshold | Respectful Casual | No authority, total stranger, but trivial low-stakes ask. |
| 10 | Grandparent/elder relative | 0.7 | 0.1 | 0.2 | 0.33* | `formal_warm` tag | Formal-Warm | High age/family-role deference is owed despite closeness and low institutional stakes. |
| 11 | Landlord | 0.5 | 0.6 | 0.5 | 0.53 | Threshold | Formal | Some authority through control of housing, fairly distant, moderately transactional. |
| 12 | Bank employee | 0.4 | 0.8 | 0.7 | 0.63 | Threshold | Formal | Institutional authority, stranger, high stakes involving money. |
| 13 | Receptionist | 0.3 | 0.8 | 0.6 | 0.57 | Threshold | Formal | Mild institutional authority, stranger, moderately formal setting. |
| 14 | Hotel/front-desk staff | 0.3 | 0.7 | 0.6 | 0.53 | Threshold | Formal | Service interaction with a stranger in an institutional setting. |
| 15 | Boss/manager | 0.8 | 0.5 | 0.6 | 0.63 | Threshold | Formal | Direct authority over the speaker combined with workplace stakes. |
| 16 | Teacher | 0.9 | 0.5 | 0.7 | 0.70 | Threshold | Full Formal | High authority, moderate familiarity, educational stakes. |
| 17 | Doctor | 0.7 | 0.6 | 0.8 | 0.70 | Threshold | Full Formal | Expert authority, social distance, and high health-related stakes. |
| 18 | Job interviewer | 0.8 | 0.9 | 0.9 | 0.87 | Threshold | Full Formal | High authority over the outcome, stranger, maximum stakes. |
| 19 | Police officer | 0.9 | 0.7 | 0.9 | 0.83 | Threshold | Full Formal | Institutional/legal authority, stranger, high stakes. |
| 20 | Customer (you as service worker) | 0.2 | 0.6 | 0.6 | — | `reverse_formal` tag | Reverse Formal | The speaker owes deference because the ordinary power direction is inverted. |

`*` The Formal-Warm dial is displayed only for reference; the tag overrides
the threshold result.

## Special cases

- **Formal-Warm:** An elder relative is socially close and the interaction is
  low-stakes, but age and family role still require deference. The numeric
  value `0.33` would otherwise misclassify this as Respectful Casual.
- **Reverse Formal:** A service worker owes deference to the customer because
  the power direction is inverted. It is an interaction type, so its dial is
  intentionally `null` and assignment is tag-based.

## Data contract

`scenario_lookup.json` is the authoritative machine-readable source. Each
scenario contains its stable numeric ID, snake-case API key, name, P/D/C
scores, rounded dial (or `null`), final tier, assignment mode, optional special
tag, and one-line justification. Threshold and special-tag definitions live in
the same file but outside the scenario records so later code can load them
without hard-coding. Step 3.1's `scenario_to_dial()` reads this file directly.

Step 3.3's second-person morphology is kept separately in
`pronoun_rules.json`, keyed by the six final tier labels defined here.

Day 4's tier×gender verb patterns and address-term policies are kept in
`verb_rules.json` and `address_term_rules.json` respectively.

The locked Day 4 test corpus is `register_test_sentences.json`, with eight
ground-truth sentences for each of the six tiers.
