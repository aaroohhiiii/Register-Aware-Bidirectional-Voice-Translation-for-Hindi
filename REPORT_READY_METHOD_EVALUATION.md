# Report-ready methodology, evaluation, and limitations

## Register selection framework

Hindi address grammar is influenced by more than a binary formal/informal
choice. The prototype models the intended register from three social
dimensions: Power (P), Distance (D), and Context (C). Power represents the
addressee's authority or seniority over the speaker. Distance represents
social unfamiliarity independently of authority. Context represents how
institutional, transactional, or high-stakes the interaction is independently
of the participants' relationship. Each dimension is scored from 0 to 1.

For ordinary scenarios, the register dial is the equally weighted mean:

```text
dial = round((P + D + C) / 3, 2)
```

The boundaries were selected from the clusters formed by the scored scenarios,
rather than as arbitrary labels. The intervals are half-open so every value has
exactly one deterministic assignment; only the final endpoint is inclusive.

| Dial interval | Register tier |
|---|---|
| 0.00 ≤ dial < 0.25 | Full Informal |
| 0.25 ≤ dial < 0.45 | Respectful Casual |
| 0.45 ≤ dial < 0.65 | Formal |
| 0.65 ≤ dial ≤ 1.00 | Full Formal |
| Tag-based | Formal-Warm |
| Tag-based | Reverse Formal |

### Scored scenario table

| # | Scenario | P | D | C | Dial | Assignment | Final tier |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | Friend | 0.0 | 0.1 | 0.0 | 0.03 | Threshold | Full Informal |
| 2 | Younger sibling | 0.0 | 0.0 | 0.0 | 0.00 | Threshold | Full Informal |
| 3 | Spouse/partner | 0.0 | 0.0 | 0.0 | 0.00 | Threshold | Full Informal |
| 4 | Classmate | 0.1 | 0.3 | 0.2 | 0.20 | Threshold | Full Informal |
| 5 | Auto driver | 0.1 | 0.6 | 0.3 | 0.33 | Threshold | Respectful Casual |
| 6 | Shopkeeper | 0.1 | 0.6 | 0.4 | 0.37 | Threshold | Respectful Casual |
| 7 | Delivery person | 0.1 | 0.6 | 0.3 | 0.33 | Threshold | Respectful Casual |
| 8 | Colleague (peer) | 0.2 | 0.4 | 0.5 | 0.37 | Threshold | Respectful Casual |
| 9 | Stranger asking directions | 0.1 | 0.8 | 0.2 | 0.37 | Threshold | Respectful Casual |
| 10 | Grandparent/elder relative | 0.7 | 0.1 | 0.2 | 0.33* | `formal_warm` tag | Formal-Warm |
| 11 | Landlord | 0.5 | 0.6 | 0.5 | 0.53 | Threshold | Formal |
| 12 | Bank employee | 0.4 | 0.8 | 0.7 | 0.63 | Threshold | Formal |
| 13 | Receptionist | 0.3 | 0.8 | 0.6 | 0.57 | Threshold | Formal |
| 14 | Hotel/front-desk staff | 0.3 | 0.7 | 0.6 | 0.53 | Threshold | Formal |
| 15 | Boss/manager | 0.8 | 0.5 | 0.6 | 0.63 | Threshold | Formal |
| 16 | Teacher | 0.9 | 0.5 | 0.7 | 0.70 | Threshold | Full Formal |
| 17 | Doctor | 0.7 | 0.6 | 0.8 | 0.70 | Threshold | Full Formal |
| 18 | Job interviewer | 0.8 | 0.9 | 0.9 | 0.87 | Threshold | Full Formal |
| 19 | Police officer | 0.9 | 0.7 | 0.9 | 0.83 | Threshold | Full Formal |
| 20 | Customer (speaker is service worker) | 0.2 | 0.6 | 0.6 | — | `reverse_formal` tag | Reverse Formal |

`*` The value is shown only to demonstrate why a threshold-only method fails;
the special tag determines the final tier.

### Why the two tag-based tiers are necessary

Formal-Warm captures deference toward an elder relative. Such a relationship
combines high age/family-role power with very low social distance and a
personal, low-stakes context. Its numerical mean is 0.33, which would place it
in Respectful Casual even though the appropriate Hindi paradigm uses `आप`,
respectful-plural agreement, and a kinship term plus `जी`. The relationship is
therefore assigned by the `elder_relative` tag.

Reverse Formal captures the direction of deference when the speaker is a
service worker addressing a customer. The important fact is not simply the
customer's position on the same P/D/C scale; it is that the ordinary direction
of power is inverted. The `customer_facing` interaction is therefore treated
as a distinct type and assigned by tag.

## Evaluation

The transformation layer was evaluated on 48 tagged Hindi sentences—eight per
tier—covering subject, possessive, dative, ergative and postpositional pronouns;
gender agreement; habitual, progressive, past, future, modal and imperative
forms; honorifics; and address terms. Each sentence was transformed into three
other tiers, producing 144 reviewed cross-tier cases. After adding the missing
instrumental forms (`तुझसे/तुमसे/आपसे`), all 144 transformations pass. The
combined detection-to-transformation smoke test passes all 48 sentences.

Standalone detection correctly classifies 36 of 48 sentences (75%).

| Ground truth ↓ / Detected → | Full Informal | Respectful Casual | Formal | Full Formal | Formal-Warm | Reverse Formal |
|---|---:|---:|---:|---:|---:|---:|
| Full Informal | 5 | 3 | 0 | 0 | 0 | 0 |
| Respectful Casual | 0 | 8 | 0 | 0 | 0 | 0 |
| Formal | 0 | 0 | 8 | 0 | 0 | 0 |
| Full Formal | 0 | 0 | 3 | 5 | 0 | 0 |
| Formal-Warm | 0 | 0 | 4 | 2 | 2 | 0 |
| Reverse Formal | 0 | 0 | 0 | 0 | 0 | 8 |

The 12 misses comprise four directions: Full Informal → Respectful Casual (3),
Full Formal → Formal (3), Formal-Warm → Formal (4), and Formal-Warm → Full
Formal (2).

## Limitations and interpretation

The detection result demonstrates a linguistic identifiability ceiling rather
than twelve unrelated implementation bugs. Three Full Informal samples use
`तुम`, which is also the defining pronoun of Respectful Casual. Three Full
Formal samples contain `आप` agreement but no obligatory overt honorific, so
their surface grammar is indistinguishable from Formal. Six Formal-Warm samples
do not name the elder-relative relationship; without a kinship term, the same
`आप` morphology can only support Formal or Full Formal.

More generally, all four upper tiers share the `आप` pronoun paradigm and
respectful-plural agreement. Power, distance, family relationship, and
participant roles are properties of the situation, not guaranteed tokens in a
single sentence. Sentence-specific lexical memorization could inflate the
score, but would not solve this general problem.

For that reason, the interface requires explicit scenario selection. The
selected scenario is resolved through `scenario_to_dial()` and supplies the
authoritative target tier. `detect_register()` remains useful as a diagnostic
description of the source sentence or as a user-confirmed fallback when
scenario metadata is unavailable, but its 75% score is not load-bearing for
the transformation pipeline.
