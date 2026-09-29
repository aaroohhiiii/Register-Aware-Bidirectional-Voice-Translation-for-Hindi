# Day 4 verb agreement and address-term rules

Status: **Drafted for Steps 4.1–4.4.** Machine-readable sources:
`verb_rules.json` and `address_term_rules.json`.

## Verb agreement matrix

Gender is independent of register. Every tier therefore has both masculine and
feminine cells across four categories.

| Tier / subject | Gender | Habitual | Present progressive | Past completive | Modal/potential |
| --- | --- | --- | --- | --- | --- |
| Full Informal / तू | Masculine | करता है | कर रहा है | खा चुका है | कर सकता है |
| Full Informal / तू | Feminine | करती है | कर रही है | खा चुकी है | कर सकती है |
| Full Informal accepted variant / तुम | Masculine | करते हो | कर रहे हो | खा चुके हो | कर सकते हो |
| Full Informal accepted variant / तुम | Feminine | करती हो | कर रही हो | खा चुकी हो | कर सकती हो |
| Respectful Casual / तुम | Masculine | करते हो | कर रहे हो | खा चुके हो | कर सकते हो |
| Respectful Casual / तुम | Feminine | करती हो | कर रही हो | खा चुकी हो | कर सकती हो |
| Formal / आप | Masculine | करते हैं | कर रहे हैं | खा चुके हैं | कर सकते हैं |
| Formal / आप | Feminine | करती हैं | कर रही हैं | खा चुकी हैं | कर सकती हैं |
| Full Formal / आप | Masculine | करते हैं | कर रहे हैं | खा चुके हैं | कर सकते हैं |
| Full Formal / आप | Feminine | करती हैं | कर रही हैं | खा चुकी हैं | कर सकती हैं |
| Formal-Warm / आप | Masculine | करते हैं | कर रहे हैं | खा चुके हैं | कर सकते हैं |
| Formal-Warm / आप | Feminine | करती हैं | कर रही हैं | खा चुकी हैं | कर सकती हैं |
| Reverse Formal / आप | Masculine | करते हैं | कर रहे हैं | खा चुके हैं | कर सकते हैं |
| Reverse Formal / आप | Feminine | करती हैं | कर रही हैं | खा चुकी हैं | कर सकती हैं |

The four `आप` tiers legitimately share verb agreement. Full Formal,
Formal-Warm, and Reverse Formal remain distinct through address-term policy,
imperatives, and honorific behavior; Hindi does not provide six separate
second-person verb paradigms.

## Address-term policy

| Tier | Presence | Terms |
| --- | --- | --- |
| Full Informal | Optional | name, यार, भाई |
| Respectful Casual | Contextual | भैया, भाई साहब, दीदी |
| Formal | Recommended | सर, मैडम, जी, name + जी |
| Full Formal | Required | सर, मैडम, जी, name + जी |
| Formal-Warm | Required and scenario-specific | true kinship term + जी |
| Reverse Formal | Required | सर, मैडम, जी, name + जी |

Important scenario overrides include `डॉक्टर साहब/डॉक्टर मैडम/डॉक्टर जी` for
a doctor, a required formal title for a teacher, true kinship forms such as
`दादी जी` for an elder relative, and a required customer-facing formal term
from the service worker's side.

## ASR reliability warning

Day 2 testing flagged short address terms and honorific words as the least
reliably preserved elements in transcription. These rules must not assume ASR
alone will retain `जी`, `सर`, `मैडम`, or a kinship term. The later normalization
layer must recover or validate address terms before transformation.

## Ground-truth sentence set

Steps 4.5 and 4.6 are captured in `register_test_sentences.json`: 48 tagged
Hindi sentences, exactly eight for each tier. Every record includes its tier,
gender and agreement target, tense/mood, human-readable feature under test, and
machine-usable feature tags.

