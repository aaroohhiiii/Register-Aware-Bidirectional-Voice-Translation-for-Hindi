# Hindi pronoun rules by register tier

Status: **closed after the Step 6.2 case audit.** The authoritative machine-readable table is
`pronoun_rules.json`.

## Scope

This table covers the second-person forms the register transformation layer
will rewrite: subject, possessive, dative/accusative, ergative, and the common
postpositional cases used with second-person pronouns. It does not rewrite
first- or third-person pronouns.

The six register tiers use three real Hindi pronoun paradigms. Upper-tier
differences will come from verb agreement, imperatives, politeness markers,
and honorific handling in later rule tables; they should not be represented by
invented pronoun forms.

## Canonical forms

| Tier | Subject | Possessive: masc. singular direct | Possessive: feminine | Possessive: masc. plural/oblique | Dative | Ergative |
| --- | --- | --- | --- | --- | --- | --- |
| Full Informal | तू | तेरा | तेरी | तेरे | तुझे (`तुझको` accepted) | तूने |
| Respectful Casual | तुम | तुम्हारा | तुम्हारी | तुम्हारे | तुम्हें (`तुमको` accepted) | तुमने |
| Formal | आप | आपका | आपकी | आपके | आपको | आपने |
| Full Formal | आप | आपका | आपकी | आपके | आपको | आपने |
| Formal-Warm | आप | आपका | आपकी | आपके | आपको | आपने |
| Reverse Formal | आप | आपका | आपकी | आपके | आपको | आपने |

## Postpositional case forms

| Tier/paradigm | Instrumental, ablative, or `से`-comitative | Interior locative | Surface locative | Terminative |
| --- | --- | --- | --- | --- |
| Full Informal (`तू`) | तुझसे | तुझमें | तुझ पर | तुझ तक |
| Respectful Casual (`तुम`) | तुमसे | तुममें | तुम पर | तुम तक |
| Four formal tiers (`आप`) | आपसे | आपमें | आप पर | आप तक |

The transformer also recognizes common split/joined spelling variants such as
`आप से`, `तुमपर`, `तुझ पे`, and `आपतक`, then emits the canonical form above.

## Agreement rules

- Possessive forms agree with the possessed noun: `तेरा/तुम्हारा/आपका घर`,
  `तेरी/तुम्हारी/आपकी गाड़ी`, and `तेरे/तुम्हारे/आपके कमरे`.
- The masculine plural/oblique form also covers masculine singular before a
  postposition, for example `आपके घर में`.
- `तुझे`, `तुम्हें`, and `आपको` are the canonical generated dative forms.
  `तुझको` and `तुमको` are recognized variants but are not the default output.
- `तू` is intentionally restricted to Full Informal. Because it can sound rude
  without genuine intimacy, the scenario tier must be resolved before this
  table is applied.
- The four `आप` tiers remain distinct even though their pronoun forms match;
  later verb, imperative, and honorific tables carry that distinction.

## Step 6.2 completeness audit

| Construction | Coverage | Implementation |
| --- | --- | --- |
| Nominative/subject | Complete | `तू / तुम / आप` |
| Accusative/dative | Complete | `तुझे / तुम्हें / आपको`, including `-को` variants |
| Ergative | Complete | `तूने / तुमने / आपने` |
| Genitive/possessive | Complete | Three-way possessed-noun agreement |
| Instrumental/ablative | Added in 6.1 | Contracted `-से` forms |
| Interior locative | Added in 6.2 | `-में`, joined and split spellings |
| Surface locative | Added in 6.2 | `पर/पे`, joined and split spellings |
| Terminative | Added in 6.2 | `तक`, joined and split spellings |
| Comitative (`के साथ`) | Already covered | Oblique possessive form + unchanged `साथ` |
| Benefactive (`के लिए`) | Already covered | Oblique possessive form + unchanged `लिए` |
| Proximity (`के पास`) | Already covered | Oblique possessive form + unchanged `पास` |
| Topic (`के बारे में`) | Already covered | Oblique possessive form + unchanged postposition sequence |

Vocatives are handled by the address-term table, not the pronoun table.
Reflexive `अपना` agrees with the possessed noun but does not encode the
addressee's register, so it correctly remains outside this second-person map.
