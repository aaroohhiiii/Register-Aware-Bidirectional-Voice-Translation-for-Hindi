# Step 5 register transformation review

## 5.1 Core transformer

`register_transformer.transform(sentence, target_tier, gender=None)` applies
the locked data tables in this order:

1. second-person subject, possessive, dative, and ergative pronouns;
2. addressee verb agreement by target tier, gender, and supported tense/mood;
3. `जी` / `कृपया` policy;
4. tier-specific address terms;
5. unchanged-token fallback for unsupported forms.

It accepts either raw Hindi text or the tagged-token list produced by
`hindi_tokenizer.tokenize_and_tag()`.

Formal-Warm is intentionally conservative. If a real kinship term such as
`दादी` or `नानी` is present, the transformer preserves it and adds `जी`. If
the relationship is absent, it does not guess one from gender because the
locked `elder_relative` override explicitly prohibits that.

## 5.2 Cross-tier review

All 48 locked sentences were converted to three tiers other than their source
tier: **144 cross-tier conversions** in total. The original review found two
instrumental-case failures; Step 6.1 closed both, so the current result is
**144/144 (100%)**. The complete reproducible output is printed by:

```bash
.venv/bin/python cross_tier_review.py
```

| Source IDs | Source tier | Three reviewed target tiers | Result |
|---|---|---|---|
| 1–8 | Full Informal | Respectful Casual, Formal, Full Formal | Pass |
| 9–16 | Respectful Casual | Full Informal, Formal, Reverse Formal | Pass |
| 17–24 | Formal | Full Informal, Respectful Casual, Full Formal | Pass |
| 25–32 | Full Formal | Full Informal, Respectful Casual, Formal | 24/24 pass after Step 6.1 |
| 33–40 | Formal-Warm | Full Informal, Formal, Full Formal | Pass |
| 41–48 | Reverse Formal | Full Informal, Respectful Casual, Formal | Pass |

Representative reviewed outputs:

| ID | Target | Output |
|---:|---|---|
| 1 | Respectful Casual | तुम क्या कर रहे हो? |
| 1 | Formal | आप क्या कर रहे हैं? |
| 1 | Full Formal | सर, आप क्या कर रहे हैं? |
| 4 | Formal | आपने रोटी खा ली? |
| 6 | Formal | आप कल आएंगी? |
| 10 | Reverse Formal | सर, आपके पास खुले पैसे हैं क्या? |
| 14 | Full Informal | तू ये कर सकता है क्या? |
| 16 | Formal | आप कृपया जल्दी आ जाइए |
| 23 | Respectful Casual | तुम कहाँ रहती हो? |
| 25 | Full Informal | भाई, क्या मैं अंदर आ सकता हूं? |
| 28 | Full Informal | तू दोबारा समझा |
| 32 | Respectful Casual | तुम अपना परिचय दो |
| 33 | Full Informal | दादी, तू कैसी है? |
| 38 | Full Formal | सर, आप कब आ रहे हैं? |
| 41 | Respectful Casual | दीदी, तुम्हारे लिए और कुछ लाऊं? |
| 43 | Full Informal | यहां बैठ |
| 47 | Formal | क्या मैं आपकी कोई सहायता कर सकता हूं? |
| 48 | Respectful Casual | तुम थोड़ा इंतज़ार करो |

Manual review checks confirmed that first-person verbs in IDs 25, 45, and 47
remain first-person, while object-controlled participles in IDs 3, 4, and 13
retain their object agreement. These scope checks prevent the transformer from
blindly rewriting every apparent verb ending.

ID 30 originally exposed missing instrumental `आपसे`. Step 6.1 added
`तुझसे/तुमसे/आपसे`, closing both failures. Their original evidence and
resolution remain preserved in `transformation_failure_log.json`.
