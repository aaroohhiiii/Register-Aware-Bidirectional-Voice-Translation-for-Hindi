# Solution Pitch — Register-Aware Bidirectional Voice Translation for Hindi

**The problem.** Hindi grammar encodes respect. The same question is *तू कहाँ जा रहा है?* to a close friend, *तुम कहाँ जा रहे हो?* to an auto driver, and *सर, आप कहाँ जा रहे हैं?* to a doctor — pronoun, verb agreement, imperatives and honorifics all change. Voice assistants and translation tools ignore this, so their Hindi comes out either rude or stiffly formal. Non-native speakers, travellers, and support or healthcare staff feel this every day.

**Our solution.** A voice app that re-speaks what you say in the register that fits the listener. Pick who you are talking to (Friend, Auto driver, Receptionist, Doctor), speak naturally, and hear your sentence back in the right tone. It works in **both directions** of a conversation, so each side hears the register appropriate to them.

**How it works**

1. **Gnani Prisma** converts Hindi speech to text.
2. A cleanup stage removes fillers and fixes punctuation.
3. Our NLP layer detects which of **six register tiers** was used (Full Informal → Respectful Casual → Formal → Full Formal, plus Formal-Warm and Reverse Formal) using Stanza POS tags.
4. A rule-based transformer rewrites pronouns, verb agreement, imperatives, honorifics and address terms for the target tier. The chosen scenario is authoritative; detection is shown for transparency and never overrides it.
5. **Gnani Timbre** (`timbre-v2.5`, voice *Nalini*) speaks the result, switching between `hi-IN` and `hi-en` for code-mixed sentences.

Local Whisper, Parler and MMS models act as an offline fallback if the cloud is unreachable.

**Why Gnani.** In our listening tests Timbre was clearly the best at carrying the तू/तुम/आप distinction in intonation, ahead of the open-source Hindi TTS models we compared. Prisma runs Hindi transcription in the cloud, so the app needs no GPU.

**Results**

| Check | Result |
|---|---|
| Cross-tier rewrites (every source tier × 3 other tiers) | **144 / 144** correct |
| End-to-end smoke sentences | **48 / 48** pass |
| Register detection from text alone | 36 / 48 (75%) — why the user picks the scenario |
| Automated tests | 179 passing |

**Who it helps.** Travellers and non-native speakers navigating Indian cities; customer-support and hospital front-desk staff who must sound respectful; and voice-bot builders who need register-correct Hindi output.

**Next steps.** Streaming (turn-by-turn live) conversation, more scenarios such as elders and teachers, gender-aware verb agreement from speaker voice, and extending the rule set to other Indic languages with honorific systems (Marathi, Bengali).

**Try it:** `python app.py` → record a sentence → choose scenario and direction. Setup steps are in the [README](README.md).
