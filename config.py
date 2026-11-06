# config.py — Central configuration for the Hindi voice-translation project.
# Edit values here; do NOT hardcode secrets in this file.

# ── ASR ──────────────────────────────────────────────────────────────────────
ASR_BACKEND     = "whisper"      # "whisper" | "prisma"
PRISMA_LANGUAGE = "hi-IN"
ASR_TIMEOUT_S   = 15

# ── TTS ──────────────────────────────────────────────────────────────────────
TTS_BACKEND = "timbre"           # primary backend: "timbre" | "parler" | "mms"

# Listening-test result (2026-10-07):
#   1st  timbre @ speed=1.0  — best register capture; Nalini voice inflects
#                              आप/तुम/तू distinctions naturally.
#   2nd  parler             — good but slightly flat on honorific sentences.
#   3rd  mms               — intelligible fallback; no speed/voice control.
# Fallback order mirrors the ranking above.
TTS_FALLBACK_ORDER = ["timbre", "parler", "mms"]

# ── Timbre (Gnani AI cloud) ───────────────────────────────────────────────────
TIMBRE_VOICE    = "Nalini"       # chosen voice — best register inflection in listening test
TIMBRE_LANGUAGE       = "hi-IN"
TIMBRE_MIXED_LANGUAGE = "hi-en"  # for sentences containing Latin-script words; set after API test
TIMBRE_SPEED    = 1.0            # locked at 1.0 — natural pace, best register capture
                                  # (0.85 was slightly sluggish; 1.15 clipped intonation)

# API key is read from .env (GNANI_API_KEY).  Never put the real key here.
CLOUD_TIMEOUT_S = 20             # seconds before a cloud request is abandoned

# ── Parler TTS (local, ai4bharat/indic-parler-tts) ───────────────────────────
PARLER_DEVICE = "cpu"            # "cpu" is safe everywhere; change to "mps" on Apple Silicon

# Natural-language voice description.  Pace wording is adjusted at runtime
# based on the `speed` argument passed to speak().
PARLER_DESCRIPTION = (
    "Divya's voice is clear and moderate in pace, with a very close recording "
    "and almost no background noise."
)

# ── MMS TTS (local, facebook/mms-tts-hin) ────────────────────────────────────
MMS_MODEL_ID = "facebook/mms-tts-hin"

# ── Scenarios & Bidirectional Targets ─────────────────────────────────────────
# Four demo personas, one per tone, used in both directions. Keys and tiers
# match scenario_lookup.json (checked in tests/test_directions.py).
DEFAULT_SCENARIO = "friend"
SCENARIOS = {
    "friend":       {"A_target": "Full Informal",     "B_target": "Full Informal",     "label": "Friend"},
    "auto_driver":  {"A_target": "Respectful Casual", "B_target": "Respectful Casual", "label": "Auto driver"},
    "receptionist": {"A_target": "Formal",            "B_target": "Formal",            "label": "Receptionist"},
    "doctor":       {"A_target": "Full Formal",       "B_target": "Full Formal",       "label": "Doctor"},
}

# ── UI voice list ─────────────────────────────────────────────────────────────
# Starts with the locked primary voice. Extend this list with other valid
# Gnani Hindi voice names once you have verified them on the Gnani voice list.
# Do NOT invent names — only real voices from the API catalogue belong here.
UI_TIMBRE_VOICES = [TIMBRE_VOICE]  # → ["Nalini"]
