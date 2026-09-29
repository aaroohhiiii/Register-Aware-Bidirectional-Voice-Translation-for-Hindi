"""Rule-based detection of the current Hindi register tier.

Detection is intentionally separate from transformation.  It tokenizes and
POS-tags untagged input, then scores the six locked tiers from observable
pronoun paradigms, verb agreement, honorific/address terms, and the explicit
special-case cues needed for Formal-Warm and Reverse Formal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from hindi_tokenizer import tokenize_and_tag


TIERS = (
    "Full Informal",
    "Respectful Casual",
    "Formal",
    "Full Formal",
    "Formal-Warm",
    "Reverse Formal",
)

TU_FORMS = frozenset(
    {
        "तू", "तूने", "तुझे", "तुझको", "तेरा", "तेरी", "तेरे", "तुझ",
        "तुझसे", "तुझमें", "तुझपर", "तुझपे", "तुझतक",
    }
)
TUM_FORMS = frozenset(
    {
        "तुम", "तुमने", "तुम्हें", "तुमको", "तुम्हारा", "तुम्हारी", "तुम्हारे",
        "तुमसे", "तुममें", "तुमपर", "तुमपे", "तुमतक",
    }
)
AAP_FORMS = frozenset(
    {
        "आप", "आपने", "आपको", "आपका", "आपकी", "आपके", "आपसे", "आपमें",
        "आपपर", "आपपे", "आपतक",
    }
)
KINSHIP_TERMS = frozenset(
    {"दादी", "नानी", "दादा", "नाना", "बुआ", "मौसी", "चाचा", "मामा"}
)
STRICT_FORMAL_TERMS = frozenset({"सर", "मैडम", "साहब", "डॉक्टर", "कृपया"})
FORMAL_IMPERATIVES = frozenset(
    {"कीजिए", "दीजिए", "समझाइए", "बैठिए", "करें", "जाइए"}
)
SERVICE_CUES = frozenset(
    {"ऑर्डर", "बिल", "सहायता", "परेशानी", "इंतज़ार", "लाऊं", "लाता"}
)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    tier: str
    confidence: float
    scores: dict[str, int]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _contains_sequence(words: list[str], sequence: Iterable[str]) -> bool:
    wanted = list(sequence)
    width = len(wanted)
    return any(words[index : index + width] == wanted for index in range(len(words) - width + 1))


def detect_register_details(sentence: str) -> DetectionResult:
    """Return the detected tier plus transparent scoring evidence."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("sentence must be a non-empty string")

    tagged = tokenize_and_tag(sentence)
    words = [token.text for token in tagged]
    word_set = set(words)
    scores = {tier: 0 for tier in TIERS}
    evidence: list[str] = []

    tu_matches = sorted(word_set & TU_FORMS)
    tum_matches = sorted(word_set & TUM_FORMS)
    aap_matches = sorted(word_set & AAP_FORMS)
    if tu_matches:
        scores["Full Informal"] += 10
        evidence.append(f"तू paradigm: {', '.join(tu_matches)}")
    if tum_matches:
        scores["Respectful Casual"] += 8
        evidence.append(f"तुम paradigm: {', '.join(tum_matches)}")
    if aap_matches:
        # आप morphology proves respectful-plural agreement but cannot by
        # itself distinguish the four tiers that share it.  Formal is the
        # unmarked default; observable special-case markers can override it.
        scores["Formal"] += 7
        scores["Full Formal"] += 2
        scores["Formal-Warm"] += 2
        scores["Reverse Formal"] += 2
        evidence.append(f"आप paradigm: {', '.join(aap_matches)}")

    # Agreement patterns strengthen the pronoun evidence without rewriting
    # nominal copulas such as "रसीद तैयार है" as register cues.
    for index in range(len(words) - 1):
        agreement, auxiliary = words[index], words[index + 1]
        if auxiliary == "हो" and (
            agreement in {"रहे", "रही", "सकते", "सकती"}
            or agreement.endswith(("ते", "ती"))
        ):
            scores["Respectful Casual"] += 4
            evidence.append("casual-plural verb agreement")
            break
        if auxiliary == "हैं" and (
            agreement in {"रहे", "रही", "सकते", "सकती", "चुके", "चुकी"}
            or agreement.endswith(("ते", "ती"))
        ):
            for tier in ("Formal", "Full Formal", "Formal-Warm", "Reverse Formal"):
                scores[tier] += 2
            evidence.append("respectful-plural verb agreement")
            break
        if auxiliary == "है" and agreement in {"रहा", "रही", "सकता", "सकती"}:
            scores["Full Informal"] += 4
            evidence.append("intimate-singular verb agreement")
            break

    kinship_matches = sorted(word_set & KINSHIP_TERMS)
    if kinship_matches:
        scores["Formal-Warm"] += 14
        evidence.append(f"elder-relative address: {', '.join(kinship_matches)}")

    service_matches = sorted(word_set & SERVICE_CUES)
    service_offer = "मैं" in word_set and (
        bool(word_set & {"लाऊं", "लाता", "सहायता"})
        or _contains_sequence(words, ("कर", "सकता", "हूं"))
    )
    polite_seating = "कृपया" in word_set and "बैठिए" in word_set and not aap_matches
    if service_matches or service_offer or polite_seating:
        scores["Reverse Formal"] += 14
        cues = service_matches or (["first-person service offer"] if service_offer else ["polite seating instruction"])
        evidence.append(f"service-context cue: {', '.join(cues)}")

    strict_matches = sorted(word_set & STRICT_FORMAL_TERMS)
    imperative_matches = sorted(word_set & FORMAL_IMPERATIVES)
    if strict_matches or imperative_matches:
        scores["Full Formal"] += 6
        scores["Reverse Formal"] += 2
        markers = strict_matches + imperative_matches
        evidence.append(f"strict formal marker: {', '.join(dict.fromkeys(markers))}")

    # A leading/trailing जी is characteristic of customer-facing service in
    # this project's locked examples; medial जी after kinship is handled by
    # the higher-priority Formal-Warm evidence above.
    if "जी" in word_set and not kinship_matches:
        scores["Reverse Formal"] += 5
        evidence.append("standalone जी")

    # Stable tie order makes the conservative unmarked defaults explicit.
    tie_priority = {
        "Formal-Warm": 6,
        "Reverse Formal": 5,
        "Full Formal": 4,
        "Formal": 3,
        "Respectful Casual": 2,
        "Full Informal": 1,
    }
    tier = max(TIERS, key=lambda candidate: (scores[candidate], tie_priority[candidate]))
    ordered = sorted(scores.values(), reverse=True)
    best = ordered[0]
    runner_up = ordered[1]
    confidence = 0.0 if best == 0 else round((best - runner_up) / best, 3)
    if not evidence:
        evidence.append("no registered cue; conservative Formal default")
    return DetectionResult(tier, confidence, scores, tuple(evidence))


def detect_register(sentence: str) -> str:
    """Infer the current tier of one untagged Hindi sentence."""
    return detect_register_details(sentence).tier


__all__ = ["DetectionResult", "detect_register", "detect_register_details"]
