"""Deterministic Hindi register transformation.

The transformer consumes either raw Hindi text or the tagged tokens returned by
``hindi_tokenizer.tokenize_and_tag``.  It applies the project's JSON rule
tables in the order fixed in Step 5.1: pronouns, verb agreement, politeness
particles, and address terms.  Unknown forms are preserved verbatim.

This is deliberately rule based.  It makes every supported rewrite
explainable and, more importantly, prevents a generative fallback from
silently changing the meaning of a sentence when a grammatical form is not in
the project's small rule inventory.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Sequence

from hindi_tokenizer import TaggedToken, tokenize_and_tag


PROJECT_DIR = Path(__file__).resolve().parent
PUNCTUATION = frozenset({",", ".", "?", "!", "।", ";", ":"})
GENDERS = frozenset({"masculine", "feminine"})

KINSHIP_TERMS = frozenset(
    {"दादी", "नानी", "दादा", "नाना", "बुआ", "मौसी", "चाचा", "मामा"}
)
MASCULINE_ADDRESS_TERMS = frozenset(
    {"सर", "साहब", "भैया", "भाई", "दादा", "नाना", "चाचा", "मामा"}
)
FEMININE_ADDRESS_TERMS = frozenset(
    {"मैडम", "दीदी", "दादी", "नानी", "बुआ", "मौसी"}
)

# Formal imperatives have lexical stem changes, so a small explicit inventory
# is safer than suffix stripping.  Forms not listed here follow the fallback
# rule and remain untouched.
IMPERATIVE_FORMS = {
    "समझाइए": {
        "intimate_singular": "समझा",
        "casual_plural": "समझाओ",
        "respectful_plural": "समझाइए",
    },
    "दीजिए": {
        "intimate_singular": "दे",
        "casual_plural": "दो",
        "respectful_plural": "दीजिए",
    },
    "कीजिए": {
        "intimate_singular": "कर",
        "casual_plural": "करो",
        "respectful_plural": "कीजिए",
    },
    "बैठिए": {
        "intimate_singular": "बैठ",
        "casual_plural": "बैठो",
        "respectful_plural": "बैठिए",
    },
    "करें": {
        "intimate_singular": "कर",
        "casual_plural": "करो",
        "respectful_plural": "करें",
    },
    "रखिए": {
        "intimate_singular": "रख",
        "casual_plural": "रखो",
        "respectful_plural": "रखिए",
    },
    "रुकिए": {
        "intimate_singular": "रुक",
        "casual_plural": "रुको",
        "respectful_plural": "रुकिए",
    },
    "लीजिए": {
        "intimate_singular": "ले",
        "casual_plural": "लो",
        "respectful_plural": "लीजिए",
    },
    "बोलिए": {
        "intimate_singular": "बोल",
        "casual_plural": "बोलो",
        "respectful_plural": "बोलिए",
    },
    "सुनिए": {
        "intimate_singular": "सुन",
        "casual_plural": "सुनो",
        "respectful_plural": "सुनिए",
    },
    "देखिए": {
        "intimate_singular": "देख",
        "casual_plural": "देखो",
        "respectful_plural": "देखिए",
    },
    "चलिए": {
        "intimate_singular": "चल",
        "casual_plural": "चलो",
        "respectful_plural": "चलिए",
    },
    "आइए": {
        "intimate_singular": "आ",
        "casual_plural": "आओ",
        "respectful_plural": "आइए",
    },
    "जाइए": {
        "intimate_singular": "जा",
        "casual_plural": "जाओ",
        "respectful_plural": "जाइए",
    },
    "पढ़िए": {
        "intimate_singular": "पढ़",
        "casual_plural": "पढ़ो",
        "respectful_plural": "पढ़िए",
    },
    "लिखिए": {
        "intimate_singular": "लिख",
        "casual_plural": "लिखो",
        "respectful_plural": "लिखिए",
    },
    "दिखाइए": {
        "intimate_singular": "दिखा",
        "casual_plural": "दिखाओ",
        "respectful_plural": "दिखाइए",
    },
    "बताइए": {
        "intimate_singular": "बता",
        "casual_plural": "बताओ",
        "respectful_plural": "बताइए",
    },
    "लाइए": {
        "intimate_singular": "ला",
        "casual_plural": "लाओ",
        "respectful_plural": "लाइए",
    },
    "खाइए": {
        "intimate_singular": "खा",
        "casual_plural": "खाओ",
        "respectful_plural": "खाइए",
    },
    "पीजिए": {
        "intimate_singular": "पी",
        "casual_plural": "पियो",
        "respectful_plural": "पीजिए",
    },
    "रहिए": {
        "intimate_singular": "रह",
        "casual_plural": "रहो",
        "respectful_plural": "रहिए",
    },
    "उठिए": {
        "intimate_singular": "उठ",
        "casual_plural": "उठो",
        "respectful_plural": "उठिए",
    },
    "भेजिए": {
        "intimate_singular": "भेज",
        "casual_plural": "भेजो",
        "respectful_plural": "भेजिए",
    },
    "मिलिए": {
        "intimate_singular": "मिल",
        "casual_plural": "मिलो",
        "respectful_plural": "मिलिए",
    },
}

# Reverse entries allow cross-tier conversion in either direction.  The short
# intimate forms are context-checked below because some can also be lexical
# stems in a larger verb phrase.
AMBIGUOUS_SHORT_IMPERATIVES = frozenset(
    _forms["intimate_singular"] for _forms in IMPERATIVE_FORMS.values()
)
for _formal, _forms in list(IMPERATIVE_FORMS.items()):
    for _variant in set(_forms.values()):
        IMPERATIVE_FORMS.setdefault(_variant, _forms)


@lru_cache(maxsize=None)
def _load_json(filename: str) -> dict:
    with (PROJECT_DIR / filename).open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _pronoun_rules() -> tuple[dict[str, dict], dict[str, tuple[str, str]]]:
    """Return target tier rules and source form -> (case, subtype)."""
    tiers = {
        entry["tier"]: entry for entry in _load_json("pronoun_rules.json")["tiers"]
    }
    source_forms: dict[str, tuple[str, str]] = {}
    for entry in tiers.values():
        forms = entry["forms"]
        source_forms[forms["subject"]] = ("subject", "canonical")
        source_forms[forms["ergative"]] = ("ergative", "canonical")
        source_forms[forms["dative"]["canonical"]] = ("dative", "canonical")
        variant = forms["dative"].get("accepted_variant")
        if variant:
            source_forms[variant] = ("dative", "canonical")
        for subtype, form in forms["possessive"].items():
            source_forms[form] = ("possessive", subtype)
        for subtype, specification in forms["postpositional"].items():
            source_forms[specification["canonical"]] = ("postpositional", subtype)
            for variant in specification["accepted_variants"]:
                source_forms[variant] = ("postpositional", subtype)
    return tiers, source_forms


@lru_cache(maxsize=1)
def _verb_rules() -> dict[str, dict]:
    return {
        entry["tier"]: entry for entry in _load_json("verb_rules.json")["tiers"]
    }


@lru_cache(maxsize=1)
def _address_config() -> dict:
    return _load_json("address_term_rules.json")


@lru_cache(maxsize=1)
def _address_rules() -> dict[str, dict]:
    return {
        entry["tier"]: entry
        for entry in _address_config()["tier_rules"]
    }


def _resolve_special_case_context(
    target_tier: str,
    scenario: str | None,
    address_term: str | None,
) -> tuple[str | None, str | None]:
    """Validate the two tag-based tiers against scenario-level overrides."""
    special_scenarios = {
        "Formal-Warm": "elder_relative",
        "Reverse Formal": "customer_facing",
    }
    required_scenario = special_scenarios.get(target_tier)
    if required_scenario is None and (scenario is not None or address_term is not None):
        raise ValueError(
            "scenario/address_term overrides are only valid for "
            "Formal-Warm and Reverse Formal targets"
        )
    effective_scenario = scenario or required_scenario
    overrides = _address_config()["scenario_overrides"]

    if required_scenario and effective_scenario != required_scenario:
        raise ValueError(
            f"{target_tier} requires scenario={required_scenario!r}, "
            f"not {effective_scenario!r}"
        )
    if effective_scenario is not None and effective_scenario not in overrides:
        raise ValueError(f"Unknown address-term scenario: {effective_scenario}")
    if address_term is not None:
        if effective_scenario is None:
            raise ValueError("address_term requires a scenario or special-case target tier")
        allowed = overrides[effective_scenario]["terms"]
        if address_term not in allowed:
            raise ValueError(
                f"Address term {address_term!r} is not allowed for "
                f"scenario {effective_scenario!r}"
            )
    return effective_scenario, address_term


def _normalise_gender(gender: str | None, words: Sequence[str]) -> str:
    if gender is not None:
        aliases = {"m": "masculine", "male": "masculine", "f": "feminine", "female": "feminine"}
        normalised = aliases.get(gender.strip().lower(), gender.strip().lower())
        if normalised not in GENDERS:
            raise ValueError("gender must be masculine, feminine, M, F, or None")
        return normalised

    feminine_cues = ("रही", "सकती", "आती", "रहती", "आओगी", "जाओगी")
    if any(word in feminine_cues or word.endswith("ओगी") for word in words):
        return "feminine"
    return "masculine"


def _address_gender_hint(gender: str | None, words: Sequence[str]) -> str | None:
    """Return only explicit/surface-supported gender for title selection."""
    if gender is not None:
        return _normalise_gender(gender, words)
    feminine_cues = {"रही", "सकती", "आती", "रहती", "आओगी", "जाओगी"}
    masculine_cues = {"रहा", "सकता", "आता", "रहता", "आओगे", "जाओगे"}
    if any(word in feminine_cues for word in words):
        return "feminine"
    if any(word in masculine_cues for word in words):
        return "masculine"
    return None


def _coerce_tokens(sentence: str | Sequence[TaggedToken]) -> tuple[list[str], list[TaggedToken]]:
    if isinstance(sentence, str):
        tagged = tokenize_and_tag(sentence)
    else:
        tagged = list(sentence)
        if not tagged or not all(isinstance(token, TaggedToken) for token in tagged):
            raise TypeError("sentence must be non-empty text or TaggedToken sequence")
    return [token.text for token in tagged], tagged


def _replace_pronouns(words: list[str], target_tier: str) -> list[str]:
    tiers, source_forms = _pronoun_rules()
    target = tiers[target_tier]["forms"]
    replaced: list[str] = []
    source_sequences = sorted(
        ((form.split(), classification) for form, classification in source_forms.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    index = 0
    while index < len(words):
        matched_words: list[str] | None = None
        classification: tuple[str, str] | None = None
        for candidate, candidate_classification in source_sequences:
            if words[index : index + len(candidate)] == candidate:
                matched_words = candidate
                classification = candidate_classification
                break
        if classification is None:
            replaced.append(words[index])
            index += 1
            continue
        case, subtype = classification
        if case == "possessive":
            target_form = target["possessive"][subtype]
        elif case == "dative":
            target_form = target["dative"]["canonical"]
        elif case == "postpositional":
            target_form = target["postpositional"][subtype]["canonical"]
        else:
            target_form = target[case]
        replaced.extend(target_form.split())
        index += len(matched_words or ())
    return replaced


def _pattern_parts(pattern: str) -> tuple[str, str]:
    parts = pattern.split()
    return parts[-2].replace("{stem}", ""), parts[-1]


def _has_direct_addressee(words: Sequence[str], target_tier: str) -> bool:
    subject = _pronoun_rules()[0][target_tier]["forms"]["subject"]
    return subject in words


def _future_form(word: str, agreement_class: str, gender: str) -> str | None:
    known_stems = {
        "आओगे": "आ",
        "आओगी": "आ",
        "जाओगे": "जा",
        "जाओगी": "जा",
    }
    stem = known_stems.get(word)
    if stem is None:
        return None
    if agreement_class == "intimate_singular":
        return stem + ("एगा" if gender == "masculine" else "एगी")
    if agreement_class == "casual_plural":
        return stem + ("ओगे" if gender == "masculine" else "ओगी")
    return stem + ("एंगे" if gender == "masculine" else "एंगी")


def _replace_verbs(words: list[str], target_tier: str, gender: str) -> list[str]:
    target = _verb_rules()[target_tier]
    agreement_class = target["agreement_class"]
    patterns = target["gender_patterns"][gender]
    direct_addressee = _has_direct_addressee(words, target_tier)
    result = list(words)

    # Multiword agreement constructions from verb_rules.json.  Requiring a
    # direct addressee prevents changing first-person forms such as "मैं ...
    # कर सकता हूं" and noun predicates such as "खाता नंबर ... है".
    if direct_addressee:
        categories = {
            "present_progressive": {"रहा", "रही", "रहे"},
            "past_completive": {"चुका", "चुकी", "चुके"},
            "modal_potential": {"सकता", "सकती", "सकते"},
        }
        auxiliaries = {"है", "हो", "हैं"}
        first_person_auxiliaries = {"हूं", "हूँ"}
        for index in range(len(result) - 1):
            # Reverse Formal commonly contains service-worker self-reference
            # alongside an addressee form. Never apply addressee agreement to
            # a first-person clause such as "मैं ... कर सकता हूं".
            if result[index + 1] in first_person_auxiliaries:
                continue
            for category, source_agreements in categories.items():
                if result[index] in source_agreements and result[index + 1] in auxiliaries:
                    target_agreement, target_auxiliary = _pattern_parts(patterns[category])
                    result[index] = target_agreement
                    result[index + 1] = target_auxiliary
                    break

        habitual_suffixes = ("ता", "ती", "ते")
        target_suffix, target_auxiliary = _pattern_parts(patterns["present_habitual"])
        for index in range(len(result) - 1):
            if result[index + 1] in first_person_auxiliaries:
                continue
            if result[index + 1] not in auxiliaries:
                continue
            for suffix in habitual_suffixes:
                if result[index].endswith(suffix) and len(result[index]) > len(suffix):
                    result[index] = result[index][: -len(suffix)] + target_suffix
                    result[index + 1] = target_auxiliary
                    break

        for index, word in enumerate(result):
            future = _future_form(word, agreement_class, gender)
            if future is not None:
                result[index] = future

        # Predicate adjective + copula, represented in the test set by कैसी.
        for index in range(len(result) - 1):
            if result[index] in {"कैसा", "कैसी", "कैसे"} and result[index + 1] in auxiliaries:
                result[index + 1] = {
                    "intimate_singular": "है",
                    "casual_plural": "हो",
                    "respectful_plural": "हैं",
                }[agreement_class]

    for index, word in enumerate(result):
        forms = IMPERATIVE_FORMS.get(word)
        if forms is not None:
            following = result[index + 1] if index + 1 < len(result) else None
            previous = result[index - 1] if index else None
            if (
                word in AMBIGUOUS_SHORT_IMPERATIVES
                and following is not None
                and following not in PUNCTUATION
                and previous != "मत"
            ):
                continue
            result[index] = forms[agreement_class]

    # The soft imperative "आ जाना" is present in the locked corpus.
    for index in range(len(result) - 1):
        if result[index : index + 2] == ["आ", "जाना"]:
            replacement = {
                "intimate_singular": ["आ", "जाना"],
                "casual_plural": ["आ", "जाना"],
                "respectful_plural": ["आ", "जाइए"],
            }[agreement_class]
            result[index : index + 2] = replacement
    return result


def _has_imperative(words: Sequence[str]) -> bool:
    unambiguous = {
        "समझाइए", "दीजिए", "कीजिए", "बैठिए", "करें", "जाइए",
        "समझाओ", "करो", "दो", "बैठो",
    }
    if any(word in unambiguous for word in words):
        return True
    for index, word in enumerate(words):
        if word != "कर":
            continue
        previous = words[index - 1] if index else None
        following = words[index + 1] if index + 1 < len(words) else None
        if previous == "मत" or following is None or following in PUNCTUATION:
            return True
    return False


def _adjust_particles(words: list[str], target_tier: str) -> list[str]:
    if target_tier in {"Full Informal", "Respectful Casual"}:
        result = [word for word in words if word not in {"जी", "कृपया"}]
        if result and result[0] == ",":
            result.pop(0)
        return result

    result = list(words)
    if _has_imperative(result) and "कृपया" not in result:
        subject = _pronoun_rules()[0][target_tier]["forms"]["subject"]
        insertion_index = result.index(subject) + 1 if subject in result else 0
        result.insert(insertion_index, "कृपया")
    return result


def _leading_vocative(words: Sequence[str]) -> tuple[list[str], int]:
    """Return tokens before an early comma and the body start index."""
    try:
        comma = words.index(",")
    except ValueError:
        return [], 0
    candidates = list(words[:comma])
    address_lexicon = (
        KINSHIP_TERMS
        | MASCULINE_ADDRESS_TERMS
        | FEMININE_ADDRESS_TERMS
        | {"जी", "डॉक्टर"}
    )
    if 0 < comma <= 3 and any(word in address_lexicon for word in candidates):
        return candidates, comma + 1
    return [], 0


def _address_gender(vocative: Sequence[str], fallback: str | None) -> str | None:
    if any(word in FEMININE_ADDRESS_TERMS for word in vocative):
        return "feminine"
    if any(word in MASCULINE_ADDRESS_TERMS for word in vocative):
        return "masculine"
    return fallback


def _adjust_address_terms(
    words: list[str],
    target_tier: str,
    gender: str | None,
    scenario: str | None,
    address_term: str | None,
) -> list[str]:
    _address_rules()[target_tier]  # Validate tier coverage in the data table.
    vocative, body_start = _leading_vocative(words)
    body = words[body_start:] if vocative else list(words)
    address_gender = _address_gender(vocative, gender)
    kinship = next((word for word in vocative if word in KINSHIP_TERMS), None)
    is_doctor = "डॉक्टर" in vocative

    if target_tier == "Full Informal":
        if kinship:
            return [kinship, ",", *body]
        if is_doctor:
            return ["डॉक्टर", ",", *body]
        if vocative:
            # A casual masculine vocative is supported by the table; for a
            # feminine/unknown addressee, dropping a title is safer than
            # inventing a name or relationship.
            return (["भाई", ",", *body] if address_gender == "masculine" else body)
        return body

    if target_tier == "Respectful Casual":
        if kinship:
            return [kinship, ",", *body]
        if is_doctor:
            return ["डॉक्टर", "साहब", ",", *body]
        if vocative:
            term = "दीदी" if address_gender == "feminine" else "भैया"
            return [term, ",", *body]
        return body

    if target_tier == "Formal-Warm":
        if address_term:
            return [*address_term.split(), ",", *body]
        if kinship:
            return [kinship, "जी", ",", *body]
        # The elder_relative override says the true relationship must be
        # selected and never guessed from gender. Scenario alone identifies
        # the tier, not whether the person is दादी, नानी, etc.
        return body

    if target_tier == "Formal":
        if not vocative:
            return body
        if any(word in {"सर", "मैडम", "जी", "साहब"} for word in vocative):
            return [*vocative, ",", *body]
        term = (
            "मैडम" if address_gender == "feminine"
            else "सर" if address_gender == "masculine"
            else "जी"
        )
        return [term, ",", *body]

    # Full Formal and Reverse Formal require an address marker.  Preserve a
    # suitable formal vocative, otherwise select a gendered generic term when
    # known and the neutral जी when it is not.
    if target_tier == "Reverse Formal" and address_term:
        return [*address_term.split(), ",", *body]
    if vocative and any(word in {"सर", "मैडम", "जी", "साहब"} for word in vocative):
        return [*vocative, ",", *body]
    if vocative:
        term = "मैडम" if address_gender == "feminine" else "सर"
    elif gender == "feminine":
        term = "मैडम"
    elif gender == "masculine":
        term = "सर"
    else:
        term = "जी"
    return [term, ",", *body]


def _detokenize(words: Sequence[str]) -> str:
    output = ""
    for word in words:
        if not output:
            output = word
        elif word in PUNCTUATION:
            output += word
        else:
            output += " " + word
    return output


def transform(
    sentence: str | Sequence[TaggedToken],
    target_tier: str,
    gender: str | None = None,
    *,
    scenario: str | None = None,
    address_term: str | None = None,
) -> str:
    """Transform a Hindi sentence into ``target_tier``.

    ``sentence`` may be raw text or the output of ``tokenize_and_tag``.
    ``gender`` controls addressee agreement and accepts masculine/feminine,
    M/F, or ``None`` (surface-form inference).  Unsupported tokens and
    grammatical constructions are preserved unchanged. ``scenario`` and
    ``address_term`` activate validated scenario-level overrides for the two
    tag-based tiers. Formal-Warm never guesses a kinship relation.
    """
    pronoun_tiers, _ = _pronoun_rules()
    if target_tier not in pronoun_tiers:
        choices = ", ".join(pronoun_tiers)
        raise ValueError(f"Unknown target tier: {target_tier}. Expected one of: {choices}")

    scenario, address_term = _resolve_special_case_context(
        target_tier, scenario, address_term
    )

    words, _tagged = _coerce_tokens(sentence)
    address_gender = _address_gender_hint(gender, words)
    resolved_gender = _normalise_gender(gender, words)
    words = _replace_pronouns(words, target_tier)
    words = _replace_verbs(words, target_tier, resolved_gender)
    words = _adjust_particles(words, target_tier)
    words = _adjust_address_terms(
        words, target_tier, address_gender, scenario, address_term
    )
    return _detokenize(words)


__all__ = ["transform"]
