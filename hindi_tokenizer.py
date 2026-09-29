"""Hindi tokenization and transformation-focused POS tagging.

Indic NLP Library supplies the token boundaries. Stanza supplies Universal
Dependencies POS tags over those pretokenized words. A small explicit lexicon
overrides the rule-facing role for honorific/politeness particles because a
statistical POS tagger can interpret context-sensitive forms such as ``जी`` as
a verb. The original POS tag is retained for inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import stanza
from indicnlp.tokenize import indic_tokenize


STANZA_MODEL_DIR = Path(__file__).with_name(".cache") / "stanza"

TokenRole = Literal["pronoun", "verb", "honorific_particle", "other"]

# These are grammatical politeness/honorific markers that later rule tables
# may add, remove, or preserve. Lexicon membership takes precedence over the
# statistical POS tag so "जी" is not accidentally treated as a lexical verb.
HONORIFIC_PARTICLES = frozenset(
    {
        "जी",
        "साहब",
        "महोदय",
        "महोदया",
        "श्री",
        "श्रीमती",
        "सर",
        "मैडम",
        "जनाब",
        "कृपया",
    }
)


@dataclass(frozen=True, slots=True)
class TaggedToken:
    """One Hindi token with POS metadata and rule-facing category flags."""

    index: int
    text: str
    upos: str
    xpos: str | None
    features: dict[str, str]
    role: TokenRole
    is_pronoun: bool
    is_verb: bool
    is_auxiliary: bool
    is_honorific_particle: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@lru_cache(maxsize=1)
def get_hindi_pos_pipeline() -> stanza.Pipeline:
    """Load the Hindi POS pipeline once, without downloading at runtime."""
    if not STANZA_MODEL_DIR.exists():
        raise FileNotFoundError(
            "Hindi POS resources are missing. Download Stanza's Hindi "
            f"tokenize,pos models into {STANZA_MODEL_DIR}."
        )

    return stanza.Pipeline(
        lang="hi",
        processors="tokenize,pos",
        package={"tokenize": "hdtb", "pos": "hdtb_charlm"},
        dir=str(STANZA_MODEL_DIR),
        tokenize_pretokenized=True,
        download_method=None,
        use_gpu=False,
        verbose=False,
    )


def _parse_features(raw_features: str | None) -> dict[str, str]:
    if not raw_features:
        return {}
    return dict(feature.split("=", 1) for feature in raw_features.split("|"))


def _rule_role(token: str, upos: str) -> TokenRole:
    if token in HONORIFIC_PARTICLES:
        return "honorific_particle"
    if upos == "PRON":
        return "pronoun"
    if upos in {"VERB", "AUX"}:
        return "verb"
    return "other"


def tokenize_and_tag(sentence: str) -> list[TaggedToken]:
    """Tokenize one Hindi sentence and flag transformation-relevant roles.

    Pronouns are identified from the Universal POS tag ``PRON``. Both lexical
    verbs (``VERB``) and auxiliaries (``AUX``) receive the rule-facing role
    ``verb`` while ``is_auxiliary`` preserves the distinction. Honorific and
    politeness markers are identified by the explicit project lexicon.

    Tokens outside those three categories are returned unchanged with role
    ``other`` so downstream transformation can degrade gracefully.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("sentence must be a non-empty string")

    indic_tokens = indic_tokenize.trivial_tokenize(sentence, lang="hi")
    document = get_hindi_pos_pipeline()([indic_tokens])
    stanza_words = [
        word for tagged_sentence in document.sentences for word in tagged_sentence.words
    ]

    stanza_tokens = [word.text for word in stanza_words]
    if stanza_tokens != indic_tokens:
        raise RuntimeError(
            "Stanza changed the Indic NLP token boundaries: "
            f"Indic={indic_tokens!r}, Stanza={stanza_tokens!r}"
        )

    tagged_tokens: list[TaggedToken] = []
    for index, word in enumerate(stanza_words):
        is_honorific = word.text in HONORIFIC_PARTICLES
        # Rule-facing flags are mutually exclusive. Keep the raw UPOS above,
        # but do not expose a lexicon-confirmed honorific such as "जी" to a
        # downstream verb rule even if the statistical tagger labels it VERB.
        is_pronoun = word.upos == "PRON" and not is_honorific
        is_verb = word.upos in {"VERB", "AUX"} and not is_honorific
        tagged_tokens.append(
            TaggedToken(
                index=index,
                text=word.text,
                upos=word.upos,
                xpos=word.xpos,
                features=_parse_features(word.feats),
                role=_rule_role(word.text, word.upos),
                is_pronoun=is_pronoun,
                is_verb=is_verb,
                is_auxiliary=word.upos == "AUX" and not is_honorific,
                is_honorific_particle=is_honorific,
            )
        )
    return tagged_tokens
