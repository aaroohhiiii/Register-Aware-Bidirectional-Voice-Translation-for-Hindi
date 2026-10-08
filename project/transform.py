"""Transformation and Register Detection wrapper module.

Imports and exposes Aarohi's validated transformation and detection functions,
and scenario-to-tier mappings with consistent signatures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is on sys.path so original modules and rule JSONs resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from register_detector import (
    DetectionResult,
    detect_register,
    detect_register_details,
    TIERS,
)
from register_transformer import transform
from scenario_dial import scenario_to_dial


def transform_text(
    sentence: str,
    target_tier: str,
    gender: str | None = None,
    *,
    scenario: str | None = None,
    address_term: str | None = None,
) -> str:
    """Transform Hindi text to a target register tier.

    Inputs:
        sentence (str): Source Hindi sentence.
        target_tier (str): Target register tier ("Full Informal", "Respectful Casual",
                          "Formal", "Full Formal", "Formal-Warm", "Reverse Formal").
        gender (str | None): Addressee gender hint ("masculine", "feminine", or None).
        scenario (str | None): Optional scenario key (e.g. "elder_relative", "customer_facing").
        address_term (str | None): Real kinship/address term (e.g. "दादी जी", "सर").

    Output:
        str: Transformed Hindi sentence.
    """
    return transform(
        sentence,
        target_tier,
        gender=gender,
        scenario=scenario,
        address_term=address_term,
    )


def infer_register(sentence: str) -> str:
    """Infer the register tier of an input Hindi sentence.

    Inputs:
        sentence (str): Source Hindi sentence.

    Output:
        str: Detected tier ("Full Informal", "Respectful Casual", "Formal",
             "Full Formal", "Formal-Warm", "Reverse Formal").
    """
    return detect_register(sentence)


def infer_register_with_details(sentence: str) -> dict[str, Any]:
    """Infer the register tier and return confidence and diagnostic cues.

    Inputs:
        sentence (str): Source Hindi sentence.

    Output:
        dict: {"tier": str, "confidence": float, "scores": dict, "evidence": tuple}
    """
    return detect_register_details(sentence).to_dict()


def resolve_scenario_tier(scenario_key: str) -> str:
    """Map a scenario key to its authoritative register tier.

    Inputs:
        scenario_key (str): Scenario ID (e.g., "doctor_consultation", "street_vendor").

    Output:
        str: Authoritative register tier.
    """
    return scenario_to_dial(scenario_key)


__all__ = [
    "DetectionResult",
    "TIERS",
    "detect_register",
    "detect_register_details",
    "infer_register",
    "infer_register_with_details",
    "resolve_scenario_tier",
    "scenario_to_dial",
    "transform",
    "transform_text",
]
