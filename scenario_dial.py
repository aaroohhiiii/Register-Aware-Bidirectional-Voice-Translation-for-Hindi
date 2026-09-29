"""Resolve a conversation scenario to its locked register tier."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


SCENARIO_LOOKUP_PATH = Path(__file__).with_name("scenario_lookup.json")


@lru_cache(maxsize=1)
def _load_lookup() -> dict[str, Any]:
    """Load the code-independent scenario configuration once per process."""
    with SCENARIO_LOOKUP_PATH.open(encoding="utf-8") as lookup_file:
        return json.load(lookup_file)


@lru_cache(maxsize=1)
def _scenario_index() -> dict[str, dict[str, Any]]:
    """Index scenario records by their stable public key."""
    scenarios = _load_lookup()["scenarios"]
    index = {scenario["key"]: scenario for scenario in scenarios}
    if len(index) != len(scenarios):
        raise ValueError("scenario_lookup.json contains duplicate scenario keys")
    return index


def _threshold_tier(dial_value: float) -> str:
    """Return the one threshold tier containing ``dial_value``."""
    matches: list[str] = []
    for threshold in _load_lookup()["thresholds"]:
        if threshold["minimum_inclusive"]:
            meets_minimum = dial_value >= threshold["minimum"]
        else:
            meets_minimum = dial_value > threshold["minimum"]

        if threshold["maximum_inclusive"]:
            meets_maximum = dial_value <= threshold["maximum"]
        else:
            meets_maximum = dial_value < threshold["maximum"]

        if meets_minimum and meets_maximum:
            matches.append(threshold["tier"])

    if len(matches) != 1:
        raise ValueError(
            f"Dial value {dial_value!r} matched {len(matches)} threshold tiers"
        )
    return matches[0]


def scenario_to_dial(scenario_name: str) -> str:
    """Return the locked register tier for a stable scenario key.

    Ordinary scenarios are classified from the average of their Power,
    Distance, and Context scores using thresholds in ``scenario_lookup.json``.
    Formal-Warm and Reverse Formal bypass threshold math and return their
    explicitly tagged tiers.

    Args:
        scenario_name: A stable snake-case key such as ``"friend"`` or
            ``"customer_facing"``.

    Raises:
        ValueError: If the scenario key is unknown or lookup data is invalid.
    """
    entry = _scenario_index().get(scenario_name)
    if entry is None:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    if entry["assignment_mode"] == "special_tag":
        if not entry.get("special_tag"):
            raise ValueError(f"Special scenario has no tag: {scenario_name}")
        return entry["tier"]

    scores = entry["scores"]
    dial_value = (scores["power"] + scores["distance"] + scores["context"]) / 3
    tier = _threshold_tier(dial_value)
    if tier != entry["tier"]:
        raise ValueError(
            f"Computed tier {tier!r} disagrees with the locked tier "
            f"{entry['tier']!r} for {scenario_name!r}"
        )
    return tier

