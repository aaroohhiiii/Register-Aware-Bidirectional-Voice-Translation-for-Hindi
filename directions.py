"""directions.py — Direction and scenario mapping for Hindi register conversion.

Public API
----------
direction_to_target(direction: str, scenario: Optional[str] = None) -> str
    Map a direction ("A" | "B") and scenario name to a target register ("formal" | "informal").
"""

from __future__ import annotations

from typing import Optional

import config


def direction_to_target(direction: str, scenario: Optional[str] = None) -> str:
    """Map direction ("A" | "B") and scenario to target register.

    Parameters
    ----------
    direction : str
        "A" (User -> Interlocutor) or "B" (Interlocutor -> User), case-insensitive.
    scenario : str | None, optional
        Scenario key in config.SCENARIOS. Defaults to config.DEFAULT_SCENARIO.

    Returns
    -------
    str
        Target register string (e.g. "formal" or "informal").

    Raises
    ------
    ValueError
        If direction is not "A" or "B", or if scenario is not recognized.
    """
    if not isinstance(direction, str):
        raise ValueError(f"direction must be a string 'A' or 'B', got: {direction!r}")

    dir_clean = direction.strip().upper()
    if dir_clean not in ("A", "B"):
        raise ValueError(f"Invalid direction: {direction!r}. Must be 'A' or 'B'.")

    scenarios = getattr(config, "SCENARIOS", {})
    default_scen = getattr(config, "DEFAULT_SCENARIO", "rickshaw")
    scen_key = (scenario or default_scen).strip().lower()

    if scen_key not in scenarios:
        allowed = list(scenarios.keys())
        raise ValueError(f"Unknown scenario: {scenario!r}. Allowed scenarios: {allowed}")

    scen_data = scenarios[scen_key]
    target_key = "A_target" if dir_clean == "A" else "B_target"
    return scen_data[target_key]


__all__ = ["direction_to_target"]
