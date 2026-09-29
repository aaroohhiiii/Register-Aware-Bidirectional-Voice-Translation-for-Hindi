"""Generate the locked Step 5.2 cross-tier review matrix.

Run ``python cross_tier_review.py`` to print all 144 reviewed conversions.
The target schedule deliberately uses three *other* tiers for every source.
"""

from __future__ import annotations

import json
from pathlib import Path

from register_transformer import transform


PROJECT_DIR = Path(__file__).resolve().parent
TARGET_SCHEDULE = {
    "Full Informal": ("Respectful Casual", "Formal", "Full Formal"),
    "Respectful Casual": ("Full Informal", "Formal", "Reverse Formal"),
    "Formal": ("Full Informal", "Respectful Casual", "Full Formal"),
    "Full Formal": ("Full Informal", "Respectful Casual", "Formal"),
    "Formal-Warm": ("Full Informal", "Formal", "Full Formal"),
    "Reverse Formal": ("Full Informal", "Respectful Casual", "Formal"),
}


def build_review_cases() -> list[dict]:
    with (PROJECT_DIR / "register_test_sentences.json").open(encoding="utf-8") as source:
        corpus = json.load(source)

    cases: list[dict] = []
    for record in corpus["sentences"]:
        addressee_gender = (
            record["gender"]["value"]
            if record["gender"]["applies_to"] == "addressee"
            else None
        )
        for target_tier in TARGET_SCHEDULE[record["tier"]]:
            cases.append(
                {
                    "sentence_id": record["id"],
                    "source_tier": record["tier"],
                    "target_tier": target_tier,
                    "gender": addressee_gender,
                    "source": record["sentence"],
                    "output": transform(
                        record["sentence"], target_tier, addressee_gender
                    ),
                }
            )
    return cases


if __name__ == "__main__":
    print(json.dumps(build_review_cases(), ensure_ascii=False, indent=2))

