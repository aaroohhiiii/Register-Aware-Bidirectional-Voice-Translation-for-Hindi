"""Evaluate register detection and the first detection→transform smoke test."""

from __future__ import annotations

import json
from pathlib import Path

from register_detector import AAP_FORMS, TU_FORMS, TUM_FORMS, detect_register_details
from register_transformer import transform


PROJECT_DIR = Path(__file__).resolve().parent
TARGET_BY_DETECTED_TIER = {
    "Full Informal": "Formal",
    "Respectful Casual": "Full Formal",
    "Formal": "Full Informal",
    "Full Formal": "Respectful Casual",
    "Formal-Warm": "Formal",
    "Reverse Formal": "Respectful Casual",
}


def _corpus() -> list[dict]:
    with (PROJECT_DIR / "register_test_sentences.json").open(encoding="utf-8") as source:
        return json.load(source)["sentences"]


def evaluate_detection() -> dict:
    rows = []
    correct = 0
    tier_totals: dict[str, dict[str, int]] = {}
    confusion_pairs: dict[str, int] = {}
    for record in _corpus():
        result = detect_register_details(record["sentence"])
        is_correct = result.tier == record["tier"]
        correct += is_correct
        if not is_correct:
            pair = f"{record['tier']} -> {result.tier}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        totals = tier_totals.setdefault(record["tier"], {"correct": 0, "total": 0})
        totals["correct"] += is_correct
        totals["total"] += 1
        rows.append(
            {
                "sentence_id": record["id"],
                "sentence": record["sentence"],
                "expected": record["tier"],
                "detected": result.tier,
                "correct": is_correct,
                "confidence": result.confidence,
                "evidence": list(result.evidence),
            }
        )
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy_percent": round(100 * correct / len(rows), 2),
        "per_tier": tier_totals,
        "confusion_pairs": confusion_pairs,
        "rows": rows,
    }


def _smoke_assessment(output: str, target_tier: str) -> tuple[bool, str]:
    words = {token.strip(",.?!।") for token in output.split()}
    if not output.strip() or output.startswith(","):
        return False, "empty output or orphan leading punctuation"
    if target_tier == "Full Informal" and words & (TUM_FORMS | AAP_FORMS):
        return False, "source pronoun survived a Full Informal conversion"
    if target_tier == "Respectful Casual" and words & (TU_FORMS | AAP_FORMS):
        return False, "source pronoun survived a Respectful Casual conversion"
    if target_tier in {"Formal", "Full Formal", "Formal-Warm", "Reverse Formal"} and words & (TU_FORMS | TUM_FORMS):
        return False, "source pronoun survived an आप-tier conversion"
    if target_tier in {"Full Informal", "Respectful Casual"} and words & {"जी", "कृपया"}:
        return False, "low-register output retained a formal particle"
    return True, "pronoun/particle invariants satisfied"


def run_round_trip_smoke_test() -> dict:
    rows = []
    passed = 0
    for record in _corpus():
        detected = detect_register_details(record["sentence"])
        target = TARGET_BY_DETECTED_TIER[detected.tier]
        addressee_gender = (
            record["gender"]["value"]
            if record["gender"]["applies_to"] == "addressee"
            else None
        )
        output = transform(record["sentence"], target, addressee_gender)
        sensible, note = _smoke_assessment(output, target)
        passed += sensible
        rows.append(
            {
                "sentence_id": record["id"],
                "expected_source_tier": record["tier"],
                "detected_source_tier": detected.tier,
                "target_tier": target,
                "input": record["sentence"],
                "output": output,
                "sensible": sensible,
                "note": note,
            }
        )
    return {
        "total": len(rows),
        "passed": passed,
        "pass_percent": round(100 * passed / len(rows), 2),
        "rows": rows,
    }


if __name__ == "__main__":
    report = {
        "detection_baseline": evaluate_detection(),
        "round_trip_smoke_test": run_round_trip_smoke_test(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
