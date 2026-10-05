"""Transparent, configurable ANVAYA operational warning decision rule."""
from __future__ import annotations

import math
from typing import Any


UNAVAILABLE = "UNAVAILABLE"


def evaluate_warning(
    probabilities: dict[str, Any] | None,
    policy: dict[str, Any],
    *,
    coverage_fraction: float | None = None,
) -> dict[str, Any]:
    """Evaluate configured probability cutoffs; this is not an official IMD warning."""
    minimum_coverage = float(policy.get("minimum_coverage_fraction", 0.0))
    if probabilities is None:
        return {
            "level": UNAVAILABLE,
            "label": policy.get("label", "ANVAYA OPERATIONAL DECISION RULE"),
            "reasons": [],
            "coverage_fraction": coverage_fraction,
            "disclaimer": policy.get("disclaimer"),
        }
    if coverage_fraction is not None and coverage_fraction < minimum_coverage:
        return {
            "level": UNAVAILABLE,
            "label": policy.get("label", "ANVAYA OPERATIONAL DECISION RULE"),
            "reasons": [],
            "coverage_fraction": coverage_fraction,
            "minimum_coverage_fraction": minimum_coverage,
            "disclaimer": policy.get("disclaimer"),
        }
    values: dict[str, float] = {}
    for name in ("heavy_probability", "very_heavy_probability", "extreme_probability"):
        value = probabilities.get(name)
        if value is None:
            return {
                "level": UNAVAILABLE,
                "label": policy.get("label", "ANVAYA OPERATIONAL DECISION RULE"),
                "reasons": [],
                "coverage_fraction": coverage_fraction,
                "disclaimer": policy.get("disclaimer"),
            }
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return {
                "level": UNAVAILABLE,
                "label": policy.get("label", "ANVAYA OPERATIONAL DECISION RULE"),
                "reasons": [],
                "coverage_fraction": coverage_fraction,
                "disclaimer": policy.get("disclaimer"),
            }
        values[name] = numeric

    for level in policy.get("levels", []):
        name = str(level.get("name", "")).upper()
        matched = []
        for criterion in level.get("any", []):
            field = criterion.get("probability")
            threshold = float(criterion.get("threshold", 2.0))
            if field in values and values[field] >= threshold:
                matched.append({"probability": field, "value": values[field], "threshold": threshold})
        if matched:
            return {
                "level": name,
                "label": policy.get("label", "ANVAYA OPERATIONAL DECISION RULE"),
                "reasons": matched,
                "coverage_fraction": coverage_fraction,
                "disclaimer": policy.get("disclaimer"),
            }
    return {
        "level": str(policy.get("default_level", "GREEN")).upper(),
        "label": policy.get("label", "ANVAYA OPERATIONAL DECISION RULE"),
        "reasons": [],
        "coverage_fraction": coverage_fraction,
        "disclaimer": policy.get("disclaimer"),
    }
