"""Build valid-time-matched NWP consensus without hiding missing sources."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from ml.operational.common import LIVE, UNAVAILABLE


def _provider_data(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result or result.get("status") != LIVE:
        return None
    data = result.get("data")
    return data if isinstance(data, dict) else None


def build_consensus(
    gfs: dict[str, Any] | None,
    gefs: dict[str, Any] | None,
    ecmwf: dict[str, Any] | None,
) -> dict[str, Any]:
    gfs_data = _provider_data(gfs)
    gefs_data = _provider_data(gefs)
    ecmwf_data = _provider_data(ecmwf)
    target = (gfs_data or {}).get("valid_time") or (gefs_data or {}).get("valid_time")
    if target is None and ecmwf_data:
        models = ecmwf_data.get("models") or {}
        target = next(
            (item.get("data", {}).get("valid_time") for item in models.values() if item.get("data")),
            None,
        )
    if target is None:
        return {
            "status": UNAVAILABLE,
            "reason": "No live provider supplied a forecast valid time",
            "independent_model_families": 0,
        }

    candidates: dict[str, list[float]] = {}
    family_bounds: dict[str, list[tuple[str, str]]] = {}
    excluded: dict[str, str] = {}

    def parse_time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("time metadata must include a timezone")
        return parsed.astimezone(timezone.utc)

    def add(family: str, source: str, data: dict[str, Any] | None, value_key: str) -> None:
        if not data:
            return
        if data.get("valid_time") != target:
            excluded[source] = "valid time does not match consensus target"
            return
        try:
            start_text, end_text = data.get("accumulation_start"), data.get("accumulation_end")
            if not start_text or not end_text:
                raise ValueError("accumulation window metadata missing")
            start_time, end_time = parse_time(start_text), parse_time(end_text)
            if end_time != parse_time(target):
                raise ValueError("accumulation end does not equal the forecast valid time")
            if end_time - start_time != timedelta(hours=24):
                raise ValueError("accumulation is not exactly 24 hours")
            value = float(data[value_key])
            if not math.isfinite(value) or value < 0:
                raise ValueError("negative precipitation total")
            candidates.setdefault(family, []).append(value)
            family_bounds.setdefault(family, []).append((start_text, end_text))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            excluded[source] = str(exc)

    if gfs_data:
        add("NOAA", "GFS", gfs_data, "gfs_daily_rain_mm")
    if gefs_data:
        add("NOAA", "GEFS", gefs_data, "precipitation_24h_mean_mm")
    ecmwf_models = (ecmwf_data or {}).get("models") or {}
    for name, item in ecmwf_models.items():
        model_data = item.get("data") if isinstance(item, dict) and item.get("status") == LIVE else None
        if model_data:
            add("ECMWF", name, model_data, "precipitation_24h_mm")

    canonical_bounds: dict[str, tuple[str, str]] = {}
    for family, bounds in family_bounds.items():
        if len(set(bounds)) == 1:
            canonical_bounds[family] = bounds[0]
        else:
            candidates.pop(family, None)
            excluded[family] = "providers in this model family used different accumulation windows"
    all_bounds = set(canonical_bounds.values())
    if len(all_bounds) > 1:
        # The between-model mean is meaningful only when every contributing
        # source spans the same physical period.
        # Keep one consistent window group if a provider returned incompatible
        # durations; deterministic order is NOAA before ECMWF.
        selected_family = next((family for family in ("NOAA", "ECMWF") if family in candidates), None)
        selected_bounds = canonical_bounds.get(selected_family) if selected_family else None
        for family in list(candidates):
            if canonical_bounds.get(family) != selected_bounds:
                candidates.pop(family, None)
                excluded[family] = "not included because its accumulation window differs"
        all_bounds = {selected_bounds} if selected_bounds else set()

    family_values = {family: float(mean(values)) for family, values in candidates.items() if values}
    family_count = len(family_values)
    rain_values = list(family_values.values())
    if family_count < 2:
        return {
            "status": UNAVAILABLE,
            "reason": "A consensus requires at least two independent live model families",
            "valid_time": target,
            "independent_model_families": family_count,
            "single_family_estimate_mm": rain_values[0] if rain_values else None,
            "family_estimates_mm": family_values,
            "excluded_sources": excluded,
        }
    return {
        "status": LIVE,
        "valid_time": target,
        "accumulation_start": next(iter(all_bounds))[0] if all_bounds else None,
        "accumulation_end": next(iter(all_bounds))[1] if all_bounds else None,
        "precipitation_24h_mean_mm": float(mean(rain_values)),
        "precipitation_24h_spread_mm": float(pstdev(rain_values)),
        "family_estimates_mm": family_values,
        "independent_model_families": family_count,
        "excluded_sources": excluded,
        "method": "equal-weight mean of independent model-family means; GEFS members and ECMWF model variants do not get extra family weight",
    }
