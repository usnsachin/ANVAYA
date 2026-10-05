"""Read-only endpoint for the frozen Phase 3F held-out 2024 benchmark."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter


router = APIRouter(tags=["verification"])
ROOT = Path(__file__).resolve().parents[3]
TEST_METRICS = ROOT / "data/results/test_metrics.csv"
THRESHOLD_METRICS = ROOT / "data/results/threshold_metrics.csv"
REGIME_METRICS = ROOT / "data/results/regime_metrics.csv"
SYSTEMS = {
    "RAW_GFS": "raw_gfs",
    "GLOBAL_ML": "global_ml",
    "ANVAYA": "anvaya",
}
PROBABILITY_SYSTEM = "ANVAYA_CALIBRATED_PROBABILITY"


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "metrics": {},
        "period": "Frozen Phase 3F held-out 2024 JJAS test partition",
        "metadata": {
            "mode": "SCIENTIFIC_MODE",
            "source": "PHASE3F_FROZEN",
            "benchmark_frozen": True,
            "sample_size": None,
            "target_dates": None,
            "reason": reason,
        },
        "comparison": {name: {} for name in SYSTEMS.values()},
        "thresholds": {},
        "regime_stratified": {},
        "probability_scores": {},
        "interpretation": "Frozen benchmark files are unavailable; no demo, inferred, or synthetic metric values are substituted.",
    }


def _build_verification() -> dict[str, Any]:
    if not TEST_METRICS.is_file() or not THRESHOLD_METRICS.is_file():
        return _unavailable("One or more frozen Phase 3F metric files are missing")
    try:
        deterministic_rows = _read_csv(TEST_METRICS)
        threshold_rows = _read_csv(THRESHOLD_METRICS)
    except (OSError, csv.Error) as exc:
        return _unavailable(f"Unable to read frozen Phase 3F metrics: {type(exc).__name__}")

    try:
        regime_rows = _read_csv(REGIME_METRICS) if REGIME_METRICS.is_file() else []
    except (OSError, csv.Error):
        regime_rows = []

    deterministic: dict[str, dict[str, Any]] = {}
    for row in deterministic_rows:
        key = row.get("system", "").upper()
        if key not in SYSTEMS:
            continue
        deterministic[SYSTEMS[key]] = {
            "rmse": _number(row.get("rmse")),
            "mae": _number(row.get("mae")),
            "bias": _number(row.get("mean_bias")),
            "pearson_r": _number(row.get("pearson_r")),
        }
    if set(deterministic) != set(SYSTEMS.values()):
        return _unavailable("Frozen deterministic metric file does not contain all three Phase 3F systems")

    thresholds: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in SYSTEMS.values()}
    probability_scores: dict[str, dict[str, Any]] = {}
    for row in threshold_rows:
        system = row.get("system", "").upper()
        threshold = row.get("threshold", "").lower()
        if system in SYSTEMS:
            system_name = SYSTEMS[system]
            thresholds[system_name][threshold] = {
                "threshold_mm": _number(row.get("threshold_mm")),
                "hits": int(row.get("hits") or 0),
                "misses": int(row.get("misses") or 0),
                "false_alarms": int(row.get("false_alarms") or 0),
                "correct_negatives": int(row.get("correct_negatives") or 0),
                "observed_events": int(row.get("observed_events") or 0),
                "forecast_events": int(row.get("forecast_events") or 0),
                "csi": _number(row.get("csi")),
                "ets": _number(row.get("ets")),
                "pod": _number(row.get("pod")),
                "far": _number(row.get("far")),
                "fss_5x5": _number(row.get("fss_5x5")),
                "fss_window_degrees_approx": _number(row.get("fss_window_degrees_approx")),
            }
        elif system == PROBABILITY_SYSTEM:
            probability_scores[threshold] = {
                "brier_score": _number(row.get("brier_score")),
                "log_loss": _number(row.get("log_loss")),
                "roc_auc": _number(row.get("roc_auc")),
                "probability_csi_at_0p5": _number(row.get("probability_csi_at_0p5")),
            }

    for name in thresholds:
        for kind in ("heavy", "very_heavy", "extreme"):
            if kind not in thresholds[name]:
                return _unavailable(f"Frozen threshold metrics are missing {kind} for {name}")
            deterministic[name][kind] = thresholds[name][kind]

    # Expose the frozen heavy-threshold regime table without recomputation or
    # mutation. RMSE is repeated across threshold rows; use the heavy row once.
    regime_stratified: dict[str, dict[str, Any]] = {}
    for row in regime_rows:
        if row.get("threshold", "").lower() != "heavy":
            continue
        regime = row.get("regime", "").strip()
        system = SYSTEMS.get(row.get("system", "").upper())
        if not regime or not system:
            continue
        entry = regime_stratified.setdefault(regime, {
            "sample_count": int(row.get("sample_count") or 0),
            "threshold_mm": _number(row.get("threshold_mm")),
            "systems": {},
        })
        entry["systems"][system] = {
            "rmse": _number(row.get("rmse")),
            "csi": _number(row.get("csi")),
            "ets": _number(row.get("ets")),
            "pod": _number(row.get("pod")),
            "far": _number(row.get("far")),
            "fss_5x5": _number(row.get("fss_5x5")),
            "observed_event_count": int(row.get("observed_event_count") or 0),
        }
    regime_stratified = {
        regime: entry for regime, entry in regime_stratified.items()
        if set(entry["systems"]) == set(SYSTEMS.values())
    }

    anvaya = deterministic["anvaya"]
    global_ml = deterministic["global_ml"]
    interpretation = (
        "This is the frozen 2024 scientific benchmark, separate from live operational feeds. "
        "Global ML has lower RMSE than ANVAYA; ANVAYA has slightly lower MAE and absolute bias. "
        "No live provider result changes or augments these historical metrics."
    )
    return {
        "status": "REFERENCE",
        "metrics": anvaya,
        "period": "Held-out 2024 JJAS test partition (Phase 3F; 2024-06-01 through 2024-09-30)",
        "metadata": {
            "mode": "SCIENTIFIC_MODE",
            "source": "PHASE3F_FROZEN",
            "model_version": "phase3f_frozen",
            "readiness": "SCIENCE_READY_YES",
            "benchmark_frozen": True,
            "sample_size": 604876,
            "target_dates": 122,
            "splits_modified": False,
            "metrics_files": [
                "data/results/test_metrics.csv",
                "data/results/threshold_metrics.csv",
                "data/results/regime_metrics.csv",
            ],
            "regime_metrics_available": bool(regime_stratified),
        },
        "comparison": deterministic,
        "thresholds": thresholds,
        "regime_stratified": regime_stratified,
        "probability_scores": probability_scores,
        "interpretation": interpretation,
        "comparison_summary": {
            "lower_rmse": "global_ml" if (global_ml["rmse"] or float("inf")) < (anvaya["rmse"] or float("inf")) else "anvaya",
            "lower_mae": "global_ml" if (global_ml["mae"] or float("inf")) < (anvaya["mae"] or float("inf")) else "anvaya",
            "anvaya_beats_global_ml_overall": False,
        },
    }


@router.get("/verification")
async def get_verification():
    """Return only frozen Phase 3F results; never synthesize fallback scores."""
    return _build_verification()
