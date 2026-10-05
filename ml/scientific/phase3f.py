"""Shared inference and verification utilities for the Phase 3F model stack."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score


THRESHOLDS = {"heavy": 64.5, "very_heavy": 115.6, "extreme": 204.5}
REGIME_LABELS = [
    "ACTIVE_MONSOON", "BREAK_MONSOON", "LOW_DEPRESSION", "COASTAL_RAINFALL",
    "OROGRAPHIC_RAINFALL", "WESTERN_DISTURBANCE", "OTHER",
]
REGRESSION_SYSTEMS = ("RAW_GFS", "GLOBAL_ML", "ANVAYA")
FEATURE_VERSION = "feature_registry_v1_phase3e_24_features"


def calibrate_regime_probabilities(model: Any, X: Any, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to model posterior probabilities."""
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    raw = np.clip(raw, 1e-12, 1.0)
    return softmax(np.log(raw) / float(temperature), axis=1)


def predict_moe(bundle: Mapping[str, Any], X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative soft-gated rainfall predictions and regime posterior."""
    posterior = calibrate_regime_probabilities(
        bundle["regime_model"], X, bundle["temperature"]
    )
    regime_models = bundle["experts"]
    fallback = bundle["global_fallback"]
    prediction = np.zeros(len(X), dtype=np.float64)
    for index, regime in enumerate(bundle["regime_labels"]):
        model = regime_models.get(regime, fallback)
        prediction += posterior[:, index] * np.asarray(model.predict(X), dtype=np.float64)
    return np.maximum(prediction, 0.0), posterior


def predict_probability_bundle(
    bundle: Mapping[str, Any], X: Any, posterior: np.ndarray
) -> np.ndarray:
    """Blend supported regime heads with global fallbacks, then calibrate."""
    result = np.zeros(len(X), dtype=np.float64)
    for index, regime in enumerate(bundle["regime_labels"]):
        model = bundle["experts"].get(regime, bundle["global_model"])
        result += posterior[:, index] * np.asarray(model.predict_proba(X))[:, 1]
    result = np.clip(result, 1e-8, 1.0 - 1e-8)
    logits = np.log(result / (1.0 - result)).reshape(-1, 1)
    calibrated = bundle["calibrator"].predict_proba(logits)[:, 1]
    return np.clip(calibrated, 0.0, 1.0)


def enforce_probability_order(probabilities: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Project independent calibrated heads onto the required nested ordering."""
    heavy = np.clip(np.asarray(probabilities["heavy"], dtype=float), 0.0, 1.0)
    very_heavy = np.minimum(
        np.clip(np.asarray(probabilities["very_heavy"], dtype=float), 0.0, 1.0), heavy
    )
    extreme = np.minimum(
        np.clip(np.asarray(probabilities["extreme"], dtype=float), 0.0, 1.0), very_heavy
    )
    return {"heavy": heavy, "very_heavy": very_heavy, "extreme": extreme}


def predict_all_probabilities(
    stack: Mapping[str, Any], X: Any, posterior: np.ndarray
) -> dict[str, np.ndarray]:
    """Run the saved multi-threshold stack and enforce nested probabilities."""
    raw = {
        name: predict_probability_bundle(bundle, X, posterior)
        for name, bundle in stack["heads"].items()
    }
    return enforce_probability_order(raw)


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    error = p - y
    correlation = None
    if y.size > 1 and np.std(y) > 0 and np.std(p) > 0:
        correlation = float(np.corrcoef(y, p)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_bias": float(np.mean(error)),
        "pearson_r": correlation,
    }


def contingency_metrics(observed_event: Sequence[bool], forecast_event: Sequence[bool]) -> dict[str, float | int | None]:
    observed = np.asarray(observed_event, dtype=bool)
    forecast = np.asarray(forecast_event, dtype=bool)
    hits = int(np.count_nonzero(observed & forecast))
    misses = int(np.count_nonzero(observed & ~forecast))
    false_alarms = int(np.count_nonzero(~observed & forecast))
    correct_negatives = int(np.count_nonzero(~observed & ~forecast))
    total = hits + misses + false_alarms + correct_negatives
    random_hits = ((hits + misses) * (hits + false_alarms) / total) if total else 0.0
    csi_den = hits + misses + false_alarms
    ets_den = hits + misses + false_alarms - random_hits
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "observed_events": hits + misses,
        "forecast_events": hits + false_alarms,
        "csi": float(hits / csi_den) if csi_den else None,
        "ets": float((hits - random_hits) / ets_den) if ets_den else None,
        "pod": float(hits / (hits + misses)) if hits + misses else None,
        "far": float(false_alarms / (hits + false_alarms)) if hits + false_alarms else None,
    }


def probability_metrics(observed_event: Sequence[bool], probability: Sequence[float]) -> dict[str, float | None]:
    y = np.asarray(observed_event, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1 - 1e-8)
    auc = None
    if np.unique(y).size == 2:
        auc = float(roc_auc_score(y, p))
    return {
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": auc,
        "probability_csi_at_0p5": contingency_metrics(y, p >= 0.5)["csi"],
    }


def spatial_fss(
    frame: pd.DataFrame,
    forecast_column: str,
    threshold: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    window_cells: int = 5,
) -> float | None:
    """Compute edge/mask-aware neighborhood FSS over complete target-date grids.

    For each target date, neighborhood fractions use only valid RF25 grid
    cells. Numerator and denominator are accumulated over dates, preserving
    the standard ratio-of-sums FSS definition. The odd window is centered.
    """
    if window_cells < 1 or window_cells % 2 != 1:
        raise ValueError("FSS neighborhood width must be a positive odd number")
    kernel = np.ones((window_cells, window_cells), dtype=np.float64)
    numerator = 0.0
    denominator = 0.0
    for _, group in frame.groupby("target_date", sort=False):
        forecast = np.full((len(latitudes), len(longitudes)), np.nan, dtype=np.float32)
        observed = np.full_like(forecast, np.nan)
        lat_idx = np.searchsorted(latitudes, group["latitude"].to_numpy())
        lon_idx = np.searchsorted(longitudes, group["longitude"].to_numpy())
        forecast[lat_idx, lon_idx] = group[forecast_column].to_numpy(dtype=np.float32)
        observed[lat_idx, lon_idx] = group["observed_rain_mm"].to_numpy(dtype=np.float32)
        valid = np.isfinite(observed) & np.isfinite(forecast)
        mask = valid.astype(np.float64)
        valid_count = convolve(mask, kernel, mode="constant", cval=0.0)
        obs_event = (observed >= threshold).astype(np.float64) * mask
        fcst_event = (forecast >= threshold).astype(np.float64) * mask
        obs_fraction = np.divide(
            convolve(obs_event, kernel, mode="constant", cval=0.0), valid_count,
            out=np.zeros_like(valid_count), where=valid_count > 0,
        )
        fcst_fraction = np.divide(
            convolve(fcst_event, kernel, mode="constant", cval=0.0), valid_count,
            out=np.zeros_like(valid_count), where=valid_count > 0,
        )
        centers = valid_count > 0
        numerator += float(np.sum((fcst_fraction[centers] - obs_fraction[centers]) ** 2))
        denominator += float(np.sum(fcst_fraction[centers] ** 2 + obs_fraction[centers] ** 2))
    if denominator == 0:
        return None
    return float(1.0 - numerator / denominator)


def probability_order_is_valid(probabilities: Mapping[str, Sequence[float]], atol: float = 1e-12) -> bool:
    heavy = np.asarray(probabilities["heavy"])
    very_heavy = np.asarray(probabilities["very_heavy"])
    extreme = np.asarray(probabilities["extreme"])
    return bool(
        np.all(extreme <= very_heavy + atol)
        and np.all(very_heavy <= heavy + atol)
        and all(np.all((p >= -atol) & (p <= 1 + atol)) for p in (heavy, very_heavy, extreme))
    )
