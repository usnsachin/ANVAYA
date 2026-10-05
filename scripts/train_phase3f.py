"""Train and evaluate the authorized Phase 3F scientific model stack.

The script fits on train_verified, uses validation_verified for early stopping
and probability calibration, freezes all model artifacts, and only then opens
test_verified for one final evaluation. It never reads the legacy test.parquet.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import ctypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss

from ml.features.engineer import FEATURE_SCHEMA
from ml.regime.labels import REGIME_LABELS
from ml.scientific.phase3f import (
    FEATURE_VERSION,
    REGRESSION_SYSTEMS,
    THRESHOLDS,
    contingency_metrics,
    enforce_probability_order,
    predict_moe,
    predict_all_probabilities,
    predict_probability_bundle,
    probability_metrics,
    probability_order_is_valid,
    regression_metrics,
    spatial_fss,
)


TARGET = "observed_rain_mm"
TRAIN_PATH = ROOT / "data/processed/train_verified.parquet"
VALIDATION_PATH = ROOT / "data/processed/validation_verified.parquet"
TEST_PATH = ROOT / "data/processed/test_verified.parquet"
ARTIFACT_DIR = ROOT / "models/scientific/phase3f"
RESULT_DIR = ROOT / "data/results"
SEED = 42
N_JOBS = min(8, os.cpu_count() or 1)
REGRESSION_SUPPORT_MIN_ROWS = 10_000
REGRESSION_SUPPORT_MIN_DATES = 30
PROBABILITY_SUPPORT_MIN_POSITIVES = 50
PROBABILITY_SUPPORT_MIN_NEGATIVES = 100
PROBABILITY_SUPPORT_MIN_POSITIVE_DATES = 20

# Fixed before test access. Validation is used for early stopping and calibration.
MODEL_CONFIG = {
    "regression": {
        "objective": "regression_l2", "n_estimators": 500,
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
        "min_child_samples": 100, "max_bin": 127, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 1, "reg_lambda": 0.1,
    },
    "classification": {
        "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": -1, "min_child_samples": 100, "max_bin": 127,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
        "reg_lambda": 0.1,
    },
    "early_stopping_rounds": 40,
    "seeds": {"random_state": SEED, "bagging_seed": SEED, "feature_fraction_seed": SEED,
              "data_random_seed": SEED, "drop_seed": SEED},
    "threads": N_JOBS,
    "deterministic": True,
    "force_col_wise": True,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_memory_gib() -> tuple[float, float]:
    """Return physical RAM available/total in GiB on Windows."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Unable to inspect available system memory")
    return status.ullAvailPhys / 2**30, status.ullTotalPhys / 2**30


def write_json(path: Path, payload: Any) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        if pd.isna(value):
            return None
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_feature_registry() -> dict[str, list[str]]:
    registry_path = ROOT / "config/feature_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    allowed = {
        item["feature_name"] for item in registry["features"]
        if item.get("allowed_for_live_model") is True
        and item.get("forecast_available") is True
        and item.get("target_related") is False
    }
    unexpected = sorted(set(FEATURE_SCHEMA) - allowed)
    if unexpected:
        raise RuntimeError(f"Feature schema includes unapproved/unavailable predictors: {unexpected}")
    forbidden = {TARGET, "regime", "heavy_rain", "very_heavy_rain", "extreme_rain",
                 "gfs_apcp_interval_mm", "gfs_rain_mm", "rain_last_6h"}
    overlap = sorted(set(FEATURE_SCHEMA) & forbidden)
    if overlap:
        raise RuntimeError(f"Leakage/retired fields appear in FEATURE_SCHEMA: {overlap}")
    categories = {
        "raw_nwp": FEATURE_SCHEMA[:10],
        "antecedent_rainfall": ["rain_last_24h"],
        "geographic_topographic": [
            "latitude", "longitude", "coast_distance_approx", "terrain_elevation_proxy",
            "orographic_flow_proxy",
        ],
        "temporal": ["lead_time_h", "month", "day_of_year"],
        "derived_wind_thermodynamic": [
            name for name in FEATURE_SCHEMA if name not in set(FEATURE_SCHEMA[:10])
            and name not in {"rain_last_24h", "latitude", "longitude", "coast_distance_approx",
                             "terrain_elevation_proxy", "orographic_flow_proxy", "lead_time_h",
                             "month", "day_of_year"}
        ],
        "regime": ["regime (classifier target only; excluded from predictor matrix)"],
    }
    flattened = [name for group in categories.values() for name in group
                 if not name.startswith("regime (")]
    if sorted(flattened) != sorted(FEATURE_SCHEMA):
        raise RuntimeError("Feature category inventory does not partition FEATURE_SCHEMA")
    return categories


def audit_temporal_firewall(path: Path, expected_split: str, expected_years: set[int]) -> dict[str, Any]:
    """Check every row's split, valid-time arithmetic, and antecedent timing in batches."""
    parquet = pq.ParquetFile(path)
    needed = ["target_date", "init_time", "valid_time", "lead_hours", "split",
              "antecedent_target_date", "antecedent_valid_time"]
    absent = sorted(set(needed) - set(parquet.schema_arrow.names))
    if absent:
        raise RuntimeError(f"{path.name} lacks temporal audit fields: {absent}")
    date_min, date_max = None, None
    rows = 0
    for batch in parquet.iter_batches(columns=needed, batch_size=50_000):
        frame = batch.to_pandas()
        if not frame["split"].eq(expected_split).all():
            raise RuntimeError(f"Unexpected split labels in {path.name}")
        targets = pd.to_datetime(frame["target_date"], utc=True, errors="raise")
        init = pd.to_datetime(frame["init_time"], utc=True, errors="raise")
        valid = pd.to_datetime(frame["valid_time"], utc=True, errors="raise")
        antecedent_target = pd.to_datetime(frame["antecedent_target_date"], utc=True, errors="raise")
        antecedent_valid = pd.to_datetime(frame["antecedent_valid_time"], utc=True, errors="raise")
        expected_valid = init + pd.to_timedelta(frame["lead_hours"].astype("int64"), unit="h")
        if not valid.equals(expected_valid):
            raise RuntimeError(f"valid_time != init_time + lead_hours in {path.name}")
        if not (antecedent_valid < init).all() or not (antecedent_target < targets).all():
            raise RuntimeError(f"Antecedent rainfall is not strictly before forecast initialization in {path.name}")
        years = set(targets.dt.year.unique().tolist())
        if not years.issubset(expected_years):
            raise RuntimeError(f"Unexpected target years in {path.name}: {sorted(years - expected_years)}")
        low, high = targets.min().date().isoformat(), targets.max().date().isoformat()
        date_min = low if date_min is None or low < date_min else date_min
        date_max = high if date_max is None or high > date_max else date_max
        rows += len(frame)
    return {"rows_audited": rows, "target_date_min": date_min, "target_date_max": date_max,
            "split": expected_split, "antecedent_strictly_before_init": True,
            "valid_time_equals_init_plus_lead": True}


def load_model_split(path: Path, split_name: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    columns = list(dict.fromkeys(list(FEATURE_SCHEMA) + [TARGET, "regime", "target_date", "latitude", "longitude"]))
    frame = pd.read_parquet(path, columns=columns)
    if frame[columns].isna().any().any():
        bad = frame[columns].isna().sum()
        raise RuntimeError(f"Missing values in model columns for {split_name}: {bad[bad > 0].to_dict()}")
    if not frame["regime"].isin(REGIME_LABELS).all():
        unknown = sorted(set(frame["regime"]) - set(REGIME_LABELS))
        raise RuntimeError(f"Unexpected regime labels in {split_name}: {unknown}")
    unknown_regimes = sorted(set(REGIME_LABELS) - set(frame["regime"]))
    if unknown_regimes:
        raise RuntimeError(f"Regimes missing from {split_name}: {unknown_regimes}")
    if (frame[TARGET].to_numpy(dtype=np.float32) < 0).any():
        raise RuntimeError(f"Negative rainfall target in {split_name}")
    X = np.ascontiguousarray(frame[FEATURE_SCHEMA].to_numpy(dtype=np.float32, copy=True))
    y = frame[TARGET].to_numpy(dtype=np.float32, copy=True)
    regimes = frame["regime"].to_numpy(dtype=str, copy=True)
    dates = frame["target_date"].to_numpy(dtype=str, copy=True)
    latitudes = frame["latitude"].to_numpy(dtype=np.float32, copy=True)
    longitudes = frame["longitude"].to_numpy(dtype=np.float32, copy=True)
    meta = {
        "rows": int(len(frame)),
        "dates": int(frame["target_date"].nunique()),
        "target_date_min": str(frame["target_date"].min()),
        "target_date_max": str(frame["target_date"].max()),
        "latitude_unique": int(frame["latitude"].nunique()),
        "longitude_unique": int(frame["longitude"].nunique()),
        "regime_counts": {str(k): int(v) for k, v in frame["regime"].value_counts().items()},
    }
    if np.isinf(X).any() or not np.isfinite(y).all():
        raise RuntimeError(f"Non-finite model values in {split_name}")
    del frame
    return X, y, regimes, dates, latitudes, longitudes, meta


def base_lgb_params(kind: str, objective: str, n_estimators: int) -> dict[str, Any]:
    common = {
        "objective": objective, "n_estimators": n_estimators,
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
        "min_child_samples": 100, "max_bin": 127, "colsample_bytree": 0.9,
        "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 0.1,
        "random_state": SEED, "bagging_seed": SEED, "feature_fraction_seed": SEED,
        "data_random_seed": SEED, "drop_seed": SEED, "n_jobs": N_JOBS,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
    }
    if kind == "regression":
        common["metric"] = "rmse"
    return common


def callbacks() -> list[Any]:
    return [lgb.early_stopping(MODEL_CONFIG["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=100)]


def save_model_smoke(name: str, model: Any, X: np.ndarray) -> Path:
    path = ARTIFACT_DIR / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    loaded = joblib.load(path)
    prediction = loaded.predict(X[:min(128, len(X))])
    if not np.isfinite(np.asarray(prediction, dtype=float)).all():
        raise RuntimeError(f"Model smoke prediction failed for {name}")
    print(f"Saved and smoke-tested {path.relative_to(ROOT)} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return path


def fit_regime_classifier(
    X_train: np.ndarray, regimes_train: np.ndarray,
    X_val: np.ndarray, regimes_val: np.ndarray,
) -> tuple[Any, float, np.ndarray, dict[str, Any], pd.DataFrame]:
    train_labels = np.asarray([REGIME_LABELS.index(value) for value in regimes_train], dtype=np.int8)
    val_labels = np.asarray([REGIME_LABELS.index(value) for value in regimes_val], dtype=np.int8)
    model = LGBMClassifier(
        **base_lgb_params("classification", "multiclass", MODEL_CONFIG["classification"]["n_estimators"]),
        num_class=len(REGIME_LABELS),
    )
    model.fit(X_train, train_labels, eval_set=[(X_val, val_labels)], eval_metric="multi_logloss",
              callbacks=callbacks())
    path = save_model_smoke("regime_classifier", model, X_val)
    raw = np.clip(model.predict_proba(X_val), 1e-12, 1.0)

    def objective(log_temperature: float) -> float:
        scaled = softmax(np.log(raw) / np.exp(log_temperature), axis=1)
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(val_labels)), val_labels], 1e-12, 1.0))))

    raw_nll = objective(0.0)
    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-5})
    temperature = float(np.exp(result.x))
    calibrated = softmax(np.log(raw) / temperature, axis=1)
    calibrated_nll = objective(result.x)
    brier = float(np.mean(np.sum((calibrated - np.eye(len(REGIME_LABELS))[val_labels]) ** 2, axis=1)))
    matrix = confusion_matrix(val_labels, np.argmax(calibrated, axis=1), labels=np.arange(len(REGIME_LABELS)))
    confusion = pd.DataFrame(matrix, index=REGIME_LABELS, columns=REGIME_LABELS)
    confusion.index.name = "observed_regime"
    confusion.columns.name = "predicted_regime"
    info = {"model_path": str(path.relative_to(ROOT)), "temperature": temperature,
            "validation_nll_before": raw_nll, "validation_nll_after": calibrated_nll,
            "validation_multiclass_brier": brier, "validation_rows": len(val_labels),
            "calibration_fit": "single scalar temperature, optimized on validation only"}
    return model, temperature, calibrated, info, confusion


def fit_regressor(
    name: str, X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    model = LGBMRegressor(**base_lgb_params("regression", "regression", MODEL_CONFIG["regression"]["n_estimators"]))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=callbacks())
    save_model_smoke(name, model, X_val)
    return model, {"best_iteration": int(model.best_iteration_ or MODEL_CONFIG["regression"]["n_estimators"]),
                   "train_rows": int(len(y_train)), "validation_rows": int(len(y_val))}


def fit_binary_classifier(
    name: str, X_train: np.ndarray, labels_train: np.ndarray,
    X_val: np.ndarray | None, labels_val: np.ndarray | None,
) -> tuple[Any, dict[str, Any]]:
    model = LGBMClassifier(
        **base_lgb_params("classification", "binary", MODEL_CONFIG["classification"]["n_estimators"])
    )
    use_early_stopping = (X_val is not None and labels_val is not None
                          and len(labels_val) > 0 and np.unique(labels_val).size == 2)
    if use_early_stopping:
        model.fit(X_train, labels_train, eval_set=[(X_val, labels_val)], eval_metric="binary_logloss",
                  callbacks=callbacks())
    else:
        model.fit(X_train, labels_train)
    smoke_x = X_val if X_val is not None and len(X_val) else X_train
    save_model_smoke(name, model, smoke_x)
    return model, {"best_iteration": int(model.best_iteration_ or MODEL_CONFIG["classification"]["n_estimators"]),
                   "train_rows": int(len(labels_train)), "validation_rows": int(len(labels_val)) if labels_val is not None else 0,
                   "early_stopping_used": bool(use_early_stopping)}


def append_importance(rows: list[dict[str, Any]], name: str, model: Any) -> None:
    booster = model.booster_
    for feature, gain, split in zip(FEATURE_SCHEMA,
                                    booster.feature_importance(importance_type="gain"),
                                    booster.feature_importance(importance_type="split")):
        rows.append({"model": name, "feature": feature, "gain": float(gain), "split": int(split)})


def validation_blend(bundle: dict[str, Any], X: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    values = np.zeros(len(X), dtype=np.float64)
    for index, regime in enumerate(REGIME_LABELS):
        model = bundle["experts"].get(regime, bundle["global_model"])
        values += posterior[:, index] * model.predict_proba(X)[:, 1]
    return np.clip(values, 1e-8, 1 - 1e-8)


def fit_probability_stack(
    X_train: np.ndarray, y_train: np.ndarray, regimes_train: np.ndarray, dates_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, regimes_val: np.ndarray, posterior_val: np.ndarray,
    importance_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    fitted: dict[str, dict[str, Any]] = {}
    details: dict[str, Any] = {}
    for name, threshold in THRESHOLDS.items():
        print(f"Training {name} >= {threshold} mm probability stack", flush=True)
        target_train = (y_train >= threshold).astype(np.int8)
        target_val = (y_val >= threshold).astype(np.int8)
        global_model, global_info = fit_binary_classifier(
            f"global_{name}_classifier", X_train, target_train, X_val, target_val
        )
        append_importance(importance_rows, f"global_{name}", global_model)
        regime_models: dict[str, Any] = {}
        support: dict[str, Any] = {}
        for regime in REGIME_LABELS:
            train_mask = regimes_train == regime
            val_mask = regimes_val == regime
            labels = target_train[train_mask]
            positives = int(labels.sum())
            negatives = int(len(labels) - positives)
            positive_dates = int(np.unique(dates_train[train_mask & (target_train == 1)]).size)
            eligible = (positives >= PROBABILITY_SUPPORT_MIN_POSITIVES
                        and negatives >= PROBABILITY_SUPPORT_MIN_NEGATIVES
                        and positive_dates >= PROBABILITY_SUPPORT_MIN_POSITIVE_DATES)
            support[regime] = {
                "train_rows": int(train_mask.sum()), "positive_rows": positives,
                "negative_rows": negatives, "positive_dates": positive_dates,
                "dedicated_head": bool(eligible),
                "fallback": None if eligible else "global_threshold_classifier",
            }
            if not eligible:
                continue
            model_name = f"{name}_expert_{regime.lower()}_classifier"
            model, info = fit_binary_classifier(
                model_name, X_train[train_mask], labels,
                X_val[val_mask], target_val[val_mask],
            )
            regime_models[regime] = model
            support[regime]["best_iteration"] = info["best_iteration"]
            append_importance(importance_rows, model_name, model)

        bundle: dict[str, Any] = {
            "threshold_name": name, "threshold_mm": threshold,
            "regime_labels": list(REGIME_LABELS), "global_model": global_model,
            "experts": regime_models, "support": support,
            "calibrator": None,
        }
        uncalibrated = validation_blend(bundle, X_val, posterior_val)
        logits = np.log(uncalibrated / (1 - uncalibrated)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
        calibrator.fit(logits, target_val)
        bundle["calibrator"] = calibrator
        fitted[name] = bundle
        artifact_path = ARTIFACT_DIR / f"{name}_probability.joblib"
        joblib.dump(bundle, artifact_path)
        loaded = joblib.load(artifact_path)
        smoke_p = validation_blend(loaded, X_val[:128], posterior_val[:128])
        if not np.isfinite(smoke_p).all():
            raise RuntimeError(f"Probability bundle reload smoke test failed for {name}")
        details[name] = {
            "threshold_mm": threshold, "artifact": str(artifact_path.relative_to(ROOT)),
            "global_classifier": global_info, "dedicated_regime_heads": sorted(regime_models),
            "support": support,
            "calibration": "Platt sigmoid on validation-only soft-gated probabilities",
            "validation_events": int(target_val.sum()),
        }
        print(f"Saved {artifact_path.relative_to(ROOT)}; dedicated heads={len(regime_models)}", flush=True)
    return fitted, details


def fss_by_system(frame: pd.DataFrame, latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for system in REGRESSION_SYSTEMS:
        column = {"RAW_GFS": "raw_gfs_rain_mm", "GLOBAL_ML": "global_ml_rain_mm",
                  "ANVAYA": "anvaya_rain_mm"}[system]
        result[system] = {
            name: spatial_fss(frame, column, threshold, latitudes, longitudes, window_cells=5)
            for name, threshold in THRESHOLDS.items()
        }
    return result


def score_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    system_columns = {"RAW_GFS": "raw_gfs_rain_mm", "GLOBAL_ML": "global_ml_rain_mm",
                      "ANVAYA": "anvaya_rain_mm"}
    latitudes = np.sort(frame["latitude"].unique())
    longitudes = np.sort(frame["longitude"].unique())
    fss = fss_by_system(frame, latitudes, longitudes)
    overall_rows, threshold_rows, regime_rows = [], [], []
    for system, column in system_columns.items():
        regression = regression_metrics(frame[TARGET], frame[column])
        overall_rows.append({"system": system, **regression})
        for threshold_name, threshold in THRESHOLDS.items():
            categorical = contingency_metrics(frame[TARGET] >= threshold, frame[column] >= threshold)
            threshold_rows.append({
                "system": system, "threshold": threshold_name, "threshold_mm": threshold,
                **categorical, "fss_5x5": fss[system][threshold_name],
                "fss_window_cells": 5, "fss_window_degrees_approx": 1.25,
            })
    for regime in REGIME_LABELS:
        regime_frame = frame[frame["regime"] == regime]
        for system, column in system_columns.items():
            regression = regression_metrics(regime_frame[TARGET], regime_frame[column])
            for threshold_name, threshold in THRESHOLDS.items():
                categorical = contingency_metrics(regime_frame[TARGET] >= threshold,
                                                   regime_frame[column] >= threshold)
                regime_rows.append({
                    "regime": regime, "system": system, "threshold": threshold_name,
                    "threshold_mm": threshold, "sample_count": len(regime_frame),
                    "observed_event_count": int((regime_frame[TARGET] >= threshold).sum()),
                    **regression, **categorical,
                })

    # Add calibrated ANVAYA event-probability diagnostics without conflating
    # them with deterministic thresholded-rainfall contingency scores.
    y = frame[TARGET].to_numpy()
    prob_columns = {"heavy": "p_heavy", "very_heavy": "p_very_heavy", "extreme": "p_extreme"}
    for name, threshold in THRESHOLDS.items():
        target = y >= threshold
        p = frame[prob_columns[name]].to_numpy()
        threshold_rows.append({
            "system": "ANVAYA_CALIBRATED_PROBABILITY", "threshold": name,
            "threshold_mm": threshold, **contingency_metrics(target, p >= 0.5),
            "fss_5x5": None, "fss_window_cells": None,
            **probability_metrics(target, p),
        })
    overall = pd.DataFrame(overall_rows)
    thresholds = pd.DataFrame(threshold_rows)
    regimes = pd.DataFrame(regime_rows)
    extremes = {}
    for name, threshold in THRESHOLDS.items():
        y_event = y >= threshold
        extremes[name] = {"threshold_mm": threshold, "observed_events": int(y_event.sum()),
                          "observed_unique_dates": int(frame.loc[y_event, "target_date"].nunique()),
                          "systems": {system: {
                              "forecast_events": int((frame[column] >= threshold).sum()),
                              "hits": contingency_metrics(y_event, frame[column] >= threshold)["hits"],
                              "detected_observed_fraction": contingency_metrics(y_event, frame[column] >= threshold)["pod"],
                          } for system, column in system_columns.items()}}
    return overall, thresholds, regimes, {"fss_5x5": fss, "extreme_events": extremes}


def main() -> None:
    for path in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    categories = audit_feature_registry()
    print(f"Phase 3F features: {len(FEATURE_SCHEMA)} approved predictors; target excluded", flush=True)
    available_gib, total_gib = available_memory_gib()
    print(f"Physical RAM available={available_gib:.2f}/{total_gib:.2f} GiB; arrays use float32; LightGBM threads={N_JOBS}", flush=True)
    if available_gib < 2.0:
        raise MemoryError("Less than 2 GiB physical RAM is available; stopping before split loading")

    # Full-row-count temporal firewall before any fit.
    train_audit = audit_temporal_firewall(TRAIN_PATH, "train", {2020, 2021, 2022})
    validation_audit = audit_temporal_firewall(VALIDATION_PATH, "validation", {2023})
    train_hash = sha256_file(TRAIN_PATH)
    validation_hash = sha256_file(VALIDATION_PATH)
    X_train, y_train, regimes_train, dates_train, _, _, train_meta = load_model_split(TRAIN_PATH, "train")
    X_val, y_val, regimes_val, dates_val, _, _, validation_meta = load_model_split(VALIDATION_PATH, "validation")
    print(f"Loaded train {X_train.shape} and validation {X_val.shape}; test remains unopened", flush=True)

    # Fit regime posterior and temperature calibration on validation only.
    regime_model, temperature, posterior_val, regime_calibration, confusion = fit_regime_classifier(
        X_train, regimes_train, X_val, regimes_val
    )
    confusion.to_csv(RESULT_DIR / "confusion_matrix_regime.csv")
    write_json(RESULT_DIR / "regime_probability_calibration.json", regime_calibration)
    importance_rows: list[dict[str, Any]] = []
    append_importance(importance_rows, "regime_classifier", regime_model)

    # Global rainfall correction receives approved features only, never regime labels.
    print("Training global rainfall correction", flush=True)
    global_regressor, global_fit_info = fit_regressor(
        "global_correction", X_train, y_train, X_val, y_val
    )
    append_importance(importance_rows, "global_correction", global_regressor)

    # Each expert is fitted only on train rows carrying that weak label.
    expert_models: dict[str, Any] = {}
    expert_support: dict[str, Any] = {}
    for regime in REGIME_LABELS:
        train_mask = regimes_train == regime
        val_mask = regimes_val == regime
        rows = int(train_mask.sum())
        unique_dates = int(np.unique(dates_train[train_mask]).size)
        supported = rows >= REGRESSION_SUPPORT_MIN_ROWS and unique_dates >= REGRESSION_SUPPORT_MIN_DATES
        expert_support[regime] = {"train_rows": rows, "train_unique_dates": unique_dates,
                                  "dedicated_expert": bool(supported),
                                  "fallback": None if supported else "global_correction"}
        if not supported:
            continue
        slug = regime.lower()
        print(f"Training rainfall expert {regime}: rows={rows}, dates={unique_dates}", flush=True)
        model, fit_info = fit_regressor(
            f"expert_{slug}", X_train[train_mask], y_train[train_mask],
            X_val[val_mask], y_val[val_mask],
        )
        expert_models[regime] = model
        expert_support[regime].update(fit_info)
        append_importance(importance_rows, f"expert_{slug}", model)

    moe_bundle = {
        "regime_model": regime_model, "temperature": temperature,
        "regime_labels": list(REGIME_LABELS), "experts": expert_models,
        "global_fallback": global_regressor, "support": expert_support,
        "feature_names": list(FEATURE_SCHEMA), "feature_version": FEATURE_VERSION,
    }
    moe_path = ARTIFACT_DIR / "anvaya_moe.joblib"
    joblib.dump(moe_bundle, moe_path)
    reloaded_moe = joblib.load(moe_path)
    smoke_prediction, smoke_posterior = predict_moe(reloaded_moe, X_val[:128])
    if not np.isfinite(smoke_prediction).all() or not np.allclose(smoke_posterior.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("ANVAYA MoE reload smoke test failed")
    print(f"Saved and smoke-tested {moe_path.relative_to(ROOT)}", flush=True)

    # Calibrated regime-aware multi-threshold classifier heads.
    probability_bundles, probability_details = fit_probability_stack(
        X_train, y_train, regimes_train, dates_train,
        X_val, y_val, regimes_val, posterior_val, importance_rows,
    )
    val_probability_raw: dict[str, np.ndarray] = {}
    val_probability_ordered: dict[str, np.ndarray] = {}
    val_calibration_scores: dict[str, Any] = {}
    for name, bundle in probability_bundles.items():
        val_probability_raw[name] = predict_probability_bundle(bundle, X_val, posterior_val)
        target = (y_val >= THRESHOLDS[name]).astype(np.int8)
        val_calibration_scores[name] = probability_metrics(target, val_probability_raw[name])
    val_probability_ordered = enforce_probability_order(val_probability_raw)
    if not probability_order_is_valid(val_probability_ordered):
        raise RuntimeError("Validation probability order enforcement failed")
    for name, values in val_probability_ordered.items():
        probability_bundles[name]["ordered_with"] = "heavy >= very_heavy >= extreme"
        probability_bundles[name]["calibrator_fit_rows"] = len(y_val)
        probability_bundles[name]["calibration_diagnostics_validation_in_sample"] = val_calibration_scores[name]
        joblib.dump(probability_bundles[name], ARTIFACT_DIR / f"{name}_probability.joblib")
        reloaded = joblib.load(ARTIFACT_DIR / f"{name}_probability.joblib")
        if not np.isfinite(predict_probability_bundle(reloaded, X_val[:128], posterior_val[:128])).all():
            raise RuntimeError(f"Calibrated probability reload smoke test failed: {name}")
    probability_stack = {
        "heads": probability_bundles,
        "thresholds_mm": dict(THRESHOLDS),
        "order": "p_extreme <= p_very_heavy <= p_heavy",
        "feature_names": list(FEATURE_SCHEMA),
    }
    stack_path = ARTIFACT_DIR / "multi_threshold_probabilities.joblib"
    joblib.dump(probability_stack, stack_path)
    probability_stack = joblib.load(stack_path)
    if not probability_order_is_valid(predict_all_probabilities(
        probability_stack, X_val[:128], posterior_val[:128]
    )):
        raise RuntimeError("Reloaded multi-threshold probability stack failed monotonicity smoke test")
    write_json(RESULT_DIR / "probability_calibration.json", {
        "method": "Platt sigmoid calibration on the validation split, applied to soft-gated per-regime probabilities",
        "validation_rows": len(y_val), "diagnostics_are_in_sample": True,
        "heads": val_calibration_scores,
        "monotonic_projection": "p_extreme=min(p_extreme,p_very_heavy); p_very_heavy=min(p_very_heavy,p_heavy)",
    })

    pd.DataFrame(importance_rows).to_csv(RESULT_DIR / "feature_importance.csv", index=False)
    # Validation comparison/calibration is now complete. Write a freeze marker
    # and artifact hashes before the first access to any test row.
    frozen_paths = sorted(p for p in ARTIFACT_DIR.glob("*.joblib") if p.is_file())
    frozen = {
        "phase": "3F", "frozen_at_utc": utc_now(), "test_accessed": False,
        "train_sha256": train_hash, "validation_sha256": validation_hash,
        "train_rows": len(y_train), "validation_rows": len(y_val),
        "feature_names": list(FEATURE_SCHEMA), "feature_version": FEATURE_VERSION,
        "model_config": MODEL_CONFIG,
        "frozen_models": {str(p.relative_to(ROOT)): sha256_file(p) for p in frozen_paths},
        "calibration_split": "validation_verified.parquet",
        "test_path": str(TEST_PATH.relative_to(ROOT)),
    }
    write_json(RESULT_DIR / "phase3f_frozen_models.json", frozen)
    print("All model fitting, early stopping, and calibration are frozen; opening held-out 2024 test now.", flush=True)

    # Sole test access: final predictions and metrics only, no fit/calibration/tuning below.
    test_audit = audit_temporal_firewall(TEST_PATH, "test", {2024})
    X_test, y_test, regimes_test, dates_test, test_latitude, test_longitude, test_meta = load_model_split(TEST_PATH, "test")
    if set(dates_train) & set(dates_test) or set(dates_val) & set(dates_test):
        raise RuntimeError("Temporal split overlap detected at final test evaluation")
    anvaya_prediction, posterior_test = predict_moe(moe_bundle, X_test)
    global_prediction = np.maximum(global_regressor.predict(X_test), 0.0)
    probabilities = predict_all_probabilities(probability_stack, X_test, posterior_test)
    if not probability_order_is_valid(probabilities):
        raise RuntimeError("Final probability ordering check failed")
    if not np.isfinite(posterior_test).all() or not np.allclose(posterior_test.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Final regime probabilities are not finite normalized posteriors")

    prediction_frame = pd.DataFrame({
        "target_date": dates_test,
        "latitude": test_latitude,
        "longitude": test_longitude,
        TARGET: y_test,
        "regime": regimes_test,
        "raw_gfs_rain_mm": X_test[:, FEATURE_SCHEMA.index("gfs_daily_rain_mm")],
        "global_ml_rain_mm": global_prediction,
        "anvaya_rain_mm": anvaya_prediction,
        "p_heavy": probabilities["heavy"],
        "p_very_heavy": probabilities["very_heavy"],
        "p_extreme": probabilities["extreme"],
    })
    regime_cols = [f"p_regime_{name.lower()}" for name in REGIME_LABELS]
    for index, column in enumerate(regime_cols):
        prediction_frame[column] = posterior_test[:, index]
    predictions_path = RESULT_DIR / "test_predictions.parquet"
    prediction_frame.to_parquet(predictions_path, index=False, compression="zstd")
    overall, threshold_metrics, regime_metrics, event_report = score_test(prediction_frame)
    overall.to_csv(RESULT_DIR / "test_metrics.csv", index=False)
    threshold_metrics.to_csv(RESULT_DIR / "threshold_metrics.csv", index=False)
    regime_metrics.to_csv(RESULT_DIR / "regime_metrics.csv", index=False)

    # Probability confusion matrix diagnoses weak-label reconstruction, not physical truth.
    true_regime_index = np.asarray([REGIME_LABELS.index(value) for value in regimes_test])
    predicted_regime_index = np.argmax(posterior_test, axis=1)
    test_confusion = confusion_matrix(true_regime_index, predicted_regime_index,
                                      labels=np.arange(len(REGIME_LABELS)))
    pd.DataFrame(test_confusion, index=REGIME_LABELS, columns=REGIME_LABELS).to_csv(
        RESULT_DIR / "test_confusion_matrix_regime.csv", index_label="observed_regime"
    )

    test_hash = sha256_file(TEST_PATH)
    frozen["test_accessed"] = True
    frozen["test_opened_after_freeze"] = True
    frozen["test_sha256"] = test_hash
    frozen["test_rows"] = int(len(y_test))
    frozen["test_date_min"] = test_meta["target_date_min"]
    frozen["test_date_max"] = test_meta["target_date_max"]
    write_json(RESULT_DIR / "phase3f_frozen_models.json", frozen)

    artifact_manifest = {
        "phase": "ANVAYA Phase 3F", "created_at_utc": utc_now(),
        "dataset": {
            "train": {**train_meta, "path": str(TRAIN_PATH.relative_to(ROOT)), "sha256": train_hash},
            "validation": {**validation_meta, "path": str(VALIDATION_PATH.relative_to(ROOT)), "sha256": validation_hash},
            "test": {**test_meta, "path": str(TEST_PATH.relative_to(ROOT)), "sha256": test_hash},
            "test_opened_after_models_frozen": True,
            "legacy_test_parquet_used": False,
        },
        "feature_version": FEATURE_VERSION, "feature_registry": str((ROOT / "config/feature_registry.yaml").relative_to(ROOT)),
        "feature_registry_sha256": sha256_file(ROOT / "config/feature_registry.yaml"),
        "features": list(FEATURE_SCHEMA), "feature_categories": categories,
        "regime_labels": list(REGIME_LABELS), "target": TARGET,
        "thresholds_mm": THRESHOLDS, "random_seed": SEED,
        "software": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "pyarrow": __import__("pyarrow").__version__},
        "model_config": MODEL_CONFIG,
        "training_audits": {"train": train_audit, "validation": validation_audit, "test": test_audit},
        "regime_classifier_calibration": regime_calibration,
        "regression_fit": {"global": global_fit_info, "regime_experts": expert_support},
        "probability_models": probability_details,
        "artifact_sha256": {str(p.relative_to(ROOT)): sha256_file(p)
                             for p in sorted(ARTIFACT_DIR.glob("*.joblib"))},
        "validation_probability_diagnostics_in_sample": val_calibration_scores,
    }
    write_json(RESULT_DIR / "experiment_manifest.json", artifact_manifest)

    report = {
        "phase": "ANVAYA Phase 3F", "status": "COMPLETE_TEST_EVALUATED",
        "test_rows": int(len(y_test)), "test_dates": int(test_meta["dates"]),
        "test_period": [test_meta["target_date_min"], test_meta["target_date_max"]],
        "headline_regression": overall.to_dict(orient="records"),
        "headline_heavy_threshold": threshold_metrics[
            (threshold_metrics["threshold"] == "heavy")
            & (threshold_metrics["system"].isin(list(REGRESSION_SYSTEMS)))
        ].to_dict(orient="records"),
        "threshold_skill": threshold_metrics.to_dict(orient="records"),
        "regime_stratified_metrics": regime_metrics.to_dict(orient="records"),
        "event_analysis": event_report,
        "regime_support": expert_support,
        "regime_classifier_calibration": regime_calibration,
        "probability_calibration_validation_in_sample": val_calibration_scores,
        "feature_categories": categories,
        "data_limitations": [
            "Regime labels are heuristic weak-supervision labels; classifier skill measures label reconstruction, not official synoptic diagnosis.",
            "Grid-cell samples are spatially correlated; 122 test dates do not make 604,876 independent observations.",
            "Calibration is fit on one calendar-year validation partition; reported validation calibration scores are in-sample diagnostics.",
            "FSS is a 5x5 grid-cell (approximately 1.25 degree) neighborhood score with valid-cell mask normalization.",
            "This test covers 2024 JJAS only; the separate legacy 2025 test artifact was not read or modified.",
        ],
        "artifacts": {
            "models": str(ARTIFACT_DIR.relative_to(ROOT)),
            "predictions": str(predictions_path.relative_to(ROOT)),
            "test_metrics": str((RESULT_DIR / "test_metrics.csv").relative_to(ROOT)),
            "threshold_metrics": str((RESULT_DIR / "threshold_metrics.csv").relative_to(ROOT)),
            "regime_metrics": str((RESULT_DIR / "regime_metrics.csv").relative_to(ROOT)),
            "manifest": str((RESULT_DIR / "experiment_manifest.json").relative_to(ROOT)),
        },
    }
    write_json(RESULT_DIR / "model_training_report.json", report)
    print("Final test regression metrics:", flush=True)
    print(overall.to_string(index=False), flush=True)
    print("Final test threshold metrics (deterministic rain fields):", flush=True)
    print(threshold_metrics[threshold_metrics["system"].isin(list(REGRESSION_SYSTEMS))].to_string(index=False), flush=True)
    print(f"Phase 3F complete. Test predictions: {predictions_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
