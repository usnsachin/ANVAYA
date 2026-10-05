"""Guarded scientific training and held-out evaluation for ANVAYA.

The command refuses to train when the readiness gate is NO unless the caller
explicitly selects ``--allow-insufficient-fallback``. That mode trains a
documented global fallback for unsupported regimes and labels the resulting
artifacts PROVISIONAL_UNVERIFIED_LOCAL_DATA when the source manifest is not
externally authenticated.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features.engineer import FEATURE_SCHEMA
from ml.regime.labels import REGIME_LABELS


THRESHOLDS = {"heavy": 64.5, "very_heavy": 115.6, "extreme": 204.5}
READY_THRESHOLD = 100


def dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(str(path.stat().st_mtime_ns).encode())
    return digest.hexdigest()


def _features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_SCHEMA if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing registered features: {missing}")
    return df[FEATURE_SCHEMA].astype(float)


def _weighted_samples(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    total = len(y)
    classes = len(counts)
    weights = {key: total / (classes * count) for key, count in counts.items()}
    return y.map(weights).to_numpy(dtype=float)


def train_regime_classifier(train_df: pd.DataFrame):
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(_features(train_df), train_df["regime"])
    return model


def train_regressor(df: pd.DataFrame):
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(_features(df), np.log1p(df["observed_rain_mm"].astype(float)))
    return model


def train_probability_head(train_df: pd.DataFrame, validation_df: pd.DataFrame, threshold: float):
    y_train = (train_df["observed_rain_mm"] >= threshold).astype(int)
    y_validation = (validation_df["observed_rain_mm"] >= threshold).astype(int)
    if y_train.nunique() < 2 or y_validation.nunique() < 2:
        raise ValueError(f"Threshold {threshold} lacks both classes in train or validation")
    base = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    base.fit(_features(train_df), y_train, sample_weight=_weighted_samples(y_train))
    method = "isotonic" if int(y_validation.sum()) >= 100 else "sigmoid"
    calibrated = CalibratedClassifierCV(base, method=method, cv="prefit")
    calibrated.fit(_features(validation_df), y_validation)
    return calibrated, method, int(y_train.sum()), int(y_validation.sum())


def predict_probability_heads(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    output = {}
    x = _features(df)
    for name, model in models.items():
        output[name] = model.predict_proba(x)[:, 1]
    probabilities = pd.DataFrame(output, index=df.index).clip(0.0, 1.0)
    # Enforce the physical nesting constraint after calibration.
    probabilities["very_heavy"] = np.minimum(probabilities["very_heavy"], probabilities["heavy"])
    probabilities["extreme"] = np.minimum(probabilities["extreme"], probabilities["very_heavy"])
    return probabilities


def _predict_regime_probabilities(model, df: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_proba(_features(df))
    classes = list(model.classes_)
    output = pd.DataFrame(0.0, index=df.index, columns=REGIME_LABELS)
    for index, label in enumerate(classes):
        if label in output.columns:
            output[label] = raw[:, index]
    return output.div(output.sum(axis=1).replace(0.0, 1.0), axis=0)


def _predict_regressor(model, df: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, np.expm1(model.predict(_features(df))))


def soft_moe_predictions(df: pd.DataFrame, regime_probabilities: pd.DataFrame,
                         global_model, experts: dict) -> np.ndarray:
    outputs = {regime: _predict_regressor(expert, df) for regime, expert in experts.items()}
    global_output = _predict_regressor(global_model, df)
    mixed = np.zeros(len(df), dtype=float)
    for regime in REGIME_LABELS:
        mixed += regime_probabilities[regime].to_numpy() * outputs.get(regime, global_output)
    return mixed


def _safe_ratio(numerator: float, denominator: float):
    return None if denominator == 0 else float(numerator / denominator)


def event_metrics(obs: np.ndarray, forecast: np.ndarray, threshold: float) -> dict:
    obs_event = obs >= threshold
    forecast_event = forecast >= threshold
    hits = int(np.sum(obs_event & forecast_event))
    misses = int(np.sum(obs_event & ~forecast_event))
    false_alarms = int(np.sum(~obs_event & forecast_event))
    correct_negatives = int(np.sum(~obs_event & ~forecast_event))
    total = hits + misses + false_alarms + correct_negatives
    random_hits = ((hits + misses) * (hits + false_alarms) / total) if total else 0.0
    ets_denominator = hits + misses + false_alarms - random_hits
    return {
        "threshold_mm": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "csi": _safe_ratio(hits, hits + misses + false_alarms),
        "pod": _safe_ratio(hits, hits + misses),
        "far": _safe_ratio(false_alarms, hits + false_alarms),
        "ets": _safe_ratio(hits - random_hits, ets_denominator),
    }


def continuous_metrics(obs: np.ndarray, forecast: np.ndarray) -> dict:
    error = forecast - obs
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def evaluate_systems(test_df: pd.DataFrame, predictions: dict) -> dict:
    obs = test_df["observed_rain_mm"].to_numpy(dtype=float)
    result = {}
    for system, forecast in predictions.items():
        system_result = continuous_metrics(obs, forecast)
        system_result["events"] = {
            name: event_metrics(obs, forecast, threshold)
            for name, threshold in THRESHOLDS.items()
        }
        result[system] = system_result

    regime_result = {}
    for regime in REGIME_LABELS:
        subset = test_df["regime"] == regime
        if not subset.any():
            regime_result[regime] = {"rows": 0}
            continue
        regime_result[regime] = {
            "rows": int(subset.sum()),
            "unique_dates": int(pd.to_datetime(test_df.loc[subset, "valid_time"], utc=True).dt.date.nunique()),
            "unique_cycles": int(test_df.loc[subset, "init_time"].nunique()),
            "systems": {
                system: {
                    **continuous_metrics(obs[subset], forecast[subset]),
                    "heavy": event_metrics(obs[subset], forecast[subset], THRESHOLDS["heavy"]),
                }
                for system, forecast in predictions.items()
            },
        }
    return {"overall": result, "regime_stratified": regime_result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate guarded ANVAYA scientific models.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/scientific")
    parser.add_argument("--allow-insufficient-fallback", action="store_true",
                        help="Permit explicit global fallback for regimes below the readiness threshold.")
    parser.add_argument("--provisional", action="store_true",
                        help="Mark artifacts provisional when local source provenance is unverified.")
    args = parser.parse_args()

    processed = ROOT / args.processed_dir
    paths = [processed / f"{split}.parquet" for split in ("train", "validation", "test")]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("All train.parquet, validation.parquet, and test.parquet files are required")
    train_df, validation_df, test_df = [pd.read_parquet(path) for path in paths]

    readiness_path = ROOT / "data/results/science_readiness_report.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    decision = readiness.get("decision", {}).get("SCIENCE_READY", "NO")
    if decision != "YES" and not args.allow_insufficient_fallback:
        raise RuntimeError(
            "SCIENCE_READY is not YES. Re-run after data expansion or explicitly pass "
            "--allow-insufficient-fallback to train a provisional global fallback."
        )

    train_counts = train_df["regime"].value_counts().to_dict()
    insufficient = [regime for regime in REGIME_LABELS if train_counts.get(regime, 0) < READY_THRESHOLD]
    if insufficient and not args.allow_insufficient_fallback:
        raise RuntimeError(f"Insufficient regimes require explicit fallback permission: {insufficient}")

    model_dir = ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = model_dir / "probability"
    probability_dir.mkdir(parents=True, exist_ok=True)
    regime_model = train_regime_classifier(train_df)
    joblib.dump({"model": regime_model, "classes": list(regime_model.classes_), "features": FEATURE_SCHEMA}, model_dir / "regime_classifier.joblib")

    global_model = train_regressor(train_df)
    joblib.dump(global_model, model_dir / "global_correction.joblib")
    experts = {}
    dedicated_regimes = []
    for regime in REGIME_LABELS:
        regime_df = train_df[train_df["regime"] == regime]
        if len(regime_df) >= READY_THRESHOLD:
            expert = train_regressor(regime_df)
            experts[regime] = expert
            dedicated_regimes.append(regime)
            joblib.dump(expert, model_dir / f"expert_{regime.lower()}.joblib")

    probability_models = {}
    calibration = {}
    for name, threshold in THRESHOLDS.items():
        model, method, train_events, validation_events = train_probability_head(train_df, validation_df, threshold)
        probability_models[name] = model
        calibration[name] = {
            "method": method,
            "train_events": train_events,
            "validation_events": validation_events,
        }
        joblib.dump(model, probability_dir / f"{name}_model.joblib")

    val_regime_probabilities = _predict_regime_probabilities(regime_model, validation_df)
    test_regime_probabilities = _predict_regime_probabilities(regime_model, test_df)
    predictions = {
        "raw_gfs": test_df["gfs_rain_mm"].to_numpy(dtype=float),
        "global_ml": _predict_regressor(global_model, test_df),
        "anvaya_moe": soft_moe_predictions(test_df, test_regime_probabilities, global_model, experts),
    }
    probability_test = predict_probability_heads(probability_models, test_df)
    prediction_frame = test_df[["valid_time", "init_time", "latitude", "longitude", "regime", "observed_rain_mm", "gfs_rain_mm"]].copy()
    for name, values in predictions.items():
        prediction_frame[f"{name}_mm"] = values
    for name in THRESHOLDS:
        prediction_frame[f"p_{name}"] = probability_test[name].to_numpy()
    prediction_path = ROOT / "data/results/scientific_test_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)

    metrics = evaluate_systems(test_df, predictions)
    run_status = "PROVISIONAL_UNVERIFIED_LOCAL_DATA" if args.provisional else "SCIENTIFIC_CANDIDATE"
    manifest = {
        "model_version": "anvaya-scientific-v1",
        "status": run_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash(paths),
        "training_period": "2020-2023 partition",
        "validation_period": "2024 partition",
        "test_period": "2025 partition preserved",
        "features": FEATURE_SCHEMA,
        "random_seed": 42,
        "dedicated_regimes": dedicated_regimes,
        "fallback_regimes": insufficient,
        "fallback_strategy": "global_correction_expert",
        "probability_calibration": calibration,
        "metrics_path": "data/results/scientific_metrics.json",
        "prediction_path": "data/results/scientific_test_predictions.parquet",
        "source_manifest": "data/source_manifest.json",
    }
    (ROOT / "models/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "data/results/scientific_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": run_status,
        "dedicated_regimes": dedicated_regimes,
        "fallback_regimes": insufficient,
        "test_rows": len(test_df),
        "metrics": metrics["overall"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
