"""Artifact and leakage-firewall checks for the Phase 3F trained stack."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from ml.features.engineer import FEATURE_SCHEMA
from ml.regime.labels import REGIME_LABELS
from ml.scientific.phase3f import (
    THRESHOLDS,
    enforce_probability_order,
    predict_moe,
    predict_all_probabilities,
    predict_probability_bundle,
    probability_order_is_valid,
)


ARTIFACT_DIR = ROOT / "models/scientific/phase3f"
RESULT_DIR = ROOT / "data/results"
TRAIN_PATH = ROOT / "data/processed/train_verified.parquet"
VALIDATION_PATH = ROOT / "data/processed/validation_verified.parquet"
TEST_PATH = ROOT / "data/processed/test_verified.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    report_path = RESULT_DIR / "model_training_report.json"
    manifest_path = RESULT_DIR / "experiment_manifest.json"
    freeze_path = RESULT_DIR / "phase3f_frozen_models.json"
    predictions_path = RESULT_DIR / "test_predictions.parquet"
    for path in (report_path, manifest_path, freeze_path, predictions_path):
        require(path.is_file(), f"Missing Phase 3F result: {path.relative_to(ROOT)}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    forbidden = {"observed_rain_mm", "regime", "heavy_rain", "very_heavy_rain", "extreme_rain",
                 "gfs_rain_mm", "gfs_apcp_interval_mm", "rain_last_6h"}
    require(not (set(FEATURE_SCHEMA) & forbidden), "Target, weak label, or retired field leaked into predictors")
    require(manifest["features"] == FEATURE_SCHEMA, "Saved feature schema differs from the live registry")
    require(manifest["dataset"]["legacy_test_parquet_used"] is False,
            "Legacy test artifact must not enter Phase 3F")
    require(freeze["test_opened_after_freeze"] is True and freeze["test_accessed"] is True,
            "Held-out test was not opened strictly after model freeze")
    require(manifest["dataset"]["test_opened_after_models_frozen"] is True,
            "Experiment manifest lacks test freeze evidence")
    require(manifest["dataset"]["train"]["target_date_max"] < manifest["dataset"]["validation"]["target_date_min"],
            "Train and validation periods overlap")
    require(manifest["dataset"]["validation"]["target_date_max"] < manifest["dataset"]["test"]["target_date_min"],
            "Validation and test periods overlap")
    for split in ("train", "validation", "test"):
        require(manifest["training_audits"][split]["antecedent_strictly_before_init"],
                f"Antecedent timing audit failed for {split}")
        require(manifest["training_audits"][split]["valid_time_equals_init_plus_lead"],
                f"Forecast time audit failed for {split}")
    require(manifest["dataset"]["train"]["sha256"] == sha256_file(TRAIN_PATH), "Train split checksum mismatch")
    require(manifest["dataset"]["validation"]["sha256"] == sha256_file(VALIDATION_PATH),
            "Validation split checksum mismatch")
    require(manifest["dataset"]["test"]["sha256"] == sha256_file(TEST_PATH), "Test split checksum mismatch")

    expected = ["regime_classifier", "global_correction", "anvaya_moe"]
    for regime, detail in manifest["regression_fit"]["regime_experts"].items():
        if detail["dedicated_expert"]:
            expected.append(f"expert_{regime.lower()}")
    for name, head in manifest["probability_models"].items():
        expected.extend([f"global_{name}_classifier", f"{name}_probability"])
        expected.extend(f"{name}_expert_{regime.lower()}_classifier"
                        for regime in head["dedicated_regime_heads"])
    expected.append("multi_threshold_probabilities")
    for stem in expected:
        path = ARTIFACT_DIR / f"{stem}.joblib"
        require(path.is_file(), f"Missing required model artifact {path.relative_to(ROOT)}")
        loaded = joblib.load(path)
        require(loaded is not None, f"Model did not load: {path.name}")
        del loaded

    # Verify every model that was frozen pre-test still has the same digest.
    for relative, digest in freeze["frozen_models"].items():
        path = ROOT / relative
        require(path.is_file(), f"Frozen model disappeared: {relative}")
        require(sha256_file(path) == digest, f"Frozen model changed after test evaluation: {relative}")

    predictions = pd.read_parquet(predictions_path)
    require(len(predictions) == manifest["dataset"]["test"]["rows"], "Prediction count differs from test rows")
    rain_columns = ["raw_gfs_rain_mm", "global_ml_rain_mm", "anvaya_rain_mm"]
    probability_columns = ["p_heavy", "p_very_heavy", "p_extreme"]
    require(predictions[rain_columns].notna().all().all(), "Missing rainfall predictions")
    require(np.isfinite(predictions[rain_columns].to_numpy()).all(), "Non-finite rainfall predictions")
    require((predictions[["global_ml_rain_mm", "anvaya_rain_mm"]].to_numpy() >= 0).all(),
            "Post-processed rainfall contains negative values")
    require(predictions[probability_columns].notna().all().all(), "Missing probability predictions")
    require(np.isfinite(predictions[probability_columns].to_numpy()).all(), "Non-finite probabilities")
    probability_map = {
        "heavy": predictions["p_heavy"].to_numpy(),
        "very_heavy": predictions["p_very_heavy"].to_numpy(),
        "extreme": predictions["p_extreme"].to_numpy(),
    }
    require(probability_order_is_valid(probability_map), "Probability heads violate range/order constraints")
    posterior_cols = [f"p_regime_{name.lower()}" for name in REGIME_LABELS]
    posterior = predictions[posterior_cols].to_numpy()
    require(np.isfinite(posterior).all(), "Non-finite regime posterior")
    require(np.allclose(posterior.sum(axis=1), 1.0, atol=1e-6), "Regime posterior does not sum to one")
    require(np.all((posterior >= 0) & (posterior <= 1)), "Regime probabilities out of range")
    require(predictions["observed_rain_mm"].notna().all(), "Missing test targets in prediction artifact")

    # A real model reload/inference smoke test uses validation predictors only;
    # the held-out test output above remains final-evaluation-only.
    validation = pd.read_parquet(VALIDATION_PATH, columns=FEATURE_SCHEMA).head(128)
    X_smoke = validation[FEATURE_SCHEMA].to_numpy(dtype=np.float32)
    moe = joblib.load(ARTIFACT_DIR / "anvaya_moe.joblib")
    rain_smoke, regime_smoke = predict_moe(moe, X_smoke)
    require(np.isfinite(rain_smoke).all() and (rain_smoke >= 0).all(), "ANVAYA reload inference failed")
    require(np.allclose(regime_smoke.sum(axis=1), 1, atol=1e-6), "Reloaded regime posterior is invalid")
    for name in THRESHOLDS:
        bundle = joblib.load(ARTIFACT_DIR / f"{name}_probability.joblib")
        p = predict_probability_bundle(bundle, X_smoke, regime_smoke)
        require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), f"Reloaded {name} probability invalid")
    stack = joblib.load(ARTIFACT_DIR / "multi_threshold_probabilities.joblib")
    ordered_smoke = predict_all_probabilities(stack, X_smoke, regime_smoke)
    require(probability_order_is_valid(ordered_smoke), "Reloaded probability ordering failed")

    for result_name in ("test_metrics.csv", "threshold_metrics.csv", "regime_metrics.csv",
                        "confusion_matrix_regime.csv", "feature_importance.csv"):
        require((RESULT_DIR / result_name).is_file(), f"Missing result artifact: {result_name}")
    test_metrics = pd.read_csv(RESULT_DIR / "test_metrics.csv")
    require(set(test_metrics["system"]) == {"RAW_GFS", "GLOBAL_ML", "ANVAYA"},
            "Three-system comparison is incomplete")
    require(set(predictions["regime"].unique()) == set(REGIME_LABELS), "Test predictions omit a regime")
    print("Phase 3F artifact/leakage tests: PASS")
    print(f"Loaded {len(expected)} required model files; checked {len(predictions):,} test predictions.")


if __name__ == "__main__":
    main()
