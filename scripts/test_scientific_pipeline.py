"""Fast checks for the governed ANVAYA scientific checkpoint.

These checks validate artifact contracts and safety properties. They do not
turn the provisional local-data experiment into a scientific readiness pass.
"""
import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.correction.moe import MixtureOfExpertsCorrectionModel
from ml.features.engineer import FEATURE_SCHEMA
from ml.probability.multi_threshold import MultiThresholdProbabilityModel


class TestScientificPipeline(unittest.TestCase):
    def test_registry_and_validation_contract(self):
        registry = yaml.safe_load((ROOT / "config/feature_registry.yaml").read_text(encoding="utf-8"))
        names = {row["feature_name"] for row in registry["features"]}
        self.assertTrue(set(FEATURE_SCHEMA) <= names)
        self.assertNotIn("observed_rain_mm", FEATURE_SCHEMA)
        self.assertFalse(any(
            row.get("target_related") and row.get("allowed_for_live_model")
            for row in registry["features"]
        ))

        report = json.loads((ROOT / "data/results/validation_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["pass"])

    def test_manifest_declares_provisional_fallbacks(self):
        manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "PROVISIONAL_UNVERIFIED_LOCAL_DATA")
        self.assertIn("BREAK_MONSOON", manifest["fallback_regimes"])
        self.assertIn("WESTERN_DISTURBANCE", manifest["fallback_regimes"])
        self.assertEqual(manifest["test_period"], "2025 partition preserved")

    def test_probability_artifact_is_monotone(self):
        path = ROOT / "data/results/scientific_test_predictions.parquet"
        df = pd.read_parquet(path)
        self.assertTrue((df["p_extreme"] <= df["p_very_heavy"]).all())
        self.assertTrue((df["p_very_heavy"] <= df["p_heavy"]).all())

        model = MultiThresholdProbabilityModel(str(ROOT / "models/scientific/probability"))
        self.assertEqual(set(model.models), {"heavy", "very_heavy", "extreme"})
        row = df.iloc[0].to_dict()
        probabilities = model.predict(row)
        self.assertGreaterEqual(probabilities["heavy"], probabilities["very_heavy"])
        self.assertGreaterEqual(probabilities["very_heavy"], probabilities["extreme"])

    def test_moe_marks_missing_regime_expert_as_fallback(self):
        model = MixtureOfExpertsCorrectionModel(str(ROOT / "models/scientific"))
        self.assertIn("ACTIVE_MONSOON", model.experts)
        self.assertIsNotNone(model.global_model)
        result = model.correct_rainfall(
            {"gfs_rain_mm": 20.0},
            {"BREAK_MONSOON": 0.8, "ACTIVE_MONSOON": 0.2},
        )
        self.assertEqual(result["prediction_mode"], "fallback_global")
        self.assertIn("BREAK_MONSOON", result["fallback_regimes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
