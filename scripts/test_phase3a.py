"""
Phase 3A Verification and Unit Test Suite for ANVAYA.

Tests:
  1. Verification endpoint logic & schema
  2. Synthetic data provider and end-to-end inference
  3. Real-data provider imports and initialization
  4. Feature engineering calculations and topological proxies
  5. Mixture-of-Experts (MoE) soft-gating logic
  6. Multi-threshold probability predictions & monotonicity
  7. Probabilistic regime classifier output
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ml.data.providers import (
    BaseProvider, DemoProvider, GFSProvider, IMDRainfallProvider, MERAProvider, get_provider
)
from ml.features.engineer import (
    engineer_features, dataframe_from_features,
    calculate_wind_speed, calculate_wind_direction, calculate_mslp_anomaly,
    calculate_moisture_transport_proxy, calculate_orographic_flow_proxy
)
from ml.regime.labels import REGIME_LABELS, REGIME_METADATA, assign_weak_regime_label
from ml.regime.classifier import RegimeClassifier
from ml.correction.moe import MixtureOfExpertsCorrectionModel
from ml.probability.multi_threshold import MultiThresholdProbabilityModel


class TestPhase3A(unittest.TestCase):

    def setUp(self):
        self.sample_features = {
            "gfs_rain_mm": 52.4,
            "gfs_mslp": 1002.5,
            "gfs_t2m": 28.5,
            "gfs_rh2m": 88.0,
            "gfs_u10": 5.5,
            "gfs_v10": 2.5,
            "gfs_u850": 14.0,
            "gfs_v850": 4.0,
            "gfs_cape": 1100.0,
            "gfs_precipitable_water": 62.0,
            "rain_last_6h": 18.0,
            "rain_last_24h": 45.0,
            "latitude": 19.076,
            "longitude": 72.8777
        }

    def test_1_verification_endpoint(self):
        """Test verification API reads real metrics file and comparison schema."""
        import asyncio
        from backend.app.api.verification import get_verification
        data = asyncio.run(get_verification())
        self.assertIn("metrics", data)
        self.assertIn("comparison", data)
        self.assertIn("raw_gfs", data["comparison"])
        self.assertIn("global_ml", data["comparison"])
        self.assertIn("anvaya", data["comparison"])
        self.assertGreater(data["metrics"]["rmse"], 0.0)
        self.assertGreater(data["metrics"]["csi"], 0.0)
        print("  [PASS] Test 1: Verification endpoint returns non-zero metrics & 3-system comparison.")

    def test_2_synthetic_mode(self):
        """Test DemoProvider and ml_service end-to-end forecast."""
        demo = DemoProvider()
        feat = demo.get_features(lat=19.076, lon=72.8777)
        self.assertEqual(feat["source"], "SYNTHETIC_DEMO")
        self.assertIn("gfs_daily_rain_mm", feat)
        self.assertNotIn("gfs_rain_mm", feat)

        from backend.app.services.ml_service import ml_service
        self.assertTrue(ml_service.is_ready())
        fcst = ml_service.run_forecast(self.sample_features)
        self.assertIn("corrected_forecast_mm", fcst)
        self.assertIn("regime", fcst)
        self.assertIn("regime_probabilities", fcst)
        self.assertIn("heavy_rain_probability", fcst)
        self.assertIn("explanation", fcst)
        print(f"  [PASS] Test 2: Synthetic inference -> Regime={fcst['regime']}, Raw={fcst['raw_forecast_mm']}mm, Corrected={fcst['corrected_forecast_mm']}mm, P(Heavy)={fcst['heavy_rain_probability']}")

    def test_3_real_data_providers(self):
        """Test real data provider imports and modes."""
        p_demo = get_provider("demo")
        p_hist = get_provider("historical")
        p_live = get_provider("live")

        self.assertIsInstance(p_demo, DemoProvider)
        self.assertIsInstance(p_hist, GFSProvider)
        self.assertIsInstance(p_live, GFSProvider)
        self.assertEqual(p_hist.mode, "historical")
        self.assertEqual(p_live.mode, "live")

        imd = IMDRainfallProvider()
        mera = MERAProvider()
        self.assertIsNotNone(imd.source_name)
        self.assertIsNotNone(mera.source_name)
        print("  [PASS] Test 3: Data provider hierarchy (DemoProvider, GFSProvider, IMDProvider, MERAProvider) verified.")

    def test_4_feature_engineering(self):
        """Test meteorological feature derivation and topological proxies."""
        ws = calculate_wind_speed(3.0, 4.0)
        self.assertAlmostEqual(ws, 5.0)

        mslp_anom = calculate_mslp_anomaly(1000.0)
        self.assertAlmostEqual(mslp_anom, -13.25)

        eng = engineer_features(self.sample_features, lead_time_h=24)
        self.assertIn("wind_speed_10m", eng)
        self.assertIn("moisture_transport_850", eng)
        self.assertIn("mslp_anomaly", eng)
        self.assertIn("terrain_elevation_proxy", eng)
        self.assertIn("orographic_flow_proxy", eng)
        self.assertGreater(eng["moisture_transport_850"], 0.0)

        # Weak regime label
        regime = assign_weak_regime_label(eng)
        self.assertIn(regime, REGIME_LABELS)
        print(f"  [PASS] Test 4: Feature engineering verified. Mumbai conditions weak label: {regime}")

    def test_5_moe_soft_gating(self):
        """Test Mixture-of-Experts soft continuous gating."""
        models_dir = os.path.join(root_dir, "ml/correction/models")
        moe = MixtureOfExpertsCorrectionModel(models_dir=models_dir)
        self.assertGreater(len(moe.experts), 0)

        # Test soft gating with multiple non-zero regime probabilities
        probs = {
            "ACTIVE_MONSOON": 0.60,
            "COASTAL_RAINFALL": 0.30,
            "BREAK_MONSOON": 0.05,
            "OTHER": 0.05,
            "LOW_DEPRESSION": 0.0,
            "OROGRAPHIC_RAINFALL": 0.0,
            "WESTERN_DISTURBANCE": 0.0
        }
        res = moe.correct_rainfall(self.sample_features, probs)
        self.assertIn("raw_forecast_mm", res)
        self.assertIn("corrected_forecast_mm", res)
        self.assertIn("adjustment_mm", res)
        self.assertIn("regime_corrections", res)
        self.assertGreater(res["corrected_forecast_mm"], 0.0)
        print(f"  [PASS] Test 5: MoE Soft Gating -> Raw={res['raw_forecast_mm']}mm, Weighted MoE={res['corrected_forecast_mm']}mm (Adjustment: {res['adjustment_mm']:+}mm)")

    def test_6_multi_threshold_probability(self):
        """Test multi-threshold probabilities and monotonicity."""
        prob_model_path = os.path.join(root_dir, "ml/probability/heavy_rain_model.joblib")
        mt = MultiThresholdProbabilityModel(fallback_model_path=prob_model_path)
        probs = mt.predict(self.sample_features)

        self.assertIn("heavy", probs)
        self.assertIn("very_heavy", probs)
        self.assertIn("extreme", probs)

        # Monotonicity check
        self.assertGreaterEqual(probs["heavy"], probs["very_heavy"])
        self.assertGreaterEqual(probs["very_heavy"], probs["extreme"])
        print(f"  [PASS] Test 6: Multi-threshold Calibrated Probabilities -> Heavy(>=64.5mm): {probs['heavy']:.1%}, Very Heavy(>=115.6mm): {probs['very_heavy']:.1%}, Extreme(>=204.5mm): {probs['extreme']:.1%}")

    def test_7_probabilistic_regime_classifier(self):
        """Test RegimeClassifier probability output vector."""
        regime_model_path = os.path.join(root_dir, "ml/regime/regime_model.joblib")
        clf = RegimeClassifier(model_path=regime_model_path)
        probs = clf.predict_proba(self.sample_features)
        top_regime, conf = clf.predict(self.sample_features)

        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
        self.assertIn(top_regime, REGIME_LABELS)
        self.assertEqual(probs[top_regime], conf)
        print(f"  [PASS] Test 7: Probabilistic Regime Classifier -> Top: {top_regime} ({conf:.1%}), Vector: { {k: round(v, 2) for k, v in probs.items() if v > 0.01} }")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("RUNNING ANVAYA PHASE 3A VERIFICATION & ARCHITECTURE TESTS")
    print("=" * 65)
    unittest.main(verbosity=2)
