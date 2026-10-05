"""
Phase 3B Verification & Scientific Integrity Test Suite for ANVAYA.

Tests:
  1. Timestamp & accumulation window alignment
  2. Anti-leakage guarantees (no target in features, no future look-ahead)
  3. Duplicate prevention on spatio-temporal primary keys
  4. IMD rainfall threshold definitions & monotonicity
  5. Strict temporal split separation (Train -> Validation -> Test)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ml.features.engineer import FEATURE_SCHEMA, MINIMUM_FEATURES


class TestPhase3B(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(root_dir, "data/processed")
        cls.train_path = os.path.join(cls.data_dir, "train.parquet")
        cls.val_path = os.path.join(cls.data_dir, "validation.parquet")
        cls.test_path = os.path.join(cls.data_dir, "test.parquet")
        cls.full_path = os.path.join(cls.data_dir, "real_training_data.parquet")

        cls.df_train = pd.read_parquet(cls.train_path)
        cls.df_val = pd.read_parquet(cls.val_path)
        cls.df_test = pd.read_parquet(cls.test_path)
        cls.df_full = pd.read_parquet(cls.full_path)

    def test_1_timestamp_alignment(self):
        """Verify initialization time, lead hours, and valid time alignment."""
        sample = self.df_full.sample(100, random_state=42)
        for _, row in sample.iterrows():
            v_time = pd.Timestamp(row["valid_time"])
            i_time = pd.Timestamp(row["init_time"])
            lead = row["lead_hours"]
            # Physical requirement: valid_time == init_time + lead_hours
            expected_v = i_time + pd.Timedelta(hours=lead)
            self.assertEqual(v_time, expected_v)
            # Physical requirement: valid_time corresponds to 03:00 UTC IMD daily accumulation
            self.assertEqual(v_time.hour, 3)
            self.assertEqual(v_time.minute, 0)
        print("  [PASS] Test 1: Timestamp & accumulation window physical alignment verified.")

    def test_2_no_future_leakage(self):
        """Verify that observed_rain_mm is strictly a target and never in features."""
        # 1. Target never in feature schema
        self.assertNotIn("observed_rain_mm", FEATURE_SCHEMA)
        self.assertNotIn("observed_rain_mm", MINIMUM_FEATURES)
        self.assertNotIn("actual_rain_mm", FEATURE_SCHEMA)
        self.assertNotIn("heavy_rain", FEATURE_SCHEMA)

        # 2. Check that antecedent observations do not correlate identically with target
        corr = self.df_full[["rain_last_24h", "observed_rain_mm"]].corr().iloc[0, 1]
        self.assertLess(corr, 0.90) # Should be reasonable physical correlation, not an identity leak
        print(f"  [PASS] Test 2: Anti-leakage guarantees confirmed. Antecedent-to-target corr: {corr:.3f}")

    def test_3_duplicate_prevention(self):
        """Verify zero duplicates in all spatio-temporal partitions."""
        dup_full = self.df_full.duplicated(subset=["valid_time", "latitude", "longitude"]).sum()
        dup_train = self.df_train.duplicated(subset=["valid_time", "latitude", "longitude"]).sum()
        dup_val = self.df_val.duplicated(subset=["valid_time", "latitude", "longitude"]).sum()
        dup_test = self.df_test.duplicated(subset=["valid_time", "latitude", "longitude"]).sum()

        self.assertEqual(dup_full, 0)
        self.assertEqual(dup_train, 0)
        self.assertEqual(dup_val, 0)
        self.assertEqual(dup_test, 0)
        print("  [PASS] Test 3: Spatio-temporal uniqueness verified (0 duplicates across all partitions).")

    def test_4_threshold_correctness_and_monotonicity(self):
        """Verify IMD warning threshold assignments and subset monotonicity."""
        obs = self.df_full["observed_rain_mm"]
        h = self.df_full["heavy_rain"]
        vh = self.df_full["very_heavy_rain"]
        ext = self.df_full["extreme_rain"]

        # Exact boundary logic
        self.assertTrue(((obs >= 64.5) == (h == 1)).all())
        self.assertTrue(((obs >= 115.6) == (vh == 1)).all())
        self.assertTrue(((obs >= 204.5) == (ext == 1)).all())

        # Physical monotonicity: extreme <= very_heavy <= heavy
        self.assertTrue((ext <= vh).all())
        self.assertTrue((vh <= h).all())
        print(f"  [PASS] Test 4: Threshold accuracy & monotonicity verified (Heavy: {h.sum()}, Very Heavy: {vh.sum()}, Extreme: {ext.sum()}).")

    def test_5_temporal_split_separation(self):
        """Verify strictly chronological temporal partitions with no cross-split leakage."""
        train_max = pd.Timestamp(self.df_train["valid_time"].max())
        val_min = pd.Timestamp(self.df_val["valid_time"].min())
        val_max = pd.Timestamp(self.df_val["valid_time"].max())
        test_min = pd.Timestamp(self.df_test["valid_time"].min())

        # Strict chronological ordering
        self.assertLess(train_max, val_min)
        self.assertLess(val_max, test_min)

        # Zero set intersection between timestamps
        train_times = set(self.df_train["valid_time"].unique())
        val_times = set(self.df_val["valid_time"].unique())
        test_times = set(self.df_test["valid_time"].unique())

        self.assertEqual(len(train_times.intersection(val_times)), 0)
        self.assertEqual(len(val_times.intersection(test_times)), 0)
        self.assertEqual(len(train_times.intersection(test_times)), 0)
        print(f"  [PASS] Test 5: Chronological temporal separation confirmed ({train_max} < {val_min} and {val_max} < {test_min}).")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("RUNNING ANVAYA PHASE 3B INTEGRITY & ALIGNMENT TESTS")
    print("=" * 65)
    unittest.main(verbosity=2)
