"""Offline tests for verified target dates, intervals, and split isolation."""
from __future__ import annotations

from datetime import date, datetime, timezone
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.verified_dataset import (
    _wind_direction_10m,
    candidate_target_dates,
    split_for_target_date,
    target_window,
)
from ml.features.engineer import FEATURE_SCHEMA, engineer_features


class VerifiedDatasetScheduleTests(unittest.TestCase):
    def test_bounded_sample_covers_each_month_and_year_with_two_dates(self) -> None:
        dates = candidate_target_dates("sample")
        self.assertEqual(len(dates), 40)
        for year in range(2020, 2025):
            for month in (6, 7, 8, 9):
                self.assertEqual(sum(d.year == year and d.month == month for d in dates), 2)
        self.assertIn(date(2024, 7, 16), dates)

    def test_full_schedule_is_all_610_jjas_dates(self) -> None:
        dates = candidate_target_dates("full")
        self.assertEqual(len(dates), 610)
        self.assertEqual(len(set(dates)), 610)
        self.assertTrue(all(d.month in (6, 7, 8, 9) for d in dates))

    def test_target_interval_is_exact_and_forecast_is_prospective(self) -> None:
        init, start, end, lead = target_window("2024-07-16")
        self.assertEqual(init, datetime(2024, 7, 15, 0, tzinfo=timezone.utc))
        self.assertEqual(start, datetime(2024, 7, 15, 3, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2024, 7, 16, 3, tzinfo=timezone.utc))
        self.assertEqual((end - start).total_seconds(), 24 * 3600)
        self.assertEqual(end - init, __import__("datetime").timedelta(hours=lead))
        self.assertEqual(lead, 27)
        self.assertGreater(start, init)

    def test_scientific_partitions_are_calendar_separated(self) -> None:
        self.assertEqual(split_for_target_date("2020-06-07"), "train")
        self.assertEqual(split_for_target_date("2022-09-30"), "train")
        self.assertEqual(split_for_target_date("2023-08-12"), "validation")
        self.assertEqual(split_for_target_date("2024-07-16"), "test")
        with self.assertRaises(ValueError):
            split_for_target_date("2025-07-12")

    def test_primary_daily_rain_field_replaces_deprecated_name(self) -> None:
        self.assertIn("gfs_daily_rain_mm", FEATURE_SCHEMA)
        self.assertNotIn("gfs_rain_mm", FEATURE_SCHEMA)
        self.assertNotIn("observed_rain_mm", FEATURE_SCHEMA)
        self.assertNotIn("rain_last_6h", FEATURE_SCHEMA)

    def test_vectorized_wind_direction_matches_canonical_float_precision(self) -> None:
        u = np.asarray([0.03126220777630806, 2.671262264251709], dtype=np.float32)
        v = np.asarray([0.9419604539871216, 0.8919604420661926], dtype=np.float32)
        vectorized = _wind_direction_10m(u, v)
        canonical = np.asarray([
            engineer_features({"gfs_u10": float(uu), "gfs_v10": float(vv)})["wind_dir_10m"]
            for uu, vv in zip(u, v)
        ])
        np.testing.assert_allclose(vectorized, canonical, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
