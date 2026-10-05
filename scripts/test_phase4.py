"""Phase 4 district, warning, route, and scientific-freeze acceptance checks."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient

from app.main import app
from app.services.district_service import DistrictService, district_service
from app.services.operational_service import operational_service
from ml.geospatial.boundaries import DistrictBoundaryRegistry
from ml.geospatial.sampling import area_weighted_grid_samples
from ml.operational.common import LIVE, UNAVAILABLE, ProviderResult
from ml.operational.warnings import evaluate_warning


RUN = datetime(2026, 10, 3, 0, tzinfo=timezone.utc)
PROTECTED_HASHES = {
    "data/processed/train_verified.parquet": "be08bfa10133040ce77fb7f3694794fe6d05a7a124e63b64d3e887ee7ad79328",
    "data/processed/validation_verified.parquet": "34f86655f6125e2365415c9d9f2ca66acd40b586117781d6135c496b926ad377",
    "data/processed/test_verified.parquet": "45d3676ded169f95e4582ff73ce764497cfbfa689d55a2c65b25b52779a5569b",
    "data/processed/test.parquet": "c12585ab1882014bc5005767fc88c4a6270d11d80f5d043d9797c99574e8f5e6",
    "models/scientific/phase3f/anvaya_moe.joblib": "c9bfe4048234dd51a79272a9fe5e60902aa2aebf179bd059bd203ee10b59f996",
    "models/scientific/phase3f/multi_threshold_probabilities.joblib": "4a20f89692cc45f82b1d464ed23f36d5d1186ca9ca67c261bcc38a075c9e195c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mock_forecast(latitude: float, longitude: float, run_time=None, include_imerg=True):
    valid = RUN.replace(tzinfo=timezone.utc)
    rain = 32.0 + latitude * 0.1 + longitude * 0.01
    return {
        "mode": "LIVE_OPERATIONAL_MODE",
        "location": {"latitude": latitude, "longitude": longitude},
        "raw_gfs": {"status": LIVE, "data": {"gfs_daily_rain_mm": rain}},
        "gefs_mean": rain + 2.0,
        "gefs_spread": 5.0,
        "ecmwf_forecast": {"status": LIVE, "data": {"models": {
            "IFS 0.25-degree": {"status": LIVE, "data": {"precipitation_24h_mm": rain + 1.0}},
            "AIFS single 0.25-degree": {"status": LIVE, "data": {"precipitation_24h_mm": rain + 3.0}},
        }}},
        "ensemble_consensus": {"status": LIVE, "precipitation_24h_mean_mm": rain + 1.5,
                                "precipitation_24h_spread_mm": 2.0, "independent_model_families": 3},
        "anvaya_corrected": rain + 0.5,
        "global_ml_rain_mm": rain + 0.4,
        "anvaya_minus_global_mm": 0.1,
        "anvaya_explanation": {"status": LIVE, "all_feature_contributions": [
            {"feature": "gfs_u850", "live_feature_value": 10.0, "correction_contribution_mm": 0.1}
        ], "top_feature_contributions": []},
        "regime": "Active",
        "regime_probability": {"Active": 0.8, "Break": 0.2},
        "heavy_probability": 0.3,
        "very_heavy_probability": 0.12,
        "extreme_probability": 0.01,
        "forecast_confidence": {"score": 0.7, "calibrated": False},
        "forecast_reference_time": valid.isoformat().replace("+00:00", "Z"),
        "ensemble_consensus": {"status": LIVE, "precipitation_24h_mean_mm": rain + 1.5,
                               "precipitation_24h_spread_mm": 2.0, "independent_model_families": 3},
        "source_metadata": {
            "gfs": {"status": LIVE, "source": "test GFS"},
            "gefs": {"status": LIVE, "source": "test GEFS"},
            "ecmwf": {"status": LIVE, "source": "test ECMWF"},
            "anvaya": {"status": LIVE, "source": "frozen model"},
        },
        "timestamp": valid.isoformat().replace("+00:00", "Z"),
    }


class Phase4AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = DistrictBoundaryRegistry()
        cls.records = cls.registry.all()

    def test_district_geojson_and_point_lookup(self):
        self.assertEqual(len(self.records), 733)
        self.assertEqual(len({item.district_id for item in self.records}), 733)
        for item in self.records:
            self.assertTrue(item.geometry.is_valid)
            self.assertIn(item.geometry.geom_type, {"Polygon", "MultiPolygon"})
            self.assertTrue(-180 <= item.bounds[0] <= item.bounds[2] <= 180)
            self.assertTrue(-90 <= item.bounds[1] <= item.bounds[3] <= 90)
        target = next(item for item in self.records if 10 <= item.geometry.representative_point().y <= 35
                      and 70 <= item.geometry.representative_point().x <= 95)
        point = target.geometry.representative_point()
        resolved, matches = self.registry.lookup_point(point.y, point.x)
        self.assertIn(target.district_id, matches)
        self.assertEqual(resolved.district_id, target.district_id)
        self.assertEqual(self.registry.metadata["output_crs"].split()[0], "EPSG:4326")

    def test_area_weighted_spatial_sampling_and_summary(self):
        polygon = box(77.1, 19.1, 78.1, 20.1)
        result = area_weighted_grid_samples(
            polygon, spacing_degrees=0.25, max_samples=4,
            domain={"west": 68, "east": 97, "south": 8, "north": 37},
        )
        self.assertGreater(result["intersecting_grid_cell_count"], len(result["samples"]))
        self.assertLessEqual(len(result["samples"]), 4)
        self.assertAlmostEqual(sum(sample.area_weight_m2 for sample in result["samples"]), result["area_inside_domain_m2"], delta=0.01)
        service = DistrictService.__new__(DistrictService)
        summary = service._weighted_summary(
            [{"area_weight_m2": 1.0, "value": 10.0}, {"area_weight_m2": 3.0, "value": 20.0}],
            4.0, lambda row: row["value"], 0.75,
        )
        self.assertEqual(summary["status"], LIVE)
        self.assertEqual(summary["mean_mm"], 17.5)
        self.assertEqual(summary["max_mm"], 20.0)
        self.assertEqual(summary["p90_mm"], 20.0)

    def test_warning_policy_all_levels_and_missing_coverage(self):
        policy = yaml.safe_load((ROOT / "config/operational.yaml").read_text(encoding="utf-8"))["warning_policy"]
        def evaluate(heavy, very_heavy, extreme, coverage=1.0):
            return evaluate_warning({"heavy_probability": heavy, "very_heavy_probability": very_heavy,
                                     "extreme_probability": extreme}, policy, coverage_fraction=coverage)["level"]
        self.assertEqual(evaluate(0.05, 0.01, 0.0), "GREEN")
        self.assertEqual(evaluate(0.3, 0.05, 0.0), "YELLOW")
        self.assertEqual(evaluate(0.55, 0.3, 0.0), "ORANGE")
        self.assertEqual(evaluate(0.8, 0.5, 0.25), "RED")
        self.assertEqual(evaluate(0.8, 0.5, 0.25, coverage=0.5), UNAVAILABLE)
        self.assertEqual(evaluate(None, 0.1, 0.0), UNAVAILABLE)

    def test_operational_routes_and_reference_catalog(self):
        item = next(item for item in self.records if 10 <= item.geometry.representative_point().y <= 35
                    and 70 <= item.geometry.representative_point().x <= 95)
        point = item.geometry.representative_point()
        with patch.object(operational_service, "forecast", new=AsyncMock(side_effect=mock_forecast)), \
             patch.object(operational_service.imerg, "fetch", new=AsyncMock(return_value=ProviderResult("test IMERG", UNAVAILABLE, error="offline test"))):
            with TestClient(app) as client:
                catalog = client.get("/api/operational/districts", params={"search": item.district, "limit": 10})
                lookup = client.get("/api/operational/districts/lookup", params={"latitude": point.y, "longitude": point.x})
                outline = client.get("/api/operational/map")
                district = client.get(f"/api/operational/districts/{item.district_id}")
                point_forecast = client.get("/api/operational/forecast", params={"latitude": point.y, "longitude": point.x})
                selected_map = client.get("/api/operational/map", params={"district_id": item.district_id, "layer": "anvaya"})
                scientific = client.get("/api/verification")
        for response in (catalog, lookup, outline, district, point_forecast, selected_map, scientific):
            self.assertEqual(response.status_code, 200, response.text[:500])
        self.assertEqual(catalog.json()["status"], "REFERENCE")
        self.assertEqual(lookup.json()["district"]["district_id"], item.district_id)
        self.assertEqual(outline.json()["status"], "REFERENCE")
        self.assertEqual(len(outline.json()["features"]), 733)
        self.assertTrue(district.json()["source_metadata"]["scientific_mode"]["benchmark_modified"] is False)
        self.assertEqual(point_forecast.json()["mode"], "LIVE_OPERATIONAL_MODE")
        self.assertIsInstance(selected_map.json()["features"][0]["geometry"], dict)
        self.assertIsNotNone(selected_map.json()["features"][0]["properties"]["forecast_value"])
        verify = scientific.json()
        self.assertEqual(verify["metadata"]["mode"], "SCIENTIFIC_MODE")
        self.assertTrue(verify["metadata"]["benchmark_frozen"])
        self.assertAlmostEqual(verify["comparison"]["raw_gfs"]["rmse"], 16.668700451285506)
        self.assertAlmostEqual(verify["comparison"]["global_ml"]["rmse"], 14.079197516719614)
        self.assertAlmostEqual(verify["comparison"]["anvaya"]["rmse"], 14.106332209718092)
        self.assertFalse(verify["comparison_summary"]["anvaya_beats_global_ml_overall"])
        self.assertTrue(verify["metadata"]["regime_metrics_available"])
        self.assertEqual(set(verify["regime_stratified"]), {
            "ACTIVE_MONSOON", "BREAK_MONSOON", "LOW_DEPRESSION", "COASTAL_RAINFALL",
            "OROGRAPHIC_RAINFALL", "WESTERN_DISTURBANCE", "OTHER",
        })
        active = verify["regime_stratified"]["ACTIVE_MONSOON"]
        self.assertEqual(active["sample_count"], 86223)
        self.assertAlmostEqual(active["systems"]["global_ml"]["rmse"], 18.51710794771139)

    def test_live_provider_failure_never_returns_mock_rainfall(self):
        async def unavailable(*args, **kwargs):
            return ProviderResult("test provider", UNAVAILABLE, error="offline acceptance test")
        with patch.object(operational_service.gfs, "fetch", new=AsyncMock(side_effect=unavailable)), \
             patch.object(operational_service.gefs, "fetch", new=AsyncMock(side_effect=unavailable)), \
             patch.object(operational_service.ecmwf, "fetch", new=AsyncMock(side_effect=unavailable)), \
             patch.object(operational_service.imerg, "fetch", new=AsyncMock(side_effect=unavailable)):
            with TestClient(app) as client:
                response = client.get("/api/operational/forecast", params={
                    "latitude": 20.321, "longitude": 78.432, "run": "2026-10-03T00:00:00Z",
                })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsNone(result["raw_gfs"]["data"])
        self.assertIsNone(result["anvaya_corrected"])
        self.assertIsNone(result["global_ml_rain_mm"])
        self.assertEqual(result["warning_level"], UNAVAILABLE)

    def test_protected_dataset_and_model_hashes(self):
        for relative, expected in PROTECTED_HASHES.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), f"Protected artifact missing: {relative}")
            self.assertEqual(sha256(path), expected, f"Protected SHA-256 changed: {relative}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
