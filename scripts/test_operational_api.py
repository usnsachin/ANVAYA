"""Offline smoke tests for Phase 3G adapters and GET /api/operational/forecast."""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient

from ml.operational.common import LIVE, UNAVAILABLE, ProviderResult
from ml.operational.consensus import build_consensus
from ml.operational.ecmwf import ECMWFProvider
from ml.operational.ge_f_s import GEFSProvider
from ml.operational.gfs_live import GFSLiveProvider
from ml.operational.imerg import IMERGProvider, _variable_schema
from ml.operational.risk import aggregate_district_risk
from app.main import app
from app.services.operational_service import operational_service


RUN = datetime(2026, 10, 3, 0, tzinfo=timezone.utc)


def openmeteo_payload(run: datetime, rain_per_hour: float = 0.25) -> dict:
    times = [run + timedelta(hours=index) for index in range(28)]
    values = {
        "time": [value.strftime("%Y-%m-%dT%H:%M") for value in times],
        "precipitation": [rain_per_hour] * len(times),
        "temperature_2m": [27.0] * len(times),
        "relative_humidity_2m": [72.0] * len(times),
        "pressure_msl": [1004.0] * len(times),
        "wind_speed_10m": [4.0] * len(times),
        "wind_direction_10m": [270.0] * len(times),
        "wind_speed_850hPa": [12.0] * len(times),
        "wind_direction_850hPa": [250.0] * len(times),
        "cape": [650.0] * len(times),
        "total_column_integrated_water_vapour": [55.0] * len(times),
    }
    return {"latitude": 20.0, "longitude": 78.0, "elevation": 100.0, "hourly": values}


def gfs_result(run: datetime, rain: float) -> ProviderResult:
    valid = run + timedelta(hours=27)
    start = valid - timedelta(hours=24)
    return ProviderResult(
        "fake NOAA GFS",
        LIVE,
        data={
            "gfs_daily_rain_mm": rain,
            "gfs_mslp": 1004.0,
            "gfs_t2m": 27.0,
            "gfs_rh2m": 72.0,
            "gfs_u10": 4.0,
            "gfs_v10": 0.0,
            "gfs_u850": 10.0,
            "gfs_v850": 5.0,
            "gfs_cape": 650.0,
            "gfs_precipitable_water": 55.0,
            "init_time": run.isoformat().replace("+00:00", "Z"),
            "valid_time": valid.isoformat().replace("+00:00", "Z"),
            "lead_hours": 27,
            "accumulation_start": start.isoformat().replace("+00:00", "Z"),
            "accumulation_end": valid.isoformat().replace("+00:00", "Z"),
        },
        metadata={"test": True},
    )


def ecmwf_result(run: datetime) -> ProviderResult:
    valid = run + timedelta(hours=27)
    start = valid - timedelta(hours=24)
    model_data = {
        "precipitation_24h_mm": 38.0,
        "gfs_mslp": 1006.0,
        "gfs_t2m": 26.0,
        "gfs_rh2m": 75.0,
        "gfs_u10": 3.0,
        "gfs_v10": 1.0,
        "gfs_u850": 11.0,
        "gfs_v850": 4.0,
        "gfs_cape": 500.0,
        "gfs_precipitable_water": 52.0,
        "init_time": run.isoformat().replace("+00:00", "Z"),
        "valid_time": valid.isoformat().replace("+00:00", "Z"),
        "accumulation_start": start.isoformat().replace("+00:00", "Z"),
        "accumulation_end": valid.isoformat().replace("+00:00", "Z"),
        "lead_hours": 27,
    }
    return ProviderResult(
        "fake ECMWF",
        LIVE,
        data={"models": {
            "IFS 0.25-degree": {"status": LIVE, "data": model_data},
            "AIFS single 0.25-degree": {"status": LIVE, "data": {**model_data, "precipitation_24h_mm": 40.0}},
        }},
        metadata={"test": True},
    )


class OperationalAdapterTests(unittest.TestCase):
    def test_gfs_sums_exact_24_hour_window_at_plus_27(self):
        async def run_test():
            provider = GFSLiveProvider()
            with patch("ml.operational.gfs_live.get_json", new=AsyncMock(return_value=openmeteo_payload(RUN))):
                result = await provider.fetch(20.0, 78.0, RUN)
            self.assertEqual(result.status, LIVE)
            self.assertAlmostEqual(result.data["gfs_daily_rain_mm"], 6.0)
            self.assertEqual(result.data["lead_hours"], 27)
            self.assertEqual(result.data["accumulation_start"], "2026-10-03T03:00:00Z")
            self.assertEqual(result.data["accumulation_end"], "2026-10-04T03:00:00Z")
            self.assertEqual(result.metadata["precipitation_unit"], "mm")
        asyncio.run(run_test())

    def test_gfs_missing_required_interval_fails_closed(self):
        async def run_test():
            payload = openmeteo_payload(RUN)
            payload["hourly"]["precipitation"][27] = None
            with patch("ml.operational.gfs_live.get_json", new=AsyncMock(return_value=payload)):
                result = await GFSLiveProvider().fetch(20.0, 78.0, RUN)
            self.assertEqual(result.status, UNAVAILABLE)
            self.assertIn("Missing numeric field", result.error)
        asyncio.run(run_test())

    def test_gefs_spread_uses_returned_members(self):
        async def run_test():
            target = RUN + timedelta(hours=27)
            times = [RUN + timedelta(hours=i) for i in range(120)]
            payload = {
                "hourly": {
                    "time": [value.strftime("%Y-%m-%dT%H:%M") for value in times],
                    "precipitation_member01": [0.5] * len(times),
                    "precipitation_member02": [1.5] * len(times),
                }
            }
            with patch("ml.operational.ge_f_s.get_json", new=AsyncMock(return_value=payload)):
                result = await GEFSProvider().fetch(20.0, 78.0, target)
            self.assertEqual(result.status, LIVE)
            self.assertEqual(result.data["member_count"], 2)
            self.assertAlmostEqual(result.data["precipitation_24h_mean_mm"], 24.0)
            self.assertAlmostEqual(result.data["precipitation_24h_spread_mm"], 12.0)
        asyncio.run(run_test())

    def test_ecmwf_parallel_model_failure_is_partial_not_fatal(self):
        async def run_test():
            async def fetch(url, params):
                if params["models"] == "ecmwf_aifs025":
                    raise RuntimeError("AIFS temporarily unavailable")
                return openmeteo_payload(RUN)
            with patch("ml.operational.ecmwf.get_json", new=fetch):
                result = await ECMWFProvider().fetch(20.0, 78.0, RUN)
            self.assertEqual(result.status, LIVE)
            self.assertIn("IFS 0.25-degree", result.data["models"])
            self.assertEqual(result.metadata["model_status"]["AIFS single 0.25-degree"], UNAVAILABLE)
        asyncio.run(run_test())

    def test_consensus_refuses_different_accumulation_windows(self):
        gfs = gfs_result(RUN, 6.0).as_dict()
        gefs = ProviderResult(
            "GEFS", LIVE,
            data={
                "precipitation_24h_mean_mm": 20.0,
                "valid_time": "2026-10-04T03:00:00Z",
                "accumulation_start": "2026-10-03T02:00:00Z",
                "accumulation_end": "2026-10-04T03:00:00Z",
            },
        ).as_dict()
        ecmwf = ecmwf_result(RUN).as_dict()
        for item in ecmwf["data"]["models"].values():
            item["data"]["accumulation_start"] = "2026-10-03T02:00:00Z"
        result = build_consensus(gfs, gefs, ecmwf)
        self.assertEqual(result["status"], UNAVAILABLE)
        self.assertEqual(result["independent_model_families"], 1)

    def test_imerg_without_credentials_is_unavailable_not_demo(self):
        async def run_test():
            with patch.dict(os.environ, {}, clear=True):
                result = await IMERGProvider().fetch(
                    20.0, 78.0, target_start=RUN + timedelta(hours=3), now=RUN + timedelta(hours=6)
                )
            self.assertEqual(result.status, UNAVAILABLE)
            self.assertIn("credentials", result.error)
        asyncio.run(run_test())

    def test_imerg_metadata_requires_rate_units_and_keeps_dimension_order(self):
        dmr = (
            '<Dataset><Group name="Grid">'
            '<Float32 name="precipitationCal">'
            '<Dim name="/Grid/time"/><Dim name="/Grid/lon"/><Dim name="/Grid/lat"/>'
            '<Attribute name="units" type="String" value="mm/hr"/>'
            '</Float32></Group></Dataset>'
        )
        dimensions, units = _variable_schema(dmr)
        self.assertEqual(dimensions, ["time", "lon", "lat"])
        self.assertEqual(units, "mm/hr")

    def test_district_risk_requires_real_geometry(self):
        result = aggregate_district_risk(
            "district-1", None, [], threshold=0.5,
            levels=[{"name": "HEAVY", "probability": "heavy_probability"}],
        )
        self.assertEqual(result["status"], UNAVAILABLE)

    def test_geojson_district_aggregation_uses_only_inside_points(self):
        polygon = {
            "type": "Polygon",
            "coordinates": [[[77.0, 19.0], [79.0, 19.0], [79.0, 21.0], [77.0, 21.0], [77.0, 19.0]]],
        }
        result = aggregate_district_risk(
            "district-1", polygon,
            [
                {"latitude": 20.0, "longitude": 78.0, "probabilities": {"heavy_probability": 0.7}},
                {"latitude": 25.0, "longitude": 80.0, "probabilities": {"heavy_probability": 0.9}},
            ],
            threshold=0.5,
            levels=[{"name": "HEAVY", "probability": "heavy_probability"}],
        )
        self.assertEqual(result["status"], LIVE)
        self.assertEqual(result["grid_point_count"], 1)
        self.assertEqual(result["warning_level"], "HEAVY")


class OperationalEndpointSmokeTest(unittest.TestCase):
    def test_live_endpoint_returns_model_and_probability_fields(self):
        event_order = []

        async def gfs_fetch(latitude, longitude, run_time):
            event_order.append("gfs")
            if run_time == RUN:
                return gfs_result(RUN, 35.0)
            return gfs_result(RUN - timedelta(hours=48), 18.0)

        async def gefs_fetch(latitude, longitude, valid_time):
            valid = valid_time
            start = valid - timedelta(hours=24)
            return ProviderResult("fake GEFS", LIVE, data={
                "precipitation_24h_mean_mm": 36.0,
                "precipitation_24h_spread_mm": 8.0,
                "valid_time": valid.isoformat().replace("+00:00", "Z"),
                "accumulation_start": start.isoformat().replace("+00:00", "Z"),
                "accumulation_end": valid.isoformat().replace("+00:00", "Z"),
                "member_count": 31,
            })

        async def ecmwf_fetch(latitude, longitude, run_time):
            return ecmwf_result(run_time)

        async def imerg_fetch(latitude, longitude, *, target_start):
            event_order.append("imerg")
            return ProviderResult("fake IMERG", LIVE, data={
                "precipitation_24h_mm": 12.0,
                "window_start": (target_start - timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
                "window_end": target_start.isoformat().replace("+00:00", "Z"),
                "rate_sample_count": 48,
            }, metadata={"post_prediction_context": True})

        # Use a unique location to avoid a cache entry from another run of this test.
        real_predict = operational_service._predict

        def tracked_predict(row):
            event_order.append("anvaya")
            return real_predict(row)

        with patch.object(operational_service.gfs, "fetch", new=AsyncMock(side_effect=gfs_fetch)), \
             patch.object(operational_service.gefs, "fetch", new=AsyncMock(side_effect=gefs_fetch)), \
             patch.object(operational_service.ecmwf, "fetch", new=AsyncMock(side_effect=ecmwf_fetch)), \
             patch.object(operational_service.imerg, "fetch", new=AsyncMock(side_effect=imerg_fetch)), \
             patch.object(operational_service, "_predict", side_effect=tracked_predict):
            with TestClient(app) as client:
                response = client.get(
                    "/api/operational/forecast",
                    params={"latitude": 19.876, "longitude": 77.654, "run": "2026-10-03T00:00:00Z"},
                )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["mode"], "LIVE_OPERATIONAL_MODE")
        for field in (
            "raw_gfs", "gefs_mean", "gefs_spread", "ecmwf_forecast", "ensemble_consensus",
            "anvaya_corrected", "regime", "regime_probability", "heavy_probability",
            "very_heavy_probability", "extreme_probability", "imerg_current",
            "forecast_confidence", "warning_level", "source_metadata", "timestamp",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["source_metadata"]["anvaya"]["status"], LIVE, payload["source_metadata"]["anvaya"])
        self.assertIsNotNone(payload["anvaya_corrected"])
        self.assertEqual(payload["source_metadata"]["imerg"]["status"], LIVE)
        self.assertEqual(payload["ensemble_consensus"]["status"], LIVE)
        self.assertEqual(payload["district_risk"]["status"], UNAVAILABLE)
        self.assertFalse(payload["forecast_confidence"]["calibrated"])
        self.assertLess(event_order.index("anvaya"), event_order.index("imerg"))

    def test_provider_failures_keep_http_route_available(self):
        async def fail_gfs(*args, **kwargs):
            return ProviderResult("gfs", UNAVAILABLE, error="offline smoke test")

        async def fail_gefs(*args, **kwargs):
            return ProviderResult("gefs", UNAVAILABLE, error="offline smoke test")

        async def fail_ecmwf(*args, **kwargs):
            return ProviderResult("ecmwf", UNAVAILABLE, error="offline smoke test")

        async def fail_imerg(*args, **kwargs):
            return ProviderResult("imerg", UNAVAILABLE, error="offline smoke test")

        with patch.object(operational_service.gfs, "fetch", new=AsyncMock(side_effect=fail_gfs)), \
             patch.object(operational_service.gefs, "fetch", new=AsyncMock(side_effect=fail_gefs)), \
             patch.object(operational_service.ecmwf, "fetch", new=AsyncMock(side_effect=fail_ecmwf)), \
             patch.object(operational_service.imerg, "fetch", new=AsyncMock(side_effect=fail_imerg)):
            with TestClient(app) as client:
                response = client.get(
                    "/api/operational/forecast",
                    params={"latitude": 20.123, "longitude": 78.456, "run": "2026-10-03T00:00:00Z"},
                )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["anvaya_corrected"])
        self.assertEqual(response.json()["warning_level"], UNAVAILABLE)

    def test_operational_route_enforces_india_bounds(self):
        with TestClient(app) as client:
            response = client.get("/api/operational/forecast", params={"latitude": 5.0, "longitude": 78.0})
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main(verbosity=2)
