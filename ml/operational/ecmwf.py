"""ECMWF IFS/AIFS open-model forecasts via Open-Meteo Single Runs API."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from ml.operational.common import LIVE, UNAVAILABLE, ProviderResult, as_finite_number, get_json
from ml.operational.gfs_live import HOURLY_FIELDS, _iso_hour, _series_value, _wind_components


SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"


async def _fetch_model(
    *,
    latitude: float,
    longitude: float,
    run_time: datetime,
    model: str,
    model_name: str,
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "models": model,
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "hourly": HOURLY_FIELDS,
        "forecast_hours": 28,
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    payload = await get_json(SINGLE_RUNS_URL, params)
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError(f"{model_name} response has no hourly time coordinate")
    times = [_iso_hour(value) for value in hourly["time"]]
    if len(set(times)) != len(times):
        raise ValueError(f"{model_name} response contains duplicate valid times")
    time_index = {value: index for index, value in enumerate(times)}
    valid_time = run_time + timedelta(hours=27)
    window_start = valid_time - timedelta(hours=24)
    interval_ends = [window_start + timedelta(hours=step) for step in range(1, 25)]
    if any(value not in time_index for value in interval_ends):
        raise ValueError(f"{model_name} lacks full hourly precipitation coverage for the 24-hour target window")
    rain = [
        _series_value(hourly, "precipitation", time_index[value])
        for value in interval_ends
    ]
    if any(value < 0 for value in rain):
        raise ValueError(f"{model_name} returned negative precipitation")
    valid_index = time_index.get(valid_time)
    if valid_index is None:
        raise ValueError(f"{model_name} lacks target valid time {valid_time.isoformat()}")
    at_valid = {
        key: _series_value(hourly, key, valid_index)
        for key in (
            "temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m",
            "wind_direction_10m", "wind_speed_850hPa", "wind_direction_850hPa", "cape",
            "total_column_integrated_water_vapour",
        )
    }
    u10, v10 = _wind_components(at_valid["wind_speed_10m"], at_valid["wind_direction_10m"])
    u850, v850 = _wind_components(at_valid["wind_speed_850hPa"], at_valid["wind_direction_850hPa"])
    if not 0 <= at_valid["relative_humidity_2m"] <= 100:
        raise ValueError(f"{model_name} relative humidity is outside 0-100 percent")
    if at_valid["cape"] < 0 or at_valid["total_column_integrated_water_vapour"] < 0:
        raise ValueError(f"{model_name} CAPE or precipitable water is negative")
    return {
        "status": LIVE,
        "data": {
            "precipitation_24h_mm": float(sum(rain)),
            "gfs_mslp": at_valid["pressure_msl"],
            "gfs_t2m": at_valid["temperature_2m"],
            "gfs_rh2m": at_valid["relative_humidity_2m"],
            "gfs_u10": u10,
            "gfs_v10": v10,
            "gfs_u850": u850,
            "gfs_v850": v850,
            "gfs_cape": at_valid["cape"],
            "gfs_precipitable_water": at_valid["total_column_integrated_water_vapour"],
            "init_time": run_time.isoformat().replace("+00:00", "Z"),
            "valid_time": valid_time.isoformat().replace("+00:00", "Z"),
            "accumulation_start": window_start.isoformat().replace("+00:00", "Z"),
            "accumulation_end": valid_time.isoformat().replace("+00:00", "Z"),
            "lead_hours": 27,
        },
        "metadata": {
            "api_provider": "Open-Meteo Single Runs API",
            "upstream": "ECMWF Open Data",
            "model": model_name,
            "api_model_selector": model,
            "retrieval_url": SINGLE_RUNS_URL,
            "run_time": run_time.isoformat().replace("+00:00", "Z"),
            "precipitation_method": "sum of 24 preceding-hour amounts; exact UTC bounds checked",
            "precipitation_unit": "mm",
            "pressure_unit": "hPa",
            "temperature_unit": "degC",
            "wind_unit": "m s-1",
            "wind_conversion": "meteorological direction-from converted to eastward/northward components",
            "grid_latitude": payload.get("latitude"),
            "grid_longitude": payload.get("longitude"),
        },
        "error": None,
    }


class ECMWFProvider:
    source = "ECMWF IFS/AIFS open data via Open-Meteo Single Runs API"

    async def fetch(self, latitude: float, longitude: float, run_time: datetime) -> ProviderResult:
        run = run_time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        selectors = (
            (os.getenv("ANVAYA_ECMWF_IFS_MODEL", "ecmwf_ifs025"), "IFS 0.25-degree"),
            (os.getenv("ANVAYA_ECMWF_AIFS_MODEL", "ecmwf_aifs025"), "AIFS single 0.25-degree"),
        )
        results = await asyncio.gather(
            *(
                _fetch_model(
                    latitude=latitude,
                    longitude=longitude,
                    run_time=run,
                    model=selector,
                    model_name=name,
                )
                for selector, name in selectors
            ),
            return_exceptions=True,
        )
        models: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for (_, model_name), result in zip(selectors, results):
            if isinstance(result, BaseException):
                errors[model_name] = f"{type(result).__name__}: {result}"
            else:
                models[model_name] = result
        if not models:
            return ProviderResult(
                self.source,
                UNAVAILABLE,
                metadata={
                    "api_provider": "Open-Meteo Single Runs API",
                    "upstream": "ECMWF Open Data",
                    "run_time": run.isoformat().replace("+00:00", "Z"),
                    "retrieval_url": SINGLE_RUNS_URL,
                },
                error="; ".join(f"{name}: {error}" for name, error in errors.items()),
            )
        return ProviderResult(
            self.source,
            LIVE,
            data={"models": models},
            metadata={
                "api_provider": "Open-Meteo Single Runs API",
                "upstream": "ECMWF Open Data",
                "requested_run": run.isoformat().replace("+00:00", "Z"),
                "retrieval_url": SINGLE_RUNS_URL,
                "model_status": {name: ("LIVE" if name in models else "UNAVAILABLE") for _, name in selectors},
                "model_errors": errors,
                "precipitation_units": "mm",
                "nonhistorical_evaluation_note": "operational source fusion is separate from the frozen Phase 3F benchmark",
            },
            error=None,
        )
