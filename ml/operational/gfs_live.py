"""NOAA GFS live run adapter, queried through Open-Meteo's single-run API.

The API broker and upstream model are recorded separately in every response;
this adapter never substitutes a generic best-match forecast for GFS.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from ml.operational.common import LIVE, UNAVAILABLE, ProviderResult, as_finite_number, get_json


SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = (
    "precipitation,temperature_2m,relative_humidity_2m,pressure_msl,"
    "wind_speed_10m,wind_direction_10m,wind_speed_850hPa,"
    "wind_direction_850hPa,cape,total_column_integrated_water_vapour"
)


def select_gfs_00z_run(now: datetime | None = None) -> datetime:
    """Select the latest 00Z run after the configured processing-latency guard."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    try:
        latency = max(4, min(int(os.getenv("ANVAYA_GLOBAL_RUN_LATENCY_HOURS", "5")), 12))
    except ValueError:
        latency = 5
    candidate = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now < candidate + timedelta(hours=latency):
        candidate -= timedelta(days=1)
    return candidate


def _iso_hour(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _wind_components(speed: float, direction_from_degrees: float) -> tuple[float, float]:
    """Convert meteorological direction-from and m/s speed to east/north winds."""
    import math

    angle = math.radians(direction_from_degrees)
    return -speed * math.sin(angle), -speed * math.cos(angle)


def _series_value(hourly: dict[str, Any], key: str, index: int) -> float:
    values = hourly.get(key)
    if not isinstance(values, list) or index >= len(values):
        raise ValueError(f"GFS response lacks required hourly field {key}")
    return as_finite_number(values[index], label=key)


class GFSLiveProvider:
    source = "NOAA/NCEP GFS 0.25-degree family via Open-Meteo Single Runs API"

    async def fetch(
        self,
        latitude: float,
        longitude: float,
        run_time: datetime | None = None,
    ) -> ProviderResult:
        run = run_time or select_gfs_00z_run()
        run = run.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if run.hour != 0:
            return ProviderResult(self.source, UNAVAILABLE, error="Phase 3F live inference requires a 00Z GFS run")
        valid_time = run + timedelta(hours=27)
        window_start = valid_time - timedelta(hours=24)
        model = os.getenv("ANVAYA_GFS_MODEL", "gfs_global")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "models": model,
            "run": run.strftime("%Y-%m-%dT%H:%M"),
            "hourly": HOURLY_FIELDS,
            "forecast_hours": 28,
            "timezone": "GMT",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        try:
            payload = await get_json(SINGLE_RUNS_URL, params)
            hourly = payload.get("hourly")
            if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
                raise ValueError("GFS response has no hourly time coordinate")
            times = [_iso_hour(value) for value in hourly["time"]]
            time_index = {value: index for index, value in enumerate(times)}
            if len(time_index) != len(times):
                raise ValueError("GFS response contains duplicate hourly timestamps")
            if valid_time not in time_index:
                raise ValueError(f"GFS run lacks the required +27 h valid time {valid_time.isoformat()}")
            end_index = time_index[valid_time]
            interval_ends = [window_start + timedelta(hours=step) for step in range(1, 25)]
            if any(ending not in time_index for ending in interval_ends):
                raise ValueError("GFS hourly precipitation does not fully cover the 24-hour target window")
            rain = [
                _series_value(hourly, "precipitation", time_index[ending])
                for ending in interval_ends
            ]
            if any(value < 0 for value in rain):
                raise ValueError("GFS precipitation contains a negative accumulation")

            at_valid: dict[str, float] = {}
            for key in (
                "temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m",
                "wind_direction_10m", "wind_speed_850hPa", "wind_direction_850hPa", "cape",
                "total_column_integrated_water_vapour",
            ):
                at_valid[key] = _series_value(hourly, key, end_index)
            u10, v10 = _wind_components(at_valid["wind_speed_10m"], at_valid["wind_direction_10m"])
            u850, v850 = _wind_components(at_valid["wind_speed_850hPa"], at_valid["wind_direction_850hPa"])
            if not 0 <= at_valid["relative_humidity_2m"] <= 100:
                raise ValueError("GFS 2 m relative humidity is outside 0-100 percent")
            if at_valid["cape"] < 0 or at_valid["total_column_integrated_water_vapour"] < 0:
                raise ValueError("GFS CAPE or precipitable-water value is negative")
            data = {
                "gfs_daily_rain_mm": float(sum(rain)),
                "gfs_mslp": at_valid["pressure_msl"],
                "gfs_t2m": at_valid["temperature_2m"],
                "gfs_rh2m": at_valid["relative_humidity_2m"],
                "gfs_u10": u10,
                "gfs_v10": v10,
                "gfs_u850": u850,
                "gfs_v850": v850,
                "gfs_cape": at_valid["cape"],
                "gfs_precipitable_water": at_valid["total_column_integrated_water_vapour"],
                "init_time": run.isoformat().replace("+00:00", "Z"),
                "valid_time": valid_time.isoformat().replace("+00:00", "Z"),
                "lead_hours": 27,
                "accumulation_start": window_start.isoformat().replace("+00:00", "Z"),
                "accumulation_end": valid_time.isoformat().replace("+00:00", "Z"),
            }
            return ProviderResult(
                self.source,
                LIVE,
                data=data,
                metadata={
                    "api_provider": "Open-Meteo Single Runs API",
                    "upstream": "NOAA/NCEP GFS; API model selector " + model,
                    "retrieval_url": SINGLE_RUNS_URL,
                    "run_selection": "explicit 00Z single run after a configurable publication-latency guard",
                    "precipitation_method": "sum of 24 non-overlapping preceding-hour precipitation values; exact UTC bounds checked",
                    "precipitation_unit": "mm",
                    "pressure_unit": "hPa",
                    "temperature_unit": "degC",
                    "wind_unit": "m s-1",
                    "wind_conversion": "API wind speed and meteorological direction-from converted to eastward/northward components",
                    "grid_latitude": payload.get("latitude"),
                    "grid_longitude": payload.get("longitude"),
                    "grid_elevation_m": payload.get("elevation"),
                },
            )
        except Exception as exc:
            return ProviderResult(
                self.source,
                UNAVAILABLE,
                metadata={
                    "api_provider": "Open-Meteo Single Runs API",
                    "upstream": "NOAA/NCEP GFS",
                    "run_time": run.isoformat().replace("+00:00", "Z"),
                    "retrieval_url": SINGLE_RUNS_URL,
                },
                error=f"{type(exc).__name__}: {exc}",
            )
