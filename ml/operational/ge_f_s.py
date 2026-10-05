"""NOAA GEFS 0.25-degree ensemble adapter via Open-Meteo's ensemble API."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from ml.operational.common import LIVE, UNAVAILABLE, ProviderResult, as_finite_number, get_json


ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MEMBER_FIELD = re.compile(r"^precipitation_member_?(\d+)$", re.IGNORECASE)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class GEFSProvider:
    source = "NOAA/NCEP GEFS atmosphere 0.25-degree via Open-Meteo Ensemble API"

    async def fetch(
        self,
        latitude: float,
        longitude: float,
        target_valid_time: datetime,
    ) -> ProviderResult:
        target_valid_time = target_valid_time.astimezone(timezone.utc)
        window_start = target_valid_time - timedelta(hours=24)
        model = os.getenv("ANVAYA_GEFS_MODEL", "ncep_gefs025")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "models": model,
            "hourly": "precipitation",
            "forecast_days": 10,
            "timezone": "GMT",
            "precipitation_unit": "mm",
        }
        try:
            payload = await get_json(ENSEMBLE_URL, params)
            hourly = payload.get("hourly")
            if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
                raise ValueError("GEFS response has no hourly time coordinate")
            times = [_parse_time(value) for value in hourly["time"]]
            if len(set(times)) != len(times):
                raise ValueError("GEFS response contains duplicate hourly timestamps")
            member_keys = sorted(
                (key for key in hourly if MEMBER_FIELD.match(key)),
                key=lambda key: int(MEMBER_FIELD.match(key).group(1)),
            )
            if len(member_keys) < 2:
                raise ValueError(
                    "GEFS response did not include at least two explicit precipitation members; refusing to invent spread"
                )
            index_by_time = {value: index for index, value in enumerate(times)}
            required_ends = [window_start + timedelta(hours=step) for step in range(1, 25)]
            if any(value not in index_by_time for value in required_ends):
                raise ValueError("GEFS member precipitation does not fully cover the requested 24-hour UTC window")
            member_totals: list[float] = []
            for key in member_keys:
                series = hourly[key]
                if not isinstance(series, list):
                    raise ValueError(f"GEFS member field {key} is not a time series")
                increments = [
                    as_finite_number(series[index_by_time[ending]], label=key)
                    for ending in required_ends
                ]
                if any(value < 0 for value in increments):
                    raise ValueError(f"GEFS member field {key} contains negative precipitation")
                member_totals.append(float(sum(increments)))
            ensemble_mean = float(mean(member_totals))
            ensemble_spread = float(pstdev(member_totals))
            return ProviderResult(
                self.source,
                LIVE,
                data={
                    "precipitation_24h_mean_mm": ensemble_mean,
                    "precipitation_24h_spread_mm": ensemble_spread,
                    "precipitation_24h_member_min_mm": min(member_totals),
                    "precipitation_24h_member_max_mm": max(member_totals),
                    "member_count": len(member_totals),
                    "valid_time": target_valid_time.isoformat().replace("+00:00", "Z"),
                    "accumulation_start": window_start.isoformat().replace("+00:00", "Z"),
                    "accumulation_end": target_valid_time.isoformat().replace("+00:00", "Z"),
                },
                metadata={
                    "api_provider": "Open-Meteo Ensemble API",
                    "upstream": "NOAA/NCEP GEFS 0.25 degree",
                    "api_model_selector": model,
                    "retrieval_url": ENSEMBLE_URL,
                    "precipitation_unit": "mm",
                    "spread_definition": "population standard deviation across explicitly returned member 24-hour totals",
                    "temporal_method": "sum 24 hourly preceding-interval API values at exact UTC interval ends; source model output is 3-hourly and API time-step handling is provider-managed",
                    "grid_latitude": payload.get("latitude"),
                    "grid_longitude": payload.get("longitude"),
                },
            )
        except Exception as exc:
            return ProviderResult(
                self.source,
                UNAVAILABLE,
                metadata={
                    "api_provider": "Open-Meteo Ensemble API",
                    "upstream": "NOAA/NCEP GEFS 0.25 degree",
                    "api_model_selector": model,
                    "retrieval_url": ENSEMBLE_URL,
                    "valid_time": target_valid_time.isoformat().replace("+00:00", "Z"),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
