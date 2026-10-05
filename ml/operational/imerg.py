"""NASA GPM IMERG Early Run point-subset adapter using GES DISC OPeNDAP.

Only one point from each of the 48 half-hourly granules is requested. The
adapter requires NASA Earthdata credentials and fails closed on absent files,
missing intervals, unknown units, or an unrecognized OPeNDAP schema.
"""
from __future__ import annotations

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ml.operational.common import LIVE, UNAVAILABLE, ProviderResult, request_timeout


GES_DISC_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHHE.07"
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _numeric_payload(text: str) -> list[float]:
    marker = "data:"
    offset = text.lower().find(marker)
    if offset < 0:
        raise ValueError("NASA OPeNDAP ASCII response has no Data section")
    return [float(value) for value in FLOAT_PATTERN.findall(text[offset + len(marker):])]


def _variable_schema(dmr_text: str) -> tuple[list[str], str]:
    root = ET.fromstring(dmr_text)
    variable = None
    for element in root.iter():
        if element.attrib.get("name") == "precipitationCal":
            variable = element
            break
    if variable is None:
        raise ValueError("IMERG OPeNDAP metadata lacks Grid/precipitationCal")
    dimensions = [
        (node.attrib.get("name") or (node.text or "")).split("/")[-1]
        for node in variable if node.tag.endswith("Dim")
    ]
    if not dimensions:
        raise ValueError("IMERG precipitation variable has no dimensions in metadata")
    units = None
    for attribute in variable.iter():
        if attribute.attrib.get("name", "").lower() == "units":
            units = attribute.attrib.get("value") or next(
                (child.text for child in attribute if child.text), None
            )
            break
    if not units:
        raise ValueError("IMERG precipitationCal units are absent from OPeNDAP metadata")
    normalized = units.lower().replace(" ", "")
    if not ("mm/hr" in normalized or "mmh-1" in normalized or "mmhour-1" in normalized):
        raise ValueError(f"Unsupported IMERG precipitation rate units: {units}")
    return dimensions, units


def _point_index(values: list[float], target: float, *, axis: str) -> int:
    if not values:
        raise ValueError(f"IMERG {axis} coordinate vector is empty")
    index = min(range(len(values)), key=lambda item: abs(values[item] - target))
    if abs(values[index] - target) > 0.051:
        raise ValueError(f"IMERG {axis} coordinate coverage is inconsistent with the requested point")
    return index


class IMERGProvider:
    source = "NASA GPM IMERG Early Run V07B via NASA GES DISC"

    @staticmethod
    def _credentials() -> tuple[dict[str, str], Any]:
        token = os.getenv("NASA_EARTHDATA_TOKEN") or os.getenv("EARTHDATA_TOKEN")
        username = os.getenv("NASA_EARTHDATA_USERNAME") or os.getenv("EARTHDATA_USERNAME")
        password = os.getenv("NASA_EARTHDATA_PASSWORD") or os.getenv("EARTHDATA_PASSWORD")
        if token:
            return {"Authorization": f"Bearer {token}"}, None
        if username and password:
            return {}, (username, password)
        return {}, None

    @staticmethod
    def _granule_url(start: datetime) -> str:
        year = start.strftime("%Y")
        day_of_year = start.strftime("%j")
        date = start.strftime("%Y%m%d")
        minute_of_day = start.hour * 60 + start.minute
        end = start + timedelta(minutes=29, seconds=59)
        name = (
            f"3B-HHR.MS.MRG.3IMERG.{date}-S{start:%H%M%S}-E{end:%H%M%S}."
            f"{minute_of_day:04d}.V07B.HDF5"
        )
        return f"{GES_DISC_BASE}/{year}/{day_of_year}/{name}"

    async def fetch(
        self,
        latitude: float,
        longitude: float,
        *,
        target_start: datetime,
        now: datetime | None = None,
    ) -> ProviderResult:
        now = now or datetime.now(timezone.utc)
        now = now.astimezone(timezone.utc)
        headers, auth = self._credentials()
        if auth is None and not headers:
            return ProviderResult(
                self.source,
                UNAVAILABLE,
                metadata={
                    "product": "GPM_3IMERGHHE V07B (Early Run, half-hourly)",
                    "provider": "NASA GES DISC",
                    "requires_authentication": True,
                    "latency_hours": 4,
                },
                error="NASA Earthdata credentials are not configured",
            )

        # Respect IMERG Early's nominal four-hour delivery lag, and do not use
        # any observation whose interval reaches into the forecast target.
        aligned_now = now - timedelta(hours=4)
        aligned_now = aligned_now.replace(minute=(aligned_now.minute // 30) * 30, second=0, microsecond=0)
        target_start = target_start.astimezone(timezone.utc)
        window_end = min(aligned_now, target_start)
        window_start = window_end - timedelta(hours=24)
        starts = [window_start + timedelta(minutes=30 * index) for index in range(48)]

        timeout = request_timeout()
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "ANVAYA-operational/1.0", **headers},
                auth=auth,
            ) as client:
                first_url = self._granule_url(starts[0])
                dmr_response = await client.get(first_url + ".dmr")
                dmr_response.raise_for_status()
                dimensions, units = _variable_schema(dmr_response.text)

                async def coordinate(name: str) -> list[float]:
                    dmr = ET.fromstring(dmr_response.text)
                    size = None
                    for element in dmr.iter():
                        if element.attrib.get("name", "").split("/")[-1] == name and element.tag.endswith("Dimension"):
                            size = int(element.attrib["size"])
                            break
                    if size is None:
                        # DAP metadata schemas may encode coordinate vector size
                        # on the variable's Dim nodes instead of Dimension nodes.
                        coord_var = next(
                            (element for element in dmr.iter() if element.attrib.get("name") == name),
                            None,
                        )
                        if coord_var is None:
                            raise ValueError(f"IMERG OPeNDAP metadata lacks {name} coordinate")
                        dims = [node for node in coord_var if node.tag.endswith("Dim")]
                        size = int(dims[0].attrib.get("size", "0")) if dims else 0
                    if size <= 0:
                        raise ValueError(f"Unable to determine IMERG {name} coordinate size")
                    response = await client.get(first_url + f".ascii?Grid/{name}[0:{size - 1}]")
                    response.raise_for_status()
                    return _numeric_payload(response.text)

                latitudes, longitudes = await asyncio.gather(coordinate("lat"), coordinate("lon"))
                lat_index = _point_index(latitudes, latitude, axis="latitude")
                normalized_lon = longitude if longitude <= 180 else longitude - 360
                lon_index = _point_index(longitudes, normalized_lon, axis="longitude")
                if "time" in dimensions:
                    time_index = dimensions.index("time")
                else:
                    raise ValueError("IMERG precipitationCal has no time dimension")
                if "lat" not in dimensions or "lon" not in dimensions:
                    raise ValueError(f"Unrecognized IMERG precipitation dimensions: {dimensions}")
                indices = []
                for name in dimensions:
                    if name == "time":
                        indices.append(0)
                    elif name == "lat":
                        indices.append(lat_index)
                    elif name == "lon":
                        indices.append(lon_index)
                    else:
                        raise ValueError(f"Unsupported IMERG precipitation dimension: {name}")
                constraint = "".join(f"[{index}]" for index in indices)
                semaphore = asyncio.Semaphore(8)

                async def read_rate(start: datetime) -> float:
                    url = self._granule_url(start)
                    async with semaphore:
                        response = await client.get(url + f".ascii?Grid/precipitationCal{constraint}")
                    response.raise_for_status()
                    values = _numeric_payload(response.text)
                    if len(values) != 1:
                        raise ValueError(f"IMERG point subset returned {len(values)} values instead of one")
                    rate = values[0]
                    if rate < 0:
                        raise ValueError("IMERG point value is negative or a missing-value sentinel")
                    return rate

                rates = await asyncio.gather(*(read_rate(start) for start in starts))
            # precipitationCal is a half-hourly mean rate in mm/hour.
            amount_mm = float(sum(rates) * 0.5)
            return ProviderResult(
                self.source,
                LIVE,
                data={
                    "precipitation_24h_mm": amount_mm,
                    "window_start": window_start.isoformat().replace("+00:00", "Z"),
                    "window_end": window_end.isoformat().replace("+00:00", "Z"),
                    "rate_sample_count": len(rates),
                    "grid_latitude": latitudes[lat_index],
                    "grid_longitude": longitudes[lon_index],
                },
                metadata={
                    "provider": "NASA GES DISC OPeNDAP",
                    "product": "GPM_3IMERGHHE V07B Early Run",
                    "granule_variable": "Grid/precipitationCal",
                    "granule_count": len(starts),
                    "retrieval_mode": "server-side OPeNDAP point subset; no global granules downloaded",
                    "native_units": units,
                    "output_units": "mm per exact 24-hour window",
                    "conversion": "sum of 48 half-hour samples; mm/hour rate multiplied by 0.5 hour",
                    "nominal_latency_hours": 4,
                    "model_feature_use": "operational proxy for the Phase 3F antecedent 24-hour rain feature; value is verified to end before forecast target start; source/time-distribution shift is not benchmark-validated",
                    "forecast_target_start": target_start.isoformat().replace("+00:00", "Z"),
                    "nasa_access_url": "https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHHE_07/summary",
                },
            )
        except Exception as exc:
            return ProviderResult(
                self.source,
                UNAVAILABLE,
                metadata={
                    "provider": "NASA GES DISC OPeNDAP",
                    "product": "GPM_3IMERGHHE V07B Early Run",
                    "window_start": window_start.isoformat().replace("+00:00", "Z"),
                    "window_end": window_end.isoformat().replace("+00:00", "Z"),
                    "retrieval_mode": "server-side OPeNDAP point subset",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
