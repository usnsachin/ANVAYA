"""Metadata-backed temporal alignment for archived GFS APCP accumulations.

The GFS APCP values are interval accumulations, not necessarily daily totals.
This module converts metadata-bounded records into non-overlapping increments
and sums only an exact IMD-style 24-hour target window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import xarray as xr


class GFSPrecipitationAlignmentError(ValueError):
    """Raised when APCP metadata cannot prove exact, safe temporal coverage."""


def _utc(value: datetime | np.datetime64) -> datetime:
    if isinstance(value, np.datetime64):
        value = datetime.fromisoformat(np.datetime_as_string(value, unit="s"))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class GFSAPCPRecord:
    """One GFS APCP value with its GRIB-derived temporal metadata."""

    values: xr.DataArray
    interval_start: datetime
    interval_end: datetime
    forecast_reference_time: datetime
    forecast_hour: int
    units: str
    statistical_process_type: str
    source_type: str = "forecast"

    def __post_init__(self) -> None:
        object.__setattr__(self, "interval_start", _utc(self.interval_start))
        object.__setattr__(self, "interval_end", _utc(self.interval_end))
        object.__setattr__(self, "forecast_reference_time", _utc(self.forecast_reference_time))

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset, variable: str, forecast_hour: int) -> "GFSAPCPRecord":
        """Read a single NCSS APCP subset, requiring CF time bounds and reftime."""
        if variable not in dataset:
            raise GFSPrecipitationAlignmentError(f"Missing requested APCP variable {variable!r}")
        field = dataset[variable]
        generation = str(field.attrs.get("Grib2_Generating_Process_Type", ""))
        if generation.casefold() != "forecast":
            raise GFSPrecipitationAlignmentError(
                f"GRIB generating-process type is {generation!r}, not Forecast"
            )
        time_coord = next(
            (name for name in field.coords if dataset[name].attrs.get("bounds")), None
        )
        if time_coord is None:
            raise GFSPrecipitationAlignmentError("APCP time coordinate has no metadata bounds")
        bounds_name = dataset[time_coord].attrs["bounds"]
        if bounds_name not in dataset:
            raise GFSPrecipitationAlignmentError(f"Missing APCP time-bounds variable {bounds_name!r}")
        bounds = np.asarray(dataset[bounds_name].values).reshape(-1)
        if bounds.size != 2:
            raise GFSPrecipitationAlignmentError("Expected exactly one APCP interval with two endpoints")
        if "reftime" not in dataset:
            raise GFSPrecipitationAlignmentError("Missing GRIB forecast reference time (reftime)")
        ref = np.asarray(dataset["reftime"].values).reshape(-1)
        if ref.size != 1:
            raise GFSPrecipitationAlignmentError("Expected one GRIB forecast reference time")
        stat = str(field.attrs.get("Grib2_Statistical_Process_Type", ""))
        units = str(field.attrs.get("units", ""))
        spatial = field.squeeze(drop=True)
        if set(spatial.dims) != {"latitude", "longitude"}:
            raise GFSPrecipitationAlignmentError(
                f"APCP value does not reduce to a two-dimensional latitude/longitude grid: {field.dims}"
            )
        spatial = spatial.transpose("latitude", "longitude")
        # Keep per-record temporal metadata on this record, not as auxiliary
        # scalar coordinates on the grid. Those time labels legitimately vary
        # by forecast lead and must not make identical spatial grids compare
        # unequal during nested accumulation differencing.
        values = xr.DataArray(
            np.asarray(spatial.values).copy(),
            dims=("latitude", "longitude"),
            coords={
                "latitude": np.asarray(spatial.latitude.values).copy(),
                "longitude": np.asarray(spatial.longitude.values).copy(),
            },
            name=field.name,
            attrs=dict(field.attrs),
        )
        return cls(
            values=values,
            interval_start=_utc(bounds[0]),
            interval_end=_utc(bounds[1]),
            forecast_reference_time=_utc(ref[0]),
            forecast_hour=int(forecast_hour),
            units=units,
            statistical_process_type=stat,
            source_type="forecast",
        )


@dataclass(frozen=True)
class DerivedPrecipitationInterval:
    """A disjoint increment, either direct or derived from nested bounds."""

    values: xr.DataArray
    interval_start: datetime
    interval_end: datetime
    derivation: str
    forecast_hours: tuple[int, ...]


def _is_water_equivalent_mm(units: str) -> bool:
    normalized = units.lower().replace(" ", "").replace("**", "^")
    return normalized in {
        "mm", "millimeter", "millimeters", "millimetre", "millimetres",
        "kg.m-2", "kgm-2", "kg/m2", "kgm^-2", "kg/m^2",
    }


def _as_mm(record: GFSAPCPRecord) -> xr.DataArray:
    if not _is_water_equivalent_mm(record.units):
        raise GFSPrecipitationAlignmentError(
            f"Unsupported APCP units {record.units!r}; expected mm or kg m-2 water equivalent"
        )
    if record.statistical_process_type.casefold() != "accumulation":
        raise GFSPrecipitationAlignmentError(
            f"APCP GRIB statistical processing is {record.statistical_process_type!r}, not Accumulation"
        )
    values = record.values.astype(float)
    finite = np.asarray(values.values)[np.isfinite(values.values)]
    if finite.size and float(finite.min()) < -1e-4:
        raise GFSPrecipitationAlignmentError("Negative raw APCP accumulation encountered")
    if finite.size and float(finite.max()) > 2000.0:
        raise GFSPrecipitationAlignmentError("Raw APCP accumulation exceeds the physical sanity limit of 2000 mm")
    return values


def build_24h_precipitation(
    records: Sequence[GFSAPCPRecord],
    target_start: datetime,
    target_end: datetime,
    init_time: datetime,
) -> xr.DataArray:
    """Build a daily rainfall field from APCP records using metadata bounds.

    Same-start nested accumulations (for example 00–03Z and 00–06Z) are
    differenced into the next disjoint interval. All other overlap is rejected.
    The resulting disjoint intervals must cover the target exactly, with no
    gaps or clipping. ``init_time`` is the forecast-reference time and is
    checked against every record; observations are not accepted as inputs.
    GFS ``kg m-2`` is numerically equivalent to mm liquid-water equivalent.
    """
    start, end, init = map(_utc, (target_start, target_end, init_time))
    if end - start != timedelta(hours=24):
        raise GFSPrecipitationAlignmentError("Target interval must be exactly 24 hours")
    if start < init:
        raise GFSPrecipitationAlignmentError("Target starts before forecast initialization; forecast cannot cover the full target")
    if not records:
        raise GFSPrecipitationAlignmentError("No GFS APCP records supplied")

    ordered = sorted(records, key=lambda r: (r.interval_end, r.interval_start, r.forecast_hour))
    ref_times = {r.forecast_reference_time for r in ordered}
    if ref_times != {init}:
        raise GFSPrecipitationAlignmentError("Forecast reference time mismatch; possible future/foreign-cycle data")

    # Build disjoint intervals from direct records or nested cumulative windows.
    derived: list[DerivedPrecipitationInterval] = []
    latest_by_start: dict[datetime, GFSAPCPRecord] = {}
    for record in ordered:
        if record.source_type != "forecast":
            raise GFSPrecipitationAlignmentError("Only forecast APCP records are allowed; observations are target-only")
        if record.interval_end <= record.interval_start:
            raise GFSPrecipitationAlignmentError("APCP interval end must follow its start")
        if record.interval_end - init != timedelta(hours=record.forecast_hour):
            raise GFSPrecipitationAlignmentError("Forecast hour disagrees with metadata-defined interval end")
        value = _as_mm(record)
        previous = latest_by_start.get(record.interval_start)
        if previous is not None:
            if record.interval_end <= previous.interval_end:
                raise GFSPrecipitationAlignmentError("Duplicate or reverse-ordered overlapping APCP accumulation")
            earlier = _as_mm(previous)
            if record.values.shape != previous.values.shape or not record.values.coords.equals(previous.values.coords):
                raise GFSPrecipitationAlignmentError("Nested APCP grids do not align exactly")
            increment = value - earlier
            finite = np.asarray(increment.values)[np.isfinite(increment.values)]
            if finite.size and float(finite.min()) < -0.05:
                raise GFSPrecipitationAlignmentError("Nested APCP difference is materially negative")
            increment = increment.where(increment >= 0, 0.0)
            derived.append(DerivedPrecipitationInterval(
                increment, previous.interval_end, record.interval_end,
                "nested_accumulation_difference", (previous.forecast_hour, record.forecast_hour),
            ))
        else:
            # A new raw interval must not overlap an already established one.
            if any(record.interval_start < item.interval_end and record.interval_end > item.interval_start
                   for item in derived):
                raise GFSPrecipitationAlignmentError("APCP intervals overlap without an exact shared accumulation start")
            derived.append(DerivedPrecipitationInterval(
                value, record.interval_start, record.interval_end, "direct_interval", (record.forecast_hour,),
            ))
        latest_by_start[record.interval_start] = record

    relevant = [item for item in derived if item.interval_end > start and item.interval_start < end]
    relevant.sort(key=lambda item: item.interval_start)
    cursor = start
    total: xr.DataArray | None = None
    for item in relevant:
        if item.interval_start != cursor:
            raise GFSPrecipitationAlignmentError(
                f"APCP coverage gap or overlap at {cursor.isoformat()}; next interval is "
                f"[{item.interval_start.isoformat()}, {item.interval_end.isoformat()}]"
            )
        if item.interval_end > end:
            raise GFSPrecipitationAlignmentError("APCP interval extends beyond the exact 24-hour target")
        if total is None:
            total = item.values.copy(deep=True)
        else:
            if total.shape != item.values.shape or not total.coords.equals(item.values.coords):
                raise GFSPrecipitationAlignmentError("APCP interval grids do not align exactly")
            total = total + item.values
        cursor = item.interval_end
    if cursor != end or total is None:
        raise GFSPrecipitationAlignmentError(
            f"APCP intervals end at {cursor.isoformat()}, not target end {end.isoformat()}"
        )

    total.name = "gfs_daily_rain_mm"
    total.attrs = {
        "long_name": "GFS precipitation accumulated over the exact IMD daily target window",
        "units": "mm",
        "source_product": "GFS GRIB2 APCP (parameter 0-1-8), statistical process Accumulation",
        "forecast_reference_time": init.isoformat().replace("+00:00", "Z"),
        "target_interval_start": start.isoformat().replace("+00:00", "Z"),
        "target_interval_end": end.isoformat().replace("+00:00", "Z"),
        "aggregation": "metadata-bounded disjoint intervals; nested shared-start accumulations differenced",
        "interval_count": len(relevant),
        "forecast_hours": ",".join(str(h) for item in relevant for h in item.forecast_hours),
    }
    return total


