"""Offline safeguards and bounded real-case validation for GFS APCP alignment.

Run deterministic unit tests with:
    backend/venv/Scripts/python.exe scripts/test_gfs_precip_alignment.py

Run the four explicitly bounded UCAR cases (about 36 small NCSS APCP queries):
    backend/venv/Scripts/python.exe scripts/test_gfs_precip_alignment.py --real
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import requests
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.gfs_precip import (  # noqa: E402
    GFSAPCPRecord,
    GFSPrecipitationAlignmentError,
    build_24h_precipitation,
)


APCP_PARAM = "0 1 8"
BBOX = {"south": 8.0, "north": 37.0, "west": 68.0, "east": 97.0}
CASES = (
    {"init": "2024-07-15T00:00:00Z", "target_date": "2024-07-16", "cycle": "00"},
    {"init": "2024-07-16T00:00:00Z", "target_date": "2024-07-17", "cycle": "00"},
    {"init": "2024-08-15T00:00:00Z", "target_date": "2024-08-16", "cycle": "00"},
    # The next complete window for this 06Z cycle ends at Jul 17 03Z (lead 45h).
    {"init": "2024-07-15T06:00:00Z", "target_date": "2024-07-17", "cycle": "06"},
)


def utc(text: str | datetime | np.datetime64) -> datetime:
    if isinstance(text, np.datetime64):
        text = datetime.fromisoformat(np.datetime_as_string(text, unit="s"))
    elif isinstance(text, str):
        text = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return text.replace(tzinfo=timezone.utc) if text.tzinfo is None else text.astimezone(timezone.utc)


def _source_template() -> str:
    config = yaml.safe_load((ROOT / "config/data_sources.yaml").read_text(encoding="utf-8"))
    source = next(item for item in config["sources"] if item["id"] == "gfs_0p25")
    return source["historical_archive"]["ucar_ncss_template"]


def _archive_file_url(init: datetime, lead: int) -> str:
    date = init.strftime("%Y%m%d")
    return _source_template().format(
        YYYY=init.strftime("%Y"), YYYYMMDD=date, CC=init.strftime("%H"), FFF=f"{lead:03d}"
    )


def _metadata_candidates(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    found: list[dict[str, str]] = []
    for grid in (node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "grid"):
        attrs: dict[str, str] = {"name": grid.attrib.get("name", "")}
        for node in grid.iter():
            if node.tag.rsplit("}", 1)[-1] == "attribute" and node.attrib.get("name"):
                attrs[node.attrib["name"]] = node.attrib.get("value", "")
        if attrs.get("Grib2_Parameter") == APCP_PARAM and attrs.get("Grib2_Level_Type") == "1":
            found.append(attrs)
    return found


def _choose_apcp_variable(candidates: list[dict[str, str]], lead: int) -> str:
    accumulated = [
        item for item in candidates
        if item.get("Grib2_Statistical_Process_Type") == "Accumulation"
        and item.get("units") == "kg.m-2"
    ]
    preferred = (
        "3_Hour_Accumulation" if lead == 3 else
        "6_Hour_Accumulation" if lead == 6 else
        "Mixed_intervals_Accumulation"
    )
    matches = [item["name"] for item in accumulated if item["name"].endswith(preferred)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Metadata did not identify exactly one APCP Accumulation message for f{lead:03d}; "
            f"expected suffix {preferred!r}, found {[item['name'] for item in accumulated]}"
        )
    return matches[0]


def _query_record(session: requests.Session, init: datetime, lead: int) -> tuple[GFSAPCPRecord, dict[str, Any]]:
    dataset_url = _archive_file_url(init, lead)
    meta_response = session.get(dataset_url + "/dataset.xml", timeout=90)
    meta_response.raise_for_status()
    if len(meta_response.content) > 10_000_000:
        raise RuntimeError("UCAR dataset metadata response is unexpectedly large")
    candidates = _metadata_candidates(meta_response.content)
    variable = _choose_apcp_variable(candidates, lead)
    valid = init + timedelta(hours=lead)
    params = {
        "var": variable,
        "north": BBOX["north"], "south": BBOX["south"],
        "west": BBOX["west"], "east": BBOX["east"],
        "accept": "netcdf3",
        "time": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    response = session.get(dataset_url, params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"NCSS f{lead:03d} failed ({response.status_code}): {response.text[:400]}")
    if len(response.content) > 2_000_000:
        raise RuntimeError("A bounded APCP response exceeded 2 MB; refusing unexpectedly large data")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "apcp_subset.nc"
        temp_path.write_bytes(response.content)
        with xr.open_dataset(temp_path, engine="netcdf4") as opened:
            dataset = opened.load()
    record = GFSAPCPRecord.from_dataset(dataset, variable, lead)
    lat, lon = np.asarray(dataset.latitude.values), np.asarray(dataset.longitude.values)
    if (lat.size, lon.size) != (117, 117):
        raise RuntimeError(f"Unexpected India-grid dimensions {(lat.size, lon.size)} at f{lead:03d}")
    if lat.min() < BBOX["south"] or lat.max() > BBOX["north"] or lon.min() < BBOX["west"] or lon.max() > BBOX["east"]:
        raise RuntimeError("NCSS returned coordinates outside the requested India bbox")
    if record.interval_end != valid:
        raise RuntimeError(
            f"Metadata APCP bound end {record.interval_end.isoformat()} != requested valid time {valid.isoformat()}"
        )
    if record.units != "kg.m-2" or record.statistical_process_type != "Accumulation":
        raise RuntimeError("APCP units/statistical processing differ from metadata-backed expectations")
    finite = np.asarray(record.values.values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size or float(finite.min()) < -1e-5 or float(finite.max()) > 2000.0:
        raise RuntimeError("APCP values are missing or outside the physical sanity range")
    info = {
        "forecast_hour": lead,
        "variable": variable,
        "parameter": APCP_PARAM,
        "level_type": "1",
        "units": record.units,
        "statistical_process_type": record.statistical_process_type,
        "accumulation_start": record.interval_start.isoformat().replace("+00:00", "Z"),
        "accumulation_end": record.interval_end.isoformat().replace("+00:00", "Z"),
        "duration_hours": (record.interval_end - record.interval_start).total_seconds() / 3600,
        "domain_mean_value": float(record.values.mean(skipna=True)),
        "domain_max_value": float(record.values.max(skipna=True)),
        "subset_bytes": len(response.content),
        "retrieval_url": response.url,
        "metadata_url": dataset_url + "/dataset.xml",
        "subset_sha256": hashlib.sha256(response.content).hexdigest(),
        "grid_shape": [int(lat.size), int(lon.size)],
        "latitude_bounds": [float(lat.min()), float(lat.max())],
        "longitude_bounds": [float(lon.min()), float(lon.max())],
    }
    return record, info


def _interval_descriptions(records: list[GFSAPCPRecord], target_start: datetime, target_end: datetime) -> list[dict[str, Any]]:
    """Mirror the nested-window subtraction for a printable, independently checked audit."""
    ordered = sorted(records, key=lambda r: (r.interval_end, r.interval_start, r.forecast_hour))
    result: list[dict[str, Any]] = []
    latest: dict[datetime, GFSAPCPRecord] = {}
    for record in ordered:
        prior = latest.get(record.interval_start)
        if prior is None:
            start, end, values = record.interval_start, record.interval_end, record.values
            method, hours = "direct APCP interval", [record.forecast_hour]
        else:
            start, end, values = prior.interval_end, record.interval_end, record.values - prior.values
            method, hours = "nested accumulation difference", [prior.forecast_hour, record.forecast_hour]
        latest[record.interval_start] = record
        if end <= target_start or start >= target_end:
            continue
        if start < target_start or end > target_end:
            continue
        result.append({
            "forecast_hours": hours,
            "start": start,
            "end": end,
            "duration_hours": (end - start).total_seconds() / 3600,
            "method": method,
            "domain_mean_mm": float(values.mean(skipna=True)),
        })
    return sorted(result, key=lambda item: item["start"])


def _find_imd_file(target_date: str) -> Path:
    day = np.datetime64(target_date)
    for path in sorted((ROOT / "data/raw/imd").glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            if "rain" in dataset and "time" in dataset.coords and day in dataset.time.values:
                return path
    raise FileNotFoundError(f"No local IMD rain record for {target_date}")


def _compare_imd(daily: xr.DataArray, target_date: str) -> dict[str, Any]:
    target_end = utc(target_date + "T03:00:00Z")
    target_start = target_end - timedelta(hours=24)
    try:
        path = _find_imd_file(target_date)
    except FileNotFoundError:
        return {
            "status": "NO_LOCAL_IMD_RECORD_FOR_DATE",
            "imd_file": None,
            "imd_time_coordinate": None,
            "imd_target_start": target_start.isoformat().replace("+00:00", "Z"),
            "imd_target_end": target_end.isoformat().replace("+00:00", "Z"),
            "imd_rainfall": None,
            "note": "No corresponding local IMD NetCDF record exists; no substitute date was used.",
        }
    with xr.open_dataset(path) as opened:
        imd = opened["rain"].sel(time=np.datetime64(target_date)).load()
        dataset = opened.load()
    # Existing project convention: local midnight date D labels the daily total ending D 03Z.
    if imd.latitude.size not in (59, 117) or imd.longitude.size not in (59, 117):
        raise RuntimeError(f"Unexpected local IMD grid shape {imd.shape}")
    # Local IMD episodes are 0.5 degree; GFS is 0.25 degree. Compare shared centers.
    gfs_on_imd = daily.sel(latitude=imd.latitude, longitude=imd.longitude, method="nearest")
    if not np.allclose(gfs_on_imd.latitude.values, imd.latitude.values) or not np.allclose(
        gfs_on_imd.longitude.values, imd.longitude.values
    ):
        raise RuntimeError("Could not align GFS and local IMD grid centers")
    gfs_values, imd_values = np.asarray(gfs_on_imd.values, dtype=float), np.asarray(imd.values, dtype=float)
    common = np.isfinite(gfs_values) & np.isfinite(imd_values)
    if not common.any():
        raise RuntimeError("No shared finite GFS/IMD cells for numerical comparison")
    imd_units = imd.attrs.get("units") or dataset["rain"].attrs.get("units")
    return {
        "status": "NUMERIC_COMPARISON_AVAILABLE",
        "imd_file": str(path.relative_to(ROOT)),
        "imd_time_coordinate": target_date + "T00:00:00 (naive local daily label)",
        "imd_target_start": (utc(target_date + "T03:00:00Z") - timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
        "imd_target_end": target_date + "T03:00:00Z",
        "imd_units_metadata": imd_units,
        "imd_units_status": "verified_from_file_metadata" if imd_units else "MISSING_FROM_LOCAL_FILE; project convention treats rain as mm, not independently verified",
        "gfs_grid_shape": list(daily.shape),
        "imd_grid_shape": list(imd.shape),
        "comparison_grid_shape": list(gfs_on_imd.shape),
        "shared_finite_cells": int(common.sum()),
        "gfs_region_mean_mm": float(gfs_values[common].mean()),
        "imd_region_mean_value": float(imd_values[common].mean()),
        "gfs_region_max_mm": float(gfs_values[common].max()),
        "imd_region_max_value": float(imd_values[common].max()),
        "reference_cell_latitude": float(imd.latitude.values[np.argmin(np.abs(imd.latitude.values - 20.0))]),
        "reference_cell_longitude": float(imd.longitude.values[np.argmin(np.abs(imd.longitude.values - 78.0))]),
        "gfs_reference_cell_mm": float(gfs_on_imd.sel(latitude=20.0, longitude=78.0, method="nearest")),
        "imd_reference_cell_value": float(imd.sel(latitude=20.0, longitude=78.0, method="nearest")),
    }


def run_real_cases() -> dict[str, Any]:
    """Retrieve only the four specified test windows, aggregate, compare, and save."""
    out_dir = ROOT / "data/raw/gfs/test_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest: dict[str, Any] = {
        "status": "RUNNING_SINGLE_CASES_ONLY",
        "retrieved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": "UCAR GDEX d084001 historical GFS via THREDDS NCSS",
        "source_dataset": "NCEP GFS 0.25 Degree Global Forecast Grids",
        "bbox": BBOX,
        "longitude_convention": "0_to_360 degrees_east; bbox west=68 east=97",
        "target_definition": "IMD daily rainfall for date D is a 24-hour accumulation ending at D 03:00 UTC; interval [D-1 03:00Z, D 03:00Z]",
        "aggregation_method": "use time-bounds and GRIB stat type; convert same-start nested APCP accumulations to disjoint increments by subtraction, then sum only exact target coverage",
        "bulk_download_started": False,
        "cases": [],
    }
    for case in CASES:
        init = utc(case["init"])
        target_end = utc(case["target_date"] + "T03:00:00Z")
        target_start = target_end - timedelta(hours=24)
        first_lead = int((target_start - init).total_seconds() / 3600)
        last_lead = int((target_end - init).total_seconds() / 3600)
        if first_lead <= 0 or last_lead > 240 or first_lead % 3 or last_lead % 3:
            raise RuntimeError(f"Unsupported test-case lead window {first_lead}..{last_lead} hours")
        leads = list(range(first_lead, last_lead + 1, 3))
        print(f"\nCASE init={init.isoformat()} target=[{target_start.isoformat()}, {target_end.isoformat()}] leads={leads}", flush=True)
        records: list[GFSAPCPRecord] = []
        raw_manifest: list[dict[str, Any]] = []
        for lead in leads:
            record, info = _query_record(session, init, lead)
            records.append(record)
            raw_manifest.append(info)
            print(
                f"  f{lead:03d} {info['accumulation_start']}..{info['accumulation_end']} "
                f"{info['duration_hours']:g}h APCP mean={info['domain_mean_value']:.3f} {info['units']} "
                f"({info['variable']})",
                flush=True,
            )
        daily = build_24h_precipitation(records, target_start, target_end, init)
        derived = _interval_descriptions(records, target_start, target_end)
        expected_start = target_start
        for item in derived:
            if item["start"] != expected_start:
                raise RuntimeError(f"Audit interval gap/overlap: expected {expected_start}, got {item['start']}")
            expected_start = item["end"]
        if expected_start != target_end or sum(item["duration_hours"] for item in derived) != 24:
            raise RuntimeError("Derived disjoint increments do not cover exactly 24 hours")
        imd = _compare_imd(daily, case["target_date"])
        if imd["imd_target_start"] != daily.attrs["target_interval_start"] or imd["imd_target_end"] != daily.attrs["target_interval_end"]:
            raise RuntimeError("GFS forecast window does not exactly match the IMD target definition")
        slug = f"gfs_apcp_alignment_{init:%Y%m%d}_{init:%H}"
        path = out_dir / f"{slug}_india.nc"
        raw_data = np.stack([np.asarray(r.values.values, dtype=np.float32) for r in records])
        bounds = np.array(
            [[np.datetime64(r.interval_start.replace(tzinfo=None)), np.datetime64(r.interval_end.replace(tzinfo=None))]
             for r in records], dtype="datetime64[ns]"
        )
        result_ds = xr.Dataset(
            {
                "gfs_apcp_interval_mm": (("record", "latitude", "longitude"), raw_data),
                "apcp_time_bounds": (("record", "bounds"), bounds),
                "gfs_daily_rain_mm": daily.astype(np.float32),
            },
            coords={
                "record": ("record", np.array(leads, dtype=np.int16)),
                "latitude": records[0].values.latitude.values,
                "longitude": records[0].values.longitude.values,
                "bounds": [0, 1],
                "reftime": np.datetime64(init.replace(tzinfo=None)),
            },
            attrs={
                "source": manifest["source"],
                "forecast_reference_time": init.isoformat().replace("+00:00", "Z"),
                "target_start": target_start.isoformat().replace("+00:00", "Z"),
                "target_end": target_end.isoformat().replace("+00:00", "Z"),
                "bbox_south": BBOX["south"], "bbox_north": BBOX["north"],
                "bbox_west": BBOX["west"], "bbox_east": BBOX["east"],
                "aggregation": "metadata-defined nested APCP differences converted to non-overlapping increments; exact 24h sum",
            },
        )
        result_ds["gfs_apcp_interval_mm"].attrs.update({
            "long_name": "Raw GFS APCP records with their own metadata-defined accumulation bounds (records may overlap)",
            "units": "mm",
            "source_units": "kg.m-2",
            "Grib2_Parameter": APCP_PARAM,
            "Grib2_Statistical_Process_Type": "Accumulation",
            "record_variable_names": json.dumps([item["variable"] for item in raw_manifest]),
        })
        result_ds["apcp_time_bounds"].attrs["long_name"] = "Metadata-defined accumulation interval start/end for each record"
        result_ds["gfs_daily_rain_mm"].attrs.update(daily.attrs)
        result_ds.to_netcdf(path, engine="netcdf4")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        case_report = {
            "initialization": init.isoformat().replace("+00:00", "Z"),
            "cycle_utc": init.strftime("%H"),
            "valid_time": target_end.isoformat().replace("+00:00", "Z"),
            "target_date": case["target_date"],
            "target_start": target_start.isoformat().replace("+00:00", "Z"),
            "target_end": target_end.isoformat().replace("+00:00", "Z"),
            "first_forecast_lead": first_lead,
            "last_forecast_lead": last_lead,
            "valid_time_equals_init_plus_lead": target_end == init + timedelta(hours=last_lead),
            "raw_apcp_records": raw_manifest,
            "non_overlapping_aggregation_intervals": [
                {**item, "start": item["start"].isoformat().replace("+00:00", "Z"),
                 "end": item["end"].isoformat().replace("+00:00", "Z")}
                for item in derived
            ],
            "gfs_daily_rain_region_mean_mm": float(daily.mean(skipna=True)),
            "imd_comparison": imd,
            "output": {
                "path": str(path.relative_to(ROOT)),
                "file_size_bytes": path.stat().st_size,
                "sha256": checksum,
            },
            "validation": {
                "exact_24h": (target_end - target_start) == timedelta(hours=24),
                "non_overlapping_interval_coverage": True,
                "gfs_units_verified_kg_m2": all(item["units"] == "kg.m-2" for item in raw_manifest),
                "statistical_process_verified_accumulation": all(item["statistical_process_type"] == "Accumulation" for item in raw_manifest),
                "rainfall_finite_and_nonnegative": bool(np.isfinite(daily.values).any() and float(daily.min()) >= 0),
                "rainfall_max_below_2000mm": float(daily.max()) <= 2000.0,
                "requested_domain_verified": all(item["grid_shape"] == [117, 117] for item in raw_manifest),
            },
        }
        manifest["cases"].append(case_report)
        if imd["status"] == "NUMERIC_COMPARISON_AVAILABLE":
            comparison_text = (
                f"IMD domain mean={imd['imd_region_mean_value']:.3f} "
                f"{imd['imd_units_metadata'] or '[units absent in local file]'}; "
                f"GFS/IMD reference cell={imd['gfs_reference_cell_mm']:.2f}/{imd['imd_reference_cell_value']:.2f}"
            )
        else:
            comparison_text = "IMD rainfall=UNAVAILABLE (no local record for this date)"
        print(
            f"  MATCH {target_start:%Y-%m-%d %H:%MZ}..{target_end:%Y-%m-%d %H:%MZ} "
            f"GFS domain mean={case_report['gfs_daily_rain_region_mean_mm']:.3f} mm; "
            f"{comparison_text}; file={path.stat().st_size:,} bytes",
            flush=True,
        )
    manifest["status"] = "GFS_ALIGNMENT_VALIDATED_LOCAL_IMD_VALUE_COVERAGE_INCOMPLETE"
    manifest_path = ROOT / "data/raw/manifests/gfs_precip_alignment_test.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path.relative_to(ROOT)}", flush=True)
    return manifest


import unittest


class GFSPrecipitationAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.init = datetime(2024, 7, 15, tzinfo=timezone.utc)
        self.start = self.init + timedelta(hours=3)
        self.end = self.start + timedelta(hours=24)
        self.coords = {"latitude": [20.0, 20.25], "longitude": [78.0, 78.25]}

    def record(self, start_h: int, end_h: int, total: float, lead: int | None = None, **kw: Any) -> GFSAPCPRecord:
        values = xr.DataArray(
            np.full((2, 2), total, dtype=float), dims=("latitude", "longitude"), coords=self.coords
        )
        return GFSAPCPRecord(
            values=values,
            interval_start=self.init + timedelta(hours=start_h),
            interval_end=self.init + timedelta(hours=end_h),
            forecast_reference_time=kw.get("forecast_reference_time", self.init),
            forecast_hour=end_h if lead is None else lead,
            units=kw.get("units", "kg.m-2"),
            statistical_process_type=kw.get("stat", "Accumulation"),
            source_type=kw.get("source_type", "forecast"),
        )

    def test_exact_24h_target_and_nonoverlap_after_nested_difference(self) -> None:
        records = [
            self.record(0, 3, 2), self.record(0, 6, 5),
            self.record(6, 9, 1), self.record(6, 12, 4),
            self.record(12, 15, 2), self.record(12, 18, 5),
            self.record(18, 21, 1), self.record(18, 24, 4),
            self.record(24, 27, 3),
        ]
        daily = build_24h_precipitation(records, self.start, self.end, self.init)
        self.assertEqual(daily.name, "gfs_daily_rain_mm")
        self.assertEqual(daily.attrs["units"], "mm")
        np.testing.assert_allclose(daily.values, 19.0)

    def test_rejects_target_window_not_exactly_24h(self) -> None:
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "exactly 24 hours"):
            build_24h_precipitation([self.record(0, 24, 10)], self.start, self.end - timedelta(hours=1), self.init)

    def test_rejects_overlap_that_cannot_be_differenced_as_nested_accumulations(self) -> None:
        records = [self.record(0, 6, 5), self.record(3, 9, 7)]
        with self.assertRaises(GFSPrecipitationAlignmentError):
            build_24h_precipitation(records, self.start, self.end, self.init)

    def test_rejects_incorrect_units_and_non_accumulation_statistic(self) -> None:
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "units"):
            build_24h_precipitation([self.record(3, 27, 12, units="m")], self.start, self.end, self.init)
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "not Accumulation"):
            build_24h_precipitation([self.record(3, 27, 12, stat="Average")], self.start, self.end, self.init)

    def test_rejects_observation_or_future_cycle_data_as_predictor_input(self) -> None:
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "Only forecast"):
            build_24h_precipitation(
                [self.record(3, 27, 12, source_type="observation")], self.start, self.end, self.init
            )
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "reference time mismatch"):
            build_24h_precipitation(
                [self.record(3, 27, 12, forecast_reference_time=self.init + timedelta(hours=6))],
                self.start, self.end, self.init,
            )

    def test_rejects_target_window_that_begins_before_forecast_initialization(self) -> None:
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "before forecast initialization"):
            build_24h_precipitation(
                [self.record(0, 24, 12)], self.init - timedelta(hours=3),
                self.init + timedelta(hours=21), self.init,
            )

    def test_dataset_adapter_rejects_nonforecast_or_unbounded_data(self) -> None:
        bounds = np.array([["2024-07-15T00:00:00", "2024-07-15T03:00:00"]], dtype="datetime64[ns]")
        data = xr.Dataset(
            {"bounds": (("time", "bnds"), bounds),
             "apcp": (("time", "latitude", "longitude"), np.ones((1, 2, 2)))},
            coords={"time": [np.datetime64("2024-07-15T01:30:00")],
                    "latitude": self.coords["latitude"], "longitude": self.coords["longitude"],
                    "reftime": np.datetime64("2024-07-15T00:00:00")},
        )
        data["time"].attrs["bounds"] = "bounds"
        data["apcp"].attrs.update({"Grib2_Generating_Process_Type": "Analysis", "units": "kg.m-2"})
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "not Forecast"):
            GFSAPCPRecord.from_dataset(data, "apcp", 3)
        data["apcp"].attrs["Grib2_Generating_Process_Type"] = "Forecast"
        del data["time"].attrs["bounds"]
        with self.assertRaisesRegex(GFSPrecipitationAlignmentError, "no metadata bounds"):
            GFSAPCPRecord.from_dataset(data, "apcp", 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="retrieve and validate the four explicitly bounded real cases")
    args, remaining = parser.parse_known_args()
    if args.real:
        run_real_cases()
    else:
        unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
