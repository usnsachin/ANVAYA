"""Build provenance-backed GFS/IMD JJAS cases on the exact RF25 grid.

This module is intentionally distinct from the legacy episode/demo builder.
It writes only real UCAR THREDDS subsets paired to official local IMD RF25
targets; a case is committed only after all metadata, time, grid, and value
checks succeed.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock, local
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
import xarray as xr
from urllib3.util.retry import Retry

from ml.data.gfs_precip import GFSAPCPRecord, build_24h_precipitation
from ml.data.providers import INDIA_BBOX
from ml.data.ucar_gfs_subset import (
    FEATURES,
    UCARGFSSSubsetError,
    UCARGFSSSubsetProvider,
    _coordinate_time_utc,
    _metadata_grid_records,
    _sha256,
)
from ml.features.engineer import FEATURE_SCHEMA, engineer_features
from ml.regime.labels import REGIME_LABELS, assign_weak_regime_label


ROOT = Path(__file__).resolve().parents[2]
YEARS = tuple(range(2020, 2025))
MONTHS = (6, 7, 8, 9)
PROCESSING_VERSION = "anvaya-verified-jjas-v1"
IMD_SOURCE = "IMD_RF25_0p25_1901_2024"
GFS_ARCHIVE_VERSION = "NCEP_GFS_0p25_GDEX_d084001_DOI_10.5065_D65D8PWK"
RAW_CASE_ROOT = Path("data/raw/gfs/verified_jjas_2020_2024/cases")
RAW_MANIFEST_ROOT = Path("data/raw/manifests/verified_jjas_2020_2024")
APCP_PARAMETER = "0 1 8"
MAX_APCP_RESPONSE_BYTES = 2_000_000
APCP_FETCH_WORKERS = 3
APCP_DECODE_ATTEMPTS = 3
APCP_NETCDF_LOCK = Lock()
MAX_ATMOSPHERIC_MISSING_FRACTION = 0.0
LEGACY_TEST_PATH = Path("data/processed/test.parquet")
LEGACY_TEST_EXPECTED_SHA256 = "c12585ab1882014bc5005767fc88c4a6270d11d80f5d043d9797c99574e8f5e6"


class VerifiedDatasetError(RuntimeError):
    """Raised when a source case cannot be proven safe for scientific use."""


@dataclass
class CaseBuildResult:
    target_date: str
    split: str
    path: Path
    manifest_path: Path
    resumed: bool
    valid_target_cells: int


def split_for_target_date(value: date | pd.Timestamp | str) -> str:
    """Assign disjoint year-based scientific partitions; never use legacy 2025."""
    year = pd.Timestamp(value).year
    if year <= 2022 and year >= 2020:
        return "train"
    if year == 2023:
        return "validation"
    if year == 2024:
        return "test"
    raise ValueError(f"Target year {year} is outside the verified 2020-2024 split")


def target_window(target_date: date | pd.Timestamp | str) -> tuple[datetime, datetime, datetime, int]:
    """Return (forecast init, exact IMD start, exact IMD end, lead) for date D.

    The selected cycle is the latest standard cycle strictly before the IMD
    interval begins: 00Z on D-1. Its lead-3..27 APCP records cover the 24h
    target exactly, and lead 27 supplies the forecast atmospheric fields.
    """
    d = pd.Timestamp(target_date).date()
    end = datetime(d.year, d.month, d.day, 3, tzinfo=timezone.utc)
    start = end - timedelta(hours=24)
    init = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(days=1)
    lead = int((end - init).total_seconds() // 3600)
    if start < init or lead != 27:
        raise VerifiedDatasetError("Chosen cycle cannot prospectively cover the exact IMD daily interval")
    return init, start, end, lead


def candidate_target_dates(stage: str) -> list[date]:
    """All JJAS target dates or a 40-case seasonal sample (2/month/year)."""
    if stage not in {"sample", "full"}:
        raise ValueError("stage must be 'sample' or 'full'")
    result: list[date] = []
    for year in YEARS:
        for month in MONTHS:
            if stage == "full":
                days: Iterable[int] = range(1, int(pd.Period(f"{year}-{month:02d}").days_in_month) + 1)
            elif month == 7:
                # Include the already audited 2024-07-16 target date.
                days = (16, 24)
            else:
                days = (7, 21)
            result.extend(date(year, month, day) for day in days)
    return result


def configure_session() -> requests.Session:
    """Use bounded retries and serial request pacing for UCAR service safety."""
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "ANVAYA verified research subset pipeline/1.0"})
    session.hooks["response"].append(lambda response, *args, **kwargs: (time.sleep(0.12), response)[1])
    return session


class IMDRF25Store:
    """Read local annual RF25 data, expose exact India subset in GFS order."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.latitude: np.ndarray | None = None
        self.longitude: np.ndarray | None = None
        self.values_by_date: dict[date, np.ndarray] = {}
        for year in YEARS:
            path = root / "data" / "raw" / "imd" / f"RF25_ind{year}_rfp25.nc"
            if not path.is_file():
                raise VerifiedDatasetError(f"Required official IMD RF25 annual file is absent: {path}")
            with xr.open_dataset(path, engine="netcdf4") as ds:
                if "RAINFALL" not in ds or "TIME" not in ds.coords:
                    raise VerifiedDatasetError(f"RF25 schema mismatch in {path}")
                if not np.isclose(float(ds["RAINFALL"].attrs.get("missing_value", -999.0)), -999.0):
                    raise VerifiedDatasetError(f"Unexpected RF25 missing value in {path}")
                subset = ds["RAINFALL"].sel(
                    LATITUDE=slice(INDIA_BBOX["min_lat"], INDIA_BBOX["max_lat"]),
                    LONGITUDE=slice(INDIA_BBOX["min_lon"], INDIA_BBOX["max_lon"]),
                )
                lat = np.asarray(subset["LATITUDE"].values, dtype=float)
                lon = np.asarray(subset["LONGITUDE"].values, dtype=float)
                if self.latitude is None:
                    self.latitude = lat[::-1].copy()
                    self.longitude = lon.copy()
                if not np.allclose(lat[::-1], self.latitude, atol=1e-8) or not np.allclose(lon, self.longitude, atol=1e-8):
                    raise VerifiedDatasetError("IMD annual file grids are not identical")
                if subset.attrs.get("units", "").lower() != "mm":
                    raise VerifiedDatasetError(f"RF25 rainfall units are not mm in {path}")
                values = np.asarray(subset.values, dtype=np.float32)[:, ::-1, :]
                for index, raw_time in enumerate(subset["TIME"].values):
                    stamp = pd.Timestamp(raw_time)
                    self.values_by_date[stamp.date()] = values[index].copy()
        if self.latitude is None or self.longitude is None:
            raise VerifiedDatasetError("No RF25 coordinate arrays loaded")
        expected_lat = np.arange(37.0, 7.9999, -0.25)
        expected_lon = np.arange(68.0, 97.0001, 0.25)
        if not np.allclose(self.latitude, expected_lat) or not np.allclose(self.longitude, expected_lon):
            raise VerifiedDatasetError("RF25 India subset is not the expected 117 x 117 GFS grid")

    def values(self, target_date: date | pd.Timestamp | str) -> np.ndarray:
        d = pd.Timestamp(target_date).date()
        if d not in self.values_by_date:
            raise VerifiedDatasetError(f"No official RF25 value exists for target date {d}")
        return self.values_by_date[d]

    @staticmethod
    def valid_mask(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        return np.isfinite(arr) & (arr != -999.0) & (arr >= 0.0)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata_apcp_variable(grids: dict[str, dict[str, str]], lead: int) -> tuple[str, dict[str, str]]:
    candidates = [
        (name, attrs) for name, attrs in grids.items()
        if attrs.get("Grib2_Parameter") == APCP_PARAMETER
        and attrs.get("Grib2_Level_Type") == "1"
        and attrs.get("Grib2_Generating_Process_Type", "").casefold() == "forecast"
        and attrs.get("Grib2_Statistical_Process_Type", "").casefold() == "accumulation"
        and attrs.get("units") == "kg.m-2"
    ]
    suffix = "3_Hour_Accumulation" if lead == 3 else "6_Hour_Accumulation" if lead == 6 else "Mixed_intervals_Accumulation"
    matches = [(name, attrs) for name, attrs in candidates if name.endswith(suffix)]
    if len(matches) != 1:
        raise VerifiedDatasetError(
            f"Forecast f{lead:03d} metadata did not identify exactly one APCP accumulation "
            f"ending {suffix}; candidates={[name for name, _ in candidates]}"
        )
    return matches[0]


def _fetch_apcp_record(
    provider: UCARGFSSSubsetProvider, init_time: datetime, lead: int,
) -> tuple[GFSAPCPRecord, dict[str, Any]]:
    """Fetch exactly one metadata-identified APCP interval over the India box."""
    dataset_url = provider._dataset_url(init_time, lead)
    metadata_url = dataset_url + "/dataset.xml"
    metadata_response = provider.session.get(metadata_url, timeout=90)
    metadata_response.raise_for_status()
    if len(metadata_response.content) > 10_000_000:
        raise VerifiedDatasetError("UCAR APCP metadata exceeded the 10 MB safety limit")
    _, grids = _metadata_grid_records(metadata_response.content)
    variable, field_metadata = _metadata_apcp_variable(grids, lead)
    valid_time = init_time + timedelta(hours=lead)
    params = {
        "var": variable,
        "north": INDIA_BBOX["max_lat"], "south": INDIA_BBOX["min_lat"],
        "west": INDIA_BBOX["min_lon"], "east": INDIA_BBOX["max_lon"],
        "accept": "netcdf3", "time": _iso(valid_time),
    }
    decode_retry_failures: list[dict[str, Any]] = []
    dataset: xr.Dataset | None = None
    response: requests.Response | None = None
    for attempt in range(1, APCP_DECODE_ATTEMPTS + 1):
        response = provider.session.get(dataset_url, params=params, timeout=120)
        if response.status_code != 200:
            response_body = response.content
            body_preview = response.text[:400].replace("\n", " ").replace("\r", " ")
            raise VerifiedDatasetError(
                f"UCAR APCP f{lead:03d} request failed: status={response.status_code}; "
                f"url={response.url}; response_bytes={len(response_body)}; "
                f"response_sha256={hashlib.sha256(response_body).hexdigest()}; "
                f"body_preview={body_preview!r}"
            )
        if len(response.content) > MAX_APCP_RESPONSE_BYTES:
            raise VerifiedDatasetError("A bounded APCP response exceeded the 2 MB subset safety limit")
        if len(response.content) < 8 or "text/html" in response.headers.get("Content-Type", "").lower():
            raise VerifiedDatasetError("UCAR APCP response is empty or an error document")
        try:
            with tempfile.TemporaryDirectory(prefix="anvaya-apcp-") as temporary:
                path = Path(temporary) / "record.nc"
                path.write_bytes(response.content)
                # netCDF4/HDF5 decode operations are serialized; independent
                # THREDDS requests remain concurrent, but the native library
                # must not open temporary NetCDF files from several workers at once.
                with APCP_NETCDF_LOCK:
                    with xr.open_dataset(path, engine="netcdf4") as opened:
                        dataset = opened.load()
            break
        except Exception as exc:
            decode_retry_failures.append({
                "attempt": attempt,
                "response_bytes": len(response.content),
                "response_sha256": hashlib.sha256(response.content).hexdigest(),
                "decode_error": f"{type(exc).__name__}: {exc}",
            })
            if attempt == APCP_DECODE_ATTEMPTS:
                raise VerifiedDatasetError(
                    f"UCAR APCP f{lead:03d} response could not decode after {attempt} bounded attempts "
                    f"({response.url}); failed response hashes/errors={decode_retry_failures}"
                ) from exc
            time.sleep(0.25 * attempt)
    if response is None or dataset is None:
        raise VerifiedDatasetError(f"UCAR APCP f{lead:03d} request produced no decoded subset")
    record = GFSAPCPRecord.from_dataset(dataset, variable, lead)
    if record.forecast_reference_time != init_time or record.interval_end != valid_time:
        raise VerifiedDatasetError("APCP reference or interval end does not match requested forecast lead")
    if record.interval_start >= record.interval_end or record.units != "kg.m-2":
        raise VerifiedDatasetError("APCP interval bounds or units are invalid")
    if record.statistical_process_type.casefold() != "accumulation":
        raise VerifiedDatasetError("Requested GFS precipitation message is not an accumulation")
    lat = np.asarray(record.values.latitude.values, dtype=float)
    lon = np.asarray(record.values.longitude.values, dtype=float)
    if lat.shape != (117,) or lon.shape != (117,):
        raise VerifiedDatasetError(f"Unexpected cropped APCP grid shape {(lat.size, lon.size)}")
    if lat.min() < 8.0 or lat.max() > 37.0 or lon.min() < 68.0 or lon.max() > 97.0:
        raise VerifiedDatasetError("APCP subset coordinates are outside requested India bounds")
    numeric = np.asarray(record.values.values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if not finite.size or finite.min() < -1e-5 or finite.max() > 2000.0:
        raise VerifiedDatasetError("APCP values are missing or outside physical sanity limits")
    info = {
        "forecast_hour": lead,
        "variable": variable,
        "metadata": field_metadata,
        "accumulation_start": _iso(record.interval_start),
        "accumulation_end": _iso(record.interval_end),
        "duration_hours": (record.interval_end - record.interval_start).total_seconds() / 3600.0,
        "units": record.units,
        "statistical_process_type": record.statistical_process_type,
        "retrieval_url": response.url,
        "metadata_url": metadata_url,
        "metadata_response_sha256": hashlib.sha256(metadata_response.content).hexdigest(),
        "subset_response_bytes": len(response.content),
        "subset_sha256": hashlib.sha256(response.content).hexdigest(),
        "decode_retry_count": len(decode_retry_failures),
        "discarded_decode_attempts": decode_retry_failures,
        "grid_shape": [int(lat.size), int(lon.size)],
        "latitude_bounds": [float(lat.min()), float(lat.max())],
        "longitude_bounds": [float(lon.min()), float(lon.max())],
        "domain_mean_mm": float(np.nanmean(numeric)),
        "domain_max_mm": float(np.nanmax(numeric)),
    }
    return record, info


def _record_from_provider_output(ds: xr.Dataset, init: datetime, lead: int, manifest: dict[str, Any]) -> tuple[GFSAPCPRecord, dict[str, Any]]:
    variable = FEATURES["gfs_apcp_interval_mm"].variable
    if variable not in ds:
        raise VerifiedDatasetError("The validated lead-27 subset lacks the APCP interval field")
    record = GFSAPCPRecord.from_dataset(ds, variable, lead)
    if record.forecast_reference_time != init or record.interval_end != init + timedelta(hours=lead):
        raise VerifiedDatasetError("Lead-27 APCP metadata does not match forecast initialization")
    apcp_group = next(
        (group for group in manifest["subset"]["request_groups"] if "gfs_apcp_interval_mm" in group["features"]),
        None,
    )
    if apcp_group is None:
        raise VerifiedDatasetError("Lead-27 manifest does not record the APCP subset request")
    info = {
        "forecast_hour": lead,
        "variable": variable,
        "metadata": manifest["subset"]["variables"]["gfs_apcp_interval_mm"]["metadata"],
        "accumulation_start": _iso(record.interval_start),
        "accumulation_end": _iso(record.interval_end),
        "duration_hours": (record.interval_end - record.interval_start).total_seconds() / 3600.0,
        "units": record.units,
        "statistical_process_type": record.statistical_process_type,
        "retrieval_url": apcp_group["url"],
        "metadata_url": manifest["source"]["source_metadata"]["metadata_url"],
        "subset_response_bytes": apcp_group["bytes"],
        "subset_sha256": apcp_group["sha256"],
        "grid_shape": [int(record.values.sizes["latitude"]), int(record.values.sizes["longitude"])],
        "latitude_bounds": [float(record.values.latitude.min()), float(record.values.latitude.max())],
        "longitude_bounds": [float(record.values.longitude.min()), float(record.values.longitude.max())],
        "domain_mean_mm": float(record.values.mean(skipna=True)),
        "domain_max_mm": float(record.values.max(skipna=True)),
    }
    return record, info


def _extract_2d(ds: xr.Dataset, feature: str) -> np.ndarray:
    spec = FEATURES[feature]
    if spec.variable not in ds.data_vars:
        raise VerifiedDatasetError(f"Atmospheric subset is missing registered raw field {spec.variable}")
    field = ds[spec.variable]
    if field.attrs.get("units") != spec.units:
        raise VerifiedDatasetError(
            f"Raw {feature} units {field.attrs.get('units')!r} do not match registered {spec.units!r}"
        )
    if field.attrs.get("Grib2_Generating_Process_Type", "").casefold() != "forecast":
        raise VerifiedDatasetError(f"{feature} is not metadata-identified as a GFS forecast field")
    squeezed = field.squeeze(drop=True)
    if set(squeezed.dims) != {"latitude", "longitude"}:
        raise VerifiedDatasetError(f"{feature} has unexpected dimensions after level/time squeeze: {field.dims}")
    values = np.asarray(squeezed.transpose("latitude", "longitude").values, dtype=np.float32)
    if values.shape != (117, 117):
        raise VerifiedDatasetError(f"{feature} is not a 117 x 117 India subset")
    return values


def _validated_cached_case(path: Path, manifest_path: Path, target_date: date) -> dict[str, Any] | None:
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.exists() or not manifest_path.exists():
        raise VerifiedDatasetError(f"Incomplete existing case cache for {target_date}; refusing implicit overwrite")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "VERIFIED_GFS_CASE_READY":
        raise VerifiedDatasetError(f"Cached case {target_date} is not marked ready")
    if manifest.get("target_date") != target_date.isoformat():
        raise VerifiedDatasetError(f"Cached case manifest date mismatch for {target_date}")
    if manifest.get("output", {}).get("sha256") != _sha256(path):
        raise VerifiedDatasetError(f"Cached case checksum mismatch for {target_date}")
    return manifest


def build_or_resume_case(
    target_date: date,
    imd: IMDRF25Store,
    provider: UCARGFSSSubsetProvider,
    root: Path = ROOT,
) -> CaseBuildResult:
    """Retrieve/cache one exact-window, grid-validated GFS case."""
    init, target_start, target_end, lead = target_window(target_date)
    split = split_for_target_date(target_date)
    relative = RAW_CASE_ROOT / str(target_date.year) / f"gfs_imd_window_{target_date:%Y%m%d}.nc"
    path = root / relative
    manifest_path = path.with_suffix(".json")
    cache_hit = _validated_cached_case(path, manifest_path, target_date)
    if cache_hit is not None:
        valid_count = int(cache_hit["validation"]["valid_paired_cells"])
        if valid_count <= 0:
            raise VerifiedDatasetError(f"Cached case {target_date} has no paired target cells")
        return CaseBuildResult(target_date.isoformat(), split, path, manifest_path, True, valid_count)

    path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = root / "data" / "raw" / "gfs" / "verified_jjas_2020_2024" / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"case-{target_date:%Y%m%d}-", dir=staging_root) as temporary:
        temp_dir = Path(temporary)
        atmospheric_path = temp_dir / "forecast_f027.nc"
        atmospheric_manifest_path = temp_dir / "forecast_f027.json"
        atmospheric_manifest = provider.retrieve_forecast(
            init, lead_hours=lead, output_path=atmospheric_path, manifest_path=atmospheric_manifest_path,
        )
        if atmospheric_manifest.get("status") != "VALIDATED_SINGLE_CASE":
            raise VerifiedDatasetError("UCAR provider did not validate the lead-27 atmospheric forecast")
        with xr.open_dataset(atmospheric_path, engine="netcdf4") as opened:
            atmosphere = opened.load()
        observed_init = _coordinate_time_utc(np.asarray(atmosphere["reftime"].values).reshape(-1)[0])
        observed_valid = _coordinate_time_utc(np.asarray(atmosphere["forecast_valid_time"].values).reshape(-1)[0])
        if observed_init != init or observed_valid != target_end:
            raise VerifiedDatasetError("Atmospheric GFS reftime/time do not equal the selected init/IMD target end")
        lat = np.asarray(atmosphere["latitude"].values, dtype=float)
        lon = np.asarray(atmosphere["longitude"].values, dtype=float)
        if lat.shape != (117,) or lon.shape != (117,):
            raise VerifiedDatasetError("Atmospheric GFS subset is not 117 x 117")
        if not np.allclose(lat, imd.latitude, atol=1e-8) or not np.allclose(lon, imd.longitude, atol=1e-8):
            raise VerifiedDatasetError("GFS and RF25 latitude/longitude coordinates do not match exactly")

        records: list[GFSAPCPRecord] = []
        apcp_info: list[dict[str, Any]] = []
        apcp_leads = list(range(3, lead, 3))
        thread_state = local()

        def fetch_interval(interval_lead: int) -> tuple[GFSAPCPRecord, dict[str, Any]]:
            worker_provider = getattr(thread_state, "provider", None)
            if worker_provider is None:
                worker_provider = UCARGFSSSubsetProvider(root=root, session=configure_session())
                thread_state.provider = worker_provider
            return _fetch_apcp_record(worker_provider, init, interval_lead)

        # Parallelize only the eight independent bounded APCP requests. Each
        # thread has its own session; a small fixed pool avoids a burst of the
        # full date range against UCAR.
        with ThreadPoolExecutor(max_workers=APCP_FETCH_WORKERS) as executor:
            apcp_results = list(executor.map(fetch_interval, apcp_leads))
        for interval_lead, (record, info) in zip(apcp_leads, apcp_results):
            if not np.allclose(record.values.latitude.values, lat, atol=1e-8) or not np.allclose(record.values.longitude.values, lon, atol=1e-8):
                raise VerifiedDatasetError(f"APCP f{interval_lead:03d} grid differs from atmospheric GFS grid")
            records.append(record)
            apcp_info.append(info)
        final_record, final_info = _record_from_provider_output(atmosphere, init, lead, atmospheric_manifest)
        records.append(final_record)
        apcp_info.append(final_info)
        daily = build_24h_precipitation(records, target_start, target_end, init)
        if daily.attrs.get("target_interval_start") != _iso(target_start) or daily.attrs.get("target_interval_end") != _iso(target_end):
            raise VerifiedDatasetError("Metadata-built GFS daily rainfall does not equal the RF25 UTC target interval")
        daily_values = np.asarray(daily.transpose("latitude", "longitude").values, dtype=np.float32)
        if daily_values.shape != (117, 117):
            raise VerifiedDatasetError("Derived daily GFS rainfall is not a 117 x 117 field")
        finite_daily = daily_values[np.isfinite(daily_values)]
        if not finite_daily.size or finite_daily.min() < -1e-5 or finite_daily.max() > 2000.0:
            raise VerifiedDatasetError("Derived GFS daily rainfall is missing or physically implausible")

        atmospheric_fields = {name: _extract_2d(atmosphere, name) for name in FEATURES if name != "gfs_apcp_interval_mm"}
        target_values = imd.values(target_date)
        target_valid = imd.valid_mask(target_values)
        if target_values.shape != (117, 117):
            raise VerifiedDatasetError("Selected RF25 daily target is not 117 x 117")
        paired_mask = target_valid & np.isfinite(daily_values)
        if int(paired_mask.sum()) == 0:
            raise VerifiedDatasetError(f"No valid IMD/GFS cells for target date {target_date}")
        apcp_data = np.stack([np.asarray(record.values.transpose("latitude", "longitude").values, dtype=np.float32) for record in records])
        bounds = np.asarray([
            [np.datetime64(record.interval_start.replace(tzinfo=None)), np.datetime64(record.interval_end.replace(tzinfo=None))]
            for record in records
        ], dtype="datetime64[ns]")
        data_vars: dict[str, Any] = {
            "gfs_apcp_interval_mm": (("record", "latitude", "longitude"), apcp_data),
            "apcp_time_bounds": (("record", "bounds"), bounds),
            "gfs_daily_rain_mm": (("latitude", "longitude"), daily_values),
        }
        for feature, values in atmospheric_fields.items():
            data_vars[feature] = (("latitude", "longitude"), values)
        case_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "record": np.asarray([record.forecast_hour for record in records], dtype=np.int16),
                "latitude": lat.astype(np.float32),
                "longitude": lon.astype(np.float32),
                "bounds": np.asarray([0, 1], dtype=np.int8),
                "reftime": np.datetime64(init.replace(tzinfo=None)),
                "valid_time": np.datetime64(target_end.replace(tzinfo=None)),
            },
            attrs={
                "source": provider.source_name,
                "source_dataset": "NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive",
                "source_dataset_doi": provider.dataset_doi,
                "source_archive_id": "UCAR GDEX d084001",
                "target_source": IMD_SOURCE,
                "target_date": target_date.isoformat(),
                "initialization_time_utc": _iso(init),
                "valid_time_utc": _iso(target_end),
                "lead_hours": lead,
                "target_interval_start_utc": _iso(target_start),
                "target_interval_end_utc": _iso(target_end),
                "bbox_south": 8.0, "bbox_north": 37.0, "bbox_west": 68.0, "bbox_east": 97.0,
                "longitude_convention": "0_to_360_positive_east",
                "apcp_aggregation": "metadata-defined non-overlapping intervals; same-start nested accumulations differenced before summation",
            },
        )
        case_ds["gfs_apcp_interval_mm"].attrs.update({
            "units": "kg.m-2 water equivalent (numerically mm)",
            "Grib2_Parameter": APCP_PARAMETER,
            "Grib2_Statistical_Process_Type": "Accumulation",
            "record_variable_names": json.dumps([record.values.name for record in records]),
        })
        case_ds["apcp_time_bounds"].attrs["long_name"] = "Source metadata-defined APCP accumulation bounds per forecast hour"
        case_ds["gfs_daily_rain_mm"].attrs.update(daily.attrs)
        for feature, spec in FEATURES.items():
            if feature == "gfs_apcp_interval_mm":
                continue
            case_ds[feature].attrs.update(atmosphere[spec.variable].attrs)
            case_ds[feature].attrs["normalized_feature_name"] = feature

        partial_path = path.with_suffix(path.suffix + ".part")
        partial_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".part")
        try:
            case_ds.to_netcdf(partial_path, engine="netcdf4")
            file_hash = _sha256(partial_path)
            manifest = {
                "status": "VERIFIED_GFS_CASE_READY",
                "processing_version": PROCESSING_VERSION,
                "target_date": target_date.isoformat(),
                "split": split,
                "source": {
                    "provider": provider.source_name,
                    "dataset": "NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive",
                    "archive_id": "UCAR GDEX d084001",
                    "doi": provider.dataset_doi,
                    "dataset_url": "https://gdex.k8s.ucar.edu/datasets/d084001/",
                    "gfs_archive_version": GFS_ARCHIVE_VERSION,
                    "target_source": IMD_SOURCE,
                    "imd_target_file": f"data/raw/imd/RF25_ind{target_date.year}_rfp25.nc",
                    "imd_variable": "RAINFALL",
                    "imd_units": "mm",
                },
                "forecast": {
                    "initialization": _iso(init),
                    "cycle": "00Z",
                    "valid_time": _iso(target_end),
                    "lead_hours": lead,
                    "target_interval_start": _iso(target_start),
                    "target_interval_end": _iso(target_end),
                },
                "subset": {
                    "bbox": {"south": 8.0, "north": 37.0, "west": 68.0, "east": 97.0,
                             "longitude_convention": "0_to_360_positive_east"},
                    "grid_dimensions": [117, 117],
                    "latitude_order": "descending north-to-south",
                    "longitude_order": "ascending east-positive 0-360 convention",
                    "atmospheric_fields": atmospheric_manifest["subset"]["variables"],
                    "atmospheric_requests": atmospheric_manifest["subset"]["request_groups"],
                    "apcp_interval_records": apcp_info,
                },
                "validation": {
                    "forecast_reftime_verified": True,
                    "valid_time_equals_init_plus_lead": target_end == init + timedelta(hours=lead),
                    "gfs_target_window_exact": daily.attrs["target_interval_start"] == _iso(target_start)
                        and daily.attrs["target_interval_end"] == _iso(target_end),
                    "apcp_record_count": len(records),
                    "all_apcp_intervals_metadata_defined": all(
                        info["statistical_process_type"].casefold() == "accumulation" for info in apcp_info
                    ),
                    "gfs_grid_dimensions": [int(lat.size), int(lon.size)],
                    "imd_gfs_grid_coordinate_match": True,
                    "imd_valid_cells": int(target_valid.sum()),
                    "valid_paired_cells": int(paired_mask.sum()),
                    "rainfall_units": {"gfs_daily_rain_mm": "mm", "imd_observed_rain_mm": "mm"},
                    "gfs_daily_rain_min_mm": float(finite_daily.min()),
                    "gfs_daily_rain_max_mm": float(finite_daily.max()),
                    "imd_missing_value": -999.0,
                },
                "output": {
                    "path": relative.as_posix(),
                    "manifest_path": manifest_path.relative_to(root).as_posix(),
                    "file_size_bytes": partial_path.stat().st_size,
                    "sha256": file_hash,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
            partial_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            partial_path.replace(path)
            partial_manifest_path.replace(manifest_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            partial_manifest_path.unlink(missing_ok=True)
            raise
        finally:
            case_ds.close()
    return CaseBuildResult(target_date.isoformat(), split, path, manifest_path, False, int(paired_mask.sum()))


def _weak_regimes(fields: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of assign_weak_regime_label, preserving precedence."""
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    mslp = fields["gfs_mslp"]
    rh = fields["gfs_rh2m"]
    u850 = fields["gfs_u850"]
    v850 = fields["gfs_v850"]
    pw = fields["gfs_precipitable_water"]
    labels = np.full(mslp.shape, "OTHER", dtype=object)
    break_mask = (lat_grid >= 15.0) & (lat_grid <= 26.0) & ((u850 <= 4.0) | (rh <= 60.0))
    labels[break_mask] = "BREAK_MONSOON"
    active = (u850 >= 6.0) & (pw >= 50.0) & (rh >= 72.0)
    labels[active] = "ACTIVE_MONSOON"
    northeast = (lat_grid >= 24.5) & (lon_grid >= 88.0)
    ghats = (lon_grid >= 73.5) & (lon_grid <= 75.5) & (lat_grid >= 10.0) & (lat_grid <= 19.5)
    orographic = (northeast | ghats) & ((u850 >= 7.0) | (v850 >= 5.0)) & (pw >= 48.0)
    labels[orographic] = "OROGRAPHIC_RAINFALL"
    coastal = (lon_grid <= 75.5) & (lat_grid >= 8.0) & (lat_grid <= 20.0) & (u850 >= 9.0) & (rh >= 78.0)
    labels[coastal] = "COASTAL_RAINFALL"
    western = (lat_grid >= 28.0) & (v850 < -2.0)
    labels[western] = "WESTERN_DISTURBANCE"
    low = mslp < 998.0
    labels[low] = "LOW_DEPRESSION"
    return labels


def _wind_direction_10m(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute 10 m wind direction in float64 to match the canonical row path.

    UCAR arrays are stored as float32. Applying arctan2 before promotion can
    differ from the canonical Python-float calculation by several 1e-5
    degrees, which is numerically harmless but defeats strict feature parity.
    """
    u64 = np.asarray(u, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    return (np.degrees(np.arctan2(u64, v64)) + 360.0) % 360.0


def case_to_frame(
    result: CaseBuildResult,
    target_date: date,
    imd: IMDRF25Store,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a validated raw case and construct valid gridded target rows."""
    init, target_start, target_end, lead = target_window(target_date)
    target = imd.values(target_date)
    antecedent_date = target_date - timedelta(days=2)
    antecedent = imd.values(antecedent_date)
    with xr.open_dataset(result.path, engine="netcdf4") as opened:
        ds = opened.load()
    lat = np.asarray(ds.latitude.values, dtype=float)
    lon = np.asarray(ds.longitude.values, dtype=float)
    if ds.attrs.get("target_interval_start_utc") != _iso(target_start) or ds.attrs.get("target_interval_end_utc") != _iso(target_end):
        raise VerifiedDatasetError(f"Cached GFS daily rainfall target interval mismatch for {target_date}")
    if not np.allclose(lat, imd.latitude, atol=1e-8) or not np.allclose(lon, imd.longitude, atol=1e-8):
        raise VerifiedDatasetError(f"Cached GFS grid differs from RF25 for {target_date}")
    predictor_fields: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        if feature == "gfs_apcp_interval_mm":
            continue
        values = np.asarray(ds[feature].values, dtype=np.float32)
        if values.shape != (117, 117):
            raise VerifiedDatasetError(f"Cached atmospheric field {feature} has wrong shape")
        predictor_fields[feature] = values
    # Normalize archive units to the feature pipeline's documented model units.
    predictor_fields["gfs_mslp"] = predictor_fields["gfs_mslp"] / 100.0  # Pa -> hPa
    predictor_fields["gfs_t2m"] = predictor_fields["gfs_t2m"] - 273.15  # K -> Celsius
    daily = np.asarray(ds["gfs_daily_rain_mm"].values, dtype=np.float32)
    target_valid = imd.valid_mask(target)
    antecedent_valid = imd.valid_mask(antecedent)
    all_gfs_valid = np.ones((117, 117), dtype=bool)
    for name, values in predictor_fields.items():
        valid = np.isfinite(values)
        if name in {"gfs_rh2m"}:
            valid &= (values >= 0.0) & (values <= 100.0)
        elif name == "gfs_mslp":
            valid &= (values >= 800.0) & (values <= 1100.0)
        elif name == "gfs_cape":
            valid &= values >= 0.0
        all_gfs_valid &= valid
    all_gfs_valid &= np.isfinite(daily) & (daily >= 0.0) & (daily <= 2000.0)
    paired = target_valid & antecedent_valid & all_gfs_valid
    if not paired.any():
        raise VerifiedDatasetError(f"No grid cells have complete target and predictor data for {target_date}")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    feature_arrays: dict[str, np.ndarray] = {
        "gfs_daily_rain_mm": daily,
        **predictor_fields,
        "rain_last_24h": antecedent,
        "latitude": lat_grid,
        "longitude": lon_grid,
    }
    feature_arrays["wind_speed_10m"] = np.hypot(predictor_fields["gfs_u10"], predictor_fields["gfs_v10"])
    feature_arrays["wind_dir_10m"] = _wind_direction_10m(
        predictor_fields["gfs_u10"], predictor_fields["gfs_v10"]
    )
    feature_arrays["wind_speed_850"] = np.hypot(predictor_fields["gfs_u850"], predictor_fields["gfs_v850"])
    feature_arrays["moisture_transport_850"] = feature_arrays["wind_speed_850"] * np.maximum(0.0, predictor_fields["gfs_precipitable_water"])
    feature_arrays["mslp_anomaly"] = predictor_fields["gfs_mslp"] - 1013.25
    feature_arrays["lead_time_h"] = np.full((117, 117), float(lead), dtype=np.float32)
    west_coast_distance = np.minimum(np.abs(lon_grid - 73.0), np.abs(lon_grid - 83.0))
    north_distance = np.abs(lat_grid - 24.0) * 0.8 + 2.0
    feature_arrays["coast_distance_approx"] = np.where(lat_grid < 24.0, west_coast_distance, north_distance)
    terrain = np.full((117, 117), 100.0)
    terrain[(lat_grid > 30.0)] = 1500.0 + (lat_grid[(lat_grid > 30.0)] - 30.0) * 400.0
    ghats_ridge = (lat_grid >= 10.0) & (lat_grid <= 20.0) & (lon_grid >= 73.5) & (lon_grid <= 75.5)
    terrain[ghats_ridge] = 850.0
    deccan = (lat_grid >= 12.0) & (lat_grid <= 22.0) & (lon_grid >= 74.0) & (lon_grid <= 79.0) & ~ghats_ridge
    terrain[deccan] = 500.0
    feature_arrays["terrain_elevation_proxy"] = terrain
    ghats_flow = (lat_grid >= 8.0) & (lat_grid <= 21.0) & (lon_grid >= 72.0) & (lon_grid <= 77.0)
    himalayan_flow = (lat_grid >= 26.0) & ~ghats_flow
    oro = np.zeros((117, 117), dtype=np.float32)
    oro[ghats_flow] = np.maximum(0.0, predictor_fields["gfs_u850"][ghats_flow]) * 1.5
    oro[himalayan_flow] = np.maximum(0.0, predictor_fields["gfs_v850"][himalayan_flow]) * 1.2
    feature_arrays["orographic_flow_proxy"] = oro
    feature_arrays["month"] = np.full((117, 117), target_date.month, dtype=np.int8)
    feature_arrays["day_of_year"] = np.full((117, 117), target_date.timetuple().tm_yday, dtype=np.int16)

    missing_schema = set(FEATURE_SCHEMA).difference(feature_arrays)
    if missing_schema:
        raise VerifiedDatasetError(f"Verified feature builder does not produce schema columns: {sorted(missing_schema)}")
    regimes = _weak_regimes(predictor_fields, lat, lon)
    init_text, valid_text = _iso(init), _iso(target_end)
    antecedent_valid_time = datetime(antecedent_date.year, antecedent_date.month, antecedent_date.day, 3, tzinfo=timezone.utc)
    if not antecedent_valid_time < init:
        raise VerifiedDatasetError("Antecedent IMD rainfall is not strictly earlier than GFS initialization")
    flattened = {key: np.asarray(value)[paired].reshape(-1) for key, value in feature_arrays.items()}
    frame = pd.DataFrame(flattened)
    frame["observed_rain_mm"] = target[paired].astype(np.float32)
    frame["regime"] = regimes[paired].astype(str)
    frame["heavy_rain"] = (frame["observed_rain_mm"] >= 64.5).astype(np.int8)
    frame["very_heavy_rain"] = (frame["observed_rain_mm"] >= 115.6).astype(np.int8)
    frame["extreme_rain"] = (frame["observed_rain_mm"] >= 204.5).astype(np.int8)
    frame["target_date"] = target_date.isoformat()
    frame["init_time"] = init_text
    frame["valid_time"] = valid_text
    frame["lead_hours"] = np.int16(lead)
    frame["cycle"] = "00Z"
    frame["forecast_cycle"] = "00Z"
    frame["forecast_cycle_id"] = init_text
    frame["source"] = "UCAR_GFS_0p25_THREDDS_NCSS + IMD_RF25"
    frame["source_version"] = f"{GFS_ARCHIVE_VERSION}+IMD_RF25_catalogue_1901_2024"
    frame["gfs_source"] = "UCAR_GFS_0p25_THREDDS_NCSS"
    frame["gfs_archive_version"] = GFS_ARCHIVE_VERSION
    frame["imd_target_source"] = IMD_SOURCE
    frame["processing_version"] = PROCESSING_VERSION
    frame["split"] = result.split
    frame["antecedent_target_date"] = antecedent_date.isoformat()
    frame["antecedent_valid_time"] = _iso(antecedent_valid_time)
    # A daily target cannot substantiate a 6-hour antecedent predictor.
    if "rain_last_6h" in frame.columns:
        raise VerifiedDatasetError("Unsupported rain_last_6h was unexpectedly included")
    if frame["observed_rain_mm"].isna().any() or frame[list(FEATURE_SCHEMA)].isna().any().any():
        raise VerifiedDatasetError(f"Missing values remain after the paired-cell validity mask for {target_date}")
    if not frame["observed_rain_mm"].between(0.0, 2000.0).all() or not frame["gfs_daily_rain_mm"].between(0.0, 2000.0).all():
        raise VerifiedDatasetError(f"Rainfall values outside physical limits for {target_date}")
    if frame.duplicated(["init_time", "valid_time", "latitude", "longitude"]).any():
        raise VerifiedDatasetError(f"Duplicate coordinate/time rows in GFS/IMD case {target_date}")
    if not ((pd.to_datetime(frame["antecedent_valid_time"], utc=True) < pd.to_datetime(frame["init_time"], utc=True)).all()):
        raise VerifiedDatasetError("A future antecedent observation was detected")
    # Cross-check vectorized production features/labels against the project's
    # canonical row-wise implementations at representative cells per case.
    check_indices = sorted({0, len(frame) // 2, len(frame) - 1})
    for idx in check_indices:
        feature_row = frame.iloc[idx].to_dict()
        canonical = engineer_features(feature_row, lead_time_h=lead, valid_date=target_date)
        for name in FEATURE_SCHEMA:
            if not np.isclose(float(canonical[name]), float(feature_row[name]), rtol=1e-6, atol=1e-6):
                raise VerifiedDatasetError(f"Vectorized {name} differs from canonical feature pipeline at {target_date}")
        if assign_weak_regime_label(feature_row) != feature_row["regime"]:
            raise VerifiedDatasetError(f"Vectorized regime label differs from canonical weak-label logic at {target_date}")
    counts = {
        "target_valid_cells": int(target_valid.sum()),
        "antecedent_valid_cells": int(antecedent_valid.sum()),
        "all_gfs_fields_valid_cells": int(all_gfs_valid.sum()),
        "valid_paired_cells": int(paired.sum()),
        "missing_target_cells_dropped": int((~target_valid).sum()),
        "invalid_antecedent_cells_dropped": int((target_valid & ~antecedent_valid).sum()),
        "invalid_gfs_cells_dropped": int((target_valid & antecedent_valid & ~all_gfs_valid).sum()),
    }
    return frame, counts


def _case_statistics(frame: pd.DataFrame, counts: dict[str, Any], split_stats: dict[str, Any]) -> None:
    split = str(frame["split"].iloc[0])
    stats = split_stats[split]
    stats["rows"] += len(frame)
    stats["dates"].add(str(frame["target_date"].iloc[0]))
    stats["forecast_initializations"].add(str(frame["init_time"].iloc[0]))
    stats["cycle_hours"].add(str(frame["cycle"].iloc[0]))
    stats["grid_cells"].update(
        frame[["latitude", "longitude"]].drop_duplicates().itertuples(index=False, name=None)
    )
    stats["duplicate_rows"] += int(frame.duplicated(["init_time", "valid_time", "latitude", "longitude"]).sum())
    stats["missing_values"] += int(frame.isna().sum().sum())
    stats["valid_cells_by_date"][str(frame["target_date"].iloc[0])] = int(counts["valid_paired_cells"])
    stats["imd_valid_cells_by_date"][str(frame["target_date"].iloc[0])] = int(counts["target_valid_cells"])
    stats["observed_values"].append(frame["observed_rain_mm"].to_numpy(dtype=np.float32))
    stats["gfs_values"].append(frame["gfs_daily_rain_mm"].to_numpy(dtype=np.float32))
    for label in REGIME_LABELS:
        subset = frame.loc[frame["regime"] == label]
        if len(subset):
            stats["regimes"][label]["rows"] += int(len(subset))
            stats["regimes"][label]["dates"].add(str(subset["target_date"].iloc[0]))
            stats["regimes"][label]["forecast_initializations"].add(str(subset["forecast_cycle_id"].iloc[0]))


def _empty_split_stats() -> dict[str, Any]:
    return {
        "rows": 0, "dates": set(), "forecast_initializations": set(), "cycle_hours": set(),
        "grid_cells": set(), "duplicate_rows": 0, "missing_values": 0,
        "valid_cells_by_date": {}, "imd_valid_cells_by_date": {}, "observed_values": [], "gfs_values": [],
        "regimes": {label: {"rows": 0, "dates": set(), "forecast_initializations": set()} for label in REGIME_LABELS},
    }


def _distribution(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    data = np.concatenate(values).astype(np.float64)
    return {
        "count": int(data.size), "min": float(np.min(data)), "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)), "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)), "max": float(np.max(data)), "mean": float(np.mean(data)),
    }


def summarize_splits(stats: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    thresholds = (("heavy", 64.5), ("very_heavy", 115.6), ("extreme", 204.5))
    for split, raw in stats.items():
        target_dist = _distribution(raw["observed_values"])
        gfs_dist = _distribution(raw["gfs_values"])
        events = {
            key: {
                "observed_count": int(sum(np.count_nonzero(values >= threshold) for values in raw["observed_values"])),
                "gfs_daily_count": int(sum(np.count_nonzero(values >= threshold) for values in raw["gfs_values"])),
            }
            for key, threshold in thresholds
        }
        regimes = {
            label: {
                "rows": value["rows"],
                "unique_dates": len(value["dates"]),
                "unique_forecast_cycles": len(value["forecast_initializations"]),
            }
            for label, value in raw["regimes"].items()
        }
        report[split] = {
            "date_range": [min(raw["dates"]) if raw["dates"] else None, max(raw["dates"]) if raw["dates"] else None],
            "number_of_forecast_cases": len(raw["forecast_initializations"]),
            "number_of_rows": raw["rows"],
            "number_of_unique_dates": len(raw["dates"]),
            "number_of_unique_forecast_cycles": len(raw["forecast_initializations"]),
            "forecast_cycle_hours": sorted(raw["cycle_hours"]),
            "number_of_unique_grid_cells": len(raw["grid_cells"]),
            "paired_cells_per_date": raw["valid_cells_by_date"],
            "missing_values": raw["missing_values"],
            "duplicate_rows": raw["duplicate_rows"],
            "observed_rainfall_mm": target_dist,
            "gfs_daily_rainfall_mm": gfs_dist,
            "extreme_event_counts": events,
            "regime_distribution": regimes,
        }
        row: dict[str, Any] = {
            "split": split,
            "date_start": report[split]["date_range"][0], "date_end": report[split]["date_range"][1],
            "forecast_cases": report[split]["number_of_forecast_cases"], "rows": raw["rows"],
            "unique_dates": len(raw["dates"]), "unique_forecast_cycles": len(raw["forecast_initializations"]),
            "cycle_hours": ";".join(sorted(raw["cycle_hours"])), "unique_grid_cells": len(raw["grid_cells"]),
            "missing_values": raw["missing_values"], "duplicate_rows": raw["duplicate_rows"],
            "observed_mean_mm": target_dist["mean"], "observed_p95_mm": target_dist["p95"],
            "observed_max_mm": target_dist["max"], "gfs_daily_mean_mm": gfs_dist["mean"],
            "gfs_daily_p95_mm": gfs_dist["p95"], "gfs_daily_max_mm": gfs_dist["max"],
        }
        for key, _ in thresholds:
            row[f"observed_{key}_count"] = events[key]["observed_count"]
            row[f"gfs_daily_{key}_count"] = events[key]["gfs_daily_count"]
        for label, value in regimes.items():
            prefix = "regime_" + label.lower()
            row[f"{prefix}_rows"] = value["rows"]
            row[f"{prefix}_unique_dates"] = value["unique_dates"]
            row[f"{prefix}_unique_forecast_cycles"] = value["unique_forecast_cycles"]
        csv_rows.append(row)
    return report, csv_rows


def write_split_partitions(
    dates: list[date], imd: IMDRF25Store, provider: UCARGFSSSubsetProvider,
    root: Path = ROOT, output_dir: Path | None = None, stage: str = "full",
    limit: int | None = None,
) -> dict[str, Any]:
    """Build or resume raw cases, then write sample/full splits atomically."""
    if stage not in {"sample", "full"}:
        raise ValueError("stage must be sample or full")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        dates = dates[:limit]
    if not dates:
        raise VerifiedDatasetError("No candidate target dates were supplied")
    legacy_test_path = root / LEGACY_TEST_PATH
    legacy_test_hash_before = _sha256(legacy_test_path) if legacy_test_path.exists() else None
    if legacy_test_hash_before != LEGACY_TEST_EXPECTED_SHA256:
        raise VerifiedDatasetError(
            "Legacy 2025 test.parquet is missing or its hash differs from the recorded pre-phase value; "
            "stopping before any GFS retrieval."
        )
    split_stats = {name: _empty_split_stats() for name in ("train", "validation", "test")}
    output_dir = output_dir or (root / "data" / "processed" / ("verified_sample" if stage == "sample" else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_sample" if stage == "sample" else "_verified"
    final_paths = {name: output_dir / f"{name}{suffix}.parquet" for name in split_stats}
    if any(path.exists() for path in final_paths.values()):
        raise FileExistsError("Verified split output exists; refusing overwrite")
    partial_paths = {name: path.with_suffix(path.suffix + ".part") for name, path in final_paths.items()}
    for path in partial_paths.values():
        path.unlink(missing_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    failures: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = []
    progress_path = root / RAW_MANIFEST_ROOT / f"{stage}_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, target_date in enumerate(dates, start=1):
            try:
                result = build_or_resume_case(target_date, imd, provider, root)
                frame, cell_counts = case_to_frame(result, target_date, imd)
                _case_statistics(frame, cell_counts, split_stats)
                split = result.split
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if split not in writers:
                    writers[split] = pq.ParquetWriter(partial_paths[split], table.schema, compression="zstd")
                elif not writers[split].schema.equals(table.schema, check_metadata=False):
                    raise VerifiedDatasetError(f"Parquet schema drift between cases in {split}")
                writers[split].write_table(table, row_group_size=len(frame))
                completed.append({
                    "target_date": target_date.isoformat(), "split": split,
                    "raw_case": result.path.relative_to(root).as_posix(), "raw_manifest": result.manifest_path.relative_to(root).as_posix(),
                    "resumed_from_cache": result.resumed, "paired_cells": cell_counts["valid_paired_cells"],
                    "rows": len(frame),
                })
                if index == 1 or index % 10 == 0 or index == len(dates):
                    print(f"[{stage}] {index}/{len(dates)} target dates complete; latest={target_date}; rows={len(frame):,}", flush=True)
            except Exception as exc:
                failures.append({"target_date": target_date.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{stage}] FAIL {target_date}: {type(exc).__name__}: {exc}", flush=True)
            progress_path.write_text(json.dumps({
                "status": "RUNNING", "stage": stage, "expected_cases": len(dates),
                "completed_cases": len(completed), "failed_cases": len(failures),
                "completed": completed, "failures": failures,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n", encoding="utf-8")
    finally:
        for writer in writers.values():
            writer.close()
    if failures or len(completed) != len(dates):
        for path in partial_paths.values():
            path.unlink(missing_ok=True)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["status"] = "INCOMPLETE_STOPPED"
        progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        raise VerifiedDatasetError(
            f"Only {len(completed)}/{len(dates)} cases succeeded; {len(failures)} failed. "
            f"No split was committed. See {progress_path.relative_to(root)}; rerun to resume cached cases."
        )

    split_report, csv_rows = summarize_splits(split_stats)
    for split, final in final_paths.items():
        partial = partial_paths[split]
        if not partial.exists() or pq.ParquetFile(partial).metadata.num_rows <= 0:
            raise VerifiedDatasetError(f"Non-empty {split} split was not produced")
        partial.replace(final)
    file_hashes = {split: _sha256(path) for split, path in final_paths.items()}
    legacy_test_hash_after = _sha256(legacy_test_path) if legacy_test_path.exists() else None
    if legacy_test_hash_after != legacy_test_hash_before:
        raise VerifiedDatasetError("Legacy 2025 test.parquet changed during verified dataset construction")
    report = {
        "status": "BOUNDED_SAMPLE_VALIDATED" if stage == "sample" else "VERIFIED_DATASET_READY_FOR_READINESS_GATE",
        "stage": stage,
        "processing_version": PROCESSING_VERSION,
        "target_source": IMD_SOURCE,
        "forecast_source": GFS_ARCHIVE_VERSION,
        "period": {"years": list(YEARS), "months": list(MONTHS), "stage_candidate_dates": len(dates)},
        "split_definition": {"train": "target years 2020-2022", "validation": "target year 2023", "test": "target year 2024"},
        "forecast_selection": "One latest standard forecast cycle prospectively covering the complete IMD day: 00Z on D-1; atmosphere at D 03Z lead 27; APCP records selected by their GRIB accumulation bounds.",
        "target_window_utc": "For IMD date D: [D-1 03:00Z, D 03:00Z)",
        "spatial_mapping": {
            "method": "Exact 0.25-degree grid-to-grid coordinate equality; no regridding or domain-mean comparison.",
            "bbox": {"south": 8.0, "north": 37.0, "west": 68.0, "east": 97.0},
            "gfs_grid_dimensions": [117, 117], "latitude_order": "descending", "longitude_convention": "positive east, 0-360",
            "gfs_grid_cells": 117 * 117,
            "imd_common_valid_grid_cells_per_date": sorted(set(v for s in split_stats.values() for v in s["imd_valid_cells_by_date"].values())),
            "valid_paired_cell_counts_per_date": {k: v for s in split_stats.values() for k, v in s["valid_cells_by_date"].items()},
        },
        "predictor_policy": {
            "primary_rain_feature": "gfs_daily_rain_mm", "observed_target": "observed_rain_mm",
            "deprecated_gfs_rain_mm_used": False, "raw_apcp_intervals_saved_in_case_netcdf": True,
            "rain_last_24h": "RF25 observation labelled D-2; its 03Z end is strictly earlier than initialization 00Z on D-1.",
            "rain_last_6h": "Excluded: daily RF25 cannot support a sub-daily 6h antecedent feature.",
            "target_excluded_from_FEATURE_SCHEMA": "observed_rain_mm is label-only and never a predictor.",
            "feature_schema": list(FEATURE_SCHEMA),
        },
        "data_quality": split_report,
        "case_inventory": completed,
        "case_failures": failures,
        "split_files": {split: {"path": path.relative_to(root).as_posix(), "rows": split_report[split]["number_of_rows"], "sha256": file_hashes[split]} for split, path in final_paths.items()},
        "legacy_2025_test_artifact": {
            "path": LEGACY_TEST_PATH.as_posix(), "sha256_before": legacy_test_hash_before,
            "sha256_after": legacy_test_hash_after, "used": False, "modified": False,
        },
        "model_training_started": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    csv_path = root / "data" / "results" / ("verified_dataset_sample_summary.csv" if stage == "sample" else "verified_dataset_summary.csv")
    json_path = root / "data" / "results" / ("verified_dataset_sample_report.json" if stage == "sample" else "verified_dataset_report.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    progress_path.write_text(json.dumps({
        "status": report["status"], "stage": stage, "expected_cases": len(dates),
        "completed_cases": len(completed), "failed_cases": 0, "report": json_path.relative_to(root).as_posix(),
        "split_files": report["split_files"], "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    return report
