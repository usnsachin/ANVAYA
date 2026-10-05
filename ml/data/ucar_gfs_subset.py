"""Strict historical GFS retrieval through UCAR THREDDS NCSS.

This provider is deliberately separate from :class:`GFSProvider`. It inspects
the selected archive file's live THREDDS metadata, makes only bounded NCSS
requests, rejects missing variables, and records every request and checksum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

from netCDF4 import num2date
import numpy as np
import requests
import xarray as xr
import yaml

from .providers import BaseProvider, INDIA_BBOX


ROOT = Path(__file__).resolve().parents[2]
FEATURES: dict[str, "FieldSpec"]


@dataclass(frozen=True)
class FieldSpec:
    feature: str
    variable: str
    parameter: str
    level_type: str
    units: str
    vert_coord: float | None = None


FEATURES = {
    "gfs_apcp_interval_mm": FieldSpec(
        "gfs_apcp_interval_mm", "Total_precipitation_surface_Mixed_intervals_Accumulation",
        "0 1 8", "1", "kg.m-2",
    ),
    "gfs_mslp": FieldSpec(
        "gfs_mslp", "Pressure_reduced_to_MSL_msl", "0 3 1", "101", "Pa",
    ),
    "gfs_t2m": FieldSpec(
        "gfs_t2m", "Temperature_height_above_ground", "0 0 0", "103", "K",
        vert_coord=2.0,
    ),
    "gfs_rh2m": FieldSpec(
        "gfs_rh2m", "Relative_humidity_height_above_ground", "0 1 1", "103", "%",
        vert_coord=2.0,
    ),
    "gfs_u10": FieldSpec(
        "gfs_u10", "u-component_of_wind_height_above_ground", "0 2 2", "103", "m/s",
        vert_coord=10.0,
    ),
    "gfs_v10": FieldSpec(
        "gfs_v10", "v-component_of_wind_height_above_ground", "0 2 3", "103", "m/s",
        vert_coord=10.0,
    ),
    "gfs_u850": FieldSpec(
        "gfs_u850", "u-component_of_wind_isobaric", "0 2 2", "100", "m/s",
        vert_coord=85000.0,
    ),
    "gfs_v850": FieldSpec(
        "gfs_v850", "v-component_of_wind_isobaric", "0 2 3", "100", "m/s",
        vert_coord=85000.0,
    ),
    "gfs_cape": FieldSpec(
        "gfs_cape", "Convective_available_potential_energy_surface", "0 7 6", "1", "J/kg",
    ),
    "gfs_precipitable_water": FieldSpec(
        "gfs_precipitable_water", "Precipitable_water_entire_atmosphere_single_layer",
        "0 1 3", "200", "kg.m-2",
    ),
}

# NCSS grid requests can combine fields only when their vertical selection is
# unambiguous. Live bounded checks established that the single-layer surface
# fields and same-level wind components can be grouped. 2 m TMP and RH have
# different level sets, so keep them separate.
REQUEST_GROUPS: tuple[tuple[str, ...], ...] = (
    ("gfs_apcp_interval_mm",),
    ("gfs_mslp", "gfs_cape", "gfs_precipitable_water"),
    ("gfs_t2m",),
    ("gfs_rh2m",),
    ("gfs_u10", "gfs_v10"),
    ("gfs_u850", "gfs_v850"),
)


class UCARGFSSSubsetError(RuntimeError):
    """Raised for metadata, retrieval, or validation failures; no demo fallback."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("init_time must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata_grid_records(xml_content: bytes) -> tuple[ET.Element, dict[str, dict[str, str]]]:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise UCARGFSSSubsetError(f"UCAR returned invalid dataset XML: {exc}") from exc
    records: dict[str, dict[str, str]] = {}
    for grid in (node for node in root.iter() if _local_name(node.tag) == "grid"):
        name = grid.attrib.get("name")
        if not name:
            continue
        attrs = {"description": grid.attrib.get("desc", "")}
        for node in grid.iter():
            if _local_name(node.tag) == "attribute" and node.attrib.get("name"):
                attrs[node.attrib["name"]] = node.attrib.get("value", "")
        records[name] = attrs
    return root, records


def _metadata_axes(root: ET.Element) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for axis in (node for node in root.iter() if _local_name(node.tag) == "axis"):
        attrs: dict[str, str] = {}
        values: ET.Element | None = None
        for child in axis:
            if _local_name(child.tag) == "attribute" and child.attrib.get("name"):
                attrs[child.attrib["name"]] = child.attrib.get("value", "")
            elif _local_name(child.tag) == "values":
                values = child
        values_attrs = dict(values.attrib) if values is not None else {}
        points: list[float] = []
        if values is not None:
            raw_points = (values.text or "").split()
            for raw in raw_points:
                try:
                    points.append(float(raw))
                except ValueError:
                    pass
            for key in ("start", "end"):
                if key in values_attrs:
                    try:
                        points.append(float(values_attrs[key]))
                    except ValueError:
                        pass
        axes.append({
            "name": axis.attrib.get("name"),
            "axis_type": axis.attrib.get("axisType"),
            "units": attrs.get("units"),
            "udunits": attrs.get("udunits", attrs.get("units", "")),
            "values": sorted(set(points)),
            "range_start": values_attrs.get("start"),
            "range_end": values_attrs.get("end"),
        })
    return axes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data: xr.DataArray) -> float:
    values = np.asarray(data.values, dtype=float).squeeze()
    if values.size != 1:
        raise UCARGFSSSubsetError(
            f"Expected one value after spatial selection for {data.name}; got shape {data.shape}"
        )
    return float(values.reshape(-1)[0])


def _coordinate_time_utc(value: Any) -> datetime:
    """Convert a decoded NetCDF time coordinate to an aware UTC datetime."""
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="s")
    elif isinstance(value, datetime):
        parsed = value
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    elif hasattr(value, "isoformat"):
        text = value.isoformat()
    else:
        raise UCARGFSSSubsetError(f"Unsupported NetCDF time coordinate value: {value!r}")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _decode_numeric_cf_time_variables(dataset: xr.Dataset) -> xr.Dataset:
    """Decode CF time axes/bounds using netCDF4 when xarray lacks cftime.

    Some archived GFS messages use a capitalized ``Hour since ...`` unit
    string. Xarray's normal CF decoder may require the optional cftime package
    for that spelling, although netCDF4 can decode the same standard-calendar
    coordinates deterministically.
    """
    referenced_bounds = {
        str(coord.attrs["bounds"])
        for coord in dataset.coords.values()
        if coord.attrs.get("bounds")
    }
    result = dataset
    for name in list(result.variables):
        variable = result[name]
        units = variable.attrs.get("units")
        if not isinstance(units, str) or " since " not in units.casefold():
            continue
        is_time = (
            variable.attrs.get("axis") == "T"
            or variable.attrs.get("standard_name") == "time"
            or name.casefold() in {"time", "time1", "time2", "reftime"}
            or name in referenced_bounds
            or name.casefold().endswith("_bounds")
        )
        if not is_time:
            continue
        calendar = str(variable.attrs.get("calendar", "standard"))
        raw = np.asarray(variable.values)
        try:
            decoded = num2date(
                raw,
                units=units,
                calendar=calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            flat: list[np.datetime64] = []
            for value in np.asarray(decoded, dtype=object).reshape(-1):
                if isinstance(value, datetime):
                    parsed = value.replace(tzinfo=None)
                else:
                    parsed = datetime.fromisoformat(value.isoformat().replace("Z", "+00:00")).replace(tzinfo=None)
                flat.append(np.datetime64(parsed, "ns"))
            values = np.asarray(flat, dtype="datetime64[ns]").reshape(raw.shape)
        except Exception as exc:
            raise UCARGFSSSubsetError(
                f"Could not decode numeric CF time variable {name!r} "
                f"(units={units!r}, calendar={calendar!r}): {exc}"
            ) from exc
        attrs = dict(variable.attrs)
        attrs["source_time_units"] = attrs.pop("units")
        attrs["source_calendar"] = attrs.pop("calendar", calendar)
        if name in result.coords:
            result = result.assign_coords({name: (variable.dims, values, attrs)})
        else:
            result[name] = xr.DataArray(values, dims=variable.dims, attrs=attrs)
    return result


class UCARGFSSSubsetProvider(BaseProvider):
    """Strict one-forecast historical provider using UCAR THREDDS NCSS."""

    source_name = "UCAR_GFS_0p25_THREDDS_NCSS"
    mode = "historical"

    def __init__(
        self,
        root: Path | str = ROOT,
        bbox: dict[str, float] | None = None,
        timeout: int = 120,
        session: requests.Session | None = None,
        dataset_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.bbox = dict(bbox or INDIA_BBOX)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.dataset_path = self._workspace_path(
            dataset_path, Path("data/raw/gfs/test_subset/gfs_0p25_20240715_00_f024_india.nc")
        )
        self.manifest_path = self._workspace_path(
            manifest_path, Path("data/raw/manifests/gfs_test_subset.json")
        )
        config_path = self.root / "config" / "data_sources.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source = next((x for x in config["sources"] if x["id"] == "gfs_0p25"), None)
        if source is None:
            raise UCARGFSSSubsetError("config/data_sources.yaml has no gfs_0p25 source")
        archive = source.get("historical_archive", {})
        if not archive.get("ucar_ncss_template"):
            raise UCARGFSSSubsetError("UCAR NCSS endpoint is missing from the existing GFS source registry")
        self.dataset_template = archive["ucar_ncss_template"]
        self.dataset_doi = "10.5065/D65D8PWK"

    def _workspace_path(self, value: Path | str | None, default: Path) -> Path:
        path = Path(value) if value is not None else default
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if self.root not in path.parents:
            raise ValueError("Provider cache paths must remain inside the project workspace")
        return path

    def _dataset_url(self, init_time: datetime, lead_hours: int) -> str:
        date = init_time.strftime("%Y%m%d")
        return self.dataset_template.format(
            YYYY=init_time.strftime("%Y"),
            YYYYMMDD=date,
            CC=init_time.strftime("%H"),
            FFF=f"{lead_hours:03d}",
        )

    def _inspect_metadata(
        self, dataset_url: str, init_time: datetime, lead_hours: int
    ) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
        response = self.session.get(dataset_url + "/dataset.xml", timeout=self.timeout)
        response.raise_for_status()
        if len(response.content) > 10 * 1024 * 1024:
            raise UCARGFSSSubsetError("Unexpectedly large THREDDS metadata response; refusing to continue")
        root, grids = _metadata_grid_records(response.content)
        missing = [spec.variable for spec in FEATURES.values() if spec.variable not in grids]
        mismatched: list[str] = []
        for spec in FEATURES.values():
            record = grids.get(spec.variable)
            if not record:
                continue
            expected = {
                "Grib2_Parameter": spec.parameter,
                "Grib2_Level_Type": spec.level_type,
                "units": spec.units,
            }
            mismatched.extend(
                f"{spec.feature}: {key}={record.get(key)!r}, expected {value!r}"
                for key, value in expected.items()
                if record.get(key) != value
            )
        if missing or mismatched:
            details = []
            if missing:
                details.append("missing metadata variables: " + ", ".join(missing))
            if mismatched:
                details.append("metadata mapping mismatch: " + "; ".join(mismatched))
            raise UCARGFSSSubsetError("; ".join(details))

        axes = _metadata_axes(root)
        longitude_axes = [a for a in axes if a["axis_type"] == "Lon"]
        if not longitude_axes:
            raise UCARGFSSSubsetError("THREDDS metadata has no longitude axis")
        lon_axis = longitude_axes[0]
        if lon_axis["units"] != "degrees_east" or not lon_axis["values"]:
            raise UCARGFSSSubsetError("Cannot establish longitude convention from THREDDS metadata")
        lon_convention = "0_to_360_east" if min(lon_axis["values"]) >= 0 else "-180_to_180"
        if lon_convention != "0_to_360_east":
            raise UCARGFSSSubsetError(
                f"Unexpected longitude convention {lon_convention}; refusing positive-east bbox request"
            )

        lat_axes = [a for a in axes if a["axis_type"] == "Lat"]
        if not lat_axes or lat_axes[0]["units"] != "degrees_north":
            raise UCARGFSSSubsetError("Cannot establish latitude convention from THREDDS metadata")

        for spec in FEATURES.values():
            if spec.vert_coord is None:
                continue
            axis_type = "Pressure" if spec.level_type == "100" else "Height"
            candidates = [a for a in axes if a["axis_type"] == axis_type]
            if not any(any(np.isclose(point, spec.vert_coord) for point in a["values"]) for a in candidates):
                raise UCARGFSSSubsetError(
                    f"Requested level {spec.vert_coord} ({axis_type}) is absent from THREDDS coordinates for {spec.feature}"
                )

        init_text = _utc_iso(init_time)
        time_axes = [a for a in axes if a["axis_type"] == "Time"]
        reference_matches = any(init_text.removesuffix("Z") in a["udunits"] for a in axes)
        valid_offset_matches = any(
            a["range_end"] is not None and np.isclose(float(a["range_end"]), lead_hours)
            for a in time_axes
        )
        if not reference_matches or not valid_offset_matches:
            raise UCARGFSSSubsetError(
                "THREDDS time coordinates do not confirm the requested initialization and lead"
            )

        return grids, axes, {
            "longitude_convention": lon_convention,
            "longitude_metadata_units": lon_axis["units"],
            "source_grid_axis_endpoints": {
                "latitude": next((a.get("values") for a in lat_axes), []),
                "longitude": lon_axis["values"],
            },
            "forecast_reference_metadata": init_text,
            "valid_lead_hours_confirmed": lead_hours,
            "metadata_url": dataset_url + "/dataset.xml",
            "dataset_html_url": dataset_url + "/dataset.html",
        }

    def _query_groups(self) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
        result = []
        for group in REQUEST_GROUPS:
            specs = [FEATURES[name] for name in group]
            params: dict[str, Any] = {
                "var": ",".join(spec.variable for spec in specs),
                "north": self.bbox["max_lat"],
                "south": self.bbox["min_lat"],
                "west": self.bbox["min_lon"],
                "east": self.bbox["max_lon"],
                # This historical GDEX NCSS instance advertises netcdf3 only;
                # xarray's netcdf4 engine can read both classic and NetCDF4 files.
                "accept": "netcdf3",
            }
            levels = {spec.vert_coord for spec in specs if spec.vert_coord is not None}
            if len(levels) > 1:
                raise UCARGFSSSubsetError(f"Incompatible levels in NCSS request group {group}")
            if levels:
                params["vertCoord"] = next(iter(levels))
            result.append((group, params))
        return result

    def _download_group(
        self,
        dataset_url: str,
        init_time: datetime,
        lead_hours: int,
        group: tuple[str, ...],
        params: dict[str, Any],
        temp_path: Path,
    ) -> dict[str, Any]:
        valid_time = init_time + timedelta(hours=lead_hours)
        query = dict(params)
        query["time"] = _utc_iso(valid_time)
        request = requests.Request("GET", dataset_url, params=query).prepare()
        with self.session.get(dataset_url, params=query, stream=True, timeout=self.timeout) as response:
            if response.status_code != 200:
                preview = response.text[:500].replace("\n", " ")
                raise UCARGFSSSubsetError(
                    f"NCSS request failed ({response.status_code}) for {group}: {preview}"
                )
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "text/plain" in content_type:
                preview = response.text[:500].replace("\n", " ")
                raise UCARGFSSSubsetError(f"NCSS returned an error document for {group}: {preview}")
            size = 0
            with temp_path.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        stream.write(chunk)
                        size += len(chunk)
            if size < 8:
                raise UCARGFSSSubsetError(f"NCSS returned an empty response for {group}")

        # Open with the explicit NetCDF4 engine; no GRIB or demo fallback.
        try:
            with xr.open_dataset(temp_path, engine="netcdf4") as opened:
                downloaded = opened.load()
        except Exception as exc:
            if "decode time units" not in str(exc).casefold():
                raise UCARGFSSSubsetError(
                    f"NCSS response for {group} is not readable NetCDF4: status=200; "
                    f"url={request.url}; response_bytes={temp_path.stat().st_size}; "
                    f"response_sha256={_sha256(temp_path)}; error={exc}"
                ) from exc
            try:
                with xr.open_dataset(temp_path, engine="netcdf4", decode_times=False) as opened:
                    downloaded = _decode_numeric_cf_time_variables(opened.load())
            except Exception as fallback_exc:
                raise UCARGFSSSubsetError(
                    f"NCSS CF-time decode failed for {group}: status=200; url={request.url}; "
                    f"response_bytes={temp_path.stat().st_size}; response_sha256={_sha256(temp_path)}; "
                    f"error={fallback_exc}"
                ) from fallback_exc
        requested_vars = {FEATURES[name].variable for name in group}
        missing = requested_vars.difference(downloaded.data_vars)
        unexpected = set(downloaded.data_vars).difference(requested_vars)
        bounds_vars = {name for name in unexpected if name.endswith("_bounds")}
        coordinate_bounds = {
            coord.attrs.get("bounds")
            for coord in downloaded.coords.values()
            if coord.attrs.get("bounds")
        }
        grid_mapping_refs = {
            downloaded[name].attrs.get("grid_mapping")
            for name in requested_vars
            if name in downloaded.data_vars and downloaded[name].attrs.get("grid_mapping")
        }
        mapping_vars = {
            name for name in unexpected
            if name.startswith("LatLon_") and downloaded[name].ndim == 0
        }
        referenced_mappings = mapping_vars.intersection(grid_mapping_refs)
        unrecognized = unexpected.difference(bounds_vars).difference(referenced_mappings)
        unreferenced_bounds = bounds_vars.difference(coordinate_bounds)
        unreferenced_mappings = mapping_vars.difference(grid_mapping_refs)
        if missing or unrecognized or unreferenced_bounds or unreferenced_mappings:
            raise UCARGFSSSubsetError(
                f"NCSS variable mismatch for {group}: missing={sorted(missing)}, "
                f"unexpected={sorted(unrecognized)}, unreferenced_CF_bounds={sorted(unreferenced_bounds)}, "
                f"unreferenced_grid_mappings={sorted(unreferenced_mappings)}; status=200; "
                f"url={request.url}; response_bytes={temp_path.stat().st_size}; "
                f"response_sha256={_sha256(temp_path)}"
            )
        ancillary_coords = bounds_vars.union(referenced_mappings)
        if ancillary_coords:
            downloaded = downloaded.set_coords(sorted(ancillary_coords))
        return {
            "features": list(group),
            "variables": sorted(requested_vars),
            "url": request.url,
            "bytes": size,
            "sha256": _sha256(temp_path),
            "dataset": downloaded,
        }

    def retrieve_forecast(
        self,
        init_time: datetime,
        lead_hours: int = 24,
        output_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Fetch and validate exactly one forecast initialization/lead."""
        if init_time.tzinfo is None:
            raise ValueError("init_time must include a timezone")
        init_time = init_time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if lead_hours < 0 or lead_hours > 240:
            raise ValueError("lead_hours must be between 0 and 240 for this archive workflow")
        valid_time = init_time + timedelta(hours=lead_hours)
        stem = f"gfs_0p25_{init_time:%Y%m%d}_{init_time:%H}_f{lead_hours:03d}_india"
        output = Path(output_path) if output_path else self.root / "data/raw/gfs/test_subset" / f"{stem}.nc"
        manifest = (
            Path(manifest_path)
            if manifest_path
            else self.root / "data/raw/manifests" / (
                "gfs_test_subset.json"
                if stem == "gfs_0p25_20240715_00_f024_india"
                else f"{stem}.json"
            )
        )
        if not output.is_absolute():
            output = self.root / output
        if not manifest.is_absolute():
            manifest = self.root / manifest
        output = output.resolve()
        manifest = manifest.resolve()
        if self.root not in output.parents or self.root not in manifest.parents:
            raise ValueError("Output and manifest paths must remain inside the project workspace")
        if (output.exists() or manifest.exists()) and not force:
            raise FileExistsError("Refusing to overwrite subset or manifest; pass force=True explicitly")

        dataset_url = self._dataset_url(init_time, lead_hours)
        grids, axes, source_meta = self._inspect_metadata(dataset_url, init_time, lead_hours)
        requests_used: list[dict[str, Any]] = []
        datasets: list[xr.Dataset] = []
        transfer_bytes = 0
        with tempfile.TemporaryDirectory(prefix="anvaya-ucar-gfs-") as work_dir:
            work = Path(work_dir)
            for index, (group, base_params) in enumerate(self._query_groups(), start=1):
                response = self._download_group(
                    dataset_url, init_time, lead_hours, group, base_params,
                    work / f"request_{index:02d}.nc",
                )
                requests_used.append({k: v for k, v in response.items() if k != "dataset"})
                datasets.append(response["dataset"])
                transfer_bytes += response["bytes"]

        try:
            merged = xr.merge(
                datasets,
                compat="no_conflicts",
                join="exact",
                combine_attrs="drop_conflicts",
            )
        except Exception as exc:
            raise UCARGFSSSubsetError(f"Could not combine compatible NCSS subset groups: {exc}") from exc

        expected_names = {spec.variable for spec in FEATURES.values()}
        missing_names = expected_names.difference(merged.data_vars)
        extra_names = set(merged.data_vars).difference(expected_names)
        if missing_names or extra_names:
            raise UCARGFSSSubsetError(
                f"Final subset variable mismatch: missing={sorted(missing_names)}, unexpected={sorted(extra_names)}"
            )

        lat = np.asarray(merged["latitude"].values, dtype=float)
        lon = np.asarray(merged["longitude"].values, dtype=float)
        if lat.ndim != 1 or lon.ndim != 1 or lat.size == 0 or lon.size == 0:
            raise UCARGFSSSubsetError("Expected nonempty one-dimensional latitude/longitude coordinates")
        if lat.size >= 721 or lon.size >= 1440 or np.isclose(np.max(lat), 90.0) or np.isclose(np.min(lat), -90.0):
            raise UCARGFSSSubsetError("NCSS returned a global or unbounded latitude grid")
        if np.min(lat) < self.bbox["min_lat"] - 1e-6 or np.max(lat) > self.bbox["max_lat"] + 1e-6:
            raise UCARGFSSSubsetError("Returned latitude coordinates exceed the requested bbox")
        if np.min(lon) < self.bbox["min_lon"] - 1e-6 or np.max(lon) > self.bbox["max_lon"] + 1e-6:
            raise UCARGFSSSubsetError("Returned longitude coordinates exceed the requested bbox")
        if lat.size > 125 or lon.size > 125:
            raise UCARGFSSSubsetError("Returned grid is larger than expected for the requested India bbox")

        precip_name = FEATURES["gfs_apcp_interval_mm"].variable
        rain = np.asarray(merged[precip_name].values, dtype=float)
        finite_rain = rain[np.isfinite(rain)]
        if finite_rain.size == 0:
            raise UCARGFSSSubsetError("Precipitation field contains no finite values")
        if float(finite_rain.min()) < -1e-5 or float(finite_rain.max()) > 2000.0:
            raise UCARGFSSSubsetError(
                f"Precipitation values are physically implausible for kg m-2: min={finite_rain.min()}, max={finite_rain.max()}"
            )

        init_text = _utc_iso(init_time)
        valid_text = _utc_iso(valid_time)
        if datetime.fromisoformat(valid_text.replace("Z", "+00:00")) != init_time + timedelta(hours=lead_hours):
            raise UCARGFSSSubsetError("valid_time is inconsistent with init_time + lead_hours")
        if "reftime" not in merged.coords:
            raise UCARGFSSSubsetError("Subset output lacks forecast-reference time coordinates")
        observed_reftime = _coordinate_time_utc(np.asarray(merged["reftime"].values).reshape(-1)[0])
        # NCSS can rename valid-time coordinates when compatible variables are
        # grouped (for example time1/time2). Validate every retained CF time
        # axis against the same requested instant before adding a canonical
        # scalar `time` coordinate for downstream readers.
        time_coordinate_audit: list[dict[str, Any]] = []
        invalid_time_axes: dict[str, Any] = {}
        for name, coordinate in merged.coords.items():
            if name == "reftime" or name.endswith("_bounds") or coordinate.ndim > 1:
                continue
            if (
                coordinate.attrs.get("axis") == "T"
                or coordinate.attrs.get("standard_name") == "time"
                or name.lower() in {"time", "time1", "time2"}
            ):
                values = [
                    _coordinate_time_utc(value)
                    for value in np.asarray(coordinate.values).reshape(-1)
                ]
                bounds_name = coordinate.attrs.get("bounds")
                if bounds_name:
                    if bounds_name not in merged:
                        invalid_time_axes[name] = f"bounds variable {bounds_name!r} is missing"
                        continue
                    bounds_values = np.asarray(merged[bounds_name].values).reshape(-1)
                    if bounds_values.size != 2:
                        invalid_time_axes[name] = f"expected 2 bounds, got {bounds_values.size}"
                        continue
                    interval_start, interval_end = map(_coordinate_time_utc, bounds_values)
                    if interval_start >= interval_end or interval_end != valid_time:
                        invalid_time_axes[name] = {
                            "interval_start": interval_start.isoformat(),
                            "interval_end": interval_end.isoformat(),
                        }
                    time_coordinate_audit.append({
                        "coordinate": name,
                        "coordinate_values": [value.isoformat() for value in values],
                        "metadata_bounds": [interval_start.isoformat(), interval_end.isoformat()],
                        "semantics": "bounded_accumulation; end must equal requested valid time",
                    })
                else:
                    if not values or any(value != valid_time for value in values):
                        invalid_time_axes[name] = [value.isoformat() for value in values]
                    time_coordinate_audit.append({
                        "coordinate": name,
                        "coordinate_values": [value.isoformat() for value in values],
                        "semantics": "instantaneous_forecast_valid_time",
                    })
        if not time_coordinate_audit:
            raise UCARGFSSSubsetError("Subset output lacks metadata-identified valid-time coordinates")
        if observed_reftime != init_time or invalid_time_axes:
            raise UCARGFSSSubsetError(
                f"NetCDF time coordinates disagree with request: reftime={observed_reftime.isoformat()}, "
                f"invalid_axes={invalid_time_axes}, expected init={init_text}, valid={valid_text}"
            )
        if "time" in merged.coords:
            time_coordinate = merged["time"]
            if not time_coordinate.attrs.get("bounds"):
                normalized_values = [
                    _coordinate_time_utc(value)
                    for value in np.asarray(time_coordinate.values).reshape(-1)
                ]
                if not normalized_values or any(value != valid_time for value in normalized_values):
                    raise UCARGFSSSubsetError("Existing instantaneous time coordinate disagrees with request")
            # If this is an accumulation time coordinate, the CF bounds audit
            # above already proved its interval end equals the forecast valid time.
        elif "time" in merged.dims:
            if merged.sizes["time"] != 1:
                raise UCARGFSSSubsetError("Cannot normalize multi-valued unnamed forecast time axis")
            merged = merged.assign_coords(time=("time", [np.datetime64(valid_time.replace(tzinfo=None))]))
        elif "time" in merged.variables:
            raise UCARGFSSSubsetError("An unrelated data variable named 'time' blocks coordinate normalization")
        else:
            merged = merged.assign_coords(time=np.datetime64(valid_time.replace(tzinfo=None)))
        merged = merged.assign_coords(
            forecast_valid_time=np.datetime64(valid_time.replace(tzinfo=None))
        )
        precip_dims = [dim for dim in merged[precip_name].dims if dim not in {"latitude", "longitude"}]
        if len(precip_dims) != 1:
            raise UCARGFSSSubsetError(f"Cannot identify precipitation time axis: {merged[precip_name].dims}")
        precip_time_coord = merged.coords.get(precip_dims[0])
        bounds_name = precip_time_coord.attrs.get("bounds") if precip_time_coord is not None else None
        if not bounds_name or bounds_name not in merged:
            raise UCARGFSSSubsetError("Precipitation output lacks CF time bounds; accumulation interval is unknown")
        bounds_values = np.asarray(merged[bounds_name].values).reshape(-1)
        if bounds_values.size != 2:
            raise UCARGFSSSubsetError("Expected one precipitation interval with start/end time bounds")
        precip_start = _coordinate_time_utc(bounds_values[0])
        precip_end = _coordinate_time_utc(bounds_values[-1])
        if precip_end != valid_time or precip_start >= precip_end:
            raise UCARGFSSSubsetError(
                f"Precipitation bounds {precip_start.isoformat()}..{precip_end.isoformat()} "
                f"do not terminate at forecast valid time {valid_text}"
            )
        precip_interval_hours = (precip_end - precip_start).total_seconds() / 3600.0
        merged.attrs.update({
            "source": self.source_name,
            "source_dataset": "NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive",
            "source_dataset_doi": self.dataset_doi,
            "source_dataset_url": "https://gdex.k8s.ucar.edu/datasets/d084001/",
            "source_file_url": dataset_url,
            "init_time": init_text,
            "valid_time": valid_text,
            "lead_hours": int(lead_hours),
            "bbox_south": float(self.bbox["min_lat"]),
            "bbox_north": float(self.bbox["max_lat"]),
            "bbox_west": float(self.bbox["min_lon"]),
            "bbox_east": float(self.bbox["max_lon"]),
            "longitude_convention": source_meta["longitude_convention"],
            "precipitation_units_note": "kg m-2 liquid-water equivalent; numerically equivalent to mm",
            "precipitation_interval_note": f"Accumulation interval {precip_start.isoformat()} to {precip_end.isoformat()} ({precip_interval_hours:g} hours)",
        })

        output.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        partial_output = output.with_suffix(output.suffix + ".part")
        partial_manifest = manifest.with_suffix(manifest.suffix + ".part")
        try:
            merged.to_netcdf(partial_output, engine="netcdf4")
            with xr.open_dataset(partial_output, engine="netcdf4") as check:
                check.load()
                final_names = set(check.data_vars)
                if final_names != expected_names:
                    raise UCARGFSSSubsetError("Written NetCDF failed strict variable round-trip validation")
            if output.exists() and force:
                output.unlink()
            partial_output.replace(output)

            variable_manifest = {
                feature: {
                    **asdict(spec),
                    "metadata": grids[spec.variable],
                }
                for feature, spec in FEATURES.items()
            }
            manifest_data = {
                "status": "VALIDATED_SINGLE_CASE",
                "source": {
                    "provider": self.source_name,
                    "dataset": "NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive",
                    "doi": self.dataset_doi,
                    "dataset_url": "https://gdex.k8s.ucar.edu/datasets/d084001/",
                    "source_file_ncss_url": dataset_url,
                    "source_metadata": source_meta,
                },
                "forecast": {
                    "initialization": init_text,
                    "valid_time": valid_text,
                    "lead_hours": int(lead_hours),
                    "cycle_utc": init_time.strftime("%H"),
                },
                "subset": {
                    "bbox": {
                        "south": self.bbox["min_lat"],
                        "north": self.bbox["max_lat"],
                        "west": self.bbox["min_lon"],
                        "east": self.bbox["max_lon"],
                        "longitude_convention": source_meta["longitude_convention"],
                    },
                    "variables": variable_manifest,
                    "request_groups": requests_used,
                    "metadata_axes": axes,
                },
                "validation": {
                    "opened_with": "xarray + netCDF4 engine",
                    "dimensions": dict(merged.sizes),
                    "latitude_bounds": [float(np.min(lat)), float(np.max(lat))],
                    "longitude_bounds": [float(np.min(lon)), float(np.max(lon))],
                    "global_grid_rejected": True,
                    "all_registered_gfs_fields_present": True,
                    "missing_variables": [],
                    "rainfall_units": grids[precip_name]["units"],
                    "rainfall_min": float(finite_rain.min()),
                    "rainfall_max": float(finite_rain.max()),
                    "rainfall_finite_cell_count": int(finite_rain.size),
                    "rainfall_check": "finite values present; nonnegative within tolerance; <= 2000 kg m-2",
                    "observed_reftime": _utc_iso(observed_reftime),
                    "observed_time_coordinate_audit": time_coordinate_audit,
                    "precipitation_time_bounds": [_utc_iso(precip_start), _utc_iso(precip_end)],
                    "precipitation_accumulation_hours": precip_interval_hours,
                    "valid_time_equals_init_plus_lead": True,
                },
                "output": {
                    "path": output.relative_to(self.root).as_posix(),
                    "manifest_path": manifest.relative_to(self.root).as_posix(),
                    "file_size_bytes": output.stat().st_size,
                    "file_size_mib": round(output.stat().st_size / (1024**2), 4),
                    "downloaded_subset_response_bytes": transfer_bytes,
                    "sha256": _sha256(output),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                "scale_estimate": {
                    "one_case_stored_bytes": output.stat().st_size,
                    "one_case_downloaded_subset_response_bytes": transfer_bytes,
                    "cycles_2021_2025_june_september": 2440,
                    "estimated_stored_bytes_2021_2025": output.stat().st_size * 2440,
                    "estimated_stored_decimal_gb_2021_2025": round(output.stat().st_size * 2440 / 1e9, 3),
                    "estimated_transfer_bytes_2021_2025": transfer_bytes * 2440,
                    "estimated_transfer_decimal_gb_2021_2025": round(transfer_bytes * 2440 / 1e9, 3),
                    "method": "single-case stored size and summed NCSS response bytes multiplied by 122 days/year x 5 years x 4 cycles/day; approximate only",
                    "bulk_download_started": False,
                },
            }
            partial_manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
            if manifest.exists() and force:
                manifest.unlink()
            partial_manifest.replace(manifest)
        except Exception:
            partial_output.unlink(missing_ok=True)
            partial_manifest.unlink(missing_ok=True)
            raise
        finally:
            for dataset in datasets:
                dataset.close()

        self.dataset_path = output
        self.manifest_path = manifest
        return manifest_data

    def get_features(
        self,
        lat: float,
        lon: float,
        valid_time: datetime | None = None,
        lead_hours: int = 24,
    ) -> dict[str, Any]:
        """Extract a normalized point from the validated test subset only.

        This method never downloads and never falls back to synthetic values.
        """
        path = self.dataset_path
        manifest_path = self.manifest_path
        if not path.exists() or not manifest_path.exists():
            raise FileNotFoundError("Run scripts/ingest_ucar_gfs_subset.py for the explicit one-case subset first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "VALIDATED_SINGLE_CASE":
            raise UCARGFSSSubsetError("Cached subset manifest is not marked as validated")
        expected_sha256 = manifest.get("output", {}).get("sha256")
        if not expected_sha256 or _sha256(path) != expected_sha256:
            raise UCARGFSSSubsetError("Cached subset checksum does not match its manifest")
        expected_valid = datetime.fromisoformat(manifest["forecast"]["valid_time"].replace("Z", "+00:00"))
        if lead_hours != manifest["forecast"]["lead_hours"]:
            raise ValueError("Requested lead_hours does not match the cached single forecast")
        if valid_time is not None:
            requested = valid_time if valid_time.tzinfo else valid_time.replace(tzinfo=timezone.utc)
            if requested.astimezone(timezone.utc) != expected_valid:
                raise ValueError("Requested valid_time does not match the cached single forecast")
        if not (
            self.bbox["min_lat"] <= lat <= self.bbox["max_lat"]
            and self.bbox["min_lon"] <= lon <= self.bbox["max_lon"]
        ):
            raise ValueError("Point is outside the configured India subset bbox")
        with xr.open_dataset(path, engine="netcdf4") as opened:
            point = opened.sel(latitude=lat, longitude=lon, method="nearest")
            result: dict[str, Any] = {
                "init_time": manifest["forecast"]["initialization"],
                "valid_time": manifest["forecast"]["valid_time"],
                "lead_hours": lead_hours,
                "latitude": float(point.latitude.values),
                "longitude": float(point.longitude.values),
                "source": self.source_name,
            }
            for feature, spec in FEATURES.items():
                if spec.variable not in point.data_vars:
                    raise UCARGFSSSubsetError(f"Cached dataset is missing required field {spec.variable}")
                value = _scalar(point[spec.variable])
                if feature == "gfs_t2m":
                    value -= 273.15
                elif feature == "gfs_mslp":
                    value /= 100.0
                result[feature] = value
            result["observed_rain_mm"] = float("nan")
            return result
