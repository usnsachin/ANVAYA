"""Streaming loader and spatial index for attributed India district polygons."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from json import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import Iterator, TextIO

from shapely.geometry import Point, shape
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
FULL_BOUNDARY_PATH = ROOT / "data/geospatial/india_districts.geojson"
DISPLAY_BOUNDARY_PATH = ROOT / "data/geospatial/india_districts_map.geojson"
BOUNDARY_METADATA_PATH = ROOT / "data/geospatial/district_boundaries_metadata.json"
CHUNK_SIZE = 1024 * 1024


def _runtime_boundary_path() -> Path:
    """Resolve an opt-in deployment geometry without changing the local scientific default."""
    configured = os.environ.get("ANVAYA_DISTRICT_BOUNDARY_PATH", "").strip()
    if not configured:
        return FULL_BOUNDARY_PATH
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def iter_feature_collection(reader: TextIO) -> Iterator[dict]:
    """Decode a FeatureCollection one feature at a time from a text stream."""
    decoder = JSONDecoder()
    buffer = ""
    marker = '"features"'
    while marker not in buffer:
        chunk = reader.read(CHUNK_SIZE)
        if not chunk:
            raise ValueError("GeoJSON has no features array")
        buffer += chunk
    marker_at = buffer.index(marker) + len(marker)
    bracket = buffer.find("[", marker_at)
    while bracket < 0:
        chunk = reader.read(CHUNK_SIZE)
        if not chunk:
            raise ValueError("GeoJSON features array is truncated")
        buffer += chunk
        bracket = buffer.find("[", marker_at)
    buffer = buffer[bracket + 1:]

    while True:
        buffer = buffer.lstrip()
        if buffer.startswith(","):
            buffer = buffer[1:]
            continue
        if buffer.startswith("]"):
            return
        if not buffer:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                raise ValueError("Unexpected EOF in GeoJSON features array")
            buffer += chunk
            continue
        try:
            feature, end = decoder.raw_decode(buffer)
        except JSONDecodeError:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                raise ValueError("Truncated GeoJSON feature")
            buffer += chunk
            continue
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("FeatureCollection contains a non-Feature value")
        yield feature
        buffer = buffer[end:]


@dataclass(frozen=True)
class DistrictBoundary:
    district_id: str
    district: str
    state: str
    state_code: str
    district_code: str
    geometry: object
    properties: dict
    bounds: tuple[float, float, float, float]

    def summary(self) -> dict:
        point = self.geometry.representative_point()
        return {
            "district_id": self.district_id,
            "district": self.district,
            "state": self.state,
            "state_code": self.state_code,
            "district_code": self.district_code,
            "representative_point": {"latitude": point.y, "longitude": point.x},
            "bounds_lon_lat": list(self.bounds),
            "boundary_status": "REFERENCE",
        }


class DistrictBoundaryRegistry:
    """Lazy in-memory STRtree; full-resolution polygons are used for analysis."""

    def __init__(self, path: Path | None = None, expected_count: int = 733):
        self.path = Path(path) if path is not None else _runtime_boundary_path()
        self.expected_count = expected_count
        self._records: list[DistrictBoundary] | None = None
        self._by_id: dict[str, DistrictBoundary] = {}
        self._by_name: dict[str, list[DistrictBoundary]] = {}
        self._tree: STRtree | None = None
        self._display_by_id: dict[str, dict] | None = None
        self._metadata: dict | None = None

    def _load(self) -> None:
        if self._records is not None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"District boundary GeoJSON not found: {self.path}")
        records: list[DistrictBoundary] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for feature in iter_feature_collection(handle):
                props = feature.get("properties") or {}
                identifier = str(props.get("anvaya_district_id") or "").strip()
                name = str(props.get("district") or "").strip()
                state = str(props.get("state_name") or "").strip()
                if not identifier or not name or not state:
                    raise ValueError("District boundary has missing ANVAYA ID/name/state attributes")
                geometry_data = feature.get("geometry")
                if not isinstance(geometry_data, dict):
                    raise ValueError(f"District {identifier} has no polygon geometry")
                geometry = shape(geometry_data)
                if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
                    raise ValueError(f"District {identifier} has unsupported or invalid geometry")
                bounds = tuple(float(value) for value in geometry.bounds)
                if not all(-180 <= value <= 180 for value in (bounds[0], bounds[2])):
                    raise ValueError(f"District {identifier} is not in WGS84 longitude coordinates")
                if not all(-90 <= value <= 90 for value in (bounds[1], bounds[3])):
                    raise ValueError(f"District {identifier} is not in WGS84 latitude coordinates")
                record = DistrictBoundary(
                    district_id=identifier,
                    district=name,
                    state=state,
                    state_code=str(props.get("stcode") or ""),
                    district_code=str(props.get("dtcode") or ""),
                    geometry=geometry,
                    properties=props,
                    bounds=bounds,
                )
                if identifier in self._by_id:
                    raise ValueError(f"Duplicate ANVAYA district ID: {identifier}")
                self._by_id[identifier] = record
                self._by_name.setdefault(name.casefold(), []).append(record)
                records.append(record)
        if self.expected_count and len(records) != self.expected_count:
            raise ValueError(f"Expected {self.expected_count} district polygons; loaded {len(records)}")
        self._records = sorted(records, key=lambda item: (item.state.casefold(), item.district.casefold()))
        self._tree = STRtree([item.geometry for item in self._records])

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            if not BOUNDARY_METADATA_PATH.is_file():
                return {"status": "UNAVAILABLE", "reason": "Boundary provenance manifest is missing"}
            self._metadata = json.loads(BOUNDARY_METADATA_PATH.read_text(encoding="utf-8"))
        metadata = dict(self._metadata)
        try:
            runtime_path = str(self.path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        except ValueError:
            runtime_path = str(self.path)
        metadata["runtime_boundary_path"] = runtime_path
        if self.path.resolve() == DISPLAY_BOUNDARY_PATH.resolve():
            metadata["runtime_geometry_role"] = (
                "deployment compact geometry for operational district lookup and aggregation; "
                "the full-resolution local source remains the Phase 4 reference"
            )
        else:
            metadata["runtime_geometry_role"] = "full-resolution Phase 4 reference geometry"
        return metadata

    def all(self) -> list[DistrictBoundary]:
        self._load()
        return list(self._records or [])

    def listing(
        self, *, offset: int = 0, limit: int = 100, search: str | None = None, state: str | None = None
    ) -> dict:
        records = self.all()
        if search:
            term = search.casefold().strip()
            records = [r for r in records if term in r.district.casefold() or term in r.state.casefold()]
        if state:
            records = [r for r in records if r.state.casefold() == state.casefold() or r.state_code.casefold() == state.casefold()]
        return {
            "total": len(records),
            "offset": offset,
            "limit": limit,
            "districts": [item.summary() for item in records[offset: offset + limit]],
        }

    def resolve(self, identifier: str) -> DistrictBoundary:
        self._load()
        key = identifier.strip().casefold()
        exact = self._by_id.get(key)
        if exact is not None:
            return exact
        matches = self._by_name.get(key, [])
        if len(matches) == 1:
            return matches[0]
        normalized = key.replace("_", "-").replace(" ", "-")
        matches = [
            item for item in (self._records or [])
            if f"{item.state}-{item.district}".casefold().replace(" ", "-").replace("&", "and") == normalized.replace("&", "and")
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"District name {identifier!r} is ambiguous; use its district_id")
        raise KeyError(identifier)

    def lookup_point(self, latitude: float, longitude: float) -> tuple[DistrictBoundary | None, list[str]]:
        self._load()
        point = Point(longitude, latitude)
        if self._tree is None:
            return None, []
        indexes = self._tree.query(point, predicate="intersects")
        candidates = [self._records[int(index)] for index in indexes]
        interior = [record for record in candidates if record.geometry.contains(point)]
        matches = interior or [record for record in candidates if record.geometry.covers(point)]
        if not matches:
            return None, []
        matches.sort(key=lambda item: (item.state_code, item.district_code, item.district_id))
        return matches[0], [item.district_id for item in matches]

    def display_feature(self, identifier: str) -> dict:
        record = self.resolve(identifier)
        if self._display_by_id is None:
            if not DISPLAY_BOUNDARY_PATH.is_file():
                raise FileNotFoundError(f"Simplified district display GeoJSON not found: {DISPLAY_BOUNDARY_PATH}")
            with DISPLAY_BOUNDARY_PATH.open("r", encoding="utf-8") as handle:
                collection = json.load(handle)
            self._display_by_id = {
                str(feature.get("properties", {}).get("anvaya_district_id")): feature
                for feature in collection.get("features", [])
            }
        feature = self._display_by_id.get(record.district_id)
        if feature is None:
            raise KeyError(f"No display geometry for district {record.district_id}")
        return feature

    def display_collection(self) -> dict:
        if self._display_by_id is None:
            if not DISPLAY_BOUNDARY_PATH.is_file():
                raise FileNotFoundError(f"Simplified district display GeoJSON not found: {DISPLAY_BOUNDARY_PATH}")
            with DISPLAY_BOUNDARY_PATH.open("r", encoding="utf-8") as handle:
                collection = json.load(handle)
            self._display_by_id = {
                str(feature.get("properties", {}).get("anvaya_district_id")): feature
                for feature in collection.get("features", [])
            }
        return {
            "type": "FeatureCollection",
            "name": "india_districts_map_wgs84_simplified_500m",
            "features": list(self._display_by_id.values()),
        }
