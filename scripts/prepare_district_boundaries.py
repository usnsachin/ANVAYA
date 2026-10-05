"""Convert the NWIC projected district GeoJSON to WGS84 GeoJSON products.

The original ZIP is preserved as the provenance source.  The full boundary file
is used for lookup/aggregation; a separately generalized file is for map
rendering only.  The GeoJSON decoder streams one feature at a time so the
168 MB source does not need to be materialized as one large Python object.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from json import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import Iterator, TextIO

from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/geospatial/source/district_nwic_geojson.zip"
FULL_OUTPUT = ROOT / "data/geospatial/india_districts.geojson"
MAP_OUTPUT = ROOT / "data/geospatial/india_districts_map.geojson"
METADATA_OUTPUT = ROOT / "data/geospatial/district_boundaries_metadata.json"
SOURCE_ENTRY = "district_nwic.GeoJSON"
SOURCE_CRS = CRS.from_epsg(7755)
TARGET_CRS = CRS.from_epsg(4326)
MAP_SIMPLIFICATION_METRES = 500.0
EXPECTED_DISTRICT_COUNT = 733
CHUNK_SIZE = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_features(reader: TextIO) -> Iterator[dict]:
    decoder = JSONDecoder()
    buffer = ""
    marker = re.compile(r'"features"\s*:\s*\[')
    while True:
        match = marker.search(buffer)
        if match:
            buffer = buffer[match.end():]
            break
        chunk = reader.read(CHUNK_SIZE)
        if not chunk:
            raise ValueError("Source GeoJSON has no features array")
        buffer += chunk

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
                raise ValueError("Unexpected EOF inside GeoJSON features array")
            buffer += chunk
            continue
        try:
            feature, end = decoder.raw_decode(buffer)
        except JSONDecodeError:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                raise ValueError("Truncated feature in source GeoJSON")
            buffer += chunk
            continue
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Source features array contains a non-Feature item")
        yield feature
        buffer = buffer[end:]


def _feature_id(properties: dict) -> str:
    state_code = str(properties.get("stcode", "")).strip()
    district_code = str(properties.get("dtcode", "")).strip()
    district = re.sub(r"[^a-z0-9]+", "-", str(properties.get("district", "")).lower()).strip("-")
    if state_code and district_code:
        # NWIC uses the shared district code 999 for both Mirpur and
        # Muzaffarabad; append the name slug to keep those features distinct.
        return f"{state_code.zfill(2)}-{district_code.zfill(3)}-{district}"
    state = re.sub(r"[^a-z0-9]+", "-", str(properties.get("state_name", "")).lower()).strip("-")
    if not state or not district:
        raise ValueError("Boundary feature lacks stable district/state identifiers")
    return f"{state}-{district}"


def _repair_polygonal(geometry):
    """Repair topology without fabricating area; discard only collapsed lines/points."""
    repaired = make_valid(geometry)
    polygons = []

    def collect(item) -> None:
        if isinstance(item, Polygon):
            polygons.append(item)
        elif isinstance(item, (MultiPolygon, GeometryCollection)):
            for part in item.geoms:
                collect(part)

    collect(repaired)
    if not polygons:
        raise ValueError("Validity repair produced no polygonal area")
    result = unary_union(polygons)
    if result.geom_type not in {"Polygon", "MultiPolygon"} or not result.is_valid:
        raise ValueError("Validity repair did not produce a valid polygonal geometry")
    return result


def _write_collection_header(handle: TextIO, name: str) -> None:
    handle.write(
        '{"type":"FeatureCollection","name":'
        + json.dumps(name)
        + ',"features":[\n'
    )


def _write_feature(handle: TextIO, feature: dict, first: bool) -> None:
    if not first:
        handle.write(",\n")
    json.dump(feature, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing boundary source archive: {ARCHIVE}")
    FULL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    full_tmp = FULL_OUTPUT.with_suffix(".geojson.tmp")
    map_tmp = MAP_OUTPUT.with_suffix(".geojson.tmp")
    to_wgs84 = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    count = 0
    invalid_source = 0
    geometry_types: dict[str, int] = {}
    district_ids: set[str] = set()
    repaired_source_districts: list[str] = []
    repaired_after_transform: list[str] = []
    full_bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            if SOURCE_ENTRY not in archive.namelist():
                raise ValueError(f"Expected {SOURCE_ENTRY!r} in boundary archive")
            with archive.open(SOURCE_ENTRY) as binary_source:
                reader = io.TextIOWrapper(binary_source, encoding="utf-8")
                with full_tmp.open("w", encoding="utf-8", newline="\n") as full_handle, \
                     map_tmp.open("w", encoding="utf-8", newline="\n") as map_handle:
                    _write_collection_header(full_handle, "india_districts_wgs84")
                    _write_collection_header(map_handle, "india_districts_map_wgs84_simplified_500m")
                    first_full = first_map = True
                    for feature in _iter_features(reader):
                        properties = feature.get("properties") or {}
                        district_id = _feature_id(properties)
                        if district_id in district_ids:
                            raise ValueError(f"Duplicate district identifier in source: {district_id}")
                        district_ids.add(district_id)
                        geometry = feature.get("geometry")
                        if not isinstance(geometry, dict):
                            raise ValueError(f"Missing geometry for district {district_id}")
                        source_shape = shape(geometry)
                        if source_shape.geom_type not in {"Polygon", "MultiPolygon"}:
                            raise ValueError(f"Unexpected geometry type for {district_id}: {source_shape.geom_type}")
                        if not source_shape.is_valid:
                            invalid_source += 1
                            source_shape = _repair_polygonal(source_shape)
                            repaired_source_districts.append(district_id)
                        full_shape = transform_geometry(to_wgs84.transform, source_shape)
                        if not full_shape.is_valid:
                            full_shape = _repair_polygonal(full_shape)
                            repaired_after_transform.append(district_id)
                        geometry_types[full_shape.geom_type] = geometry_types.get(full_shape.geom_type, 0) + 1
                        properties["anvaya_district_id"] = district_id
                        full_feature = {**feature, "properties": properties, "geometry": mapping(full_shape)}
                        _write_feature(full_handle, full_feature, first_full)
                        first_full = False
                        min_x, min_y, max_x, max_y = full_shape.bounds
                        full_bounds[0] = min(full_bounds[0], min_x)
                        full_bounds[1] = min(full_bounds[1], min_y)
                        full_bounds[2] = max(full_bounds[2], max_x)
                        full_bounds[3] = max(full_bounds[3], max_y)

                        display_shape = source_shape.simplify(MAP_SIMPLIFICATION_METRES, preserve_topology=True)
                        display_shape = transform_geometry(to_wgs84.transform, display_shape)
                        if not display_shape.is_valid:
                            raise ValueError(f"Map simplification produced invalid geometry for {district_id}")
                        map_feature = {**full_feature, "geometry": mapping(display_shape)}
                        _write_feature(map_handle, map_feature, first_map)
                        first_map = False
                        count += 1
                        if count % 50 == 0:
                            print(f"Converted {count} district features", flush=True)
                    full_handle.write("\n]}\n")
                    map_handle.write("\n]}\n")
        if count != EXPECTED_DISTRICT_COUNT:
            raise ValueError(f"Expected {EXPECTED_DISTRICT_COUNT} source districts; parsed {count}")
        full_tmp.replace(FULL_OUTPUT)
        map_tmp.replace(MAP_OUTPUT)
    except Exception:
        full_tmp.unlink(missing_ok=True)
        map_tmp.unlink(missing_ok=True)
        raise

    metadata = {
        "dataset": "NWIC India District Boundary GeoJSON",
        "source_portal": "https://www.nwdp.nwic.gov.in/dataset/district-boundary",
        "source_download_url": "https://nwdp.nwic.gov.in/dataset/6c1af675-1dec-4927-882c-c1ba9d73f76b/resource/8d9aa2e9-9806-4f26-a4ac-48ba21e9b96d/download/district_nwic_geojson.zip",
        "portal_boundary_count": EXPECTED_DISTRICT_COUNT,
        "portal_last_update_utc": "2025-05-05T07:23:00Z",
        "portal_data_producer_agency": "Geological Survey of India (portal metadata); feature attribute src_agency says Survey of India (SOI)",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_archive": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "source_archive_sha256": _sha256(ARCHIVE),
        "source_internal_file": SOURCE_ENTRY,
        "source_crs": "EPSG:7755 — WGS 84 / India NSF LCC, metres",
        "output_crs": "EPSG:4326 — WGS 84 longitude/latitude (RFC 7946 coordinate order)",
        "full_output": str(FULL_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "full_output_sha256": _sha256(FULL_OUTPUT),
        "display_output": str(MAP_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "display_output_sha256": _sha256(MAP_OUTPUT),
        "display_generalization": f"Topology-preserving Shapely simplify at {MAP_SIMPLIFICATION_METRES:g} metres in source CRS; display only, never used for lookup or rainfall aggregation.",
        "feature_count": count,
        "geometry_types": geometry_types,
        "source_invalid_geometry_count": invalid_source,
        "geometry_repairs": {
            "method": "Shapely make_valid followed by polygon-only union; the original source archive remains unchanged",
            "source_invalid_features_repaired": repaired_source_districts,
            "repaired_after_transform": repaired_after_transform,
            "collapsed_nonpolygonal_artifacts_dropped": True,
        },
        "wgs84_bounds_lon_lat": full_bounds,
        "license": "No per-resource license text was present in the portal metadata at acquisition; retain attribution and confirm portal terms before redistribution.",
        "attribution": "National Water Informatics Centre (NWIC), Ministry of Jal Shakti, Government of India; district resource attributed to GSI by portal metadata and Survey of India in feature attributes.",
    }
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
