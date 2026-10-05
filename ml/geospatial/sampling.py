"""Deterministic equal-area, spatially stratified samples of 0.25-degree cells."""
from __future__ import annotations

import math
from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as transform_geometry


_TO_EQUAL_AREA = Transformer.from_crs(4326, 6933, always_xy=True)
_TO_WGS84 = Transformer.from_crs(6933, 4326, always_xy=True)


@dataclass(frozen=True)
class GridSample:
    latitude: float
    longitude: float
    area_weight_m2: float
    represented_cell_count: int
    sample_row: int
    sample_column: int


def area_weighted_grid_samples(
    district_geometry,
    *,
    spacing_degrees: float = 0.25,
    max_samples: int = 9,
    domain: dict | None = None,
) -> dict:
    """Build area weights from every intersecting cell, then choose spatial strata.

    For large districts, one observed forecast point represents the total
    equal-area coverage of its spatial stratum. Returned coverage is separately
    tracked, so a sparse sample is never presented as exhaustive gridded data.
    """
    domain = domain or {"west": 68.0, "east": 97.0, "south": 8.0, "north": 37.0}
    if not math.isfinite(spacing_degrees) or spacing_degrees <= 0:
        raise ValueError("Grid spacing must be a positive finite number")
    max_samples = max(1, int(max_samples))
    study_area = district_geometry.intersection(
        box(domain["west"], domain["south"], domain["east"], domain["north"])
    )
    if study_area.is_empty or study_area.area == 0:
        return {"samples": [], "intersecting_grid_cell_count": 0, "area_inside_domain_m2": 0.0}
    projected_district = transform_geometry(_TO_EQUAL_AREA.transform, study_area)
    min_lon, min_lat, max_lon, max_lat = study_area.bounds
    step = spacing_degrees
    col_start = max(0, math.floor((min_lon - domain["west"]) / step))
    col_stop = min(math.ceil((domain["east"] - domain["west"]) / step), math.ceil((max_lon - domain["west"]) / step))
    row_start = max(0, math.floor((min_lat - domain["south"]) / step))
    row_stop = min(math.ceil((domain["north"] - domain["south"]) / step), math.ceil((max_lat - domain["south"]) / step))
    candidates: list[dict] = []
    for row in range(row_start, row_stop):
        south = domain["south"] + row * step
        north = min(domain["north"], south + step)
        for column in range(col_start, col_stop):
            west = domain["west"] + column * step
            east = min(domain["east"], west + step)
            cell = transform_geometry(_TO_EQUAL_AREA.transform, box(west, south, east, north))
            intersection = projected_district.intersection(cell)
            area = float(intersection.area)
            if area <= 0 or intersection.is_empty:
                continue
            point = intersection.representative_point()
            longitude, latitude = _TO_WGS84.transform(point.x, point.y)
            candidates.append({
                "latitude": float(latitude),
                "longitude": float(longitude),
                "area": area,
                "row": row,
                "column": column,
                "projected_x": point.x,
                "projected_y": point.y,
            })
    if not candidates:
        return {"samples": [], "intersecting_grid_cell_count": 0, "area_inside_domain_m2": 0.0}

    total_area = sum(item["area"] for item in candidates)
    width = max(max_lon - min_lon, step)
    height = max(max_lat - min_lat, step)
    bins_x = max(1, min(max_samples, int(math.sqrt(max_samples * width / height))))
    bins_y = max(1, max_samples // bins_x)
    strata: dict[tuple[int, int], list[dict]] = {}
    for candidate in candidates:
        bx = min(bins_x - 1, int((candidate["longitude"] - min_lon) / width * bins_x))
        by = min(bins_y - 1, int((candidate["latitude"] - min_lat) / height * bins_y))
        strata.setdefault((by, bx), []).append(candidate)

    samples: list[GridSample] = []
    for (by, bx), group in sorted(strata.items()):
        weight = sum(item["area"] for item in group)
        target_x = sum(item["projected_x"] * item["area"] for item in group) / weight
        target_y = sum(item["projected_y"] * item["area"] for item in group) / weight
        selected = min(
            group,
            key=lambda item: (item["projected_x"] - target_x) ** 2 + (item["projected_y"] - target_y) ** 2,
        )
        samples.append(GridSample(
            latitude=selected["latitude"],
            longitude=selected["longitude"],
            area_weight_m2=weight,
            represented_cell_count=len(group),
            sample_row=by,
            sample_column=bx,
        ))
    return {
        "samples": samples,
        "intersecting_grid_cell_count": len(candidates),
        "area_inside_domain_m2": total_area,
        "selected_sample_count": len(samples),
        "grid_spacing_degrees": step,
        "aggregation": "all intersecting 0.25-degree cell polygons weighted by EPSG:6933 intersection area; one representative in-district grid-cell point selected per spatial stratum; stratum area assigned to its sample",
    }
