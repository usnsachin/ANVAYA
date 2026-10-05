"""Point alerts and optional GeoJSON-backed district risk aggregation."""
from __future__ import annotations

from typing import Any

from ml.operational.common import LIVE, UNAVAILABLE


def warning_level(
    probabilities: dict[str, float] | None,
    *,
    threshold: float,
    levels: list[dict[str, str]],
    below_threshold: str = "MONITOR",
) -> str:
    if not probabilities:
        return UNAVAILABLE
    for entry in levels:
        field = entry.get("probability")
        name = entry.get("name")
        if field and name and float(probabilities.get(field, 0.0)) >= threshold:
            return name
    return below_threshold


def _inside_ring(longitude: float, latitude: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 4:
        return False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crosses = ((y1 > latitude) != (y2 > latitude)) and (
            longitude < (x2 - x1) * (latitude - y1) / ((y2 - y1) or 1e-15) + x1
        )
        if crosses:
            inside = not inside
        previous = current
    return inside


def _geometry_contains(geometry: dict[str, Any], longitude: float, latitude: float) -> bool:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon" and isinstance(coordinates, list) and coordinates:
        if not _inside_ring(longitude, latitude, coordinates[0]):
            return False
        return not any(_inside_ring(longitude, latitude, ring) for ring in coordinates[1:])
    if kind == "MultiPolygon" and isinstance(coordinates, list):
        return any(_geometry_contains({"type": "Polygon", "coordinates": polygon}, longitude, latitude)
                   for polygon in coordinates)
    raise ValueError(f"Unsupported or malformed district GeoJSON geometry: {kind}")


def aggregate_district_risk(
    district_id: str | None,
    boundary_geometry: dict[str, Any] | None,
    point_forecasts: list[dict[str, Any]],
    *,
    threshold: float,
    levels: list[dict[str, str]],
    below_threshold: str = "MONITOR",
) -> dict[str, Any]:
    """Aggregate point forecasts with a district polygon when one is verified.

    The result is the maximum pointwise event probability, not a calibrated
    probability that any part of the district is affected.
    """
    if not boundary_geometry:
        return {
            "status": UNAVAILABLE,
            "district_id": district_id,
            "warning_level": UNAVAILABLE,
            "reason": "No verified district boundary geometry is configured; only point-level warning is available.",
        }
    if not point_forecasts:
        return {
            "status": UNAVAILABLE,
            "district_id": district_id,
            "warning_level": UNAVAILABLE,
            "reason": "No forecast grid points were supplied for district aggregation.",
        }
    try:
        selected = [
            item for item in point_forecasts
            if _geometry_contains(boundary_geometry, float(item["longitude"]), float(item["latitude"]))
            and isinstance(item.get("probabilities"), dict)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": UNAVAILABLE,
            "district_id": district_id,
            "warning_level": UNAVAILABLE,
            "reason": f"District geometry/point validation failed: {exc}",
        }
    if not selected:
        return {
            "status": UNAVAILABLE,
            "district_id": district_id,
            "warning_level": UNAVAILABLE,
            "reason": "No forecast points fall inside the supplied district polygon.",
        }
    fields = sorted({
        entry["probability"] for entry in levels if entry.get("probability")
    })
    maxima = {
        field: max(float(item["probabilities"].get(field, 0.0)) for item in selected)
        for field in fields
    }
    level = warning_level(maxima, threshold=threshold, levels=levels, below_threshold=below_threshold)
    return {
        "status": LIVE,
        "district_id": district_id,
        "warning_level": level,
        "max_point_probabilities": maxima,
        "grid_point_count": len(selected),
        "aggregation": "maximum pointwise probability within supplied polygon",
        "probability_interpretation": "not a calibrated district-area or any-cell probability",
    }
