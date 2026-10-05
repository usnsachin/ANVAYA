"""Live-only operational multi-source forecast endpoint."""
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, HTTPException, Query
from typing import Literal

from app.services.operational_service import operational_service
from app.services.district_service import district_service


logger = logging.getLogger(__name__)
router = APIRouter(tags=["operational"])


@router.get("/operational/forecast")
async def get_operational_forecast(
    latitude: float = Query(..., ge=8.0, le=37.0),
    longitude: float = Query(..., ge=68.0, le=97.0),
    run: datetime | None = Query(None, description="Optional explicit 00Z UTC GFS initialization; otherwise selected after the configured publication-latency guard."),
):
    """Return a LIVE_OPERATIONAL_MODE point forecast; never reads or writes splits."""
    if run is not None and run.tzinfo is None:
        raise HTTPException(status_code=422, detail="run must include a timezone, for example 2026-10-04T00:00:00Z")
    try:
        return await operational_service.forecast(latitude, longitude, run_time=run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Even an unexpected adapter/service failure returns a safe operational
        # envelope rather than propagating a provider error as a backend crash.
        logger.exception("Operational forecast assembly failed")
        unavailable = {
            "status": "UNAVAILABLE",
            "source": "operational forecast assembly",
            "data": None,
            "metadata": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "mode": "LIVE_OPERATIONAL_MODE",
            "location": {"latitude": latitude, "longitude": longitude},
            "raw_gfs": unavailable,
            "gefs_mean": None,
            "gefs_spread": None,
            "ecmwf_forecast": unavailable,
            "ensemble_consensus": {"status": "UNAVAILABLE", "reason": "provider assembly failure"},
            "anvaya_corrected": None,
            "global_ml_rain_mm": None,
            "anvaya_minus_raw_mm": None,
            "anvaya_minus_global_mm": None,
            "anvaya_explanation": {"label": "MODEL-BASED EXPLANATION", "status": "UNAVAILABLE", "causal_claim": False},
            "regime": None,
            "regime_probability": None,
            "heavy_probability": None,
            "very_heavy_probability": None,
            "extreme_probability": None,
            "imerg_current": unavailable,
            "district_risk": {"status": "UNAVAILABLE", "warning_level": "UNAVAILABLE"},
            "forecast_confidence": {"status": "UNAVAILABLE", "score": None},
            "warning_level": "UNAVAILABLE",
            "warnings": {"level": "UNAVAILABLE", "label": "ANVAYA OPERATIONAL DECISION RULE", "reasons": []},
            "source_metadata": {
                "mode": "LIVE_OPERATIONAL_MODE",
                "error": f"{type(exc).__name__}: {exc}",
                "scientific_mode": {"mode": "SCIENTIFIC_MODE", "status": "REFERENCE", "historical_benchmark_modified": False},
            },
            "timestamp": now,
        }


@router.get("/operational/districts")
async def list_operational_districts(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    search: str | None = Query(None, max_length=100),
    state: str | None = Query(None, max_length=80),
):
    """Return the sourced district catalogue; forecasts are computed on demand."""
    try:
        listing = district_service.list_districts(offset=offset, limit=limit, search=search, state=state)
        boundary_metadata = district_service.registry.metadata
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            **listing,
            "mode": "LIVE_OPERATIONAL_MODE",
            "status": "REFERENCE",
            "confidence": {"status": "UNAVAILABLE", "reason": "This catalogue response contains boundary metadata, not a forecast."},
            "warnings": [],
            "source_metadata": {
                "boundaries": {"status": "REFERENCE", "metadata": boundary_metadata},
                "forecast_values": "not requested; call /api/operational/districts/{district_id}",
                "scientific_mode": {"mode": "SCIENTIFIC_MODE", "benchmark_modified": False},
            },
            "timestamp": timestamp,
        }
    except Exception as exc:
        logger.exception("Operational district catalogue unavailable")
        return {
            "total": 0,
            "offset": offset,
            "limit": limit,
            "districts": [],
            "mode": "LIVE_OPERATIONAL_MODE",
            "status": "UNAVAILABLE",
            "confidence": {"status": "UNAVAILABLE"},
            "warnings": ["District boundary catalogue is unavailable"],
            "source_metadata": {"boundaries": {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}},
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


@router.get("/operational/districts/lookup")
async def lookup_operational_district(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
):
    """Resolve a WGS84 point against the acquired district polygon catalogue."""
    try:
        return {
            **district_service.lookup_point(latitude, longitude),
            "mode": "LIVE_OPERATIONAL_MODE",
            "confidence": {"status": "REFERENCE", "meaning": "geometric polygon membership; not forecast confidence"},
            "warnings": [],
            "source_metadata": {"boundaries": district_service.registry.metadata},
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    except Exception as exc:
        logger.exception("District point lookup failed")
        return {
            "status": "UNAVAILABLE",
            "district": None,
            "mode": "LIVE_OPERATIONAL_MODE",
            "confidence": {"status": "UNAVAILABLE"},
            "warnings": ["District point lookup unavailable"],
            "source_metadata": {"boundaries": {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}},
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


@router.get("/operational/districts/{district_id}")
async def get_operational_district_forecast(
    district_id: str,
    run: datetime | None = Query(None, description="Optional explicit 00Z UTC model initialization"),
):
    """Compute a cached, spatially area-weighted forecast for one selected district."""
    if run is not None and run.tzinfo is None:
        raise HTTPException(status_code=422, detail="run must include a timezone, for example 2026-10-04T00:00:00Z")
    try:
        return await district_service.forecast(district_id, run_time=run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown district {district_id!r}; use /api/operational/districts for valid IDs") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("District operational forecast failed")
        return {
            "mode": "LIVE_OPERATIONAL_MODE",
            "status": "UNAVAILABLE",
            "district": {"district_id": district_id},
            "warning_level": "UNAVAILABLE",
            "warnings": [{"level": "UNAVAILABLE", "reason": "District forecast service failed safely"}],
            "confidence": {"status": "UNAVAILABLE"},
            "forecast_confidence": {"status": "UNAVAILABLE", "score": None, "calibrated": False},
            "source_metadata": {"error": f"{type(exc).__name__}: {exc}", "scientific_mode": {"benchmark_modified": False}},
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


@router.get("/operational/map")
async def get_operational_map(
    district_id: str | None = Query(None, max_length=120),
    layer: Literal["raw_gfs", "global_ml", "anvaya", "probability", "district_warning"] = Query("district_warning"),
):
    """Return generalized district outlines, or one selected district's live layer."""
    try:
        boundary_metadata = district_service.registry.metadata
        if district_id is None:
            collection = district_service.registry.display_collection()
            for feature in collection["features"]:
                feature.setdefault("properties", {})["forecast_value"] = None
                feature["properties"]["forecast_status"] = "NOT_REQUESTED"
            return {
                **collection,
                "mode": "LIVE_OPERATIONAL_MODE",
                "status": "REFERENCE",
                "layer": "district_boundaries",
                "selected_forecast_layer": layer,
                "confidence": {"status": "UNAVAILABLE", "reason": "Select a district to request its forecast."},
                "warnings": [],
                "source_metadata": {"boundaries": {"status": "REFERENCE", "metadata": boundary_metadata}},
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        forecast = await district_service.forecast(district_id)
        feature = district_service.registry.display_feature(district_id)
        values = {
            "raw_gfs": forecast.get("raw_gfs_rain_mm"),
            "global_ml": forecast.get("global_ml_rain_mm"),
            "anvaya": forecast.get("anvaya_rain_mm"),
            "probability": forecast.get("heavy_probability"),
            "district_warning": forecast.get("warning_level"),
        }
        properties = dict(feature.get("properties") or {})
        properties.update({
            "forecast_value": values[layer],
            "forecast_layer": layer,
            "forecast_status": forecast.get("status", "UNAVAILABLE"),
            "warning_level": forecast.get("warning_level", "UNAVAILABLE"),
            "rainfall_unit": "mm" if layer in {"raw_gfs", "global_ml", "anvaya"} else None,
            "probability_unit": "fraction" if layer == "probability" else None,
        })
        feature = {**feature, "properties": properties}
        return {
            "type": "FeatureCollection",
            "name": "anvaya_selected_district_operational_layer",
            "features": [feature],
            "mode": "LIVE_OPERATIONAL_MODE",
            "status": forecast.get("status", "UNAVAILABLE"),
            "layer": layer,
            "summary": forecast,
            "confidence": forecast.get("forecast_confidence"),
            "warnings": forecast.get("warnings", []),
            "source_metadata": forecast.get("source_metadata", {}),
            "timestamp": forecast.get("timestamp"),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown district {district_id!r}") from exc
    except Exception as exc:
        logger.exception("Operational map unavailable")
        return {
            "type": "FeatureCollection",
            "name": "anvaya_operational_map_unavailable",
            "features": [],
            "mode": "LIVE_OPERATIONAL_MODE",
            "status": "UNAVAILABLE",
            "layer": layer,
            "confidence": {"status": "UNAVAILABLE"},
            "warnings": ["Map data unavailable"],
            "source_metadata": {"error": f"{type(exc).__name__}: {exc}"},
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
