"""On-demand district forecast, equal-area aggregation, and cache service."""
from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ml.geospatial.boundaries import DistrictBoundary, DistrictBoundaryRegistry
from ml.geospatial.sampling import GridSample, area_weighted_grid_samples
from ml.operational.common import LIVE, REFERENCE, UNAVAILABLE, ProviderResult
from ml.operational.warnings import evaluate_warning

from app.services.operational_service import operational_service


logger = logging.getLogger(__name__)


class DistrictService:
    def __init__(self, registry: DistrictBoundaryRegistry | None = None) -> None:
        self.registry = registry or DistrictBoundaryRegistry()
        self.config = operational_service.config
        district_config = self.config.get("district_risk", {})
        self.grid_spacing = float(district_config.get("grid_spacing_degrees", 0.25))
        self.max_samples = max(1, int(district_config.get("max_forecast_samples_per_district", 9)))
        self.minimum_coverage = float(district_config.get("minimum_forecast_area_coverage_fraction", 0.75))
        self.timeout_seconds = max(1, min(int(district_config.get("district_request_timeout_seconds", 60)), 120))
        self.cache_ttl = max(0, int(district_config.get("district_cache_ttl_seconds", 600)))
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._cache_lock = asyncio.Lock()

    def list_districts(self, *, offset: int, limit: int, search: str | None, state: str | None) -> dict:
        return self.registry.listing(offset=offset, limit=limit, search=search, state=state)

    def lookup_point(self, latitude: float, longitude: float) -> dict[str, Any]:
        district, matches = self.registry.lookup_point(latitude, longitude)
        return {
            "status": REFERENCE if district else UNAVAILABLE,
            "location": {"latitude": latitude, "longitude": longitude},
            "district": district.summary() if district else None,
            "boundary_matches": matches,
            "boundary_ambiguous": len(matches) > 1,
            "boundary_metadata": self.registry.metadata,
        }

    async def _collect_sample(
        self, semaphore: asyncio.Semaphore, sample: GridSample, run: datetime
    ) -> dict[str, Any]:
        async with semaphore:
            try:
                payload = await asyncio.wait_for(
                    operational_service.forecast(
                        sample.latitude,
                        sample.longitude,
                        run_time=run,
                        include_imerg=False,
                    ),
                    timeout=self.timeout_seconds,
                )
                return {"sample": sample, "forecast": payload, "error": None}
            except Exception as exc:
                logger.warning(
                    "District grid sample failed at %.4f, %.4f: %s",
                    sample.latitude,
                    sample.longitude,
                    exc,
                )
                return {"sample": sample, "forecast": None, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _weighted_summary(
        rows: list[dict[str, Any]],
        total_area: float,
        getter: Callable[[dict[str, Any]], Any],
        minimum_coverage: float,
    ) -> dict[str, Any]:
        valid: list[tuple[float, float]] = []
        for row in rows:
            try:
                value = float(getter(row))
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(value):
                valid.append((value, float(row["area_weight_m2"])))
        valid_area = sum(weight for _, weight in valid)
        coverage = valid_area / total_area if total_area > 0 else 0.0
        result: dict[str, Any] = {
            "status": LIVE if valid and coverage >= minimum_coverage else UNAVAILABLE,
            "coverage_fraction": min(1.0, coverage),
            "sample_count": len(valid),
            "mean_mm": None,
            "max_mm": None,
            "p90_mm": None,
        }
        if not valid or coverage < minimum_coverage:
            return result
        result["mean_mm"] = sum(value * weight for value, weight in valid) / valid_area
        result["max_mm"] = max(value for value, _ in valid)
        ordered = sorted(valid)
        target = 0.9 * valid_area
        cumulative = 0.0
        percentile = ordered[-1][0]
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= target:
                percentile = value
                break
        result["p90_mm"] = percentile
        return result

    @staticmethod
    def _sample_row(item: dict[str, Any]) -> dict[str, Any] | None:
        forecast = item.get("forecast")
        if not isinstance(forecast, dict):
            return None
        sample: GridSample = item["sample"]
        raw = forecast.get("raw_gfs") or {}
        raw_data = raw.get("data") if raw.get("status") == LIVE else None
        source_meta = forecast.get("source_metadata") or {}
        return {
            "latitude": sample.latitude,
            "longitude": sample.longitude,
            "area_weight_m2": sample.area_weight_m2,
            "represented_cell_count": sample.represented_cell_count,
            "forecast": forecast,
            "raw_gfs_rain_mm": (raw_data or {}).get("gfs_daily_rain_mm"),
            "global_ml_rain_mm": forecast.get("global_ml_rain_mm"),
            "anvaya_rain_mm": forecast.get("anvaya_corrected"),
            "heavy_probability": forecast.get("heavy_probability"),
            "very_heavy_probability": forecast.get("very_heavy_probability"),
            "extreme_probability": forecast.get("extreme_probability"),
            "regime": forecast.get("regime"),
            "regime_probability": forecast.get("regime_probability"),
            "confidence_score": (forecast.get("forecast_confidence") or {}).get("score"),
            "consensus_mm": (forecast.get("ensemble_consensus") or {}).get("precipitation_24h_mean_mm"),
            "consensus_spread_mm": (forecast.get("ensemble_consensus") or {}).get("precipitation_24h_spread_mm"),
            "consensus_families": (forecast.get("ensemble_consensus") or {}).get("independent_model_families"),
            "source_metadata": source_meta,
            "explanation": forecast.get("anvaya_explanation") or {},
        }

    @staticmethod
    def _weighted_regime(rows: list[dict[str, Any]], total_area: float, min_coverage: float) -> dict[str, Any]:
        weights: dict[str, float] = {}
        probabilities: dict[str, float] = {}
        valid_area = 0.0
        for row in rows:
            distribution = row.get("regime_probability")
            if row.get("anvaya_rain_mm") is None or not isinstance(distribution, dict):
                continue
            area = row["area_weight_m2"]
            valid_area += area
            for name, probability in distribution.items():
                try:
                    value = float(probability)
                except (TypeError, ValueError):
                    continue
                if 0 <= value <= 1 and math.isfinite(value):
                    probabilities[name] = probabilities.get(name, 0.0) + area * value
            regime = row.get("regime")
            if regime:
                weights[regime] = weights.get(regime, 0.0) + area
        coverage = valid_area / total_area if total_area else 0.0
        if not probabilities or coverage < min_coverage:
            return {"status": UNAVAILABLE, "dominant_regime": None, "probability_vector": None, "coverage_fraction": coverage}
        normalized = {name: value / valid_area for name, value in probabilities.items()}
        return {
            "status": LIVE,
            "dominant_regime": max(weights, key=weights.get) if weights else max(normalized, key=normalized.get),
            "probability_vector": normalized,
            "dominant_regime_area_fraction": (max(weights.values()) / valid_area) if weights else None,
            "coverage_fraction": coverage,
        }

    async def forecast(self, identifier: str, *, run_time: datetime | None = None) -> dict[str, Any]:
        boundary = self.registry.resolve(identifier)
        run = operational_service._valid_run(run_time)
        cache_key = (boundary.district_id, run.isoformat())
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and now - cached[0] <= self.cache_ttl:
                result = copy.deepcopy(cached[1])
                result["source_metadata"]["cache"] = {"hit": True, "ttl_seconds": self.cache_ttl}
                return result

        domain = self.config.get("india_domain", {})
        sample_info = area_weighted_grid_samples(
            boundary.geometry,
            spacing_degrees=self.grid_spacing,
            max_samples=self.max_samples,
            domain=domain,
        )
        samples: list[GridSample] = sample_info["samples"]
        sample_errors: list[str] = []
        if samples:
            semaphore = asyncio.Semaphore(3)
            try:
                collected = await asyncio.wait_for(
                    asyncio.gather(*(self._collect_sample(semaphore, sample, run) for sample in samples)),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                collected = []
                sample_errors.append(f"District forecast exceeded {self.timeout_seconds}s total timeout")
            for item in collected:
                if item.get("error"):
                    sample_errors.append(item["error"])
        else:
            collected = []
            sample_errors.append("District has no 0.25-degree cell intersection inside the supported India forecast domain")

        rows = [row for item in collected if (row := self._sample_row(item)) is not None]
        total_area = float(sample_info.get("area_inside_domain_m2", 0.0))
        min_coverage = self.minimum_coverage
        summaries = {
            field: self._weighted_summary(rows, total_area, lambda row, name=field: row.get(name), min_coverage)
            for field in ("raw_gfs_rain_mm", "global_ml_rain_mm", "anvaya_rain_mm", "heavy_probability", "very_heavy_probability", "extreme_probability", "consensus_mm", "consensus_spread_mm", "confidence_score")
        }
        gefs_summary = self._weighted_summary(
            rows, total_area, lambda row: (row.get("forecast") or {}).get("gefs_mean"), min_coverage
        )
        gefs_spread_summary = self._weighted_summary(
            rows, total_area, lambda row: (row.get("forecast") or {}).get("gefs_spread"), min_coverage
        )
        model_coverage = summaries["anvaya_rain_mm"]["coverage_fraction"]
        regime = self._weighted_regime(rows, total_area, min_coverage)
        probability = {
            "heavy_probability": summaries["heavy_probability"]["mean_mm"],
            "very_heavy_probability": summaries["very_heavy_probability"]["mean_mm"],
            "extreme_probability": summaries["extreme_probability"]["mean_mm"],
        }
        warning = evaluate_warning(
            probability if all(value is not None for value in probability.values()) else None,
            self.config.get("warning_policy", {}),
            coverage_fraction=model_coverage,
        )

        family_counts = self._weighted_summary(
            rows, total_area, lambda row: row.get("consensus_families"), min_coverage
        )
        # Source-specific ECMWF forecasts are retained independently; IFS/AIFS
        # are not silently folded into a NOAA or independent-family count.
        ecmwf_models: dict[str, dict[str, Any]] = {}
        for label, value_key in (("IFS 0.25-degree", "ifs"), ("AIFS single 0.25-degree", "aifs")):
            ecmwf_models[label] = self._weighted_summary(
                rows,
                total_area,
                lambda row, target=label: self._ecmwf_value(row, target),
                min_coverage,
            )
            ecmwf_models[label]["forecast_reference_time"] = run.isoformat().replace("+00:00", "Z")

        # Satellite context is fetched after all NWP/model inference and only at
        # one disclosed in-district sample location; it is not treated as a
        # spatially complete district observation.
        target_start = run + timedelta(hours=3)
        if samples:
            try:
                imerg_result = await asyncio.wait_for(
                    operational_service._safe_call(
                        operational_service.imerg.fetch(
                            samples[0].latitude,
                            samples[0].longitude,
                            target_start=target_start,
                        )
                    ),
                    timeout=operational_service._imerg_timeout,
                )
            except asyncio.TimeoutError:
                imerg_result = ProviderResult(
                    operational_service.imerg.source,
                    UNAVAILABLE,
                    error="NASA IMERG exceeded its configured total request deadline",
                )
        else:
            imerg_result = ProviderResult(
                operational_service.imerg.source,
                UNAVAILABLE,
                error="No supported district sample point is available for the IMERG context request",
            )
        imerg_context = imerg_result.as_dict()
        imerg_context["metadata"]["spatial_scope"] = "one representative in-district 0.25-degree sample; not a district area mean"

        correction_raw = self._weighted_summary(
            rows, total_area,
            lambda row: (row.get("anvaya_rain_mm") - row.get("raw_gfs_rain_mm"))
            if row.get("anvaya_rain_mm") is not None and row.get("raw_gfs_rain_mm") is not None else None,
            min_coverage,
        )
        correction_global = self._weighted_summary(
            rows, total_area,
            lambda row: (row.get("anvaya_rain_mm") - row.get("global_ml_rain_mm"))
            if row.get("anvaya_rain_mm") is not None and row.get("global_ml_rain_mm") is not None else None,
            min_coverage,
        )
        explanation = self._aggregate_explanation(rows, total_area, min_coverage, correction_global["mean_mm"])

        source_summary = self._source_summary(
            summaries, gefs_summary, gefs_spread_summary, ecmwf_models, imerg_context, run
        )
        consensus = {
            "status": summaries["consensus_mm"]["status"],
            "precipitation_24h_mean_mm": summaries["consensus_mm"]["mean_mm"],
            "spatial_p90_mm": summaries["consensus_mm"]["p90_mm"],
            "independent_model_families_area_mean": family_counts["mean_mm"],
            "coverage_fraction": summaries["consensus_mm"]["coverage_fraction"],
            "method": "area-weighted mean of valid pointwise, accumulation-window-matched NOAA/ECMWF family consensus values",
        }
        forecast_spread = {
            "between_model_spread_area_mean_mm": summaries["consensus_spread_mm"]["mean_mm"],
            "consensus_spatial_p90_minus_p10_mm": self._weighted_range(rows, total_area, "consensus_mm", min_coverage),
            "gefs_member_spread_area_mean_mm": self._weighted_summary(
                rows, total_area,
                lambda row: (row.get("forecast", {}).get("gefs_spread")),
                min_coverage,
            )["mean_mm"],
            "calibrated": False,
        }
        source_agreement = {
            "status": LIVE if consensus["status"] == LIVE else UNAVAILABLE,
            "consensus_area_coverage_fraction": consensus["coverage_fraction"],
            "independent_model_families_area_mean": family_counts["mean_mm"],
            "independent_model_families_required": 2,
            "interpretation": "fraction of sampled district area with valid same-window consensus; not empirical forecast skill",
        }

        target_end = run + timedelta(hours=27)
        response = {
            "mode": "LIVE_OPERATIONAL_MODE",
            "status": LIVE if summaries["anvaya_rain_mm"]["status"] == LIVE else UNAVAILABLE,
            "district": boundary.summary(),
            "forecast_reference_time": run.isoformat().replace("+00:00", "Z"),
            "valid_time": target_end.isoformat().replace("+00:00", "Z"),
            "target_window": {
                "start": target_start.isoformat().replace("+00:00", "Z"),
                "end": target_end.isoformat().replace("+00:00", "Z"),
                "duration_hours": 24,
            },
            "raw_gfs_rain_mm": summaries["raw_gfs_rain_mm"]["mean_mm"],
            "global_ml_rain_mm": summaries["global_ml_rain_mm"]["mean_mm"],
            "anvaya_rain_mm": summaries["anvaya_rain_mm"]["mean_mm"],
            "anvaya_minus_raw_mm": correction_raw["mean_mm"],
            "anvaya_minus_global_mm": correction_global["mean_mm"],
            "heavy_probability": probability["heavy_probability"],
            "very_heavy_probability": probability["very_heavy_probability"],
            "extreme_probability": probability["extreme_probability"],
            "regime": regime["dominant_regime"],
            "regime_probability": regime["probability_vector"],
            "regime_area_fraction": regime["dominant_regime_area_fraction"],
            "district_metrics": {
                "raw_gfs": summaries["raw_gfs_rain_mm"],
                "global_ml": summaries["global_ml_rain_mm"],
                "anvaya": summaries["anvaya_rain_mm"],
                "anvaya_minus_raw": correction_raw,
                "anvaya_minus_global": correction_global,
                "heavy_probability": summaries["heavy_probability"],
                "very_heavy_probability": summaries["very_heavy_probability"],
                "extreme_probability": summaries["extreme_probability"],
            },
            "gefs_mean": gefs_summary["mean_mm"],
            "gefs_spread": gefs_spread_summary["mean_mm"],
            "ecmwf_forecast": {"status": self._combined_status(ecmwf_models), "models": ecmwf_models},
            "forecast_consensus": consensus,
            "forecast_spread": forecast_spread,
            "source_agreement": source_agreement,
            "anvaya_explanation": explanation,
            "imerg_current": imerg_context,
            "forecast_confidence": {
                "status": summaries["confidence_score"]["status"],
                "score": summaries["confidence_score"]["mean_mm"],
                "coverage_fraction": summaries["confidence_score"]["coverage_fraction"],
                "calibrated": False,
                "method": "area-weighted average of the point forecast source-stability diagnostic; not calibrated skill",
            },
            "warning_level": warning["level"],
            "warnings": warning,
            "sampling": {
                "method": sample_info.get("aggregation"),
                "grid_spacing_degrees": self.grid_spacing,
                "intersecting_grid_cell_count": sample_info.get("intersecting_grid_cell_count", 0),
                "selected_sample_count": len(samples),
                "maximum_samples_per_district": self.max_samples,
                "forecast_area_coverage_fraction": model_coverage,
                "minimum_forecast_area_coverage_fraction": min_coverage,
                "samples": [
                    {
                        "latitude": round(sample.latitude, 6),
                        "longitude": round(sample.longitude, 6),
                        "represented_grid_cells": sample.represented_cell_count,
                        "area_weight_m2": sample.area_weight_m2,
                        "status": LIVE if any(row["latitude"] == sample.latitude and row["anvaya_rain_mm"] is not None for row in rows) else UNAVAILABLE,
                    }
                    for sample in samples
                ],
            },
            "source_metadata": {
                "mode": "LIVE_OPERATIONAL_MODE",
                "boundaries": {
                    "status": REFERENCE,
                    "source": self.registry.metadata,
                },
                "gfs": source_summary["gfs"],
                "gefs": source_summary["gefs"],
                "ecmwf": source_summary["ecmwf"],
                "anvaya": source_summary["anvaya"],
                "imerg": source_summary["imerg"],
                "imd_reference": operational_service._reference_metadata(),
                "scientific_mode": {
                    "mode": "SCIENTIFIC_MODE",
                    "status": REFERENCE,
                    "benchmark_modified": False,
                    "historical_rmse_claim": False,
                },
                "warning_policy": self.config.get("warning_policy", {}),
                "sample_errors": sample_errors[:20],
                "cache": {"hit": False, "ttl_seconds": self.cache_ttl},
            },
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        async with self._cache_lock:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[cache_key] = (time.monotonic(), copy.deepcopy(response))
        return response

    @staticmethod
    def _ecmwf_value(row: dict[str, Any], model_name: str) -> float | None:
        payload = (row.get("forecast") or {}).get("ecmwf_forecast") or {}
        model = ((payload.get("data") or {}).get("models") or {}).get(model_name) or {}
        if model.get("status") != LIVE:
            return None
        value = (model.get("data") or {}).get("precipitation_24h_mm")
        return float(value) if value is not None else None

    @staticmethod
    def _combined_status(items: dict[str, dict[str, Any]]) -> str:
        return LIVE if any(value.get("status") == LIVE for value in items.values()) else UNAVAILABLE

    @staticmethod
    def _weighted_range(rows: list[dict[str, Any]], total_area: float, field: str, minimum_coverage: float) -> float | None:
        summary = DistrictService._weighted_summary(rows, total_area, lambda row: row.get(field), minimum_coverage)
        if summary["status"] != LIVE:
            return None
        valid = []
        for row in rows:
            value = row.get(field)
            if value is not None:
                valid.append((float(value), row["area_weight_m2"]))
        valid.sort()
        area = sum(weight for _, weight in valid)
        if not area:
            return None
        def percentile(q: float) -> float:
            target = q * area
            seen = 0.0
            for value, weight in valid:
                seen += weight
                if seen >= target:
                    return value
            return valid[-1][0]
        return percentile(0.9) - percentile(0.1)

    @staticmethod
    def _aggregate_explanation(
        rows: list[dict[str, Any]], total_area: float, minimum_coverage: float, correction_mm: float | None
    ) -> dict[str, Any]:
        weights: dict[str, float] = {}
        feature_values: dict[str, float] = {}
        available_area = 0.0
        for row in rows:
            explanation = row.get("explanation") or {}
            if explanation.get("status") == UNAVAILABLE:
                continue
            area = row["area_weight_m2"]
            available_area += area
            for item in explanation.get("all_feature_contributions", explanation.get("top_feature_contributions", [])):
                feature = str(item.get("feature"))
                try:
                    contribution = float(item["correction_contribution_mm"])
                    value = float(item["live_feature_value"])
                except (KeyError, TypeError, ValueError):
                    continue
                weights[feature] = weights.get(feature, 0.0) + area * contribution
                feature_values[feature] = feature_values.get(feature, 0.0) + area * value
        coverage = available_area / total_area if total_area else 0.0
        if not weights or coverage < minimum_coverage:
            return {
                "label": "MODEL-BASED EXPLANATION",
                "status": UNAVAILABLE,
                "error": "Insufficient forecast-area coverage for spatially aggregated local model attributions",
                "coverage_fraction": coverage,
                "causal_claim": False,
            }
        physical_names = {
            "gfs_daily_rain_mm": "GFS daily rainfall",
            "rain_last_24h": "antecedent 24-hour rainfall proxy",
            "moisture_transport_850": "850-hPa moisture-transport proxy",
            "gfs_u850": "850-hPa zonal wind",
            "gfs_v850": "850-hPa meridional wind",
            "gfs_cape": "CAPE instability",
            "gfs_precipitable_water": "precipitable water",
            "gfs_rh2m": "2-m relative humidity",
            "terrain_elevation_proxy": "terrain-elevation proxy",
            "orographic_flow_proxy": "orographic-flow proxy",
        }
        features = sorted(weights, key=lambda name: abs(weights[name] / available_area), reverse=True)[:5]
        top = [
            {
                "feature": name,
                "physical_driver": physical_names.get(name, name.replace("_", " ")),
                "area_mean_feature_value": feature_values[name] / available_area,
                "correction_contribution_mm": weights[name] / available_area,
            }
            for name in features
        ]
        return {
            "label": "MODEL-BASED EXPLANATION",
            "status": LIVE,
            "method": "Area-weighted aggregation of frozen Phase 3F expert TreeSHAP-style local contributions to ANVAYA minus Global ML.",
            "comparison": "ANVAYA minus Global ML rainfall forecast",
            "correction_amount_mm": correction_mm,
            "dominant_physical_drivers": [item["physical_driver"] for item in top[:3]],
            "top_feature_contributions": top,
            "summary": "The frozen model attributed the area-mean ANVAYA−Global ML adjustment to the listed predictors; these predictive associations are not causal explanations.",
            "coverage_fraction": coverage,
            "causal_claim": False,
            "caveat": "Live operational input substitution is not scientifically validated against the frozen historical benchmark.",
        }

    def _source_summary(
        self,
        summaries: dict[str, dict[str, Any]],
        gefs_summary: dict[str, Any],
        gefs_spread_summary: dict[str, Any],
        ecmwf_models: dict[str, dict[str, Any]],
        imerg: dict[str, Any],
        run: datetime,
    ) -> dict[str, dict[str, Any]]:
        common_valid = (run + timedelta(hours=27)).isoformat().replace("+00:00", "Z")
        common_start = (run + timedelta(hours=3)).isoformat().replace("+00:00", "Z")
        def item(status: str, source: str, summary: dict[str, Any] | None, uncertainty: Any = None) -> dict[str, Any]:
            return {
                "status": status,
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "forecast_reference_time": run.isoformat().replace("+00:00", "Z"),
                "valid_time": common_valid,
                "accumulation_start": common_start,
                "accumulation_end": common_valid,
                "forecast_value_mm": (summary or {}).get("mean_mm"),
                "uncertainty": uncertainty,
                "area_coverage_fraction": (summary or {}).get("coverage_fraction", 0.0),
                "spatial_statistics": summary,
            }
        return {
            "gfs": item(summaries["raw_gfs_rain_mm"]["status"], "NOAA/NCEP GFS via configured operational delivery API", summaries["raw_gfs_rain_mm"], summaries["raw_gfs_rain_mm"].get("p90_mm")),
            "gefs": item(gefs_summary["status"], "NOAA/NCEP GEFS ensemble members", gefs_summary, gefs_spread_summary.get("mean_mm")),
            "ecmwf": {
                "status": self._combined_status(ecmwf_models),
                "source": "ECMWF IFS/AIFS operational products",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "forecast_reference_time": run.isoformat().replace("+00:00", "Z"),
                "valid_time": common_valid,
                "models": ecmwf_models,
            },
            "anvaya": item(summaries["anvaya_rain_mm"]["status"], "Frozen Phase 3F ANVAYA MoE inference", summaries["anvaya_rain_mm"], {
                "status": "UNAVAILABLE" if summaries["anvaya_rain_mm"]["status"] != LIVE else "LIVE",
                "calibrated_skill_claim": False,
            }),
            "imerg": {
                "status": imerg.get("status", UNAVAILABLE),
                "source": imerg.get("source"),
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "forecast_reference_time": None,
                "observation_window": (imerg.get("data") or {}).get("window_start"),
                "observed_context_value_mm": (imerg.get("data") or {}).get("precipitation_24h_mm"),
                "unit": "mm" if (imerg.get("data") or {}).get("precipitation_24h_mm") is not None else None,
                "spatial_scope": (imerg.get("metadata") or {}).get("spatial_scope"),
                "uncertainty": None,
            },
        }


district_service = DistrictService()
