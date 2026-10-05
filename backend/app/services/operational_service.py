"""Fault-tolerant LIVE_OPERATIONAL_MODE service, isolated from science mode."""
from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features.engineer import (  # noqa: E402
    FEATURE_SCHEMA,
    calculate_coast_distance_approx,
    calculate_orographic_flow_proxy,
    calculate_terrain_elevation_proxy,
    calculate_wind_direction,
    calculate_wind_speed,
    calculate_moisture_transport_proxy,
    calculate_mslp_anomaly,
)
from ml.operational.common import REFERENCE, UNAVAILABLE, ProviderResult  # noqa: E402
from ml.operational.consensus import build_consensus  # noqa: E402
from ml.operational.ecmwf import ECMWFProvider  # noqa: E402
from ml.operational.ge_f_s import GEFSProvider  # noqa: E402
from ml.operational.gfs_live import GFSLiveProvider, select_gfs_00z_run  # noqa: E402
from ml.operational.imerg import IMERGProvider  # noqa: E402
from ml.operational.risk import aggregate_district_risk  # noqa: E402
from ml.operational.warnings import evaluate_warning  # noqa: E402
from ml.scientific.phase3f import predict_all_probabilities, predict_moe  # noqa: E402


logger = logging.getLogger(__name__)
MODEL_DIR = ROOT / "models/scientific/phase3f"
CONFIG_PATH = ROOT / "config/operational.yaml"
FEATURE_VERSION = "feature_registry_v1_phase3e_24_features"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


class OperationalService:
    def __init__(self) -> None:
        self.gfs = GFSLiveProvider()
        self.gefs = GEFSProvider()
        self.ecmwf = ECMWFProvider()
        self.imerg = IMERGProvider()
        self.config = _load_config()
        self._models: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._model_error: str | None = None
        self._cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
        self._cache_lock = asyncio.Lock()
        self._ttl = max(0, int(self.config.get("cache_ttl_seconds", 600)))
        self._imerg_timeout = max(1, min(int(self.config.get("imerg_total_timeout_seconds", 20)), 60))

    def _frozen_models(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._models is not None:
            return self._models
        if self._model_error:
            raise RuntimeError(self._model_error)
        try:
            moe_path = MODEL_DIR / "anvaya_moe.joblib"
            probability_path = MODEL_DIR / "multi_threshold_probabilities.joblib"
            if not moe_path.is_file() or not probability_path.is_file():
                raise FileNotFoundError("Frozen Phase 3F model bundle is incomplete")
            moe = joblib.load(moe_path)
            probabilities = joblib.load(probability_path)
            expected = list(FEATURE_SCHEMA)
            if moe.get("feature_names") != expected or probabilities.get("feature_names") != expected:
                raise ValueError("Frozen Phase 3F feature names do not match the runtime feature schema")
            if moe.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Frozen Phase 3F feature version is not the expected version")
            self._models = (moe, probabilities)
            return self._models
        except Exception as exc:
            self._model_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(self._model_error) from exc

    @staticmethod
    def _valid_run(run_time: datetime | None) -> datetime:
        if run_time is None:
            return select_gfs_00z_run()
        if run_time.tzinfo is None:
            raise ValueError("run must include a timezone (UTC Z recommended)")
        run = run_time.astimezone(timezone.utc)
        if run.hour != 0 or run.minute or run.second or run.microsecond:
            raise ValueError("Only a 00Z run is supported by this frozen Phase 3F model interface")
        return run

    def _build_model_features(
        self,
        gfs_data: dict[str, Any],
        antecedent_gfs_data: dict[str, Any],
        latitude: float,
        longitude: float,
        target_time: datetime,
    ) -> np.ndarray:
        required_raw = (
            "gfs_daily_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
            "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
        )
        missing = [key for key in required_raw if key not in gfs_data or gfs_data[key] is None]
        if missing:
            raise ValueError(f"GFS lacks required frozen-model inputs: {missing}")
        rain_last_24h = antecedent_gfs_data.get("gfs_daily_rain_mm")
        if rain_last_24h is None:
            raise ValueError("Prior-run GFS antecedent rainfall is unavailable")
        target_start = datetime.fromisoformat(gfs_data["accumulation_start"].replace("Z", "+00:00"))
        gfs_start = datetime.fromisoformat(gfs_data["init_time"].replace("Z", "+00:00"))
        gfs_end = datetime.fromisoformat(gfs_data["valid_time"].replace("Z", "+00:00"))
        if gfs_end - gfs_start != timedelta(hours=27):
            raise ValueError("GFS lead time is not the Phase 3F 27-hour target lead")
        antecedent_start = datetime.fromisoformat(antecedent_gfs_data["accumulation_start"].replace("Z", "+00:00"))
        antecedent_end = datetime.fromisoformat(antecedent_gfs_data["accumulation_end"].replace("Z", "+00:00"))
        if antecedent_end - antecedent_start != timedelta(hours=24):
            raise ValueError("Prior-run GFS antecedent accumulation is not exactly 24 hours")
        if antecedent_end >= gfs_start:
            raise ValueError("Prior-run GFS antecedent window does not end before current initialization")
        if target_start - antecedent_end != timedelta(hours=24):
            raise ValueError("Prior-run GFS antecedent window does not match the frozen feature timestamp lag")

        u10, v10 = float(gfs_data["gfs_u10"]), float(gfs_data["gfs_v10"])
        u850, v850 = float(gfs_data["gfs_u850"]), float(gfs_data["gfs_v850"])
        feature_map = {
            **{key: float(gfs_data[key]) for key in required_raw},
            "rain_last_24h": float(rain_last_24h),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "wind_speed_10m": calculate_wind_speed(u10, v10),
            "wind_dir_10m": calculate_wind_direction(u10, v10),
            "wind_speed_850": calculate_wind_speed(u850, v850),
            "moisture_transport_850": calculate_moisture_transport_proxy(
                calculate_wind_speed(u850, v850), float(gfs_data["gfs_precipitable_water"])
            ),
            "mslp_anomaly": calculate_mslp_anomaly(float(gfs_data["gfs_mslp"])),
            "lead_time_h": 27.0,
            "coast_distance_approx": calculate_coast_distance_approx(latitude, longitude),
            "terrain_elevation_proxy": calculate_terrain_elevation_proxy(latitude, longitude),
            "orographic_flow_proxy": calculate_orographic_flow_proxy(u850, v850, latitude, longitude),
            "month": float(target_time.month),
            "day_of_year": float(target_time.timetuple().tm_yday),
        }
        if set(feature_map) != set(FEATURE_SCHEMA):
            absent = sorted(set(FEATURE_SCHEMA) - set(feature_map))
            unexpected = sorted(set(feature_map) - set(FEATURE_SCHEMA))
            raise ValueError(f"Live feature map/schema mismatch; missing={absent}, unexpected={unexpected}")
        row = np.asarray([[feature_map[name] for name in FEATURE_SCHEMA]], dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError("Live feature vector contains missing or non-finite values")
        return row

    def _predict(self, row: np.ndarray) -> dict[str, Any]:
        moe, probability_stack = self._frozen_models()
        prediction, posterior = predict_moe(moe, row)
        global_model = moe["global_fallback"]
        global_prediction = np.asarray(global_model.predict(row), dtype=np.float64)
        probabilities = predict_all_probabilities(probability_stack, row, posterior)
        regime_distribution = {
            str(label): float(posterior[0, index])
            for index, label in enumerate(moe["regime_labels"])
        }
        top_regime = max(regime_distribution, key=regime_distribution.get)
        result = {
            "anvaya_corrected": float(prediction[0]),
            "global_ml_rain_mm": float(global_prediction[0]),
            "regime": top_regime,
            "regime_probability": regime_distribution,
            "heavy_probability": float(probabilities["heavy"][0]),
            "very_heavy_probability": float(probabilities["very_heavy"][0]),
            "extreme_probability": float(probabilities["extreme"][0]),
            "model_version": "Phase 3F frozen inference bundles",
            "feature_version": moe["feature_version"],
        }
        for key in ("anvaya_corrected", "global_ml_rain_mm", "heavy_probability", "very_heavy_probability", "extreme_probability"):
            if not math.isfinite(result[key]):
                raise ValueError(f"Frozen model returned a non-finite value for {key}")
        result["anvaya_minus_global_mm"] = result["anvaya_corrected"] - result["global_ml_rain_mm"]
        result["explanation"] = self._explain_correction(moe, row, posterior, regime_distribution, result)
        return result

    @staticmethod
    def _explain_correction(
        moe: dict[str, Any],
        row: np.ndarray,
        posterior: np.ndarray,
        regime_distribution: dict[str, float],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """Return local TreeSHAP-style attributions for ANVAYA minus Global ML."""
        names = list(FEATURE_SCHEMA)
        try:
            global_booster = moe["global_fallback"].booster_
            global_parts = np.asarray(global_booster.predict(row, pred_contrib=True), dtype=np.float64)[0]
            mixture_parts = np.zeros(len(names) + 1, dtype=np.float64)
            for index, regime in enumerate(moe["regime_labels"]):
                model = moe["experts"].get(regime, moe["global_fallback"])
                parts = np.asarray(model.booster_.predict(row, pred_contrib=True), dtype=np.float64)[0]
                if len(parts) != len(names) + 1:
                    raise ValueError("TreeSHAP contribution vector does not match frozen feature schema")
                mixture_parts += float(posterior[0, index]) * parts
            delta_parts = mixture_parts - global_parts
            feature_values = row[0].astype(float)
            ordered = sorted(range(len(names)), key=lambda i: abs(float(delta_parts[i])), reverse=True)[:5]
            labels = {
                "gfs_daily_rain_mm": "GFS daily rainfall",
                "rain_last_24h": "antecedent 24-hour rainfall proxy",
                "moisture_transport_850": "850-hPa moisture-transport proxy",
                "gfs_u850": "850-hPa zonal wind",
                "gfs_v850": "850-hPa meridional wind",
                "gfs_cape": "CAPE instability",
                "gfs_precipitable_water": "precipitable water",
                "gfs_mslp": "mean sea-level pressure",
                "mslp_anomaly": "mean sea-level pressure anomaly",
                "terrain_elevation_proxy": "terrain-elevation proxy",
                "orographic_flow_proxy": "orographic-flow proxy",
                "gfs_rh2m": "2-m relative humidity",
                "gfs_t2m": "2-m temperature",
            }
            contributions = [
                {
                    "feature": names[index],
                    "physical_driver": labels.get(names[index], names[index].replace("_", " ")),
                    "live_feature_value": float(feature_values[index]),
                    "correction_contribution_mm": float(delta_parts[index]),
                }
                for index in ordered
            ]
            all_contributions = [
                {
                    "feature": names[index],
                    "physical_driver": labels.get(names[index], names[index].replace("_", " ")),
                    "live_feature_value": float(feature_values[index]),
                    "correction_contribution_mm": float(delta_parts[index]),
                }
                for index in range(len(names))
            ]
            drivers = [item for item in contributions if abs(item["correction_contribution_mm"]) >= 1e-8]
            driver_text = ", ".join(
                f"{item['physical_driver']} ({item['correction_contribution_mm']:+.2f} mm)"
                for item in drivers[:3]
            ) or "no individual feature contribution exceeds numerical tolerance"
            correction = prediction["anvaya_minus_global_mm"]
            return {
                "label": "MODEL-BASED EXPLANATION",
                "method": "LightGBM pred_contrib (TreeSHAP-style) values for each frozen expert, soft-weighted by the live regime posterior and differenced from the frozen global correction model.",
                "comparison": "ANVAYA minus Global ML rainfall forecast",
                "correction_amount_mm": correction,
                "base_value_difference_mm": float(delta_parts[-1]),
                "top_feature_contributions": contributions,
                "all_feature_contributions": all_contributions,
                "dominant_physical_drivers": [item["physical_driver"] for item in drivers[:3]],
                "summary": f"The frozen model attributed the {correction:+.2f} mm ANVAYA−Global ML adjustment mainly to {driver_text}.",
                "regime_probability_vector": regime_distribution,
                "causal_claim": False,
                "caveat": "Predictive feature attributions describe the frozen model's local output decomposition, not physical causation. Live input-source substitution remains unvalidated against the historical benchmark.",
            }
        except Exception as exc:
            return {
                "label": "MODEL-BASED EXPLANATION",
                "status": UNAVAILABLE,
                "method": "Frozen-model local feature contributions",
                "error": f"{type(exc).__name__}: {exc}",
                "causal_claim": False,
                "caveat": "No attribution values are fabricated when the frozen model cannot supply them.",
            }

    def _confidence(self, consensus: dict[str, Any]) -> dict[str, Any]:
        family_count = int(consensus.get("independent_model_families") or 0)
        if consensus.get("status") != "LIVE" or family_count < 2:
            return {
                "status": UNAVAILABLE,
                "score": None,
                "independent_model_families": family_count,
                "calibrated": False,
                "reason": "Insufficient independent model families for a consensus stability diagnostic",
            }
        mean_rain = float(consensus["precipitation_24h_mean_mm"])
        spread = float(consensus["precipitation_24h_spread_mm"])
        normalized_spread = spread / (abs(mean_rain) + spread) if abs(mean_rain) + spread else 0.0
        coverage = min(family_count / 3.0, 1.0)
        score = max(0.0, min(1.0, coverage * (1.0 - normalized_spread)))
        return {
            "status": "LIVE",
            "score": score,
            "independent_model_families": family_count,
            "between_family_spread_mm": spread,
            "normalized_spread": normalized_spread,
            "coverage_fraction": coverage,
            "method": self.config.get("confidence_policy", {}).get("method"),
            "calibrated": False,
            "historical_skill_claim": False,
        }

    def _warning(self, predictions: dict[str, Any] | None) -> str:
        return evaluate_warning(predictions, self.config.get("warning_policy", {}))["level"]

    def warning_detail(
        self, predictions: dict[str, Any] | None, *, coverage_fraction: float | None = None
    ) -> dict[str, Any]:
        return evaluate_warning(
            predictions,
            self.config.get("warning_policy", {}),
            coverage_fraction=coverage_fraction,
        )

    @staticmethod
    def _reference_metadata() -> dict[str, Any]:
        return {
            "status": REFERENCE,
            "provider": "India Meteorological Department, Climate Research and Services, Pune",
            "product": "RF25 0.25-degree daily gridded rainfall, locally audited annual files 2020-2024",
            "role": "historical target/reference only; never used as a live predictor or future observation",
            "current_live_value_available": False,
            "citation": "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html",
        }

    async def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        run_time: datetime | None = None,
        include_imerg: bool = True,
    ) -> dict[str, Any]:
        run = self._valid_run(run_time)
        key = (round(latitude, 4), round(longitude, 4), run.isoformat(), bool(include_imerg))
        now_monotonic = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(key)
            if cached and now_monotonic - cached[0] <= self._ttl:
                result = copy.deepcopy(cached[1])
                result["source_metadata"]["cache"] = {"status": "LIVE", "hit": True, "ttl_seconds": self._ttl}
                return result

        target_start = run + timedelta(hours=3)
        target_end = run + timedelta(hours=27)
        gfs_raw, antecedent_raw, gefs_result, ecmwf_result = await asyncio.gather(
            self._safe_call(self.gfs.fetch(latitude, longitude, run)),
            self._safe_call(self.gfs.fetch(latitude, longitude, run - timedelta(hours=48))),
            self._safe_call(self.gefs.fetch(latitude, longitude, target_end)),
            self._safe_call(self.ecmwf.fetch(latitude, longitude, run)),
        )
        gfs_payload = gfs_raw.as_dict()
        gefs_payload = gefs_result.as_dict()
        ecmwf_payload = ecmwf_result.as_dict()
        antecedent_payload = antecedent_raw.as_dict()
        consensus = build_consensus(gfs_payload, gefs_payload, ecmwf_payload)

        gfs_data = gfs_payload.get("data") if gfs_payload.get("status") == "LIVE" else None
        antecedent_gfs_data = antecedent_payload.get("data") if antecedent_payload.get("status") == "LIVE" else None
        predictions: dict[str, Any] | None = None
        model_error: str | None = None
        if not gfs_data:
            model_error = "Frozen model inference requires a live NOAA GFS single run"
        elif not antecedent_gfs_data:
            model_error = "Frozen model inference requires the earlier explicit GFS run used to construct the time-matched antecedent 24-hour feature"
        else:
            try:
                row = self._build_model_features(gfs_data, antecedent_gfs_data, latitude, longitude, target_end)
                predictions = self._predict(row)
            except Exception as exc:
                model_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Live Phase 3F inference unavailable: %s", model_error)

        # Satellite data is deliberately fetched after NWP consensus and ANVAYA
        # inference; it is observation context, not a frozen-model input.
        if include_imerg:
            try:
                imerg_result = await asyncio.wait_for(
                    self._safe_call(self.imerg.fetch(latitude, longitude, target_start=target_start)),
                    timeout=self._imerg_timeout,
                )
            except asyncio.TimeoutError:
                imerg_result = ProviderResult(
                    source=self.imerg.source,
                    status=UNAVAILABLE,
                    error=f"NASA IMERG exceeded the configured {self._imerg_timeout}s total request deadline",
                )
        else:
            imerg_result = ProviderResult(
                source=self.imerg.source,
                status=UNAVAILABLE,
                metadata={"request_skipped": True, "scope": "district samples request one separate post-model satellite context sample"},
                error="IMERG intentionally skipped for spatial NWP samples",
            )
        imerg_payload = imerg_result.as_dict()
        district_risk = aggregate_district_risk(
            district_id=None,
            boundary_geometry=None,
            point_forecasts=(
                [{
                    "latitude": latitude,
                    "longitude": longitude,
                    "probabilities": predictions,
                }]
                if predictions else []
            ),
            threshold=0.5,
            levels=[],
            below_threshold="GREEN",
        )

        source_metadata = {
            "mode": "LIVE_OPERATIONAL_MODE",
            "scientific_mode": {
                "mode": "SCIENTIFIC_MODE",
                "status": "REFERENCE",
                "description": "Frozen Phase 3F experiment and its historical benchmark remain unchanged; this endpoint does not write to scientific datasets or metrics.",
            },
            "gfs": {"status": gfs_payload["status"], "source": gfs_payload["source"], "metadata": gfs_payload["metadata"], "error": gfs_payload["error"]},
            "gfs_antecedent": {
                "status": antecedent_payload["status"],
                "source": antecedent_payload["source"],
                "metadata": antecedent_payload["metadata"],
                "error": antecedent_payload["error"],
                "feature_role": "rain_last_24h proxy; explicitly prior 00Z GFS run and exact frozen antecedent window",
                "training_source_shift": "Phase 3F used prior-day observed IMD RF25; live mode uses a prior-run GFS 24-hour forecast for the same physical interval. This operational substitution is not benchmark-validated.",
            },
            "gefs": {"status": gefs_payload["status"], "source": gefs_payload["source"], "metadata": gefs_payload["metadata"], "error": gefs_payload["error"]},
            "ecmwf": {"status": ecmwf_payload["status"], "source": ecmwf_payload["source"], "metadata": ecmwf_payload["metadata"], "error": ecmwf_payload["error"]},
            "imerg": {"status": imerg_payload["status"], "source": imerg_payload["source"], "metadata": imerg_payload["metadata"], "error": imerg_payload["error"]},
            "imd_reference": self._reference_metadata(),
            "district_boundaries": {
                "status": REFERENCE,
                "source": "National Water Data Portal district boundary release",
                "metadata_path": "data/geospatial/district_boundaries_metadata.json",
                "feature_count": 733,
                "boundary_vintage": "portal last update 2025-05-05; not asserted current after later administrative changes",
            },
            "anvaya": {
                "status": "LIVE" if predictions else UNAVAILABLE,
                "model_artifacts": [str((MODEL_DIR / "anvaya_moe.joblib").relative_to(ROOT)), str((MODEL_DIR / "multi_threshold_probabilities.joblib").relative_to(ROOT))],
                "feature_version": FEATURE_VERSION,
                "error": model_error,
                "historical_rmse_claim": False,
                "live_input_note": "IMERG is post-prediction context only; rain_last_24h is constructed from a time-matched earlier GFS run and is not the training-time IMD observation source.",
            },
            "consensus": {"status": consensus.get("status"), "reason": consensus.get("reason")},
            "target_window": {
                "start": target_start.isoformat().replace("+00:00", "Z"),
                "end": target_end.isoformat().replace("+00:00", "Z"),
                "duration_hours": 24,
            },
            "alert_policy": {
                "label": self.config.get("warning_policy", {}).get("label"),
                "levels": self.config.get("warning_policy", {}).get("levels", []),
                "scope": "this endpoint is point-level; request /api/operational/districts/{district_id} for area-weighted district aggregation",
                "disclaimer": self.config.get("warning_policy", {}).get("disclaimer"),
            },
            "cache": {"status": "LIVE", "hit": False, "ttl_seconds": self._ttl},
        }
        warning = self.warning_detail(predictions)
        result = {
            "mode": "LIVE_OPERATIONAL_MODE",
            "location": {"latitude": latitude, "longitude": longitude},
            "raw_gfs": gfs_payload,
            "gefs_mean": (gefs_payload.get("data") or {}).get("precipitation_24h_mean_mm"),
            "gefs_spread": (gefs_payload.get("data") or {}).get("precipitation_24h_spread_mm"),
            "ecmwf_forecast": ecmwf_payload,
            "ensemble_consensus": consensus,
            "anvaya_corrected": predictions.get("anvaya_corrected") if predictions else None,
            "global_ml_rain_mm": predictions.get("global_ml_rain_mm") if predictions else None,
            "anvaya_minus_raw_mm": (
                predictions.get("anvaya_corrected") - (gfs_data or {}).get("gfs_daily_rain_mm")
                if predictions and gfs_data else None
            ),
            "anvaya_minus_global_mm": predictions.get("anvaya_minus_global_mm") if predictions else None,
            "anvaya_explanation": predictions.get("explanation") if predictions else {
                "label": "MODEL-BASED EXPLANATION", "status": UNAVAILABLE,
                "error": model_error, "causal_claim": False,
            },
            "regime": predictions.get("regime") if predictions else None,
            "regime_probability": predictions.get("regime_probability") if predictions else None,
            "heavy_probability": predictions.get("heavy_probability") if predictions else None,
            "very_heavy_probability": predictions.get("very_heavy_probability") if predictions else None,
            "extreme_probability": predictions.get("extreme_probability") if predictions else None,
            "imerg_current": imerg_payload,
            "district_risk": district_risk,
            "forecast_confidence": self._confidence(consensus),
            "warning_level": self._warning(predictions),
            "warnings": warning,
            "source_metadata": source_metadata,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        async with self._cache_lock:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = (time.monotonic(), copy.deepcopy(result))
        return result

    @staticmethod
    async def _safe_call(awaitable: Any) -> ProviderResult:
        try:
            return await awaitable
        except Exception as exc:
            return ProviderResult(
                source="operational-provider",
                status=UNAVAILABLE,
                error=f"{type(exc).__name__}: {exc}",
            )


operational_service = OperationalService()
