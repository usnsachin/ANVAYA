"""
ANVAYA ML Service

Loads trained models once at startup and exposes a single inference pipeline.
Supports:
  - Regime classification (soft probabilities)
  - Regime-aware Mixture-of-Experts correction
  - Multi-threshold heavy rainfall probability
  - Explainability signals

Falls back gracefully when models are missing.
"""
import os
import sys
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Add the project root to sys.path so ml.* modules are importable
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def _configured_model_dir():
    """Return an explicitly selected model directory, if configured."""
    configured = os.environ.get("ANVAYA_MODEL_DIR")
    if not configured:
        return None
    return configured if os.path.isabs(configured) else os.path.join(root_dir, configured)


def _load_regime_model():
    """Load the best available regime classifier (v2 first, then legacy)."""
    configured_dir = _configured_model_dir()
    if configured_dir:
        configured_path = os.path.join(configured_dir, "regime_classifier.joblib")
        if os.path.exists(configured_path):
            try:
                from ml.regime.classifier import RegimeClassifier
                return RegimeClassifier(configured_path)
            except Exception as e:
                logger.warning(f"Configured regime model load failed: {e}")

    v2_path = os.path.join(root_dir, "ml/regime/regime_classifier_v2.joblib")
    legacy_path = os.path.join(root_dir, "ml/regime/regime_model.joblib")

    # Try new RegimeClassifier
    try:
        from ml.regime.classifier import RegimeClassifier
        if os.path.exists(v2_path):
            clf = RegimeClassifier(v2_path)
            logger.info("Loaded regime_classifier_v2 (RegimeClassifier)")
            return clf
        if os.path.exists(legacy_path):
            clf = RegimeClassifier(legacy_path)
            logger.info("Loaded legacy regime_model as RegimeClassifier")
            return clf
    except Exception as e:
        logger.warning(f"RegimeClassifier load failed: {e}")

    # Fall back to old RegimeModel
    try:
        from ml.regime.model import RegimeModel
        if os.path.exists(legacy_path):
            m = RegimeModel(legacy_path)
            logger.info("Loaded legacy RegimeModel")
            return _LegacyRegimeAdapter(m)
    except Exception as e:
        logger.warning(f"Legacy RegimeModel load failed: {e}")

    return None


class _LegacyRegimeAdapter:
    """Wraps old RegimeModel to present the same interface as RegimeClassifier."""
    def __init__(self, legacy_model):
        self._m = legacy_model

    @property
    def model(self):
        return self._m.model

    def predict(self, features):
        return self._m.predict(features)

    def predict_proba(self, features):
        regime, conf = self._m.predict(features)
        # Distribute remaining probability uniformly
        from ml.regime.classifier import REGIME_LABELS
        n = len(REGIME_LABELS)
        rest = (1.0 - conf) / max(1, n - 1)
        return {r: conf if r == regime else rest for r in REGIME_LABELS}


def _load_correction_model():
    """Load MoE correction model if available, else fall back to legacy."""
    models_dir = _configured_model_dir() or os.path.join(root_dir, "ml/correction/models")

    try:
        from ml.correction.moe import MixtureOfExpertsCorrectionModel
        moe = MixtureOfExpertsCorrectionModel(models_dir)
        if moe.experts:
            logger.info(f"Loaded MoE correction model with {len(moe.experts)} experts")
            return moe
    except Exception as e:
        logger.warning(f"MoE correction load failed: {e}")

    # Fall back to legacy CorrectionModel
    try:
        from ml.correction.model import CorrectionModel
        m = CorrectionModel(models_dir)
        logger.info("Loaded legacy CorrectionModel")
        return _LegacyCorrectionAdapter(m)
    except Exception as e:
        logger.warning(f"Legacy CorrectionModel load failed: {e}")

    return None


class _LegacyCorrectionAdapter:
    """Wraps legacy CorrectionModel to expose correct_rainfall(features, regime_proba)."""
    def __init__(self, legacy_model):
        self._m = legacy_model

    @property
    def experts(self):
        return self._m.models

    @property
    def global_model(self):
        return None

    def correct_rainfall(self, features, regime_proba):
        # Pick top regime for hard dispatch
        top_regime = max(regime_proba, key=regime_proba.get) if regime_proba else "OTHER"
        raw, corrected, adj = self._m.correct_rainfall(features, top_regime)
        return {
            "raw_forecast_mm": raw,
            "corrected_forecast_mm": corrected,
            "global_corrected_mm": raw,
            "adjustment_mm": adj,
            "regime_corrections": {},
        }


def _load_probability_model():
    """Load multi-threshold model if available, else legacy single-threshold."""
    configured_dir = _configured_model_dir()
    multi_dir = os.path.join(configured_dir, "probability") if configured_dir else os.path.join(root_dir, "ml/probability/models")

    try:
        from ml.probability.multi_threshold import MultiThresholdProbabilityModel
        mt = MultiThresholdProbabilityModel(multi_dir)
        if mt.models:
            logger.info(f"Loaded multi-threshold probability model ({len(mt.models)} thresholds)")
            return mt
    except Exception as e:
        logger.warning(f"Multi-threshold load failed: {e}")

    # Fall back
    try:
        from ml.probability.model import ProbabilityModel
        legacy_path = os.path.join(root_dir, "ml/probability/heavy_rain_model.joblib")
        m = ProbabilityModel(legacy_path)
        logger.info("Loaded legacy ProbabilityModel")
        return _LegacyProbAdapter(m)
    except Exception as e:
        logger.warning(f"Legacy ProbabilityModel load failed: {e}")

    return None


class _LegacyProbAdapter:
    """Wraps legacy single-threshold model to return multi-threshold dict."""
    def __init__(self, legacy_model):
        self._m = legacy_model

    @property
    def models(self):
        return {"heavy": self._m.model} if self._m.model else {}

    def predict(self, features):
        heavy = self._m.predict(features)
        return {
            "heavy": heavy,
            "very_heavy": max(0.0, heavy - 0.25),
            "extreme": max(0.0, heavy - 0.5),
        }


class MLService:
    """Central inference service for the ANVAYA pipeline.

    Loaded once at startup. Exposes run_forecast() for single-point inference.
    """

    def __init__(self):
        logger.info("Loading ANVAYA ML models...")
        self.data_mode = os.environ.get("DATA_MODE", "demo").upper()
        self.model_version = "prototype"
        # Never advertise the scientific manifest while the service is using
        # the legacy/demo model directories.  Scientific artifacts are opt-in
        # through ANVAYA_MODEL_DIR so the default demo path remains honest.
        configured_dir = _configured_model_dir()
        if configured_dir:
            manifest_path = os.path.join(root_dir, "models/manifest.json")
            if os.path.exists(manifest_path):
                try:
                    import json
                    with open(manifest_path, "r", encoding="utf-8") as handle:
                        self.model_version = json.load(handle).get("model_version", self.model_version)
                except Exception:
                    pass
        self.regime_model = _load_regime_model()
        self.correction_model = _load_correction_model()
        self.probability_model = _load_probability_model()
        ready = self.is_ready()
        logger.info(f"MLService ready={ready}")

    def is_ready(self) -> bool:
        return (
            self.regime_model is not None
            and self.correction_model is not None
            and self.probability_model is not None
        )

    def generate_explanation(
        self,
        features: Dict[str, float],
        regime: str,
        regime_proba: Dict[str, float],
        adjustment: float,
    ) -> Dict[str, Any]:
        signals = []
        if features.get("gfs_u850", 0) > 8:
            signals.append("Strong low-level moisture transport (850 hPa)")
        if features.get("gfs_rh2m", 0) > 80:
            signals.append("High surface relative humidity")
        if features.get("gfs_mslp", 1013) < 1000:
            signals.append("Low-pressure system detected (MSLP anomaly)")
        if features.get("gfs_cape", 0) > 1000:
            signals.append("Elevated atmospheric instability (CAPE)")
        if features.get("rain_last_24h", 0) > 30:
            signals.append("High antecedent rainfall (last 24 h)")
        if features.get("gfs_precipitable_water", 0) > 55:
            signals.append("High precipitable water column")
        if not signals:
            signals.append("Typical seasonal conditions")

        direction = "increased" if adjustment >= 0 else "decreased"
        amt = abs(round(adjustment, 1))
        reason = (
            f"Detected regime: {regime} "
            f"(confidence {regime_proba.get(regime, 0):.0%}). "
            f"AI {direction} raw NWP by {amt} mm based on regime-specific correction."
        )

        return {
            "detected_regime": regime,
            "regime_confidence": round(regime_proba.get(regime, 0.0), 3),
            "signals": signals[:3],
            "reason": reason,
            "data_mode": self.data_mode,
            "model_version": self.model_version,
        }

    def run_forecast(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Full ANVAYA inference pipeline for one location.

        Pipeline:
          1. Regime classifier → soft probability vector
          2. MoE correction (weighted by regime probs)
          3. Multi-threshold probability heads
          4. Explainability signals
        """
        # Engineer any missing derived features on-the-fly
        import math
        u10 = features.get("gfs_u10", 0.0)
        v10 = features.get("gfs_v10", 0.0)
        u850 = features.get("gfs_u850", 0.0)
        v850 = features.get("gfs_v850", 0.0)
        features.setdefault("wind_speed_10m", math.sqrt(u10**2 + v10**2))
        features.setdefault("wind_speed_850", math.sqrt(u850**2 + v850**2))
        features.setdefault(
            "moisture_transport_850",
            features["wind_speed_850"] * features.get("gfs_precipitable_water", 0.0),
        )
        features.setdefault("mslp_anomaly", features.get("gfs_mslp", 1013.25) - 1013.25)
        features.setdefault("lead_time_h", 24.0)
        features.setdefault("month", 7.0)

        # 1. Regime classification
        if self.regime_model:
            regime, regime_conf = self.regime_model.predict(features)
            try:
                regime_proba = self.regime_model.predict_proba(features)
            except Exception:
                regime_proba = {regime: regime_conf}
        else:
            regime, regime_conf = "OTHER", 0.5
            regime_proba = {regime: regime_conf}

        # 2. Rainfall correction
        if self.correction_model:
            correction = self.correction_model.correct_rainfall(features, regime_proba)
        else:
            raw = features.get("gfs_rain_mm", 0.0)
            correction = {
                "raw_forecast_mm": raw,
                "corrected_forecast_mm": raw,
                "global_corrected_mm": raw,
                "adjustment_mm": 0.0,
                "regime_corrections": {},
                "prediction_mode": "raw_fallback",
                "fallback_regimes": list(regime_proba),
            }

        # 3. Heavy rain probabilities
        if self.probability_model:
            probs = self.probability_model.predict(features)
        else:
            probs = {"heavy": 0.0, "very_heavy": 0.0, "extreme": 0.0}

        heavy_prob = probs.get("heavy", 0.0)
        risk_level = "High" if heavy_prob > 0.7 else "Medium" if heavy_prob > 0.4 else "Low"

        # 4. Explanation
        explanation = self.generate_explanation(
            features, regime, regime_proba, correction["adjustment_mm"]
        )

        return {
            "raw_forecast_mm": round(correction["raw_forecast_mm"], 2),
            "corrected_forecast_mm": round(correction["corrected_forecast_mm"], 2),
            "global_corrected_mm": round(correction.get("global_corrected_mm", correction["raw_forecast_mm"]), 2),
            "adjustment_mm": round(correction["adjustment_mm"], 2),
            "regime": regime,
            "regime_confidence": round(regime_conf, 3),
            "regime_probabilities": {k: round(v, 3) for k, v in regime_proba.items()},
            "heavy_rain_probability": round(heavy_prob, 3),
            "very_heavy_probability": round(probs.get("very_heavy", 0.0), 3),
            "extreme_probability": round(probs.get("extreme", 0.0), 3),
            "prediction_mode": correction.get("prediction_mode", "legacy"),
            "fallback_regimes": correction.get("fallback_regimes", []),
            "data_mode": self.data_mode,
            "model_version": self.model_version,
            "risk_level": risk_level,
            "explanation": explanation,
        }


ml_service = MLService()
