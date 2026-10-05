"""
Multi-Threshold Heavy Rainfall Probability Model for ANVAYA.
Estimates calibrated probabilities for meteorological hazard warning thresholds:
  - Heavy Rainfall: >= 64.5 mm/day
  - Very Heavy Rainfall: >= 115.6 mm/day
  - Extremely Heavy Rainfall: >= 204.5 mm/day
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

IMD_THRESHOLDS = {
    "heavy": 64.5,
    "very_heavy": 115.6,
    "extreme": 204.5
}

PROB_FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m",
    "gfs_u10", "gfs_v10", "gfs_u850", "gfs_v850",
    "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h",
    "latitude", "longitude"
]

class MultiThresholdProbabilityModel:
    """
    Calibrated multi-threshold heavy rainfall probability predictor.
    """
    def __init__(self, models_dir: Optional[str] = None, fallback_model_path: Optional[str] = None):
        self.models = {}

        # 1. Look for threshold-specific models inside models_dir
        if models_dir and os.path.exists(models_dir):
            for key in IMD_THRESHOLDS:
                p = os.path.join(models_dir, f"{key}_model.joblib")
                if os.path.exists(p):
                    try:
                        self.models[key] = joblib.load(p)
                    except Exception as e:
                        print(f"Warning: could not load {key} model: {e}")

        # 2. If single fallback model supplied, use it for heavy (>= 64.5mm)
        if "heavy" not in self.models and fallback_model_path and os.path.exists(fallback_model_path):
            try:
                self.models["heavy"] = joblib.load(fallback_model_path)
            except Exception as e:
                print(f"Warning: could not load fallback model: {e}")

    def _prepare_df(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        row = dict(features_dict)
        df = pd.DataFrame([row])
        return df

    @staticmethod
    def _target_features(model) -> list:
        target_features = getattr(model, "feature_names_in_", None)
        if target_features is None and hasattr(model, "estimator"):
            target_features = getattr(model.estimator, "feature_names_in_", None)
        return list(target_features) if target_features is not None else list(PROB_FEATURES)

    def _with_target_features(self, df: pd.DataFrame, model) -> pd.DataFrame:
        target_features = self._target_features(model)
        for col in target_features:
            if col not in df.columns:
                df[col] = 0.0
        return df[target_features]

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Return calibrated probabilities of exceeding each threshold.
        
        Returns:
            Dict: {'heavy': p1, 'very_heavy': p2, 'extreme': p3}
            where 0.0 <= p <= 1.0. Monotonically decreasing by definition.
        """
        df = self._prepare_df(features_dict)
        raw_rain = float(features_dict.get("gfs_rain_mm", 0.0))

        # 1. Base probability for heavy rain (>= 64.5 mm)
        if "heavy" in self.models:
            try:
                model = self.models["heavy"]
                probs = model.predict_proba(self._with_target_features(df.copy(), model))[0]
                if hasattr(model, "classes_") and 1 in model.classes_:
                    idx = list(model.classes_).index(1)
                    heavy_p = float(probs[idx])
                else:
                    heavy_p = float(probs[-1])
            except Exception:
                heavy_p = float(min(1.0, max(0.0, raw_rain / (IMD_THRESHOLDS["heavy"] * 1.5))))
        else:
            # Physical heuristic fallback based on raw NWP precipitation
            heavy_p = float(min(1.0, max(0.0, raw_rain / (IMD_THRESHOLDS["heavy"] * 1.5))))

        # 2. Probability for Very Heavy (>= 115.6 mm)
        if "very_heavy" in self.models:
            try:
                model = self.models["very_heavy"]
                probs = model.predict_proba(self._with_target_features(df.copy(), model))[0]
                idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
                very_heavy_p = float(probs[idx])
            except Exception:
                very_heavy_p = heavy_p * 0.4
        else:
            # Calibrated tail approximation
            ratio = max(0.0, (raw_rain - IMD_THRESHOLDS["heavy"]) / (IMD_THRESHOLDS["very_heavy"] - IMD_THRESHOLDS["heavy"]))
            very_heavy_p = float(heavy_p * min(1.0, max(0.05, ratio * 0.8)))

        # 3. Probability for Extremely Heavy (>= 204.5 mm)
        if "extreme" in self.models:
            try:
                model = self.models["extreme"]
                probs = model.predict_proba(self._with_target_features(df.copy(), model))[0]
                idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
                extreme_p = float(probs[idx])
            except Exception:
                extreme_p = very_heavy_p * 0.25
        else:
            extreme_p = float(very_heavy_p * 0.25)

        # Enforce physical monotonicity: P(extreme) <= P(very_heavy) <= P(heavy)
        heavy_p = max(0.0, min(1.0, heavy_p))
        very_heavy_p = max(0.0, min(heavy_p, very_heavy_p))
        extreme_p = max(0.0, min(very_heavy_p, extreme_p))

        return {
            "heavy": round(heavy_p, 4),
            "very_heavy": round(very_heavy_p, 4),
            "extreme": round(extreme_p, 4)
        }
