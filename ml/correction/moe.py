"""
Regime-Aware Mixture of Experts (MoE) Rainfall Correction Model for ANVAYA.

CORE ARCHITECTURAL INNOVATION:
Instead of a discontinuous hard regime switch, the final post-processed rainfall
is a soft-gated continuous expectation over all regime experts:

    y_corrected = sum( P(regime_i | x) * expert_i(x) )

This completely prevents step-discontinuities at regime transitions while retaining
regime-specific physical error corrections.
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from ml.regime.labels import REGIME_LABELS

CORRECTION_FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m",
    "gfs_u10", "gfs_v10", "gfs_u850", "gfs_v850",
    "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h",
    "latitude", "longitude"
]

class MixtureOfExpertsCorrectionModel:
    """
    Soft-gated Mixture-of-Experts rainfall bias corrector.
    Combines predictions from regime-specialized regressors weighted by regime posteriors.
    """
    def __init__(self, models_dir: Optional[str] = None):
        self.experts = {}
        self.global_model = None

        if models_dir and os.path.exists(models_dir):
            for file_path in glob.glob(os.path.join(models_dir, "*.joblib")):
                base_name = os.path.basename(file_path).replace(".joblib", "").upper()
                try:
                    loaded = joblib.load(file_path)
                    if base_name in {"GLOBAL", "GLOBAL_CORRECTION"}:
                        self.global_model = loaded
                    elif base_name.startswith("EXPERT_"):
                        self.experts[base_name[len("EXPERT_"):]] = loaded
                    elif base_name in REGIME_LABELS:
                        # Backward-compatible legacy artifact naming.
                        self.experts[base_name] = loaded
                    else:
                        # Ignore unrelated artifacts in a shared model directory.
                        continue
                except Exception as e:
                    print(f"Warning: could not load expert {file_path}: {e}")

    def _prepare_features(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        row = dict(features_dict)
        df = pd.DataFrame([row])
        for col in CORRECTION_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        return df[CORRECTION_FEATURES]

    def _predict_single_expert(self, expert, df: pd.DataFrame) -> float:
        """Evaluate a single expert model in expm1 space."""
        try:
            target_cols = list(getattr(expert, "feature_names_in_", CORRECTION_FEATURES))
            sub_df = df.copy()
            for col in target_cols:
                if col not in sub_df.columns:
                    sub_df[col] = 0.0
            pred_log = expert.predict(sub_df[target_cols])[0]
            val = float(np.expm1(pred_log))
            return max(0.0, val)
        except Exception:
            # Fallback if prediction fails
            return float(df.get("gfs_rain_mm", 0.0).iloc[0])

    def correct_rainfall(
        self,
        features_dict: Dict[str, Any],
        regime_probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compute soft-gated MoE corrected rainfall forecast.
        
        Args:
            features_dict: Meteorological input vector.
            regime_probabilities: Posterior P(regime | x) for each regime.
            
        Returns:
            Dict containing raw_forecast_mm, corrected_forecast_mm,
            adjustment_mm, global_corrected_mm, and individual expert contributions.
        """
        raw_forecast = float(features_dict.get("gfs_rain_mm", 0.0))
        df = self._prepare_features(features_dict)

        # Baseline: Global ML correction (if available)
        if self.global_model is not None:
            global_pred = self._predict_single_expert(self.global_model, df)
        else:
            global_pred = raw_forecast

        if not self.experts:
            return {
                "raw_forecast_mm": raw_forecast,
                "corrected_forecast_mm": raw_forecast,
                "global_corrected_mm": global_pred,
                "adjustment_mm": 0.0,
                "regime_corrections": {},
                "prediction_mode": "raw_fallback",
                "fallback_regimes": list(regime_probabilities),
            }

        # Mixture of experts soft blending
        moe_sum = 0.0
        weight_sum = 0.0
        expert_outputs = {}
        fallback_regimes = []

        for regime, prob in regime_probabilities.items():
            if prob <= 0.0001:
                continue

            # Look up expert model (case-insensitive normalized)
            regime_key = regime.upper().strip()
            expert = self.experts.get(regime_key)

            if expert is not None:
                exp_pred = self._predict_single_expert(expert, df)
            else:
                # If specific expert not loaded, fall back to global or raw
                exp_pred = global_pred
                fallback_regimes.append(regime_key)

            expert_outputs[regime_key] = exp_pred
            moe_sum += prob * exp_pred
            weight_sum += prob

        if weight_sum > 0:
            final_corrected = max(0.0, moe_sum / weight_sum)
        else:
            final_corrected = raw_forecast

        adjustment = final_corrected - raw_forecast

        return {
            "raw_forecast_mm": round(raw_forecast, 2),
            "corrected_forecast_mm": round(final_corrected, 2),
            "global_corrected_mm": round(global_pred, 2),
            "adjustment_mm": round(adjustment, 2),
            "regime_corrections": {k: round(v, 2) for k, v in expert_outputs.items()},
            "prediction_mode": "fallback_global" if fallback_regimes else "dedicated_moe",
            "fallback_regimes": fallback_regimes,
        }

    def predict(
        self,
        features_dict: Dict[str, Any],
        regime_probabilities: Dict[str, float]
    ) -> float:
        """Convenience method returning the corrected rainfall float directly."""
        res = self.correct_rainfall(features_dict, regime_probabilities)
        return res["corrected_forecast_mm"]
