"""
Probabilistic Weather Regime Classifier for ANVAYA.
Estimates posterior distribution P(regime | features) for soft mixture-of-experts gating.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from ml.regime.labels import REGIME_LABELS

CLASSIFIER_FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m",
    "gfs_u10", "gfs_v10", "gfs_u850", "gfs_v850",
    "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h",
    "latitude", "longitude",
    "wind_speed_10m", "wind_speed_850", "moisture_transport_850",
    "mslp_anomaly", "lead_time_h", "month"
]

class RegimeClassifier:
    """
    Probabilistic weather regime classifier.
    Outputs a soft probability vector over 7 regimes used as gating weights in MoE.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.classes_ = list(REGIME_LABELS)
        
        if model_path and os.path.exists(model_path):
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict):
                self.model = loaded.get("model")
                self.classes_ = list(loaded.get("classes", REGIME_LABELS))
            else:
                self.model = loaded
                if hasattr(loaded, "classes_"):
                    self.classes_ = list(loaded.classes_)

    def _prepare_df(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        row = dict(features_dict)
        # Compute dynamic derived features if absent
        u10 = float(row.get("gfs_u10", 0.0))
        v10 = float(row.get("gfs_v10", 0.0))
        u850 = float(row.get("gfs_u850", 0.0))
        v850 = float(row.get("gfs_v850", 0.0))
        pw = float(row.get("gfs_precipitable_water", 0.0))
        mslp = float(row.get("gfs_mslp", 1013.25))

        row.setdefault("wind_speed_10m", float(np.sqrt(u10**2 + v10**2)))
        w850 = float(np.sqrt(u850**2 + v850**2))
        row.setdefault("wind_speed_850", w850)
        row.setdefault("moisture_transport_850", float(w850 * max(0.0, pw)))
        row.setdefault("mslp_anomaly", float(mslp - 1013.25))
        row.setdefault("lead_time_h", 24.0)
        row.setdefault("month", 7.0)

        df = pd.DataFrame([row])
        # Use exact feature names the model was trained on if available
        target_features = list(getattr(self.model, "feature_names_in_", CLASSIFIER_FEATURES))
        for col in target_features:
            if col not in df.columns:
                df[col] = 0.0
        return df[target_features]

    def predict_proba(self, features_dict: Dict[str, Any]) -> Dict[str, float]:
        """Return calibrated probability for each weather regime."""
        if self.model is None:
            n = len(REGIME_LABELS)
            return {r: 1.0 / n for r in REGIME_LABELS}

        df = self._prepare_df(features_dict)
        probs = self.model.predict_proba(df)[0]
        classes = self.classes_

        prob_dict = {r: 0.0 for r in REGIME_LABELS}
        for cls_name, p in zip(classes, probs):
            prob_dict[str(cls_name)] = float(p)

        # Normalize sum to 1.0
        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v / total for k, v in prob_dict.items()}
        return prob_dict

    def predict(self, features_dict: Dict[str, Any]) -> Tuple[str, float]:
        """Return dominant regime and its confidence probability."""
        probs = self.predict_proba(features_dict)
        best_regime = max(probs, key=probs.get)
        return best_regime, float(probs[best_regime])

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train classifier with class weighting for imbalanced regimes."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        
        # Ensure feature columns
        X_sub = X.copy()
        for col in CLASSIFIER_FEATURES:
            if col not in X_sub.columns:
                X_sub[col] = 0.0
        X_sub = X_sub[CLASSIFIER_FEATURES]

        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            random_state=42,
            class_weight="balanced"
        )
        clf.fit(X_sub, y)
        self.model = clf
        self.classes_ = list(clf.classes_)

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_}, output_path)
