import os
import joblib
import pandas as pd

FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
    "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h", "latitude", "longitude"
]

class RegimeModel:
    def __init__(self, model_path="ml/regime/regime_model.joblib"):
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
    def predict(self, features_dict):
        if not self.model:
            return "OTHER", 0.0
            
        df = pd.DataFrame([features_dict], columns=FEATURES)
        probs = self.model.predict_proba(df)[0]
        classes = self.model.classes_
        
        max_idx = probs.argmax()
        return classes[max_idx], float(probs[max_idx])
