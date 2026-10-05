import os
import joblib
import pandas as pd

FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
    "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h", "latitude", "longitude"
]

class ProbabilityModel:
    def __init__(self, model_path="ml/probability/heavy_rain_model.joblib"):
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
    def predict(self, features_dict):
        if not self.model:
            return 0.0
            
        df = pd.DataFrame([features_dict], columns=FEATURES)
        probs = self.model.predict_proba(df)[0]
        
        # Class 1 is heavy_rain = 1
        if 1 in self.model.classes_:
            idx = list(self.model.classes_).index(1)
            return float(probs[idx])
        return 0.0
