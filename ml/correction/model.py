import os
import glob
import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
    "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h", "latitude", "longitude"
]

class CorrectionModel:
    def __init__(self, models_dir="ml/correction/models"):
        self.models = {}
        if os.path.exists(models_dir):
            for path in glob.glob(f"{models_dir}/*.joblib"):
                regime_name = os.path.basename(path).replace(".joblib", "").upper()
                self.models[regime_name] = joblib.load(path)
                
    def correct_rainfall(self, features_dict, regime):
        raw_forecast = features_dict.get("gfs_rain_mm", 0.0)
        
        if regime not in self.models:
            return raw_forecast, raw_forecast, 0.0
            
        df = pd.DataFrame([features_dict], columns=FEATURES)
        pred_log = self.models[regime].predict(df)[0]
        corrected = np.expm1(pred_log)
        
        # Ensure non-negative
        corrected = max(0.0, float(corrected))
        adjustment_mm = corrected - raw_forecast
        
        return float(raw_forecast), corrected, adjustment_mm
