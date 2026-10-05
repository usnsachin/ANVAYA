import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
    "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h", "latitude", "longitude"
]

def train_correction_models(train_df):
    os.makedirs("ml/correction/models", exist_ok=True)
    models = {}
    
    regimes = train_df["regime"].unique()
    for regime in regimes:
        regime_df = train_df[train_df["regime"] == regime]
        if len(regime_df) < 50:
            continue
            
        X = regime_df[FEATURES]
        y = np.log1p(regime_df["actual_rain_mm"])
        
        reg = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        reg.fit(X, y)
        
        model_name = regime.lower()
        joblib.dump(reg, f"ml/correction/models/{model_name}.joblib")
        models[regime] = reg
        
    return models

if __name__ == "__main__":
    df = pd.read_csv("data/sample/training_data.csv")
    train_correction_models(df)
