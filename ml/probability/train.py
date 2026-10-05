import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = [
    "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
    "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
    "rain_last_6h", "rain_last_24h", "latitude", "longitude"
]

def train_probability_model(train_df):
    X = train_df[FEATURES]
    y = train_df["heavy_rain"]
    
    clf = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    clf.fit(X, y)
    
    os.makedirs("ml/probability", exist_ok=True)
    joblib.dump(clf, "ml/probability/heavy_rain_model.joblib")
    return clf

if __name__ == "__main__":
    df = pd.read_csv("data/sample/training_data.csv")
    train_probability_model(df)
