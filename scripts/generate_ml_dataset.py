import os
import json
import numpy as np
import pandas as pd

def generate_dataset(num_samples=10000, output_path="data/sample/training_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    # Features
    latitude = np.random.uniform(8.0, 37.0, num_samples)
    longitude = np.random.uniform(68.0, 97.0, num_samples)
    gfs_t2m = np.random.normal(30.0, 4.0, num_samples)
    gfs_rh2m = np.random.uniform(30.0, 100.0, num_samples)
    gfs_mslp = np.random.normal(1005.0, 5.0, num_samples)
    gfs_u10 = np.random.normal(0.0, 5.0, num_samples)
    gfs_v10 = np.random.normal(0.0, 5.0, num_samples)
    gfs_u850 = np.random.normal(5.0, 8.0, num_samples)
    gfs_v850 = np.random.normal(2.0, 5.0, num_samples)
    gfs_cape = np.random.exponential(500.0, num_samples)
    gfs_precipitable_water = np.random.uniform(10.0, 70.0, num_samples)
    rain_last_6h = np.random.exponential(5.0, num_samples)
    rain_last_24h = rain_last_6h + np.random.exponential(10.0, num_samples)
    
    # Base rain based on RH and CAPE
    gfs_rain_mm = np.where(gfs_rh2m > 70, np.random.exponential(15.0, num_samples), np.random.exponential(2.0, num_samples))
    gfs_rain_mm += (gfs_cape / 1000) * 2.0

    df = pd.DataFrame({
        "latitude": latitude,
        "longitude": longitude,
        "gfs_t2m": gfs_t2m,
        "gfs_rh2m": gfs_rh2m,
        "gfs_mslp": gfs_mslp,
        "gfs_u10": gfs_u10,
        "gfs_v10": gfs_v10,
        "gfs_u850": gfs_u850,
        "gfs_v850": gfs_v850,
        "gfs_cape": gfs_cape,
        "gfs_precipitable_water": gfs_precipitable_water,
        "rain_last_6h": rain_last_6h,
        "rain_last_24h": rain_last_24h,
        "gfs_rain_mm": gfs_rain_mm
    })

    # Regime assignment
    regimes = []
    actual_rain_mm = []
    
    for i in range(num_samples):
        r = df.iloc[i]
        
        # Simple deterministic rules for regimes
        is_coastal = r["longitude"] < 75.0 and r["latitude"] < 20.0
        is_himalayan = r["latitude"] > 30.0
        
        if is_coastal and r["gfs_u850"] > 10.0 and r["gfs_rh2m"] > 80:
            regime = "COASTAL_RAINFALL"
            bias = 1.5
        elif is_himalayan and r["gfs_v850"] < -2.0:
            regime = "WESTERN_DISTURBANCE"
            bias = 1.2
        elif r["gfs_mslp"] < 998.0:
            regime = "LOW_DEPRESSION"
            bias = 2.0
        elif r["gfs_u850"] > 5.0 and r["gfs_precipitable_water"] > 50:
            regime = "ACTIVE_MONSOON"
            bias = 1.3
        elif r["gfs_u850"] <= 5.0 and r["gfs_rh2m"] < 60:
            regime = "BREAK_MONSOON"
            bias = 0.5
        elif r["latitude"] > 25.0 and r["longitude"] > 85.0:
            regime = "OROGRAPHIC_RAINFALL"
            bias = 1.4
        else:
            regime = "OTHER"
            bias = 1.0
            
        regimes.append(regime)
        
        # Add some noise to actual rain
        true_rain = r["gfs_rain_mm"] * bias + np.random.normal(0, 2.0)
        actual_rain_mm.append(max(0.0, true_rain))
        
    df["regime"] = regimes
    df["actual_rain_mm"] = actual_rain_mm
    df["heavy_rain"] = (df["actual_rain_mm"] >= 64.5).astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} samples at {output_path}")

if __name__ == "__main__":
    generate_dataset()
