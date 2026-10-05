from fastapi import APIRouter
from app.services.ml_service import ml_service

router = APIRouter(tags=["districts"])

# Deterministic districts with sample features
DEMO_DISTRICTS = [
    {"id": "MH-MU", "name": "Mumbai", "state": "Maharashtra", "lat": 19.076, "lon": 72.8777, "features": {"gfs_rain_mm": 45.0, "gfs_mslp": 1002.0, "gfs_t2m": 29.5, "gfs_rh2m": 85.0, "gfs_u10": 4.5, "gfs_v10": 2.1, "gfs_u850": 12.5, "gfs_v850": 3.0, "gfs_cape": 800.0, "gfs_precipitable_water": 60.0, "rain_last_6h": 15.0, "rain_last_24h": 40.0}},
    {"id": "MH-PU", "name": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "features": {"gfs_rain_mm": 15.0, "gfs_mslp": 1005.0, "gfs_t2m": 27.5, "gfs_rh2m": 75.0, "gfs_u10": 2.5, "gfs_v10": 1.1, "gfs_u850": 8.5, "gfs_v850": 1.0, "gfs_cape": 500.0, "gfs_precipitable_water": 45.0, "rain_last_6h": 5.0, "rain_last_24h": 12.0}},
    {"id": "MH-RN", "name": "Ratnagiri", "state": "Maharashtra", "lat": 16.9902, "lon": 73.3120, "features": {"gfs_rain_mm": 65.0, "gfs_mslp": 1000.0, "gfs_t2m": 28.5, "gfs_rh2m": 90.0, "gfs_u10": 6.5, "gfs_v10": 3.1, "gfs_u850": 15.5, "gfs_v850": 4.0, "gfs_cape": 1200.0, "gfs_precipitable_water": 65.0, "rain_last_6h": 25.0, "rain_last_24h": 70.0}},
    {"id": "GA-NO", "name": "Goa North", "state": "Goa", "lat": 15.4909, "lon": 73.8278, "features": {"gfs_rain_mm": 55.0, "gfs_mslp": 1001.0, "gfs_t2m": 28.0, "gfs_rh2m": 88.0, "gfs_u10": 5.5, "gfs_v10": 2.5, "gfs_u850": 14.0, "gfs_v850": 3.5, "gfs_cape": 1000.0, "gfs_precipitable_water": 62.0, "rain_last_6h": 20.0, "rain_last_24h": 55.0}},
    {"id": "KL-KO", "name": "Kochi", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "features": {"gfs_rain_mm": 35.0, "gfs_mslp": 1004.0, "gfs_t2m": 27.0, "gfs_rh2m": 82.0, "gfs_u10": 3.5, "gfs_v10": 1.5, "gfs_u850": 10.0, "gfs_v850": 2.0, "gfs_cape": 600.0, "gfs_precipitable_water": 55.0, "rain_last_6h": 10.0, "rain_last_24h": 25.0}},
    {"id": "KA-MA", "name": "Mangaluru", "state": "Karnataka", "lat": 12.9141, "lon": 74.8560, "features": {"gfs_rain_mm": 40.0, "gfs_mslp": 1003.0, "gfs_t2m": 27.5, "gfs_rh2m": 84.0, "gfs_u10": 4.0, "gfs_v10": 1.8, "gfs_u850": 11.0, "gfs_v850": 2.5, "gfs_cape": 700.0, "gfs_precipitable_water": 58.0, "rain_last_6h": 12.0, "rain_last_24h": 30.0}},
    {"id": "AS-GU", "name": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "features": {"gfs_rain_mm": 25.0, "gfs_mslp": 1006.0, "gfs_t2m": 26.0, "gfs_rh2m": 80.0, "gfs_u10": 2.0, "gfs_v10": 1.0, "gfs_u850": 6.0, "gfs_v850": 1.5, "gfs_cape": 400.0, "gfs_precipitable_water": 50.0, "rain_last_6h": 8.0, "rain_last_24h": 20.0}},
    {"id": "WB-KO", "name": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "features": {"gfs_rain_mm": 30.0, "gfs_mslp": 998.0, "gfs_t2m": 28.0, "gfs_rh2m": 85.0, "gfs_u10": 3.0, "gfs_v10": 2.0, "gfs_u850": 8.0, "gfs_v850": 3.0, "gfs_cape": 900.0, "gfs_precipitable_water": 60.0, "rain_last_6h": 10.0, "rain_last_24h": 28.0}},
    {"id": "UK-DE", "name": "Dehradun", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322, "features": {"gfs_rain_mm": 20.0, "gfs_mslp": 1008.0, "gfs_t2m": 22.0, "gfs_rh2m": 70.0, "gfs_u10": 1.5, "gfs_v10": 1.0, "gfs_u850": 4.0, "gfs_v850": 1.0, "gfs_cape": 300.0, "gfs_precipitable_water": 40.0, "rain_last_6h": 5.0, "rain_last_24h": 15.0}},
    {"id": "JK-SR", "name": "Srinagar", "state": "Jammu & Kashmir", "lat": 34.0837, "lon": 74.7973, "features": {"gfs_rain_mm": 10.0, "gfs_mslp": 1010.0, "gfs_t2m": 15.0, "gfs_rh2m": 60.0, "gfs_u10": 1.0, "gfs_v10": 0.5, "gfs_u850": 2.0, "gfs_v850": -3.0, "gfs_cape": 100.0, "gfs_precipitable_water": 20.0, "rain_last_6h": 2.0, "rain_last_24h": 5.0}},
    {"id": "BR-PA", "name": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "features": {"gfs_rain_mm": 15.0, "gfs_mslp": 1005.0, "gfs_t2m": 29.0, "gfs_rh2m": 75.0, "gfs_u10": 2.0, "gfs_v10": 1.0, "gfs_u850": 5.0, "gfs_v850": 1.5, "gfs_cape": 450.0, "gfs_precipitable_water": 48.0, "rain_last_6h": 4.0, "rain_last_24h": 10.0}},
    {"id": "OR-BH", "name": "Bhubaneswar", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "features": {"gfs_rain_mm": 45.0, "gfs_mslp": 997.0, "gfs_t2m": 28.5, "gfs_rh2m": 86.0, "gfs_u10": 4.5, "gfs_v10": 2.5, "gfs_u850": 10.0, "gfs_v850": 3.5, "gfs_cape": 1100.0, "gfs_precipitable_water": 63.0, "rain_last_6h": 18.0, "rain_last_24h": 45.0}},
    {"id": "TN-CH", "name": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "features": {"gfs_rain_mm": 12.0, "gfs_mslp": 1007.0, "gfs_t2m": 31.0, "gfs_rh2m": 72.0, "gfs_u10": 3.0, "gfs_v10": 1.5, "gfs_u850": 6.0, "gfs_v850": 2.0, "gfs_cape": 550.0, "gfs_precipitable_water": 45.0, "rain_last_6h": 2.0, "rain_last_24h": 8.0}},
    {"id": "GJ-AH", "name": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "features": {"gfs_rain_mm": 5.0, "gfs_mslp": 1006.0, "gfs_t2m": 32.0, "gfs_rh2m": 65.0, "gfs_u10": 2.5, "gfs_v10": 1.0, "gfs_u850": 4.0, "gfs_v850": 1.0, "gfs_cape": 350.0, "gfs_precipitable_water": 35.0, "rain_last_6h": 0.0, "rain_last_24h": 2.0}},
    {"id": "RJ-JA", "name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "features": {"gfs_rain_mm": 2.0, "gfs_mslp": 1008.0, "gfs_t2m": 34.0, "gfs_rh2m": 50.0, "gfs_u10": 2.0, "gfs_v10": 0.5, "gfs_u850": 3.0, "gfs_v850": 0.5, "gfs_cape": 200.0, "gfs_precipitable_water": 25.0, "rain_last_6h": 0.0, "rain_last_24h": 0.0}},
]

@router.get("/districts")
async def get_districts():
    """Get districts with their mock forecasts from the prototype ML pipeline."""
    districts = []
    
    for d in DEMO_DISTRICTS:
        features = dict(d["features"])
        features["latitude"] = d["lat"]
        features["longitude"] = d["lon"]
        
        result = ml_service.run_forecast(features)
        
        districts.append({
            "id": d["id"],
            "name": d["name"],
            "state": d["state"],
            "lat": d["lat"],
            "lon": d["lon"],
            "elevation_m": 0,
            "avg_annual_rainfall_mm": 0,
            "raw_forecast_mm": result["raw_forecast_mm"],
            "corrected_forecast_mm": result["corrected_forecast_mm"],
            "heavy_rain_probability": result["heavy_rain_probability"],
            "very_heavy_probability": result["very_heavy_probability"],
            "extreme_probability": result["extreme_probability"],
            "regime": result["regime"],
            "regime_confidence": result["regime_confidence"],
            "regime_probabilities": result["regime_probabilities"],
            "prediction_mode": result["prediction_mode"],
            "fallback_regimes": result["fallback_regimes"],
            "data_mode": result["data_mode"],
            "model_version": result["model_version"],
            "risk_level": result["risk_level"]
        })
        
    return {
        "districts": districts,
        "metadata": {
            "source": f"{ml_service.data_mode} / {ml_service.model_version}",
            "model_version": ml_service.model_version,
            "data_mode": ml_service.data_mode,
            "count": len(districts),
        }
    }
