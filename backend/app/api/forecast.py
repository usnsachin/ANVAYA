from fastapi import APIRouter
from datetime import datetime
from app.services.ml_service import ml_service
from app.api.districts import DEMO_DISTRICTS

router = APIRouter(tags=["forecast"])

@router.get("/forecast")
async def get_forecasts():
    """Get forecast predictions from the ML pipeline."""
    forecasts = []
    
    for d in DEMO_DISTRICTS:
        features = dict(d["features"])
        features["latitude"] = d["lat"]
        features["longitude"] = d["lon"]
        
        result = ml_service.run_forecast(features)
        
        forecasts.append({
            "district": d["name"],
            "date": "2024-07-15",
            "lat": d["lat"],
            "lon": d["lon"],
            "raw_forecast_mm": result["raw_forecast_mm"],
            "corrected_forecast_mm": result["corrected_forecast_mm"],
            "regime": result["regime"],
            "confidence": result["heavy_rain_probability"], # Used in UI as probability
            "heavy_probability": result["heavy_rain_probability"],
            "very_heavy_probability": result["very_heavy_probability"],
            "extreme_probability": result["extreme_probability"],
            "global_corrected_mm": result["global_corrected_mm"],
            "regime_probabilities": result["regime_probabilities"],
            "prediction_mode": result["prediction_mode"],
            "data_mode": result["data_mode"],
            "model_version": result["model_version"],
            "risk_level": result["risk_level"],
            "explanation": result["explanation"]
        })
        
    return {
        "forecasts": forecasts,
        "metadata": {
            "source": f"{ml_service.data_mode} / {ml_service.model_version}", 
            "generated_at": datetime.utcnow().isoformat() + "Z", 
            "model_version": ml_service.model_version
        }
    }
