from pydantic import BaseModel
from typing import Dict, List, Optional

class ForecastItem(BaseModel):
    district: str
    date: str
    raw_forecast_mm: float
    corrected_forecast_mm: float
    regime: str
    confidence: float
    very_heavy_probability: Optional[float] = None
    extreme_probability: Optional[float] = None
    global_corrected_mm: Optional[float] = None
    regime_probabilities: Optional[Dict[str, float]] = None
    prediction_mode: Optional[str] = None
    fallback_regimes: Optional[List[str]] = None
    data_mode: Optional[str] = None
    model_version: Optional[str] = None

class Metadata(BaseModel):
    source: str
    generated_at: str
    model_version: str
    data_mode: Optional[str] = None

class ForecastResponse(BaseModel):
    forecasts: List[ForecastItem]
    metadata: Metadata
