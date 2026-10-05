from pydantic import BaseModel
from typing import List, Dict, Any

class District(BaseModel):
    id: str
    name: str
    state: str
    lat: float
    lon: float
    elevation_m: int
    avg_annual_rainfall_mm: int

class DistrictResponse(BaseModel):
    districts: List[District]
    metadata: Dict[str, Any]
