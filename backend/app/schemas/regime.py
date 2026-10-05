from pydantic import BaseModel
from typing import List, Dict, Any

class Regime(BaseModel):
    name: str
    description: str
    probability: float
    color: str

class RegimeResponse(BaseModel):
    current_regime: str
    regimes: List[Regime]
    metadata: Dict[str, Any]
