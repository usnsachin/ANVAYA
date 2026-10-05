from pydantic import BaseModel
from typing import Dict, Any

class VerificationMetrics(BaseModel):
    rmse: float
    ets: float
    csi: float
    pod: float
    far: float
    fss: float

class VerificationResponse(BaseModel):
    metrics: VerificationMetrics
    period: str
    metadata: Dict[str, Any]
