from fastapi import APIRouter
from app.schemas.regime import RegimeResponse

router = APIRouter(tags=["regimes"])

@router.get("/regimes", response_model=RegimeResponse)
async def get_regimes():
    """Get active monsoon regime status."""
    return {
      "current_regime": "active",
      "regimes": [
        {"name": "active", "description": "Enhanced monsoon phase with above-normal rainfall", "probability": 0.55, "color": "#2563eb"},
        {"name": "moderate", "description": "Near-normal monsoon conditions", "probability": 0.30, "color": "#16a34a"},
        {"name": "break", "description": "Suppressed monsoon phase with below-normal rainfall", "probability": 0.15, "color": "#dc2626"}
      ],
      "metadata": {"source": "demo", "classification_method": "placeholder"}
    }
