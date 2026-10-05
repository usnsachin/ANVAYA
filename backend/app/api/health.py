from fastapi import APIRouter
from datetime import datetime
from app.services.ml_service import ml_service

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """Return health status of the service."""
    model_ready = ml_service.is_ready()
    return {
        "status": "ok",
        "service": "monsoon-ai-backend",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_status": "ready" if model_ready else "error",
        "data_mode": ml_service.data_mode,
        "model_version": ml_service.model_version,
    }
