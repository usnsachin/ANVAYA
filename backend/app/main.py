import os
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, forecast, districts, verification, regimes, operational

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Monsoon AI Backend...")
    yield
    logger.info("Shutting down Monsoon AI Backend...")

app = FastAPI(title="ANVAYA Backend", lifespan=lifespan)


def _cors_origins() -> list[str]:
    """Use an explicit frontend origin in production and a local default only in development."""
    environment = os.environ.get("ENVIRONMENT", "development").strip().casefold()
    configured = os.environ.get("FRONTEND_URL", "").strip()
    if not configured:
        if environment in {"production", "prod"}:
            raise RuntimeError("FRONTEND_URL must be configured when ENVIRONMENT=production")
        configured = "http://localhost:3000"
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins:
        raise RuntimeError("FRONTEND_URL must contain at least one origin")
    return origins


FRONTEND_URLS = _cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(districts.router, prefix="/api")
app.include_router(verification.router, prefix="/api")
app.include_router(regimes.router, prefix="/api")
app.include_router(operational.router, prefix="/api")
