# ANVAYA

## Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Smart India Hackathon 2026 — Problem Statement SIH26080**

**Team:** Samvidanveshan

ANVAYA is a decision-support prototype for India’s monsoon rainfall. It pairs
live numerical-weather-prediction (NWP) inputs with a frozen, regime-aware
scientific post-processing stack, rainfall probabilities, district context,
and a clear separation between live operations and historical verification.

## Project links

- **GitHub:** https://github.com/usnsachin/ANVAYA
- **Public prototype:** PENDING_RENDER_DEPLOYMENT
- **Backend API:** PENDING_RENDER_DEPLOYMENT

The URLs remain placeholders until an actual deployment is completed.

## 1. Problem statement

Heavy monsoon rainfall varies substantially by weather regime and location.
Raw numerical forecasts can have systematic errors, while a single global
correction may not represent active/break monsoon, coastal, orographic,
depression, and other rainfall regimes equally well. ANVAYA explores a
regime-aware post-processing approach and presents operational information at
locations and districts.

## 2. Solution overview

The frozen scientific experiment uses historical GFS predictors and verified
IMD gridded daily rainfall targets. Its Phase 3F model stack estimates corrected
rainfall and heavy-rain probabilities. The separate live operational layer can
query GFS, GEFS, ECMWF IFS/AIFS, and GPM IMERG context, then exposes forecast
source status, consensus/spread, district summaries, and warning categories.
Provider failure is represented as unavailable rather than replaced with
fabricated data.

The operational path is not part of the historical benchmark and does not
change or improve its reported metrics.

## 3. Architecture

    Browser (Next.js)
      ├─ MapLibre + OpenFreeMap/OpenStreetMap
      ├─ Open-Meteo location search
      └─ Forecast, district, alert, and verification views
              ├─ Local: same-origin /api through the Next.js proxy
              └─ Render: NEXT_PUBLIC_API_URL → FastAPI /api
                           ├─ operational provider adapters (LIVE/REFERENCE/UNAVAILABLE)
                           ├─ district risk aggregation
                           └─ frozen Phase 3F inference artifacts

    Scientific verification: frozen metrics and methodology, isolated from live APIs

## 4. Frontend

The frontend is in frontend/ and uses Next.js, TypeScript, React, Tailwind,
and MapLibre GL JS. The main product areas are Home (decision support), Map
(geographic exploration), Alerts (actionable warnings), Insights (scientific
graphs/verification), and Forecast (location forecasts).

Google Maps is not required. The keyless open map shows visible
**© OpenStreetMap contributors** attribution and requests tiles for the visible
viewport. Search uses the Open-Meteo Geocoding API; browser-native geolocation
is used for “Locate Me.” Satellite imagery is unavailable and is not
simulated. Map and risk colors are ANVAYA visualization categories, not
official IMD warning classifications.

## 5. Backend

The FastAPI application is in backend/, with entry point app.main:app and
API prefix /api. The deployment health endpoint is /api/health; frozen
verification is served at /api/verification. Operational endpoints are
fault-tolerant and label source provenance/status in their responses.

## 6. Scientific pipeline

The frozen Phase 3F experiment uses verified 0.25° GFS/IMD samples for
2020–2022 training, 2023 validation, and a held-out 2024 test partition.
The 2024 partition contains **604,876 samples across 122 JJAS target dates**.
Model files, result summaries, and a checksum manifest are retained for
inference/verification. The larger train/validation/test datasets are not
needed by the deployed API and are intentionally not stored in Git.

| System | RMSE (mm) | MAE (mm) | Mean bias (mm) |
| --- | ---: | ---: | ---: |
| Raw GFS | 16.669 | 7.830 | 0.495 |
| Global ML | 14.079 | 7.280 | 0.714 |
| ANVAYA | 14.106 | 7.277 | 0.636 |

ANVAYA does **not** beat Global ML overall: Global ML has lower RMSE and
stronger held-out heavy-rain categorical scores. The intended contribution is
regime-aware specialization, calibrated extreme-rain probability, geography-
aware delivery, and local decision support—not a claim of best overall
accuracy. /api/verification serves the frozen metric files; it does not
recompute or alter the benchmark.

## 7. Operational/live forecast path

LIVE_OPERATIONAL_MODE is separate from SCIENTIFIC_MODE. Depending on
availability, the operational layer queries NOAA GFS and GEFS, ECMWF IFS/AIFS
open data, and NASA GPM IMERG context. It preserves source labels and degrades
gracefully when an upstream provider is unavailable. IMERG is observation
context; it is not silently substituted into the frozen scientific target or
benchmark.

## 8. Data sources

- **Historical predictors:** NOAA/NCEP GFS 0.25°; the validated archive and
  precipitation interval handling are documented in DATA_SOURCES.md.
- **Historical target/reference:** provenance-verified IMD 0.25° gridded daily
  rainfall used for the frozen experiment.
- **Live model inputs:** NOAA GFS, NOAA GEFS, and ECMWF IFS/AIFS open data,
  subject to provider availability.
- **Observation context:** NASA GPM IMERG (optional credentials may be required
  for some access paths).
- **Map and search:** OpenFreeMap/OpenStreetMap tiles and Open-Meteo Geocoding.

Raw weather archives, downloaded forecast caches, and large processed
partitions are intentionally excluded. See DATA_SOURCES.md and
SCIENTIFIC_METHOD.md for provenance and methodology.

## 9. Local development

Use PowerShell from the repository root.

### Backend (terminal 1)

Create the project environment once if needed, install the declared
dependencies, then start FastAPI:

    py -3.13 -m venv backend/venv
    backend/venv/Scripts/python.exe -m pip install -r backend/requirements.txt
    $env:FRONTEND_URL = "http://localhost:3000"
    $env:ENVIRONMENT = "development"
    backend/venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

If backend/venv already exists, skip the first two commands.

### Frontend (terminal 2)

    cd frontend
    npm ci
    npm run dev -- --hostname 127.0.0.1 --port 3000

In local development, /api/* is proxied by Next.js to
http://127.0.0.1:8000/api/*; browser requests stay relative to the frontend.
The optional ANVAYA_BACKEND_ORIGIN setting in
frontend/.env.local.example overrides that server-side proxy target.

Open http://localhost:3000. The application does not start ingestion or
training as part of local startup.

## 10. Environment variables

| Variable | Service | Purpose |
| --- | --- | --- |
| NEXT_PUBLIC_API_URL | Render frontend | Public FastAPI origin, e.g. https://<api-service>.onrender.com; frontend adds /api. This is embedded at build time. |
| FRONTEND_URL | Render backend | Exact deployed frontend origin allowed by CORS, e.g. https://<frontend-service>.onrender.com. |
| ENVIRONMENT | Backend | Set to production on Render; production requires FRONTEND_URL. |
| ANVAYA_DISTRICT_BOUNDARY_PATH | Backend | Render uses the committed compact 733-district geometry. |
| ANVAYA_BACKEND_ORIGIN | Next.js server, optional | Server-side proxy destination for local/same-origin setups; no browser key or credential. |
| NASA_EARTHDATA_TOKEN | Backend, optional secret | Enables IMERG access where required; without it, the source may be unavailable. |

Do not commit .env, .env.local, tokens, passwords, or API keys. The frontend
API origin is public configuration, not a secret.

## 11–12. Deployment and Render

The repository contains a root render.yaml Blueprint for separate Render
Web Services. Next.js is rooted at frontend/. FastAPI stays rooted at the
repository level because its runtime requires sibling ml/, models/, config/,
and data/ files; PYTHONPATH=backend lets Uvicorn use the existing app.main:app
entry point from the repository root. Both services have automatic deploys
disabled. The
Blueprint does not deploy until you create it in Render.

Because Render assigns the public URLs, fill in FRONTEND_URL and
NEXT_PUBLIC_API_URL after the initial service creation. During bootstrap, use
the temporary reserved .invalid origins shown in DEPLOYMENT.md if Render
requires values before it can create the services; then replace them with the
actual service origins. After changing NEXT_PUBLIC_API_URL, redeploy the
frontend so Next.js rebuilds with the public value. See DEPLOYMENT.md for the
GitHub connection, exact Render steps, health checks, and troubleshooting.

The expected backend start command is:

    uvicorn app.main:app --host 0.0.0.0 --port $PORT

## 13. Important limitations

- Historical skill is established only on the frozen 2024 test partition; it
  does not prove future operational performance.
- Live provider availability, latency, source cadence, and optional Earthdata
  authentication can affect operational responses.
- District forecasts depend on a bounded sample/coverage policy and are not an
  exhaustive gridded district field or official IMD warning.
- The compact committed boundary file is about 4.8 MiB. The full-resolution
  114.87 MiB reference GeoJSON is excluded from Git; compact geometry affects
  boundary precision only, not science artifacts or metrics.
- OpenFreeMap’s public tile service has no availability SLA.
- Browser geolocation requires user permission and HTTPS (or localhost).
- No public Render deployment has been performed by this repository-preparation
  task.

## 14. Prototype usage

Open the application, search for an Indian city/district or select a supported
district, and review the forecast/source status and warning context. The Map
page is for geographic exploration, Alerts for warning summaries, and Insights
for the frozen verification results. Unqueried districts remain neutral; a
provider failure is shown as unavailable rather than as a fabricated forecast.

## 15. Project structure

    backend/                  FastAPI routes and services
    config/                   Feature, source, and operational configuration
    data/geospatial/           Compact runtime district boundaries and provenance
    data/results/              Frozen Phase 3F metrics/manifests for verification
    frontend/                 Next.js application, map, search, and product pages
    ml/                       Feature, operational, scientific inference code
    models/scientific/phase3f/ Frozen inference artifacts and checksum manifest
    scripts/                  Bounded tests and readiness checks
    render.yaml               Two-service Render Blueprint (manual deployment)
