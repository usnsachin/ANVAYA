# ANVAYA frontend

## Local development

Start the FastAPI backend in a first PowerShell terminal from the repository root:

```powershell
$env:FRONTEND_URL = "http://localhost:3000"
$env:ENVIRONMENT = "development"
backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Then start the frontend in a second PowerShell terminal:

```powershell
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

For local production-mode testing, set `ANVAYA_BACKEND_ORIGIN` to
`http://127.0.0.1:8000`, run `npm run build`, and then
`npm run start -- --hostname 127.0.0.1 --port 3000`. Development defaults the
server-side proxy to that local backend; production has no localhost fallback.
For Render, set `NEXT_PUBLIC_API_URL` to the deployed FastAPI service origin
without `/api`; the browser then calls that API directly and the backend CORS
allowlist must include the deployed frontend origin. Rebuild the frontend after
changing `NEXT_PUBLIC_API_URL`.

## Keyless map and location search

Google Maps is not required. The interactive map uses MapLibre GL JS with the
keyless OpenFreeMap styles backed by OpenStreetMap data. MapLibre displays the
provider and map-data attribution, including **© OpenStreetMap contributors**.
Only the current viewport's normal map resources are requested; the app does
not prefetch or bulk-cache geographic areas. The public map service is provided
as-is and does not offer an availability SLA, so production deployments with
uptime requirements should evaluate self-hosting or a contracted tile service.

Location search uses the Open-Meteo Geocoding API without a key. Recent query
results are cached in browser local storage for up to 24 hours. If geocoding is
unavailable, the local district directory, manual district selection and
browser-native **Locate Me** remain available. Locate Me asks the browser for
permission only after the user activates it; it does not call Google's
Geolocation API.

Available map styles are Standard, Light and Dark. Satellite imagery is
unavailable from the configured credential-free/open source and is not
simulated. Forecast coloring and warning colors are ANVAYA visualizations,
not official IMD warning classifications. Unqueried districts remain neutral.

## Checks

```powershell
npm run lint
npm run build
```
