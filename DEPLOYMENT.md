# ANVAYA deployment guide — Render

ANVAYA is a monorepo with a Next.js frontend in frontend/ and a FastAPI
backend in backend/. The root render.yaml Blueprint defines separate Render
Web Services. The frontend uses rootDir: frontend. The backend remains rooted
at the repository root because its runtime also imports ml/ and reads models/,
config/, and data/ outside backend/; Render does not expose files outside a
service rootDir. Its PYTHONPATH=backend setting preserves the existing FastAPI
entry point. The Blueprint sets autoDeployTrigger: off;
adding the Blueprint does not deploy from this local preparation task, and
pushing commits will not auto-deploy these services.

No GitHub remote, Render service, or public URL is created by this document.

## 1. Publish the repository

1. Create or use the GitHub repository https://github.com/usnsachin/ANVAYA.
2. Push the main branch from the local project.
3. Confirm the repository contains the root render.yaml,
   frontend/package-lock.json, backend requirements, compact boundary file,
   frozen inference artifacts, and frozen verification summaries.
4. Confirm .env files, raw archives, processed training partitions, caches,
   and the full-resolution boundary source are absent from the repository.

## 2. Create the Render Blueprint

1. In the Render Dashboard, choose New → Blueprint and connect the existing
   usnsachin/ANVAYA repository and main branch.
2. Review the two services from render.yaml:
   - anvaya-api: Python, root directory repository root, build command
     pip install -r backend/requirements.txt, start command
     uvicorn app.main:app --host 0.0.0.0 --port $PORT, with PYTHONPATH=backend.
   - anvaya-frontend: Node, root directory frontend, build command
     npm ci && npm run build, start command npm start.
3. The Blueprint requests FRONTEND_URL and NEXT_PUBLIC_API_URL because the
   service URLs do not exist yet. If Render requires values during initial
   creation, enter temporary reserved origins:

       FRONTEND_URL=https://frontend-not-set.example.invalid
       NEXT_PUBLIC_API_URL=https://backend-not-set.example.invalid

   These .invalid origins are placeholders only, not service URLs. The first
   frontend can render its pages, but its API calls will not work until the
   variables are replaced with the actual service URLs. Do not publish these
   placeholder values as a working deployment.
4. Create the Blueprint and wait for both services to complete their initial
   deploys. This is a user-triggered deployment action in Render, not performed
   by this repository preparation.

## 3. Set the real service URLs

After Render assigns each service its onrender.com URL:

1. In anvaya-api → Environment, set FRONTEND_URL to the exact frontend
   origin, without a trailing slash, for example
   https://<actual-frontend-service>.onrender.com.
2. In anvaya-frontend → Environment, set NEXT_PUBLIC_API_URL to the
   FastAPI service origin, without /api or a trailing slash, for example
   https://<actual-api-service>.onrender.com.
3. Save both settings. Redeploy/restart the backend to apply CORS, and trigger a
   new frontend build/deploy: Next.js embeds NEXT_PUBLIC_API_URL into the
   client bundle at build time.
4. Do not guess Render URLs from the service names; copy the actual URLs shown
   in the dashboard.

The frontend appends /api to NEXT_PUBLIC_API_URL. Local development leaves
that variable unset and uses the relative /api path through Next.js’s
same-origin proxy.

## 4. Backend environment and health

The Blueprint sets:

- ENVIRONMENT=production
- ANVAYA_DISTRICT_BOUNDARY_PATH=data/geospatial/india_districts_map.geojson
- FRONTEND_URL (manual value described above)
- PYTHON_VERSION=3.14.6

NASA_EARTHDATA_TOKEN is optional and should only be added as a secret in the
Render Dashboard if needed. Without usable Earthdata credentials, IMERG may be
returned as unavailable; it is not fabricated or substituted into the
scientific benchmark.

The configured health check is /api/health. After deploy, verify these URLs
in a browser or with Invoke-WebRequest:

    https://<actual-api-service>.onrender.com/api/health
    https://<actual-api-service>.onrender.com/api/verification
    https://<actual-frontend-service>.onrender.com/
    https://<actual-frontend-service>.onrender.com/map
    https://<actual-frontend-service>.onrender.com/alerts
    https://<actual-frontend-service>.onrender.com/insights
    https://<actual-frontend-service>.onrender.com/forecast

Then use the deployed frontend to verify one bounded location/district request.
The browser makes cross-origin API calls to the configured FastAPI origin, and
the backend CORS allowlist is the exact FRONTEND_URL value.

## 5. Redeployments

Automatic deploys are disabled in the Blueprint. To release a later commit,
review it and use Manual Deploy → Deploy latest commit for the appropriate
Render service. The frontend must rebuild after changes to NEXT_PUBLIC_API_URL.
The backend must restart after FRONTEND_URL changes.

## 6. Local checks before deployment

From the repository root in PowerShell:

    cd frontend
    npm ci
    npm run lint
    npm run build
    cd ..
    backend/venv/Scripts/python.exe scripts/test_phase4.py
    backend/venv/Scripts/python.exe scripts/test_operational_api.py

For local startup commands, see the README. These checks do not start data
ingestion, download weather archives, or train models.

## 7. Troubleshooting

- Frontend pages load but API calls fail: check that NEXT_PUBLIC_API_URL is the
  actual backend origin and that the frontend was rebuilt after changing it.
  The frontend adds /api automatically.
- Browser reports CORS errors: set backend FRONTEND_URL to the exact frontend
  origin including https://, with no path or trailing slash, then restart the
  backend.
- Backend health check fails immediately: confirm ENVIRONMENT=production,
  FRONTEND_URL is a non-empty valid origin, Render is using the repository
  root for the backend service,
  and the service start command uses $PORT.
- District endpoint cannot load geometry: confirm the compact
  data/geospatial/india_districts_map.geojson is in Git and
  ANVAYA_DISTRICT_BOUNDARY_PATH is set as shown in the Blueprint.
- Scientific model loading fails: confirm all files under
  models/scientific/phase3f/ and the root ml/ package were pushed. The release
  audit checks the Phase 3F SHA-256 manifest; do not retrain or replace these
  files to work around runtime issues.
- Live provider is unavailable: check the response’s source status and
  provider metadata. External provider failure does not change frozen
  verification results.

## Security and scientific boundaries

- Never commit .env, .env.local, credentials, API keys, or tokens.
- The deployed frontend URL and API URL are public configuration, not secrets.
- Raw weather archives, downloaded forecast caches, training partitions, and
  interrupted checkpoints are excluded from Git.
- Phase 3F model artifacts, frozen dataset checksums, benchmark values, and
  verification methodology must remain unchanged.
- Operational APIs do not claim to improve historical RMSE absent a new
  controlled experiment.
- Google Maps is not required.
