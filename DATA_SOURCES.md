# ANVAYA data-source and provenance register

This document is the human-readable companion to `config/data_sources.yaml`,
`config/feature_registry.yaml`, `data/source_manifest.json`, and
`data/results/source_health.json`. A source being listed here does not mean it
has been downloaded, authenticated, or scientifically validated.

## Current source state (2026-10-04)

The source-health probe found three local caches and nine endpoints temporarily
unavailable from this execution environment. The network failures were socket
permission failures (`WinError 10013`), not evidence that the upstream datasets
do not exist. No external source was silently substituted.

| Source | Intended role | Current state | Used in current scientific artifacts? |
|---|---|---|---|
| NOAA/NCEP GFS 0.25° | NWP baseline and predictors | One UCAR historical India subset validated; earlier local episode cache provenance remains unverified | Yes, local prototype input; test subset not yet added to scientific training data |
| NOAA/NCEP GEFS 0.25° | Ensemble spread/uncertainty | Temporarily unavailable | No |
| IMD RF25 0.25° daily gridded rainfall | Grid-matched target | Official 1901–2024 product identified; annual 2020–2024 NetCDF files audited; legacy episode files remain unverified | RF25 validated for one cached GFS comparison; legacy prototype targets remain unverified |
| NCMRWF MERA | Independent reference | Temporarily unavailable | No |
| NASA GPM IMERG V07 | Satellite cross-check | Temporarily unavailable | No |
| ISRO MOSDAC INSAT QPE | High-resolution reference | Temporarily unavailable | No |
| ECMWF ERA5 | Retrospective context/diagnostics | Temporarily unavailable | No |
| NASA MERRA-2 | Redundant retrospective context | Temporarily unavailable | No |
| SRTM/NASADEM | Static terrain features | Temporarily unavailable | No; current features retain documented proxies |
| NOAA IBTrACS | Retrospective storm/depression labels | Temporarily unavailable; no file cached | No |
| District boundaries | Spatial aggregation | Temporarily unavailable/source selection pending | No |

The exact machine-readable counts and errors are in
`data/results/source_health.json`. File hashes and local file inventory are in
`data/source_manifest.json`. The manifest records legacy GFS/IMD episode files
as local and unverified. The separately listed RF25 annual files match the
official IMD catalogue's product identity and metadata; the user supplied them
as provenance-verified. Their retrieval receipt/URL and semantic release
identifier are not embedded in the files.

## Intended external source references

- GFS and GEFS: NOAA/NCEP NOMADS filter endpoints for the 0.25° products:
  [GFS](https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl) and
  [GEFS](https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl).
- IMD rainfall: the product identity, release, access method, and retrieval
  timestamp must be captured for every future download. The registry points to
  the [IMD Climate Services reference](https://mausam.imd.gov.in/imd_latest/contents/pdf/pubbrochures/Climate%20Services.pdf).
- IBTrACS: retrospective storm-track context only. The official archive and
  version information are maintained by [NOAA NCEI](https://www.ncei.noaa.gov/products/international-best-track-archive).
- Reanalysis and satellite products: see the URLs in
  `config/data_sources.yaml`; credentials and product versions are mandatory
  before ingestion.

## UCAR historical GFS subset retrieval (one case validated)

The historical GFS source remains under the existing `gfs_0p25` registry
entry. For archived cases use the NCAR GDEX d084001 THREDDS NetCDF Subset
Service (NCSS), not the global AWS objects. The NCSS dataset URL is
`https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001/{YYYY}/{YYYYMMDD}/gfs.0p25.{YYYYMMDD}{CC}.f{FFF}.grib2`;
the associated `dataset.xml` is fetched first and must confirm the forecast
reference time, lead, longitude convention, GRIB2 parameter identifiers,
level types, units, and availability of all requested fields. The longitude
axis is `0..359.75 degrees_east`, so India bounds are sent as positive
`west=68`, `east=97` (not negative-west longitudes). The endpoint advertises
NetCDF-3 output (`accept=netcdf3`) for this dataset.

The implemented provider is separate from the existing `GFSProvider`:
`ml/data/ucar_gfs_subset.py:UCARGFSSSubsetProvider`. It makes strict requests
for registry-listed variables, refuses missing/unexpected fields, and has no
synthetic fallback. The one-case command is
`backend/venv/Scripts/python.exe scripts/ingest_ucar_gfs_subset.py`; its
defaults select only 2024-07-15 00Z f024. The NCSS service requires requested
grid variables to share vertical levels/time coordinates. For the selected
case it explicitly rejected co-requesting 2 m temperature and 2 m relative
humidity because their vertical-level sets differ. Therefore this tested
implementation makes ten single-variable requests, each with its exact
`vertCoord` where applicable, and merges only after validating the individual
responses. It does not contain a date-range loop.

### Validated case

| Item | Result |
|---|---|
| Forecast | Init `2024-07-15T00:00Z`, valid `2024-07-16T00:00Z`, lead 24 h |
| Requested bbox | 8–37°N, 68–97°E; returned grid-center bounds match exactly |
| Returned grid | 117 × 117; no global 90°N/90°S grid |
| NetCDF file | `data/raw/gfs/test_subset/gfs_0p25_20240715_00_f024_india.nc` (616,601 bytes; 0.588 MiB) |
| Manifest | `data/raw/manifests/gfs_test_subset.json`, including source metadata, all ten request URLs, variable mappings, dimensions, validation, and SHA-256 |
| Checksum | `590c3f6e497a42a5000ecda9d134c077261f4aa89d5e0310afffa653491b68bb` |
| Feature completeness | All 10 registered raw GFS fields retrieved; no missing-variable fallback |
| Rainfall check | 0–144.625 kg m⁻², 13,689 finite grid cells; accumulation bounds are 2024-07-15 18Z–2024-07-16 00Z (6 h) |

`kg m-2` precipitation is numerically equivalent to millimetres of liquid
water. The f024 `Total_precipitation` message is the 18Z–00Z six-hour interval,
not a 24-hour accumulation. It is now represented as
`gfs_apcp_interval_mm`; it must never be relabeled as `gfs_rain_mm` or used as
the primary predictor against an IMD daily target.

The output opens with xarray's `netcdf4` engine (the engine also reads the
endpoint's NetCDF-3 format). Estimated storage is 616,601 bytes for this one
full India-domain case. Extrapolating its stored size to 122 June–September
days × 5 years × 4 cycles gives approximately **1.505 decimal GB** for
2021–2025. The ten NCSS responses total 583,924 bytes for the sample, giving
a rough transfer extrapolation of **1.425 decimal GB**. These are one-case
estimates, not a measured archive total, and assume every cycle exists with
similar output size. No multi-year retrieval has been started. Review
precipitation-window alignment and the sample size before separately
authorizing any expansion.

## Ingestion contract

1. For ordinary full-object sources, use `scripts/download_data.py` with an
   explicit URL and, when known, an expected SHA-256 checksum; downloads are
   resumable and never overwrite without `--force`. For historical GFS India
   subsets, use `UCARGFSSSubsetProvider` so THREDDS applies variable/time/level/
   bbox filtering before delivery; keep each run to an explicitly selected
   forecast and validate its manifest before considering further dates.
2. Record source version, retrieval time, URL, checksum, spatial/temporal
   subset, and access/license notes in the source manifest.
3. Keep retrospective sources separate from live predictors. ERA5, MERRA-2,
   MERA, IMERG, INSAT, and IBTrACS cannot become operational features merely
   because they are available after an event.
4. Do not merge an independent reference into the IMD target column. It is a
   validation/reference stream unless a separate scientific decision and
   lineage record says otherwise.
5. Re-run `scripts/validate_data.py --operational` after adding features. The
   target `observed_rain_mm` and any future-observation field must fail the
   operational registry check.

## Current limitation

The only processed data available in this checkpoint are July episodes from
the local GFS/IMD cache. June, August, September, and non-monsoon windows were
preflighted but not fabricated when their paired source files were absent.

## GFS APCP alignment validation (2026-10-04)

The controlling IMD daily window for date D is `[D-1 03:00Z, D 03:00Z)`.
GFS APCP is GRIB2 parameter 0-1-8 at level type 1, statistical process
`Accumulation`, source units `kg.m-2` (equal numerically to mm water
equivalent). Forecast endpoint time alone is insufficient: the original
2024-07-15 00Z f024 record has explicit bounds 2024-07-15 18Z–2024-07-16 00Z,
only six hours. The 00Z cycle must use lead 27h to end at the next-day IMD 03Z
boundary; f024 is not the daily window.

NCSS metadata and NetCDF time bounds were inspected for each downloaded lead.
At the tested 00Z cycles the archive provides f003 00–03Z and f006 00–06Z,
then mixed records such as f009 06–09Z / f012 06–12Z,
f015 12–15Z / f018 12–18Z, and f021 18–21Z / f024 18–00Z. Paired same-start
accumulations are nested, not disjoint. `build_24h_precipitation` subtracts the
shorter from the longer only when metadata proves the same start and exact
nested endpoints, yielding disjoint three-hour increments; it then checks
gap-free, overlap-free coverage and sums exactly 24 hours. Unsupported
overlap, metadata gaps, non-forecast/observation data, units other than water
equivalent, or a forecast reference-time mismatch cause a hard failure.

Four bounded real cases were retrieved through the India-box NCSS service (no
global objects and no archive-scale download):

| Forecast initialization | Target interval (UTC) | End lead | GFS daily domain mean | Local IMD comparison |
|---|---|---:|---:|---|
| 2024-07-15 00Z | Jul 15 03Z–Jul 16 03Z | 27h | 10.316 mm | 29.989 local values; units absent in file |
| 2024-07-16 00Z | Jul 16 03Z–Jul 17 03Z | 27h | 10.086 mm | 30.002 local values; units absent in file |
| 2024-08-15 00Z | Aug 15 03Z–Aug 16 03Z | 27h | 6.442 mm | No local IMD record for Aug 16 |
| 2024-07-15 06Z | Jul 16 03Z–Jul 17 03Z | 45h | 9.746 mm | 30.002 local values; units absent in file |

The July local IMD values are from the existing 2024 validation NetCDF at
59×59 shared grid centers; the GFS subset is 117×117. The files have no units
attribute and are documented as provenance-unverified. They are reported for
traceability, not as authoritative numerical validation. The August comparison
is not available locally; no other date was substituted. All four GFS time
windows, spatial bounds (8–37°N, 68–97°E), GFS units, non-overlap, and
valid-time/end-lead checks passed. Each saved four-case test file is about
569 KB and contains raw metadata-bounded records plus the derived
`gfs_daily_rain_mm` field. Per-request URLs, variable names, time bounds,
metadata, checksums, comparisons, and limitations are in
`data/raw/manifests/gfs_precip_alignment_test.json`.

The feature registry now separates `gfs_apcp_interval_mm` (disabled as a live
model feature) from `gfs_daily_rain_mm` (derived, enabled only after exact
alignment). Existing processed datasets and trained artifacts still using
`gfs_rain_mm` have not been rebuilt and are not cleared for renewed scientific
training. Multi-year ingestion remains stopped.

## Phase 3D JJAS expansion inventory (2026-10-04)

The requested 2021–2025 JJAS schedule contains 2,440 date/cycle slots for
00Z/06Z/12Z/18Z. The UCAR THREDDS subset route is available and was validated
in the prior phase, but this phase made **zero** GFS requests. The local source
manifest marks existing raw GFS/IMD cache files `present_unverified`; the IMD
rain variable has no units metadata and daily-midnight time coordinates without
time bounds. The dataset builder also contains a synthetic demo data path, so
an IMD label or filename alone is not accepted as observational provenance.

At the original Phase 3D inventory (before the annual RF25 files arrived), six
local IMD-labelled episode files covered 48 unique July dates but were
unverified. That historical preflight found 192 `TARGET_UNVERIFIED` and 2,248
`TARGET_UNAVAILABLE` cycle slots. No full-grid GRIB2 object was downloaded,
no aligned dataset was written, and the held-out partition was not modified.
The subsequent annual RF25 audit below supersedes that target-availability
finding for 2020–2024. Historical Phase 3D accounting remains in
`data/results/gfs_bulk_ingestion_report.json` and
`data/results/gfs_bulk_ingestion_summary.csv`.

The target-source prerequisite is now met for 2020–2024 by the annual RF25
files audited below. Before any future coverage expansion, build and validate
the normalized aligned splits from `gfs_daily_rain_mm`; separately obtain a
provenance-verified 2025 target without altering the held-out partition. The
legacy processed partitions still contain `gfs_rain_mm`; they are not a valid
Phase 3D aligned dataset.

## IMD RF25 annual rainfall audit (2026-10-04)

The official product is IMD Climate Research and Services, Pune's *New High
Spatial Resolution (0.25 x 0.25 degree) Long Period Daily Gridded Rainfall Data
Set Over India*, catalogue coverage 1901–2024. The local annual files cover
2020–2024. The archive does not assign a semantic release number; NetCDF
conversion history attributes report FERRET V6.82 on 20-Feb-2026 for 2020–2023
and 9-Apr-2026 for 2024. The product page documents 0.25° daily rainfall in mm,
135×129 grid points, longitude 66.5–100°E and latitude 6.5–38.5°N. See the
[official IMD NetCDF catalogue](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html).

Annual file inventory: `RF25_ind2020_rfp25.nc` (366 days),
`RF25_ind2021_rfp25.nc` (365), `RF25_ind2022_rfp25.nc` (365),
`RF25_ind2023_rfp25.nc` (365), and `RF25_ind2024_rfp25.nc` (366). All are
NetCDF classic (CDF-1), with `RAINFALL(TIME,LATITUDE,LONGITUDE)`, units mm,
`-999` fill/missing value, ascending latitude 6.5–38.5°N, ascending eastward
longitude 66.5–100°E, and 0.25° spacing. `TIME` is daily and labeled at
00:00Z; there are no per-record time bounds. Calendar is explicitly
`GREGORIAN` in 2024 and is not encoded in 2020–2023. Per-file sizes and
SHA-256 digests are in the RF25 source entry in `data/source_manifest.json`
and in `data/results/imd_rf25_target_audit.json`.

The time label is not treated as the accumulation endpoint by itself. IMD's
[official API reference](https://api.imd.gov.in/public/api_reference.html)
defines daily `Past_24_hrs_Rainfall` for observation date D as 08:30 IST on
D−1 through 08:30 IST on D. Published product methodology describes the IMD
gridded gauge rainfall as 24-hour rainfall ending 08:30 IST (03:00 UTC); see
[Kumar et al. (2022)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002595).
Thus the RF25 value labeled date D represents `[D−1 03:00Z, D 03:00Z)`,
matching the target window used by `ml/data/gfs_precip.py`. The NetCDF does not
carry bounds; the physical-period assignment is sourced from IMD's daily
observation convention and product methodology, not inferred from midnight.

For the already-cached GFS cycle 2024-07-15 00Z, the pipeline rebuilt
`gfs_daily_rain_mm` from metadata-bounded APCP records for 2024-07-15 03Z to
2024-07-16 03Z and matched it to `TIME=2024-07-16` in the 2024 RF25 file. The
grids are both 117×117 at 0.25° over 8–37°N, 68–97°E after reversing IMD's
ascending latitude axis to match GFS's descending axis. Longitudes match
exactly in positive-east/0–360 convention. Units are mm on both sides (GFS
source APCP is kg m⁻² water equivalent). There are 4,958 common non-missing
grid cells. Cellwise diagnostic means are 11.876 mm GFS and 10.403 mm IMD;
these are a single-case data audit, not model verification skill.

The seven older `imd_*` episode files remain separately classified as
unverified: they lack rainfall units and time bounds, and several use a 0.5° grid.
They were not used for this comparison. This audit does not rebuild
train/validation/test data; no GFS downloads, training, or 2025 test changes
were made. At that earlier single-case checkpoint readiness remained
`SCIENCE_READY = NO` until normalized aligned splits were built. That status is
superseded by the full verified dataset and readiness result in the Phase 3E
section below.

## Phase 3E verified JJAS dataset build checkpoint (2026-10-04)

The resumable builder targets official local IMD RF25 annual files for
2020–2024 and requests only registered GFS predictors from UCAR GDEX d084001
through geographic/time/variable/level NCSS subsets. Target date D uses the
D−1 00Z forecast at +27h; metadata-bounded APCP intervals are combined without
overlap to cover exactly `[D−1 03:00Z, D 03:00Z)`. The selected domain is
8–37°N, 68–97°E in positive-east 0–360 longitude convention (117×117 at
0.25°). Split dates are 2020–2022 train, 2023 validation, and 2024 test. The
legacy 2025 test artifact is outside this build.

The 40-case bounded sample passed: 24 train dates, eight validation dates, and
eight test dates, with 4,958 paired valid RF25/GFS grid cells per date
(198,320 rows total). All registered GFS predictors, exact daily intervals,
spatial grid nodes, no-missing/no-duplicate rows, and prospective timing were
validated. All seven weak-label regimes occur in every split; BREAK_MONSOON
and WESTERN_DISTURBANCE occur on every sample date. LOW_DEPRESSION appears on
10 train, 2 validation, and 4 test sample dates; full-period distinct-date
adequacy remains to be assessed. The sample report, split checksums, and
extreme/regime counts are in
`data/results/verified_dataset_sample_report.json` and
`data/results/verified_dataset_sample_summary.csv`.

The full 610-date 2020–2024 JJAS build completed with zero unresolved case
failures. It reused checksum-verified cached cases and fetched only missing
dates through this same bounded UCAR NCSS path. Final verified partitions
contain 366 train dates (2020–2022), 122 validation dates (2023), and 122 test
dates (2024), with 4,958 exact paired grid cells per date. The full case
inventory/checksums and partition hashes are recorded in
`data/results/verified_dataset_report.json`; live completion is recorded in
`data/raw/manifests/verified_jjas_2020_2024/full_progress.json`.

The UCAR subset reader now includes a deterministic CF-time decoding path for
standard-calendar coordinates encoded as e.g. `Hour since ...` when xarray's
optional cftime decoder is unavailable. It decodes the numeric coordinate and
its bounds using netCDF4 while retaining source units/calendar in attributes;
it does not relax forecast-time or APCP-bound validation. This path was tested
against the 2024-09-26 850-hPa wind response. The source-variable mismatch
path now reports the exact missing/unexpected fields, response URL, byte count,
and SHA-256.

The full-data readiness gate returned `SCIENCE_READY = YES`; it evaluated only
the new verified partitions. No model training has started. The protected
legacy 2025 test artifact remains outside the dataset and byte-identical to
SHA-256 `c12585ab1882014bc5005767fc88c4a6270d11d80f5d043d9797c99574e8f5e6`.

## Phase 3G operational source-fusion path (2026-10-04)

`GET /api/operational/forecast?latitude=...&longitude=...` is an explicitly
separate `LIVE_OPERATIONAL_MODE`. It is point-based within 8–37°N, 68–97°E.
At the Phase 3G checkpoint there was no selected district polygon source;
Phase 4 superseded that limitation with sourced NWIC geometry and a separate
area-weighted district endpoint, while this point forecast remains point-level.
`ml/operational/risk.py` implements GeoJSON Polygon/MultiPolygon point
selection and risk aggregation for verified polygons plus multiple forecast
points; it reports the maximum cell probability, not a calibrated district
area probability. Existing scientific
routes, verified parquet splits, 2024 metrics/predictions, frozen model files,
and the legacy 2025 test artifact are not changed by this endpoint. Existing
source IDs remain in `config/data_sources.yaml`; Phase 3G adds operational
access metadata and the separate `config/operational.yaml`.

The live model path uses explicit-run APIs delivered by Open-Meteo. Open-Meteo
is the API delivery/interpolation layer, not the originating weather agency:
responses preserve both the delivery provider and upstream NOAA/ECMWF model
identity. NOAA GFS uses an explicit 00Z single run and lead +27 h, matching the
Phase 3F valid-time convention. Rainfall is derived from 24 returned
preceding-hour amounts only after confirming the exact UTC interval. GEFS
spread is computed across returned member series; if explicit members or full
window coverage are absent, no synthetic spread is returned. ECMWF IFS and
AIFS are requested as distinct model selectors for the same run and valid
window. See the [GFS API documentation](https://open-meteo.com/en/docs/gfs-api),
[Single Runs API](https://open-meteo.com/en/docs/single-runs-api),
[GEFS Ensemble API](https://open-meteo.com/en/docs/ensemble-api), and
[ECMWF API](https://open-meteo.com/en/docs/ecmwf-api).

Consensus groups NOAA GFS/GEFS together and ECMWF IFS/AIFS together so
multiple products from one provider family do not receive extra family weight.
It only averages precipitation totals if all contributing independent model
families have the same valid time and accumulation bounds. Between-family
spread and source coverage drive an explicitly uncalibrated stability score;
it is not a calibrated forecast-skill estimate.

The frozen Phase 3F model is loaded from
`models/scientific/phase3f/anvaya_moe.joblib` and
`multi_threshold_probabilities.joblib`; its schema/version must match before
inference. `rain_last_24h` is taken from a second, earlier explicit GFS 00Z
run whose +27 h accumulation window matches the frozen antecedent time window.
This is a real, time-matched forecast proxy, but Phase 3F was trained with
observed IMD RF25 rainfall for that field. The source shift is reported and
has not been validated or scored; if this prior run is missing, ANVAYA
probabilities return unavailable. NASA IMERG is requested after model
inference as observation/context and is not a Phase 3F input. The IMERG
adapter requests 48 half-hour point subsets from GES DISC OPeNDAP, applies the
nominal Early Run latency, checks units/coverage, and requires Earthdata
credentials. It never downloads global granules. NASA documents the Early,
Late, and Final latencies on the [IMERG product page](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHHE_07/summary).

Local IMD RF25 files remain `REFERENCE` (2020–2024), not current observations
in 2026. They are never used as live predictors or modified by this endpoint.
Provider states are reported as `LIVE`, `REFERENCE`, `DEMO`, or `UNAVAILABLE`.
Timeouts/errors stay source-local; cached responses have a configurable
600-second TTL. Warning levels use the configurable 0.5 probability threshold
in `config/operational.yaml`; these are ANVAYA operational rules, not official
IMD warning levels. No result from Phase 3G is evidence of lower historical
RMSE or improved scientific performance.

Offline API smoke suite: `scripts/test_operational_api.py` (12 tests). It uses
mocked provider responses to exercise exact-window GFS aggregation,
returned-member GEFS spread, partial ECMWF availability, accumulation-window
mismatch rejection, missing NASA credentials, polygon selection, model-before-IMERG
ordering, geographic validation, the frozen-model endpoint path, and all-provider
failure fallback. A passing offline smoke test does not
assert live third-party endpoint availability or NASA credential validity.
After integration, SHA-256 spot checks matched the recorded manifests for
`train_verified.parquet`, `validation_verified.parquet`, `test_verified.parquet`,
the protected legacy `test.parquet` (2025), and the two loaded Phase 3F inference
bundles. The external provider probe did not yield a usable response in this
execution, so network/API availability remains unconfirmed.

Deployment configuration: store `NASA_EARTHDATA_TOKEN` in a secret manager (or
use `NASA_EARTHDATA_USERNAME` plus `NASA_EARTHDATA_PASSWORD`); do not commit
credentials. `ANVAYA_OPERATIONAL_TIMEOUT_SECONDS`,
`ANVAYA_GLOBAL_RUN_LATENCY_HOURS`, the four `ANVAYA_*_MODEL` selector overrides,
and `ANVAYA_WARNING_PROBABILITY_THRESHOLD` are optional environment overrides.
Cache TTL and IMERG total deadline are in `config/operational.yaml`. Confirm
Open-Meteo service terms/limits and NASA authorization in the deployment
environment before enabling live traffic.

## District boundary reference geometry

Phase 4 uses the public [NWIC India district boundary resource](https://www.nwdp.nwic.gov.in/dataset/district-boundary), downloaded on 2026-10-04. The portal listed 733 features and a last update of 2025-05-05. The portal metadata names the Geological Survey of India (GSI) as producer, while the feature-level `src_agency` attribute says Survey of India (SOI); both attributions are preserved rather than silently reconciled. The source archive is `data/geospatial/source/district_nwic_geojson.zip` (SHA-256 `44c734cc72139f2447dcebfe2791cac862dc5ba265e158912d797cf3410d5c37`).

The source GeoJSON declares EPSG:7755 (WGS 84 / India NSF LCC, metres). `scripts/prepare_district_boundaries.py` transforms coordinates to WGS 84 / EPSG:4326 for RFC 7946 GeoJSON. It repairs 13 invalid source geometries with Shapely `make_valid` and polygon-only union; the acquired archive remains unchanged and repaired feature IDs are recorded in `data/geospatial/district_boundaries_metadata.json`. The full-resolution output is used for lookup and analysis. A separate display-only file is topology-preserving simplified by 500 m in the source CRS and is never used for analysis. Full and display output SHA-256 checksums are recorded in the same metadata file and `data/source_manifest.json`.

District operational summaries intersect the polygons with the supported GFS domain (8–37°N, 68–97°E) and 0.25° cells, project intersections to equal-area EPSG:6933, and aggregate at most nine spatially stratified point forecasts with represented-area weights. Forecast values are suppressed when live forecast coverage is below 75%; districts outside the forecast domain remain unavailable. This sparse area-weighted sampling is not a complete district grid or an official warning. The boundary vintage is 2025 and may omit later administrative changes. No per-resource license text was present in the portal metadata at acquisition; attribution is retained and redistribution terms must be confirmed with NWIC.
