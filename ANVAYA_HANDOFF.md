# ANVAYA Project Handoff & Architecture Checkpoint

> **Project Title:** ANVAYA — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts  
> **Status:** Phase 4 district operational product implemented; Phase 3F benchmark remains frozen and unchanged
>
> **FINAL_STATUS:** `PHASE_4_DISTRICT_OPERATIONAL_PRODUCT_OFFLINE_TESTED`
> **Date:** October 5, 2026  
> **Workspace:** ANVAYA repository root

---

## 1. Project Purpose
ANVAYA resolves a fundamental meteorological limitation in Numerical Weather Prediction (NWP) during the Indian Summer Monsoon: global NWP models (specifically NOAA GFS 0.25°) exhibit state-dependent, regime-conditioned systematic errors that vary across synoptic setups (Active vs. Break phases, Offshore Trough coastal rains, Cyclonic Lows/Depressions, Himalayan and Western Ghats orographic uplift).

Rather than applying a uniform global bias correction or a hard categorical regime switch, ANVAYA implements:
1. **Probabilistic Weather-Regime Classification** ($P(\text{regime} \mid \mathbf{x})$) across 7 transparent meteorological regimes.
2. **Regime-Aware Mixture of Experts (MoE)** where expert predictions are soft-gated by regime posterior probabilities:
   $$\hat{y}_{\text{ANVAYA}} = \sum_{r} P(r \mid \mathbf{x}) \cdot f_r(\mathbf{x})$$
3. **Calibrated Multi-Threshold Heavy Rainfall Probabilities** ($\ge 64.5\text{ mm}$ Heavy, $\ge 115.6\text{ mm}$ Very Heavy, $\ge 204.5\text{ mm}$ Extremely Heavy).
4. **District-Level Aggregations & Explainability** for actionable disaster-preparedness intelligence.
5. **Rigorous Temporal Verification** comparing Raw GFS vs. Global ML Correction vs. ANVAYA MoE across meteorological metrics (CSI, ETS, POD, FAR, RMSE, FSS).

---

## 2. Phase 3B Alignment & Leakage Audit

### Physical Alignment Specification
* **IMD Daily Rainfall Truth:** Standard IMD 0.25° daily rainfall measures 24-hour rainfall accumulation ending at 08:30 IST (03:00 UTC) on date $D$. The true accumulation interval is $[D-1\text{ 03:00 UTC} \to D\text{ 03:00 UTC}]$.
* **GFS Forecast Initialization & Valid Time:** GFS cycles are 00/06/12/18Z; there is no standard 03Z cycle. A matching valid-time endpoint does not prove the APCP interval matches. For a 00Z initialization on D-1, the daily target ending D 03Z is at lead 27h, and every APCP record must be validated against its metadata bounds. The former `D-1 03Z + f024` assertion was incorrect and is retired.
* **Spatial Matching:** Standardized to the Indian meteorological domain ($8.0^\circ-37.0^\circ\text{N}$, $68.0^\circ-97.0^\circ\text{E}$) with coordinate alignment.

### Anti-Leakage Verification
* **Target Isolation:** `observed_rain_mm` is strictly isolated as the regression/classification ground-truth target $y$ and is **never** included in `FEATURE_SCHEMA` or `MINIMUM_FEATURES`. `gfs_apcp_interval_mm` is a raw bounded accumulation and is disabled in the live feature registry; only metadata-aligned `gfs_daily_rain_mm` may be the primary predictor against the IMD daily target.
* **Antecedent Predictors:** `rain_last_6h` and `rain_last_24h` strictly reflect prior observations from $T \le \text{init\_time}$ (the antecedent day $D-1$ observation cycle), ensuring zero look-ahead leakage.
* **Duplicate Prevention:** Evaluated on `(valid_time, latitude, longitude)` — exactly 0 duplicate rows across all splits.

---

## 3. Dataset Summary & Temporal Partitions

The locally assembled, externally unverified dataset spans multiple July
episodes across the temporal partitions below. It must not be described as an
independently verified multi-source scientific dataset:
* **Combined Real Dataset:** `data/processed/real_training_data.parquet` (194,936 rows, 32 variables)
* **Train Partition (2020–2023):** `data/processed/train.parquet` (118,354 rows, 60.7%)
  * Time range: `2020-07-05 03:00:00` to `2023-07-16 03:00:00` UTC
* **Validation Partition (2024):** `data/processed/validation.parquet` (38,291 rows, 19.6%)
  * Time range: `2024-07-10 03:00:00` to `2024-07-20 03:00:00` UTC
* **Test Partition (2025):** `data/processed/test.parquet` (38,291 rows, 19.6%)
  * Time range: `2025-07-12 03:00:00` to `2025-07-22 03:00:00` UTC

### Extreme Event Counts (Target `observed_rain_mm`)
* **Heavy Rainfall ($\ge 64.5\text{ mm}$):** 11,023 events (5.65%)
  * Train: 6,684 | Validation: 2,158 | Test: 2,181
* **Very Heavy Rainfall ($\ge 115.6\text{ mm}$):** 1,495 events (0.77%)
  * Train: 888 | Validation: 310 | Test: 297
* **Extremely Heavy Rainfall ($\ge 204.5\text{ mm}$):** 86 events (0.044%)
  * Train: 55 | Validation: 19 | Test: 12
*(Note: In the evaluated dataset and test period, raw GFS predicted 0 events $\ge 204.5\text{ mm}$ while observations contained 55 train, 19 validation, and 12 test events. This finding is scoped to these evaluated splits and should not be generalized beyond them.)*

### Data Quality Metrics
* Total records: 194,936
* Unique grid points: 3,481
* Missing cells: 0
* Duplicate coordinate-time rows: 0
* Negative rainfall values: 0
* Percentiles (mm/day): p25=21.48, p50=26.09, p75=31.14, p90=37.84, p95=71.45, p99=113.22, Max=291.82

---

## 4. Current Architecture & File Structure
```
monsoon-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry with lifespan & CORS
│   │   ├── api/
│   │   │   ├── health.py            # Dynamic model readiness status
│   │   │   ├── forecast.py          # ML inference across 15 demo districts
│   │   │   ├── districts.py         # District metadata + ML forecast aggregates
│   │   │   ├── verification.py      # 3-system comparison verification metrics
│   │   │   └── regimes.py           # Active regime status & descriptions
│   │   ├── services/
│   │   │   └── ml_service.py        # Centralized ML inference & explanation engine
│   │   ├── schemas/                 # Pydantic v2 validation models
│   │   └── models/                  # Data models
│   └── venv/                        # Python 3.14 virtual environment
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Entry screen ("Enter Anvaya")
│   │   │   ├── dashboard/page.tsx   # Main intelligence dashboard
│   │   │   ├── forecast/page.tsx    # Filterable forecast explorer
│   │   │   ├── districts/page.tsx   # Searchable district table
│   │   │   └── verification/page.tsx# Verification metrics & CSI bar chart
│   │   └── components/              # Layout, Map, Explainability, Forecast, Regime, Verification
├── ml/
│   ├── data/
│   │   ├── __init__.py
│   │   └── providers.py             # GFSProvider, IMDRainfallProvider, MERAProvider, DemoProvider
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py              # Dynamical, thermodynamic & topological proxies
│   ├── regime/
│   │   ├── labels.py                # Weak-supervision rules & metadata
│   │   ├── classifier.py            # Probabilistic RegimeClassifier (P(regime|x))
│   │   ├── model.py                 # Legacy RegimeModel wrapper
│   │   └── train.py                 # Regime training routine
│   ├── correction/
│   │   ├── moe.py                   # Soft-gated Mixture of Experts
│   │   ├── model.py                 # Legacy single-regime dispatcher
│   │   └── models/                  # 7 HistGradientBoostingRegressor models
│   ├── probability/
│   │   ├── multi_threshold.py       # Multi-threshold calibrated predictor (monotonic)
│   │   ├── model.py                 # Legacy single-threshold model wrapper
│   │   └── train.py                 # Heavy rain classifier training script
│   └── verification/
│       └── metrics.py               # Meteorological metrics (CSI, ETS, POD, FAR, RMSE, FSS)
├── data/
│   ├── raw/                         # Raw GFS & IMD NetCDF grids (2020-2025 episodes)
│   ├── processed/                   # Real aligned training data
│   │   ├── real_training_data.parquet (194,936 rows)
│   │   ├── real_training_data.csv
│   │   ├── train.parquet            (118,354 rows, 2020-2023)
│   │   ├── validation.parquet       (38,291 rows, 2024)
│   │   └── test.parquet             (38,291 rows, 2025)
│   ├── results/                     # Profiling and verification reports
│   │   ├── data_quality_report.json
│   │   └── data_quality_summary.csv
│   └── sample/                      # Sample datasets & metrics
├── scripts/
│   ├── build_real_dataset.py        # Configurable multi-period ingestion & temporal splitting
│   ├── profile_real_dataset.py      # Statistical profiling & data quality validator
│   ├── test_phase3a.py              # Phase 3A unit tests (7/7 passed)
│   ├── test_phase3b.py              # Phase 3B alignment & integrity tests (5/5 passed)
│   ├── train_all.py                 # Synthetic training & evaluation script
│   └── setup_demo.py                # Idempotent demo setup script
├── ANVAYA_HANDOFF.md                # This comprehensive handoff document
└── PROJECT_STATUS.json              # Machine-readable status file
```

---

## 5. Test Suite Verification

### Phase 3A Test Suite (`scripts/test_phase3a.py`) — 7/7 PASSED
* `test_1_verification_endpoint`: PASS
* `test_2_synthetic_mode`: PASS
* `test_3_real_data_providers`: PASS
* `test_4_feature_engineering`: PASS
* `test_5_moe_soft_gating`: PASS
* `test_6_multi_threshold_probability`: PASS
* `test_7_probabilistic_regime_classifier`: PASS

### Phase 3B Test Suite (`scripts/test_phase3b.py`) — 5/5 PASSED
* `test_1_timestamp_alignment`: PASS for endpoint equality only. This did **not** validate the GFS APCP accumulation bounds; see the correction below.
* `test_2_no_future_leakage`: PASS (Zero target leakage; antecedent correlation $r = 0.789$)
* `test_3_duplicate_prevention`: PASS (0 duplicate coordinates across all splits)
* `test_4_threshold_correctness_and_monotonicity`: PASS ($P(\text{extreme}) \le P(\text{very\_heavy}) \le P(\text{heavy})$)
* `test_5_temporal_split_separation`: PASS (Strict chronological order: Train 2023 < Val 2024 < Test 2025)

### GFS precipitation-window correction (2026-10-04)

The old handoff conflated valid-time endpoint equality with precipitation-window
equality. It also assumed a 03Z GFS initialization. Both assumptions were
wrong. IMD date D represents `[D-1 03Z, D 03Z)`. The verified GFS APCP product
is GRIB2 0-1-8, level 1, statistical process `Accumulation`, source units
`kg.m-2`. Original 2024-07-15 00Z f024 bounds are 18Z–00Z (6 h), not 24 h.

Added `ml/data/gfs_precip.py:build_24h_precipitation` to require metadata time
bounds/reference time and construct exact 24-hour rainfall. It differences
same-start nested accumulations only (e.g. f006−f003 for 03–06Z), rejects other
overlap/gaps and non-forecast inputs, then sums disjoint intervals. Raw provider
rainfall is now `gfs_apcp_interval_mm`; only derived `gfs_daily_rain_mm` is
enabled in `config/feature_registry.yaml`. Existing processed Parquet/model
artifacts still use `gfs_rain_mm` and are not validated against the APCP
interval definition; do not resume training or make skill claims from them
until rebuilt.

Real tests in `scripts/test_gfs_precip_alignment.py --real` cover 2024-07-15
00Z, 2024-07-16 00Z, 2024-08-15 00Z, and 2024-07-15 06Z. All four GFS windows
pass exact bounds, 24h duration, non-overlap, unit, domain, and valid-time
checks. Their targets are respectively Jul15 03Z–Jul16 03Z, Jul16 03Z–Jul17
03Z, Aug15 03Z–Aug16 03Z, and Jul16 03Z–Jul17 03Z. Saved test subsets are
approximately 569 KB each under `data/raw/gfs/test_subset/`; retrieval URLs,
record tables, checksums, and comparisons are in
`data/raw/manifests/gfs_precip_alignment_test.json`. The local IMD cache has no
Aug16 record; its July values lack a units attribute and are provenance
unverified. GFS-to-IMD interval arithmetic is verified, but independent IMD
value/unit validation remains limited. No bulk ingestion, training, or other
source integration was started. Offline validation passed 7/7 alignment
guard tests and 6/6 UCAR provider tests, including an assertion that raw APCP
is never exposed as either `gfs_daily_rain_mm` or deprecated `gfs_rain_mm`.

### Live Backend Endpoints (Port 8000)
* `/api/health`: HTTP 200 (`model_status: "ready"`)
* `/api/regimes`: HTTP 200
* `/api/forecast`: HTTP 200
* `/api/districts`: HTTP 200
* `/api/verification`: HTTP 200

---

## 6. Phase 3C-0 Scientific Readiness Gate

**SCIENCE_READY = NO**

The gate completed on the real temporal splits and wrote `data/results/science_readiness_report.json`.

### Regime counts by split

| Regime | Train | Validation | Test |
|---|---:|---:|---:|
| ACTIVE_MONSOON | 79,763 | 25,828 | 25,811 |
| BREAK_MONSOON | 11 | 3 | 4 |
| LOW_DEPRESSION | 3,825 | 1,221 | 1,208 |
| COASTAL_RAINFALL | 8,203 | 2,660 | 2,666 |
| OROGRAPHIC_RAINFALL | 18,034 | 5,845 | 5,829 |
| WESTERN_DISTURBANCE | 5 | 1 | 0 |
| OTHER | 8,513 | 2,733 | 2,773 |

Readiness by training coverage: ACTIVE_MONSOON, LOW_DEPRESSION, COASTAL_RAINFALL, OROGRAPHIC_RAINFALL, and OTHER are READY. BREAK_MONSOON and WESTERN_DISTURBANCE are INSUFFICIENT. No rows were duplicated or synthetically inflated.

### Unique dates, cycles, and grid locations

| Split | Rows | Unique dates | Unique forecast cycles | Unique grid locations |
|---|---:|---:|---:|---:|
| Train | 118,354 | 34 | 34 | 3,481 |
| Validation | 38,291 | 11 | 11 | 3,481 |
| Test | 38,291 | 11 | 11 | 3,481 |

All three splits contain July only; June, August, and September are absent. This limits independent synoptic-event coverage, especially for Break Monsoon and Western Disturbance conditions.

### Extreme-event audit

| Threshold | Train observed / raw GFS | Validation observed / raw GFS | Test observed / raw GFS |
|---|---:|---:|---:|
| >= 64.5 mm | 6,684 / 2,480 | 2,158 / 789 | 2,181 / 803 |
| >= 115.6 mm | 888 / 0 | 310 / 0 | 297 / 0 |
| >= 204.5 mm | 55 / 0 | 19 / 0 | 12 / 0 |

The zero raw-GFS counts at 115.6 mm and 204.5 mm are findings for this dataset and evaluated periods only. All metric implementations handled zero denominators without crashing; undefined ratios currently return 0.0 and should be treated cautiously.

### Leakage status

**PASS.** `observed_rain_mm` is absent from `FEATURE_SCHEMA` and `MINIMUM_FEATURES`; `scripts/build_real_dataset.py` uses prior-day observations for antecedent rainfall after the first episode row and a non-target historical estimate for that first row; all rows have `valid_time = init_time + 24h`; and train/validation/test init and valid times are disjoint.

### Exact next action

Do not train Phase 3C final models yet. Add additional real historical periods containing targeted August Break Monsoon episodes and September/early-October Western Disturbance episodes. Re-run `scripts/science_readiness_gate.py`; only proceed to final model training after the insufficient regimes have adequate real coverage, or after an explicitly documented fallback/global-expert decision.

## 7. Phase 3B-2 Targeted Regime Coverage Expansion

The builder now supports configurable windows through `scripts/build_real_dataset.py`:

* `--start-date` and `--end-date` for a custom episode
* `--year-range` for reproducible year selection
* `--expanded-coverage` for configured June-September candidate windows
* `--monsoon-only` to restrict windows to June-September
* `--include-winter` for documented October-April non-monsoon candidate windows
* `--replace-test` as an explicit opt-in; the existing 2025 `test.parquet` is preserved by default

The builder now preflights paired GFS and IMD source files, validates missing targets, invalid rainfall, timestamp alignment, duplicate coordinate-time rows, and cross-split valid-time overlap. Missing scientific source files are not synthesized unless `DATA_MODE=demo` or `--allow-synthetic` is explicitly used.

### Expansion result

| Metric | Old dataset | New dataset |
|---|---:|---:|
| Total rows | 194,936 | 194,936 |
| Train rows | 118,354 | 118,354 |
| Validation rows | 38,291 | 38,291 |
| Test rows | 38,291 | 38,291 (preserved) |
| Date coverage | July only | July only |
| Non-monsoon coverage | None | None |

The requested scientific expansion was not executed because the raw directory contains no paired GFS/IMD files for the configured June, August, September, or October-April windows. The preflight listed the missing files and exited before modifying processed data. This means regime counts and extreme-event counts remain those recorded in Section 6.

### Current coverage limitation

Break Monsoon remains insufficient at 11 training samples, and Western Disturbance remains insufficient at 5 training samples. Low / Depression coverage remains 3,825 training samples across the existing July episodes; no new event labels were created.

### Exact next action

Provide or ingest real paired GFS and IMD NetCDF/GRIB source files for the configured June-September monsoon windows and selected October-April non-monsoon windows. Run:

`python scripts/build_real_dataset.py --expanded-coverage --include-winter`

Then run `python scripts/profile_real_dataset.py` and `python scripts/science_readiness_gate.py`. Do not use `--allow-synthetic` for the scientific expansion and do not train models until the resulting gate is reviewed.

## 8. Exact Next Task (Phase 3C / Training Phase)
1. **After the readiness gate passes**, train the probabilistic regime classifier on `data/processed/train.parquet`.
2. Train regime-specific experts only for regimes with adequate real coverage; use a documented fallback for any insufficient regime.
3. Train the global ML baseline and calibrated multi-threshold probability heads.
4. Evaluate Raw GFS, Global ML Correction, and ANVAYA Soft-Gated MoE on held-out `test.parquet`.
5. Compute regime-stratified verification metrics and update `data/sample/verification_metrics.json`.

## 9. Full-scale multi-source mission checkpoint

The source-governance and provisional scientific implementation requested for
the full-scale mission is now present, with the following boundary:

* `config/data_sources.yaml` is the source registry and
  `config/feature_registry.yaml` is the feature/leakage registry.
* `scripts/discover_data_sources.py` writes the hashed local inventory at
  `data/source_manifest.json`.
* `scripts/source_health.py` writes endpoint/cache status at
  `data/results/source_health.json`.
* `scripts/download_data.py` provides explicit, resumable, checksum-aware
  downloads. It does not invent credentials or silently substitute sources.
* `scripts/validate_data.py --operational` is the data-quality and operational
  leakage firewall.
* `scripts/run_full_pipeline.py` orchestrates source discovery, health,
  profiling, validation, readiness, and opt-in training.
* `scripts/train_scientific.py` produces a global correction model, dedicated
  experts for adequately represented regimes, a documented global fallback for
  insufficient regimes, calibrated heavy/very-heavy/extreme probabilities,
  and overall/regime-stratified verification outputs.
* `backend/app/services/ml_service.py` and the forecast API expose
  `model_version`, `data_mode`, `prediction_mode`, and `fallback_regimes`.
  Scientific artifacts are selected only when `ANVAYA_MODEL_DIR` is explicit;
  the default demo path remains unchanged.
* `DATA_SOURCES.md`, `SCIENTIFIC_METHOD.md`, and `EXPERIMENT_REPORT.md` define
  the source, method, limitation, and interpretation contracts.

### Current source and model boundary

The source-health run recorded 3 local-cache sources and 9 temporarily
unavailable endpoints. The local GFS/IMD files are present but their external
provenance is unverified. GEFS, MERA, IMERG, INSAT, ERA5, MERRA-2, SRTM,
IBTrACS, and district boundaries were not integrated into the scientific
features or targets in this checkpoint. The failed IBTrACS retrieval left no
partial file in the raw directory.

The explicit provisional training run completed with model manifest
`anvaya-scientific-v1`. It trained dedicated experts for Active, Low/Depression,
Coastal, Orographic, and Other; Break Monsoon and Western Disturbance use the
global correction fallback because their training counts are 11 and 5. The
artifact status is `PROVISIONAL_UNVERIFIED_LOCAL_DATA`. These metrics are
reproducibility diagnostics only, not a publishable or operational skill claim.

### Final checkpoint commands

```powershell
python scripts/discover_data_sources.py
python scripts/source_health.py
python scripts/profile_real_dataset.py
python scripts/validate_data.py --operational
python scripts/science_readiness_gate.py
python scripts/test_phase3a.py
python scripts/test_phase3b.py
python scripts/test_scientific_pipeline.py
python scripts/run_full_pipeline.py --skip-training
```

The readiness command is expected to exit non-zero until paired real source
windows add adequate Break Monsoon and Western Disturbance coverage. The
2025 `test.parquet` remains preserved.

### Re-entry procedure

After real paired GFS/IMD windows are supplied, run
`python scripts/build_real_dataset.py --expanded-coverage --include-winter`,
review `data/results/science_readiness_report.json`, rerun the operational
validator, and only then run training without the provisional fallback flag.
Independent sources must be downloaded and versioned separately before being
used for validation, labels, uncertainty, or live features.

## 10. UCAR historical GFS single-case subset checkpoint

`ml/data/ucar_gfs_subset.py` adds `UCARGFSSSubsetProvider` alongside the
existing `GFSProvider`; the latter remains unchanged. Before each retrieval the
new provider reads the exact THREDDS `dataset.xml`, verifies all registry
field names against GRIB2 parameter/level/unit metadata, confirms the archive
time coordinates, and checks that the longitude convention is 0–360 degrees
east. Missing or unexpected fields fail explicitly; there is no demo fallback.

Validated only the requested case: 2024-07-15 00Z, f024, valid
2024-07-16 00Z, bbox 8–37°N/68–97°E. THREDDS requires matching vertical-level
and time coordinates for multi-variable grid requests; this file's 2 m
temperature and RH variables were rejected when grouped because their level
sets differ. The implementation therefore makes ten variable-specific
NCSS requests using NetCDF-3 and merges the validated fields. Its optional
2-D latitude/longitude output was excluded; the scalar CF grid-mapping value
and precipitation time bounds are retained as coordinates.

The resulting xarray-validated file is
`data/raw/gfs/test_subset/gfs_0p25_20240715_00_f024_india.nc`; its manifest is
`data/raw/manifests/gfs_test_subset.json`. The file is 616,601 bytes with
SHA-256 `590c3f6e497a42a5000ecda9d134c077261f4aa89d5e0310afffa653491b68bb`.
All ten raw GFS registry fields were present on a 117×117 India grid. The
precipitation accumulation is 18Z–00Z (six hours), ending at valid time; do
not treat it as a daily total without resolving the observation-window join.

The one-case estimates extrapolate to approximately 1.505 decimal GB stored
and 1.425 decimal GB transferred for the 2,440 2021–2025 June–September cycle
candidates. These estimates are not a retrieval plan: **no multi-year ingest
has started and none should run automatically**. Reproduce the case with
`backend/venv/Scripts/python.exe scripts/ingest_ucar_gfs_subset.py`; that
command defaults to this one date/cycle/lead only. Offline tests are in
`scripts/test_ucar_gfs_subset.py`.

## 11. Phase 3D bulk-ingestion source gate (2026-10-04)

The Phase 3D inventory enumerated 2,440 requested 2021–2025 JJAS date/cycle
slots (610 dates × four cycles). It issued zero bulk GFS requests: no locally
cached IMD target is provenance-verified. The six IMD-labelled files intersecting
2021–2025 cover 48 unique July dates, but their `rain` variable has no units,
their daily-midnight coordinates have no time bounds, and the source manifest
marks the raw cache `present_unverified`. Those 48 dates therefore account for
192 `TARGET_UNVERIFIED` cycle slots; the other 562 dates (2,248 slots) are
`TARGET_UNAVAILABLE`. These categories are not treated as observations.

Phase 3D result: zero requests, zero download failures, zero successful GFS
cases, and zero aligned samples. No normalized dataset was written. Regime
coverage and observed/raw daily-GFS extreme counts are N/A, not zero, because
there are no eligible samples. The legacy parquet splits contain deprecated
`gfs_rain_mm` and unverified targets; they were not used for coverage or event
statistics. `scripts/science_readiness_gate.py` now fails closed on missing
source verification, exact target-window equality, spatial matching, or the
required aligned `gfs_daily_rain_mm` and per-row provenance fields, and reports
`SCIENCE_READY = NO` without scoring the legacy rows.

The held-out 2025 test partition (`data/processed/test.parquet`) was preserved
byte-for-byte (SHA-256
`C12585AB1882014BC5005767FC88C4A6270D11D80F5D043D9797C99574E8F5E6`). No
regridding or point-vs-domain comparison was attempted. Details are in
`data/results/gfs_bulk_ingestion_report.json` and
`data/results/gfs_bulk_ingestion_summary.csv`. This was the pre-RF25 checkpoint;
the annual RF25 files documented below now satisfy the target-source
prerequisite for 2020–2024. Still do not begin bulk retrieval: first the
dataset-builder/alignment path and a normalized dataset need validation, and a
separate verified 2025 target is required without changing the held-out split.

## 12. Follow-on IMD RF25 target audit (2026-10-04)

The user-provided annual files identify as the official IMD CR&S RF25 product
(catalogue vintage 1901–2024); local full-year NetCDFs cover 2020–2024. They
use `RAINFALL` in mm on a 129×135, 0.25° grid, with `-999` missing values.
Their `TIME` coordinate is daily and labeled at 00Z, without time bounds. The
official IMD daily observation convention maps date D to 24-hour rainfall
ending at 08:30 IST / 03Z: `[D−1 03:00Z, D 03:00Z)`. The seven older episode
files remain unverified and are not substituted for RF25.

One already-cached case, GFS initialization 2024-07-15 00Z, was rebuilt with
`build_24h_precipitation` and matched to the IMD 2024-07-16 record. The exact
window is Jul 15 03Z–Jul 16 03Z. After reversing the IMD latitude axis, both
grids match exactly at 117×117 and 0.25°; longitude convention and units match.
There are 4,958 common valid grid cells. See
`data/results/imd_rf25_target_audit.json` for inventory and SHA-256 values.

The source-aware readiness gate was rerun and returned `SCIENCE_READY = NO` at
this historical, pre-Phase-3E checkpoint:
the one-case target/window audit passes, but no normalized aligned train,
validation, and test dataset exists. No GFS download, model training, or 2025
test modification occurred. Continue to hold multi-year ingestion until an
explicit ingestion plan is approved and the aligned dataset is ready for a
meaningful readiness assessment. This status was superseded by the completed
full-period Phase 3E build and gate in Section 13.

## 13. Phase 3E verified RF25/GFS dataset build checkpoint (2026-10-04)

Phase 3E adds a resumable source-backed builder for 2020–2024 June–September:
610 IMD RF25 target dates on the selected 117×117 India grid. It keeps the
existing `GFSProvider` intact and uses `UCARGFSSSubsetProvider` with UCAR GDEX
THREDDS/NCSS for the registered atmospheric predictors. Each target date D is
paired prospectively with the D−1 00Z forecast at +27h, valid at D03Z. GFS APCP
is assembled from metadata-bounded, non-overlapping intervals to exactly
match the RF25 window `[D−1 03:00Z, D 03:00Z)`. Splits are calendar-separated:
2020–2022 train, 2023 validation, and 2024 test. This new test split does not
alter legacy `data/processed/test.parquet`.

The bounded sample passed 40/40 cases: 24 train dates, eight validation, and
eight test dates, with 4,958 paired cells per date (198,320 rows total). All
ten registered raw GFS fields and `gfs_daily_rain_mm` are present; exact
metadata-backed windows, spatial grid nodes, no-missing/no-duplicate rows,
prospective lead +27 timing, and antecedent-rain leakage checks pass. All
seven weak-label regimes occur in each split; BREAK_MONSOON and
WESTERN_DISTURBANCE occur on every sample date. LOW_DEPRESSION covers 10 train,
2 validation, and 4 test sample dates, so final adequacy is deferred to the
full-period distinct-date gate. Sample outputs and QA are in
`data/processed/verified_sample/` and
`data/results/verified_dataset_sample_report.json`.

The full build completed 610/610 dates with zero unresolved failures. Final
partitions contain 1,814,628 train rows (366 dates, 2020–2022), 604,876
validation rows (122 dates, 2023), and 604,876 test rows (122 dates, 2024),
with 4,958 exact paired RF25/GFS cells per date. All 610 case manifests and
source checksums passed validation. The three split SHA-256 values and case
inventory are recorded in `data/results/verified_dataset_report.json`.

The readiness gate evaluated only `train_verified.parquet`,
`validation_verified.parquet`, and `test_verified.parquet` and returned
`SCIENCE_READY = YES`. Every weak-label regime has at least 167 distinct
training dates (minimum is LOW_DEPRESSION); all split-separation, target
isolation, antecedent-time, and per-row forecast-time checks passed. Regime
coverage below is unique target dates, not spatial row counts:

| Regime | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| ACTIVE_MONSOON | 363 | 122 | 122 |
| BREAK_MONSOON | 366 | 122 | 122 |
| LOW_DEPRESSION | 167 | 46 | 65 |
| COASTAL_RAINFALL | 272 | 99 | 98 |
| OROGRAPHIC_RAINFALL | 349 | 114 | 115 |
| WESTERN_DISTURBANCE | 365 | 122 | 121 |
| OTHER | 366 | 122 | 122 |

Observed rainfall cell/date counts at 64.5 / 115.6 / 204.5 mm are respectively
28,267 / 6,016 / 786 cells and 366 / 316 / 131 dates in train;
9,130 / 1,990 / 231 cells and 117 / 92 / 47 dates in validation; and
11,193 / 2,561 / 338 cells and 120 / 107 / 55 dates in test. The full regime,
extreme, leakage, and readiness reports are in
`data/results/science_readiness_report.json`.

At the Phase 3E checkpoint no model had been trained. The legacy 2025 test parquet
remains protected at SHA-256 `c12585ab1882014bc5005767fc88c4a6270d11d80f5d043d9797c99574e8f5e6`;
Phase 3F did not read or modify it.
The interrupted, footer-incomplete Parquet staging artifacts were preserved
under `data/processed/checkpoints/interrupted_full_build_20261004/`; they are
archival only and are not dataset inputs.

## 14. Phase 3F scientific model stack (2026-10-04)

Phase 3F used only the verified Phase 3E partitions: train 2020–2022
(1,814,628 rows), validation 2023 (604,876), and the 2024 test (604,876 rows;
122 target dates). The training script audited every row for
`valid_time = init_time + lead_hours` and antecedent observation time strictly
before initialization. It fit and calibrated on train/validation, saved and
hashed every model, wrote the freeze marker, and only then opened the 2024 test
partition. It did not read or change the separate legacy
`data/processed/test.parquet` artifact from 2025.

The 24 approved predictors are the current `FEATURE_SCHEMA` and live feature
allow-list. `observed_rain_mm`, regime labels, threshold labels,
`gfs_apcp_interval_mm`, legacy `gfs_rain_mm`, and unavailable `rain_last_6h`
were excluded from predictor matrices. The global baseline receives no regime
label. Seven dedicated rainfall experts were supported; probability experts
use global threshold fallbacks where a regime/threshold fails the train-only
support rule. ANVAYA uses a validation-temperature-scaled regime posterior,
soft-gates regression experts, and applies validation-only Platt calibration
to threshold heads. The three probabilities are constrained at inference to
`P(extreme) <= P(very heavy) <= P(heavy)`.

### Held-out 2024 test headline

Deterministic categorical scores use the predicted rainfall field thresholded
at 64.5 mm. FSS is the 5×5 grid-cell neighborhood score (approximately
1.25°), normalized over valid RF25 cells.

| System | RMSE (mm) | MAE (mm) | Bias (mm) | Pearson r | Heavy CSI | ETS | POD | FAR | FSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw GFS | 16.669 | 7.830 | 0.495 | 0.494 | 0.149 | 0.142 | 0.219 | 0.681 | 0.450 |
| Global ML | **14.079** | 7.280 | 0.714 | **0.607** | **0.196** | **0.191** | **0.233** | **0.447** | **0.527** |
| ANVAYA | 14.106 | **7.277** | **0.636** | 0.603 | 0.188 | 0.183 | 0.224 | 0.459 | 0.510 |

ANVAYA does not win overall: Global ML has slightly lower RMSE and stronger
heavy-event CSI/ETS/POD/FAR/FSS. ANVAYA has marginally lower MAE and bias.
Both learned systems improve RMSE over raw GFS.

At 115.6 mm, CSI/ETS/POD/FAR/FSS were Raw GFS
0.085/0.083/0.133/0.809/0.304, Global ML
0.059/0.058/0.068/0.682/0.193, and ANVAYA
0.044/0.043/0.048/0.655/0.136. At 204.5 mm, the respective scores were
0.047/0.047/0.074/0.887/0.174, 0.037/0.036/0.038/0.581/0.052, and
0/0/0/undefined/0. ANVAYA’s regression field issued no 204.5 mm events,
despite its separate probability head assigning six cells at or above 0.5
(four hits). This is a divergence between point estimates and event
probabilities, not extreme-event recovery by the rainfall regressor.

### Regime-stratified findings

Test cell counts were Active 86,223; Break 170,122; Low/Depression 35,324;
Coastal 16,548; Orographic 8,501; Western Disturbance 10,335; Other 277,823.
ANVAYA reduced RMSE versus Global ML in Active (18.393 vs 18.517 mm), Low
(16.818 vs 17.006), Coastal (22.820 vs 22.939), and Other (11.642 vs
11.662); Global ML was better in Break, Orographic, and Western Disturbance.
ANVAYA's heavy CSI was higher in Active (0.173 vs 0.170), Low (0.239 vs
0.215), and Other (0.076 vs 0.069), tied in Coastal (0.425), and lower in
Break (0.055 vs 0.080), Orographic (0.218 vs 0.339), and Western Disturbance
(0 vs 0.327). Sparse-event counts remain visible in
`data/results/regime_metrics.csv`; extreme values in Western Disturbance, for
example, are based on only two observed cells.

Validation temperature scaling reduced regime-label NLL from 0.009143 to
0.009062 (multiclass Brier 0.006002). Test probability-head ROC AUC was
0.923/0.948/0.912 for Heavy/Very Heavy/Extreme, with Brier
0.01475/0.00387/0.000543. These probabilistic scores are distinct from the
deterministic rainfall-threshold comparison. The regime task reproduces
heuristic weak labels; these scores are not an independent official synoptic
classification assessment.

Models are isolated under `models/scientific/phase3f/`. Predictions, test,
threshold and regime metrics, feature importance, validation confusion
matrix, frozen-model hashes and experiment manifest are under `data/results/`.
`scripts/test_phase3f.py` passed, loading 35 model artifacts and checking all
604,876 test predictions. The legacy models at the parent
`models/scientific/` path remain untouched. No frontend integration or new
data ingestion was started.

## 15. Phase 3G operational multi-source layer (2026-10-04)

Added a distinct `LIVE_OPERATIONAL_MODE` endpoint at
`GET /api/operational/forecast`, restricted to India point coordinates
8–37°N, 68–97°E. Existing scientific routes remain unchanged. The live path
requests an explicit NOAA GFS 00Z run through the Open-Meteo Single Runs API,
NOAA GEFS members through the Ensemble API, and independent ECMWF IFS/AIFS
single-run data. Open-Meteo is identified as the delivery layer; upstream
model/agency attribution is retained. NOAA GFS precipitation is a sum of 24
preceding-hour values over `[init+3h, init+27h)`. Cross-source consensus is
withheld unless independent model families match both valid time and exact
accumulation bounds. GEFS spread comes from the returned members, not a
synthetic fallback.

The backend loads only the frozen Phase 3F MoE and probability bundles for
inference. The `rain_last_24h` input is built from an earlier GFS 00Z +27 h
accumulation matching the frozen antecedent physical interval. Phase 3F used
observed IMD RF25 at this feature; the operational forecast proxy is a source
shift and remains unvalidated. NASA IMERG Early Run is fetched after inference
as satellite context, using 48 OPeNDAP point subsets, and never changes the
frozen model input or target. It requires NASA Earthdata access. Local IMD RF25
2020–2024 is identified only as `REFERENCE`, not a current operational
observation.

At the Phase 3G checkpoint no verified district polygons were present and
district risk was point-level only. This is superseded by the Phase 4 district
integration documented at the end of this handoff.

The API degrades per source and returns `LIVE`, `REFERENCE`, `DEMO`, or
`UNAVAILABLE` states. Responses are cached for 600 seconds by default. The
warning threshold is configurable and explicitly not an official IMD warning
protocol. Confidence is a source-coverage/spread stability diagnostic, not
calibrated skill. Phase 3G makes no claim of historical RMSE improvement.

Offline route/adapter smoke tests passed: `scripts/test_operational_api.py`,
12/12, including polygon behavior, model-before-IMERG ordering, geographic validation,
all-provider success (mocked provider inputs and real
frozen model artifacts) and all-provider failure fallback. This confirms route
assembly and fallback behavior, not current third-party endpoint availability
or NASA credential validity. At the Phase 3G checkpoint no frontend, verified
dataset, test partition, training artifact, or Phase 3F benchmark result was
modified. Phase 4 later integrated the frontend while retaining those
scientific artifacts unchanged.

## 16. Phase 4 district operational product (2026-10-05)

The five explicit outcomes are kept visible: weak-label regime classification,
bias-corrected rainfall, calibrated heavy/very-heavy/extreme probabilities,
district-level area-weighted product, and held-out scientific verification.
The NWIC district resource was acquired from the official portal and retained
with its original archive, source URL, vintage, and checksums. It contains 733
features and declares EPSG:7755; the full analytical geometry is transformed
to WGS84. Thirteen invalid polygons were repaired in derived files, with IDs
and method recorded in `data/geospatial/district_boundaries_metadata.json`.
The display-only map geometry is simplified separately and is never used for
lookup or aggregation.

The API adds `GET /api/operational/districts`, point lookup at
`GET /api/operational/districts/lookup`, on-demand area-weighted district
forecasts at `GET /api/operational/districts/{district_id}`, and
`GET /api/operational/map`. Aggregation clips 0.25-degree cells to the
district/domain intersection and uses EPSG:6933 area weights; at most nine
spatial strata are sampled, with represented area and coverage reported.
Summaries below configured coverage are suppressed. The warning policy is
configurable, named `ANVAYA OPERATIONAL DECISION RULE`, and is not an official
IMD warning. Only queried/computed districts carry forecast values; the map
does not imply an uncomputed nationwide forecast.

The live forecast panel exposes GFS, GEFS, ECMWF, frozen-model outputs,
probabilities, regime, confidence, warnings, source metadata, and a
model-based (non-causal) local feature attribution when available. The
verification endpoint reads only the frozen Phase 3F 2024 test, threshold,
and regime metric files;
it never substitutes synthetic/demo scores. It reports the three systems,
including Global ML's lower RMSE and stronger heavy CSI/ETS/POD/FAR/FSS than
ANVAYA, while ANVAYA's MAE and absolute bias are marginally lower.
LIVE_OPERATIONAL_MODE is kept distinct from SCIENTIFIC_MODE.

Phase 4 passed `scripts/test_phase4.py` (6/6), the existing
`scripts/test_operational_api.py` (12/12), and the frontend production build
(`npm run build`, Webpack with one static-generation worker). Live
third-party access and NASA Earthdata authorization remain deployment checks;
offline tests establish failure-safe behavior but do not prove upstream
availability. Verified data/model hashes and all Phase 3F metric files remain
unchanged.

## 17. Phase 5 judge-facing product polish (2026-10-05)

The entry page now states the ANVAYA product name, positioning, one-sentence
purpose, SIH26080 problem statement, and the complete raw-NWP-to-warning
pipeline. The dark operational-console layout carries a persistent API and
frozen-science status strip.

`/dashboard` is the district command center. It loads the existing reference
boundary map, defaults the selection to Wayanad when present, and computes
only a user-requested district through the existing map/forecast API. KPI and
table values remain blank until those API results exist. The warning table is
limited to computed districts and includes search plus warning, regime, and
state filters. Neutral reference polygons are never represented as forecast
values.

`/forecast` keeps the location-based LIVE point-forecast panel and adds the
requested six-control Wayanad scenario presentation. This scenario is
explicitly `DEMO · SYNTHETIC`: its browser-side scripted calculator is not a
model inference, calibrated probability, real event replay, current forecast,
or district warning. The current operational API accepts a location rather
than these scenario overrides; terrain slope is not in the frozen 24-feature
registry. The page states this before the controls and keeps the LIVE provider
query in a separate section. The Wayanad and neighboring district labels in
the demo are geography-only; the dashboard provides the actual on-demand live
district query.

`/districts` retains its reference catalog and on-demand area-weighted
district workflow. `/verification` presents the frozen 2024 Raw GFS / Global
ML / ANVAYA comparison and seven regime-stratified heavy-threshold rows. The
existing `GET /api/verification` response now exposes a read-only
`regime_stratified` view from `data/results/regime_metrics.csv`; no scores are
recomputed or substituted. That source CSV does not include regime-level FSS,
so the table displays it as unavailable rather than deriving a new number.

No new API route was added. Existing endpoints remain:
`GET /api/health`, `GET /api/operational/forecast`,
`GET /api/operational/districts`, `GET /api/operational/districts/lookup`,
`GET /api/operational/districts/{district_id}`, `GET /api/operational/map`,
and `GET /api/verification`.

Phase 5 verification: `npm run lint` passed with no warnings;
`npm run build` passed. `scripts/test_operational_api.py` passed 12/12 and
`scripts/test_phase4.py` passed 6/6, including checks that the frozen regime
rows are served and all six protected dataset/model hashes remain unchanged.
Live provider connectivity and NASA Earthdata credentials remain deployment
checks. No training, dataset, or model-artifact changes occurred and no new
historical skill claim is made.

## 18. Phase 6 keyless frontend mapping

The frontend map renderer is MapLibre GL JS using OpenFreeMap's Standard
(Bright), Light (Positron), and Dark styles over OpenStreetMap data. MapLibre's
attribution control is enabled and explicitly includes `© OpenStreetMap
contributors`. Only normal style/source/tile requests for the visible
viewport are made; there is no geographic-area prefetch or bulk tile cache.
The public map service requires no API key but is provided without an SLA.
Satellite imagery is explicitly unavailable and is not simulated.

Location search now uses the Open-Meteo Geocoding API, debounced with 24-hour
local recent-query caching and India-first sorting. Search returns place
coordinates and available country/admin subdivisions. A local 733-district
directory remains available in parallel when external geocoding fails.
Manual district-map selection and the user-initiated browser geolocation path
remain; no Google Maps, Places, or Geolocation API is used. **Google Maps is not required.**

The existing operational workspace and API flow are preserved. District
polygons remain neutral until queried, and forecast markers are only drawn
when the selected coordinate has an actual returned ANVAYA rainfall value.
Map-only warning colors are green (low), yellow/amber (moderate), red
(heavy/severe), and pink (exceptional/extreme); these are visual categories,
not official IMD warning classifications. Home, Map, Alerts, Insights and
Forecast retain their separate product roles.

Scientific models, verified partitions, frozen Phase 3F artifacts and metrics,
Phase 4 district aggregation, API contracts and verification outputs were not
changed. Frontend validation and route smoke checks are recorded in
`PROJECT_STATUS.json` after execution. Browser geolocation permission is
user-controlled and must be granted by the user for an end-to-end Locate Me
check.
