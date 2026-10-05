# ANVAYA scientific method and safeguards

## Objective

Estimate grid-point rainfall correction and multi-threshold event risk from
forecast-time predictors while allowing the correction to vary by an
interpretable weather-regime posterior. The current implementation is a
reproducible engineering checkpoint, not a claim of operational skill.

For feature vector `x`, the classifier produces `P(r | x)` over seven weakly
supervised labels. Each sufficiently represented regime has an expert
`f_r(x)`. The correction is a soft mixture:

`y_hat_moe = sum_r P(r | x) f_r(x)`.

Insufficient regimes do not get a high-variance dedicated expert. They share a
global correction expert and are marked in the model manifest and inference
response as fallbacks.

## Target and physical timing

The target is `observed_rain_mm` for the IMD-style daily accumulation ending
at 03:00 UTC (08:30 IST) on date D: the physical window is
`[D-1 03:00Z, D 03:00Z)`. Equality of forecast valid time and the target end is
necessary but is not sufficient to establish precipitation alignment. GFS
cycles are at 00/06/12/18Z, not 03Z; for a 00Z initialization on D-1, the
first complete target window ends at lead 27h, not f024. More generally, the
forecast initialization must precede or equal the target start and every
accumulation record must be validated from its metadata-defined bounds.

The archived GFS APCP GRIB2 field is parameter 0-1-8, level type 1, with
statistical processing `Accumulation` and units `kg.m-2` (numerically
millimetres of liquid-water equivalent). The f024 value in the original test
subset covers 18Z–00Z (6h), not a daily total. The archive also supplies
overlapping nested intervals: e.g. f003 is 00–03Z, f006 is 00–06Z, f009 is
06–09Z, and f012 is 06–12Z. `ml/data/gfs_precip.py` differences only exact
same-start nested accumulations (f006−f003, f012−f009, etc.), rejects other
overlap, gaps, wrong units/statistics, observations, and mismatched forecast
reference times, then sums the resulting disjoint intervals. The resulting
`gfs_daily_rain_mm` must cover exactly the IMD 24-hour window. Raw records are
named `gfs_apcp_interval_mm`; they are not directly eligible as daily target
predictors. Only `gfs_daily_rain_mm` is enabled in the live feature registry.

Real NCSS tests cover 2024-07-15 00Z, 2024-07-16 00Z, 2024-08-15 00Z, plus
2024-07-15 06Z. Exact GFS bounds and the target-window arithmetic pass for all
four. At that time, only legacy local IMD episode files were available; their
`rain` variable has no units attribute and their provenance remains
unverified. The later RF25 annual archive audit and independent one-date
comparison are recorded below. Existing processed datasets/model artifacts using
`gfs_rain_mm` predate this APCP interval audit and must not be used for renewed
training or scientific skill claims until rebuilt with `gfs_daily_rain_mm`.

The builder still validates `valid_time = init_time + lead_hours` and the
split validator rejects temporal overlap, but those checks alone do not prove
that the GFS precipitation accumulation window matches the IMD target.

Antecedent rainfall features use the prior observation cycle. The first row of
an episode uses a non-target historical estimate, and the target is never part
of `FEATURE_SCHEMA`, `MINIMUM_FEATURES`, or the operational feature registry.

## Features and regimes

The current 24-feature schema combines GFS fields, antecedent rainfall,
location, derived wind/moisture/pressure features, terrain proxies, lead time,
and calendar fields. Each feature is registered in
`config/feature_registry.yaml` with source time, forecast availability, and
target-related flags.

Regime labels are transparent weak-supervision rules from
`ml/regime/labels.py`: `ACTIVE_MONSOON`, `BREAK_MONSOON`, `LOW_DEPRESSION`,
`COASTAL_RAINFALL`, `OROGRAPHIC_RAINFALL`, `WESTERN_DISTURBANCE`, and `OTHER`.
They are not official IMD event labels. Adding ERA5, IBTrACS, MERA, or any
other retrospective source requires a separate label-lineage audit so that
future information cannot enter a live feature path.

## Training and calibration

Phase 3F uses only the verified Phase 3E calendar partitions: 2020–2022
train, 2023 validation, and 2024 held-out test. The separate legacy 2025
`test.parquet` is not a model input and was left untouched. Random seed is 42.
All 24 predictors must be live-allowed and forecast-available in
`config/feature_registry.yaml`; `observed_rain_mm` is target-only, and the
weak regime is a classifier label, not an input to the global regressor.

LightGBM fits the global regression baseline, seven supported regime experts,
the seven-class regime posterior, and global plus supported per-regime binary
heads at 64.5, 115.6, and 204.5 mm. The posterior is scalar-temperature scaled
on validation. Threshold heads are soft-gated by the calibrated posterior and
their outputs receive validation-only Platt sigmoid calibration. A regime
threshold head is trained only with at least 50 positive rows, 100 negative
rows, and 20 distinct positive dates; otherwise that gate uses the global
threshold head. No oversampling or test-derived weights are used. The final
three probabilities are projected to
`P(extreme) <= P(very_heavy) <= P(heavy)`.

Validation drives early stopping and calibration only. A freeze marker records
SHA-256 hashes for all model artifacts before test rows are opened. Test
evaluation reports RMSE, MAE, bias, Pearson correlation, CSI, ETS, POD, FAR,
and FSS at each threshold for Raw GFS, Global ML, and ANVAYA. FSS is a
5×5-cell (~1.25°) neighborhood score with valid-grid mask normalization.
Undefined denominators remain null, not zero. Regime-level tables retain all
regimes and sample/event counts, including sparse slices.

## Leakage and provenance firewall

The minimum reproducible checks are:

```powershell
python scripts/discover_data_sources.py
python scripts/profile_real_dataset.py
python scripts/validate_data.py --operational
python scripts/science_readiness_gate.py
```

These checks cover missing/negative targets, timestamp alignment, duplicate
spatio-temporal keys, cross-split overlap, target isolation, and source
registry completeness. Phase 3F models use the verified Phase 3E RF25/GFS
partitions. Older provisional artifacts remain separate and are not loaded by
the Phase 3F trainer.

## Uncertainty and fallback semantics

The current model exposes calibrated threshold probabilities and soft regime
probabilities. A production uncertainty layer should add ensemble spread (GEFS
or another independently ingested source), prediction intervals, reliability
diagrams, and event-level bootstrap intervals. Those are not fabricated in
this checkpoint because GEFS and the independent reference products are not
available here.

The inference response exposes `prediction_mode`, `fallback_regimes`,
`data_mode`, and `model_version`. This prevents a consumer from confusing a
fallback correction with a dedicated expert or confusing the default demo
models with the explicitly selected scientific artifacts.

## Phase 3D source-gated expansion outcome

The 2021–2025 JJAS expansion was preflighted against the local target inventory
before any GFS request. The local IMD-labelled cache covers 48 unique July
dates in this period, but does not provide verifiable provider provenance,
rainfall units, or time bounds for the exact `[D-1 03:00Z, D 03:00Z)` target
window. The source manifest reports the local cache as unverified, and the
project builder has a synthetic demo-data generation path; consequently none
of these cached values is accepted as observed rainfall for this phase.

Result: 2,440 planned date/cycle slots, zero eligible targets, zero GFS
requests, zero aligned samples, and `SCIENCE_READY = NO`. For audit purposes,
48 local-but-unverified dates account for 192 cycle slots, while 562 dates
without local records account for 2,248 slots. Regime and extreme-event
distributions are N/A because no verified aligned records exist. The old
processed splits use deprecated `gfs_rain_mm`, so their event counts are not
reused. The 2025 test partition is preserved unchanged.

The readiness gate now requires a verified-source report, exact temporal and
spatial alignment evidence, and nonempty splits with `gfs_daily_rain_mm`,
observed targets, and GFS source/archive, cycle, IMD target-source, and
processing-version fields before computing regime/extreme summaries. A failed
source/alignment preflight returns NO and does not score legacy partitions. No
model training, regridding, or target fabrication occurred in this phase.

## Follow-on IMD RF25 target verification

The local annual files match IMD's official RF25 0.25° daily rainfall product
for 2020–2024. Their `RAINFALL` variable is in mm; date D follows IMD's daily
08:30 IST observation convention, giving `[D-1 03:00Z, D 03:00Z)`. The
NetCDFs use daily midnight date labels but do not contain time bounds, so the
observation convention is taken from IMD documentation rather than inferred
from the timestamp.

For target date 2024-07-16, an existing 2024-07-15 00Z GFS case was rebuilt
through `build_24h_precipitation` and compared cell-by-cell on 4,958 common
valid 0.25° grid points. Units, exact window, latitude order (after reversing
IMD's ascending axis), and positive-east longitude coordinates all match. This
is a single-case source audit, not a trained/evaluated dataset. At that
historical checkpoint the gate returned `SCIENCE_READY = NO` because no
aligned train/validation/test splits using `gfs_daily_rain_mm` existed. That
result is superseded by the completed full Phase 3E dataset and readiness gate
below; the 2025 test partition remains unchanged.

## Phase 3E dataset construction checkpoint

The planned supervised dataset covers every June–September RF25 date in
2020–2024 (610 target dates). For target date D, only the preceding D−1 00Z
GFS initialization is used, with endpoint lead +27h. Its valid time is D03Z,
and APCP predictors are constructed only from GRIB/CF metadata intervals that
tile the IMD interval `[D−1 03Z, D 03Z)` exactly; accumulation fields are not
interpreted from forecast-hour labels alone. Spatial pairing uses the exact
117×117, 0.25° grid over 8–37°N and 68–97°E, with RF25 latitude reversed to
GFS's descending order. Rainfall is mm; RF25 fill value -999 is excluded from
paired rows. The primary daily predictor is `gfs_daily_rain_mm`; raw APCP is
never exposed as daily rain.

Calendar splits are fixed before row creation: 2020–2022 train, 2023
validation, 2024 test. Antecedent rainfall features must come from records
strictly before forecast initialization; observed target rainfall remains
target-only. Regime coverage must be reported by distinct target dates as well
as cell rows, with event counts at 64.5, 115.6, and 204.5 mm. The 2025 legacy
test partition is protected by a pre/post SHA-256 and is outside these splits.

The 40-date bounded test passed: 24 train dates, eight validation, and eight
test dates, with 4,958 paired cells per case (198,320 rows total). All ten
registered raw fields and `gfs_daily_rain_mm` were present, all seven weak
regimes appeared in each split, and exact window, paired-grid, split-date,
missing/duplicate, and antecedent timing checks passed. BREAK_MONSOON and
WESTERN_DISTURBANCE were represented on every sampled date; LOW_DEPRESSION
appeared on 10 train, 2 validation, and 4 test sample dates. Thus sample
presence is confirmed, while adequacy is reserved for the full unique-date
assessment. Results are recorded in
`data/results/verified_dataset_sample_report.json`.

The full 610-date dataset is complete: 366 train dates (2020–2022), 122
validation dates (2023), and 122 test dates (2024), each with 4,958 paired
cells. All cases use D−1 00Z at lead +27h and metadata-validated APCP
accumulations matching `[D−1 03:00Z, D 03:00Z)`. Source checksums, split
hashes, feature/schema checks, exact grid checks, target isolation, antecedent
timing, and calendar separation passed. The legacy 2025 test parquet was
excluded and its protected SHA-256 remains unchanged.

The readiness gate (`data/results/science_readiness_report.json`) evaluated
only the three verified partitions and returned `SCIENCE_READY = YES`. Its
minimum training support is 167 unique dates for LOW_DEPRESSION; all seven
regimes exceed the 30-date threshold. Leakage audit and full JJAS coverage
checks passed. Observed extreme-event counts at 64.5 / 115.6 / 204.5 mm are
28,267 / 6,016 / 786 cells in train, 9,130 / 1,990 / 231 in validation, and
11,193 / 2,561 / 338 in test. This paragraph records the Phase 3E checkpoint;
the model fit and evaluation are now completed as described below.

## Phase 3F held-out evaluation (2026-10-04)

The fit used the fixed train 2020–2022, validation 2023, and test 2024
partitions. The saved model freeze and hashes precede test access. On the
604,876-row, 122-date test, Raw GFS / Global ML / ANVAYA RMSE was
16.669 / 14.079 / 14.106 mm. At 64.5 mm, CSI / ETS / POD / FAR / FSS was
0.149 / 0.142 / 0.219 / 0.681 / 0.450 for Raw GFS,
0.196 / 0.191 / 0.233 / 0.447 / 0.527 for Global ML, and
0.188 / 0.183 / 0.224 / 0.459 / 0.510 for ANVAYA. Thus the MoE did not
beat the global correction overall on RMSE or heavy categorical skill. It
reduced MAE slightly (7.277 vs 7.280 mm) and mean bias (0.636 vs 0.714 mm).

ANVAYA point rainfall predicted no test cells at or above 204.5 mm. Its
separate extreme probability head had test ROC AUC 0.912 and Brier 0.000543;
at probability 0.5 it forecast six cells, of which four were observed extreme
(338 observed cells total). The probability ranking and point-forecast event
detection therefore tell different stories; neither should be summarized as
extreme deterministic skill. The probability threshold results and all
regime-specific metrics are in `data/results/threshold_metrics.csv` and
`data/results/regime_metrics.csv`. Regime labels are heuristic weak labels,
and 2024 JJAS is a single held-out year; cellwise scores are spatially
correlated and do not imply independent-event significance.

Artifacts, code, exact split hashes, validation calibration diagnostics, and
software/model configuration are recorded in
`data/results/experiment_manifest.json` and
`data/results/model_training_report.json`. All 35 model artifacts loaded in
the Phase 3F artifact/leakage test script.

## Phase 4 operational product boundary

The problem outcomes are: (1) weak-label weather-regime classification,
(2) rainfall bias correction, (3) calibrated heavy/very-heavy/extreme event
probabilities, (4) district-level operational summaries, and (5) verification
using RMSE, ETS, CSI, POD, FAR, and FSS. Phase 4 adds district geometry and
on-demand live aggregation without changing the frozen Phase 3F datasets,
models, split assignment, predictions, or metrics.

Live providers are available only in `LIVE_OPERATIONAL_MODE`; the held-out
2024 benchmark is served separately in `SCIENTIFIC_MODE` from the frozen
Phase 3F result files. No current GFS/GEFS/ECMWF/IMERG response enters the
historical comparison, and operational model-input source substitution is
not evidence of improved historical RMSE. The benchmark shows Global ML has
lower RMSE and stronger heavy CSI/ETS/POD/FAR/FSS than ANVAYA; ANVAYA has
marginally lower MAE and absolute bias. No overall ANVAYA win is claimed.

District polygons are the 2025-vintage NWIC reference release, not a
meteorological observation. Cell/polygon intersections are measured in
EPSG:6933 and area-weighted over spatially stratified live forecast samples
(maximum nine sample points per district). Coverage is explicit and numerical
district summaries are unavailable below the configured 75% coverage floor.
This sampling is not an exhaustive district grid. Warning thresholds are a
configurable `ANVAYA OPERATIONAL DECISION RULE`, not an official IMD alert
protocol. The implementation and provenance details are in
`DATA_SOURCES.md` and `ANVAYA_HANDOFF.md`.
