# ANVAYA experiment report — historical provisional checkpoint, superseded by Phase 3F

The legacy July-only results below are retained for audit history only. The
authoritative verified-data model results are in the Phase 3F section at the
end of this report; do not combine the two experiments.

Date: 2026-10-03  
Model manifest: `models/manifest.json` (`anvaya-scientific-v1`)  
Data status: `PROVISIONAL_UNVERIFIED_LOCAL_DATA`  
Readiness: `SCIENCE_READY = NO`

## Dataset used

The current processed dataset has 194,936 rows over 3,481 grid points:

| Split | Rows | Period | Unique dates |
|---|---:|---|---:|
| Train | 118,354 | 2020–2023 | 34 |
| Validation | 38,291 | 2024 | 11 |
| Test | 38,291 | 2025, preserved | 11 |

All splits are July-only. The source-expansion preflight found no paired local
GFS/IMD files for the requested June, August, September, or non-monsoon
windows, so it stopped before writing new scientific outputs.

Training coverage is adequate for Active, Low/Depression, Coastal, Orographic,
and Other regimes. Break Monsoon has 11 training rows and Western Disturbance
has 5; both use the documented global correction fallback. Western Disturbance
has zero test rows. These counts are effective event coverage warnings, not
evidence of generalization.

## Held-out test diagnostics

These values are emitted by `data/results/scientific_metrics.json`. They are
included for reproducibility only and must not be presented as publishable
skill because the local raw inputs are unverified, coverage is July-only, and
the readiness gate is negative.

| System | RMSE (mm) | MAE (mm) | Bias (mm) | Heavy CSI | Very Heavy CSI | Extreme CSI |
|---|---:|---:|---:|---:|---:|---:|
| Raw GFS | 10.558 | 5.568 | -4.994 | 0.368 | 0.000 | 0.000 |
| Global ML | 4.475 | 2.580 | -0.248 | 0.931 | 0.689 | 0.000 |
| ANVAYA MoE | 4.535 | 2.621 | -0.228 | 0.922 | 0.639 | 0.000 |

The Extreme test slice contains 12 observed events and no hits from either
learned system. The very-heavy and extreme results therefore require expanded
event coverage before any scientific conclusion. Row counts also overstate
independent sample size because each date contains thousands of spatially
correlated grid cells.

## Regime-stratified interpretation

The full regime table is in `data/results/scientific_metrics.json`. It reports
RMSE, MAE, bias, and event metrics for Raw GFS, Global ML, and ANVAYA MoE in
each regime. For regimes with no observed events or no issued forecasts, the
corresponding ratio is `null`; this is undefined, not zero skill. In the
current July-only test set, the main supported comparisons are Active,
Coastal, Orographic, Low/Depression, and Other. Break has only four rows and
Western has none, so neither supports a regime claim.

## Reproducibility commands

```powershell
python scripts/discover_data_sources.py
python scripts/source_health.py
python scripts/profile_real_dataset.py
python scripts/validate_data.py --operational
python scripts/science_readiness_gate.py
python scripts/train_scientific.py --allow-insufficient-fallback --provisional
```

The last command documents the historical provisional workflow only. Phase
3F artifacts live separately under `models/scientific/phase3f/`; the existing
legacy models were not loaded or overwritten and remain opt-in in the backend:

```powershell
$env:ANVAYA_MODEL_DIR = "models/scientific"
$env:DATA_MODE = "historical"
uvicorn backend.app.main:app --reload --port 8000
```

The default backend configuration remains the existing demo/legacy path. The
Phase 3F artifacts have not been integrated into the backend and should not be
presented as operational forecasts.

## Historical next-step note (superseded by Phase 3E and Phase 3F)

Ingest authenticated or otherwise verifiable paired GFS and IMD data for
targeted August Break Monsoon and September/early-October Western Disturbance
episodes, plus enough June–September and winter coverage to make temporal and
regime results meaningful. Then rerun the readiness gate, regenerate the
manifest, retrain without the provisional fallback flag, and review calibration
and regime-stratified verification before deployment.

## Phase 3F — verified GFS/RF25 scientific model stack (2026-10-04)

### Scope and protocol

The only training inputs were `train_verified.parquet` (2020–2022,
1,814,628 rows), `validation_verified.parquet` (2023, 604,876 rows), and the
fixed 2024 test partition (604,876 rows over 122 target dates). All rows passed
the timing audit (`valid_time = init_time + lead_hours`; antecedent rainfall
strictly before initialization), feature registry audit, and finite/nonmissing
predictor checks. The model artifacts were saved and SHA-256 frozen before the
test partition was first opened. The legacy 2025 `test.parquet`, all previous
models, and unverified legacy datasets were not loaded. No new data were
downloaded and no frontend code was changed.

The 24 predictors match the live `FEATURE_SCHEMA`; target, regime target,
threshold labels, retired `gfs_rain_mm`/`gfs_apcp_interval_mm`, and unavailable
`rain_last_6h` were excluded. The global baseline does not ingest the regime
label. LightGBM models were trained sequentially with seed 42 and validation
early stopping. Seven rainfall experts passed the train-only support rule.
Threshold probability experts require at least 50 positive and 100 negative
training rows and 20 positive dates; unsupported regime/threshold pairs use
the global head. Regime probabilities were temperature-scaled and the
soft-gated binary probabilities Platt-calibrated on validation only.

### Overall 2024 test comparison

| System | RMSE (mm) | MAE (mm) | Mean bias (mm) | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Raw GFS | 16.669 | 7.830 | 0.495 | 0.494 |
| Global ML | **14.079** | 7.280 | 0.714 | **0.607** |
| ANVAYA | 14.106 | **7.277** | **0.636** | 0.603 |

The table below gives deterministic rainfall-field verification at each
threshold. FSS uses a 5×5-cell (~1.25°) neighborhood and valid-cell mask
normalization.

| System | Threshold | CSI | ETS | POD | FAR | FSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw GFS | 64.5 mm | 0.149 | 0.142 | 0.219 | 0.681 | 0.450 |
| Global ML | 64.5 mm | **0.196** | **0.191** | **0.233** | **0.447** | **0.527** |
| ANVAYA | 64.5 mm | 0.188 | 0.183 | 0.224 | 0.459 | 0.510 |
| Raw GFS | 115.6 mm | **0.085** | **0.083** | **0.133** | 0.809 | **0.304** |
| Global ML | 115.6 mm | 0.059 | 0.058 | 0.068 | 0.682 | 0.193 |
| ANVAYA | 115.6 mm | 0.044 | 0.043 | 0.048 | **0.655** | 0.136 |
| Raw GFS | 204.5 mm | **0.047** | **0.047** | **0.074** | 0.887 | **0.174** |
| Global ML | 204.5 mm | 0.037 | 0.036 | 0.038 | **0.581** | 0.052 |
| ANVAYA | 204.5 mm | 0 | 0 | 0 | undefined | 0 |

Conclusion: the trained ANVAYA MoE does not win overall. Global ML is best on
RMSE, Pearson r, and all reported heavy-threshold categorical measures at
64.5 mm. ANVAYA has slightly lower MAE and mean bias. Both learned systems
improve overall RMSE against raw GFS. ANVAYA's 204.5 mm regression field
predicts no extreme cells; its separate probability head at p>=0.5 detects
4/338 observed extreme cells from six forecasts.

### Calibrated ANVAYA probability heads

| Event threshold | Brier | Log loss | ROC AUC | Hits at p>=0.5 | Forecast cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heavy >=64.5 mm | 0.01475 | 0.06012 | 0.923 | 1,812 | 3,103 |
| Very heavy >=115.6 mm | 0.00387 | 0.01813 | 0.948 | 65 | 182 |
| Extreme >=204.5 mm | 0.000543 | 0.003673 | 0.912 | 4 | 6 |

Temperature scaling reduced validation regime-label NLL from 0.009143 to
0.009062; validation multiclass Brier was 0.006002. These labels are
heuristic weak-supervision classes, so this is label reconstruction, not an
independent meteorological regime verification. Test probability discrimination
is not the same as deterministic rainfall event detection.

### Regime-wise result summary

Test cell counts: Active 86,223; Break 170,122; Low/Depression 35,324; Coastal
16,548; Orographic 8,501; Western Disturbance 10,335; Other 277,823. Compared
with Global ML, ANVAYA RMSE improved in Active (18.393 vs 18.517 mm), Low
(16.818 vs 17.006), Coastal (22.820 vs 22.939), and Other (11.642 vs
11.662). Global ML was better in Break, Orographic, and Western Disturbance.
For heavy CSI ANVAYA was better in Active (0.173 vs 0.170), Low (0.239 vs
0.215), and Other (0.076 vs 0.069); tied in Coastal (0.425); and worse in
Break (0.055 vs 0.080), Orographic (0.218 vs 0.339), and Western
Disturbance (0 vs 0.327). Every regime and threshold remains in
`data/results/regime_metrics.csv` with sample/event counts; extreme results
for rare slices (e.g. two Western Disturbance extreme events) are highly
uncertain.

### Artifacts and validation

Models are in `models/scientific/phase3f/`. Outputs include
`data/results/model_training_report.json`, `test_metrics.csv`,
`threshold_metrics.csv`, `regime_metrics.csv`, `confusion_matrix_regime.csv`,
`feature_importance.csv`, `test_predictions.parquet`,
`experiment_manifest.json`, `probability_calibration.json`, and
`phase3f_frozen_models.json`. `scripts/test_phase3f.py` passed: all 35 model
files loaded, all 604,876 test prediction rows were checked, regression
outputs were finite/nonnegative, probabilities were in range and monotone,
and split/hash/freeze checks passed. No training/deployment claim beyond this
2024 JJAS experiment is warranted; cell samples are spatially correlated and
only one year is held out.
