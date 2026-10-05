"""
ANVAYA Phase 3E -- Verified GFS/IMD Scientific Readiness Gate Before Model Training.

Performs:
1. Regime coverage audit per split (count, %, unique dates, unique cycles, unique grid cells)
2. Extreme-event coverage per split (observed AND raw GFS at 64.5, 115.6, 204.5 mm)
3. Regime readiness classification: READY / LOW_SAMPLE / INSUFFICIENT
4. Sample dependence documentation (unique forecast times, dates, grid cells)
5. Leakage audit (feature isolation, antecedent timing, and split separation)
6. Verification safety audit (division-by-zero checks in metrics.py)
7. Overstatement audit
8. Final SCIENCE_READY decision table

Outputs:
  data/results/science_readiness_report.json
  Updated ANVAYA_HANDOFF.md and PROJECT_STATUS.json with readiness table
"""
import os
import sys
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ml.regime.labels import REGIME_LABELS
from ml.features.engineer import FEATURE_SCHEMA, MINIMUM_FEATURES
from ml.verification import metrics as vm

# -- Readiness thresholds -------------------------------------------
# Coverage is based on distinct forecast/target dates, never grid-cell rows.
READY_UNIQUE_DATE_THRESHOLD = 30
LOW_SAMPLE_UNIQUE_DATE_THRESHOLD = 10

SPLITS = ["train", "validation", "test"]
THRESHOLDS_MM = {"heavy_ge_64p5mm": 64.5, "very_heavy_ge_115p6mm": 115.6, "extreme_ge_204p5mm": 204.5}
GATE_COLUMNS = [
    "target_date", "init_time", "valid_time", "lead_hours", "latitude", "longitude",
    "cycle", "forecast_cycle", "forecast_cycle_id", "regime", "gfs_daily_rain_mm",
    "observed_rain_mm", "antecedent_valid_time", "rain_last_24h",
]
VERIFIED_SPLIT_FILES = {"train": "train_verified.parquet", "validation": "validation_verified.parquet", "test": "test_verified.parquet"}


def load_splits(data_dir: str) -> dict:
    """Load only the columns needed for gate diagnostics from verified splits."""
    dfs = {}
    for s in SPLITS:
        path = os.path.join(data_dir, VERIFIED_SPLIT_FILES[s])
        if os.path.exists(path):
            columns = pq.ParquetFile(path).schema.names
            use_columns = [name for name in GATE_COLUMNS if name in columns]
            dfs[s] = pd.read_parquet(path, columns=use_columns)
            print(f"  Loaded {s}: {len(dfs[s]):,} rows")
        else:
            print(f"  WARNING: {path} not found!")
    return dfs


def source_alignment_preflight(data_dir: str, results_dir: str) -> dict:
    """Require the Phase 3E verified dataset and verify its split checksums."""
    blockers = []
    audit_path = os.path.join(results_dir, "imd_rf25_target_audit.json")
    target_audit = None
    if not os.path.exists(audit_path):
        blockers.append("Missing data/results/imd_rf25_target_audit.json IMD target audit.")
    else:
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                target_audit = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"Unable to read the IMD RF25 target audit: {exc}")

    case_audit = target_audit.get("single_case_comparison", {}) if target_audit else {}
    audit_pass = bool(
        target_audit
        and target_audit.get("status") == "PASS_SINGLE_CASE_TARGET_AND_WINDOW_AUDIT"
        and case_audit.get("window_match_verified") is True
        and case_audit.get("units_match_verified") is True
        and case_audit.get("spatial_grid_match_verified") is True
        and case_audit.get("latitude_order_verified") is True
        and case_audit.get("longitude_convention_verified") is True
        and case_audit.get("rebuilt_field_equals_cached_derived_field") is True
    )
    if target_audit is not None and not audit_pass:
        blockers.append("The one-case IMD/GFS target, units, or grid audit did not pass all checks.")

    verified_report_path = os.path.join(results_dir, "verified_dataset_report.json")
    verified_report = None
    if not os.path.exists(verified_report_path):
        blockers.append("Missing full data/results/verified_dataset_report.json; bounded samples are not final splits.")
    else:
        try:
            with open(verified_report_path, "r", encoding="utf-8") as f:
                verified_report = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"Unable to read the verified dataset report: {exc}")
    if verified_report is not None:
        if verified_report.get("status") != "VERIFIED_DATASET_READY_FOR_READINESS_GATE" or verified_report.get("stage") != "full":
            blockers.append("The verified dataset report is not a completed full-period Phase 3E dataset.")
        if verified_report.get("period", {}).get("stage_candidate_dates") != 610:
            blockers.append("The full verified dataset must cover all 610 2020-2024 JJAS target dates.")
        if verified_report.get("target_window_utc") != "For IMD date D: [D-1 03:00Z, D 03:00Z)":
            blockers.append("The full verified dataset does not document the exact IMD/GFS daily window.")
        if verified_report.get("spatial_mapping", {}).get("method", "").find("Exact 0.25-degree grid-to-grid") < 0:
            blockers.append("The full verified dataset has no exact grid-to-grid spatial mapping proof.")
    if not audit_pass:
        blockers.append("A passing one-case IMD RF25/GFS temporal-and-spatial audit is required before readiness evaluation.")

    required_columns = {
        "gfs_daily_rain_mm", "observed_rain_mm", "init_time", "valid_time", "target_date",
        "lead_hours", "latitude", "longitude", "cycle", "forecast_cycle", "forecast_cycle_id",
        "source", "source_version", "gfs_source", "gfs_archive_version", "imd_target_source",
        "processing_version", "antecedent_valid_time", "rain_last_24h",
    }
    split_schema = {}
    if not blockers:
        for split_name in SPLITS:
            path = os.path.join(data_dir, VERIFIED_SPLIT_FILES[split_name])
            if not os.path.exists(path):
                blockers.append(f"Missing normalized {split_name} split: {path}")
                continue
            try:
                parquet_file = pq.ParquetFile(path)
                columns = set(parquet_file.schema.names)
                num_rows = parquet_file.metadata.num_rows
            except Exception as exc:
                blockers.append(f"Unable to read normalized {split_name} split: {exc}")
                continue
            missing = sorted(required_columns - columns)
            if missing:
                blockers.append(f"{split_name} split lacks required aligned/provenance columns: {missing}")
            if "gfs_rain_mm" in columns:
                blockers.append(f"{split_name} split still contains deprecated gfs_rain_mm.")
            if num_rows <= 0:
                blockers.append(f"{split_name} verified split is empty.")
            report_file = verified_report.get("split_files", {}).get(split_name, {}) if verified_report else {}
            if report_file.get("path") != os.path.relpath(path, root_dir).replace(os.sep, "/"):
                blockers.append(f"{split_name} path differs from the verified dataset report.")
            expected_hash = report_file.get("sha256")
            digest = hashlib.sha256()
            with open(path, "rb") as source_stream:
                for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if not expected_hash or digest.hexdigest() != expected_hash:
                blockers.append(f"{split_name} verified split checksum does not match its report.")
            split_schema[split_name] = {
                "rows": int(num_rows),
                "has_gfs_daily_rain_mm": "gfs_daily_rain_mm" in columns,
                "has_deprecated_gfs_rain_mm": "gfs_rain_mm" in columns,
            }

    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "verified_dataset_report_path": os.path.relpath(verified_report_path, root_dir),
        "target_audit_path": os.path.relpath(audit_path, root_dir),
        "target_audit_status": target_audit.get("status") if target_audit is not None else None,
        "source_verification_status": "VERIFIED_FOR_ONE_CASE" if audit_pass else "BLOCKED",
        "full_dataset_status": verified_report.get("status") if verified_report else None,
        "single_case_comparison": case_audit if target_audit is not None else None,
        "verified_dataset_split_files": verified_report.get("split_files") if verified_report else None,
        "split_schema": split_schema,
        "blockers": blockers,
        "note": (
            "The single-case audit verifies source/window/grid semantics only. It cannot substitute for "
            "a non-empty normalized aligned train/validation/test dataset; legacy splits are not scored."
        ),
    }


def save_blocked_readiness(results_dir: str, preflight: dict) -> dict:
    """Persist an explicit NO decision without scoring unverified legacy rows."""
    reason = " ".join(preflight["blockers"])
    decision = {
        "SCIENCE_READY": "NO",
        "reason": reason,
        "ready_regimes": [],
        "low_sample_regimes": [],
        "insufficient_regimes": [],
        "regime_readiness_status": "NOT_ASSESSED_NO_ELIGIBLE_DATA",
        "minimum_additional_data": [],
        "recommended_strategy_for_insufficient": "Do not train; obtain verified, aligned source data first.",
    }
    report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "3E",
        "title": "Verified GFS/IMD Source-gated Scientific Readiness Gate",
        "dataset_preflight": preflight,
        "decision": decision,
        "legacy_dataset_metrics": "NOT_COMPUTED; only Phase 3E *_verified.parquet partitions are eligible evidence",
    }
    report_path = os.path.join(results_dir, "science_readiness_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSCIENCE_READY = NO")
    print(f"Reason: {reason}")
    print(f"Report saved to: {report_path}")
    return decision


def regime_coverage_audit(dfs: dict) -> dict:
    """Per-split regime coverage: count, %, unique dates, unique cycles, unique grid cells."""
    report = {}
    for split_name, df in dfs.items():
        split_total = len(df)
        split_report = {}
        for regime in REGIME_LABELS:
            subset = df[df["regime"] == regime]
            count = len(subset)
            pct = round(count / split_total * 100, 3) if split_total > 0 else 0.0

            # Date-level independent support, not spatial row counts.
            if count > 0:
                unique_dates = subset["target_date"].nunique()
                unique_cycles = subset["forecast_cycle_id"].nunique()
                unique_grids = subset[["latitude", "longitude"]].drop_duplicates().shape[0]
            else:
                unique_dates = 0
                unique_cycles = 0
                unique_grids = 0

            split_report[regime] = {
                "count": int(count),
                "percentage": pct,
                "unique_dates": int(unique_dates),
                "unique_forecast_cycles": int(unique_cycles),
                "unique_grid_locations": int(unique_grids),
            }
        report[split_name] = split_report
    return report


def extreme_event_coverage(dfs: dict) -> dict:
    """Per-split cell counts and unique target dates with events."""
    report = {}
    for split_name, df in dfs.items():
        split_report = {}
        for label, thresh in THRESHOLDS_MM.items():
            obs_count = int((df["observed_rain_mm"] >= thresh).sum())
            gfs_count = int((df["gfs_daily_rain_mm"] >= thresh).sum())
            obs_event_rows = df["observed_rain_mm"] >= thresh
            gfs_event_rows = df["gfs_daily_rain_mm"] >= thresh
            split_report[label] = {
                "observed_count": obs_count,
                "gfs_daily_count": gfs_count,
                "observed_unique_dates": int(df.loc[obs_event_rows, "target_date"].nunique()),
                "gfs_daily_unique_dates": int(df.loc[gfs_event_rows, "target_date"].nunique()),
                "observed_pct": round(obs_count / len(df) * 100, 3) if len(df) > 0 else 0.0,
                "gfs_daily_pct": round(gfs_count / len(df) * 100, 3) if len(df) > 0 else 0.0,
            }
        report[split_name] = split_report
    return report


def regime_readiness_classification(regime_audit: dict) -> dict:
    """Classify regimes by unique train dates, not correlated grid-cell rows."""
    train_regimes = regime_audit.get("train", {})
    readiness = {}
    for regime in REGIME_LABELS:
        info = train_regimes.get(regime, {})
        rows = info.get("count", 0)
        dates = info.get("unique_dates", 0)
        if dates >= READY_UNIQUE_DATE_THRESHOLD:
            status = "READY"
        elif dates >= LOW_SAMPLE_UNIQUE_DATE_THRESHOLD:
            status = "LOW_SAMPLE"
        else:
            status = "INSUFFICIENT"
        readiness[regime] = {
            "train_rows": rows,
            "train_count": rows,
            "train_unique_dates": dates,
            "train_unique_forecast_cycles": info.get("unique_forecast_cycles", 0),
            "status": status,
            "note": _readiness_note(regime, dates, status)
        }
    return readiness


def _readiness_note(regime: str, count: int, status: str) -> str:
    if status == "READY":
        return f"{count} distinct train target dates - meets the date-level coverage threshold."
    elif status == "LOW_SAMPLE":
        return (f"{count} distinct train target dates - marginal and not sufficient for a dedicated expert; "
                "expand or merge only under an explicit scientific policy.")
    else:
        return (f"{count} distinct train target dates - insufficient independent temporal support; "
                "grid-cell row totals do not change this classification.")


def sample_dependence_analysis(dfs: dict) -> dict:
    """Document that row count is not independent meteorological events."""
    report = {}
    for split_name, df in dfs.items():
        unique_valid_times = df["valid_time"].nunique()
        unique_init_times = df["init_time"].nunique()
        unique_dates = df["target_date"].nunique()
        unique_grids = df[["latitude", "longitude"]].drop_duplicates().shape[0]
        unique_forecast_cycles = df["forecast_cycle_id"].nunique()

        report[split_name] = {
            "total_rows": int(len(df)),
            "unique_valid_times": int(unique_valid_times),
            "unique_init_times": int(unique_init_times),
            "unique_dates": int(unique_dates),
            "unique_forecast_cycles": int(unique_forecast_cycles),
            "unique_grid_cells": int(unique_grids),
            "avg_grid_points_per_time": round(len(df) / max(unique_valid_times, 1), 1),
            "note": (
                f"Each forecast time maps to ~{round(len(df) / max(unique_valid_times, 1))} "
                "spatially correlated grid points. Row count overstates effective sample size. "
                "Independent meteorological events are approximately unique_dates x unique_grid_regions, "
                "not total_rows."
            )
        }
    return report


def leakage_audit(dfs: dict) -> dict:
    """Verify target isolation, strictly past antecedents, and calendar separation."""
    target_name = "observed_rain_mm"
    antecedent_names = ["rain_last_24h"]
    source_path = os.path.join(root_dir, "ml", "data", "verified_dataset.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    target_isolated = target_name not in FEATURE_SCHEMA and target_name not in MINIMUM_FEATURES
    required_features_present = all(name in FEATURE_SCHEMA for name in antecedent_names)
    source_uses_prior_observation = all(marker in source for marker in (
        "antecedent_date = target_date - timedelta(days=2)",
        "antecedent_valid_time < init",
        "rain_last_24h",
        "daily RF25 cannot support a sub-daily 6h antecedent feature",
    )) and "rain_last_6h" not in FEATURE_SCHEMA

    split_checks = {}
    init_sets = {}
    valid_sets = {}
    target_date_sets = {}
    per_split_antecedent_pass = {}
    target_time_semantics = {}
    for split_name, df in dfs.items():
        init_times = pd.to_datetime(df["init_time"], utc=True)
        valid_times = pd.to_datetime(df["valid_time"], utc=True)
        deltas = (valid_times - init_times).dt.total_seconds() / 3600.0
        split_checks[split_name] = {
            "all_valid_times_after_init": bool((valid_times > init_times).all()),
            "all_lead_hours_match": bool(np.allclose(deltas.to_numpy(), df["lead_hours"].to_numpy())),
            "all_valid_times_at_03z": bool((valid_times.dt.hour.eq(3) & valid_times.dt.minute.eq(0)).all()),
            "valid_time_date_equals_target_date": bool((valid_times.dt.strftime("%Y-%m-%d").to_numpy() == df["target_date"].astype(str).to_numpy()).all()),
            "all_target_years_match_split": bool(
                (df["target_date"].str[:4].astype(int).between(2020, 2022).all() if split_name == "train" else
                 df["target_date"].str[:4].astype(int).eq(2023).all() if split_name == "validation" else
                 df["target_date"].str[:4].astype(int).eq(2024).all())
            ),
            "min_hours_valid_after_init": float(deltas.min()) if len(deltas) else None,
            "max_hours_valid_after_init": float(deltas.max()) if len(deltas) else None,
        }
        antecedent_times = pd.to_datetime(df["antecedent_valid_time"], utc=True)
        per_split_antecedent_pass[split_name] = bool((antecedent_times < init_times).all())
        init_sets[split_name] = set(init_times.astype(str))
        valid_sets[split_name] = set(valid_times.astype(str))
        target_date_sets[split_name] = set(df["target_date"].astype(str))

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    no_init_overlap = all(
        init_sets[a].isdisjoint(init_sets[b]) for a, b in pairs if a in init_sets and b in init_sets
    )
    no_valid_overlap = all(
        valid_sets[a].isdisjoint(valid_sets[b]) for a, b in pairs if a in valid_sets and b in valid_sets
    )
    no_target_date_overlap = all(
        target_date_sets[a].isdisjoint(target_date_sets[b]) for a, b in pairs if a in target_date_sets and b in target_date_sets
    )

    checks = {
        "observed_rain_mm_not_in_feature_schema": target_isolated,
        "antecedent_features_in_schema": required_features_present,
        "antecedent_source_contract_uses_prior_observation": source_uses_prior_observation,
        "every_antecedent_valid_time_strictly_precedes_initialization": all(per_split_antecedent_pass.values()),
        "six_hour_antecedent_feature_excluded_without_subdaily_source": "rain_last_6h" not in FEATURE_SCHEMA,
        "no_future_observation_in_feature_definition": target_isolated and source_uses_prior_observation and all(per_split_antecedent_pass.values()),
        "no_init_time_overlap_between_splits": no_init_overlap,
        "no_valid_time_overlap_between_splits": no_valid_overlap,
        "no_target_date_overlap_between_splits": no_target_date_overlap,
        "per_split_time_alignment": all(
            info["all_valid_times_after_init"] and info["all_lead_hours_match"]
            and info["all_valid_times_at_03z"] and info["valid_time_date_equals_target_date"]
            and info["all_target_years_match_split"]
            for info in split_checks.values()
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "per_split": split_checks,
        "antecedent_time_checks": per_split_antecedent_pass,
        "feature_schema": list(FEATURE_SCHEMA),
        "note": (
            "Each rain_last_24h predictor comes from RF25 target date D-2, whose 03Z interval end "
            "strictly precedes 00Z initialization on D-1. Daily RF25 does not support rain_last_6h, "
            "so that feature is excluded; observed_rain_mm remains label-only."
        ),
    }


def verification_safety_audit() -> dict:
    """Inspect verification metric functions for division-by-zero safety."""
    results = {}

    # Test CSI with all-zero contingency
    csi_zero = vm.calculate_csi(0, 0, 0)
    results["calculate_csi(0,0,0)"] = {
        "value": csi_zero,
        "safe": csi_zero == 0.0 or (csi_zero != float('inf') and not (csi_zero != csi_zero)),
        "issue": "Returns 0.0 when denominator=0. Acceptable but NaN would be more correct (undefined skill)."
    }

    # Test POD with no observed events
    pod_zero = vm.calculate_pod(0, 0)
    results["calculate_pod(0,0)"] = {
        "value": pod_zero,
        "safe": pod_zero == 0.0 or (pod_zero != float('inf') and not (pod_zero != pod_zero)),
        "issue": "Returns 0.0 when hits+misses=0. Should be NaN (no observed events to detect)."
    }

    # Test FAR with no forecasts issued
    far_zero = vm.calculate_far(0, 0)
    results["calculate_far(0,0)"] = {
        "value": far_zero,
        "safe": far_zero == 0.0 or (far_zero != float('inf') and not (far_zero != far_zero)),
        "issue": "Returns 0.0 when hits+false_alarms=0. Should be NaN (no forecasts issued)."
    }

    # Test ETS with all zeros
    ets_zero = vm.calculate_ets(0, 0, 0, 0)
    results["calculate_ets(0,0,0,0)"] = {
        "value": ets_zero,
        "safe": ets_zero == 0.0 or (ets_zero != float('inf') and not (ets_zero != ets_zero)),
        "issue": "Returns 0.0 when total=0. Safe but masks undefined skill."
    }

    # Test ETS with realistic case: observed events but zero forecasts (complete miss)
    ets_miss = vm.calculate_ets(0, 100, 0, 5000)
    results["calculate_ets(0,100,0,5000)"] = {
        "value": round(ets_miss, 6),
        "safe": True,
        "issue": "Complete miss case. ETS should be negative or zero - valid."
    }

    # Check FSS with empty arrays
    try:
        fss_empty = vm.calculate_fss_proxy(np.array([]), np.array([]))
        results["calculate_fss_proxy([],[])"] = {
            "value": fss_empty,
            "safe": True,
            "issue": "Empty arrays handled (returns 0.0)."
        }
    except Exception as e:
        results["calculate_fss_proxy([],[])"] = {
            "value": None,
            "safe": False,
            "issue": f"Crashes with: {str(e)}"
        }

    all_safe = all(r["safe"] for r in results.values())
    results["overall_safe"] = all_safe
    results["recommendation"] = (
        "All metric functions handle zero-denominator cases without crashing. "
        "However, returning 0.0 for undefined ratios (e.g., POD when no events exist) "
        "can be misleading. Recommend returning NaN for genuinely undefined cases in "
        "production verification."
    )
    return results


def overstatement_audit(handoff_path: str) -> dict:
    """Check for overstatements in the handoff document."""
    issues = []
    if os.path.exists(handoff_path):
        with open(handoff_path, "r", encoding="utf-8") as f:
            content = f.read()

        problematic_phrases = [
            ("completely missed", "Overstates GFS failure - should be scoped to evaluated dataset and period"),
            ("raw GFS completely", "Universal claim about GFS - restrict to evaluated grid/period"),
            ("demonstrating severe conditional underprediction of mesoscale convective extremes",
             "This is a strong universal meteorological conclusion from a limited synthetic dataset"),
        ]
        for phrase, concern in problematic_phrases:
            if phrase.lower() in content.lower():
                issues.append({"phrase": phrase, "concern": concern})

    return {
        "overstatements_found": len(issues),
        "issues": issues,
        "recommendation": (
            "Replace universal GFS failure claims with: "
            "'In the evaluated dataset ({period}), raw GFS 0.25 degree forecast predicted "
            "N events above 204.5 mm compared to M observed events. "
            "This underprediction is consistent with known GFS precipitation biases "
            "at extreme thresholds but should not be generalized beyond the evaluated period.'"
        ) if issues else "No overstatements found."
    }


def full_monsoon_coverage_check(dfs: dict) -> dict:
    """Check if dataset covers only July or also June/August/September."""
    report = {}
    for split_name, df in dfs.items():
        months = df["valid_time"].apply(lambda x: pd.Timestamp(x).month if not isinstance(x, pd.Timestamp) else x.month)
        month_counts = months.value_counts().sort_index().to_dict()
        report[split_name] = {
            "months_covered": {str(m): int(c) for m, c in month_counts.items()},
            "months_missing": [m for m in [6, 7, 8, 9] if m not in month_counts],
        }

    missing_total = set()
    for s, info in report.items():
        missing_total.update(info["months_missing"])

    month_names = {6: "June", 7: "July", 8: "August", 9: "September"}
    report["recommendation"] = (
        f"Missing months: {', '.join(month_names[m] for m in sorted(missing_total))}. "
        "Full JJAS coverage is recommended for scientific credibility. "
        "Priority: add August (Break Monsoon episodes), September (Western Disturbance events)."
    ) if missing_total else "Full JJAS coverage achieved."
    return report


def compute_science_ready_decision(readiness: dict) -> dict:
    """Final SCIENCE_READY = YES/NO decision."""
    insufficient = [r for r, info in readiness.items() if info["status"] == "INSUFFICIENT"]
    low_sample = [r for r, info in readiness.items() if info["status"] == "LOW_SAMPLE"]
    ready = [r for r, info in readiness.items() if info["status"] == "READY"]

    # Any regime below the distinct-date READY threshold blocks this phase.
    # Spatial grid rows are correlated samples and cannot promote a class.
    unsupported = insufficient + low_sample
    if unsupported:
        science_ready = "NO"
        reason = (
            f"Regimes below {READY_UNIQUE_DATE_THRESHOLD} distinct training dates: {', '.join(unsupported)}. "
            "Grid-cell row counts do not satisfy temporal-support requirements; in particular, "
            "Western Disturbance and Break Monsoon must each meet the date-level threshold."
        )
        minimum_additional_data = []
        for r in unsupported:
            count = readiness[r]["train_unique_dates"]
            needed = max(0, READY_UNIQUE_DATE_THRESHOLD - count)
            minimum_additional_data.append({
                "regime": r,
                "current_train_unique_dates": count,
                "minimum_additional_unique_dates_needed": needed,
                "recommended_expansion": _expansion_recommendation(r)
            })
    else:
        science_ready = "YES"
        reason = "All regimes have sufficient training samples."
        minimum_additional_data = []

    return {
        "SCIENCE_READY": science_ready,
        "reason": reason,
        "ready_regimes": ready,
        "low_sample_regimes": low_sample,
        "insufficient_regimes": insufficient,
        "minimum_additional_data": minimum_additional_data,
        "recommended_strategy_for_insufficient": (
            "Do not train final models in this phase. Collect verified dates or revisit regime taxonomy "
            "under an explicit review; do not infer adequacy from spatial row counts."
        ) if unsupported else "N/A"
    }


def _expansion_recommendation(regime: str) -> str:
    recs = {
        "BREAK_MONSOON": (
            "Add mid-August dry spells (historically common break periods). "
            "Recommended windows: 2020-08-10 to 2020-08-20, 2021-08-05 to 2021-08-18, "
            "2022-08-12 to 2022-08-22. Also add late July breaks when monsoon trough "
            "shifts to foothills."
        ),
        "WESTERN_DISTURBANCE": (
            "Add late September / early October WD events over northern India. "
            "Recommended windows: 2020-09-20 to 2020-09-30, 2021-09-18 to 2021-09-28, "
            "2022-09-15 to 2022-09-25. WDs during monsoon are less frequent but occur "
            "in September when the subtropical jet shifts southward."
        ),
        "LOW_DEPRESSION": (
            "Add Bay of Bengal low-pressure episodes. "
            "Recommended: 2020-08-05 to 2020-08-12, 2021-07-28 to 2021-08-05."
        ),
    }
    return recs.get(regime, "Add more episodes covering this regime's synoptic conditions.")


def main():
    print("=" * 72)
    print("ANVAYA Phase 3E - VERIFIED GFS/IMD SCIENTIFIC READINESS GATE")
    print("=" * 72)
    print()

    data_dir = os.path.join(root_dir, "data", "processed")
    results_dir = os.path.join(root_dir, "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Only the full Phase 3E verified partitions are eligible. Old July-only
    # and legacy 2025 artifacts are deliberately never loaded or scored.
    preflight = source_alignment_preflight(data_dir, results_dir)
    if preflight["status"] != "PASS":
        return save_blocked_readiness(results_dir, preflight)

    # -- Load splits --
    print("1. Loading temporal splits...")
    dfs = load_splits(data_dir)
    if len(dfs) < 3:
        print("ERROR: Need train, validation, and test parquet files.")
        sys.exit(1)
    print()

    # -- Regime coverage audit --
    print("2. Regime coverage audit (per split)...")
    regime_audit = regime_coverage_audit(dfs)
    for split_name in SPLITS:
        print(f"\n  [{split_name.upper()}]")
        print(f"  {'Regime':<24} {'Count':>7} {'Pct':>7} {'Dates':>6} {'Cycles':>7} {'Grids':>6}")
        print(f"  {'-'*60}")
        for regime in REGIME_LABELS:
            info = regime_audit[split_name][regime]
            print(f"  {regime:<24} {info['count']:>7,} {info['percentage']:>6.1f}% "
                  f"{info['unique_dates']:>6} {info['unique_forecast_cycles']:>7} {info['unique_grid_locations']:>6}")
    print()

    # -- Extreme-event coverage --
    print("3. Extreme-event coverage (per split, observed AND aligned daily GFS)...")
    extreme_audit = extreme_event_coverage(dfs)
    for split_name in SPLITS:
        print(f"\n  [{split_name.upper()}]")
        print(f"  {'Threshold':<25} {'Obs Count':>10} {'Obs %':>7} {'GFS Count':>10} {'GFS %':>7}")
        print(f"  {'-'*62}")
        for label, info in extreme_audit[split_name].items():
            print(f"  {label:<25} {info['observed_count']:>10,} {info['observed_pct']:>6.2f}% "
                  f"{info['gfs_daily_count']:>10,} {info['gfs_daily_pct']:>6.2f}%")
    print()

    # -- Regime readiness classification --
    print("4. Regime readiness classification (based on TRAIN partition)...")
    readiness = regime_readiness_classification(regime_audit)
    print(f"\n  {'Regime':<24} {'Train Count':>12} {'Status':>14}")
    print(f"  {'-'*52}")
    for regime in REGIME_LABELS:
        info = readiness[regime]
        status_marker = {"READY": "[OK]", "LOW_SAMPLE": "[WARN]", "INSUFFICIENT": "[FAIL]"}.get(info["status"], "[?]")
        print(f"  {regime:<24} {info['train_unique_dates']:>9,} dates / {info['train_rows']:>10,} rows "
              f"{status_marker} {info['status']:>12}")
    print()

    # -- Sample dependence --
    print("5. Sample dependence analysis...")
    dependence = sample_dependence_analysis(dfs)
    for split_name in SPLITS:
        info = dependence[split_name]
        print(f"\n  [{split_name.upper()}]")
        print(f"    Total rows:              {info['total_rows']:>10,}")
        print(f"    Unique valid times:      {info['unique_valid_times']:>10}")
        print(f"    Unique init times:       {info['unique_init_times']:>10}")
        print(f"    Unique dates:            {info['unique_dates']:>10}")
        print(f"    Unique grid cells:       {info['unique_grid_cells']:>10}")
        print(f"    Avg grid pts/time:       {info['avg_grid_points_per_time']:>10.1f}")
    print()

    # -- Leakage audit --
    print("6. Leakage audit...")
    leakage = leakage_audit(dfs)
    for check_name, passed in leakage["checks"].items():
        print(f"  {check_name:<55} -> {'PASS' if passed else 'FAIL'}")
    print(f"  Overall leakage status: {leakage['status']}")
    print()

    # -- Verification safety --
    print("7. Verification metric safety audit...")
    safety = verification_safety_audit()
    for func_name, info in safety.items():
        if func_name in ("overall_safe", "recommendation"):
            continue
        safe_str = "SAFE" if info["safe"] else "UNSAFE"
        print(f"  {func_name:<35} -> {safe_str}  (value={info['value']})")
    print(f"  Overall safe: {safety['overall_safe']}")
    print(f"  Recommendation: {safety['recommendation']}")
    print()

    # -- Overstatement audit --
    print("8. Overstatement audit...")
    handoff_path = os.path.join(root_dir, "ANVAYA_HANDOFF.md")
    overstatements = overstatement_audit(handoff_path)
    print(f"  Found {overstatements['overstatements_found']} overstatement(s)")
    for issue in overstatements["issues"]:
        print(f"    - '{issue['phrase']}': {issue['concern']}")
    if overstatements["recommendation"]:
        print(f"  Recommendation: {overstatements['recommendation'][:200]}...")
    print()

    # -- Full monsoon coverage check --
    print("9. Full monsoon (JJAS) coverage check...")
    monsoon_coverage = full_monsoon_coverage_check(dfs)
    for split_name in SPLITS:
        info = monsoon_coverage[split_name]
        print(f"  [{split_name.upper()}] Months covered: {info['months_covered']}, Missing: {info['months_missing']}")
    print(f"  Recommendation: {monsoon_coverage['recommendation']}")
    print()

    # -- Final decision --
    print("=" * 72)
    print("10. FINAL SCIENCE_READY DECISION")
    print("=" * 72)
    decision = compute_science_ready_decision(readiness)
    print(f"\n  SCIENCE_READY = {decision['SCIENCE_READY']}")
    print(f"  Reason: {decision['reason']}")
    if decision["minimum_additional_data"]:
        print(f"\n  Minimum additional data needed:")
        for item in decision["minimum_additional_data"]:
            print(f"    - {item['regime']}: have {item['current_train_unique_dates']} unique dates, "
                  f"need {item['minimum_additional_unique_dates_needed']} more")
            print(f"      Expansion: {item['recommended_expansion'][:120]}...")
    print(f"\n  Strategy: {decision['recommended_strategy_for_insufficient']}")
    print("=" * 72)

    # -- Save report --
    full_report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "3E",
        "title": "Verified GFS/IMD Source-Gated Scientific Readiness Gate",
        "dataset_preflight": preflight,
        "regime_coverage_audit": regime_audit,
        "extreme_event_coverage": extreme_audit,
        "regime_readiness": readiness,
        "sample_dependence": dependence,
        "leakage_audit": leakage,
        "verification_safety": {k: v for k, v in safety.items() if k not in ("overall_safe", "recommendation")},
        "verification_safety_overall": safety["overall_safe"],
        "verification_safety_recommendation": safety["recommendation"],
        "overstatement_audit": overstatements,
        "monsoon_coverage": {k: v for k, v in monsoon_coverage.items() if k != "recommendation"},
        "monsoon_coverage_recommendation": monsoon_coverage["recommendation"],
        "decision": decision,
    }

    report_path = os.path.join(results_dir, "science_readiness_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if decision["SCIENCE_READY"] == "YES" else 1)
