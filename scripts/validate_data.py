"""Run the ANVAYA data-quality and operational leakage firewall."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features.engineer import FEATURE_SCHEMA
from ml.regime.labels import REGIME_LABELS


def validate_dataframe(df: pd.DataFrame, split: str) -> dict:
    valid = pd.to_datetime(df["valid_time"], utc=True)
    init = pd.to_datetime(df["init_time"], utc=True)
    delta_hours = (valid - init).dt.total_seconds() / 3600.0
    return {
        "split": split,
        "rows": int(len(df)),
        "unique_dates": int(valid.dt.date.nunique()),
        "unique_forecast_cycles": int(df["init_time"].nunique()),
        "unique_grid_locations": int(df[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_target": int(df["observed_rain_mm"].isna().sum()),
        "negative_observed": int((df["observed_rain_mm"] < 0).sum()),
        "negative_gfs": int((df["gfs_rain_mm"] < 0).sum()),
        "duplicate_coordinate_time": int(df.duplicated(["valid_time", "latitude", "longitude"]).sum()),
        "timestamp_mismatch": int((delta_hours != df["lead_hours"]).sum()),
        "valid_not_after_init": int((valid <= init).sum()),
        "regime_counts": {regime: int((df["regime"] == regime).sum()) for regime in REGIME_LABELS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ANVAYA processed data and feature leakage firewall.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="data/results/validation_report.json")
    parser.add_argument("--operational", action="store_true", help="Reject any registry feature not allowed for live use.")
    args = parser.parse_args()

    registry = yaml.safe_load((ROOT / "config" / "feature_registry.yaml").read_text(encoding="utf-8"))
    registry_rows = registry.get("features", [])
    registry_names = {row["feature_name"] for row in registry_rows}
    target_related = [row["feature_name"] for row in registry_rows if row.get("target_related")]
    live_disallowed = [row["feature_name"] for row in registry_rows if not row.get("allowed_for_live_model")]
    feature_checks = {
        "all_feature_schema_registered": set(FEATURE_SCHEMA) <= registry_names,
        "observed_target_not_in_feature_schema": "observed_rain_mm" not in FEATURE_SCHEMA,
        "no_registry_target_related_feature_allowed": not any(
            row.get("target_related") and row.get("allowed_for_live_model") for row in registry_rows
        ),
        "operational_registry_clean": not live_disallowed if args.operational else True,
    }

    processed = ROOT / args.processed_dir
    frames = {}
    split_reports = {}
    for split in ("train", "validation", "test"):
        path = processed / f"{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames[split] = df
            split_reports[split] = validate_dataframe(df, split)

    valid_sets = {
        split: set(pd.to_datetime(df["valid_time"], utc=True).astype(str))
        for split, df in frames.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlap = {
        f"{a}_vs_{b}": len(valid_sets.get(a, set()) & valid_sets.get(b, set()))
        for a, b in pairs
        if a in valid_sets and b in valid_sets
    }
    split_reports["cross_split_valid_time_overlap"] = overlap

    data_checks = {
        "all_expected_splits_present": len(frames) == 3,
        "no_cross_split_valid_time_overlap": all(value == 0 for value in overlap.values()),
        "all_split_integrity_checks_pass": all(
            all(report[key] == 0 for key in (
                "missing_cells", "missing_target", "negative_observed", "negative_gfs",
                "duplicate_coordinate_time", "timestamp_mismatch", "valid_not_after_init"
            ))
            for split, report in split_reports.items()
            if split in frames
        ),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_registry": "config/feature_registry.yaml",
        "feature_checks": feature_checks,
        "data_checks": data_checks,
        "pass": all(feature_checks.values()) and all(data_checks.values()),
        "target_related_registry_features": target_related,
        "live_disallowed_registry_features": live_disallowed,
        "splits": split_reports,
        "note": "PASS means structural integrity and registry checks passed; it does not establish that local raw files are externally authentic.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "feature_checks": feature_checks, "data_checks": data_checks}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
