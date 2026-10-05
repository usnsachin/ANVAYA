"""
Comprehensive Meteorological Dataset Profiler for ANVAYA.

Profiles the expanded real dataset (and temporal partitions), validating:
  - Data integrity & missingness
  - Duplicate detection
  - Coordinate & temporal spans
  - Physical validity (non-negativity of precipitation)
  - Rainfall percentile distributions (p10 to p99.9)
  - Severe & extreme event frequencies (>= 64.5mm, >= 115.6mm, >= 204.5mm)
  - Regime proxy distributions across splits
Exports:
  - data/results/data_quality_report.json
  - data/results/data_quality_summary.csv
"""
import os
import sys
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ml.regime.labels import REGIME_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("profile_real_dataset")


def profile_dataframe(df: pd.DataFrame, partition_name: str = "full") -> dict:
    """Generate detailed descriptive metrics for a partition."""
    n_rows = len(df)
    n_unique_coords = len(df.drop_duplicates(subset=["latitude", "longitude"]))
    time_min = str(df["valid_time"].min())
    time_max = str(df["valid_time"].max())
    time_steps = df["valid_time"].nunique()
    init_steps = df["init_time"].nunique() if "init_time" in df.columns else 0
    unique_dates = pd.to_datetime(df["valid_time"], utc=True).dt.date.nunique()

    # Missing values
    missing_dict = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Duplicates check on (valid_time, latitude, longitude)
    dup_count = int(df.duplicated(subset=["valid_time", "latitude", "longitude"]).sum())

    # Target & NWP Rainfall distributions
    obs = df["observed_rain_mm"].values
    gfs = df["gfs_rain_mm"].values

    negative_obs = int((obs < 0.0).sum())
    negative_gfs = int((gfs < 0.0).sum())

    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    obs_pcts = {f"p{p}": float(np.percentile(obs, p)) for p in percentiles}
    gfs_pcts = {f"p{p}": float(np.percentile(gfs, p)) for p in percentiles}

    # Extreme rainfall thresholds
    heavy_count = int((obs >= 64.5).sum())
    very_heavy_count = int((obs >= 115.6).sum())
    extreme_count = int((obs >= 204.5).sum())

    gfs_heavy = int((gfs >= 64.5).sum())
    gfs_very_heavy = int((gfs >= 115.6).sum())
    gfs_extreme = int((gfs >= 204.5).sum())

    # Regime distribution
    observed_regimes = df["regime"].value_counts().to_dict() if "regime" in df.columns else {}
    regime_counts = {regime: int(observed_regimes.get(regime, 0)) for regime in REGIME_LABELS}
    regime_pcts = {k: round(v / n_rows * 100, 2) for k, v in regime_counts.items()}

    init = pd.to_datetime(df["init_time"], utc=True)
    valid = pd.to_datetime(df["valid_time"], utc=True)

    # Lead time distribution
    lead_time_dist = df["lead_hours"].value_counts().to_dict() if "lead_hours" in df.columns else {}

    regime_examples = {}
    for regime in REGIME_LABELS:
        subset = df[df["regime"] == regime]
        regime_examples[regime] = {
            "sample_dates": sorted(pd.to_datetime(subset["valid_time"], utc=True).dt.strftime("%Y-%m-%d").unique())[:5],
            "sample_forecast_cycles": sorted(subset["init_time"].astype(str).unique())[:5],
            "sample_grid_locations": [
                {"latitude": float(row.latitude), "longitude": float(row.longitude)}
                for row in subset[["latitude", "longitude"]].drop_duplicates().head(5).itertuples(index=False)
            ],
        }

    return {
        "partition": partition_name,
        "total_records": n_rows,
        "unique_grid_points": n_unique_coords,
        "unique_dates": unique_dates,
        "time_steps_count": time_steps,
        "forecast_cycles_count": init_steps,
        "time_range": {"min": time_min, "max": time_max},
        "spatial_bounds": {
            "lat_min": float(df["latitude"].min()),
            "lat_max": float(df["latitude"].max()),
            "lon_min": float(df["longitude"].min()),
            "lon_max": float(df["longitude"].max())
        },
        "lead_time_distribution": lead_time_dist,
        "data_integrity": {
            "total_missing_cells": total_missing,
            "duplicate_coordinate_time_rows": dup_count,
            "negative_observed_rain_count": negative_obs,
            "negative_gfs_rain_count": negative_gfs,
            "missing_target_count": int(df["observed_rain_mm"].isna().sum()),
            "timestamp_mismatch_count": int(
                ((valid - init).dt.total_seconds() / 3600.0 != df["lead_hours"]).sum()
            )
        },
        "rainfall_percentiles": {
            "observed_rain_mm": obs_pcts,
            "gfs_forecast_rain_mm": gfs_pcts
        },
        "extreme_event_counts": {
            "heavy_ge_64p5mm": {
                "observed_count": heavy_count,
                "observed_prevalence_pct": round(heavy_count / n_rows * 100, 3),
                "gfs_predicted_count": gfs_heavy,
                "gfs_prevalence_pct": round(gfs_heavy / n_rows * 100, 3)
            },
            "very_heavy_ge_115p6mm": {
                "observed_count": very_heavy_count,
                "observed_prevalence_pct": round(very_heavy_count / n_rows * 100, 3),
                "gfs_predicted_count": gfs_very_heavy,
                "gfs_prevalence_pct": round(gfs_very_heavy / n_rows * 100, 3)
            },
            "extremely_heavy_ge_204p5mm": {
                "observed_count": extreme_count,
                "observed_prevalence_pct": round(extreme_count / n_rows * 100, 4),
                "gfs_predicted_count": gfs_extreme,
                "gfs_prevalence_pct": round(gfs_extreme / n_rows * 100, 4)
            }
        },
        "regime_distribution": {
            "counts": regime_counts,
            "percentages": regime_pcts,
            "examples": regime_examples,
        },
    }


def main():
    results_dir = os.path.join(root_dir, "data/results")
    processed_dir = os.path.join(root_dir, "data/processed")
    os.makedirs(results_dir, exist_ok=True)

    full_parquet = os.path.join(processed_dir, "real_training_data.parquet")
    train_parquet = os.path.join(processed_dir, "train.parquet")
    val_parquet = os.path.join(processed_dir, "validation.parquet")
    test_parquet = os.path.join(processed_dir, "test.parquet")

    if not os.path.exists(full_parquet):
        logger.error(f"Cannot find {full_parquet}. Run scripts/build_real_dataset.py first.")
        return

    logger.info("Profiling full real dataset...")
    df_full = pd.read_parquet(full_parquet)
    report_full = profile_dataframe(df_full, "full")

    # Profile partitions if present
    report_partitions = {}
    for p_name, p_file in [("train", train_parquet), ("validation", val_parquet), ("test", test_parquet)]:
        if os.path.exists(p_file):
            logger.info(f"Profiling {p_name} partition...")
            df_p = pd.read_parquet(p_file)
            report_partitions[p_name] = profile_dataframe(df_p, p_name)

    partition_valid_times = {
        name: set(pd.to_datetime(df["valid_time"], utc=True).astype(str))
        for name, df in [("full", df_full)]
    }
    for p_name in report_partitions:
        df_p = pd.read_parquet(os.path.join(processed_dir, f"{p_name}.parquet"))
        partition_valid_times[p_name] = set(pd.to_datetime(df_p["valid_time"], utc=True).astype(str))
    split_pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    cross_split_overlap = {
        f"{left}_vs_{right}": len(partition_valid_times.get(left, set()) & partition_valid_times.get(right, set()))
        for left, right in split_pairs
        if left in partition_valid_times and right in partition_valid_times
    }

    final_report = {
        "dataset_name": "ANVAYA Multi-Period Real Aligned Dataset (GFS + IMD)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": report_full,
        "partitions": report_partitions,
        "scientific_integrity": {
            "cross_split_valid_time_overlap": cross_split_overlap,
            "pass": (
                all(value == 0 for value in cross_split_overlap.values())
                and all(
                    report["data_integrity"]["missing_target_count"] == 0
                    and report["data_integrity"]["timestamp_mismatch_count"] == 0
                    and report["data_integrity"]["duplicate_coordinate_time_rows"] == 0
                    for report in [report_full] + list(report_partitions.values())
                )
            )
        }
    }

    # 1. Export JSON report
    json_path = os.path.join(results_dir, "data_quality_report.json")
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Saved data quality report to JSON: {json_path}")

    # 2. Export compact CSV summary
    summary_rows = []
    all_reports = [("full", report_full)] + list(report_partitions.items())

    for name, r in all_reports:
        ext = r["extreme_event_counts"]
        obs_p = r["rainfall_percentiles"]["observed_rain_mm"]
        summary_rows.append({
            "partition": name,
            "records": r["total_records"],
            "grid_points": r["unique_grid_points"],
            "time_steps": r["time_steps_count"],
            "start_time": r["time_range"]["min"],
            "end_time": r["time_range"]["max"],
            "missing_cells": r["data_integrity"]["total_missing_cells"],
            "duplicate_rows": r["data_integrity"]["duplicate_coordinate_time_rows"],
            "negative_rain_values": r["data_integrity"]["negative_observed_rain_count"],
            "obs_p50_mm": obs_p["p50"],
            "obs_p90_mm": obs_p["p90"],
            "obs_p99_mm": obs_p["p99"],
            "obs_max_mm": obs_p["p100"],
            "heavy_events (>=64.5mm)": ext["heavy_ge_64p5mm"]["observed_count"],
            "very_heavy_events (>=115.6mm)": ext["very_heavy_ge_115p6mm"]["observed_count"],
            "extreme_events (>=204.5mm)": ext["extremely_heavy_ge_204p5mm"]["observed_count"]
        })

    csv_path = os.path.join(results_dir, "data_quality_summary.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    logger.info(f"Saved compact data quality summary to CSV: {csv_path}")

    # Terminal output
    full_ext = report_full["extreme_event_counts"]
    full_obs_p = report_full["rainfall_percentiles"]["observed_rain_mm"]
    print("\n" + "=" * 75)
    print("ANVAYA DATA QUALITY & EXTREME EVENT PROFILE REPORT")
    print("=" * 75)
    print(f"Total Records:                    {report_full['total_records']:,}")
    print(f"Unique Grid Points:               {report_full['unique_grid_points']:,}")
    print(f"Time Range:                       {report_full['time_range']['min']} -> {report_full['time_range']['max']}")
    print(f"Duplicate Coordinate-Time Rows:   {report_full['data_integrity']['duplicate_coordinate_time_rows']}")
    print(f"Missing Values:                   {report_full['data_integrity']['total_missing_cells']}")
    print(f"Negative Rainfall Values:         {report_full['data_integrity']['negative_observed_rain_count']}")
    print("-" * 75)
    print("Observed Rainfall Percentiles (mm/day):")
    print("  p25:  {:<6.2f} | p50 (Median): {:<6.2f} | p75: {:<6.2f} | p90: {:<6.2f}".format(
        full_obs_p["p25"], full_obs_p["p50"], full_obs_p["p75"], full_obs_p["p90"]))
    print("  p95:  {:<6.2f} | p99:          {:<6.2f} | p99.9: {:<6.2f} | Max: {:<6.2f}".format(
        full_obs_p["p95"], full_obs_p["p99"], full_obs_p["p99.9"], full_obs_p["p100"]))
    print("-" * 75)
    print("Extreme Event Coverage (Observed vs GFS Forecast):")
    print("  - Heavy (>= 64.5 mm):        Observed: {:>6,} ({:>5.2f}%) | GFS Forecast: {:>6,} ({:>5.2f}%)".format(
        full_ext["heavy_ge_64p5mm"]["observed_count"], full_ext["heavy_ge_64p5mm"]["observed_prevalence_pct"],
        full_ext["heavy_ge_64p5mm"]["gfs_predicted_count"], full_ext["heavy_ge_64p5mm"]["gfs_prevalence_pct"]))
    print("  - Very Heavy (>= 115.6 mm):  Observed: {:>6,} ({:>5.2f}%) | GFS Forecast: {:>6,} ({:>5.2f}%)".format(
        full_ext["very_heavy_ge_115p6mm"]["observed_count"], full_ext["very_heavy_ge_115p6mm"]["observed_prevalence_pct"],
        full_ext["very_heavy_ge_115p6mm"]["gfs_predicted_count"], full_ext["very_heavy_ge_115p6mm"]["gfs_prevalence_pct"]))
    print("  - Extreme (>= 204.5 mm):     Observed: {:>6,} ({:>5.3f}%) | GFS Forecast: {:>6,} ({:>5.3f}%)".format(
        full_ext["extremely_heavy_ge_204p5mm"]["observed_count"], full_ext["extremely_heavy_ge_204p5mm"]["observed_prevalence_pct"],
        full_ext["extremely_heavy_ge_204p5mm"]["gfs_predicted_count"], full_ext["extremely_heavy_ge_204p5mm"]["gfs_prevalence_pct"]))
    print("-" * 75)
    print("Temporal Partition Breakdown:")
    for p_name, r in report_partitions.items():
        p_ext = r["extreme_event_counts"]
        print(f"  [{p_name.upper():<10}] {r['total_records']:>7,} rows | {r['time_range']['min']} to {r['time_range']['max']} | Heavy: {p_ext['heavy_ge_64p5mm']['observed_count']} | Very Heavy: {p_ext['very_heavy_ge_115p6mm']['observed_count']} | Extreme: {p_ext['extremely_heavy_ge_204p5mm']['observed_count']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
