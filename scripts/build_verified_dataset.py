"""Build the bounded real-data sample, then the verified 2020-2024 JJAS splits.

This command never trains a model and never reads or writes the legacy 2025
test partition. Each real source case is cached with its own manifest and is
resumed only after checksum validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.ucar_gfs_subset import UCARGFSSSubsetError, UCARGFSSSubsetProvider
from ml.data.verified_dataset import (
    IMDRF25Store,
    VerifiedDatasetError,
    candidate_target_dates,
    configure_session,
    write_split_partitions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sample", "full"), default="sample",
                        help="sample = 40 real dates (2 per month/year); full = all 610 JJAS dates.")
    args = parser.parse_args()
    try:
        dates = candidate_target_dates(args.stage)
        print(f"Stage {args.stage}: {len(dates)} target dates across 2020-2024 JJAS", flush=True)
        print("Split: target year 2020-2022 train, 2023 validation, 2024 test", flush=True)
        print("Forecast: latest eligible 00Z on D-1; target interval [D-1 03Z, D 03Z)", flush=True)
        imd = IMDRF25Store(ROOT)
        provider = UCARGFSSSubsetProvider(root=ROOT, session=configure_session())
        report = write_split_partitions(dates, imd, provider, root=ROOT, stage=args.stage)
        print(f"Status: {report['status']}", flush=True)
        print(f"Reports: {report['stage']} JSON/CSV written under data/results", flush=True)
        for split, info in report["data_quality"].items():
            print(f"{split}: {info['number_of_rows']:,} rows; {info['number_of_unique_dates']} target dates; "
                  f"{info['number_of_unique_grid_cells']} grid cells", flush=True)
        return 0
    except (VerifiedDatasetError, UCARGFSSSubsetError, OSError, ValueError) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
