"""Retrieve one explicitly requested UCAR GFS forecast subset and stop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.ucar_gfs_subset import UCARGFSSSubsetProvider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch one India-domain UCAR GFS THREDDS subset; this command never iterates over date ranges."
    )
    parser.add_argument("--date", default="2024-07-15", help="Forecast initialization date (YYYY-MM-DD).")
    parser.add_argument("--cycle", choices=("00", "06", "12", "18"), default="00")
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--output", help="Workspace-relative or absolute NetCDF output path.")
    parser.add_argument("--manifest", help="Workspace-relative or absolute JSON manifest path.")
    parser.add_argument("--force", action="store_true", help="Explicitly replace this one case's output and manifest.")
    args = parser.parse_args()
    init_time = datetime.strptime(f"{args.date} {args.cycle}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    result = UCARGFSSSubsetProvider().retrieve_forecast(
        init_time,
        lead_hours=args.lead_hours,
        output_path=args.output,
        manifest_path=args.manifest,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
