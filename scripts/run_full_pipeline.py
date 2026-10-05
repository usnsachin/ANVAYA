"""Reproducible ANVAYA pipeline orchestrator.

Expensive stages are explicit. The orchestrator never trains through a failed
readiness gate unless ``--allow-insufficient-fallback`` is supplied.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], allow_failure: bool = False) -> int:
    print(f"\n=== {name} ===")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"Pipeline step failed: {name} ({result.returncode})")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible ANVAYA pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Skip any future explicit download stage.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-ui", action="store_true", help="UI is not rebuilt by this backend/data pipeline.")
    parser.add_argument("--force-rebuild", action="store_true", help="Attempt configured real-source rebuild; never synthesize unless explicitly requested.")
    parser.add_argument("--allow-insufficient-fallback", action="store_true")
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args()

    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "options": vars(args),
    }

    run["steps"].append({"name": "discover_data_sources", "exit": run_step(
        "Source manifest", [sys.executable, "scripts/discover_data_sources.py"]
    )})
    run["steps"].append({"name": "source_health", "exit": run_step(
        "Source health", [sys.executable, "scripts/source_health.py"]
    )})

    if args.force_rebuild:
        run["steps"].append({"name": "build_real_dataset", "exit": run_step(
            "Real-source data ingestion", [sys.executable, "scripts/build_real_dataset.py", "--expanded-coverage", "--include-winter"],
            allow_failure=False,
        )})

    run["steps"].append({"name": "profile_real_dataset", "exit": run_step(
        "Data profile", [sys.executable, "scripts/profile_real_dataset.py"]
    )})
    run["steps"].append({"name": "validate_data", "exit": run_step(
        "Leakage firewall and integrity", [sys.executable, "scripts/validate_data.py", "--operational"]
    )})
    gate_exit = run_step("Scientific readiness gate", [sys.executable, "scripts/science_readiness_gate.py"], allow_failure=True)
    run["steps"].append({"name": "science_readiness_gate", "exit": gate_exit})

    if not args.skip_training:
        if gate_exit and not args.allow_insufficient_fallback:
            print("Training skipped: readiness gate is not ready. Pass --allow-insufficient-fallback for a provisional global fallback.")
            run["steps"].append({"name": "train_scientific", "exit": "SKIPPED_READINESS_NOT_READY"})
        else:
            command = [sys.executable, "scripts/train_scientific.py"]
            if args.allow_insufficient_fallback:
                command.append("--allow-insufficient-fallback")
            if args.provisional:
                command.append("--provisional")
            run["steps"].append({"name": "train_scientific", "exit": run_step("Scientific training/evaluation", command)})
    else:
        run["steps"].append({"name": "train_scientific", "exit": "SKIPPED_BY_FLAG"})

    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    output = ROOT / "data/results/pipeline_run.json"
    output.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nPipeline checkpoint written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
