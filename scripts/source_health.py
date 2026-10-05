"""Check configured source endpoints and local cache health.

Network checks are lightweight HEAD/GET metadata checks only. This script does
not download scientific data and never maps an unavailable source to zeros.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
STATUS_VALUES = {"AVAILABLE", "AUTH_REQUIRED", "TEMPORARILY_UNAVAILABLE", "NOT_FOUND"}


def _local_files(source: dict) -> int:
    count = 0
    for raw_path in str(source.get("local_cache_path", "")).split(";"):
        path = ROOT / raw_path.strip()
        if path.exists():
            count += sum(1 for item in path.rglob("*") if item.is_file())
    return count


def check_source(source: dict, timeout: int = 8) -> dict:
    source_id = source["id"]
    local_count = _local_files(source)
    if local_count:
        return {
            "id": source_id,
            "status": "AVAILABLE",
            "basis": "local_cache",
            "local_file_count": local_count,
            "url": source.get("citation"),
            "provenance_warning": "local cache presence does not verify provider provenance or product version",
        }

    url = source.get("citation")
    if not url:
        return {"id": source_id, "status": "NOT_FOUND", "basis": "no_endpoint_configured"}
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code in (405, 501):
            response = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
        code = response.status_code
        if code in (401, 403):
            status = "AUTH_REQUIRED"
        elif 200 <= code < 400:
            status = "AVAILABLE"
        elif code == 404:
            status = "NOT_FOUND"
        else:
            status = "TEMPORARILY_UNAVAILABLE"
        return {
            "id": source_id,
            "status": status if status in STATUS_VALUES else "TEMPORARILY_UNAVAILABLE",
            "basis": "endpoint_check",
            "http_status": code,
            "final_url": response.url,
            "url": url,
        }
    except requests.RequestException as exc:
        return {
            "id": source_id,
            "status": "TEMPORARILY_UNAVAILABLE",
            "basis": "endpoint_check",
            "url": url,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ANVAYA data source availability.")
    parser.add_argument("--output", default="data/results/source_health.json")
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text(encoding="utf-8"))
    sources = config.get("sources", [])
    results = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as executor:
        futures = [executor.submit(check_source, source, args.timeout) for source in sources]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["id"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statuses": results,
        "counts": {status: sum(item["status"] == status for item in results) for status in sorted(STATUS_VALUES)},
        "note": "Endpoint availability is not equivalent to successful scientific ingestion; inspect source manifests and credentials.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
