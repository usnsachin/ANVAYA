"""Inventory configured data sources and local cache files.

This script is intentionally read-only with respect to scientific data. It
creates a provenance manifest from ``config/data_sources.yaml`` and local file
metadata, but never treats missing data as zero or as a successful download.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def discover(manifest_path: Path) -> dict:
    config_path = ROOT / "config" / "data_sources.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources = []
    for source in config.get("sources", []):
        cache_value = source.get("local_cache_path") or ""
        cache_paths = [part.strip() for part in cache_value.split(";") if part.strip()]
        files = []
        for cache_path in cache_paths:
            cache_dir = ROOT / cache_path
            if cache_dir.exists():
                files.extend(file_record(path) for path in sorted(cache_dir.rglob("*")) if path.is_file())
        sources.append({
            "id": source["id"],
            "name": source["name"],
            "provider": source["provider"],
            "configured_status": source.get("availability_status"),
            "local_cache_path": source.get("local_cache_path"),
            "local_file_count": len(files),
            "local_files": files,
        })

    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "source_config": "config/data_sources.yaml",
        "sources": sources,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ANVAYA local source manifest.")
    parser.add_argument("--output", default="data/source_manifest.json")
    args = parser.parse_args()
    manifest = discover(ROOT / args.output)
    print(json.dumps({
        "manifest": args.output,
        "sources": len(manifest["sources"]),
        "cached_files": sum(item["local_file_count"] for item in manifest["sources"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
