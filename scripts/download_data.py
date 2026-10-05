"""Cached, resumable downloader for explicitly selected public source files.

The downloader requires an explicit URL and destination. It never guesses a
scientific product URL from a landing page, and it refuses to overwrite an
existing file unless ``--force`` is given.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None,
             force: bool = False, timeout: int = 30) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        existing_hash = sha256_file(destination)
        if expected_sha256 and existing_hash != expected_sha256:
            raise RuntimeError(f"Existing file hash mismatch: {destination}")
        return {"status": "CACHED", "path": str(destination.relative_to(ROOT)), "sha256": existing_hash}

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Download failed with HTTP {response.status_code}: {url}")
        if offset and response.status_code == 200:
            partial.unlink()
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)

    digest = sha256_file(destination)
    if expected_sha256 and digest != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file hash mismatch: {destination}")
    return {
        "status": "DOWNLOADED",
        "path": str(destination.relative_to(ROOT)),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one explicitly selected ANVAYA source file with caching.")
    parser.add_argument("--source-id", required=True, help="Source id from config/data_sources.yaml.")
    parser.add_argument("--url", required=True, help="Exact file URL; landing pages are not download URLs.")
    parser.add_argument("--output", required=True, help="Workspace-relative destination path.")
    parser.add_argument("--sha256", help="Expected SHA-256 checksum.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text(encoding="utf-8"))
    configured = {source["id"] for source in config.get("sources", [])}
    if args.source_id not in configured:
        raise SystemExit(f"Unknown source id: {args.source_id}")
    destination = ROOT / args.output
    if ROOT not in destination.parents:
        raise SystemExit("Output must be inside the project workspace")

    result = download(args.url, destination, args.sha256, args.force, args.timeout)
    result["source_id"] = args.source_id
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
