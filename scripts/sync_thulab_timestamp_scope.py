#!/usr/bin/env python3
"""Download and verify the frozen THULab timestamp-dataset scope.

The input scope is a JSON document containing ``timestamp_datasets`` rows with
``id``, ``repo_id`` and a 40-character Hugging Face revision.  Each repository
is materialized under its own directory.  Only TsFile data and the small source
metadata files are fetched.  Per-dataset SHA-256 receipts make a resumed run
fail closed if an existing file no longer matches the completed receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from huggingface_hub import snapshot_download


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
ALLOW_PATTERNS = ("*.tsfile", "README.md", ".gitattributes")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_scope(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    scope = json.loads(path.read_text(encoding="utf-8"))
    rows = scope.get("timestamp_datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scope has no timestamp_datasets")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        dataset_id = str(row.get("id", "")).strip()
        repo_id = str(row.get("repo_id", "")).strip()
        revision = str(row.get("revision", "")).strip()
        if not dataset_id or dataset_id in seen:
            raise ValueError(f"duplicate or empty dataset id: {dataset_id!r}")
        if repo_id != f"THULab/{dataset_id}":
            raise ValueError(f"unexpected repo id for {dataset_id}: {repo_id}")
        if not REVISION_RE.fullmatch(revision):
            raise ValueError(f"invalid frozen revision for {dataset_id}: {revision}")
        if "/" in dataset_id or dataset_id in {".", ".."}:
            raise ValueError(f"unsafe dataset id: {dataset_id}")
        seen.add(dataset_id)
        normalized.append({"id": dataset_id, "repo_id": repo_id, "revision": revision})
    expected = int(scope.get("timestamp_dataset_count", len(normalized)))
    if len(normalized) != expected:
        raise ValueError(
            f"timestamp dataset count mismatch: rows={len(normalized)} expected={expected}"
        )
    return scope, normalized


def receipt_files(dataset_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(dataset_dir).parts:
            continue
        relative = path.relative_to(dataset_dir).as_posix()
        if path.suffix == ".tsfile" or relative in {"README.md", ".gitattributes"}:
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def verify_receipt(dataset_dir: Path, receipt: dict[str, Any], row: dict[str, str]) -> bool:
    if receipt.get("status") != "complete":
        return False
    if receipt.get("repo_id") != row["repo_id"] or receipt.get("revision") != row["revision"]:
        return False
    files = receipt.get("files")
    if not isinstance(files, list) or not any(str(item.get("path", "")).endswith(".tsfile") for item in files):
        return False
    # Rebuild the complete eligible-file inventory.  Equality rejects missing,
    # modified, and extra stale TsFile/metadata objects, not just bad listed
    # objects.
    return receipt_files(dataset_dir) == files


def sync_one(
    row: dict[str, str], destination: Path, receipts: Path, verify_only: bool
) -> dict[str, Any]:
    dataset_dir = destination / row["id"]
    receipt_path = receipts / f"{row['id']}.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if verify_receipt(dataset_dir, prior, row):
            return prior | {"resume_status": "verified_existing"}
    if verify_only:
        raise RuntimeError(f"missing or invalid receipt for {row['id']}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=row["repo_id"],
        repo_type="dataset",
        revision=row["revision"],
        local_dir=str(dataset_dir),
        allow_patterns=list(ALLOW_PATTERNS),
        max_workers=4,
        etag_timeout=30,
    )
    files = receipt_files(dataset_dir)
    tsfiles = [item for item in files if item["path"].endswith(".tsfile")]
    if not tsfiles:
        raise RuntimeError(f"no TsFile objects downloaded for {row['repo_id']}")
    receipt = {
        "schema": "thulab-timestamp-dataset-download-receipt-v1",
        "status": "complete",
        "dataset_id": row["id"],
        "repo_id": row["repo_id"],
        "revision": row["revision"],
        "dataset_dir": str(dataset_dir.resolve()),
        "file_count": len(files),
        "tsfile_count": len(tsfiles),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "tsfile_bytes": sum(int(item["bytes"]) for item in tsfiles),
        "files": files,
        "completed_unix_seconds": time.time(),
    }
    atomic_json(receipt_path, receipt)
    return receipt | {"resume_status": "downloaded_and_verified"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    scope_path = args.scope.resolve()
    destination = args.destination.resolve()
    if destination in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"unsafe destination: {destination}")
    scope, rows = load_scope(scope_path)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".sync.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another sync process holds {lock_path}") from exc
    receipts = destination / "_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    expected_directories = {row["id"] for row in rows}
    unexpected_directories = sorted(
        child.name
        for child in destination.iterdir()
        if child.is_dir()
        and not child.name.startswith(".")
        and child.name != "_receipts"
        and child.name not in expected_directories
    )
    if unexpected_directories:
        raise RuntimeError(
            f"unexpected dataset directories in mirror: {unexpected_directories}"
        )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(sync_one, row, destination, receipts, args.verify_only): row
            for row in rows
        }
        for future in as_completed(pending):
            row = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "dataset_id": row["id"],
                            "status": result["resume_status"],
                            "tsfile_count": result["tsfile_count"],
                            "tsfile_bytes": result["tsfile_bytes"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:  # keep other independent downloads progressing
                failures.append({"dataset_id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps(failures[-1], ensure_ascii=False, sort_keys=True), flush=True)

    results.sort(key=lambda item: item["dataset_id"])
    top = {
        "schema": "thulab-timestamp-scope-download-v1",
        "status": "complete" if not failures and len(results) == len(rows) else "incomplete",
        "scope_id": scope.get("scope_id"),
        "scope_path": str(scope_path),
        "scope_sha256": sha256_file(scope_path),
        "dataset_count": len(results),
        "expected_dataset_count": len(rows),
        "tsfile_count": sum(int(item["tsfile_count"]) for item in results),
        "total_bytes": sum(int(item["total_bytes"]) for item in results),
        "tsfile_bytes": sum(int(item["tsfile_bytes"]) for item in results),
        "failures": failures,
        "inventory_verification": "exact eligible-file set, size, and SHA-256 per receipt",
        "datasets": [
            {
                "dataset_id": item["dataset_id"],
                "repo_id": item["repo_id"],
                "revision": item["revision"],
                "receipt": str((receipts / f"{item['dataset_id']}.json").resolve()),
            }
            for item in results
        ],
        "completed_unix_seconds": time.time(),
    }
    atomic_json(destination / "DOWNLOAD_MANIFEST.json", top)
    print(json.dumps(top | {"datasets": []}, ensure_ascii=False, sort_keys=True))
    return 0 if top["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
