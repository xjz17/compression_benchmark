#!/usr/bin/env python3
"""Materialize already-present THULab files into one verified local mirror.

The tool is deliberately offline.  It searches only explicitly supplied
candidate roots, accepts a source only after its byte size and SHA-256 match a
frozen per-dataset receipt, and never replaces a different destination file.
On APFS it first attempts a clone copy (``cp -c``); otherwise it falls back to
an ordinary metadata-preserving copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Iterable


SCHEMA = "thulab-local-existing-materialization-progress-v1"
PRUNED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
}


class MaterializationError(RuntimeError):
    """Raised when the frozen input contract is malformed or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read JSON {path}: {exc}") from exc


def _safe_receipt_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise MaterializationError("receipt file path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise MaterializationError(f"unsafe receipt file path: {value!r}")
    normalized = pure.as_posix()
    if normalized != value:
        raise MaterializationError(f"non-canonical receipt file path: {value!r}")
    return normalized


def load_contract(scope_path: Path, receipts_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scope = _load_json(scope_path)
    if not isinstance(scope, dict):
        raise MaterializationError("scope must be a JSON object")
    datasets = scope.get("timestamp_datasets")
    if not isinstance(datasets, list):
        raise MaterializationError("scope.timestamp_datasets must be a list")
    expected_count = scope.get("timestamp_dataset_count")
    if expected_count != len(datasets):
        raise MaterializationError("scope timestamp_dataset_count does not match its dataset list")

    seen_ids: set[str] = set()
    expected: list[dict[str, Any]] = []
    for row in datasets:
        if not isinstance(row, dict):
            raise MaterializationError("scope dataset entry must be an object")
        dataset_id = row.get("id")
        repo_id = row.get("repo_id")
        revision = row.get("revision")
        if not all(isinstance(value, str) and value for value in (dataset_id, repo_id, revision)):
            raise MaterializationError("scope dataset id, repo_id, and revision must be non-empty strings")
        if PurePosixPath(dataset_id).parts != (dataset_id,) or dataset_id in {".", ".."}:
            raise MaterializationError(f"dataset id is not a safe path component: {dataset_id!r}")
        if dataset_id in seen_ids:
            raise MaterializationError(f"duplicate dataset id in scope: {dataset_id}")
        seen_ids.add(dataset_id)

        receipt_path = receipts_dir / f"{dataset_id}.json"
        receipt = _load_json(receipt_path)
        if not isinstance(receipt, dict) or receipt.get("status") != "complete":
            raise MaterializationError(f"receipt is not complete: {receipt_path}")
        if receipt.get("dataset_id") != dataset_id:
            raise MaterializationError(f"receipt dataset_id mismatch: {receipt_path}")
        if receipt.get("repo_id") != repo_id or receipt.get("revision") != revision:
            raise MaterializationError(f"receipt identity mismatch: {receipt_path}")
        files = receipt.get("files")
        if not isinstance(files, list) or not files:
            raise MaterializationError(f"receipt files must be a non-empty list: {receipt_path}")

        seen_paths: set[str] = set()
        normalized_files: list[dict[str, Any]] = []
        for file_row in files:
            if not isinstance(file_row, dict):
                raise MaterializationError(f"receipt file entry is not an object: {receipt_path}")
            relative_path = _safe_receipt_path(file_row.get("path"))
            size = file_row.get("bytes")
            digest = file_row.get("sha256")
            if relative_path in seen_paths:
                raise MaterializationError(f"duplicate receipt path {dataset_id}/{relative_path}")
            seen_paths.add(relative_path)
            if not isinstance(size, int) or size < 0:
                raise MaterializationError(f"invalid byte size for {dataset_id}/{relative_path}")
            if not isinstance(digest, str) or len(digest) != 64:
                raise MaterializationError(f"invalid SHA-256 for {dataset_id}/{relative_path}")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise MaterializationError(f"invalid SHA-256 for {dataset_id}/{relative_path}") from exc
            normalized_files.append(
                {
                    "dataset_id": dataset_id,
                    "path": relative_path,
                    "bytes": size,
                    "sha256": digest.lower(),
                }
            )
        if receipt.get("file_count") != len(normalized_files):
            raise MaterializationError(f"receipt file_count mismatch: {receipt_path}")
        if receipt.get("total_bytes") != sum(row["bytes"] for row in normalized_files):
            raise MaterializationError(f"receipt total_bytes mismatch: {receipt_path}")
        expected.extend(normalized_files)

    return scope, expected


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_candidate_roots(candidate_roots: Iterable[Path], target_root: Path) -> list[Path]:
    target_resolved = target_root.expanduser().resolve(strict=False)
    normalized: list[Path] = []
    seen: set[Path] = set()
    for raw in candidate_roots:
        path = raw.expanduser().resolve(strict=True)
        if not path.is_dir() and not path.is_file():
            raise MaterializationError(f"candidate root is not a file or directory: {path}")
        if (
            path == target_resolved
            or _is_relative_to(path, target_resolved)
            or _is_relative_to(target_resolved, path)
        ):
            raise MaterializationError(f"candidate root and target overlap: {path}")
        if path not in seen:
            normalized.append(path)
            seen.add(path)
    if not normalized:
        raise MaterializationError("at least one --candidate-root is required")
    return normalized


def iter_candidate_files(roots: Iterable[Path], interesting_names: set[str]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            if root.name in interesting_names and not root.is_symlink():
                yield root
            continue
        for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in PRUNED_DIRECTORY_NAMES
                and not (Path(directory) / name).is_symlink()
            )
            for name in sorted(file_names):
                if name not in interesting_names:
                    continue
                path = Path(directory) / name
                if path.is_file() and not path.is_symlink():
                    yield path


def discover_sources(
    expected: list[dict[str, Any]], candidate_roots: list[Path]
) -> tuple[dict[tuple[str, str], Path], dict[str, int]]:
    by_key = {(row["dataset_id"], row["path"]): row for row in expected}
    dataset_ids = {row["dataset_id"] for row in expected}
    by_loose_tsfile: dict[tuple[str, int], list[dict[str, Any]]] = {}
    interesting_names: set[str] = set()
    for row in expected:
        name = PurePosixPath(row["path"]).name
        interesting_names.add(name)
        if row["path"].endswith(".tsfile"):
            by_loose_tsfile.setdefault((name, row["bytes"]), []).append(row)

    sources: dict[tuple[str, str], Path] = {}
    digest_cache: dict[Path, str] = {}
    scanned = 0
    hashed = 0
    for path in iter_candidate_files(candidate_roots, interesting_names):
        scanned += 1
        try:
            size = path.stat().st_size
        except OSError:
            continue
        possible: dict[tuple[str, str], dict[str, Any]] = {}

        parts = path.parts
        for index, component in enumerate(parts):
            if component not in dataset_ids:
                continue
            relative_parts = parts[index + 1 :]
            if not relative_parts:
                continue
            relative_path = PurePosixPath(*relative_parts).as_posix()
            row = by_key.get((component, relative_path))
            if row is not None and row["bytes"] == size:
                possible[(component, relative_path)] = row

        if path.name.endswith(".tsfile"):
            for row in by_loose_tsfile.get((path.name, size), []):
                possible[(row["dataset_id"], row["path"])] = row
        if not possible:
            continue

        digest = digest_cache.get(path)
        if digest is None:
            digest = sha256_file(path)
            digest_cache[path] = digest
            hashed += 1
        for key, row in possible.items():
            if digest == row["sha256"]:
                current = sources.get(key)
                if current is None or str(path) < str(current):
                    sources[key] = path
    return sources, {"candidate_files_scanned": scanned, "candidate_files_hashed": hashed}


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.materialize-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    method = "copy"
    try:
        try:
            clone = subprocess.run(
                ["/bin/cp", "-c", "-p", str(source), str(temporary)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            clone = None
        if clone is not None and clone.returncode == 0:
            method = "apfs_clone"
        else:
            shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise MaterializationError(f"copied file failed SHA-256 verification: {destination}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_file() and not destination.is_symlink() and sha256_file(destination) == expected_sha256:
                return "already_present"
            raise MaterializationError(f"destination appeared with different content: {destination}")
        return method
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_progress(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.write-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _target_ancestry_is_safe(target_root: Path, destination: Path) -> bool:
    """Return false when an existing component below target_root is a symlink."""
    relative = destination.relative_to(target_root)
    current = target_root
    if current.is_symlink():
        return False
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            return False
    return True


def materialize(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    scope_path = args.scope.expanduser().resolve(strict=True)
    receipts_dir = args.receipts_dir.expanduser().resolve(strict=True)
    if not receipts_dir.is_dir():
        raise MaterializationError(f"receipts directory is not a directory: {receipts_dir}")
    target_root = args.target.expanduser().resolve(strict=False)
    progress_path = args.progress_json.expanduser().resolve(strict=False)
    if progress_path == target_root or _is_relative_to(progress_path, target_root):
        raise MaterializationError("progress JSON must be outside the materialized target")
    candidate_roots = normalize_candidate_roots(args.candidate_root, target_root)
    scope, expected = load_contract(scope_path, receipts_dir)
    sources, scan_stats = discover_sources(expected, candidate_roots)

    scope_sha256 = sha256_file(scope_path)
    entries: list[dict[str, Any]] = []
    counts = {
        "already_present": 0,
        "apfs_clone": 0,
        "copy": 0,
        "conflict": 0,
        "missing_source": 0,
        "would_materialize": 0,
    }
    materialized_bytes = 0

    for row in expected:
        key = (row["dataset_id"], row["path"])
        source = sources.get(key)
        destination = target_root / row["dataset_id"] / Path(*PurePosixPath(row["path"]).parts)
        entry = dict(row)
        entry["destination"] = str(destination)
        if source is not None:
            entry["source"] = str(source)

        if not _target_ancestry_is_safe(target_root, destination):
            status = "conflict"
        elif destination.exists() or destination.is_symlink():
            if (
                destination.is_file()
                and not destination.is_symlink()
                and destination.stat().st_size == row["bytes"]
                and sha256_file(destination) == row["sha256"]
            ):
                status = "already_present"
                materialized_bytes += row["bytes"]
            else:
                status = "conflict"
        elif source is None:
            status = "missing_source"
        elif args.dry_run:
            status = "would_materialize"
        else:
            try:
                status = _copy_verified(source, destination, row["sha256"])
                materialized_bytes += row["bytes"]
            except MaterializationError:
                status = "conflict"
        counts[status] += 1
        entry["status"] = status
        entries.append(entry)

    per_dataset: list[dict[str, Any]] = []
    for dataset in scope["timestamp_datasets"]:
        dataset_id = dataset["id"]
        dataset_entries = [entry for entry in entries if entry["dataset_id"] == dataset_id]
        per_dataset.append(
            {
                "dataset_id": dataset_id,
                "expected_files": len(dataset_entries),
                "source_files_found": sum("source" in entry for entry in dataset_entries),
                "materialized_or_existing_files": sum(
                    entry["status"] in {"already_present", "apfs_clone", "copy"}
                    for entry in dataset_entries
                ),
                "conflicts": sum(entry["status"] == "conflict" for entry in dataset_entries),
            }
        )

    if counts["conflict"]:
        overall_status = "blocked_conflicts"
        exit_code = 1
    elif args.dry_run:
        overall_status = "dry_run"
        exit_code = 0
    else:
        overall_status = "local_existing_materialized"
        exit_code = 0
    report = {
        "schema": SCHEMA,
        "status": overall_status,
        "generated_unix_seconds": time.time(),
        "dry_run": bool(args.dry_run),
        "scope": str(scope_path),
        "scope_sha256": scope_sha256,
        "scope_dataset_count": len(scope["timestamp_datasets"]),
        "receipts_dir": str(receipts_dir),
        "target_root": str(target_root),
        "candidate_roots": [str(path) for path in candidate_roots],
        "expected_files": len(expected),
        "expected_bytes": sum(row["bytes"] for row in expected),
        "source_files_found": len(sources),
        "source_bytes_found": sum(row["bytes"] for row in expected if (row["dataset_id"], row["path"]) in sources),
        "materialized_or_existing_bytes": materialized_bytes,
        "scan": scan_stats,
        "counts": counts,
        "datasets": per_dataset,
        "files": entries,
    }
    _write_progress(args.progress_json, report)
    return report, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--receipts-dir", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, action="append", type=Path)
    parser.add_argument("--progress-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, exit_code = materialize(args)
    except (MaterializationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("schema", "status", "expected_files", "source_files_found", "counts")}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
