#!/usr/bin/env python3
"""Publish a verified legacy train/validation artifact with explicit split roles.

Only split_contract, missing entity.test_rows=0, and provenance_sha256 may
change. Values, masks, source identity, ordering, and split boundaries remain
unchanged. The source directory and all previous receipts are preserved.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import prepare_weighted_grpo_timeseries_v1 as preparation


SCHEMA = "weighted-grpo-prepared-role-migration-v1"
RECEIPT_NAME = "migration_receipt.json"


class MigrationError(ValueError):
    """A legacy artifact cannot be upgraded without changing its data contract."""


def _publish_new_directory(source: Path, destination: Path) -> None:
    """Atomically publish without replacing even a concurrently created empty directory."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux" and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(source), os.fsencode(destination), 4)
    else:
        raise MigrationError("platform lacks atomic no-replace directory publication")
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _load_legacy(root: Path) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    path = root / "dataset_manifest.json"
    original_bytes = path.read_bytes()
    manifest = json.loads(original_bytes)
    if not isinstance(manifest, dict):
        raise MigrationError("input manifest must be an object")
    if "split_contract" in manifest:
        raise MigrationError("input already has split_contract; refusing to relabel it")
    verification = preparation.verify_prepared_dataset(root, verify_source=True)
    if (path.read_bytes() != original_bytes
            or verification["provenance_sha256"] != manifest.get("provenance_sha256")):
        raise MigrationError("input manifest changed during legacy verification")
    cohort = preparation.load_cohort_manifest()
    entry = next((row for row in cohort["datasets"]
                  if row["id"] == manifest.get("dataset_id")), None)
    if entry is None or entry["role"] != "train_validation":
        raise MigrationError("input is not a frozen train_validation source")
    expected_binding = {
        "version": preparation.COHORT_VERSION,
        "assignment_sha256": preparation.COHORT_ASSIGNMENT_SHA256,
        "policy_sha256": preparation.COHORT_POLICY_SHA256,
        "role": "train_validation",
    }
    if manifest.get("cohort") != expected_binding:
        raise MigrationError("input has no exact frozen train_validation cohort binding")
    if any(manifest.get(field) != entry[field] for field in ("domain", "source_group")):
        raise MigrationError("input source family/domain differs from the frozen cohort")
    return original_bytes, manifest, verification


def _upgraded_manifest(original: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(original)
    result["split_contract"] = copy.deepcopy(
        preparation.SPLIT_CONTRACTS["train_validation"]
    )
    for entity in result["entities"]:
        selected = int(entity["selected_rows"])
        if (int(entity["train_rows"]) != selected * 4 // 5
                or int(entity["train_rows"]) + int(entity["validation_rows"]) != selected
                or entity.get("test_rows", 0) != 0):
            raise MigrationError("legacy entity counts cannot prove an empty test partition")
        entity.setdefault("test_rows", 0)
    result["provenance_sha256"] = hashlib.sha256(preparation._canonical_json(
        preparation._manifest_fingerprint_basis(result)
    )).hexdigest()
    preparation.validate_prepared_cohort_manifests(
        [result], preparation.load_cohort_manifest(), require_complete_train=False,
    )
    return result


def migrate_prepared(
    input_path: Path, output_path: Path, *, transfer: str = "copy",
) -> dict[str, Any]:
    if transfer not in {"copy", "hardlink"}:
        raise MigrationError("transfer must be copy or hardlink")
    supplied = input_path.expanduser().resolve()
    if not supplied.exists():
        raise MigrationError(f"input does not exist: {supplied}")
    if supplied.is_file() and supplied.name != "dataset_manifest.json":
        raise MigrationError("input file must be named dataset_manifest.json")
    source = supplied if supplied.is_dir() else supplied.parent
    output = output_path.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise MigrationError("output must not exist; original artifacts are never overwritten")
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise MigrationError("input and output must be separate non-nested directories")
    original_bytes, original, legacy_verification = _load_legacy(source)
    upgraded = _upgraded_manifest(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.role-migration.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise MigrationError(f"migration already holds output lock: {lock_path}") from error
    os.close(lock_fd)
    temporary: Path | None = None
    published = False
    try:
        if output.exists():
            raise MigrationError("output appeared before migration")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.migration-", dir=output.parent))
        payloads = []
        for block in original["blocks"]:
            for prefix in ("values", "missing_bitmap"):
                relative = str(block[f"{prefix}_path"])
                source_file = preparation._verified_child(source, relative)
                expected_bytes = int(block[f"{prefix}_bytes"])
                expected_hash = str(block[f"{prefix}_sha256"])
                if (not source_file.is_file() or source_file.stat().st_size != expected_bytes
                        or preparation.sha256_file(source_file) != expected_hash):
                    raise MigrationError(f"source payload changed before transfer: {relative}")
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if transfer == "hardlink":
                    os.link(source_file, destination)
                else:
                    shutil.copyfile(source_file, destination)
                if (destination.stat().st_size != expected_bytes
                        or preparation.sha256_file(destination) != expected_hash):
                    raise MigrationError(f"payload changed during transfer: {relative}")
                payloads.append({"path": relative, "bytes": expected_bytes,
                                 "sha256": expected_hash, "transfer": transfer})
        new_manifest_bytes = _json_bytes(upgraded)
        (temporary / "dataset_manifest.json").write_bytes(new_manifest_bytes)
        # Preserve exact prior manifest bytes outside the new provenance basis.
        audit_dir = temporary / "migration"
        audit_dir.mkdir()
        (audit_dir / "input_dataset_manifest.json").write_bytes(original_bytes)
        strict_verification = preparation.verify_prepared_dataset(
            temporary, verify_source=True, expected_role="train_validation",
        )
        if (source / "dataset_manifest.json").read_bytes() != original_bytes:
            raise MigrationError("source manifest changed during migration")
        receipt = {
            "schema": SCHEMA, "status": "verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": original["dataset_id"],
            "input": {
                "path": str(source),
                "manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "provenance_sha256": original["provenance_sha256"],
            },
            "output": {
                "path": str(output),
                "manifest_sha256": hashlib.sha256(new_manifest_bytes).hexdigest(),
                "provenance_sha256": upgraded["provenance_sha256"],
            },
            "allowed_changes": ["split_contract", "entities[].test_rows=0",
                                "provenance_sha256"],
            "source_directory_preserved": True,
            "source_and_blocks_unchanged": True,
            "implementation": {
                "migration_script_sha256": preparation.sha256_file(Path(__file__).resolve()),
                "prepared_verifier_sha256": preparation.sha256_file(
                    Path(preparation.__file__).resolve()
                ),
                "publication": "atomic_rename_no_replace",
            },
            "legacy_verification": legacy_verification,
            "strict_verification": strict_verification,
            "payloads": payloads,
            "payloads_sha256": hashlib.sha256(preparation._canonical_json(payloads)).hexdigest(),
            "payload_bytes": sum(row["bytes"] for row in payloads),
        }
        (temporary / RECEIPT_NAME).write_bytes(_json_bytes(receipt))
        if output.exists():
            raise MigrationError("output appeared before atomic publication")
        _publish_new_directory(temporary, output)
        published = True
        return receipt
    finally:
        if temporary is not None and not published:
            shutil.rmtree(temporary)
        lock_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transfer", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    try:
        receipt = migrate_prepared(args.input, args.output, transfer=args.transfer)
    except (MigrationError, preparation.PreparationError, OSError,
            ValueError, KeyError, TypeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({
        "schema": SCHEMA, "status": receipt["status"],
        "dataset_id": receipt["dataset_id"], "output": receipt["output"],
        "receipt": str(Path(receipt["output"]["path"]) / RECEIPT_NAME),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
