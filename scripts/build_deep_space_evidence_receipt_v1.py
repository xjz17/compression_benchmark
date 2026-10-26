#!/usr/bin/env python3
"""Build a verified, path-portable evidence receipt for deep-space-probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping, Sequence


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rebuild_weighted_grpo_deep_space_v1 as deep_space  # noqa: E402
from prepare_weighted_grpo_timeseries_v1 import (  # noqa: E402
    PreparationError,
    verify_prepared_dataset,
)


SCHEMA = "weighted-grpo-deep-space-rebuild-receipts-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PreparationError(f"JSON document must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_receipt(
    *,
    spec_path: Path,
    audit_path: Path,
    canonical_root: Path,
    prepared_root: Path,
    output_path: Path,
    repository_root: Path,
) -> Mapping[str, Any]:
    spec_path = spec_path.resolve()
    audit_path = audit_path.resolve()
    canonical_root = canonical_root.resolve()
    prepared_root = prepared_root.resolve()
    repository_root = repository_root.resolve()
    contract = deep_space.load_spec(spec_path)
    source_evidence = deep_space.validate_audit_receipt(contract, audit_path)
    audit_document = _load(audit_path)
    canonical_verification = deep_space.verify_canonical(
        canonical_root, verify_source=True
    )
    prepared_verification = verify_prepared_dataset(
        prepared_root, verify_source=True
    )
    canonical_manifest_path = canonical_root / "canonical_manifest.json"
    prepared_manifest_path = prepared_root / "dataset_manifest.json"
    canonical = _load(canonical_manifest_path)
    prepared = _load(prepared_manifest_path)
    if canonical.get("preparation_spec_sha256") != contract.spec_sha256:
        raise PreparationError("canonical artifact is bound to a different formal contract")
    if prepared.get("preparation_spec_sha256") != contract.spec_sha256:
        raise PreparationError("prepared artifact is bound to a different formal contract")
    if prepared_verification.get("provenance_sha256") != prepared.get(
        "provenance_sha256"
    ):
        raise PreparationError("prepared provenance verification mismatch")

    def repository_relative(path: Path) -> str:
        try:
            return str(path.relative_to(repository_root))
        except ValueError as error:
            raise PreparationError(
                f"evidence path is outside repository root: {path}"
            ) from error

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_date": "2026-09-17",
        "status": "passed",
        "dataset_id": deep_space.DATASET_ID,
        "formal_use_boundary": (
            "This receipt proves one completed exact-upstream rebuild and preparation. "
            "A formal run must independently verify its own prepared directory and bind "
            "that run's manifest and provenance hashes; filesystem paths are not source identity."
        ),
        "source_overlay": {
            "repo_id": deep_space.UPSTREAM_REPO,
            "revision": deep_space.UPSTREAM_REVISION,
            "repo_path": deep_space.UPSTREAM_REPO_PATH,
            "bytes": deep_space.SOURCE_BYTES,
            "sha256": deep_space.SOURCE_SHA256,
            "git_oid": deep_space.SOURCE_GIT_OID,
            "readme_bytes": deep_space.README_BYTES,
            "readme_sha256": deep_space.README_SHA256,
        },
        "invalid_mirror_rejection": {
            "repo_id": "BenchmarkHost/deep-space-probes",
            "revision": "2ce6a7c5b0e22c3b6aa19459e4fa16a433f7bc8c",
            "policy": "never use, rescale, repair, or admit the frozen TsFile mirror",
            "reasons": [
                "mirror Time values are Unix seconds while its frozen card claims milliseconds",
                "the readable numeric query path omits timestamp-only rows",
                "the readable numeric row count 978408 differs from the upstream 1183368 rows",
            ],
        },
        "formal_spec": {
            "path": repository_relative(spec_path),
            "file_sha256": _sha256(spec_path),
            "contract_sha256": contract.spec_sha256,
        },
        "source_audit": {
            "path": repository_relative(audit_path),
            "file_sha256": _sha256(audit_path),
            "rows": source_evidence["rows"],
            "timestamp_only_rows": source_evidence["timestamp_only_rows"],
            "numeric_schema_sha256": audit_document["numeric_schema_sha256"],
        },
        "verified_local_canonical_rebuild": {
            "manifest_file_sha256": _sha256(canonical_manifest_path),
            "provenance_sha256": canonical_verification["provenance_sha256"],
            "blocks": canonical["totals"]["blocks"],
            "rows": canonical_verification["rows"],
            "timestamp_only_rows": canonical_verification["timestamp_only_rows"],
            "original_numeric_bytes": canonical["totals"]["original_numeric_bytes"],
            "source_verified": True,
        },
        "verified_local_prepared_artifact": {
            "manifest_file_sha256": _sha256(prepared_manifest_path),
            "provenance_sha256": prepared_verification["provenance_sha256"],
            "blocks": prepared["totals"]["blocks"],
            "source_rows": prepared["selection"]["source_rows"],
            "selected_rows": prepared["selection"]["selected_rows"],
            "original_numeric_bytes": prepared["totals"]["original_numeric_bytes"],
            "prepared_block_bytes": prepared["totals"]["prepared_block_bytes"],
            "source_verified": True,
            "split": prepared["selection"]["split"],
        },
    }
    _atomic_json(output_path.resolve(), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=SCRIPTS.parent)
    args = parser.parse_args(argv)
    try:
        receipt = build_receipt(
            spec_path=args.spec,
            audit_path=args.audit,
            canonical_root=args.canonical,
            prepared_root=args.prepared,
            output_path=args.output,
            repository_root=args.repository_root,
        )
    except (PreparationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DEEP_SPACE_RECEIPT_BLOCKER: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
