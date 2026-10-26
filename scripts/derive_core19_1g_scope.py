#!/usr/bin/env python3
"""Derive the current 19-dataset, whole-column 1 GiB active input view.

The source Paper20 preview remains immutable.  This utility copies the 19
selected dataset manifests byte-for-byte, removes the retired
``poisson_timestamps`` dataset at the cohort level, and emits a new top
manifest and audit receipt.  The chunk objects referenced by the dataset
manifests are reused and can optionally be verified in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


EXPECTED_SOURCE_TOP_SHA256 = (
    "14430177a4539011f028f2800271e19d993d6f16e4b5d80fbc1f3bd6f4b8b09a"
)


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


def copy_immutable(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"source manifest hash mismatch: {source}")
    if destination.is_file():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"refusing to replace conflicting manifest: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def verify_chunks(dataset_manifest: dict[str, Any]) -> tuple[int, int]:
    checked_files = 0
    checked_bytes = 0
    for column in dataset_manifest.get("columns", []):
        for chunk in column.get("chunks", []):
            path = Path(str(chunk["path"]))
            expected_bytes = int(chunk["bytes"])
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"chunk size or presence mismatch: {path}")
            if sha256_file(path) != str(chunk["sha256"]):
                raise ValueError(f"chunk hash mismatch: {path}")
            checked_files += 1
            checked_bytes += expected_bytes
    return checked_files, checked_bytes


def derive(
    source_top_path: Path,
    scope_path: Path,
    output: Path,
    *,
    expected_source_top_sha256: str,
    verify_all_chunks: bool,
) -> dict[str, Any]:
    source_top_path = source_top_path.resolve()
    scope_path = scope_path.resolve()
    output = output.resolve()
    if output in {Path("/"), Path.home().resolve(), source_top_path.parent}:
        raise ValueError(f"unsafe output directory: {output}")

    source_top_sha256 = sha256_file(source_top_path)
    if expected_source_top_sha256 and source_top_sha256 != expected_source_top_sha256:
        raise ValueError(
            "unexpected source top manifest: "
            f"{source_top_sha256} != {expected_source_top_sha256}"
        )
    source_top = json.loads(source_top_path.read_text(encoding="utf-8"))
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope_sha256 = sha256_file(scope_path)
    datasets = list(map(str, scope.get("core_csv_datasets", [])))
    if len(datasets) != 19 or len(set(datasets)) != 19:
        raise ValueError("scope must contain exactly 19 distinct core CSV datasets")
    if "poisson_timestamps" in datasets:
        raise ValueError("retired poisson_timestamps remains in the core scope")

    source_entries = {
        str(entry["dataset"]): entry for entry in source_top.get("datasets", [])
    }
    missing = [dataset for dataset in datasets if dataset not in source_entries]
    if missing:
        raise ValueError(f"source top manifest is missing datasets: {missing}")

    source_selection = source_top["identity"]["preview_selection"]["datasets"]
    limit_bytes = int(source_top["preview_protocol"]["limit_bytes"])
    selected_entries: list[dict[str, Any]] = []
    totals = {
        "blocking_source_numeric_failures": 0,
        "chunks": 0,
        "provenance_only_nonready_entries": 0,
        "raw_bytes": 0,
        "ready_columns": 0,
        "retained_rows": 0,
    }
    verified_chunk_count = 0
    verified_chunk_bytes = 0
    trimmed_datasets: list[str] = []

    for dataset in datasets:
        entry = dict(source_entries[dataset])
        selection = source_selection[dataset]
        if int(selection["selected_bytes"]) > limit_bytes:
            raise ValueError(f"{dataset} exceeds the {limit_bytes}-byte active cap")
        selected_columns = int(selection["selected_columns"])
        selected_source_indices = list(map(int, selection["selected_source_indices"]))
        if selected_source_indices != list(range(selected_columns)):
            raise ValueError(f"{dataset} selection is not a whole-column prefix")
        if int(selection["selected_columns"]) < int(selection["source_columns"]):
            trimmed_datasets.append(dataset)

        source_dataset_manifest = (
            source_top_path.parent / "datasets" / dataset / "manifest.json"
        )
        destination_manifest = output / "datasets" / dataset / "manifest.json"
        copy_immutable(
            source_dataset_manifest,
            destination_manifest,
            str(entry["manifest_sha256"]),
        )
        dataset_manifest = json.loads(
            source_dataset_manifest.read_text(encoding="utf-8")
        )
        manifest_indices = [
            int(column["source_index"]) for column in dataset_manifest.get("columns", [])
        ]
        if manifest_indices != selected_source_indices:
            raise ValueError(f"{dataset} dataset manifest disagrees with top selection")
        manifest_raw_bytes = sum(
            int(column["raw_bytes"]) for column in dataset_manifest.get("columns", [])
        )
        if manifest_raw_bytes != int(selection["selected_bytes"]):
            raise ValueError(f"{dataset} selected-byte total mismatch")
        if verify_all_chunks:
            count, size = verify_chunks(dataset_manifest)
            verified_chunk_count += count
            verified_chunk_bytes += size

        entry["manifest_path"] = str(destination_manifest)
        selected_entries.append(entry)
        for key in totals:
            totals[key] += int(entry["totals"].get(key, 0))

    selection = {dataset: source_selection[dataset] for dataset in datasets}
    derived_scope = {
        "schema": "core19-1g-column-prefix-derived-scope-v1",
        "scope_id": scope["scope_id"],
        "scope_path": str(scope_path),
        "scope_sha256": scope_sha256,
        "source_top_manifest_path": str(source_top_path),
        "source_top_manifest_sha256": source_top_sha256,
        "excluded_datasets": ["poisson_timestamps"],
        "trimmed_dataset_count": len(trimmed_datasets),
        "trimmed_datasets": trimmed_datasets,
        "deletion_semantics": (
            "columns after the longest whole-column prefix at or below 1 GiB "
            "are absent from the active manifest; immutable source bytes remain "
            "available only for provenance and historical result reproducibility"
        ),
    }
    producer = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "argv": list(sys.argv),
    }

    identity = dict(source_top["identity"])
    identity["datasets"] = datasets
    identity["preview_selection"] = dict(identity["preview_selection"])
    identity["preview_selection"]["datasets"] = selection
    identity["derived_scope"] = derived_scope

    preview_protocol = dict(source_top["preview_protocol"])
    preview_protocol["selection"] = selection
    preview_protocol["derived_scope"] = derived_scope

    top = dict(source_top)
    top["identity"] = identity
    top["preview_protocol"] = preview_protocol
    top["datasets"] = selected_entries
    top["totals"] = totals
    top["derived_scope"] = derived_scope
    atomic_json(output / "manifest.json", top)
    atomic_json(output / "PREVIEW_PROTOCOL.json", preview_protocol)

    receipt = {
        "schema": "core19-1g-column-prefix-source-contract-v1",
        "status": "complete",
        "active_manifest_path": str(output / "manifest.json"),
        "active_manifest_sha256": sha256_file(output / "manifest.json"),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "selection": selection,
        "totals": totals,
        "derived_scope": derived_scope,
        "dataset_manifest_hashes": {
            entry["dataset"]: entry["manifest_sha256"] for entry in selected_entries
        },
        "chunk_verification": {
            "performed": verify_all_chunks,
            "checked_files": verified_chunk_count,
            "checked_bytes": verified_chunk_bytes,
        },
        "producer": producer,
    }
    atomic_json(output / "SOURCE_CONTRACT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-source-top-sha256", default=EXPECTED_SOURCE_TOP_SHA256
    )
    parser.add_argument("--verify-all-chunks", action="store_true")
    args = parser.parse_args()
    receipt = derive(
        args.source_top,
        args.scope,
        args.output,
        expected_source_top_sha256=args.expected_source_top_sha256,
        verify_all_chunks=args.verify_all_chunks,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
