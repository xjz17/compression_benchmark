#!/usr/bin/env python3
"""Enumerate the complete bounded PV2F physical pipeline space for one column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from physical_operator_runtime_v2 import (
    COMPRESSORS,
    PACKERS,
    TRANSFORMS,
    ACTION_BY_NAME,
    TransformState,
    apply_transform_pipeline,
    decode_frame,
    encode_from_packed,
    pack_values,
    raw_i64,
    registry,
    self_check,
    sha256_bytes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_exact_complete_output(output_dir: Path, args: argparse.Namespace) -> bool:
    """Accept only a complete output produced by these exact runner/runtime bytes."""
    manifest_path = output_dir / "manifest.json"
    measurements_path = output_dir / "pipeline_measurements.csv"
    catalog_path = output_dir / "action_catalog.json"
    if not (manifest_path.is_file() and measurements_path.is_file() and catalog_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (
            len(transform_prefixes(args.max_transforms, tuple(spec.name for spec in TRANSFORMS)))
            * len(PACKERS)
            * len(COMPRESSORS)
        )
        source = manifest["source_sha256"]
        return bool(
            manifest["dataset"] == args.dataset
            and int(manifest["column_index"]) == args.column_index
            and int(manifest["max_transforms"]) == args.max_transforms
            and int(manifest["expected_candidates"]) == expected
            and int(manifest["written_candidates"]) == expected
            and bool(manifest["all_candidates_accounted"])
            and source["runner"] == sha256_file(Path(__file__).resolve())
            and source["runtime"]
            == sha256_file(Path(__file__).with_name("physical_operator_runtime_v2.py"))
            and source["pipeline_measurements"] == sha256_file(measurements_path)
            and source["action_catalog"] == sha256_file(catalog_path)
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
        repo_root / "orignial_datasets" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data_path = next((path for path in data_candidates if path.is_file()), None)
    info_path = next((path for path in info_candidates if path.is_file()), None)
    if data_path is None or info_path is None:
        raise FileNotFoundError(f"missing data or column metadata for {dataset}")
    return data_path, info_path


def normalize_column(series: pd.Series, metadata: dict[str, str]) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    logical_type = str(metadata.get("logical_type", "")).strip().lower()
    decimal_places = int(metadata.get("max_point_number", 0) or 0)
    if logical_type in {"float", "double"} and decimal_places > 0:
        finite = np.rint(finite * (10**decimal_places))
    return np.asarray(np.rint(finite), dtype=np.int64)


def load_column(repo_root: Path, dataset: str, column_index: int, max_values: int) -> tuple[np.ndarray, dict[str, str], Path, Path]:
    data_path, info_path = locate_dataset(repo_root, dataset)
    header = pd.read_csv(data_path, nrows=0)
    if column_index < 0 or column_index >= len(header.columns):
        raise KeyError(f"column index {column_index} not found in {data_path}")
    column_name = str(header.columns[column_index])
    metadata_by_name = {row["column_name"]: row for row in read_csv(info_path)}
    matched_metadata = metadata_by_name.get(column_name)
    metadata = dict(matched_metadata or {})
    metadata.update({"column_index": str(column_index), "column_name": column_name})
    # The benchmark CSVs contain deterministic epoch columns that replace the
    # raw Air Date/Time pair and the raw Metro date_time string.  Their current
    # column_info snapshots still describe the pre-normalization schema.
    if column_name in {"timestamp", "date_time"}:
        metadata["logical_type"] = "integer"
        metadata["max_point_number"] = "0"
    elif matched_metadata is None:
        raise KeyError(f"column {column_name!r} has no metadata in {info_path}")
    logical_type = str(metadata.get("logical_type", "")).strip().lower()
    if logical_type not in {"integer", "int", "long", "float", "double"}:
        raise ValueError(f"column {column_index} has unsupported logical type {logical_type!r}")
    frame = pd.read_csv(data_path, usecols=[column_name], nrows=max_values or None, low_memory=False)
    values = normalize_column(frame[column_name], metadata)
    if values.size == 0:
        raise ValueError("normalized column is empty")
    return values, metadata, data_path, info_path


def transform_prefixes(max_transforms: int, transform_names: tuple[str, ...]) -> list[tuple[str, ...]]:
    prefixes: list[tuple[str, ...]] = [()]
    for depth in range(1, max_transforms + 1):
        prefixes.extend(itertools.permutations(transform_names, depth))
    return prefixes


def failure_row(
    dataset: str,
    column_index: int,
    column_name: str,
    logical_type: str,
    values: np.ndarray,
    transforms: tuple[str, ...],
    packer: str,
    compressor: str,
    error: Exception,
) -> dict[str, object]:
    pipeline = transforms + (packer, compressor)
    return {
        "dataset": dataset,
        "column_index": column_index,
        "column_name": column_name,
        "logical_type": logical_type,
        "sample_values": int(values.size),
        "original_bytes": int(values.size * 8),
        "input_sha256": sha256_bytes(raw_i64(values)),
        "transform_count": len(transforms),
        "transform_pipeline": " -> ".join(transforms),
        "packer": packer,
        "compressor": compressor,
        "pipeline": " -> ".join(pipeline),
        "outer_header_bytes": "",
        "action_table_bytes": "",
        "metadata_length_prefix_bytes": "",
        "transform_metadata_bytes": "",
        "packer_payload_bytes": "",
        "compressed_payload_bytes": "",
        "outer_frame_bytes": "",
        "checksum_bytes": "",
        "final_bytes": "",
        "compression_rate": "",
        "encoded_sha256": "",
        "lossless": 0,
        "status": f"{type(error).__name__}:{str(error)[:240]}",
        "elapsed_ms": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--column-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-values", type=int, default=0)
    parser.add_argument("--max-transforms", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--skip-runtime-self-check", action="store_true")
    args = parser.parse_args()
    if args.column_index < 0 or args.max_values < 0 or args.progress_every <= 0:
        parser.error("column-index and max-values must be non-negative; progress-every must be positive")
    return args


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if is_exact_complete_output(output_dir, args):
            print(
                json.dumps(
                    {
                        "reused_complete_output": str(output_dir),
                        "dataset": args.dataset,
                        "column_index": args.column_index,
                        "max_transforms": args.max_transforms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    runtime_check = {"status": "SKIPPED"} if args.skip_runtime_self_check else self_check(repo_root)
    values, metadata, data_path, info_path = load_column(
        repo_root, args.dataset, args.column_index, args.max_values
    )
    transform_names = tuple(spec.name for spec in TRANSFORMS)
    packer_names = tuple(spec.name for spec in PACKERS)
    compressor_names = tuple(spec.name for spec in COMPRESSORS)
    prefixes = transform_prefixes(args.max_transforms, transform_names)
    expected_rows = len(prefixes) * len(packer_names) * len(compressor_names)
    measurements_path = output_dir / "pipeline_measurements.csv"
    fieldnames = list(
        failure_row(
            args.dataset,
            args.column_index,
            metadata["column_name"],
            metadata["logical_type"],
            values,
            (),
            packer_names[0],
            compressor_names[0],
            RuntimeError("schema"),
        )
    )
    completed = 0
    lossless = 0
    failures = 0
    best: dict[str, object] | None = None
    with measurements_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for prefix_index, transforms in enumerate(prefixes):
            try:
                state = apply_transform_pipeline(values, transforms)
                state_error: Exception | None = None
            except Exception as exc:
                state = TransformState(np.asarray([], dtype=np.int64), ())
                state_error = exc
            for packer_name in packer_names:
                if state_error is None:
                    try:
                        packed = pack_values(state.values, ACTION_BY_NAME[packer_name])
                        packer_error: Exception | None = None
                    except Exception as exc:
                        packed = b""
                        packer_error = exc
                else:
                    packed = b""
                    packer_error = state_error
                for compressor_name in compressor_names:
                    candidate_started = time.perf_counter_ns()
                    if packer_error is not None:
                        row = failure_row(
                            args.dataset,
                            args.column_index,
                            metadata["column_name"],
                            metadata["logical_type"],
                            values,
                            transforms,
                            packer_name,
                            compressor_name,
                            packer_error,
                        )
                        failures += 1
                    else:
                        try:
                            encoded = encode_from_packed(
                                values, state, packer_name, compressor_name, packed
                            )
                            restored, decoded_pipeline = decode_frame(encoded.data)
                            if decoded_pipeline != encoded.pipeline or not np.array_equal(restored, values):
                                raise AssertionError("decoder-only round trip mismatch")
                            components = encoded.components
                            if components["outer_frame_bytes"] + components["compressed_payload_bytes"] != components["final_bytes"]:
                                raise AssertionError("component byte sum mismatch")
                            row = {
                                "dataset": args.dataset,
                                "column_index": args.column_index,
                                "column_name": metadata["column_name"],
                                "logical_type": metadata["logical_type"],
                                "sample_values": int(values.size),
                                "original_bytes": int(values.size * 8),
                                "input_sha256": sha256_bytes(raw_i64(values)),
                                "transform_count": len(transforms),
                                "transform_pipeline": " -> ".join(transforms),
                                "packer": packer_name,
                                "compressor": compressor_name,
                                "pipeline": " -> ".join(encoded.pipeline),
                                **components,
                                "compression_rate": components["final_bytes"] / float(values.size * 8),
                                "encoded_sha256": sha256_bytes(encoded.data),
                                "lossless": 1,
                                "status": "ok",
                                "elapsed_ms": (time.perf_counter_ns() - candidate_started) / 1_000_000.0,
                            }
                            lossless += 1
                            if best is None or (int(row["final_bytes"]), str(row["pipeline"])) < (int(best["final_bytes"]), str(best["pipeline"])):
                                best = row.copy()
                        except Exception as exc:
                            row = failure_row(
                                args.dataset,
                                args.column_index,
                                metadata["column_name"],
                                metadata["logical_type"],
                                values,
                                transforms,
                                packer_name,
                                compressor_name,
                                exc,
                            )
                            failures += 1
                    writer.writerow(row)
                    completed += 1
                    if completed % args.progress_every == 0 or completed == expected_rows:
                        print(
                            json.dumps(
                                {
                                    "dataset": args.dataset,
                                    "column_index": args.column_index,
                                    "completed": completed,
                                    "expected": expected_rows,
                                    "lossless": lossless,
                                    "failures": failures,
                                    "prefix": prefix_index + 1,
                                    "prefixes": len(prefixes),
                                    "elapsed_sec": time.perf_counter() - started,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
    if completed != expected_rows or best is None:
        raise AssertionError((completed, expected_rows, best))
    registry_path = output_dir / "action_catalog.json"
    registry_path.write_text(json.dumps(registry(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "elapsed_sec": time.perf_counter() - started,
        "evaluation_type": "real_gt_normalized_int64_physical_prototype",
        "dataset": args.dataset,
        "column_index": args.column_index,
        "column_name": metadata["column_name"],
        "logical_type": metadata["logical_type"],
        "sample_values": int(values.size),
        "original_bytes": int(values.size * 8),
        "input_sha256": sha256_bytes(raw_i64(values)),
        "max_transforms": args.max_transforms,
        "no_repeated_exact_transform": True,
        "transform_prefixes": len(prefixes),
        "transform_variants": len(transform_names),
        "packer_variants": len(packer_names),
        "compressor_variants": len(compressor_names),
        "expected_candidates": expected_rows,
        "written_candidates": completed,
        "lossless_candidates": lossless,
        "failed_candidates": failures,
        "all_candidates_accounted": completed == expected_rows,
        "byte_objective": "final PV2F bytes / normalized int64 bytes; includes transform metadata, action table, outer frame, inner compressor frame, and mandatory outer CRC32",
        "best": best,
        "runtime_self_check": runtime_check,
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "runtime": sha256_file(Path(__file__).with_name("physical_operator_runtime_v2.py")),
            "data_csv": sha256_file(data_path),
            "column_info_csv": sha256_file(info_path),
            "action_catalog": sha256_file(registry_path),
            "pipeline_measurements": sha256_file(measurements_path),
        },
        "claim_boundary": "standalone normalized-int64 research wire format; no production kernel, random-access, query, or original-CSV missing-value claim",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"finished": manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
