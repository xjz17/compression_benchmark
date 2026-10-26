#!/usr/bin/env python3
"""Deduplicate registered actions and method-scoped C++ candidates by ratio vector."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np

import primitive_operator_action_runtime as primitive


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_outputs/cpp_codec_action_split_20260902_v1"
COLUMNS = {
    "Metro-Traffic": (
        "temp", "rain_1h", "snow_1h", "clouds_all", "date_time", "traffic_volume",
    ),
    "Air-Quality": (
        "timestamp", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
        "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    ),
}
RLE_INTEGER_COLUMNS = (
    ("Metro-Traffic", "clouds_all"),
    ("Metro-Traffic", "date_time"),
    ("Metro-Traffic", "traffic_volume"),
    ("Air-Quality", "timestamp"),
    ("Air-Quality", "PT08.S1(CO)"),
    ("Air-Quality", "NMHC(GT)"),
    ("Air-Quality", "PT08.S2(NMHC)"),
    ("Air-Quality", "NOx(GT)"),
    ("Air-Quality", "PT08.S3(NOx)"),
    ("Air-Quality", "NO2(GT)"),
    ("Air-Quality", "PT08.S4(NO2)"),
    ("Air-Quality", "PT08.S5(O3)"),
)
METHOD_NAMES = {
    "Chimp": "CHIMP", "Gorilla": "GORILLA", "2DIFF": "TS_2DIFF",
    "SVB": "Stream VByte", "FastPFor": "fastpfor", "Simple8b": "simple8b",
    "Sprintz": "SPRINTZ", "DBP": "DELTA_BINARY_PACKED", "Camel": "camel",
    "Elf": "elf", "Elf+": "elf_plus", "FPC": "fpc", "PFPC": "pfpc",
    "SPDP": "spdp", "BUFF": "buff", "Zstd": "zstd", "Snappy": "snappy",
    "Zlib": "zlib",
}
OMISSION_NAMES = {
    "frame_of_reference": "for", "bitpack_int": "bit-packing",
    "value_reorder": "value reorder", "regression_predictor": "regression",
    "integer_code": "integer coding", "run_length": "run length",
    "block_transform": "block transform", "entropy_special": "special code",
    "entropy_range": "range code", "container_frame": "container framing",
    "entropy_huffman": "huffman code", "literal_entropy": "literal code",
    "entropy_fse": "fse/ans code", "block_state_control": "block/state",
}


def preferred_implementation(method: str, operator_id: str) -> str | None:
    if method in {"gzip", "Zlib"}:
        return (
            "native_streams_upstream"
            if operator_id == "container_frame"
            else "native_deflate_operator_upstream"
        )
    if method == "Zstd":
        return "native_zstd_operator_upstream"
    if method == "LZ4":
        return (
            "native_streams_upstream"
            if operator_id == "container_frame"
            else "native_lz4_operator_upstream"
        )
    if method == "Snappy":
        return "native_snappy_operator_upstream"
    if method == "Blosc":
        return "native_blosc_operator_upstream"
    if method == "Blosc2":
        return "native_blosc2_operator_upstream"
    if method == "Bzip2":
        return "project_patched_bzip2_source_ablation"
    return None


def candidate_row_key(candidate: dict[str, str]) -> tuple[str, str, str | None]:
    method = candidate["method"]
    operator_id = candidate["operator_id"]
    benchmark_method = METHOD_NAMES.get(method, method)
    omission = operator_id if method == "Bzip2" else OMISSION_NAMES.get(operator_id, operator_id)
    return benchmark_method, omission, preferred_implementation(method, operator_id)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_measurement(
    rows: list[dict[str, str]], candidate: dict[str, str]
) -> tuple[int, int]:
    method, omission, implementation = candidate_row_key(candidate)
    matches = [
        row for row in rows
        if row["method"] == method
        and row["omitted_operator"] == omission
        and (implementation is None or row["implementation"] == implementation)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one benchmark row for {candidate['action_name']}, got {len(matches)}"
        )
    row = matches[0]
    if row["lossless"] != "1":
        raise RuntimeError(f"non-lossless candidate row: {candidate['action_name']}")
    return int(row["original_bytes"]), int(row["compressed_bytes"])


def ratio(value: tuple[int, int]) -> Fraction | None:
    original, compressed = value
    if original == 0 and compressed == 0:
        return None
    if original <= 0:
        raise RuntimeError("candidate measurement has compressed bytes but no input bytes")
    return Fraction(compressed, original)


def single_column_benchmark(
    executable: Path, dataset: str, column: str
) -> tuple[str, list[dict[str, str]]]:
    with tempfile.TemporaryDirectory(prefix="wc_action_dedup_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "column.csv"
        output_path = tmp_path / "results.csv"
        with (ROOT / "orignial_datasets" / f"{dataset}.csv").open(
            newline="", encoding="utf-8"
        ) as source, input_path.open("w", newline="", encoding="utf-8") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=["value"])
            writer.writeheader()
            for row in reader:
                writer.writerow({"value": row[column]})
        completed = subprocess.run(
            [str(executable), str(input_path), str(output_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"single-column benchmark failed for {dataset}:{column}: {completed.stderr}"
            )
        rows = load_rows(output_path)
        if len(rows) != 304 or any(row["lossless"] != "1" for row in rows):
            raise RuntimeError(f"invalid 304-row benchmark for {dataset}:{column}")
        return f"{dataset}:{column}", rows


def audit_integer_rle_implementations() -> dict[str, object]:
    """Compare the two remaining RLE wires only on integer-input columns."""

    reference_bytes: list[int] = []
    tsfile_bytes: list[int] = []
    for dataset, column in RLE_INTEGER_COLUMNS:
        with (ROOT / "orignial_datasets" / f"{dataset}.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            values = np.asarray(
                [int(row[column]) for row in csv.DictReader(handle)], dtype="<i8"
            )
        reference = primitive.encode_operator_pipeline(
            values, ("op.run_length.reference_v1",)
        )
        tsfile = primitive.encode_registered_exact_action_pipeline(
            values.tobytes(),
            ("op.run_length.tsfile_int64.v1",),
            logical_type="int",
        )
        decoded_reference, decoded_actions = primitive.decode_operator_pipeline(reference)
        if decoded_actions != ("op.run_length.reference_v1",) or not np.array_equal(
            decoded_reference, values
        ):
            raise RuntimeError(f"reference RLE round-trip failed: {dataset}:{column}")
        if primitive.decode_rle_int64(tsfile) != values.tolist():
            raise RuntimeError(f"TsFile int64 RLE round-trip failed: {dataset}:{column}")
        reference_bytes.append(len(reference))
        tsfile_bytes.append(len(tsfile))
    return {
        "columns": [f"{dataset}:{column}" for dataset, column in RLE_INTEGER_COLUMNS],
        "reference_v1_compressed_bytes": reference_bytes,
        "tsfile_int64_v1_compressed_bytes": tsfile_bytes,
        "equal_ratio_columns": sum(
            left == right for left, right in zip(reference_bytes, tsfile_bytes)
        ),
        "ratio_vectors_equal": reference_bytes == tsfile_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=BASE / "cpp_action_candidates.csv")
    parser.add_argument("--air-results", type=Path, default=BASE / "operator_ablation_air.csv")
    parser.add_argument("--metro-results", type=Path, default=BASE / "operator_ablation_metro.csv")
    parser.add_argument("--ans-audit", type=Path, default=BASE / "ans_cpp_python_wire_audit.json")
    parser.add_argument("--operator-bench", type=Path, default=ROOT / "cpp/build/operator_ablation_bench")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = load_rows(args.catalog)
    if len(candidates) != 106:
        raise RuntimeError(f"expected 106 raw candidates, got {len(candidates)}")
    measurable = [row for row in candidates if row["evidence"] == "codec_ablation_cpp"]
    if len(measurable) != 94:
        raise RuntimeError(f"expected 94 codec-ablation candidates, got {len(measurable)}")

    aggregate_rows = {
        "Air-Quality": load_rows(args.air_results),
        "Metro-Traffic": load_rows(args.metro_results),
    }
    aggregate_groups: dict[tuple[str, tuple[Fraction, ...]], list[dict[str, str]]] = defaultdict(list)
    for candidate in measurable:
        signature = tuple(
            ratio(find_measurement(aggregate_rows[dataset], candidate))
            for dataset in ("Metro-Traffic", "Air-Quality")
        )
        aggregate_groups[(candidate["operator_id"], signature)].append(candidate)
    possible_duplicates = [rows for rows in aggregate_groups.values() if len(rows) > 1]

    column_order = [
        (dataset, column) for dataset, columns in COLUMNS.items() for column in columns
    ]
    per_column_rows: dict[str, list[dict[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(single_column_benchmark, args.operator_bench, dataset, column):
            (dataset, column)
            for dataset, column in column_order
        }
        for future in as_completed(futures):
            key, rows = future.result()
            per_column_rows[key] = rows

    confirmed_groups = []
    for group in possible_duplicates:
        signatures = []
        for candidate in group:
            vector = [
                find_measurement(per_column_rows[f"{dataset}:{column}"], candidate)
                for dataset, column in column_order
            ]
            signatures.append(tuple(ratio(value) for value in vector))
        if len(set(signatures)) == 1:
            confirmed_groups.append({
                "operator_id": group[0]["operator_id"],
                "canonical": group[0]["action_name"],
                "aliases": [row["action_name"] for row in group[1:]],
                "compressed_bytes_20cols": [
                    find_measurement(
                        per_column_rows[f"{dataset}:{column}"], group[0]
                    )[1]
                    for dataset, column in column_order
                ],
            })

    duplicate_candidate_records = sum(len(group["aliases"]) for group in confirmed_groups)
    ans_audit = json.loads(args.ans_audit.read_text(encoding="utf-8"))
    ans_equivalent = (
        ans_audit["columns"] == 20
        and ans_audit["cpp_lossless_columns"] == 20
        and ans_audit["python_lossless_columns"] == 20
        and ans_audit["wire_identical_columns"] == 20
    )
    policy_names = {action.name for action in primitive.POLICY_ACTIONS}
    if "op.entropy_fse.project_rans_bytes.v1" not in policy_names or not ans_equivalent:
        raise RuntimeError("ANS cross-catalog deduplication is not supported by 20/20 evidence")

    removed_float_rle = [
        f"op.run_length.tsfile_float64.maxpoint{point}.v1" for point in (1, 2, 4, 14)
    ]
    if policy_names.intersection(removed_float_rle):
        raise RuntimeError("float-input run_length actions remain registered")
    if "op.run_length.tsfile_int64.v1" not in policy_names:
        raise RuntimeError("integer run_length action is missing")
    rle_integer_audit = audit_integer_rle_implementations()
    if rle_integer_audit["ratio_vectors_equal"]:
        raise RuntimeError("the two remaining integer-input RLE actions should be merged")

    registered_count = len(primitive.POLICY_ACTIONS)
    deduplicated_candidates = len(candidates) - duplicate_candidate_records
    cross_catalog_duplicates = 1
    combined_count = registered_count + deduplicated_candidates - cross_catalog_duplicates
    payload = {
        "schema": "operator-implementation-name-ratio-dedup-v1",
        "identity_rule": (
            "same operator_id and identical exact compression-ratio vector on the fixed 20 columns"
        ),
        "candidate_ratio_definition": (
            "for non-standalone C++ candidates, the vector is the containing codec's compressed "
            "size with that operator omitted; it is a method-scoped ablation signature, not a "
            "standalone action payload"
        ),
        "columns": [f"{dataset}:{column}" for dataset, column in column_order],
        "registered_actions_after_integer_only_rle": registered_count,
        "registered_stop_action_id": primitive.STOP_ACTION_ID,
        "removed_float_input_run_length_actions": removed_float_rle,
        "remaining_run_length_actions": sorted(
            action.name for action in primitive.POLICY_ACTIONS
            if action.operator_id == "run_length"
        ),
        "run_length_integer_input_audit": rle_integer_audit,
        "raw_cpp_candidate_records": len(candidates),
        "measured_codec_ablation_candidates": len(measurable),
        "unmeasured_candidates_kept_distinct": sum(
            row["evidence"] not in {"codec_ablation_cpp", "atomic_cpp"}
            for row in candidates
        ),
        "confirmed_cpp_candidate_duplicate_groups": confirmed_groups,
        "duplicate_cpp_candidate_records_removed": duplicate_candidate_records,
        "deduplicated_cpp_candidate_records": deduplicated_candidates,
        "cross_catalog_duplicate_groups": [{
            "operator_id": "entropy_fse",
            "registered_action": "op.entropy_fse.project_rans_bytes.v1",
            "cpp_candidate": "op.entropy_fse.ans.cpp_v1",
            "evidence": "20/20 lossless and byte-identical encoded payload",
        }],
        "cross_catalog_duplicate_records_removed": cross_catalog_duplicates,
        "combined_method_scoped_implementation_records": combined_count,
        "strict_registered_trainable_actions": registered_count,
        "count_formula": (
            f"{registered_count} + ({len(candidates)} - {duplicate_candidate_records}) "
            f"- {cross_catalog_duplicates} = {combined_count}"
        ),
        "claim_boundary": (
            "the combined count includes method-scoped ablation, proxy, blocked, and unwired "
            "records; only strict_registered_trainable_actions are currently registered actions"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "registered": registered_count,
        "raw_cpp_candidates": len(candidates),
        "cpp_duplicates_removed": duplicate_candidate_records,
        "deduplicated_cpp_candidates": deduplicated_candidates,
        "cross_duplicates_removed": cross_catalog_duplicates,
        "combined": combined_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
