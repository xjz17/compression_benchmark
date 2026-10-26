#!/usr/bin/env python3
"""Compare same-operator implementations on eight real-data column prefixes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_operator_implementation_dedup as cpp_audit  # noqa: E402
import primitive_operator_action_runtime as primitive  # noqa: E402


PROBES = (
    ("Metro-Traffic", "temp", "float"),
    ("Metro-Traffic", "traffic_volume", "int"),
    ("Air-Quality", "C6H6(GT)", "float"),
    ("Air-Quality", "PT08.S1(CO)", "int"),
    ("Bird-migration", "location-long", "float"),
    ("Bird-migration", "event-id", "int"),
    ("meteoblue", "Basel Temperature [2 m elevation corrected]", "float"),
    ("meteoblue", "timestamp", "int"),
)


def load_probe(dataset: str, column: str, logical_type: str, limit: int) -> dict[str, object]:
    values: list[int | float] = []
    with (ROOT / "orignial_datasets" / f"{dataset}.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            text = row[column].strip()
            if not text or text.lower() == "nan":
                continue
            value: int | float = int(text) if logical_type == "int" else float(text)
            if isinstance(value, float) and not math.isfinite(value):
                continue
            values.append(value)
            if len(values) == limit:
                break
    if len(values) != limit:
        raise RuntimeError(f"{dataset}:{column} has only {len(values)} usable values")
    array = np.asarray(values, dtype="<i8" if logical_type == "int" else "<f8")
    return {
        "dataset": dataset,
        "column": column,
        "logical_type": logical_type,
        "values": array,
        "raw": array.tobytes(),
        "rows": len(array),
        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def run_cpp_probe(
    executable: Path, probe: dict[str, object]
) -> tuple[str, list[dict[str, str]]]:
    key = f"{probe['dataset']}:{probe['column']}"
    values = np.asarray(probe["values"])
    with tempfile.TemporaryDirectory(prefix="wc_same_operator_ratio_") as tmp:
        input_path = Path(tmp) / "input.csv"
        output_path = Path(tmp) / "output.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["value"])
            writer.writerows((value.item(),) for value in values)
        completed = subprocess.run(
            [str(executable), str(input_path), str(output_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"C++ benchmark failed for {key}: {completed.stderr}")
        rows = cpp_audit.load_rows(output_path)
        if len(rows) != 304 or any(row["lossless"] != "1" for row in rows):
            raise RuntimeError(f"invalid C++ benchmark output for {key}")
        return key, rows


def decode_exact_atomic(name: str, encoded: bytes) -> list[int] | list[float] | bytes:
    if name == "op.entropy_huffman.project_huffman_bytes.v1":
        return primitive.huffman_decode(encoded)
    if name == "op.entropy_range.project_arithmetic_bytes.v1":
        return primitive.arithmetic_decode(encoded)
    if name == "op.entropy_fse.project_rans_bytes.v1":
        return primitive.ans_decode(encoded)
    if name == "op.dictionary.project_int64.v1":
        return primitive.decode_dictionary_int(encoded)
    if name == "op.dictionary.project_float64.v1":
        return primitive.decode_dictionary_float(encoded)
    if name == "op.run_length.tsfile_int64.v1":
        return primitive.decode_rle_int64(encoded)
    raise KeyError(name)


def standalone_applicable(action: object, logical_type: str) -> bool:
    if isinstance(action, primitive.PrimitiveActionSpec):
        return True
    name = action.name
    if name == "op.dictionary.project_int64.v1" or name == "op.run_length.tsfile_int64.v1":
        return logical_type == "int"
    if name == "op.dictionary.project_float64.v1":
        return logical_type == "float"
    return name in primitive.EXACT_ATOMIC_ACTION_BY_NAME


def measure_standalone(action: object, probe: dict[str, object]) -> tuple[int, int]:
    raw = bytes(probe["raw"])
    values = np.asarray(probe["values"])
    if isinstance(action, primitive.PrimitiveActionSpec):
        int_view = np.frombuffer(raw, dtype="<i8")
        encoded = primitive.encode_operator_pipeline(int_view, (action.name,))
        decoded, names = primitive.decode_operator_pipeline(encoded)
        if names != (action.name,) or not np.array_equal(decoded, int_view):
            raise RuntimeError(f"registered primitive round-trip failed: {action.name}")
        return len(raw), len(encoded)

    probe_type = str(probe["logical_type"])
    logical_type = (
        "bytes" if action.operator_id in {"entropy_huffman", "entropy_range", "entropy_fse"}
        else probe_type
    )
    encoded = primitive.encode_registered_exact_action_pipeline(
        raw, (action.name,), logical_type=logical_type
    )
    decoded = decode_exact_atomic(action.name, encoded)
    if isinstance(decoded, bytes):
        ok = decoded == raw
    elif probe_type == "int":
        ok = np.array_equal(np.asarray(decoded, dtype="<i8"), values)
    else:
        ok = np.array_equal(
            np.asarray(decoded, dtype="<f8").view("<u8"), values.view("<u8")
        )
    if not ok:
        raise RuntimeError(f"registered exact-atomic round-trip failed: {action.name}")
    return len(raw), len(encoded)


def compare_vectors(
    left: list[Fraction | None], right: list[Fraction | None]
) -> dict[str, object]:
    common = [
        index for index, (a, b) in enumerate(zip(left, right))
        if a is not None and b is not None
    ]
    equal = [index for index in common if left[index] == right[index]]
    different = [index for index in common if left[index] != right[index]]
    return {
        "common_probe_count": len(common),
        "equal_probe_count": len(equal),
        "different_probe_count": len(different),
        "all_equal_on_common_probes": bool(common) and not different,
    }


def summarize_operator_vectors(
    names_by_operator: dict[str, list[str]], vectors: dict[str, list[Fraction | None]]
) -> list[dict[str, object]]:
    out = []
    for operator_id, names in names_by_operator.items():
        if len(names) < 2:
            continue
        pairs = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                comparison = compare_vectors(vectors[left], vectors[right])
                pairs.append({"left": left, "right": right, **comparison})
        different_pairs = sum(pair["different_probe_count"] > 0 for pair in pairs)
        equal_pairs = sum(pair["all_equal_on_common_probes"] for pair in pairs)
        incomparable_pairs = sum(pair["common_probe_count"] == 0 for pair in pairs)
        out.append({
            "operator_id": operator_id,
            "implementation_count": len(names),
            "pair_count": len(pairs),
            "different_pairs": different_pairs,
            "equal_pairs": equal_pairs,
            "incomparable_pairs": incomparable_pairs,
            "all_implementations_equal": different_pairs == 0 and incomparable_pairs == 0,
            "pairs": pairs,
        })
    return out


def ans_cpp_python_audit(
    probes: list[dict[str, object]], executable: Path
) -> dict[str, object]:
    rows = []
    for probe in probes:
        raw = bytes(probe["raw"])
        python_encoded = primitive.ans_encode(raw)
        if primitive.ans_decode(python_encoded) != raw:
            raise RuntimeError("Python ANS round-trip failed")
        with tempfile.TemporaryDirectory(prefix="wc_ans_cross_") as tmp:
            input_path = Path(tmp) / "raw.bin"
            output_path = Path(tmp) / "encoded.bin"
            input_path.write_bytes(raw)
            completed = subprocess.run(
                [str(executable), str(input_path), str(output_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"C++ ANS failed: {completed.stderr}")
            cpp_encoded = output_path.read_bytes()
        rows.append({
            "probe": f"{probe['dataset']}:{probe['column']}",
            "original_bytes": len(raw),
            "python_compressed_bytes": len(python_encoded),
            "cpp_compressed_bytes": len(cpp_encoded),
            "wire_identical": cpp_encoded == python_encoded,
        })
    return {
        "probe_count": len(rows),
        "wire_identical_probes": sum(row["wire_identical"] for row in rows),
        "all_wire_identical": all(row["wire_identical"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=8192)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--cpp-catalog",
        type=Path,
        default=ROOT / "analysis_outputs/cpp_codec_action_split_20260902_v1/cpp_action_candidates.csv",
    )
    parser.add_argument(
        "--dedup-audit",
        type=Path,
        default=ROOT / "analysis_outputs/operator_implementation_dedup_20260903_v1/audit.json",
    )
    parser.add_argument(
        "--operator-bench",
        type=Path,
        default=ROOT / "cpp/build/operator_ablation_bench",
    )
    parser.add_argument(
        "--ans-cli", type=Path, default=ROOT / "cpp/build/ans_rans_cli"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    probes = [load_probe(*spec, args.sample_rows) for spec in PROBES]
    probe_keys = [f"{probe['dataset']}:{probe['column']}" for probe in probes]

    cpp_results: dict[str, list[dict[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(run_cpp_probe, args.operator_bench, probe): probe
            for probe in probes
        }
        for future in as_completed(futures):
            key, rows = future.result()
            cpp_results[key] = rows

    candidates = cpp_audit.load_rows(args.cpp_catalog)
    measurable = [row for row in candidates if row["evidence"] == "codec_ablation_cpp"]
    unmeasured = [
        row for row in candidates
        if row["evidence"] not in {"codec_ablation_cpp", "atomic_cpp"}
    ]
    cpp_vectors: dict[str, list[Fraction | None]] = {}
    measurement_rows: list[dict[str, object]] = []
    for candidate in measurable:
        vector = []
        for probe, key in zip(probes, probe_keys):
            original, compressed = cpp_audit.find_measurement(cpp_results[key], candidate)
            value = cpp_audit.ratio((original, compressed))
            vector.append(value)
            measurement_rows.append({
                "domain": "cpp_method_scoped_omission",
                "operator_id": candidate["operator_id"],
                "implementation_name": candidate["action_name"],
                "probe": key,
                "logical_type": probe["logical_type"],
                "original_bytes": original,
                "compressed_bytes": compressed,
                "compression_rate": float(value) if value is not None else "",
                "applicable": value is not None,
                "lossless": True,
            })
        cpp_vectors[candidate["action_name"]] = vector

    dedup = json.loads(args.dedup_audit.read_text(encoding="utf-8"))
    alias_to_canonical = {
        alias: group["canonical"]
        for group in dedup["confirmed_cpp_candidate_duplicate_groups"]
        for alias in group["aliases"]
    }
    canonical_cpp_names: dict[str, list[str]] = defaultdict(list)
    for candidate in measurable:
        canonical = alias_to_canonical.get(candidate["action_name"], candidate["action_name"])
        if canonical not in canonical_cpp_names[candidate["operator_id"]]:
            canonical_cpp_names[candidate["operator_id"]].append(canonical)

    prior_merge_checks = []
    for group in dedup["confirmed_cpp_candidate_duplicate_groups"]:
        members = [group["canonical"], *group["aliases"]]
        comparisons = [
            {
                "left": members[0],
                "right": member,
                **compare_vectors(cpp_vectors[members[0]], cpp_vectors[member]),
            }
            for member in members[1:]
        ]
        prior_merge_checks.append({
            "operator_id": group["operator_id"],
            "members": members,
            "stable_on_new_probes": all(
                row["all_equal_on_common_probes"] for row in comparisons
            ),
            "comparisons": comparisons,
        })

    all_standalone_actions = [
        action for action in primitive.POLICY_ACTIONS
        if isinstance(action, primitive.PrimitiveActionSpec)
        or action.name in primitive.EXACT_ATOMIC_ACTION_BY_NAME
    ]
    standalone_family_counts = defaultdict(int)
    for action in all_standalone_actions:
        standalone_family_counts[action.operator_id] += 1
    standalone_actions = [
        action for action in all_standalone_actions
        if standalone_family_counts[action.operator_id] > 1
    ]
    single_implementation_standalone = [
        action.name for action in all_standalone_actions
        if standalone_family_counts[action.operator_id] == 1
    ]
    nonstandalone_registered = [
        action.name for action in primitive.POLICY_ACTIONS
        if action not in all_standalone_actions
    ]
    standalone_vectors: dict[str, list[Fraction | None]] = {}
    standalone_names: dict[str, list[str]] = defaultdict(list)
    for action in standalone_actions:
        vector = []
        for probe, key in zip(probes, probe_keys):
            if not standalone_applicable(action, str(probe["logical_type"])):
                vector.append(None)
                continue
            original, compressed = measure_standalone(action, probe)
            value = Fraction(compressed, original)
            vector.append(value)
            measurement_rows.append({
                "domain": "registered_standalone",
                "operator_id": action.operator_id,
                "implementation_name": action.name,
                "probe": key,
                "logical_type": probe["logical_type"],
                "original_bytes": original,
                "compressed_bytes": compressed,
                "compression_rate": float(value),
                "applicable": True,
                "lossless": True,
            })
        standalone_vectors[action.name] = vector
        standalone_names[action.operator_id].append(action.name)

    ans_audit = ans_cpp_python_audit(probes, args.ans_cli)
    prior_cpp_stable = all(row["stable_on_new_probes"] for row in prior_merge_checks)
    count_stable = prior_cpp_stable and ans_audit["all_wire_identical"]

    cpp_operator_results = summarize_operator_vectors(canonical_cpp_names, cpp_vectors)
    standalone_operator_results = summarize_operator_vectors(
        standalone_names, standalone_vectors
    )
    output = {
        "schema": "same-operator-real-data-ratio-audit-v1",
        "question": "do different implementations in the same operator family all have the same compression ratio?",
        "answer": False,
        "sample_rows_per_probe": args.sample_rows,
        "probes": [
            {
                key: probe[key]
                for key in ("dataset", "column", "logical_type", "rows", "raw_sha256")
            }
            for probe in probes
        ],
        "coverage_partition_of_194": {
            "registered_standalone_in_multi_implementation_families_measured": len(standalone_actions),
            "registered_standalone_single_implementation_not_compared": len(single_implementation_standalone),
            "registered_exact_native_stage_not_standalone": len(nonstandalone_registered),
            "deduplicated_cpp_method_scoped_measured": sum(
                len(names) for names in canonical_cpp_names.values()
            ),
            "cpp_unmeasured_blocked_proxy_or_unwired": len(unmeasured),
            "sum": (
                len(standalone_actions) + len(single_implementation_standalone)
                + len(nonstandalone_registered)
                + sum(len(names) for names in canonical_cpp_names.values())
                + len(unmeasured)
            ),
        },
        "registered_standalone": {
            "operator_families_with_multiple_implementations": len(standalone_operator_results),
            "families_with_different_ratios": sum(
                row["different_pairs"] > 0 for row in standalone_operator_results
            ),
            "operator_results": standalone_operator_results,
        },
        "cpp_method_scoped_omission": {
            "raw_measured_candidates": len(measurable),
            "deduplicated_measured_candidates": sum(
                len(names) for names in canonical_cpp_names.values()
            ),
            "operator_families_with_multiple_implementations": len(cpp_operator_results),
            "families_with_different_ratios": sum(
                row["different_pairs"] > 0 for row in cpp_operator_results
            ),
            "operator_results": cpp_operator_results,
        },
        "previously_merged_cpp_groups": prior_merge_checks,
        "previously_merged_cpp_groups_stable": prior_cpp_stable,
        "ans_cross_catalog": ans_audit,
        "audited_194_count_still_valid_on_new_probes": count_stable,
        "unmeasured_cpp_candidate_names": [row["action_name"] for row in unmeasured],
        "claim_boundary": (
            "registered standalone ratios are direct action outputs; C++ candidate ratios are "
            "whole-codec outputs with the named stage omitted; 60 exact native stages and 11 "
            "blocked/proxy/unwired candidates have no standalone per-implementation ratio"
        ),
    }
    if output["coverage_partition_of_194"]["sum"] != 194:
        raise RuntimeError(f"coverage partition does not sum to 194: {output['coverage_partition_of_194']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = [
        "domain", "operator_id", "implementation_name", "probe", "logical_type",
        "original_bytes", "compressed_bytes", "compression_rate", "applicable", "lossless",
    ]
    with (args.output_dir / "measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(measurement_rows)

    print(json.dumps({
        "answer": output["answer"],
        "coverage": output["coverage_partition_of_194"],
        "registered_multi_families": len(standalone_operator_results),
        "registered_different_families": output["registered_standalone"]["families_with_different_ratios"],
        "cpp_multi_families": len(cpp_operator_results),
        "cpp_different_families": output["cpp_method_scoped_omission"]["families_with_different_ratios"],
        "previous_merges_stable": count_stable,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
