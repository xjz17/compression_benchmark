#!/usr/bin/env python3
"""Fail-closed finalizer for the independent active97 timing campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


RESPONSE_SCHEMA = "operator102-timing-column-response-v1"
CAMEL_INT64_BOUNDARY_REASON = (
    "unsupported_input_domain: Camel int64-to-float64 exactness precondition is false"
)
CAMEL_INT64_BOUNDARY_REJECTION = {
    "class": "native_timing_error",
    "message": "Camel integer is not exactly representable as float64",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_not_applicable_result(row: dict[str, Any]) -> None:
    """Fail closed unless this is the exact frozen A004 support boundary."""
    require(row.get("implementation_id") == "A004",
            "only A004 may report not_applicable in active97 timing")
    require(row.get("bitexact") is False and row.get("rankable") is False,
            "A004 support boundary must be explicitly non-bitexact/non-rankable")
    require(row.get("reason") == CAMEL_INT64_BOUNDARY_REASON,
            "A004 support-boundary reason drift")
    require(row.get("native_domain_rejection") == CAMEL_INT64_BOUNDARY_REJECTION,
            "A004 native support-boundary evidence drift")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-datasets", type=int, default=200)
    ap.add_argument("--expected-columns", type=int, default=3723)
    ap.add_argument("--expected-implementations", type=int, default=97)
    args = ap.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == args.expected_columns,
            "plan does not have the expected column count")
    expected = {(str(x["job_id"]), str(x["dataset_id"]), str(x["column_id"]))
                for x in jobs}
    require(len(expected) == args.expected_columns, "plan identities are not unique")
    require(len({x[1] for x in expected}) == args.expected_datasets,
            "plan does not have the expected dataset count")

    seen_columns: set[tuple[str, str, str]] = set()
    implementation_ids: set[str] | None = None
    column_rows: list[dict[str, Any]] = []
    dataset_totals: dict[tuple[str, str], dict[str, Any]] = {}
    input_receipts = []
    for input_path in [p.resolve() for p in args.inputs]:
        require(input_path.is_file(), f"missing shard output: {input_path}")
        input_receipts.append({"path": str(input_path), "sha256": sha256(input_path)})
        with input_path.open() as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                response = json.loads(line)
                require(response.get("schema") == RESPONSE_SCHEMA,
                        f"wrong schema: {input_path}:{number}")
                identity = (str(response["request_id"]), str(response["dataset_id"]),
                            str(response["column_id"]))
                require(identity in expected and identity not in seen_columns,
                        f"unexpected/duplicate column: {identity}")
                seen_columns.add(identity)
                results = response.get("results")
                require(isinstance(results, list) and
                        len(results) == args.expected_implementations,
                        f"wrong implementation count: {identity}")
                ids = {str(row["implementation_id"]) for row in results}
                if implementation_ids is None:
                    implementation_ids = ids
                require(ids == implementation_ids and len(ids) == args.expected_implementations,
                        f"implementation union drift: {identity}")
                for row in results:
                    status = row.get("status")
                    require(status in {"ok", "not_applicable"},
                            f"failed/nonterminal result: {identity} {row}")
                    flat = {"dataset_id": identity[1], "column_id": identity[2],
                            "implementation_id": row["implementation_id"],
                            "status": status, "bitexact": row.get("bitexact")}
                    if status == "ok":
                        timing = row.get("timing")
                        require(row.get("bitexact") is True and isinstance(timing, dict) and
                                timing.get("warmups_per_block") == 1 and
                                timing.get("repeats") == 3 and timing.get("threads") == 1 and
                                timing.get("ranking_view") == "full_o2r1" and
                                timing.get("bitexact_each_warmup_and_repeat") is True,
                                f"invalid timing evidence: {identity} {row['implementation_id']}")
                        flat.update({
                            "valid_count": row["valid_count"],
                            "full_encode_ns_samples": timing["full_encode_ns"],
                            "full_decode_ns_samples": timing["full_decode_ns"],
                            "full_encode_median_ns": timing["full_encode_median_ns"],
                            "full_decode_median_ns": timing["full_decode_median_ns"],
                            "full_encode_median_ns_per_valid_value":
                                timing["full_encode_median_ns_per_valid_value"],
                            "full_decode_median_ns_per_valid_value":
                                timing["full_decode_median_ns_per_valid_value"],
                            "kernel_encode_ns_samples": timing["kernel_encode_ns"],
                            "kernel_decode_ns_samples": timing["kernel_decode_ns"],
                            "kernel_encode_median_ns": timing["kernel_encode_median_ns"],
                            "kernel_decode_median_ns": timing["kernel_decode_median_ns"],
                        })
                        key = (identity[1], str(row["implementation_id"]))
                        total = dataset_totals.setdefault(key, {
                            "dataset_id": identity[1],
                            "implementation_id": row["implementation_id"],
                            "timed_columns": 0, "valid_count": 0,
                            "full_encode_median_ns_sum": 0,
                            "full_decode_median_ns_sum": 0})
                        total["timed_columns"] += 1
                        total["valid_count"] += int(row["valid_count"])
                        total["full_encode_median_ns_sum"] += int(
                            timing["full_encode_median_ns"])
                        total["full_decode_median_ns_sum"] += int(
                            timing["full_decode_median_ns"])
                    else:
                        validate_not_applicable_result(row)
                        flat.update({
                            "rankable": False,
                            "reason": row["reason"],
                            "native_domain_rejection": row["native_domain_rejection"],
                        })
                    column_rows.append(flat)
    require(seen_columns == expected, f"missing columns: {len(expected - seen_columns)}")
    assert implementation_ids is not None
    require(len(column_rows) == args.expected_columns * args.expected_implementations,
            "column x implementation matrix is incomplete")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    column_path = output / "COLUMN_OPERATOR_TIMING.jsonl"
    with column_path.open("w") as f:
        for row in sorted(column_rows,
                          key=lambda x: (x["dataset_id"], x["column_id"],
                                         x["implementation_id"])):
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    dataset_path = output / "DATASET_OPERATOR_TIMING.csv"
    fields = ["dataset_id", "implementation_id", "timed_columns", "valid_count",
              "full_encode_median_ns_sum", "full_decode_median_ns_sum",
              "full_encode_ns_per_valid_value", "full_decode_ns_per_valid_value"]
    with dataset_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for total in sorted(dataset_totals.values(),
                            key=lambda x: (x["dataset_id"], x["implementation_id"])):
            count = total["valid_count"]
            total["full_encode_ns_per_valid_value"] = (
                total["full_encode_median_ns_sum"] / count if count else None)
            total["full_decode_ns_per_valid_value"] = (
                total["full_decode_median_ns_sum"] / count if count else None)
            writer.writerow(total)
    ok_count = sum(row["status"] == "ok" for row in column_rows)
    receipt = {
        "schema": "operator-active97-timing200-acceptance-v1",
        "status": "accepted",
        "datasets": args.expected_datasets,
        "columns": args.expected_columns,
        "implementations": args.expected_implementations,
        "terminal_column_implementation_keys": len(column_rows),
        "timed_bitexact_keys": ok_count,
        "not_applicable_support_boundary_keys": len(column_rows) - ok_count,
        "failed_keys": 0,
        "timing_contract": "formal full timing is operator forward + exact O2R1 pack and exact O2R1 parse + operator inverse; W1/R3/T1; per-repeat bit-exact; samplewise sum across entity blocks then column median; persistent native process; rotated implementation order; kernel-only timing retained only for diagnosis",
        "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "inputs": input_receipts,
        "column_output": str(column_path), "column_output_sha256": sha256(column_path),
        "dataset_output": str(dataset_path), "dataset_output_sha256": sha256(dataset_path),
        "old_results_preserved": True,
    }
    atomic_json(output / "ACCEPTANCE.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
