#!/usr/bin/env python3
"""End-to-end contract test for whole-column decimal scaling campaigns."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Iterable


METHODS = ("DBP", "RLBE", "REGER")
SCALED_PROFILE = "baseline86-decimal-scaling-int64-exact-v1"
DEFAULT_PROFILE = "baseline86-fixed-profiles-20260919-v3"
COLUMN_SCHEMA = "column-scales-exact-decimal-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def execute(
    cli: Path, arguments: Iterable[object], *, expected: int = 0, timeout: int = 120,
) -> subprocess.CompletedProcess[bytes]:
    command = [str(cli), *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {command!r}\n"
            f"stdout={result.stdout.decode(errors='replace')}\n"
            f"stderr={result.stderr.decode(errors='replace')}"
        )
    return result


def json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid JSONL line {number}: {line!r}") from error
        require(isinstance(value, dict), f"JSONL line {number} is not an object")
        rows.append(value)
    return rows


def write_block(
    root: Path, *, block_id: str, dtype: str, values: bytes,
    table_name: str, source_column: str, column: str,
    entity_id: str, chunk_index: int,
) -> dict[str, Any]:
    width = 8 if dtype in ("float64", "int64") else 4
    require(len(values) % width == 0, f"unaligned fixture {block_id}")
    row_count = len(values) // width
    missing = bytes((row_count + 7) // 8)
    values_name = f"{block_id}.values.bin"
    missing_name = f"{block_id}.missing.bitmap"
    (root / values_name).write_bytes(values)
    (root / missing_name).write_bytes(missing)
    return {
        "block_id": block_id,
        "dtype": dtype,
        "table_name": table_name,
        "role": "value",
        "source_column": source_column,
        "column": column,
        "entity_id": entity_id,
        "chunk_index": chunk_index,
        "row_count": row_count,
        "original_numeric_bytes": len(values),
        "values_path": values_name,
        "missing_bitmap_path": missing_name,
        "values_sha256": sha256(values),
        "missing_bitmap_sha256": sha256(missing),
    }


def build_manifest(root: Path) -> tuple[Path, dict[str, Any], dict[str, tuple[bytes, bytes]]]:
    first_values = struct.pack("<2d", 1.2, -3.4)       # P=1 block.
    second_values = struct.pack("<2d", 2.345, -6.125)  # P=3 block.
    third_values = struct.pack("<2f", 7.25, -8.5)      # Separate P=2 column.
    integer_values = struct.pack("<3q", 1, 2, -3)
    blocks = [
        write_block(
            root, block_id="temp-entity-z-chunk-9", dtype="float64",
            values=first_values, table_name="metrics",
            source_column="temperature", column="c0001",
            entity_id="entity-z", chunk_index=9,
        ),
        write_block(
            root, block_id="temp-entity-a-chunk-1", dtype="float64",
            values=second_values, table_name="metrics",
            source_column="temperature", column="c0001",
            entity_id="entity-a", chunk_index=1,
        ),
        write_block(
            root, block_id="humidity-entity-q-chunk-4", dtype="float32",
            values=third_values, table_name="metrics",
            source_column="humidity", column="c0002",
            entity_id="entity-q", chunk_index=4,
        ),
        write_block(
            root, block_id="count-entity-q-chunk-4", dtype="int64",
            values=integer_values, table_name="metrics",
            source_column="count", column="c0003",
            entity_id="entity-q", chunk_index=4,
        ),
    ]
    manifest = {"dataset_id": "fixture-dataset", "blocks": blocks}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    fixture_bytes = {
        block["block_id"]: (
            (root / block["values_path"]).read_bytes(),
            (root / block["missing_bitmap_path"]).read_bytes(),
        )
        for block in blocks
    }
    return manifest_path, manifest, fixture_bytes


def identity(source_column: str, column: str, dtype: str) -> dict[str, str]:
    return {
        "dataset_id": "fixture-dataset",
        "table_name": "metrics",
        "role": "value",
        "source_column": source_column,
        "column": column,
        "dtype": dtype,
    }


def validate_column_receipt(
    receipt: dict[str, Any], manifest_path: Path, cli: Path,
) -> dict[str, dict[str, Any]]:
    require(receipt.get("schema") == COLUMN_SCHEMA, "column receipt schema drift")
    require(receipt.get("manifest_sha256") == sha256(manifest_path.read_bytes()),
            "column receipt manifest SHA drift")
    require(receipt.get("scanner_cli_sha256") == sha256(cli.read_bytes()),
            "column receipt CLI SHA drift")
    require(receipt.get("warmups") == 1 and receipt.get("repeats") == 3,
            "scan-column-scales defaults are not 1 warmup / 3 repeats")
    columns = receipt.get("columns")
    require(isinstance(columns, list) and len(columns) == 2,
            "column receipt does not contain exactly the two float columns")
    by_source: dict[str, dict[str, Any]] = {}
    for column_receipt in columns:
        key = column_receipt.get("column_key")
        require(isinstance(key, str), "column_key is not a JSON string")
        try:
            parsed_key = json.loads(key)
        except json.JSONDecodeError as error:
            raise AssertionError("column_key is not valid JSON") from error
        require(parsed_key == column_receipt.get("column_identity"),
                "column_key and column_identity differ")
        source = parsed_key.get("source_column")
        require(source not in by_source, "duplicate scientific column receipt")
        by_source[source] = column_receipt
        samples = column_receipt.get("precision_derivation_ns_samples")
        require(isinstance(samples, list) and len(samples) == 3,
                f"{source}: precision sample count drift")
        require(all(isinstance(sample, int) and sample >= 0 for sample in samples),
                f"{source}: invalid precision sample")

    temperature = by_source.get("temperature")
    require(temperature is not None, "temperature column receipt missing")
    require(temperature.get("column_identity") == identity("temperature", "c0001", "float64"),
            "temperature identity includes entity/chunk or otherwise drifted")
    require(temperature.get("decimal_places") == 3,
            "two temperature blocks did not receive whole-column P=3")
    sorted_temperature_ids = ["temp-entity-a-chunk-1", "temp-entity-z-chunk-9"]
    require(temperature.get("block_ids") == sorted_temperature_ids,
            "temperature block IDs are not the sorted exact union")
    require(temperature.get("values_sha256s") == [
        sha256((manifest_path.parent / f"{block_id}.values.bin").read_bytes())
        for block_id in sorted_temperature_ids
    ], "temperature value digests do not follow sorted block IDs")
    require(temperature.get("missing_bitmap_sha256s") == [
        sha256((manifest_path.parent / f"{block_id}.missing.bitmap").read_bytes())
        for block_id in sorted_temperature_ids
    ], "temperature missing digests do not follow sorted block IDs")
    require(temperature.get("original_numeric_bytes") == 32,
            "temperature original byte total drift")

    humidity = by_source.get("humidity")
    require(humidity is not None, "humidity column receipt missing")
    require(humidity.get("column_identity") == identity("humidity", "c0002", "float32"),
            "humidity identity drift")
    require(humidity.get("decimal_places") == 2, "humidity P=2 drift")
    require(humidity.get("block_ids") == ["humidity-entity-q-chunk-4"],
            "humidity block union drift")
    require(humidity.get("original_numeric_bytes") == 8,
            "humidity original byte total drift")
    return by_source


def assert_samples(row: dict[str, Any], field: str, size: int = 3) -> list[int]:
    samples = row.get(field)
    require(isinstance(samples, list) and len(samples) == size,
            f"{row.get('block_id')}/{row.get('method')}: {field} length drift")
    require(all(isinstance(value, int) and value >= 0 for value in samples),
            f"{row.get('block_id')}/{row.get('method')}: invalid {field}")
    return samples


def validate_campaign(
    cli: Path, root: Path, manifest: dict[str, Any], fixture_bytes: dict[str, tuple[bytes, bytes]],
    columns: dict[str, dict[str, Any]], results_path: Path, frame_dir: Path,
) -> None:
    rows = json_lines(results_path)
    require(len(rows) == len(manifest["blocks"]) * len(METHODS),
            "scaled campaign result cardinality drift")
    by_key = {(row.get("block_id"), row.get("method")): row for row in rows}
    require(len(by_key) == len(rows), "scaled campaign emitted duplicate result keys")
    block_by_id = {block["block_id"]: block for block in manifest["blocks"]}
    first_success: set[tuple[str, str]] = set()
    for block in manifest["blocks"]:
        block_id = block["block_id"]
        for method in METHODS:
            row = by_key.get((block_id, method))
            require(row is not None, f"missing result {block_id}/{method}")
            require(row.get("status") in ("ok", "N/A"),
                    f"nonterminal successful fixture {block_id}/{method}: {row}")
            require(row.get("dtype") == block["dtype"],
                    f"{block_id}/{method}: dtype drift")
            require(row.get("original_numeric_bytes") == block["original_numeric_bytes"],
                    f"{block_id}/{method}: original byte denominator drift")
            relative = row.get("complete_frame_relative_path")
            require(isinstance(relative, str) and relative,
                    f"{block_id}/{method}: content-addressed frame path missing")
            frame_path = frame_dir / relative
            require(frame_path.is_file(), f"{block_id}/{method}: frame was not stored")
            require(row.get("complete_frame_bytes") == frame_path.stat().st_size,
                    f"{block_id}/{method}: frame byte count drift")
            require(row.get("complete_frame_sha256") == sha256(frame_path.read_bytes()),
                    f"{block_id}/{method}: frame digest drift")
            decoded_values = root / f"decoded-{block_id}-{method}.values.bin"
            decoded_missing = root / f"decoded-{block_id}-{method}.missing.bitmap"
            execute(cli, ["decode", frame_path, decoded_values, decoded_missing])
            require((decoded_values.read_bytes(), decoded_missing.read_bytes()) == fixture_bytes[block_id],
                    f"{block_id}/{method}: cold-decoded block differs")

            if block["dtype"] == "int64":
                require(row.get("profile_version") == DEFAULT_PROFILE,
                        f"{block_id}/{method}: integer block did not use the original profile")
                require("decimal_places" not in row,
                        f"{block_id}/{method}: integer block was silently scaled")
                assert_samples(row, "compression_ns_samples")
                assert_samples(row, "decompression_ns_samples")
                continue

            source = block["source_column"]
            column_receipt = columns[source]
            require(row.get("profile_version") == SCALED_PROFILE,
                    f"{block_id}/{method}: float block did not use scaled profile")
            require(row.get("column_key") == column_receipt["column_key"],
                    f"{block_id}/{method}: column key drift")
            require(row.get("decimal_places") == column_receipt["decimal_places"],
                    f"{block_id}/{method}: block did not use whole-column P")
            column_samples = column_receipt["precision_derivation_ns_samples"]
            require(row.get("column_precision_derivation_ns_samples") == column_samples,
                    f"{block_id}/{method}: original calibration samples missing")
            encode_only = assert_samples(row, "compression_encode_only_ns_samples")
            precision = assert_samples(row, "precision_derivation_ns_samples")
            combined = assert_samples(row, "compression_ns_samples")
            assert_samples(row, "decompression_ns_samples")
            charge_key = (method, column_receipt["column_key"])
            should_charge = charge_key not in first_success
            require(row.get("precision_charge_once") is should_charge,
                    f"{block_id}/{method}: one-time precision charge flag drift")
            if should_charge:
                require(precision == column_samples,
                        f"{block_id}/{method}: first block did not receive calibration charge")
                first_success.add(charge_key)
            else:
                require(precision == [0, 0, 0],
                        f"{block_id}/{method}: later block was charged twice")
            require(combined == [left + right for left, right in zip(encode_only, precision)],
                    f"{block_id}/{method}: compression samples omit or double-count calibration")


def benchmark_command(
    manifest_path: Path, output_path: Path, receipt_path: Path,
    *, frame_dir: Path | None = None,
) -> list[object]:
    command: list[object] = [
        "benchmark-manifest-scaled", manifest_path, output_path, "1", "3",
        "--column-scales", receipt_path,
    ]
    if frame_dir is not None:
        command.extend(("--frame-dir", frame_dir))
    command.extend(METHODS)
    return command


def expect_pre_output_rejection(
    cli: Path, root: Path, manifest_path: Path, receipt: dict[str, Any], name: str,
) -> None:
    receipt_path = root / f"tampered-{name}.json"
    output_path = root / f"tampered-{name}.jsonl"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result = subprocess.run(
        [str(cli), *map(str, benchmark_command(manifest_path, output_path, receipt_path))],
        capture_output=True, check=False, timeout=120,
    )
    require(result.returncode == 2,
            f"tampered {name} returned {result.returncode}, not validation error 2: "
            f"{result.stderr.decode(errors='replace')}")
    require(not output_path.exists(),
            f"tampered {name} created JSONL before validation completed")


def test_tamper_rejection(
    cli: Path, root: Path, manifest_path: Path, manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> int:
    cases: list[tuple[str, dict[str, Any]]] = []

    changed_precision = copy.deepcopy(receipt)
    changed_precision["columns"][0]["decimal_places"] += 1
    cases.append(("precision", changed_precision))

    changed_cli = copy.deepcopy(receipt)
    changed_cli["scanner_cli_sha256"] = "0" * 64
    cases.append(("cli-sha", changed_cli))

    missing_column = copy.deepcopy(receipt)
    missing_column["columns"].pop()
    cases.append(("missing-column", missing_column))

    extra_column = copy.deepcopy(receipt)
    extra = copy.deepcopy(extra_column["columns"][0])
    extra_identity = dict(extra["column_identity"])
    extra_identity["column"] = "c9999"
    extra["column_identity"] = extra_identity
    extra["column_key"] = json.dumps(extra_identity, sort_keys=True, separators=(",", ":"))
    extra_column["columns"].append(extra)
    cases.append(("extra-column", extra_column))

    duplicate_block = copy.deepcopy(receipt)
    multi = next(column for column in duplicate_block["columns"] if len(column["block_ids"]) == 2)
    multi["block_ids"][1] = multi["block_ids"][0]
    cases.append(("duplicate-block", duplicate_block))

    short_samples = copy.deepcopy(receipt)
    short_samples["columns"][0]["precision_derivation_ns_samples"].pop()
    cases.append(("sample-length", short_samples))

    for name, tampered in cases:
        expect_pre_output_rejection(cli, root, manifest_path, tampered, name)

    # Changing frozen values without changing the manifest-declared digest must
    # also be rejected before a result stream is opened.
    values_path = root / manifest["blocks"][0]["values_path"]
    original_values = values_path.read_bytes()
    changed_values = bytearray(original_values)
    changed_values[0] ^= 1
    values_path.write_bytes(changed_values)
    try:
        expect_pre_output_rejection(cli, root, manifest_path, receipt, "manifest-values")
    finally:
        values_path.write_bytes(original_values)

    duplicate_manifest = copy.deepcopy(manifest)
    duplicate_manifest["blocks"].append(copy.deepcopy(duplicate_manifest["blocks"][0]))
    duplicate_manifest_path = root / "duplicate-block-manifest.json"
    duplicate_manifest_path.write_text(json.dumps(duplicate_manifest, indent=2) + "\n")
    expect_pre_output_rejection(
        cli, root, duplicate_manifest_path, receipt, "duplicate-manifest-block",
    )
    return len(cases) + 2


def test_iteration_overflow_rejection(
    cli: Path, root: Path, manifest_path: Path, receipt_path: Path,
) -> int:
    cases = (
        ("warmups-u32-overflow", "4294967297", "3"),
        ("repeats-u64-max", "1", "18446744073709551615"),
    )
    for name, warmups, repeats in cases:
        output_path = root / f"{name}.jsonl"
        result = subprocess.run(
            [
                str(cli), "benchmark-manifest-scaled", str(manifest_path),
                str(output_path), warmups, repeats, "--column-scales",
                str(receipt_path), "DBP",
            ],
            capture_output=True, check=False, timeout=120,
        )
        require(result.returncode == 2,
                f"{name} returned {result.returncode}, not validation error 2: "
                f"{result.stderr.decode(errors='replace')}")
        require(not output_path.exists(),
                f"{name} created JSONL before validating the 64-bit count")
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True,
                        help="path to native_baseline86_cli")
    args = parser.parse_args()
    cli = args.cli.resolve()
    require(cli.is_file(), f"CLI does not exist: {cli}")
    with tempfile.TemporaryDirectory(prefix="decimal-scaling-manifest-") as temporary:
        root = Path(temporary)
        manifest_path, manifest, fixture_bytes = build_manifest(root)
        receipt_path = root / "column-scales.json"
        execute(cli, ["scan-column-scales", manifest_path, receipt_path])
        receipt = json.loads(receipt_path.read_text())
        columns = validate_column_receipt(receipt, manifest_path, cli)

        results_path = root / "results.jsonl"
        frame_dir = root / "frames"
        execute(
            cli,
            benchmark_command(
                manifest_path, results_path, receipt_path, frame_dir=frame_dir,
            ),
            timeout=180,
        )
        validate_campaign(
            cli, root, manifest, fixture_bytes, columns, results_path, frame_dir,
        )
        iteration_rejections = test_iteration_overflow_rejection(
            cli, root, manifest_path, receipt_path,
        )
        rejected = test_tamper_rejection(
            cli, root, manifest_path, manifest, receipt,
        )
    print(
        "decimal scaling manifest contract passed: "
        f"blocks=4 methods={len(METHODS)} results=12 "
        f"tamper_rejections={rejected} iteration_rejections={iteration_rejections}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
