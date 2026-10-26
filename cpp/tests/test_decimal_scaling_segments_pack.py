#!/usr/bin/env python3
"""Cold-process contract test for sliced inputs and packed scaled frames."""

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


METHODS = (
    "2DIFF", "BOS", "REGER", "SVB", "FastPFor", "Simple8b", "RLBE", "DBP",
    "LECO",
)
PACK_STORAGE = "length_prefixed_ngf1_custody_pack_v1"
COMPOUND_STORAGE = "P172B861_scientific_compound_v1"
COMPOUND_MAGIC = b"P172B861"
COMPOUND_HEADER_BYTES = 16
COMPOUND_PREFIX_BYTES = 8
VERIFY_SCHEMA = "scaled-packed-cold-verification-v1"


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid JSONL line {number}") from error
        require(isinstance(row, dict), f"JSONL line {number} is not an object")
        rows.append(row)
    return rows


def segment_row(
    *, block_id: str, dtype: str, values: bytes, missing: bytes,
    values_offset: int, missing_offset: int, source_column: str, column: str,
    entity_id: str, chunk_index: int, row_count: int,
) -> dict[str, Any]:
    width = 8 if dtype == "float64" else 4
    require(dtype in ("float32", "float64"), "fixture dtype is not floating point")
    require(len(values) % width == 0, f"{dtype} fixture is unaligned")
    require(len(missing) == (row_count + 7) // 8, "missing fixture size drift")
    valid_count = row_count - sum(byte.bit_count() for byte in missing)
    require(len(values) == valid_count * width,
            "valid numeric payload differs from missing bitmap")
    return {
        "block_id": block_id,
        "dtype": dtype,
        "table_name": "metrics",
        "role": "value",
        "source_column": source_column,
        "column": column,
        "entity_id": entity_id,
        "chunk_index": chunk_index,
        "row_count": row_count,
        "original_numeric_bytes": len(values),
        "values_path": "shared.values.bin",
        "values_offset": values_offset,
        "values_length": len(values),
        "missing_bitmap_path": "shared.missing.bitmap",
        "missing_bitmap_offset": missing_offset,
        "missing_bitmap_length": len(missing),
        "values_sha256": sha256(values),
        "missing_bitmap_sha256": sha256(missing),
    }


def build_sliced_manifest(
    root: Path,
) -> tuple[Path, dict[str, Any], dict[str, tuple[bytes, bytes]]]:
    values1 = struct.pack("<2d", 1.2, -3.4)
    missing1 = b"\x04"  # The third logical row is NULL and has no payload word.
    values2 = struct.pack("<3d", 2.345, -6.125, 7.0)
    missing2 = b"\x00"
    values3 = struct.pack("<3f", 7.25, -8.5, 9.0)
    missing3 = b"\x00"

    values_prefix, values_gap, values_suffix = b"values-prefix", b"gap", b"tail"
    values_offset1 = len(values_prefix)
    values_offset2 = values_offset1 + len(values1) + len(values_gap)
    values_offset3 = values_offset2 + len(values2) + len(values_gap)
    (root / "shared.values.bin").write_bytes(
        values_prefix + values1 + values_gap + values2 + values_gap + values3
        + values_suffix
    )

    missing_prefix, missing_gap, missing_suffix = b"MP", b"G", b"T"
    missing_offset1 = len(missing_prefix)
    missing_offset2 = missing_offset1 + len(missing1) + len(missing_gap)
    missing_offset3 = missing_offset2 + len(missing2) + len(missing_gap)
    (root / "shared.missing.bitmap").write_bytes(
        missing_prefix + missing1 + missing_gap + missing2 + missing_gap
        + missing3 + missing_suffix
    )

    blocks = [
        segment_row(
            block_id="temp-entity-b-chunk-3", dtype="float64", values=values1,
            missing=missing1, values_offset=values_offset1,
            missing_offset=missing_offset1, source_column="temperature",
            column="c0001", entity_id="entity-b", chunk_index=3, row_count=3,
        ),
        segment_row(
            block_id="temp-entity-a-chunk-1", dtype="float64", values=values2,
            missing=missing2, values_offset=values_offset2,
            missing_offset=missing_offset2, source_column="temperature",
            column="c0001", entity_id="entity-a", chunk_index=1, row_count=3,
        ),
        segment_row(
            block_id="humidity-entity-c-chunk-2", dtype="float32", values=values3,
            missing=missing3, values_offset=values_offset3,
            missing_offset=missing_offset3, source_column="humidity",
            column="c0002", entity_id="entity-c", chunk_index=2, row_count=3,
        ),
    ]
    manifest = {"dataset_id": "sliced-fixture-dataset", "blocks": blocks}
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    fixtures = {
        blocks[0]["block_id"]: (values1, missing1),
        blocks[1]["block_id"]: (values2, missing2),
        blocks[2]["block_id"]: (values3, missing3),
    }
    return manifest_path, manifest, fixtures


def verify_command(
    manifest: Path, results: Path, pack: Path, scales: Path, receipt: Path,
    methods: Iterable[str] = METHODS,
) -> list[object]:
    return [
        "verify-scaled-pack", manifest, results, pack, scales, receipt, *methods,
    ]


def require_verify_rejection(
    cli: Path, root: Path, name: str, *, manifest: Path, results: Path,
    pack: Path, scales: Path, methods: Iterable[str] = METHODS,
) -> None:
    receipt = root / f"rejected-{name}.json"
    result = subprocess.run(
        [str(cli), *map(str, verify_command(
            manifest, results, pack, scales, receipt, methods,
        ))],
        capture_output=True, check=False, timeout=120,
    )
    require(result.returncode == 2,
            f"{name}: verifier returned {result.returncode}, not 2; "
            f"stderr={result.stderr.decode(errors='replace')}")
    require(not receipt.exists(), f"{name}: failed verifier wrote a receipt")


def test_segment_rejections(
    cli: Path, root: Path, manifest: dict[str, Any],
) -> int:
    cases: list[tuple[str, dict[str, Any]]] = []

    out_of_bounds = copy.deepcopy(manifest)
    out_of_bounds["blocks"][0]["values_offset"] = 1 << 62
    cases.append(("values-out-of-bounds", out_of_bounds))

    missing_length = copy.deepcopy(manifest)
    del missing_length["blocks"][0]["values_length"]
    cases.append(("offset-without-length", missing_length))

    negative_offset = copy.deepcopy(manifest)
    negative_offset["blocks"][0]["missing_bitmap_offset"] = -1
    cases.append(("negative-missing-offset", negative_offset))

    unaligned_length = copy.deepcopy(manifest)
    unaligned_length["blocks"][0]["values_length"] -= 1
    cases.append(("unaligned-segment", unaligned_length))

    wrong_segment_sha = copy.deepcopy(manifest)
    wrong_segment_sha["blocks"][0]["values_sha256"] = "0" * 64
    cases.append(("segment-sha", wrong_segment_sha))

    for name, bad_manifest in cases:
        manifest_path = root / f"invalid-{name}.manifest.json"
        output_path = root / f"invalid-{name}.scales.json"
        write_json(manifest_path, bad_manifest)
        result = subprocess.run(
            [str(cli), "scan-column-scales", str(manifest_path), str(output_path), "0", "1"],
            capture_output=True, check=False, timeout=120,
        )
        require(result.returncode == 2,
                f"{name}: scanner returned {result.returncode}, not 2; "
                f"stderr={result.stderr.decode(errors='replace')}")
        require(not output_path.exists(), f"{name}: rejected scanner wrote a receipt")
    return len(cases)


def validate_pack_and_cold_decode(
    cli: Path, root: Path, rows: list[dict[str, Any]], pack_path: Path,
    fixtures: dict[str, tuple[bytes, bytes]],
) -> tuple[int, int]:
    pack = pack_path.read_bytes()
    expected_offset = 0
    frame_bytes = 0
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        key = (row.get("block_id"), row.get("method"))
        require(key not in seen, f"duplicate packed result key {key}")
        seen.add(key)
        require(row.get("status") == "ok", f"fixture result is not numeric: {row}")
        require(row.get("complete_frame_storage") == PACK_STORAGE,
                f"{key}: custody format drift")
        require(row.get("complete_frame_pack_record_offset") == expected_offset,
                f"{key}: noncontiguous pack offset")
        frame_length = row.get("complete_frame_pack_length")
        require(isinstance(frame_length, int) and frame_length > 0,
                f"{key}: invalid pack frame length")
        require(frame_length == row.get("complete_frame_bytes"),
                f"{key}: packed length differs from scientific NGF1 bytes")
        require(expected_offset + 8 + frame_length <= len(pack),
                f"{key}: pack member exceeds custody file")
        prefix_length = struct.unpack_from("<Q", pack, expected_offset)[0]
        require(prefix_length == frame_length, f"{key}: custody prefix length drift")
        frame = pack[expected_offset + 8:expected_offset + 8 + frame_length]
        require(sha256(frame) == row.get("complete_frame_sha256"),
                f"{key}: packed NGF1 digest drift")

        frame_path = root / f"cold-frame-{index}.ngf1"
        values_out = root / f"cold-values-{index}.bin"
        missing_out = root / f"cold-missing-{index}.bitmap"
        frame_path.write_bytes(frame)
        execute(cli, ["decode", frame_path, values_out, missing_out])
        require((values_out.read_bytes(), missing_out.read_bytes()) == fixtures[key[0]],
                f"{key}: independent cold decode differs from sliced source")

        frame_bytes += frame_length
        expected_offset += 8 + frame_length
    require(expected_offset == len(pack), "custody pack has unindexed trailing bytes")
    require(seen == {(block, method) for block in fixtures for method in METHODS},
            "packed result key union drift")
    return frame_bytes, expected_offset


def validate_legacy_frame_equivalence(
    packed_rows: list[dict[str, Any]], pack_path: Path,
    legacy_rows: list[dict[str, Any]], legacy_frame_dir: Path,
) -> None:
    legacy_by_key = {
        (row.get("block_id"), row.get("method")): row for row in legacy_rows
    }
    require(len(legacy_by_key) == len(legacy_rows),
            "legacy frame-directory campaign contains duplicate keys")
    pack = pack_path.read_bytes()
    for row in packed_rows:
        key = (row.get("block_id"), row.get("method"))
        legacy = legacy_by_key.get(key)
        require(legacy is not None, f"{key}: missing legacy custody result")
        offset = row["complete_frame_pack_record_offset"]
        length = row["complete_frame_pack_length"]
        packed_frame = pack[offset + 8:offset + 8 + length]
        relative = legacy.get("complete_frame_relative_path")
        require(isinstance(relative, str) and relative,
                f"{key}: legacy content-addressed path missing")
        legacy_frame = (legacy_frame_dir / relative).read_bytes()
        require(packed_frame == legacy_frame,
                f"{key}: packing changed the exact NGF1 object")
        require(row.get("complete_frame_bytes") == legacy.get("complete_frame_bytes")
                and row.get("complete_frame_sha256") == legacy.get("complete_frame_sha256"),
                f"{key}: scientific byte identity differs by custody format")
    require(set(legacy_by_key) == {
        (row.get("block_id"), row.get("method")) for row in packed_rows
    }, "legacy and packed campaign key unions differ")


def test_verifier_rejections(
    cli: Path, root: Path, manifest_path: Path, results_path: Path,
    pack_path: Path, scales_path: Path, rows: list[dict[str, Any]],
) -> int:
    cases = 0

    payload_tamper = bytearray(pack_path.read_bytes())
    payload_tamper[8] ^= 1
    payload_path = root / "tampered-payload.pack"
    payload_path.write_bytes(payload_tamper)
    require_verify_rejection(
        cli, root, "pack-payload", manifest=manifest_path, results=results_path,
        pack=payload_path, scales=scales_path,
    )
    cases += 1

    prefix_tamper = bytearray(pack_path.read_bytes())
    prefix_tamper[0] ^= 1
    prefix_path = root / "tampered-prefix.pack"
    prefix_path.write_bytes(prefix_tamper)
    require_verify_rejection(
        cli, root, "pack-prefix", manifest=manifest_path, results=results_path,
        pack=prefix_path, scales=scales_path,
    )
    cases += 1

    trailing_path = root / "tampered-trailing.pack"
    trailing_path.write_bytes(pack_path.read_bytes() + b"trailing")
    require_verify_rejection(
        cli, root, "pack-trailing", manifest=manifest_path, results=results_path,
        pack=trailing_path, scales=scales_path,
    )
    cases += 1

    missing_prefix_path = root / "tampered-missing-prefix.pack"
    missing_prefix_path.write_bytes(pack_path.read_bytes()[:4])
    require_verify_rejection(
        cli, root, "pack-missing-prefix", manifest=manifest_path,
        results=results_path, pack=missing_prefix_path, scales=scales_path,
    )
    cases += 1

    result_cases: list[tuple[str, list[dict[str, Any]]]] = []
    wrong_precision = copy.deepcopy(rows)
    wrong_precision[0]["decimal_places"] += 1
    result_cases.append(("decimal-places", wrong_precision))

    wrong_offset = copy.deepcopy(rows)
    wrong_offset[0]["complete_frame_pack_record_offset"] += 1
    result_cases.append(("pack-offset", wrong_offset))

    wrong_length = copy.deepcopy(rows)
    wrong_length[0]["complete_frame_pack_length"] += 1
    result_cases.append(("pack-length", wrong_length))

    wrong_digest = copy.deepcopy(rows)
    wrong_digest[0]["complete_frame_sha256"] = "0" * 64
    result_cases.append(("frame-sha", wrong_digest))

    missing_key = copy.deepcopy(rows)
    del missing_key[0]["complete_frame_pack_length"]
    result_cases.append(("missing-result-key", missing_key))

    wrong_stats = copy.deepcopy(rows)
    wrong_stats[0]["scaled_value_count"] += 1
    result_cases.append(("scaling-stats", wrong_stats))

    wrong_charge = copy.deepcopy(rows)
    wrong_charge[0]["precision_charge_once"] = not wrong_charge[0]["precision_charge_once"]
    result_cases.append(("timing-charge", wrong_charge))

    negative_ns = copy.deepcopy(rows)
    negative_ns[0]["compression_encode_only_ns_samples"][0] = -1
    result_cases.append(("negative-timing-sample", negative_ns))

    fractional_ns = copy.deepcopy(rows)
    fractional_ns[0]["decompression_ns_samples"][0] += 0.5
    result_cases.append(("fractional-timing-sample", fractional_ns))

    wrong_decompression_median = copy.deepcopy(rows)
    wrong_decompression_median[0]["decompression_ns"] += 1
    result_cases.append(("decompression-median", wrong_decompression_median))

    wrong_normalized_time = copy.deepcopy(rows)
    wrong_normalized_time[0]["compression_ns_per_original_numeric_byte"] += 1.0
    result_cases.append(("normalized-time", wrong_normalized_time))

    result_cases.append(("duplicate-result", rows + [copy.deepcopy(rows[0])]))
    result_cases.append(("missing-result", rows[:-1]))

    for name, bad_rows in result_cases:
        bad_results = root / f"tampered-{name}.jsonl"
        write_jsonl(bad_results, bad_rows)
        require_verify_rejection(
            cli, root, name, manifest=manifest_path, results=bad_results,
            pack=pack_path, scales=scales_path,
        )
        cases += 1

    require_verify_rejection(
        cli, root, "wrong-method-union", manifest=manifest_path,
        results=results_path, pack=pack_path, scales=scales_path,
        methods=("DBP", "REGER"),
    )
    return cases + 1


def test_custody_option_rejections(
    cli: Path, root: Path, manifest_path: Path, scales_path: Path,
) -> int:
    conflict_output = root / "conflicting-custody.jsonl"
    conflict_pack = root / "conflicting-custody.pack"
    conflict_dir = root / "conflicting-custody-frames"
    result = subprocess.run(
        [
            str(cli), "benchmark-manifest-scaled", str(manifest_path),
            str(conflict_output), "0", "1", "--column-scales", str(scales_path),
            "--frame-pack", str(conflict_pack), "--frame-dir", str(conflict_dir),
            "DBP",
        ],
        capture_output=True, check=False, timeout=120,
    )
    require(result.returncode == 2, "simultaneous pack/frame-dir custody was accepted")
    require(not conflict_output.exists() and not conflict_pack.exists()
            and not conflict_dir.exists(),
            "custody conflict created output before validation")

    existing_pack = root / "existing.pack"
    existing_pack.write_bytes(b"preserve-me")
    existing_output = root / "existing-pack.jsonl"
    result = subprocess.run(
        [
            str(cli), "benchmark-manifest-scaled", str(manifest_path),
            str(existing_output), "0", "1", "--column-scales", str(scales_path),
            "--frame-pack", str(existing_pack), "DBP",
        ],
        capture_output=True, check=False, timeout=120,
    )
    require(result.returncode == 2, "pre-existing custody pack was overwritten")
    require(existing_pack.read_bytes() == b"preserve-me",
            "pre-existing custody pack content changed")
    require(not existing_output.exists(), "exclusive-pack rejection created JSONL")
    return 2


def build_single_column_compound_manifest(
    root: Path, manifest: dict[str, Any],
    fixtures: dict[str, tuple[bytes, bytes]],
) -> tuple[Path, dict[str, Any], dict[str, tuple[bytes, bytes]]]:
    """Make a three-block, one-whole-column fixture for compound framing.

    Compound mode deliberately rejects a manifest containing more than one
    logical column.  Reusing a verified segment under a fresh block identity is
    sufficient here: the test is for framing/accounting and all source SHA and
    range checks still run in both the producer and cold verifier.
    """
    compound = copy.deepcopy(manifest)
    compound["blocks"] = copy.deepcopy(manifest["blocks"][:2])
    third = copy.deepcopy(manifest["blocks"][0])
    third["block_id"] = "temp-entity-c-chunk-5"
    third["entity_id"] = "entity-c"
    third["chunk_index"] = 5
    compound["blocks"].append(third)
    manifest_path = root / "compound-manifest.json"
    write_json(manifest_path, compound)
    compound_fixtures = {
        row["block_id"]: fixtures[manifest["blocks"][index]["block_id"]]
        for index, row in enumerate(compound["blocks"][:2])
    }
    compound_fixtures[third["block_id"]] = fixtures[manifest["blocks"][0]["block_id"]]
    return manifest_path, compound, compound_fixtures


def validate_compound_rows(
    rows: list[dict[str, Any]], pack: bytes, method: str,
    fixtures: dict[str, tuple[bytes, bytes]],
) -> int:
    require(pack[:8] == COMPOUND_MAGIC, f"{method}: compound magic drift")
    require(len(pack) >= COMPOUND_HEADER_BYTES,
            f"{method}: compound header is truncated")
    require(struct.unpack_from("<Q", pack, 8)[0] == len(rows),
            f"{method}: compound member count drift")
    require(len(rows) == len(fixtures) == 3,
            f"{method}: compound fixture cardinality drift")

    expected_offset = COMPOUND_HEADER_BYTES
    scientific_total = 0
    for index, row in enumerate(rows):
        key = (row.get("block_id"), row.get("method"))
        require(key == (list(fixtures)[index], method),
                f"{method}: compound result ordering drift: {key}")
        require(row.get("status") == "ok", f"{key}: compound result is not numeric")
        require(row.get("complete_frame_storage") == COMPOUND_STORAGE,
                f"{key}: compound storage label drift")
        require(row.get("compound_frame_schema") == "P172B861",
                f"{key}: compound schema drift")
        require(row.get("complete_frame_pack_record_offset") == expected_offset,
                f"{key}: compound pack offset drift")
        ngf1_bytes = row.get("complete_frame_pack_length")
        require(isinstance(ngf1_bytes, int) and ngf1_bytes > 0,
                f"{key}: compound child length invalid")
        require(row.get("complete_ngf1_frame_bytes") == ngf1_bytes,
                f"{key}: child NGF1 byte accounting drift")
        require(row.get("compound_length_prefix_bytes") == COMPOUND_PREFIX_BYTES,
                f"{key}: compound prefix charge drift")
        first = index == 0
        header_charge = COMPOUND_HEADER_BYTES if first else 0
        require(row.get("compound_header_charge_once") is first,
                f"{key}: global-header charge-once drift")
        require(row.get("compound_header_bytes_charged") == header_charge,
                f"{key}: global-header byte charge drift")
        scientific_bytes = ngf1_bytes + COMPOUND_PREFIX_BYTES + header_charge
        require(row.get("complete_frame_bytes") == scientific_bytes,
                f"{key}: full compound scientific byte count drift")
        original_bytes = len(fixtures[row["block_id"]][0])
        require(row.get("compression_ratio") == scientific_bytes / original_bytes,
                f"{key}: compound compression ratio excludes framing bytes")

        require(expected_offset + COMPOUND_PREFIX_BYTES + ngf1_bytes <= len(pack),
                f"{key}: compound member exceeds pack")
        require(struct.unpack_from("<Q", pack, expected_offset)[0] == ngf1_bytes,
                f"{key}: compound child prefix length drift")
        frame = pack[
            expected_offset + COMPOUND_PREFIX_BYTES:
            expected_offset + COMPOUND_PREFIX_BYTES + ngf1_bytes
        ]
        require(sha256(frame) == row.get("complete_frame_sha256"),
                f"{key}: compound child SHA drift")

        ngf1_encode = row.get("ngf1_compression_ns_samples")
        ngf1_decode = row.get("ngf1_decompression_ns_samples")
        encode = row.get("compression_encode_only_ns_samples")
        decode = row.get("decompression_ns_samples")
        require(all(isinstance(values, list) and len(values) == 1
                    for values in (ngf1_encode, ngf1_decode, encode, decode)),
                f"{key}: compound timing sample shape drift")
        prefix_encode = row.get("compound_prefix_encode_ns")
        prefix_decode = row.get("compound_prefix_decode_ns")
        require(isinstance(prefix_encode, int) and prefix_encode >= 0
                and isinstance(prefix_decode, int) and prefix_decode >= 0,
                f"{key}: compound prefix timing is not unsigned ns")
        require(encode[0] == ngf1_encode[0] + prefix_encode,
                f"{key}: compound encode timing arithmetic drift")
        require(decode[0] == ngf1_decode[0] + prefix_decode,
                f"{key}: compound decode timing arithmetic drift")
        precision = row.get("precision_derivation_ns_samples")
        total = row.get("compression_ns_samples")
        require(isinstance(precision, list) and len(precision) == 1
                and isinstance(total, list) and len(total) == 1
                and total[0] == encode[0] + precision[0],
                f"{key}: precision cost is not separate from compound prefix cost")

        expected_offset += COMPOUND_PREFIX_BYTES + ngf1_bytes
        scientific_total += scientific_bytes
    require(expected_offset == len(pack),
            f"{method}: compound pack has unindexed trailing bytes")
    require(scientific_total == len(pack),
            f"{method}: sum of charged compound bytes differs from physical pack")
    return scientific_total


def test_compound_frames(
    cli: Path, root: Path, manifest: dict[str, Any],
    fixtures: dict[str, tuple[bytes, bytes]],
) -> tuple[int, int]:
    manifest_path, compound, compound_fixtures = \
        build_single_column_compound_manifest(root, manifest, fixtures)
    scales_path = root / "compound-column-scales.json"
    execute(cli, ["scan-column-scales", manifest_path, scales_path, "0", "1"])
    scales = json.loads(scales_path.read_text())
    require(len(scales.get("columns", [])) == 1,
            "compound fixture did not form exactly one whole logical column")
    require(scales["columns"][0].get("decimal_places") == 3,
            "compound fixture whole-column precision drift")

    record_count = 0
    rejection_count = 0
    for method_index, method in enumerate(METHODS):
        results_path = root / f"compound-{method}.jsonl"
        pack_path = root / f"compound-{method}.pack"
        execute(
            cli,
            [
                "benchmark-manifest-scaled", manifest_path, results_path,
                "0", "1", "--column-scales", scales_path, "--frame-pack",
                pack_path, "--compound-frame", method,
            ],
            timeout=180,
        )
        rows = read_jsonl(results_path)
        pack = pack_path.read_bytes()
        scientific_bytes = validate_compound_rows(
            rows, pack, method, compound_fixtures,
        )
        record_count += len(rows)

        receipt_path = root / f"compound-{method}.cold.json"
        execute(
            cli,
            verify_command(
                manifest_path, results_path, pack_path, scales_path,
                receipt_path, (method,),
            ),
            timeout=180,
        )
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("status") == "verified"
                and receipt.get("compound_frame_schema") == "P172B861",
                f"{method}: compound cold receipt identity drift")
        require(receipt.get("scientific_compound_frame_bytes") == scientific_bytes
                and receipt.get("scientific_compound_frame_bytes") == len(pack),
                f"{method}: compound cold byte total drift")
        require(receipt.get("custody_pack_bytes") == len(pack),
                f"{method}: compound custody byte total drift")
        require(receipt.get("complete_ngf1_bytes") == sum(
            row["complete_ngf1_frame_bytes"] for row in rows
        ), f"{method}: compound cold child-NGF1 total drift")

        # One method is enough to exercise each independent header/timing guard.
        if method_index == 0:
            bad_magic = bytearray(pack)
            bad_magic[0] ^= 1
            bad_magic_path = root / "compound-tampered-magic.pack"
            bad_magic_path.write_bytes(bad_magic)
            require_verify_rejection(
                cli, root, "compound-magic", manifest=manifest_path,
                results=results_path, pack=bad_magic_path, scales=scales_path,
                methods=(method,),
            )
            rejection_count += 1

            bad_count = bytearray(pack)
            struct.pack_into("<Q", bad_count, 8, len(rows) + 1)
            bad_count_path = root / "compound-tampered-count.pack"
            bad_count_path.write_bytes(bad_count)
            require_verify_rejection(
                cli, root, "compound-count", manifest=manifest_path,
                results=results_path, pack=bad_count_path, scales=scales_path,
                methods=(method,),
            )
            rejection_count += 1

            timing_rows = copy.deepcopy(rows)
            timing_rows[0]["compound_prefix_encode_ns"] += 1
            timing_path = root / "compound-tampered-prefix-timing.jsonl"
            write_jsonl(timing_path, timing_rows)
            require_verify_rejection(
                cli, root, "compound-prefix-timing", manifest=manifest_path,
                results=timing_path, pack=pack_path, scales=scales_path,
                methods=(method,),
            )
            rejection_count += 1

            header_charge_rows = copy.deepcopy(rows)
            header_charge_rows[0]["compound_header_bytes_charged"] = 0
            charge_path = root / "compound-tampered-header-charge.jsonl"
            write_jsonl(charge_path, header_charge_rows)
            require_verify_rejection(
                cli, root, "compound-header-charge", manifest=manifest_path,
                results=charge_path, pack=pack_path, scales=scales_path,
                methods=(method,),
            )
            rejection_count += 1

    require(record_count == len(METHODS) * len(compound["blocks"]) == 27,
            "compound campaign did not test 9 methods x 3 blocks")

    # The compound option itself must fail closed for a multi-column manifest
    # and for a multi-method invocation before producing scientific output.
    multi_column_scales = root / "multi-column-compound-scales.json"
    execute(
        cli,
        ["scan-column-scales", root / "manifest.json", multi_column_scales, "0", "1"],
    )
    for name, bad_manifest, bad_scales, methods in (
        ("multi-column", root / "manifest.json", multi_column_scales, ("DBP",)),
        ("multi-method", manifest_path, scales_path, ("DBP", "RLBE")),
    ):
        output_path = root / f"rejected-compound-{name}.jsonl"
        pack_path = root / f"rejected-compound-{name}.pack"
        result = subprocess.run(
            [
                str(cli), "benchmark-manifest-scaled", str(bad_manifest),
                str(output_path), "0", "1", "--column-scales",
                str(bad_scales), "--frame-pack", str(pack_path),
                "--compound-frame", *methods,
            ],
            capture_output=True, check=False, timeout=120,
        )
        require(result.returncode == 2,
                f"compound {name} invocation returned {result.returncode}, not 2")
        require(not output_path.exists() and not pack_path.exists(),
                f"compound {name} rejection created outputs")
        rejection_count += 1
    return record_count, rejection_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True,
                        help="path to native_baseline86_cli")
    args = parser.parse_args()
    cli = args.cli.resolve()
    require(cli.is_file(), f"CLI does not exist: {cli}")

    with tempfile.TemporaryDirectory(prefix="decimal-scaling-packed-") as temporary:
        root = Path(temporary)
        manifest_path, manifest, fixtures = build_sliced_manifest(root)
        segment_rejections = test_segment_rejections(cli, root, manifest)

        scales_path = root / "column-scales.json"
        execute(cli, ["scan-column-scales", manifest_path, scales_path, "0", "1"])
        scales = json.loads(scales_path.read_text())
        require(scales.get("schema") == "column-scales-exact-decimal-v1",
                "precision receipt schema drift")
        require(len(scales.get("columns", [])) == 2,
                "sliced blocks did not form two scientific columns")
        scales_by_source = {
            column["column_identity"]["source_column"]: column
            for column in scales["columns"]
        }
        require(scales_by_source["temperature"].get("decimal_places") == 3,
                "two float64 slices did not receive whole-column P=3")
        require(scales_by_source["humidity"].get("decimal_places") == 2,
                "float32 sliced column did not receive P=2")

        custody_rejections = test_custody_option_rejections(
            cli, root, manifest_path, scales_path,
        )

        compound_records, compound_rejections = test_compound_frames(
            cli, root, manifest, fixtures,
        )

        legacy_results_path = root / "legacy-results.jsonl"
        legacy_frame_dir = root / "legacy-frames"
        execute(
            cli,
            [
                "benchmark-manifest-scaled", manifest_path, legacy_results_path,
                "0", "1", "--column-scales", scales_path, "--frame-dir",
                legacy_frame_dir, *METHODS,
            ],
            timeout=180,
        )
        legacy_rows = read_jsonl(legacy_results_path)

        results_path = root / "results.jsonl"
        pack_path = root / "frames.pack"
        execute(
            cli,
            [
                "benchmark-manifest-scaled", manifest_path, results_path, "0", "1",
                "--column-scales", scales_path, "--frame-pack", pack_path, *METHODS,
            ],
            timeout=180,
        )
        rows = read_jsonl(results_path)
        require(len(rows) == len(manifest["blocks"]) * len(METHODS),
                "packed campaign cardinality drift")
        frame_bytes, pack_bytes = validate_pack_and_cold_decode(
            cli, root, rows, pack_path, fixtures,
        )
        validate_legacy_frame_equivalence(
            rows, pack_path, legacy_rows, legacy_frame_dir,
        )

        receipt_path = root / "cold-verification.json"
        execute(
            cli,
            verify_command(
                manifest_path, results_path, pack_path, scales_path, receipt_path,
            ),
            timeout=180,
        )
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("schema") == VERIFY_SCHEMA, "cold receipt schema drift")
        require(receipt.get("status") == "verified", "cold receipt is not verified")
        require(receipt.get("dataset_id") == manifest["dataset_id"],
                "cold receipt dataset identity drift")
        require(receipt.get("manifest_sha256") == sha256(manifest_path.read_bytes()),
                "cold receipt manifest SHA drift")
        require(receipt.get("results_sha256") == sha256(results_path.read_bytes()),
                "cold receipt result SHA drift")
        require(receipt.get("pack_sha256") == sha256(pack_path.read_bytes()),
                "cold receipt pack SHA drift")
        require(receipt.get("precision_receipt_sha256") == sha256(scales_path.read_bytes()),
                "cold receipt precision SHA drift")
        require(receipt.get("cli_sha256") == sha256(cli.read_bytes()),
                "cold receipt CLI SHA drift")
        require(set(receipt.get("methods", [])) == set(METHODS),
                "cold receipt method union drift")
        require(receipt.get("expected_rows") == len(rows)
                and receipt.get("verified_rows") == len(rows)
                and receipt.get("numeric_rows") == len(rows),
                "cold receipt row counters drift")
        require(receipt.get("not_applicable_rows") == 0
                and receipt.get("unsupported_rows") == 0,
                "cold receipt unexpected support boundary")
        require(receipt.get("numeric_original_bytes") ==
                sum(len(values) for values, _ in fixtures.values()) * len(METHODS),
                "cold receipt source byte total drift")
        require(receipt.get("complete_ngf1_bytes") == frame_bytes,
                "cold receipt scientific NGF1 byte total drift")
        require(receipt.get("custody_pack_bytes") == pack_bytes,
                "cold receipt custody byte total drift")
        require(receipt.get("source_cache_used_by_decoder") is False,
                "cold verifier reported a source cache")

        verifier_rejections = test_verifier_rejections(
            cli, root, manifest_path, results_path, pack_path, scales_path, rows,
        )

    print(
        "decimal scaling packed contract passed: "
        f"blocks=3 methods={len(METHODS)} results={3 * len(METHODS)} "
        f"segment_rejections={segment_rejections} "
        f"custody_rejections={custody_rejections} "
        f"verifier_rejections={verifier_rejections} "
        f"compound_records={compound_records} "
        f"compound_rejections={compound_rejections}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
