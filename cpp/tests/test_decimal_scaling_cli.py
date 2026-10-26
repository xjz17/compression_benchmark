#!/usr/bin/env python3
"""Cold-process contract tests for the decimal-scaled int64 baseline profile.

The default baseline profile is deliberately kept separate: the nine methods
below remain int64-only there.  The opt-in scaled profile must preserve the
original float32/float64 bytes, including raw exceptional values, across an
encode process and a distinct decode process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Iterable


METHODS = (
    "2DIFF",
    "BOS",
    "REGER",
    "SVB",
    "FastPFor",
    "Simple8b",
    "DBP",
    "RLBE",
    "LECO",
)
SCALED_PROFILE = "baseline86-decimal-scaling-int64-exact-v1"
SCALED_INVENTORY_SCHEMA = "native-baseline86-scaled-inventory-v1"
OLD_CONFIGURATIONS = {
    "2DIFF": "compose_int64_default",
    "BOS": "adapter_compose_int64",
    "REGER": "reorder_delta_zigzag_for_bitpack_residual_core",
    "SVB": "compose_int64_default",
    "FastPFor": "FastPFor8_prefix_then_variablebyte_tail",
    "Simple8b": "bitpack_int64_default",
    "DBP": "compose_int64_default",
    "RLBE": "rlbe_compose_int64",
    "LECO": "regression_zigzag_bitpack",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def execute(cli: Path, arguments: Iterable[str], *, expected: int = 0) -> subprocess.CompletedProcess[bytes]:
    command = [str(cli), *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {command!r}\n"
            f"stdout={result.stdout.decode(errors='replace')}\n"
            f"stderr={result.stderr.decode(errors='replace')}"
        )
    return result


def json_stdout(result: subprocess.CompletedProcess[bytes], context: str) -> dict[str, Any]:
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{context}: stdout is not one JSON object: {result.stdout!r}") from error
    require(isinstance(value, dict), f"{context}: JSON stdout is not an object")
    return value


def float_word(value: float, dtype: str) -> int:
    if dtype == "float64":
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    raise AssertionError(dtype)


def words_bytes(words: Iterable[int], dtype: str) -> bytes:
    code = "Q" if dtype == "float64" else "I"
    mask = (1 << (64 if dtype == "float64" else 32)) - 1
    values = tuple(word & mask for word in words)
    return struct.pack(f"<{len(values)}{code}", *values)


def mixed_fixture(dtype: str) -> tuple[int, bytes, bytes, int]:
    """Return rows, compact valid values, missing bitmap, decimal places.

    The finite low-magnitude values all have canonical decimal precision at
    most two.  Negative zero, a NaN payload, infinity, and a finite value whose
    scaled integer exceeds int64 exercise the raw-bit exception path.
    """

    decimals = (
        0.0, 0.1, -0.1, 1.25, -2.5, 3.0, 3.0, 4.25, -5.75,
        8.0, 10.24, -9.99, 0.07, 0.07, 9.0, 12.34, -12.34,
    )
    words = [float_word(value, dtype) for value in decimals]
    if dtype == "float64":
        words.extend((
            0x8000000000000000,  # -0.0, whose sign must survive.
            0x7FF8000000001234,  # quiet NaN with a non-default payload.
            0x7FF0000000000000,  # +infinity.
            float_word(1.0e20, dtype),  # finite, but 1e20 * 1e2 exceeds int64.
        ))
    else:
        words.extend((
            0x80000000,
            0x7FC01234,
            0x7F800000,
            float_word(1.0e20, dtype),
        ))
    # Two rows are NULL.  The values stream contains only the 21 valid rows.
    row_count = len(words) + 2
    missing = bytearray((row_count + 7) // 8)
    for row in (3, 17):
        missing[row // 8] |= 1 << (row % 8)
    return row_count, words_bytes(words, dtype), bytes(missing), 2


def pure_exception_fixture(dtype: str) -> tuple[int, bytes, bytes, int]:
    if dtype == "float64":
        words = (
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF800000000BEEF,
            0x7FEFFFFFFFFFFFFF,
        )
    else:
        words = (
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC0BEEF,
            0x7F7FFFFF,
        )
    return len(words), words_bytes(words, dtype), b"\x00", 0


def assert_common_scaled_receipt(
    receipt: dict[str, Any], *, method: str, dtype: str, rows: int,
    decimal_places: int, values: bytes, frame: Path,
) -> None:
    require(receipt.get("status") == "ok", f"{method}/{dtype}: status is not ok")
    require(receipt.get("method") == method, f"{method}/{dtype}: method drift")
    require(receipt.get("dtype") == dtype, f"{method}/{dtype}: dtype drift")
    require(receipt.get("row_count") == rows, f"{method}/{dtype}: row count drift")
    require(receipt.get("profile_version") == SCALED_PROFILE,
            f"{method}/{dtype}: scaled profile is not explicit")
    require(receipt.get("decimal_places") == decimal_places,
            f"{method}/{dtype}: decimal precision is not explicit")
    require(isinstance(receipt.get("base_configuration"), str) and receipt["base_configuration"],
            f"{method}/{dtype}: base configuration missing")
    require(isinstance(receipt.get("scaling_configuration"), str)
            and receipt["scaling_configuration"],
            f"{method}/{dtype}: scaling configuration missing")
    require(receipt.get("original_numeric_bytes") == len(values),
            f"{method}/{dtype}: numeric denominator drift")
    frame_bytes = frame.read_bytes()
    require(receipt.get("complete_frame_bytes") == len(frame_bytes),
            f"{method}/{dtype}: receipt does not count the complete frame")
    require(receipt.get("complete_frame_sha256") == sha256(frame_bytes),
            f"{method}/{dtype}: complete-frame digest mismatch")


def assert_mixed_scaling_stats(receipt: dict[str, Any], dtype: str, context: str) -> None:
    width = 4 if dtype == "float32" else 8
    require(receipt.get("scaled_value_count") == 17,
            f"{context}: scaled lane count drift")
    require(receipt.get("exception_value_count") == 4,
            f"{context}: exception lane count drift")
    require(receipt.get("raw_exception_bytes") == 4 * width,
            f"{context}: raw exception byte count drift")
    metadata_bytes = receipt.get("scaling_metadata_bytes")
    require(isinstance(metadata_bytes, int) and metadata_bytes >= 4 * width,
            f"{context}: scaling metadata omits raw exception bytes")
    fraction = receipt.get("exception_fraction")
    require(isinstance(fraction, (int, float)) and abs(fraction - 4 / 21) < 1e-15,
            f"{context}: exception fraction drift")


def test_inventories(cli: Path) -> None:
    default = json_stdout(execute(cli, ["inventory"]), "default inventory")
    default_methods = {item["label"]: item for item in default.get("methods", [])}
    for method in METHODS:
        require(method in default_methods, f"default inventory lost {method}")
        item = default_methods[method]
        require(item.get("dtypes") == ["int64"],
                f"default {method} silently admitted float input")
        require(item.get("configuration") == OLD_CONFIGURATIONS[method],
                f"default {method} configuration changed")

    scaled = json_stdout(execute(cli, ["inventory-scaled"]), "scaled inventory")
    require(scaled.get("schema") == SCALED_INVENTORY_SCHEMA,
            "scaled inventory schema drift")
    require(scaled.get("profile_version") == SCALED_PROFILE,
            "scaled inventory profile drift")
    methods = scaled.get("methods")
    require(isinstance(methods, list), "scaled inventory methods is not an array")
    labels = [item.get("label") for item in methods]
    require(len(labels) == len(METHODS) and set(labels) == set(METHODS),
            "scaled inventory is not the exact nine-method catalog")
    for item in methods:
        method = item["label"]
        require(item.get("linked") is True, f"scaled {method} is not linked")
        require(item.get("dtypes") == ["float32", "float64"],
                f"scaled {method} dtype contract drift")
        require(item.get("base_configuration") == OLD_CONFIGURATIONS[method],
                f"scaled {method} does not identify its integer backend")
        require(isinstance(item.get("scaling_configuration"), str)
                and item["scaling_configuration"],
                f"scaled {method} omits its scaling configuration")


def test_cold_roundtrips(cli: Path, directory: Path) -> int:
    count = 0
    for dtype in ("float32", "float64"):
        rows, values, missing, decimal_places = mixed_fixture(dtype)
        values_path = directory / f"{dtype}.values.bin"
        missing_path = directory / f"{dtype}.missing.bitmap"
        values_path.write_bytes(values)
        missing_path.write_bytes(missing)
        for method in METHODS:
            stem = f"{method.lower()}-{dtype}"
            frame = directory / f"{stem}.ngf1"
            encoded = json_stdout(
                execute(cli, [
                    "encode-scaled", method, dtype, str(rows), values_path,
                    missing_path, str(decimal_places), frame,
                ]),
                f"encode-scaled {method}/{dtype}",
            )
            assert_common_scaled_receipt(
                encoded, method=method, dtype=dtype, rows=rows,
                decimal_places=decimal_places, values=values, frame=frame,
            )
            assert_mixed_scaling_stats(encoded, dtype, f"encode {method}/{dtype}")

            decoded_values = directory / f"{stem}.decoded.values.bin"
            decoded_missing = directory / f"{stem}.decoded.missing.bitmap"
            decoded = json_stdout(
                execute(cli, ["decode", frame, decoded_values, decoded_missing]),
                f"cold decode {method}/{dtype}",
            )
            require(decoded.get("dtype") == dtype, f"{method}/{dtype}: decoded dtype drift")
            require(decoded.get("row_count") == rows,
                    f"{method}/{dtype}: decoded row count drift")
            require(decoded_values.read_bytes() == values,
                    f"{method}/{dtype}: float bit pattern did not round-trip")
            require(decoded_missing.read_bytes() == missing,
                    f"{method}/{dtype}: NULL bitmap did not round-trip")

            measured_frame = directory / f"{stem}.measured.ngf1"
            measured = json_stdout(
                execute(cli, [
                    "measure-scaled", method, dtype, str(rows), values_path,
                    missing_path, str(decimal_places), "0", "1", measured_frame,
                ]),
                f"measure-scaled {method}/{dtype}",
            )
            assert_common_scaled_receipt(
                measured, method=method, dtype=dtype, rows=rows,
                decimal_places=decimal_places, values=values, frame=measured_frame,
            )
            assert_mixed_scaling_stats(measured, dtype, f"measure {method}/{dtype}")
            for field in ("compression_ns_samples", "decompression_ns_samples"):
                samples = measured.get(field)
                require(isinstance(samples, list) and len(samples) == 1,
                        f"{method}/{dtype}: {field} is not the requested sample set")
                require(isinstance(samples[0], int) and samples[0] >= 0,
                        f"{method}/{dtype}: {field} contains an invalid duration")
            verify_values = directory / f"{stem}.measured.decoded.values.bin"
            verify_missing = directory / f"{stem}.measured.decoded.missing.bitmap"
            execute(cli, ["decode", measured_frame, verify_values, verify_missing])
            require(verify_values.read_bytes() == values,
                    f"{method}/{dtype}: measured frame changed float bits")
            require(verify_missing.read_bytes() == missing,
                    f"{method}/{dtype}: measured frame changed NULLs")
            count += 1
    return count


def test_all_null(cli: Path, directory: Path) -> int:
    """All-NULL columns are N/A but still have a decodable complete frame."""

    count = 0
    row_count = 9
    values = b""
    missing = b"\xff\x01"
    values_path = directory / "all-null.values.bin"
    missing_path = directory / "all-null.missing.bitmap"
    values_path.write_bytes(values)
    missing_path.write_bytes(missing)
    for dtype in ("float32", "float64"):
        for method in METHODS:
            stem = f"all-null-{method.lower()}-{dtype}"
            frame = directory / f"{stem}.ngf1"
            receipt = json_stdout(
                execute(cli, [
                    "encode-scaled", method, dtype, str(row_count), values_path,
                    missing_path, "0", frame,
                ]),
                f"all-null encode-scaled {method}/{dtype}",
            )
            require(receipt.get("status") == "N/A",
                    f"{method}/{dtype}: all-NULL column was not classified N/A")
            require(receipt.get("method") == method,
                    f"{method}/{dtype}: all-NULL method drift")
            require(receipt.get("dtype") == dtype,
                    f"{method}/{dtype}: all-NULL dtype drift")
            require(receipt.get("row_count") == row_count,
                    f"{method}/{dtype}: all-NULL row count drift")
            require(receipt.get("profile_version") == SCALED_PROFILE,
                    f"{method}/{dtype}: all-NULL profile drift")
            require(receipt.get("decimal_places") == 0,
                    f"{method}/{dtype}: all-NULL precision drift")
            require(receipt.get("original_numeric_bytes") == 0,
                    f"{method}/{dtype}: all-NULL denominator is not zero")
            require(receipt.get("scaled_value_count") == 0,
                    f"{method}/{dtype}: all-NULL scaled lane count is not zero")
            require(receipt.get("exception_value_count") == 0,
                    f"{method}/{dtype}: all-NULL exception count is not zero")
            require(receipt.get("raw_exception_bytes") == 0,
                    f"{method}/{dtype}: all-NULL raw exception bytes are not zero")
            require(receipt.get("exception_fraction") is None,
                    f"{method}/{dtype}: all-NULL exception fraction is not null")
            frame_bytes = frame.read_bytes()
            require(receipt.get("complete_frame_bytes") == len(frame_bytes),
                    f"{method}/{dtype}: all-NULL complete-frame size mismatch")
            require(receipt.get("complete_frame_sha256") == sha256(frame_bytes),
                    f"{method}/{dtype}: all-NULL complete-frame digest mismatch")

            decoded_values = directory / f"{stem}.decoded.values.bin"
            decoded_missing = directory / f"{stem}.decoded.missing.bitmap"
            decoded = json_stdout(
                execute(cli, ["decode", frame, decoded_values, decoded_missing]),
                f"all-null cold decode {method}/{dtype}",
            )
            require(decoded.get("dtype") == dtype,
                    f"{method}/{dtype}: all-NULL decoded dtype drift")
            require(decoded.get("row_count") == row_count,
                    f"{method}/{dtype}: all-NULL decoded row count drift")
            require(decoded_values.read_bytes() == values,
                    f"{method}/{dtype}: all-NULL decode fabricated values")
            require(decoded_missing.read_bytes() == missing,
                    f"{method}/{dtype}: all-NULL bitmap did not round-trip")
            count += 1
    return count


def test_integer_kernel_domain_boundary(cli: Path, directory: Path) -> None:
    """A valid scale may still exceed an integer backend's support domain."""

    values = struct.pack("<2d", -9.0e18, 9.0e18)
    values_path = directory / "delta-overflow.values.bin"
    missing_path = directory / "delta-overflow.missing.bitmap"
    values_path.write_bytes(values)
    missing_path.write_bytes(b"\x00")
    rejected_frame = directory / "delta-overflow-2diff.ngf1"
    rejected = execute(
        cli,
        [
            "encode-scaled", "2DIFF", "float64", "2", values_path,
            missing_path, "0", rejected_frame,
        ],
        expected=3,
    )
    require(not rejected_frame.exists(), "2DIFF published a delta-overflow frame")
    require(b"adjacent delta" in rejected.stderr and b"support domain" in rejected.stderr,
            "2DIFF did not expose its precise scaled-int64 support boundary")

    # The same two scaled integers are legal for BOS.  This guards against
    # treating every large magnitude as a scaling exception or broadly hiding
    # backend-specific behavior.
    admitted_frame = directory / "delta-overflow-bos.ngf1"
    admitted = json_stdout(
        execute(cli, [
            "encode-scaled", "BOS", "float64", "2", values_path,
            missing_path, "0", admitted_frame,
        ]),
        "BOS large-delta control",
    )
    assert_common_scaled_receipt(
        admitted, method="BOS", dtype="float64", rows=2,
        decimal_places=0, values=values, frame=admitted_frame,
    )
    decoded_values = directory / "delta-overflow-bos.decoded.values.bin"
    decoded_missing = directory / "delta-overflow-bos.decoded.missing.bitmap"
    execute(cli, ["decode", admitted_frame, decoded_values, decoded_missing])
    require(decoded_values.read_bytes() == values,
            "BOS large-delta control did not preserve float bits")
    require(decoded_missing.read_bytes() == b"\x00",
            "BOS large-delta control changed NULLs")


def test_fail_closed(cli: Path, directory: Path) -> None:
    # The default profile must remain int64-only for all nine methods.
    rows, values, missing, _ = mixed_fixture("float64")
    values_path = directory / "default-rejection.values.bin"
    missing_path = directory / "default-rejection.missing.bitmap"
    values_path.write_bytes(values)
    missing_path.write_bytes(missing)
    for method in METHODS:
        frame = directory / f"default-{method.lower()}.ngf1"
        result = execute(
            cli,
            ["encode", method, "float64", str(rows), values_path, missing_path, frame],
            expected=3,
        )
        require(not frame.exists(), f"default {method} wrote a float frame")
        require(b"unsupported:" in result.stderr,
                f"default {method} did not report an unsupported profile")

    # A nonempty block with no scalable lane is not an integer-codec result.
    for dtype in ("float32", "float64"):
        exception_rows, exception_values, exception_missing, decimal_places = (
            pure_exception_fixture(dtype)
        )
        exception_values_path = directory / f"pure-exception-{dtype}.values.bin"
        exception_missing_path = directory / f"pure-exception-{dtype}.missing.bitmap"
        exception_values_path.write_bytes(exception_values)
        exception_missing_path.write_bytes(exception_missing)
        for method in METHODS:
            frame = directory / f"pure-exception-{method.lower()}-{dtype}.ngf1"
            result = execute(
                cli,
                [
                    "encode-scaled", method, dtype, str(exception_rows),
                    exception_values_path, exception_missing_path,
                    str(decimal_places), frame,
                ],
                expected=3,
            )
            require(not frame.exists(), f"{method}/{dtype}: pure exceptions produced a frame")
            require(b"unsupported:" in result.stderr,
                    f"{method}/{dtype}: pure exceptions were not classified unsupported")

    # Precision below the canonical requirement and precision above the wire
    # cap must fail before publishing a frame.  Either is a configuration/input
    # error rather than a valid compression result.
    low_values = words_bytes((float_word(0.1, "float64"),), "float64")
    low_values_path = directory / "low-precision.values.bin"
    no_missing_path = directory / "one-row.missing.bitmap"
    low_values_path.write_bytes(low_values)
    no_missing_path.write_bytes(b"\x00")
    for precision in (0, 325):
        frame = directory / f"invalid-precision-{precision}.ngf1"
        command = [
            "encode-scaled", "RLBE", "float64", "1", low_values_path,
            no_missing_path, str(precision), frame,
        ]
        result = subprocess.run(
            [str(cli), *map(str, command)], capture_output=True, check=False, timeout=60,
        )
        require(result.returncode in (2, 3),
                f"invalid precision {precision} was accepted: {result.stdout!r}")
        require(not frame.exists(), f"invalid precision {precision} published a frame")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True,
                        help="path to native_baseline86_cli")
    args = parser.parse_args()
    cli = args.cli.resolve()
    require(cli.is_file(), f"CLI does not exist: {cli}")
    test_inventories(cli)
    with tempfile.TemporaryDirectory(prefix="decimal-scaling-cli-") as temporary:
        directory = Path(temporary)
        roundtrips = test_cold_roundtrips(cli, directory)
        all_null_roundtrips = test_all_null(cli, directory)
        test_integer_kernel_domain_boundary(cli, directory)
        test_fail_closed(cli, directory)
    print(
        f"decimal scaling CLI contract passed: cold_roundtrips={roundtrips} "
        f"all_null_roundtrips={all_null_roundtrips} "
        f"methods={len(METHODS)} dtypes=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
