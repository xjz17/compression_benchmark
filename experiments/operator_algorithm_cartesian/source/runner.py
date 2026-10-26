#!/usr/bin/env python3
"""Isolated carrier and baseline-CLI runner for real joint cells.

This file deliberately does not synthesize a joint result from independent
operator/algorithm winners.  It accepts the actual O2R1 bytes for exactly one
operator block, carrierizes them, executes the bound algorithm binary, verifies
the algorithm decode back to those exact O2R1 bytes, and requires a separate
operator-inverse receipt before it emits a JOA1 frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from carrier import ACTIVE85, carrier_kind_for_algorithm, pack_bytes, unpack_bytes
from joint_frame import build_frame, parse_frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run_checked(argv: list[str], *, env: dict[str, str] | None = None
                ) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv, text=True, capture_output=True, check=False, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result


def carrierize(args: argparse.Namespace) -> int:
    payload = args.o2r1.read_bytes()
    kind = carrier_kind_for_algorithm(args.algorithm)
    packed = pack_bytes(payload, kind)
    args.output.mkdir(parents=True, exist_ok=False)
    write_exclusive(args.output / "values.bin", packed.values)
    write_exclusive(args.output / "missing.bin", packed.missing)
    write_exclusive(args.output / "carrier.dbc1", packed.metadata)
    receipt = {
        "schema": "operator-algorithm-byte-carrier-v1",
        "status": "ready",
        "algorithm": args.algorithm,
        "carrier_kind": packed.kind,
        "dtype": packed.dtype,
        "row_count": packed.row_count,
        "source_o2r1_bytes": len(payload),
        "source_o2r1_sha256": hashlib.sha256(payload).hexdigest(),
        "carrier_numeric_bytes": len(packed.values),
        "carrier_missing_bitmap_bytes": len(packed.missing),
        "carrier_metadata_bytes": len(packed.metadata),
        "carrier_padding_bytes": packed.padding_bytes,
        "lane_bytes": packed.lane_bytes,
        "storage_width": packed.storage_width,
        "values_sha256": hashlib.sha256(packed.values).hexdigest(),
        "missing_sha256": hashlib.sha256(packed.missing).hexdigest(),
        "metadata_sha256": hashlib.sha256(packed.metadata).hexdigest(),
        "contract": (
            "O2R1 bytes -> declared reversible native-dtype carrier; all carrier "
            "numeric bytes, zero missing bitmap, DBC1 metadata and final algorithm "
            "frame are charged"
        ),
    }
    write_exclusive(
        args.output / "CARRIER.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _one_json_line(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("native CLI did not emit exactly one JSON line")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise RuntimeError("native CLI response is not an object")
    return value


def _median3(samples: list[int]) -> int:
    if len(samples) != 3:
        raise ValueError("W1/R3 requires exactly three measured samples")
    return sorted(samples)[1]


def _time_carrier_pack(payload: bytes, kind: str):
    # One untimed warmup, then three complete pack operations including DBC1
    # SHA/metadata and the canonical all-present missing bitmap.
    warm = pack_bytes(payload, kind)
    samples: list[int] = []
    packed = warm
    for iteration in range(4):
        begin = time.perf_counter_ns()
        candidate = pack_bytes(payload, kind)
        elapsed = time.perf_counter_ns() - begin
        if candidate != warm:
            raise RuntimeError("carrier pack is not deterministic")
        if iteration:
            samples.append(elapsed)
        packed = candidate
    return packed, samples


def _time_carrier_unpack(values: bytes, missing: bytes, metadata: bytes,
                         expected: bytes) -> list[int]:
    samples: list[int] = []
    for iteration in range(4):
        begin = time.perf_counter_ns()
        restored = unpack_bytes(values, missing, metadata)
        elapsed = time.perf_counter_ns() - begin
        if restored != expected:
            raise RuntimeError("carrier inverse changed O2R1 bytes")
        if iteration:
            samples.append(elapsed)
    return samples


def measure_one(args: argparse.Namespace) -> int:
    if args.algorithm not in ACTIVE85:
        raise ValueError("algorithm is not in frozen active85")
    if sha256(args.binary) != args.expected_binary_sha256:
        raise ValueError("algorithm binary SHA-256 mismatch")
    native_env = os.environ.copy()
    if args.ld_library_path:
        prior = native_env.get("LD_LIBRARY_PATH")
        native_env["LD_LIBRARY_PATH"] = (
            args.ld_library_path if not prior else f"{args.ld_library_path}:{prior}"
        )
    inverse = read_json(args.operator_inverse_receipt)
    required_inverse = {
        "status": "passed",
        "bit_exact": True,
        "applicable": True,
        "strict": True,
        "fallback_used": False,
        "operator_id": args.operator_id,
        "source_column_sha256": args.source_column_sha256,
        "o2r1_sha256": sha256(args.o2r1),
    }
    for key, expected in required_inverse.items():
        if inverse.get(key) != expected:
            raise ValueError(f"operator inverse receipt differs at {key}")

    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="joint-cell-", dir=args.output) as temp:
        temp_path = Path(temp)
        payload = args.o2r1.read_bytes()
        packed, carrier_pack_samples = _time_carrier_pack(
            payload, carrier_kind_for_algorithm(args.algorithm)
        )
        values = temp_path / "values.bin"
        missing = temp_path / "missing.bin"
        metadata = temp_path / "carrier.dbc1"
        frame = temp_path / "algorithm.ngf"
        decoded_values = temp_path / "decoded.values"
        decoded_missing = temp_path / "decoded.missing"
        values.write_bytes(packed.values)
        missing.write_bytes(packed.missing)
        metadata.write_bytes(packed.metadata)

        measured = run_checked([
            str(args.binary), "measure", args.algorithm, packed.dtype,
            str(packed.row_count), str(values), str(missing), "1", "3", str(frame),
        ], env=native_env)
        measurement = _one_json_line(measured.stdout)
        if measurement.get("status") not in {"ok", "N/A"}:
            raise RuntimeError("algorithm measurement did not return ok")
        if measurement.get("status") == "N/A":
            receipt = {
                "schema": "operator-algorithm-joint-cell-result-v1",
                "status": "not_applicable",
                "bit_exact": False,
                "ranking_eligible": False,
                "algorithm": args.algorithm,
                "operator_id": args.operator_id,
                "carrier_kind": packed.kind,
                "algorithm_measurement": measurement,
                "reason": measurement.get(
                    "reason", "algorithm rejected the reversible carrier domain"
                ),
                "independent_optima_combined": False,
            }
            write_exclusive(
                args.output / "RESULT.json",
                (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
            )
            print(json.dumps(receipt, sort_keys=True))
            return 0
        run_checked([
            str(args.binary), "decode", str(frame), str(decoded_values),
            str(decoded_missing),
        ], env=native_env)
        carrier_unpack_samples = _time_carrier_unpack(
            decoded_values.read_bytes(), decoded_missing.read_bytes(),
            packed.metadata, payload,
        )

        binding = {
            "schema": "operator-algorithm-joint-binding-v1",
            "scope": args.scope,
            "dataset_id": args.dataset_id,
            "column_id": args.column_id,
            "operator_id": args.operator_id,
            "operator_catalog_sha256": args.operator_catalog_sha256,
            "operator_inverse_receipt_sha256": sha256(args.operator_inverse_receipt),
            "algorithm": args.algorithm,
            "algorithm_binary_sha256": args.expected_binary_sha256,
            "algorithm_profile": args.algorithm_profile,
            "carrier_kind": packed.kind,
            "carrier_lane_bytes": packed.lane_bytes,
            "carrier_storage_width": packed.storage_width,
            "o2r1_sha256": hashlib.sha256(payload).hexdigest(),
            "timing_contract": "algorithm native W1/R3/T1; process and file IO excluded",
        }
        joint = build_frame(
            binding, packed.metadata, frame.read_bytes(), args.source_column_sha256
        )
        parsed = parse_frame(joint)
        write_exclusive(args.output / "result.joa1", joint)
        receipt = {
            "schema": "operator-algorithm-joint-cell-result-v1",
            "status": "ok",
            "bit_exact": True,
            "source_to_operator_inverse_bit_exact": True,
            "algorithm_frame_to_o2r1_bit_exact": True,
            "binding": binding,
            "source_numeric_bytes": args.source_numeric_bytes,
            "o2r1_bytes": len(payload),
            "carrier_numeric_bytes": len(packed.values),
            "carrier_missing_bitmap_bytes": len(packed.missing),
            "carrier_metadata_bytes": len(packed.metadata),
            "carrier_padding_bytes": packed.padding_bytes,
            "algorithm_complete_frame_bytes": frame.stat().st_size,
            "joint_complete_frame_bytes": len(joint),
            "compression_ratio": (
                len(joint) / args.source_numeric_bytes
                if args.source_numeric_bytes else None
            ),
            "algorithm_measurement": measurement,
            "carrier_timing": {
                "clock": "time.perf_counter_ns",
                "warmups": 1,
                "repeats": 3,
                "pack_ns_samples": carrier_pack_samples,
                "pack_median_ns": _median3(carrier_pack_samples),
                "unpack_ns_samples": carrier_unpack_samples,
                "unpack_median_ns": _median3(carrier_unpack_samples),
                "boundary": (
                    "pack includes O2R1-to-native-dtype values, canonical missing "
                    "bitmap, DBC1 metadata and SHA; unpack includes validation, "
                    "native-dtype-to-O2R1 reconstruction and SHA"
                ),
            },
            "joint_frame_sha256": parsed["complete_frame_sha256"],
            "ranking_eligible": True,
            "independent_optima_combined": False,
        }
        write_exclusive(
            args.output / "RESULT.json",
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    carrier = sub.add_parser("carrierize")
    carrier.add_argument("--algorithm", required=True)
    carrier.add_argument("--o2r1", type=Path, required=True)
    carrier.add_argument("--output", type=Path, required=True)
    carrier.set_defaults(function=carrierize)

    measure = sub.add_parser("measure-one")
    measure.add_argument("--algorithm", required=True)
    measure.add_argument("--binary", type=Path, required=True)
    measure.add_argument("--expected-binary-sha256", required=True)
    measure.add_argument("--algorithm-profile", required=True)
    measure.add_argument("--ld-library-path")
    measure.add_argument("--operator-id", required=True)
    measure.add_argument("--operator-catalog-sha256", required=True)
    measure.add_argument("--operator-inverse-receipt", type=Path, required=True)
    measure.add_argument("--o2r1", type=Path, required=True)
    measure.add_argument("--source-column-sha256", required=True)
    measure.add_argument("--source-numeric-bytes", type=int, required=True)
    measure.add_argument("--scope", default="current200")
    measure.add_argument("--dataset-id", required=True)
    measure.add_argument("--column-id", required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.set_defaults(function=measure_one)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
