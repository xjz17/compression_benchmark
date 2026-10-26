#!/usr/bin/env python3
"""Exact old/new differential checks for the v6 native feature fast path.

This is a CLI semantics test.  Its wall time includes process startup and must
not be reported as feature-extraction latency; the companion C++ harness times
the feature functions in process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from typing import Iterable


WEB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD = WEB_ROOT / "cpp/build-native-features-before-v6/native_features_cli"
DEFAULT_NEW = WEB_ROOT / "cpp/build-native-features-v6/native_features_cli"
DEFAULT_RECEIPT = Path(tempfile.gettempdir()) / "native_features_equivalence_v6.json"
FUZZ_SEED = 0x16620260922


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flow(kind: str, width: int, view: str, payload: bytes) -> tuple[str, int, str, bytes]:
    return kind, width, view, payload


def task_request(flows: Iterable[tuple[str, int, str, bytes]]) -> str:
    values = list(flows)
    lines = [str(len(values))]
    for kind, width, view, payload in values:
        lines.append(f"{kind} {width} {view} {payload.hex() or '-'}")
    return "\n".join(lines) + "\n"


def state_request(flows: list[tuple[str, int, str, bytes]], *, objective: str = "2.75") -> str:
    total = sum(len(value[3]) for value in flows) or 1
    lines = [task_request(flows).rstrip("\n")]
    # raw action_count nodes policy_edits replayed edit_count continuation
    # candidate_used candidate_limit pending_kind pending_index objective
    lines.append(f"{total} 6341 12 1 9 1 0 7 64 replace 3 {objective}")
    for index in range(len(flows)):
        lines.append(f"{index % 10} {(-1 if index == 0 else (index * 37) % 6341)}")
    return "\n".join(lines) + "\n"


def pack(fmt: str, values: Iterable[int | float]) -> bytes:
    values = list(values)
    return struct.pack("<" + fmt * len(values), *values)


def splitmix_hash(key: int) -> int:
    mask = (1 << 64) - 1
    value = key & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def collision_f64_payload(count: int = 24) -> tuple[bytes, int]:
    # count=24 gives table capacity 64.  Search values whose *binary64 keys*
    # land in one initial bucket so linear-probing collision handling is tested.
    buckets: dict[int, list[float]] = {}
    chosen: list[float] | None = None
    chosen_bucket = -1
    for integer in range(1, 200_000):
        value = float(integer)
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        bucket = splitmix_hash(bits) & 63
        group = buckets.setdefault(bucket, [])
        group.append(value)
        if len(group) == count:
            chosen, chosen_bucket = group, bucket
            break
    if chosen is None:
        raise AssertionError("failed to construct collision chain")
    return pack("d", chosen), chosen_bucket


def explicit_cases() -> tuple[list[tuple[str, str, str]], int]:
    cases: list[tuple[str, str, str]] = []

    integer_specs = [
        ("int8", "b", 1, [-(1 << 7), -1, 0, 1, (1 << 7) - 1, 0]),
        ("uint8", "B", 1, [0, 1, (1 << 8) - 1, 0, (1 << 8) - 1]),
        ("int16", "h", 2, [-(1 << 15), -1, 0, 1, (1 << 15) - 1]),
        ("uint16", "H", 2, [0, 1, (1 << 16) - 1, 0, (1 << 16) - 1]),
        ("int32", "i", 4, [-(1 << 31), -1, 0, 1, (1 << 31) - 1]),
        ("uint32", "I", 4, [0, 1, (1 << 32) - 1, 0, (1 << 32) - 1]),
        ("int64", "q", 8, [-(1 << 63), -(1 << 53) - 1, -1, 0, 1, (1 << 53) + 1, (1 << 63) - 1]),
        ("uint64", "Q", 8, [0, 1, (1 << 53) - 1, (1 << 53), (1 << 53) + 1, (1 << 64) - 1]),
    ]
    for view, fmt, width, values in integer_specs:
        request = task_request([flow("numeric", width, view, pack(fmt, values))])
        cases.append((f"integer_extrema_{view}", "task", request))

    # Raw floating bit patterns retain signed zero and exact NaN/Inf payloads.
    floating_specs = [
        ("float16", "H", 2, [0x0000, 0x8000, 0x0001, 0x3C00, 0xC100, 0x7C00, 0xFC00, 0x7E01, 0x3C00]),
        ("float32", "I", 4, [0, 0x80000000, 1, 0x3F800000, 0xC0200000, 0x7F800000, 0xFF800000, 0x7FC12345, 0x3F800000]),
        ("float64", "Q", 8, [0, 0x8000000000000000, 1, 0x3FF0000000000000, 0xC004000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8123456789ABC, 0x3FF0000000000000]),
    ]
    for view, fmt, width, bits in floating_specs:
        cases.append((f"signed_zero_nonfinite_{view}", "task",
                      task_request([flow("numeric", width, view, pack(fmt, bits))])))

    for count in (8191, 8192, 8193):
        values = [((index * 104729) % 2_000_003) - 1_000_001 for index in range(count)]
        cases.append((f"sampling_boundary_int64_{count}", "task",
                      task_request([flow("numeric", 8, "int64", pack("q", values))])))

    collision_payload, collision_bucket = collision_f64_payload()
    cases.append((f"hash_collision_chain_bucket_{collision_bucket}", "task",
                  task_request([flow("numeric", 8, "float64", collision_payload)])))

    roots: list[tuple[str, int, str, bytes]] = []
    for index in range(8):
        roots.extend([
            flow("numeric", 8, "int64", pack("q", [index, -index, index << 54, index])),
            flow("numeric", 4, "uint32", pack("I", [index, 0, (1 << 32) - 1, index])),
            flow("numeric", 4, "float32", pack("I", [0, 0x80000000, 0x3F800000 + index, 0x7FC00001])),
            flow("serial", 1, "bytes", bytes([index, index, 255 - index])),
        ])
    cases.append(("multiple_roots_32_mixed", "task", task_request(roots)))
    cases.append(("state_multiple_roots_bucket_collisions", "state", state_request(roots[:14])))
    bytes_only = [flow("serial", 1, "bytes", bytes([index, index, 255 - index]))
                  for index in range(32)]
    cases.append(("bytes_only_32_tiny_roots", "task", task_request(bytes_only)))
    cases.append(("tiny_numeric_single_value", "task",
                  task_request([flow("numeric", 4, "int32", pack("i", [17]))])))

    # Expected-rejection parity: partial elements, mismatched view, and an
    # overflowing finite-data feature must fail closed in both binaries.
    cases.extend([
        ("reject_partial_element", "task", "1\nnumeric 4 int32 0000\n"),
        ("reject_view_width_mismatch", "task", "1\nnumeric 4 float64 00000000\n"),
        ("reject_nonfinite_state_objective", "state",
         state_request([flow("numeric", 4, "int32", pack("i", [1, 2]))], objective="nan")),
        ("reject_overflow_feature", "task", task_request([
            flow("numeric", 8, "float64", pack("Q", [0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF] * 2))
        ])),
    ])
    return cases, collision_bucket


def fuzz_cases(count: int) -> list[tuple[str, str, str]]:
    rng = random.Random(FUZZ_SEED)
    specs = [
        ("int8", "B", 1), ("uint8", "B", 1),
        ("int16", "H", 2), ("uint16", "H", 2), ("float16", "H", 2),
        ("int32", "I", 4), ("uint32", "I", 4), ("float32", "I", 4),
        ("int64", "Q", 8), ("uint64", "Q", 8), ("float64", "Q", 8),
    ]
    output: list[tuple[str, str, str]] = []
    for case_index in range(count):
        roots = []
        for _ in range(rng.randint(1, 8)):
            view, fmt, width = rng.choice(specs)
            length = rng.randint(0, 96)
            mask = (1 << (width * 8)) - 1
            words = [rng.getrandbits(width * 8) & mask for _ in range(length)]
            roots.append(flow("numeric", width, view, pack(fmt, words)))
        output.append((f"fuzz_{case_index:03d}", "task", task_request(roots)))
    return output


def execute(cli: Path, mode: str, request: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(cli), mode], input=request.encode(), capture_output=True, timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--fuzz-cases", type=int, default=80)
    args = parser.parse_args()
    for cli in (args.old, args.new):
        if not cli.is_file():
            raise SystemExit(f"missing CLI: {cli}")

    cases, collision_bucket = explicit_cases()
    cases += fuzz_cases(args.fuzz_cases)
    passed = []
    mismatches = []
    rejected = []
    for name, mode, request in cases:
        old = execute(args.old, mode, request)
        new = execute(args.new, mode, request)
        same = (old.returncode, old.stdout, old.stderr) == (new.returncode, new.stdout, new.stderr)
        if not same:
            mismatches.append({
                "name": name, "mode": mode,
                "old_returncode": old.returncode, "new_returncode": new.returncode,
                "old_stdout_sha256": hashlib.sha256(old.stdout).hexdigest(),
                "new_stdout_sha256": hashlib.sha256(new.stdout).hexdigest(),
                "old_stderr": old.stderr.decode(errors="replace"),
                "new_stderr": new.stderr.decode(errors="replace"),
            })
        else:
            passed.append(name)
            if old.returncode:
                rejected.append(name)

    receipt = {
        "schema": "native-features-old-new-equivalence-v6",
        "status": "pass" if not mismatches else "fail",
        "comparison": "exact return code, stdout bytes, and stderr bytes",
        "timing_scope": "none; CLI process wall time is deliberately not measured",
        "old_cli": str(args.old.resolve()),
        "old_cli_sha256": sha256(args.old),
        "new_cli": str(args.new.resolve()),
        "new_cli_sha256": sha256(args.new),
        "case_count": len(cases),
        "explicit_case_count": len(cases) - args.fuzz_cases,
        "fuzz_case_count": args.fuzz_cases,
        "fuzz_seed": FUZZ_SEED,
        "collision_initial_bucket": collision_bucket,
        "coverage": [
            "signed and unsigned widths", "float16/32/64", "+0 and -0",
            "NaN and +/-Inf", "8191/8192/8193 sampling", "multiple roots",
            "linear-probing hash collisions", "state buckets", "fail-closed inputs",
        ],
        "passed_cases": passed,
        "expected_rejections": rejected,
        "mismatches": mismatches,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "case_count", "explicit_case_count", "fuzz_case_count", "mismatches")}, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
