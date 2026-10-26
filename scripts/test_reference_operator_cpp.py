#!/usr/bin/env python3
"""Freeze original Python OPA1 streams, then test the independent C++ port.

Capture must run before deleting the old kernels. Verification imports no
Python codec and uses only saved raw bytes, action names and original wires.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import warnings
import zlib

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "analysis_outputs/reference_operator_cpp_migration_v1/python_goldens.json"
CLI = ROOT / "cpp/build/reference_operator_actions_cli"
RETIRED_FORWARD_ACTIONS = {"op.scaling.reference_v1"}


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def capture(path: Path) -> None:
    raise RuntimeError("Historical Python reference kernels retired; verify the existing frozen oracle.")
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen oracle: {path}")
    import numpy as np
    import primitive_operator_action_runtime as original
    import audit_86_composability_realdata as realdata

    names = [action.name for action in original.PRIMITIVE_ACTIONS]
    sequences = [[name] for name in names]
    sequences += [[], [names[7], names[11], names[6]], [names[3], names[7]],
                  [names[4], names[19]], [names[28], names[9]],
                  [names[14], names[10]], [names[0], names[20]],
                  [names[2], names[6]], [names[17], names[26]],
                  [names[27], names[15], names[21]]]
    inputs = []
    for rows in (128, 2048):
        columns, _ = realdata.load_external_columns(rows)
        inputs.extend((f"real/{col.dataset}/{col.name}/{rows}/{col.logical_type}", col.raw)
                      for col in columns)
    rng = random.Random(20260905)
    edge_words = [0, (1 << 64) - 1, (1 << 63), (1 << 63) - 1,
                  (1 << 53) + 1, ((-(1 << 53) - 1) & ((1 << 64) - 1)),
                  0x7ff0000000000000, 0xfff0000000000000, 0x7ff800000000002a]
    cases = [[], [0], [1], [(1 << 63)], edge_words,
             [0, 127, 128, 255, 256, 32767, 32768, (1 << 31) - 1, 1 << 31],
             [42] * 2048, list(range(65)), [9, 7, 7, 7, 7, -2, -2, 0] * 32]
    for count in (7, 31, 64, 129, 251):
        cases.append([rng.getrandbits(64) for _ in range(count)])
    for index, words in enumerate(cases):
        inputs.append((f"edge/{index}", struct.pack(
            f"<{len(words)}Q", *(word & ((1 << 64) - 1) for word in words))))
    result = {"schema": "frozen-python-reference-opa1-v1", "actions": names,
              "source_sha256": {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                                for file in (ROOT / "scripts/primitive_operator_action_runtime.py",
                                             ROOT / "scripts/generate_air_operator_pair_heatmap.py",
                                             ROOT / "scripts/measure_selected_codec_cost.py")},
              "cases": []}
    counts = {"wire_records": 0, "lossless_records": 0, "encode_errors": 0, "decode_errors": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for label, raw in inputs:
            case = {"label": label, "raw": b64(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(), "records": []}
            values = np.frombuffer(raw, dtype="<i8")
            for sequence in sequences:
                record = {"actions": sequence}
                try:
                    wire = original.encode_operator_pipeline(values, sequence)
                    record["wire"] = b64(wire)
                    counts["wire_records"] += 1
                    try:
                        decoded, decoded_names = original.decode_operator_pipeline(wire)
                        record["lossless"] = decoded.tobytes() == raw and tuple(sequence) == decoded_names
                        counts["lossless_records"] += record["lossless"]
                    except Exception as error:
                        record["lossless"] = False
                        record["decode_error"] = f"{type(error).__name__}: {error}"
                        counts["decode_errors"] += 1
                except Exception as error:
                    record["encode_error"] = f"{type(error).__name__}: {error}"
                    counts["encode_errors"] += 1
                case["records"].append(record)
            result["cases"].append(case)
    result["counts"] = counts
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"goldens": str(path), "cases": len(inputs), **counts}))


def verify(path: Path, cli: Path) -> None:
    golden = json.loads(path.read_text())
    failures, wire_checks, roundtrips, legacy_negatives, state_checks = [], 0, 0, 0, 0
    retired_forward_rejections, retired_wire_decodes = 0, 0
    for case_index, case in enumerate(golden["cases"]):
        raw = base64.b64decode(case["raw"])
        assert hashlib.sha256(raw).hexdigest() == case["raw_sha256"]
        for record in case["records"]:
            if "wire" not in record:
                legacy_negatives += 1
                rejected = subprocess.run([str(cli), "encode", *record["actions"]], input=raw, capture_output=True)
                if rejected.returncode == 0:
                    failures.append({"case": case["label"], "actions": record["actions"],
                                     "kind": "accepted-legacy-encode-error"})
                continue
            command = [str(cli), "encode", *record["actions"]]
            encoded = subprocess.run(command, input=raw, capture_output=True)
            expected = base64.b64decode(record["wire"])
            if RETIRED_FORWARD_ACTIONS.intersection(record["actions"]):
                if encoded.returncode == 0:
                    failures.append({"case": case["label"], "actions": record["actions"],
                                     "kind": "accepted-retired-forward-action"})
                else:
                    retired_forward_rejections += 1
                if record.get("lossless"):
                    decoded = subprocess.run([str(cli), "decode"], input=expected, capture_output=True)
                    names_output = subprocess.run([str(cli), "actions"], input=expected, capture_output=True)
                    expected_state = zlib.decompress(expected)[6 + len(record["actions"]):]
                    inverse_output = subprocess.run(
                        [str(cli), "invert-state", *record["actions"]],
                        input=expected_state, capture_output=True)
                    if decoded.returncode or decoded.stdout != raw:
                        failures.append({"case": case["label"], "actions": record["actions"],
                                         "kind": "retired-cold-decode"})
                    elif names_output.returncode or names_output.stdout.decode().splitlines() != record["actions"]:
                        failures.append({"case": case["label"], "actions": record["actions"],
                                         "kind": "retired-action-name"})
                    elif inverse_output.returncode or inverse_output.stdout != raw:
                        failures.append({"case": case["label"], "actions": record["actions"],
                                         "kind": "retired-state-inverse"})
                    else:
                        retired_wire_decodes += 1
                continue
            if encoded.returncode or encoded.stdout != expected:
                failures.append({"case": case["label"], "actions": record["actions"],
                                 "kind": "encode", "expected_bytes": len(expected),
                                 "actual_bytes": len(encoded.stdout), "error": encoded.stderr.decode(errors="replace")})
                continue
            wire_checks += 1
            if record.get("lossless"):
                decoded = subprocess.run([str(cli), "decode"], input=expected, capture_output=True)
                if decoded.returncode or decoded.stdout != raw:
                    failures.append({"case": case["label"], "actions": record["actions"],
                                     "kind": "decode", "error": decoded.stderr.decode(errors="replace")})
                else:
                    roundtrips += 1
            else:
                legacy_negatives += 1
                rejected = subprocess.run([str(cli), "decode"], input=expected, capture_output=True)
                if rejected.returncode == 0:
                    failures.append({"case": case["label"], "actions": record["actions"],
                                     "kind": "accepted-legacy-invalid-roundtrip"})
            if case_index in (0, 8, 16, len(golden["cases"]) - 1):
                original_uncompressed = zlib.decompress(expected)
                expected_state = original_uncompressed[6 + len(record["actions"]):]
                for command, expected_output in (
                    ("state", expected_state), ("raw-encode", original_uncompressed),
                ):
                    output = subprocess.run([str(cli), command, *record["actions"]], input=raw, capture_output=True)
                    if output.returncode or output.stdout != expected_output:
                        failures.append({"case": case["label"], "actions": record["actions"], "kind": command})
                    else:
                        state_checks += 1
                # Only parse the frozen format here: no Python transform or
                # inverse implementation is imported during verification.
                offset = 0
                for _ in record["actions"]:
                    arrays, extra = struct.unpack_from("<HI", expected_state, offset + 12)
                    offset += 18 + extra
                    for _ in range(arrays):
                        _count, byte_count = struct.unpack_from("<II", expected_state, offset)
                        offset += 8 + byte_count
                tag = expected_state[offset:offset + 2]
                count = struct.unpack_from("<I", expected_state, offset + 2)[0]
                formats = {b"U1": "B", b"I1": "b", b"I2": "h", b"I4": "i", b"I8": "q"}
                values = (() if tag == b"I0" else struct.unpack_from(f"<{count}{formats[tag]}", expected_state, offset + 6))
                value_output = subprocess.run([str(cli), "values", *record["actions"]], input=raw, capture_output=True)
                names_output = subprocess.run([str(cli), "actions"], input=expected, capture_output=True)
                if value_output.returncode or value_output.stdout != struct.pack(f"<{count}q", *values):
                    failures.append({"case": case["label"], "actions": record["actions"], "kind": "values"})
                elif names_output.returncode or names_output.stdout.decode().splitlines() != record["actions"]:
                    failures.append({"case": case["label"], "actions": record["actions"], "kind": "actions"})
                else:
                    state_checks += 2
                if record.get("lossless"):
                    inverse_output = subprocess.run([str(cli), "invert-state", *record["actions"]], input=expected_state, capture_output=True)
                    if inverse_output.returncode or inverse_output.stdout != raw:
                        failures.append({"case": case["label"], "actions": record["actions"], "kind": "invert-state"})
                    else:
                        state_checks += 1
    print(json.dumps({"status": "FAIL" if failures else "PASS", "wire_checks": wire_checks,
                      "bitwise_roundtrips": roundtrips, "state_bridge_checks": state_checks,
                      "retired_forward_rejections": retired_forward_rejections,
                      "retired_wire_decodes": retired_wire_decodes,
                      "legacy_negative_cases": legacy_negatives,
                      "failures": failures[:60], "failure_count": len(failures)}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--goldens", type=Path, default=GOLDENS)
    parser.add_argument("--cli", type=Path, default=CLI)
    args = parser.parse_args()
    capture(args.goldens) if args.capture else verify(args.goldens, args.cli)
