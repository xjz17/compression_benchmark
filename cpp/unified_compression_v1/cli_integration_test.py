#!/usr/bin/env python3
"""Cross-process checks for the unified compression CLI.

Python only coordinates the executable and verifies its wire contract. All
compression, decompression, timing, framing, and bit-exact checks happen in C++.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(cli: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(cli), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def one_json(process: subprocess.CompletedProcess[str]) -> dict:
    require(process.returncode == 0, f"CLI failed: {process.stdout} {process.stderr}")
    lines = process.stdout.splitlines()
    require(len(lines) == 1, f"expected one JSON response, received {len(lines)}")
    return json.loads(lines[0])


def select_int64_algorithm(inventory: dict) -> str:
    eligible = []
    for method in inventory["methods"]:
        if (
            method["kind"] == "algorithm"
            and method["enabled"]
            and method["available"]
            and "int64" in method["dtypes"]
        ):
            eligible.append(method)
            if "Zstd" in method["aliases"]:
                return "Zstd"
    for method in eligible:
        if {"int32", "int64", "float32", "float64"}.issubset(method["dtypes"]):
            return method["name"]
    raise AssertionError("inventory has no enabled, available four-dtype algorithm")


def main() -> None:
    require(len(sys.argv) >= 2, "usage: cli_integration_test.py CLI [CATALOG]")
    cli = pathlib.Path(sys.argv[1]).resolve()
    common = ["--operator-catalog", str(pathlib.Path(sys.argv[2]).resolve())] if len(sys.argv) > 2 else []

    inventory = one_json(run(cli, *common, "inventory"))
    require(inventory["status"] == "ok", "inventory did not succeed")
    require(inventory["method_count"] == len(inventory["methods"]), "inventory count mismatch")
    require(all("contract" in item for item in inventory["methods"]), "contract missing")
    method = select_int64_algorithm(inventory)

    markdown = run(cli, *common, "inventory", "--markdown")
    require(markdown.returncode == 0, "Markdown inventory failed")
    require("input/output contract" in markdown.stdout, "Markdown contract column missing")

    with tempfile.TemporaryDirectory(prefix="unified-compression-cli-") as directory:
        root = pathlib.Path(directory)
        frame = root / "frame.ucc1"
        request_path = root / "request.json"
        request = {
            "method": method,
            "dtype": "int64",
            "values": [-(2**63), None, 2**63 - 1, 0],
            "warmups": 0,
            "repeats": 1,
            "output_path": str(frame),
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        compressed = one_json(run(cli, *common, "compress", str(request_path)))
        require(compressed["status"] == "ok", "compression did not succeed")
        require(compressed["bit_exact"] is True, "compression was not bit-exact")
        require(compressed["compressed_bytes"] == frame.stat().st_size, "frame size mismatch")
        require(
            compressed["compressed_sha256"] == hashlib.sha256(frame.read_bytes()).hexdigest(),
            "frame digest mismatch",
        )
        for field in (
            "compression_ratio",
            "compression_time_ns",
            "compression_time_ms",
            "decompression_time_ns",
            "decompression_time_ms",
        ):
            require(field in compressed, f"missing metric {field}")

        restored = one_json(run(cli, *common, "decompress", str(frame)))
        expected_values = struct.pack("<q", -(2**63)) + struct.pack("<q", 2**63 - 1) + struct.pack("<q", 0)
        require(restored["dtype"] == "int64", "restored dtype mismatch")
        require(restored["row_count"] == 4, "restored row count mismatch")
        require(restored["values_le_hex"] == expected_values.hex(), "int64 bytes lost precision")
        require(restored["missing_bitmap_hex"] == "02", "missing bitmap mismatch")
        cold = one_json(
            run(
                cli,
                "--operator-catalog",
                str(root / "deliberately-absent-catalog.json"),
                "decompress",
                str(frame),
            )
        )
        require(cold["values_le_hex"] == expected_values.hex(), "cold decode read a catalog")

        operator_frame = root / "operator-a013.ucc1"
        operator_request_path = root / "operator-request.json"
        operator_request_path.write_text(
            json.dumps(
                {
                    "method": "operator:A013",
                    "dtype": "int64",
                    "values": [10, None, 12],
                    "warmups": 0,
                    "repeats": 1,
                    "output_path": str(operator_frame),
                }
            ),
            encoding="utf-8",
        )
        operator_compressed = one_json(
            run(cli, *common, "compress", str(operator_request_path))
        )
        require(operator_compressed["bit_exact"] is True, "A013 compression was not bit-exact")
        operator_cold = one_json(
            run(
                cli,
                "--operator-catalog",
                str(root / "deliberately-absent-operator-catalog.json"),
                "decompress",
                str(operator_frame),
            )
        )
        expected_operator_values = struct.pack("<q", 10) + struct.pack("<q", 12)
        require(operator_cold["dtype"] == "int64", "A013 cold dtype mismatch")
        require(operator_cold["row_count"] == 3, "A013 cold row count mismatch")
        require(
            operator_cold["values_le_hex"] == expected_operator_values.hex(),
            "A013 cold numeric bytes mismatch",
        )
        require(operator_cold["missing_bitmap_hex"] == "02", "A013 cold null bitmap mismatch")

        frame_base64 = base64.b64encode(frame.read_bytes()).decode("ascii")
        lines = [
            "{",
            json.dumps({"command": "compress", "method": method, "dtype": "int32", "values": [2**31]}),
            json.dumps({"command": "decompress", "compressed_base64": frame_base64}),
            json.dumps({"command": "compress", "method": method, "dtype": "int64", "values": [None, None]}),
        ]
        served = subprocess.run(
            [str(cli), *common, "serve"],
            input="\n".join(lines) + "\n",
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(served.returncode == 0, f"serve failed: {served.stderr}")
        responses = [json.loads(line) for line in served.stdout.splitlines()]
        require(len(responses) == len(lines), "serve did not answer every input line")
        require(responses[0]["error"]["code"] == "invalid_request", "malformed JSON accepted")
        require(responses[1]["error"]["code"] == "invalid_request", "int32 overflow accepted")
        require(responses[2]["status"] == "ok", "serve stopped after an error")
        require(responses[2]["values_le_hex"] == expected_values.hex(), "base64 frame decode mismatch")
        require(responses[3]["compression_ratio"] is None, "all-null ratio must be null")
        require(responses[3]["bit_exact"] is True, "all-null round trip failed")

    print("unified compression CLI integration checks passed")


if __name__ == "__main__":
    main()
