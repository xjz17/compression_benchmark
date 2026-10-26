#!/usr/bin/env python3
"""Local ordinary + all-ten-special timing wire smoke test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile


SPECIAL = {"A057", "A065", "A086", "A094", "A096",
           "A097", "A098", "A099", "A100", "A102"}
BOUNDARY_REASON = (
    "unsupported_input_domain: Camel int64-to-float64 exactness precondition is false"
)
BOUNDARY_REJECTION = {
    "class": "native_timing_error",
    "message": "Camel integer is not exactly representable as float64",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    args = ap.parse_args()
    catalog = json.loads(args.catalog.read_text())
    rows = catalog.get("resolved_implementations", catalog.get("implementations"))
    by_id = {row["implementation_id"]: row for row in rows}
    ordinary = next(key for key in by_id if key not in SPECIAL and
                    key not in {"A106", "A107", "A108", "A109", "A110"})
    ids = [ordinary] + sorted(SPECIAL)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        values = root / "values.bin"
        # 64 aligned uint8 values with repeated and changing regions exercise
        # both literal and structured paths without relying on a data fixture.
        values.write_bytes(bytes((index // 4) & 0xff for index in range(64)))
        manifest = root / "column_manifest.json"
        manifest.write_text(json.dumps({
            "schema": "operator-timing-smoke-column-v1",
            "dataset_id": "smoke",
            "column_id": "u8",
            "byte_order": "little",
            "values_are_valid_only": True,
            "blocks": [{
                "block_id": "b0", "dtype": "uint8", "width": 1,
                "valid_count": 64, "values_path": str(values),
                "values_sha256": digest(values), "missing_count": 0,
            }],
        }))
        request = {
            "schema": "operator102-timing-column-request-v1",
            "request_id": "smoke", "dataset_id": "smoke", "column_id": "u8",
            "manifest_path": str(manifest),
            "expected_manifest_sha256": digest(manifest),
            "implementations": [by_id[key] for key in ids],
        }

        # 2**53+1 is an exact int64 but cannot round-trip through float64.
        # The frozen A004 integer Camel path must preserve that input-domain
        # rejection as a narrow, non-rankable support boundary.
        boundary_values = root / "a004_boundary.bin"
        boundary_values.write_bytes(struct.pack("<q", (1 << 53) + 1))
        boundary_manifest = root / "a004_boundary_manifest.json"
        boundary_manifest.write_text(json.dumps({
            "schema": "operator-timing-smoke-column-v1",
            "dataset_id": "smoke", "column_id": "a004-boundary",
            "byte_order": "little", "values_are_valid_only": True,
            "blocks": [{
                "block_id": "b0", "dtype": "int64", "width": 8,
                "valid_count": 1, "values_path": str(boundary_values),
                "values_sha256": digest(boundary_values), "missing_count": 0,
            }],
        }))
        boundary_request = {
            "schema": "operator102-timing-column-request-v1",
            "request_id": "a004-boundary", "dataset_id": "smoke",
            "column_id": "a004-boundary",
            "manifest_path": str(boundary_manifest),
            "expected_manifest_sha256": digest(boundary_manifest),
            "implementations": [by_id["A004"]],
        }

        # Prove that implementation_id=A004 does not broadly relabel other
        # failures.  An invalid stage is a hard failure, not a support boundary.
        invalid_a004 = copy.deepcopy(by_id["A004"])
        invalid_a004["present_pipeline"][0]["action"] = (
            "op.invalid.a004_must_remain_failed"
        )
        invalid_request = copy.deepcopy(boundary_request)
        invalid_request["request_id"] = "a004-other-error"
        invalid_request["column_id"] = "a004-other-error"
        invalid_request["implementations"] = [invalid_a004]

        wire = "".join(json.dumps(item) + "\n" for item in (
            request, boundary_request, invalid_request
        ))
        completed = subprocess.run([str(args.driver), "--jsonl"],
                                   input=wire, text=True,
                                   capture_output=True, check=True)
        responses = [json.loads(line) for line in completed.stdout.splitlines()
                     if line.strip()]
        assert len(responses) == 3, completed.stdout
        response, boundary_response, invalid_response = responses
        assert response["schema"] == "operator102-timing-column-response-v1"
        assert [row["implementation_id"] for row in response["results"]] == ids
        for row in response["results"]:
            assert row["status"] == "ok", row
            assert row["bitexact"] is True
            timing = row["timing"]
            assert timing["warmups_per_block"] == 1
            assert timing["repeats"] == 3 and timing["threads"] == 1
            assert timing["ranking_view"] == "full_o2r1"
            for key in ("kernel_encode_ns", "kernel_decode_ns",
                        "full_encode_ns", "full_decode_ns"):
                assert len(timing[key]) == 3
            assert all(full >= kernel for full, kernel in zip(
                timing["full_encode_ns"], timing["kernel_encode_ns"]))
            assert all(full >= kernel for full, kernel in zip(
                timing["full_decode_ns"], timing["kernel_decode_ns"]))
            assert timing["bitexact_each_warmup_and_repeat"] is True
        assert {row["implementation_id"] for row in response["results"]
                if row["special_bridge"]} == SPECIAL

        boundary_row = boundary_response["results"][0]
        assert boundary_row == {
            "implementation_id": "A004",
            "map_key": by_id["A004"].get("map_key", ""),
            "status": "not_applicable",
            "bitexact": False,
            "rankable": False,
            "reason": BOUNDARY_REASON,
            "native_domain_rejection": BOUNDARY_REJECTION,
        }, boundary_row

        invalid_row = invalid_response["results"][0]
        assert invalid_row["implementation_id"] == "A004"
        assert invalid_row["status"] == "failed", invalid_row
        assert invalid_row["bitexact"] is False
        assert invalid_row["error"]["class"] == "native_timing_error"
        assert invalid_row["error"]["message"] != BOUNDARY_REJECTION["message"]
        assert "native_domain_rejection" not in invalid_row

        # Both the append-only runner and the finalizer independently reject
        # any broadened or mutated N/A result.
        for filename in ("run_timing_shard.py", "finalize_timing.py"):
            module = load_source_module(Path(__file__).with_name(filename),
                                        f"smoke_{filename.replace('.', '_')}")
            module.validate_not_applicable_result(boundary_row)
            for key, value in (
                ("implementation_id", "A005"),
                ("reason", "generic unsupported"),
                ("rankable", True),
                ("native_domain_rejection", {
                    "class": "native_timing_error", "message": "other"}),
            ):
                mutated = copy.deepcopy(boundary_row)
                mutated[key] = value
                try:
                    module.validate_not_applicable_result(mutated)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(
                        f"{filename} accepted broadened support boundary: {key}"
                    )
    print("operator102_timing_smoke PASS ordinary=1 special=10 "
          "a004_boundary=1 a004_other_error_failed=1")


if __name__ == "__main__":
    main()
