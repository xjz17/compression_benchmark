#!/usr/bin/env python3
"""Freeze exact active85 carrier routes from the accepted campaign routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old84", type=Path, required=True)
    parser.add_argument("--bos", type=Path, required=True)
    parser.add_argument("--corad-binary", required=True)
    parser.add_argument("--corad-binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old84, bos = read(args.old84.resolve()), read(args.bos.resolve())
    if old84.get("schema") != "native85-nonnull-method-routes-v1":
        raise ValueError("unexpected old84 route schema")
    if bos.get("schema") != "new15-algorithm85-method-routes-v1":
        raise ValueError("unexpected BOS route schema")
    routes = old84.get("routes")
    if not isinstance(routes, list) or len(routes) != 7:
        raise ValueError("old84 must contain seven reviewed route groups")
    bos_routes = bos.get("routes")
    if not isinstance(bos_routes, list) or len(bos_routes) != 1:
        raise ValueError("BOS overlay must contain one route")

    assignments: dict[str, dict] = {}
    for route in routes:
        for method in route["methods"]:
            if method in assignments:
                raise ValueError(f"duplicate method route: {method}")
            effective = dict(route)
            # The carrier contract fixes one dtype per algorithm.  The reviewed
            # typed route uses its float64 override for the exact finite carrier.
            assignments[method] = effective
    for route in bos_routes:
        for method in route["methods"]:
            if method in assignments:
                raise ValueError(f"BOS unexpectedly exists in old84: {method}")
            assignments[method] = dict(route)

    root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(root / "source"))
    from carrier import ACTIVE85, carrier_kind_for_algorithm  # type: ignore

    if set(assignments) != set(ACTIVE85):
        missing = sorted(set(ACTIVE85) - set(assignments))
        extra = sorted(set(assignments) - set(ACTIVE85))
        raise ValueError(f"active85 route mismatch missing={missing} extra={extra}")

    records: list[dict] = []
    for method in sorted(ACTIVE85):
        source = assignments[method]
        carrier = carrier_kind_for_algorithm(method)
        dtype = {"direct_u8": "uint8", "int64_u56": "int64",
                 "float64_u48": "float64", "float64_u8": "float64"}[carrier]
        effective = dict(source)
        override = source.get("dtype_overrides", {}).get(dtype)
        if isinstance(override, dict):
            effective.update(override)
        if method == "CORAD":
            effective.update({
                "binary": args.corad_binary,
                "binary_sha256": args.corad_binary_sha256,
                "ld_library_path": None,
                "source_route_id": source["route_id"],
            })
        records.append({
            "algorithm": method,
            "carrier_kind": carrier,
            "dtype": dtype,
            "binary": effective["binary"],
            "binary_sha256": effective["binary_sha256"],
            "ld_library_path": effective.get("ld_library_path"),
            "cli_profile": "standard_measure_decode",
            "profile": effective.get("configuration", effective.get(
                "adapter", f"reviewed-route:{source['route_id']}")),
            "route_id": source["route_id"],
            "source_command_profile": source["command_profile"],
            "algorithm_profile": effective.get("configuration", effective.get("adapter")),
            "warmups": 1,
            "repeats": 3,
            "threads": 1,
            "cold_decode_required": True,
            "bit_exact_required": True,
        })
    result = {
        "schema": "operator-algorithm-active85-carrier-routes-v1",
        "status": "frozen",
        "algorithms": 85,
        "carrier_counts": {
            kind: sum(row["carrier_kind"] == kind for row in records)
            for kind in ("direct_u8", "int64_u56", "float64_u48", "float64_u8")
        },
        "records": records,
        "records_canonical_sha256": canonical_sha(records),
        "source_routes": {
            "old84_path": str(args.old84.resolve()),
            "old84_sha256": hashlib.sha256(args.old84.read_bytes()).hexdigest(),
            "bos_path": str(args.bos.resolve()),
            "bos_sha256": hashlib.sha256(args.bos.read_bytes()).hexdigest(),
        },
        "contract": {
            "complete_algorithm_frame_charged": True,
            "carrier_metadata_and_missing_bitmap_charged": True,
            "native_W1_R3_T1": True,
            "cold_decode": True,
            "independent_optima_combined": False,
            "no_substitute_codec": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "algorithms": 85,
                      "carrier_counts": result["carrier_counts"],
                      "records_canonical_sha256": result["records_canonical_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
