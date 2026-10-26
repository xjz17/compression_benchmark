#!/usr/bin/env python3
"""Execute one carrier through its exact active85 route and cold-decode it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from carrier import pack_bytes, unpack_bytes  # type: ignore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_routes(path: Path) -> dict[str, dict]:
    value = json.loads(path.read_text())
    if value.get("schema") != "operator-algorithm-active85-carrier-routes-v1":
        raise ValueError("unexpected carrier route schema")
    rows = value.get("records")
    if not isinstance(rows, list) or len(rows) != 85:
        raise ValueError("carrier route manifest is not exact active85")
    result = {row["algorithm"]: row for row in rows}
    if len(result) != 85:
        raise ValueError("duplicate algorithm route")
    return result


def one_json(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("algorithm CLI did not emit one JSON object")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise RuntimeError("algorithm CLI response is not an object")
    return value


def run(argv: list[str], environment: dict[str, str]) -> dict:
    completed = subprocess.run(argv, text=True, capture_output=True,
                               env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"algorithm command exited {completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    return one_json(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--o2r1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    routes = load_routes(args.routes.resolve())
    route = routes.get(args.algorithm)
    if route is None:
        raise ValueError("algorithm is outside frozen active85")
    binary = Path(route["binary"])
    if not binary.is_file() or sha256(binary) != route["binary_sha256"]:
        raise ValueError("algorithm binary missing or SHA-256 mismatch")
    payload = args.o2r1.read_bytes()
    carrier = pack_bytes(payload, route["carrier_kind"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    values, missing = output / "values.bin", output / "missing.bin"
    frame = output / "algorithm.frame"
    restored_values, restored_missing = output / "decoded.values", output / "decoded.missing"
    values.write_bytes(carrier.values)
    missing.write_bytes(carrier.missing)
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
    })
    if route.get("ld_library_path"):
        environment["LD_LIBRARY_PATH"] = route["ld_library_path"]
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    measured = run([
        str(binary), "measure", args.algorithm, carrier.dtype,
        str(carrier.row_count), str(values), str(missing), "1", "3", str(frame),
    ], environment)
    if measured.get("status") not in {"ok", "N/A"}:
        raise RuntimeError("algorithm measure did not return ok")
    encoded_samples = measured.get("compression_ns_samples")
    decoded_samples = measured.get("decompression_ns_samples")
    if not (isinstance(encoded_samples, list) and len(encoded_samples) == 3 and
            isinstance(decoded_samples, list) and len(decoded_samples) == 3):
        raise RuntimeError("algorithm result is not W1/R3")
    cold = run([
        str(binary), "decode", str(frame), str(restored_values), str(restored_missing),
    ], environment)
    restored = unpack_bytes(restored_values.read_bytes(), restored_missing.read_bytes(),
                            carrier.metadata)
    if restored != payload:
        raise RuntimeError("cold algorithm decode + carrier inverse changed O2R1")
    receipt = {
        "schema": "operator-algorithm-route-execution-v1",
        "status": "ok",
        "algorithm": args.algorithm,
        "route_id": route["route_id"],
        "binary": str(binary),
        "binary_sha256": route["binary_sha256"],
        "carrier_kind": carrier.kind,
        "dtype": carrier.dtype,
        "carrier_numeric_bytes": len(carrier.values),
        "carrier_missing_bitmap_bytes": len(carrier.missing),
        "carrier_metadata_bytes": len(carrier.metadata),
        "algorithm_frame_bytes": frame.stat().st_size,
        "algorithm_frame_sha256": sha256(frame),
        "source_o2r1_bytes": len(payload),
        "source_o2r1_sha256": hashlib.sha256(payload).hexdigest(),
        "measurement": measured,
        "cold_decode": cold,
        "bit_exact": True,
        "warmups": 1,
        "repeats": 3,
        "threads": 1,
    }
    (output / "RESULT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
