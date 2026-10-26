#!/usr/bin/env python3
"""Validate current PEC1 binary and write an auditable capability receipt."""
from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys

import prediction_entropy_codec as codec


def main() -> int:
    command = [sys.executable, "scripts/test_prediction_entropy_codec.py", "-v"]
    result = subprocess.run(command, cwd=codec.ROOT, capture_output=True, text=True,
                            timeout=600)
    print(result.stdout + result.stderr, end="")
    passed = result.returncode == 0
    receipt = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": "passed" if passed else "failed",
        "host_platform": platform.platform(),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "binary": str(codec._binary()),
        "binary_sha256": codec.sha256(codec._binary()),
        "source_fingerprints": codec.source_fingerprints(),
        "validated_providers": ["ppmd", "dmc", "paq"] if passed else [],
        "unvalidated_providers": {
            "cmix": "complete path is implemented but no large-memory cold-decode probe was run"
        },
        "contract": {
            "frame": "PEC1",
            "event_order": "predict -> arithmetic encode/decode current bit -> observe",
            "bit_order": "MSB-first",
            "probability_quantizer": "one=1+floor(P(1)*65534), total=65536",
            "cold_decode": "fresh native process regenerates provider state",
        },
    }
    codec.VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    codec.VALIDATION.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(codec.VALIDATION)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
