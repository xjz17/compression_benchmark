#!/usr/bin/env python3
"""Validate A110 as a complete NNP1 typed-graph action and write evidence."""
from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys

from build_nncp_probability import DRIVER, ROOT, sha256
from nncp_entropy_codec import (
    ACTION,
    PROFILE,
    VALIDATION,
    source_fingerprints,
)


def main() -> int:
    command = [sys.executable, "scripts/test_nncp_entropy_graph.py", "-v"]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=900
    )
    print(result.stdout + result.stderr, end="")
    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": "passed" if result.returncode == 0 else "failed",
        "host_platform": platform.platform(),
        "host_architecture": platform.machine(),
        "action": ACTION,
        "provider": "nncp",
        "frame": "NNP1",
        "profile": PROFILE,
        "driver_sha256": sha256(DRIVER),
        "source_fingerprints": source_fingerprints(),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "contract": {
            "role": "complete terminal entropy path, not an atomic probability edge",
            "cold_decode": "fresh Python interpreter and fresh NNCP native process",
            "decoder_inputs": "TGF1 frame, pinned action identity, and NNP1 leaf only",
        },
        "limitations": [
            "NNP1 uses the explicit transformer-stream1-v1 profile",
            "raw payload is bounded by the existing 4096-byte NNCP adapter limit",
            "this receipt is correctness evidence, not a performance claim",
        ],
    }
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(json.dumps(report, indent=2) + "\n")
    print(VALIDATION)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
