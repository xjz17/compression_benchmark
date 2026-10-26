#!/usr/bin/env python3
"""Run bounded genuine-model tests and record reproducible A110 evidence."""
import datetime
import json
import os
import platform
import subprocess
import sys

from build_nncp_probability import DOCKER_IMAGE, DRIVER, ROOT, VALIDATION, source_fingerprints, sha256


def main():
    records = []
    commands = [[sys.executable, "scripts/test_nncp_probability.py", "-v"],
                [sys.executable, "scripts/test_prediction_probability.py", "-v"]]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        print(result.stdout + result.stderr, end="")
        records.append({"command": command, "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
    docker = os.environ.get("WEB_COMPRESSION_NNCP_USE_DOCKER") == "1"
    image_id = None
    if docker:
        image_id = subprocess.check_output(["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"], text=True).strip()
    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": "passed" if all(item["returncode"] == 0 for item in records) else "failed",
        "host_platform": platform.platform(), "host_architecture": platform.machine(),
        "execution": "explicit Linux amd64 Docker" if docker else "native Linux x86-64",
        "environment": {"WEB_COMPRESSION_NNCP_USE_DOCKER": os.environ.get("WEB_COMPRESSION_NNCP_USE_DOCKER")},
        "docker_image": DOCKER_IMAGE if docker else None, "docker_image_id": image_id,
        "source_fingerprints": source_fingerprints(), "driver_sha256": sha256(DRIVER),
        "upstream": "Fabrice Bellard NNCP 3.3, release 2024-06-05",
        "profile": "transformer-stream1-v1", "commands": records,
        "contract": {"input": "one byte symbol after prediction, 0..255", "output": "float32[256]",
                     "order": "predict -> encode/decode event -> observe -> gradient -> model_update",
                     "max_symbols": 4096, "seed": 123, "batch_size": 1, "segment_length": 1},
        "limitations": ["Local functional tests only; no 20-dataset or server performance claim",
                        "NNCP default file-profile schedule differs from the explicit stream1 profile",
                        "Original LibNC binary requires Linux x86-64", "Probability provider is not a graph entropy-coder integration claim"],
    }
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(json.dumps(report, indent=2) + "\n")
    print(VALIDATION)
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
