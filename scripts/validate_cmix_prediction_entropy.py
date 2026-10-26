#!/usr/bin/env python3
"""Run one bounded, low-priority full-cmix PEC1 encode and cold decode.

The script is intentionally Linux-only and fail-closed.  It writes a separate
resource receipt and promotes ``cmix`` in the ordinary PEC1 capability receipt
only after byte-exact decode with the current binary and source fingerprints.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

import prediction_entropy_codec as codec


RECEIPT = (
    codec.ROOT
    / "analysis_outputs/weight_conditioned_grpo_20260917_v1/preflight/"
      "cmix_prediction_entropy_validation.json"
)
MIN_AVAILABLE_BYTES = 64 * 1024**3
MAX_LOAD_PER_CPU = 0.75


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is unavailable")


def _time_command(arguments: list[str], payload: bytes, timeout_seconds: int) -> dict:
    with tempfile.NamedTemporaryFile(prefix="cmix-time-", delete=False) as handle:
        time_path = Path(handle.name)
    command = [
        "/usr/bin/time", "-v", "-o", str(time_path),
        "nice", "-n", "19", "ionice", "-c", "3",
        "timeout", "--signal=TERM", str(timeout_seconds),
        str(codec._binary()), *arguments,
    ]
    started = time.perf_counter_ns()
    result = subprocess.run(command, input=payload, capture_output=True, check=False)
    wall_ns = time.perf_counter_ns() - started
    try:
        report = time_path.read_text(errors="replace")
    finally:
        time_path.unlink(missing_ok=True)
    values: dict[str, str] = {}
    for line in report.splitlines():
        match = re.match(r"\s*([^:]+):\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2)
    max_rss = values.get("Maximum resident set size (kbytes)")
    return {
        "command": command,
        "returncode": result.returncode,
        "wall_time_ns": wall_ns,
        "max_rss_kib": int(max_rss) if max_rss and max_rss.isdigit() else None,
        "time_v": values,
        "stderr": result.stderr.decode("utf-8", errors="replace"),
        "stdout": result.stdout,
    }


def _json_record(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "stdout"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-large-memory", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not args.allow_large_memory:
        raise SystemExit("refusing full cmix without --allow-large-memory")
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise SystemExit("full cmix receipt is restricted to Linux x86-64")
    if not 60 <= args.timeout_seconds <= 3600:
        raise SystemExit("timeout must be in [60,3600] seconds")

    available_before = _mem_available()
    load_before = os.getloadavg()
    cpus = os.cpu_count() or 1
    safety = {
        "mem_available_bytes_before": available_before,
        "load_average_before": list(load_before),
        "logical_cpus": cpus,
        "minimum_available_bytes": MIN_AVAILABLE_BYTES,
        "maximum_load_per_cpu": MAX_LOAD_PER_CPU,
    }
    if available_before < MIN_AVAILABLE_BYTES:
        raise SystemExit("refusing full cmix: less than 64 GiB MemAvailable")
    if load_before[0] / cpus > MAX_LOAD_PER_CPU:
        raise SystemExit("refusing full cmix: one-minute load exceeds safety threshold")
    duplicate = subprocess.run(
        ["pgrep", "-af", "prediction_entropy_codec_cli.*(encode|decode).*cmix"],
        capture_output=True, text=True, check=False,
    )
    if duplicate.returncode == 0 and duplicate.stdout.strip():
        raise SystemExit("refusing full cmix: another full-cmix codec process is running")

    raw = b"A"
    encoded = _time_command(
        ["encode", "cmix", "--allow-large-memory"], raw, args.timeout_seconds
    )
    decoded = None
    passed = encoded["returncode"] == 0
    if passed:
        try:
            passed = codec.frame_provider(encoded["stdout"]) == "cmix"
        except ValueError:
            passed = False
    if passed:
        # This is a distinct native process with no encoder state.
        decoded = _time_command(
            ["decode", "--allow-large-memory"], encoded["stdout"],
            args.timeout_seconds,
        )
        passed = decoded["returncode"] == 0 and decoded["stdout"] == raw

    receipt = {
        "schema": codec.CMIX_RECEIPT_SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": "passed" if passed else "failed",
        "host_platform": platform.platform(),
        "hostname": platform.node(),
        "provider": "cmix",
        "action": codec.PROVIDER_TO_ACTION["cmix"],
        "frame": "PEC1",
        "complete_reversible_path": True,
        "cold_decode_independent_process": True,
        "explicit_large_memory_opt_in": True,
        "single_process_sequential": True,
        "input_size": len(raw),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "frame_size": len(encoded["stdout"]) if encoded["returncode"] == 0 else None,
        "frame_sha256": (
            hashlib.sha256(encoded["stdout"]).hexdigest()
            if encoded["returncode"] == 0 else None
        ),
        "binary": str(codec._binary()),
        "binary_sha256": codec.sha256(codec._binary()),
        "source_fingerprints": codec.source_fingerprints(),
        "safety": safety,
        "encode": _json_record(encoded),
        "cold_decode": _json_record(decoded) if decoded is not None else None,
        "mem_available_bytes_after": _mem_available(),
        "load_average_after": list(os.getloadavg()),
        "claim_boundary": (
            "one-byte correctness/resource probe only; no dataset compression or timing claim"
        ),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    if passed:
        base = json.loads(codec.VALIDATION.read_text())
        current = (
            base.get("result") == "passed"
            and base.get("source_fingerprints") == codec.source_fingerprints()
            and base.get("binary_sha256") == codec.sha256(codec._binary())
        )
        if not current:
            receipt["result"] = "failed"
            receipt["promotion_error"] = "base PEC1 receipt is not current"
            RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            passed = False
        else:
            providers = set(base.get("validated_providers", ()))
            providers.add("cmix")
            base["validated_providers"] = sorted(providers)
            base["unvalidated_providers"] = {
                key: value for key, value in base.get("unvalidated_providers", {}).items()
                if key != "cmix"
            }
            base["cmix_probe_receipt"] = str(RECEIPT.relative_to(codec.ROOT))
            base["cmix_probe_sha256"] = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
            temporary = codec.VALIDATION.with_suffix(".tmp")
            temporary.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, codec.VALIDATION)

    print(RECEIPT)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
