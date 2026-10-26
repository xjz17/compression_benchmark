#!/usr/bin/env python3
"""Verify frozen entropy, dictionary and snap111 wires; historical freeze is disabled."""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "analysis_outputs/python_to_cpp_action_migration_v1"
GOLDEN = DIR / "atomic_goldens.json"


def b64(data):
    return base64.b64encode(data).decode()


def freeze():
    raise RuntimeError(
        "Historical oracle is immutable: freeze ran before Python-kernel removal. "
        "Use verify; archived sources are in "
        "analysis_outputs/python_to_cpp_action_migration_v1/"
        "python_implementation_source_backup.tar.gz"
    )

def verify():
    records = json.loads(GOLDEN.read_text())
    for row in records:
        codec = row["codec"]
        raw, expected = base64.b64decode(row["raw"]), base64.b64decode(row["encoded"])
        if codec.startswith("dictionary"):
            cli, forward, reverse = "dictionary64_action_cli", "encode", "decode"
        elif codec == "snap111":
            cli, forward, reverse = "snappy_exact_stages_cli", "snap111-forward", "lz-inverse"
        else:
            cli, forward, reverse = "general_codec_reference_stages_cli", codec + "-forward", codec + "-inverse"
        path = ROOT / "cpp/build" / cli
        encoded = subprocess.run([str(path), forward], input=raw, capture_output=True, check=True).stdout
        assert encoded == expected, (row["case"], codec, len(encoded), len(expected))
        restored = subprocess.run([str(path), reverse], input=encoded, capture_output=True, check=True).stdout
        assert restored == raw, (row["case"], codec, "inverse")
    result = dict(records=len(records), identical_payloads=len(records), roundtrips=len(records),
                  golden_sha256=hashlib.sha256(GOLDEN.read_bytes()).hexdigest())
    (DIR / "atomic_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else verify()
