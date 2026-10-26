#!/usr/bin/env python3
"""Targeted C++ regression for retiring op.scaling.reference_v1.

The fixtures were frozen with the pre-retirement C++ CLI.  This test imports no
Python operator runtime: every check starts the rebuilt native CLI and compares
bytes against those frozen artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "analysis_outputs/identity_scaling_retirement_20260915"
DEFAULT_CLI = ROOT / "cpp/build/reference_operator_actions_cli"
RETIRED = "op.scaling.reference_v1"
RETIRED_RECIPE = "op.integer_code.reference_v1"


def invoke(cli: Path, command: str, payload: bytes, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(cli), command, *arguments], input=payload, capture_output=True, check=False
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    args = parser.parse_args()
    cli = args.cli.resolve()
    fixture_root = args.fixtures.resolve()
    fixture_dir = fixture_root / "fixtures"
    manifest = json.loads((fixture_root / "fixtures_manifest.json").read_text())
    raw = (fixture_dir / "input_i64.bin").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest["input"]["sha256"],
            "frozen input hash mismatch")

    listed = invoke(cli, "all-actions", b"")
    require(listed.returncode == 0, listed.stderr.decode(errors="replace"))
    current_actions = listed.stdout.decode().splitlines()
    require(len(current_actions) == len(set(current_actions)) == 27,
            f"expected 27 unique current reference actions, got {len(current_actions)}")
    require(RETIRED not in current_actions, "retired scaling remains forward-selectable")
    require(RETIRED_RECIPE not in current_actions,
            "retired ZigZag plus ULEB128 recipe remains forward-selectable")

    rejected_commands = (
        "encode", "raw-encode", "state", "values", "inspect", "legacy-encode"
    )
    rejected = 0
    for command in rejected_commands:
        result = invoke(cli, command, raw, RETIRED)
        require(result.returncode != 0, f"{command} accepted retired scaling")
        require(b"retired from forward execution" in result.stderr,
                f"{command} returned the wrong retirement error")
        rejected += 1

    identity = manifest["records"]["identity"]
    identity_wire = (fixture_dir / "identity_wire.bin").read_bytes()
    identity_raw_opa1 = (fixture_dir / "identity_raw_opa1.bin").read_bytes()
    identity_state = (fixture_dir / "identity_state.bin").read_bytes()
    require(identity_raw_opa1[:7] == b"OPA1\x01\x01\x01",
            "frozen identity OPA1 does not contain immutable wire ID 1")
    require(identity_state[:12].rstrip(b"\0") == b"scaling",
            "frozen identity state lost its scaling metadata tag")

    decoded = invoke(cli, "decode", identity_wire)
    require(decoded.returncode == 0 and decoded.stdout == raw,
            "cold decode of frozen identity OPA1 failed")
    names = invoke(cli, "actions", identity_wire)
    require(names.returncode == 0 and names.stdout.decode().splitlines() == [RETIRED],
            "frozen identity OPA1 no longer reports its historical action name")
    inverted = invoke(cli, "invert-state", identity_state, RETIRED)
    require(inverted.returncode == 0 and inverted.stdout == raw,
            "read-only inverse of frozen identity state failed")
    inverted_tag = invoke(cli, "inverse-step", identity_state, "scaling")
    require(inverted_tag.returncode == 0 and inverted_tag.stdout == raw,
            "metadata-tag inverse of frozen identity state failed")
    legacy_wire = (fixture_dir / "identity_legacy_wire.bin").read_bytes()
    legacy_decoded = invoke(cli, "legacy-decode", legacy_wire)
    require(legacy_decoded.returncode == 0 and legacy_decoded.stdout == raw,
            "cold decode of frozen identity OPC1 failed")

    byte_checks = 0
    roundtrips = 0
    for label in ("nonidentity_delta", "nonidentity_zigzag"):
        record = manifest["records"][label]
        action = record["actions"][0]
        for suffix, command in (
            ("wire", "encode"),
            ("raw_opa1", "raw-encode"),
            ("state", "state"),
            ("values", "values"),
            ("inspect", "inspect"),
        ):
            expected = (fixture_dir / f"{label}_{suffix}.bin").read_bytes()
            actual = invoke(cli, command, raw, action)
            require(actual.returncode == 0, actual.stderr.decode(errors="replace"))
            require(actual.stdout == expected,
                    f"{label} {command} bytes changed after retirement")
            byte_checks += 1
        wire = (fixture_dir / f"{label}_wire.bin").read_bytes()
        restored = invoke(cli, "decode", wire)
        require(restored.returncode == 0 and restored.stdout == raw,
                f"{label} cold round trip failed")
        decoded_names = invoke(cli, "actions", wire)
        require(decoded_names.returncode == 0 and
                decoded_names.stdout.decode().splitlines() == [action],
                f"{label} wire ID/name mapping changed")
        roundtrips += 1

    delta_legacy = (fixture_dir / "nonidentity_delta_legacy_wire.bin").read_bytes()
    delta_legacy_now = invoke(cli, "legacy-encode", raw, "op.delta.reference_v1")
    require(delta_legacy_now.returncode == 0 and delta_legacy_now.stdout == delta_legacy,
            "non-identity legacy wire bytes changed")
    byte_checks += 1
    delta_legacy_decoded = invoke(cli, "legacy-decode", delta_legacy)
    require(delta_legacy_decoded.returncode == 0 and delta_legacy_decoded.stdout == raw,
            "non-identity legacy wire no longer decodes")
    roundtrips += 1

    print(json.dumps({
        "status": "PASS",
        "current_reference_actions": len(current_actions),
        "retired_action": RETIRED,
        "retired_forward_rejections": rejected,
        "frozen_identity_opa1_decode": True,
        "frozen_identity_opc1_decode": True,
        "frozen_identity_actions_name": identity["decoded_actions"],
        "frozen_identity_state_inverse": True,
        "nonidentity_byte_identity_checks": byte_checks,
        "nonidentity_roundtrips": roundtrips,
        "fixture_manifest": str(fixture_root / "fixtures_manifest.json"),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
