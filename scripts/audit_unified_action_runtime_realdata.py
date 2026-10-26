#!/usr/bin/env python3
"""Real-column round-trip audit for the unified 247-record action interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import audit_86_composability_realdata as external_data
import primitive_operator_action_runtime as primitive
from unified_action_runtime import (
    ACTIONS,
    ActionData,
    ActionNotExecutableError,
    _unpack_result,
    describe_action,
    run_action,
    run_actions,
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def audit(rows: int) -> dict[str, Any]:
    columns, manifest = external_data.load_external_columns(rows)
    standalone_checks = 0
    native_method_checks = 0
    native_stage_checks = 0
    native_wire_checks = 0
    covered_native_actions: set[str] = set()
    float_rle_checks = 0

    for column in columns:
        typed = (
            ActionData.ints(column.values)
            if column.logical_type == "integer"
            else ActionData.floats(column.values)
        )
        raw = ActionData.bytes(column.raw)
        for action in ACTIONS:
            if action.execution_mode != "standalone":
                continue
            source = raw if "bytes" in action.accepted_input_kinds else typed
            if source.kind not in action.accepted_input_kinds:
                continue
            encoded = run_action(action.name, source)
            if run_action(action.name, encoded, reverse=True) != source:
                raise AssertionError(
                    f"standalone real-data round trip failed: {action.name} "
                    f"on {column.dataset}:{column.name}"
                )
            standalone_checks += 1

        for method, algorithm_key in primitive.EXACT_NATIVE_METHOD_KEYS.items():
            names = tuple(
                action.name
                for action in primitive.exact_native_lz_action_pipeline(algorithm_key)
            )
            encoded = raw
            for name in names:
                stage_input = encoded
                encoded = run_action(name, stage_input)
                if run_action(name, encoded, reverse=True) != stage_input:
                    raise AssertionError(
                        f"native-LZ stage real-data round trip failed: {name} "
                        f"on {column.dataset}:{column.name}"
                    )
                covered_native_actions.add(name)
                native_stage_checks += 1
            _header, wire = _unpack_result(encoded)
            expected_wire = primitive.encode_exact_native_lz_pipeline(
                raw.payload, algorithm_key
            )
            if wire != expected_wire:
                raise AssertionError(
                    f"native-LZ final wire mismatch: {method} "
                    f"on {column.dataset}:{column.name}"
                )
            native_wire_checks += 1
            if run_actions(names, encoded, reverse=True) != raw:
                raise AssertionError(
                    f"native-LZ real-data round trip failed: {method} "
                    f"on {column.dataset}:{column.name}"
                )
            native_method_checks += 1

        if column.logical_type == "float" and column.max_point in {1, 2, 4, 14}:
            names = primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS
            encoded = run_actions(names, typed, max_point=column.max_point)
            if run_actions(names, encoded, reverse=True) != typed:
                raise AssertionError(
                    f"float scaling/RLE round trip failed on {column.dataset}:{column.name}"
                )
            float_rle_checks += 1

    if covered_native_actions != set(primitive.EXACT_NATIVE_ACTION_BY_NAME):
        raise AssertionError("real-data audit did not cover all 60 native-LZ actions")

    descriptor_rejections = 0
    probe = ActionData.bytes(b"descriptor-boundary")
    for action in ACTIONS:
        if action.execution_mode != "descriptor_only":
            continue
        try:
            run_action(action.name, probe)
        except ActionNotExecutableError:
            descriptor_rejections += 1
        else:
            raise AssertionError(f"descriptor-only action executed: {action.name}")

    result = {
        "schema": "unified-action-realdata-audit-v2",
        "status": "PASS",
        "claim_boundary": (
            "interface and lossless round-trip only; no new named-upstream "
            "stage-composition or compression-ratio claim"
        ),
        "action_count": len(ACTIONS),
        "native_lz_action_contracts": [
            {
                "name": name,
                "role": primitive.exact_native_lz_stage_role(name),
                "execution_mode": describe_action(name).execution_mode,
                "accepted_input_kinds": list(
                    describe_action(name).accepted_input_kinds
                ),
                "output_kinds": list(describe_action(name).output_kinds),
                "implementation_source": describe_action(name).implementation_source,
            }
            for name in sorted(primitive.EXACT_NATIVE_ACTION_BY_NAME)
        ],
        "columns": manifest,
        "counts": {
            "standalone_realdata_round_trips": standalone_checks,
            "native_lz_method_realdata_round_trips": native_method_checks,
            "native_lz_stage_realdata_round_trips": native_stage_checks,
            "native_lz_final_wire_exact_matches": native_wire_checks,
            "native_lz_unique_actions_covered": len(covered_native_actions),
            "float_scaling_then_int_rle_round_trips": float_rle_checks,
            "descriptor_only_rejections": descriptor_rejections,
        },
    }
    result["audit_sha256"] = _json_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.rows)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
