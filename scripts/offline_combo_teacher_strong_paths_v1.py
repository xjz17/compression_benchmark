#!/usr/bin/env python3
"""Add strong, already-registered LZ4 numeric paths to the six-weight teacher.

This module changes only the offline candidate inventory.  It reuses existing
6341-action IDs and the frozen 6373-output per-flow STOP adapter, so existing
models and action masks keep their identity.  The LZ4 stages are the registered
vendored exact stages; the preceding byte shuffle is the project's reversible
reference byte-plane transform.  No standard-Zstd or Brotli equivalence is
claimed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import offline_combo_teacher_sixweight_v2 as inherited


base = inherited.base
fit3 = inherited.fit3
ActionDescriptor = inherited.ActionDescriptor
ExactCatalog = inherited.ExactCatalog
Plan = inherited.Plan
_INHERITED_FROZEN_PLANS = inherited.frozen_plans

EXTENSION_SCHEMA = "native-online-offline-combo-strong-registered-paths-v1"
INHERITED_PLAN_COUNT = 92
STRONG_PLAN_COUNT = 2
TOTAL_PLAN_COUNT = INHERITED_PLAN_COUNT + STRONG_PLAN_COUNT


def descriptor(runtime_action: str, implementation_id: str) -> ActionDescriptor:
    return ActionDescriptor(runtime_action, implementation_id=implementation_id,
                            io_variant=runtime_action)


LZ4_EXACT_BACKEND = (
    descriptor("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
               "A083"),
    descriptor("op.match_token_format.lz4.cpp_v1", "A069"),
    descriptor("op.integer_code.lz4.cpp_v1", "A075"),
)


def frozen_plans() -> tuple[Plan, ...]:
    plans = list(_INHERITED_FROZEN_PLANS())
    base.require(len(plans) == INHERITED_PLAN_COUNT,
                 "six-weight inherited plan count drift")
    for width in (4, 8):
        shuffle = descriptor(
            f"op.block_transform.blosc_shuffle.typesize{width}.cpp_reference_v1",
            "A049")
        plans.append(Plan(
            f"strong_registered_byte_shuffle{width}+lz4_exact_stages",
            (shuffle,), LZ4_EXACT_BACKEND, True,
            coverage_origin="strong_registered_existing_actions"))
    result = tuple(plans)
    base.require(len(result) == TOTAL_PLAN_COUNT and
                 len({plan.name for plan in result}) == TOTAL_PLAN_COUNT,
                 "strong-path candidate inventory drift")
    return result


def _rewrite_jsonl(path: Path, extra: Mapping[str, Any]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    base.atomic_jsonl(path, ({**row, **extra} for row in rows))


def _augment(output: Path) -> None:
    shared = {
        "strong_path_schema": EXTENSION_SCHEMA,
        "strong_path_candidates": [
            plan.name for plan in frozen_plans()
            if plan.coverage_origin == "strong_registered_existing_actions"
        ],
        "strong_path_action_registry_changed": False,
        "standard_zstd_claimed": False,
        "standard_brotli_claimed": False,
    }
    search_path = output / "search_space.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    search.update(shared)
    search.setdefault("artifacts", {})[str(Path(__file__).resolve())] = base.sha256(
        Path(__file__))
    base.atomic_json(search_path, search)
    for filename in ("candidate_measurements.jsonl", "teacher.jsonl"):
        _rewrite_jsonl(output / filename,
                       {"strong_path_schema": EXTENSION_SCHEMA})
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(shared)
    summary.setdefault("limitations", []).append(
        "The added byte-shuffle plus LZ4 paths use existing registered stages; "
        "they do not fill the standard-Zstd/Brotli graph-action gaps.")
    base.atomic_json(summary_path, summary)
    receipt_path = output / "teacher.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(shared)
    receipt.update({"teacher_sha256": base.sha256(output / "teacher.jsonl"),
                    "search_space_sha256": base.sha256(search_path)})
    base.atomic_json(receipt_path, receipt)


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    candidate_values: list[int] = []
    for index, value in enumerate(forwarded):
        if value == "--candidate-limit":
            base.require(index + 1 < len(forwarded),
                         "--candidate-limit requires a value")
            candidate_values.append(int(forwarded[index + 1]))
        elif value.startswith("--candidate-limit="):
            candidate_values.append(int(value.split("=", 1)[1]))
    base.require(len(candidate_values) <= 1, "duplicate --candidate-limit")
    if not candidate_values:
        forwarded.extend(("--candidate-limit", str(TOTAL_PLAN_COUNT)))
    else:
        base.require(TOTAL_PLAN_COUNT <= candidate_values[0] <= 128,
                     "candidate limit must retain inherited and strong paths")

    inherited.frozen_plans = frozen_plans
    inherited.base.frozen_plans = frozen_plans
    result = inherited.main(forwarded)
    output_value = next((forwarded[index + 1]
                         for index, value in enumerate(forwarded[:-1])
                         if value == "--output"), None)
    if output_value is None:
        output_value = next((value.split("=", 1)[1] for value in forwarded
                             if value.startswith("--output=")), None)
    base.require(output_value is not None, "--output is required")
    _augment(Path(output_value).resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
