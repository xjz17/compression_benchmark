#!/usr/bin/env python3
"""Bounded train-only pilot for strong paths already in the frozen registry.

The output uses the established offline-teacher measurement and baseline
schemas.  It is an audit input, not a replacement teacher: candidates are
restricted to seven explicit recipes, all labels come from admitted train
blocks, and direct Zstd/LZ4 are measured in the same invocation.
"""
from __future__ import annotations

import sys
from typing import Sequence

import offline_combo_teacher_v6 as base


LZ4 = (
    "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
    "op.match_token_format.lz4.cpp_v1",
    "op.integer_code.lz4.cpp_v1",
)
SNAPPY = (
    "op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",
    "op.match_token_format.snappy.cpp_v1",
)


def pilot_plans() -> tuple[base.Plan, ...]:
    plans = (
        base.Plan("raw"),
        base.Plan("registered_lz4_exact_stages", LZ4),
        base.Plan("registered_snappy_exact_stages", SNAPPY),
        base.Plan(
            "registered_shuffle4+lz4_exact_stages",
            ("op.block_transform.blosc_shuffle.typesize4.cpp_reference_v1",),
            LZ4, True),
        base.Plan(
            "registered_shuffle8+lz4_exact_stages",
            ("op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1",),
            LZ4, True),
        base.Plan(
            "registered_xor_gorilla+lz4_exact_stages",
            ("op.xor.gorilla.cpp_v1",), LZ4, True),
        base.Plan(
            "registered_delta2+lz4_exact_stages",
            ("op.delta.ts_2diff.cpp_v1",), LZ4, True),
    )
    base.require(len({plan.name for plan in plans}) == len(plans),
                 "strong pilot candidate names are not unique")
    return plans


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    plans = pilot_plans()
    base.frozen_plans = pilot_plans
    if not any(value == "--candidate-limit" or
               value.startswith("--candidate-limit=") for value in forwarded):
        forwarded.extend(("--candidate-limit", str(len(plans))))
    return base.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
