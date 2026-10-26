#!/usr/bin/env python3
"""Isolated 102-template strong-path teacher using only registered actions.

The 94 inherited plans stay byte-for-byte available and eight paths are added
without changing the 6,341-action registry or 6,373-output online adapter.
Reference-chain names are intentional: no upstream Bzip2, Brotli, Zstd, Huff0,
or FSE wire equivalence is claimed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import offline_combo_teacher_strong_paths_v1 as inherited


base = inherited.base
Plan = inherited.Plan
ActionDescriptor = inherited.ActionDescriptor
_INHERITED_FROZEN_PLANS = inherited.frozen_plans

EXTENSION_SCHEMA = "native-online-offline-combo-strong-registered-paths-v2"
INHERITED_PLAN_COUNT = 94
ADDED_PLAN_COUNT = 8
TOTAL_PLAN_COUNT = INHERITED_PLAN_COUNT + ADDED_PLAN_COUNT


def descriptor(runtime_action: str, implementation_id: str) -> ActionDescriptor:
    # Runtime action + concrete implementation ID is the stable identity here.
    # Some frozen rows intentionally have no io_variant; current logical view
    # resolves their typed alternative fail-closed in ExactCatalog.exact().
    return ActionDescriptor(runtime_action, implementation_id=implementation_id)


LZ4_EXACT_BACKEND = (
    descriptor("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", "A083"),
    descriptor("op.match_token_format.lz4.cpp_v1", "A069"),
    descriptor("op.integer_code.lz4.cpp_v1", "A075"),
)
BZIP2_REFERENCE_PREFIX = (
    descriptor("op.block_transform.bzip2_bwt.cpp_reference_v1", "A050"),
    descriptor("op.symbol_recode.bzip2_mtf.cpp_reference_v1", "A054"),
    descriptor("op.run_length.byte_optimal.cpp_v1", "A055"),
)
def frozen_plans() -> tuple[Plan, ...]:
    plans = list(_INHERITED_FROZEN_PLANS())
    base.require(len(plans) == INHERITED_PLAN_COUNT,
                 "strong v1 inherited plan count drift")
    origin = "strong_registered_reference_chains_v2"
    for name, action, implementation_id in (
        ("delta2", "op.delta.ts_2diff.cpp_v1", "A028"),
        ("delta_streamvbyte", "op.delta.stream_vbyte.cpp_v1", "A028"),
        ("xor_previous", "op.xor.previous_word.cpp_v2", "A031"),
        ("xor_gorilla", "op.xor.gorilla.cpp_v1", "A031"),
    ):
        plans.append(Plan(
            f"strong_registered_{name}+lz4_exact_stages",
            (descriptor(action, implementation_id),), LZ4_EXACT_BACKEND, True,
            coverage_origin=origin))
    plans.append(Plan(
        "strong_registered_dictionary_words64+each_lz4_exact_stages",
        (descriptor("op.dictionary.project_words64.cpp_v1", "A042"),),
        LZ4_EXACT_BACKEND, True, coverage_origin=origin))
    plans.append(Plan(
        "strong_registered_for_minimum+each_lz4_exact_stages",
        (descriptor("op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
                    "A013"),),
        LZ4_EXACT_BACKEND, True, coverage_origin=origin))
    plans.append(Plan(
        "strong_registered_xor_semantic_ports+each_lz4_exact_stages",
        (descriptor("op.xor.previous_word.semantic_ports.cpp_v1", "A031"),),
        LZ4_EXACT_BACKEND, True, coverage_origin=origin))
    plans.extend((
        Plan("strong_registered_bzip2_reference_chain",
             BZIP2_REFERENCE_PREFIX,
             (descriptor("op.entropy_huffman.project_bytes.cpp_v1", "A077"),),
             True, coverage_origin=origin),
    ))
    result = tuple(plans)
    base.require(len(result) == TOTAL_PLAN_COUNT and
                 len({plan.name for plan in result}) == TOTAL_PLAN_COUNT,
                 "strong v2 candidate inventory drift")
    return result


def _rewrite_jsonl(path: Path, extra: Mapping[str, Any]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    base.atomic_jsonl(path, ({**row, **extra} for row in rows))


def _augment(output: Path) -> None:
    new_candidates = [plan.name for plan in frozen_plans()
                      if plan.coverage_origin ==
                      "strong_registered_reference_chains_v2"]
    shared = {
        "strong_path_schema": EXTENSION_SCHEMA,
        "strong_path_candidate_count": TOTAL_PLAN_COUNT - 92,
        "new_reference_chain_candidate_count": ADDED_PLAN_COUNT,
        "strong_path_total_template_count": TOTAL_PLAN_COUNT,
        "strong_reference_chain_candidates": new_candidates,
        "strong_path_action_registry_changed": False,
        "standard_bzip2_claimed": False,
        "standard_brotli_claimed": False,
        "standard_zstd_claimed": False,
        "upstream_wire_equivalence_claimed": False,
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
    summary.setdefault("limitations", []).extend((
        "The BWT/MTF/RLE/Huffman path is a registered reference composition, "
        "not an upstream Bzip2 wire claim.",
        "The LZ4 paths use registered exact LZ4 stages after independent "
        "project transforms; they do not claim a new upstream composite wire.",
    ))
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
                     "candidate limit must retain all 102 strong-v2 templates")

    inherited.frozen_plans = frozen_plans
    inherited.inherited.frozen_plans = frozen_plans
    inherited.inherited.base.frozen_plans = frozen_plans
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
