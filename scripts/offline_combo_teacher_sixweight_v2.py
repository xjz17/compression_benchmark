#!/usr/bin/env python3
"""Bounded typed-multiflow teacher extension for the six-weight pilot.

The frozen 83-plan union pilot is retained in full.  Nine current-runtime
connections are added for FOR, dictionary, and the typed two-port previous-word
XOR.  These are executable graph-stage seeds; they are neither upstream method
aliases nor a standard-Zstd construction.  Type-inapplicable seeds remain
explicit failed measurements in the v6 teacher output.

The v6 teacher/search/summary wire schemas are intentionally preserved for
``train_combo_policy_v6.py``.  ``schema_extension`` records the additional
candidate identity and the still-partial 86-method/PDF102 coverage boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import offline_combo_teacher_union_pilot_v1 as union


base = union.base
fit3 = union.fit3
ActionDescriptor = union.ActionDescriptor
CatalogAction = union.CatalogAction
ExactCatalog = union.ExactCatalog
Plan = union.Plan

EXTENSION_SCHEMA = "native-online-offline-combo-sixweight-typed-multiflow-v2"
# Keep the established union-binding value consumed by the campaign admission
# code.  The typed-multiflow scope is versioned by EXTENSION_SCHEMA and the
# explicit candidate counts below; changing this shared field would make the
# otherwise wire-compatible v6 teacher fail admission.
COVERAGE_STATUS = "partial_current_runtime_only"
CATALOG_SHA256 = "4ef157e9a1f24f9b851cfd81ed57def6be717d27a6ad9d478dfef3c3a569148c"
INHERITED_PLAN_COUNT = 83
HISTORICAL_PLAN_COUNT = 76
REGISTERED_SEED_COUNT = 7
TYPED_MULTIFLOW_PLAN_COUNT = 9
TOTAL_PLAN_COUNT = INHERITED_PLAN_COUNT + TYPED_MULTIFLOW_PLAN_COUNT


def descriptor(runtime_action: str, implementation_id: str, *,
               io_variant: str | None = None) -> ActionDescriptor:
    return ActionDescriptor(runtime_action, implementation_id=implementation_id,
                            io_variant=io_variant)


MAX_VALUE_BITPACK = (
    descriptor("op.bitpack_int.max_value_width.cpp_v1", "A020"),
)
SNAPPY_STAGE_PAIR = (
    descriptor("op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",
               "A082",
               io_variant="op.block_state_control.snappy_matchfinder_residual_core.cpp_v1"),
    descriptor("op.match_token_format.snappy.cpp_v1", "A068",
               io_variant="op.match_token_format.snappy.cpp_v1"),
)
FOR_MINIMUM = descriptor(
    "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1", "A013",
    io_variant="op.frame_of_reference.minimum.int64_uint64_le.cpp_v1")
DICTIONARY_WORDS64 = descriptor(
    "op.dictionary.project_words64.cpp_v1", "A042",
    io_variant="op.dictionary.project_words64.cpp_v1")
XOR_SEMANTIC_PORTS = descriptor(
    "op.xor.previous_word.semantic_ports.cpp_v1", "A031")

LEAF_BACKENDS: tuple[tuple[str, tuple[ActionDescriptor, ...]], ...] = (
    ("max_value_bitpack", MAX_VALUE_BITPACK),
    ("rle", (descriptor("op.run_length.byte_optimal.cpp_v1", "A055"),)),
    ("huffman", (descriptor("op.entropy_huffman.project_bytes.cpp_v1", "A077",
                            io_variant="op.entropy_huffman.project_bytes.cpp_v1"),)),
    ("rans_reference", (
        descriptor("op.entropy_fse.zstd_rans.cpp_reference_v1", "A080",
                   io_variant="op.entropy_fse.zstd_rans.cpp_reference_v1"),)),
    ("max_value_bitpack+snappy_stages", MAX_VALUE_BITPACK + SNAPPY_STAGE_PAIR),
)


def frozen_plans() -> tuple[Plan, ...]:
    inherited = union.frozen_plans()
    base.require(len(inherited) == INHERITED_PLAN_COUNT,
                 "frozen 83-plan union pilot count drift")
    plans = list(inherited)
    for name, prefix in (("for_minimum", FOR_MINIMUM),
                         ("dictionary_words64", DICTIONARY_WORDS64)):
        plans.append(Plan(
            f"typed_multiflow_{name}+each_max_value_bitpack",
            (prefix,), MAX_VALUE_BITPACK, True,
            "typed_multiflow_registered_connection"))
        plans.append(Plan(
            f"typed_multiflow_{name}+each_max_value_bitpack+snappy_stages",
            (prefix,), MAX_VALUE_BITPACK + SNAPPY_STAGE_PAIR, True,
            "typed_multiflow_registered_connection"))
    for backend_name, backend in LEAF_BACKENDS:
        plans.append(Plan(
            f"typed_multiflow_xor_semantic_ports+each_{backend_name}",
            (XOR_SEMANTIC_PORTS,), backend, True,
            "typed_multiflow_registered_connection"))
    result = tuple(plans)
    base.require(len(result) == TOTAL_PLAN_COUNT and
                 len({plan.name for plan in result}) == len(result),
                 "invalid six-weight typed-multiflow candidate inventory")
    base.require(sum(plan.coverage_origin == "all_76_v6_templates"
                     for plan in result) == HISTORICAL_PLAN_COUNT and
                 sum(plan.coverage_origin == "registered_specialist_seed"
                     for plan in result) == REGISTERED_SEED_COUNT and
                 sum(plan.coverage_origin == "typed_multiflow_registered_connection"
                     for plan in result) == TYPED_MULTIFLOW_PLAN_COUNT,
                 "candidate provenance accounting drift")
    return result


def output_views(descriptor_value: ActionDescriptor, produced: int) -> list[str]:
    """Mirror the frozen StageDispatch policy-view contract, not physical kind."""
    action = descriptor_value.runtime_action
    if action == "op.xor.previous_word.semantic_ports.cpp_v1":
        base.require(produced == 2, "semantic XOR must produce two typed ports")
        return ["uint64", "xor_residual64"]
    if action in {
            "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            "op.dictionary.project_words64.cpp_v1"}:
        base.require(produced == 2,
                     "FOR/dictionary must produce two physical Numeric8 ports")
        # The frozen C++ policy contract intentionally exposes these as bytes.
        return ["bytes", "bytes"]
    if action == "op.outlier_separation.high_bits.uint32.cpp_v1":
        return ["uint32"] * produced
    return ["bytes"] * produced


def _apply(backend: fit3.OnlineBackend, catalog: ExactCatalog,
           snapshot: fit3.Snapshot, descriptor_value: ActionDescriptor,
           views: list[str], slot: int) -> tuple[fit3.Snapshot, list[str]]:
    base.require(not snapshot.terminal and 0 <= slot < len(views),
                 "invalid stage application")
    before = len(views)
    action = catalog.exact(snapshot.legal_action_ids, descriptor_value,
                           target_slot=slot,
                           current_input_view=views[slot])
    next_snapshot = backend.step(action)
    after = (0 if next_snapshot.terminal else
             int(next_snapshot.response["open_flow_count"]))
    produced = after - (before - 1)
    base.require(produced >= 0, "ordinary stage open-flow count drift")
    next_views = (views[:slot] + views[slot + 1:] +
                  output_views(descriptor_value, produced))
    return next_snapshot, next_views


def estimated_two_root_steps(plan: Plan) -> int:
    """Upper bound for the frozen numeric-root plus missing-bitmap probe shape."""
    if plan.name == "raw":
        return 2
    # One auxiliary-root STOP.  Every new multi-flow prefix yields two leaves.
    if plan.coverage_origin == "typed_multiflow_registered_connection":
        return 1 + len(plan.prefix) + 2 * (len(plan.leaf_backend) + 1)
    # Existing templates are already admitted by the inherited pilot.
    return 1 + len(plan.prefix) + (len(plan.leaf_backend) + 1 if
                                  plan.backend_each_leaf else 1)


def materialize_plan(backend: fit3.OnlineBackend, block: fit3.TrainingBlock,
                     contract: fit3.OnlineContract, catalog: ExactCatalog,
                     plan: Plan) -> tuple[int, ...]:
    base.require(block.roots[0].get("original_numeric_payload") is True,
                 "typed-multiflow teacher requires numeric root zero")
    snapshot = backend.reset(block, fit3.FIXED_WEIGHTS[0], contract)
    views = [str(root["logical_view"]) for root in block.roots]
    initial = int(snapshot.response.get("open_flow_count", len(views)))
    base.require(initial == len(views), "C++ initial open-flow count drift")
    for slot in range(initial - 1, 0, -1):
        snapshot, views = union._stop(backend, snapshot, views, slot)
    if plan.name == "raw":
        snapshot, views = union._stop(backend, snapshot, views, 0)
        recipe = fit3._recipe(snapshot.response)
        base.require(len(recipe) <= contract.max_policy_steps,
                     "raw candidate exceeded policy budget")
        return recipe
    for descriptor_value in plan.prefix:
        snapshot, views = _apply(backend, catalog, snapshot, descriptor_value,
                                 views, 0)
        base.require(not snapshot.terminal,
                     f"candidate {plan.name} exhausted budget in prefix")
    if plan.backend_each_leaf:
        while not snapshot.terminal:
            slot = len(views) - 1
            for descriptor_value in plan.leaf_backend:
                snapshot, views = _apply(backend, catalog, snapshot,
                                         descriptor_value, views, slot)
                base.require(not snapshot.terminal,
                             f"candidate {plan.name} exceeded the 10-step budget "
                             "before leaf STOP")
                slot = len(views) - 1
            snapshot, views = union._stop(backend, snapshot, views, slot)
    else:
        while not snapshot.terminal:
            snapshot, views = union._stop(
                backend, snapshot, views, len(views) - 1)
    recipe = fit3._recipe(snapshot.response)
    base.require(len(recipe) <= contract.max_policy_steps,
                 f"candidate {plan.name} exceeded the 10-step budget")
    return recipe


def _rewrite_jsonl(path: Path, extra: Mapping[str, Any]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    base.atomic_jsonl(path, ({**row, **extra} for row in rows))


def _extension(union_path: Path, union_document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": EXTENSION_SCHEMA,
        "union_space_sha256": str(union_document["space_sha256"]),
        "union_catalog": str(union_path.resolve()),
        "union_catalog_sha256": base.sha256(union_path),
        "runtime_catalog_sha256": CATALOG_SHA256,
        "coverage_status": COVERAGE_STATUS,
        "full_union_training_ready": False,
        "full_union_coverage_claimed": False,
        "candidate_counts": {
            "all_76_v6_templates": HISTORICAL_PLAN_COUNT,
            "registered_specialist_seeds": REGISTERED_SEED_COUNT,
            "typed_multiflow_registered_connections": TYPED_MULTIFLOW_PLAN_COUNT,
            "total": TOTAL_PLAN_COUNT,
        },
        "typed_multiflow_connections": [
            plan.name for plan in frozen_plans()
            if plan.coverage_origin == "typed_multiflow_registered_connection"],
        "policy_horizon": 10,
        "maximum_estimated_two_root_steps": max(
            estimated_two_root_steps(plan) for plan in frozen_plans()),
        "method_equivalence_claimed": False,
        "standard_zstd_claimed": False,
    }


def _augment(output: Path, union_path: Path,
             union_document: Mapping[str, Any]) -> None:
    identity = str(union_document["space_sha256"])
    extension = _extension(union_path, union_document)
    shared = {"union_space_sha256": identity,
              "coverage_status": COVERAGE_STATUS,
              "full_union_coverage_claimed": False,
              "schema_extension": extension}
    search_path = output / "search_space.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    search.update(shared)
    search.setdefault("artifacts", {}).update({
        str(Path(__file__).resolve()): base.sha256(Path(__file__)),
        str(union_path.resolve()): base.sha256(union_path),
    })
    base.atomic_json(search_path, search)
    row_fields = {"union_space_sha256": identity,
                  "coverage_status": COVERAGE_STATUS,
                  "candidate_schema_extension": EXTENSION_SCHEMA}
    for filename in ("candidate_measurements.jsonl", "teacher.jsonl"):
        _rewrite_jsonl(output / filename, row_fields)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(shared)
    summary.setdefault("limitations", []).extend((
        "This 92-plan pilot remains a partial current-runtime subset of the "
        "86-method/PDF102 union.",
        "Typed seeds that are illegal for a source type remain explicit failures.",
        "rANS-reference and Snappy-named stages are registered stage chains, not "
        "a standard-Zstd or upstream-method equivalence claim."))
    base.atomic_json(summary_path, summary)
    receipt_path = output / "teacher.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(shared)
    receipt.update({"teacher_sha256": base.sha256(output / "teacher.jsonl"),
                    "search_space_sha256": base.sha256(search_path)})
    base.atomic_json(receipt_path, receipt)


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    union_path, forwarded = union._consume_option(forwarded, "--union-catalog")
    base.require(union_path.is_file(), f"union catalog absent: {union_path}")
    union_document = json.loads(union_path.read_text(encoding="utf-8"))
    base.require(union_document.get("schema") ==
                 "operator86-pdf102-candidate-union-v1",
                 "unexpected union catalog schema")
    base.require(fit3.digest(fit3.canonical_json(union_document["payload"])) ==
                 union_document.get("space_sha256"),
                 "union catalog payload hash drift")
    catalog_path, _ = union._consume_option(list(forwarded), "--catalog")
    base.require(base.sha256(catalog_path) == CATALOG_SHA256,
                 "six-weight v2 requires the frozen v7 runtime catalog")

    plans = frozen_plans()
    candidate_values = []
    for index, value in enumerate(forwarded):
        if value == "--candidate-limit":
            base.require(index + 1 < len(forwarded),
                         "--candidate-limit requires a value")
            candidate_values.append(int(forwarded[index + 1]))
        elif value.startswith("--candidate-limit="):
            candidate_values.append(int(value.split("=", 1)[1]))
    base.require(len(candidate_values) <= 1, "duplicate --candidate-limit")
    if not candidate_values:
        forwarded.extend(("--candidate-limit", str(len(plans))))
    else:
        base.require(len(plans) <= candidate_values[0] <= 128,
                     "candidate limit must retain all 83 inherited and 9 typed "
                     "multiflow plans and remain bounded by 128")

    base.ExactCatalog = ExactCatalog
    base.Plan = Plan
    base.frozen_plans = frozen_plans
    base.materialize_plan = materialize_plan
    result = base.main(forwarded)
    output_value = next((forwarded[index + 1]
                         for index, value in enumerate(forwarded[:-1])
                         if value == "--output"), None)
    if output_value is None:
        output_value = next((value.split("=", 1)[1] for value in forwarded
                             if value.startswith("--output=")), None)
    base.require(output_value is not None, "--output is required")
    _augment(Path(output_value).resolve(), union_path, union_document)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
