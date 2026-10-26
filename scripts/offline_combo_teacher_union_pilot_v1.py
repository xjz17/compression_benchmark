#!/usr/bin/env python3
"""Bounded partial-union teacher built on the frozen v6 execution contract.

This pilot deliberately keeps the v6 teacher/search/summary schemas so that
``train_combo_policy_v6.py`` can consume its output.  Extra fields bind every
artifact to the separately frozen 86-method/PDF102 union.  The binding is a
coverage statement, not a claim that the current 6,373-action runtime executes
the full union.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import offline_combo_teacher_v6 as base
import online_matched_train_v6 as fit3


EXTENSION_SCHEMA = "native-online-offline-combo-union-pilot-v1"
COVERAGE_STATUS = "partial_current_runtime_only"
_ORIGINAL_FROZEN_PLANS = base.frozen_plans


@dataclass(frozen=True)
class ActionDescriptor:
    runtime_action: str
    catalog_key: str | None = None
    implementation_id: str | None = None
    input_view: str | None = None
    io_variant: str | None = None
    parameter_mode: str | None = None
    parameter_value: int | None = None
    max_point: int | None = None
    parameter_items: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        base.require(bool(self.runtime_action), "empty runtime action descriptor")
        base.require(tuple(sorted(self.parameter_items)) == self.parameter_items,
                     "descriptor parameter_items must be sorted")


@dataclass(frozen=True)
class CatalogAction:
    action_id: int
    runtime_action: str
    implementation_id: str | None
    input_view: str
    target_slot: int | None
    ordinary_operator: bool
    key: str
    io_variant: str | None
    parameter_mode: str
    parameter_value: int | None
    max_point: int | None
    parameter_items: tuple[tuple[str, str], ...]


class ExactCatalog:
    """Resolve a legal action by current flow type and full descriptor.

    No action-ID ordering participates in resolution.  Ambiguity is an error.
    ``bytes`` is accepted as a generic byte-view fallback only when the current
    logical view has no exact catalog row for the requested descriptor.
    """

    def __init__(self, catalog_path: Path, profile_path: Path):
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
        base.require(document.get("registry_fingerprint") ==
                     "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd",
                     "base catalog registry fingerprint drift")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        excluded = frozenset(str(value) for value in
                             profile.get("excluded_implementation_ids", ()))
        base.require({"A109", "A110"}.issubset(excluded),
                     "pilot profile must exclude cmix and NNCP")
        actions: dict[int, CatalogAction] = {}
        for row in document["registry_payload"]["actions"]:
            items = tuple(sorted((str(item[0]), str(item[1]))
                                 for item in row.get("parameter_items", ())))
            action = CatalogAction(
                int(row["action_id"]), str(row["runtime_action"]),
                None if row.get("implementation_id") is None else
                str(row["implementation_id"]), str(row.get("input_view", "")),
                None if row.get("target_slot") is None else int(row["target_slot"]),
                bool(row.get("ordinary_operator")), str(row["key"]),
                None if row.get("io_variant") is None else str(row["io_variant"]),
                str(row.get("parameter_mode", "none")),
                None if row.get("parameter_value") is None else
                int(row["parameter_value"]),
                None if row.get("max_point") is None else int(row["max_point"]),
                items)
            base.require(action.action_id not in actions,
                         "duplicate catalog action ID")
            actions[action.action_id] = action
        base.require(len(actions) == fit3.FLOW_STOP_BEGIN,
                     "base catalog action count drift")
        self.actions = actions
        self.catalog_path = catalog_path.resolve()
        self.profile_path = profile_path.resolve()
        self.excluded_implementation_ids = excluded

    @staticmethod
    def _specified_match(action: CatalogAction,
                         descriptor: ActionDescriptor) -> bool:
        checks = (
            (descriptor.catalog_key, action.key),
            (descriptor.implementation_id, action.implementation_id),
            (descriptor.input_view, action.input_view),
            (descriptor.io_variant, action.io_variant),
            (descriptor.parameter_mode, action.parameter_mode),
            (descriptor.parameter_value, action.parameter_value),
            (descriptor.max_point, action.max_point),
        )
        return all(expected is None or expected == actual
                   for expected, actual in checks) and (
                       not descriptor.parameter_items or
                       descriptor.parameter_items == action.parameter_items)

    def exact(self, legal: Sequence[int], descriptor: ActionDescriptor, *,
              target_slot: int, current_input_view: str) -> int:
        candidates = [self.actions[action_id] for action_id in legal
                      if action_id < fit3.FLOW_STOP_BEGIN and
                      self.actions[action_id].runtime_action ==
                      descriptor.runtime_action and
                      self.actions[action_id].target_slot == target_slot and
                      self._specified_match(self.actions[action_id], descriptor)]
        exact_view = [row for row in candidates
                      if row.input_view == current_input_view]
        if descriptor.input_view is not None:
            selected = candidates
        elif exact_view:
            selected = exact_view
        else:
            selected = [row for row in candidates if row.input_view == "bytes"]
        base.require(len(selected) == 1,
                     f"descriptor {descriptor!r} at slot {target_slot} view "
                     f"{current_input_view!r} has {len(selected)} legal matches")
        action = selected[0]
        base.require(action.implementation_id not in
                     self.excluded_implementation_ids,
                     "excluded implementation became a teacher action")
        return action.action_id

    def describe(self, recipe: Sequence[int]) -> list[dict[str, Any]]:
        rows = []
        for position, action_id in enumerate(recipe):
            if action_id >= fit3.FLOW_STOP_BEGIN:
                rows.append({"position": position, "action_id": action_id,
                             "mechanism": "per_flow_stop",
                             "flow_slot": action_id - fit3.FLOW_STOP_BEGIN})
                continue
            action = self.actions[action_id]
            rows.append({"position": position, "action_id": action_id,
                         "mechanism": action.runtime_action,
                         "implementation_id": action.implementation_id,
                         "input_view": action.input_view,
                         "target_slot": action.target_slot,
                         "catalog_key": action.key,
                         "io_variant": action.io_variant,
                         "parameter_mode": action.parameter_mode,
                         "parameter_value": action.parameter_value,
                         "max_point": action.max_point,
                         "parameter_items": [list(item)
                                             for item in action.parameter_items]})
        return rows


@dataclass(frozen=True)
class Plan:
    name: str
    prefix: tuple[ActionDescriptor, ...] = ()
    leaf_backend: tuple[ActionDescriptor, ...] = ()
    backend_each_leaf: bool = False
    coverage_origin: str = "v6_template"

    def __post_init__(self) -> None:
        base.require(bool(self.name), "empty candidate name")
        base.require(len(self.prefix) <= 3, "pilot prefix depth exceeds three")
        base.require(bool(self.prefix) or self.name == "raw",
                     "non-raw plan is empty")
        base.require(not self.backend_each_leaf or self.leaf_backend,
                     "multi-flow plan requires a leaf backend")


def _descriptor(runtime_action: str, implementation_id: str | None = None
                ) -> ActionDescriptor:
    return ActionDescriptor(runtime_action, implementation_id=implementation_id)


def frozen_plans() -> tuple[Plan, ...]:
    historical = tuple(Plan(
        plan.name, tuple(_descriptor(value) for value in plan.prefix),
        tuple(_descriptor(value) for value in plan.leaf_backend),
        plan.backend_each_leaf, "all_76_v6_templates")
        for plan in _ORIGINAL_FROZEN_PLANS())
    base.require(len(historical) == 76, "frozen v6 template count drift")
    # These are registered C++ stage chains with explicit parent ordering.
    # They are executable seeds, not full-union or upstream-equivalence claims.
    extra = (
        Plan("union_seed_gorilla_registered",
             (_descriptor("op.xor.gorilla.cpp_v1", "A031"),
              _descriptor("op.trailing_zero_pack.gorilla.cpp_v1", "A038")),
             coverage_origin="registered_specialist_seed"),
        Plan("union_seed_chimp128_registered",
             (_descriptor("op.xor.chimp128.cpp_v1", "A032"),
              _descriptor("op.trailing_zero_pack.chimp128.cpp_v1", "A039")),
             coverage_origin="registered_specialist_seed"),
        Plan("union_seed_svb_registered",
             (_descriptor("op.delta.stream_vbyte.cpp_v1", "A028"),
              _descriptor("op.integer_code.stream_vbyte.cpp_v1", "A072")),
             coverage_origin="registered_specialist_seed"),
        Plan("union_seed_fastpfor_registered",
             (_descriptor("op.bitpack_int.fastpfor_pfor.cpp_v1", "A020"),
              _descriptor("op.integer_code.fastpfor_variablebyte.cpp_v1", "A073")),
             coverage_origin="registered_specialist_seed"),
        Plan("union_seed_pongo_registered",
             (_descriptor("op.eraser.pongo.cpp_v1", "A002"),
              _descriptor("op.xor.pongo.cpp_v1", "A031"),
              _descriptor("op.trailing_zero_pack.pongo.cpp_v1", "A038")),
             coverage_origin="registered_specialist_seed"),
        Plan("union_seed_camel_registered",
             (_descriptor("op.eraser.camel.cpp_v1", "A004"),
              _descriptor("op.xor.camel.cpp_v1", "A031"),
              _descriptor("op.bitstream_pack.camel.cpp_v1", "A076")),
             coverage_origin="registered_specialist_seed"),
        Plan("union_seed_fpzip_registered",
             (_descriptor("op.delta.fpzip.cpp_v1", "A030"),
              _descriptor("op.entropy_range.fpzip.cpp_v1", "A079")),
             coverage_origin="registered_specialist_seed"),
    )
    plans = historical + extra
    base.require(len(plans) <= 160 and
                 len({plan.name for plan in plans}) == len(plans),
                 "invalid union pilot candidate inventory")
    return plans


def _stop(backend: fit3.OnlineBackend, snapshot: fit3.Snapshot,
          views: list[str], slot: int) -> tuple[fit3.Snapshot, list[str]]:
    action = fit3.FLOW_STOP_BEGIN + slot
    base.require(action in snapshot.legal_action_ids,
                 "requested per-flow STOP is illegal")
    return backend.step(action), views[:slot] + views[slot + 1:]


def _apply(backend: fit3.OnlineBackend, catalog: ExactCatalog,
           snapshot: fit3.Snapshot, descriptor: ActionDescriptor,
           views: list[str], slot: int) -> tuple[fit3.Snapshot, list[str]]:
    base.require(not snapshot.terminal and 0 <= slot < len(views),
                 "invalid stage application")
    before = len(views)
    action = catalog.exact(snapshot.legal_action_ids, descriptor,
                           target_slot=slot,
                           current_input_view=views[slot])
    next_snapshot = backend.step(action)
    after = (0 if next_snapshot.terminal else
             int(next_snapshot.response["open_flow_count"]))
    produced = after - (before - 1)
    base.require(produced >= 0, "ordinary stage open-flow count drift")
    # Current online.cpp assigns bytes to every ordinary output except typed
    # semantic/split operations; this pilot's plans use only ordinary stages.
    next_views = views[:slot] + views[slot + 1:] + ["bytes"] * produced
    return next_snapshot, next_views


def materialize_plan(backend: fit3.OnlineBackend, block: fit3.TrainingBlock,
                     contract: fit3.OnlineContract, catalog: ExactCatalog,
                     plan: Plan) -> tuple[int, ...]:
    base.require(block.roots[0].get("original_numeric_payload") is True,
                 "pilot requires the primary numeric root at slot zero")
    snapshot = backend.reset(block, fit3.FIXED_WEIGHTS[0], contract)
    views = [str(root["logical_view"]) for root in block.roots]
    initial = int(snapshot.response.get("open_flow_count", len(views)))
    base.require(initial == len(views), "C++ initial open-flow count drift")
    for slot in range(initial - 1, 0, -1):
        snapshot, views = _stop(backend, snapshot, views, slot)
    if plan.name == "raw":
        snapshot, views = _stop(backend, snapshot, views, 0)
        return fit3._recipe(snapshot.response)
    for descriptor in plan.prefix:
        snapshot, views = _apply(backend, catalog, snapshot, descriptor,
                                 views, 0)
        base.require(not snapshot.terminal,
                     f"candidate {plan.name} exhausted budget in prefix")
    if plan.backend_each_leaf:
        while not snapshot.terminal:
            slot = len(views) - 1
            for descriptor in plan.leaf_backend:
                snapshot, views = _apply(backend, catalog, snapshot,
                                         descriptor, views, slot)
                base.require(not snapshot.terminal,
                             f"candidate {plan.name} exhausted before leaf STOP")
                slot = len(views) - 1
            snapshot, views = _stop(backend, snapshot, views, slot)
    else:
        while not snapshot.terminal:
            snapshot, views = _stop(backend, snapshot, views, len(views) - 1)
    return fit3._recipe(snapshot.response)


def _rewrite_jsonl(path: Path, extra: Mapping[str, Any]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    base.atomic_jsonl(path, ({**row, **extra} for row in rows))


def _augment(output: Path, union_path: Path, union: Mapping[str, Any]) -> None:
    identity = str(union["space_sha256"])
    extension = {"schema": EXTENSION_SCHEMA,
                 "union_space_sha256": identity,
                 "union_catalog": str(union_path.resolve()),
                 "union_catalog_sha256": base.sha256(union_path),
                 "coverage_status": COVERAGE_STATUS,
                 "full_union_training_ready": False,
                 "historical_v6_templates_considered": 76,
                 "registered_specialist_seeds_considered":
                 len(frozen_plans()) - 76}
    search_path = output / "search_space.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    search.update({"schema_extension": extension,
                   "union_space_sha256": identity,
                   "coverage_status": COVERAGE_STATUS,
                   "full_union_coverage_claimed": False})
    search.setdefault("artifacts", {}).update({
        str(Path(__file__).resolve()): base.sha256(Path(__file__)),
        str(union_path.resolve()): base.sha256(union_path),
    })
    base.atomic_json(search_path, search)
    for filename in ("candidate_measurements.jsonl", "teacher.jsonl"):
        _rewrite_jsonl(output / filename,
                       {"union_space_sha256": identity,
                        "coverage_status": COVERAGE_STATUS})
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"schema_extension": extension,
                    "union_space_sha256": identity,
                    "coverage_status": COVERAGE_STATUS,
                    "full_union_coverage_claimed": False})
    summary.setdefault("limitations", []).append(
        "The pilot measures only current 6,373-action executable paths; the "
        "86-method/PDF102 union remains partially unwired and is not claimed covered.")
    base.atomic_json(summary_path, summary)
    receipt_path = output / "teacher.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update({"teacher_sha256": base.sha256(output / "teacher.jsonl"),
                    "search_space_sha256": base.sha256(search_path),
                    "schema_extension": extension,
                    "union_space_sha256": identity,
                    "coverage_status": COVERAGE_STATUS,
                    "full_union_coverage_claimed": False})
    base.atomic_json(receipt_path, receipt)


def _consume_option(argv: list[str], name: str) -> tuple[Path, list[str]]:
    prefix = name + "="
    for index, value in enumerate(argv):
        if value.startswith(prefix):
            return Path(value[len(prefix):]), argv[:index] + argv[index + 1:]
        if value == name:
            base.require(index + 1 < len(argv), f"{name} requires a path")
            return Path(argv[index + 1]), argv[:index] + argv[index + 2:]
    raise ValueError(f"required option absent: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    union_path, forwarded = _consume_option(forwarded, "--union-catalog")
    base.require(union_path.is_file(), f"union catalog absent: {union_path}")
    union = json.loads(union_path.read_text(encoding="utf-8"))
    base.require(union.get("schema") == "operator86-pdf102-candidate-union-v1",
                 "unexpected union catalog schema")
    calculated = fit3.digest(fit3.canonical_json(union["payload"]))
    base.require(calculated == union.get("space_sha256"),
                 "union catalog payload hash drift")
    plans = frozen_plans()
    candidate_values = []
    for index, value in enumerate(forwarded):
        if value == "--candidate-limit":
            base.require(index + 1 < len(forwarded),
                         "--candidate-limit requires a value")
            candidate_values.append(int(forwarded[index + 1]))
        elif value.startswith("--candidate-limit="):
            candidate_values.append(int(value.split("=", 1)[1]))
    base.require(len(candidate_values) <= 1,
                 "duplicate --candidate-limit")
    if not candidate_values:
        forwarded.extend(("--candidate-limit", str(len(plans))))
    else:
        base.require(len(plans) <= candidate_values[0] <= 128,
                     "union pilot candidate limit must retain all 76 templates "
                     "and all registered seeds")
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
    _augment(Path(output_value).resolve(), union_path, union)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
