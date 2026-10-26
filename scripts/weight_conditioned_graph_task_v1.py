#!/usr/bin/env python3
"""Executable typed-graph task for :mod:`weight_conditioned_grpo_v1`.

This adapter derives its mechanism catalog from the PDF-aligned 67-mechanism
inventory while keeping profiles, recipes and framing adapters non-atomic.  A
policy action is a fixed tuple of (operation, target leaf slot,
parameters, input view), not merely an operator-family name.  The catalog is
global and deterministic; task-specific legality masks only remove entries
whose typed preconditions or graph budgets are not satisfied.

Python owns scheduling, features, accounting and timing.  Every transform and
inverse is delegated to ``typed_graph_runtime`` and its current native C/C++
bridges.  Terminal evaluation replays the recipe from the original roots,
builds a complete TGF1 frame, and restores that frame without access to the
encoder state.  There is no Python compression fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

import exception_separation
import nncp_entropy_codec
import operator_implementation_catalog
import operator_taxonomy_v30_v7 as taxonomy
import prediction_entropy_codec
import residual_window_pack
import typed_graph_runtime as graph_runtime
import unified_action_runtime as unified_runtime
import weight_conditioned_operator_gate_v1 as operator_gate
from weight_conditioned_grpo_v1 import (
    CompressionMetrics,
    TerminalOutcome,
    objective_value,
)
from weighted_objective_pruning_v1 import BoundState


SCHEMA = "weight-conditioned-graph-task-v1"
CATALOG_SCHEMA = "weight-conditioned-fixed-graph-actions-v1"
STATE_FEATURE_SCHEMA = "weight-conditioned-graph-state-features-v2"
STATE_FEATURE_DIM = 28
STATE_CURRENT_OBJECTIVE_INDEX = 15
FLOW_FEATURE_BUCKETS = 6
MEASUREMENT_PROTOCOL_SCHEMA = "typed-graph-timing-samples-v2"
EXPECTED_OPERATOR_PDF_SHA256 = (
    "d32fb153988d1e0afe702acbf0279a14183ef38ca130e157a9c7a846a3f020f7"
)
MAX_ORDINARY_DEPTH = 10
MAX_GRAPH_NODES = 64
MAX_ACTIVE_STREAMS = 32
FIRST_BLOCK_CANDIDATE_BUDGET = 64
CONTINUATION_CANDIDATE_BUDGET = 16


class GraphTaskError(RuntimeError):
    """The fixed action contract or executable graph task was violated."""


class EvaluationBudgetExceeded(GraphTaskError):
    """A complete candidate was requested after the block budget was spent."""


@dataclass(frozen=True)
class GraphTaskConfig:
    max_ordinary_depth: int = MAX_ORDINARY_DEPTH
    max_graph_nodes: int = MAX_GRAPH_NODES
    max_active_streams: int = MAX_ACTIVE_STREAMS
    max_policy_edits: int = MAX_GRAPH_NODES
    timing_warmups: int = 1
    timing_repetitions: int = 5
    independent_cold_validation: bool = True
    cold_validation_timeout_seconds: float = 30.0
    allow_large_memory: bool = False
    require_formal_training_ready: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_ordinary_depth <= MAX_ORDINARY_DEPTH:
            raise ValueError("max_ordinary_depth must be in [1,10]")
        if not 1 <= self.max_graph_nodes <= MAX_GRAPH_NODES:
            raise ValueError("max_graph_nodes must be in [1,64]")
        if not 1 <= self.max_active_streams <= MAX_ACTIVE_STREAMS:
            raise ValueError("max_active_streams must be in [1,32]")
        if not 1 <= self.max_policy_edits <= self.max_graph_nodes:
            raise ValueError("max_policy_edits must fit the graph-node budget")
        if (isinstance(self.timing_warmups, bool)
                or not isinstance(self.timing_warmups, int)
                or not 0 <= self.timing_warmups <= 31):
            raise ValueError("timing_warmups must be in [0,31]")
        if (isinstance(self.timing_repetitions, bool)
                or not isinstance(self.timing_repetitions, int)
                or not 1 <= self.timing_repetitions <= 31):
            raise ValueError("timing_repetitions must be in [1,31]")
        if (not math.isfinite(self.cold_validation_timeout_seconds)
                or self.cold_validation_timeout_seconds <= 0.0):
            raise ValueError("cold-validation timeout must be finite and positive")
        if type(self.allow_large_memory) is not bool:
            raise ValueError("allow_large_memory must be boolean")
        if type(self.require_formal_training_ready) is not bool:
            raise ValueError("require_formal_training_ready must be boolean")


class CandidateBudget:
    """Thread-safe complete-candidate accounting shared by one block search."""

    def __init__(self, limit: int, *, hard_limit: bool = True):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("candidate budget must be a positive integer")
        self.limit = limit
        self.hard_limit = bool(hard_limit)
        self._used = 0
        self._lock = threading.Lock()

    def reserve(self) -> int:
        with self._lock:
            if self.hard_limit and self._used >= self.limit:
                raise EvaluationBudgetExceeded(
                    f"complete-candidate budget exhausted ({self._used}/{self.limit})"
                )
            self._used += 1
            return self._used

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._used, self.limit

    def reset(self, limit: int | None = None) -> None:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                  or limit < 1):
            raise ValueError("candidate budget must be a positive integer")
        with self._lock:
            if limit is not None:
                self.limit = limit
            self._used = 0


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    key: str
    runtime_action: str
    target_slot: int | None
    other_slot: int | None = None
    input_view: str = "bytes"
    parameter_mode: str = "none"
    parameter_value: int | None = None
    parameter_items: tuple[tuple[str, int | bool], ...] = ()
    max_point: int | None = None
    io_variant: str | None = None
    implementation_id: str | None = None
    required_parent_runtime_actions: tuple[str, ...] = ()
    required_logical_views: tuple[str, ...] = ()
    input_byte_multiple: int = 1
    required_nonserial_width: int | None = None
    allowed_physical_widths: tuple[int, ...] = ()
    input_nonempty: bool = False
    edit_kind: str | None = None
    ordinary_operator: bool = True
    category: str = "operator"

    def json_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskState:
    graph: graph_runtime.GraphState
    leaf_depths: tuple[tuple[str, int], ...]
    leaf_views: tuple[tuple[str, str], ...]
    parent_actions: tuple[tuple[str, int], ...]
    recipe: tuple[int, ...]
    node_count: int
    policy_edits: int
    replayed_nodes: int
    continuation_edit_mode: bool = False
    pending_edit_kind: str | None = None
    pending_edit_index: int | None = None
    edit_count: int = 0
    edit_history: tuple[tuple[str, int, int | None], ...] = ()

    def depths(self) -> dict[str, int]:
        return dict(self.leaf_depths)

    def views(self) -> dict[str, str]:
        return dict(self.leaf_views)

    def parents(self) -> dict[str, int]:
        return dict(self.parent_actions)


@dataclass(frozen=True)
class RecipeEdit:
    """One bounded edit to a replayed prior-block recipe."""

    kind: str
    index: int
    action_id: int | None = None


_NNCP_COMPLETE_ENTROPY_OPERATOR = nncp_entropy_codec.ACTION
_A111_WIDTHS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
_BITUNPACK_WIDTHS = (1, 2, 4, 8)
_DECIMAL_MAX_POINTS = tuple(range(19))

# One mechanism may need more than one explicit I/O variant because a later
# mechanism consumes the packet produced by that variant.  These names are all
# current taxonomy/policy registry names; profile/recipe/adapter identities are
# deliberately absent from the action catalog.
_MECHANISM_ACTION_OVERRIDES: dict[str, tuple[str, ...]] = {
    "A007": ("op.scaling.exact_decimal.cpp_v2",),
    "A009": ("op.scaling.alp.cpp_v1", "op.scaling.buff.cpp_v1"),
    "A013": (
        "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
        "op.frame_of_reference.bos.cpp_v1",
        "op.frame_of_reference.buff.cpp_v2",
    ),
    "A014": ("op.block_transform.reference_v1",),
    "A020": (
        "op.bitpack_int.max_value_width.cpp_v1",
        "op.bitpack_int.fastpfor_pfor.cpp_v1",
    ),
    "A023": ("op.integer_code.uleb128.reference_values.cpp_v2",),
    "A028": (
        "op.delta.reference_v1",
        "op.delta.ts_2diff.cpp_v1",
        "op.delta.bos.cpp_v1",
        "op.delta.spdp.cpp_v1",
        "op.delta.stream_vbyte.cpp_v1",
    ),
    "A031": (
        graph_runtime.ADJACENT_XOR_PORT_ACTION,
        "op.xor.reference_v1",
        "op.xor.previous_word.cpp_v2",
        "op.xor.gorilla.cpp_v1",
        "op.xor.camel.cpp_v1",
        "op.xor.pongo.cpp_v1",
    ),
    "A032": ("op.xor.chimp_history128.cpp_v2", "op.xor.chimp128.cpp_v1"),
    "A033": ("op.xor.fpc.cpp_v1", "op.xor.pfpc.cpp_v1"),
    "A038": (
        "op.trailing_zero_pack.gorilla.cpp_v1",
        "op.trailing_zero_pack.pongo.cpp_v1",
    ),
    "A040": (
        "op.trailing_zero_pack.fpc.cpp_v1",
        "op.trailing_zero_pack.pfpc.cpp_v1",
    ),
    "A046": ("op.zigzag.signed64.raw.cpp_v2",),
    "A049": (
        "op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1",
        "op.block_transform.blosc_shuffle.typesize4.cpp_reference_v1",
        "op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1",
    ),
    "A060": (
        "op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1",
        "op.lz_match.native_lz.aligned8.min3.max18",
        "op.match_parse_policy.native_lz.aligned8_greedy",
    ),
    "A067": ("op.match_token_format.native_lz.ctrl8_offset16_len8.bias3",),
}

_DIRECT_MECHANISM_ACTIONS = {
    "A035": residual_window_pack.ELF_PLUS_ACTION,
    "A103": "float_deconstruct",
    "A104": "bitsplit",
    "A105": "bitunpack",
    "A106": "op.prediction_probability.ppm.provider_v1",
    "A107": "op.prediction_probability.dmc.provider_v1",
    "A108": "op.prediction_probability.paq.provider_v1",
    "A109": "op.prediction_probability.cmix.provider_v1",
    "A110": _NNCP_COMPLETE_ENTROPY_OPERATOR,
    "A111": exception_separation.ALGORITHM_NAME,
}

_EXPLICIT_IO_VARIANTS: dict[str, tuple[str, ...]] = {
    # A007 is one scaling mechanism with two historical packet carriers.  The
    # carrier is an explicit fixed-catalog choice; neither carrier is promoted
    # to a separate mechanism.
    "A007": (
        "op.scaling.ts_2diff.cpp_v1",
        "op.scaling.simple8b.cpp_v1",
    ),
}

# A few exact stages intentionally use ActionData.bytes on the wire even
# though their native kernels consume logical 64-bit words.  The UAR
# descriptor therefore cannot by itself distinguish them from arbitrary-byte
# entropy stages.  These inventory-backed semantic overrides become fixed
# catalog preconditions; execution still passes ``input_view="bytes"``.
_WORD64_BYTE_ACTIONS = {
    "op.integer_code.uleb128.reference_values.cpp_v2": ("int64",),
    "op.delta.spdp.cpp_v1": ("int64", "float64"),
    "op.xor.fpc.cpp_v1": ("int64", "float64"),
    "op.xor.pfpc.cpp_v1": ("int64", "float64"),
}
_RAW_UINT64_LANE_ACTIONS = {
    "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
    "op.bitpack_int.max_value_width.cpp_v1",
}

_ACTION_REQUIRED_PARENTS: dict[str, tuple[str, ...]] = {
    "op.xor.camel.cpp_v1": ("op.eraser.camel.cpp_v1",),
    "op.xor.pongo.cpp_v1": ("op.eraser.pongo.cpp_v1",),
    "op.frame_of_reference.bos.cpp_v1": ("op.delta.bos.cpp_v1",),
    "op.frame_of_reference.buff.cpp_v2": ("op.scaling.buff.cpp_v1",),
    "op.outlier_separation.bos.cpp_v1": ("op.frame_of_reference.bos.cpp_v1",),
    "op.trailing_zero_pack.gorilla.cpp_v1": ("op.xor.gorilla.cpp_v1",),
    "op.trailing_zero_pack.pongo.cpp_v1": ("op.xor.pongo.cpp_v1",),
    "op.trailing_zero_pack.chimp128.cpp_v1": ("op.xor.chimp128.cpp_v1",),
    "op.trailing_zero_pack.fpc.cpp_v1": ("op.xor.fpc.cpp_v1",),
    "op.trailing_zero_pack.pfpc.cpp_v1": ("op.xor.pfpc.cpp_v1",),
    "op.dictionary.corad_tricklet_xor.cpp_v2": (
        "op.scaling.corad_normalize_exact.cpp_v2",
    ),
    "op.block_transform.buff_byte_plane_split.cpp_v2": (
        "op.frame_of_reference.buff.cpp_v2",
    ),
    "op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1": (
        "op.block_transform.int64_lo_hi32.cpp_v1",
    ),
    "op.lz_match.native_lz.aligned8.min3.max18": (
        "op.match_window_model.native_lz.window4096",
    ),
    "op.match_parse_policy.native_lz.aligned8_greedy": (
        "op.lz_match.native_lz.aligned8.min3.max18",
    ),
    "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3": (
        "op.match_parse_policy.native_lz.aligned8_greedy",
    ),
    "op.match_token_format.snappy.cpp_v1": (
        "op.lz_match.snappy_snap_1_1_1.cpp_v1",
        "op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",
    ),
    "op.match_token_format.lz4.cpp_v1": (
        "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
    ),
    "op.match_token_format.blosclz.ctrl8.cpp_reference_v1": (
        "op.lz_match.blosclz.greedy.cpp_reference_v1",
    ),
    "op.integer_code.stream_vbyte.cpp_v1": ("op.delta.stream_vbyte.cpp_v1",),
    "op.integer_code.fastpfor_variablebyte.cpp_v1": (
        "op.bitpack_int.fastpfor_pfor.cpp_v1",
    ),
    "op.integer_code.lz4.cpp_v1": ("op.match_token_format.lz4.cpp_v1",),
    "op.bitstream_pack.camel.cpp_v1": ("op.xor.camel.cpp_v1",),
    "op.entropy_range.fpzip.cpp_v1": ("op.delta.fpzip.cpp_v1",),
    "op.entropy_special.spdp.cpp_v1": ("op.block_transform.spdp.cpp_v1",),
}

# These are deterministic runtime adapters required by three retained mechanisms.
# They are profiles/recipe wiring in the audited 102-entry directory, not policy
# operators.  Consequently they are serialized as ordinary TGF1 runtime steps
# inside the selected mechanism action, but never receive a policy action ID and
# can never appear in a replayable recipe.
_INTERNAL_PARENT_ADAPTERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "op.lz_match.native_lz.aligned8.min3.max18": (
        "op.match_window_model.native_lz.window4096", (),
    ),
    "op.match_token_format.blosclz.ctrl8.cpp_reference_v1": (
        "op.lz_match.blosclz.greedy.cpp_reference_v1",
        ("op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1",),
    ),
    "op.entropy_special.spdp.cpp_v1": (
        "op.block_transform.spdp.cpp_v1", ("op.delta.spdp.cpp_v1",),
    ),
}
_INTERNAL_ADAPTER_RUNTIME_ACTIONS = frozenset(
    action for action, _parents in _INTERNAL_PARENT_ADAPTERS.values()
)

_MECHANISM_EXECUTION_BLOCKERS = {
    "A109": (
        "cmix requires explicit large-memory opt-in and a matching complete "
        "cold-decode validation receipt"
    ),
    "A110": (
        "NNCP requires Linux x86-64 (or verified Docker) plus matching "
        "standalone and typed-graph NNP1 receipts"
    ),
}


def _mechanism_runtime_names(item: Mapping[str, Any]) -> tuple[str, ...]:
    implementation_id = item["implementation_id"]
    if implementation_id in _DIRECT_MECHANISM_ACTIONS:
        return (_DIRECT_MECHANISM_ACTIONS[implementation_id],)
    if implementation_id in _MECHANISM_ACTION_OVERRIDES:
        return _MECHANISM_ACTION_OVERRIDES[implementation_id]
    names = tuple(item.get("source_action_names", ()))
    return names[:1]


def _input_views(implementation_id: str, runtime_action: str) -> tuple[str, ...]:
    if runtime_action == graph_runtime.ADJACENT_XOR_PORT_ACTION:
        return ("int64", "float64")
    if implementation_id == "A035":
        return ("physical_word",)
    if implementation_id == "A103":
        return ("float32",)
    if implementation_id in {"A104", "A105", "A106", "A107", "A108", "A109", "A110"}:
        return ("bytes",)
    if implementation_id == "A111":
        return ("uint32",)
    try:
        descriptor = unified_runtime.describe_action(runtime_action)
    except (KeyError, ValueError):
        return ()
    return tuple(kind for kind in descriptor.accepted_input_kinds
                 if kind in {"bytes", "int64", "float64"})


def _parameter_variants(implementation_id: str) -> tuple[dict[str, Any], ...]:
    if implementation_id in {"A006", "A007", "A008"}:
        return tuple({"max_point": value} for value in _DECIMAL_MAX_POINTS)
    if implementation_id == "A103":
        return ({"parameter_mode": "fp32"},)
    if implementation_id == "A104":
        return ({"parameter_mode": "top8"},)
    if implementation_id == "A105":
        return tuple({"parameter_mode": "bits", "parameter_value": bits}
                     for bits in _BITUNPACK_WIDTHS)
    if implementation_id == "A106":
        return ({"parameter_items": (("ppmd_memory_mib", 16), ("ppmd_order", 6))},)
    if implementation_id == "A108":
        return ({"parameter_items": (("paq_level", 0),)},)
    if implementation_id == "A109":
        return ({"parameter_items": (("allow_large_memory", True),)},)
    if implementation_id == "A111":
        return tuple({"parameter_mode": "width", "parameter_value": width}
                     for width in _A111_WIDTHS)
    return ({},)


def _semantic_input_contract(
    implementation_id: str, runtime_action: str, input_view: str
) -> dict[str, Any]:
    logical_views = _WORD64_BYTE_ACTIONS.get(runtime_action, ())
    if not logical_views and input_view in {
        "int64", "float64", "float32", "uint32",
        "physical_word",
    }:
        logical_views = (input_view,)
    widths = {
        "int64": 8, "float64": 8, "float32": 4, "uint32": 4,
        "physical_word": 4,
    }
    multiples = [widths[view] for view in logical_views if view in widths]
    if runtime_action in _RAW_UINT64_LANE_ACTIONS:
        multiples.append(8)
    return {
        "required_logical_views": tuple(logical_views),
        "input_byte_multiple": max(multiples, default=1),
        "required_nonserial_width": (
            8 if runtime_action == "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1"
            else None
        ),
        "allowed_physical_widths": (1, 2, 4, 8) if runtime_action == "bitsplit" else (),
        # A successful positive fixture does not certify empty-input support;
        # fail closed for every ordinary mechanism action.
        "input_nonempty": bool(implementation_id),
    }


def _mechanism_templates() -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for item in operator_implementation_catalog.MECHANISMS:
        implementation_id = item["implementation_id"]
        for runtime_action in _mechanism_runtime_names(item):
            try:
                graph_descriptor = taxonomy.describe_action(runtime_action)
                if implementation_id in _EXPLICIT_IO_VARIANTS:
                    io_variants: tuple[str | None, ...] = _EXPLICIT_IO_VARIANTS[
                        implementation_id
                    ]
                else:
                    io_variants = (
                        runtime_action if runtime_action in graph_descriptor.io_variants
                        else None,
                    )
            except ValueError:
                io_variants = (None,)
            for io_variant in io_variants:
                for view in _input_views(implementation_id, runtime_action):
                    for variant in _parameter_variants(implementation_id):
                        parameters = tuple(variant.get("parameter_items", ()))
                        suffix = ",".join(
                            f"{key}={int(value)}" for key, value in parameters
                        )
                        key = f"{implementation_id}:{runtime_action}"
                        if io_variant is not None and io_variant != runtime_action:
                            key += f"(io={io_variant})"
                        if suffix:
                            key += f"({suffix})"
                        if variant.get("parameter_value") is not None:
                            key += (
                                f"({variant.get('parameter_mode')}="
                                f"{variant['parameter_value']})"
                            )
                        if variant.get("max_point") is not None:
                            key += f"(max_point={variant['max_point']})"
                        templates.append({
                            "key": key,
                            "runtime_action": runtime_action,
                            "input_view": view,
                            "implementation_id": implementation_id,
                            "io_variant": io_variant,
                            "required_parent_runtime_actions": _ACTION_REQUIRED_PARENTS.get(
                                runtime_action, ()
                            ),
                            "category": "mechanism",
                            **_semantic_input_contract(
                                implementation_id, runtime_action, view
                            ),
                            **variant,
                        })
    return templates


def _catalog_rows() -> tuple[ActionSpec, ...]:
    rows: list[ActionSpec] = []

    def add(key: str, runtime_action: str, target: int | None, **values: Any) -> None:
        rows.append(ActionSpec(len(rows), key, runtime_action, target, **values))

    add("STOP", "stop", None, ordinary_operator=False, category="stop")
    templates = _mechanism_templates()
    templates.extend((
        dict(key="split_segments.half", runtime_action="split_segments",
             input_view="preserve", parameter_mode="half",
             ordinary_operator=False, category="split_segments"),
        dict(key="split_segments.quarters", runtime_action="split_segments",
             input_view="preserve", parameter_mode="quarters",
             ordinary_operator=False, category="split_segments"),
    ))
    templates.sort(key=lambda row: (
        row.get("implementation_id") or "ZZZ", row["key"],
        row.get("input_view", ""), row.get("parameter_value", -1),
    ))
    for target in range(MAX_ACTIVE_STREAMS):
        for template in templates:
            item = dict(template)
            base_key = item.pop("key")
            key = f"{base_key}[{item.get('input_view', 'bytes')}]@{target}"
            runtime_action = item.pop("runtime_action")
            add(key, runtime_action, target, **item)
    for left in range(MAX_ACTIVE_STREAMS - 1):
        add(f"concat@{left},{left + 1}", "concat", left,
            other_slot=left + 1, input_view="preserve", ordinary_operator=False,
            category="concat")
    for kind in ("insert", "delete", "replace", "parameter_change"):
        add(
            f"edit_type.{kind}", f"edit_type:{kind}", None,
            input_view="preserve", ordinary_operator=False,
            category="edit_type", edit_kind=kind,
        )
    for position in range(MAX_GRAPH_NODES + 1):
        add(
            f"edit_position@{position}", "edit_position", None,
            input_view="preserve", parameter_value=position,
            ordinary_operator=False, category="edit_position",
        )
    return tuple(rows)


ACTION_CATALOG = _catalog_rows()
STOP_ACTION = 0
ACTION_COUNT = len(ACTION_CATALOG)
ACTION_BY_KEY = {row.key: row for row in ACTION_CATALOG}
EDIT_TYPE_ACTIONS = {
    row.edit_kind: row.action_id for row in ACTION_CATALOG
    if row.category == "edit_type" and row.edit_kind is not None
}
EDIT_POSITION_ACTIONS = {
    int(row.parameter_value): row.action_id for row in ACTION_CATALOG
    if row.category == "edit_position" and row.parameter_value is not None
}
if len(ACTION_BY_KEY) != ACTION_COUNT or any(row.action_id != index
                                             for index, row in enumerate(ACTION_CATALOG)):
    counts: dict[str, int] = {}
    for row in ACTION_CATALOG:
        counts[row.key] = counts.get(row.key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    raise RuntimeError(
        "fixed graph action catalog is not unique and contiguous: "
        f"duplicate_keys={duplicates[:20]}"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


REGISTRY_FINGERPRINT = hashlib.sha256(_canonical_json({
    "schema": CATALOG_SCHEMA,
    "state_feature_schema": STATE_FEATURE_SCHEMA,
    "typed_graph_registry": graph_runtime.CURRENT_GRAPH_REGISTRY,
    "operator_directory_registry": operator_implementation_catalog.REGISTRY_VERSION,
    "operator_directory_sha256": operator_implementation_catalog.MANIFEST_SHA256,
    "expected_operator_pdf_sha256": EXPECTED_OPERATOR_PDF_SHA256,
    "internal_parent_adapters": {
        consumer: {
            "adapter": adapter,
            "required_policy_parents": list(parents),
        }
        for consumer, (adapter, parents) in sorted(
            _INTERNAL_PARENT_ADAPTERS.items()
        )
    },
    "actions": [row.json_record() for row in ACTION_CATALOG],
})).hexdigest()


_PROBE_PREFIXES: dict[str, tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...]] = {
    "op.xor.camel.cpp_v1": (("op.eraser.camel.cpp_v1", "float64", ()),),
    "op.xor.pongo.cpp_v1": (("op.eraser.pongo.cpp_v1", "float64", ()),),
    "op.frame_of_reference.bos.cpp_v1": (("op.delta.bos.cpp_v1", "int64", ()),),
    "op.frame_of_reference.buff.cpp_v2": (("op.scaling.buff.cpp_v1", "float64", ()),),
    "op.outlier_separation.bos.cpp_v1": (
        ("op.delta.bos.cpp_v1", "int64", ()),
        ("op.frame_of_reference.bos.cpp_v1", "bytes", ()),
    ),
    "op.trailing_zero_pack.gorilla.cpp_v1": (
        ("op.xor.gorilla.cpp_v1", "float64", ()),
    ),
    "op.trailing_zero_pack.pongo.cpp_v1": (
        ("op.eraser.pongo.cpp_v1", "float64", ()),
        ("op.xor.pongo.cpp_v1", "bytes", ()),
    ),
    "op.trailing_zero_pack.chimp128.cpp_v1": (
        ("op.xor.chimp128.cpp_v1", "float64", ()),
    ),
    "op.trailing_zero_pack.fpc.cpp_v1": (("op.xor.fpc.cpp_v1", "bytes", ()),),
    "op.trailing_zero_pack.pfpc.cpp_v1": (("op.xor.pfpc.cpp_v1", "bytes", ()),),
    "op.dictionary.corad_tricklet_xor.cpp_v2": (
        ("op.scaling.corad_normalize_exact.cpp_v2", "int64", ()),
    ),
    "op.block_transform.buff_byte_plane_split.cpp_v2": (
        ("op.scaling.buff.cpp_v1", "float64", ()),
        ("op.frame_of_reference.buff.cpp_v2", "bytes", ()),
    ),
    "op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1": (
        ("op.scaling.legacy_exact_decimal.cpp_v1", "int64", ()),
        ("op.block_transform.int64_lo_hi32.cpp_v1", "bytes", ()),
    ),
    "op.lz_match.native_lz.aligned8.min3.max18": (
        ("op.match_window_model.native_lz.window4096", "bytes", ()),
    ),
    "op.match_parse_policy.native_lz.aligned8_greedy": (
        ("op.match_window_model.native_lz.window4096", "bytes", ()),
        ("op.lz_match.native_lz.aligned8.min3.max18", "bytes", ()),
    ),
    "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3": (
        ("op.match_window_model.native_lz.window4096", "bytes", ()),
        ("op.lz_match.native_lz.aligned8.min3.max18", "bytes", ()),
        ("op.match_parse_policy.native_lz.aligned8_greedy", "bytes", ()),
    ),
    "op.match_token_format.snappy.cpp_v1": (
        ("op.lz_match.snappy_snap_1_1_1.cpp_v1", "bytes", ()),
    ),
    "op.match_token_format.lz4.cpp_v1": (
        ("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", "bytes", ()),
    ),
    "op.match_token_format.blosclz.ctrl8.cpp_reference_v1": (
        ("op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1", "bytes", ()),
        ("op.lz_match.blosclz.greedy.cpp_reference_v1", "bytes", ()),
    ),
    "op.integer_code.stream_vbyte.cpp_v1": (
        ("op.delta.stream_vbyte.cpp_v1", "int64", ()),
    ),
    "op.integer_code.fastpfor_variablebyte.cpp_v1": (
        ("op.bitpack_int.fastpfor_pfor.cpp_v1", "int64", ()),
    ),
    "op.integer_code.lz4.cpp_v1": (
        ("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", "bytes", ()),
        ("op.match_token_format.lz4.cpp_v1", "bytes", ()),
    ),
    "op.bitstream_pack.camel.cpp_v1": (
        ("op.eraser.camel.cpp_v1", "float64", ()),
        ("op.xor.camel.cpp_v1", "bytes", ()),
    ),
    "op.entropy_range.fpzip.cpp_v1": (("op.delta.fpzip.cpp_v1", "float64", ()),),
    "op.entropy_special.spdp.cpp_v1": (
        ("op.delta.spdp.cpp_v1", "bytes", ()),
        ("op.block_transform.spdp.cpp_v1", "bytes", ()),
    ),
    residual_window_pack.ELF_PLUS_ACTION: (
        (graph_runtime.ADJACENT_XOR_PORT_ACTION, "int64", ()),
    ),
}


def _probe_key(spec: ActionSpec) -> str:
    return hashlib.sha256(_canonical_json({
        "implementation_id": spec.implementation_id,
        "runtime_action": spec.runtime_action,
        "input_view": spec.input_view,
        "parameter_mode": spec.parameter_mode,
        "parameter_value": spec.parameter_value,
        "parameter_items": spec.parameter_items,
        "max_point": spec.max_point,
        "io_variant": spec.io_variant,
    })).hexdigest()


def _probe_fixture(spec: ActionSpec) -> tuple[graph_runtime.Stream, str]:
    if spec.implementation_id in {"A006", "A007", "A008"}:
        # Zero is exactly representable under every cataloged max_point.  This
        # fixture certifies the parameter channel and cold inverse; real block
        # legality is still checked against the current block before exposure.
        return graph_runtime.Stream(
            "numeric", 8, struct.pack("<16d", *([0.0] * 16))
        ), "float64"
    if spec.implementation_id == "A035":
        words = (0, 1, 1, 0x101, 0x10101, 0x100010101, 0x7fffffffffffffff)
        return graph_runtime.Stream("numeric", 8, struct.pack("<7q", *words)), "int64"
    if spec.implementation_id == "A103":
        words = (0x00000000, 0x80000000, 0x3f800000, 0x7fc01234, 0x00000001)
        return graph_runtime.Stream("numeric", 4, struct.pack("<5I", *words)), "float32"
    if spec.implementation_id == "A104":
        return graph_runtime.Stream("numeric", 4, bytes(range(64))), "bytes"
    if spec.implementation_id == "A111":
        words = (0, 1, 2, 7, 255, 1025, 0xffffffff)
        return graph_runtime.Stream("numeric", 4, struct.pack("<7I", *words)), "uint32"
    prefixes = _PROBE_PREFIXES.get(spec.runtime_action, ())
    root_view = prefixes[0][1] if prefixes else (
        spec.required_logical_views[0]
        if spec.required_logical_views else spec.input_view
    )
    if root_view == "int64":
        values = tuple(range(64))
        return graph_runtime.Stream("numeric", 8, struct.pack("<64q", *values)), "int64"
    if root_view == "float64":
        values = tuple(float(index) / 4.0 for index in range(64))
        return graph_runtime.Stream("numeric", 8, struct.pack("<64d", *values)), "float64"
    raw = (b"ABCDEFGH" * 64)
    return graph_runtime.Stream("serial", 1, raw), "bytes"


def _probe_parameters(spec: ActionSpec) -> dict[str, Any] | None:
    values = dict(spec.parameter_items)
    if spec.parameter_mode == "fp32":
        values.update(format="fp32")
    elif spec.parameter_mode == "top8":
        values.update(mode="top8", width=4)
    elif spec.parameter_mode == "bits":
        values.update(bits=spec.parameter_value)
    elif spec.parameter_mode == "width":
        values.update(width=spec.parameter_value)
    return values or None


def _apply_fixed_graph_action(
    graph: graph_runtime.GraphState,
    spec: ActionSpec,
    input_ids: Sequence[str],
    *,
    parameters: Mapping[str, Any] | None,
    input_view: str,
) -> graph_runtime.GraphState:
    """Apply one catalog action, including the audited max_point channel.

    ``typed_graph_runtime`` v7 does not expose ``max_point`` on
    ``GraphState.apply``.  For the three decimal-scaling mechanisms we invoke
    the same versioned UAR selector directly and construct the identical
    legacy graph edge.  UAR stores the chosen/effective scale inside its UAR1
    packet, so the unmodified TGF1 cold decoder reconstructs it independently.
    """
    if spec.max_point is None:
        return graph.apply(
            spec.runtime_action, list(input_ids), parameters=parameters,
            input_view=input_view, io_variant=spec.io_variant,
        )
    if (spec.implementation_id not in {"A006", "A007", "A008"}
            or spec.max_point not in _DECIMAL_MAX_POINTS
            or len(input_ids) != 1
            or parameters):
        raise GraphTaskError("invalid cataloged max_point graph action")
    input_id = input_ids[0]
    source = graph.streams[input_id]
    if input_view != "float64" or source.kind != "numeric" or source.width != 8:
        raise GraphTaskError("max_point scaling requires one float64 Numeric flow")
    taxonomy_registry = unified_runtime._graph_taxonomy(graph.registry_version)
    descriptor = taxonomy_registry.describe_action(spec.runtime_action)
    selector = descriptor.legacy_policy_action_id
    if selector is None:
        raise GraphTaskError("max_point scaling lacks a versioned policy selector")
    policy_registry = getattr(
        taxonomy_registry, "POLICY_REGISTRY_VERSION",
        taxonomy_registry.LEGACY_REGISTRY_VERSION,
    )
    result = unified_runtime.run_policy_action(
        selector,
        unified_runtime.ActionData("float64", source.data, len(source.data) // 8),
        registry_version=policy_registry,
        io_variant=spec.io_variant,
        max_point=spec.max_point,
    )
    output_id = f"s{graph.next_id}"
    output = graph_runtime.Stream("serial", 1, result.payload)
    record = {
        "action": spec.runtime_action,
        "inputs": [input_id],
        "input_specs": [source.spec()],
        "kind": "legacy",
        "outputs": [output_id],
        "output_specs": [output.spec()],
        "input_view": "float64",
    }
    live = {key: value for key, value in graph.streams.items() if key != input_id}
    live[output_id] = output
    return graph_runtime.GraphState(
        live, dict(graph.roots), graph.steps + (record,), graph.next_id + 1,
        graph.registry_version,
    )


def _cold_probe(frame: bytes, root: graph_runtime.Stream, *, timeout: float = 60.0) -> None:
    script = r'''
import hashlib, json, sys
from typed_graph_runtime import restore_frame
roots = restore_frame(sys.stdin.buffer.read())
result = [{"id": key, "kind": value.kind, "width": value.width,
           "size": len(value.data), "sha256": hashlib.sha256(value.data).hexdigest()}
          for key, value in roots.items()]
sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
'''
    result = subprocess.run(
        [sys.executable, "-c", script], input=frame, capture_output=True,
        cwd=Path(__file__).resolve().parent,
        env={**os.environ, "PYTHONHASHSEED": "0"}, check=False, timeout=timeout,
    )
    if result.returncode:
        raise GraphTaskError(
            "cold restore failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    actual = json.loads(result.stdout)
    expected = [{
        "id": "s0", "kind": root.kind, "width": root.width,
        "size": len(root.data), "sha256": _sha256_bytes(root.data),
    }]
    if actual != expected:
        raise GraphTaskError("cold restore receipt mismatched probe root")


def _run_action_probe(spec: ActionSpec) -> dict[str, Any]:
    record = {
        "probe_key": _probe_key(spec),
        "implementation_id": spec.implementation_id,
        "runtime_action": spec.runtime_action,
        "input_view": spec.input_view,
        "io_variant": spec.io_variant,
        "max_point": spec.max_point,
        "parameters": dict(spec.parameter_items),
        "required_parent_runtime_actions": list(spec.required_parent_runtime_actions),
        "required_logical_views": list(spec.required_logical_views),
        "input_byte_multiple": spec.input_byte_multiple,
        "required_nonserial_width": spec.required_nonserial_width,
        "allowed_physical_widths": list(spec.allowed_physical_widths),
        "input_nonempty": spec.input_nonempty,
        "passed": False,
    }
    if spec.implementation_id == "A109":
        evidence = prediction_entropy_codec.cmix_receipt_status()
        record["external_large_memory_receipt"] = evidence
        if (evidence.get("current") is True
                and evidence.get("action") == spec.runtime_action
                and evidence.get("provider") == "cmix"
                and evidence.get("frame") == "PEC1"
                and evidence.get("cold_decode_independent_process") is True
                and isinstance(evidence.get("frame_size"), int)
                and evidence.get("frame_size") >= 64):
            record.update(
                passed=True,
                evidence_kind="current_external_full_cmix_large_memory_receipt",
                frame_bytes=evidence["frame_size"],
                root_view="bytes",
                output_flow_count=1,
            )
        else:
            record["error"] = evidence.get(
                "reason", "cmix requires a current explicit large-memory cold-decode receipt"
            )
        return record
    if spec.implementation_id == "A110" and not nncp_entropy_codec.availability().get(
        "graph_eligible"
    ):
        record["error"] = nncp_entropy_codec.availability()["reason"]
        return record
    root, root_view = _probe_fixture(spec)
    state = graph_runtime.GraphState.from_streams(
        [root], registry_version=graph_runtime.CURRENT_GRAPH_REGISTRY
    )
    try:
        for action, view, items in _PROBE_PREFIXES.get(spec.runtime_action, ()):
            if len(state.streams) != 1:
                raise GraphTaskError("probe prefix unexpectedly produced multiple data flows")
            state = state.apply(
                action, [next(iter(state.streams))], parameters=dict(items) or None,
                input_view=view,
            )
        if spec.implementation_id == "A035" and len(state.streams) == 2:
            target_id = next(
                key for key, stream in state.streams.items()
                if stream.kind == "numeric" and stream.width == 8 and len(stream.data) != 8
            )
        elif len(state.streams) == 1:
            target_id = next(iter(state.streams))
        else:
            raise GraphTaskError("probe target requires exactly one active input data flow")
        target = state.streams[target_id]
        record["target_input_spec"] = target.spec()
        record["target_input_view"] = (
            spec.required_logical_views[0]
            if spec.required_logical_views else spec.input_view
        )
        input_view = spec.input_view if spec.input_view in {"int64", "float64"} else "bytes"
        state = _apply_fixed_graph_action(
            state, spec, [target_id],
            parameters=_probe_parameters(spec), input_view=input_view,
        )
        frame = state.stop()
        restored = graph_runtime.restore_frame(frame)
        if restored != {"s0": root}:
            raise GraphTaskError("in-process restore mismatched probe root")
        _cold_probe(frame, root)
        record.update(
            passed=True,
            frame_bytes=len(frame),
            root_view=root_view,
            output_flow_count=len(state.streams),
        )
    except (OSError, RuntimeError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        record.update(error_type=type(error).__name__, error=str(error))
    return record


@lru_cache(maxsize=1)
def mechanism_probe_report() -> dict[str, Any]:
    """Execute and cold-restore every fixed mechanism action configuration."""
    specs: dict[str, ActionSpec] = {}
    for spec in ACTION_CATALOG:
        if spec.implementation_id is not None:
            specs.setdefault(_probe_key(spec), spec)
    records = [_run_action_probe(spec) for spec in specs.values()]
    passed_keys = sorted(record["probe_key"] for record in records if record["passed"])
    by_mechanism: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_mechanism.setdefault(record["implementation_id"], []).append(record)
    mechanisms = []
    for item in operator_implementation_catalog.MECHANISMS:
        implementation_id = item["implementation_id"]
        action_records = by_mechanism.get(implementation_id, [])
        mechanisms.append({
            "implementation_id": implementation_id,
            "passed": any(record["passed"] for record in action_records),
            "action_probes": action_records,
        })
    return {
        "schema": "weight-conditioned-mechanism-probes-v1",
        "registry_fingerprint": REGISTRY_FINGERPRINT,
        "mechanism_count": len(mechanisms),
        "passed_mechanism_count": sum(row["passed"] for row in mechanisms),
        "failed_mechanism_count": sum(not row["passed"] for row in mechanisms),
        "passed_action_probe_keys": passed_keys,
        "mechanisms": mechanisms,
    }


def _expanded_taxonomy_names(names: set[str]) -> set[str]:
    """Expand source names through v7 aliases and I/O variants.

    This is used only for the coverage audit.  It does not make an alias a
    policy action and it never changes the fixed action catalog.
    """
    result = {name for name in names if name}
    changed = True
    while changed:
        changed = False
        for descriptor in taxonomy.ACTIONS:
            group = {
                descriptor.name,
                descriptor.mechanism,
                *descriptor.io_variants,
                *descriptor.aliases,
            }
            if result & group and not group <= result:
                result.update(group)
                changed = True
    return result


def _mechanism_coverage() -> dict[str, Any]:
    """Join the 67-mechanism inventory to actual bit-exact cold probes."""
    gate = operator_gate.build_report()
    gate_rows = {
        row["implementation_id"]: row
        for row in gate["entries"] if row["layer"] == "mechanism"
    }
    probe = mechanism_probe_report()
    probe_rows = {row["implementation_id"]: row for row in probe["mechanisms"]}
    catalog_rows: dict[str, list[ActionSpec]] = {}
    policy_runtime_actions = {spec.runtime_action for spec in ACTION_CATALOG}
    raw_reachability_ids = {"A035", "A060", "A067", "A070", "A081"}
    for spec in ACTION_CATALOG:
        if spec.implementation_id is not None:
            catalog_rows.setdefault(spec.implementation_id, []).append(spec)

    mechanisms: list[dict[str, Any]] = []
    missing_catalog: list[dict[str, Any]] = []
    failed_probes: list[dict[str, Any]] = []
    upstream_blocked: list[dict[str, Any]] = []
    policy_blocked: list[dict[str, Any]] = []
    represented_selectable: list[dict[str, Any]] = []
    for item in operator_implementation_catalog.MECHANISMS:
        implementation_id = item["implementation_id"]
        upstream = gate_rows[implementation_id]
        actions = catalog_rows.get(implementation_id, [])
        probe_row = probe_rows.get(implementation_id, {
            "implementation_id": implementation_id,
            "passed": False,
            "action_probes": [],
        })
        passed_actions = [
            row for row in probe_row["action_probes"] if row["passed"]
        ]
        failed_actions = [
            row for row in probe_row["action_probes"] if not row["passed"]
        ]
        record = {
            "implementation_id": implementation_id,
            "name": item["name"],
            "cataloged": bool(actions),
            "fixed_configuration_count": len({
                _probe_key(spec) for spec in actions
            }),
            "runtime_actions": sorted({spec.runtime_action for spec in actions}),
            "upstream_status": upstream["status"],
            "upstream_selectable": bool(upstream["selectable"]),
            "bit_exact_cold_probe_passed": bool(probe_row["passed"]),
            "passed_action_probe_count": len(passed_actions),
            "failed_action_probe_count": len(failed_actions),
            "failed_action_probes": failed_actions,
        }
        if implementation_id == "A035":
            record["conditional_reachability"] = (
                "reachable from raw int64/float64 through the explicit A031 C++ "
                "adjacent-XOR semantic-port action; the first word remains a sibling "
                "leaf and A035 consumes only xor_residuals"
            )
        if implementation_id in raw_reachability_ids:
            reachable_probes = []
            for row in passed_actions:
                target = str(row.get("runtime_action", ""))
                prefixes = [entry[0] for entry in _PROBE_PREFIXES.get(target, ())]
                if target in policy_runtime_actions and all(
                    prefix in policy_runtime_actions
                    or prefix in _INTERNAL_ADAPTER_RUNTIME_ACTIONS
                    for prefix in prefixes
                ):
                    reachable_probes.append({"target": target, "prefixes": prefixes})
            record["raw_column_policy_reachability"] = reachable_probes
            if not reachable_probes:
                reason = (
                    "no passing complete-frame cold probe is composed solely from "
                    "current policy actions starting at a raw data flow"
                )
                policy_blocked.append({
                    "implementation_id": implementation_id,
                    "reason": reason,
                    "kind": "raw_column_policy_reachability",
                })
        if not actions:
            missing = dict(record)
            missing["reason"] = "no fixed action configuration generated from inventory"
            missing_catalog.append(missing)
        if not probe_row["passed"]:
            failed = dict(record)
            failed["reason"] = _MECHANISM_EXECUTION_BLOCKERS.get(
                implementation_id,
                "no fixed configuration completed a bit-exact in-process and cold restore",
            )
            failed_probes.append(failed)
        if not upstream["selectable"]:
            blocked = dict(record)
            blocked["reason"] = upstream.get(
                "blocker", "upstream operator gate marks mechanism not selectable"
            )
            upstream_blocked.append(blocked)
        if upstream["selectable"] and actions:
            represented_selectable.append(record)
        mechanisms.append(record)

    non_atomic = {"profile": [], "recipe": [], "adapter": []}
    for item in operator_implementation_catalog.IMPLEMENTATIONS:
        layer = item["accounting"]["layer"]
        if layer in non_atomic:
            non_atomic[layer].append({
                "implementation_id": item["implementation_id"],
                "name": item["name"],
                "source_action_names": list(item.get("source_action_names", ())),
            })

    omitted_selectable = [
        row for row in missing_catalog if row["upstream_selectable"]
    ]
    executable_ids = [
        row["implementation_id"] for row in mechanisms
        if row["bit_exact_cold_probe_passed"]
    ]
    return {
        "source_gate_schema": gate["schema"],
        "pdf": gate["pdf"],
        "trace_entry_count": gate["counts"]["trace_entries"],
        "mechanism_count": gate["counts"]["mechanisms"],
        "cataloged_mechanism_count": sum(row["cataloged"] for row in mechanisms),
        "upstream_selectable_mechanism_count": sum(
            row["upstream_selectable"] for row in mechanisms
        ),
        "represented_selectable_mechanism_count": len(represented_selectable),
        "omitted_selectable_mechanism_count": len(omitted_selectable),
        "represented_selectable_mechanisms": represented_selectable,
        "omitted_selectable_mechanisms": omitted_selectable,
        "missing_catalog_mechanisms": missing_catalog,
        "probe_passed_mechanism_count": len(executable_ids),
        "probe_passed_mechanism_ids": executable_ids,
        "probe_failed_mechanism_count": len(failed_probes),
        "probe_failed_mechanisms": failed_probes,
        "policy_reachability_blockers": policy_blocked,
        "upstream_blocked_mechanisms": upstream_blocked,
        "upstream_formal_training_gate_passed": gate["formal_training_gate_passed"],
        "exhaustive_bit_exact_probe_receipt": (
            len(executable_ids) == gate["counts"]["mechanisms"]
        ),
        "probe_schema": probe["schema"],
        "probe_registry_fingerprint": probe["registry_fingerprint"],
        "non_atomic_identity_layers": non_atomic,
        "coverage_claim": (
            "A mechanism is covered only after at least one fixed configuration "
            "executes through typed_graph_runtime, produces a complete TGF1 frame, "
            "and restores the exact root in-process and in a fresh interpreter. "
            "Profiles, recipes and adapters remain trace identities, not atomic actions."
        ),
        "mechanisms": mechanisms,
    }


def catalog_report() -> dict[str, Any]:
    """Return exact bounded coverage and make the formal-training gate explicit."""
    represented = {row.runtime_action for row in ACTION_CATALOG}
    unsupported = []
    for descriptor in taxonomy.ACTIONS:
        names = {descriptor.name, descriptor.mechanism, *descriptor.io_variants,
                 *descriptor.aliases}
        if names & (represented | set(_INTERNAL_ADAPTER_RUNTIME_ACTIONS)):
            continue
        if descriptor.kind == "retired":
            reason = "retired_current_slot"
        elif descriptor.kind == "model_provider":
            reason = ("complete_graph_entropy_path_not_cold_decoded"
                      if descriptor.name in prediction_entropy_codec.ACTION_TO_PROVIDER
                      else "probability_provider_without_complete_graph_entropy_edge")
        elif descriptor.kind == "special":
            reason = "special_parameterization_not_in_bounded_v1_catalog"
        else:
            reason = "registered_descriptor_not_in_bounded_raw_input_catalog"
        unsupported.append({
            "graph_action_id": descriptor.action_id,
            "name": descriptor.name,
            "kind": descriptor.kind,
            "operator_id": descriptor.operator_id,
            "reason": reason,
        })
    capability = prediction_entropy_codec.availability()
    nncp_capability = nncp_entropy_codec.availability()
    mechanisms = _mechanism_coverage()
    formal_blockers: list[dict[str, Any]] = []
    if not mechanisms["pdf"].get("matches_expected"):
        formal_blockers.append({
            "kind": "operator_pdf_fingerprint",
            "reason": "the configured 102-entry PDF is missing or its SHA-256 does not match",
        })
    if mechanisms["upstream_blocked_mechanisms"]:
        formal_blockers.append({
            "kind": "upstream_operator_gate",
            "count": len(mechanisms["upstream_blocked_mechanisms"]),
            "implementation_ids": [
                row["implementation_id"]
                for row in mechanisms["upstream_blocked_mechanisms"]
            ],
            "reason": "the PDF-aligned operator gate marks these mechanisms not selectable",
        })
    if mechanisms["omitted_selectable_mechanisms"]:
        formal_blockers.append({
            "kind": "adapter_fixed_catalog_coverage",
            "count": len(mechanisms["omitted_selectable_mechanisms"]),
            "implementation_ids": [
                row["implementation_id"]
                for row in mechanisms["omitted_selectable_mechanisms"]
            ],
            "reason": "selectable mechanisms have no fixed action configuration in adapter v1",
        })
    if mechanisms["probe_failed_mechanisms"]:
        formal_blockers.append({
            "kind": "adapter_exhaustive_probe",
            "count": len(mechanisms["probe_failed_mechanisms"]),
            "implementation_ids": [
                row["implementation_id"]
                for row in mechanisms["probe_failed_mechanisms"]
            ],
            "reason": (
                "these mechanisms have no fixed configuration with a successful "
                "typed-graph bit-exact cold probe"
            ),
        })
    if mechanisms["policy_reachability_blockers"]:
        formal_blockers.append({
            "kind": "adapter_policy_reachability",
            "count": len(mechanisms["policy_reachability_blockers"]),
            "implementation_ids": [
                row["implementation_id"]
                for row in mechanisms["policy_reachability_blockers"]
            ],
            "reason": (
                "the mechanisms execute only behind a conditional or non-atomic "
                "producer that a raw-column policy cannot currently construct"
            ),
        })
    return {
        "schema": CATALOG_SCHEMA,
        "state_feature_schema": STATE_FEATURE_SCHEMA,
        "state_feature_dim": STATE_FEATURE_DIM,
        "registry_fingerprint": REGISTRY_FINGERPRINT,
        "typed_graph_registry": graph_runtime.CURRENT_GRAPH_REGISTRY,
        "operator_directory_registry": operator_implementation_catalog.REGISTRY_VERSION,
        "expected_operator_pdf_sha256": EXPECTED_OPERATOR_PDF_SHA256,
        "fixed_policy_action_count": ACTION_COUNT,
        "stop_action": STOP_ACTION,
        "continuation_edit_action_space": {
            "type_action_ids": dict(EDIT_TYPE_ACTIONS),
            "position_action_id_range": [
                min(EDIT_POSITION_ACTIONS.values()),
                max(EDIT_POSITION_ACTIONS.values()),
            ],
            "position_range": [
                min(EDIT_POSITION_ACTIONS), max(EDIT_POSITION_ACTIONS),
            ],
            "first_block_max_completed_edits": MAX_GRAPH_NODES,
            "continuation_max_completed_edits": 2,
            "protocol": (
                "edit_type -> edit_position -> fixed operator/parameter action; "
                "delete completes after edit_position"
            ),
            "parameter_change_identity": (
                "a fixed catalog action for the same mechanism, runtime action and "
                "target slot with a different parameter tuple"
            ),
            "replay_contract": (
                "after every completed edit the entire edited recipe is replayed "
                "from the original root data flows before it becomes a legal state"
            ),
        },
        "represented_runtime_actions": sorted(represented - {"stop"}),
        "internal_non_policy_adapters": [
            {
                "consumer_runtime_action": consumer,
                "adapter_runtime_action": adapter,
                "required_policy_parent_runtime_actions": list(parents),
            }
            for consumer, (adapter, parents) in sorted(
                _INTERNAL_PARENT_ADAPTERS.items()
            )
        ],
        "unsupported_v7_descriptors": unsupported,
        "prediction_entropy_capability": capability,
        "nncp_complete_entropy_capability": nncp_capability,
        "mechanism_coverage": mechanisms,
        "formal_training_ready": not formal_blockers,
        "formal_training_blockers": formal_blockers,
        "direct_bridge_constraints": {
            "A006-A008": (
                "Every fixed action carries an explicit max_point in [0,18]. "
                "The value is part of the action identity, is passed to the current "
                "UAR policy selector without consulting future blocks, and is exposed "
                "only when the current data flow encodes exactly. The resulting UAR1 "
                "packet records the effective scale and is restored by the ordinary "
                "independent TGF1 cold decoder."
            ),
            "A035": (
                "Raw 64-bit columns use the explicit A031 adjacent-XOR semantic-port "
                "edge. It exposes first and xor_residuals as separate leaves; A035 is "
                "legal only on the latter and does not own or duplicate the anchor."
            ),
            "A111": "Selectable only for a Numeric width-4 uint32 data flow and a cataloged b.",
            "A106-A108": (
                "Selectable only when the PEC1 C++ binary and current cold-decode validation "
                "receipt match. The probability vector itself is never scored as compressed bytes."
            ),
            "A109": (
                "A fixed cmix action is cataloged, but its legal mask remains false unless "
                "allow_large_memory is explicit and a matching large-memory cold-decode "
                "validation receipt exists."
            ),
            "A110": (
                "A fixed action represents the complete NNP1 terminal entropy path, not an "
                "atomic probability-prediction edge. Its legal mask requires matching standalone "
                "and typed-graph cold-decode receipts."
            ),
        },
        "claim_boundary": (
            "The PDF's 102 trace entries, 67 retained mechanisms and this fixed policy "
            "catalog are different counts. Omitted descriptors never fall back to a "
            "different compression implementation."
        ),
    }


def assert_formal_training_ready() -> dict[str, Any]:
    """Fail closed unless the full 67-mechanism execution gate is satisfied."""
    report = catalog_report()
    if not report["formal_training_ready"]:
        coverage = report["mechanism_coverage"]
        omitted = [row["implementation_id"]
                   for row in coverage["omitted_selectable_mechanisms"]]
        blocked = [row["implementation_id"]
                   for row in coverage["upstream_blocked_mechanisms"]]
        probe_failed = [row["implementation_id"]
                        for row in coverage["probe_failed_mechanisms"]]
        unreachable = [row["implementation_id"]
                       for row in coverage["policy_reachability_blockers"]]
        raise GraphTaskError(
            "formal-training readiness gate failed: "
            f"omitted_selectable={omitted}, upstream_blocked={blocked}, "
            f"probe_failed={probe_failed}, policy_unreachable={unreachable}"
        )
    return report


def find_action(
    key_prefix: str,
    *,
    target_slot: int = 0,
    input_view: str | None = None,
    parameter_value: int | None = None,
    parameter_items: Sequence[tuple[str, int | bool]] | None = None,
    max_point: int | None = None,
    io_variant: str | None = None,
    other_slot: int | None = None,
) -> int:
    """Resolve one fixed action for orchestration, tests and recipe replay."""
    matches = [row for row in ACTION_CATALOG
               if row.target_slot == target_slot
               and (row.key == key_prefix or row.key.startswith(key_prefix + "[")
                    or row.runtime_action == key_prefix)
               and (input_view is None or row.input_view == input_view)
               and (parameter_value is None or row.parameter_value == parameter_value)
               and (parameter_items is None
                    or row.parameter_items == tuple(parameter_items))
               and (max_point is None or row.max_point == max_point)
               and (io_variant is None or row.io_variant == io_variant)
               and (other_slot is None or row.other_slot == other_slot)]
    if len(matches) != 1:
        raise KeyError(f"action lookup is not unique: {key_prefix!r}, matches={len(matches)}")
    return matches[0].action_id


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root_records(streams: Sequence[graph_runtime.Stream], views: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"kind": stream.kind, "width": stream.width, "size": len(stream.data),
         "sha256": _sha256_bytes(stream.data), "view": view}
        for stream, view in zip(streams, views)
    ]


def _infer_view(stream: graph_runtime.Stream) -> str:
    return "bytes"


def _validate_view(stream: graph_runtime.Stream, view: str) -> None:
    allowed = {
        "bytes",
        "int8", "uint8", "int16", "uint16", "int32", "uint32",
        "int64", "uint64", "float16", "float32", "float64",
        "xor_residual32", "xor_residual64",
    }
    if view not in allowed:
        raise ValueError(f"unsupported root view {view!r}")
    expected = {
        "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
        "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
        "float16": 2, "float32": 4, "float64": 8,
        "xor_residual32": 4, "xor_residual64": 8,
    }.get(view)
    if expected is not None and (stream.kind != "numeric" or stream.width != expected):
        raise ValueError(f"{view} requires a Numeric width-{expected} root")


def _numeric_samples(streams: Sequence[graph_runtime.Stream], views: Sequence[str]) -> np.ndarray:
    values: list[np.ndarray] = []
    dtypes = {
        "int8": "<i1", "uint8": "<u1", "int16": "<i2", "uint16": "<u2",
        "int32": "<i4", "uint32": "<u4", "int64": "<i8", "uint64": "<u8",
        "float16": "<f2", "float32": "<f4", "float64": "<f8",
    }
    for stream, view in zip(streams, views):
        dtype = dtypes.get(view)
        if dtype is None or not stream.data:
            continue
        array = np.frombuffer(stream.data, dtype=dtype)
        if array.size > 8192:
            indices = np.linspace(0, array.size - 1, 8192, dtype=np.int64)
            array = array[indices]
        values.append(array.astype(np.float64, copy=False))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _task_features(streams: Sequence[graph_runtime.Stream], views: Sequence[str]) -> np.ndarray:
    raw = b"".join(stream.data[:65536] for stream in streams)[:65536]
    byte_values = np.frombuffer(raw, dtype=np.uint8) if raw else np.empty(0, dtype=np.uint8)
    if byte_values.size:
        counts = np.bincount(byte_values, minlength=256).astype(np.float64)
        probabilities = counts[counts > 0] / byte_values.size
        entropy = float(-np.sum(probabilities * np.log2(probabilities)) / 8.0)
        zero_bytes = float(np.mean(byte_values == 0))
        equal_bytes = float(np.mean(byte_values[1:] == byte_values[:-1])) if byte_values.size > 1 else 0.0
    else:
        entropy = zero_bytes = equal_bytes = 0.0
    numeric = _numeric_samples(streams, views)
    finite = numeric[np.isfinite(numeric)]
    if finite.size:
        zero_values = float(np.mean(finite == 0.0))
        unique_fraction = float(np.unique(finite[:8192]).size / min(finite.size, 8192))
        if finite.size > 1:
            scale = float(np.median(np.abs(finite))) + 1.0e-12
            smoothness = 1.0 / (1.0 + float(np.median(np.abs(np.diff(finite)))) / scale)
        else:
            smoothness = 1.0
    else:
        zero_values, unique_fraction, smoothness = 0.0, 1.0, 0.0
    total = sum(len(stream.data) for stream in streams)
    widths = np.asarray([stream.width for stream in streams], dtype=np.float64)
    kinds = [stream.kind for stream in streams]
    result = np.asarray([
        math.log2(total + 1.0) / 32.0,
        len(streams) / MAX_ACTIVE_STREAMS,
        float(np.mean(widths)) / 8.0,
        float(np.max(widths)) / 8.0,
        kinds.count("serial") / len(kinds),
        kinds.count("numeric") / len(kinds),
        kinds.count("struct") / len(kinds),
        views.count("int64") / len(views),
        views.count("float64") / len(views),
        views.count("float32") / len(views),
        views.count("uint32") / len(views),
        entropy,
        zero_bytes,
        equal_bytes,
        zero_values,
        1.0 - unique_fraction,
        smoothness,
        float(np.mean(~np.isfinite(numeric))) if numeric.size else 0.0,
    ], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise GraphTaskError("task feature extraction produced non-finite values")
    return result


class TypedGraphCompressionTask:
    """A real ``GraphCompressionTask`` backed by typed graph runtime v7."""

    action_count = ACTION_COUNT
    stop_action = STOP_ACTION
    registry_fingerprint = REGISTRY_FINGERPRINT

    def __init__(
        self,
        task_id: str,
        root_streams: Sequence[graph_runtime.Stream],
        *,
        root_views: Sequence[str] | None = None,
        compression_reference_ns_per_byte: float,
        decompression_reference_ns_per_byte: float,
        objective_weight: Sequence[float] | None = None,
        config: GraphTaskConfig = GraphTaskConfig(),
        previous_recipe: Sequence[int] = (),
        candidate_budget: CandidateBudget | None = None,
        block_index: int = 0,
        original_numeric_bytes: int | None = None,
        continuation_edit_mode: bool = False,
        max_recipe_edits: int = 2,
    ) -> None:
        if config.require_formal_training_ready:
            assert_formal_training_ready()
        self.task_id = str(task_id)
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        self._roots = tuple(root_streams)
        if not self._roots:
            raise ValueError("at least one root data flow is required")
        if len(self._roots) > config.max_active_streams:
            raise ValueError("root count exceeds active-data-flow limit")
        self._root_views = tuple(root_views or (_infer_view(stream) for stream in self._roots))
        if len(self._root_views) != len(self._roots):
            raise ValueError("one root view is required per root data flow")
        for stream, view in zip(self._roots, self._root_views):
            _validate_view(stream, view)
        all_root_bytes = sum(len(stream.data) for stream in self._roots)
        numeric_root_bytes = sum(
            len(stream.data) for stream in self._roots if stream.kind == "numeric"
        )
        if original_numeric_bytes is None:
            # Backward-compatible behavior for existing one-root tasks.  A
            # prepared block carrying a missing bitmap must pass the explicit
            # valid-value denominator below.
            self.original_numeric_bytes = all_root_bytes
            self._denominator_source = "legacy_all_root_bytes"
        else:
            if (isinstance(original_numeric_bytes, bool)
                    or not isinstance(original_numeric_bytes, int)
                    or original_numeric_bytes <= 0
                    or original_numeric_bytes > numeric_root_bytes):
                raise ValueError(
                    "original_numeric_bytes must be a positive integer no larger "
                    "than the bytes in numeric value roots"
                )
            self.original_numeric_bytes = original_numeric_bytes
            self._denominator_source = "explicit_valid_numeric_value_bytes"
        if self.original_numeric_bytes <= 0:
            raise ValueError("GRPO tasks require positive original numeric bytes")
        refs = (float(compression_reference_ns_per_byte),
                float(decompression_reference_ns_per_byte))
        if any(not math.isfinite(value) or value <= 0.0 for value in refs):
            raise ValueError("time references must be finite and positive")
        self.compression_reference_ns_per_byte = refs[0]
        self.decompression_reference_ns_per_byte = refs[1]
        if objective_weight is None:
            self.objective_weight: tuple[float, float, float] | None = None
        else:
            if len(objective_weight) != 3:
                raise ValueError("objective_weight must contain exactly three values")
            weight = tuple(float(value) for value in objective_weight)
            if (any(not math.isfinite(value) or value < 0.0 for value in weight)
                    or not math.isclose(sum(weight), 1.0, rel_tol=0.0, abs_tol=1.0e-12)):
                raise ValueError("objective_weight must be finite, non-negative, and sum to one")
            self.objective_weight = weight
        self.config = config
        if type(continuation_edit_mode) is not bool:
            raise ValueError("continuation_edit_mode must be boolean")
        if (isinstance(max_recipe_edits, bool) or not isinstance(max_recipe_edits, int)
                or not 1 <= max_recipe_edits <= config.max_graph_nodes):
            raise ValueError(
                "max_recipe_edits must be in [1,max_graph_nodes]; production "
                "continuation blocks separately enforce the two-edit limit"
            )
        self._continuation_edit_mode = continuation_edit_mode
        self._max_recipe_edits = max_recipe_edits
        self._edited_recipe_cache: dict[tuple[Any, ...], TaskState | None] = {}
        self._edit_prefix_cache: dict[
            tuple[tuple[int, ...], int], TaskState | None
        ] = {}
        self._prediction_capability = prediction_entropy_codec.availability()
        self._nncp_capability = nncp_entropy_codec.availability()
        self._max_point_graph_cache: dict[
            tuple[tuple[int, ...], int], graph_runtime.GraphState | None
        ] = {}
        self._measurement_lock = threading.Lock()
        self._current_objective_by_recipe: dict[tuple[int, ...], float] = {}
        self._measured_action_costs: dict[
            int, list[tuple[float, float, float]]
        ] = {}
        probe_report = mechanism_probe_report()
        self._passed_action_probe_keys = frozenset(
            probe_report["passed_action_probe_keys"]
        )
        self._probe_output_counts = {
            row["probe_key"]: int(row.get("output_flow_count", 1))
            for mechanism in probe_report["mechanisms"]
            for row in mechanism["action_probes"]
            if row["passed"]
        }
        self.block_index = int(block_index)
        if self.block_index < 0:
            raise ValueError("block_index must be non-negative")
        default_budget = (FIRST_BLOCK_CANDIDATE_BUDGET if self.block_index == 0
                          else CONTINUATION_CANDIDATE_BUDGET)
        # Training calls the same task over many GRPO iterations. Its default
        # budget is therefore an accounting threshold rather than a lifetime
        # hard stop. Frozen first-block search should pass CandidateBudget(64)
        # explicitly; continuation_task installs a hard 16-candidate budget.
        self.candidate_budget = candidate_budget or CandidateBudget(
            default_budget, hard_limit=False
        )
        self._static_features = _task_features(self._roots, self._root_views)
        root_records = _root_records(self._roots, self._root_views)
        self.data_fingerprint = hashlib.sha256(_canonical_json(root_records)).hexdigest()
        self._previous_recipe = tuple(int(action) for action in previous_recipe)
        if any(action == STOP_ACTION or not 0 <= action < ACTION_COUNT
               or ACTION_CATALOG[action].category not in {
                   "mechanism", "split_segments", "concat"
               }
               for action in self._previous_recipe):
            raise ValueError(
                "previous recipe contains STOP, an edit token, or an unknown action"
            )
        self._root_state = self._replay_root(self._previous_recipe)
        self.root_fingerprint = hashlib.sha256(_canonical_json({
            "data": self.data_fingerprint,
            "previous_recipe": list(self._previous_recipe),
            "original_numeric_bytes": self.original_numeric_bytes,
            "objective_weight": (
                list(self.objective_weight) if self.objective_weight is not None else None
            ),
            "continuation_edit_mode": self._continuation_edit_mode,
            "max_recipe_edits": self._max_recipe_edits,
            "limits": asdict(config),
        })).hexdigest()
        self.max_steps = (
            self._max_recipe_edits * 3 + 1
            if self._continuation_edit_mode else config.max_policy_edits + 1
        )

    @property
    def previous_recipe(self) -> tuple[int, ...]:
        return self._previous_recipe

    def task_features(self) -> np.ndarray:
        return self._static_features.copy()

    def initial_state(self) -> TaskState:
        return self._root_state

    def reset_evaluation_budget(self, limit: int | None = None) -> None:
        self.candidate_budget.reset(limit)

    def action_record(self, action: int) -> ActionSpec:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        if not 0 <= int(action) < ACTION_COUNT:
            raise IndexError("unknown fixed graph action")
        return ACTION_CATALOG[int(action)]

    def state_features(self, state: TaskState) -> np.ndarray:
        self._require_state(state)
        streams = list(state.graph.streams.values())
        depths = state.depths()
        parents = state.parents()
        sizes = np.asarray([len(stream.data) for stream in streams], dtype=np.float64)
        depth_values = np.asarray([depths[key] for key in state.graph.streams], dtype=np.float64)
        used, limit = self.candidate_budget.snapshot()
        kinds = [stream.kind for stream in streams]
        raw = self.original_numeric_bytes
        edit_limit = (
            self._max_recipe_edits if state.continuation_edit_mode
            else self.config.max_policy_edits
        )
        pending_kind_code = {
            None: 0.0, "insert": 0.25, "delete": 0.5,
            "replace": 0.75, "parameter_change": 1.0,
        }[state.pending_edit_kind]
        with self._measurement_lock:
            current_objective = self._current_objective_by_recipe.get(state.recipe)
        # -1 is an explicit "not measured on this block" sentinel.  A measured
        # J is non-negative and is bounded to [0,1) for stable policy input.
        current_objective_feature = (
            -1.0 if current_objective is None
            else current_objective / (1.0 + current_objective)
        )
        parent_buckets = np.zeros(FLOW_FEATURE_BUCKETS, dtype=np.float64)
        byte_buckets = np.zeros(FLOW_FEATURE_BUCKETS, dtype=np.float64)
        # Stable graph stream order gives every active data flow one deterministic
        # parent/operator and byte contribution without changing the frozen
        # 18+28+3=49 observation width.  Matching bucket indices retain the
        # parent/byte association; collisions are deterministic sums.
        for index, (stream_id, stream) in enumerate(state.graph.streams.items()):
            bucket = index % FLOW_FEATURE_BUCKETS
            parent_buckets[bucket] += (
                (parents[stream_id] + 1) / max(ACTION_COUNT, 1)
            ) / len(streams)
            byte_buckets[bucket] += math.log1p(len(stream.data) / raw)
        scalar_features = [
            state.node_count / self.config.max_graph_nodes,
            float(np.max(depth_values)) / self.config.max_ordinary_depth,
            float(np.mean(depth_values)) / self.config.max_ordinary_depth,
            len(streams) / self.config.max_active_streams,
            float(np.sum(sizes)) / raw,
            kinds.count("serial") / len(kinds),
            kinds.count("numeric") / len(kinds),
            float(np.mean([stream.width for stream in streams])) / 8.0,
            state.policy_edits / edit_limit,
            min(used, limit) / limit,
            state.replayed_nodes / self.config.max_graph_nodes,
            float(state.continuation_edit_mode),
            state.edit_count / self._max_recipe_edits,
            pending_kind_code,
            ((state.pending_edit_index + 1) / (MAX_GRAPH_NODES + 1)
             if state.pending_edit_index is not None else 0.0),
            current_objective_feature,
        ]
        result = np.asarray(
            scalar_features + parent_buckets.tolist() + byte_buckets.tolist(),
            dtype=np.float64,
        )
        if result.shape != (STATE_FEATURE_DIM,):
            raise GraphTaskError("state feature schema width changed unexpectedly")
        if not np.all(np.isfinite(result)):
            raise GraphTaskError("state feature extraction produced non-finite values")
        return result

    def measured_action_cost(
        self, action: int
    ) -> tuple[float, float, float] | None:
        """Median measured rho/E/D for complete recipes containing ``action``."""
        self.action_record(action)
        with self._measurement_lock:
            rows = tuple(self._measured_action_costs.get(int(action), ()))
        if not rows:
            return None
        return tuple(
            float(statistics.median(row[index] for row in rows))
            for index in range(3)
        )  # type: ignore[return-value]

    def action_measurement_count(self, action: int) -> int:
        self.action_record(action)
        with self._measurement_lock:
            return len(self._measured_action_costs.get(int(action), ()))

    def ordering_features(self) -> dict[str, float]:
        """Current-block features consumed only by heuristic branch ordering."""
        floating = sum(
            view.startswith("float") for view in self._root_views
        ) / len(self._root_views)
        return {
            "smoothness": float(self._static_features[16]),
            "repeat_fraction": float(max(
                self._static_features[13], self._static_features[15]
            )),
            "floating_fraction": float(floating),
        }

    def certified_bound_state(self, state: TaskState) -> BoundState:
        """Return a conservative suffix-safe byte/ratio lower bound.

        Future operators may shrink every current leaf, so no live payload byte
        is counted.  Roots, root order and registry version survive every edit.
        Executed steps are immutable only in append-only mode: recipe edits may
        delete or replace them, so their metadata contributes no editing bound.
        A real terminal frame also needs a non-empty leaf table, hence using
        ``leaves=[]`` is conservative.
        """
        self._require_state(state)
        header = {
            "schema": "typed-graph-frame-v1",
            "roots": state.graph.roots,
            "root_order": list(state.graph.roots),
            "registry_version": state.graph.registry_version,
            "steps": [] if state.continuation_edit_mode else state.graph.steps,
            "leaves": [],
        }
        raw = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return BoundState(
            original_numeric_bytes=self.original_numeric_bytes,
            fixed_frame_bytes=12 + 32,
            mandatory_metadata_bytes=len(raw),
            exhausted_leaf_logical_bits=(),
        )

    def legal_actions(self, state: TaskState) -> Sequence[int]:
        self._require_state(state)
        if (not state.continuation_edit_mode
                and state.policy_edits >= self.config.max_policy_edits):
            return (STOP_ACTION,)
        if (state.continuation_edit_mode
                and state.edit_count >= self._max_recipe_edits
                and state.pending_edit_kind is None):
            return (STOP_ACTION,)
        return tuple(row.action_id for row in ACTION_CATALOG if self._is_legal(state, row))

    def transition(self, state: TaskState, action: int) -> TaskState | None:
        self._require_state(state)
        spec = self.action_record(action)
        if spec.action_id == STOP_ACTION or not self._is_legal(state, spec):
            return None
        try:
            return self._apply(state, spec, count_policy_edit=True)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise GraphTaskError(
                f"fixed action {spec.key} passed its static legality mask but failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def evaluate_terminal(self, state: TaskState) -> TerminalOutcome:
        self._require_state(state)
        if state.pending_edit_kind is not None:
            raise GraphTaskError("cannot evaluate a continuation with an incomplete edit")
        ordinal = self.candidate_budget.reserve()
        compression_samples: list[int] = []
        decompression_samples: list[int] = []
        frame: bytes | None = None
        restored: dict[str, graph_runtime.Stream] | None = None
        for _ in range(self.config.timing_warmups):
            warm_graph = self._replay_recipe(state.recipe)
            warm_frame = warm_graph.graph.stop()
            if frame is None:
                frame = warm_frame
            elif warm_frame != frame:
                raise GraphTaskError("recipe warmup produced non-deterministic complete frames")
        for _ in range(self.config.timing_repetitions):
            started = time.perf_counter_ns()
            replayed = self._replay_recipe(state.recipe)
            candidate = replayed.graph.stop()
            compression_samples.append(time.perf_counter_ns() - started)
            if frame is None:
                frame = candidate
            elif candidate != frame:
                raise GraphTaskError("recipe replay produced non-deterministic complete frames")
        assert frame is not None
        for _ in range(self.config.timing_warmups):
            decoded = graph_runtime.restore_frame(
                frame, allow_large_memory=self.config.allow_large_memory
            )
            if restored is None:
                restored = decoded
        for _ in range(self.config.timing_repetitions):
            started = time.perf_counter_ns()
            decoded = graph_runtime.restore_frame(
                frame, allow_large_memory=self.config.allow_large_memory
            )
            decompression_samples.append(time.perf_counter_ns() - started)
            if restored is None:
                restored = decoded
        assert restored is not None
        self._verify_restored(restored)
        cold_validated = False
        if self.config.independent_cold_validation:
            self._independent_cold_validate(frame)
            cold_validated = True
        compression_ns = int(statistics.median(compression_samples))
        decompression_ns = int(statistics.median(decompression_samples))
        used, limit = self.candidate_budget.snapshot()
        graph_description = {
            "schema": SCHEMA,
            "registry_fingerprint": REGISTRY_FINGERPRINT,
            "typed_graph_registry": graph_runtime.CURRENT_GRAPH_REGISTRY,
            "recipe_action_ids": list(state.recipe),
            "recipe": [ACTION_CATALOG[action].json_record() for action in state.recipe],
            "continuation_edit_mode": state.continuation_edit_mode,
            "starting_recipe_action_ids": list(self._previous_recipe),
            "completed_recipe_edits": state.edit_count,
            "recipe_edit_limit": (
                self._max_recipe_edits if state.continuation_edit_mode else 0
            ),
            "first_block_unbounded_local_edit_mode": bool(
                state.continuation_edit_mode
                and self.block_index == 0
                and self._max_recipe_edits == self.config.max_graph_nodes
            ),
            "edit_history": [
                {"kind": kind, "position": position, "action_id": action_id}
                for kind, position, action_id in state.edit_history
            ],
            "ordinary_max_depth": max(state.depths().values()),
            "node_count": state.node_count,
            "policy_operator_node_count": state.node_count,
            "serialized_runtime_step_count": len(state.graph.steps),
            "internal_non_policy_adapter_steps": [
                step["action"] for step in state.graph.steps
                if step.get("action") in _INTERNAL_ADAPTER_RUNTIME_ACTIONS
            ],
            "active_data_flows": len(state.graph.streams),
            "candidate_ordinal": ordinal,
            "candidate_budget_used": used,
            "candidate_budget_limit": limit,
            "candidate_budget_hard_limit": self.candidate_budget.hard_limit,
            "measurement_protocol_schema": MEASUREMENT_PROTOCOL_SCHEMA,
            "timing_aggregation": "median",
            "timing_warmups": self.config.timing_warmups,
            "timing_repetitions": self.config.timing_repetitions,
            "compression_repeat_time_ns": list(compression_samples),
            "decompression_repeat_time_ns": list(decompression_samples),
            "independent_cold_validation": cold_validated,
            "allow_large_memory": self.config.allow_large_memory,
            "original_numeric_bytes": self.original_numeric_bytes,
            "objective_weight": (
                list(self.objective_weight) if self.objective_weight is not None else None
            ),
            "denominator_source": self._denominator_source,
            "complete_frame_includes_all_root_flows": True,
            "compression_timing_boundary": (
                "Python scheduling + native child-process operators + TGF1 framing; "
                "excludes source-file reads and policy inference"
            ),
            "decompression_timing_boundary": (
                "in-process frame-only typed_graph_runtime.restore_frame; independent "
                "fresh-interpreter validation is correctness-only and is not included"
            ),
            "cold_worker_launch_in_decompression_time": False,
        }
        outcome = TerminalOutcome(
            operator_graph=graph_description,
            compressed_payload=frame,
            metrics=CompressionMetrics(
                original_numeric_bytes=self.original_numeric_bytes,
                compression_time_ns=compression_ns,
                decompression_time_ns=decompression_ns,
                compression_reference_ns_per_byte=self.compression_reference_ns_per_byte,
                decompression_reference_ns_per_byte=self.decompression_reference_ns_per_byte,
            ),
        )
        rho = outcome.compression_ratio
        e_cost = outcome.metrics.compression_ns_per_byte
        d_cost = outcome.metrics.decompression_ns_per_byte
        with self._measurement_lock:
            if self.objective_weight is not None:
                self._current_objective_by_recipe[state.recipe] = objective_value(
                    outcome, self.objective_weight
                )
            for action in set(state.recipe):
                rows = self._measured_action_costs.setdefault(action, [])
                rows.append((rho, e_cost, d_cost))
                # One online search evaluates at most 64 complete candidates.
                # Bounding the history also prevents accidental cross-run growth
                # when a training task is reused for several GRPO iterations.
                if len(rows) > FIRST_BLOCK_CANDIDATE_BUDGET:
                    del rows[:-FIRST_BLOCK_CANDIDATE_BUDGET]
        return outcome

    def continuation_task(
        self,
        task_id: str,
        root_streams: Sequence[graph_runtime.Stream],
        *,
        root_views: Sequence[str] | None = None,
        previous_recipe: Sequence[int],
        max_local_edits: int = 2,
        previous_edits: Sequence[RecipeEdit] = (),
        original_numeric_bytes: int | None = None,
    ) -> "TypedGraphCompressionTask":
        """Create a later-block task rooted at a replayed/locally edited recipe."""
        edited_recipe = self.edit_recipe(
            previous_recipe, previous_edits, max_edits=max_local_edits
        )
        config = GraphTaskConfig(
            max_ordinary_depth=self.config.max_ordinary_depth,
            max_graph_nodes=self.config.max_graph_nodes,
            max_active_streams=self.config.max_active_streams,
            max_policy_edits=max_local_edits,
            timing_warmups=self.config.timing_warmups,
            timing_repetitions=self.config.timing_repetitions,
            independent_cold_validation=self.config.independent_cold_validation,
            cold_validation_timeout_seconds=self.config.cold_validation_timeout_seconds,
            allow_large_memory=self.config.allow_large_memory,
            require_formal_training_ready=self.config.require_formal_training_ready,
        )
        return TypedGraphCompressionTask(
            task_id, root_streams, root_views=root_views,
            compression_reference_ns_per_byte=self.compression_reference_ns_per_byte,
            decompression_reference_ns_per_byte=self.decompression_reference_ns_per_byte,
            objective_weight=self.objective_weight,
            config=config, previous_recipe=edited_recipe,
            candidate_budget=CandidateBudget(CONTINUATION_CANDIDATE_BUDGET),
            block_index=self.block_index + 1,
            original_numeric_bytes=original_numeric_bytes,
            continuation_edit_mode=True,
            max_recipe_edits=max_local_edits,
        )

    @staticmethod
    def edit_recipe(
        recipe: Sequence[int], edits: Sequence[RecipeEdit], *, max_edits: int = 2
    ) -> tuple[int, ...]:
        """Apply at most two structural/parameter edits before replay.

        ``parameter_change`` is represented by replacing one fixed action with
        another action for the same mechanism/runtime action but a different
        fixed parameter tuple.  Legality of the edited graph is checked by the
        normal replay path on the new block.
        """
        if isinstance(max_edits, bool) or not isinstance(max_edits, int):
            raise ValueError("max_edits must be an integer")
        if not 0 <= max_edits <= 2:
            raise ValueError("continuation recipe edits are bounded to [0,2]")
        if len(edits) > max_edits:
            raise ValueError("continuation edit list exceeds its bounded budget")
        result = [int(action) for action in recipe]
        if any(action == STOP_ACTION or not 0 <= action < ACTION_COUNT
               or ACTION_CATALOG[action].category not in {
                   "mechanism", "split_segments", "concat"
               }
               for action in result):
            raise ValueError("recipe contains STOP, an edit token, or an unknown action")
        for edit in edits:
            if not isinstance(edit, RecipeEdit):
                raise TypeError("previous_edits must contain RecipeEdit values")
            if edit.kind == "insert":
                if not 0 <= edit.index <= len(result) or edit.action_id is None:
                    raise ValueError("invalid insert edit")
                replacement = int(edit.action_id)
                if (replacement == STOP_ACTION or not 0 <= replacement < ACTION_COUNT
                        or ACTION_CATALOG[replacement].category not in {
                            "mechanism", "split_segments", "concat"
                        }):
                    raise ValueError("insert edit references an unknown action")
                result.insert(edit.index, replacement)
            elif edit.kind == "delete":
                if not 0 <= edit.index < len(result) or edit.action_id is not None:
                    raise ValueError("invalid delete edit")
                result.pop(edit.index)
            elif edit.kind in {"replace", "parameter_change"}:
                if not 0 <= edit.index < len(result) or edit.action_id is None:
                    raise ValueError(f"invalid {edit.kind} edit")
                replacement = int(edit.action_id)
                if (replacement == STOP_ACTION or not 0 <= replacement < ACTION_COUNT
                        or ACTION_CATALOG[replacement].category not in {
                            "mechanism", "split_segments", "concat"
                        }):
                    raise ValueError(f"{edit.kind} edit references an unknown action")
                if edit.kind == "parameter_change":
                    old = ACTION_CATALOG[result[edit.index]]
                    new = ACTION_CATALOG[replacement]
                    if (old.implementation_id, old.runtime_action) != (
                        new.implementation_id, new.runtime_action
                    ):
                        raise ValueError(
                            "parameter_change must keep the mechanism and runtime action"
                        )
                    if (_probe_key(old) == _probe_key(new)):
                        raise ValueError("parameter_change must select a different fixed tuple")
                result[edit.index] = replacement
            else:
                raise ValueError(f"unknown continuation edit kind {edit.kind!r}")
        return tuple(result)

    def _require_state(self, state: TaskState) -> None:
        if not isinstance(state, TaskState):
            raise TypeError("expected TaskState")
        if state.graph.registry_version != graph_runtime.CURRENT_GRAPH_REGISTRY:
            raise GraphTaskError("task state graph registry changed")
        if state.continuation_edit_mode != self._continuation_edit_mode:
            raise GraphTaskError("task state continuation mode changed")

    def _empty_state(self) -> TaskState:
        graph = graph_runtime.GraphState.from_streams(
            self._roots, registry_version=graph_runtime.CURRENT_GRAPH_REGISTRY
        )
        ids = tuple(graph.streams)
        return TaskState(
            graph=graph,
            leaf_depths=tuple((key, 0) for key in ids),
            leaf_views=tuple(zip(ids, self._root_views)),
            parent_actions=tuple((key, STOP_ACTION) for key in ids),
            recipe=(), node_count=0, policy_edits=0, replayed_nodes=0,
            continuation_edit_mode=self._continuation_edit_mode,
        )

    def _replay_root(self, recipe: Sequence[int]) -> TaskState:
        state = self._empty_state()
        for action in recipe:
            spec = self.action_record(action)
            if not self._is_legal(
                state, spec, ignore_policy_edit_limit=True, replay_mode=True
            ):
                raise GraphTaskError(f"previous recipe is illegal at {spec.key}")
            state = self._apply_graph_operator(
                state, spec, count_policy_edit=False
            )
        return replace(
            state, policy_edits=0, replayed_nodes=state.node_count,
            continuation_edit_mode=self._continuation_edit_mode,
            pending_edit_kind=None, pending_edit_index=None, edit_count=0,
            edit_history=(),
        )

    def _replay_recipe(self, recipe: Sequence[int]) -> TaskState:
        state = self._empty_state()
        for action in recipe:
            spec = self.action_record(action)
            if not self._is_legal(
                state, spec, ignore_policy_edit_limit=True, replay_mode=True
            ):
                raise GraphTaskError(f"stored recipe became illegal at {spec.key}")
            state = self._apply_graph_operator(
                state, spec, count_policy_edit=False
            )
        return state

    def _parameter_change_candidates(self, action_id: int) -> tuple[int, ...]:
        old = ACTION_CATALOG[action_id]
        if old.implementation_id is None:
            return ()
        return tuple(
            row.action_id for row in ACTION_CATALOG
            if row.category == "mechanism"
            and row.implementation_id == old.implementation_id
            and row.runtime_action == old.runtime_action
            and row.target_slot == old.target_slot
            and _probe_key(row) != _probe_key(old)
        )

    def _edited_candidate_state(
        self, state: TaskState, kind: str, index: int,
        action_id: int | None,
    ) -> TaskState | None:
        key = (
            state.recipe, state.edit_count, state.edit_history,
            kind, index, action_id,
        )
        if key in self._edited_recipe_cache:
            return self._edited_recipe_cache[key]
        try:
            edited = self.edit_recipe(
                state.recipe, (RecipeEdit(kind, index, action_id),), max_edits=1
            )
            if len(edited) > self.config.max_graph_nodes:
                raise GraphTaskError("edited recipe exceeds graph-node budget")
            replayed = self._replay_recipe(edited)
            result = replace(
                replayed,
                policy_edits=state.policy_edits + 1,
                replayed_nodes=replayed.node_count,
                continuation_edit_mode=True,
                pending_edit_kind=None,
                pending_edit_index=None,
                edit_count=state.edit_count + 1,
                edit_history=state.edit_history + ((kind, index, action_id),),
            )
        except (OSError, RuntimeError, ValueError, TypeError):
            result = None
        self._edited_recipe_cache[key] = result
        return result

    def _edit_prefix_state(
        self, recipe: tuple[int, ...], index: int
    ) -> TaskState | None:
        key = (recipe, index)
        if key not in self._edit_prefix_cache:
            try:
                self._edit_prefix_cache[key] = self._replay_recipe(recipe[:index])
            except (OSError, RuntimeError, ValueError, TypeError):
                self._edit_prefix_cache[key] = None
        return self._edit_prefix_cache[key]

    def _is_legal(
        self, state: TaskState, spec: ActionSpec, *,
        ignore_policy_edit_limit: bool = False,
        replay_mode: bool = False,
    ) -> bool:
        if replay_mode or not state.continuation_edit_mode:
            if spec.category in {"edit_type", "edit_position"}:
                return False
            return self._is_graph_action_legal(
                state, spec, ignore_policy_edit_limit=ignore_policy_edit_limit
            )

        if spec.action_id == STOP_ACTION:
            return state.pending_edit_kind is None
        if state.edit_count >= self._max_recipe_edits:
            return False
        if spec.category == "edit_type":
            if state.pending_edit_kind is not None:
                return False
            if spec.edit_kind == "insert":
                return len(state.recipe) < self.config.max_graph_nodes
            if spec.edit_kind in {"delete", "replace"}:
                return bool(state.recipe)
            if spec.edit_kind == "parameter_change":
                return any(
                    self._parameter_change_candidates(action)
                    for action in state.recipe
                )
            return False
        if spec.category == "edit_position":
            if (state.pending_edit_kind is None
                    or state.pending_edit_index is not None
                    or spec.parameter_value is None):
                return False
            position = int(spec.parameter_value)
            kind = state.pending_edit_kind
            if kind == "insert":
                return 0 <= position <= len(state.recipe)
            if not 0 <= position < len(state.recipe):
                return False
            if kind == "parameter_change":
                return bool(self._parameter_change_candidates(state.recipe[position]))
            if kind == "delete":
                return self._edited_candidate_state(
                    state, kind, position, None
                ) is not None
            return kind == "replace"
        if spec.category not in {"mechanism", "split_segments", "concat"}:
            return False
        if (state.pending_edit_kind not in {
                "insert", "replace", "parameter_change"
            } or state.pending_edit_index is None):
            return False
        if state.pending_edit_kind == "parameter_change":
            old_action = state.recipe[state.pending_edit_index]
            if spec.action_id not in self._parameter_change_candidates(old_action):
                return False
        prefix = self._edit_prefix_state(state.recipe, state.pending_edit_index)
        if (prefix is None
                or not self._is_graph_action_legal(
                    prefix, spec, ignore_policy_edit_limit=True
                )):
            return False
        return self._edited_candidate_state(
            state, state.pending_edit_kind, state.pending_edit_index,
            spec.action_id,
        ) is not None

    def _is_graph_action_legal(
        self, state: TaskState, spec: ActionSpec, *,
        ignore_policy_edit_limit: bool = False,
    ) -> bool:
        if spec.action_id == STOP_ACTION:
            return True
        if (spec.implementation_id is not None
                and _probe_key(spec) not in self._passed_action_probe_keys):
            return False
        if (state.node_count >= self.config.max_graph_nodes
                or (not ignore_policy_edit_limit
                    and state.policy_edits >= self.config.max_policy_edits)):
            return False
        ids = tuple(state.graph.streams)
        if spec.target_slot is None or spec.target_slot >= len(ids):
            return False
        target_id = ids[spec.target_slot]
        target = state.graph.streams[target_id]
        depths = state.depths()
        views = state.views()
        parents = state.parents()
        if spec.input_nonempty and not target.data:
            return False
        if len(target.data) % spec.input_byte_multiple:
            return False
        if (spec.required_nonserial_width is not None and target.kind != "serial"
                and target.width != spec.required_nonserial_width):
            return False
        if spec.allowed_physical_widths and target.width not in spec.allowed_physical_widths:
            return False
        if spec.required_logical_views:
            logical_view = views[target_id]
            if "physical_word" in spec.required_logical_views:
                if not logical_view.startswith("xor_residual"):
                    return False
            elif logical_view not in spec.required_logical_views:
                return False
        if spec.runtime_action == "concat":
            if spec.other_slot is None or spec.other_slot >= len(ids):
                return False
            other = state.graph.streams[ids[spec.other_slot]]
            return (target.kind, target.width) == (other.kind, other.width)
        resulting_depth = depths[target_id] + int(spec.ordinary_operator)
        if resulting_depth > self.config.max_ordinary_depth:
            return False
        if spec.input_view in {"int64", "float64", "float32", "uint32"}:
            if views[target_id] != spec.input_view:
                return False
        if (spec.runtime_action == "op.bitpack_int.max_value_width.cpp_v1"
                and spec.input_view == "int64"
                and np.any(np.frombuffer(target.data, dtype="<i8") < 0)):
            return False
        if spec.input_view == "physical_word" and not (
            target.kind == "numeric" and target.width in (4, 8)
            and views[target_id] == f"xor_residual{target.width * 8}"
        ):
            return False
        if spec.required_parent_runtime_actions:
            parent_action = ACTION_CATALOG[parents[target_id]].runtime_action
            if parent_action not in spec.required_parent_runtime_actions:
                adapter = _INTERNAL_PARENT_ADAPTERS.get(spec.runtime_action)
                if (adapter is None
                        or adapter[0] not in spec.required_parent_runtime_actions
                        or (adapter[1] and parent_action not in adapter[1])):
                    return False
        if spec.runtime_action == exception_separation.ALGORITHM_NAME and not (
            target.kind == "numeric" and target.width == 4
        ):
            return False
        if spec.runtime_action == "float_deconstruct" and not (
            target.kind == "numeric" and target.width == 4
            and views[target_id] == "float32"
        ):
            return False
        if spec.runtime_action == "columnize" and target.width <= 1:
            return False
        if spec.runtime_action == "bitsplit" and target.width <= 1:
            return False
        if spec.runtime_action == "bitunpack" and target.kind != "serial":
            return False
        if spec.runtime_action == "split_segments" and len(target.data) // target.width < 2:
            return False
        if spec.runtime_action in prediction_entropy_codec.ACTION_TO_PROVIDER:
            provider = prediction_entropy_codec.ACTION_TO_PROVIDER[spec.runtime_action]
            capability = self._prediction_capability.get("providers", {}).get(provider, {})
            if not capability.get("graph_eligible"):
                return False
            if provider == "cmix" and not self.config.allow_large_memory:
                return False
        if spec.runtime_action == nncp_entropy_codec.ACTION:
            if not self._nncp_capability.get("graph_eligible"):
                return False
            # The frozen transformer-stream1-v1 bridge is a bounded validation
            # profile, not a silent chunker for the 1 MiB task blocks.
            if len(target.data) > 4096:
                return False
        predicted = self._predicted_active_count(state, spec, target)
        if predicted > self.config.max_active_streams:
            return False
        if spec.max_point is not None:
            key = (state.recipe, spec.action_id)
            if key not in self._max_point_graph_cache:
                try:
                    input_view = "float64"
                    self._max_point_graph_cache[key] = _apply_fixed_graph_action(
                        state.graph, spec, [target_id],
                        parameters=self._parameters(spec, target),
                        input_view=input_view,
                    )
                except (OSError, RuntimeError, ValueError, TypeError):
                    self._max_point_graph_cache[key] = None
            if self._max_point_graph_cache[key] is None:
                return False
        return True

    def _predicted_active_count(
        self, state: TaskState, spec: ActionSpec, target: graph_runtime.Stream
    ) -> int:
        count = len(state.graph.streams)
        if spec.runtime_action == "concat":
            return count - 1
        if spec.runtime_action == exception_separation.ALGORITHM_NAME:
            return count + 2
        if spec.runtime_action == "columnize":
            return count - 1 + target.width
        if spec.runtime_action == "split_segments":
            return count + (3 if spec.parameter_mode == "quarters" else 1)
        if spec.runtime_action in {"float_deconstruct", "bitsplit"}:
            return count + 1
        probed_outputs = self._probe_output_counts.get(_probe_key(spec))
        if probed_outputs is not None:
            return count - 1 + probed_outputs
        return count

    def _parameters(self, spec: ActionSpec, stream: graph_runtime.Stream) -> dict[str, Any] | None:
        values: dict[str, Any] = dict(spec.parameter_items)
        mode = spec.parameter_mode
        if mode == "none":
            return values or None
        if mode == "stream_width":
            values.update(mode="transpose", width=stream.width)
            return values
        if mode == "fp32":
            values.update(format="fp32")
            return values
        if mode == "top8":
            values.update(mode="top8", width=stream.width)
            return values
        if mode == "bits":
            values.update(bits=spec.parameter_value)
            return values
        if mode == "width":
            values.update(width=spec.parameter_value)
            return values
        if mode == "allow_large_memory":
            values.update(allow_large_memory=True)
            return values
        elements = len(stream.data) // stream.width
        if mode == "half":
            values.update(mode="continuous", sizes=[elements // 2, 0])
            return values
        if mode == "quarters":
            first = max(1, elements // 4)
            second = max(1, (elements - first) // 3)
            third = max(1, (elements - first - second) // 2)
            if first + second + third >= elements:
                values.update(mode="continuous", sizes=[elements // 2, 0])
                return values
            values.update(mode="continuous", sizes=[first, second, third, 0])
            return values
        raise GraphTaskError(f"unknown fixed parameter mode {mode}")

    def _apply(self, state: TaskState, spec: ActionSpec, *, count_policy_edit: bool) -> TaskState:
        if state.continuation_edit_mode:
            if spec.category == "edit_type":
                assert spec.edit_kind is not None
                return replace(
                    state, pending_edit_kind=spec.edit_kind,
                    pending_edit_index=None,
                )
            if spec.category == "edit_position":
                assert spec.parameter_value is not None
                position = int(spec.parameter_value)
                assert state.pending_edit_kind is not None
                if state.pending_edit_kind == "delete":
                    edited = self._edited_candidate_state(
                        state, "delete", position, None
                    )
                    if edited is None:
                        raise GraphTaskError("legal delete edit failed replay")
                    return edited
                return replace(state, pending_edit_index=position)
            if (state.pending_edit_kind is not None
                    and state.pending_edit_index is not None):
                edited = self._edited_candidate_state(
                    state, state.pending_edit_kind,
                    state.pending_edit_index, spec.action_id,
                )
                if edited is None:
                    raise GraphTaskError("legal recipe edit failed replay")
                return edited
            raise GraphTaskError("continuation operator action lacks edit type/position")
        return self._apply_graph_operator(
            state, spec, count_policy_edit=count_policy_edit
        )

    def _apply_graph_operator(
        self, state: TaskState, spec: ActionSpec, *, count_policy_edit: bool
    ) -> TaskState:
        ids = tuple(state.graph.streams)
        target_id = ids[spec.target_slot]  # type: ignore[index]
        original_input_ids = [target_id]
        input_ids = [target_id]
        if spec.runtime_action == "concat":
            assert spec.other_slot is not None
            original_input_ids.append(ids[spec.other_slot])
            input_ids.append(ids[spec.other_slot])
        working_graph = state.graph
        if spec.required_parent_runtime_actions:
            parent_runtime_action = ACTION_CATALOG[
                state.parents()[target_id]
            ].runtime_action
            adapter = _INTERNAL_PARENT_ADAPTERS.get(spec.runtime_action)
            if (parent_runtime_action not in spec.required_parent_runtime_actions
                    and adapter is not None):
                adapter_action, required_policy_parents = adapter
                if (required_policy_parents
                        and parent_runtime_action not in required_policy_parents):
                    raise GraphTaskError(
                        "internal adapter policy-parent precondition changed after legality"
                    )
                before = set(working_graph.streams)
                working_graph = working_graph.apply(
                    adapter_action, [target_id], input_view="bytes"
                )
                adapter_outputs = [
                    key for key in working_graph.streams if key not in before
                ]
                if len(adapter_outputs) != 1:
                    raise GraphTaskError(
                        "internal non-policy adapter must produce exactly one data flow"
                    )
                target_id = adapter_outputs[0]
                input_ids = [target_id]
        target_stream = working_graph.streams[target_id]
        parameters = self._parameters(spec, target_stream)
        input_view = spec.input_view if spec.input_view in {"int64", "float64"} else "bytes"
        cache_key = (state.recipe, spec.action_id)
        following = self._max_point_graph_cache.get(cache_key)
        if following is None:
            following = _apply_fixed_graph_action(
                working_graph, spec, input_ids, parameters=parameters,
                input_view=input_view,
            )
        old_depths, old_views = state.depths(), state.views()
        old_ids = set(state.graph.streams)
        new_ids = [key for key in following.streams if key not in old_ids]
        consumed = set(original_input_ids)
        kept_ids = [key for key in state.graph.streams if key not in consumed]
        parent_depth = max(old_depths[key] for key in original_input_ids)
        output_depth = parent_depth + int(spec.ordinary_operator)
        if spec.runtime_action == "split_segments":
            output_views = [old_views[target_id]] * len(new_ids)
        elif spec.runtime_action == "concat":
            selected_views = {old_views[key] for key in original_input_ids}
            output_views = [selected_views.pop() if len(selected_views) == 1 else "bytes"]
        elif spec.runtime_action == graph_runtime.ADJACENT_XOR_PORT_ACTION:
            output_views = ["uint64", "xor_residual64"]
        elif spec.runtime_action == exception_separation.ALGORITHM_NAME:
            output_views = ["uint32"] * len(new_ids)
        else:
            output_views = ["bytes"] * len(new_ids)
        depths = [(key, old_depths[key]) for key in kept_ids]
        depths.extend((key, output_depth) for key in new_ids)
        views = [(key, old_views[key]) for key in kept_ids]
        views.extend(zip(new_ids, output_views))
        old_parents = state.parents()
        parents = [(key, old_parents[key]) for key in kept_ids]
        parents.extend((key, spec.action_id) for key in new_ids)
        result = TaskState(
            graph=following,
            leaf_depths=tuple(depths), leaf_views=tuple(views),
            parent_actions=tuple(parents), recipe=state.recipe + (spec.action_id,),
            node_count=state.node_count + 1,
            policy_edits=state.policy_edits + int(count_policy_edit),
            replayed_nodes=state.replayed_nodes,
            continuation_edit_mode=state.continuation_edit_mode,
            pending_edit_kind=state.pending_edit_kind,
            pending_edit_index=state.pending_edit_index,
            edit_count=state.edit_count,
            edit_history=state.edit_history,
        )
        if (result.node_count > self.config.max_graph_nodes
                or len(result.graph.streams) > self.config.max_active_streams
                or max(result.depths().values()) > self.config.max_ordinary_depth):
            raise GraphTaskError("runtime result exceeded a statically checked graph limit")
        return result

    def _verify_restored(self, restored: Mapping[str, graph_runtime.Stream]) -> None:
        expected = {f"s{index}": stream for index, stream in enumerate(self._roots)}
        if dict(restored) != expected:
            raise GraphTaskError("complete graph frame did not restore root bytes and types exactly")

    def _independent_cold_validate(self, frame: bytes) -> None:
        script = r'''
import hashlib, json, sys
from typed_graph_runtime import restore_frame
roots = restore_frame(
    sys.stdin.buffer.read(), allow_large_memory=(sys.argv[1] == "1")
)
result = [{"id": key, "kind": value.kind, "width": value.width,
           "size": len(value.data), "sha256": hashlib.sha256(value.data).hexdigest()}
          for key, value in roots.items()]
sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
'''
        try:
            result = subprocess.run(
                [sys.executable, "-c", script,
                 "1" if self.config.allow_large_memory else "0"],
                input=frame, capture_output=True,
                cwd=Path(__file__).resolve().parent,
                env={**os.environ, "PYTHONHASHSEED": "0"}, check=False,
                timeout=self.config.cold_validation_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise GraphTaskError("independent cold decoder timed out") from error
        if result.returncode:
            raise GraphTaskError(
                "independent cold decoder failed: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
        try:
            actual = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise GraphTaskError("independent cold decoder returned invalid receipt") from error
        expected = [
            {"id": f"s{index}", "kind": stream.kind, "width": stream.width,
             "size": len(stream.data), "sha256": _sha256_bytes(stream.data)}
            for index, stream in enumerate(self._roots)
        ]
        if actual != expected:
            raise GraphTaskError("independent cold decoder receipt mismatched root data flows")


__all__ = [
    "ACTION_BY_KEY", "ACTION_CATALOG", "ACTION_COUNT", "CATALOG_SCHEMA",
    "CONTINUATION_CANDIDATE_BUDGET", "CandidateBudget", "EvaluationBudgetExceeded",
    "EDIT_POSITION_ACTIONS", "EDIT_TYPE_ACTIONS",
    "EXPECTED_OPERATOR_PDF_SHA256", "FIRST_BLOCK_CANDIDATE_BUDGET",
    "GraphTaskConfig", "GraphTaskError", "MAX_ACTIVE_STREAMS", "MAX_GRAPH_NODES",
    "MAX_ORDINARY_DEPTH", "REGISTRY_FINGERPRINT", "SCHEMA",
    "STATE_CURRENT_OBJECTIVE_INDEX", "STATE_FEATURE_DIM", "STATE_FEATURE_SCHEMA",
    "STOP_ACTION",
    "BoundState", "RecipeEdit", "TaskState", "TypedGraphCompressionTask",
    "assert_formal_training_ready", "catalog_report", "find_action",
    "mechanism_probe_report",
]
