"""Deterministic, policy-gated registry for the first operator harness.

The layered 102-entry catalog is the identity/accounting authority.  This
module adds a small execution contract for an explicitly qualified subset; it
does not infer executability from ``source_action_names`` or from a catalog
``execution_status`` string.  In particular, NNCP (A110) is stopped by policy
and no request field can enable it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import operator_implementation_catalog as catalog


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "operator-harness-registry-v1"
POLICY_REGISTRY = "operator-harness-v1"
NNCP_ID = "A110"
REPRESENTATIVE_IDS = (
    "A013", "A020", "A023", "A031", "A046", "A048", "A077", "A080", "A111",
)
BUILTIN_STEP_KINDS = frozenset({"stop", "split_segments", "concat"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINE_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?$")


@dataclass(frozen=True)
class PortSpec:
    """One typed logical port; cardinality is descriptive, not executable code."""

    name: str
    role: str
    kind: str
    element_width_bytes: int
    logical_view: str
    cardinality: str


@dataclass(frozen=True)
class ActionBinding:
    """A harness/runtime action binding without a performance or correctness claim."""

    action: str
    variant: str
    profiles: tuple[str, ...]
    parameter_names: tuple[str, ...]
    input_ports: tuple[PortSpec, ...]
    output_ports: tuple[PortSpec, ...]
    output_port_multiplicity: Mapping[str, Any]
    evidence_status: str = "registered_unmeasured"


@dataclass(frozen=True)
class OpSpec:
    """Uniform harness view of one historical A-id trace entry."""

    trace_id: str
    name: str
    layer: str
    mechanism_refs: tuple[str, ...]
    family_numbers: tuple[int, ...]
    source_entry_ids: tuple[str, ...]
    source_actions: tuple[str, ...]
    composed_source_actions: tuple[str, ...]
    source_refs: tuple[str, ...]
    typed_port_status: str
    action_bindings: tuple[ActionBinding, ...]
    state: Mapping[str, Any]
    resources: Mapping[str, Any]
    gate: Mapping[str, Any]
    versions: Mapping[str, str]
    hashes: Mapping[str, Any]
    catalog_execution_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _port(
    name: str,
    role: str,
    kind: str,
    width: int,
    logical_view: str,
    cardinality: str,
) -> PortSpec:
    return PortSpec(name, role, kind, width, logical_view, cardinality)


def _binding(
    action: str,
    variant: str,
    inputs: tuple[PortSpec, ...],
    outputs: tuple[PortSpec, ...],
    *,
    profiles: tuple[str, ...] = (),
    parameters: tuple[str, ...] = (),
    output_multiplicity: Mapping[str, Any] | None = None,
) -> ActionBinding:
    multiplicity = output_multiplicity or {"kind": "fixed", "count": len(outputs)}
    return ActionBinding(
        action, variant, profiles, parameters, inputs, outputs, multiplicity
    )


_BINDINGS: dict[str, tuple[ActionBinding, ...]] = {
    "A013": (_binding(
        "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
        "minimum_for_typed_ports",
        (_port("values", "data", "numeric", 8, "int64_or_uint64", "N"),),
        (
            _port("minimum", "inverse_operand", "numeric", 8, "same_as_input", "1"),
            _port("offsets", "data", "numeric", 8, "uint64", "N"),
        ),
        profiles=("int64", "uint64"),
    ),),
    "A020": (_binding(
        "op.bitpack_int.max_value_width.cpp_v1",
        "max_value_selected_fixed_width",
        (_port("values", "data", "numeric", 8, "uint64", "N"),),
        (_port("packed", "data", "serial", 1, "bytes", "variable"),),
        profiles=("uint64",),
        parameters=("bit_order", "alignment_bytes", "minimum_width", "wide_cutoff"),
    ),),
    "A023": (_binding(
        "op.integer_code.uleb128.reference_values.cpp_v2",
        "raw_uint64_values",
        (_port("values", "data", "numeric", 8, "uint64", "N"),),
        (_port("encoded", "data", "serial", 1, "bytes", "variable"),),
        profiles=("uint64",),
    ),),
    "A031": (_binding(
        "op.xor.previous_word.semantic_ports.cpp_v1",
        "first_and_adjacent_xor_tail",
        (_port("values", "data", "numeric", 8, "word64", "N"),),
        (
            _port("first", "anchor", "numeric", 8, "uint64", "0_or_1"),
            _port("xor_tail", "data", "numeric", 8, "uint64", "max(N-1,0)"),
        ),
        profiles=("int64", "float64", "uint64"),
    ),),
    "A046": (_binding(
        "op.zigzag.signed64.raw.cpp_v2",
        "whole_stream",
        (_port("signed", "data", "numeric", 8, "int64", "N"),),
        (_port("codes", "data", "numeric", 8, "uint64", "N"),),
        profiles=("int64",),
    ),),
    "A048": (_binding(
        "harness.a048.byte_planes.v1",
        "harness_typed_io_bridge_over_existing_buff_plane_kernels",
        (_port("offsets", "data", "numeric", 8, "uint64", "N"),),
        (_port(
            "byte_planes", "data_or_zero_plane_carrier", "serial", 1, "uint8_plane",
            "ceil(bit_width/8) ports of N bytes; bit_width=0 uses one empty carrier",
        ),),
        profiles=("uint64",),
        parameters=("bit_width",),
        output_multiplicity={
            "kind": "parameterized",
            "physical_expression": "max(1,ceil(bit_width/8))",
            "semantic_expression": "ceil(bit_width/8)",
            "zero_width_carrier": True,
        },
    ),),
    "A077": (_binding(
        "op.entropy_huffman.project_bytes.cpp_v1",
        "project_byte_huffman",
        (_port(
            "payload", "data_with_preserved_input_descriptor", "any_physical_kind", 0,
            "byte_payload_any_physical_shape", "input_bytes",
        ),),
        (_port("encoded", "data", "serial", 1, "bytes", "variable"),),
        profiles=("*",),
    ),),
    "A080": (_binding(
        "op.entropy_fse.zstd_rans.cpp_reference_v1",
        "project_static_byte_rans_reference",
        (_port(
            "payload", "data_with_preserved_input_descriptor", "any_physical_kind", 0,
            "byte_payload_any_physical_shape", "input_bytes",
        ),),
        (_port("encoded", "data", "serial", 1, "bytes", "variable"),),
        profiles=("*",),
    ),),
    "A111": (_binding(
        "op.outlier_separation.high_bits.uint32.cpp_v1",
        "low_positions_high_typed_ports",
        (_port("values", "data", "numeric", 4, "uint32", "N"),),
        (
            _port("low", "data", "numeric", 4, "uint32", "N"),
            _port("positions", "exception_index", "numeric", 4, "uint32", "E"),
            _port("high", "exception_value", "numeric", 4, "uint32", "E"),
        ),
        profiles=("uint32",),
        parameters=("width",),
    ),),
}


_STATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "A013": {"scope": "call", "reset": "per_call", "inverse_state": "minimum_port"},
    "A020": {"scope": "call", "reset": "per_call", "inverse_state": "bit_width_side_metadata"},
    "A023": {"scope": "call", "reset": "per_call", "inverse_state": "value_count_side_metadata"},
    "A031": {"scope": "stream", "reset": "first_input_word", "inverse_state": "first_port"},
    "A046": {"scope": "call", "reset": "not_applicable", "inverse_state": "none"},
    "A048": {"scope": "call", "reset": "per_call", "inverse_state": "bit_width_and_N"},
    "A077": {"scope": "call", "reset": "per_call", "inverse_state": "codec_frame"},
    "A080": {"scope": "call", "reset": "per_call", "inverse_state": "codec_frame"},
    "A111": {"scope": "call", "reset": "per_call", "inverse_state": "width_and_N"},
}


def _resolved_source_hashes(source_refs: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    """Hash existing referenced files; never claim unresolved prose is file evidence."""
    paths: dict[str, Path] = {}
    for ref in source_refs:
        relative = _LINE_SUFFIX_RE.sub("", ref)
        candidate = ROOT / relative
        if candidate.is_file():
            paths[relative] = candidate
    return tuple(
        {"path": relative, "sha256": _sha256(path.read_bytes())}
        for relative, path in sorted(paths.items())
    )


def _make_spec(item: Mapping[str, Any]) -> OpSpec:
    trace_id = item["implementation_id"]
    source_refs = tuple(item.get("source_refs", ()))
    resolved_hashes = _resolved_source_hashes(source_refs)
    entry_hash = _sha256(_canonical_bytes(item))
    refs_hash = _sha256(_canonical_bytes(source_refs))
    files_hash = _sha256(_canonical_bytes(resolved_hashes))
    bindings = _BINDINGS.get(trace_id, ())

    if trace_id == NNCP_ID:
        typed_port_status = "disabled_by_policy"
        gate = {
            "status": "disabled_by_policy",
            "selectable": False,
            "overridable": False,
            "reason": "NNCP remains stopped for operator-harness-v1",
        }
    elif bindings:
        typed_port_status = "qualified_binding_unmeasured"
        gate = {
            "status": "bound_unmeasured",
            "selectable": True,
            "overridable": False,
            "reason": "explicit v1 harness binding; registry is not measurement evidence",
        }
    else:
        typed_port_status = "unqualified"
        gate = {
            "status": "unqualified",
            "selectable": False,
            "overridable": False,
            "reason": "no typed v1 harness binding",
        }

    state = _STATE_OVERRIDES.get(trace_id, {
        "scope": "unqualified",
        "reset": "unqualified",
        "inverse_state": "unqualified",
    })
    resources = {
        "execution_site": "local_process_only",
        "isolation": "unspecified",
        "cpu_threads": "unspecified",
        "memory_bound": "unspecified",
        "remote_hosts": [],
    }
    if trace_id == NNCP_ID:
        resources = {
            **resources,
            "platform_note": "catalog records Linux x86-64 or explicit Docker",
            "policy_allows_execution": False,
        }

    return OpSpec(
        trace_id=trace_id,
        name=item["name"],
        layer=item["accounting"]["layer"],
        mechanism_refs=tuple(item["accounting"].get("mechanism_ids", ())),
        family_numbers=tuple(item.get("family_numbers", ())),
        source_entry_ids=tuple(item.get("source_entry_ids", ())),
        source_actions=tuple(item.get("source_action_names", ())),
        composed_source_actions=tuple(item.get("composed_source_action_names", ())),
        source_refs=source_refs,
        typed_port_status=typed_port_status,
        action_bindings=bindings,
        state=state,
        resources=resources,
        gate=gate,
        versions={
            "harness_schema": SCHEMA,
            "policy_registry": POLICY_REGISTRY,
            "catalog_registry": catalog.REGISTRY_VERSION,
        },
        hashes={
            "catalog_sha256": catalog.MANIFEST_SHA256,
            "catalog_entry_sha256": entry_hash,
            "source_refs_sha256": refs_hash,
            "resolved_source_files_sha256": files_hash,
            "resolved_source_files": resolved_hashes,
        },
        catalog_execution_status=item.get("execution_status", "unqualified"),
    )


def _all_specs() -> tuple[OpSpec, ...]:
    specs = tuple(_make_spec(item) for item in catalog.IMPLEMENTATIONS)
    if tuple(sorted(spec.trace_id for spec in specs)) != tuple(sorted(catalog.BY_ID)):
        raise RuntimeError("harness registry ids do not match layered catalog")
    if set(_BINDINGS) != set(REPRESENTATIVE_IDS):
        raise RuntimeError("representative ids and explicit bindings diverged")
    return specs


def load_registry() -> dict[str, Any]:
    """Return a detached deterministic JSON-compatible 102-entry registry."""
    operators = [spec.to_dict() for spec in _all_specs()]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_registry": POLICY_REGISTRY,
        "catalog_registry_version": catalog.REGISTRY_VERSION,
        "catalog_sha256": catalog.MANIFEST_SHA256,
        "layer_counts": dict(catalog.LAYER_COUNTS),
        "operator_count": len(operators),
        "representative_ids": list(REPRESENTATIVE_IDS),
        "operators": operators,
    }
    payload["registry_sha256"] = _sha256(_canonical_bytes(payload))
    # Dataclass conversion intentionally preserves tuples.  Normalize through
    # JSON so an on-disk registry readback compares equal to this in-memory
    # value during resume checks.
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def registry_by_id(registry: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Index a registry by trace id and return detached operator records."""
    source = load_registry() if registry is None else registry
    result: dict[str, dict[str, Any]] = {}
    for record in source["operators"]:
        trace_id = record["trace_id"]
        if trace_id in result:
            raise ValueError(f"duplicate harness trace id {trace_id}")
        result[trace_id] = deepcopy(record)
    return result


def get_spec(trace_id: str) -> OpSpec:
    """Return the typed registry record for one current (non-retired) A id."""
    for spec in _all_specs():
        if spec.trace_id == trace_id:
            return spec
    if trace_id in catalog.RETIRED_BY_ID:
        raise ValueError(f"retired implementation {trace_id}")
    raise ValueError(f"unknown implementation {trace_id!r}")


def _stage_action(step: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    stage = step.get("stage")
    if not isinstance(stage, Mapping):
        raise ValueError("stage step requires a stage mapping")
    action = stage.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("stage action must be a non-empty string")
    return action, stage


@dataclass(frozen=True)
class _Flow:
    kind: str
    width: int
    logical_view: str
    count: int | str


def _root_flow(root: Mapping[str, Any], ordinal: int) -> _Flow:
    kind = root.get("kind")
    width = root.get("width")
    if kind not in {"serial", "numeric", "struct"}:
        raise ValueError(f"root {ordinal} has invalid stream kind {kind!r}")
    if type(width) is not int or width <= 0:
        raise ValueError(f"root {ordinal} width must be a positive integer")
    if kind == "serial" and width != 1:
        raise ValueError(f"root {ordinal} serial stream width must be 1")
    if kind == "numeric" and width not in {1, 2, 4, 8}:
        raise ValueError(f"root {ordinal} numeric width must be 1, 2, 4 or 8")
    data_hex = root.get("data_hex")
    if not isinstance(data_hex, str):
        raise ValueError(f"root {ordinal} data_hex must be a string")
    try:
        payload = bytes.fromhex(data_hex)
    except ValueError as exc:
        raise ValueError(f"root {ordinal} data_hex is invalid") from exc
    if len(payload) % width:
        raise ValueError(f"root {ordinal} payload is not width aligned")
    view = root.get("logical_view")
    if view is None or view == "":
        view = {"serial": "bytes", "numeric": "numeric", "struct": "struct"}[kind]
    if not isinstance(view, str):
        raise ValueError(f"root {ordinal} logical_view must be a string")
    return _Flow(kind, width, view, len(payload) // width)


def _decimal_parameter(
    parameters: Mapping[str, str], name: str, *, minimum: int, maximum: int,
    required: bool = False, default: int | None = None,
) -> int:
    if name not in parameters:
        if required:
            raise ValueError(f"missing required parameter {name!r}")
        if default is None:
            raise ValueError(f"parameter {name!r} has no default")
        return default
    text = parameters[name]
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"parameter {name!r} must be an unsigned decimal integer")
    value = int(text)
    if not minimum <= value <= maximum:
        raise ValueError(f"parameter {name!r} must be in [{minimum},{maximum}]")
    return value


def _require_flow(
    flow: _Flow, action: str, *, kind: str, width: int, logical_view: str | None = None
) -> None:
    if flow.kind != kind or flow.width != width:
        raise ValueError(
            f"{action!r} expects {kind} width {width}, got {flow.kind} width {flow.width}"
        )
    if logical_view is not None and flow.logical_view != logical_view:
        raise ValueError(
            f"{action!r} expects logical view {logical_view!r}, got {flow.logical_view!r}"
        )


def _symbolic_sum(counts: list[int | str]) -> int | str:
    if all(type(value) is int for value in counts):
        return sum(counts)  # type: ignore[arg-type]
    return "+".join(str(value) for value in counts)


def _stage_transfer(action: str, stage: Mapping[str, Any], flow: _Flow) -> list[_Flow]:
    """Finite typed transfer function for the nine qualified v1 bindings."""
    profile = stage.get("profile")
    parameters = stage.get("parameters", {})

    if action == "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1":
        expected_view = "int64" if profile == "int64" else "uint64"
        _require_flow(flow, action, kind="numeric", width=8, logical_view=expected_view)
        return [
            _Flow("numeric", 8, expected_view, 1),
            _Flow("numeric", 8, "uint64", flow.count),
        ]
    if action == "op.bitpack_int.max_value_width.cpp_v1":
        _require_flow(flow, action, kind="numeric", width=8, logical_view="uint64")
        bit_order = parameters.get("bit_order", "msb")
        if bit_order not in {"msb", "lsb"}:
            raise ValueError("parameter 'bit_order' must be 'msb' or 'lsb'")
        alignment = _decimal_parameter(
            parameters, "alignment_bytes", minimum=1, maximum=8, default=1
        )
        if alignment not in {1, 8}:
            raise ValueError("parameter 'alignment_bytes' must be 1 or 8")
        _decimal_parameter(parameters, "minimum_width", minimum=0, maximum=64, default=0)
        _decimal_parameter(parameters, "wide_cutoff", minimum=0, maximum=64, default=64)
        return [_Flow("serial", 1, "bytes", f"packed_bytes({flow.count})")]
    if action == "op.integer_code.uleb128.reference_values.cpp_v2":
        _require_flow(flow, action, kind="numeric", width=8, logical_view="uint64")
        return [_Flow("serial", 1, "bytes", f"uleb128_bytes({flow.count})")]
    if action == "op.xor.previous_word.semantic_ports.cpp_v1":
        expected_view = str(profile)
        _require_flow(flow, action, kind="numeric", width=8, logical_view=expected_view)
        first = 0 if flow.count == 0 else 1
        tail = max(flow.count - 1, 0) if type(flow.count) is int else f"max({flow.count}-1,0)"
        return [
            _Flow("numeric", 8, "uint64", first),
            _Flow("numeric", 8, "uint64", tail),
        ]
    if action == "op.zigzag.signed64.raw.cpp_v2":
        _require_flow(flow, action, kind="numeric", width=8, logical_view="int64")
        return [_Flow("numeric", 8, "uint64", flow.count)]
    if action == "harness.a048.byte_planes.v1":
        _require_flow(flow, action, kind="numeric", width=8, logical_view="uint64")
        bit_width = _decimal_parameter(
            parameters, "bit_width", minimum=0, maximum=64, required=True
        )
        semantic_planes = (bit_width + 7) // 8
        if semantic_planes == 0:
            return [_Flow("serial", 1, "a048_zero_plane_carrier", 0)]
        return [
            _Flow("serial", 1, f"a048_byte_plane_{index}", flow.count)
            for index in range(semantic_planes)
        ]
    if action in {
        "op.entropy_huffman.project_bytes.cpp_v1",
        "op.entropy_fse.zstd_rans.cpp_reference_v1",
    }:
        # Packet metadata records the original descriptor, so any valid
        # physical kind/width is accepted and reconstructed exactly on inverse.
        return [_Flow("serial", 1, "bytes", f"entropy_bytes({flow.count}x{flow.width})")]
    if action == "op.outlier_separation.high_bits.uint32.cpp_v1":
        _require_flow(flow, action, kind="numeric", width=4, logical_view="uint32")
        width = _decimal_parameter(
            parameters, "width", minimum=0, maximum=32, required=True
        )
        exceptions = f"E(A111,width={width},N={flow.count})"
        return [
            _Flow("numeric", 4, "uint32", flow.count),
            _Flow("numeric", 4, "uint32", exceptions),
            _Flow("numeric", 4, "uint32", exceptions),
        ]
    raise ValueError(f"no typed transfer function for action {action!r}")


def _take_flows(
    step: Mapping[str, Any], active: dict[int, _Flow], ordinal: int
) -> tuple[list[int], list[_Flow]]:
    inputs = step.get("inputs", [])
    if not isinstance(inputs, list) or not all(type(value) is int and value >= 0 for value in inputs):
        raise ValueError(f"recipe step {ordinal} inputs must be nonnegative integer ids")
    if len(inputs) != len(set(inputs)):
        raise ValueError(f"recipe step {ordinal} has duplicate input ids")
    flows = []
    for flow_id in inputs:
        if flow_id not in active:
            raise ValueError(f"recipe step {ordinal} input flow {flow_id} is not active")
        flows.append(active[flow_id])
    return inputs, flows


def _validate_typed_topology(case: Mapping[str, Any]) -> None:
    roots = case["roots"]
    if not roots:
        raise ValueError("harness case requires at least one root flow")
    active = {ordinal: _root_flow(root, ordinal) for ordinal, root in enumerate(roots)}
    next_id = len(active)
    steps = case["recipe"]["steps"]
    if steps[-1].get("kind") != "stop":
        raise ValueError("recipe requires an explicit terminal stop")
    if sum(step.get("kind") == "stop" for step in steps if isinstance(step, Mapping)) != 1:
        raise ValueError("recipe must contain exactly one terminal stop")

    for ordinal, step in enumerate(steps):
        kind = step["kind"]
        if kind == "stop":
            if ordinal != len(steps) - 1:
                raise ValueError("operation after stop is not allowed")
            inputs, _ = _take_flows(step, active, ordinal)
            if inputs or step.get("segment_sizes") or "stage" in step:
                raise ValueError("stop cannot carry inputs, segment sizes or stage data")
            continue

        inputs, flows = _take_flows(step, active, ordinal)
        if kind == "stage":
            if len(flows) != 1:
                raise ValueError("qualified harness stages require exactly one input flow")
            if step.get("segment_sizes"):
                raise ValueError("stage cannot carry segment_sizes")
            action, stage = _stage_action(step)
            outputs = _stage_transfer(action, stage, flows[0])
        elif kind == "split_segments":
            if "stage" in step:
                raise ValueError("split_segments cannot carry stage data")
            if len(flows) != 1:
                raise ValueError("split_segments requires exactly one input flow")
            sizes = step.get("segment_sizes")
            if (
                not isinstance(sizes, list) or not sizes
                or not all(type(value) is int and value >= 0 for value in sizes)
            ):
                raise ValueError("split_segments requires nonnegative integer segment_sizes")
            if type(flows[0].count) is int and sum(sizes) != flows[0].count:
                raise ValueError("split_segments sizes must exactly cover the input flow")
            outputs = [
                _Flow(flows[0].kind, flows[0].width, flows[0].logical_view, size)
                for size in sizes
            ]
        elif kind == "concat":
            if "stage" in step or step.get("segment_sizes"):
                raise ValueError("concat cannot carry stage data or segment_sizes")
            if len(flows) < 2:
                raise ValueError("concat requires at least two input flows")
            first = flows[0]
            if any((flow.kind, flow.width) != (first.kind, first.width) for flow in flows[1:]):
                raise ValueError("concat requires identical input kinds and widths")
            view = first.logical_view
            if any(flow.logical_view != view for flow in flows[1:]):
                view = "bytes"
            outputs = [_Flow(first.kind, first.width, view, _symbolic_sum([f.count for f in flows]))]
        else:
            raise ValueError(f"unsupported recipe step kind {kind!r}")

        for flow_id in inputs:
            del active[flow_id]
        for output in outputs:
            active[next_id] = output
            next_id += 1


def validate_case(
    case: Mapping[str, Any], registry: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate and detach a JSON harness case without executing its graph.

    All non-builtin stage actions must be explicit bindings of a claimed
    ``implementation_ids`` entry, and every claimed id must occur.  Extra
    case-level receipt fields (for example ``block_id``) are preserved.
    """
    if not isinstance(case, Mapping):
        raise ValueError("harness case must be a mapping")
    normalized = deepcopy(dict(case))
    case_id = normalized.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")

    claimed = normalized.get("implementation_ids")
    if not isinstance(claimed, list) or not claimed or not all(isinstance(x, str) for x in claimed):
        raise ValueError("implementation_ids must be a non-empty string list")
    if len(set(claimed)) != len(claimed):
        raise ValueError("implementation_ids must be unique")

    # A runner should freeze load_registry() once and pass that snapshot here;
    # this avoids re-reading and hashing all referenced source files per case.
    specs = registry_by_id(registry)
    for trace_id in claimed:
        if trace_id not in specs:
            raise ValueError(f"unknown implementation {trace_id!r}")
        gate = specs[trace_id]["gate"]
        if gate["status"] == "disabled_by_policy":
            raise ValueError(f"{trace_id} disabled_by_policy: {gate['reason']}")
        if not gate["selectable"]:
            raise ValueError(f"{trace_id} unqualified for {POLICY_REGISTRY}")

    roots = normalized.get("roots")
    if not isinstance(roots, list):
        raise ValueError("roots must be a list")
    if not all(isinstance(root, Mapping) for root in roots):
        raise ValueError("each root must be a mapping")

    recipe = normalized.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("recipe must be a mapping")
    if recipe.get("policy_registry") != POLICY_REGISTRY:
        raise ValueError(f"recipe policy_registry must be {POLICY_REGISTRY!r}")
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("recipe steps must be a non-empty list")

    action_owner: dict[str, str] = {}
    binding_by_action: dict[str, Mapping[str, Any]] = {}
    for trace_id in claimed:
        for binding in specs[trace_id]["action_bindings"]:
            action = binding["action"]
            if action in action_owner and action_owner[action] != trace_id:
                raise ValueError(f"ambiguous action binding {action}")
            action_owner[action] = trace_id
            binding_by_action[action] = binding

    used: set[str] = set()
    for ordinal, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"recipe step {ordinal} must be a mapping")
        kind = step.get("kind")
        if kind in BUILTIN_STEP_KINDS:
            continue
        if kind != "stage":
            raise ValueError(f"unsupported recipe step kind {kind!r}")
        action, stage = _stage_action(step)
        if action in BUILTIN_STEP_KINDS or action == "split":
            raise ValueError(
                f"builtin {action!r} must be represented by its recipe step kind"
            )
        owner = action_owner.get(action)
        if owner is None:
            raise ValueError(f"stage action {action!r} is not bound to a claimed implementation")
        declared_owner = stage.get("implementation_id")
        if declared_owner is not None and declared_owner != owner:
            raise ValueError(f"stage action {action!r} belongs to {owner}, not {declared_owner}")
        parameters = stage.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"stage parameters for {action!r} must be a mapping")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in parameters.items()):
            raise ValueError(f"stage parameters for {action!r} must be string pairs")
        allowed = set(binding_by_action[action]["parameter_names"])
        unexpected = set(parameters) - allowed
        if unexpected:
            raise ValueError(f"unknown parameters for {action!r}: {sorted(unexpected)}")
        profile = stage.get("profile")
        if profile is not None and not isinstance(profile, str):
            raise ValueError(f"stage profile for {action!r} must be a string")
        profiles = set(binding_by_action[action]["profiles"])
        if "*" not in profiles and profile not in profiles:
            raise ValueError(
                f"profile {profile!r} is not declared for {action!r}; "
                f"expected one of {sorted(profiles)}"
            )
        used.add(owner)

    unused = set(claimed) - used
    if unused:
        raise ValueError(f"claimed implementations are unused: {sorted(unused)}")
    _validate_typed_topology(normalized)
    return normalized


if __name__ == "__main__":
    print(json.dumps(load_registry(), ensure_ascii=False, indent=2, sort_keys=True))
