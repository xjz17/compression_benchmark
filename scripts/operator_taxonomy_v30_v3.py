"""Versioned 30-family graph-action taxonomy over the frozen canonical129 IDs.

This is a registry, not an implementation detector or a codec implementation.
The legacy runtime remains unchanged. Runtime-owned capability overrides must
distinguish a working probability getter (available) from a provider connected
to graph entropy coding (selectable); source presence proves neither.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import canonical129_registry as legacy


REGISTRY_VERSION = "operator-taxonomy-30-v3"
LEGACY_REGISTRY_VERSION = "cpp-canonical-129-v1"
LEGACY_CODEC_COUNT = 129
LEGACY_STOP_ACTION_ID = 129
COLUMNIZE_ACTION_ID = 59
STOP_ACTION_ID = 137
CONCAT_ACTION_ID = 138
SPLIT_SEGMENTS_ACTION_ID = 139
ACTION_COUNT = 140

# Keep a value snapshot, never mutate canonical129.GROUPS or its manifest.
_LEGACY_JSON = json.dumps(legacy.GROUPS, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
LEGACY_SNAPSHOT_SHA256 = hashlib.sha256(_LEGACY_JSON.encode()).hexdigest()

_FAMILY_IDS = (
    "eraser", "scaling", "frame_of_reference", "value_reorder",
    "regression_predictor", "outlier_separation", "bitpack_int", "delta",
    "xor", "trailing_zero_pack", "dictionary", "zigzag", "block_transform",
    "symbol_recode", "run_length", "lz_match", "match_parse_policy",
    "match_window_model", "match_token_format", "integer_code",
    "bitstream_pack", "entropy_huffman", "entropy_range", "entropy_fse",
    "entropy_special", "literal_entropy", "block_state_control",
    "container_frame", "split_columns", "prediction_probability",
)


@dataclass(frozen=True)
class OperatorFamily:
    number: int
    operator_id: str


FAMILIES = tuple(OperatorFamily(i + 1, name) for i, name in enumerate(_FAMILY_IDS))
FAMILY_BY_ID = {f.operator_id: f for f in FAMILIES}
FAMILY_BY_NUMBER = {f.number: f for f in FAMILIES}


@dataclass(frozen=True)
class ActionDescriptor:
    action_id: int
    name: str
    kind: str
    operator_id: str | None
    mechanism: str
    io_variants: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    legacy_policy_action_id: int | None = None
    legacy_operator_id: str | None = None
    default_available: bool = False
    availability_note: str = "No connected implementation is declared by this registry."
    requires_opt_in: tuple[str, ...] = ()
    input_contract: str = "typed data stream"
    output_contract: str = "typed data stream"

    @property
    def family_number(self) -> int | None:
        return FAMILY_BY_ID[self.operator_id].number if self.operator_id else None


@dataclass(frozen=True)
class Availability:
    available: bool
    selectable: bool
    status: str
    reason: str
    requires_opt_in: tuple[str, ...] = ()


def legacy_snapshot() -> tuple[dict[str, Any], ...]:
    """Return detached historical records; editing them cannot change the registry."""
    return tuple(json.loads(_LEGACY_JSON))


def _legacy_descriptors() -> tuple[ActionDescriptor, ...]:
    records = legacy_snapshot()
    if legacy.REGISTRY_VERSION != LEGACY_REGISTRY_VERSION or len(records) != LEGACY_CODEC_COUNT:
        raise RuntimeError("canonical129 snapshot/version changed; explicit migration is required")
    if tuple(row["policy_action_id"] for row in records) != tuple(range(LEGACY_CODEC_COUNT)):
        raise RuntimeError("canonical129 IDs are not the expected frozen dense namespace")
    if records[COLUMNIZE_ACTION_ID]["name"] != "op.block_transform.byte_shuffle.cpp_v2":
        raise RuntimeError("legacy byte-shuffle ID changed")
    out = []
    for row in records:
        columnize = row["policy_action_id"] == COLUMNIZE_ACTION_ID
        out.append(ActionDescriptor(
            action_id=row["policy_action_id"], name=row["name"], kind="codec",
            operator_id="split_columns" if columnize else row["operator_id"],
            mechanism="columnize" if columnize else row["name"],
            io_variants=tuple(v["name"] for v in row["io_variants"]),
            modes=("transpose", "split-struct") if columnize else (),
            aliases=tuple(dict.fromkeys(
                (("op.split_columns.columnize.cpp_v1",) if columnize else ()) +
                tuple(old for old, current in legacy.PREVIOUS_ALGORITHM_TO_CURRENT.items()
                      if current == row["name"] and old != row["name"]
                      and old not in [v["name"] for v in row["io_variants"]]))),
            legacy_policy_action_id=row["policy_action_id"], legacy_operator_id=row["operator_id"],
            default_available=True,
            availability_note=(
                "Existing ID/name/wire replay retained. Legacy byte shuffle supports widths 1/4/8; "
                "the graph columnize adapter must declare availability for new modes."
                if columnize else "Registered by frozen canonical129; runtime-specific legality still applies."
            ),
            output_contract=("column streams and inverse layout metadata" if columnize else
                "minimum:int64/uint64[1] + offsets:uint64[N]; source adapters preserve outer metadata" if row["operator_id"] == "frame_of_reference" else "legacy typed/wire result"),
        ))
    return tuple(out)


_NEW_COLUMNS = (
    ActionDescriptor(129, "op.split_columns.float_deconstruct.cpp_v1", "codec", "split_columns", "float_deconstruct",
                     modes=("fp16", "bf16", "fp32"), input_contract="raw floating-point bit patterns",
                     output_contract="exponent and sign/fraction streams plus exact format metadata"),
    ActionDescriptor(130, "op.split_columns.bitsplit.cpp_v1", "codec", "split_columns", "bitsplit",
                     modes=("generic", "top8", "fp", "bf16"), input_contract="fixed-width raw numeric bit patterns",
                     output_contract="bit-field streams plus non-overlapping complete bit layout"),
    ActionDescriptor(131, "op.split_columns.bitunpack.cpp_v1", "codec", "split_columns", "bitunpack",
                     modes=("fixed-width",), input_contract="packed bits with explicit bit width/order",
                     output_contract="unpacked values plus original valid-bit/padding metadata"),
)

_PROVIDERS = tuple(ActionDescriptor(
    132 + i, f"op.prediction_probability.{key}.provider_v1", "model_provider", "prediction_probability", label,
    modes=("causal",), requires_opt_in=("allow_large_memory",) if key == "cmix" else (),
    availability_note=(
        "Declare causal getter availability separately from graph entropy selectability; "
        "this descriptor does not provide an encoder, a neural model, or a fallback."
    ),
    input_contract="symbols and decoder-reconstructible causal context",
    output_contract="model resource consumed by model-conditioned entropy coding; not compressed bytes",
) for i, (key, label) in enumerate((("ppm", "PPM"), ("dmc", "DMC"), ("paq", "PAQ"), ("cmix", "cmix"), ("nncp", "NNCP"))))

CODEC_ACTIONS = _legacy_descriptors() + _NEW_COLUMNS
MODEL_PROVIDERS = _PROVIDERS
OPERATOR_ACTIONS = CODEC_ACTIONS + MODEL_PROVIDERS
SPECIAL_ACTIONS = (
    ActionDescriptor(STOP_ACTION_ID, "stop", "special", None, "stop", default_available=True,
                     availability_note="Only legal when the graph is independently decodable and closed.",
                     input_contract="closed graph state", output_contract="final independently decodable frame"),
    ActionDescriptor(CONCAT_ACTION_ID, "concat", "special", None, "concat",
                     input_contract="explicit ordered stream IDs", output_contract="joined stream and inverse boundaries"),
    ActionDescriptor(SPLIT_SEGMENTS_ACTION_ID, "split_segments", "special", None, "split_segments",
                     modes=("continuous", "split_by_range"), aliases=("split_by_range",),
                     input_contract="one stream and explicit or encoder-detected boundaries",
                     output_contract="segments and serialized exact inverse boundaries"),
)
ACTIONS = OPERATOR_ACTIONS + SPECIAL_ACTIONS
ACTION_BY_ID = {a.action_id: a for a in ACTIONS}
ACTION_BY_NAME = {name: a for a in ACTIONS for name in (a.name, *a.aliases)}
IO_VARIANT_TO_ACTION = {v: a for a in CODEC_ACTIONS for v in a.io_variants}
COLUMN_MECHANISMS = ("columnize", "float_deconstruct", "bitsplit", "bitunpack")

if tuple(a.action_id for a in ACTIONS) != tuple(range(ACTION_COUNT)):
    raise RuntimeError("new graph action IDs are not dense")
if len(ACTION_BY_NAME) != sum(1 + len(a.aliases) for a in ACTIONS):
    raise RuntimeError("duplicate graph action name/alias")
if set(a.operator_id for a in OPERATOR_ACTIONS) != set(_FAMILY_IDS):
    raise RuntimeError("30-family taxonomy has uncovered or unknown families")


def describe_action(action: int | str) -> ActionDescriptor:
    """Numeric IDs belong to this version only; migrate historical IDs first."""
    if isinstance(action, bool):
        raise ValueError("boolean is not an action ID")
    if type(action) is int:
        try:
            return ACTION_BY_ID[action]
        except KeyError as exc:
            raise ValueError(f"unknown graph action ID: {action}") from exc
    if isinstance(action, str):
        result = ACTION_BY_NAME.get(action) or IO_VARIANT_TO_ACTION.get(action)
        if result is not None:
            return result
    raise ValueError(f"unknown graph action: {action!r}")


def availability(action: int | str, capabilities: Mapping[str, Any] | None = None, *,
                 allow_large_memory: bool = False, mode: str | None = None) -> Availability:
    """Apply runtime-owned name->capability declarations without changing IDs.

    A value may be bool, Availability, or a mapping containing available,
    selectable, status, reason, and requires_opt_in. A model-provider True means
    the runtime has connected BOTH the causal predictor and its entropy contract.
    This function never probes binaries and never substitutes another provider.
    """
    descriptor = describe_action(action)
    if mode is not None and mode not in descriptor.modes:
        raise ValueError(f"unsupported {descriptor.name} mode: {mode}")
    declarations = capabilities or {}
    unknown = set(declarations) - set(ACTION_BY_NAME)
    if unknown:
        raise ValueError(f"unknown capability action names: {sorted(unknown)}")
    names = (descriptor.name, *descriptor.aliases)
    supplied = [declarations[n] for n in names if n in declarations]
    if len(supplied) > 1 and any(value != supplied[0] for value in supplied[1:]):
        raise ValueError("conflicting capability declarations for aliases")
    present = bool(supplied)
    value = supplied[0] if present else None
    if isinstance(value, Availability):
        value = asdict(value)
    if type(value) is bool:
        value = {"available": value}
    if present and not isinstance(value, Mapping):
        raise ValueError("capability must be bool, Availability, or a mapping")
    item = dict(value or {})
    allowed = {"available", "selectable", "status", "reason", "requires_opt_in"}
    if set(item) - allowed:
        raise ValueError(f"unknown capability fields: {sorted(set(item) - allowed)}")
    available = item.get("available", descriptor.default_available)
    requested_selectable = item.get("selectable", available)
    if type(available) is not bool or type(requested_selectable) is not bool:
        raise ValueError("available/selectable must be booleans")
    # The frozen byte-shuffle binding cannot implement the newly requested struct mode.
    if descriptor.action_id == COLUMNIZE_ACTION_ID and mode == "split-struct" and not present:
        available = False
    opt_in = item.get("requires_opt_in", ())
    if isinstance(opt_in, str) or not isinstance(opt_in, (tuple, list)) or any(not isinstance(x, str) for x in opt_in):
        raise ValueError("requires_opt_in must be a sequence of strings")
    requirements = tuple(dict.fromkeys((*descriptor.requires_opt_in, *opt_in)))
    if set(requirements) - {"allow_large_memory"}:
        raise ValueError("unknown opt-in requirement")
    opted_in = not requirements or allow_large_memory
    reason = item.get("reason", descriptor.availability_note)
    status = item.get("status", "connected" if present and available else
                      "legacy_registered" if available and descriptor.legacy_policy_action_id is not None else
                      "available" if available else "descriptor_only")
    if not isinstance(status, str) or not isinstance(reason, str):
        raise ValueError("capability status/reason must be strings")
    if available and not opted_in:
        reason = f"{reason} Explicit allow_large_memory opt-in is required."
    return Availability(available, available and requested_selectable and opted_in, status, reason, requirements)


def selectable_actions(capabilities: Mapping[str, Any] | None = None, *, allow_large_memory: bool = False,
                       include_special: bool = True) -> tuple[ActionDescriptor, ...]:
    candidates = ACTIONS if include_special else OPERATOR_ACTIONS
    return tuple(a for a in candidates if availability(a.name, capabilities,
                 allow_large_memory=allow_large_memory).selectable)


def migrate_action(action_id: int, registry_version: str, *, io_variant: str | None = None,
                   parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Migrate into frozen graph v3, independently of the current atomic IDs."""
    from graph_registry_migration import frozen_atomic
    import operator_taxonomy_v30 as v1
    import operator_taxonomy_v30_v2 as v2
    if type(action_id) is not int: raise ValueError("action_id must be an integer")
    source_version, source_id = registry_version, action_id
    if registry_version == REGISTRY_VERSION:
        target = describe_action(action_id)
        if target.kind == 'codec' and target.operator_id != 'split_columns':
            migrated = frozen_atomic(action_id, LEGACY_REGISTRY_VERSION, legacy,
                                     io_variant=io_variant, parameters=parameters)
            io_variant, parameters = migrated['io_variant'], migrated['parameters']
    else:
        old_graph = {v1.REGISTRY_VERSION:v1, v2.REGISTRY_VERSION:v2}.get(registry_version)
        if old_graph:
            old = old_graph.describe_action(action_id)
            if old.legacy_policy_action_id is None or old.operator_id == 'split_columns':
                target = describe_action(old.name)
                if io_variant is not None and io_variant not in old.io_variants:
                    raise ValueError('I/O variant does not belong to source graph action')
            else:
                migrated = frozen_atomic(old.legacy_policy_action_id, old_graph.LEGACY_REGISTRY_VERSION, legacy,
                                         io_variant=io_variant, parameters=parameters)
                target = describe_action(migrated['policy_action_id'])
                io_variant, parameters = migrated['io_variant'], migrated['parameters']
        else:
            migrated = frozen_atomic(action_id, registry_version, legacy,
                                     io_variant=io_variant, parameters=parameters)
            target = describe_action(STOP_ACTION_ID if migrated['policy_action_id'] == LEGACY_STOP_ACTION_ID else migrated['policy_action_id'])
            io_variant, parameters = migrated['io_variant'], migrated['parameters']
    if target.kind == 'special':
        if io_variant is not None or (target.name == 'stop' and parameters):
            raise ValueError('invalid special operation parameters')
    elif io_variant is not None and io_variant not in target.io_variants:
        raise ValueError('I/O variant does not belong to target')
    copied = json.loads(json.dumps(dict(parameters or {}), sort_keys=True, allow_nan=False))
    return {'registry_version':REGISTRY_VERSION, 'action_id':target.action_id, 'name':target.name,
            'kind':target.kind, 'operator_id':target.operator_id, 'io_variant':io_variant, 'parameters':copied,
            'source_registry_version':source_version, 'source_action_id':source_id}


def registry(capabilities: Mapping[str, Any] | None = None, *, allow_large_memory: bool = False) -> dict[str, Any]:
    """Export descriptors and live capability status separately from inventories."""
    result = []
    for action in ACTIONS:
        status = availability(action.name, capabilities, allow_large_memory=allow_large_memory)
        result.append({**asdict(action), "family_number": action.family_number, "availability": asdict(status)})
    counts = Counter(a.operator_id for a in OPERATOR_ACTIONS)
    return {
        "registry_version": REGISTRY_VERSION, "legacy_registry_version": LEGACY_REGISTRY_VERSION,
        "legacy_snapshot_sha256": LEGACY_SNAPSHOT_SHA256,
        "families": [asdict(f) for f in FAMILIES], "actions": result,
        "counts": {"operator_families": 30, "legacy_codec_actions": 129,
                   "codec_descriptors": len(CODEC_ACTIONS), "model_provider_descriptors": len(MODEL_PROVIDERS),
                   "special_action_kinds": 3, "all_action_descriptors": len(ACTIONS),
                   "selectable_actions": sum(a["availability"]["selectable"] for a in result)},
        "operator_action_counts": dict(counts),
        "special_ids": {a.name: a.action_id for a in SPECIAL_ACTIONS},
        "legacy_stop_migration": {"source_id": 129, "target_id": STOP_ACTION_ID},
        "columnize_reclassification": {"legacy_id": COLUMNIZE_ACTION_ID,
            "preserved_name": ACTION_BY_ID[COLUMNIZE_ACTION_ID].name,
            "old_operator_id": "block_transform", "new_operator_id": "split_columns",
            "adds_algorithm": False},
        "evidence_boundary": "Descriptor counts do not assert adapter availability or policy legality. Special actions are not operator families.",
    }
