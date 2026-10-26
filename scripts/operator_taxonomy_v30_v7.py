"""Current graph view with explicit ZigZag stages and unsigned ULEB128.

All graph slots remain stable. Old graph descriptors are imported only for
versioned migration; current retired slots are never forward selectable.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, replace
import importlib
import json
from typing import Any, Mapping
import operator_taxonomy_v30_v6 as previous
import operator_taxonomy_v30_v4 as frozen
import operator_implementation_catalog as implementation_catalog
import primitive_operator_action_runtime as primitive

REGISTRY_VERSION = "operator-taxonomy-30-v7"
PREVIOUS_REGISTRY_VERSION = previous.REGISTRY_VERSION
POLICY_REGISTRY_VERSION = primitive.ACTION_REGISTRY_VERSION
LEGACY_REGISTRY_VERSION = frozen.LEGACY_REGISTRY_VERSION
LEGACY_CODEC_COUNT = frozen.LEGACY_CODEC_COUNT
LEGACY_STOP_ACTION_ID = frozen.LEGACY_STOP_ACTION_ID
COLUMNIZE_ACTION_ID = frozen.COLUMNIZE_ACTION_ID
STOP_ACTION_ID = frozen.STOP_ACTION_ID
CONCAT_ACTION_ID = frozen.CONCAT_ACTION_ID
SPLIT_SEGMENTS_ACTION_ID = frozen.SPLIT_SEGMENTS_ACTION_ID
ACTION_COUNT = frozen.ACTION_COUNT
LEGACY_SNAPSHOT_SHA256 = frozen.LEGACY_SNAPSHOT_SHA256
OperatorFamily = frozen.OperatorFamily
ActionDescriptor = frozen.ActionDescriptor
Availability = frozen.Availability
FAMILIES = frozen.FAMILIES
FAMILY_BY_ID = frozen.FAMILY_BY_ID
FAMILY_BY_NUMBER = frozen.FAMILY_BY_NUMBER
COLUMN_MECHANISMS = frozen.COLUMN_MECHANISMS
MODEL_PROVIDERS = frozen.MODEL_PROVIDERS
SPECIAL_ACTIONS = frozen.SPECIAL_ACTIONS
RETIRED_IDENTITY_ACTION_NAMES = implementation_catalog.RETIRED_IDENTITY_ACTION_NAMES

def _current_descriptor(old):
    if old.action_id in primitive.RETIRED_POLICY_IDS:
        return replace(old, kind="retired", default_available=False,
            availability_note="Retired implementation or composite; migrate the old selector to an explicit current recipe.")
    policy = primitive.POLICY_ACTION_BY_ID.get(old.action_id)
    if policy is None:
        return old
    changed = policy.name != old.name
    result = replace(old, name=policy.name,
        operator_id="split_columns" if old.action_id == COLUMNIZE_ACTION_ID else policy.operator_id,
        mechanism=policy.name if changed else old.mechanism,
        io_variants=primitive.POLICY_IO_VARIANTS[policy.name],
        aliases=() if changed else tuple(a for a in old.aliases if a not in primitive.RETIRED_COMPOSITE_ACTION_NAMES),
        availability_note="Registered current split stage; runtime input and I/O profile checks still apply.")
    if old.action_id == 17:
        result = replace(result, input_contract="values:int64[N]; LECO block profile",
            output_contract="2 generated semantic streams: coefficients:float64[blocks,2], residuals:int64[N]; block lengths as recovery metadata")
    elif old.action_id == 25:
        result = replace(result, input_contract="unsigned values:uint64[N], or scalar uint32 length; carrier/profile explicit",
            output_contract="1 generated semantic stream: encoded:bytes; incoming anchors/token streams bypass unchanged")
    elif old.action_id == 55:
        result = replace(result, input_contract="selected values or residuals:int64[N]; explicit I/O profile",
            output_contract="1 transformed semantic stream: codes:uint64[N]; unselected recovery streams bypass unchanged")
    if old.action_id == 58:
        result = replace(result,input_contract="offsets:uint64[N]; bit_width:uint8 in 0..64; base/scale bypass",
            output_contract="ceil(bit_width/8) independent byte planes:uint8[N], high plane first; base:int64[1] bypass; width/count/scale are recovery metadata")
    return result

ACTIONS = tuple(_current_descriptor(a) if a.kind == "codec" else a for a in frozen.ACTIONS)
CODEC_ACTIONS = tuple(a for a in ACTIONS if a.kind == "codec")
RETIRED_ACTIONS = tuple(a for a in ACTIONS if a.kind == "retired")
OPERATOR_ACTIONS = CODEC_ACTIONS + MODEL_PROVIDERS
ACTION_BY_ID = {a.action_id:a for a in ACTIONS}
ACTION_BY_NAME = {name:a for a in ACTIONS for name in (a.name,*a.aliases)}
IO_VARIANT_TO_ACTION = {v:a for a in CODEC_ACTIONS for v in a.io_variants}
if tuple(a.action_id for a in ACTIONS) != tuple(range(ACTION_COUNT)):
    raise RuntimeError("v6 graph must preserve every frozen numeric slot")
if {a.action_id for a in RETIRED_ACTIONS} != {5,18,35,87,90}:
    raise RuntimeError("v6 graph retirement slots changed")
if {a.operator_id for a in OPERATOR_ACTIONS} != set(FAMILY_BY_ID):
    raise RuntimeError("v6 graph must retain all 30 operator families")

def legacy_snapshot():
    return frozen.legacy_snapshot()

def describe_action(action):
    if type(action) is int and action in ACTION_BY_ID:
        return ACTION_BY_ID[action]
    if isinstance(action,str):
        result = ACTION_BY_NAME.get(action) or IO_VARIANT_TO_ACTION.get(action)
        if result is not None:return result
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
    if descriptor.kind == "retired":
        return Availability(False, False, "retired", descriptor.availability_note)
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


def migrate_action(action_id, registry_version, *, io_variant=None, parameters=None):
    """Return one current step or an explicit, ordered replacement recipe."""
    if type(action_id) is not int:raise ValueError("action_id must be an integer")
    source_version,source_id=registry_version,action_id
    if registry_version.startswith("operator-taxonomy-30-v"):
        modules={f"operator-taxonomy-30-v{i}":("operator_taxonomy_v30" if i==1 else f"operator_taxonomy_v30_v{i}") for i in range(1,8)}
        if registry_version not in modules:raise ValueError("unknown graph registry version")
        graph=importlib.import_module(modules[registry_version])
        old=graph.describe_action(action_id)
        if registry_version==REGISTRY_VERSION and old.kind=="retired":
            raise ValueError("retired current graph slot is not selectable")
        if old.legacy_policy_action_id is None or old.operator_id=="split_columns":
            target=describe_action(old.name)
            if io_variant is not None and io_variant not in target.io_variants:
                raise ValueError("I/O variant does not belong to source graph action")
            if target.kind=="special" and (io_variant is not None or (target.name=="stop" and parameters)):
                raise ValueError("invalid special operation parameters")
            return _step(target,io_variant,parameters,source_version,source_id)
        policy_version=getattr(graph,"POLICY_REGISTRY_VERSION",graph.LEGACY_REGISTRY_VERSION)
        migrated=primitive.migrate_policy_action(old.legacy_policy_action_id,policy_version,io_variant=io_variant,parameters=parameters)
    else:
        migrated=primitive.migrate_policy_action(action_id,registry_version,io_variant=io_variant,parameters=parameters)
    if migrated.get("migration_kind") in {"replacement_recipe","retired_identity_bypass"}:
        ids=migrated["replacement_policy_action_ids"]
        variants=migrated["replacement_io_variants"]
        settings=migrated["replacement_parameters"]
        steps=[_step(describe_action(i),v,p,source_version,source_id) for i,v,p in zip(ids,variants,settings)]
        return dict(registry_version=REGISTRY_VERSION,kind="retired_recipe",
            migration_kind=migrated["migration_kind"],replacement_action_ids=ids,
            replacement_action_names=[x["name"] for x in steps],replacement_io_variants=variants,
            replacement_parameters=settings,replacement_steps=steps,
            source_registry_version=source_version,source_action_id=source_id)
    target=describe_action(STOP_ACTION_ID if migrated["policy_action_id"]==primitive.STOP_ACTION_ID else migrated["policy_action_id"])
    result=_step(target,migrated["io_variant"],migrated["parameters"],source_version,source_id)
    for key in ("migration_kind","requires_remeasurement"):
        if key in migrated:result[key]=migrated[key]
    return result

def _step(target,variant,parameters,source_version,source_id):
    if variant is not None and variant not in target.io_variants:
        raise ValueError("I/O variant does not belong to target graph stage")
    return dict(registry_version=REGISTRY_VERSION,action_id=target.action_id,name=target.name,
        kind=target.kind,operator_id=target.operator_id,io_variant=variant,
        parameters=json.loads(json.dumps(dict(parameters or {}),allow_nan=False)),
        source_registry_version=source_version,source_action_id=source_id)

def registry(capabilities=None, *, allow_large_memory=False):
    rows=[{**asdict(a),"family_number":a.family_number,"availability":asdict(availability(a.name,capabilities,allow_large_memory=allow_large_memory))} for a in ACTIONS]
    return dict(registry_version=REGISTRY_VERSION,previous_registry_version=PREVIOUS_REGISTRY_VERSION,
        legacy_registry_version=LEGACY_REGISTRY_VERSION,policy_registry_version=POLICY_REGISTRY_VERSION,
        legacy_snapshot_sha256=LEGACY_SNAPSHOT_SHA256,families=[asdict(f) for f in FAMILIES],actions=rows,
        counts=dict(operator_families=30,legacy_codec_action_slots=LEGACY_CODEC_COUNT,
            active_codec_descriptors=len(CODEC_ACTIONS),retired_descriptor_slots=len(RETIRED_ACTIONS),
            model_provider_descriptors=len(MODEL_PROVIDERS),special_action_kinds=3,
            all_reserved_action_slots=len(ACTIONS),active_action_descriptors=len(ACTIONS)-len(RETIRED_ACTIONS),
            selectable_actions=sum(x["availability"]["selectable"] for x in rows)),
        operator_action_counts=dict(Counter(a.operator_id for a in OPERATOR_ACTIONS)),
        special_ids={a.name:a.action_id for a in SPECIAL_ACTIONS},
        retirements=[dict(reserved_action_id=a.action_id,name=a.name,
            old_frame_decode_supported=True,new_graph_forward_supported=False) for a in RETIRED_ACTIONS],
        implementation_directory=dict(registry_version=implementation_catalog.REGISTRY_VERSION,
            implementation_count=implementation_catalog.IMPLEMENTATION_COUNT,
            trace_entry_count=implementation_catalog.TRACE_ENTRY_COUNT,
            mechanism_count=implementation_catalog.MECHANISM_COUNT,
            layer_counts=implementation_catalog.LAYER_COUNTS,
            count_is_graph_action_count=False),
        legacy_stop_migration=dict(source_id=LEGACY_STOP_ACTION_ID,target_id=STOP_ACTION_ID),
        columnize_reclassification=dict(legacy_id=COLUMNIZE_ACTION_ID,preserved_name=ACTION_BY_ID[COLUMNIZE_ACTION_ID].name,
            old_operator_id="block_transform",new_operator_id="split_columns",adds_algorithm=False),
        evidence_boundary="The 102 trace entries, 67 retained mechanisms, graph descriptors, reserved IDs and executable adapters are separate counts.")
