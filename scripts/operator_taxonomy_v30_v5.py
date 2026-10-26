"""Current 30-family graph view with the identity Scaling proxy retired.

The v4 numeric namespace remains frozen for old graph frames.  V5 preserves
every numeric slot but makes ``op.scaling.reference_v1`` a non-executable
retirement descriptor, so no saved ID is silently reinterpreted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
from typing import Any, Mapping

import operator_taxonomy_v30_v4 as frozen
import operator_implementation_catalog as implementation_catalog


REGISTRY_VERSION = "operator-taxonomy-30-v5"
PREVIOUS_REGISTRY_VERSION = frozen.REGISTRY_VERSION
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

_frozen_identity = next(
    action for action in frozen.CODEC_ACTIONS
    if action.name in RETIRED_IDENTITY_ACTION_NAMES
)
RETIRED_ACTIONS = (
    replace(
        _frozen_identity,
        kind="retired",
        default_available=False,
        availability_note=(
            "Identity Scaling was removed from the 108-item implementation directory; "
            "the numeric slot remains reserved only for old-frame decoding."
        ),
    ),
)
CODEC_ACTIONS = tuple(
    action for action in frozen.CODEC_ACTIONS
    if action.name not in RETIRED_IDENTITY_ACTION_NAMES
)
OPERATOR_ACTIONS = CODEC_ACTIONS + MODEL_PROVIDERS
ACTIONS = tuple(
    RETIRED_ACTIONS[0] if action.name in RETIRED_IDENTITY_ACTION_NAMES else action
    for action in frozen.ACTIONS
)
ACTION_BY_ID = {action.action_id: action for action in ACTIONS}
ACTION_BY_NAME = {
    name: action for action in ACTIONS for name in (action.name, *action.aliases)
}
IO_VARIANT_TO_ACTION = {
    variant: action for action in CODEC_ACTIONS for variant in action.io_variants
}

if tuple(action.action_id for action in ACTIONS) != tuple(range(ACTION_COUNT)):
    raise RuntimeError("v5 must preserve every v4 numeric action slot")
if len(RETIRED_ACTIONS) != 1 or RETIRED_ACTIONS[0].action_id != 5:
    raise RuntimeError("identity Scaling retirement slot changed")
if set(action.operator_id for action in OPERATOR_ACTIONS) != set(frozen.FAMILY_BY_ID):
    raise RuntimeError("retirement unexpectedly removed a 30-family taxonomy category")


def legacy_snapshot() -> tuple[dict[str, Any], ...]:
    return frozen.legacy_snapshot()


def describe_action(action: int | str) -> ActionDescriptor:
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
    descriptor = describe_action(action)
    if descriptor.kind == "retired":
        if mode is not None:
            raise ValueError("retired identity Scaling has no modes")
        declarations = capabilities or {}
        unknown = set(declarations) - set(ACTION_BY_NAME)
        if unknown:
            raise ValueError(f"unknown capability action names: {sorted(unknown)}")
        return Availability(
            False, False, "retired",
            descriptor.availability_note,
        )
    return frozen.availability(
        descriptor.name, capabilities,
        allow_large_memory=allow_large_memory, mode=mode,
    )


def selectable_actions(capabilities: Mapping[str, Any] | None = None, *,
                       allow_large_memory: bool = False,
                       include_special: bool = True) -> tuple[ActionDescriptor, ...]:
    candidates = ACTIONS if include_special else OPERATOR_ACTIONS
    return tuple(
        action for action in candidates
        if availability(action.name, capabilities,
                        allow_large_memory=allow_large_memory).selectable
    )


def migrate_action(action_id: int, registry_version: str, *,
                   io_variant=None, parameters=None):
    """Migrate non-retired IDs; identity Scaling migrates to an empty recipe."""
    source_version, source_id = registry_version, action_id
    if registry_version not in (REGISTRY_VERSION, frozen.REGISTRY_VERSION, LEGACY_REGISTRY_VERSION):
        import operator_taxonomy_v30 as v1
        import operator_taxonomy_v30_v2 as v2
        import operator_taxonomy_v30_v3 as v3
        source_graph = {
            v1.REGISTRY_VERSION: v1,
            v2.REGISTRY_VERSION: v2,
            v3.REGISTRY_VERSION: v3,
        }.get(registry_version)
        if source_graph is not None:
            source_descriptor = source_graph.describe_action(action_id)
            if source_descriptor.name in RETIRED_IDENTITY_ACTION_NAMES:
                if io_variant not in (None, source_descriptor.name) or parameters not in (None, {}):
                    raise ValueError("retired identity Scaling has no parameters")
                return {
                    "registry_version": REGISTRY_VERSION,
                    "kind": "retired_recipe",
                    "replacement_action_ids": [],
                    "replacement_action_names": [],
                    "source_registry_version": source_version,
                    "source_action_id": source_id,
                }
    if registry_version in (REGISTRY_VERSION, frozen.REGISTRY_VERSION, LEGACY_REGISTRY_VERSION):
        source_descriptor = describe_action(action_id)
        if source_descriptor.kind == "retired":
            if io_variant not in (None, source_descriptor.name) or parameters not in (None, {}):
                raise ValueError("retired identity Scaling has no parameters")
            return {
                "registry_version": REGISTRY_VERSION,
                "kind": "retired_recipe",
                "replacement_action_ids": [],
                "replacement_action_names": [],
                "source_registry_version": source_version,
                "source_action_id": source_id,
            }
    delegated_version = (
        frozen.REGISTRY_VERSION if registry_version == REGISTRY_VERSION
        else registry_version
    )
    try:
        result = frozen.migrate_action(
            action_id, delegated_version,
            io_variant=io_variant, parameters=parameters,
        )
    except ValueError as exc:
        if str(exc) != "identity Scaling is retired; omit the stage from new recipes":
            raise
        if io_variant not in (None, "op.scaling.reference_v1") or parameters not in (None, {}):
            raise ValueError("retired identity Scaling has no parameters") from exc
        return {
            "registry_version": REGISTRY_VERSION,
            "kind": "retired_recipe",
            "replacement_action_ids": [],
            "replacement_action_names": [],
            "source_registry_version": source_version,
            "source_action_id": source_id,
        }
    if result["name"] in RETIRED_IDENTITY_ACTION_NAMES:
        if io_variant not in (None, result["name"]) or parameters not in (None, {}):
            raise ValueError("retired identity Scaling has no parameters")
        return {
            "registry_version": REGISTRY_VERSION,
            "kind": "retired_recipe",
            "replacement_action_ids": [],
            "replacement_action_names": [],
            "source_registry_version": source_version,
            "source_action_id": source_id,
        }
    result["registry_version"] = REGISTRY_VERSION
    result["source_registry_version"] = source_version
    result["source_action_id"] = source_id
    return result


def registry(capabilities: Mapping[str, Any] | None = None, *,
             allow_large_memory: bool = False) -> dict[str, Any]:
    rows = []
    for action in ACTIONS:
        status = availability(
            action.name, capabilities,
            allow_large_memory=allow_large_memory,
        )
        rows.append({
            **asdict(action),
            "family_number": action.family_number,
            "availability": asdict(status),
        })
    counts = Counter(action.operator_id for action in OPERATOR_ACTIONS)
    return {
        "registry_version": REGISTRY_VERSION,
        "previous_registry_version": PREVIOUS_REGISTRY_VERSION,
        "legacy_registry_version": LEGACY_REGISTRY_VERSION,
        "legacy_snapshot_sha256": LEGACY_SNAPSHOT_SHA256,
        "families": [asdict(family) for family in FAMILIES],
        "actions": rows,
        "counts": {
            "operator_families": 30,
            "legacy_codec_action_slots": LEGACY_CODEC_COUNT,
            "active_codec_descriptors": len(CODEC_ACTIONS),
            "retired_descriptor_slots": len(RETIRED_ACTIONS),
            "model_provider_descriptors": len(MODEL_PROVIDERS),
            "special_action_kinds": 3,
            "all_reserved_action_slots": len(ACTIONS),
            "active_action_descriptors": len(ACTIONS) - len(RETIRED_ACTIONS),
            "selectable_actions": sum(row["availability"]["selectable"] for row in rows),
        },
        "operator_action_counts": dict(counts),
        "special_ids": {action.name: action.action_id for action in SPECIAL_ACTIONS},
        "retirement": {
            "reserved_action_id": RETIRED_ACTIONS[0].action_id,
            "name": RETIRED_ACTIONS[0].name,
            "replacement_recipe": [],
            "old_frame_decode_supported": True,
            "new_graph_forward_supported": False,
        },
        "implementation_directory": {
            "registry_version": implementation_catalog.REGISTRY_VERSION,
            "implementation_count": implementation_catalog.IMPLEMENTATION_COUNT,
            "count_is_graph_action_count": False,
        },
        "legacy_stop_migration": {
            "source_id": LEGACY_STOP_ACTION_ID,
            "target_id": STOP_ACTION_ID,
        },
        "columnize_reclassification": {
            "legacy_id": COLUMNIZE_ACTION_ID,
            "preserved_name": ACTION_BY_ID[COLUMNIZE_ACTION_ID].name,
            "old_operator_id": "block_transform",
            "new_operator_id": "split_columns",
            "adds_algorithm": False,
        },
        "evidence_boundary": (
            "The 108-item equivalence directory, active graph descriptors, reserved IDs, "
            "and executable adapters are separate counts."
        ),
    }
