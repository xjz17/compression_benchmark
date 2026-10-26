#!/usr/bin/env python3
"""Build the deterministic, local-only batch plan for operator harness v2.

This module is metadata only.  It does not import or execute a codec, contact a
remote host, or mutate either source catalog.  The frozen resolved catalog is
the execution-plan authority and the layered catalog is the trace/layer
authority.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAYERED_CATALOG = (
    ROOT / "benchmark/reference/actions/operator_implementations_layered_v1.json"
)
RESOLVED_CATALOG = (
    ROOT
    / "analysis_outputs/operator102_current135_20261017_v1/native/RESOLVED_CATALOG.json"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmark/reference/actions/operator_harness_batch_catalog_v2.json"
)
SPECIAL_BRIDGE_SOURCE = ROOT / "cpp/native_graph_v1/operator102_special_bridges.cpp"

SCHEMA = "operator-harness-batch-catalog-v2"
POLICY_REGISTRY = "operator-harness-v2"

V1_CORE_IDS = (
    "A013", "A020", "A023", "A031", "A046", "A048", "A077", "A080", "A111",
)

# These exclusions are policy, not a runtime capability probe.  There is no
# request or CLI override.  Keeping every reason explicit prevents a later
# catalog capability change from silently admitting one of the stopped rows.
DISABLED_POLICY: dict[str, dict[str, str]] = {
    "A106": {"family": "PPMd", "reason": "PPMd is outside harness-v2 scope"},
    "A107": {"family": "DMC", "reason": "DMC is outside harness-v2 scope"},
    "A108": {"family": "PAQ8", "reason": "PAQ8 is outside harness-v2 scope"},
    "A109": {
        "family": "CMIX/LSTM",
        "reason": "CMIX/LSTM is outside harness-v2 scope",
    },
    "A110": {
        "family": "NNCP",
        "reason": "NNCP remains stopped and disabled by harness-v2 policy",
    },
}

BATCH_ORDER = (
    "v1core9",
    "ordinary_source_pipelines",
    "source_exact_dispatcher",
    "special_residual_deps",
)

SOURCE_EXACT_DISPATCHER = "operator102_dispatcher_source_exact_v1"
SPECIAL_BRIDGE_DISPATCHER = "operator102_special_bridge_v1"
RESIDUAL_DEPS_DISPATCHER = "operator102_support_deps_general_residual_v1"
ORDINARY_DISPATCHER = "native_graph_v1_stage_dispatch"

# Mirrors supports_special_bridge() in the source named above.  The generator
# checks that the frozen dispatcher rows are exactly this set, so unsupported
# IDs cannot be mislabeled as native special bridges.
CPP_SPECIAL_BRIDGE_IDS = (
    "A057", "A065", "A086", "A094", "A096",
    "A097", "A098", "A099", "A100", "A102",
)

_EXPECTED_ALL_COUNT = 102
_EXPECTED_ELIGIBLE_COUNT = 97
_EXPECTED_LAYER_COUNTS = {
    "mechanism": 67,
    "profile": 7,
    "recipe": 14,
    "adapter": 14,
}
_EXPECTED_ELIGIBLE_LAYER_COUNTS = {
    "mechanism": 62,
    "profile": 7,
    "recipe": 14,
    "adapter": 14,
}
_EXPECTED_BATCH_COUNTS = {
    "v1core9": 9,
    "ordinary_source_pipelines": 66,
    "source_exact_dispatcher": 5,
    "special_residual_deps": 17,
}
_FALLBACK_WRAPPER_ID = "operator102.lossless_domain_fallback.selector_raw.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value, _sha256(raw)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _batch_id(row: Mapping[str, Any]) -> str:
    implementation_id = row["implementation_id"]
    if implementation_id in V1_CORE_IDS:
        return "v1core9"
    dispatcher = row["dispatcher"]
    if dispatcher == ORDINARY_DISPATCHER:
        return "ordinary_source_pipelines"
    if dispatcher == SOURCE_EXACT_DISPATCHER:
        return "source_exact_dispatcher"
    if dispatcher in {SPECIAL_BRIDGE_DISPATCHER, RESIDUAL_DEPS_DISPATCHER}:
        return "special_residual_deps"
    raise ValueError(
        f"{implementation_id}: unknown frozen dispatcher {dispatcher!r}"
    )


def _layer_semantics(layer: str) -> str:
    return {
        "mechanism": "mechanism_trace",
        "profile": "parameter_or_io_profile_trace",
        "recipe": "composite_recipe_trace",
        "adapter": "compatibility_adapter_trace",
    }[layer]


def _v1_binding_semantics(implementation_id: str) -> str | None:
    if implementation_id not in V1_CORE_IDS:
        return None
    if implementation_id == "A048":
        return "harness_composite_adapter_over_existing_cpp_byte_plane_kernels"
    return "direct_typed_cpp_stage_binding"


def _pipeline_with_roles(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    pipeline = row["present_pipeline"]
    focal = row["focal_step_index"]
    _require(isinstance(pipeline, list) and pipeline, "present_pipeline must be nonempty")
    _require(type(focal) is int and 0 <= focal < len(pipeline), "invalid focal_step_index")
    result: list[dict[str, Any]] = []
    for index, source_step in enumerate(pipeline):
        _require(isinstance(source_step, Mapping), "pipeline step must be an object")
        role = "prefix" if index < focal else "focal" if index == focal else "suffix"
        step = deepcopy(dict(source_step))
        step["step_index"] = index
        step["role"] = role
        result.append(step)
    return result


def _domain_execution(row: Mapping[str, Any]) -> dict[str, Any]:
    contract = row.get("domain_fallback_contract")
    focal = row["present_pipeline"][row["focal_step_index"]]
    if contract is None:
        return {
            "strict_core_action": focal["action"],
            "measurement_wrapper": None,
            "wrapper_counts_as_operator": False,
            "support_accounting": "strict_core_only",
        }
    _require(
        contract.get("source_id") == _FALLBACK_WRAPPER_ID,
        f"{row['implementation_id']}: unexpected lossless fallback wrapper",
    )
    _require(
        focal.get("parameters", {}).get("exact_raw_fallback") is True,
        f"{row['implementation_id']}: fallback contract lacks explicit parameter",
    )
    return {
        "strict_core_action": focal["action"],
        "measurement_wrapper": _FALLBACK_WRAPPER_ID,
        "wrapper_counts_as_operator": False,
        "support_accounting": "strict_target_and_raw_fallback_reported_separately",
        "contract": deepcopy(contract),
    }


def _dispatch_support(row: Mapping[str, Any]) -> dict[str, Any]:
    implementation_id = row["implementation_id"]
    dispatcher = row["dispatcher"]
    if dispatcher == SPECIAL_BRIDGE_DISPATCHER:
        _require(
            implementation_id in CPP_SPECIAL_BRIDGE_IDS,
            f"{implementation_id}: frozen special bridge is unsupported by C++ bridge",
        )
        return {
            "kind": "cpp_special_bridge",
            "support_status": "supported",
            "dependency_scope": "operator102_special_bridges_and_private_support_deps",
        }
    if dispatcher == RESIDUAL_DEPS_DISPATCHER:
        return {
            "kind": "cpp_general_residual_support_dependencies",
            "support_status": "planned",
            "dependency_scope": "private_general_codec_residual_dependencies",
        }
    if dispatcher == SOURCE_EXACT_DISPATCHER:
        return {
            "kind": "cpp_source_exact_dispatcher",
            "support_status": "planned",
            "dependency_scope": "source_exact_cli_backends",
        }
    return {
        "kind": "cpp_stage_dispatch",
        "support_status": "planned",
        "dependency_scope": "native_graph_v1_stages",
    }


def _operator_row(
    resolved: Mapping[str, Any], layered: Mapping[str, Any]
) -> dict[str, Any]:
    implementation_id = resolved["implementation_id"]
    _require(
        resolved["name"] == layered["name"],
        f"{implementation_id}: name differs between source catalogs",
    )
    layer = layered["accounting"]["layer"]
    disabled = DISABLED_POLICY.get(implementation_id)
    batch_id = None if disabled else _batch_id(resolved)
    pipeline = _pipeline_with_roles(resolved)
    semantics = {
        "catalog_layer": _layer_semantics(layer),
        "pipeline": (
            "single_focal_stage"
            if len(pipeline) == 1
            else "prefix_focal_suffix_composite"
        ),
        "v1_registry_binding": _v1_binding_semantics(implementation_id),
    }
    selection: dict[str, Any]
    if disabled:
        selection = {
            "state": "disabled_by_policy",
            "selectable": False,
            "overridable": False,
            "family": disabled["family"],
            "reason": disabled["reason"],
        }
    else:
        selection = {
            "state": "planned_unverified",
            "selectable": True,
            "overridable": False,
            "family": None,
            "reason": "eligible for bounded harness-v2 validation",
        }
    dispatch_support = _dispatch_support(resolved)
    if disabled:
        dispatch_support = {
            **dispatch_support,
            "support_status": "disabled_by_policy",
        }
    return {
        "implementation_id": implementation_id,
        "name": resolved["name"],
        "layer": layer,
        "batch_id": batch_id,
        "selection": selection,
        "initial_kind": resolved["initial_kind"],
        "initial_width": resolved["initial_width"],
        "input_view": resolved["input_view"],
        "input_adapter_kind": resolved["input_adapter_kind"],
        "present_pipeline": pipeline,
        "focal_step_index": resolved["focal_step_index"],
        "pipeline_partitions": {
            "prefix_step_indexes": [
                step["step_index"] for step in pipeline if step["role"] == "prefix"
            ],
            "focal_step_index": resolved["focal_step_index"],
            "suffix_step_indexes": [
                step["step_index"] for step in pipeline if step["role"] == "suffix"
            ],
        },
        "backend": resolved["execution_backend"],
        "dispatcher": resolved["dispatcher"],
        "dispatch_support": dispatch_support,
        "execution_semantics": semantics,
        "domain_execution": _domain_execution(resolved),
        "source_refs": deepcopy(resolved["source_refs"]),
    }


def _validate_sources(
    layered_document: Mapping[str, Any], resolved_document: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    layered_rows = layered_document.get("implementations")
    resolved_rows = resolved_document.get("resolved_implementations")
    _require(isinstance(layered_rows, list), "layered catalog lacks implementations")
    _require(isinstance(resolved_rows, list), "resolved catalog lacks resolved_implementations")
    _require(len(layered_rows) == _EXPECTED_ALL_COUNT, "layered catalog is not exact102")
    _require(len(resolved_rows) == _EXPECTED_ALL_COUNT, "resolved catalog is not exact102")
    layered = {row["implementation_id"]: row for row in layered_rows}
    resolved = {row["implementation_id"]: row for row in resolved_rows}
    _require(len(layered) == _EXPECTED_ALL_COUNT, "layered IDs are duplicated")
    _require(len(resolved) == _EXPECTED_ALL_COUNT, "resolved IDs are duplicated")
    _require(set(layered) == set(resolved), "layered/resolved ID sets differ")
    _require(
        layered_document.get("layer_counts") == _EXPECTED_LAYER_COUNTS,
        "layered catalog counts drifted",
    )
    _require(
        set(DISABLED_POLICY) == {"A106", "A107", "A108", "A109", "A110"},
        "disabled policy must remain exact PPMd/DMC/PAQ8/CMIX-LSTM/NNCP set",
    )
    _require(set(V1_CORE_IDS).isdisjoint(DISABLED_POLICY), "v1 core includes disabled ID")
    _require(set(CPP_SPECIAL_BRIDGE_IDS).isdisjoint(DISABLED_POLICY), "special bridge includes disabled ID")
    frozen_special = {
        implementation_id
        for implementation_id, row in resolved.items()
        if row.get("dispatcher") == SPECIAL_BRIDGE_DISPATCHER
    }
    _require(
        frozen_special == set(CPP_SPECIAL_BRIDGE_IDS),
        "frozen/C++ special bridge sets differ",
    )
    for implementation_id, row in resolved.items():
        _require(row.get("applicable") is True, f"{implementation_id}: frozen row not applicable")
        _require(
            row.get("source_refs") == layered[implementation_id].get("source_refs")
            or implementation_id == "A110",
            f"{implementation_id}: source refs differ between catalogs",
        )
    return layered, resolved


def load_batch_catalog() -> dict[str, Any]:
    """Return a detached deterministic JSON-primitives-only exact102 plan."""
    layered_document, layered_sha = _load_json(LAYERED_CATALOG)
    resolved_document, resolved_sha = _load_json(RESOLVED_CATALOG)
    layered, resolved = _validate_sources(layered_document, resolved_document)

    ordered_ids = [row["implementation_id"] for row in layered_document["implementations"]]
    operators = [_operator_row(resolved[aid], layered[aid]) for aid in ordered_ids]
    eligible = [row for row in operators if row["selection"]["selectable"]]
    disabled = [row for row in operators if not row["selection"]["selectable"]]
    _require(len(eligible) == _EXPECTED_ELIGIBLE_COUNT, "eligible count is not exact97")
    _require(len(disabled) == len(DISABLED_POLICY), "disabled count is not exact5")

    batch_ids: dict[str, list[str]] = {
        batch_id: [
            row["implementation_id"]
            for row in eligible
            if row["batch_id"] == batch_id
        ]
        for batch_id in BATCH_ORDER
    }
    batch_counts = {key: len(value) for key, value in batch_ids.items()}
    _require(batch_counts == _EXPECTED_BATCH_COUNTS, "batch partition counts drifted")
    _require(
        sum(batch_counts.values()) == _EXPECTED_ELIGIBLE_COUNT,
        "batch partition does not cover exact97",
    )

    eligible_layer_counts = {
        layer: sum(row["layer"] == layer for row in eligible)
        for layer in _EXPECTED_LAYER_COUNTS
    }
    _require(
        eligible_layer_counts == _EXPECTED_ELIGIBLE_LAYER_COUNTS,
        "eligible layer counts drifted",
    )

    special_source_sha = _sha256(SPECIAL_BRIDGE_SOURCE.read_bytes())
    source_catalogs = [
        {
            "role": "layer_and_trace_authority",
            "path": str(LAYERED_CATALOG.relative_to(ROOT)),
            "schema": layered_document.get("schema"),
            "sha256": layered_sha,
        },
        {
            "role": "frozen_execution_pipeline_authority",
            "path": str(RESOLVED_CATALOG.relative_to(ROOT)),
            "schema": resolved_document.get("schema"),
            "sha256": resolved_sha,
        },
    ]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_registry": POLICY_REGISTRY,
        "scope": (
            "local metadata plan for bounded real-data harness validation; "
            "all codec execution belongs to the C++ runtime"
        ),
        "execution_language": "C++",
        "remote_execution": False,
        "source_catalogs": source_catalogs,
        "source_hashes": {
            item["path"]: item["sha256"] for item in source_catalogs
        },
        "counts": {
            "all_trace_entries": len(operators),
            "eligible": len(eligible),
            "disabled_by_policy": len(disabled),
            "eligible_layers": eligible_layer_counts,
            "batches": batch_counts,
        },
        "policy": {
            "disabled_ids": list(DISABLED_POLICY),
            "runtime_override_allowed": False,
            "nncp_state": "disabled_by_policy",
        },
        "batch_order": list(BATCH_ORDER),
        "batches": [
            {
                "batch_id": batch_id,
                "implementation_count": len(batch_ids[batch_id]),
                "implementation_ids": batch_ids[batch_id],
            }
            for batch_id in BATCH_ORDER
        ],
        "special_bridge_contract": {
            "dispatcher": SPECIAL_BRIDGE_DISPATCHER,
            "cpp_source": str(SPECIAL_BRIDGE_SOURCE.relative_to(ROOT)),
            "cpp_source_sha256": special_source_sha,
            "supported_ids": list(CPP_SPECIAL_BRIDGE_IDS),
            "unsupported_frozen_dispatcher_ids": [],
            "support_set_is_exact": True,
            "separate_dispatchers": {
                "source_exact_ids": [
                    row["implementation_id"]
                    for row in operators
                    if row["dispatcher"] == SOURCE_EXACT_DISPATCHER
                ],
                "residual_dependency_ids": [
                    row["implementation_id"]
                    for row in operators
                    if row["dispatcher"] == RESIDUAL_DEPS_DISPATCHER
                ],
            },
            "note": (
                "source-exact and residual-dependency dispatchers are separate; "
                "they are not counted as operator102_special_bridge_v1 support"
            ),
        },
        "domain_fallback_accounting": {
            "wrapper_id": _FALLBACK_WRAPPER_ID,
            "implementation_ids": [
                row["implementation_id"]
                for row in operators
                if row["domain_execution"]["measurement_wrapper"] is not None
            ],
            "wrapper_counts_as_operator": False,
        },
        "operators": operators,
    }
    document["catalog_sha256"] = _sha256(_canonical_bytes(document))
    # Normalize tuples and mappings and guarantee callers cannot mutate module
    # constants through a returned object.
    return json.loads(json.dumps(document, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    document = load_batch_catalog()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "all": document["counts"]["all_trace_entries"],
                "eligible": document["counts"]["eligible"],
                "disabled": document["counts"]["disabled_by_policy"],
                "catalog_sha256": document["catalog_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
