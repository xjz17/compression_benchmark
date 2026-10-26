#!/usr/bin/env python3
"""Derive the exact102 operator-tree catalog with pure typed-plane A048.

The v2 catalog remains immutable and compatible.  This metadata-only builder
changes one eligible row: A048 becomes the direct typed uint64 to independent
byte-plane stage.  Codec execution stays in C++.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CATALOG = (
    ROOT / "benchmark/reference/actions/operator_harness_batch_catalog_v2.json"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmark/reference/actions/operator_harness_tree_catalog_v3.json"
)

SCHEMA = "operator-harness-tree-catalog-v3"
POLICY_REGISTRY = "operator-harness-tree-v3"
A048_ACTION = "harness.a048.byte_planes.v1"
A048_PROFILE = "uint64"
A048_DEFAULT_BIT_WIDTH = "64"
DISABLED_IDS = ["A106", "A107", "A108", "A109", "A110"]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_base(document: Mapping[str, Any]) -> None:
    _require(
        document.get("schema") == "operator-harness-batch-catalog-v2",
        "base catalog has the wrong schema",
    )
    counts = document.get("counts", {})
    _require(
        counts.get("all_trace_entries") == 102
        and counts.get("eligible") == 97
        and counts.get("disabled_by_policy") == 5,
        "base catalog is not exact102/exact97/exact5",
    )
    _require(
        document.get("policy", {}).get("disabled_ids") == DISABLED_IDS
        and document.get("policy", {}).get("runtime_override_allowed") is False,
        "base exclusion policy drifted",
    )
    operators = document.get("operators")
    _require(isinstance(operators, list) and len(operators) == 102,
             "base catalog does not contain 102 operators")
    ids = [row.get("implementation_id") for row in operators]
    _require(len(set(ids)) == 102 and ids.count("A048") == 1,
             "base operator IDs are invalid")
    stored_sha = document.get("catalog_sha256")
    unhashed = deepcopy(dict(document))
    unhashed.pop("catalog_sha256", None)
    _require(
        stored_sha == _sha256(_canonical_bytes(unhashed)),
        "base catalog canonical content hash mismatch",
    )


def _a048_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(row))
    _require(result.get("implementation_id") == "A048", "wrong A048 row")
    _require(result.get("selection", {}).get("selectable") is True,
             "A048 must remain eligible")
    result.update(
        {
            "initial_kind": "numeric",
            "initial_width": 8,
            "input_view": "uint64",
            "input_adapter_kind": "canonical_uint64_or_raw_bit_pattern",
            "present_pipeline": [
                {
                    "action": A048_ACTION,
                    "profile": A048_PROFILE,
                    "parameters": {"bit_width": A048_DEFAULT_BIT_WIDTH},
                    "role": "focal",
                    "step_index": 0,
                }
            ],
            "focal_step_index": 0,
            "pipeline_partitions": {
                "prefix_step_indexes": [],
                "focal_step_index": 0,
                "suffix_step_indexes": [],
            },
            "backend": "operator_harness_v2_typed_plane_cpp",
            "dispatcher": "operator_harness_v2_policy_dispatch",
            "dispatch_support": {
                "kind": "cpp_pure_typed_plane_stage",
                "support_status": "supported",
                "dependency_scope": (
                    "operator_harness_v2_a048_typed_planes_and_existing_"
                    "buff_split_join_kernel"
                ),
            },
            "execution_semantics": {
                "catalog_layer": "mechanism_trace",
                "pipeline": "single_focal_stage",
                "v1_registry_binding": "direct_typed_cpp_stage_binding",
                "execution_variant": "pure_typed_plane",
            },
            "domain_execution": {
                "strict_core_action": A048_ACTION,
                "measurement_wrapper": None,
                "wrapper_counts_as_operator": False,
                "support_accounting": "strict_core_only",
            },
            "harness_parameter_domains": {
                "bit_width": {
                    "type": "canonical_decimal_integer_string",
                    "minimum": 0,
                    "maximum": 64,
                    "default": A048_DEFAULT_BIT_WIDTH,
                    "step_index": 0,
                    "action": A048_ACTION,
                    "profile": A048_PROFILE,
                }
            },
            "source_refs": [
                "cpp/src/learning_codec_exact_stages.cpp:1112-1153",
                "cpp/operator_harness_v2/a048_typed_planes.cpp",
            ],
        }
    )
    return result


def load_tree_catalog() -> dict[str, Any]:
    base_raw = BASE_CATALOG.read_bytes()
    base = json.loads(base_raw)
    _require(isinstance(base, dict), "base catalog must be an object")
    _validate_base(base)

    document = deepcopy(base)
    document.pop("catalog_sha256", None)
    document["schema"] = SCHEMA
    document["policy_registry"] = POLICY_REGISTRY
    document["scope"] = (
        "exact102 operator-tree catalog for bounded and stress validation; "
        "all kernels and timed execution are C++"
    )
    document["remote_execution"] = False
    document["catalog_derivation"] = {
        "base_path": str(BASE_CATALOG.relative_to(ROOT)),
        "base_file_sha256": _sha256(base_raw),
        "base_catalog_content_sha256": base["catalog_sha256"],
        "changed_implementation_ids": ["A048"],
        "change": "replace composite BUFF prefix with pure typed-plane stage",
    }
    document["operator_tree_contract"] = {
        "terminology": "operator_tree",
        "runtime_internal_representation": "native_graph_v1",
        "recipes_are_rooted_ordered_operator_trees": True,
        "terminal_step": "stop",
    }
    document["parameterized_operator_ids"] = ["A048"]
    document["operators"] = [
        _a048_row(row) if row["implementation_id"] == "A048" else deepcopy(row)
        for row in base["operators"]
    ]

    for row in document["operators"]:
        if row["implementation_id"] != "A048":
            _require(
                "harness_parameter_domains" not in row,
                f"{row['implementation_id']}: unexpected parameter domain",
            )
    _require(
        sum(row["selection"]["selectable"] for row in document["operators"]) == 97,
        "v3 eligible count drifted",
    )
    _require(
        {
            row["implementation_id"]
            for row in document["operators"]
            if not row["selection"]["selectable"]
        }
        == set(DISABLED_IDS),
        "v3 disabled set drifted",
    )
    document["catalog_sha256"] = _sha256(_canonical_bytes(document))
    return json.loads(json.dumps(document, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    document = load_tree_catalog()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "schema": document["schema"],
                "all": document["counts"]["all_trace_entries"],
                "eligible": document["counts"]["eligible"],
                "disabled": document["counts"]["disabled_by_policy"],
                "catalog_sha256": document["catalog_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
