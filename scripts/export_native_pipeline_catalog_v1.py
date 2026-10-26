#!/usr/bin/env python3
"""Freeze the complete production catalog; not a native capability claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import weight_conditioned_graph_task_v1 as graph


def export(destination: Path) -> dict:
    payload = {
        "schema": graph.CATALOG_SCHEMA,
        "state_feature_schema": graph.STATE_FEATURE_SCHEMA,
        "typed_graph_registry": graph.graph_runtime.CURRENT_GRAPH_REGISTRY,
        "operator_directory_registry": graph.operator_implementation_catalog.REGISTRY_VERSION,
        "operator_directory_sha256": graph.operator_implementation_catalog.MANIFEST_SHA256,
        "expected_operator_pdf_sha256": graph.EXPECTED_OPERATOR_PDF_SHA256,
        "internal_parent_adapters": {
            consumer: {"adapter": adapter, "required_policy_parents": list(parents)}
            for consumer, (adapter, parents) in sorted(graph._INTERNAL_PARENT_ADAPTERS.items())
        },
        "actions": [row.json_record() for row in graph.ACTION_CATALOG],
    }
    fingerprint = hashlib.sha256(graph._canonical_json(payload)).hexdigest()
    expected = "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd"
    if (graph.ACTION_COUNT, graph.STOP_ACTION, fingerprint, graph.REGISTRY_FINGERPRINT) != (
            6341, 0, expected, expected):
        raise ValueError("catalog changed; explicit native mapping review is required")
    supported = [0]
    for slot in range(32):
        for offset, view in ((100, "bytes"), (101, "int64")):
            action = offset + 195 * slot
            row = graph.ACTION_CATALOG[action]
            if (row.runtime_action, row.target_slot, row.input_view,
                    row.parameter_items, row.parameter_mode, row.input_byte_multiple,
                    row.input_nonempty) != (
                    "op.bitpack_int.max_value_width.cpp_v1", slot, view, (), "none", 8, True):
                raise ValueError(f"native bitpack mapping mismatch at action {action}")
            supported.append(action)
    document = {
        "schema": "native-pipeline-v1-frozen-complete-action-catalog",
        "registry_fingerprint": fingerprint,
        "action_count": graph.ACTION_COUNT,
        "stop_action": graph.STOP_ACTION,
        "registry_payload": payload,
        "native_pipeline_development_supported_action_ids": supported,
        "scope": "Full identity/provenance artifact, not full execution or formal-training readiness.",
    }
    raw = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True,
                      allow_nan=False) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return {"action_count": graph.ACTION_COUNT, "registry_fingerprint": fingerprint,
            "native_development_action_count": len(supported),
            "artifact_bytes": len(raw), "artifact_sha256": hashlib.sha256(raw).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json")
    print(json.dumps(export(parser.parse_args().output), sort_keys=True))
