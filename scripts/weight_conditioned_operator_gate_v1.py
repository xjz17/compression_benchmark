#!/usr/bin/env python3
"""Audit the PDF-aligned 102-entry directory before graph-policy training.

This report keeps trace entries, mechanisms, profiles, recipes and adapters
separate.  A probability provider is selectable only with a matching complete
entropy path and independent cold decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import operator_implementation_catalog as catalog
import operator_taxonomy_v30_v7 as taxonomy
import primitive_operator_action_runtime as policy


SCHEMA = "weight-conditioned-operator-gate-v1"
EXPECTED_PDF_SHA256 = "d32fb153988d1e0afe702acbf0279a14183ef38ca130e157a9c7a846a3f020f7"
ROOT = Path(__file__).resolve().parents[1]
PDF_FILENAME = "30类算子_102项实现_审查定稿.pdf"
DEFAULT_PDF = ROOT / "docs" / PDF_FILENAME
DESKTOP_PDF = Path.home() / "Desktop/benchmark_paper" / PDF_FILENAME
DIRECT_GRAPH_BRIDGES = {
    "A035": ("op.trailing_zero_pack.elf_plus_residual.cpp_v1",
             ROOT / "cpp/build/residual_window_pack_cli"),
    "A111": ("op.outlier_separation.high_bits.uint32.cpp_v1",
             ROOT / "cpp/build/exception_separation_cli"),
}
LAYOUT_MECHANISMS = {
    "A103": "float_deconstruct",
    "A104": "bitsplit",
    "A105": "bitunpack",
}
PROBABILITY_ONLY = {"A106", "A107", "A108", "A109"}
PROBABILITY_MECHANISMS = {*PROBABILITY_ONLY, "A110"}
PROBABILITY_PROVIDER_KEYS = {
    "A106": "ppmd",
    "A107": "dmc",
    "A108": "paq",
    "A109": "cmix",
    "A110": "nncp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_pdf_path(pdf: Path | None = None) -> Path:
    if pdf is not None:
        return Path(pdf).expanduser()
    override = os.environ.get("WEIGHTED_GRPO_OPERATOR_PDF")
    if override is not None:
        return Path(override).expanduser()
    # An invalid explicit or bundled PDF must fail its hash gate, not fall back.
    if DEFAULT_PDF.exists() or not DESKTOP_PDF.is_file():
        return DEFAULT_PDF
    return DESKTOP_PDF


def _nncp() -> dict[str, Any]:
    try:
        from build_nncp_probability import nncp_availability
        return nncp_availability()
    except Exception as error:  # Capability reporting must not hide the cause.
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _prediction_entropy() -> dict[str, Any]:
    try:
        from prediction_entropy_codec import availability
        return availability()
    except Exception as error:
        return {"binary_exists": False, "validated_current_build": False,
                "providers": {}, "reason": f"{type(error).__name__}: {error}"}


def _nncp_graph_entropy() -> dict[str, Any]:
    try:
        from nncp_entropy_codec import availability
        return availability()
    except Exception as error:
        return {"graph_eligible": False, "cold_decode_validated": False,
                "reason": f"{type(error).__name__}: {error}"}


def build_report(pdf: Path | None = None) -> dict[str, Any]:
    pdf = resolve_pdf_path(pdf)
    policy_names = {item.name for item in policy.POLICY_ACTIONS}
    graph_names = {item.name for item in taxonomy.OPERATOR_ACTIONS}
    layout_binary = ROOT / "cpp/build/openzl_layout_stages_cli"
    nncp = _nncp()
    nncp_graph = _nncp_graph_entropy()
    entropy = _prediction_entropy()
    entries = []
    for item in catalog.IMPLEMENTATIONS:
        implementation_id = item["implementation_id"]
        layer = item["accounting"]["layer"]
        names = tuple(item.get("source_action_names", ()))
        row: dict[str, Any] = {
            "implementation_id": implementation_id,
            "layer": layer,
            "family_numbers": item["family_numbers"],
            "source_action_names": list(names),
            "mechanism_ids": item["accounting"]["mechanism_ids"],
            "policy_registered_names": [name for name in names if name in policy_names],
            "graph_registered_names": [name for name in names if name in graph_names],
            "selectable": False,
        }
        if layer != "mechanism":
            row["status"] = {
                "profile": "profile_preserved_not_atomic_action",
                "recipe": "recipe_or_complete_backend_preserved",
                "adapter": "framing_or_compatibility_preserved",
            }[layer]
        elif implementation_id in DIRECT_GRAPH_BRIDGES:
            name, binary = DIRECT_GRAPH_BRIDGES[implementation_id]
            row.update(selectable=binary.is_file(), status=(
                "direct_graph_bridge_ready" if binary.is_file()
                else "direct_graph_bridge_binary_missing"), direct_graph_action=name,
                required_binary=str(binary), formal_evidence_ready=False,
                formal_evidence_status="binary_present_but_no_current_per_mechanism_probe_receipt")
        elif implementation_id in LAYOUT_MECHANISMS:
            row.update(selectable=layout_binary.is_file(), status=(
                "layout_graph_action_ready" if layout_binary.is_file()
                else "layout_graph_binary_missing"),
                direct_graph_action=LAYOUT_MECHANISMS[implementation_id],
                required_binary=str(layout_binary), formal_evidence_ready=False,
                formal_evidence_status="binary_present_but_no_current_per_mechanism_probe_receipt")
        elif implementation_id in PROBABILITY_ONLY:
            provider = PROBABILITY_PROVIDER_KEYS[implementation_id]
            evidence = entropy.get("providers", {}).get(provider, {})
            eligible = bool(evidence.get("graph_eligible"))
            row.update(
                selectable=eligible,
                status=("complete_prediction_entropy_graph_path_ready" if eligible else
                        "prediction_entropy_path_unvalidated_for_graph_policy"),
                formal_evidence_ready=eligible,
                blocker=(None if eligible else
                         "complete graph path lacks a current bit-exact cold-decode receipt"),
                direct_graph_action=names[0] if names else None,
                prediction_entropy=entropy,
                probability_evidence={
                    "provider": provider,
                    "provider_api_implemented": True,
                    "standalone_reversible_entropy_path": bool(
                        evidence.get("complete_reversible_path")),
                    "independent_cold_decode_validated": bool(
                        evidence.get("cold_decode_validated")),
                    "graph_entropy_integrated": bool(evidence.get("graph_eligible")),
                    "formal_policy_eligible": eligible,
                },
            )
        elif implementation_id == "A110":
            # A110 is admitted only as its complete NNP1 terminal entropy path.
            # It is never relabeled as an atomic probability vector transform.
            validated = bool(nncp.get("validated_current_build"))
            eligible = bool(nncp_graph.get("graph_eligible"))
            row.update(
                selectable=eligible,
                status=(
                    "complete_nncp_nnp1_graph_path_ready"
                    if eligible else "complete_nncp_nnp1_graph_path_unvalidated"
                ),
                formal_evidence_ready=eligible,
                blocker=(None if eligible else
                         "complete NNP1 typed-graph path lacks a current bit-exact cold-decode receipt"),
                direct_graph_action=names[0] if names else None,
                nncp=nncp,
                nncp_graph_entropy=nncp_graph,
                probability_evidence={
                    "provider": PROBABILITY_PROVIDER_KEYS[implementation_id],
                    "provider_api_implemented": True,
                    "standalone_reversible_entropy_path": True,
                    "independent_cold_decode_validated": validated,
                    "standalone_runtime_available": bool(nncp.get("available")),
                    "graph_path_role": "complete_nnp1_terminal_entropy_path",
                    "graph_entropy_integrated": eligible,
                    "formal_policy_eligible": eligible,
                },
            )
        elif row["graph_registered_names"] and row["policy_registered_names"]:
            row.update(selectable=True, status="registered_graph_and_policy_action_unprobed",
                       formal_evidence_ready=False,
                       formal_evidence_status="no_current_per_mechanism_bit_exact_probe_receipt")
        else:
            row.update(status="missing_graph_or_policy_registration",
                       formal_evidence_ready=False)
        entries.append(row)

    mechanisms = [row for row in entries if row["layer"] == "mechanism"]
    blocked = [row for row in mechanisms if not row["selectable"]]
    evidence_blocked = [
        row for row in mechanisms
        if row["selectable"] and not row.get("formal_evidence_ready", False)
    ]
    non_probability_blocked = [
        row for row in blocked if row["implementation_id"] not in PROBABILITY_MECHANISMS]
    pdf_status: dict[str, Any] = {"path": str(pdf), "exists": pdf.is_file()}
    if pdf.is_file():
        pdf_sha256 = sha256(pdf)
        pdf_status.update(sha256=pdf_sha256, matches_expected=pdf_sha256 == EXPECTED_PDF_SHA256)
    else:
        pdf_status.update(sha256=None, matches_expected=False)
    return {
        "schema": SCHEMA,
        "catalog_registry_version": catalog.REGISTRY_VERSION,
        "catalog_sha256": catalog.MANIFEST_SHA256,
        "pdf": pdf_status,
        "counts": {
            "trace_entries": catalog.TRACE_ENTRY_COUNT,
            "mechanisms": catalog.MECHANISM_COUNT,
            "layers": catalog.LAYER_COUNTS,
            "operator_groups": 30,
            "special_operations": 3,
            "selectable_mechanisms": sum(row["selectable"] for row in mechanisms),
            "blocked_mechanisms": len(blocked),
            "formally_validated_mechanisms": sum(
                bool(row.get("formal_evidence_ready")) for row in mechanisms
            ),
            "formal_evidence_blockers": len(evidence_blocked),
        },
        "structural_gate_passed": (not non_probability_blocked and pdf_status["matches_expected"]),
        "formal_training_gate_passed": (
            not blocked and not evidence_blocked and pdf_status["matches_expected"]
        ),
        "formal_training_blockers": [
            {"implementation_id": row["implementation_id"], "status": row["status"],
             "blocker": row.get("blocker")}
            for row in blocked
        ] + [
            {"implementation_id": row["implementation_id"], "status": row["status"],
             "blocker": row.get(
                 "formal_evidence_status",
                 "selectable runtime lacks a current per-mechanism bit-exact receipt",
             )}
            for row in evidence_blocked
        ],
        "probability_training_scope": {
            "requested_mechanisms": sorted(PROBABILITY_MECHANISMS),
            "eligible_mechanisms": sorted(
                row["implementation_id"] for row in mechanisms
                if row["implementation_id"] in PROBABILITY_MECHANISMS and row["selectable"]),
            "reduced_scope_requires_explicit_plan_change": True,
            "note": (
                "A106-A108 use the validated PEC1 causal prediction plus arithmetic-coding graph "
                "path. A109 is eligible only while its explicit full-cmix large-memory cold-decode "
                "receipt matches the current binary and sources. A110 is eligible only through its "
                "distinct complete NNP1 terminal entropy path and matching graph receipt."
            ),
        },
        "claim_boundary": (
            "registered or source-present is not execution evidence; formal training remains blocked "
            "until every selectable mechanism has a successful bit-exact probe and every probability "
            "provider used by the policy has a matching entropy path and cold decoder"
        ),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-formal-gate", action="store_true")
    args = parser.parse_args()
    report = build_report(args.pdf)
    encoded = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return int(args.require_formal_gate and not report["formal_training_gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
