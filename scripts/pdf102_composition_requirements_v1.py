#!/usr/bin/env python3
"""Traceable composition requirements from the reviewed 102-operator handbook.

This module describes *documented candidates*.  It deliberately does not
produce a runtime action mask: exact-value compatibility, numeric adaptation,
packet-specific hints, state links, and byte typed views have different
requirements and remain distinguishable in the returned records.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "operator_102"
CATALOG_PATH = ROOT / "benchmark" / "reference" / "actions" / "operator_implementations_102.json"
LAYERED_CATALOG_PATH = ROOT / "benchmark" / "reference" / "actions" / "operator_implementations_layered_v1.json"
REVIEW_VALIDATION_PATH = DOCS / "review_pdf_validation.json"
HANDBOOK_VALIDATION_PATH = DOCS / "pdf_validation.json"

# flow_compatibility is the source used to render the reviewed per-output
# compatibility chapter.  It lives beside generated documentation rather than
# in a Python package, so make that location importable without changing it.
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))
import flow_compatibility as flow  # noqa: E402
import dependency_appendix  # noqa: E402


SCHEMA_VERSION = "pdf102-composition-requirements-v1"
EXCLUDED = {
    "A109": "cmix excluded by the current experiment scope because it is too slow",
    "A110": "NNCP excluded by the current experiment scope because it is too slow",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _aid_number(aid: str) -> int:
    return int(aid[1:])


def _contract_source(aid: str) -> Path:
    number = _aid_number(aid)
    if aid in {"A110", "A111"}:
        return DOCS / "interface_contract_revision.py"
    if number <= 40:
        return DOCS / "stream_contracts_001_040.json"
    if number <= 76:
        return DOCS / "flow_contracts_041_076.json"
    return DOCS / "source_inputs" / "flow_contracts_077_111.json"


def _validation_output(path: Path, path_suffix: str | None = None) -> dict[str, Any]:
    outputs = _load_json(path)["outputs"]
    if path_suffix is None:
        if len(outputs) != 1:
            raise ValueError(f"expected one output in {path}")
        return outputs[0]
    selected = [row for row in outputs if str(row["path"]).endswith(path_suffix)]
    if len(selected) != 1:
        raise ValueError(f"cannot resolve {path_suffix!r} in {path}")
    return selected[0]


def _resolved_document(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha(path)
    if actual_sha != row["sha256"]:
        raise ValueError(f"PDF hash differs from reviewed receipt: {path}")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "pages": int(row["pages"]),
        "receipt": str(REVIEW_VALIDATION_PATH if row["pages"] > 200 else HANDBOOK_VALIDATION_PATH),
    }


def _matching_accepts(target: dict[str, Any], kind: str, edge_kind: str) -> list[tuple[int, dict[str, Any]]]:
    matches: list[tuple[int, dict[str, Any]]] = []
    for index, accepted in enumerate(target["accepts"]):
        types = set(accepted["types"])
        mode = accepted["mode"]
        if edge_kind == "direct" and mode == "value" and (
            kind in types or (kind.startswith("struct:") and "struct" in types)
        ):
            matches.append((index, accepted))
        elif edge_kind == "adapted" and mode == "value" and flow.is_integer(kind) and any(
            flow.is_integer(item) for item in types
        ) and kind not in types:
            matches.append((index, accepted))
        elif edge_kind == "state" and mode == "state" and kind in types:
            matches.append((index, accepted))
        elif edge_kind == "view" and mode == "value" and kind == "bytes" and "u8" in types and "bytes" not in types:
            matches.append((index, accepted))
    return matches


def _edge_id(edge: dict[str, Any]) -> str:
    identity = {
        key: edge[key]
        for key in (
            "from",
            "to",
            "kind",
            "producer_port",
            "consumer_accept_index",
            "types",
            "accepted_types",
            "condition",
        )
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "pdf102-" + sha256(payload).hexdigest()[:20]


def _scope_fields(source: str, target: str) -> dict[str, Any]:
    excluded = list(dict.fromkeys(aid for aid in (source, target) if aid in EXCLUDED))
    return {
        "allowed_by_scope": not excluded,
        "scope_exclusions": [{"id": aid, "reason": EXCLUDED[aid]} for aid in excluded],
    }


def _base_edge(
    producer: dict[str, Any],
    output: dict[str, Any],
    target: dict[str, Any],
    *,
    kind: str,
    types: list[str],
    accepted_types: list[str],
    consumer_accept_index: int | None,
    target_requires: str,
    condition: str,
    review_bookmarks: dict[str, int],
    handbook_bookmarks: dict[str, int],
) -> dict[str, Any]:
    edge = {
        "from": producer["id"],
        "to": target["id"],
        "direction": "forward",
        "kind": kind,
        "relation": {
            "direct": "exact_value_type_candidate",
            "adapted": "explicit_lossless_numeric_adaptation_required",
            "hint": "documented_specific_consumer",
            "state": "state_channel_candidate",
            "view": "explicit_reversible_typed_view_required",
        }[kind],
        "producer_port": output["name"],
        "producer_role": output["role"],
        "producer_output_types": list(output["types"]),
        "types": sorted(types),
        "accepted_types": sorted(accepted_types),
        "consumer_accept_index": consumer_accept_index,
        "consumer_input": target["input_text"],
        "condition": condition,
        "producer_detail": output["detail"],
        "consumer_requires": target_requires,
        "runtime_status": "documented_candidate_unverified",
        "runtime_legal": False,
        "source": {
            "contract": str(_contract_source(producer["id"]).relative_to(ROOT)),
            "producer_evidence": list(producer["evidence"]),
            "consumer_contract": str(_contract_source(target["id"]).relative_to(ROOT)),
            "consumer_evidence": list(target["evidence"]),
            "review_pdf_operator_page": review_bookmarks[producer["id"]],
            "review_pdf_compatibility_page": review_bookmarks["flow-" + producer["id"]],
            "consumer_pdf_operator_page": review_bookmarks[target["id"]],
            "consumer_pdf_compatibility_page": review_bookmarks["flow-" + target["id"]],
            "handbook_pdf_operator_page": handbook_bookmarks[producer["id"]],
        },
        **_scope_fields(producer["id"], target["id"]),
    }
    edge["edge_id"] = _edge_id(edge)
    return edge


def _deduplicate(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for edge in edges:
        prior = by_id.get(edge["edge_id"])
        if prior is not None and prior != edge:
            raise ValueError(f"edge id collision: {edge['edge_id']}")
        by_id[edge["edge_id"]] = edge
    return sorted(
        by_id.values(),
        key=lambda edge: (
            _aid_number(edge["from"]),
            edge["producer_port"],
            edge["kind"],
            _aid_number(edge["to"]),
            edge["consumer_accept_index"] if edge["consumer_accept_index"] is not None else -1,
            edge["edge_id"],
        ),
    )


def _dependency_pairs(record: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Expand only directions explicitly represented by an appendix recipe."""
    spec = record["pair_or_sequence"]
    kind = spec["kind"]
    sequences: list[tuple[str | None, list[str]]] = []
    if kind == "pair":
        sequences.append((None, spec["sequence"]))
    elif kind == "alternative_pairs":
        sequences.extend((None, pair) for pair in spec["pairs"])
        if spec.get("other_current_pair"):
            sequences.append((None, spec["other_current_pair"]))
    elif kind == "sequence":
        sequences.append((None, spec["sequence"]))
    elif kind == "method_sequences":
        sequences.extend((name, sequence) for name, sequence in spec["sequences"].items())
    elif kind == "ordered_record_dispatch":
        # The two middle actions are per-record alternatives sharing state.
        return [
            ("A004", "A031", "literal branch"),
            ("A004", "A028", "decimal branch"),
            ("A031", "A076", "literal branch rejoin"),
            ("A028", "A076", "decimal branch rejoin"),
        ]
    elif kind in {"branched_recipe", "event_sequence"}:
        # A111's three branches already occur as exact, port-specific hints in
        # the flow contracts.  The probability event sequence has no entropy
        # consumer A-ID, so both remain in dependency_requirements rather than
        # inventing a misleading sequential edge.
        return []
    else:
        raise ValueError(f"unknown dependency shape {kind!r}")

    pairs: list[tuple[str, str, str | None]] = []
    for context, sequence in sequences:
        ids = []
        for step in sequence:
            match = re.search(r"A\d{3}", step)
            if not match:
                raise ValueError(f"non-A step in dependency sequence {record['dependency_id']}: {step}")
            ids.append(match.group(0))
        pairs.extend((left, right, context) for left, right in zip(ids, ids[1:]))
    return pairs


def _dependency_edge(
    record: dict[str, Any],
    group: str,
    source: str,
    target: str,
    context: str | None,
    *,
    target_contract: dict[str, Any],
    review_bookmarks: dict[str, int],
    handbook_bookmarks: dict[str, int],
) -> dict[str, Any]:
    ports = record["ports"]
    relevant_ports = [port for port in ports if port.get("operator") == source] or ports
    contracts = [
        port.get("contract") or f"{port['from']} -> {port['to']}"
        for port in relevant_ports
    ]
    scope = record["hard_pruning_eligibility"]
    alternative = record["alternative_predecessor_or_intervening_step"]
    condition_parts = [record["title"], record["reason"], *contracts, scope["scope"]]
    if context:
        condition_parts.insert(1, f"recipe context: {context}")
    condition = " | ".join(condition_parts)
    edge = {
        "from": source,
        "to": target,
        "direction": "forward",
        "kind": "hint",
        "relation": "documented_order_dependency",
        "producer_port": relevant_ports[0]["from"],
        "producer_role": "dependency_contract",
        "producer_output_types": [],
        "types": [],
        "accepted_types": [],
        "consumer_accept_index": None,
        "consumer_input": relevant_ports[0]["to"],
        "condition": condition,
        "producer_detail": record["reason"],
        "consumer_requires": " ; ".join(contracts),
        "runtime_status": "documented_candidate_unverified",
        "runtime_legal": False,
        "dependency": {
            "dependency_id": record["dependency_id"],
            "group": group,
            "recipe_context": context,
            "ports": ports,
            "pair_or_sequence": record["pair_or_sequence"],
            "alternative_predecessor_or_intervening_step": alternative,
            "inverse": record["inverse"],
            "hard_pruning_eligibility": scope,
            "source_evidence": record["source_evidence"],
        },
        "source": {
            "contract": str((DOCS / "dependency_appendix.py").relative_to(ROOT)),
            "producer_evidence": record["source_evidence"],
            "consumer_contract": str(_contract_source(target).relative_to(ROOT)),
            "consumer_evidence": list(target_contract["evidence"]),
            "review_pdf_operator_page": review_bookmarks[source],
            "review_pdf_compatibility_page": review_bookmarks["flow-" + source],
            "consumer_pdf_operator_page": review_bookmarks[target],
            "consumer_pdf_compatibility_page": review_bookmarks["flow-" + target],
            "review_pdf_dependency_page": review_bookmarks["dependency-" + group],
            "handbook_pdf_operator_page": handbook_bookmarks[source],
        },
        **_scope_fields(source, target),
    }
    edge["edge_id"] = _edge_id(edge)
    return edge


def build_pdf_requirements() -> dict[str, Any]:
    """Return the reviewed PDF102 nodes and typed candidate edges.

    ``runtime_legal`` is always false here.  A caller must separately prove
    action registration, carrier availability, inverse wiring, and a lossless
    cold-decode test before admitting an edge to a runtime mask.
    """

    contracts = flow.load_contracts()
    flow.validate_contracts(contracts)
    by_id = {row["id"]: row for row in contracts}

    catalog = _load_json(CATALOG_PATH)
    layered = _load_json(LAYERED_CATALOG_PATH)
    catalog_by_id = {row["implementation_id"]: row for row in catalog["implementations"]}
    layered_by_id = {row["implementation_id"]: row for row in layered["implementations"]}
    expected_ids = set(catalog_by_id)
    if set(by_id) != expected_ids or set(layered_by_id) != expected_ids or len(expected_ids) != 102:
        raise ValueError("102 catalog, layered inventory, and flow contracts differ")

    review_row = _validation_output(REVIEW_VALIDATION_PATH)
    handbook_row = _validation_output(
        HANDBOOK_VALIDATION_PATH, "30类算子_102项实现与86算法组合_独立手册.pdf"
    )
    review_bookmarks = review_row["bookmarks"]
    handbook_bookmarks = handbook_row["bookmarks"]
    for aid in expected_ids:
        if aid not in review_bookmarks or "flow-" + aid not in review_bookmarks or aid not in handbook_bookmarks:
            raise ValueError(f"missing PDF page bookmark for {aid}")

    operators: list[dict[str, Any]] = []
    for aid in sorted(expected_ids, key=_aid_number):
        base = catalog_by_id[aid]
        layer = layered_by_id[aid].get("accounting", {})
        contract = by_id[aid]
        operators.append(
            {
                "id": aid,
                "name": base["name"],
                "family_numbers": list(base["family_numbers"]),
                "classification": base["classification"],
                "layer": layer.get("layer"),
                "mechanism_ids": list(layer.get("mechanism_ids", [])),
                "input_text": contract["input_text"],
                "accepts": contract["accepts"],
                "outputs": contract["outputs"],
                "exposure": contract["exposure"],
                "source_evidence": contract["evidence"],
                "eligible": aid not in EXCLUDED,
                "exclusion_reason": EXCLUDED.get(aid),
                "runtime_status": "catalogued_not_admitted_by_this_document",
                "source": {
                    "contract": str(_contract_source(aid).relative_to(ROOT)),
                    "review_pdf_operator_page": review_bookmarks[aid],
                    "review_pdf_compatibility_page": review_bookmarks["flow-" + aid],
                    "handbook_pdf_operator_page": handbook_bookmarks[aid],
                },
            }
        )

    edges: list[dict[str, Any]] = []
    unbound_state_outputs: list[dict[str, Any]] = []
    for producer in contracts:
        for output in producer["outputs"]:
            links = flow.output_links(output, contracts)
            if output["role"] == "preserve":
                continue

            if output["role"] == "data":
                for kind, target_ids in links["direct"].items():
                    for target_id in target_ids:
                        target = by_id[target_id]
                        for accept_index, accepted in _matching_accepts(target, kind, "direct"):
                            condition = accepted["requires"] or f"exact value type {kind}"
                            edges.append(
                                _base_edge(
                                    producer,
                                    output,
                                    target,
                                    kind="direct",
                                    types=[kind],
                                    accepted_types=accepted["types"],
                                    consumer_accept_index=accept_index,
                                    target_requires=accepted["requires"],
                                    condition=condition,
                                    review_bookmarks=review_bookmarks,
                                    handbook_bookmarks=handbook_bookmarks,
                                )
                            )

                for kind, target_ids in links["adapted"].items():
                    for target_id in target_ids:
                        target = by_id[target_id]
                        for accept_index, accepted in _matching_accepts(target, kind, "adapted"):
                            accepted_numeric = [item for item in accepted["types"] if flow.is_integer(item)]
                            requirement = accepted["requires"] or "target integer value contract"
                            edges.append(
                                _base_edge(
                                    producer,
                                    output,
                                    target,
                                    kind="adapted",
                                    types=[kind],
                                    accepted_types=accepted_numeric,
                                    consumer_accept_index=accept_index,
                                    target_requires=accepted["requires"],
                                    condition=f"explicit lossless {kind} conversion to one of {accepted_numeric}; {requirement}",
                                    review_bookmarks=review_bookmarks,
                                    handbook_bookmarks=handbook_bookmarks,
                                )
                            )

                for target_id in links["view_consumers"]:
                    target = by_id[target_id]
                    for accept_index, accepted in _matching_accepts(target, "bytes", "view"):
                        edges.append(
                            _base_edge(
                                producer,
                                output,
                                target,
                                kind="view",
                                types=["bytes"],
                                accepted_types=["u8"],
                                consumer_accept_index=accept_index,
                                target_requires=accepted["requires"],
                                condition="explicit reversible bytes-to-u8 typed view; " + (
                                    accepted["requires"] or "preserve byte order and length"
                                ),
                                review_bookmarks=review_bookmarks,
                                handbook_bookmarks=handbook_bookmarks,
                            )
                        )

            if output["role"] == "state":
                state_edge_count_before = len(edges)
                for target_id in links["state_consumers"]:
                    target = by_id[target_id]
                    for output_type in output["types"]:
                        for accept_index, accepted in _matching_accepts(target, output_type, "state"):
                            edges.append(
                                _base_edge(
                                    producer,
                                    output,
                                    target,
                                    kind="state",
                                    types=[output_type],
                                    accepted_types=accepted["types"],
                                    consumer_accept_index=accept_index,
                                    target_requires=accepted["requires"],
                                    condition=accepted["requires"] or f"state type {output_type}; shared ordered state required",
                                    review_bookmarks=review_bookmarks,
                                    handbook_bookmarks=handbook_bookmarks,
                                )
                            )
                if len(edges) == state_edge_count_before:
                    unbound_state_outputs.append(
                        {
                            "from": producer["id"],
                            "producer_port": output["name"],
                            "types": list(output["types"]),
                            "condition": output["detail"],
                            "reason": "the reviewed contracts contain no matching state-mode consumer",
                            "runtime_legal": False,
                            "source": {
                                "contract": str(_contract_source(producer["id"]).relative_to(ROOT)),
                                "review_pdf_compatibility_page": review_bookmarks["flow-" + producer["id"]],
                            },
                            **_scope_fields(producer["id"], producer["id"]),
                        }
                    )

            for hint in links["hints"]:
                target = by_id[hint["id"]]
                edges.append(
                    _base_edge(
                        producer,
                        output,
                        target,
                        kind="hint",
                        types=list(output["types"]),
                        accepted_types=sorted({item for accepted in target["accepts"] for item in accepted["types"]}),
                        consumer_accept_index=None,
                        target_requires="",
                        condition=hint["condition"],
                        review_bookmarks=review_bookmarks,
                        handbook_bookmarks=handbook_bookmarks,
                    )
                )

    dependency_requirements: list[dict[str, Any]] = []
    for group in dependency_appendix.GROUP_ORDER:
        for raw_record in dependency_appendix.DEPENDENCY_GROUPS[group]:
            # Normalize tuples and other immutable containers to a JSON-native
            # snapshot so the returned result has no module-owned objects.
            record = json.loads(json.dumps(raw_record, ensure_ascii=False))
            dependency_requirements.append({"group": group, **record})
            for source, target, context in _dependency_pairs(record):
                edges.append(
                    _dependency_edge(
                        record,
                        group,
                        source,
                        target,
                        context,
                        target_contract=by_id[target],
                        review_bookmarks=review_bookmarks,
                        handbook_bookmarks=handbook_bookmarks,
                    )
                )

    edges = _deduplicate(edges)
    kind_counts: dict[str, int] = defaultdict(int, {kind: 0 for kind in ("direct", "adapted", "hint", "state", "view")})
    for edge in edges:
        kind_counts[edge["kind"]] += 1

    reciprocal: set[tuple[str, str]] = set()
    directions = {(edge["from"], edge["to"]) for edge in edges}
    for source, target in directions:
        if source != target and (target, source) in directions:
            reciprocal.add(tuple(sorted((source, target), key=_aid_number)))

    source_files = [
        CATALOG_PATH,
        LAYERED_CATALOG_PATH,
        REVIEW_VALIDATION_PATH,
        HANDBOOK_VALIDATION_PATH,
        DOCS / "flow_compatibility.py",
        DOCS / "dependency_appendix.py",
        DOCS / "handbook_content.py",
        DOCS / "current_handbook.json",
        DOCS / "interface_contract_revision.py",
        *flow.CONTRACT_FILES,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "purpose": "PDF102 documented before/after composition requirements",
            "edge_direction": "from producer output to candidate consumer input",
            "runtime_admission": "No edge is runtime legal solely because it appears here.",
            "edge_kinds": {
                "direct": "exact value-type match",
                "adapted": "integer types differ; an explicit lossless conversion is required",
                "hint": "the handbook names this specific consumer and preserves its condition",
                "state": "state channel; not an ordinary byte/value edge",
                "view": "bytes require an explicit reversible u8 typed view",
            },
            "reciprocal_edges": "Two separately documented directions are retained and are not a direction conflict.",
            "family_expansion": "No family label is expanded into unconditional all-to-all edges.",
        },
        "source_documents": {
            "reviewed_final": _resolved_document(review_row),
            "independent_handbook": _resolved_document(handbook_row),
        },
        "source_hashes": {str(path.relative_to(ROOT)): _sha(path) for path in sorted(set(source_files))},
        "catalog": {
            "operator_count": len(operators),
            "catalog_version": catalog["registry_version"],
            "layered_catalog_version": layered["registry_version"],
            "excluded_but_accounted": dict(EXCLUDED),
            "edge_count": len(edges),
            "edge_kind_counts": dict(sorted(kind_counts.items())),
            "unbound_state_output_count": len(unbound_state_outputs),
        },
        "operators": operators,
        "edges": edges,
        "dependency_requirements": dependency_requirements,
        "unbound_state_outputs": unbound_state_outputs,
        "reciprocal_pairs": [list(pair) for pair in sorted(reciprocal, key=lambda pair: (_aid_number(pair[0]), _aid_number(pair[1])))],
        "conflicts": [],
    }


def main() -> None:
    result = build_pdf_requirements()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
