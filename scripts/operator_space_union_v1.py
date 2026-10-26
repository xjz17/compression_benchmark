#!/usr/bin/env python3
"""Freeze the requested 86-method/PDF102 candidate union without claiming execution.

This is a coverage and candidate-grammar artifact, not a replacement policy
registry.  A document edge may require an adapter, a branch or shared state;
the existence of two action IDs does not prove that the edge can execute.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "operator86-pdf102-candidate-union-v1"
EXCLUDED = frozenset({"A109", "A110"})


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def binding_inventory(layered: Mapping[str, Any], catalog: Mapping[str, Any]) -> list[dict]:
    """Keep all I/O/config variants; never guess a smallest action ID."""
    templates: dict[str, list[dict]] = {}
    for row in catalog["registry_payload"]["actions"]:
        if row.get("category") == "mechanism" and row.get("target_slot") == 0:
            templates.setdefault(row["implementation_id"], []).append(row)
    records = []
    for item in layered["implementations"]:
        identity = item["implementation_id"]
        accounting = item["accounting"]
        parts = accounting.get("mechanism_ids", [])
        direct = templates.get(identity, [])
        part_options = {part: templates.get(part, []) for part in parts}
        if identity in EXCLUDED:
            status = "excluded_by_user"
        elif direct:
            status = "configurations_present_requires_native_qualification"
        elif accounting["layer"] == "profile":
            status = "profile_requires_explicit_parameter_binding"
        elif accounting["layer"] == "recipe":
            status = "recipe_requires_order_branch_and_state_binding"
        else:
            status = "adapter_requires_explicit_packet_binding"
        records.append({
            "implementation_id": identity,
            "accounting": accounting,
            "source_classification": item.get("classification"),
            "status": status,
            "direct_slot0_configurations": direct,
            "component_slot0_configurations": part_options,
            "selectable_proven": False,
            "reason": "Catalog presence does not certify source equivalence, port wiring or cold decode.",
        })
    return records


def seal(payload: dict) -> dict:
    return {"schema": SCHEMA, "space_sha256": digest(payload), "payload": payload}


def validate_document(document: Mapping[str, Any]) -> None:
    require(document.get("schema") == SCHEMA, "candidate union schema mismatch")
    payload = document["payload"]
    require(document.get("space_sha256") == digest(payload), "candidate union fingerprint mismatch")
    obligations = payload["obligations"]
    identifiers = [row["id"] for row in obligations]
    require(len(identifiers) == len(set(identifiers)), "duplicate coverage obligation")
    require(all(row["kind"] in {"method_path", "implementation", "pdf_edge", "state_contract_review", "dependency_review"}
                for row in obligations), "unknown obligation kind would disappear from the queue")
    methods = [row for row in obligations if row["kind"] == "method_path"]
    operators = [row for row in obligations if row["kind"] == "implementation"]
    require(len(methods) == 86, "all 86 original method paths must remain in the denominator")
    require(len({row["label"] for row in methods}) == 86, "duplicate original method label")
    require(all(row["excluded"] is False and row.get("required", True) is True for row in methods),
            "required original method path cannot be excluded or optional")
    require(len(operators) == 102, "all 102 PDF trace entries must remain in the denominator")
    require({row["implementation_id"] for row in operators if row["excluded"]} == EXCLUDED,
            "cmix/NNCP exclusion drift")
    require(payload["online_search"] is False, "candidate grammar cannot introduce online search")
    if "original_method_paths" in payload:
        require({row["label"] for row in methods} ==
                {row["label"] for row in payload["original_method_paths"]},
                "method obligations differ from source paths")
    if "pdf102" in payload:
        require({row["implementation_id"] for row in operators} ==
                {row["id"] for row in payload["pdf102"]["operators"]},
                "implementation obligations differ from PDF102")
        required_edges = [row for row in obligations if row["kind"] == "pdf_edge"]
        require({row["edge_sha256"] for row in required_edges} ==
                {digest(row) for row in payload["pdf102"]["edges"]},
                "PDF port edges silently omitted or collapsed")
        require({row["id"] for row in obligations if row["kind"] == "state_contract_review"} ==
                {"state-review:" + digest(row) for row in payload["pdf102"]["unbound_state_outputs"]},
                "unbound PDF state outputs omitted from review queue")
        require(all(row["excluded"] == (row["from"] in EXCLUDED or row["to"] in EXCLUDED)
                    for row in required_edges), "PDF edge scope exclusion drift")
        require({row["id"] for row in obligations if row["kind"] == "dependency_review"} ==
                {"dependency:" + row["dependency_id"] for row in payload["pdf102"]["dependency_requirements"]},
                "PDF order or shared-state requirements omitted")
        expected = make_obligations(payload["original_method_paths"],
                                    payload["current_registry_bindings"], payload["pdf102"])
        require(obligations == expected, "coverage obligations differ from the canonical source projection")


def coverage_schedule(document: Mapping[str, Any], *, cursor: int, budget: int) -> dict:
    """Round-robin requirement queue, with a resumable monotonically increasing cursor.

    Items are coverage tasks, NOT action IDs or recipes to blindly execute.
    Unwired tasks are included so they cannot disappear behind an executable
    subset.  Actual exploration status belongs in the run's separate ledger.
    """
    validate_document(document)
    require(cursor >= 0 and budget > 0, "cursor must be nonnegative and budget positive")
    groups = []
    for kind in ("method_path", "implementation", "state_contract_review", "dependency_review", "pdf_edge"):
        groups.append(deque(row for row in document["payload"]["obligations"]
                            if row["kind"] == kind and not row["excluded"]))
    order = []
    while any(groups):
        for group in groups:
            if group:
                order.append(group.popleft())
    require(bool(order), "empty coverage queue")
    take = min(budget, len(order))
    selected = [order[(cursor + offset) % len(order)] for offset in range(take)]
    return {"schema": "operator-union-coverage-schedule-v1",
            "space_sha256": document["space_sha256"],
            "queue_kind": "coverage_requirements_not_executable_recipes",
            "cursor": cursor, "next_cursor": cursor + take,
            "cycle_size": len(order), "cycle": cursor // len(order),
            "selected_count": take, "unique_not_selected_this_batch": len(order) - take,
            "obligations": selected}


def make_obligations(methods: Sequence[Mapping[str, Any]], bindings: Sequence[Mapping[str, Any]],
                     pdf: Mapping[str, Any]) -> list[dict]:
    """One canonical projection; mutable queue flags cannot silently shrink scope."""
    obligations = []
    for method in methods:
        label = method["label"]
        obligations.append({"id": "method:" + label, "kind": "method_path", "label": label,
                            "excluded": False, "required": True,
                            "status": "requires_original_algorithm_native_path_qualification"})
    for row in bindings:
        identity = row["implementation_id"]
        obligations.append({"id": "implementation:" + identity, "kind": "implementation",
                            "implementation_id": identity, "excluded": identity in EXCLUDED,
                            "required": identity not in EXCLUDED, "status": row["status"]})
    for edge in pdf["edges"]:
        # Full condition/port identity is part of the key. Two different ports
        # connecting the same A IDs must not collapse to an untyped edge.
        key = digest(edge)
        excluded = edge["from"] in EXCLUDED or edge["to"] in EXCLUDED
        obligations.append({"id": "pdf-edge:" + key, "kind": "pdf_edge", "edge_sha256": key,
                            "from": edge["from"], "to": edge["to"],
                            "excluded": excluded, "required": not excluded,
                            "status": "excluded_by_user" if excluded else "conditional_document_candidate"})
    for state in pdf["unbound_state_outputs"]:
        excluded = state["from"] in EXCLUDED
        obligations.append({"id": "state-review:" + digest(state),
                            "kind": "state_contract_review", "from": state["from"],
                            "producer_port": state["producer_port"],
                            "excluded": excluded, "required": not excluded,
                            "status": "excluded_by_user" if excluded else "review_state_consumer_or_declared_internal_fusion"})
    for dependency in pdf["dependency_requirements"]:
        identities = set(dependency["implementation_ids"])
        in_scope = sorted(identities - EXCLUDED)
        excluded = not in_scope
        obligations.append({"id": "dependency:" + dependency["dependency_id"],
                            "kind": "dependency_review", "dependency_sha256": digest(dependency),
                            "in_scope_implementation_ids": in_scope,
                            "excluded_implementation_ids": sorted(identities & EXCLUDED),
                            "excluded": excluded, "required": not excluded,
                            "status": "excluded_by_user" if excluded else "review_order_branches_shared_state_and_inverse"})
    return obligations


def build_union(root: Path = ROOT) -> dict:
    from operator86_required_paths_v1 import build_method_requirements, build_source_receipt
    from pdf102_composition_requirements_v1 import build_pdf_requirements

    root = root.resolve()
    methods = build_method_requirements(root)
    pdf = build_pdf_requirements()
    layered_path = root / "benchmark/reference/actions/operator_implementations_layered_v1.json"
    catalog_path = root / "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json"
    scope_path = root / "cpp/native_pipeline_v1/experiment_no_cmix_nncp_v1.json"
    layered = json.loads(layered_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    scope = json.loads(scope_path.read_text())
    require(set(scope["excluded_implementation_ids"]) == EXCLUDED, "scope exclusions changed")
    require(scope["catalog_sha256"] == file_sha(catalog_path), "scope/catalog binding mismatch")
    require(layered["count"] == 102, "PDF102 catalog count changed")
    bindings = binding_inventory(layered, catalog)
    obligations = make_obligations(methods, bindings, pdf)
    payload = {
        "definition": "All 86 original algorithm paths union all conditional PDF102 port relationships",
        "online_search": False,
        "special_operations": ["per_flow_STOP", "Concat", "continuous_Split"],
        "excluded_implementation_ids": sorted(EXCLUDED),
        "original_method_paths": methods,
        "pdf102": pdf,
        "current_registry_bindings": bindings,
        "obligations": obligations,
        "source_receipts": {
            "methods": build_source_receipt(root),
            "layered_catalog": {"path": str(layered_path), "sha256": file_sha(layered_path)},
            "frozen_policy_catalog": {"path": str(catalog_path), "sha256": file_sha(catalog_path),
                                      "registry_fingerprint": catalog["registry_fingerprint"]},
            "experiment_scope": {"path": str(scope_path), "sha256": file_sha(scope_path)},
            "builders": {name: file_sha(root / "scripts" / name) for name in (
                "operator_space_union_v1.py", "operator86_required_paths_v1.py",
                "pdf102_composition_requirements_v1.py")},
        },
        "execution_admission": {
            "status": "not_qualified", "ready_for_full_union_training": False,
            "reason": "Required candidate grammar is recorded; complete native path and per-port qualification remains separate.",
            "required_evidence": ["original_method_source_equivalence", "typed_ports_and_explicit_adapters",
                                  "branch_and_shared_state_contract", "complete_frame_bit_exact_roundtrip",
                                  "independent_cold_decode", "native_mask_training_deployment_parity",
                                  "new_registry_and_model_identity_when_semantics_change"],
        },
    }
    document = seal(payload)
    validate_document(document)
    return document


def render_report(document: Mapping[str, Any]) -> str:
    validate_document(document)
    payload = document["payload"]
    edges = [row for row in payload["obligations"] if row["kind"] == "pdf_edge"]
    counts = Counter(row["status"] for row in payload["current_registry_bindings"])
    rows = ["# 86 原算法路径与 PDF102 条件衔接候选目录", "",
            f"候选身份：`{document['space_sha256']}`。", "",
            "已完成候选范围编译；尚未完成全部路径的 C++ 接入与冷解码验收，本目录不能被当成新模型已训练的证明。", "",
            "- 86/86 原算法路径保留为必须覆盖项；完整算法基线不代替拆分路径。",
            "- 102/102 历史实现条目保留；cmix A109、NNCP A110 排除后，100 项参与候选范围。",
            f"- PDF 条件连接共 {len(edges)} 条，其中 {sum(not row['excluded'] for row in edges)} 条未涉及用户排除项。",
            "- 每条连接保留输出端口、类型、条件和来源；数值适配、视图转换和状态连接不作为无条件直接连接。",
            "- 有限队列批次使用可恢复游标，不固定截取前 64 项；未测量/未接线不是不支持。",
            "- 多步组合由条件有向图表示，不穷举无限长度序列；原算法路径的状态和封包要求另行保留。",
            "- 在线仍为逐步选择及增量执行，零候选实测；新动作或语义需要新 registry 和重新训练。", "",
            "## 当前 102 项在冻结策略目录中的绑定状态", "",
            "以下仅核查配置是否存在，不是运行时覆盖认证。", "",
            "| 状态 | 条目数 |", "|---|---:|"]
    rows.extend(f"| `{key}` | {value} |" for key, value in sorted(counts.items()))
    rows.extend(["", "## PDF 连接分类", "", "| 连接类别 | 条数 |", "|---|---:|"])
    rows.extend(f"| `{key}` | {value} |" for key, value in
                sorted(payload["pdf102"]["catalog"]["edge_kind_counts"].items()))
    rows.extend(["", f"另有 {payload['pdf102']['catalog']['unbound_state_output_count']} 个状态输出未找到合同中的消费端口，已保留为接口缺口。",
                 f"依赖附录的 {len(payload['pdf102']['dependency_requirements'])} 项顺序/分支/共享状态约束也单独保留。",
                 "这些连接是带条件的单边要求，不是等量的完整可执行组合。"])
    rows.extend(["", "## 产物", "", "- `candidate_union.json`：完整原算法要求、PDF 条件边与配置映射。",
                 "- `method_paths_86.tsv`：86 个算法逐项机制路径及额外精确性要求，便于检查覆盖。",
                 "- `pdf102_edges.tsv`：全部带方向、端口、条件与 PDF 页码的候选连接。",
                 "- `coverage_schedule.json`：下一批覆盖任务，包含未接入项及恢复游标；不是可直接执行的配方。",
                 "- `current_runtime_gate.json`：全范围训练准入当前为未通过；不修改旧模型身份。", ""])
    return "\n".join(rows)


def write_tables(document: Mapping[str, Any], destination: Path) -> None:
    payload = document["payload"]
    with (destination / "method_paths_86.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["ordinal", "method", "group", "documentary_mechanism_path",
                         "additional_exactness_requirements", "complete_native_path_proven"])
        for method in payload["original_method_paths"]:
            requirements = method["original_algorithm_requirement"]["additional_required_exactness"]
            writer.writerow([method["ordinal"], method["label"], method["category"],
                             " -> ".join(method["required_path"]),
                             " ; ".join(row["requirement"] for row in requirements), "not_established"])
    with (destination / "pdf102_edges.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["edge_id", "producer", "output_port", "consumer", "kind",
                         "types", "condition", "producer_detail", "allowed_by_scope",
                         "review_pdf_page", "runtime_status"])
        for edge in payload["pdf102"]["edges"]:
            writer.writerow([edge["edge_id"], edge["from"], edge["producer_port"], edge["to"],
                             edge["kind"], ",".join(edge["types"]), edge["condition"],
                             edge["producer_detail"], edge["allowed_by_scope"],
                             edge["source"]["review_pdf_compatibility_page"], edge["runtime_status"]])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--budget", type=int, default=64)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--space", type=Path, required=True)
    schedule.add_argument("--cursor", type=int, default=0)
    schedule.add_argument("--budget", type=int, default=64)
    schedule.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--space", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        document = build_union()
        require(not args.output.exists(), "use a new versioned output directory")
        # Validate the requested budget before creating any files.
        batch = coverage_schedule(document, cursor=0, budget=args.budget)
        args.output.mkdir(parents=True)
        for name, data in (("candidate_union.json", document), ("coverage_schedule.json", batch),
                           ("current_runtime_gate.json", {"space_sha256": document["space_sha256"],
                                                          **document["payload"]["execution_admission"]})):
            (args.output / name).write_bytes(canonical(data) + b"\n")
        (args.output / "REPORT.md").write_text(render_report(document))
        write_tables(document, args.output)
        print(json.dumps({"space_sha256": document["space_sha256"], "output": str(args.output),
                          "method_paths": 86, "implementations": 102,
                          "pdf_edges": len(document["payload"]["pdf102"]["edges"]),
                          "ready_for_full_union_training": False}, ensure_ascii=False))
        return 0
    document = json.loads(args.space.read_text())
    validate_document(document)
    if args.command == "schedule":
        args.output.write_bytes(canonical(coverage_schedule(document, cursor=args.cursor,
                                                           budget=args.budget)) + b"\n")
        return 0
    print(json.dumps({"space_integrity": "passed", **document["payload"]["execution_admission"]}))
    # A metadata-only export cannot authorize execution even if this boolean
    # were edited and the public content hash recomputed.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
