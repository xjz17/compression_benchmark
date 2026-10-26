#!/usr/bin/env python3
"""Audit the boundary between C++ codec-stage candidates and RL policy actions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import primitive_operator_action_runtime as primitive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    ROOT / "analysis_outputs/cpp_codec_action_split_20260902_v1/cpp_action_candidates.csv"
)
DEFAULT_ANS_AUDIT = (
    ROOT / "analysis_outputs/cpp_codec_action_split_20260902_v1/ans_cpp_python_wire_audit.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--ans-audit", type=Path, default=DEFAULT_ANS_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.catalog.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    ans_audit = json.loads(args.ans_audit.read_text(encoding="utf-8"))
    cpp_lossless = ans_audit.get("cpp_lossless_columns", ans_audit.get("lossless_columns"))
    python_lossless = ans_audit.get("python_lossless_columns", cpp_lossless)
    if (ans_audit["wire_identical_columns"] != 20 or cpp_lossless != 20 or
            python_lossless != 20):
        raise RuntimeError("ANS C++/Python equivalence requires 20/20 wire and roundtrip")

    policy_names = {action.name for action in primitive.POLICY_ACTIONS}
    policy_by_name = {action.name: action for action in primitive.POLICY_ACTIONS}
    candidate_names = {row["action_name"] for row in candidates}
    if len(candidates) != 106 or len(candidate_names) != 106:
        raise RuntimeError("expected 106 unique C++ candidate names")
    evidence = Counter(row["evidence"] for row in candidates)
    standalone = [row for row in candidates if row["standalone_composable"] == "1"]
    if len(standalone) != 1 or standalone[0]["method"] != "ANS":
        raise RuntimeError("only ANS may be standalone-composable in this catalog")
    registered_by_exact_name = sorted(policy_names & candidate_names)
    if registered_by_exact_name:
        raise RuntimeError(
            "C++ catalog candidates must not be silently registered as policy actions: "
            + ", ".join(registered_by_exact_name)
        )
    equivalent_policy_name = "op.entropy_fse.project_rans_bytes.v1"
    equivalent_policy = policy_by_name.get(equivalent_policy_name)
    if equivalent_policy is None or equivalent_policy.operator_id != standalone[0]["operator_id"]:
        raise RuntimeError("missing same-operator policy action for exact ANS wire deduplication")
    equivalent_existing = [
        {
            "cpp_candidate": standalone[0]["action_name"],
            "policy_action": equivalent_policy_name,
            "dedup_rule": (
                "same entropy_fse operator and byte-identical payload on all 20 columns"
            ),
        }
    ]
    net_new_policy_actions = len(standalone) - len(equivalent_existing)

    payload = {
        "schema": "cpp-action-registration-boundary-v1",
        "policy_action_count": len(primitive.POLICY_ACTIONS),
        "policy_stop_action_id": primitive.STOP_ACTION_ID,
        "cpp_codec_stage_candidate_count": len(candidates),
        "cpp_candidate_evidence_counts": dict(sorted(evidence.items())),
        "cpp_standalone_composable_count": len(standalone),
        "cpp_candidates_registered_by_exact_name": len(registered_by_exact_name),
        "cpp_candidates_are_policy_actions": candidate_names.issubset(policy_names),
        "net_new_policy_actions_from_cpp_catalog": net_new_policy_actions,
        "equivalent_existing_policy_actions": equivalent_existing,
        "claim_boundary": (
            "codec_ablation_cpp means a real method-scoped omission path, not an independently "
            "composable or registered training action"
        ),
        "provenance": {
            "candidate_catalog": {
                "path": str(args.catalog.resolve()),
                "sha256": sha256(args.catalog),
            },
            "ans_wire_audit": {
                "path": str(args.ans_audit.resolve()),
                "sha256": sha256(args.ans_audit),
            },
            "policy_runtime": {
                "path": str(Path(primitive.__file__).resolve()),
                "sha256": sha256(Path(primitive.__file__)),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "policy_action_count",
        "cpp_codec_stage_candidate_count",
        "cpp_standalone_composable_count",
        "cpp_candidates_registered_by_exact_name",
        "net_new_policy_actions_from_cpp_catalog",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
