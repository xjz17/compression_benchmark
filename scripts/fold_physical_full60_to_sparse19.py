#!/usr/bin/env python3
"""Fold saved Full60 PV2F actors into exactly equivalent Sparse19 actors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXTERNAL_COLUMNS = (
    ("Bird-migration", 0), ("Bird-migration", 1), ("Bird-migration", 2),
    ("Bird-migration", 3), ("Bird-migration", 4),
    ("meteoblue", 0), ("meteoblue", 5), ("meteoblue", 10),
    ("meteoblue", 15), ("meteoblue", 20), ("meteoblue", 25),
    ("meteoblue", 30), ("meteoblue", 35), ("meteoblue", 40),
    ("meteoblue", 45), ("meteoblue", 50), ("meteoblue", 55),
    ("meteoblue", 60),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def legal_nonterminal_prefixes(base) -> list[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = {()}
    frontier = [()]
    while frontier:
        prefix = frontier.pop()
        for action in base.legal_action_names(prefix):
            candidate = prefix + (action,)
            if not base.is_terminal(candidate) and candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return sorted(seen)


def policy_identity(path: Path) -> tuple[str, int, int]:
    match = re.fullmatch(r"(ppo|grpo)_seed(\d+)_fold(\d+)", path.stem)
    if match is None:
        raise ValueError(f"expected <ppo|grpo>_seedN_foldN policy name, got {path.name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def proposal_set(base, actor, builder, column: int, seed: int, fold: int, budget: int, sparse: bool):
    propose = base.propose_pipeline_sparse19 if sparse else base.propose_pipeline
    rng_seed = seed + 10_000_019 * (fold + 1) + 1009 * (column + 1)
    rng = np.random.default_rng(rng_seed)
    proposals = {propose(actor, builder, column, None), ("raw.i64", "none")}
    for _ in range(budget):
        proposals.add(propose(actor, builder, column, rng))
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--full-policy-dir", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposal-budget", type=int, default=512)
    args = parser.parse_args()

    script = Path(__file__).resolve()
    sys.path.insert(0, str(script.parent))
    base_path = script.with_name("search_physical_operator_variants_v2.py")
    base = load_module(base_path, "fold_sparse_base")
    repo_root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "policies").mkdir()

    tables = base.load_tables(args.measurement_root.resolve())
    datasets = sorted({table.dataset for table in tables})
    feature_rows = read_csv(args.feature_csv.resolve())
    if len(feature_rows) != len(tables):
        raise AssertionError((args.feature_csv, len(feature_rows), len(tables)))
    for table, row in zip(tables, feature_rows):
        if (
            row["dataset"] != table.dataset
            or int(row["column_index"]) != table.column_index
            or row["column_name"] != table.column_name
        ):
            raise AssertionError((table.key, row))
    raw = np.asarray(
        [[float(row[f"feature_{index}"]) for index in range(17)] for row in feature_rows],
        dtype=np.float64,
    )
    external_raw: list[np.ndarray] = []
    external_rows: list[tuple[str, int, str]] = []
    if not args.skip_external:
        for dataset, column in EXTERNAL_COLUMNS:
            values, metadata, _data, _info = base.load_column(repo_root, dataset, column, 0)
            external_raw.append(base.column_features(values))
            external_rows.append((dataset, column, metadata["column_name"]))
    combined_raw = np.vstack((raw, np.vstack(external_raw))) if external_raw else raw
    duplicate_violations = int(np.sum(combined_raw[:, 6] != combined_raw[:, 8]))
    if duplicate_violations:
        raise AssertionError(f"f6/f8 differ on {duplicate_violations} audited columns")
    all_columns = np.arange(combined_raw.shape[0], dtype=np.int32)
    prefixes = legal_nonterminal_prefixes(base)
    policy_paths = sorted(args.full_policy_dir.resolve().glob("*.npz"))
    if not policy_paths:
        raise FileNotFoundError(args.full_policy_dir)

    recorded: dict[tuple[str, int, int, str, int], dict[str, str]] = {}
    if args.evaluation_csv:
        for row in read_csv(args.evaluation_csv.resolve()):
            recorded[(
                row["algorithm"], int(row["seed"]), int(row["fold"]),
                row["test_dataset"], int(row["column_index"]),
            )] = row

    audit_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    external_proposal_rows: list[dict[str, Any]] = []
    max_logit_error = 0.0
    for policy_path in policy_paths:
        algorithm, seed, fold = policy_identity(policy_path)
        if not 0 <= fold < len(datasets):
            raise AssertionError((policy_path, datasets))
        test_dataset = datasets[fold]
        train = np.asarray(
            [index for index, table in enumerate(tables) if table.dataset != test_dataset],
            dtype=np.int32,
        )
        test = np.asarray(
            [index for index, table in enumerate(tables) if table.dataset == test_dataset],
            dtype=np.int32,
        )
        full_builder = base.FeatureBuilder(combined_raw, train)
        sparse_builder = base.Sparse19FeatureBuilder(combined_raw, train)
        with np.load(policy_path) as arrays:
            params = [arrays[f"arr_{index}"].copy() for index in range(4)]
        full_actor = base.MLP(60, int(params[1].shape[0]), len(base.ACTION_NAMES), np.random.default_rng(0))
        full_actor.params = params
        sparse_actor = base.fold_full60_actor(full_actor)
        folded_path = output / "policies" / policy_path.name
        np.savez_compressed(folded_path, *sparse_actor.params)

        local_logit_error = 0.0
        for prefix in prefixes:
            full_x = np.vstack([full_builder.one(int(column), prefix) for column in all_columns])
            full_logits = full_actor.forward(full_x)[0]
            sparse_logits = sparse_actor.forward_columns(sparse_builder, all_columns, prefix)
            local_logit_error = max(
                local_logit_error,
                float(np.max(np.abs(full_logits - sparse_logits))),
            )
        max_logit_error = max(max_logit_error, local_logit_error)

        matched = 0
        recorded_matched = 0
        for column in test:
            column = int(column)
            table = tables[column]
            full_proposals = proposal_set(
                base, full_actor, full_builder, column, seed, fold, args.proposal_budget, False
            )
            sparse_proposals = proposal_set(
                base, sparse_actor, sparse_builder, column, seed, fold, args.proposal_budget, True
            )
            full_valid = [pipeline for pipeline in full_proposals if pipeline in table.sizes]
            sparse_valid = [pipeline for pipeline in sparse_proposals if pipeline in table.sizes]
            full_best = min(full_valid, key=lambda pipeline: (table.sizes[pipeline], pipeline))
            sparse_best = min(sparse_valid, key=lambda pipeline: (table.sizes[pipeline], pipeline))
            proposal_equal = full_proposals == sparse_proposals
            selected_equal = full_best == sparse_best
            matched += int(proposal_equal and selected_equal)
            recorded_row = recorded.get((algorithm, seed, fold, table.dataset, table.column_index))
            recorded_equal = recorded_row is None or (
                recorded_row["pipeline"] == " -> ".join(sparse_best)
                and int(recorded_row["encoded_bytes"]) == table.sizes[sparse_best]
                and int(recorded_row["unique_proposals"]) == len(sparse_proposals)
            )
            recorded_matched += int(recorded_equal)
            proposal_rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "fold": fold,
                "dataset": table.dataset,
                "column_index": table.column_index,
                "column_name": table.column_name,
                "proposal_sets_equal": int(proposal_equal),
                "selected_pipeline_equal": int(selected_equal),
                "recorded_evaluation_equal": int(recorded_equal),
                "unique_proposals": len(sparse_proposals),
                "full60_bytes": table.sizes[full_best],
                "sparse19_bytes": table.sizes[sparse_best],
                "full60_pipeline": " -> ".join(full_best),
                "sparse19_pipeline": " -> ".join(sparse_best),
            })

        external_matched = 0
        for external_offset, (dataset, source_column, column_name) in enumerate(external_rows):
            column = len(tables) + external_offset
            rng_seed = seed + 30_000_019 + 1009 * (external_offset + 1)
            full_rng = np.random.default_rng(rng_seed)
            sparse_rng = np.random.default_rng(rng_seed)
            full_proposals = {
                base.propose_pipeline(full_actor, full_builder, column, None),
                ("raw.i64", "none"),
            }
            sparse_proposals = {
                base.propose_pipeline_sparse19(sparse_actor, sparse_builder, column, None),
                ("raw.i64", "none"),
            }
            for _ in range(args.proposal_budget):
                full_proposals.add(base.propose_pipeline(full_actor, full_builder, column, full_rng))
                sparse_proposals.add(
                    base.propose_pipeline_sparse19(sparse_actor, sparse_builder, column, sparse_rng)
                )
            equal = full_proposals == sparse_proposals
            external_matched += int(equal)
            external_proposal_rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "fold": fold,
                "training_dataset": ",".join(sorted({tables[index].dataset for index in train})),
                "dataset": dataset,
                "source_column_index": source_column,
                "column_name": column_name,
                "proposal_sets_equal": int(equal),
                "full60_unique_proposals": len(full_proposals),
                "sparse19_unique_proposals": len(sparse_proposals),
            })
        audit_rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "fold": fold,
            "test_dataset": test_dataset,
            "policy": str(policy_path),
            "folded_policy": str(folded_path),
            "legal_nonterminal_prefixes": len(prefixes),
            "columns_including_external": len(all_columns),
            "max_full60_vs_sparse19_logit_abs_error": local_logit_error,
            "development_proposal_matches": matched,
            "development_columns": len(test),
            "recorded_evaluation_matches": recorded_matched,
            "external_proposal_matches": external_matched,
            "external_columns": len(external_rows),
        })

    write_csv(output / "fold_equivalence.csv", audit_rows)
    write_csv(output / "development_proposal_equivalence.csv", proposal_rows)
    if external_proposal_rows:
        write_csv(output / "external_proposal_equivalence.csv", external_proposal_rows)
    development_exact = sum(
        int(row["proposal_sets_equal"])
        and int(row["selected_pipeline_equal"])
        and int(row["recorded_evaluation_equal"])
        for row in proposal_rows
    )
    external_exact = sum(int(row["proposal_sets_equal"]) for row in external_proposal_rows)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if (
            max_logit_error <= 1.0e-12
            and development_exact == len(proposal_rows)
            and external_exact == len(external_proposal_rows)
        ) else "FAIL",
        "policies": len(policy_paths),
        "algorithms": sorted({policy_identity(path)[0] for path in policy_paths}),
        "legal_nonterminal_prefixes": len(prefixes),
        "development_columns": len(tables),
        "external_columns": len(external_rows),
        "external_datasets": sorted({dataset for dataset, _column, _name in external_rows}),
        "f6_f8_violations_all_audited_columns": duplicate_violations,
        "max_full60_vs_sparse19_logit_abs_error": max_logit_error,
        "development_proposal_rows": len(proposal_rows),
        "development_exact_proposal_rows": development_exact,
        "external_proposal_rows": len(external_proposal_rows),
        "external_exact_proposal_rows": external_exact,
        "representation": {
            "continuous_fields": 16,
            "categorical_fields": 3,
            "logical_observation_fields": 19,
            "dense_equivalent_dimensions": 53,
            "categorical_implementation": (
                "ordered transform slot 1, ordered transform slot 2, and packer use independent "
                "lookup tables; IDs are never treated as ordinal numeric features"
            ),
            "training": "unchanged Full60 optimization followed by an algebraically exact inference fold",
        },
        "source_sha256": {
            "runner": sha256(script),
            "base": sha256(base_path),
            "evaluation_csv": sha256(args.evaluation_csv.resolve()) if args.evaluation_csv else None,
            "feature_csv": sha256(args.feature_csv.resolve()),
        },
        "claim_boundary": (
            "exact for the supplied frozen policies on all legal nonterminal prefixes and the audited "
            "finite columns; removing f8 is not claimed for arbitrary int64 columns or retraining"
        ),
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise AssertionError(audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
