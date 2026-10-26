#!/usr/bin/env python3
"""Deterministically audit the information-preserving PV2F feature deletions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    base_path = script_path.with_name("search_physical_operator_variants_v2.py")
    sys.path.insert(0, str(script_path.parent))
    base = load_module(base_path, "physical_feature_redundancy_base")
    tables = base.load_tables(args.measurement_root.resolve())
    raw, _rows = base.load_feature_matrix(args.repo_root.resolve(), tables)

    prefixes: set[tuple[str, ...]] = {()}
    frontier = [()]
    while frontier:
        prefix = frontier.pop()
        for action in base.legal_action_names(prefix):
            candidate = prefix + (action,)
            if not base.is_terminal(candidate) and candidate not in prefixes:
                prefixes.add(candidate)
                frontier.append(candidate)

    phase_violations = 0
    depth_violations = 0
    for prefix in sorted(prefixes):
        feature = base.prefix_features(prefix)
        phase = feature[:3]
        depth = feature[3:6]
        transform_history = feature[6:30]
        packer_history = feature[30:43]
        derived_phase = np.asarray([float(not packer_history.any()), float(packer_history.any()), 0.0])
        slot_count = int(transform_history[:12].any()) + int(transform_history[12:].any())
        derived_depth = np.zeros(3)
        derived_depth[slot_count] = 1.0
        phase_violations += int(not np.array_equal(phase, derived_phase))
        depth_violations += int(not np.array_equal(depth, derived_depth))

    duplicate_violations = int(np.sum(raw[:, 6] != raw[:, 8]))
    edge = np.asarray([np.iinfo(np.int64).max - 1, np.iinfo(np.int64).max], dtype=np.int64)
    edge_integer_run = float(np.mean(edge[1:] == edge[:-1]))
    edge_float_delta_zero = float(np.mean(np.diff(edge.astype(np.float64)) == 0))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not (phase_violations or depth_violations or duplicate_violations) else "FAIL",
        "nonterminal_prefixes": len(prefixes),
        "phase_derivation_violations": phase_violations,
        "depth_derivation_violations": depth_violations,
        "observed_columns": len(tables),
        "feature6_feature8_observed_violations": duplicate_violations,
        "int64_edge_witness": {
            "values": [int(value) for value in edge],
            "feature6_integer_run_ratio": edge_integer_run,
            "feature8_float_delta_zero_ratio": edge_float_delta_zero,
            "float_path_is_false_duplicate": bool(edge_integer_run != edge_float_delta_zero),
        },
        "deletion": "phase3 + depth3 + feature8; retain feature6",
        "source_sha256": {"audit": sha256(script_path), "base": sha256(base_path)},
    }
    if result["status"] != "PASS" or not result["int64_edge_witness"]["float_path_is_false_duplicate"]:
        raise AssertionError(result)
    args.output.resolve().write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
