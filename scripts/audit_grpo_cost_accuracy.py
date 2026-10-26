#!/usr/bin/env python3
"""Independent count, chronology, accuracy, cost, and lossless audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--full-evaluation", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    locked = json.loads((root / "development_screen/LOCKED_DECISION.json").read_text())
    if locked["locked_coordinate_point_candidates"] != [45] or locked["locked_group_point_candidates"] != ["group_probe28"]:
        raise AssertionError(locked)
    locked_time = datetime.fromisoformat(locked["created_utc"])
    manifests = [json.loads((root / f"raw/{name}/manifest.json").read_text()) for name in ("compact_no_zigzag", "compact_no_probes")]
    if any(datetime.fromisoformat(item["created_utc"]) <= locked_time for item in manifests):
        raise AssertionError("external model predates locked decision")
    if manifests[0]["variants"][0]["disabled_dimensions"] != [45] or manifests[1]["variants"][0]["disabled_dimensions"] != list(range(18, 46)):
        raise AssertionError("frozen deletion set mismatch")

    full = [row for row in read(args.full_evaluation.resolve()) if row["algorithm"] == "C0"]
    variants = {
        "compact_no_zigzag": read(root / "raw/compact_no_zigzag/evaluation.csv"),
        "compact_no_probes": read(root / "raw/compact_no_probes/evaluation.csv"),
    }
    if len(full) != 54 or any(len(value) != 54 for value in variants.values()):
        raise AssertionError("evaluation coverage")
    expected_keys = {(row["dataset"], row["column_index"], row["seed"]) for row in full}
    if len(expected_keys) != 54 or any({(row["dataset"], row["column_index"], row["seed"]) for row in value} != expected_keys for value in variants.values()):
        raise AssertionError("evaluation keys")

    summary = {row["variant"]: row for row in read(root / "external_analysis_v2/external_variant_summary.csv")}
    recomputed = {}
    for name, values in variants.items():
        exact = 100.0 * np.mean([abs(float(row["proxy_rate"]) - float(row["oracle_rate"])) <= 1.0e-15 for row in values])
        within = 100.0 * np.mean([float(row["oracle_regret_pct"]) <= 1.0 + 1.0e-12 for row in values])
        recomputed[name] = {"pooled_exact_pct": float(exact), "pooled_within_pct": float(within)}
        if int(summary[name]["point_rule_external_pass"]) != 1:
            raise AssertionError(name)

    cost_manifest = json.loads((root / "external_cost_v2/manifest.json").read_text())
    if sorted(map(int, cost_manifest["actor_ms"])) != [50, 78, 88, 103, 105, 106]:
        raise AssertionError("compact actor widths")
    codec = read(root / "codec/per_seed_codec_cost.csv")
    codec_manifest = json.loads((root / "codec/MANIFEST.json").read_text())
    if len(codec) != 162 or sum(int(row["lossless"]) for row in codec) != 162 or codec_manifest["unique_pipeline_column_pairs"] != 83:
        raise AssertionError("codec replay")

    output = {
        "status": "PASS",
        "locked_before_external": True,
        "locked_utc": locked["created_utc"],
        "external_model_utc": [item["created_utc"] for item in manifests],
        "evaluation_rows": {"full": len(full), **{name: len(value) for name, value in variants.items()}},
        "recomputed_pooled_accuracy": recomputed,
        "codec": {"rows": len(codec), "lossless": 162, "unique_column_pipeline_pairs": 83},
        "hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("external_analysis_v2/*")) if path.is_file()},
    }
    (root / "AUDIT.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
