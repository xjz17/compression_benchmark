#!/usr/bin/env python3
"""Recompute the selector-only Beam versus exact-prefix timing summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1.0e-9:
        raise AssertionError((actual, expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    summary_path = root / "SUMMARY.json"
    samples_path = root / "samples.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = read_rows(samples_path)
    variants = {row["variant"] for row in samples}
    if variants != {"beam29_selector_only", "sparse19_unique_prefix_exact"}:
        raise AssertionError(variants)
    if not summary["beam_timer_excludes_offline_oracle_scan"]:
        raise AssertionError("Beam timer boundary missing")
    expected_rows = int(summary["groups"]) * int(summary["repeats"]) * 2
    if len(samples) != expected_rows:
        raise AssertionError((len(samples), expected_rows))

    grouped: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in samples:
        grouped[(row["dataset"], int(row["column_index"]), row["algorithm"])][
            row["variant"]
        ].append(int(row["elapsed_ns"]))
    if len(grouped) != int(summary["groups"]):
        raise AssertionError(len(grouped))
    medians = {
        key: {variant: statistics.median(values) for variant, values in by_variant.items()}
        for key, by_variant in grouped.items()
    }
    beam = [values["beam29_selector_only"] for values in medians.values()]
    exact = [values["sparse19_unique_prefix_exact"] for values in medians.values()]
    reductions = [100.0 * (old - new) / old for old, new in zip(beam, exact)]
    close(statistics.median(beam) / 1_000_000.0, summary["beam29_group_median_ms"])
    close(statistics.median(exact) / 1_000_000.0, summary["exact_group_median_ms"])
    close(statistics.median(reductions), summary["paired_exact_speedup_vs_beam_pct_median"])
    if all(value > 0.0 for value in reductions) != summary["all_groups_exact_faster"]:
        raise AssertionError("all-groups flag mismatch")

    by_dataset = {}
    for dataset in sorted({key[0] for key in grouped}):
        selected = [
            reduction for key, reduction in zip(medians, reductions) if key[0] == dataset
        ]
        by_dataset[dataset] = {
            "groups": len(selected),
            "exact_faster": sum(value > 0.0 for value in selected),
            "exact_slower": sum(value < 0.0 for value in selected),
            "paired_median_latency_reduction_pct": statistics.median(selected),
        }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checks": {
            "sample_rows": len(samples),
            "groups": len(grouped),
            "repeats": summary["repeats"],
            "selector_only_variant_names_exact": True,
            "summary_arithmetic_recomputed": True,
            "beam_timer_declares_oracle_excluded": True,
            "exact_faster_groups": sum(value > 0.0 for value in reductions),
            "exact_slower_groups": sum(value < 0.0 for value in reductions),
        },
        "by_dataset": by_dataset,
        "source_sha256": {
            summary_path.name: sha256(summary_path),
            samples_path.name: sha256(samples_path),
        },
        "claim_boundary": (
            "Arithmetic and timer-label audit only; source inspection is recorded separately and this "
            "does not establish end-to-end codec latency."
        ),
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
