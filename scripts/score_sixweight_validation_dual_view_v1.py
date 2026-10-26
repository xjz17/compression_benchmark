#!/usr/bin/env python3
"""Score a frozen validation panel against same-panel Zstd before training.

This is a checkpoint-selection diagnostic, not the B_raw-weighted final test.
The two views keep both large blocks and all admitted domains visible. The
selection rule must be fixed before any candidate in a new run is selected.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


WEIGHTS = {
    "100": (1.0, 0.0, 0.0),
    "010": (0.0, 1.0, 0.0),
    "001": (0.0, 0.0, 1.0),
    "550": (0.5, 0.5, 0.0),
    "505": (0.5, 0.0, 0.5),
    "433": (0.4, 0.3, 0.3),
}
E0 = 15.159782608695652
D0 = 7.023168103448276
MIN_LARGE_BYTES = 65536
SELECTION_RULE = "max_six_weight_relative_J_large_and_all_domain_macro_plus_0.01_mean_v1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(row: dict, name: str) -> float:
    value = float(row[name])
    require(math.isfinite(value) and value > 0, f"invalid {name}")
    return value


def objective(row: dict, weight: tuple[float, float, float]) -> float:
    ratio = metric(row, "compression_ratio")
    encode = metric(row, "compression_ns_per_byte")
    decode = metric(row, "decompression_ns_per_byte")
    a, b, c = weight
    return a * ratio + b * encode / (encode + E0) + c * decode / (decode + D0)


def near(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=1e-9, rel_tol=1e-9)


def hierarchy_masses(keys: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], float]:
    require(keys, "empty macro hierarchy")
    children: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for key in keys:
        for i, part in enumerate(key):
            children[key[:i]].add(part)
    masses = {key: math.prod(1 / len(children[key[:i]]) for i in range(len(key)))
              for key in keys}
    require(near(sum(masses.values()), 1.0), "macro masses do not sum to one")
    return masses


def macro(values: dict[str, float], metadata: dict[str, tuple[str, str, str]]) -> float:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample, value in values.items():
        groups[metadata[sample]].append(value)
    masses = hierarchy_masses(set(groups))
    return sum(masses[key] * math.fsum(rows) / len(rows)
               for key, rows in groups.items())


def score(baseline: dict, validation: dict,
          expected_validation_manifest_sha256: str | None = None) -> dict:
    require(baseline.get("schema") == "sixweight-validation-zstd-v1" and
            baseline.get("test_used") is False and
            baseline.get("cpu") == 25 and
            baseline.get("objective") == "a*R+b*E/(E+E0)+c*D/(D+D0)",
            "frozen same-panel Zstd baseline contract missing")
    if expected_validation_manifest_sha256 is not None:
        require(baseline.get("validation_manifest_sha256") ==
                expected_validation_manifest_sha256,
                "baseline does not bind the admitted validation manifest")
    require(validation.get("schema") == "native-combo-greedy-validation-v6" and
            validation.get("greedy_only_checkpoint_selection") is True and
            validation.get("validation_candidate_evaluations") == 0,
            "greedy C++ validation contract missing")
    baseline_by_sample: dict[str, dict] = {}
    for row in baseline.get("rows", []):
        sample = str(row["sample_id"])
        require(sample not in baseline_by_sample and
                int(row["B_raw"]) > 0 and
                row.get("independent_cold_decode") is True,
                "duplicate or unverified Zstd baseline sample")
        require(set(row["J_by_weight"]) == set(WEIGHTS),
                "incomplete Zstd six-weight vector")
        for name, weight in WEIGHTS.items():
            require(near(float(row["J_by_weight"][name]), objective(row, weight)),
                    "Zstd objective differs from frozen scale")
        baseline_by_sample[sample] = row
    require(baseline_by_sample, "empty Zstd baseline")
    model: dict[tuple[str, str], dict] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    names_by_weight = {weight: name for name, weight in WEIGHTS.items()}
    for row in validation.get("rows", []):
        sample = str(row["sample_id"])
        weight = tuple(float(part) for part in row["weight"])
        require(sample in baseline_by_sample and weight in names_by_weight,
                "model validation sample or weight outside frozen panel")
        name = names_by_weight[weight]
        require((sample, name) not in model and
                int(row["B_raw"]) == int(baseline_by_sample[sample]["B_raw"]),
                "duplicate validation row or B_raw mismatch")
        require(row.get("status") == "ok" and row.get("error") is None,
                "failed validation row cannot be selected")
        require(near(float(row["J_total"]), objective(row, weight)),
                "model objective differs from frozen scale")
        key = tuple(str(row[field]) for field in
                    ("domain", "dataset_id", "size_bucket"))
        require(len(key) == 3 and all(key) and
                (sample not in metadata or metadata[sample] == key),
                "sample macro hierarchy changed across weights")
        metadata[sample] = key
        model[sample, name] = row
    require(set(metadata) == set(baseline_by_sample) and
            set(model) == {(sample, name) for sample in metadata for name in WEIGHTS},
            "validation is not a complete same-panel sample by weight matrix")
    large = {sample for sample, row in baseline_by_sample.items()
             if int(row["B_raw"]) >= MIN_LARGE_BYTES}
    require(large, "no >=64KiB validation sample")
    result = {}
    relative = []
    for panel_name, samples in (("large", large), ("all", set(metadata))):
        panel = {}
        panel_metadata = {sample: metadata[sample] for sample in samples}
        for name in WEIGHTS:
            model_j = macro({sample: float(model[sample, name]["J_total"])
                             for sample in samples}, panel_metadata)
            zstd_j = macro({sample: float(baseline_by_sample[sample]["J_by_weight"][name])
                            for sample in samples}, panel_metadata)
            require(zstd_j > 0 and math.isfinite(model_j) and math.isfinite(zstd_j),
                    "nonfinite or nonpositive macro objective")
            ratio = model_j / zstd_j
            relative.append(ratio)
            panel[name] = {"model_macro_J": model_j, "zstd_macro_J": zstd_j,
                           "relative_J": ratio}
        result[panel_name] = {"samples": len(samples),
                              "domains": sorted({metadata[s][0] for s in samples}),
                              "by_weight": panel}
    return {"schema": "sixweight-dual-view-validation-score-v1",
            "selection_rule": SELECTION_RULE,
            "selection_score": max(relative) + 0.01 * math.fsum(relative) / len(relative),
            "worst_relative_J": max(relative),
            "mean_relative_J": math.fsum(relative) / len(relative),
            "views": result, "validation_row_count": len(model),
            "test_used": False, "E0": E0, "D0": D0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    baseline = json.loads(a.baseline.read_text())
    validation = json.loads(a.validation.read_text())
    value = score(baseline, validation,
                  sha(a.validation_manifest) if a.validation_manifest else None)
    value.update(baseline_sha256=sha(a.baseline),
                 validation_sha256=sha(a.validation),
                 validation_manifest_sha256=(sha(a.validation_manifest)
                                             if a.validation_manifest else None))
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if a.output:
        require(not a.output.exists(), "immutable score output already exists")
        a.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
