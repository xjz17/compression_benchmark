#!/usr/bin/env python3
"""Independently recompute the fresh-dataset semantic re-encoding evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("C0", "E1", "E2", "E3", "D1", "D2", "D3", "D4")
TEST_DATASETS = ("Bird-migration", "meteoblue")
EXPECTED_SELECTION = {"Bird-migration": list(range(5)), "meteoblue": list(range(0, 61, 5))}
EXCLUDED_SELECTION = {"Bird-migration": [5, 6]}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "orignial_datasets" / f"{dataset}.csv",
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data = next((path for path in data_candidates if path.is_file()), None)
    info = next((path for path in info_candidates if path.is_file()), None)
    if data is None or info is None:
        raise FileNotFoundError(dataset)
    return data, info


def independently_load_slices(repo_root: Path, operator_module, selected: dict[str, list[int]]):
    evidence: dict[tuple[str, int], dict[str, object]] = {}
    for dataset, column_indices in selected.items():
        data_path, info_path = locate_dataset(repo_root, dataset)
        metadata = {int(item["column_index"]): item for item in operator_module.numeric_columns(info_path)}
        chosen = [metadata[index] for index in column_indices]
        names = [item["column_name"] for item in chosen]
        frame = pd.read_csv(data_path, usecols=names, nrows=65536, low_memory=False, dtype=str)
        if len(frame) != 65536:
            raise AssertionError((dataset, len(frame)))
        for item in chosen:
            index = int(item["column_index"])
            values = operator_module.column_values(frame[item["column_name"]], item)
            if values.size != 65536:
                raise AssertionError((dataset, index, values.size))
            prefix = np.asarray(values[:32768], dtype=np.int64)
            suffix = np.asarray(values[32768:65536], dtype=np.int64)
            evidence[(dataset, index)] = {
                "column_name": item["column_name"],
                "calibration_sha256": digest(operator_module.i64_bytes(prefix)),
                "evaluation_sha256": digest(operator_module.i64_bytes(suffix)),
                "evaluation_values": suffix,
                "source_csv": str(data_path.resolve()),
            }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    raw_manifest = json.loads((run_root / "raw" / "manifest.json").read_text(encoding="utf-8"))
    codec_manifest = json.loads((run_root / "codec" / "MANIFEST.json").read_text(encoding="utf-8"))
    analysis_manifest = json.loads((run_root / "analysis" / "MANIFEST.json").read_text(encoding="utf-8"))

    measurement_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    excluded_measurement_rows: list[dict[str, str]] = []
    shard_manifests = []
    for shard in sorted((run_root / "fresh_measurements").iterdir()):
        if not shard.is_dir():
            continue
        manifest_path = shard / "manifest.json"
        if not manifest_path.is_file():
            continue
        shard_manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        shard_measurements = rows(shard / "pipeline_measurements.csv")
        shard_summary = rows(shard / "summary.csv")
        key = shard_summary[0]["dataset"], int(shard_summary[0]["column_index"])
        if key[1] in EXPECTED_SELECTION.get(key[0], []):
            measurement_rows.extend(shard_measurements)
            summary_rows.extend(shard_summary)
        elif key[1] in EXCLUDED_SELECTION.get(key[0], []):
            excluded_measurement_rows.extend(shard_measurements)
        else:
            raise AssertionError(("unexpected fresh shard", key))

    column_keys = sorted({(row["dataset"], int(row["column_index"])) for row in summary_rows})
    selected = {
        dataset: sorted(index for item_dataset, index in column_keys if item_dataset == dataset)
        for dataset in TEST_DATASETS
    }
    if selected != EXPECTED_SELECTION:
        raise AssertionError(selected)
    by_measurement_column = Counter((row["dataset"], int(row["column_index"])) for row in measurement_rows)
    if set(by_measurement_column.values()) != {22764}:
        raise AssertionError(by_measurement_column)
    if len(measurement_rows) != 18 * 22764:
        raise AssertionError(len(measurement_rows))
    if any(int(row["lossless"]) != 1 or row["status"] != "ok" for row in measurement_rows):
        raise AssertionError("fresh proxy search contains a failed pipeline")

    operator_module = load_module(
        repo_root / "scripts" / "generate_air_operator_pair_heatmap.py", "semantic_audit_operator"
    )
    codec_module = load_module(run_root / "code" / "scripts" / "measure_selected_codec_cost.py", "semantic_audit_codec")
    slice_evidence = independently_load_slices(repo_root, operator_module, selected)
    summary_by_key = {(row["dataset"], int(row["column_index"])): row for row in summary_rows}
    for key, evidence in slice_evidence.items():
        if evidence["calibration_sha256"] != summary_by_key[key]["sample_sha256"]:
            raise AssertionError((key, "calibration hash mismatch"))

    evaluation = rows(run_root / "raw" / "evaluation.csv")
    status = rows(run_root / "raw" / "run_status.csv")
    training = rows(run_root / "raw" / "training_curve.csv")
    expected_eval_keys = {
        (variant, seed, dataset, column)
        for variant in VARIANTS
        for seed in (20260825, 20260826, 20260827)
        for dataset, columns in selected.items()
        for column in columns
    }
    observed_eval_keys = {
        (row["algorithm"], int(row["seed"]), row["dataset"], int(row["column_index"]))
        for row in evaluation
    }
    if observed_eval_keys != expected_eval_keys or len(evaluation) != 432:
        raise AssertionError("RL evaluation coverage mismatch")
    if len(status) != 24 or any(int(row["completed_iterations"]) != 80 for row in status):
        raise AssertionError("training completion mismatch")
    if len(training) != 24 * 9:
        raise AssertionError(("training log row count", len(training)))
    if raw_manifest["counts"]["development_columns"] != 20 or raw_manifest["counts"]["external_test_columns"] != 18:
        raise AssertionError(raw_manifest["counts"])
    if max(raw_manifest["function_match_max_abs_logit_error"].values()) > 1.0e-12:
        raise AssertionError(raw_manifest["function_match_max_abs_logit_error"])
    expected_cardinality = {"C0": 813, "E1": 813, "E2": 813, "E3": 813, "D1": 813, "D2": 435, "D3": 3, "D4": 1}
    if raw_manifest["history_code_cardinality"] != expected_cardinality:
        raise AssertionError(raw_manifest["history_code_cardinality"])

    codec = rows(run_root / "codec" / "per_seed_codec_cost.csv")
    if len(codec) != 450 or any(int(row["lossless"]) != 1 for row in codec):
        raise AssertionError("codec coverage/lossless mismatch")
    if {int(row["original_values"]) for row in codec} != {32768}:
        raise AssertionError("codec suffix length mismatch")
    if {row["algorithm"] for row in codec} != {*VARIANTS, "Beam-29"}:
        raise AssertionError("codec algorithm mismatch")
    if int(codec_manifest["row_slice"]["offset"]) != 32768 or int(codec_manifest["row_slice"]["max_values"]) != 32768:
        raise AssertionError("codec row split mismatch")

    input_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    distinct_replays: dict[tuple[str, int, str], tuple[int, str]] = {}
    for row in codec:
        key = row["dataset"], int(row["column_index"])
        evidence = slice_evidence[key]
        input_hashes[key].add(row["input_sha256"])
        if row["input_sha256"] != evidence["evaluation_sha256"]:
            raise AssertionError((key, "evaluation hash mismatch"))
        pipeline = tuple(part.strip() for part in row["safe_pipeline"].split(" -> ") if part.strip())
        replay_key = key + (row["safe_pipeline"],)
        if replay_key not in distinct_replays:
            encoded = codec_module.encode_stream(evidence["evaluation_values"], pipeline, operator_module)
            decoded, decoded_pipeline = codec_module.decode_stream(encoded)
            if decoded_pipeline != pipeline or not np.array_equal(decoded, evidence["evaluation_values"]):
                raise AssertionError((replay_key, "independent replay failed"))
            distinct_replays[replay_key] = (len(encoded), digest(encoded))
        encoded_bytes, encoded_hash = distinct_replays[replay_key]
        if int(row["encoded_bytes"]) != encoded_bytes or row["encoded_sha256"] != encoded_hash:
            raise AssertionError((replay_key, "encoded artifact mismatch"))
    if any(len(items) != 1 for items in input_hashes.values()):
        raise AssertionError("algorithms did not receive the same suffix")

    codec_by_key = {
        (row["algorithm"], int(row["seed"]), row["dataset"], int(row["column_index"])): row
        for row in codec if row["algorithm"] != "Beam-29"
    }
    published = {row["variant"]: row for row in rows(run_root / "analysis" / "variant_results.csv")}
    recomputed_effects = {}
    max_published_effect_error = 0.0
    for variant in VARIANTS[1:]:
        dataset_effects = []
        for dataset, columns in selected.items():
            column_effects = []
            for column in columns:
                seed_effects = []
                for seed in (20260825, 20260826, 20260827):
                    candidate = codec_by_key[(variant, seed, dataset, column)]
                    reference = codec_by_key[("C0", seed, dataset, column)]
                    seed_effects.append(100.0 * (float(candidate["codec_rate"]) - float(reference["codec_rate"])))
                column_effects.append(float(np.mean(seed_effects)))
            dataset_effects.append(float(np.mean(column_effects)))
        effect = float(np.mean(dataset_effects))
        recomputed_effects[variant] = effect
        error = abs(effect - float(published[variant]["macro_delta_rate_pp"]))
        max_published_effect_error = max(max_published_effect_error, error)
    if max_published_effect_error > 1.0e-12:
        raise AssertionError(("published effect mismatch", max_published_effect_error))

    expected_verdicts = analysis_manifest["verdicts"]
    if expected_verdicts != {key: published[key]["verdict"] for key in VARIANTS[1:]}:
        raise AssertionError("manifest/table verdict mismatch")
    if not math.isfinite(max_published_effect_error):
        raise AssertionError("non-finite audit result")

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "fresh_proxy_search": {
            "datasets": selected,
            "columns": len(column_keys),
            "pipelines_per_column": 22764,
            "rows": len(measurement_rows),
            "all_lossless": True,
            "shards": len(shard_manifests),
            "grammar_ineligible_columns": EXCLUDED_SELECTION,
            "grammar_ineligible_recorded_rows": len(excluded_measurement_rows),
        },
        "split": {
            "calibration_rows": [0, 32768],
            "evaluation_rows": [32768, 65536],
            "independently_reloaded": True,
            "calibration_hashes_match": True,
            "evaluation_hashes_match": True,
        },
        "training": {
            "development_datasets": ["Metro-Traffic", "Air-Quality"],
            "external_test_datasets": list(TEST_DATASETS),
            "runs": len(status),
            "iterations_each": 80,
            "evaluation_rows": len(evaluation),
            "function_match_max_error": max(raw_manifest["function_match_max_abs_logit_error"].values()),
            "history_code_cardinality": expected_cardinality,
        },
        "codec": {
            "rows": len(codec),
            "lossless_rows": sum(int(row["lossless"]) for row in codec),
            "distinct_column_pipeline_replays": len(distinct_replays),
            "byte_and_hash_replay_match": True,
            "decoder_used_encoded_bytes_only": True,
        },
        "analysis": {
            "recomputed_macro_delta_rate_pp": recomputed_effects,
            "max_published_effect_error": max_published_effect_error,
            "verdicts": expected_verdicts,
        },
        "claim_boundary": {
            "standalone_normalized_int64_python_codec": True,
            "probe_conditioned": True,
            "production_codec": False,
            "zero_shot": False,
            "individual_bit_necessity": False,
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
