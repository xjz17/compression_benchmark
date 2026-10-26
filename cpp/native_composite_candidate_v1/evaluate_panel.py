#!/usr/bin/env python3
"""CPU-pinned end-to-end panel for the two frozen coarse codec selectors.

This script orchestrates measurements only. Compression selection, feature
extraction, model inference, complete NGF1 encode, and complete decode remain
inside the C++ executable and its timers. Every retained frame is additionally
decoded by a fresh process and compared byte-for-byte with both input roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import time
from collections import Counter
from typing import Any


WEIGHTS = {
    "100": (1.0, 0.0, 0.0),
    "010": (0.0, 1.0, 0.0),
    "001": (0.0, 0.0, 1.0),
    "550": (0.5, 0.5, 0.0),
    "505": (0.5, 0.0, 0.5),
    "433": (0.4, 0.3, 0.3),
}
SELECTORS = ("global_linear", "data_aware", "zstd")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode()).hexdigest()


def median(values: list[int]) -> float:
    require(bool(values), "timing sample list is empty")
    return float(statistics.median(values))


def command_prefix(cpu: int | None) -> list[str]:
    return [] if cpu is None else ["taskset", "-c", str(cpu)]


def run(command: list[str], cpu: int | None, env: dict[str, str],
        capture: bool = True) -> str:
    result = subprocess.run(command_prefix(cpu) + command, check=True,
                            text=True, capture_output=capture, env=env)
    return result.stdout.strip() if capture else ""


def roots(sample: dict[str, Any], manifest_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    numeric = [root for root in sample["roots"] if root["kind"] == "numeric"]
    bitmap = [root for root in sample["roots"] if root["kind"] == "serial"]
    require(len(numeric) == len(bitmap) == 1, "expected numeric+bitmap roots")
    values = manifest_dir / numeric[0]["numeric_path"]
    missing = manifest_dir / bitmap[0]["bitmap_path"]
    require(values.is_file() and missing.is_file(), "prepared payload is absent")
    require(sha256(values) == numeric[0]["numeric_sha256"],
            "numeric payload SHA-256 mismatch")
    require(sha256(missing) == bitmap[0]["bitmap_sha256"],
            "bitmap payload SHA-256 mismatch")
    require(values.stat().st_size == sample["B_raw"] == numeric[0]["B_raw"],
            "B_raw differs from original numeric root bytes")
    return values, missing


def selector_token(label: str, model: pathlib.Path) -> str:
    if label == "global_linear":
        return "train-linear-v1"
    if label == "data_aware":
        return f"data-aware-v1@{model}"
    if label == "zstd":
        return "Zstd"
    raise AssertionError(label)


def measure_one(*, cli: pathlib.Path, model: pathlib.Path, selector: str,
                weight_label: str, weight: tuple[float, float, float],
                sample: dict[str, Any], manifest_dir: pathlib.Path,
                frame: pathlib.Path, cold_dir: pathlib.Path, warmups: int,
                repeats: int, cpu: int | None, env: dict[str, str]) -> dict[str, Any]:
    values, missing = roots(sample, manifest_dir)
    frame.parent.mkdir(parents=True, exist_ok=True)
    if selector == "zstd":
        command = [str(cli), "measure-direct", "Zstd", sample["dtype"],
                   str(sample["row_count"]), str(values), str(missing),
                   str(warmups), str(repeats), str(frame)]
    else:
        command = [str(cli), "measure", selector_token(selector, model),
                   *(format(value, ".17g") for value in weight), sample["dtype"],
                   str(sample["row_count"]), str(values), str(missing),
                   str(warmups), str(repeats), str(frame)]
    output = json.loads(run(command, cpu, env))
    require(output["status"] == "ok", "non-finite panel row")
    require(output["B_raw"] == sample["B_raw"], "CLI B_raw mismatch")
    require(output["B_compressed"] == frame.stat().st_size,
            "CLI compressed byte count mismatch")
    require(len(output["compression_ns_samples"]) == repeats and
            len(output["decompression_ns_samples"]) == repeats,
            "CLI timing sample count mismatch")

    decoded_values = cold_dir / "decoded.values.bin"
    decoded_missing = cold_dir / "decoded.missing.bitmap"
    cold_dir.mkdir(parents=True, exist_ok=True)
    decode_command = "decode-direct" if selector == "zstd" else "decode"
    cold_stdout = run([str(cli), decode_command, str(frame), str(decoded_values),
                       str(decoded_missing)], cpu, env)
    require(sha256(decoded_values) == sha256(values),
            "cold decoded numeric root differs")
    require(sha256(decoded_missing) == sha256(missing),
            "cold decoded bitmap root differs")
    decoded_values.unlink()
    decoded_missing.unlink()

    b_raw = int(output["B_raw"])
    ratio = int(output["B_compressed"]) / b_raw
    encode_ns = median(output["compression_ns_samples"])
    decode_ns = median(output["decompression_ns_samples"])
    require(math.isclose(ratio, output["compression_ratio"], rel_tol=0,
                         abs_tol=1e-15), "CLI compression ratio drift")
    row = {
        "split": sample["split"],
        "sample_id": sample["sample_id"],
        "calibration_key": sample.get("calibration_key"),
        "dataset_id": sample["dataset_id"],
        "domain": sample["domain"],
        "dtype": sample["dtype"],
        "row_count": sample["row_count"],
        "weight": weight_label,
        "weights": list(weight),
        "selector": selector,
        "selected_backend": output["backend"],
        "selector_identity": output["selector_identity"],
        "B_raw": b_raw,
        "B_compressed": int(output["B_compressed"]),
        "compression_ns_samples": output["compression_ns_samples"],
        "decompression_ns_samples": output["decompression_ns_samples"],
        "compression_ratio": ratio,
        "compression_ns_per_byte": encode_ns / b_raw,
        "decompression_ns_per_byte": decode_ns / b_raw,
        "frame_path": str(frame),
        "frame_sha256": sha256(frame),
        "cold_decode": "pass",
        "cold_decode_stdout": cold_stdout,
        "values_sha256": sha256(values),
        "bitmap_sha256": sha256(missing),
    }
    return row


def objective(row: dict[str, Any], e0: float, d0: float) -> float:
    wr, we, wd = row["weights"]
    e = row["compression_ns_per_byte"]
    d = row["decompression_ns_per_byte"]
    return (wr * row["compression_ratio"] + we * e / (e + e0) +
            wd * d / (d + d0))


def aggregate(rows: list[dict[str, Any]], split: str, e0: float,
              d0: float) -> dict[str, Any]:
    split_rows = [row for row in rows if row["split"] == split]
    result: dict[str, Any] = {}
    for selector in ("global_linear", "data_aware"):
        result[selector] = {}
        for label in WEIGHTS:
            selected = [row for row in split_rows
                        if row["selector"] == selector and row["weight"] == label]
            baseline = [row for row in split_rows
                        if row["selector"] == "zstd" and row["weight"] == label]
            require(len(selected) == len(baseline) and bool(selected),
                    "unpaired selector/Zstd rows")
            require({row["sample_id"] for row in selected} ==
                    {row["sample_id"] for row in baseline},
                    "selector/Zstd sample identities differ")
            total = sum(row["B_raw"] for row in selected)
            require(total == sum(row["B_raw"] for row in baseline),
                    "selector/Zstd denominator differs")

            def mean(group: list[dict[str, Any]], key: str) -> float:
                return math.fsum(row["B_raw"] * row[key] for row in group) / total

            model_metrics = {
                "R": mean(selected, "compression_ratio"),
                "E_ns_per_byte": mean(selected, "compression_ns_per_byte"),
                "D_ns_per_byte": mean(selected, "decompression_ns_per_byte"),
                "J": math.fsum(row["B_raw"] * objective(row, e0, d0)
                                for row in selected) / total,
            }
            zstd_metrics = {
                "R": mean(baseline, "compression_ratio"),
                "E_ns_per_byte": mean(baseline, "compression_ns_per_byte"),
                "D_ns_per_byte": mean(baseline, "decompression_ns_per_byte"),
                "J": math.fsum(row["B_raw"] * objective(row, e0, d0)
                                for row in baseline) / total,
            }
            ratio = model_metrics["J"] / zstd_metrics["J"]
            result[selector][label] = {
                "B_raw": total,
                "block_count": len(selected),
                "model": model_metrics,
                "same_run_Zstd": zstd_metrics,
                "J_over_Zstd": ratio,
                "strictly_better_than_Zstd": ratio < 1.0,
                "decision_counts": dict(sorted(Counter(
                    row["selected_backend"] for row in selected).items())),
            }
        values = [result[selector][label]["J_over_Zstd"] for label in WEIGHTS]
        result[selector]["gate"] = {
            "all_six_strictly_better": all(value < 1.0 for value in values),
            "worst_J_over_Zstd": max(values),
            "mean_J_over_Zstd": math.fsum(values) / len(values),
        }
    return result


def selection(summary: dict[str, Any]) -> dict[str, Any]:
    validation = summary["validation"]
    simple = validation["global_linear"]["gate"]
    aware = validation["data_aware"]["gate"]
    if aware["all_six_strictly_better"] and not simple["all_six_strictly_better"]:
        chosen, reason = "data_aware", "only data-aware passes all-six gate"
    elif simple["all_six_strictly_better"] and not aware["all_six_strictly_better"]:
        chosen, reason = "global_linear", "only global linear passes all-six gate"
    elif not simple["all_six_strictly_better"] and not aware["all_six_strictly_better"]:
        chosen, reason = None, "neither frozen candidate passes all-six gate"
    else:
        relative_improvement = ((simple["worst_J_over_Zstd"] -
                                 aware["worst_J_over_Zstd"]) /
                                simple["worst_J_over_Zstd"])
        if relative_improvement >= 0.005:
            chosen, reason = "data_aware", "worst ratio improves by at least 0.5%"
        else:
            chosen, reason = "global_linear", "data-aware worst-ratio gain is below 0.5%"
            if (math.isclose(relative_improvement, 0.005, rel_tol=0,
                             abs_tol=1e-15) and
                    aware["mean_J_over_Zstd"] < simple["mean_J_over_Zstd"]):
                chosen, reason = "data_aware", "threshold tie resolved by lower six-ratio mean"
        return {"chosen": chosen, "reason": reason,
                "data_aware_relative_worst_ratio_improvement": relative_improvement,
                "frozen_rule": "PROJECT_CONSTRAINTS.md section 10.28"}
    return {"chosen": chosen, "reason": reason,
            "frozen_rule": "PROJECT_CONSTRAINTS.md section 10.28"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--train-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--validation-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cpu", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    require(args.cli.is_file() and os.access(args.cli, os.X_OK),
            "C++ CLI is absent or not executable")
    require(args.model.is_file(), "frozen data-aware model is absent")
    require(args.warmups >= 0 and args.repeats > 0, "invalid timing counts")
    model_json = json.loads(args.model.read_bytes())
    require(model_json["schema"] == "train-only-direct-codec-data-aware-selector-v1",
            "unexpected model schema")
    e0 = float(model_json["objective"]["E0_ns_per_byte"])
    d0 = float(model_json["objective"]["D0_ns_per_byte"])
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "BLOSC_NTHREADS": "1",
                "BLOSC2_NTHREADS": "1"})
    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "rows.jsonl"
    started = time.time_ns()
    rows: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    with rows_path.open("w", encoding="utf-8") as sink:
        for expected_split, manifest_path in (
                ("train", args.train_manifest),
                ("validation", args.validation_manifest)):
            manifest = json.loads(manifest_path.read_bytes())
            manifests[expected_split] = manifest
            require(manifest["split"] == expected_split, "manifest split mismatch")
            require(manifest["paths_relative_to"] == "manifest_parent",
                    "unsupported payload path base")
            require(manifest["test_used_for_training_or_statistics"] is False and
                    manifest["web20_used_for_training_or_statistics"] is False,
                    "manifest admits held-out data")
            require(len(manifest["samples"]) == (40 if expected_split == "train" else 32),
                    "unexpected frozen panel size")
            for sample in manifest["samples"]:
                require(sample["split"] == expected_split, "sample split mismatch")
                for weight_label, weight in WEIGHTS.items():
                    for selector in SELECTORS:
                        frame = (args.output / "frames" / expected_split /
                                 selector / weight_label /
                                 f"{sample['sample_id']}.ngf")
                        cold = (args.output / "cold" / expected_split /
                                selector / weight_label /
                                sample["sample_id"])
                        row = measure_one(
                            cli=args.cli.resolve(), model=args.model.resolve(),
                            selector=selector, weight_label=weight_label,
                            weight=weight, sample=sample,
                            manifest_dir=manifest_path.resolve().parent,
                            frame=frame.resolve(), cold_dir=cold.resolve(),
                            warmups=args.warmups, repeats=args.repeats,
                            cpu=args.cpu, env=env)
                        rows.append(row)
                        sink.write(json.dumps(row, sort_keys=True) + "\n")
                        sink.flush()

    summary = {
        split: aggregate(rows, split, e0, d0)
        for split in ("train", "validation")
    }
    summary["selection"] = selection(summary)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    inventory = run([str(args.cli.resolve()), "inventory"], args.cpu, env)
    receipt = {
        "schema": "native-composite-candidate-panel-receipt-v1",
        "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_affinity": None if args.cpu is None else [args.cpu],
        "thread_environment": {key: env[key] for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLOSC_NTHREADS", "BLOSC2_NTHREADS")},
        "warmups": args.warmups,
        "repeats": args.repeats,
        "metrics": {
            "R": "complete NGF1 bytes / original numeric bytes",
            "E": "median complete in-process C++ encode ns / original numeric bytes",
            "D": "median complete in-process C++ decode ns / original numeric bytes",
            "E0_ns_per_byte": e0, "D0_ns_per_byte": d0,
            "lower_is_better": True,
        },
        "model_path": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "model_parameter_sha256": model_json["model_parameter_sha256"],
        "cli_path": str(args.cli.resolve()),
        "cli_sha256": sha256(args.cli),
        "train_manifest_path": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256(args.train_manifest),
        "validation_manifest_path": str(args.validation_manifest.resolve()),
        "validation_manifest_sha256": sha256(args.validation_manifest),
        "inventory_stdout": inventory,
        "row_count": len(rows),
        "rows_sha256": sha256(rows_path),
        "rows_canonical_sha256": canonical_sha(rows),
        "frame_count": len(rows),
        "cold_decode_pass_count": sum(row["cold_decode"] == "pass" for row in rows),
        "claim_boundary": "train40 and validation32 only; no Web20 or independent test",
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "rows": len(rows),
                      "selection": summary["selection"]}, sort_keys=True))


if __name__ == "__main__":
    main()
