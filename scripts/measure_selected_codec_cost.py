#!/usr/bin/env python3
"""Measure a standalone byte-stream codec for selected operator pipelines.

This is deliberately narrower than a production codec.  It serializes the
pipeline IDs, every transform's forward metadata, and the compact integer
payload into a zlib envelope.  Decoding starts only from that envelope: it does
not receive the original values, a pipeline argument, or the encoder's inverse
closures. Transform, compact packing, and inversion execute in C++; Python
retains report/data handling and header/metadata parsing only. The measured
wall times include CLI process launch and IPC, not isolated C++ kernel cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MAGIC = b"OPC1"
VERSION = 1
OPERATOR_IDS = (
    "eraser",
    "scaling",
    "frame_of_reference",
    "value_reorder",
    "regression_predictor",
    "outlier_separation",
    "bitpack_int",
    "delta",
    "xor",
    "trailing_zero_pack",
    "dictionary",
    "zigzag",
    "block_transform",
    "symbol_recode",
    "run_length",
    "lz_match",
    "match_parse_policy",
    "match_window_model",
    "match_token_format",
    "integer_code",
    "bitstream_pack",
    "entropy_huffman",
    "entropy_range",
    "entropy_fse",
    "entropy_special",
    "literal_entropy",
    "block_state_control",
    "container_frame",
)
OPERATOR_TO_ID = {name: index for index, name in enumerate(OPERATOR_IDS)}
EXPECTED_TAG = {
    "eraser": "eraser",
    "scaling": "scaling",
    "frame_of_reference": "for",
    "value_reorder": "reorder",
    "regression_predictor": "regression",
    "outlier_separation": "outlier",
    "bitpack_int": "bitpack",
    "delta": "delta",
    "xor": "xor",
    "trailing_zero_pack": "trailing0",
    "dictionary": "dictionary",
    "zigzag": "zigzag",
    "block_transform": "block",
    "symbol_recode": "symbol",
    "run_length": "runlength",
    "lz_match": "lzmatch",
    "match_parse_policy": "policy",
    "match_window_model": "window",
    "match_token_format": "tokens",
    "integer_code": "intcode",
    "bitstream_pack": "bitstream",
    "entropy_huffman": "entropy_huff",
    "entropy_range": "entropy_rang",
    "entropy_fse": "entropy_fse",
    "entropy_special": "entropy_spec",
    "literal_entropy": "literal_entr",
    "block_state_control": "window",
    "container_frame": "container",
}


class CodecError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_operator_module(repo_root: Path):
    path = repo_root / "scripts" / "generate_air_operator_pair_heatmap.py"
    spec = importlib.util.spec_from_file_location("standalone_codec_operator_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load operator implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def need(raw: bytes, offset: int, count: int) -> None:
    if offset < 0 or count < 0 or offset + count > len(raw):
        raise CodecError(f"truncated stream at {offset}, need {count}, size {len(raw)}")


def parse_meta(raw: bytes, offset: int) -> tuple[tuple[str, list[np.ndarray], bytes], int]:
    need(raw, offset, 18)
    tag = raw[offset : offset + 12].rstrip(b"\0").decode("ascii")
    offset += 12
    array_count = int.from_bytes(raw[offset : offset + 2], "little")
    offset += 2
    extra_size = int.from_bytes(raw[offset : offset + 4], "little")
    offset += 4
    need(raw, offset, extra_size)
    extra = raw[offset : offset + extra_size]
    offset += extra_size
    arrays: list[np.ndarray] = []
    for _ in range(array_count):
        need(raw, offset, 8)
        count = int.from_bytes(raw[offset : offset + 4], "little")
        byte_count = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        offset += 8
        if byte_count != count * 8:
            raise CodecError(f"invalid array byte count {byte_count} for {count} int64 values")
        need(raw, offset, byte_count)
        array = np.frombuffer(raw, dtype="<i8", count=count, offset=offset).astype(np.int64, copy=True)
        arrays.append(array)
        offset += byte_count
    return (tag, arrays, extra), offset


def _reference_call(command: str, payload: bytes, names: tuple[str, ...] = ()) -> bytes:
    executable = Path(os.environ.get(
        "REFERENCE_OPERATOR_ACTIONS_CLI",
        str(Path(__file__).resolve().parents[1] / "cpp/build/reference_operator_actions_cli"),
    ))
    if not executable.is_file():
        raise CodecError(f"build reference_operator_actions_cli first: {executable}")
    result = subprocess.run(
        [str(executable), command, *names], input=payload, capture_output=True, check=False,
    )
    if result.returncode:
        raise CodecError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def parse_compact_payload(raw: bytes, offset: int) -> np.ndarray:
    need(raw, offset, 6)
    decoded = _reference_call("compact-decode", raw[offset:])
    return np.frombuffer(decoded, dtype="<i8").copy()


def invert_step(current: np.ndarray, tag: str, arrays: list[np.ndarray], extra: bytes) -> np.ndarray:
    """Legacy metadata API: construct a record, then invert only in C++."""
    record = bytearray(tag.encode("ascii")[:12].ljust(12, b"\0"))
    record.extend(len(arrays).to_bytes(2, "little"))
    record.extend(len(extra).to_bytes(4, "little"))
    record.extend(extra)
    for array in arrays:
        values = np.asarray(array, dtype="<i8")
        payload = values.tobytes()
        record.extend(len(values).to_bytes(4, "little"))
        record.extend(len(payload).to_bytes(4, "little"))
        record.extend(payload)
    compact = _reference_call("compact-encode", np.asarray(current, dtype="<i8").tobytes())
    decoded = _reference_call("inverse-step", bytes(record) + compact, (tag,))
    return np.frombuffer(decoded, dtype="<i8").copy()


def encode_stream(values: np.ndarray, pipeline: tuple[str, ...], module) -> bytes:
    """Preserve OPC1 wire; module remains an API-compatibility argument."""
    if not pipeline or len(pipeline) > 255:
        raise CodecError(f"invalid pipeline depth {len(pipeline)}")
    for name in pipeline:
        if name not in OPERATOR_TO_ID:
            raise CodecError(f"unknown operator {name}")
    names = tuple(f"op.{name}.reference_v1" for name in pipeline)
    return _reference_call("legacy-encode", np.asarray(values, dtype="<i8").tobytes(), names)


def decode_stream(encoded: bytes) -> tuple[np.ndarray, tuple[str, ...]]:
    # This reads only the historical public IDs. C++ validates records, unpacks
    # the payload and reverses the complete pipeline from the original wire.
    try:
        raw = zlib.decompress(encoded)
    except zlib.error as exc:
        raise CodecError("invalid zlib envelope") from exc
    need(raw, 0, 6)
    if raw[:4] != MAGIC or raw[4] != VERSION:
        raise CodecError("invalid codec header")
    depth = raw[5]
    need(raw, 6, depth)
    try:
        pipeline = tuple(OPERATOR_IDS[value] for value in raw[6 : 6 + depth])
    except IndexError as exc:
        raise CodecError("invalid operator ID") from exc
    decoded = _reference_call("legacy-decode", encoded)
    return np.frombuffer(decoded, dtype="<i8").copy(), pipeline


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
        repo_root / "orignial_datasets" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data = next((path for path in data_candidates if path.is_file()), None)
    info = next((path for path in info_candidates if path.is_file()), None)
    if data is None or info is None:
        raise FileNotFoundError(f"missing data or metadata for {dataset}")
    return data, info


def load_values(
    repo_root: Path,
    input_rows: list[dict[str, str]],
    module,
    row_offset: int = 0,
    max_values: int = 0,
) -> dict[tuple[str, int], np.ndarray]:
    wanted: dict[str, dict[int, str]] = defaultdict(dict)
    for row in input_rows:
        wanted[row["dataset"]][int(row["column_index"])] = row["column_name"]
    result: dict[tuple[str, int], np.ndarray] = {}
    for dataset, columns in sorted(wanted.items()):
        data_path, info_path = locate_dataset(repo_root, dataset)
        metadata = {int(row["column_index"]): row for row in read_csv(info_path)}
        frame = pd.read_csv(
            data_path,
            usecols=list(columns.values()),
            skiprows=range(1, row_offset + 1) if row_offset else None,
            nrows=max_values or None,
            low_memory=False,
            dtype=str,
        )
        for column_index, column_name in sorted(columns.items()):
            if metadata[column_index]["column_name"] != column_name:
                raise AssertionError((dataset, column_index, column_name, metadata[column_index]["column_name"]))
            if max_values and len(frame[column_name]) != max_values:
                raise AssertionError(
                    f"short source row slice {dataset}.{column_name}: {len(frame[column_name])} != {max_values}"
                )
            values = module.column_values(frame[column_name], metadata[column_index])
            result[(dataset, column_index)] = np.asarray(values, dtype=np.int64)
    return result


def median_ms(samples_ns: list[int]) -> float:
    return statistics.median(samples_ns) / 1_000_000.0


def measure_one(values: np.ndarray, pipeline: tuple[str, ...], module, repetitions: int) -> dict[str, object]:
    warm = encode_stream(values, pipeline, module)
    restored, decoded_pipeline = decode_stream(warm)
    if decoded_pipeline != pipeline or restored.shape != values.shape or not np.array_equal(restored, values):
        raise AssertionError(f"standalone round trip failed for {pipeline}")
    encode_ns: list[int] = []
    decode_ns: list[int] = []
    encoded = warm
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        candidate = encode_stream(values, pipeline, module)
        encode_ns.append(time.perf_counter_ns() - started)
        if candidate != encoded:
            raise AssertionError("codec is not deterministic")
        started = time.perf_counter_ns()
        restored, decoded_pipeline = decode_stream(candidate)
        decode_ns.append(time.perf_counter_ns() - started)
        if decoded_pipeline != pipeline or not np.array_equal(restored, values):
            raise AssertionError("timed standalone round trip failed")
    return {
        "original_values": int(values.size),
        "original_bytes": int(values.size * 8),
        "encoded_bytes": len(encoded),
        "codec_rate": len(encoded) / float(values.size * 8),
        "encode_ms_median": median_ms(encode_ns),
        "decode_ms_median": median_ms(decode_ns),
        "encode_ms_min": min(encode_ns) / 1_000_000.0,
        "decode_ms_min": min(decode_ns) / 1_000_000.0,
        "input_sha256": hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest(),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "lossless": 1,
        "timing_scope": "python_cli_wall_including_process_launch_ipc",
    }


def self_check(repo_root: Path) -> None:
    module = load_operator_module(repo_root)
    samples = (
        np.asarray([], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([-7, -7, -3, 0, 1, 1, 1, 18, 18, 99], dtype=np.int64),
        np.arange(-35, 37, dtype=np.int64),
    )
    pipelines = [(operator,) for operator in OPERATOR_IDS]
    pipelines.extend(
        tuple(OPERATOR_IDS[(start + offset) % len(OPERATOR_IDS)] for offset in range(3))
        for start in range(len(OPERATOR_IDS))
    )
    checks = 0
    for values in samples:
        for pipeline in pipelines:
            encoded = encode_stream(values, pipeline, module)
            restored, decoded_pipeline = decode_stream(encoded)
            if decoded_pipeline != pipeline or restored.shape != values.shape or not np.array_equal(restored, values):
                raise AssertionError((values.tolist(), pipeline, restored.tolist(), decoded_pipeline))
            checks += 1
    print(json.dumps({"self_check": "PASS", "round_trips": checks, "operators": len(OPERATOR_IDS)}))


def summarize(output_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per_column = []
    grouped: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in output_rows:
        grouped[(row["algorithm"], row["dataset"], int(row["column_index"]), row["column_name"])].append(row)
    for (algorithm, dataset, column_index, column_name), rows in sorted(grouped.items()):
        rates = [float(row["codec_rate"]) for row in rows]
        gains = [float(row["codec_gain_vs_baseline_pct"]) for row in rows]
        per_column.append(
            {
                "algorithm": algorithm,
                "dataset": dataset,
                "column_index": column_index,
                "column_name": column_name,
                "baseline_algorithm": rows[0]["baseline_algorithm"],
                "baseline_rate": rows[0]["baseline_rate"],
                "median_codec_rate": statistics.median(rates),
                "median_codec_gain_vs_baseline_pct": statistics.median(gains),
                "min_codec_gain_vs_baseline_pct": min(gains),
                "max_codec_gain_vs_baseline_pct": max(gains),
                "positive_seeds": sum(gain > 0 for gain in gains),
                "median_encode_ms": statistics.median(float(row["encode_ms_median"]) for row in rows),
                "median_decode_ms": statistics.median(float(row["decode_ms_median"]) for row in rows),
            }
        )
    algorithm_rows = []
    for algorithm in sorted({row["algorithm"] for row in per_column}):
        rows = [row for row in per_column if row["algorithm"] == algorithm]
        gains = [float(row["median_codec_gain_vs_baseline_pct"]) for row in rows]
        algorithm_rows.append(
            {
                "algorithm": algorithm,
                "columns": len(rows),
                "positive_columns": sum(gain > 0 for gain in gains),
                "all_three_seeds_positive": sum(int(row["positive_seeds"]) == 3 for row in rows),
                "median_column_gain_pct": statistics.median(gains),
                "mean_column_gain_pct": statistics.mean(gains),
                "min_column_gain_pct": min(gains),
                "max_column_gain_pct": max(gains),
                "median_encode_ms": statistics.median(float(row["median_encode_ms"]) for row in rows),
                "median_decode_ms": statistics.median(float(row["median_decode_ms"]) for row in rows),
            }
        )
    return per_column, algorithm_rows


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    input_rows = read_csv(args.evaluations_csv.resolve())
    required = {"algorithm", "seed", "fold", "dataset", "column_index", "column_name", "safe_pipeline", "baseline_algorithm", "baseline_rate"}
    if not input_rows or not required.issubset(input_rows[0]):
        raise ValueError(f"evaluation CSV lacks required columns: {sorted(required - set(input_rows[0] if input_rows else {}))}")
    module = load_operator_module(repo_root)
    values_by_column = load_values(
        repo_root, input_rows, module, row_offset=args.row_offset, max_values=args.max_values
    )
    cache: dict[tuple[str, int, tuple[str, ...]], dict[str, object]] = {}
    output_rows = []
    for input_row in input_rows:
        key = (input_row["dataset"], int(input_row["column_index"]))
        pipeline = tuple(part.strip() for part in input_row["safe_pipeline"].split(" -> ") if part.strip())
        cache_key = (key[0], key[1], pipeline)
        if cache_key not in cache:
            cache[cache_key] = measure_one(values_by_column[key], pipeline, module, args.repetitions)
        measured = cache[cache_key]
        baseline_rate = float(input_row["baseline_rate"])
        codec_rate = float(measured["codec_rate"])
        output_rows.append(
            {
                "algorithm": input_row["algorithm"],
                "seed": input_row["seed"],
                "fold": input_row["fold"],
                "dataset": input_row["dataset"],
                "column_index": input_row["column_index"],
                "column_name": input_row["column_name"],
                "logical_type": input_row.get("logical_type", ""),
                "safe_pipeline": input_row["safe_pipeline"],
                "baseline_algorithm": input_row["baseline_algorithm"],
                "baseline_rate": baseline_rate,
                **measured,
                "codec_gain_vs_baseline_pct": 100.0 * (baseline_rate - codec_rate) / baseline_rate,
                "proxy_safe_rate": input_row.get("safe_proxy_rate", ""),
                "codec_minus_proxy_rate": codec_rate - float(input_row.get("safe_proxy_rate") or math.nan),
                "timing_repetitions": args.repetitions,
            }
        )
    if len(output_rows) != len(input_rows) or any(int(row["lossless"]) != 1 for row in output_rows):
        raise AssertionError("incomplete or non-lossless codec results")
    per_column, algorithm_rows = summarize(output_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed_codec_cost.csv", output_rows)
    write_csv(args.output_dir / "per_column_codec_summary.csv", per_column)
    write_csv(args.output_dir / "algorithm_codec_summary.csv", algorithm_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "real_python_standalone_proxy_codec",
        "input_evaluations": str(args.evaluations_csv.resolve()),
        "input_rows": len(input_rows),
        "unique_pipeline_column_pairs": len(cache),
        "lossless_rows": sum(int(row["lossless"]) for row in output_rows),
        "timing_repetitions": args.repetitions,
        "row_slice": {
            "offset": args.row_offset,
            "max_values": args.max_values,
            "contract": "evaluation rows are disjoint from calibration rows when offset is nonzero",
        },
        "codec_envelope": "zlib level 6 over versioned pipeline IDs, transform metadata, and compact int payload",
        "decoder_contract": "decode_stream receives only encoded bytes and reconstructs the normalized int64 sequence",
        "source_semantics": "finite CSV values normalized by declared decimal scaling to int64; exact round trip is on this normalized sequence",
        "production_claim_allowed": False,
        "production_claim_reason": "conceptual Python transforms share proxy implementations and are not production kernels",
        "algorithms": algorithm_rows,
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluations-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--max-values", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.row_offset < 0 or args.max_values < 0:
        parser.error("--row-offset and --max-values must be non-negative")
    if not args.self_check and (args.evaluations_csv is None or args.output_dir is None):
        parser.error("--evaluations-csv and --output-dir are required unless --self-check is used")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check(args.repo_root.resolve())
    else:
        run(args)


if __name__ == "__main__":
    main()
