#!/usr/bin/env python3
"""Exact standalone-codec search through five operator applications.

Two modes are intentionally separated:

* ``fixed-tail`` exactly tests ``a -> b -> lz_match -> entropy_huffman ->
  container_frame`` against an exact depth<=3 oracle.
* ``full`` enumerates every canonical non-empty pipeline through depth 5.  The
  29th RL action is absorbing STOP, so canonical enumeration removes duplicate
  STOP padding and visits sum(28**d, d=1..5) pipelines.

The operator implementations remain conceptual transforms.  In particular,
``entropy_huffman`` is not a production Huffman bitstream implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import os
import socket
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TAIL = ("lz_match", "entropy_huffman", "container_frame")
PRECONDITIONERS = (
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
)


@dataclass(frozen=True)
class Candidate:
    encoded_bytes: int
    pipeline: tuple[str, ...]


_VALUES: np.ndarray | None = None
_OPERATOR_MODULE = None
_CODEC_MODULE = None
_OPERATORS: tuple[str, ...] = ()
_TOP_K = 8


def import_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_dataset(repo_root: Path, dataset: str) -> tuple[Path, Path]:
    data_candidates = (
        repo_root / "backend" / "dataset_csv" / f"{dataset}.csv",
        repo_root / "orignial_datasets" / f"{dataset}.csv",
    )
    info_candidates = (
        repo_root / "backend" / "dataset" / dataset / f"{dataset}_column_info.csv",
        repo_root / "dataset" / dataset / f"{dataset}_column_info.csv",
    )
    data_path = next((path for path in data_candidates if path.is_file()), None)
    info_path = next((path for path in info_candidates if path.is_file()), None)
    if data_path is None or info_path is None:
        raise FileNotFoundError(f"missing data or metadata for {dataset}")
    return data_path, info_path


def load_column(operator_module, repo_root: Path, dataset: str, column_index: int):
    data_path, info_path = locate_dataset(repo_root, dataset)
    with info_path.open("r", encoding="utf-8", newline="") as handle:
        metadata = {int(row["column_index"]): row for row in csv.DictReader(handle)}
    if column_index not in metadata:
        raise ValueError(f"unknown column index {column_index} for {dataset}")
    column = metadata[column_index]
    logical = str(column.get("logical_type", "")).strip().lower()
    if logical not in {"integer", "int", "long", "float", "double"}:
        raise ValueError(f"column {column_index} is not numeric: {logical}")
    frame = pd.read_csv(data_path, usecols=[column["column_name"]], low_memory=False, dtype=str)
    values = operator_module.column_values(frame[column["column_name"]], column)
    return data_path, info_path, column, np.asarray(values, dtype=np.int64)


def score_state(codec, state, pipeline: tuple[str, ...]) -> int:
    operator_bytes = bytes(codec.OPERATOR_TO_ID[name] for name in pipeline)
    raw = codec.MAGIC + bytes((codec.VERSION, len(pipeline))) + operator_bytes + state.serialized()
    return len(zlib.compress(raw, level=6))


def encoded_candidate(codec, state, pipeline: tuple[str, ...]) -> Candidate:
    return Candidate(score_state(codec, state, pipeline), pipeline)


def candidate_key(candidate: Candidate) -> tuple[int, int, tuple[str, ...]]:
    return candidate.encoded_bytes, len(candidate.pipeline), candidate.pipeline


def insert_top(top: list[Candidate], candidate: Candidate, limit: int) -> None:
    top.append(candidate)
    top.sort(key=candidate_key)
    if len(top) > limit:
        del top[limit:]


def candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "encoded_bytes": candidate.encoded_bytes,
        "pipeline": list(candidate.pipeline),
        "pipeline_text": " -> ".join(candidate.pipeline),
        "depth": len(candidate.pipeline),
    }


def verify_candidate(codec, operator_module, values: np.ndarray, candidate: Candidate) -> dict[str, Any]:
    encoded = codec.encode_stream(values, candidate.pipeline, operator_module)
    if len(encoded) != candidate.encoded_bytes:
        raise AssertionError(
            f"incremental/full encode mismatch for {candidate.pipeline}: "
            f"{candidate.encoded_bytes} != {len(encoded)}"
        )
    restored, decoded_pipeline = codec.decode_stream(encoded)
    lossless = decoded_pipeline == candidate.pipeline and np.array_equal(restored, values)
    if not lossless:
        raise AssertionError(f"standalone round trip failed for {candidate.pipeline}")
    return {
        **candidate_dict(candidate),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "lossless": 1,
    }


def exact_through_depth3(codec, operator_module, values: np.ndarray, operators: tuple[str, ...]):
    base = operator_module.initial_state(values)
    best_by_depth: dict[int, Candidate] = {}
    counts = {1: 0, 2: 0, 3: 0}
    failures = 0

    def visit(prefix: tuple[str, ...], state) -> None:
        nonlocal failures
        if len(prefix) >= 3:
            return
        for operator in operators:
            pipeline = prefix + (operator,)
            try:
                next_state = operator_module.apply_operator(state, operator)
                candidate = encoded_candidate(codec, next_state, pipeline)
            except Exception:
                failures += 1
                continue
            depth = len(pipeline)
            counts[depth] += 1
            if depth not in best_by_depth or candidate_key(candidate) < candidate_key(best_by_depth[depth]):
                best_by_depth[depth] = candidate
            visit(pipeline, next_state)

    visit((), base)
    return best_by_depth, counts, failures


def fixed_tail_search(codec, operator_module, values: np.ndarray, operators: tuple[str, ...]):
    base = operator_module.initial_state(values)
    top_all: list[Candidate] = []
    top_preconditioners: list[Candidate] = []
    failures = 0
    evaluations = 0
    preconditioner_set = set(PRECONDITIONERS)
    for first, second in itertools.product(operators, repeat=2):
        pipeline = (first, second) + TAIL
        try:
            state = base
            for operator in pipeline:
                state = operator_module.apply_operator(state, operator)
            candidate = encoded_candidate(codec, state, pipeline)
        except Exception:
            failures += 1
            continue
        evaluations += 1
        insert_top(top_all, candidate, 20)
        if first in preconditioner_set and second in preconditioner_set:
            insert_top(top_preconditioners, candidate, 20)
    return top_all, top_preconditioners, evaluations, failures


def run_fixed_tail(args, codec, operator_module, operators, input_meta, values) -> None:
    started = time.perf_counter()
    best_by_depth, depth_counts, depth3_failures = exact_through_depth3(
        codec, operator_module, values, operators
    )
    top_all, top_preconditioners, tail_evaluations, tail_failures = fixed_tail_search(
        codec, operator_module, values, operators
    )
    expected_depth_counts = {depth: len(operators) ** depth for depth in range(1, 4)}
    if depth_counts != expected_depth_counts:
        raise AssertionError((depth_counts, expected_depth_counts))
    if tail_evaluations != len(operators) ** 2:
        raise AssertionError((tail_evaluations, len(operators) ** 2))
    exact_best = min(best_by_depth.values(), key=candidate_key)
    fixed_best = top_all[0]
    preconditioner_best = top_preconditioners[0]
    verified = [
        verify_candidate(codec, operator_module, values, candidate)
        for candidate in sorted(
            set(best_by_depth.values()) | {fixed_best, preconditioner_best}, key=candidate_key
        )
    ]
    original_bytes = int(values.size * 8)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "fixed-tail",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "dataset": args.dataset,
        "column_index": args.column_index,
        "column_name": input_meta["column_name"],
        "logical_type": input_meta.get("logical_type", ""),
        "sample_values": int(values.size),
        "original_bytes": original_bytes,
        "input_sha256": hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest(),
        "operators": list(operators),
        "tail": list(TAIL),
        "depth_counts": depth_counts,
        "depth3_evaluations": sum(depth_counts.values()),
        "fixed_tail_evaluations": tail_evaluations,
        "failures": depth3_failures + tail_failures,
        "exact_best_through_depth3": candidate_dict(exact_best),
        "fixed_tail_best": candidate_dict(fixed_best),
        "preconditioner_prefix_best": candidate_dict(preconditioner_best),
        "fixed_tail_gain_vs_exact_depth3_pct": 100.0
        * (exact_best.encoded_bytes - fixed_best.encoded_bytes)
        / exact_best.encoded_bytes,
        "preconditioner_tail_gain_vs_exact_depth3_pct": 100.0
        * (exact_best.encoded_bytes - preconditioner_best.encoded_bytes)
        / exact_best.encoded_bytes,
        "beats_exact_depth3": int(fixed_best.encoded_bytes < exact_best.encoded_bytes),
        "preconditioner_prefix_beats_exact_depth3": int(
            preconditioner_best.encoded_bytes < exact_best.encoded_bytes
        ),
        "rate_exact_depth3": exact_best.encoded_bytes / original_bytes,
        "rate_fixed_tail": fixed_best.encoded_bytes / original_bytes,
        "rate_preconditioner_tail": preconditioner_best.encoded_bytes / original_bytes,
        "best_by_depth": {
            str(depth): candidate_dict(candidate) for depth, candidate in sorted(best_by_depth.items())
        },
        "top_fixed_tail": [candidate_dict(candidate) for candidate in top_all],
        "top_preconditioner_fixed_tail": [
            candidate_dict(candidate) for candidate in top_preconditioners
        ],
        "verified_candidates": verified,
        "wall_seconds": time.perf_counter() - started,
        "proxy_warning": (
            "Conceptual transforms plus zlib; entropy_huffman is a frequency-ranked "
            "dictionary transform, not a production Huffman bitstream."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": summary}, ensure_ascii=False), flush=True)


def init_full_worker(values, operator_module, codec, operators, top_k: int) -> None:
    global _VALUES, _OPERATOR_MODULE, _CODEC_MODULE, _OPERATORS, _TOP_K
    _VALUES = values
    _OPERATOR_MODULE = operator_module
    _CODEC_MODULE = codec
    _OPERATORS = operators
    _TOP_K = top_k


def full_pair_shard(pair_index: int) -> dict[str, Any]:
    if _VALUES is None:
        raise RuntimeError("worker is not initialized")
    started = time.perf_counter()
    operator_count = len(_OPERATORS)
    first_index, second_index = divmod(pair_index, operator_count)
    first = _OPERATORS[first_index]
    second = _OPERATORS[second_index]
    base = _OPERATOR_MODULE.initial_state(_VALUES)
    failures = 0
    counts = {2: 0, 3: 0, 4: 0, 5: 0}
    top_by_depth: dict[int, list[Candidate]] = {2: [], 3: [], 4: [], 5: []}

    try:
        state1 = _OPERATOR_MODULE.apply_operator(base, first)
        state2 = _OPERATOR_MODULE.apply_operator(state1, second)
        pair = (first, second)
        insert_top(top_by_depth[2], encoded_candidate(_CODEC_MODULE, state2, pair), _TOP_K)
        counts[2] += 1
    except Exception as exc:
        return {
            "pair_index": pair_index,
            "first": first,
            "second": second,
            "counts": counts,
            "failures": 1,
            "fatal": type(exc).__name__,
            "top_by_depth": {},
            "wall_seconds": time.perf_counter() - started,
        }

    def visit(prefix: tuple[str, ...], state) -> None:
        nonlocal failures
        if len(prefix) >= 5:
            return
        for operator in _OPERATORS:
            pipeline = prefix + (operator,)
            try:
                next_state = _OPERATOR_MODULE.apply_operator(state, operator)
                candidate = encoded_candidate(_CODEC_MODULE, next_state, pipeline)
            except Exception:
                failures += 1
                continue
            depth = len(pipeline)
            counts[depth] += 1
            insert_top(top_by_depth[depth], candidate, _TOP_K)
            visit(pipeline, next_state)

    visit(pair, state2)
    return {
        "pair_index": pair_index,
        "first": first,
        "second": second,
        "counts": counts,
        "failures": failures,
        "fatal": "",
        "top_by_depth": {
            str(depth): [candidate_dict(item) for item in candidates]
            for depth, candidates in top_by_depth.items()
        },
        "wall_seconds": time.perf_counter() - started,
    }


def read_completed_shards(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARN ignoring incomplete shard line {line_number}", file=sys.stderr)
                continue
            rows[int(row["pair_index"])] = row
    return rows


def select_pair_indices(args, operator_count: int) -> list[int]:
    total = operator_count**2
    if args.pair_indices:
        selected = sorted(set(args.pair_indices))
    elif args.pair_limit:
        selected = list(range(min(args.pair_limit, total)))
    else:
        selected = list(range(total))
    if any(index < 0 or index >= total for index in selected):
        raise ValueError(f"pair index must be in [0, {total})")
    return selected


def merge_full_results(
    codec,
    operator_module,
    values: np.ndarray,
    operators: tuple[str, ...],
    shard_rows: Iterable[dict[str, Any]],
):
    best_by_depth: dict[int, Candidate] = {}
    counts = {depth: 0 for depth in range(1, 6)}
    failures = 0
    base = operator_module.initial_state(values)
    for operator in operators:
        state = operator_module.apply_operator(base, operator)
        candidate = encoded_candidate(codec, state, (operator,))
        counts[1] += 1
        if 1 not in best_by_depth or candidate_key(candidate) < candidate_key(best_by_depth[1]):
            best_by_depth[1] = candidate
    for row in shard_rows:
        failures += int(row["failures"])
        for depth_text, count in row["counts"].items():
            counts[int(depth_text)] += int(count)
        for depth_text, candidates in row.get("top_by_depth", {}).items():
            depth = int(depth_text)
            for item in candidates:
                candidate = Candidate(int(item["encoded_bytes"]), tuple(item["pipeline"]))
                if depth not in best_by_depth or candidate_key(candidate) < candidate_key(
                    best_by_depth[depth]
                ):
                    best_by_depth[depth] = candidate
    verified = {
        str(depth): verify_candidate(codec, operator_module, values, candidate)
        for depth, candidate in sorted(best_by_depth.items())
    }
    return best_by_depth, counts, failures, verified


def run_full(args, codec, operator_module, operators, input_meta, values) -> None:
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("full mode requires multiprocessing fork semantics")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    shard_path = args.output_dir / "pair_shards.jsonl"
    completed = read_completed_shards(shard_path) if args.resume else {}
    selected = select_pair_indices(args, len(operators))
    pending = [index for index in selected if index not in completed]
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "mode": "full",
                "selected_pair_shards": len(selected),
                "completed_pair_shards": len(completed),
                "pending_pair_shards": len(pending),
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    if pending:
        context = mp.get_context("fork")
        with shard_path.open("a", encoding="utf-8", buffering=1) as handle:
            with context.Pool(
                processes=args.workers,
                initializer=init_full_worker,
                initargs=(values, operator_module, codec, operators, args.top_k),
            ) as pool:
                for done_count, row in enumerate(
                    pool.imap_unordered(full_pair_shard, pending, chunksize=1), start=1
                ):
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    completed[int(row["pair_index"])] = row
                    if done_count == 1 or done_count % args.progress_every == 0:
                        elapsed = time.perf_counter() - started
                        print(
                            json.dumps(
                                {
                                    "finished_now": done_count,
                                    "pending_at_start": len(pending),
                                    "total_completed": len(completed),
                                    "elapsed_seconds": elapsed,
                                    "mean_seconds_per_finished_shard": elapsed / done_count,
                                }
                            ),
                            flush=True,
                        )
    chosen_rows = [completed[index] for index in selected if index in completed]
    best_by_depth, counts, failures, verified = merge_full_results(
        codec, operator_module, values, operators, chosen_rows
    )
    expected_depth_counts = {depth: len(operators) ** depth for depth in range(1, 6)}
    complete_space = len(selected) == len(operators) ** 2 and len(chosen_rows) == len(selected)
    if complete_space and counts != expected_depth_counts:
        raise AssertionError((counts, expected_depth_counts))
    best = min(best_by_depth.values(), key=candidate_key)
    depth3_best = min(
        (candidate for depth, candidate in best_by_depth.items() if depth <= 3),
        key=candidate_key,
    )
    original_bytes = int(values.size * 8)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "dataset": args.dataset,
        "column_index": args.column_index,
        "column_name": input_meta["column_name"],
        "logical_type": input_meta.get("logical_type", ""),
        "sample_values": int(values.size),
        "original_bytes": original_bytes,
        "input_sha256": hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest(),
        "operators": list(operators),
        "workers": args.workers,
        "selected_pair_shards": len(selected),
        "completed_pair_shards": len(chosen_rows),
        "complete_canonical_space": complete_space,
        "canonical_stop_semantics": "absorbing STOP is removed as suffix padding",
        "raw_fixed_horizon_action_strings": 29**5,
        "expected_canonical_nonempty_pipelines": sum(len(operators) ** depth for depth in range(1, 6)),
        "expected_depth_counts": expected_depth_counts,
        "observed_depth_counts": counts,
        "observed_evaluations": sum(counts.values()),
        "failures": failures,
        "best_by_depth": {
            str(depth): candidate_dict(candidate) for depth, candidate in sorted(best_by_depth.items())
        },
        "best_through_depth5": candidate_dict(best),
        "best_through_depth3": candidate_dict(depth3_best),
        "gain_exact_depth5_vs_exact_depth3_pct": 100.0
        * (depth3_best.encoded_bytes - best.encoded_bytes)
        / depth3_best.encoded_bytes,
        "best_rate": best.encoded_bytes / original_bytes,
        "depth3_rate": depth3_best.encoded_bytes / original_bytes,
        "verified_best_by_depth": verified,
        "wall_seconds_current_invocation": time.perf_counter() - started,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "proxy_warning": (
            "Conceptual transforms plus zlib; entropy_huffman is a frequency-ranked "
            "dictionary transform, not a production Huffman bitstream."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": summary}, ensure_ascii=False), flush=True)


def self_check(codec, operator_module, operators: tuple[str, ...]) -> None:
    if len(operators) != 28 or len(set(operators)) != 28:
        raise AssertionError(f"expected 28 unique operators, got {len(operators)}")
    synthetic = np.asarray([-7, -7, -3, 0, 1, 1, 1, 18, 18, 99] * 8, dtype=np.int64)
    pipelines = [(operator,) for operator in operators]
    pipelines.extend(
        (operators[start], operators[(start + 7) % len(operators)], operators[(start + 13) % len(operators)])
        for start in range(len(operators))
    )
    for pipeline in pipelines:
        state = operator_module.initial_state(synthetic)
        for operator in pipeline:
            state = operator_module.apply_operator(state, operator)
        incremental = score_state(codec, state, pipeline)
        encoded = codec.encode_stream(synthetic, pipeline, operator_module)
        restored, decoded_pipeline = codec.decode_stream(encoded)
        if incremental != len(encoded) or decoded_pipeline != pipeline or not np.array_equal(
            restored, synthetic
        ):
            raise AssertionError(f"self-check failed for {pipeline}")
    expected = sum(len(operators) ** depth for depth in range(1, 6))
    if expected != 17_847_788 or len(operators) ** 5 != 17_210_368 or 29**5 != 20_511_149:
        raise AssertionError("search-space arithmetic changed")
    print(json.dumps({"self_check": "PASS", "round_trips": len(pipelines), "canonical": expected}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--codec-path",
        type=Path,
        default=Path(__file__).resolve().with_name("measure_selected_codec_cost.py"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--column-index", type=int, required=True)
    parser.add_argument("--mode", choices=("fixed-tail", "full"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--pair-indices", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.top_k < 1 or args.progress_every < 1 or args.pair_limit < 0:
        parser.error("--top-k and --progress-every must be positive; --pair-limit nonnegative")
    return args


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.codec_path = args.codec_path.resolve()
    args.output_dir = args.output_dir.resolve()
    codec = import_path(args.codec_path, "exact_depth5_codec")
    operator_module = codec.load_operator_module(args.repo_root)
    operators = tuple(codec.OPERATOR_IDS)
    self_check(codec, operator_module, operators)
    if args.self_check_only:
        return
    data_path, info_path, column, values = load_column(
        operator_module, args.repo_root, args.dataset, args.column_index
    )
    input_meta = {
        **column,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "column_info_path": str(info_path),
        "column_info_sha256": sha256_file(info_path),
    }
    if args.mode == "fixed-tail":
        run_fixed_tail(args, codec, operator_module, operators, input_meta, values)
    else:
        run_full(args, codec, operator_module, operators, input_meta, values)


if __name__ == "__main__":
    main()
