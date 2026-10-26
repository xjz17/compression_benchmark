#!/usr/bin/env python3
"""Search the family-unique PV2F depth-3 physical pipeline space.

The runner supports two deliberately separate modes:

* ``beam`` evaluates widths independently and records every real PV2F probe;
* ``exact`` enumerates every family-unique transform prefix through depth 3.

Every successful candidate is decoded and compared bitwise with the normalized
int64 input before its byte count is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import socket
import statistics
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from enumerate_physical_operator_variants_v2 import load_column
from physical_operator_runtime_v2 import (
    ACTION_BY_NAME,
    COMPRESSORS,
    PACKERS,
    TRANSFORMS,
    TransformRecord,
    TransformState,
    apply_transform,
    decode_frame,
    encode_from_packed,
    pack_values,
    raw_i64,
    registry,
    self_check,
    sha256_bytes,
)


TRANSFORM_NAMES = tuple(spec.name for spec in TRANSFORMS)
PACKER_NAMES = tuple(spec.name for spec in PACKERS)
COMPRESSOR_NAMES = tuple(spec.name for spec in COMPRESSORS)
TRANSFORM_SET = set(TRANSFORM_NAMES)
PACKER_SET = set(PACKER_NAMES)
COMPRESSOR_SET = set(COMPRESSOR_NAMES)
TRANSFORM_FAMILY = {spec.name: spec.family for spec in TRANSFORMS}
CSV_FIELDS = (
    "dataset",
    "column_index",
    "column_name",
    "logical_type",
    "sample_values",
    "original_bytes",
    "input_sha256",
    "transform_count",
    "transform_families",
    "transform_pipeline",
    "packer",
    "compressor",
    "pipeline",
    "final_bytes",
    "compression_rate",
    "encoded_sha256",
    "lossless",
    "status",
    "elapsed_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_python_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no Python sources under {root}")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_pipeline(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(" -> ") if part.strip())


def split_prefix(prefix: Sequence[str], max_transforms: int) -> tuple[tuple[str, ...], str | None, str | None]:
    transforms: list[str] = []
    packer: str | None = None
    compressor: str | None = None
    for name in prefix:
        if packer is None and name in TRANSFORM_SET:
            transforms.append(name)
        elif packer is None and name in PACKER_SET:
            packer = name
        elif packer is not None and compressor is None and name in COMPRESSOR_SET:
            compressor = name
        else:
            raise ValueError(f"illegal typed prefix: {prefix}")
    if len(transforms) > max_transforms:
        raise ValueError(f"too many transforms: {prefix}")
    families = [TRANSFORM_FAMILY[name] for name in transforms]
    if len(families) != len(set(families)):
        raise ValueError(f"repeated semantic transform family: {prefix}")
    return tuple(transforms), packer, compressor


def legal_actions(prefix: tuple[str, ...], max_transforms: int) -> tuple[str, ...]:
    transforms, packer, compressor = split_prefix(prefix, max_transforms)
    if compressor is not None:
        return ()
    if packer is not None:
        return COMPRESSOR_NAMES
    used_families = {TRANSFORM_FAMILY[name] for name in transforms}
    available = tuple(name for name in TRANSFORM_NAMES if TRANSFORM_FAMILY[name] not in used_families)
    if len(transforms) >= max_transforms:
        return PACKER_NAMES
    return available + PACKER_NAMES


def is_terminal(prefix: tuple[str, ...], max_transforms: int) -> bool:
    return split_prefix(prefix, max_transforms)[2] is not None


def default_completion(prefix: tuple[str, ...], max_transforms: int) -> tuple[str, ...]:
    transforms, packer, compressor = split_prefix(prefix, max_transforms)
    if compressor is not None:
        return prefix
    if packer is None:
        return transforms + ("raw.i64", "none")
    return transforms + (packer, "none")


def family_unique_transform_prefixes(max_transforms: int) -> list[tuple[str, ...]]:
    prefixes: list[tuple[str, ...]] = [()]
    for depth in range(1, max_transforms + 1):
        for prefix in itertools.permutations(TRANSFORM_NAMES, depth):
            families = tuple(TRANSFORM_FAMILY[name] for name in prefix)
            if len(families) == len(set(families)):
                prefixes.append(prefix)
    return prefixes


def complete_output(
    output_dir: Path,
    args: argparse.Namespace,
    values: np.ndarray,
    data_path: Path,
    info_path: Path,
) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            return False
        if manifest.get("mode") != args.mode or manifest.get("dataset") != args.dataset:
            return False
        if int(manifest.get("column_index", -1)) != args.column_index:
            return False
        if int(manifest.get("max_transforms", -1)) != args.max_transforms:
            return False
        if int(manifest.get("max_values", -1)) != args.max_values:
            return False
        expected_widths = list(args.width_list) if args.mode == "beam" else []
        if manifest.get("widths") != expected_widths:
            return False
        if manifest.get("input_sha256") != sha256_bytes(raw_i64(values)):
            return False
        sources = manifest["source_sha256"]
        if sources["runner"] != sha256_file(Path(__file__).resolve()):
            return False
        if sources["runtime"] != sha256_file(Path(__file__).with_name("physical_operator_runtime_v2.py")):
            return False
        if sources["loader"] != sha256_file(Path(__file__).with_name("enumerate_physical_operator_variants_v2.py")):
            return False
        if sources["data_csv"] != sha256_file(data_path):
            return False
        if sources["column_info_csv"] != sha256_file(info_path):
            return False
        if sources["backend_myapp_python_tree"] != sha256_python_tree(args.repo_root.resolve() / "backend" / "myapp"):
            return False
        for name, expected_hash in manifest["output_sha256"].items():
            if sha256_file(output_dir / name) != expected_hash:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


class Evaluator:
    """Real-codec evaluator with bounded packer-payload reuse."""

    def __init__(
        self,
        values: np.ndarray,
        metadata: dict[str, str],
        dataset: str,
        column_index: int,
        max_transforms: int,
        packed_cache_limit: int = 128,
    ) -> None:
        self.values = np.asarray(values, dtype=np.int64)
        self.metadata = metadata
        self.dataset = dataset
        self.column_index = column_index
        self.max_transforms = max_transforms
        self.input_sha256 = sha256_bytes(raw_i64(self.values))
        self.original_bytes = int(self.values.size * 8)
        self.state_cache: dict[tuple[str, ...], TransformState] = {
            (): TransformState(self.values.copy(), ())
        }
        self.packed_cache: OrderedDict[tuple[tuple[str, ...], str], bytes] = OrderedDict()
        self.packed_cache_limit = packed_cache_limit
        self.measurement_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.failures = 0

    def state(self, transforms: tuple[str, ...]) -> TransformState:
        split_prefix(transforms, self.max_transforms)
        cached = self.state_cache.get(transforms)
        if cached is not None:
            return cached
        parent = self.state(transforms[:-1])
        spec = ACTION_BY_NAME[transforms[-1]]
        transformed, metadata = apply_transform(parent.values, spec)
        state = TransformState(
            np.asarray(transformed, dtype=np.int64),
            parent.records + (TransformRecord(spec, metadata),),
        )
        self.state_cache[transforms] = state
        return state

    def packed(self, transforms: tuple[str, ...], packer: str) -> bytes:
        key = transforms, packer
        cached = self.packed_cache.pop(key, None)
        if cached is not None:
            self.packed_cache[key] = cached
            return cached
        payload = pack_values(self.state(transforms).values, ACTION_BY_NAME[packer])
        self.packed_cache[key] = payload
        while len(self.packed_cache) > self.packed_cache_limit:
            self.packed_cache.popitem(last=False)
        return payload

    def measure(self, pipeline: tuple[str, ...]) -> dict[str, Any]:
        cached = self.measurement_cache.get(pipeline)
        if cached is not None:
            return cached
        transforms, packer, compressor = split_prefix(pipeline, self.max_transforms)
        if packer is None or compressor is None:
            raise ValueError(f"not a complete pipeline: {pipeline}")
        started = time.perf_counter_ns()
        try:
            state = self.state(transforms)
            packed = self.packed(transforms, packer)
            frame = encode_from_packed(self.values, state, packer, compressor, packed)
            restored, decoded_pipeline = decode_frame(frame.data)
            if frame.pipeline != pipeline or decoded_pipeline != pipeline:
                raise AssertionError("pipeline identity mismatch")
            if not np.array_equal(restored, self.values):
                raise AssertionError("decoder-only bitwise round-trip mismatch")
            row: dict[str, Any] = {
                "dataset": self.dataset,
                "column_index": self.column_index,
                "column_name": self.metadata["column_name"],
                "logical_type": self.metadata["logical_type"],
                "sample_values": int(self.values.size),
                "original_bytes": self.original_bytes,
                "input_sha256": self.input_sha256,
                "transform_count": len(transforms),
                "transform_families": " -> ".join(TRANSFORM_FAMILY[name] for name in transforms),
                "transform_pipeline": " -> ".join(transforms),
                "packer": packer,
                "compressor": compressor,
                "pipeline": " -> ".join(pipeline),
                "final_bytes": int(frame.components["final_bytes"]),
                "compression_rate": int(frame.components["final_bytes"]) / self.original_bytes,
                "encoded_sha256": sha256_bytes(frame.data),
                "lossless": 1,
                "status": "ok",
                "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            }
        except Exception as exc:
            self.failures += 1
            row = {
                "dataset": self.dataset,
                "column_index": self.column_index,
                "column_name": self.metadata["column_name"],
                "logical_type": self.metadata["logical_type"],
                "sample_values": int(self.values.size),
                "original_bytes": self.original_bytes,
                "input_sha256": self.input_sha256,
                "transform_count": len(transforms),
                "transform_families": " -> ".join(TRANSFORM_FAMILY[name] for name in transforms),
                "transform_pipeline": " -> ".join(transforms),
                "packer": packer,
                "compressor": compressor,
                "pipeline": " -> ".join(pipeline),
                "final_bytes": "",
                "compression_rate": "",
                "encoded_sha256": "",
                "lossless": 0,
                "status": f"{type(exc).__name__}:{str(exc)[:240]}",
                "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            }
        self.measurement_cache[pipeline] = row
        return row

    def size(self, pipeline: tuple[str, ...]) -> int:
        row = self.measure(pipeline)
        if row["status"] != "ok" or int(row["lossless"]) != 1:
            return sys.maxsize
        return int(row["final_bytes"])


def write_measurements(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def run_beam_width(
    values: np.ndarray,
    metadata: dict[str, str],
    args: argparse.Namespace,
    width: int,
    measurement_path: Path,
    evaluator: Evaluator,
) -> dict[str, Any]:
    # Keep transform-phase and packer-phase prefixes in separate beams.  A
    # single mixed frontier collapses to cheap packer prefixes and can end the
    # search before depth 2/3 is ever visited.
    transform_frontier: list[tuple[str, ...]] = [()]
    packer_frontier: list[tuple[str, ...]] = []
    terminals: set[tuple[str, ...]] = set()
    expected_measurements: set[tuple[str, ...]] = set()
    levels: list[dict[str, int]] = []
    measurements_before = len(evaluator.measurement_cache)
    started = time.perf_counter()
    for step in range(1, args.max_transforms + 3):
        expanded_transforms: set[tuple[str, ...]] = set()
        expanded_packers: set[tuple[str, ...]] = set()
        for prefix in transform_frontier + packer_frontier:
            for action in legal_actions(prefix, args.max_transforms):
                candidate = prefix + (action,)
                if is_terminal(candidate, args.max_transforms):
                    terminals.add(candidate)
                    expected_measurements.add(candidate)
                    evaluator.measure(candidate)
                else:
                    _transforms, packer, _compressor = split_prefix(candidate, args.max_transforms)
                    if packer is None:
                        expanded_transforms.add(candidate)
                    else:
                        expanded_packers.add(candidate)
                    probe = default_completion(candidate, args.max_transforms)
                    expected_measurements.add(probe)
                    evaluator.measure(probe)
        if not expanded_transforms and not expanded_packers:
            levels.append(
                {
                    "step": step,
                    "expanded_transform": 0,
                    "expanded_packer": 0,
                    "retained_transform": 0,
                    "retained_packer": 0,
                    "terminal_seen": len(terminals),
                }
            )
            break
        transform_frontier = sorted(
            expanded_transforms,
            key=lambda prefix: (evaluator.size(default_completion(prefix, args.max_transforms)), prefix),
        )[:width]
        packer_frontier = sorted(
            expanded_packers,
            key=lambda prefix: (evaluator.size(default_completion(prefix, args.max_transforms)), prefix),
        )[:width]
        levels.append(
            {
                "step": step,
                "expanded_transform": len(expanded_transforms),
                "expanded_packer": len(expanded_packers),
                "retained_transform": len(transform_frontier),
                "retained_packer": len(packer_frontier),
                "terminal_seen": len(terminals),
            }
        )
        print(
            json.dumps(
                {
                    "mode": "beam",
                    "dataset": args.dataset,
                    "column_index": args.column_index,
                    "width": width,
                    **levels[-1],
                    "logical_width_evaluations": len(expected_measurements),
                    "unique_shared_evaluations": len(evaluator.measurement_cache),
                    "elapsed_sec": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not expected_measurements.issubset(evaluator.measurement_cache):
        raise AssertionError(
            f"candidate accounting mismatch: measured={len(evaluator.measurement_cache)} "
            f"expected={len(expected_measurements)}"
        )
    valid_terminals = [pipeline for pipeline in terminals if evaluator.size(pipeline) < sys.maxsize]
    if not valid_terminals or evaluator.failures:
        raise AssertionError(
            f"Beam width {width}: terminals={len(valid_terminals)} failures={evaluator.failures}"
        )
    best = min(valid_terminals, key=lambda pipeline: (evaluator.size(pipeline), pipeline))
    rows = sorted(
        (evaluator.measurement_cache[pipeline] for pipeline in expected_measurements),
        key=lambda row: str(row["pipeline"]),
    )
    written = write_measurements(measurement_path, rows)
    elapsed = time.perf_counter() - started
    result = {
        "width": width,
        "pipeline": " -> ".join(best),
        "transform_count": len(split_prefix(best, args.max_transforms)[0]),
        "final_bytes": evaluator.size(best),
        "compression_rate": evaluator.size(best) / evaluator.original_bytes,
        "candidate_evaluations": len(expected_measurements),
        "new_codec_evaluations_in_shared_run": len(evaluator.measurement_cache) - measurements_before,
        "all_candidates_accounted": expected_measurements.issubset(evaluator.measurement_cache),
        "written_measurements": written,
        "terminal_candidates_seen": len(terminals),
        "lossless_candidates": sum(int(row["lossless"]) for row in rows),
        "failed_candidates": evaluator.failures,
        "search_wall_sec": elapsed,
        "candidate_elapsed_ms_sum": sum(float(row["elapsed_ms"]) for row in rows),
        "levels": levels,
        "measurement_file": measurement_path.name,
    }
    print(json.dumps({"beam_finished": result}, ensure_ascii=False), flush=True)
    return result


def run_exact(
    values: np.ndarray,
    metadata: dict[str, str],
    args: argparse.Namespace,
    measurement_path: Path,
) -> dict[str, Any]:
    evaluator = Evaluator(values, metadata, args.dataset, args.column_index, args.max_transforms)
    prefixes = family_unique_transform_prefixes(args.max_transforms)
    expected = len(prefixes) * len(PACKER_NAMES) * len(COMPRESSOR_NAMES)
    started = time.perf_counter()
    written = 0
    best_row: dict[str, Any] | None = None
    elapsed_samples: list[float] = []
    with measurement_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for prefix_index, transforms in enumerate(prefixes, start=1):
            for packer in PACKER_NAMES:
                for compressor in COMPRESSOR_NAMES:
                    pipeline = transforms + (packer, compressor)
                    row = evaluator.measure(pipeline)
                    writer.writerow(row)
                    written += 1
                    elapsed_samples.append(float(row["elapsed_ms"]))
                    if row["status"] == "ok" and (
                        best_row is None
                        or (int(row["final_bytes"]), str(row["pipeline"]))
                        < (int(best_row["final_bytes"]), str(best_row["pipeline"]))
                    ):
                        best_row = row.copy()
                    evaluator.measurement_cache.pop(pipeline, None)
            if prefix_index % args.progress_every == 0 or prefix_index == len(prefixes):
                print(
                    json.dumps(
                        {
                            "mode": "exact",
                            "dataset": args.dataset,
                            "column_index": args.column_index,
                            "prefix": prefix_index,
                            "prefixes": len(prefixes),
                            "completed": written,
                            "expected": expected,
                            "failures": evaluator.failures,
                            "elapsed_sec": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if written != expected or evaluator.failures or best_row is None:
        raise AssertionError((written, expected, evaluator.failures, best_row))
    return {
        "transform_prefixes": len(prefixes),
        "expected_candidates": expected,
        "written_measurements": written,
        "lossless_candidates": written,
        "failed_candidates": evaluator.failures,
        "pipeline": best_row["pipeline"],
        "transform_count": int(best_row["transform_count"]),
        "final_bytes": int(best_row["final_bytes"]),
        "compression_rate": float(best_row["compression_rate"]),
        "search_wall_sec": time.perf_counter() - started,
        "median_candidate_elapsed_ms": statistics.median(elapsed_samples),
        "measurement_file": measurement_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=("Air-Quality", "Metro-Traffic"), required=True)
    parser.add_argument("--column-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("beam", "exact"), default="beam")
    parser.add_argument("--widths", default="29,64,256,1024")
    parser.add_argument("--max-transforms", type=int, default=3, choices=(0, 1, 2, 3))
    parser.add_argument("--max-values", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-runtime-self-check", action="store_true")
    parser.add_argument(
        "--quarantine-incomplete",
        action="store_true",
        help="recoverably rename a nonempty non-reusable output before rerunning",
    )
    args = parser.parse_args()
    try:
        args.width_list = tuple(dict.fromkeys(int(value) for value in args.widths.split(",")))
    except ValueError as exc:
        parser.error(f"invalid widths: {exc}")
    if args.column_index < 0 or args.max_values < 0 or args.progress_every <= 0:
        parser.error("column-index/max-values must be nonnegative and progress-every positive")
    if not args.width_list or any(width <= 0 for width in args.width_list):
        parser.error("widths must be positive")
    return args


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    output_dir = args.output_dir.resolve()
    values, metadata, data_path, info_path = load_column(
        repo_root, args.dataset, args.column_index, args.max_values
    )
    frozen_input_hashes = {
        "data_csv": sha256_file(data_path),
        "column_info_csv": sha256_file(info_path),
        "backend_myapp_python_tree": sha256_python_tree(repo_root / "backend" / "myapp"),
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        if complete_output(output_dir, args, values, data_path, info_path):
            print(json.dumps({"reused_complete_output": str(output_dir)}, sort_keys=True), flush=True)
            return
        if not args.quarantine_incomplete:
            raise FileExistsError(f"refusing non-empty incomplete or source-mismatched output: {output_dir}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = output_dir.with_name(f"{output_dir.name}.incomplete.{stamp}.{os.getpid()}")
        os.replace(output_dir, quarantine)
        print(json.dumps({"quarantined_output": str(quarantine)}, sort_keys=True), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    runtime_check = {"status": "SKIPPED"} if args.skip_runtime_self_check else self_check(repo_root)
    output_files: list[Path] = []
    if args.mode == "beam":
        results: list[dict[str, Any]] = []
        shared_evaluator = Evaluator(
            values, metadata, args.dataset, args.column_index, args.max_transforms
        )
        for width in args.width_list:
            measurement_path = output_dir / f"candidate_measurements_width{width}.csv"
            results.append(
                run_beam_width(
                    values, metadata, args, width, measurement_path, shared_evaluator
                )
            )
            output_files.append(measurement_path)
        results_path = output_dir / "beam_results.json"
        write_json(results_path, results)
        output_files.append(results_path)
        mode_result: Any = results
    else:
        measurement_path = output_dir / "candidate_measurements.csv"
        mode_result = run_exact(values, metadata, args, measurement_path)
        result_path = output_dir / "exact_result.json"
        write_json(result_path, mode_result)
        output_files.extend((measurement_path, result_path))
    catalog_path = output_dir / "action_catalog.json"
    write_json(
        catalog_path,
        {
            "runtime_registry": registry(),
            "experiment_grammar": {
                "max_transforms": args.max_transforms,
                "family_unique_transforms": True,
                "required_tail": ["packer", "compressor"],
            },
        },
    )
    output_files.append(catalog_path)
    source_sha256 = {
        "runner": sha256_file(Path(__file__).resolve()),
        "runtime": sha256_file(Path(__file__).with_name("physical_operator_runtime_v2.py")),
        "loader": sha256_file(Path(__file__).with_name("enumerate_physical_operator_variants_v2.py")),
        "data_csv": sha256_file(data_path),
        "column_info_csv": sha256_file(info_path),
        "backend_myapp_python_tree": sha256_python_tree(repo_root / "backend" / "myapp"),
    }
    for name, frozen_hash in frozen_input_hashes.items():
        if source_sha256[name] != frozen_hash:
            raise RuntimeError(f"source changed during run: {name}")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "elapsed_sec": time.perf_counter() - started,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "mode": args.mode,
        "dataset": args.dataset,
        "column_index": args.column_index,
        "column_name": metadata["column_name"],
        "logical_type": metadata["logical_type"],
        "sample_values": int(values.size),
        "original_bytes": int(values.size * 8),
        "input_sha256": sha256_bytes(raw_i64(values)),
        "max_transforms": args.max_transforms,
        "max_values": args.max_values,
        "widths": list(args.width_list) if args.mode == "beam" else [],
        "search_algorithm": "typed_phase_stratified_beam" if args.mode == "beam" else "exact_enumeration",
        "beam_width_execution": "ascending widths share immutable real-codec measurements; per-width wall time is marginal and not a latency comparison",
        "grammar": "0..3 transforms with unique semantic families, then exactly one packer and one compressor",
        "byte_objective": "final PV2F bytes / normalized int64 bytes including transform metadata, action table, frames, and outer CRC32",
        "runtime_self_check": runtime_check,
        "result": mode_result,
        "source_sha256": source_sha256,
        "output_sha256": {path.name: sha256_file(path) for path in output_files},
        "claim_boundary": "standalone normalized-int64 research wire format; TypedBeam minimum is best-known except at exact anchors",
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / f"manifest.json.tmp.{os.getpid()}"
    write_json(temporary_manifest, manifest)
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({"finished": manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
