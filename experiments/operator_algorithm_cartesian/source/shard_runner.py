#!/usr/bin/env python3
"""Deterministic current200 column/operator queue with O2R1 reuse.

One task is exactly one frozen column and one resolved operator.  The operator
producer runs once and its immutable O2R1 block files are then consumed by all
85 algorithm routes.  Every algorithm gets a terminal status; independently
measured winners are never joined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

from carrier import ACTIVE85, carrier_kind_for_algorithm


RETIRED_OPERATOR_IDS = {"A106", "A107", "A108", "A109", "A110"}
CELL_STATUSES = {
    "ok", "not_applicable", "operator_fallback", "carrier_rejected",
    "algorithm_input_domain", "encode_failed", "decode_failed",
    "operator_inverse_failed", "bit_mismatch", "resource_limit", "timeout",
    "dependency_unavailable",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def safe(value: str) -> str:
    require(value and all(character.isalnum() or character in "._-+" for character in value),
            f"unsafe identifier: {value!r}")
    return value


def load_plan(path: Path, expected_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(digest(path) == expected_sha, "Clean200 plan SHA-256 mismatch")
    plan = read_object(path)
    jobs = plan.get("jobs", plan.get("columns"))
    require(isinstance(jobs, list) and len(jobs) == 3723,
            "Clean200 plan must contain exactly 3,723 columns")
    datasets = {row.get("dataset_id") for row in jobs if isinstance(row, dict)}
    require(len(datasets) == 200 and None not in datasets,
            "Clean200 plan must contain exactly 200 datasets")
    identities: set[tuple[str, str]] = set()
    for row in jobs:
        require(isinstance(row, dict), "column plan row must be an object")
        identity = (safe(str(row.get("dataset_id", ""))),
                    safe(str(row.get("column_id", ""))))
        require(identity not in identities, f"duplicate column identity: {identity}")
        identities.add(identity)
    return plan, jobs


def load_operators(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    require(digest(path) == expected_sha, "resolved operator catalog SHA-256 mismatch")
    value = read_object(path)
    rows = value.get("resolved_implementations", value.get("implementations"))
    require(isinstance(rows, list), "resolved operator rows missing")
    active = [row for row in rows if isinstance(row, dict) and
              row.get("implementation_id") not in RETIRED_OPERATOR_IDS]
    ids = [row.get("implementation_id") for row in active]
    require(len(active) == 97 and len(set(ids)) == 97 and
            all(isinstance(item, str) and item for item in ids),
            "active operator catalog must be the exact 97-ID set")
    return active


def load_routes(path: Path, expected_sha: str) -> dict[str, dict[str, Any]]:
    require(digest(path) == expected_sha, "active85 route manifest SHA-256 mismatch")
    value = read_object(path)
    require(value.get("schema") == "operator-algorithm-active85-carrier-routes-v1",
            "unexpected active85 route manifest schema")
    require(value.get("status") == "frozen" and value.get("algorithms") == 85,
            "active85 route manifest is not frozen/exact")
    rows = value.get("records")
    require(isinstance(rows, list), "route manifest requires records array")
    records_sha = hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()
    require(value.get("records_canonical_sha256") == records_sha,
            "active85 route record union SHA-256 mismatch")
    contract = value.get("contract")
    require(isinstance(contract, dict) and
            contract.get("complete_algorithm_frame_charged") is True and
            contract.get("carrier_metadata_and_missing_bitmap_charged") is True and
            contract.get("native_W1_R3_T1") is True and
            contract.get("cold_decode") is True and
            contract.get("independent_optima_combined") is False and
            contract.get("no_substitute_codec") is True,
            "active85 route byte/timing contract is incomplete")
    routes: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "route row must be object")
        name = row.get("algorithm", row.get("method"))
        require(isinstance(name, str) and name in ACTIVE85 and name not in routes,
                f"invalid or duplicate algorithm route: {name}")
        routes[name] = row
    require(set(routes) == ACTIVE85, "route manifest must bind exact active85")
    return routes


def manifest_for_job(plan: dict[str, Any], job: dict[str, Any]) -> tuple[Path, str]:
    manifests = job.get("manifests")
    raw = job.get("manifest_path")
    if raw is None and isinstance(manifests, list) and len(manifests) == 1:
        raw = manifests[0]
    require(isinstance(raw, str) and raw, "column job needs exactly one manifest")
    path = Path(raw)
    sha = job.get("manifest_sha256")
    if sha is None:
        sha_map = plan.get("manifest_sha256", {})
        if isinstance(sha_map, dict):
            sha = sha_map.get(raw, sha_map.get(str(path.resolve())))
    require(isinstance(sha, str) and len(sha) == 64,
            f"manifest SHA missing: {path}")
    return path, sha


def task_key(job: dict[str, Any], operator: dict[str, Any]) -> str:
    return safe(f"{job['dataset_id']}__{job['column_id']}__{operator['implementation_id']}")


def status_cell(algorithm: str, operator_id: str, status: str, reason: str,
                carrier_kind: str | None = None) -> dict[str, Any]:
    require(status in CELL_STATUSES, f"unknown terminal status: {status}")
    return {
        "schema": "operator-algorithm-joint-cell-result-v2",
        "algorithm": algorithm, "operator_id": operator_id,
        "status": status, "reason": reason,
        "carrier_kind": carrier_kind or carrier_kind_for_algorithm(algorithm),
        "bit_exact": False, "ranking_eligible": False,
        "source_numeric_bytes": None, "joint_complete_frame_bytes": None,
        "compression_ratio": None,
        "operator_full_encode_ns_samples": None,
        "operator_full_decode_ns_samples": None,
        "algorithm_encode_ns_samples": None,
        "algorithm_decode_ns_samples": None,
        "carrier_pack_ns_samples": None,
        "carrier_unpack_ns_samples": None,
        "joint_stage_sum_encode_ns_samples": None,
        "joint_stage_sum_decode_ns_samples": None,
        "independent_optima_combined": False,
    }


def classify_failure(stderr: str) -> str:
    text = stderr.lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "cannot allocate" in text or "bad_alloc" in text or "resource" in text:
        return "resource_limit"
    if "binary sha" in text or "no such file" in text or "dependency" in text:
        return "dependency_unavailable"
    if "inverse receipt" in text:
        return "operator_inverse_failed"
    if "changed o2r1" in text or "bit" in text and "mismatch" in text:
        return "bit_mismatch"
    if "decode" in text:
        return "decode_failed"
    if "carrier" in text:
        return "carrier_rejected"
    return "encode_failed"


def samples(measurement: dict[str, Any], key: str) -> list[int]:
    value = measurement.get(key)
    require(isinstance(value, list) and len(value) == 3 and
            all(isinstance(item, int) and item >= 0 for item in value),
            f"invalid {key}")
    return value


def vector_sum(left: list[int], right: list[int]) -> list[int]:
    require(len(left) == len(right) == 3, "W1/R3 vectors require three samples")
    return [left[index] + right[index] for index in range(3)]


def column_outer_bytes(block_frame_sizes: list[int]) -> int:
    # JOC1 + uint32 count + one uint64 length descriptor per JOA1 block.
    return 8 + 8 * len(block_frame_sizes) + sum(block_frame_sizes)


class Producer:
    def __init__(self, binary: Path):
        self.process = subprocess.Popen(
            [str(binary), "--jsonl"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        require(self.process.poll() is None, "O2R1 producer exited")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(value, separators=(",", ":"),
                                            allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        require(bool(line), "O2R1 producer returned no response")
        response = json.loads(line)
        require(isinstance(response, dict), "O2R1 response must be object")
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def route_binary(route: dict[str, Any]) -> tuple[Path, str, str]:
    binary = Path(str(route.get("binary", "")))
    expected = route.get("binary_sha256")
    profile = route.get("profile", route.get("algorithm_profile", ""))
    require(binary.is_file() and os.access(binary, os.X_OK),
            f"algorithm route binary unavailable: {binary}")
    require(isinstance(expected, str) and digest(binary) == expected,
            f"algorithm route binary SHA mismatch: {binary}")
    require(isinstance(profile, str) and profile, "algorithm profile missing")
    require(route.get("cli_contract", "baseline-measure-decode-v1") ==
            "baseline-measure-decode-v1", "unsupported algorithm CLI contract")
    return binary, expected, profile


def run_algorithm(algorithm: str, route: dict[str, Any], bundle_root: Path,
                  bundle: dict[str, Any], runner: Path, task_root: Path,
                  operator_catalog_sha: str, timeout: float) -> dict[str, Any]:
    operator_id = str(bundle["operator_id"])
    carrier = carrier_kind_for_algorithm(algorithm)
    try:
        binary, binary_sha, profile = route_binary(route)
    except Exception as error:
        return status_cell(algorithm, operator_id, "dependency_unavailable", str(error), carrier)
    algorithm_encode = [0, 0, 0]
    algorithm_decode = [0, 0, 0]
    carrier_pack = [0, 0, 0]
    carrier_unpack = [0, 0, 0]
    sizes: list[int] = []
    block_shas: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="joint-algorithm-", dir=task_root) as temporary:
            temporary_root = Path(temporary)
            for block in bundle["blocks"]:
                index = int(block["block_index"])
                block_output = temporary_root / f"block_{index:08d}"
                command = [
                    sys.executable, str(runner), "measure-one",
                    "--algorithm", algorithm, "--binary", str(binary),
                    "--expected-binary-sha256", binary_sha,
                    "--algorithm-profile", profile,
                    "--operator-id", operator_id,
                    "--operator-catalog-sha256", operator_catalog_sha,
                    "--operator-inverse-receipt",
                    str(bundle_root / block["operator_inverse_receipt_path"]),
                    "--o2r1", str(bundle_root / block["o2r1_path"]),
                    "--source-column-sha256", bundle["source_column_sha256"],
                    "--source-numeric-bytes", str(block["source_numeric_bytes"]),
                    "--dataset-id", str(bundle["dataset_id"]),
                    "--column-id", str(bundle["column_id"]),
                    "--output", str(block_output),
                ]
                ld_library_path = route.get("ld_library_path")
                require(ld_library_path is None or isinstance(ld_library_path, str),
                        f"invalid LD_LIBRARY_PATH binding for {algorithm}")
                if ld_library_path:
                    command.extend(["--ld-library-path", ld_library_path])
                completed = subprocess.run(command, text=True, capture_output=True,
                                           timeout=timeout, check=False)
                if completed.returncode != 0:
                    status = classify_failure(completed.stderr)
                    return status_cell(algorithm, operator_id, status,
                                       completed.stderr[-2000:] or "joint runner failed", carrier)
                result = read_object(block_output / "RESULT.json")
                if result.get("status") == "not_applicable":
                    return status_cell(algorithm, operator_id, "algorithm_input_domain",
                                       str(result.get("reason", "algorithm domain rejected carrier")),
                                       carrier)
                require(result.get("status") == "ok" and result.get("bit_exact") is True,
                        "block result is not bit-exact ok")
                measurement = result["algorithm_measurement"]
                algorithm_encode = vector_sum(
                    algorithm_encode, samples(measurement, "compression_ns_samples"))
                algorithm_decode = vector_sum(
                    algorithm_decode, samples(measurement, "decompression_ns_samples"))
                carrier_timing = result["carrier_timing"]
                carrier_pack = vector_sum(
                    carrier_pack, samples(carrier_timing, "pack_ns_samples"))
                carrier_unpack = vector_sum(
                    carrier_unpack, samples(carrier_timing, "unpack_ns_samples"))
                sizes.append(int(result["joint_complete_frame_bytes"]))
                block_shas.append(str(result["joint_frame_sha256"]))
        operator_encode = list(bundle["timing"]["full_encode_including_o2r1_pack"]["samples_ns"])
        operator_decode = list(bundle["timing"]["full_decode_including_o2r1_parse"]["samples_ns"])
        complete_bytes = column_outer_bytes(sizes)
        source_bytes = int(bundle["source_numeric_bytes"])
        return {
            "schema": "operator-algorithm-joint-cell-result-v2",
            "algorithm": algorithm, "operator_id": operator_id,
            "status": "ok", "reason": None, "carrier_kind": carrier,
            "carrier_contract": "charged reversible byte carrier v1",
            "bit_exact": True, "ranking_eligible": True,
            "source_numeric_bytes": source_bytes,
            "joint_complete_frame_bytes": complete_bytes,
            "compression_ratio": complete_bytes / source_bytes if source_bytes else None,
            "operator_full_encode_ns_samples": operator_encode,
            "operator_full_decode_ns_samples": operator_decode,
            "algorithm_encode_ns_samples": algorithm_encode,
            "algorithm_decode_ns_samples": algorithm_decode,
            "carrier_pack_ns_samples": carrier_pack,
            "carrier_unpack_ns_samples": carrier_unpack,
            "joint_stage_sum_encode_ns_samples": vector_sum(
                operator_encode, vector_sum(carrier_pack, algorithm_encode)),
            "joint_stage_sum_decode_ns_samples": vector_sum(
                algorithm_decode, vector_sum(carrier_unpack, operator_decode)),
            "timing_contract": "paired stage sum; not contiguous end-to-end wall",
            "block_count": len(sizes), "block_joa1_sha256": block_shas,
            "outer_frame_contract": "JOC1 + u32 block_count + repeated u64 length + JOA1",
            "outer_frame_persisted": False,
            "independent_optima_combined": False,
        }
    except subprocess.TimeoutExpired as error:
        return status_cell(algorithm, operator_id, "timeout", str(error), carrier)
    except Exception as error:
        status = classify_failure(f"{type(error).__name__}: {error}")
        return status_cell(algorithm, operator_id, status,
                           f"{type(error).__name__}: {error}", carrier)


def run_task(args: argparse.Namespace, producer: Producer, plan: dict[str, Any],
             job: dict[str, Any], operator: dict[str, Any],
             routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = task_key(job, operator)
    task_root = args.output / "tasks" / key
    terminal = task_root / "TERMINAL.json"
    if terminal.exists():
        existing = read_object(terminal)
        require(existing.get("task_key") == key, "existing terminal identity mismatch")
        return existing
    task_root.mkdir(parents=True, exist_ok=True)
    attempts_root = task_root / "attempts"
    attempts_root.mkdir(exist_ok=True)
    prior_attempts = sorted(path for path in attempts_root.iterdir()
                            if path.is_dir() and path.name.startswith("attempt_"))
    attempt_number = len(prior_attempts) + 1
    attempt_root = attempts_root / f"attempt_{attempt_number:04d}"
    attempt_root.mkdir(exist_ok=False)
    manifest, manifest_sha = manifest_for_job(plan, job)
    bundle_root = attempt_root / "operator_bundle"
    request = {
        "schema": "operator-algorithm-o2r1-column-request-v1",
        "request_id": key,
        "dataset_id": job["dataset_id"], "column_id": job["column_id"],
        "manifest_path": str(manifest),
        "expected_manifest_sha256": manifest_sha,
        "input_contract": args.input_contract,
        "implementation": operator, "output_dir": str(bundle_root),
        "provenance": {
            "clean200_plan_sha256": args.plan_sha256,
            "operator_catalog_sha256": args.operator_catalog_sha256,
            "active85_routes_sha256": args.routes_sha256,
            "bundle_producer_sha256": args.producer_sha256,
        },
    }
    atomic_json(attempt_root / "REQUEST.json", request)
    started = time.time_ns()
    response = producer.request(request)
    atomic_json(attempt_root / "PRODUCER_RESPONSE.json", response)
    operator_status = response.get("status")
    cells: list[dict[str, Any]] = []
    if operator_status == "not_applicable":
        cells = [status_cell(name, operator["implementation_id"], "not_applicable",
                             str(response.get("reason", "operator not applicable")))
                 for name in sorted(ACTIVE85)]
    elif operator_status == "operator_fallback":
        cells = [status_cell(name, operator["implementation_id"], "operator_fallback",
                             "operator used charged raw fallback; excluded from ranking")
                 for name in sorted(ACTIVE85)]
    elif operator_status == "passed":
        bundle = read_object(bundle_root / "BUNDLE.json")
        require(bundle.get("strict") is True and bundle.get("bit_exact") is True and
                bundle.get("ranking_eligible") is True,
                "passed operator bundle lacks strict inverse qualification")
        for name in sorted(ACTIVE85):
            cells.append(run_algorithm(
                name, routes[name], bundle_root, bundle, args.joint_runner,
                attempt_root, args.operator_catalog_sha256, args.cell_timeout_seconds))
    else:
        reason = json.dumps(response.get("error", "operator producer failed"),
                            sort_keys=True)
        cells = [status_cell(name, operator["implementation_id"],
                             "operator_inverse_failed", reason)
                 for name in sorted(ACTIVE85)]
    require(len(cells) == 85 and all(cell["status"] in CELL_STATUSES for cell in cells),
            "task did not produce 85 terminal cell statuses")
    result = {
        "schema": "operator-algorithm-column-operator-terminal-v1",
        "task_key": key, "status": "complete",
        "dataset_id": job["dataset_id"], "column_id": job["column_id"],
        "operator_id": operator["implementation_id"],
        "attempt": attempt_number,
        "operator_status": operator_status,
        "cell_status_counts": {status: sum(cell["status"] == status for cell in cells)
                               for status in sorted(CELL_STATUSES)},
        "cells": cells, "elapsed_wall_ns_not_scientific": time.time_ns() - started,
        "old_results_preserved": True,
    }
    atomic_json(terminal, result)
    return result


def jobs_for_shard(jobs: list[dict[str, Any]], operators: list[dict[str, Any]],
                   shard_index: int, shard_count: int,
                   datasets: set[str], column_ids: set[str],
                   operator_ids: set[str]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    ordinal = 0
    for job in jobs:
        for operator in operators:
            include = ordinal % shard_count == shard_index
            ordinal += 1
            if not include:
                continue
            if datasets and job["dataset_id"] not in datasets:
                continue
            if column_ids and job["column_id"] not in column_ids:
                continue
            if operator_ids and operator["implementation_id"] not in operator_ids:
                continue
            selected.append((job, operator))
    return selected


def execute(args: argparse.Namespace) -> int:
    args.plan = args.plan.resolve(); args.operator_catalog = args.operator_catalog.resolve()
    args.routes = args.routes.resolve(); args.producer = args.producer.resolve()
    args.joint_runner = args.joint_runner.resolve(); args.output = args.output.resolve()
    require(0 <= args.shard_index < args.shard_count, "invalid shard")
    require(hasattr(os, "sched_setaffinity"),
            "production shard requires Linux CPU-affinity support")
    os.sched_setaffinity(0, {args.cpu})
    require(digest(args.producer) == args.producer_sha256,
            "O2R1 producer binary SHA mismatch")
    plan, jobs = load_plan(args.plan, args.plan_sha256)
    operators = load_operators(args.operator_catalog, args.operator_catalog_sha256)
    routes = load_routes(args.routes, args.routes_sha256)
    args.input_contract = {
        "byte_order": "little", "values_are_valid_only": True,
        "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema": "operator-algorithm-current200-shard-v1",
        "plan": str(args.plan), "plan_sha256": args.plan_sha256,
        "columns": 3723, "datasets": 200,
        "operator_catalog": str(args.operator_catalog),
        "operator_catalog_sha256": args.operator_catalog_sha256, "operators": 97,
        "routes": str(args.routes), "routes_sha256": args.routes_sha256,
        "algorithms": 85, "joint_cells": 30_696_135,
        "producer": str(args.producer), "producer_sha256": args.producer_sha256,
        "joint_runner": str(args.joint_runner),
        "joint_runner_sha256": digest(args.joint_runner),
        "shard_runner": str(Path(__file__).resolve()),
        "shard_runner_sha256": digest(Path(__file__).resolve()),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "cpu": args.cpu, "threads": 1,
        "old_results_preserved": True,
    }
    frozen_path = args.output / f"FROZEN_SHARD_{args.shard_index:04d}.json"
    if frozen_path.exists():
        require(read_object(frozen_path) == frozen, "shard frozen identity changed")
    else:
        atomic_json(frozen_path, frozen)
    selected = jobs_for_shard(
        jobs, operators, args.shard_index, args.shard_count,
        set(args.dataset or []), set(args.column_id or []),
        set(args.operator_id or []))
    if args.max_tasks is not None:
        selected = selected[:args.max_tasks]
    producer = Producer(args.producer)
    completed = 0
    try:
        for job, operator in selected:
            run_task(args, producer, plan, job, operator, routes)
            completed += 1
            atomic_json(args.output / f"PROGRESS_{args.shard_index:04d}.json", {
                "schema": "operator-algorithm-current200-progress-v1",
                "shard_index": args.shard_index, "selected_tasks": len(selected),
                "completed_tasks": completed, "remaining_tasks": len(selected) - completed,
                "completed_joint_cells": completed * 85,
            })
    finally:
        producer.close()
    return 0


def build_pilot(args: argparse.Namespace) -> int:
    _, jobs = load_plan(args.plan.resolve(), args.plan_sha256)
    ordered = sorted(jobs, key=lambda row: (
        int(row.get("scheduling_original_numeric_bytes", row.get("source_numeric_bytes", 0))),
        str(row["dataset_id"]), str(row["column_id"])))
    require(len(ordered) >= 3, "pilot requires at least three columns")
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    receipt = {
        "schema": "operator-algorithm-current200-pilot-selection-v1",
        "selection": "deterministic smallest, median and largest physical column",
        "columns": selected, "operator_scope": "all active97",
        "algorithm_scope": "all active85 including all three carrier classes",
        "expected_joint_cells": len(selected) * 97 * 85,
        "purpose": "route/carrier correctness and measured throughput; not final science",
    }
    atomic_json(args.output.resolve(), receipt)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-shard")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--operator-catalog", type=Path, required=True)
    run.add_argument("--operator-catalog-sha256", required=True)
    run.add_argument("--routes", type=Path, required=True)
    run.add_argument("--routes-sha256", required=True)
    run.add_argument("--producer", type=Path, required=True)
    run.add_argument("--producer-sha256", required=True)
    run.add_argument("--joint-runner", type=Path, default=Path(__file__).with_name("runner.py"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, required=True)
    run.add_argument("--cpu", type=int, required=True,
                     help="one reviewed physical CPU; inherited by all native children")
    run.add_argument("--dataset", action="append")
    run.add_argument("--column-id", action="append")
    run.add_argument("--operator-id", action="append")
    run.add_argument("--max-tasks", type=int)
    run.add_argument("--cell-timeout-seconds", type=float, default=3600)
    run.set_defaults(function=execute)
    pilot = sub.add_parser("build-pilot")
    pilot.add_argument("--plan", type=Path, required=True)
    pilot.add_argument("--plan-sha256", required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.set_defaults(function=build_pilot)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
