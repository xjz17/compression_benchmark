#!/usr/bin/env python3
"""Bounded, frozen validation for the explicitly NONFORMAL 11-source pilot.

This is not the independent test-set experiment. All graph methods use the same
TGF1 encoder and timing boundaries. The optional zlib6 comparison has a separate
audited complete wrapper, and is identified separately from native core timing.
Timeouts/failures are missing measurements, never synthetic rewards or zeroes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import calibrate_weighted_grpo_references_v1 as calibration
from weight_conditioned_grpo_v1 import FIXED_WEIGHTS, FrozenPolicy, load_checkpoint, objective_value, propose_frozen
import weight_conditioned_graph_task_v1 as graph

SCHEMA = "weighted-grpo-11source-pilot-frozen-validation-v1"
METHODS = ("initial", "trained", "training_seed", "raw")
DTYPES = {"int32": (1, 4), "int64": (2, 8), "float32": (3, 4), "float64": (4, 8)}
ZLIB_HEADER = struct.Struct("<4sBQQQ")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    calibration.write_atomic(path, value)


def parameter_sha256(policy: Any) -> str:
    return _sha(b"".join(np.ascontiguousarray(p, dtype="<f8").tobytes() for p in policy.params))


def validate_panel(panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = panel.get("blocks")
    if not isinstance(records, list) or not records:
        raise ValueError("validation panel must contain fixed block records")
    seen: set[tuple[str, str]] = set()
    for record in records:
        block = record["block"]
        key = (str(record["dataset_id"]), str(block["block_id"]))
        if key in seen:
            raise ValueError("duplicate validation panel block")
        seen.add(key)
        if block["split"] != "validation":
            raise ValueError("validation cannot contain training or test blocks")
        if block["dtype"] not in DTYPES or not 0 < int(block["original_numeric_bytes"]) <= 1048576:
            raise ValueError("validation requires supported positive complete blocks <=1 MiB")
        if any(not math.isfinite(float(record[k])) or float(record[k]) <= 0 for k in ("E0", "D0")):
            raise ValueError("validation references must be positive TRAIN-frozen values")
    return records


def validate_panel_scope(records: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any],
                         library: Mapping[str, Any]) -> None:
    expected = {row["dataset_id"] for row in receipt["sources"]}
    actual = {row["dataset_id"] for row in records}
    if len(expected) != 11 or actual != expected:
        raise ValueError("validation panel must cover exactly the same 11 admitted pilot sources")
    if library.get("source_scope") != "training_only":
        raise ValueError("validation seed library must be TRAIN-only")
    if any(row.get("source_split") != "train" for row in library["measurements"]):
        raise ValueError("nontraining measurement cannot enter validation seed library")


def _zlib_cli(cli: Path, command: str, payload: bytes) -> bytes:
    result = subprocess.run([str(cli), command], input=payload, capture_output=True,
                            check=False, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def _validate_zlib_roots(dtype: str, rows: int, values: bytes, bitmap: bytes) -> None:
    width = DTYPES[dtype][1]
    if len(values) % width or len(bitmap) != (rows + 7) // 8:
        raise ValueError("invalid numeric/bitmap length")
    if rows % 8 and bitmap[-1] >> (rows % 8):
        raise ValueError("missing mask padding bits must be zero")
    if len(values) // width + sum(byte.bit_count() for byte in bitmap) != rows:
        raise ValueError("missing mask does not match valid-value count")


def encode_zlib6_frame(cli: Path, dtype: str, row_count: int, values: bytes, bitmap: bytes) -> bytes:
    """ZP11 carries all root type/partition information missing from ZR61."""
    if dtype not in DTYPES or isinstance(row_count, bool) or row_count < 1:
        raise ValueError("invalid typed zlib6 root")
    _validate_zlib_roots(dtype, row_count, values, bitmap)
    header = ZLIB_HEADER.pack(b"ZP11", DTYPES[dtype][0], row_count, len(values), len(bitmap))
    body = header + _zlib_cli(cli, "encode", values + bitmap)
    return body + hashlib.sha256(body).digest()


def decode_zlib6_frame(cli: Path, frame: bytes) -> tuple[str, int, bytes, bytes]:
    if len(frame) <= ZLIB_HEADER.size + 32:
        raise ValueError("truncated ZP11 frame")
    body, digest = frame[:-32], frame[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("ZP11 integrity failure")
    magic, dtype_id, rows, numeric_bytes, bitmap_bytes = ZLIB_HEADER.unpack_from(body)
    names = {value[0]: (name, value[1]) for name, value in DTYPES.items()}
    if (magic != b"ZP11" or dtype_id not in names or rows < 1
            or bitmap_bytes != (rows + 7) // 8):
        raise ValueError("invalid ZP11 metadata")
    dtype, width = names[dtype_id]
    if numeric_bytes % width or numeric_bytes > rows * width:
        raise ValueError("invalid ZP11 valid-value length")
    restored = _zlib_cli(cli, "decode", body[ZLIB_HEADER.size:])
    if len(restored) != numeric_bytes + bitmap_bytes:
        raise ValueError("ZP11 decoded length mismatch")
    values, bitmap = restored[:numeric_bytes], restored[numeric_bytes:]
    _validate_zlib_roots(dtype, rows, values, bitmap)
    return dtype, rows, values, bitmap


def _row_count(record: Mapping[str, Any], bitmap: bytes) -> int:
    block = record["block"]
    # Prepared manifests use row_count, independently of valid-value count.
    for key in ("row_count", "rows", "count"):
        if key in block:
            value = int(block[key])
            if (value + 7) // 8 != len(bitmap):
                raise ValueError("prepared row count differs from bitmap length")
            return value
    if "row_start" in block and "row_end" in block:
        value = int(block["row_end"]) - int(block["row_start"])
        if (value + 7) // 8 == len(bitmap):
            return value
    raise ValueError("prepared validation block has no verified row count")


def _persist_frame(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise RuntimeError("saved complete frame changed")
    return {"path": str(path), "bytes": len(payload), "sha256": _sha(payload)}


def _zlib_measure(record: dict[str, Any], cli: Path, output: Path, runner: Any) -> dict[str, Any]:
    values, bitmap = runner.read_block(record)
    dtype, rows = record["block"]["dtype"], _row_count(record, bitmap)
    encode_times, decode_times = [], []
    expected = (dtype, rows, values, bitmap)
    frame = b""
    for repeat in range(6):
        started = time.perf_counter_ns()
        frame = encode_zlib6_frame(cli, dtype, rows, values, bitmap)
        encode_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        restored = decode_zlib6_frame(cli, frame)
        decode_ns = time.perf_counter_ns() - started
        if restored != expected:
            raise RuntimeError("zlib6 independent native decode changed original roots")
        if repeat:
            encode_times.append(encode_ns)
            decode_times.append(decode_ns)
    saved = _persist_frame(output.with_suffix(".zp11"), frame)
    # A separate interpreter parses only the persisted frame and the pinned CLI;
    # it has no encoder-side root/length/cache information.
    cold = subprocess.run([
        os.sys.executable, str(Path(__file__).resolve()), "--cold-zlib-frame", saved["path"],
        "--zlib6-cli", str(cli),
    ], capture_output=True, check=False, timeout=30)
    if cold.returncode:
        raise RuntimeError("cold ZP11 decoder failed: " + cold.stderr.decode(errors="replace"))
    cold_record = json.loads(cold.stdout)
    if cold_record != {"dtype": dtype, "row_count": rows,
                       "values_sha256": _sha(values), "bitmap_sha256": _sha(bitmap)}:
        raise RuntimeError("cold ZP11 decoder root hash/type mismatch")
    raw = len(values)
    core = calibration._run_native(cli, values + bitmap, warmups=1, repeats=5)
    return {
        "status": "complete", "method": "zlib6_separate_wrapper", "original_numeric_bytes": raw,
        "complete_compressed_bytes": len(frame), "rho": len(frame) / raw,
        "compression_time_ns": statistics.median(encode_times),
        "decompression_time_ns": statistics.median(decode_times),
        "E_ns_per_byte": statistics.median(encode_times) / raw,
        "D_ns_per_byte": statistics.median(decode_times) / raw,
        "compression_repeat_time_ns": encode_times, "decompression_repeat_time_ns": decode_times,
        "J": None, "J_unavailable_reason": "separate_wrapper_not_graph_candidate",
        "native_core_receipt": core, "frame": saved, "bit_exact": True,
        "independent_persisted_frame_cold_decode": True,
        "timing_boundary": "Python ZP11 framing/hash + native child process per encode/decode; source reads excluded",
        "native_core_times_are_not_graph_times": True,
        "same_graph_format": False, "format": "ZP11(dtype,row_count,numeric_bytes,bitmap_bytes,ZR61,sha256)",
        "selection_walltime_ns": None, "selection_unavailable_reason": "fixed_method_no_selection",
    }


def _graph_measure(
    record: dict[str, Any], weight: Sequence[float], method: str, policy: Any,
    library: Mapping[str, Any], output: Path, runner: Any, candidate_budget: int,
) -> dict[str, Any]:
    before = parameter_sha256(policy) if policy is not None else None
    select_started = time.perf_counter_ns()
    recipe = () if method == "raw" else runner.seed_recipe(record, weight, library)
    task = runner.make_task(record, recipe=recipe, weight=weight, max_edits=2, cold=True)
    task.candidate_budget = graph.CandidateBudget(candidate_budget if policy is not None else 1)
    candidates = []
    initial = task.evaluate_terminal(task.initial_state())
    candidates.append((objective_value(initial, weight), initial))
    proposals = []
    failures = []
    if policy is not None:
        for index in range(candidate_budget - 1):
            try:
                proposal = propose_frozen(policy, task, weight, greedy=index == 0, seed=173 + index)
                candidates.append((proposal.objective, proposal.outcome))
                proposals.append({
                    "objective": proposal.objective, "actions": list(proposal.actions),
                    **{name: getattr(proposal, name) for name in (
                        "feature_time_ns", "model_inference_time_ns", "environment_transition_time_ns",
                        "terminal_evaluation_time_ns", "prediction_time_ns")},
                })
            except (ValueError, RuntimeError, OSError) as error:
                # A failure is neither a reward nor an apparent high-quality result.
                failures.append({"candidate_index": index + 1, "error_type": type(error).__name__,
                                 "error": str(error)})
    selection_ns = time.perf_counter_ns() - select_started
    _, selected = min(candidates, key=lambda item: item[0])
    final_recipe = tuple(selected.operator_graph["recipe_action_ids"])
    replay_started = time.perf_counter_ns()
    final_task = runner.make_task(record, recipe=final_recipe, weight=weight, max_edits=2, cold=True)
    final = final_task.evaluate_terminal(final_task.initial_state())
    if final.compressed_payload != selected.compressed_payload:
        raise RuntimeError("final encoding replay differs from selected complete frame")
    saved = _persist_frame(output.with_suffix(".tgf"), final.compressed_payload)
    final_task._independent_cold_validate(Path(saved["path"]).read_bytes())
    final_replay_ns = time.perf_counter_ns() - replay_started
    after = parameter_sha256(policy) if policy is not None else None
    if before != after:
        raise RuntimeError("frozen inference mutated model parameters")
    if final.metrics.original_numeric_bytes != int(record["block"]["original_numeric_bytes"]):
        raise RuntimeError("evaluation denominator changed")
    return {
        "status": "complete_with_candidate_failures" if failures else "complete", "method": method,
        "weight": list(weight), "initial_recipe_action_ids": list(recipe),
        "recipe_action_ids": list(final_recipe), "operator_graph": final.operator_graph,
        "original_numeric_bytes": final.metrics.original_numeric_bytes,
        "complete_compressed_bytes": len(final.compressed_payload), "rho": final.compression_ratio,
        "compression_time_ns": final.metrics.compression_time_ns,
        "decompression_time_ns": final.metrics.decompression_time_ns,
        "E_ns_per_byte": final.metrics.compression_ns_per_byte,
        "D_ns_per_byte": final.metrics.decompression_ns_per_byte,
        "E_hat": final.metrics.normalized_compression_time,
        "D_hat": final.metrics.normalized_decompression_time,
        "J": objective_value(final, weight), "selection_J": objective_value(selected, weight),
        "E0": record["E0"], "D0": record["D0"],
        "selection_walltime_ns": selection_ns if policy is not None else None,
        "fixed_method_measurement_walltime_ns": selection_ns if policy is None else None,
        "selection_includes": "block loading, features, model, seed and proposal evaluations, edits, correctness checks",
        "final_replay_walltime_ns": final_replay_ns, "final_replay_excluded_from_selection_time": True,
        "candidate_evaluations": task.candidate_budget.snapshot()[0],
        "successful_complete_candidates": len(candidates), "candidate_budget": candidate_budget if policy else 1,
        "candidate_failures": failures, "proposal_timing": proposals,
        "frame": saved, "bit_exact": True, "independent_persisted_frame_cold_decode": True,
        "parameter_sha256_before": before, "parameter_sha256_after": after,
        "frozen_parameters_unchanged": before == after, "same_graph_format": True,
    }


def _isolated_child(path: str, work: Callable[[], dict[str, Any]]) -> None:
    os.setsid()
    try:
        record = work()
    except BaseException as error:
        record = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    _write(Path(path), record)


def run_isolated(work: Callable[[], dict[str, Any]], path: Path, timeout: float) -> dict[str, Any]:
    """Fork warm parent, bounded wait, and terminate only this child's group."""
    if timeout <= 0:
        return {"status": "not_run_budget_exhausted"}
    ctx = multiprocessing.get_context("fork")
    child = ctx.Process(target=_isolated_child, args=(str(path), work))
    started = time.monotonic()
    child.start()
    child.join(timeout)
    if child.is_alive():
        try:
            # setsid in the child isolates any native grandchildren as well.
            # Fall back to direct termination if setsid has not yet happened.
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            pass
        child.join(5)
        result = {"status": "timeout", "timeout_seconds": timeout,
                  "elapsed_seconds": time.monotonic() - started,
                  "metrics_unavailable": "row did not finish; no synthetic reward or zero timing"}
        _write(path, result)
        return result
    if not path.is_file():
        result = {"status": "failed", "error": "isolated worker exited without a receipt",
                  "exit_code": child.exitcode}
        _write(path, result)
        return result
    return json.loads(path.read_bytes())


def paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = {tuple(row[k] if k != "weight" else tuple(row[k])
                      for k in ("dataset_id", "block_id", "weight", "method")): row
                for row in rows if row.get("status") in ("complete", "complete_with_candidate_failures")
                and row.get("same_graph_format")}
    counts = {"win": 0, "tie": 0, "loss": 0}
    pairs = []
    for key, trained in complete.items():
        if key[-1] != "trained":
            continue
        initial = complete.get((*key[:-1], "initial"))
        if initial is None:
            continue
        difference = float(trained["J"]) - float(initial["J"])
        verdict = "tie" if abs(difference) <= 1e-12 else "win" if difference < 0 else "loss"
        counts[verdict] += 1
        pairs.append({"dataset_id": key[0], "block_id": key[1], "weight": list(key[2]),
                      "initial_J": initial["J"], "trained_J": trained["J"],
                      "initial_rho": initial["rho"], "trained_rho": trained["rho"], "verdict": verdict})
    return {"trained_vs_initial_same_graph": counts, "paired_rows": len(pairs), "pairs": pairs,
            "caution": "validation panel only; fresh timings are noisy; not independent test evidence"}


def run_evaluation(
    run_dir: Path, *, max_seconds: float = 1800, row_timeout: float = 60,
    candidate_budget: int = 4, zlib6_cli: Path | None = None,
) -> dict[str, Any]:
    import run_weighted_grpo_11source_pilot_v1 as runner
    if (not math.isfinite(max_seconds) or max_seconds <= 0
            or not math.isfinite(row_timeout) or row_timeout <= 0 or not 1 <= candidate_budget <= 64):
        raise ValueError("positive finite time budgets and 1..64 candidate evaluations required")
    run_dir = run_dir.resolve()
    output = run_dir / "validation"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    panel_path, library_path = run_dir / "validation_panel.json", run_dir / "seed_library.json"
    panel_bytes = panel_path.read_bytes()
    records = validate_panel(json.loads(panel_bytes))
    library = json.loads(library_path.read_bytes())
    validate_panel_scope(records, json.loads((run_dir / "input_receipt.json").read_bytes()), library)
    policies = {}
    checkpoint_hashes = {}
    for name, filename in (("initial", "initial_policy.json"), ("trained", "final_policy.json")):
        path = run_dir / filename
        loaded = load_checkpoint(path, expected_registry_fingerprint=graph.REGISTRY_FINGERPRINT)
        policies[name] = FrozenPolicy(loaded.policy)
        checkpoint_hashes[name] = {"file_sha256": _sha(path.read_bytes()),
                                   "parameters_sha256": parameter_sha256(policies[name])}
    identity0, identity1 = (policies[name].identity for name in ("initial", "trained"))
    if identity0 != identity1:
        raise ValueError("initial and final model identities differ")
    base = {
        "schema": SCHEMA, "formal_training_eligible": False, "data_scope": "fixed validation panel, no independent test claims",
        "panel_sha256": _sha(panel_bytes), "seed_library_sha256": _sha(library_path.read_bytes()),
        "checkpoint_hashes": checkpoint_hashes, "registry_fingerprint": graph.REGISTRY_FINGERPRINT,
        "dataset_ids": sorted({r["dataset_id"] for r in records}), "panel_blocks": len(records),
        "weights": [list(w) for w in FIXED_WEIGHTS], "candidate_budget": candidate_budget,
        "candidate_procedure": "one TRAIN-library seed + greedy trajectory + two fixed-seed sampled trajectories when budget=4",
        "max_validation_seconds": max_seconds, "row_timeout_seconds": row_timeout,
        "time_reference_scope": "TRAIN-only frozen native calibration; graph timing boundaries differ, normalized times may saturate",
        "timing_status": "NONFORMAL Python scheduling + native child operators; not unified native in-process graph timing",
        "same_graph_methods": list(METHODS), "fixed_algorithm_baseline": "separate audited ZP11 frame; not a policy action",
    }
    rows = []

    def publish(status: str) -> dict[str, Any]:
        report = {**base, "status": status, "elapsed_seconds": time.monotonic() - started,
                  "rows": rows, "summary": paired_summary(rows)}
        _write(output / "REPORT.json", report)
        return report

    publish("initializing_runtime")
    # The call is cached after runner training; standalone entry pays this time
    # within its validation budget. Native row work remains separately bounded.
    graph.mechanism_probe_report()
    jobs = []
    if zlib6_cli is not None:
        for record in records:
            jobs.append((record, None, "zlib6_separate_wrapper"))
    # Weight-major then source-major keeps partial runs visibly broad. Each
    # source/method pair alternates order to reduce systematic timing warm bias.
    for weight_index, weight in enumerate(FIXED_WEIGHTS):
        for source_index, record in enumerate(records):
            methods = METHODS if (weight_index + source_index) % 2 == 0 else ("trained", "initial", "training_seed", "raw")
            jobs.extend((record, weight, method) for method in methods)
    base["planned_rows"] = len(jobs)
    for index, (record, weight, method) in enumerate(jobs):
        path = output / f"row_{index:04d}.json"
        identity = {"dataset_id": record["dataset_id"], "block_id": record["block"]["block_id"],
                    "column": record["block"]["column"], "dtype": record["block"]["dtype"],
                    "weight": list(weight) if weight is not None else None, "method": method}
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            result = {"status": "not_run_budget_exhausted"}
        else:
            print(json.dumps({"event": "validation_row_start", "row": index, **identity}), flush=True)
            if method == "zlib6_separate_wrapper":
                work = lambda r=record, p=path: _zlib_measure(r, zlib6_cli, p, runner)
            else:
                work = lambda r=record, w=weight, m=method, p=path: _graph_measure(
                    r, w, m, policies.get(m), library, p, runner, candidate_budget)
            result = run_isolated(work, path, min(row_timeout, remaining))
        rows.append({**result, **identity})
        _write(path, rows[-1])
        publish("running")
    for name, policy in policies.items():
        if parameter_sha256(policy) != checkpoint_hashes[name]["parameters_sha256"]:
            raise RuntimeError("parent frozen policy changed during validation")
    base["frozen_parameters_unchanged"] = True
    if _sha(panel_path.read_bytes()) != base["panel_sha256"]:
        raise RuntimeError("validation panel changed during evaluation")
    status = "complete" if all(r["status"] in ("complete", "complete_with_candidate_failures") for r in rows) else "partial"
    return publish(status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run-dir", type=Path)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--row-timeout", type=float, default=60)
    parser.add_argument("--candidate-budget", type=int, default=4)
    parser.add_argument("--zlib6-cli", type=Path)
    parser.add_argument("--cold-zlib-frame", type=Path)
    args = parser.parse_args(argv)
    if args.cold_zlib_frame:
        if not args.zlib6_cli:
            parser.error("cold decoder requires --zlib6-cli")
        dtype, rows, values, bitmap = decode_zlib6_frame(args.zlib6_cli, args.cold_zlib_frame.read_bytes())
        print(json.dumps({"dtype": dtype, "row_count": rows, "values_sha256": _sha(values),
                          "bitmap_sha256": _sha(bitmap)}))
        return 0
    if not args.pilot_run_dir:
        parser.error("--pilot-run-dir is required")
    run_evaluation(args.pilot_run_dir, max_seconds=args.max_seconds, row_timeout=args.row_timeout,
                   candidate_budget=args.candidate_budget, zlib6_cli=args.zlib6_cli)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
