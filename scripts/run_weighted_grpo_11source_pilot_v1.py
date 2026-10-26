#!/usr/bin/env python3
"""Bounded, explicitly NONFORMAL 11-source weight-conditioned GRPO pilot.

This does not bypass or alter any formal campaign gate. All registered legal
actions remain available; each rollout edits its training-derived starting
recipe at most twice. Full source blocks are sampled without truncation. A
monotonic supervisor kills the training process and native descendants when
the two-hour budget expires. Validation is a separately bounded stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

import calibrate_weighted_grpo_references_v1 as calibration
import prepare_weighted_grpo_timeseries_v1 as preparation
import typed_graph_runtime as runtime
import weight_conditioned_graph_task_v1 as graph
import weight_conditioned_grpo_v1 as grpo

SCHEMA = "weighted-grpo-11source-twohour-nonformal-pilot-v1"
TIMING = "python_graph_orchestration_and_native_cli_hybrid_not_native_inprocess_graph"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def event(name: str, **fields: Any) -> None:
    print(json.dumps({"event": name, **fields}, sort_keys=True), flush=True)


def checked_document(record: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(record["path"]).resolve()
    payload = path.read_bytes()
    if sha(payload) != record["sha256"]:
        raise ValueError(f"frozen document changed: {path}")
    return path, json.loads(payload)


def read_block(record: Mapping[str, Any]) -> tuple[bytes, bytes]:
    """Read exactly the frozen full block, including its missing-value bitmap."""
    block = record["block"]
    root = Path(record["manifest_path"]).parent.resolve()
    values = calibration._safe_child(root, block["values_path"]).read_bytes()
    bitmap = calibration._safe_child(root, block["missing_bitmap_path"]).read_bytes()
    if sha(values) != block["values_sha256"] or sha(bitmap) != block["missing_bitmap_sha256"]:
        raise ValueError("frozen block payload fingerprint mismatch")
    if len(values) != int(block["original_numeric_bytes"]):
        raise ValueError("numeric denominator differs from complete value payload")
    if len(values) + len(bitmap) != int(block["prepared_block_bytes"]):
        raise ValueError("prepared bytes must include missing bitmap")
    return values, bitmap


def make_task(record: Mapping[str, Any], *, recipe: Sequence[int] = (),
              weight: Sequence[float] | None = None, max_edits: int = 2,
              cold: bool = True) -> graph.TypedGraphCompressionTask:
    values, bitmap = read_block(record)
    dtype = record["block"]["dtype"]
    width = preparation.SUPPORTED_DTYPES[dtype][0]
    return graph.TypedGraphCompressionTask(
        f"{record['dataset_id']}:{record['block']['block_id']}",
        (runtime.Stream("numeric", width, values), runtime.Stream("serial", 1, bitmap)),
        root_views=(dtype, "bytes"), previous_recipe=tuple(recipe),
        compression_reference_ns_per_byte=float(record["E0"]),
        decompression_reference_ns_per_byte=float(record["D0"]),
        objective_weight=weight, original_numeric_bytes=len(values),
        continuation_edit_mode=True, max_recipe_edits=max_edits,
        config=graph.GraphTaskConfig(
            max_policy_edits=2, timing_warmups=0, timing_repetitions=1,
            independent_cold_validation=cold, cold_validation_timeout_seconds=30,
            require_formal_training_ready=False,
        ),
    )


def seed_recipes(dtype: str) -> list[tuple[int, ...]]:
    """Mechanism IDs are protected by the exact frozen registry fingerprint."""
    recipes = [(), (151,), (167,), (170,)]
    if dtype == "int64":
        recipes.extend([(92, 100), (106, 142, 100)])
    elif dtype == "float64":
        recipes.append((116, 133))
    return recipes


def seed_recipe(record: Mapping[str, Any], weight: Sequence[float],
                seed_library: Mapping[str, Any]) -> tuple[int, ...]:
    """Choose by frozen TRAIN measurements, never by validation outcomes."""
    dtype = record["block"]["dtype"]
    bucket = calibration.block_size_bucket(int(record["block"]["prepared_block_bytes"]))
    rows = [row for row in seed_library["measurements"] if row["dtype"] == dtype]
    if not rows:
        return ()
    distance = min(abs(math.log2(row["block_size_bucket_bytes"]) - math.log2(bucket)) for row in rows)
    rows = [row for row in rows if abs(math.log2(row["block_size_bucket_bytes"]) - math.log2(bucket)) == distance]
    a, b, c = grpo.validate_weight(weight)
    def score(row: Mapping[str, Any]) -> tuple[float, tuple[int, ...]]:
        e, d = row["E_ns_per_byte"], row["D_ns_per_byte"]
        return (float(a * row["rho"] + b * e / (e + record["E0"]) +
                      c * d / (d + record["D0"])), tuple(row["recipe_action_ids"]))
    return tuple(min(rows, key=score)["recipe_action_ids"])


def _isolation_entry(connection: Any, function: Any, args: tuple[Any, ...]) -> None:
    os.setsid()
    try:
        connection.send(("ok", function(*args)))
    except BaseException as error:
        connection.send(("error", {"type": type(error).__name__, "error": str(error),
                                    "traceback": traceback.format_exc()[-6000:]}))
    finally:
        connection.close()


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS reports EPERM for a just-reaped process group; production is Linux.
        if sys.platform != "darwin":
            raise


def run_isolated(function: Any, args: tuple[Any, ...], timeout: float) -> tuple[str, Any]:
    """Fork inherits warmed native probes; timeout kills native descendants too."""
    context = mp.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_isolation_entry, args=(send, function, args))
    child.start()
    send.close()
    end = time.monotonic() + timeout
    result: tuple[str, Any] = ("timeout", {"timeout_seconds": timeout})
    try:
        while time.monotonic() < end:
            if receive.poll(min(0.1, max(0.0, end - time.monotonic()))):
                try:
                    result = receive.recv()
                except EOFError:
                    result = ("error", {"error": "isolated worker closed without a result"})
                break
            if not child.is_alive():
                result = ("error", {"exitcode": child.exitcode})
                break
    finally:
        _kill_group(child.pid)
        child.join(timeout=2)
        if child.is_alive():
            child.kill()
            child.join(timeout=2)
        receive.close()
    return result


def _reference_groups(config: Mapping[str, Any]) -> dict[tuple[str, int], tuple[float, float]]:
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    references = config.get("reference_tables", [config.get("reference_table")])
    seen: set[tuple[str, str]] = set()
    for entry in references:
        if entry is None:
            raise ValueError("frozen reference table is required")
        _, document = checked_document(entry)
        if document.get("schema") != calibration.SMOKE_SCHEMA:
            raise ValueError("pilot expects explicitly nonformal reference receipts")
        digest = document.get("calibration_sha256")
        basis = {key: value for key, value in document.items() if key != "calibration_sha256"}
        if digest != sha(calibration._canonical_json(basis)):
            raise ValueError("reference content digest mismatch")
        for row in document["blocks"]:
            if row.get("split", "train") != "train":
                raise ValueError("normalization may only use TRAIN")
            key = (row["dataset_id"], row["block_id"])
            if key in seen:
                raise ValueError("duplicate reference block")
            seen.add(key)
            grouped[(row["dtype"], int(row["block_size_bucket_bytes"]))].append((
                float(row["compression_ns_per_original_byte"]),
                float(row["decompression_ns_per_original_byte"]),
            ))
    return {key: (statistics.median(row[0] for row in values),
                  statistics.median(row[1] for row in values)) for key, values in grouped.items()}


def prepare_inputs(config: Mapping[str, Any], output: Path) -> dict[str, list[dict[str, Any]]]:
    _, admission = checked_document(config["admission"])
    approved = {row["dataset_id"]: row for row in admission["sources"] if row["status"] == "approved"}
    references = _reference_groups(config)
    reference_rows = {}
    for entry in config.get("reference_tables", [config.get("reference_table")]):
        _, document = checked_document(entry)
        reference_rows.update({(row["dataset_id"], row["block_id"]): row for row in document["blocks"]})
    train: dict[str, list[dict[str, Any]]] = {}
    panel: list[dict[str, Any]] = []
    receipt: list[dict[str, Any]] = []
    manifests = []
    for entry in config["prepared_manifests"]:
        path, manifest = checked_document(entry)
        dataset = manifest["dataset_id"]
        if dataset in train or dataset not in approved:
            raise ValueError("duplicate or unapproved pilot source")
        audit = approved[dataset]
        if (audit["source_group"] != manifest["source_group"] or
                audit["revision"] != manifest["source"].get("upstream_revision", manifest["source"].get("revision")) or
                manifest.get("cohort", {}).get("role") != "train_validation"):
            raise ValueError("pilot source role/revision differs from approved preparation")
        manifests.append(manifest)
        train[dataset] = []
        val = []
        zero = Counter()
        for block in manifest["blocks"]:
            if block["split"] not in {"train", "validation"}:
                raise ValueError("pilot must not consume test sources")
            if int(block["original_numeric_bytes"]) == 0:
                zero[block["split"]] += 1
                continue
            if block["dtype"] not in preparation.SUPPORTED_DTYPES:
                raise ValueError("unknown original numeric dtype")
            bucket = calibration.block_size_bucket(int(block["prepared_block_bytes"]))
            key = (block["dtype"], bucket)
            fallback = None
            if key not in references:
                candidates = [item for item in references if item[0] == block["dtype"]]
                if not candidates:
                    raise ValueError("no TRAIN reference for source dtype")
                nearest = min(candidates, key=lambda item: (abs(math.log2(item[1]) - math.log2(bucket)), item))
                fallback = {"requested_bucket": bucket, "used_train_bucket": nearest[1]}
                key = nearest
            e0, d0 = references[key]
            if min(e0, d0) <= 0 or not all(math.isfinite(v) for v in (e0, d0)):
                raise ValueError("nonpositive/nonfinite TRAIN reference")
            record = {"dataset_id": dataset, "manifest_path": str(path),
                      "manifest_sha256": entry["sha256"], "block": block,
                      "E0": e0, "D0": d0, "reference_nearest_bucket_fallback": fallback}
            if block["split"] == "train":
                values, bitmap = read_block(record)
                measured = reference_rows.pop((dataset, block["block_id"]), None)
                if measured is None or any((
                    measured["provenance_sha256"] != manifest["provenance_sha256"],
                    measured["original_numeric_bytes"] != block["original_numeric_bytes"],
                    measured["dtype"] != block["dtype"],
                    measured["prepared_block_bytes"] != block["prepared_block_bytes"],
                    measured["calibration_payload_sha256"] != sha(values + bitmap),
                )):
                    raise ValueError("TRAIN reference does not match frozen block provenance/payload")
                train[dataset].append(record)
            else:
                val.append(record)
        if not train[dataset] or not val:
            raise ValueError("every pilot source needs positive TRAIN and validation blocks")
        # Metadata-only panel choice is frozen before learning; no validation values read here.
        ordered = sorted(val, key=lambda row: (
            str(row["block"].get("column", "")).lower().startswith("timestamp"),
            int(row["block"]["original_numeric_bytes"]), str(row["block"]["block_id"])))
        panel.append(ordered[0])
        receipt.append({"dataset_id": dataset, "manifest_path": str(path), "manifest_sha256": entry["sha256"],
                        "train_blocks": len(train[dataset]), "validation_blocks": len(val),
                        "train_numeric_bytes": sum(int(row["block"]["original_numeric_bytes"]) for row in train[dataset]),
                        "validation_numeric_bytes": sum(int(row["block"]["original_numeric_bytes"]) for row in val),
                        "zero_numeric_blocks_not_eligible": dict(zero)})
        event("source_verified", **receipt[-1])
    if len(train) != int(config.get("expected_sources", 11)):
        raise ValueError("pilot source count differs from frozen request")
    if reference_rows:
        raise ValueError("reference calibration contains blocks outside the pilot TRAIN sources")
    preparation.validate_prepared_cohort_manifests(manifests, preparation.load_cohort_manifest(),
                                                  require_complete_train=False)
    calibration.write_atomic(output / "input_receipt.json", {
        "schema": SCHEMA, "sources": receipt, "all_positive_train_payload_hashes_verified": True,
        "validation_values_read": False, "test_values_read": False,
        "sampling": "uniform_source_then_uniform_positive_full_train_block_with_replacement",
        "reference_scope": "TRAIN_only_all_verified_native_zlib6_block_receipts_group_medians",
        "graph_reference_timing_boundary_matches": False,
        "formal_training_eligible": False,
    })
    calibration.write_atomic(output / "validation_panel.json", {
        "schema": SCHEMA, "scope": "one_complete_metadata_selected_validation_block_per_source_not_full_validation",
        "frozen_before_training": True, "blocks": panel,
    })
    return train


def _measure_seed(record: Mapping[str, Any], recipe: tuple[int, ...]) -> dict[str, Any]:
    task = make_task(record, recipe=recipe, cold=True)
    result = task.evaluate_terminal(task.initial_state())
    return {"dataset_id": record["dataset_id"], "block_id": record["block"]["block_id"],
            "dtype": record["block"]["dtype"], "source_split": "train",
            "block_size_bucket_bytes": calibration.block_size_bucket(record["block"]["prepared_block_bytes"]),
            "recipe_action_ids": list(recipe), "rho": result.compression_ratio,
            "E_ns_per_byte": result.metrics.compression_ns_per_byte,
            "D_ns_per_byte": result.metrics.decompression_ns_per_byte,
            "complete_frame_sha256": sha(result.compressed_payload), "cold_decode_passed": True}


def build_seed_library(train: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> dict[str, Any]:
    rows, failures = [], []
    for dataset, blocks in sorted(train.items()):
        # Choose one original complete TRAIN block; short blocks make bootstrap bounded.
        record = min(blocks, key=lambda row: (str(row["block"].get("column", "")).lower().startswith("timestamp"),
                                              row["block"]["original_numeric_bytes"], row["block"]["block_id"]))
        for recipe in seed_recipes(record["block"]["dtype"]):
            status, result = run_isolated(_measure_seed, (record, recipe), 30)
            if status == "ok":
                rows.append(result)
            else:
                failures.append({"dataset_id": dataset, "recipe_action_ids": list(recipe), "status": status, "detail": result})
        event("seed_source_complete", dataset_id=dataset, successes=len(rows), failures=len(failures))
    if not rows:
        raise RuntimeError("no seed candidate passed complete-frame cold validation")
    document = {"schema": SCHEMA, "source_scope": "training_only", "measurements": rows,
                "failures": failures, "native_graph_timing_boundary": TIMING,
                "raw_fallback_allowed": True, "not_formal_complete_solution_library": True}
    calibration.write_atomic(output / "seed_library.json", document)
    return document


def _collect_group(policy: Any, record: Mapping[str, Any], weight: np.ndarray,
                   recipe: tuple[int, ...], seed: int) -> dict[str, Any]:
    task = make_task(record, recipe=recipe, weight=weight, cold=False)
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(8):
        episode = grpo._rollout(policy, task, weight, rng, greedy=False, timing=True)
        outcome = episode.pop("outcome")
        episode.pop("final_state")
        episode["recipe_action_ids"] = outcome.operator_graph["recipe_action_ids"]
        episode["rho"] = outcome.compression_ratio
        episode["E_ns_per_byte"] = outcome.metrics.compression_ns_per_byte
        episode["D_ns_per_byte"] = outcome.metrics.decompression_ns_per_byte
        episodes.append(episode)
    return {"episodes": episodes, "task_id": task.task_id, "data_fingerprint": task.data_fingerprint,
            "root_fingerprint": task.root_fingerprint, "starting_recipe": list(recipe)}


def pack_batch(episodes: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    advantages = grpo.group_advantages([row["return"] for row in episodes])
    lengths = np.asarray([row["actions"].size for row in episodes], dtype=np.int64)
    episode_ids = np.repeat(np.arange(len(episodes)), lengths)
    return {**{key: np.concatenate([row[key] for row in episodes])
               for key in ("features", "legal", "actions", "old_logp")},
            "advantages": advantages[episode_ids], "token_weights": 1.0 / lengths[episode_ids]}


def parameter_hash(policy: Any) -> str:
    return sha(b"".join(np.ascontiguousarray(value, dtype="<f8").tobytes() for value in policy.params))


def _checkpoint(output: Path, policy: Any, optimizer: Any, config: Any, seed: int,
                completed: int, name: str, extra: Mapping[str, Any]) -> None:
    grpo.save_checkpoint(grpo.TrainingResult(policy, (), config, seed, completed), output / name,
                         extra={"scope": SCHEMA, "formal_training_eligible": False, **extra})
    # Atomically preserve persistent Adam state separately; policy remains standard readable format.
    temporary = output / "optimizer.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, steps=np.asarray(optimizer.steps),
                 **{f"first_{i}": value for i, value in enumerate(optimizer.first)},
                 **{f"second_{i}": value for i, value in enumerate(optimizer.second)})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "optimizer.npz")


def _training_loop(train: Mapping[str, Sequence[Mapping[str, Any]]], library: Mapping[str, Any],
                   policy: Any, output: Path, seed: int, deadline: float, group_timeout: float) -> None:
    os.setsid()
    config = grpo.GRPOConfig(groups_per_iteration=2)
    optimizer = grpo.Adam(policy.params, config.learning_rate)
    rng = np.random.default_rng(seed + 1000003)
    source_names = sorted(train)
    started = time.monotonic()
    completed = attempts = 0
    successes, attempted, failures = Counter(), Counter(), Counter()
    unique = set()
    first_hash = parameter_hash(policy)
    with (output / "training_events.jsonl").open("a", buffering=1) as log:
        while time.monotonic() < deadline - 2:
            groups, details = [], []
            # A completed update always has one fixed and one continuous weight group.
            weights = grpo.sample_training_weights(2, rng)
            for weight, weight_kind in weights:
                if time.monotonic() >= deadline - 2:
                    break
                dataset = source_names[int(rng.integers(len(source_names)))]
                record = train[dataset][int(rng.integers(len(train[dataset])))]
                recipe = seed_recipe(record, weight, library)
                attempts += 1
                attempted[dataset] += 1
                status, result = run_isolated(_collect_group,
                    (policy, record, weight, recipe, int(rng.integers(2**31))),
                    min(group_timeout, max(0.01, deadline - time.monotonic() - 1)))
                row = {"event": "training_group", "attempt": attempts, "dataset_id": dataset,
                       "block_id": record["block"]["block_id"], "weight": weight.tolist(),
                       "weight_source": weight_kind, "status": status,
                       "elapsed_seconds": time.monotonic() - started,
                       "original_numeric_bytes": record["block"]["original_numeric_bytes"]}
                if status == "ok":
                    groups.extend(result["episodes"])
                    successes[dataset] += 1
                    unique.add((dataset, record["block"]["block_id"]))
                    row.update(starting_recipe=result["starting_recipe"],
                               mean_J=float(np.mean([v["objective"] for v in result["episodes"]])),
                               terminal_recipes=[v["recipe_action_ids"] for v in result["episodes"]])
                    details.append(row)
                else:
                    failures[status] += 1
                    row["detail"] = result
                log.write(json.dumps(row, sort_keys=True) + "\n")
                event("training_group", **{k: row[k] for k in ("attempt", "dataset_id", "status", "elapsed_seconds")})
            if len(groups) != 16 or time.monotonic() >= deadline - 2:
                continue
            batch = pack_batch(groups)
            before = [value.copy() for value in policy.params]
            metrics = grpo._update(policy, optimizer, batch, config)
            if any(not np.isfinite(value).all() for value in policy.params):
                raise FloatingPointError("nonfinite pilot model")
            completed += 1
            change = math.sqrt(sum(float(np.square(old-new).sum()) for old,new in zip(before, policy.params)))
            summary = {"completed_updates": completed, "optimizer_steps": optimizer.steps,
                       "persistent_adam": True, "attempted_groups": attempts,
                       "successful_groups_by_source": dict(successes),
                       "attempted_groups_by_source": dict(attempted), "failed_groups": dict(failures),
                       "unique_train_blocks_successfully_sampled": len(unique),
                       "parameter_sha256_initial": first_hash, "parameter_sha256_current": parameter_hash(policy),
                       "last_parameter_change_l2": change, "training_elapsed_seconds": time.monotonic()-started}
            _checkpoint(output, policy, optimizer, config, seed, completed, "final_policy.json", summary)
            calibration.write_atomic(output / "training_status.json", {"status": "running", **summary})
            log.write(json.dumps({"event": "update", **summary, **metrics, "groups": details}, sort_keys=True)+"\n")
            event("update_saved", **summary)
        calibration.write_atomic(output / "training_exit.json", {
            "status": "budget_complete", "completed_updates": completed, "optimizer_steps": optimizer.steps,
            "elapsed_seconds": time.monotonic()-started, "attempted_groups_by_source": dict(attempted),
            "successful_groups_by_source": dict(successes), "failed_groups": dict(failures)})


def _descendants(pid: int) -> list[int]:
    """Linux process tree used by hard deadline supervisor, not user processes."""
    parents: dict[int, list[int]] = defaultdict(list)
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            tail = path.read_text().rsplit(")", 1)[1].split()
            parents[int(tail[1])].append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    pending, result = [pid], []
    while pending:
        current = pending.pop()
        result.append(current)
        pending.extend(parents.get(current, ()))
    return result


def supervise_training(target: Any, args: tuple[Any, ...], seconds: float) -> dict[str, Any]:
    """Hard wall clock includes optimizer/checkpoint/failed rollout time."""
    started = time.monotonic()
    deadline = started + seconds
    child = mp.get_context("fork").Process(target=target, args=(*args, deadline))
    child.start()
    while child.is_alive() and time.monotonic() < deadline:
        child.join(timeout=min(0.2, max(0.0, deadline-time.monotonic())))
    exhausted = child.is_alive()
    if exhausted:
        tree = _descendants(child.pid)
        groups = set()
        for pid in tree:
            try:
                groups.add(os.getpgid(pid))
            except ProcessLookupError:
                pass
        for group in groups - {os.getpgrp()}:
            _kill_group(group)
        for pid in reversed(tree):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.join(timeout=3)
    return {"budget_seconds": seconds, "elapsed_seconds": time.monotonic()-started,
            "deadline_enforced": exhausted, "training_worker_exitcode": child.exitcode}


def _train_entry(train: Any, library: Any, policy: Any, output: Path, seed: int,
                 group_timeout: float, deadline: float) -> None:
    _training_loop(train, library, policy, output, seed, deadline, group_timeout)


def run(config_path: Path, output: Path, *, seconds: float = 7200, seed: int = 17,
        group_timeout: float = 120, evaluate: bool = True) -> dict[str, Any]:
    if not 1 <= seconds <= 7200 or not 1 <= group_timeout <= 600:
        raise ValueError("pilot budget must be 1..7200 s and group timeout 1..600 s")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_bytes())
    if config["expected_registry_fingerprint"] != graph.REGISTRY_FINGERPRINT:
        raise ValueError("pilot registry changed")
    config.update(schema=SCHEMA, train_seconds=seconds, seed=seed, group_timeout=group_timeout,
                  graph_timing_boundary=TIMING, formal_training_eligible=False,
                  graph_reference_timing_boundary_matches=False, max_recipe_edits=2,
                  hidden_sizes=[128,128], timing_repetitions=1, evaluation_separately_timed=True)
    calibration.write_atomic(output / "config.json", config)
    calibration.write_atomic(output / "status.json", {"status": "input_verification", "formal_training_eligible": False})
    train = prepare_inputs(config, output)
    event("native_probe_warmup_start")
    probes = graph.mechanism_probe_report()
    calibration.write_atomic(output / "mechanism_probes.json", probes)
    if any(not row.get("passed", False) for row in probes["mechanisms"]):
        raise RuntimeError("native mechanism probe failed; no pilot training")
    calibration.write_atomic(output / "status.json", {"status": "train_only_seed_bootstrap"})
    library = build_seed_library(train, output)
    first = train[sorted(train)[0]][0]
    task = make_task(first, cold=False)
    policy = grpo.initialize_policy([task], seed=seed)
    fingerprints = tuple(sorted((name, blocks[0]["manifest_sha256"]) for name, blocks in train.items()))
    policy.identity = replace(policy.identity, data_fingerprints=fingerprints,
                              root_fingerprints=tuple((name, sha(json.dumps(library,sort_keys=True).encode())) for name,_ in fingerprints))
    config["model_parameters"] = sum(array.size for array in policy.params)
    config["model_parameter_bytes"] = sum(array.nbytes for array in policy.params)
    calibration.write_atomic(output / "config.json", config)
    training_config = grpo.GRPOConfig(groups_per_iteration=2)
    initial = grpo.TrainingResult(policy, (), training_config, seed, 0)
    for name in ("initial_policy.json", "final_policy.json"):
        grpo.save_checkpoint(initial, output / name, extra={"scope": SCHEMA, "formal_training_eligible": False,
                                                           "status": "initial_not_trained"})
    started_utc = datetime.now(timezone.utc)
    training_clock = {"training_start_utc": started_utc.isoformat(),
                      "expected_training_cutoff_utc": (started_utc + timedelta(seconds=seconds)).isoformat()}
    calibration.write_atomic(output / "status.json", {"status": "training", "budget_seconds": seconds,
                                                     "model_parameters": config["model_parameters"], **training_clock})
    event("training_start", budget_seconds=seconds, model_parameters=config["model_parameters"], **training_clock)
    receipt = supervise_training(_train_entry, (train, library, policy, output, seed, group_timeout), seconds)
    calibration.write_atomic(output / "training_budget_receipt.json", receipt)
    if not receipt["deadline_enforced"] and receipt["training_worker_exitcode"] != 0:
        raise RuntimeError(f"training worker exited unexpectedly: {receipt['training_worker_exitcode']}; last checkpoint retained")
    loaded = grpo.load_checkpoint(output / "final_policy.json", expected_registry_fingerprint=graph.REGISTRY_FINGERPRINT)
    status = {"status": "training_finished", "completed_updates": loaded.completed_iterations,
              "formal_training_eligible": False, **receipt, **training_clock}
    calibration.write_atomic(output / "status.json", status)
    event("training_finished", **status)
    if evaluate:
        from evaluate_weighted_grpo_11source_pilot_v1 import run_evaluation
        run_evaluation(output, zlib6_cli=Path(config["zlib6_cli"]) if config.get("zlib6_cli") else None)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seconds", type=float, default=7200)
    parser.add_argument("--group-timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args.config, args.output_dir, seconds=args.train_seconds, seed=args.seed,
            group_timeout=args.group_timeout, evaluate=not args.skip_evaluation)
    except BaseException as error:
        if args.output_dir.is_dir():
            calibration.write_atomic(args.output_dir / "failure.json", {"schema": SCHEMA,
                "status": "failed", "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
