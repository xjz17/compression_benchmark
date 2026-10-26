#!/usr/bin/env python3
"""Bounded offline teacher search for the v13 per-flow STOP policy.

The search runs only on admitted training blocks.  It uses the same C++
``OnlineTrainingServer`` state, legality mask, graph executor, complete NGF1
frame, independent decoder, and loaded 6,373-action policy forward pass as the
online policy.  Search is deliberately outside deployment: the emitted teacher
contains forced legal trajectories; it does not add candidate evaluation to
online inference.

Unlike the old quick teacher, stage identities are exact ``runtime_action``
matches.  In particular, the project rANS stage is never called Zstd and the
LZ4 match finder is never called a complete LZ4 codec.  Direct LZ4 and Zstd
references are measured separately through the frozen native baseline CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import online_matched_train_v6 as fit3


SCHEMA = "native-online-offline-combo-teacher-v6"
SEARCH_SPACE_SCHEMA = "native-online-offline-combo-space-v6"
MEASUREMENT_SCHEMA = "native-online-offline-combo-measurement-v6"
TEACHER_SCHEMA = fit3.TEACHER_SCHEMA


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return fit3.digest(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    payload = fit3.canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(fit3.canonical_json(dict(row)) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class Action:
    action_id: int
    runtime_action: str
    implementation_id: str | None
    input_view: str
    target_slot: int | None
    ordinary_operator: bool
    key: str


class ExactCatalog:
    """Frozen action lookup with no substring aliases or lowest-ID guessing."""

    def __init__(self, catalog_path: Path, profile_path: Path):
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
        require(document.get("registry_fingerprint") ==
                "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd",
                "base catalog registry fingerprint drift")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        excluded = frozenset(str(value) for value in
                             profile.get("excluded_implementation_ids", ()))
        require({"A109", "A110"}.issubset(excluded),
                "offline v4 profile must exclude cmix and NNCP")
        actions: dict[int, Action] = {}
        for row in document["registry_payload"]["actions"]:
            action = Action(
                int(row["action_id"]), str(row["runtime_action"]),
                None if row.get("implementation_id") is None else
                str(row["implementation_id"]), str(row.get("input_view", "")),
                None if row.get("target_slot") is None else int(row["target_slot"]),
                bool(row.get("ordinary_operator")), str(row["key"]))
            require(action.action_id not in actions, "duplicate catalog action ID")
            actions[action.action_id] = action
        require(len(actions) == fit3.FLOW_STOP_BEGIN,
                "base catalog action count drift")
        self.actions = actions
        self.catalog_path = catalog_path.resolve()
        self.profile_path = profile_path.resolve()
        self.excluded_implementation_ids = excluded

    def exact(self, legal: Sequence[int], runtime_action: str, *,
              target_slot: int) -> int:
        matches = [action_id for action_id in legal
                   if action_id < fit3.FLOW_STOP_BEGIN and
                   self.actions[action_id].runtime_action == runtime_action and
                   self.actions[action_id].target_slot == target_slot]
        require(len(matches) == 1,
                f"exact stage {runtime_action!r} at slot {target_slot} has "
                f"{len(matches)} legal actions")
        action = self.actions[matches[0]]
        require(action.implementation_id not in self.excluded_implementation_ids,
                "excluded implementation became a teacher action")
        return matches[0]

    def describe(self, recipe: Sequence[int]) -> list[dict[str, Any]]:
        rows = []
        for position, action_id in enumerate(recipe):
            if action_id >= fit3.FLOW_STOP_BEGIN:
                rows.append({"position": position, "action_id": action_id,
                             "mechanism": "per_flow_stop",
                             "flow_slot": action_id - fit3.FLOW_STOP_BEGIN})
                continue
            action = self.actions[action_id]
            rows.append({"position": position, "action_id": action_id,
                         "mechanism": action.runtime_action,
                         "implementation_id": action.implementation_id,
                         "input_view": action.input_view,
                         "target_slot": action.target_slot,
                         "catalog_key": action.key})
        return rows


LZ4_STAGES = (
    "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
    "op.match_token_format.lz4.cpp_v1",
    "op.integer_code.lz4.cpp_v1",
)
SNAPPY_STAGES = (
    "op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",
    "op.match_token_format.snappy.cpp_v1",
)


@dataclass(frozen=True)
class Plan:
    name: str
    prefix: tuple[str, ...] = ()
    leaf_backend: tuple[str, ...] = ()
    backend_each_leaf: bool = False

    def __post_init__(self) -> None:
        require(self.name != "", "empty candidate name")
        require(len(self.prefix) <= 3, "offline v4 prefix depth exceeds three")
        require(bool(self.prefix) or self.name == "raw", "non-raw plan is empty")
        require(not self.backend_each_leaf or self.leaf_backend,
                "multi-flow plan requires a leaf backend")


def frozen_plans() -> tuple[Plan, ...]:
    """The deterministic, auditable <=128 candidate subset.

    These are graph-stage recipes, not names for complete external methods.
    Failed type/parent combinations remain measurement records.
    """

    terminal = {
        "rle": ("op.run_length.byte_optimal.cpp_v1",),
        "huffman": ("op.entropy_huffman.project_bytes.cpp_v1",),
        "range": ("op.entropy_range.project_bytes.cpp_v1",),
        "rans_reference": ("op.entropy_fse.zstd_rans.cpp_reference_v1",),
        "lz4_stage_triplet": LZ4_STAGES,
        "snappy_stage_pair": SNAPPY_STAGES,
        "max_value_bitpack": ("op.bitpack_int.max_value_width.cpp_v1",),
        "fastpfor_pfor": ("op.bitpack_int.fastpfor_pfor.cpp_v1",),
        "simple8b": ("op.bitpack_int.simple8b.cpp_v1",),
    }
    plans: list[Plan] = [Plan("raw")]
    for label, stages in terminal.items():
        plans.append(Plan(label, stages))

    # Put the multi-flow cases before the product below so the default
    # 64-candidate budget cannot silently omit them.
    multi = {
        "for_minimum": ("op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",),
        "dictionary_words64": ("op.dictionary.project_words64.cpp_v1",),
    }
    for prefix_name, prefix in multi.items():
        for backend_name in ("rle", "huffman", "rans_reference"):
            plans.append(Plan(f"{prefix_name}+each_{backend_name}", prefix,
                              terminal[backend_name], True))

    prefixes = {
        "delta_reference": ("op.delta.reference_v1",),
        "delta2": ("op.delta.ts_2diff.cpp_v1",),
        "delta_streamvbyte": ("op.delta.stream_vbyte.cpp_v1",),
        "delta_sprintz": ("op.delta.sprintz.cpp_v1",),
        "xor_previous": ("op.xor.previous_word.cpp_v2",),
        "xor_gorilla": ("op.xor.gorilla.cpp_v1",),
        "xor_chimp128": ("op.xor.chimp128.cpp_v1",),
        "zigzag": ("op.zigzag.signed64.raw.cpp_v2",),
        "shuffle4": ("op.block_transform.blosc_shuffle.typesize4.cpp_reference_v1",),
        "shuffle8": ("op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1",),
        "byte_plane": ("op.block_transform.buff_byte_plane_split.cpp_v2",),
        "native_word_order": ("op.block_transform.buff_native_word_order.cpp_v1",),
    }
    compact_backends = {name: terminal[name] for name in
                        ("rle", "huffman", "range", "rans_reference",
                         "lz4_stage_triplet", "snappy_stage_pair")}
    for prefix_name, prefix in prefixes.items():
        for backend_name, backend in compact_backends.items():
            if len(prefix) + len(backend) <= 3:
                plans.append(Plan(f"{prefix_name}+{backend_name}", prefix + backend))

    # The bound is part of the search identity, not a silent runtime truncation.
    require(32 <= len(plans) <= 128 and len({plan.name for plan in plans}) == len(plans),
            "invalid frozen candidate inventory")
    return tuple(plans)


def _stop_slot(backend: fit3.OnlineBackend, snapshot: fit3.Snapshot,
               slot: int) -> fit3.Snapshot:
    action = fit3.FLOW_STOP_BEGIN + slot
    require(action in snapshot.legal_action_ids, "requested per-flow STOP is illegal")
    return backend.step(action)


def _apply_exact(backend: fit3.OnlineBackend, catalog: ExactCatalog,
                 snapshot: fit3.Snapshot, runtime_action: str,
                 slot: int) -> fit3.Snapshot:
    require(not snapshot.terminal, "cannot apply a stage to a terminal session")
    action = catalog.exact(snapshot.legal_action_ids, runtime_action,
                           target_slot=slot)
    return backend.step(action)


def materialize_plan(backend: fit3.OnlineBackend, block: fit3.TrainingBlock,
                     contract: fit3.OnlineContract, catalog: ExactCatalog,
                     plan: Plan) -> tuple[int, ...]:
    """Build one legal recipe while tracking the active flow count.

    Root zero is the admitted numeric value flow.  Auxiliary roots (for
    example the missing bitmap) are retained as raw terminal leaves.  Closing
    them first leaves a single unambiguous primary flow for the stage path.
    """

    require(block.roots[0].get("original_numeric_payload") is True,
            "offline v4 requires the primary numeric root at slot zero")
    snapshot = backend.reset(block, fit3.FIXED_WEIGHTS[0], contract)
    initial = int(snapshot.response.get("open_flow_count", len(block.roots)))
    require(initial == len(block.roots), "C++ initial open-flow count drift")
    for slot in range(initial - 1, 0, -1):
        snapshot = _stop_slot(backend, snapshot, slot)
    if plan.name == "raw":
        require(not snapshot.terminal, "raw primary unexpectedly terminal")
        snapshot = _stop_slot(backend, snapshot, 0)
        return fit3._recipe(snapshot.response)

    for runtime_action in plan.prefix:
        snapshot = _apply_exact(backend, catalog, snapshot, runtime_action, 0)
        require(not snapshot.terminal,
                f"candidate {plan.name} exhausted policy budget in prefix")

    if plan.backend_each_leaf:
        while not snapshot.terminal:
            count = int(snapshot.response["open_flow_count"])
            slot = count - 1
            for runtime_action in plan.leaf_backend:
                snapshot = _apply_exact(backend, catalog, snapshot,
                                        runtime_action, slot)
                require(not snapshot.terminal,
                        f"candidate {plan.name} exhausted budget before leaf STOP")
                slot = int(snapshot.response["open_flow_count"]) - 1
            snapshot = _stop_slot(backend, snapshot, slot)
    else:
        while not snapshot.terminal:
            count = int(snapshot.response["open_flow_count"])
            snapshot = _stop_slot(backend, snapshot, count - 1)
    return fit3._recipe(snapshot.response)


def _run_json(command: Sequence[str], timeout: float, *, cpu: int | None) -> dict[str, Any]:
    argv = list(command)
    if cpu is not None:
        argv = ["taskset", "-c", str(cpu), *argv]
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout,
                               env=environment, check=False)
    require(completed.returncode == 0,
            f"native baseline failed rc={completed.returncode}: " + completed.stderr[-1000:])
    return json.loads(completed.stdout)


def direct_baselines(block: fit3.TrainingBlock, baseline_cli: Path, *,
                     timeout: float, cpu: int | None) -> list[dict[str, Any]]:
    """Measure true complete LZ4/Zstd methods on the same admitted bytes."""

    numeric = [root for root in block.roots
               if root.get("original_numeric_payload") is True]
    require(len(numeric) == 1, "direct baseline requires exactly one numeric root")
    root = numeric[0]
    dtype = str(root["logical_view"])
    require(dtype in fit3.DTYPE_WIDTH and int(root["width"]) == fit3.DTYPE_WIDTH[dtype],
            "direct baseline requires a fixed-width numeric root")
    values = bytes.fromhex(str(root["data_hex"]))
    require(len(values) == block.original_numeric_bytes and
            len(values) % int(root["width"]) == 0, "baseline B_raw mismatch")
    rows = len(values) // int(root["width"])
    bitmaps = [bytes.fromhex(str(item["data_hex"])) for item in block.roots
               if item.get("original_numeric_payload") is False and
               item.get("logical_view") == "bytes" and
               len(bytes.fromhex(str(item["data_hex"]))) == (rows + 7) // 8]
    require(len(bitmaps) <= 1, "ambiguous missing bitmap root")
    missing = bitmaps[0] if bitmaps else bytes((rows + 7) // 8)
    output = []
    with tempfile.TemporaryDirectory(prefix="offline_combo_v4.") as temporary:
        folder = Path(temporary)
        values_path, missing_path = folder / "values.bin", folder / "missing.bin"
        values_path.write_bytes(values)
        missing_path.write_bytes(missing)
        for method in ("LZ4", "Zstd"):
            try:
                frame_path = folder / f"{method}.frame"
                response = _run_json(
                    [str(baseline_cli), "measure", method, dtype, str(rows),
                     str(values_path), str(missing_path), "0", "1", str(frame_path)],
                    timeout, cpu=cpu)
                require(response.get("status") == "ok" and
                        int(response["original_numeric_bytes"]) ==
                        block.original_numeric_bytes, "baseline status/denominator drift")
                require(response.get("primary_comparison") is True,
                        "baseline is not a primary direct implementation")
                decoded_values = folder / f"{method}.decoded.values"
                decoded_missing = folder / f"{method}.decoded.missing"
                _run_json([str(baseline_cli), "decode", str(frame_path),
                           str(decoded_values), str(decoded_missing)], timeout, cpu=cpu)
                require(decoded_values.read_bytes() == values and
                        decoded_missing.read_bytes() == missing,
                        "baseline independent decode mismatch")
                frame = frame_path.read_bytes()
                require(len(frame) == int(response["complete_frame_bytes"]) and
                        fit3.digest(frame) == response["complete_frame_sha256"] and
                        math.isclose(float(response["compression_ratio"]),
                                     len(frame) / block.original_numeric_bytes,
                                     rel_tol=1e-12, abs_tol=1e-12),
                        "baseline complete-frame identity/ratio drift")
                compression_samples = tuple(int(value) for value in
                                            response["compression_ns_samples"])
                decompression_samples = tuple(int(value) for value in
                                              response["decompression_ns_samples"])
                require(len(compression_samples) == 1 and
                        len(decompression_samples) == 1 and
                        all(value > 0 for value in
                            (*compression_samples, *decompression_samples)),
                        "baseline native timing samples invalid")
                output.append({"method": method, "status": "ok",
                               "frame_bytes": len(frame),
                               "compression_ratio": len(frame) /
                               block.original_numeric_bytes,
                               "compression_ns_per_byte":
                               float(response["compression_ns"]) /
                               block.original_numeric_bytes,
                               "decompression_ns_per_byte":
                               float(response["decompression_ns"]) /
                               block.original_numeric_bytes,
                               "timing_scope": "direct_native_complete_method_no_mlp",
                               "independent_decode": True,
                               "native_response": response})
            except Exception as error:
                output.append({"method": method, "status": "failed",
                               "error": type(error).__name__ + ": " + str(error)})
    return output


def measurement_row(plan: Plan, recipe: Sequence[int], catalog: ExactCatalog,
                    measurement: fit3.RecipeMeasurement,
                    block: fit3.TrainingBlock, weight: Sequence[float], *,
                    measured_timing_weight: Sequence[float] | None = None
                    ) -> dict[str, Any]:
    requested_weight = fit3._weights(weight)
    measured_weight = fit3._weights(measured_timing_weight or requested_weight)
    reused = requested_weight != measured_weight
    return {"schema": MEASUREMENT_SCHEMA, "sample_id": block.sample_id,
            "dataset_id": block.dataset_id, "domain": block.domain,
            "split": block.split, "candidate": plan.name,
            "weight": list(requested_weight),
            "status": "ok", "recipe_action_ids": list(recipe),
            "recipe_mechanisms": catalog.describe(recipe),
            "frame_bytes": measurement.frame_bytes,
            "original_numeric_bytes": measurement.original_numeric_bytes,
            "compression_ratio": measurement.compression_ratio,
            "compression_ns_per_byte": measurement.compression_ns_per_byte,
            "decompression_ns_per_byte": measurement.decompression_ns_per_byte,
            "objective": fit3.objective(measurement, block, requested_weight),
            "frame_sha256": measurement.frame_sha256,
            "canonical_recipe_key": measurement.canonical_recipe_key,
            "timing_scope": "cpp_total_forced_action_current_model_forward",
            "measured_timing_weight": list(measured_weight),
            "time_reused_across_weights": reused,
            "approximate_teacher_timing": reused,
            "in_process_roundtrip": True,
            "independent_decode": False}


def pareto_indices(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    valid = [(index, row) for index, row in enumerate(rows)
             if row.get("status") == "ok"]
    kept: set[int] = set()
    for index, row in valid:
        point = tuple(float(row[key]) for key in
                      ("compression_ratio", "compression_ns_per_byte",
                       "decompression_ns_per_byte"))
        dominated = False
        for other_index, other in valid:
            if index == other_index:
                continue
            rival = tuple(float(other[key]) for key in
                          ("compression_ratio", "compression_ns_per_byte",
                           "decompression_ns_per_byte"))
            if all(a <= b for a, b in zip(rival, point)) and any(
                    a < b for a, b in zip(rival, point)):
                dominated = True
                break
        if not dominated:
            kept.add(index)
    return kept


def quality_proof(rows: Sequence[Mapping[str, Any]],
                  baselines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-block ratio oracle and direct complete baselines."""

    ratio_rows = [row for row in rows if row.get("status") == "ok" and
                  tuple(row.get("weight", ())) == tuple(fit3.FIXED_WEIGHTS[0])]
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for row in ratio_rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    winners = {sample: min(items, key=lambda item: (
        float(item["compression_ratio"]), str(item["candidate"])))
        for sample, items in by_sample.items()}
    baseline_by_method: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in baselines:
        if row.get("status") == "ok":
            baseline_by_method.setdefault(str(row["method"]), {})[
                str(row["sample_id"])] = row
    common = set(winners)
    for method in ("RawNGF1", "LZ4", "Zstd"):
        common &= set(baseline_by_method.get(method, {}))
    if not common:
        return {"status": "insufficient_coverage", "passed": False,
                "common_samples": 0}
    selected_bytes = sum(int(winners[sample]["frame_bytes"]) for sample in common)
    raw_bytes = sum(int(winners[sample]["original_numeric_bytes"]) for sample in common)
    selected_ratio = selected_bytes / raw_bytes
    comparisons = {}
    for method in ("RawNGF1", "LZ4", "Zstd"):
        baseline_bytes = sum(int(baseline_by_method[method][sample]["frame_bytes"])
                             for sample in common)
        baseline_ratio = baseline_bytes / raw_bytes
        comparisons[method] = {
            "compression_ratio": baseline_ratio,
            "offline_oracle_is_better": selected_ratio < baseline_ratio,
            "per_block_wins": sum(
                float(winners[sample]["compression_ratio"]) <
                float(baseline_by_method[method][sample]["compression_ratio"])
                for sample in common),
        }
    return {"status": "complete", "passed":
            all(value["offline_oracle_is_better"] for value in comparisons.values()),
            "common_samples": len(common), "original_numeric_bytes": raw_bytes,
            "offline_oracle_compression_ratio": selected_ratio,
            "comparisons": comparisons,
            "winner_candidates": {sample: str(winners[sample]["candidate"])
                                  for sample in sorted(common)}}


def _raw_baseline(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"method": "RawNGF1", "sample_id": row["sample_id"],
            "dataset_id": row["dataset_id"], "domain": row["domain"],
            "status": "ok", "frame_bytes": row["frame_bytes"],
            "original_numeric_bytes": row["original_numeric_bytes"],
            "compression_ratio": row["compression_ratio"],
            "compression_ns_per_byte": row["compression_ns_per_byte"],
            "decompression_ns_per_byte": row["decompression_ns_per_byte"],
            "timing_scope": row["timing_scope"],
            "independent_decode": True}


def independent_cold_decode(cold_cli: Path, frame_hex: str,
                            roots: Sequence[Mapping[str, Any]], output: Path, *,
                            timeout: float, cpu: int | None) -> dict[str, Any]:
    frame = bytes.fromhex(frame_hex)
    frame_sha = fit3.digest(frame)
    with tempfile.TemporaryDirectory(prefix="offline-combo-cold-v4.") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_bytes(fit3.canonical_json({
            "schema": "native-graph-request-v1", "frame_hex": frame_hex}) + b"\n")
        response = _run_json([str(cold_cli), "decode", str(request_path)], timeout,
                             cpu=cpu)
    expected = [dict(root) for root in roots]
    require(response.get("schema") == "native-graph-response-v1" and
            response.get("roots") == expected,
            "independent cold decoder changed roots")
    raw = sum(len(bytes.fromhex(str(root["data_hex"]))) for root in roots
              if root.get("original_numeric_payload") is True)
    require(int(response.get("original_numeric_bytes", -1)) == raw,
            "independent cold decoder changed B_raw")
    frames = output / "winner_frames"
    frames.mkdir(parents=True, exist_ok=True)
    frame_path = frames / f"{frame_sha}.ngf1"
    if frame_path.exists():
        require(frame_path.read_bytes() == frame, "winner frame SHA collision")
    else:
        frame_path.write_bytes(frame)
    return {"schema": "native-online-winner-cold-decode-v4",
            "frame_sha256": frame_sha, "frame_bytes": len(frame),
            "frame_path": str(frame_path), "cold_cli": str(cold_cli.resolve()),
            "cold_cli_sha256": sha256(cold_cli), "roots_verified": len(roots),
            "original_numeric_bytes": raw, "independent_process": True,
            "bit_exact_roots": True}


def _teacher_rows(backend: fit3.OnlineBackend, contract: fit3.OnlineContract,
                  block_map: Mapping[str, fit3.TrainingBlock],
                  winners: Sequence[Mapping[str, Any]], *, cold_cli: Path,
                  output: Path, timeout: float, cpu: int | None
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    cold_receipts: list[dict[str, Any]] = []
    cold_keys: set[tuple[str, tuple[int, ...]]] = set()
    for winner in winners:
        require(winner.get("split") == "train", "validation cannot produce teacher labels")
        block = block_map[str(winner["sample_id"])]
        weight = fit3._weights(winner["weight"])
        recipe = tuple(int(value) for value in winner["recipe_action_ids"])
        cold_key = (block.sample_id, recipe)
        if cold_key not in cold_keys:
            response = backend.replay(block, weight, contract, recipe, warmups=0,
                                      repetitions=1, capture_transitions=False)
            terminal = fit3._terminal(response)
            require(str(terminal.get("frame_sha256")) == winner["frame_sha256"],
                    "winner frame changed before cold decode")
            cold_receipts.append({"sample_id": block.sample_id,
                                  "dataset_id": block.dataset_id,
                                  "recipe_action_ids": list(recipe),
                                  **independent_cold_decode(
                                      cold_cli, str(terminal["frame_hex"]),
                                      block.request_roots, output,
                                      timeout=timeout, cpu=cpu)})
            cold_keys.add(cold_key)
        for transition in fit3.replay_transitions(backend, block, weight,
                                                  contract, recipe):
            records.append({
                "schema": TEACHER_SCHEMA, "split": "train",
                "sample_id": block.sample_id, "dataset_id": block.dataset_id,
                "domain": block.domain, "root_fingerprint": block.root_fingerprint,
                "weight": list(weight), "recipe_action_ids": list(recipe),
                "teacher_model_sha256": backend.model_sha256,
                "teacher_timing_scope":
                "cpp_total_forced_action_current_model_forward",
                "teacher_objective": winner["objective"],
                "teacher_metrics": {
                    "compression_ratio": winner["compression_ratio"],
                    "compression_ns_per_byte": winner["compression_ns_per_byte"],
                    "decompression_ns_per_byte":
                    winner["decompression_ns_per_byte"]},
                "offline_candidate": winner["candidate"],
                "winner_independent_cold_decode": True, **transition})
    return records, cold_receipts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-cli", type=Path, required=True)
    parser.add_argument("--cold-cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--max-train-blocks", type=int, default=66)
    parser.add_argument("--sample-id", action="append",
                        help="repeat to run an exact auditable train-only probe subset")
    parser.add_argument("--request-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--deadline-seconds", type=float, default=300.0)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--evaluate-validation", action="store_true")
    parser.add_argument("--reuse-candidate-time-weights", action="store_true",
                        help="measure candidate time at weight100 once; explicitly reuse it "
                             "as an approximate teacher cost for the other five weights")
    parser.add_argument("--require-quality-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require(len(args.data) == 2, "pass exact train and validation manifests")
    require(1 <= args.candidate_limit <= 128, "candidate limit must be in [1,128]")
    require(1 <= args.max_train_blocks <= 66, "v4 permits at most 66 train blocks")
    require(math.isfinite(args.deadline_seconds) and args.deadline_seconds > 0,
            "invalid deadline")
    for path in (*args.data, args.reference, args.server, args.catalog, args.profile,
                 args.model, args.baseline_cli, args.cold_cli):
        require(path.is_file(), f"required artifact absent: {path}")
    output = args.output.resolve()
    require(not output.exists(), "offline v4 output must be a new directory")
    output.mkdir(parents=True)

    contract = fit3.OnlineContract()
    blocks = fit3.load_training_data(args.data, reference_path=args.reference,
                                     materialized_root=args.materialized_root)
    all_train = tuple(block for block in blocks if block.split == "train")
    if args.sample_id:
        requested = tuple(str(value) for value in args.sample_id)
        require(len(set(requested)) == len(requested), "duplicate --sample-id")
        indexed = {block.sample_id: block for block in all_train}
        require(all(value in indexed for value in requested),
                "--sample-id is absent or is not an admitted train block")
        train = tuple(indexed[value] for value in requested)
        require(len(train) <= args.max_train_blocks,
                "exact probe subset exceeds --max-train-blocks")
        selection = {"mode": "explicit_sample_ids", "sample_ids": list(requested)}
    else:
        # Large blocks first makes a bounded prefix informative and avoids the
        # old manifest-order bug that selected only the first tiny timestamps.
        ordered = sorted(all_train,
                         key=lambda block: (-block.original_numeric_bytes,
                                            block.domain, block.dataset_id,
                                            block.sample_id))
        train = tuple(ordered[:args.max_train_blocks])
        selection = {"mode": "descending_B_raw_then_domain_dataset_id",
                     "sample_ids": [block.sample_id for block in train]}
    selected = train + (tuple(block for block in blocks if block.split == "validation")
                        if args.evaluate_validation else ())
    require(train and all(block.split == "train" for block in train),
            "offline teacher has no admitted train blocks")
    catalog = ExactCatalog(args.catalog, args.profile)
    plans = frozen_plans()[:args.candidate_limit]
    search_space = {
        "schema": SEARCH_SPACE_SCHEMA, "candidate_limit": args.candidate_limit,
        "materialized_candidate_count": len(plans), "max_prefix_depth": 3,
        "per_candidate_timeout_seconds": args.request_timeout_seconds,
        "wall_deadline_seconds": args.deadline_seconds,
        "weights": [list(value) for value in fit3.FIXED_WEIGHTS],
        "train_only_teacher": True,
        "train_selection": selection,
        "validation_evaluation_labels_written": False,
        "online_action_space_unchanged": True,
        "online_candidate_search": False,
        "excluded_implementation_ids": sorted(catalog.excluded_implementation_ids),
        "direct_baselines": ["LZ4", "Zstd"],
        "direct_baseline_timing_scope": "native_complete_method_no_mlp",
        "graph_candidate_timing_scope":
        "cpp_total_forced_action_current_model_forward",
        "reuse_candidate_time_weights": args.reuse_candidate_time_weights,
        "measured_candidate_timing_weights":
        ([list(fit3.FIXED_WEIGHTS[0])] if args.reuse_candidate_time_weights else
         [list(value) for value in fit3.FIXED_WEIGHTS]),
        "approximate_teacher_timing": args.reuse_candidate_time_weights,
        "plans": [asdict(plan) for plan in plans],
        "artifacts": {str(path.resolve()): sha256(path) for path in
                      (*args.data, args.reference, args.server, args.catalog,
                       args.profile, args.model, args.baseline_cli, args.cold_cli)}}
    atomic_json(output / "search_space.json", search_space)

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    block_map = {block.sample_id: block for block in train}
    with fit3.OnlineTrainingServer(
            args.server, args.catalog, args.profile, cpu=args.cpu,
            timeout_seconds=args.request_timeout_seconds) as backend:
        backend.load_model(args.model)
        for block in selected:
            if time.monotonic() - started >= args.deadline_seconds:
                rows.append({"schema": MEASUREMENT_SCHEMA,
                             "sample_id": block.sample_id, "dataset_id": block.dataset_id,
                             "domain": block.domain, "split": block.split,
                             "candidate": None, "status": "not_run_deadline"})
                continue
            block_rows: list[dict[str, Any]] = []
            recipe_cache: dict[tuple[int, ...], str] = {}
            for plan in plans:
                if time.monotonic() - started >= args.deadline_seconds:
                    block_rows.append({"schema": MEASUREMENT_SCHEMA,
                        "sample_id": block.sample_id, "dataset_id": block.dataset_id,
                        "domain": block.domain, "split": block.split,
                        "candidate": plan.name, "status": "not_run_deadline"})
                    continue
                try:
                    recipe = materialize_plan(backend, block, contract, catalog, plan)
                    if recipe in recipe_cache:
                        block_rows.append({"schema": MEASUREMENT_SCHEMA,
                            "sample_id": block.sample_id, "dataset_id": block.dataset_id,
                            "domain": block.domain, "split": block.split,
                            "candidate": plan.name, "status": "duplicate_recipe",
                            "duplicate_of": recipe_cache[recipe],
                            "recipe_action_ids": list(recipe),
                            "recipe_mechanisms": catalog.describe(recipe)})
                        continue
                    recipe_cache[recipe] = plan.name
                    measured_weights = ((fit3.FIXED_WEIGHTS[0],)
                                        if args.reuse_candidate_time_weights else
                                        fit3.FIXED_WEIGHTS)
                    measured_by_weight = {}
                    for weight in measured_weights:
                        measured_by_weight[tuple(weight)] = fit3.replay_measure(
                            backend, block, weight, contract, recipe,
                            warmups=args.warmups, repetitions=args.repetitions)
                    for weight in fit3.FIXED_WEIGHTS:
                        source_weight = (fit3.FIXED_WEIGHTS[0]
                                         if args.reuse_candidate_time_weights else weight)
                        block_rows.append(measurement_row(
                            plan, recipe, catalog,
                            measured_by_weight[tuple(source_weight)], block, weight,
                            measured_timing_weight=source_weight))
                except Exception as error:
                    block_rows.append({"schema": MEASUREMENT_SCHEMA,
                        "sample_id": block.sample_id, "dataset_id": block.dataset_id,
                        "domain": block.domain, "split": block.split,
                        "candidate": plan.name, "status": "failed",
                        "error": type(error).__name__ + ": " + str(error)})
            for weight in fit3.FIXED_WEIGHTS:
                weighted = [row for row in block_rows if row.get("status") == "ok" and
                            tuple(row.get("weight", ())) == tuple(weight)]
                frontier = pareto_indices(weighted)
                for index, row in enumerate(weighted):
                    row["pareto"] = index in frontier
                if weighted and block.split == "train":
                    winners.append(min(weighted, key=lambda row: (
                        float(row["objective"]), float(row["compression_ratio"]),
                        str(row["candidate"]))))
            raw = next((row for row in block_rows if row.get("status") == "ok" and
                        row.get("candidate") == "raw" and
                        tuple(row.get("weight", ())) == tuple(fit3.FIXED_WEIGHTS[0])), None)
            if raw is not None:
                baselines.append(_raw_baseline(raw))
            for baseline in direct_baselines(
                    block, args.baseline_cli, timeout=args.request_timeout_seconds,
                    cpu=args.cpu):
                baselines.append({"sample_id": block.sample_id,
                                  "dataset_id": block.dataset_id,
                                  "domain": block.domain,
                                  "original_numeric_bytes":
                                  block.original_numeric_bytes, **baseline})
            rows.extend(block_rows)
            atomic_jsonl(output / "candidate_measurements.jsonl", rows)
            atomic_jsonl(output / "direct_baselines.jsonl", baselines)
            atomic_json(output / "progress.json", {
                "schema": SCHEMA, "elapsed_seconds": time.monotonic() - started,
                "blocks_started": len({str(row["sample_id"]) for row in rows}),
                "measurement_records": len(rows),
                "successful_measurements": sum(row.get("status") == "ok" for row in rows)})

        teacher_records, cold_receipts = _teacher_rows(
            backend, contract, block_map, winners, cold_cli=args.cold_cli,
            output=output, timeout=args.request_timeout_seconds, cpu=args.cpu)
    atomic_jsonl(output / "teacher.jsonl", teacher_records)
    atomic_jsonl(output / "winner_cold_decode_receipts.jsonl", cold_receipts)
    teacher_receipt = {
        "schema": "native-online-offline-combo-teacher-receipt-v6",
        "teacher_sha256": sha256(output / "teacher.jsonl"),
        "records": len(teacher_records),
        "train_samples": len({row["sample_id"] for row in teacher_records}),
        "validation_records": sum(row.get("split") == "validation"
                                  for row in teacher_records),
        "unique_winner_frames_cold_decoded": len(cold_receipts),
        "winner_cold_decode_receipts_sha256":
        sha256(output / "winner_cold_decode_receipts.jsonl"),
        "contract": asdict(contract), "contract_sha256": contract.fingerprint,
        "search_space_sha256": sha256(output / "search_space.json")}
    atomic_json(output / "teacher.receipt.json", teacher_receipt)
    proof = quality_proof(rows, baselines)
    summary = {"schema": SCHEMA, "completed": True,
               "elapsed_seconds": time.monotonic() - started,
               "train_blocks_requested": len(train),
               "validation_evaluated": args.evaluate_validation,
               "candidate_records": len(rows),
               "candidate_successes": sum(row.get("status") == "ok" for row in rows),
               "candidate_failures": sum(row.get("status") == "failed" for row in rows),
               "candidate_deadline_skips": sum(row.get("status") ==
                                               "not_run_deadline" for row in rows),
               "teacher_records": len(teacher_records),
               "quality_gate": proof,
               "limitations": [
                   "Direct LZ4/Zstd frames use their frozen native baseline envelope; "
                   "graph candidates use complete NGF1 frames.",
                   "The finite candidate set is an approximate offline teacher, not a "
                   "global optimum.",
                   "Current online typed carriers intentionally make some apparent "
                   "chains illegal after a stage changes the logical view to bytes."]}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.require_quality_gate and not proof.get("passed"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
