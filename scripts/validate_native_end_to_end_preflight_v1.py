#!/usr/bin/env python3
"""Small real-data preflight for the complete native training/test path.

This creates and exports a randomly initialized policy, then exercises frozen
C++ search on one or two chronological validation blocks and all six weights.
It performs zero gradient updates and never admits formal training or reports
the random policy as model-effect evidence.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np

import calibrate_native_smallest32_v1 as calibration
from export_weighted_grpo_native_policy_v1 import export_checkpoint
import native_prepared_cohort_v1 as data
import native_experiment_scope_v1 as experiment_scope
from native_grpo_task_v1 import ACTION_COUNT, REGISTRY, NativeClient, NativeGraphTask
import weight_conditioned_grpo_v1 as grpo


SCHEMA = "native-end-to-end-preflight-v1"
MAX_VALIDATION_BLOCK_COUNT = 2
MAX_TIMING_WARMUPS = 1
MAX_TIMING_REPETITIONS = 3
MAX_REQUEST_SECONDS = 1800.0
MAX_NATIVE_ADDRESS_SPACE_BYTES = 64 << 30


def _manifest_path(path: Path) -> Path:
    result = Path(path).resolve()
    if result.is_dir():
        result = result / "dataset_manifest.json"
    data.require(result.is_file(), f"missing prepared manifest: {result}")
    return result


def _load_prepared(paths: list[Path]) -> tuple[list[data.Block], list[dict]]:
    data.require(bool(paths), "at least one --prepared manifest is required")
    blocks: list[data.Block] = []
    identities: list[dict] = []
    seen_paths: set[Path] = set()
    seen_datasets: set[str] = set()
    for supplied in paths:
        path = _manifest_path(supplied)
        data.require(path not in seen_paths, f"duplicate prepared manifest: {path}")
        seen_paths.add(path)
        raw = path.read_bytes()
        document = json.loads(raw)
        cohort = document.get("cohort", {})
        data.require(
            cohort.get("role") == "train_validation"
            and cohort.get("version") == data.gate.VERSION,
            "preflight accepts only frozen TRAIN_VALIDATION manifests",
        )
        dataset_id = document.get("dataset_id")
        domain = document.get("domain")
        provenance = document.get("provenance_sha256")
        rows = document.get("blocks")
        data.require(
            isinstance(dataset_id, str) and dataset_id
            and isinstance(domain, str) and domain
            and isinstance(provenance, str) and len(provenance) == 64
            and isinstance(rows, list) and rows,
            "prepared manifest identity is incomplete",
        )
        data.require(dataset_id not in seen_datasets, "duplicate prepared dataset_id")
        seen_datasets.add(dataset_id)
        data.require(
            {row.get("split") for row in rows} == {"train", "validation"},
            "prepared manifest must contain only train and validation blocks",
        )
        manifest_sha = data.digest(raw)
        for row in rows:
            blocks.append(data.Block(
                dataset_id, domain, path, manifest_sha, provenance,
                "train_validation", row,
            ))
        identities.append({
            "path": str(path), "dataset_id": dataset_id, "domain": domain,
            "manifest_sha256": manifest_sha, "provenance_sha256": provenance,
            "block_count": len(rows),
        })
    data.require(len({block.key for block in blocks}) == len(blocks),
                 "duplicate prepared block identity")
    return sorted(blocks, key=lambda block: block.key), identities


def _select_sequence(blocks: list[data.Block]):
    grouped: dict[tuple[str, str, str], list[data.Block]] = collections.defaultdict(list)
    for block in blocks:
        grouped[(block.dataset_id, block.record["entity_id"], block.record["column"])].append(block)
    candidates = []
    for key, members in grouped.items():
        train = sorted(
            (block for block in members
             if block.split == "train" and block.record["original_numeric_bytes"] > 0),
            key=lambda block: (block.record["block_index"], block.record["block_id"]),
        )
        validation = sorted(
            (block for block in members
             if block.split == "validation" and block.record["original_numeric_bytes"] > 0),
            key=lambda block: (block.record["block_index"], block.record["block_id"]),
        )
        if train and validation:
            candidates.append((key, train[0], validation[:MAX_VALIDATION_BLOCK_COUNT]))
    data.require(bool(candidates),
                 "no source/entity/column has train and nonempty validation blocks")
    _key, train, validation = sorted(candidates, key=lambda item: item[0])[0]
    dtype = train.record["dtype"]
    data.require(all(block.record["dtype"] == dtype for block in validation),
                 "selected chronological sequence changed dtype")
    # Reading the roots rechecks payload hashes, row counts, bitmap semantics,
    # byte order, and the B_raw denominator before any native process starts.
    train.roots()
    for block in validation:
        block.roots()
    return train, validation


def _load_reference(path: Path) -> dict:
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    unsigned = {key: value for key, value in document.items()
                if key != "calibration_sha256"}
    data.require(
        document.get("calibration_sha256") == data.digest(data.canonical(unsigned)),
        "calibration content checksum mismatch",
    )
    data.require(
        document.get("schema") == calibration.SCHEMA
        and document.get("lookup_rule") == calibration.LOOKUP
        and document.get("recipe") == calibration.RECIPE,
        "calibration method/lookup identity mismatch",
    )
    data.require(
        document.get("test_used_for_training_statistics") is False
        and document.get("validation_used_for_time_statistics") is False,
        "calibration contains validation/test statistics",
    )
    return document


def _load_library(path: Path, catalog: Path, reference: dict, train: data.Block) -> tuple[dict, list[dict]]:
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    unsigned = {key: value for key, value in document.items()
                if key != "library_content_sha256"}
    data.require(
        document.get("library_content_sha256") == data.digest(data.canonical(unsigned)),
        "training library content checksum mismatch",
    )
    data.require(
        document.get("schema") == "weighted-grpo-training-solution-library-v1"
        and document.get("library_format") == "native-ngf1-smallest32-v1"
        and document.get("source_scope") == "training_only"
        and document.get("action_registry_fingerprint") == REGISTRY,
        "preflight requires the frozen training-only native library",
    )
    data.require(document.get("native_catalog_sha256") == data.digest(Path(catalog).read_bytes()),
                 "training library/catalog mismatch")
    data.require(document.get("reference_sha256") == reference["calibration_sha256"],
                 "training library/calibration mismatch")
    data.require(document.get("corpus_fingerprint") == reference.get("corpus_fingerprint"),
                 "training library/calibration corpus mismatch")
    recipes = [row for row in document.get("recipes", [])
               if row.get("dtype") == train.record["dtype"]]
    data.require(0 < len(recipes) <= 64,
                 "selected dtype needs 1..64 training-library candidates")
    recipe_keys = [tuple(row.get("recipe_action_ids", ())) for row in recipes]
    data.require(len(recipe_keys) == len(set(recipe_keys))
                 and all(key and all(type(action) is int and 0 < action < ACTION_COUNT
                                     for action in key) for key in recipe_keys),
                 "training library has duplicate or invalid recipes")
    data.require(all(row.get("source_split") == "train" for row in recipes),
                 "validation/test recipe entered the training library")
    source_blocks = {
        fingerprint for row in recipes
        for fingerprint in row.get("source_block_fingerprints", [])
    }
    data.require(train.fingerprint in source_blocks,
                 "selected train block is absent from the frozen library evidence")
    return document, recipes


def _cold_decode(cli: Path, cpu: int, frame: bytes, *, allow_large_memory: bool,
                 timeout_seconds: float):
    with tempfile.TemporaryDirectory(prefix="native-preflight-cold-") as directory:
        request = Path(directory) / "request.json"
        request.write_bytes(data.canonical({
            "schema": "native-graph-request-v1", "frame_hex": frame.hex(),
            "allow_large_memory": allow_large_memory,
        }))
        started = time.monotonic_ns()
        result = subprocess.run(
            ["taskset", "-c", str(cpu), "prlimit",
             "--as=" + str(MAX_NATIVE_ADDRESS_SPACE_BYTES), "--",
             str(Path(cli).resolve()), "decode", str(request)],
            capture_output=True, check=False, timeout=timeout_seconds,
        )
        wall_ns = time.monotonic_ns() - started
    data.require(result.returncode == 0,
                 "native cold decode failed: " + result.stderr.decode(errors="replace")[:2000])
    answer = json.loads(result.stdout)
    data.require(answer.get("schema") == "native-graph-response-v1",
                 "wrong native cold-decode response schema")
    return answer, wall_ns


def _reference(reference: dict, block: data.Block) -> dict:
    return calibration.reference_for(
        reference, block.record["dtype"], block.record["prepared_block_bytes"]
    )


def _catalog_gate(response: dict, catalog_sha256: str, expected_scope: dict) -> None:
    data.require(
        response["complete_native_graph_space"] is True
        and response["unsupported_graph_action_count"] == 0
        and response["native_executable_action_count"] == ACTION_COUNT
        and response["action_count"] == ACTION_COUNT
        and response["registry_fingerprint"] == REGISTRY
        and response["catalog_file_sha256"] == catalog_sha256
        and response["experiment_scope"] == expected_scope,
        "preflight requires the complete frozen 6,341-action native catalog",
    )


def _print_pair_progress(row: dict, weight_index: int, phase: str) -> None:
    print(json.dumps({
        "event": "native_preflight_pair_complete",
        "phase": phase,
        "weight_index": weight_index,
        "block_id": row["block_id"],
        "block_fingerprint": row["block_fingerprint"],
        "driver_wall_ns": row["pair_wall_ns_for_budget_estimate_only"],
        "native_E_total_ns": row["raw_cpp_timing_ns"]["total_compression_ns"],
    }, sort_keys=True), flush=True)


def _check_search(result: dict, *, block: data.Block, ordinal: int,
                  library_candidate_count: int) -> bytes:
    budget = 64 if ordinal == 0 else 16
    data.require(
        result.get("search_status") == "ok"
        and result.get("terminal_status") == "ok",
        "random-policy preflight did not find a complete graph",
    )
    data.require(
        result["evaluation_budget"] == budget
        and 1 <= result["evaluations"] <= budget
        and len(result["candidates"]) == result["evaluations"],
        "native complete-candidate budget accounting changed",
    )
    if ordinal == 0:
        data.require(result["evaluations"] >= library_candidate_count,
                     "not every training-library start was charged within the 64 budget")
    timing = result["timing"]
    components = (
        "task_initialization_ns", "feature_ns", "policy_ns", "action_ordering_ns", "transition_ns",
        "candidate_evaluation_ns", "certified_pruning_ns",
        "total_prediction_ns", "final_encode_ns", "total_compression_ns",
    )
    data.require(all(type(timing[name]) is int and timing[name] >= 0 for name in components),
                 "native search returned an invalid timing component")
    data.require(
        timing["total_prediction_ns"] >= sum(timing[name] for name in (
            "task_initialization_ns", "feature_ns", "policy_ns", "action_ordering_ns", "transition_ns",
            "candidate_evaluation_ns", "certified_pruning_ns",
        ))
        and timing["total_compression_ns"] >= (
            timing["total_prediction_ns"] + timing["final_encode_ns"]
        )
        and result["search_mode"] == "policy"
        and result["enable_certified_pruning"] is True
        and result["enable_objective_guided_ordering"] is True
        and result["pruning_enabled"] is True
        and result["certified_pruned_states"] >= 0,
        "native E_total interval or search-mode/pruning contract changed",
    )
    raw = block.record["original_numeric_bytes"]
    frame = bytes.fromhex(result["frame_hex"])
    data.require(
        result["original_numeric_bytes"] == raw
        and result["frame_bytes"] == len(frame)
        and result["frame_sha256"] == data.digest(frame)
        and math.isclose(result["compression_ratio"], len(frame) / raw,
                         rel_tol=0.0, abs_tol=0.0),
        "native final frame or B_raw denominator changed",
    )
    return frame


def _search_pair(client: NativeClient, block: data.Block, *, weight, ordinal: int,
                 previous_recipe, reference: dict, library_candidate_count: int,
                 cold_cli: Path, cpu: int, timing_warmups: int,
                 timing_repetitions: int, allow_large_memory: bool,
                 timeout_seconds: float) -> tuple[dict, tuple[int, ...]]:
    scale = _reference(reference, block)
    pair_started = time.monotonic_ns()
    with NativeGraphTask(
        client, block, weights=weight, E0=scale["E0"], D0=scale["D0"],
        ordinal=ordinal, previous_recipe=previous_recipe,
        require_complete=True, warmups=timing_warmups,
        repeats=timing_repetitions, allow_large_memory=allow_large_memory,
    ) as task:
        search_started = time.monotonic_ns()
        result = client.call({
            "cmd": "search", "task_handle": task.task_handle,
            "beam_width": 30, "include_frame_hex": True,
            "use_solution_library": ordinal == 0,
            "dtype": block.record["dtype"],
        })
        search_wall_ns = time.monotonic_ns() - search_started
    frame = _check_search(
        result, block=block, ordinal=ordinal,
        library_candidate_count=library_candidate_count,
    )
    cold, cold_wall_ns = _cold_decode(
        cold_cli, cpu, frame, allow_large_memory=allow_large_memory,
        timeout_seconds=timeout_seconds,
    )
    data.require(
        cold["roots"] == block.roots()
        and cold["original_numeric_bytes"] == block.record["original_numeric_bytes"],
        "independent cold decode changed the original roots",
    )
    samples = result["decode_samples_ns"]
    data.require(
        len(samples) == timing_repetitions
        and all(type(sample) is int and sample >= 0 for sample in samples),
        "native decode samples changed shape or domain",
    )
    pair_wall_ns = time.monotonic_ns() - pair_started
    timing = result["timing"]
    row = {
        **block.identity(), "weight": list(weight),
        "series_block_ordinal": ordinal,
        "previous_recipe_action_ids": list(previous_recipe),
        "best_recipe_action_ids": result["best_recipe_action_ids"],
        "best_search_action_ids": result["best_search_action_ids"],
        "evaluation_budget": result["evaluation_budget"],
        "evaluations": result["evaluations"],
        "library_candidate_count": library_candidate_count if ordinal == 0 else 0,
        "library_candidates_charged": ordinal != 0 or result["evaluations"] >= library_candidate_count,
        "candidate_status_counts": dict(collections.Counter(
            candidate["terminal_status"] for candidate in result["candidates"]
        )),
        "budget_unexplored_actions": result["budget_unexplored_actions"],
        "duplicate_states_skipped": result["duplicate_states_skipped"],
        "frame_bytes": len(frame), "frame_sha256": data.digest(frame),
        "compression_ratio": len(frame) / block.record["original_numeric_bytes"],
        "J_selection": result["J_selection"],
        "raw_cpp_timing_ns": timing,
        "cpp_total_compression_ns_per_B_raw": (
            timing["total_compression_ns"] / block.record["original_numeric_bytes"]
        ),
        "cpp_decode_samples_ns": samples,
        "cpp_median_decompression_ns_per_B_raw": (
            statistics.median(samples) / block.record["original_numeric_bytes"]
        ),
        "search_call_wall_ns_including_ipc": search_wall_ns,
        "cold_decode_process_wall_ns": cold_wall_ns,
        "pair_wall_ns_for_budget_estimate_only": pair_wall_ns,
        "cold_decode_verified": True,
    }
    return row, tuple(result["best_recipe_action_ids"])


def _continuation_probe(client: NativeClient, block: data.Block, previous_recipe,
                        reference: dict, cold_cli: Path, cpu: int,
                        timing_warmups: int, timing_repetitions: int,
                        allow_large_memory: bool, timeout_seconds: float) -> dict:
    weight = grpo.FIXED_WEIGHTS[-1]
    scale = _reference(reference, block)
    with NativeGraphTask(
        client, block, weights=weight, E0=scale["E0"], D0=scale["D0"],
        ordinal=1, previous_recipe=previous_recipe, require_complete=True,
        warmups=timing_warmups, repeats=timing_repetitions,
        allow_large_memory=allow_large_memory,
    ) as task:
        before = task.initial_state()
        before_features = task.state_features(before).tolist()
        data.require(before.payload["candidate_used"] == 0
                     and before.payload["candidate_budget"] == 16,
                     "continuation probe did not start with the 16-candidate budget")
        started = time.monotonic_ns()
        outcome = task.evaluate_terminal(before)
        evaluation_wall_ns = time.monotonic_ns() - started
        inspect_started = time.monotonic_ns()
        task.refresh_initial()
        inspect_wall_ns = time.monotonic_ns() - inspect_started
        after = task.initial_state()
        after_features = task.state_features(after).tolist()
        data.require(
            after.handle == before.handle
            and after.payload["candidate_used"] == 1
            and after.payload["candidate_budget"] == 16
            and math.isclose(before_features[9], 0.0, abs_tol=0.0)
            and math.isclose(after_features[9], 1.0 / 16.0,
                             rel_tol=0.0, abs_tol=1.0e-15),
            "prior evaluation was not exposed by immutable inspect features",
        )
        result = {
            "block": block.identity(), "weight": list(weight),
            "previous_recipe_action_ids": list(previous_recipe),
            "task_handle_reused": True, "immutable_state_handle_reused": True,
            "candidate_budget": 16, "candidate_used_before": 0,
            "candidate_used_after": 1,
            "candidate_fraction_feature_index": 9,
            "candidate_fraction_before": before_features[9],
            "candidate_fraction_after": after_features[9],
            "objective_feature_index": 15,
            "objective_feature_before": before_features[15],
            "objective_feature_after": after_features[15],
            "prior_evaluation_wall_ns_for_budget_estimate_only": evaluation_wall_ns,
            "inspect_refresh_wall_ns_for_budget_estimate_only": inspect_wall_ns,
            "legal_action_count_after_inspect": len(task.legal_actions(after)),
        }
        if isinstance(outcome, grpo.TerminalFailure):
            result.update({
                "terminal_status": "failed", "error": outcome.error,
                "original_numeric_bytes": outcome.original_numeric_bytes,
                "max_frame_bytes": outcome.max_frame_bytes,
                "cold_decode_verified": False,
            })
        else:
            cold, cold_wall_ns = _cold_decode(
                cold_cli, cpu, outcome.compressed_payload,
                allow_large_memory=allow_large_memory,
                timeout_seconds=timeout_seconds,
            )
            data.require(cold["roots"] == block.roots(),
                         "continuation prior evaluation failed cold decode")
            result.update({
                "terminal_status": "ok",
                "frame_bytes": len(outcome.compressed_payload),
                "frame_sha256": data.digest(outcome.compressed_payload),
                "compression_ratio": outcome.compression_ratio,
                "cold_decode_process_wall_ns": cold_wall_ns,
                "cold_decode_verified": True,
            })
        return result


def run(arguments) -> dict:
    data.require(os.name == "posix" and hasattr(os, "sched_getaffinity"),
                 "preflight requires Linux CPU affinity")
    available = os.sched_getaffinity(0)
    data.require(arguments.cpu in available, "requested --cpu is outside process affinity")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        data.require(os.environ.get(name) == "1", name + " must equal 1")
    data.require(0 <= arguments.timing_warmups <= MAX_TIMING_WARMUPS,
                 "preflight timing warmups must stay within 0..1")
    data.require(1 <= arguments.timing_repetitions <= MAX_TIMING_REPETITIONS,
                 "preflight timing repetitions must stay within 1..3")
    output = arguments.output.resolve()
    data.require(not output.exists(), "preflight output already exists; use a new directory")

    run_started = time.monotonic_ns()
    blocks, prepared_identity = _load_prepared(arguments.prepared)
    train, validation = _select_sequence(blocks)
    reference = _load_reference(arguments.reference)
    library, dtype_recipes = _load_library(
        arguments.library, arguments.catalog, reference, train
    )
    catalog_sha = data.digest(arguments.catalog.read_bytes())
    expected_scope = experiment_scope.load(arguments.catalog,
                                             arguments.experiment_profile)
    runtime_identity = {
        "native_server": {"path": str(arguments.cli.resolve()),
                          "sha256": data.digest(arguments.cli.read_bytes())},
        "catalog": {"path": str(arguments.catalog.resolve()), "sha256": catalog_sha},
        "experiment_scope": expected_scope,
        "cold_decode_cli": {"path": str(arguments.cold_cli.resolve()),
                            "sha256": data.digest(arguments.cold_cli.read_bytes())},
        "training_library": {
            "path": str(arguments.library.resolve()),
            "file_sha256": data.digest(arguments.library.read_bytes()),
            "content_sha256": library["library_content_sha256"],
            "corpus_fingerprint": library["corpus_fingerprint"],
            "matching_dtype_recipe_count": len(dtype_recipes),
        },
        "reference": {"path": str(arguments.reference.resolve()),
                      "sha256": data.digest(arguments.reference.read_bytes()),
                      "calibration_sha256": reference["calibration_sha256"]},
    }
    output.mkdir(parents=True)

    checkpoint = output / "random_initialized_policy.json"
    model = output / "random_initialized_policy.wgpn"
    with NativeClient(
        arguments.cli, arguments.catalog, library=arguments.library,
        cpu=arguments.cpu, timeout=arguments.timeout_seconds,
        profile=arguments.experiment_profile,
    ) as client:
        catalog_status = client.call({"cmd": "catalog"})
        _catalog_gate(catalog_status, catalog_sha, expected_scope)
        scale = _reference(reference, train)
        with NativeGraphTask(
            client, train, weights=(0.4, 0.3, 0.3),
            E0=scale["E0"], D0=scale["D0"], require_complete=True,
            warmups=arguments.timing_warmups,
            repeats=arguments.timing_repetitions,
            allow_large_memory=arguments.allow_large_memory,
        ) as task:
            identity = grpo._inspect_tasks([task], (128, 128))
    data.require(
        identity.task_feature_dim == 18 and identity.state_feature_dim == 28
        and identity.input_dim == 49 and identity.hidden_sizes == (128, 128)
        and identity.action_count == ACTION_COUNT
        and identity.registry_fingerprint == REGISTRY,
        "real native task did not produce the frozen 49D policy identity",
    )
    policy = grpo.TwoHiddenMLP(identity, np.random.default_rng(arguments.seed))
    checkpoint_payload_sha = grpo.save_checkpoint(
        grpo.TrainingResult(policy, (), grpo.GRPOConfig(), arguments.seed, 0),
        checkpoint,
        extra={
            "schema": SCHEMA, "purpose": "end_to_end_preflight_only",
            "initialization": "random_seeded_two_by_128_no_gradient",
            "policy_untrained": True, "gradient_updates": 0,
            "formal_training_admitted": False,
        },
    )
    export_receipt = export_checkpoint(checkpoint, model)
    data.require(export_receipt["identity"] == identity.to_json(),
                 "native export changed the policy identity")
    runtime_identity["random_checkpoint"] = {
        "path": str(checkpoint), "file_sha256": data.digest(checkpoint.read_bytes()),
        "payload_sha256": checkpoint_payload_sha,
    }
    runtime_identity["random_native_model"] = {
        "path": str(model), "file_sha256": data.digest(model.read_bytes()),
        "export_receipt": export_receipt,
    }

    pairs = []
    with NativeClient(
        arguments.cli, arguments.catalog, model=model, library=arguments.library,
        cpu=arguments.cpu, timeout=arguments.timeout_seconds,
        profile=arguments.experiment_profile,
    ) as client:
        schema = client.call({"cmd": "schema"})
        data.require(schema["model_loaded"] is True
                     and schema["solution_library_loaded"] is True
                     and schema["solution_library_pinned"] is True,
                     "frozen native server did not pin model and training library")
        _catalog_gate(schema["catalog"], catalog_sha, expected_scope)
        for weight_index, weight in enumerate(grpo.FIXED_WEIGHTS):
            previous: tuple[int, ...] = ()
            for ordinal, block in enumerate(validation):
                row, previous = _search_pair(
                    client, block, weight=weight, ordinal=ordinal,
                    previous_recipe=previous, reference=reference,
                    library_candidate_count=len(dtype_recipes),
                    cold_cli=arguments.cold_cli, cpu=arguments.cpu,
                    timing_warmups=arguments.timing_warmups,
                    timing_repetitions=arguments.timing_repetitions,
                    allow_large_memory=arguments.allow_large_memory,
                    timeout_seconds=arguments.timeout_seconds,
                )
                row["weight_index"] = weight_index
                pairs.append(row)
                _print_pair_progress(row, weight_index, "validation")
        # Independently exercise the chronological train -> validation API path.
        # This is protocol validation only; train data is never scored as model
        # effect and validation remains excluded from the random policy identity.
        probe_weight = grpo.FIXED_WEIGHTS[-1]
        bootstrap, probe_previous_recipe = _search_pair(
            client, train, weight=probe_weight, ordinal=0, previous_recipe=(),
            reference=reference, library_candidate_count=len(dtype_recipes),
            cold_cli=arguments.cold_cli, cpu=arguments.cpu,
            timing_warmups=arguments.timing_warmups,
            timing_repetitions=arguments.timing_repetitions,
            allow_large_memory=arguments.allow_large_memory,
            timeout_seconds=arguments.timeout_seconds,
        )
        bootstrap["weight_index"] = len(grpo.FIXED_WEIGHTS) - 1
        _print_pair_progress(bootstrap, bootstrap["weight_index"],
                             "continuation_bootstrap_train")
        continuation_search, _probe_selected_recipe = _search_pair(
            client, validation[0], weight=probe_weight, ordinal=1,
            previous_recipe=probe_previous_recipe, reference=reference,
            library_candidate_count=len(dtype_recipes),
            cold_cli=arguments.cold_cli, cpu=arguments.cpu,
            timing_warmups=arguments.timing_warmups,
            timing_repetitions=arguments.timing_repetitions,
            allow_large_memory=arguments.allow_large_memory,
            timeout_seconds=arguments.timeout_seconds,
        )
        continuation_search["weight_index"] = len(grpo.FIXED_WEIGHTS) - 1
        _print_pair_progress(continuation_search,
                             continuation_search["weight_index"],
                             "continuation_search_validation")
        continuation = _continuation_probe(
            client, validation[0], probe_previous_recipe, reference,
            arguments.cold_cli, arguments.cpu,
            arguments.timing_warmups, arguments.timing_repetitions,
            arguments.allow_large_memory,
            arguments.timeout_seconds,
        )

    expected_pairs = len(validation) * len(grpo.FIXED_WEIGHTS)
    data.require(len(pairs) == expected_pairs
                 and len({(row["block_fingerprint"], row["weight_index"])
                          for row in pairs}) == expected_pairs,
                 "preflight validation block/weight coverage changed")
    pair_wall = [row["pair_wall_ns_for_budget_estimate_only"] for row in pairs]
    all_validation_nonempty = sum(
        block.split == "validation" and block.record["original_numeric_bytes"] > 0
        for block in blocks
    )
    median_pair_wall = statistics.median(pair_wall)
    receipt = {
        "schema": SCHEMA, "passed": True,
        "scope": "real-data launch preflight; random policy; no formal model-effect claim",
        "formal_training_admitted": False, "gradient_updates": 0,
        "policy_untrained": True, "formal_model_effect_evidence": False,
        "test_data_opened": False, "validation_only_after_training_identity_block": True,
        "policy_initialization": {
            "seed": arguments.seed, "hidden_sizes": [128, 128],
            "kind": "random_seeded_initialization_without_optimizer_or_gradient",
            "identity": identity.to_json(), "observation_width": identity.input_dim,
        },
        "prepared_manifests": prepared_identity,
        "selected_train_identity_block": train.identity(),
        "selected_validation_sequence": [block.identity() for block in validation],
        "runtime_identity": runtime_identity, "catalog_gate": catalog_status,
        "cpu": arguments.cpu, "timing_warmups": arguments.timing_warmups,
        "timing_repetitions": arguments.timing_repetitions,
        "allow_large_memory": arguments.allow_large_memory,
        "native_process_limits": {
            "cpu_affinity": [arguments.cpu],
            "address_space_bytes": MAX_NATIVE_ADDRESS_SPACE_BYTES,
            "request_and_cold_decode_timeout_seconds": arguments.timeout_seconds,
        },
        "fixed_weight_count": len(grpo.FIXED_WEIGHTS),
        "measured_validation_pair_count": len(pairs),
        "measurements": pairs,
        "chronological_continuation_search_probe": {
            "scope": "train bootstrap plus validation continuation; protocol evidence only",
            "bootstrap_first_block": bootstrap,
            "continuation_block": continuation_search,
        },
        "continuation_prior_evaluate_inspect": continuation,
        "budget_estimate": {
            "scope": "driver wall-clock extrapolation for launch planning only",
            "includes": "task reset, JSONL IPC, C++ search, task release, independent cold decode",
            "excludes": "training gradient updates and full-cohort effects",
            "sample_pair_count": len(pair_wall),
            "sample_total_wall_ns": sum(pair_wall),
            "sample_median_pair_wall_ns": median_pair_wall,
            "sample_max_pair_wall_ns": max(pair_wall),
            "provided_nonempty_validation_block_count": all_validation_nonempty,
            "projected_provided_validation_six_weight_wall_ns": int(
                median_pair_wall * all_validation_nonempty * len(grpo.FIXED_WEIGHTS)
            ),
            "latency_risk": (
                "GraphTask may replay expensive recipes while constructing features, masks, "
                "and transitions; this random-policy one-source sample is a launch estimate, "
                "not an upper bound for CMIX/PAQ/NNCP-heavy full-cohort training."
            ),
        },
        "run_wall_ns": time.monotonic_ns() - run_started,
    }
    receipt["receipt_sha256"] = data.digest(data.canonical(receipt))
    calibration.write_new(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, action="append", required=True,
                        help="Frozen TRAIN_VALIDATION dataset_manifest.json; repeatable")
    for name in ("reference", "cli", "catalog", "library", "cold-cli", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment-profile", type=Path,
                        default=experiment_scope.DEFAULT_PROFILE)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--timing-warmups", type=int, default=0)
    parser.add_argument("--timing-repetitions", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--allow-large-memory", action=argparse.BooleanOptionalAction,
                        default=True)
    arguments = parser.parse_args()
    data.require(arguments.cpu >= 0 and math.isfinite(arguments.timeout_seconds)
                 and 0 < arguments.timeout_seconds <= MAX_REQUEST_SECONDS,
                 "invalid CPU or timeout; preflight timeout must be within 1800 seconds")
    receipt = run(arguments)
    print(json.dumps({
        "passed": receipt["passed"],
        "formal_training_admitted": receipt["formal_training_admitted"],
        "gradient_updates": receipt["gradient_updates"],
        "policy_untrained": receipt["policy_untrained"],
        "measurement_pairs": receipt["measured_validation_pair_count"],
        "output": str(arguments.output.resolve() / "receipt.json"),
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
