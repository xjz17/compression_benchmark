#!/usr/bin/env python3
"""Protocol/contract smoke test for the native pipeline JSONL server."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


REGISTRY = "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def synthetic_native_model(path: Path) -> None:
    """Write an identity-correct synthetic model with action 91 ranked first."""
    def u32(value: int) -> bytes:
        return struct.pack("<I", value)

    def text(value: str) -> bytes:
        encoded = value.encode()
        return u32(len(encoded)) + encoded

    def pairs(value: tuple[tuple[str, str], ...]) -> bytes:
        return u32(len(value)) + b"".join(text(k) + text(v) for k, v in value)

    body = b"".join(text(value) for value in (
        "weight-conditioned-grpo-policy-v1", "a" * 64, "b" * 64,
        REGISTRY, "task-state-plus-simplex-weights-v1",
    ))
    body += b"".join(u32(value) for value in (6341, 0, 18, 28, 128, 128))
    body += pairs((("synthetic", "data"),)) + pairs((("synthetic", "root"),))
    shapes = ((49, 128), (128,), (128, 128), (128,), (128, 6341), (6341,))
    body += u32(len(shapes))
    for index, shape in enumerate(shapes):
        count = 1
        for dimension in shape:
            count *= dimension
        body += u32(len(shape)) + b"".join(u32(value) for value in shape)
        body += struct.pack("<Q", count)
        if index == 5:
            values = [-100.0] * count
            values[0] = 0.0
            values[91] = 100.0
            body += struct.pack("<" + "d" * count, *values)
        else:
            body += bytes(8 * count)
    content = b"WGPNAT1\0" + struct.pack("<Q", len(body)) + body
    path.write_bytes(content + hashlib.sha256(content).digest())


class Client:
    def __init__(self, argv: list[str]):
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
        self.serial = 0

    def call(self, value: dict, *, succeeds: bool = True) -> dict:
        self.serial += 1
        value = dict(value, request_id=self.serial)
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        answer = json.loads(self.process.stdout.readline())
        assert answer["request_id"] == self.serial
        assert answer["ok"] is succeeds, answer
        return answer

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)
        assert self.process.returncode in (-15, -9, 0)


def base_request() -> dict:
    raw = struct.pack("<8q", *range(8))
    return {
        "task_id": "native-server-contract-smoke",
        "roots": [{
            "kind": "numeric", "width": 8, "data_hex": raw.hex(),
            "logical_view": "int64", "original_numeric_payload": True,
        }],
        "weights": [1.0, 0.0, 0.0],
        "compression_reference_ns_per_byte": 1.0,
        "decompression_reference_ns_per_byte": 1.0,
        "original_numeric_bytes": len(raw),
        "timing_warmups": 0,
        "timing_repetitions": 1,
        "require_complete_native_catalog": False,
    }


def validate_state(value: dict, *, budget: int, max_steps: int) -> None:
    assert value["candidate_budget"] == budget
    assert value["max_steps"] == max_steps
    assert len(value["task_features"]) == 18
    assert len(value["state_features"]) == 28
    legal = value["legal_action_ids"]
    assert legal == sorted(set(legal)) and legal[0] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--experiment-profile", type=Path, required=True)
    parser.add_argument("--graph-cli", type=Path, required=True)
    args = parser.parse_args()
    server_argv = [
        str(args.server.resolve()), "--catalog", str(args.catalog.resolve()),
        "--experiment-profile", str(args.experiment_profile.resolve()), "serve",
    ]
    client = Client(server_argv)
    try:
        schema = client.call({"cmd": "schema"})
        assert schema["schema"] == "native-pipeline-v1-jsonl"
        assert schema["catalog"]["action_count"] == 6341
        assert schema["catalog"]["registry_fingerprint"] == REGISTRY
        scope = schema["catalog"]["experiment_scope"]
        assert scope["schema"] == "native-experiment-action-scope-v1"
        assert scope["profile_path"] == str(args.experiment_profile.resolve())
        assert scope["profile_sha256"] == hashlib.sha256(
            args.experiment_profile.read_bytes()).hexdigest()
        assert scope["catalog_sha256"] == hashlib.sha256(
            args.catalog.read_bytes()).hexdigest()
        assert set(scope) == {
            "schema", "profile_path", "profile_sha256", "catalog_sha256",
            "excluded_action_ids", "excluded_action_count", "enabled_action_count",
            "excluded_fixed_configuration_count", "enabled_fixed_configuration_count",
        }
        assert scope["excluded_action_count"] == 64
        assert scope["enabled_action_count"] == 6277
        assert scope["excluded_fixed_configuration_count"] == 2
        assert scope["enabled_fixed_configuration_count"] == 191
        assert len(scope["excluded_action_ids"]) == 64
        assert schema["search_modes"] == ["policy", "beam30"]
        assert schema["search_defaults"] == {
            "search_mode": "policy", "beam_width": 30,
            "enable_certified_pruning": True,
            "enable_objective_guided_ordering": True,
        }
        catalog = client.call({"cmd": "catalog"})
        assert catalog["experiment_scope"] == scope
        assert len(catalog["execution_mapping_sha256"]) == 64
        assert catalog["execution_mappings"]["schema"] == "native-pipeline-v1-execution-mappings"
        blocked = catalog["unsupported_graph_actions"]
        assert len(blocked) == catalog["unsupported_graph_action_count"]
        assert catalog["complete_native_graph_space"] is (not blocked)
        assert all(set(row) == {"action_id", "key", "runtime_action", "category", "reason"}
                   for row in blocked)

        reset = client.call(dict(base_request(), cmd="reset", block_index=0,
                                 continuation_edit_mode=False))
        validate_state(reset, budget=64, max_steps=65)
        assert not (set(reset["legal_action_ids"]) & set(scope["excluded_action_ids"]))
        excluded_previous = client.call(dict(
            base_request(), cmd="reset", block_index=1,
            continuation_edit_mode=True,
            previous_recipe_action_ids=[scope["excluded_action_ids"][0]]),
            succeeds=False)
        assert "excluded by the experiment scope" in excluded_previous["error"]
        initial_handle = reset["state_handle"]
        terminal = client.call({"cmd": "evaluate_terminal", "state_handle": initial_handle,
                                "include_frame_hex": True})
        assert terminal["recipe_action_ids"] == []
        assert terminal["candidate_ordinal"] == 1 and terminal["candidate_budget"] == 64
        assert terminal["frame_bytes"] * 2 == len(terminal["frame_hex"])
        assert terminal["frame_sha256"] == hashlib.sha256(bytes.fromhex(terminal["frame_hex"])).hexdigest()
        assert len(terminal["encode_samples_ns"]) == len(terminal["decode_samples_ns"]) == 1
        refreshed = client.call({"cmd": "inspect", "state_handle": initial_handle})
        assert refreshed["state_handle"] == initial_handle and refreshed["candidate_used"] == 1
        assert refreshed["state_features"] != reset["state_features"]

        with tempfile.NamedTemporaryFile("w", suffix=".json") as request_file:
            json.dump({"schema": "native-graph-request-v1",
                       "frame_hex": terminal["frame_hex"]}, request_file)
            request_file.flush()
            cold = subprocess.run([str(args.graph_cli.resolve()), "decode", request_file.name],
                                  check=True, capture_output=True, text=True)
        restored = json.loads(cold.stdout)
        assert restored["original_numeric_bytes"] == 64
        assert restored["roots"] == base_request()["roots"]

        # A known multi-output fixed action exercises immutable step handles.
        assert 91 in reset["legal_action_ids"]
        stepped = client.call({"cmd": "step", "state_handle": initial_handle, "action_id": 91})
        assert stepped["state"]["recipe_action_ids"] == [91]
        assert stepped["active_data_flows"] == 2
        original_again = client.call({"cmd": "evaluate_terminal", "state_handle": initial_handle})
        assert original_again["recipe_action_ids"] == []

        actions = json.loads(args.catalog.read_text())["registry_payload"]["actions"]
        ids = {row["key"]: row["action_id"] for row in actions}
        # Production Concat permits equal physical types with different
        # logical views, assigning the combined leaf the logical view bytes.
        mixed = dict(base_request(), cmd="reset", block_index=0,
                     continuation_edit_mode=False)
        mixed["roots"] = [base_request()["roots"][0], {
            "kind": "numeric", "width": 8,
            "data_hex": struct.pack("<4Q", 0x7ff8000000000001, 0x8000000000000000,
                                    0x7ff0000000000000, 0xfff0000000000000).hex(),
            "logical_view": "float64", "original_numeric_payload": True,
        }]
        mixed["original_numeric_bytes"] = 96
        concat_id = ids["concat@0,1"]
        mixed_reset = client.call(mixed)
        assert concat_id in mixed_reset["legal_action_ids"]
        concatenated = client.call({"cmd": "step", "state_handle": mixed_reset["state_handle"],
                                    "action_id": concat_id})
        assert concatenated["active_data_flows"] == 1
        concat_frame = client.call({"cmd": "evaluate_terminal",
                                    "state_handle": concatenated["state_handle"]})
        assert concat_frame["recipe_action_ids"] == [concat_id]

        # All 2 carriers x 19 max_point rows must remain distinct executable
        # fixed configurations and serialize the selected carrier in NGF1.
        scale = dict(base_request(), cmd="reset", block_index=0,
                     continuation_edit_mode=False)
        scale["roots"] = [{
            "kind": "numeric", "width": 8,
            "data_hex": struct.pack("<16d", *([0.0] * 16)).hex(),
            "logical_view": "float64", "original_numeric_payload": True,
        }]
        scale["original_numeric_bytes"] = 128
        scale_reset = client.call(scale)
        carriers = [row["action_id"] for row in actions
                    if row["target_slot"] == 0
                    and row["runtime_action"] == "op.scaling.exact_decimal.cpp_v2"
                    and row["io_variant"] in {"op.scaling.simple8b.cpp_v1",
                                               "op.scaling.ts_2diff.cpp_v1"}]
        assert len(carriers) == 38 and set(carriers) <= set(scale_reset["legal_action_ids"])
        for action_id in carriers:
            scaled = client.call({"cmd": "step", "state_handle": scale_reset["state_handle"],
                                  "action_id": action_id})
            measured = client.call({"cmd": "evaluate_terminal",
                                    "state_handle": scaled["state_handle"]})
            assert measured["recipe_action_ids"] == [action_id]

        # Execution-only mappings do not alter the frozen policy identity:
        # A010 uses the explicit full 0..18 exact-scale bound, and A060 inserts
        # the reversible lo/hi32 carrier before its limb-batch matcher.
        assert 84 in scale_reset["legal_action_ids"]
        legacy_scaled = client.call({"cmd": "step", "state_handle": scale_reset["state_handle"],
                                     "action_id": 84})
        assert 156 in legacy_scaled["legal_action_ids"]
        limb_matched = client.call({"cmd": "step", "state_handle": legacy_scaled["state_handle"],
                                    "action_id": 156})
        assert client.call({"cmd": "evaluate_terminal",
                            "state_handle": limb_matched["state_handle"]})["terminal_status"] == "ok"

        # The trailing packers recover their int/float profile from the
        # predecessor packet and preserve it in complete-frame metadata.
        for xor_action, trailing_action in ((116, 133), (125, 135)):
            xor_state = client.call({"cmd": "step", "state_handle": scale_reset["state_handle"],
                                     "action_id": xor_action})
            assert trailing_action in xor_state["legal_action_ids"]
            packed = client.call({"cmd": "step", "state_handle": xor_state["state_handle"],
                                  "action_id": trailing_action})
            assert client.call({"cmd": "evaluate_terminal",
                                "state_handle": packed["state_handle"]})["terminal_status"] == "ok"

        continuation = client.call(dict(base_request(), cmd="reset", block_index=1,
                                        continuation_edit_mode=True,
                                        previous_recipe_action_ids=[91]))
        validate_state(continuation, budget=16, max_steps=7)
        delete_prefix = client.call({"cmd": "step", "state_handle": continuation["state_handle"],
                                     "action_id": ids["edit_type.delete"]})
        assert delete_prefix["terminal_evaluable"] is False
        deleted = client.call({"cmd": "step", "state_handle": delete_prefix["state_handle"],
                               "action_id": ids["edit_position@0"]})
        assert deleted["state"]["recipe_action_ids"] == []
        assert deleted["state"]["edit_count"] == 1 and deleted["terminal_evaluable"] is True

        # A previous-block recipe may be inapplicable to current values.  The
        # failure is a charged candidate with no fabricated measurements, and
        # two local edits may repair it through an invalid intermediate state.
        invalid = dict(scale, cmd="reset", block_index=1,
                       continuation_edit_mode=True,
                       previous_recipe_action_ids=[3, 4])
        invalid["roots"] = [mixed["roots"][1]]
        invalid["original_numeric_bytes"] = 32
        invalid_reset = client.call(invalid)
        assert invalid_reset["state"]["recipe_valid"] is False
        failed = client.call({"cmd": "evaluate_terminal",
                              "state_handle": invalid_reset["state_handle"]})
        assert failed["terminal_status"] == "failed" and failed["candidate_ordinal"] == 1
        assert failed["max_frame_bytes"] == 1 << 30
        assert not ({"frame_hex", "frame_bytes", "compression_ratio",
                     "encode_samples_ns", "decode_samples_ns"} & failed.keys())
        first_type = client.call({"cmd": "step", "state_handle": invalid_reset["state_handle"],
                                  "action_id": ids["edit_type.delete"]})
        first_repair = client.call({"cmd": "step", "state_handle": first_type["state_handle"],
                                    "action_id": ids["edit_position@0"]})
        assert first_repair["state"]["recipe_valid"] is False
        second_type = client.call({"cmd": "step", "state_handle": first_repair["state_handle"],
                                   "action_id": ids["edit_type.delete"]})
        repaired = client.call({"cmd": "step", "state_handle": second_type["state_handle"],
                                "action_id": ids["edit_position@0"]})
        assert repaired["state"]["recipe_valid"] is True
        repaired_terminal = client.call({"cmd": "evaluate_terminal",
                                         "state_handle": repaired["state_handle"]})
        assert repaired_terminal["terminal_status"] == "ok"
        assert repaired_terminal["candidate_ordinal"] == 2

        # Candidate accounting is hard: attempt 65 must fail for a first block.
        budget_task = client.call(dict(base_request(), cmd="reset", block_index=0,
                                       continuation_edit_mode=False))
        for ordinal in range(1, 65):
            measured = client.call({"cmd": "evaluate_terminal",
                                    "state_handle": budget_task["state_handle"]})
            assert measured["candidate_ordinal"] == ordinal
        exhausted = client.call({"cmd": "evaluate_terminal",
                                 "state_handle": budget_task["state_handle"]}, succeeds=False)
        assert "budget exhausted" in exhausted["error"]

        # Beam30 is a true model-free server mode, not a policy invocation
        # whose score is later relabeled as a beam result.
        model_free_beam_task = client.call(dict(
            base_request(), cmd="reset", block_index=0,
            continuation_edit_mode=False,
            limits={"max_policy_edits": 1}))
        model_free_beam = client.call({
            "cmd": "search", "task_handle": model_free_beam_task["task_handle"],
            "search_mode": "beam30", "beam_width": 30,
            "enable_certified_pruning": True,
            "enable_objective_guided_ordering": True,
        })
        assert model_free_beam["search_mode"] == "beam30"
        assert model_free_beam["timing"]["policy_ns"] == 0

        released = client.call({"cmd": "release_task", "task_handle": reset["task_handle"]})
        assert released["released"] is True
        client.call({"cmd": "evaluate_terminal", "state_handle": initial_handle}, succeeds=False)
    finally:
        client.close()

    # A frozen training-only library is verified on load and selected inside C++.
    document = {
        "schema": "weighted-grpo-training-solution-library-v1",
        "library_format": "measured-feature-bucket-v1",
        "source_scope": "training_only",
        "action_registry_fingerprint": REGISTRY,
        "recipes": [
            {"recipe_id": "fixture-inapplicable", "dtype": "int64",
             "recipe_action_ids": [3], "source_split": "train"},
            {"recipe_id": "fixture-a91", "dtype": "int64",
             "recipe_action_ids": [91], "source_split": "train"},
        ],
    }
    document["library_content_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
    with tempfile.NamedTemporaryFile("w", suffix=".json") as library:
        json.dump(document, library, sort_keys=True)
        library.flush()
        library_sha = hashlib.sha256(Path(library.name).read_bytes()).hexdigest()
        library_client = Client(server_argv[:-1] + ["--solution-library", library.name,
                                                     "--solution-library-sha256", library_sha,
                                                     "serve"])
        try:
            assert library_client.call({"cmd": "schema"})["solution_library_pinned"] is True
            selected = library_client.call(dict(base_request(), cmd="select_library",
                                                block_index=0, continuation_edit_mode=False,
                                                dtype="int64", include_frame_hex=True))
            assert selected["starting_recipe_action_ids"] == [91, 0]
            assert selected["recipe_action_ids"] == [91]
            assert selected["evaluations"] == selected["candidate_ordinal"] == 2
            retained = library_client.call(dict(base_request(), cmd="reset_from_library",
                                                block_index=0, continuation_edit_mode=False,
                                                dtype="int64", include_frame_hex=True))
            validate_state(retained, budget=64, max_steps=65)
            assert retained["state"]["recipe_action_ids"] == [91]
            assert retained["candidate_used"] == 2
            assert retained["library_selection"]["evaluations_charged"] == 2
            assert retained["library_selection"]["failed_count"] == 1
            second = library_client.call({"cmd": "evaluate_terminal",
                                          "state_handle": retained["state_handle"]})
            assert second["candidate_ordinal"] == 3 and second["recipe_action_ids"] == [91]
            library_client.call({"cmd": "release_task", "task_handle": retained["task_handle"]})
        finally:
            library_client.close()

        with tempfile.NamedTemporaryFile(suffix=".wgpn") as library_model:
            synthetic_native_model(Path(library_model.name))
            library_search_client = Client(
                server_argv[:-1] + ["--solution-library", library.name,
                                    "--solution-library-sha256", library_sha,
                                    "--model", library_model.name, "serve"])
            try:
                seeded = library_search_client.call(dict(
                    base_request(), cmd="reset", block_index=0,
                    continuation_edit_mode=False,
                    limits={"max_policy_edits": 1}))
                library_search = library_search_client.call({
                    "cmd": "search", "task_handle": seeded["task_handle"],
                    "use_solution_library": True, "dtype": "int64",
                    "beam_width": 1})
                assert library_search["search_status"] == "ok"
                assert library_search["evaluations"] >= 2
                assert any(row["terminal_status"] == "failed"
                           for row in library_search["candidates"])
                assert any(row["terminal_status"] == "ok"
                           for row in library_search["candidates"])
            finally:
                library_search_client.close()

    excluded_library = dict(document)
    excluded_library["recipes"] = [{
        "recipe_id": "excluded-cmix", "dtype": "int64",
        "recipe_action_ids": [183], "source_split": "train",
    }]
    excluded_library.pop("library_content_sha256", None)
    excluded_library["library_content_sha256"] = hashlib.sha256(
        canonical(excluded_library)).hexdigest()
    with tempfile.NamedTemporaryFile("w", suffix=".json") as excluded_file:
        json.dump(excluded_library, excluded_file, sort_keys=True)
        excluded_file.flush()
        rejected = subprocess.run(
            server_argv[:-1] + ["--solution-library", excluded_file.name, "serve"],
            input="", text=True, capture_output=True,
        )
    assert rejected.returncode == 1
    assert "excluded by the experiment scope" in rejected.stderr

    # Search itself runs policy inference, masks, transitions, candidate
    # evaluation and the final independent encode inside this C++ process.
    with tempfile.NamedTemporaryFile(suffix=".wgpn") as model_file:
        synthetic_native_model(Path(model_file.name))
        search_client = Client(server_argv[:-1] + ["--model", model_file.name, "serve"])
        try:
            reset = search_client.call(dict(base_request(), cmd="reset", block_index=0,
                                            continuation_edit_mode=False,
                                            limits={"max_policy_edits": 1}))
            assert reset["max_steps"] == 2
            searched = search_client.call({"cmd": "search", "task_handle": reset["task_handle"],
                                           "beam_width": 1, "include_frame_hex": True})
            assert searched["evaluations"] == 1
            assert searched["evaluation_budget"] == 64
            assert searched["candidates"][0]["recipe_action_ids"] == []
            assert searched["search_mode"] == "policy"
            assert searched["enable_certified_pruning"] is True
            assert searched["enable_objective_guided_ordering"] is True
            assert searched["pruning_enabled"] is True
            assert searched["certified_pruned_states"] > 0
            replay_cache = searched["replay_cache"]
            assert replay_cache["hits"] > 0
            assert replay_cache["misses"] > 0
            assert replay_cache["retained_bytes"] <= replay_cache["byte_limit"]
            assert replay_cache["byte_limit"] == 64 << 20
            assert replay_cache["entries"] > 0
            assert searched["frame_bytes"] * 2 == len(searched["frame_hex"])
            assert searched["frame_sha256"] == hashlib.sha256(
                bytes.fromhex(searched["frame_hex"])).hexdigest()
            timing = searched["timing"]
            assert timing["task_initialization_ns"] > 0
            assert timing["policy_ns"] > 0 and timing["feature_ns"] > 0
            prediction_parts = sum(timing[name] for name in (
                "task_initialization_ns", "feature_ns", "policy_ns",
                "action_ordering_ns", "transition_ns", "candidate_evaluation_ns",
                "certified_pruning_ns",
            ))
            assert timing["total_prediction_ns"] >= prediction_parts
            assert timing["total_compression_ns"] >= (
                timing["total_prediction_ns"] + timing["final_encode_ns"])
            after_search = search_client.call({
                "cmd": "evaluate_terminal", "state_handle": reset["state_handle"],
            })
            assert after_search["candidate_ordinal"] == searched["evaluations"] + 1

            # This ablation retains policy probabilities while disabling both
            # the certified target bound and objective-guided action ordering.
            ablated_reset = search_client.call(dict(
                base_request(), cmd="reset", block_index=0,
                continuation_edit_mode=False,
                limits={"max_policy_edits": 1}))
            ablated = search_client.call({
                "cmd": "search", "task_handle": ablated_reset["task_handle"],
                "beam_width": 1, "search_mode": "policy",
                "enable_certified_pruning": False,
                "enable_objective_guided_ordering": False,
            })
            assert ablated["search_mode"] == "policy"
            assert ablated["enable_certified_pruning"] is False
            assert ablated["enable_objective_guided_ordering"] is False
            assert ablated["pruning_enabled"] is False
            assert ablated["certified_pruned_states"] == 0
            assert ablated["evaluations"] == 2
            assert ablated["candidates"][1]["recipe_action_ids"] == [91]
            assert searched["best_recipe_action_ids"] == ablated["best_recipe_action_ids"]
            assert searched["frame_sha256"] == ablated["frame_sha256"]

            # Beam30 never invokes the frozen model, but shares the same exact
            # candidate budget, target guidance and certified bound.
            beam_reset = search_client.call(dict(
                base_request(), cmd="reset", block_index=0,
                continuation_edit_mode=False,
                limits={"max_policy_edits": 1}))
            beam = search_client.call({
                "cmd": "search", "task_handle": beam_reset["task_handle"],
                "beam_width": 30, "search_mode": "beam30",
                "enable_certified_pruning": True,
                "enable_objective_guided_ordering": True,
            })
            assert beam["search_mode"] == "beam30"
            assert beam["enable_certified_pruning"] is True
            assert beam["enable_objective_guided_ordering"] is True
            assert beam["timing"]["policy_ns"] == 0
            assert beam["evaluation_budget"] == 64
            assert beam["best_recipe_action_ids"] == searched["best_recipe_action_ids"]

            # Search reconstructs the task inside the same outer C++ interval.
            # The constructor's internal STOP removal must be reversed by the
            # fresh-request API before reconstructing a user-supplied seed.
            starting = search_client.call(dict(
                base_request(), cmd="reset", block_index=0,
                continuation_edit_mode=False,
                starting_recipe_action_ids=[91, 0],
                limits={"max_policy_edits": 1}))
            starting_search = search_client.call({
                "cmd": "search", "task_handle": starting["task_handle"],
                "beam_width": 1,
            })
            assert starting_search["search_status"] == "ok"
            assert starting_search["candidates"][0]["recipe_action_ids"] == [91]
            assert starting_search["timing"]["task_initialization_ns"] > 0

            all_failed_request = dict(invalid, cmd="reset", max_recipe_edits=1)
            all_failed_reset = search_client.call(all_failed_request)
            all_failed = search_client.call({"cmd": "search",
                                             "task_handle": all_failed_reset["task_handle"],
                                             "beam_width": 1, "include_frame_hex": True})
            assert all_failed["search_status"] == all_failed["terminal_status"] == "failed"
            assert all_failed["evaluations"] >= 1 and all_failed["max_frame_bytes"] == 1 << 30
            assert not ({"frame_hex", "frame_bytes", "compression_ratio"} & all_failed.keys())
            assert all_failed["timing"]["final_encode_ns"] == 0
        finally:
            search_client.close()


if __name__ == "__main__":
    main()
