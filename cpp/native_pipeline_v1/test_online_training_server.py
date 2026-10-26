#!/usr/bin/env python3
"""JSONL contract for the shared v13 OnlineSession training server."""
from __future__ import annotations

import argparse
import json
import struct
import subprocess


def send(process: subprocess.Popen[str], document: dict) -> dict:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(document, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise AssertionError("training server closed its output")
    return json.loads(line)


def roots() -> list[dict]:
    values = b"".join(struct.pack("<q", value) for value in range(32))
    return [
        {"kind": "numeric", "width": 8, "data_hex": values.hex(),
         "original_numeric_payload": True, "logical_view": "int64"},
        {"kind": "serial", "width": 1, "data_hex": "00000000",
         "original_numeric_payload": False, "logical_view": "bytes"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--experiment-profile", required=True)
    args = parser.parse_args()
    process = subprocess.Popen(
        [args.server, args.catalog, args.experiment_profile],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    common = {"roots": roots(), "weights": [1.0, 0.0, 0.0],
              "max_policy_steps": 10}
    try:
        reset = send(process, {"op": "reset", "request_id": 1, **common})
        assert reset["ok"] and not reset["terminal"]
        assert len(reset["observation"]) == 49
        assert reset["action_count"] == 6373 and reset["flow_stop_begin"] == 6341
        assert 0 not in reset["legal_action_ids"]
        assert 6341 in reset["legal_action_ids"] and 6342 in reset["legal_action_ids"]
        assert reset["candidate_evaluations"] == 0
        assert reset["replayed_prefix_stages"] == 0

        repeated = send(process, {"op": "observe", "request_id": 2})
        assert repeated["observation"] == reset["observation"]
        assert repeated["legal_action_ids"] == reset["legal_action_ids"]

        one_open = send(process, {"op": "step", "request_id": 3,
                                  "action_id": 6342})
        assert one_open["ok"] and not one_open["terminal"]
        assert one_open["open_flow_count"] == 1 and 6342 not in one_open["legal_action_ids"]
        terminal = send(process, {"op": "step", "request_id": 4,
                                  "action_id": 6341})
        assert terminal["ok"] and terminal["terminal"]
        assert terminal["recipe_action_ids"] == [6342, 6341]
        assert terminal["candidate_evaluations"] == 0
        assert terminal["replayed_prefix_stages"] == 0
        assert terminal["frame_bytes"] * 2 == len(terminal["frame_hex"])
        assert terminal["original_numeric_bytes"] == 256
        assert not terminal["complete_online_timing"]

        replay = send(process, {
            "op": "replay", "request_id": 5, **common,
            "forced_action_ids": [6342, 6341],
            "capture_transitions": True,
            "timing_warmups": 0, "timing_repetitions": 2,
        })
        assert replay["ok"] and replay["terminal"]
        assert replay["recipe_action_ids"] == [6342, 6341]
        assert len(replay["transitions"]) == 2
        assert all(len(row["observation"]) == 49 for row in replay["transitions"])
        assert len(replay["encode_samples_ns"]) == 2
        assert len(replay["decode_samples_ns"]) == 2
        assert not replay["reward_timing_eligible"]
        assert replay["timing_scope"] == "model_free_forced_runtime_not_reward_eligible"
        assert replay["frame_sha256"] == terminal["frame_sha256"]

        illegal = send(process, {
            "op": "replay", "request_id": 6, **common,
            "forced_action_ids": [0], "timing_repetitions": 1,
        })
        assert not illegal["ok"] and "shared online legal set" in illegal["error"]
        trailing = send(process, {
            "op": "replay", "request_id": 7, **common,
            "forced_action_ids": [6342, 6341, 6341],
            "timing_repetitions": 1,
        })
        assert not trailing["ok"] and "after terminal state" in trailing["error"]
    finally:
        if process.stdin:
            process.stdin.close()
        return_code = process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr else ""
        if return_code:
            raise AssertionError(f"training server exited {return_code}: {stderr}")


if __name__ == "__main__":
    main()
