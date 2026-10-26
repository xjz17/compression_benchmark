#!/usr/bin/env python3
"""Focused local contract tests for the Linux end-to-end preflight driver."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import validate_native_end_to_end_preflight_v1 as preflight


class PreflightContractTests(unittest.TestCase):
    def test_cold_decode_is_cpu_memory_and_deadline_bounded(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({
                "schema": "native-graph-response-v1",
                "roots": [],
                "original_numeric_bytes": 0,
            }).encode(),
            stderr=b"",
        )
        with mock.patch.object(preflight.subprocess, "run", return_value=completed) as run:
            answer, wall_ns = preflight._cold_decode(
                Path("/tmp/native_graph_request_cli"), 25, b"NGF1",
                allow_large_memory=True, timeout_seconds=17.0,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["taskset", "-c", "25", "prlimit"])
        self.assertEqual(
            command[4:6],
            ["--as=" + str(preflight.MAX_NATIVE_ADDRESS_SPACE_BYTES), "--"],
        )
        self.assertEqual(command[-2], "decode")
        self.assertEqual(command[-3], str(Path("/tmp/native_graph_request_cli").resolve()))
        self.assertEqual(Path(command[-1]).name, "request.json")
        self.assertEqual(run.call_args.kwargs["timeout"], 17.0)
        self.assertEqual(answer["schema"], "native-graph-response-v1")
        self.assertGreaterEqual(wall_ns, 0)

    def test_catalog_gate_rejects_partial_action_space(self) -> None:
        scope = {"schema": "native-experiment-action-scope-v1"}
        complete = {
            "complete_native_graph_space": True,
            "unsupported_graph_action_count": 0,
            "native_executable_action_count": preflight.ACTION_COUNT,
            "action_count": preflight.ACTION_COUNT,
            "registry_fingerprint": preflight.REGISTRY,
            "catalog_file_sha256": "a" * 64,
            "experiment_scope": scope,
        }
        preflight._catalog_gate(complete, "a" * 64, scope)
        partial = dict(complete, unsupported_graph_action_count=1)
        with self.assertRaises(ValueError):
            preflight._catalog_gate(partial, "a" * 64, scope)
        with self.assertRaises(ValueError):
            preflight._catalog_gate(complete, "a" * 64, dict(scope, drift=True))

    def test_pair_progress_reports_driver_and_native_time(self) -> None:
        row = {
            "block_id": "validation-0001",
            "block_fingerprint": "b" * 64,
            "pair_wall_ns_for_budget_estimate_only": 123,
            "raw_cpp_timing_ns": {"total_compression_ns": 45},
        }
        with mock.patch("builtins.print") as printed:
            preflight._print_pair_progress(row, 2, "validation")
        payload = json.loads(printed.call_args.args[0])
        self.assertEqual(payload, {
            "event": "native_preflight_pair_complete",
            "phase": "validation",
            "weight_index": 2,
            "block_id": "validation-0001",
            "block_fingerprint": "b" * 64,
            "driver_wall_ns": 123,
            "native_E_total_ns": 45,
        })
        self.assertTrue(printed.call_args.kwargs["flush"])


if __name__ == "__main__":
    unittest.main()
