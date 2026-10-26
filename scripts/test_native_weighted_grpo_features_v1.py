#!/usr/bin/env python3
"""Cross-language feature parity against the existing training observation code.

Build cpp/native_features_v1 first. These tests do not admit graph execution,
operator coverage, or model training.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import threading
import unittest

import numpy as np

import typed_graph_runtime as graph
import weight_conditioned_graph_task_v1 as reference

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("NATIVE_FEATURES_CLI", ROOT / "cpp/build-native-features-v1/native_features_cli"))


def text_flows(streams, views):
    return str(len(streams)) + "\n" + "\n".join(
        f"{s.kind} {s.width} {view} {s.data.hex() or '-'}"
        for s, view in zip(streams, views)
    ) + "\n"


def run(mode, request, *, ok=True):
    result = subprocess.run([str(CLI), mode], input=request, text=True,
                            capture_output=True, timeout=30)
    if ok and result.returncode:
        raise AssertionError(result.stderr)
    if not ok:
        if result.returncode == 0:
            raise AssertionError("invalid request accepted")
        return result.stderr
    return np.asarray([float(v) for v in result.stdout.split()], dtype=np.float64)


class FeaturesParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CLI.is_file():
            raise RuntimeError(f"build native feature CLI first: {CLI}")

    def compare_task(self, arrays):
        streams = [graph.Stream("numeric", a.dtype.itemsize, a.tobytes()) for a in arrays]
        views = [a.dtype.name for a in arrays]
        expected = reference._task_features(streams, views)
        actual = run("task", text_flows(streams, views))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)

    def test_all_native_widths_and_signed_boundaries(self):
        for dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
            info = np.iinfo(dtype)
            with self.subTest(dtype=dtype):
                self.compare_task([np.array([info.min, 0, 1, info.max, 0, 0], dtype=dtype)])

    def test_floating_special_values_preserve_feature_meaning(self):
        for dtype in ("float16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                values = np.array([0.0, -0.0, float("nan"), float("inf"), -float("inf"),
                                   np.finfo(dtype).smallest_subnormal, 2.5, -3.0, 2.5], dtype=dtype)
                original = values.tobytes()
                self.compare_task([values])
                self.assertEqual(values.tobytes(), original)

    def test_sampling_at_and_above_8192_matches_numpy_indices(self):
        rng = np.random.default_rng(17)
        for count in (1, 8191, 8192, 8193, 65539):
            with self.subTest(count=count):
                self.compare_task([rng.integers(-100000, 100000, count, dtype=np.int64)])

    def test_mixed_roots_bitmap_prefix_and_empty_flow(self):
        arrays = [np.arange(13001, dtype="<i4"), np.arange(19001, dtype="<f8")]
        streams = [graph.Stream("serial", 1, b"\x00\x00\xff"),
                   graph.Stream("numeric", 4, arrays[0].tobytes()),
                   graph.Stream("numeric", 8, arrays[1].tobytes()),
                   graph.Stream("numeric", 2, b"")]
        views = ["bytes", "int32", "float64", "float16"]
        np.testing.assert_allclose(run("task", text_flows(streams, views)),
                                   reference._task_features(streams, views), rtol=2e-13, atol=2e-14)

    def test_empty_and_raw_bit_view_roots(self):
        for streams, views in (([graph.Stream("numeric", 4, b"")], ["float32"]),
                               ([graph.Stream("numeric", 4, b"\xff" * 64)], ["xor_residual32"]),
                               ([graph.Stream("struct", 3, b"abcdef")], ["bytes"])):
            np.testing.assert_allclose(run("task", text_flows(streams, views)),
                                       reference._task_features(streams, views), rtol=2e-13, atol=2e-14)

    def test_state_features_match_existing_method_with_bucket_collisions(self):
        # This instantiates only the deterministic feature method's state. It
        # does not run mechanism probes or pretend to construct legal recipes.
        streams = [graph.Stream("numeric", 4, np.arange(i + 1, dtype="<i4").tobytes())
                   for i in range(13)]
        streams += [graph.Stream("serial", 1, b"\x80\x00")]
        views = ["int32"] * 13 + ["bytes"]
        state_graph = graph.GraphState.from_streams(streams)
        keys = list(state_graph.streams)
        parents = [reference.STOP_ACTION, *range(13)]
        depths = [i % 10 for i in range(14)]
        for continuation in (False, True):
            for pending in (None, "insert", "delete", "replace", "parameter_change"):
                with self.subTest(continuation=continuation, pending=pending):
                    task = object.__new__(reference.TypedGraphCompressionTask)
                    task.config = reference.GraphTaskConfig()
                    task._continuation_edit_mode = continuation
                    task.original_numeric_bytes = sum(len(s.data) for s in streams[:-1])
                    task._max_recipe_edits = 2
                    task.candidate_budget = reference.CandidateBudget(64)
                    for _ in range(7):
                        task.candidate_budget.reserve()
                    task._measurement_lock = threading.Lock()
                    task._current_objective_by_recipe = {(): 2.75}
                    state = reference.TaskState(
                        graph=state_graph, leaf_depths=tuple(zip(keys, depths)),
                        leaf_views=tuple(zip(keys, views)), parent_actions=tuple(zip(keys, parents)),
                        recipe=(), node_count=12, policy_edits=1, replayed_nodes=9,
                        continuation_edit_mode=continuation, pending_edit_kind=pending,
                        pending_edit_index=3 if pending else None, edit_count=1,
                    )
                    request = text_flows(streams, views) + (
                        f"{task.original_numeric_bytes} {reference.ACTION_COUNT} 12 1 9 1 "
                        f"{int(continuation)} 7 64 {pending or '-'} {3 if pending else -1} 2.75\n"
                    ) + "\n".join(f"{depth} {parent}" for depth, parent in zip(depths, parents)) + "\n"
                    np.testing.assert_allclose(run("state", request), task.state_features(state),
                                               rtol=2e-13, atol=2e-14)

    def test_invalid_flow_type_width_and_nonfinite_feature_are_rejected(self):
        for request in ("1\nnumeric 3 int32 000000\n", "1\nserial 1 float64 00\n",
                        "1\nnumeric 4 int32 0000\n", "0\n",
                        "1\nnumeric 8 unknown 0000000000000000\n"):
            run("task", request, ok=False)
        values = np.full(4, np.finfo(np.float64).max, dtype="<f8")
        # Median overflow can still yield finite smoothness; an alternating
        # finite extreme forces inf/inf and must match the old fail-closed rule.
        values[::2] *= -1
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(reference.GraphTaskError):
                reference._task_features([graph.Stream("numeric", 8, values.tobytes())], ["float64"])
        run("task", text_flows([graph.Stream("numeric", 8, values.tobytes())], ["float64"]), ok=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
