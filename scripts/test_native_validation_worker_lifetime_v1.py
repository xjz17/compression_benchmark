#!/usr/bin/env python3
"""Regression tests for training-pool release around frozen validation."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import train_native_smallest32_v1 as training


class _Factory:
    instances = []

    def __init__(self, corpus, *, cli, catalog, library, cpus):
        self.cpus = tuple(cpus)
        self.clients = [object() for _ in self.cpus]
        self.events = [("open", len(self.clients))]
        type(self).instances.append(self)

    def solution_library_identity(self):
        return {
            "path": "/frozen/training_library.json",
            "file_sha256": "1" * 64,
            "action_registry_fingerprint": training.REGISTRY,
        }

    def close(self):
        self.events.append(("close", len(self.clients)))
        self.clients = []

    def reopen(self):
        if self.clients:
            raise AssertionError("training pool reopened before its old workers closed")
        self.clients = [object() for _ in self.cpus]
        self.events.append(("reopen", len(self.clients)))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _Optimizer:
    execution_receipt = {
        "policy_update_device": "cuda:0",
        "policy_backend": "pytorch_float64",
    }


class ValidationWorkerLifetimeTests(unittest.TestCase):
    def _run(self, phase: str, *, evaluator_fails: bool = False):
        _Factory.instances.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission = root / "admission.json"
            admission.write_text("{}")
            runtime_paths = {
                name: root / name for name in ("cli", "catalog", "cold", "library")
            }
            for path in runtime_paths.values():
                path.write_bytes(path.name.encode())
            arguments = SimpleNamespace(
                cpus="25,26", phase=phase, admission=admission,
                cohort=root / "cohort", materialized=root / "materialized",
                reference=root / "reference", cli=runtime_paths["cli"],
                catalog=runtime_paths["catalog"], cold_cli=runtime_paths["cold"],
                library=runtime_paths["library"], output=root / "output",
                validation_interval=1, max_iterations=2, seed=0,
            )
            corpus = SimpleNamespace(fingerprint="corpus-fingerprint")
            reference = {"calibration_sha256": "reference-sha"}
            bound = SimpleNamespace(corpus_sha256="bound-corpus-sha")
            gate = {
                "corpus_fingerprint": corpus.fingerprint,
                "reference_sha256": reference["calibration_sha256"],
                "frozen_files": {
                    str(path.resolve()): "0" * 64 for path in runtime_paths.values()
                },
                "receipts": {},
            }
            identity = SimpleNamespace(to_json=lambda: {
                "input_dim": 49, "action_count": 6341,
                "registry_fingerprint": training.REGISTRY,
            })
            loaded = (
                object(), _Optimizer(), np.random.default_rng(1), 1, 0,
                None, None, [], [],
            )

            def frozen_evaluate(*args, **kwargs):
                factory = _Factory.instances[-1]
                factory.events.append(("evaluate", len(factory.clients)))
                self.assertEqual(factory.clients, [])
                if evaluator_fails:
                    raise RuntimeError("synthetic frozen validation failure")
                return {"mean_validation_cost": 0.25}

            def training_iteration(**kwargs):
                callback = kwargs.get("progress_callback")
                if callback is not None:
                    callback({"event": "synthetic_rollout_complete"})
                return {
                    "parameter_change_l2": 1.0,
                    "transitions": 1,
                    "policy_execution": dict(_Optimizer.execution_receipt),
                }

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(
                    os.environ,
                    {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                     "CUDA_VISIBLE_DEVICES": "GPU-test"},
                ))
                stack.enter_context(mock.patch.object(training, "Factory", _Factory))
                stack.enter_context(mock.patch.object(training, "admission", return_value=gate))
                stack.enter_context(mock.patch.object(training.data, "load", return_value=corpus))
                stack.enter_context(mock.patch.object(
                    training.binding, "load_reference", return_value=reference,
                ))
                stack.enter_context(mock.patch.object(training.binding, "bind", return_value=bound))
                stack.enter_context(mock.patch.object(
                    training.engine, "_build_policy_identity", return_value=identity,
                ))
                stack.enter_context(mock.patch.object(
                    training.engine, "_load_resume", return_value=loaded,
                ))
                stack.enter_context(mock.patch.object(
                    training.engine, "_training_iteration", side_effect=training_iteration,
                ))
                stack.enter_context(mock.patch.object(
                    training.engine, "_resume_document", return_value={},
                ))
                stack.enter_context(mock.patch.object(training.engine, "_atomic_json"))
                stack.enter_context(mock.patch.object(
                    training, "save_policy", side_effect=lambda *args: Path(args[4]).with_suffix(".wgpn"),
                ))
                stack.enter_context(mock.patch.object(
                    training.frozen, "evaluate", side_effect=frozen_evaluate,
                ))
                if evaluator_fails:
                    with self.assertRaisesRegex(RuntimeError, "synthetic frozen validation failure"):
                        training.train(arguments)
                else:
                    training.train(arguments)
        return _Factory.instances[-1].events

    def test_train_validation_closes_then_reopens_pool(self):
        events = self._run("train")
        self.assertEqual(events, [
            ("open", 2), ("close", 2), ("evaluate", 0),
            ("reopen", 2), ("close", 2),
        ])

    def test_final_validation_does_not_reopen_training_pool(self):
        events = self._run("frozen-validation")
        self.assertEqual(events, [
            ("open", 2), ("close", 2), ("evaluate", 0), ("close", 0),
        ])

    def test_failed_validation_does_not_reopen_and_context_closes(self):
        events = self._run("train", evaluator_fails=True)
        self.assertEqual(events, [
            ("open", 2), ("close", 2), ("evaluate", 0), ("close", 0),
        ])


if __name__ == "__main__":
    unittest.main()
