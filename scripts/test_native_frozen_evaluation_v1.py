#!/usr/bin/env python3
"""Bounded receipt/resume tests; no native worker is started."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import native_frozen_evaluation_v1 as frozen
import native_prepared_cohort_v1 as data
import weight_conditioned_grpo_v1 as grpo


class _Block:
    dataset_id = "fixture"
    fingerprint = "fixture-block"
    record = {
        "entity_id": "entity", "split": "test", "column": "value",
        "block_index": 0, "block_id": "block-0",
    }

    def identity(self):
        return {
            "dataset_id": self.dataset_id, "block_id": self.record["block_id"],
            "block_fingerprint": self.fingerprint, "split": self.record["split"],
            "original_numeric_bytes": 8,
        }


class _Corpus:
    role = "test_candidate"
    fingerprint = "fixture-corpus"

    def __init__(self):
        self.block = _Block()

    def split(self, split):
        if split != "test":
            raise ValueError("fixture exposes only test")
        return (self.block,)


def _artifacts(root: Path) -> dict[str, Path]:
    result = {}
    for name in ("model", "cli", "catalog", "library", "cold_cli"):
        if name == 'catalog':
            result[name] = frozen.experiment_scope.DEFAULT_CATALOG
            continue
        path = root / name
        path.write_bytes((name + "\n").encode())
        result[name] = path
    return result


def _identity(corpus, artifacts, reference, *, cpus=(25,), warmups=1, repeats=1):
    return {
        "schema": "native-frozen-evaluation-identity-v1", "split": "test",
        "experiment_scope": frozen.experiment_scope.load(artifacts['catalog']),
        "corpus_fingerprint": corpus.fingerprint,
        "model_sha256": data.digest(artifacts["model"].read_bytes()),
        "library_sha256": data.digest(artifacts["library"].read_bytes()),
        "native_cli_sha256": data.digest(artifacts["cli"].read_bytes()),
        "catalog_sha256": data.digest(artifacts["catalog"].read_bytes()),
        "cold_cli_sha256": data.digest(artifacts["cold_cli"].read_bytes()),
        "reference_sha256": reference["calibration_sha256"],
        "driver_sha256": data.digest(Path(frozen.__file__).read_bytes()),
        "cpus": list(cpus), "warmups": warmups, "repeats": repeats,
        "allow_large_memory": True, "complete_frames_preserved": True,
        "inference_variant": "model", "search_options": frozen.search_options('model'),
        "block_reuse": True, "candidate_budget_rule": "first_64_then_16",
        "expected_pair_count": len(corpus.split("test")) * len(grpo.FIXED_WEIGHTS),
        "expected_block_fingerprints_sha256": data.digest(
            data.canonical(sorted(block.fingerprint for block in corpus.split("test")))
        ),
    }


def _rows():
    rows = []
    for index, weight in enumerate(grpo.FIXED_WEIGHTS):
        value = 0.2 + index / 100.0
        rows.append({
            "dataset_id": "fixture", "block_id": "block-0",
            "block_fingerprint": "fixture-block", "split": "test",
            "original_numeric_bytes": 8,
            "weight_index": index, "weight": list(weight), "status": "ok",
            "inference_variant": "model",
            "J_selection": value, "J_total": value,
            "compression_ratio": 0.5, "compression_ns_per_byte": 1.0,
            "decompression_ns_per_byte": 1.0,
            "frame_bytes": 4, "frame_sha256": data.digest(b"NGF1"),
            "frame_path": "frames/"+data.digest(b"NGF1")+".ngf",
            "cold_decode_verified": True,
        })
    return rows


def _publish_completed(output: Path, identity: dict, rows=None):
    rows = list(_rows() if rows is None else rows)
    frozen.preserve_frame(output,b"NGF1")
    worker = output / "worker_0.jsonl"
    frozen.atomic_bytes(worker, b"".join(data.canonical(row) + b"\n" for row in rows))
    values = [row["J_selection"] for row in rows if row["status"] == "ok"]
    by_weight = {
        str(row["weight_index"]): row["J_selection"]
        for row in rows if row["status"] == "ok"
    }
    receipt = {
        "schema": "native-frozen-evaluation-smallest32-v1",
        "run_identity": identity, "split": "test",
        "corpus_fingerprint": identity["corpus_fingerprint"],
        "model_sha256": identity["model_sha256"],
        "library_sha256": identity["library_sha256"],
        "native_cli_sha256": identity["native_cli_sha256"],
        "catalog_sha256": identity["catalog_sha256"],
        "cpus": identity["cpus"], "warmups": identity["warmups"],
        "repeats": identity["repeats"],
        "pair_count": identity["expected_pair_count"],
        "ok_pair_count": len(values),
        "N_A_pair_count": sum(row["status"] == "N/A" for row in rows),
        "failed_pair_count": sum(row["status"] == "failed" for row in rows),
        "mean_exact_J": float(np.mean(values)) if values else None,
        "mean_validation_cost": float(np.mean(values)) if values else None,
        "mean_exact_J_by_weight": by_weight,
        "selection_statistic": None, "test_used_for_model_selection": False,
        "wall_ns": 1,
        "worker_files": [{"path": worker.name, "sha256": data.digest(worker.read_bytes())}],
    }
    receipt["receipt_sha256"] = data.digest(data.canonical(receipt))
    frozen.atomic_bytes(output / "run_identity.json", data.canonical(identity) + b"\n")
    frozen.atomic_bytes(output / "receipt.json", data.canonical(receipt) + b"\n")
    return receipt


class _NoopNativeClient:
    """Context-only fixture; any protocol call means checkpoint reuse failed."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def call(self, request):
        raise AssertionError(f"checkpoint reuse issued native request: {request}")


class FrozenReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.corpus = _Corpus()
        self.reference = {"calibration_sha256": "r" * 64}
        self.artifacts = _artifacts(self.root)
        self.identity = _identity(self.corpus, self.artifacts, self.reference)

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_receipt_reuses_without_native_worker(self):
        expected = _publish_completed(self.output, self.identity)
        with mock.patch.object(frozen, "NativeClient", side_effect=AssertionError(
            "completed receipt must return before native worker startup"
        )):
            actual = frozen.evaluate(
                self.corpus, self.reference, split="test",
                model=self.artifacts["model"], cli=self.artifacts["cli"],
                catalog=self.artifacts["catalog"], library=self.artifacts["library"],
                cold_cli=self.artifacts["cold_cli"], cpus=(25,), output=self.output,
                warmups=1, repeats=1,
            )
        self.assertEqual(actual, expected)

    def test_worker_content_tamper_is_rejected(self):
        _publish_completed(self.output, self.identity)
        with (self.output / "worker_0.jsonl").open("ab") as out:
            out.write(b"{}\n")
        with self.assertRaisesRegex(ValueError, "changed|checksum|coverage"):
            frozen.verify_receipt(self.output, self.identity)

    def test_receipt_summary_tamper_is_rejected(self):
        receipt = _publish_completed(self.output, self.identity)
        receipt["mean_exact_J"] = 999.0
        frozen.atomic_bytes(self.output / "receipt.json", data.canonical(receipt) + b"\n")
        with self.assertRaisesRegex(ValueError, "summary|checksum|changed"):
            frozen.verify_receipt(self.output, self.identity)

    def test_worker_row_coverage_is_recomputed(self):
        receipt = _publish_completed(self.output, self.identity)
        shortened = _rows()[:-1]
        worker = self.output / "worker_0.jsonl"
        frozen.atomic_bytes(worker, b"".join(data.canonical(row) + b"\n" for row in shortened))
        receipt["worker_files"][0]["sha256"] = data.digest(worker.read_bytes())
        receipt["receipt_sha256"] = data.digest(data.canonical({
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }))
        frozen.atomic_bytes(self.output / "receipt.json", data.canonical(receipt) + b"\n")
        with self.assertRaisesRegex(ValueError, "coverage|pair|row"):
            frozen.verify_receipt(self.output, self.identity)

    def test_final_test_timing_rejects_parallel_cpus_before_native_start(self):
        with mock.patch.object(frozen, "NativeClient", side_effect=AssertionError(
            "invalid final timing configuration must not start native workers"
        )):
            with self.assertRaisesRegex(ValueError, "serial"):
                frozen.evaluate(
                    self.corpus, self.reference, split="test",
                    model=Path("/does/not/matter"), cli=Path("/does/not/matter"),
                    catalog=Path("/does/not/matter"), library=Path("/does/not/matter"),
                    cold_cli=Path("/does/not/matter"), cpus=(25, 26),
                    output=self.output,
                )

    def test_variant_identity_prevents_cross_mode_resume(self):
        _publish_completed(self.output,self.identity)
        with self.assertRaisesRegex(ValueError,'different inputs/configuration'):
            frozen.evaluate(self.corpus,self.reference,split='test',**self.artifacts,
                cpus=(25,),output=self.output,warmups=1,repeats=1,variant='no_reuse')
        self.assertEqual(frozen.search_options('beam30')['search_mode'],'beam30')
        self.assertFalse(frozen.search_options('no_target_guidance')['enable_certified_pruning'])
        self.assertFalse(frozen.search_options('no_target_guidance')['enable_objective_guided_ordering'])

    def _write_checkpoint(self, job_index: int, row: dict):
        frozen.preserve_frame(self.output,b"NGF1")
        checkpoint = self.output / "jobs" / f"{job_index:08d}.json"
        identity = {
            "run": self.identity, "blocks": [self.corpus.block.fingerprint],
            "weight_index": job_index, "cpu": 25,
        }
        rows = [row]
        frozen.atomic_bytes(checkpoint, data.canonical({
            "identity": identity, "rows": rows,
            "rows_sha256": data.digest(data.canonical(rows)),
        }) + b"\n")

    def test_all_sequence_checkpoints_resume_without_graph_task(self):
        for index, row in enumerate(_rows()):
            self._write_checkpoint(index, row)
        with mock.patch.object(frozen, "NativeClient", _NoopNativeClient), mock.patch.object(
            frozen, "NativeGraphTask", side_effect=AssertionError(
                "complete sequence checkpoint must not reconstruct a graph task"
            )
        ):
            receipt = frozen.evaluate(
                self.corpus, self.reference, split="test",
                model=self.artifacts["model"], cli=self.artifacts["cli"],
                catalog=self.artifacts["catalog"], library=self.artifacts["library"],
                cold_cli=self.artifacts["cold_cli"], cpus=(25,), output=self.output,
                warmups=1, repeats=1,
            )
        self.assertEqual(receipt["pair_count"], 6)
        self.assertEqual(receipt["ok_pair_count"], 6)
        self.assertEqual(len((self.output / "worker_0.jsonl").read_text().splitlines()), 6)

    def test_preserved_frame_tamper_or_removal_invalidates_completed_receipt(self):
        _publish_completed(self.output,self.identity)
        path=self.output/_rows()[0]['frame_path']
        path.write_bytes(b'NGF1changed')
        with self.assertRaisesRegex(ValueError,'frame checksum'):
            frozen.verify_receipt(self.output,self.identity)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'frame is missing'):
            frozen.verify_receipt(self.output,self.identity)

    def test_resume_rejects_missing_frame_before_native_graph(self):
        for index,row in enumerate(_rows()):self._write_checkpoint(index,row)
        (self.output/_rows()[0]['frame_path']).unlink()
        with mock.patch.object(frozen,'NativeClient',_NoopNativeClient):
            with self.assertRaisesRegex(ValueError,'frame is missing'):
                frozen.evaluate(self.corpus,self.reference,split='test',
                    **self.artifacts,cpus=(25,),output=self.output,warmups=1,repeats=1)

    def test_checkpoint_cannot_substitute_source_row(self):
        row = copy.deepcopy(_rows()[0])
        row["block_id"] = "different-block"
        self._write_checkpoint(0, row)
        with mock.patch.object(frozen, "NativeClient", _NoopNativeClient), mock.patch.object(
            frozen, "NativeGraphTask", side_effect=AssertionError(
                "tampered checkpoint must be rejected before graph construction"
            )
        ):
            with self.assertRaisesRegex(ValueError, "substituted"):
                frozen.evaluate(
                    self.corpus, self.reference, split="test",
                    model=self.artifacts["model"], cli=self.artifacts["cli"],
                    catalog=self.artifacts["catalog"], library=self.artifacts["library"],
                    cold_cli=self.artifacts["cold_cli"], cpus=(25,), output=self.output,
                    warmups=1, repeats=1,
                )


if __name__ == "__main__":
    unittest.main()
