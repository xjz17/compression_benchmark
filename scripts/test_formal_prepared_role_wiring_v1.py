#!/usr/bin/env python3
"""Exercise role admission with real prepared files and real source hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import build_formal_weighted_grpo_gate_bundle_v1 as gate_builder
import formal_weighted_grpo_campaign_launcher_v1 as launcher
import formal_weighted_grpo_training_v1 as training
import measure_weighted_grpo_solution_seeds_v1 as seeds
import prepare_weighted_grpo_timeseries_v1 as preparation
import prepared_block_baseline86_v1 as baseline
import weighted_grpo_experiment_v1 as experiment
from test_weighted_grpo_experiment_v1 import RecordingFactory


def materialize(root: Path, entry: dict, cohort: dict, offset: int) -> Path:
    root.mkdir(parents=True)
    source = root / "source.csv"
    source.write_text(
        "ts,value\n" + "".join(
            f"{1700000000 + row},{offset + row}\n" for row in range(10)
        ), encoding="utf-8",
    )
    spec = {
        "format_version": preparation.FORMAT_VERSION,
        "dataset_id": entry["id"], "domain": entry["domain"],
        "source_group": entry["source_group"],
        "source": {
            "path": str(source), "format": "csv", "encoding": "utf-8",
            "delimiter": ",", "upstream_revision": "synthetic-contract-test",
            "upstream_locator": entry["locator"],
            "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "entity_columns": [],
        "timestamp": {
            "source_name": "ts", "output_name": "timestamp_unix",
            "source_kind": "epoch", "source_unit": "s", "output_unit": "s",
            "timezone": "UTC", "duplicate_policy": "reject",
        },
        "numeric_columns": [{"source_name": "value", "dtype": "int64"}],
        "missing_tokens": ["", "null"],
        "dataset_byte_cap": 160, "block_byte_cap": 128,
    }
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = root / "prepared"
    preparation.prepare_dataset(
        preparation.load_preparation_contract(spec_path), output,
        cohort_manifest=cohort,
    )
    return output


class FormalPreparedRoleWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.cohort = preparation.load_cohort_manifest()
        cls.train_entries = [entry for entry in cls.cohort["datasets"]
                             if entry["role"] == "train_validation"]
        cls.test_entry = next(entry for entry in cls.cohort["datasets"]
                              if entry["role"] == "test_candidate")
        cls.train = [materialize(cls.root / f"train-{index}", entry, cls.cohort, index * 100)
                     for index, entry in enumerate(cls.train_entries)]
        cls.test = materialize(cls.root / "test", cls.test_entry, cls.cohort, 10000)
        reference = {
            "schema": "weighted-grpo-zlib6-reference-table-v1",
            "objective_normalization": "E/(E+E0), D/(D+D0)",
            "reference_method": "native zlib level 6 complete ZR61 frame",
            "training_block_count": 32,
            "references": [{
                "dtype": "int64", "block_size_bucket_bytes": bucket,
                "E0_compression_ns_per_original_byte": 2.0,
                "D0_decompression_ns_per_original_byte": 3.0,
            } for bucket in (32, 128)],
        }
        reference["calibration_sha256"] = hashlib.sha256(
            experiment._canonical_json(reference)
        ).hexdigest()
        cls.reference = cls.root / "reference.json"
        cls.reference.write_text(json.dumps(reference), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_test_materialization_is_accepted_by_experiment_and_baseline(self):
        dataset = experiment._load_prepared_dataset(
            self.test, verify_source=True, expected_role="test_candidate",
        )
        metadata, blocks = baseline.load_prepared_blocks(self.test, splits=("test",))
        self.assertTrue(metadata["verification"]["source_verified"])
        self.assertEqual(len(blocks), 2)
        self.assertEqual({block.split for block in blocks}, {"test"})
        self.assertEqual(sum(block.row_count for block in blocks), 20)
        self.assertEqual(dataset.manifest["cohort"]["role"], "test_candidate")

    def test_train_and_seed_loaders_accept_complete_real_train_cohort(self):
        corpus = training._load_corpus(self.train, self.reference)
        self.assertEqual(len(corpus.train_blocks), 32)
        self.assertEqual(len(corpus.validation_blocks), 32)
        blocks, provenance = seeds._candidate_blocks(
            [self.train[0]], max_blocks_per_dtype=2,
        )
        self.assertEqual(len(provenance), 1)
        self.assertTrue(blocks)
        self.assertEqual({row["block"]["split"] for row in blocks}, {"train"})

    def test_real_test_manifest_runs_all_six_weights_without_verifier_mock(self):
        result = experiment.run_experiment(
            [self.test], self.reference, RecordingFactory(),
            config=experiment.ExperimentConfig(methods=("beam30",), splits=("test",)),
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 12)

    def test_test_data_is_rejected_by_training_and_seed_before_selection(self):
        for consumer in (
            lambda: training._load_corpus([self.test], self.reference),
            lambda: seeds._candidate_blocks([self.test], max_blocks_per_dtype=1),
        ):
            with self.subTest(consumer=consumer), self.assertRaisesRegex(
                experiment.ExperimentContractError, "verification failed",
            ):
                consumer()

    def test_train_data_and_mixed_split_request_are_rejected_by_test_baseline(self):
        with self.assertRaisesRegex(baseline.BaselineContractError, "verification failed"):
            baseline.load_prepared_blocks(self.train[0], splits=("test",))
        with self.assertRaisesRegex(baseline.BaselineContractError, "cannot be mixed"):
            baseline.load_prepared_blocks(self.test, splits=("test", "validation"))

    def _admit(self, test_path: Path):
        gate_path = self.root / "gate.json"
        gate_path.write_text("{}", encoding="utf-8")
        manifests = [json.loads((path / "dataset_manifest.json").read_text())
                     for path in self.train]
        spec = SimpleNamespace(
            repository_root=self.root, prepared=tuple(self.train),
            test_prepared=(test_path,), reference_table=self.reference,
            gate_bundle=gate_path,
        )
        gate = training.GateEvidence("source", "registry", (), "bundle")
        receipts = {"data": {
            "dataset_provenance_sha256": {
                row["dataset_id"]: row["provenance_sha256"] for row in manifests
            },
            "dataset_admission": {"source_spec_gate": {
                "complete_cohort_trainable": True, "blocked": 0,
            }},
        }}
        with mock.patch.object(training, "_load_gate_bundle", return_value=gate), \
                mock.patch.object(launcher, "_gate_receipts", return_value=receipts), \
                mock.patch.object(gate_builder, "discover_runtime_identity", return_value=(
                    SimpleNamespace(source_fingerprint="source",
                                    action_registry_fingerprint="registry", action_count=3)
                )):
            return launcher.admit_campaign(spec)

    def test_launcher_verifies_test_payload_and_joint_cohort(self):
        admission = self._admit(self.test)
        self.assertTrue(admission["passed"])
        self.assertEqual(admission["test_block_count"], 2)
        self.assertEqual(admission["combined_cohort_verification"]["role_counts"], {
            "train_validation": 16, "test_candidate": 1,
        })

    def test_launcher_rejects_cross_role_source_hash_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            reused = materialize(Path(directory) / "reused", self.test_entry, self.cohort, 0)
            with self.assertRaisesRegex(launcher.CampaignAdmissionError, "hash crosses"):
                self._admit(reused)

    def test_launcher_rejects_tampered_test_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = materialize(Path(directory) / "corrupt", self.test_entry, self.cohort, 90)
            manifest = json.loads((corrupt / "dataset_manifest.json").read_text())
            payload = corrupt / manifest["blocks"][0]["values_path"]
            data = bytearray(payload.read_bytes())
            data[0] ^= 1
            payload.write_bytes(data)
            with self.assertRaisesRegex(launcher.CampaignAdmissionError, "verification failed"):
                self._admit(corrupt)

    def test_launcher_rejects_test_source_family_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = materialize(Path(directory) / "family", self.test_entry, self.cohort, 91)
            path = corrupt / "dataset_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["source_group"] = self.train_entries[0]["source_group"]
            manifest["provenance_sha256"] = hashlib.sha256(
                preparation._canonical_json(preparation._manifest_fingerprint_basis(manifest))
            ).hexdigest()
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(launcher.CampaignAdmissionError, "source_group differs"):
                self._admit(corrupt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
