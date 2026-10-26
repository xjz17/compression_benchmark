#!/usr/bin/env python3
"""Tests for the independent THULab launch/completion gate validator."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
import sys
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_project179_thulab_launch_gate_v1 as gate  # noqa: E402


DATASET_CAP = 1024
CHUNK_CAP = 64
METADATA_CAP = 16384


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_bytes(gate._canonical_json(value) + b"\n")


class GateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ids = [*gate.PILOT_DATASET_IDS] + [
            f"dataset_{index:03d}" for index in range(4, gate.FULL_DATASET_COUNT)
        ]
        self.rows = [
            {"id": dataset_id, "repo_id": f"THULab/{dataset_id}", "revision": f"r-{index:03d}"}
            for index, dataset_id in enumerate(self.ids)
        ]
        self.scope = root / "scope.json"
        _write_json(self.scope, {
            "schema": gate.SCOPE_SCHEMA,
            "scope_id": "unit-project179",
            "timestamp_dataset_count": gate.FULL_DATASET_COUNT,
            "timestamp_datasets": self.rows,
        })
        self.policy = root / "policy.json"
        self.producer = root / "producer.py"
        self.helper_timeseries = root / "timeseries.py"
        self.helper_tsfile = root / "tsfile.py"
        self.policy.write_bytes(b'{"policy":"unit"}\n')
        self.producer.write_bytes(b"# producer\n")
        self.helper_timeseries.write_bytes(b"# timeseries\n")
        self.helper_tsfile.write_bytes(b"# tsfile\n")
        self.hashes = {
            "scope": gate.sha256_file(self.scope),
            "policy": gate.sha256_file(self.policy),
            "producer": gate.sha256_file(self.producer),
            "helper_timeseries": gate.sha256_file(self.helper_timeseries),
            "helper_tsfile": gate.sha256_file(self.helper_tsfile),
        }
        self.fixed, _scope, self.scope_rows = gate.bind_fixed_inputs(
            scope_path=self.scope,
            scope_sha256=self.hashes["scope"],
            policy_path=self.policy,
            policy_sha256=self.hashes["policy"],
            producer_path=self.producer,
            producer_sha256=self.hashes["producer"],
            helper_timeseries_path=self.helper_timeseries,
            helper_timeseries_sha256=self.hashes["helper_timeseries"],
            helper_tsfile_path=self.helper_tsfile,
            helper_tsfile_sha256=self.hashes["helper_tsfile"],
        )

    def plan_row(self, dataset_id: str) -> dict:
        row = next(item for item in self.rows if item["id"] == dataset_id)
        binding = _digest(f"binding:{dataset_id}".encode())
        policy = _digest(f"policy:{dataset_id}".encode())
        source = b"x"
        return {
            "dataset_id": dataset_id,
            "repo_id": row["repo_id"],
            "revision": row["revision"],
            "binding_sha256": binding,
            "policy_sha256": policy,
            "source_receipt_sha256": _digest(f"receipt:{dataset_id}".encode()),
            "source_files": [{"path": "source.tsfile", "bytes": len(source), "sha256": _digest(source)}],
            "source_validation": {
                "status": "validated_count_only",
                "source_rows": 1,
                "entity_count": 1,
                "source_rows_by_file": {"source.tsfile": 1},
                "payload_values_packed": False,
            },
            "candidate_stream_count": 1,
            "selected_stream_count": 1,
            "unselected_stream_count": 0,
            "selected_logical_bytes": 4,
            "longest_prefix_witness": {
                "all_candidates_selected": True,
                "first_unselected_logical_bytes": None,
                "proof_boundary": "all_candidates_selected",
            },
            "expected_payload": {
                "byte_upper_bound": 64,
                "shard_count_upper_bound": 1,
                "shard_byte_cap": CHUNK_CAP,
                "method": "unit",
            },
            "expected_stream_index": {
                "record_count": 4,
                "entity_record_count": 1,
                "stream_record_count": 1,
                "uncompressed_bytes_upper_bound": 2048,
                "compressed_bytes_conservative_skeleton": 128,
                "compressed_bytes_worst_case_upper_bound": 4096,
                "compressed_byte_cap": METADATA_CAP,
                "conservative_method": "unit",
            },
            "peak_temporary_bytes_upper_bound": 4096,
            "expected_filesystem_objects": {
                "regular_files": 3,
                "directories": 2,
                "total": 5,
                "formula": "payload_shards + stream_index + dataset_manifest + dataset_dir + payload_dir",
            },
            "layout_assertions": {
                "per_stream_payload_files": 0,
                "per_stream_payload_directories": 0,
                "payload_files_are_dataset_level_shards": True,
                "stream_metadata_is_in_single_compressed_index": True,
            },
        }

    @staticmethod
    def recalc_plan(receipt: dict) -> None:
        plans = receipt["datasets"]
        regular = sum(row["expected_filesystem_objects"]["regular_files"] for row in plans)
        directories = sum(row["expected_filesystem_objects"]["directories"] for row in plans)
        receipt["totals"] = {
            "candidate_stream_count": sum(row["candidate_stream_count"] for row in plans),
            "selected_stream_count": sum(row["selected_stream_count"] for row in plans),
            "selected_logical_bytes": sum(row["selected_logical_bytes"] for row in plans),
            "expected_payload_shard_count_upper_bound": sum(
                row["expected_payload"]["shard_count_upper_bound"] for row in plans
            ),
            "expected_index_uncompressed_bytes_upper_bound": sum(
                row["expected_stream_index"]["uncompressed_bytes_upper_bound"] for row in plans
            ),
            "expected_index_compressed_bytes_conservative_skeleton": sum(
                row["expected_stream_index"]["compressed_bytes_conservative_skeleton"] for row in plans
            ),
            "expected_index_compressed_bytes_worst_case_upper_bound": sum(
                row["expected_stream_index"]["compressed_bytes_worst_case_upper_bound"] for row in plans
            ),
            "peak_temporary_bytes_upper_bound": max(
                row["peak_temporary_bytes_upper_bound"] for row in plans
            ),
            "expected_collection_filesystem_objects": {
                "regular_files": regular + len(plans) + 1,
                "directories": directories + 1,
                "total": regular + len(plans) + 1 + directories + 1,
                "includes": "unit",
            },
            "per_stream_payload_files": 0,
            "per_stream_payload_directories": 0,
        }

    def make_plan(self, selection: str, path: Path, *, ids=None) -> dict:
        selected = list(ids if ids is not None else (
            gate.PILOT_DATASET_IDS if selection == "pilot" else self.ids
        ))
        receipt = {
            "schema": gate.PLAN_SCHEMA,
            "status": "complete",
            "mode": "count_only_no_payload_planning",
            "scope_id": "unit-project179",
            "scope_sha256": self.hashes["scope"],
            "policy_file_sha256": self.hashes["policy"],
            "mirror_manifest_sha256": "0" * 64,
            "selected_dataset_count": len(selected),
            "full_scope_dataset_count": gate.FULL_DATASET_COUNT,
            "selected_dataset_ids": selected,
            "caps": {
                "dataset_logical_byte_cap": DATASET_CAP,
                "payload_shard_byte_cap": CHUNK_CAP,
                "metadata_compressed_byte_cap": METADATA_CAP,
            },
            "storage_layout": gate.LAYOUT,
            "runtime_code_bindings": {
                "producer_sha256": self.hashes["producer"],
                "helper_timeseries_sha256": self.hashes["helper_timeseries"],
                "helper_tsfile_sha256": self.hashes["helper_tsfile"],
            },
            "datasets": [self.plan_row(dataset_id) for dataset_id in selected],
            "determinism": {"payload_materialized": False},
        }
        self.recalc_plan(receipt)
        _write_json(path, receipt)
        return receipt

    def validate_plan(self, path: Path, selection: str):
        return gate.validate_plan(
            path,
            selection=selection,
            fixed_inputs=self.fixed,
            scope_rows=self.scope_rows,
            dataset_byte_cap=DATASET_CAP,
            chunk_byte_cap=CHUNK_CAP,
            metadata_compressed_byte_cap=METADATA_CAP,
        )

    def make_output(self, output: Path, plan_receipt: dict, *, shard_count: int = 1) -> None:
        output.mkdir()
        receipts_dir = output / "_receipts"
        receipts_dir.mkdir()
        top_rows = []
        runtime = {
            "producer_sha256": self.hashes["producer"],
            "helper_timeseries_sha256": self.hashes["helper_timeseries"],
            "helper_tsfile_sha256": self.hashes["helper_tsfile"],
        }
        for plan in plan_receipt["datasets"]:
            dataset_id = plan["dataset_id"]
            dataset_dir = output / dataset_id
            payload_dir = dataset_dir / "payload"
            payload_dir.mkdir(parents=True)
            shards = []
            for index in range(shard_count):
                payload = bytes([index + 1]) * 4
                relative = f"payload/shard-{index:06d}.bin"
                path = dataset_dir / relative
                path.write_bytes(payload)
                shards.append({
                    "shard_index": index,
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": _digest(payload),
                })
            raw_index = b"{}\n"
            buffer = io.BytesIO()
            with gzip.GzipFile(filename="stream_index.jsonl", mode="wb", fileobj=buffer, mtime=0) as handle:
                handle.write(raw_index)
            compressed = buffer.getvalue()
            index_path = dataset_dir / "stream_index.jsonl.gz"
            index_path.write_bytes(compressed)
            index = {
                "schema": gate.STREAM_INDEX_SCHEMA,
                "path": "stream_index.jsonl.gz",
                "compression": "gzip",
                "record_encoding": "canonical_json_lines_utf8",
                "record_count": 4,
                "entity_record_count": 1,
                "stream_record_count": 1,
                "compressed_bytes": len(compressed),
                "compressed_sha256": _digest(compressed),
                "uncompressed_bytes": len(raw_index),
                "uncompressed_sha256": _digest(raw_index),
                "compressed_byte_cap": METADATA_CAP,
            }
            manifest = {
                "schema": gate.DATASET_SCHEMA,
                "status": "materialized",
                "storage_layout": gate.LAYOUT,
                "dataset_id": dataset_id,
                "repo_id": plan["repo_id"],
                "revision": plan["revision"],
                "binding_sha256": plan["binding_sha256"],
                "scope_id": "unit-project179",
                "scope_sha256": self.hashes["scope"],
                "source_receipt_sha256": plan["source_receipt_sha256"],
                "runtime_code_bindings": runtime,
                "source_tsfiles": plan["source_files"],
                "timestamp_policy": {
                    "policy_file_sha256": self.hashes["policy"],
                    "policy_sha256": plan["policy_sha256"],
                },
                "selection": {
                    "dataset_byte_cap": DATASET_CAP,
                    "candidate_stream_count": 1,
                    "selected_stream_count": 1,
                    "selected_logical_bytes": 4,
                    "unselected_stream_count": 0,
                    "chunk_byte_cap": CHUNK_CAP,
                    "metadata_compressed_byte_cap": METADATA_CAP,
                },
                "totals": {
                    "stream_count": 1,
                    "chunk_count": 1,
                    "rows": 1,
                    "valid_values": 1,
                    "missing_values": 0,
                    "logical_bytes": 4,
                },
                "stream_index": index,
                "payload_shards": shards,
            }
            manifest["provenance_sha256"] = _digest(gate._canonical_json(manifest))
            manifest_path = dataset_dir / "dataset_manifest.json"
            _write_json(manifest_path, manifest)
            receipt = {
                "schema": gate.DATASET_RECEIPT_SCHEMA,
                "status": "materialized",
                "dataset_id": dataset_id,
                "repo_id": plan["repo_id"],
                "revision": plan["revision"],
                "binding_sha256": plan["binding_sha256"],
                "scope_id": "unit-project179",
                "scope_sha256": self.hashes["scope"],
                "source_receipt_sha256": plan["source_receipt_sha256"],
                "runtime_code_bindings": runtime,
                "policy_file_sha256": self.hashes["policy"],
                "policy_sha256": plan["policy_sha256"],
                "storage_layout": gate.LAYOUT,
                "output_directory": dataset_id,
                "manifest_sha256": gate.sha256_file(manifest_path),
                "selected_stream_count": 1,
                "selected_logical_bytes": 4,
                "candidate_stream_count": 1,
                "files": [dict(plan["source_files"][0]) | {"status": "verified_and_materialized"}],
            }
            receipt_path = receipts_dir / f"{dataset_id}.json"
            _write_json(receipt_path, receipt)
            top_rows.append({
                "dataset_id": dataset_id,
                "status": "materialized",
                "receipt": f"_receipts/{dataset_id}.json",
                "receipt_sha256": gate.sha256_file(receipt_path),
            })
        top = {
            "schema": gate.COLLECTION_SCHEMA,
            "status": "complete",
            "mode": "prepare_or_resume",
            "scope_id": "unit-project179",
            "scope_sha256": self.hashes["scope"],
            "policy_file_sha256": self.hashes["policy"],
            "selected_dataset_count": len(plan_receipt["datasets"]),
            "full_scope_dataset_count": gate.FULL_DATASET_COUNT,
            "status_counts": {"materialized": len(plan_receipt["datasets"]), "blocked": 0},
            "dataset_byte_cap": DATASET_CAP,
            "chunk_byte_cap": CHUNK_CAP,
            "metadata_compressed_byte_cap": METADATA_CAP,
            "storage_layout": gate.LAYOUT,
            "runtime_code_bindings": runtime,
            "datasets": top_rows,
        }
        _write_json(output / "MATERIALIZATION_MANIFEST.json", top)

    def launch(self, pilot_plan: Path, full_plan: Path, pilot_output: Path, receipt: Path):
        return gate.build_launch_gate(
            pilot_planning_receipt=pilot_plan,
            full_planning_receipt=full_plan,
            pilot_output=pilot_output,
            output_receipt=receipt,
            fixed_inputs=self.fixed,
            scope_rows=self.scope_rows,
            dataset_byte_cap=DATASET_CAP,
            chunk_byte_cap=CHUNK_CAP,
            metadata_compressed_byte_cap=METADATA_CAP,
        )


class Project179ThulabGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = GateFixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _launch_fixture(self, *, shard_count=1):
        pilot_path = self.root / "pilot-plan.json"
        full_path = self.root / "full-plan.json"
        pilot = self.fixture.make_plan("pilot", pilot_path)
        self.fixture.make_plan("full", full_path)
        output = self.root / "pilot-output"
        self.fixture.make_output(output, pilot, shard_count=shard_count)
        return pilot_path, full_path, output

    def test_validate_pilot_and_full_plans(self):
        pilot_path = self.root / "pilot.json"
        full_path = self.root / "full.json"
        self.fixture.make_plan("pilot", pilot_path)
        self.fixture.make_plan("full", full_path)
        self.assertEqual(len(self.fixture.validate_plan(pilot_path, "pilot")["datasets"]), 4)
        self.assertEqual(len(self.fixture.validate_plan(full_path, "full")["datasets"]), 160)

    def test_rejects_pilot_order_change(self):
        path = self.root / "pilot.json"
        ids = list(gate.PILOT_DATASET_IDS)
        ids[0], ids[1] = ids[1], ids[0]
        self.fixture.make_plan("pilot", path, ids=ids)
        with self.assertRaisesRegex(gate.GateError, "dataset order/content"):
            self.fixture.validate_plan(path, "pilot")

    def test_rejects_index_worst_case_above_metadata_cap(self):
        path = self.root / "pilot.json"
        receipt = self.fixture.make_plan("pilot", path)
        receipt["datasets"][0]["expected_stream_index"][
            "compressed_bytes_worst_case_upper_bound"
        ] = METADATA_CAP + 1
        GateFixture.recalc_plan(receipt)
        _write_json(path, receipt)
        with self.assertRaisesRegex(gate.GateError, "invalid stream-index plan"):
            self.fixture.validate_plan(path, "pilot")

    def test_rejects_non_160_frozen_scope(self):
        scope = json.loads(self.fixture.scope.read_text())
        scope["timestamp_datasets"].pop()
        scope["timestamp_dataset_count"] = 159
        _write_json(self.fixture.scope, scope)
        with self.assertRaisesRegex(gate.GateError, "exactly 160"):
            gate.bind_fixed_inputs(
                scope_path=self.fixture.scope,
                scope_sha256=gate.sha256_file(self.fixture.scope),
                policy_path=self.fixture.policy,
                policy_sha256=self.fixture.hashes["policy"],
                producer_path=self.fixture.producer,
                producer_sha256=self.fixture.hashes["producer"],
                helper_timeseries_path=self.fixture.helper_timeseries,
                helper_timeseries_sha256=self.fixture.hashes["helper_timeseries"],
                helper_tsfile_path=self.fixture.helper_tsfile,
                helper_tsfile_sha256=self.fixture.hashes["helper_tsfile"],
            )

    def test_rejects_fixed_input_hash_mismatch(self):
        with self.assertRaisesRegex(gate.GateError, "producer SHA-256 mismatch"):
            gate.bind_fixed_inputs(
                scope_path=self.fixture.scope,
                scope_sha256=self.fixture.hashes["scope"],
                policy_path=self.fixture.policy,
                policy_sha256=self.fixture.hashes["policy"],
                producer_path=self.fixture.producer,
                producer_sha256="f" * 64,
                helper_timeseries_path=self.fixture.helper_timeseries,
                helper_timeseries_sha256=self.fixture.hashes["helper_timeseries"],
                helper_tsfile_path=self.fixture.helper_tsfile,
                helper_tsfile_sha256=self.fixture.hashes["helper_tsfile"],
            )

    def test_build_launch_gate_and_refuse_clobber(self):
        pilot, full, output = self._launch_fixture()
        receipt_path = self.root / "launch-gate.json"
        receipt, target = self.fixture.launch(pilot, full, output, receipt_path)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(target, receipt_path.resolve())
        self.assertEqual(receipt["pilot"]["dataset_count"], 4)
        self.assertTrue(receipt["pilot"]["validation"]["exact_tree"])
        self.assertEqual(
            receipt["validator"]["sha256"],
            gate.sha256_file(Path(gate.__file__).resolve()),
        )
        binding = receipt["input_bindings"]
        self.assertEqual(
            receipt["input_bindings_sha256"], _digest(gate._canonical_json(binding))
        )
        with self.assertRaisesRegex(gate.GateError, "already exists"):
            self.fixture.launch(pilot, full, output, receipt_path)

    def test_rejects_pilot_full_mirror_manifest_mismatch(self):
        pilot, full, output = self._launch_fixture()
        full_receipt = json.loads(full.read_text())
        full_receipt["mirror_manifest_sha256"] = "f" * 64
        _write_json(full, full_receipt)
        with self.assertRaisesRegex(gate.GateError, "mirror-manifest bindings differ"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_pilot_full_dataset_plan_mismatch(self):
        pilot, full, output = self._launch_fixture()
        full_receipt = json.loads(full.read_text())
        full_receipt["datasets"][0]["binding_sha256"] = "f" * 64
        _write_json(full, full_receipt)
        with self.assertRaisesRegex(gate.GateError, "dataset bindings differ"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_pilot_full_omitted_field_plan_mismatch(self):
        pilot, full, output = self._launch_fixture()
        pilot_receipt = json.loads(pilot.read_text())
        pilot_receipt["datasets"][0]["peak_temporary_bytes_upper_bound"] += 123
        pilot_receipt["datasets"][0]["source_validation"]["source_rows"] = 999999999
        GateFixture.recalc_plan(pilot_receipt)
        _write_json(pilot, pilot_receipt)
        with self.assertRaisesRegex(gate.GateError, "dataset bindings differ"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_plan_actual_binding_mismatch(self):
        pilot, full, output = self._launch_fixture()
        receipt = json.loads(pilot.read_text())
        receipt["datasets"][0]["binding_sha256"] = "f" * 64
        GateFixture.recalc_plan(receipt)
        _write_json(pilot, receipt)
        full_receipt = json.loads(full.read_text())
        full_receipt["datasets"][0]["binding_sha256"] = "f" * 64
        GateFixture.recalc_plan(full_receipt)
        _write_json(full, full_receipt)
        with self.assertRaisesRegex(gate.GateError, "plan/receipt mismatch"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_blocked_collection(self):
        pilot, full, output = self._launch_fixture()
        top_path = output / "MATERIALIZATION_MANIFEST.json"
        top = json.loads(top_path.read_text())
        top["status"] = "complete_with_blocked_datasets"
        top["status_counts"] = {"materialized": 3, "blocked": 1}
        _write_json(top_path, top)
        with self.assertRaisesRegex(gate.GateError, "binding/status"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_extra_pilot_file(self):
        pilot, full, output = self._launch_fixture()
        (output / "unexpected.txt").write_text("unexpected")
        with self.assertRaisesRegex(gate.GateError, "not an exact tree"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_symlink_in_pilot_tree(self):
        pilot, full, output = self._launch_fixture()
        (output / "link").symlink_to(output / "MATERIALIZATION_MANIFEST.json")
        with self.assertRaisesRegex(gate.GateError, "contains a symlink"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_shards_above_plan_bound(self):
        pilot, full, output = self._launch_fixture(shard_count=2)
        with self.assertRaisesRegex(gate.GateError, "shard count exceeds plan"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_rejects_index_above_plan_bound(self):
        pilot, full, output = self._launch_fixture()
        receipt = json.loads(pilot.read_text())
        for row in receipt["datasets"]:
            row["expected_stream_index"]["uncompressed_bytes_upper_bound"] = 3
            row["expected_stream_index"]["compressed_bytes_conservative_skeleton"] = 3
            row["expected_stream_index"]["compressed_bytes_worst_case_upper_bound"] = 3
        GateFixture.recalc_plan(receipt)
        _write_json(pilot, receipt)
        full_receipt = json.loads(full.read_text())
        for row in full_receipt["datasets"]:
            if row["dataset_id"] in gate.PILOT_DATASET_IDS:
                row["expected_stream_index"]["uncompressed_bytes_upper_bound"] = 3
                row["expected_stream_index"]["compressed_bytes_conservative_skeleton"] = 3
                row["expected_stream_index"]["compressed_bytes_worst_case_upper_bound"] = 3
        GateFixture.recalc_plan(full_receipt)
        _write_json(full, full_receipt)
        with self.assertRaisesRegex(gate.GateError, "compressed binding/bound"):
            self.fixture.launch(pilot, full, output, self.root / "gate.json")

    def test_completion_gate_validates_full_160(self):
        full_path = self.root / "full-plan.json"
        full_receipt = self.fixture.make_plan("full", full_path)
        output = self.root / "formal-output"
        self.fixture.make_output(output, full_receipt)
        receipt, target = gate.build_completion_gate(
            full_planning_receipt=full_path,
            formal_output=output,
            output_receipt=self.root / "completion.json",
            fixed_inputs=self.fixture.fixed,
            scope_rows=self.fixture.scope_rows,
            dataset_byte_cap=DATASET_CAP,
            chunk_byte_cap=CHUNK_CAP,
            metadata_compressed_byte_cap=METADATA_CAP,
        )
        self.assertEqual(receipt["formal"]["dataset_count"], 160)
        self.assertEqual(receipt["formal"]["blocked_dataset_count"], 0)
        self.assertEqual(
            receipt["validator"]["sha256"],
            gate.sha256_file(Path(gate.__file__).resolve()),
        )
        self.assertTrue(target.is_file())

    def test_rejects_gate_receipt_inside_materialization(self):
        pilot, full, output = self._launch_fixture()
        with self.assertRaisesRegex(gate.GateError, "outside every materialization"):
            self.fixture.launch(pilot, full, output, output / "gate.json")

    def test_rejects_duplicate_json_keys(self):
        path = self.root / "bad.json"
        path.write_text('{"schema":"a","schema":"b"}\n')
        with self.assertRaisesRegex(gate.GateError, "duplicate JSON key"):
            gate._load_json(path, "bad")

    def test_cli_validate_plan(self):
        path = self.root / "pilot.json"
        self.fixture.make_plan("pilot", path)
        args = [
            "validate-plan",
            "--scope", str(self.fixture.scope),
            "--scope-sha256", self.fixture.hashes["scope"],
            "--policy", str(self.fixture.policy),
            "--policy-sha256", self.fixture.hashes["policy"],
            "--producer", str(self.fixture.producer),
            "--producer-sha256", self.fixture.hashes["producer"],
            "--helper-timeseries", str(self.fixture.helper_timeseries),
            "--helper-timeseries-sha256", self.fixture.hashes["helper_timeseries"],
            "--helper-tsfile", str(self.fixture.helper_tsfile),
            "--helper-tsfile-sha256", self.fixture.hashes["helper_tsfile"],
            "--dataset-byte-cap", str(DATASET_CAP),
            "--chunk-byte-cap", str(CHUNK_CAP),
            "--metadata-compressed-byte-cap", str(METADATA_CAP),
            "--planning-receipt", str(path),
            "--selection", "pilot",
        ]
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(gate.main(args), 0)
        self.assertEqual(json.loads(output.getvalue())["dataset_count"], 4)


if __name__ == "__main__":
    unittest.main()
