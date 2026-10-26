#!/usr/bin/env python3

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/build_project179_numeric_utc_contract_v1.py"
SPEC = importlib.util.spec_from_file_location("project179_contract", MODULE_PATH)
assert SPEC and SPEC.loader
contract_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract_module)


class Project179NumericUtcContractTest(unittest.TestCase):
    def build(self, **overrides):
        arguments = {
            "scope_path": contract_module.DEFAULT_SCOPE,
            "core_policy_path": contract_module.DEFAULT_CORE_POLICY,
            "thulab_policy_path": contract_module.DEFAULT_THULAB_POLICY,
            "core_producer_path": contract_module.DEFAULT_CORE_PRODUCER,
            "core_auditor_path": contract_module.DEFAULT_CORE_AUDITOR,
            "thulab_producer_path": contract_module.DEFAULT_THULAB_PRODUCER,
            "thulab_helper_timeseries_path": (
                contract_module.DEFAULT_THULAB_HELPER_TIMESERIES
            ),
            "thulab_helper_tsfile_path": (
                contract_module.DEFAULT_THULAB_HELPER_TSFILE
            ),
        }
        arguments.update(overrides)
        return contract_module.build_contract(**arguments)

    def build_core_evidence_fixture(self, directory: Path) -> dict[str, object]:
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        scope = json.loads(contract_module.DEFAULT_SCOPE.read_text(encoding="utf-8"))
        core_ids = list(map(str, scope["core_csv_datasets"]))
        policy_sha256 = contract_module.sha256_file(contract_module.DEFAULT_CORE_POLICY)
        producer_sha256 = contract_module.sha256_file(contract_module.DEFAULT_CORE_PRODUCER)
        output = directory / "core-output"
        active_root = directory / "active"
        backup_root = directory / "backup"
        active_root.mkdir()
        backup_root.mkdir()
        rows = []
        audit_rows = []
        totals = {
            "timestamp_admitted": 0,
            "timestamp_selected": 0,
            "timestamp_blocked": 0,
            "time_only_dropped": 0,
            "remaining_strings_dropped": 0,
            "numeric_candidates": 0,
            "numeric_selected": 0,
            "chunks": 0,
            "chunk_bytes": 0,
        }
        for dataset_id in core_ids:
            dataset_path = output / "datasets" / dataset_id / "manifest.json"
            dataset_path.parent.mkdir(parents=True)
            dataset = {
                "schema": "project179-core19-numeric-utc-dataset-v1",
                "status": "complete",
                "dataset": dataset_id,
                "identity": {
                    "policy_sha256": policy_sha256,
                    "producer_sha256": producer_sha256,
                },
                "selection": {"limit_bytes": 1 << 30, "selected_bytes": 0},
                "columns": [],
                "counts": totals,
            }
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            dataset_binding = contract_module._file_binding(dataset_path)
            rows.append({
                "dataset": dataset_id,
                "path": str(dataset_path.resolve()),
                "sha256": dataset_binding["sha256"],
                "status": "complete",
                "counts": totals,
            })
            audit_rows.append({
                "dataset": dataset_id,
                "status": "PASS",
                "output_manifest": dataset_binding,
                "timestamp_admitted": 0,
                "timestamp_selected": 0,
                "timestamp_blocked": 0,
                "candidate_columns": 0,
                "selected_columns": 0,
                "selected_bytes": 0,
                "chunks": 0,
                "chunk_bytes": 0,
            })
        manifest_path = output / "manifest.json"
        manifest = {
            "schema": "project179-core19-numeric-utc-scope-v1",
            "status": "complete",
            "dataset_count": 19,
            "datasets": core_ids,
            "policy_sha256": policy_sha256,
            "limit_bytes": 1 << 30,
            "producer": {
                "path": str(contract_module.DEFAULT_CORE_PRODUCER.resolve()),
                "sha256": producer_sha256,
            },
            "active_root": str(active_root),
            "backup_root": str(backup_root),
            "dataset_manifests": rows,
            "totals": totals,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        zstd_path = directory / "zstd"
        zstd_path.write_bytes(b"fixture zstd\n")
        active_top = active_root / "manifest.json"
        active_top.write_text("{}\n", encoding="utf-8")
        active_contract = active_root / "SOURCE_CONTRACT.json"
        active_contract.write_text(json.dumps({
            "active_manifest_path": str(active_top),
            "active_manifest_sha256": contract_module.sha256_file(active_top),
        }), encoding="utf-8")
        bindings = {
            "scope": contract_module._file_binding(contract_module.DEFAULT_SCOPE),
            "policy": contract_module._file_binding(contract_module.DEFAULT_CORE_POLICY),
            "producer": contract_module._file_binding(contract_module.DEFAULT_CORE_PRODUCER),
            "auditor": contract_module._file_binding(contract_module.DEFAULT_CORE_AUDITOR),
            "zstd": contract_module._file_binding(zstd_path),
            "active_source_contract": contract_module._file_binding(active_contract),
            "active_top_manifest": contract_module._file_binding(active_top),
        }
        receipt = {
            "schema": contract_module.CORE_AUDIT_SCHEMA,
            "status": "PASS",
            "mode": "cold_read_only",
            "scope_id": scope["scope_id"],
            "input_bindings": bindings,
            "output_binding": {
                "root": str(output.resolve()),
                "top_manifest": contract_module._file_binding(manifest_path),
                "eligible_file_count": 20,
            },
            "dataset_count": 19,
            "datasets": audit_rows,
            "totals": totals,
            "completed_at": "2026-09-28T00:00:00+00:00",
            "release_checks": {
                **{name: True for name in contract_module.CORE_AUDIT_TRUE_CHECKS},
                "timestamp_blocked": 0,
                "limit_bytes_per_dataset": 1 << 30,
            },
        }
        receipt_path = directory / "core19.independent-audit.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return {
            "manifest": manifest_path,
            "receipt": receipt_path,
            "receipt_value": receipt,
            "active_root": active_root,
            "backup_root": backup_root,
            "zstd": zstd_path,
        }

    @staticmethod
    def core_materialization_arguments(fixture: dict[str, object]) -> dict[str, Path]:
        return {
            "core_manifest_path": fixture["manifest"],
            "core_audit_receipt_path": fixture["receipt"],
            "core_active_root": fixture["active_root"],
            "core_backup_root": fixture["backup_root"],
            "core_zstd_path": fixture["zstd"],
        }

    @staticmethod
    def core_gate_patch(
        fixture: dict[str, object], *, fresh_receipt: dict | None = None,
    ) -> tuple[mock._patch, mock.Mock]:
        zstd_path = fixture["zstd"]
        assert isinstance(zstd_path, Path)
        if fresh_receipt is None:
            fresh_receipt = copy.deepcopy(fixture["receipt_value"])
            fresh_receipt["completed_at"] = "2026-09-28T00:00:01+00:00"
        rerun = mock.Mock(return_value=fresh_receipt)
        return mock.patch.multiple(
            contract_module,
            EXPECTED_CORE_ZSTD_PATH=str(zstd_path),
            EXPECTED_CORE_ZSTD_SHA256=contract_module.sha256_file(zstd_path),
            _run_fresh_core_audit=rerun,
        ), rerun

    def test_contract_only_is_explicitly_not_a_completion_claim(self) -> None:
        value = self.build()
        contract_module.validate_contract(value)
        self.assertEqual(value["dataset_identity_counts"]["total"], 179)
        self.assertEqual(value["evidence_state"], "contract_only_no_materialization_claim")
        self.assertEqual(value["server166_materialization"]["verified_dataset_count"], 0)
        self.assertFalse(value["server166_materialization"]["core19"]["completion_claim"])
        self.assertFalse(value["server192_propagation"]["completion_claim"])

    def test_contract_binds_live_inputs_by_sha256(self) -> None:
        value = self.build()
        expected = {
            "scope": contract_module.DEFAULT_SCOPE,
            "core19_policy": contract_module.DEFAULT_CORE_POLICY,
            "thulab_policy": contract_module.DEFAULT_THULAB_POLICY,
            "core19_producer": contract_module.DEFAULT_CORE_PRODUCER,
            "core19_auditor": contract_module.DEFAULT_CORE_AUDITOR,
            "thulab_producer": contract_module.DEFAULT_THULAB_PRODUCER,
            "thulab_helper_timeseries": (
                contract_module.DEFAULT_THULAB_HELPER_TIMESERIES
            ),
            "thulab_helper_tsfile": contract_module.DEFAULT_THULAB_HELPER_TSFILE,
        }
        for name, path in expected.items():
            self.assertEqual(
                value["input_bindings"][name]["sha256"],
                contract_module.sha256_file(path),
            )

    def test_runner_hard_pins_all_release_inputs_and_passes_fresh_gate_paths(self) -> None:
        runner_path = ROOT / "scripts/run_project179_numeric_utc_166.sh"
        runner = runner_path.read_text(encoding="utf-8")
        expected = {
            "EXPECTED_SCOPE_SHA256": contract_module.DEFAULT_SCOPE,
            "EXPECTED_CORE_POLICY_SHA256": contract_module.DEFAULT_CORE_POLICY,
            "EXPECTED_THULAB_POLICY_SHA256": contract_module.DEFAULT_THULAB_POLICY,
            "EXPECTED_CORE_PRODUCER_SHA256": contract_module.DEFAULT_CORE_PRODUCER,
            "EXPECTED_CORE_AUDITOR_SHA256": contract_module.DEFAULT_CORE_AUDITOR,
            "EXPECTED_THULAB_PRODUCER_SHA256": contract_module.DEFAULT_THULAB_PRODUCER,
            "EXPECTED_THULAB_HELPER_TIMESERIES_SHA256": (
                contract_module.DEFAULT_THULAB_HELPER_TIMESERIES
            ),
            "EXPECTED_THULAB_HELPER_TSFILE_SHA256": (
                contract_module.DEFAULT_THULAB_HELPER_TSFILE
            ),
            "EXPECTED_CONTRACT_BUILDER_SHA256": contract_module.__file__,
        }
        for name, path in expected.items():
            match = re.search(rf"^readonly {name}=([0-9a-f]{{64}})$", runner, re.MULTILINE)
            self.assertIsNotNone(match, name)
            self.assertEqual(match.group(1), contract_module.sha256_file(Path(path)))
        self.assertIn(
            f"readonly EXPECTED_ZSTD_SHA256={contract_module.EXPECTED_CORE_ZSTD_SHA256}",
            runner,
        )
        for argument in ("--core-active-root", "--core-backup-root", "--core-zstd"):
            self.assertIn(argument, runner)

    def test_thulab_dataset_binding_hash_includes_both_helpers(self) -> None:
        runtime = {
            "producer_sha256": "a" * 64,
            "helper_timeseries_sha256": "b" * 64,
            "helper_tsfile_sha256": "c" * 64,
        }
        dataset = {
            "dataset_id": "fixture",
            "repo_id": "THULab/fixture",
            "revision": "d" * 40,
            "storage_layout": "sharded_v1",
            "source_tsfiles": [
                {"path": "part.tsfile", "bytes": 1, "sha256": "e" * 64}
            ],
            "selection": {
                "dataset_byte_cap": 1 << 30,
                "chunk_byte_cap": 64 << 20,
                "metadata_compressed_byte_cap": 1 << 30,
            },
        }
        receipt = {"source_receipt_sha256": "f" * 64}
        original = contract_module._expected_thulab_binding_sha256(
            dataset=dataset, receipt=receipt, scope_id="scope",
            scope_sha256="1" * 64, policy_file_sha256="2" * 64,
            policy_sha256="3" * 64, runtime=runtime,
        )
        changed = dict(runtime)
        changed["helper_tsfile_sha256"] = "4" * 64
        rebound = contract_module._expected_thulab_binding_sha256(
            dataset=dataset, receipt=receipt, scope_id="scope",
            scope_sha256="1" * 64, policy_file_sha256="2" * 64,
            policy_sha256="3" * 64, runtime=changed,
        )
        self.assertNotEqual(original, rebound)

    def test_semantic_claims_are_kept_separate(self) -> None:
        value = self.build()
        semantic = value["derived_view_contract"]["timestamp_semantics"]
        self.assertGreater(semantic["core19"]["exact_event_utc_instant_definition_count"], 0)
        self.assertGreater(semantic["core19"]["coordinate_only_definition_count"], 0)
        self.assertEqual(
            semantic["thulab"]["primary_exact_event_utc_dataset_count"]
            + semantic["thulab"]["primary_stored_coordinate_dataset_count"]
            + semantic["thulab"]["primary_dataset_adapter_count"]
            + semantic["thulab"]["primary_blocked_dataset_count"],
            160,
        )
        self.assertEqual(semantic["thulab"]["primary_dataset_adapter_count"], 3)
        self.assertEqual(semantic["thulab"]["primary_blocked_dataset_count"], 0)

    def test_builder_rejects_unknown_thulab_policy_decision_fault_injection(self) -> None:
        policy = json.loads(
            contract_module.DEFAULT_THULAB_POLICY.read_text(encoding="utf-8")
        )
        policy["datasets"][0]["decision"] = "admit_unknown_fault_injection"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thulab-policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(
                contract_module.ContractError, "unsupported decision"
            ):
                self.build(thulab_policy_path=path)

    def test_validator_rejects_unknown_primary_decision_category(self) -> None:
        value = self.build()
        tampered = copy.deepcopy(value)
        tampered["derived_view_contract"]["timestamp_semantics"]["thulab"][
            "primary_unknown_dataset_count"
        ] = 0
        with self.assertRaisesRegex(
            contract_module.ContractError, "categories are not exhaustive"
        ):
            contract_module.validate_contract(tampered)

    def test_validator_rejects_primary_decision_count_not_equal_to_160(self) -> None:
        value = self.build()
        tampered = copy.deepcopy(value)
        tampered["derived_view_contract"]["timestamp_semantics"]["thulab"][
            "primary_exact_event_utc_dataset_count"
        ] -= 1
        with self.assertRaisesRegex(contract_module.ContractError, "sum to exactly 160"):
            contract_module.validate_contract(tampered)

    def test_validator_rejects_materialization_overclaim(self) -> None:
        value = self.build()
        tampered = copy.deepcopy(value)
        tampered["evidence_state"] = "materialized_and_verified"
        with self.assertRaisesRegex(contract_module.ContractError, "overclaims"):
            contract_module.validate_contract(tampered)

    def test_builder_rejects_scope_count_tampering(self) -> None:
        scope = json.loads(contract_module.DEFAULT_SCOPE.read_text(encoding="utf-8"))
        scope["core_csv_datasets"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            path.write_text(json.dumps(scope), encoding="utf-8")
            with self.assertRaisesRegex(contract_module.ContractError, "19 Core"):
                self.build(scope_path=path)

    def test_builder_rejects_unverified_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_core_evidence_fixture(Path(directory))
            fixture["manifest"].write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            gate_patch, _rerun = self.core_gate_patch(fixture)
            with gate_patch, self.assertRaisesRegex(
                contract_module.ContractError, "Core19 materialization"
            ):
                self.build(**self.core_materialization_arguments(fixture))

    def test_core_manifest_and_audit_receipt_must_be_supplied_together(self) -> None:
        missing = Path("missing-core-evidence.json")
        with self.assertRaisesRegex(contract_module.ContractError, "supplied together"):
            self.build(core_manifest_path=missing)
        with self.assertRaisesRegex(contract_module.ContractError, "supplied together"):
            self.build(core_audit_receipt_path=missing)

    def test_valid_core_manifest_and_independent_audit_receipt_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_core_evidence_fixture(Path(directory))
            gate_patch, rerun = self.core_gate_patch(fixture)
            with gate_patch:
                value = self.build(**self.core_materialization_arguments(fixture))
                with self.assertRaisesRegex(
                    contract_module.ContractError, "verification authority"
                ):
                    contract_module.validate_contract(value)
                contract_module.validate_contract(value, materialization_verified=True)
            rerun.assert_called_once()
            evidence = value["server166_materialization"]["core19"]
            self.assertEqual(evidence["state"], "materialized_and_verified")
            self.assertTrue(evidence["completion_claim"])
            self.assertEqual(
                evidence["independent_audit_receipt"]["sha256"],
                contract_module.sha256_file(fixture["receipt"]),
            )
            self.assertEqual(
                evidence["auditor_sha256"],
                contract_module.sha256_file(contract_module.DEFAULT_CORE_AUDITOR),
            )
            receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
            self.assertEqual(
                evidence["zstd_sha256"],
                receipt["input_bindings"]["zstd"]["sha256"],
            )
            self.assertEqual(value["evidence_state"], "partially_attached")

    def test_cli_accepts_explicit_core_auditor_and_audit_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_core_evidence_fixture(root)
            output_path = root / "contract.json"
            gate_patch, rerun = self.core_gate_patch(fixture)
            with gate_patch, redirect_stdout(io.StringIO()):
                code = contract_module.main([
                    "--core-auditor", str(contract_module.DEFAULT_CORE_AUDITOR),
                    "--core-manifest", str(fixture["manifest"]),
                    "--core-audit-receipt", str(fixture["receipt"]),
                    "--core-active-root", str(fixture["active_root"]),
                    "--core-backup-root", str(fixture["backup_root"]),
                    "--core-zstd", str(fixture["zstd"]),
                    "--output", str(output_path),
                ])
            rerun.assert_called_once()
            self.assertEqual(code, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                written["server166_materialization"]["core19"]["state"],
                "materialized_and_verified",
            )

    def test_tampered_core_audit_receipt_is_rejected(self) -> None:
        cases = {
            "receipt": lambda receipt: receipt.__setitem__("status", "FAIL"),
            "hash": lambda receipt: receipt["input_bindings"]["scope"].__setitem__(
                "sha256", "0" * 64
            ),
            "auditor_hash": lambda receipt: receipt["input_bindings"][
                "auditor"
            ].__setitem__("sha256", "2" * 64),
            "path": lambda receipt: receipt["output_binding"].__setitem__(
                "root", "/tampered/core-output"
            ),
            "top_manifest_path": lambda receipt: receipt["output_binding"][
                "top_manifest"
            ].__setitem__("path", "/tampered/manifest.json"),
            "top_manifest_hash": lambda receipt: receipt["output_binding"][
                "top_manifest"
            ].__setitem__("sha256", "1" * 64),
            "checks": lambda receipt: receipt["release_checks"].__setitem__(
                "all_source_output_and_chunk_hashes_streamed", False
            ),
            "extra_check": lambda receipt: receipt["release_checks"].__setitem__(
                "unreviewed_future_check", True
            ),
            "missing_final_rescan_check": lambda receipt: receipt["release_checks"].pop(
                "final_output_tree_rescan_after_all_stable_bindings"
            ),
            "limit": lambda receipt: receipt["release_checks"].__setitem__(
                "limit_bytes_per_dataset", (1 << 30) - 1
            ),
            "blocked": lambda receipt: receipt["release_checks"].__setitem__(
                "timestamp_blocked", 1
            ),
            "dataset_identities": lambda receipt: receipt["datasets"][0].__setitem__(
                "dataset", "tampered"
            ),
            "dataset_output_manifest": lambda receipt: receipt["datasets"][0][
                "output_manifest"
            ].__setitem__("sha256", "3" * 64),
            "dataset_counts": lambda receipt: receipt["datasets"][0].__setitem__(
                "selected_columns", 1
            ),
            "totals": lambda receipt: receipt["totals"].__setitem__(
                "numeric_selected", 1
            ),
            "zstd_binding": lambda receipt: receipt["input_bindings"].pop("zstd"),
        }
        for name, tamper in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_core_evidence_fixture(Path(directory))
                receipt_path = fixture["receipt"]
                fresh_receipt = copy.deepcopy(fixture["receipt_value"])
                fresh_receipt["completed_at"] = "2026-09-28T00:00:01+00:00"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                tamper(receipt)
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                gate_patch, _rerun = self.core_gate_patch(
                    fixture, fresh_receipt=fresh_receipt
                )
                with gate_patch, self.assertRaises(contract_module.ContractError):
                    self.build(**self.core_materialization_arguments(fixture))

    def test_materialized_core_requires_all_fresh_audit_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_core_evidence_fixture(Path(directory))
            full = self.core_materialization_arguments(fixture)
            for missing in ("core_active_root", "core_backup_root", "core_zstd_path"):
                with self.subTest(missing=missing):
                    arguments = dict(full)
                    arguments.pop(missing)
                    with self.assertRaisesRegex(contract_module.ContractError, "supplied together"):
                        self.build(**arguments)

    def test_formal_auditor_and_zstd_are_hard_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alternate_auditor = root / "auditor.py"
            alternate_auditor.write_text("def audit(**kwargs): return {}\n", encoding="utf-8")
            with self.assertRaisesRegex(contract_module.ContractError, "hard-pinned formal auditor"):
                self.build(core_auditor_path=alternate_auditor)

            fixture = self.build_core_evidence_fixture(root / "fixture")
            with self.assertRaisesRegex(contract_module.ContractError, "hard-pinned /usr/bin/zstd"):
                self.build(**self.core_materialization_arguments(fixture))
            with mock.patch.object(
                contract_module, "EXPECTED_CORE_ZSTD_PATH", str(fixture["zstd"])
            ), self.assertRaisesRegex(contract_module.ContractError, "hard-pinned formal binary"):
                self.build(**self.core_materialization_arguments(fixture))

    def test_forged_receipt_must_equal_fresh_cold_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_core_evidence_fixture(Path(directory))
            fresh_receipt = copy.deepcopy(fixture["receipt_value"])
            fresh_receipt["completed_at"] = "2026-09-28T00:00:01+00:00"
            fresh_receipt["output_binding"]["eligible_file_count"] += 1
            gate_patch, rerun = self.core_gate_patch(
                fixture, fresh_receipt=fresh_receipt
            )
            with gate_patch, self.assertRaisesRegex(
                contract_module.ContractError, "differs from a fresh cold rerun"
            ):
                self.build(**self.core_materialization_arguments(fixture))
            rerun.assert_called_once()

    def test_active_bindings_are_rehashed_from_required_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_core_evidence_fixture(Path(directory))
            active_top = fixture["active_root"] / "manifest.json"
            active_top.write_text('{"tampered":true}\n', encoding="utf-8")
            gate_patch, _rerun = self.core_gate_patch(fixture)
            with gate_patch, self.assertRaisesRegex(
                contract_module.ContractError, "active top manifest path/hash binding mismatch"
            ):
                self.build(**self.core_materialization_arguments(fixture))

    def test_core_top_requires_ordered_contained_regular_dataset_manifests(self) -> None:
        cases = ("order", "path", "symlink", "counts")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_core_evidence_fixture(Path(directory))
                manifest_path = fixture["manifest"]
                top = json.loads(manifest_path.read_text(encoding="utf-8"))
                if case == "order":
                    top["dataset_manifests"][0], top["dataset_manifests"][1] = (
                        top["dataset_manifests"][1], top["dataset_manifests"][0]
                    )
                    manifest_path.write_text(json.dumps(top), encoding="utf-8")
                elif case == "path":
                    top["dataset_manifests"][0]["path"] = str(
                        (Path(directory).resolve() / "outside.json")
                    )
                    manifest_path.write_text(json.dumps(top), encoding="utf-8")
                elif case == "symlink":
                    dataset_path = Path(top["dataset_manifests"][0]["path"])
                    target = Path(directory).resolve() / "outside.json"
                    target.write_text(dataset_path.read_text(encoding="utf-8"), encoding="utf-8")
                    dataset_path.unlink()
                    dataset_path.symlink_to(target)
                else:
                    top["dataset_manifests"][0]["counts"] = dict(
                        top["dataset_manifests"][0]["counts"]
                    )
                    top["dataset_manifests"][0]["counts"]["numeric_selected"] = 1
                    manifest_path.write_text(json.dumps(top), encoding="utf-8")
                gate_patch, _rerun = self.core_gate_patch(fixture)
                with gate_patch, self.assertRaises(contract_module.ContractError):
                    self.build(**self.core_materialization_arguments(fixture))

    def test_validate_contract_checks_authority_counts_and_pins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_core_evidence_fixture(Path(directory))
            gate_patch, _rerun = self.core_gate_patch(fixture)
            with gate_patch:
                value = self.build(**self.core_materialization_arguments(fixture))
            mutations = {
                "count": lambda item: item["server166_materialization"].__setitem__(
                    "verified_dataset_count", 0
                ),
                "auditor": lambda item: item["input_bindings"]["core19_auditor"].__setitem__(
                    "sha256", "4" * 64
                ),
                "zstd": lambda item: item["server166_materialization"]["core19"].__setitem__(
                    "zstd_sha256", "5" * 64
                ),
            }
            with mock.patch.object(
                contract_module, "EXPECTED_CORE_ZSTD_PATH", str(fixture["zstd"])
            ), mock.patch.object(
                contract_module,
                "EXPECTED_CORE_ZSTD_SHA256",
                contract_module.sha256_file(fixture["zstd"]),
            ):
                for name, mutate in mutations.items():
                    with self.subTest(name=name):
                        tampered = copy.deepcopy(value)
                        mutate(tampered)
                        with self.assertRaises(contract_module.ContractError):
                            contract_module.validate_contract(
                                tampered, materialization_verified=True
                            )

    def test_atomic_round_trip(self) -> None:
        value = self.build(server192_state="authentication_blocked", server192_detail="public-key authentication rejected")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract_module._atomic_json(path, value)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            contract_module.validate_contract(loaded)
            self.assertEqual(loaded, value)
            self.assertEqual(loaded["server192_propagation"]["state"], "authentication_blocked")

    def test_sharded_dataset_stream_index_is_validated_without_manifest_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory) / "fixture"
            dataset_dir.mkdir()
            payload_dir = dataset_dir / "payload"
            payload_dir.mkdir()
            timestamps = (7).to_bytes(8, "little", signed=True)
            values = (11).to_bytes(4, "little", signed=True)
            bitmap = b"\x00"
            shard_bytes = timestamps + values + bitmap
            shard_path = payload_dir / "shard-000000.bin"
            shard_path.write_bytes(shard_bytes)
            entity_values = ['STRING:"x"']
            entity_id = hashlib.sha256(
                contract_module._canonical_json(entity_values)
            ).hexdigest()
            chunk = {
                "chunk_index": 0,
                "row_count": 1,
                "valid_count": 1,
                "missing_count": 0,
                "first_timestamp_unix_us": 7,
                "last_timestamp_unix_us": 7,
                "timestamp_unit": "us",
                "timestamp_dtype": "int64",
                "value_dtype": "int32",
                "byte_order": "little",
                "missing_bitmap_semantics": "one_bit_means_missing_lsb_first",
            }
            offset = 0
            for name, payload in (
                ("timestamps", timestamps), ("values", values),
                ("missing_bitmap", bitmap),
            ):
                digest = hashlib.sha256(payload).hexdigest()
                chunk[f"{name}_segment"] = {
                    "path": "payload/shard-000000.bin", "offset": offset,
                    "length": len(payload), "sha256": digest,
                }
                chunk[f"{name}_bytes"] = len(payload)
                chunk[f"{name}_sha256"] = digest
                offset += len(payload)
            totals = {
                "stream_count": 1, "chunk_count": 1, "rows": 1,
                "valid_values": 1, "missing_values": 0, "logical_bytes": 12,
            }
            binding = "b" * 64
            runtime_bindings = {
                "producer_sha256": "d" * 64,
                "helper_timeseries_sha256": "e" * 64,
                "helper_tsfile_sha256": "f" * 64,
            }
            records = [
                {"record_type": "header", "schema": "project179-thulab-stream-index-v1", "storage_layout": "sharded_v1", "binding_sha256": binding, "dataset_id": "fixture", **runtime_bindings},
                {"record_type": "entity", "source_path": "part.tsfile", "table_name": "t", "entity_id": entity_id, "entity_values": entity_values},
                {"record_type": "stream", "stream": {
                    "stream_index": 0, "stream_id": "s-00000000", "entity_id": entity_id,
                    "source_path": "part.tsfile", "table_name": "t", "schema_sha256": "c" * 64,
                    "field": "v", "source_field": "v", "tsfile_type": "INT32",
                    "dtype": "int32", "width_bytes": 4,
                    "conversion": "native_numeric_little_endian", "row_count": 1,
                    "valid_count": 1, "missing_count": 0, "logical_bytes": 12,
                    "chunks": [chunk],
                }},
                {"record_type": "footer", "schema": "project179-thulab-stream-index-v1", "entity_record_count": 1, "stream_record_count": 1, "totals": totals},
            ]
            raw = b"".join(contract_module._canonical_json(row) + b"\n" for row in records)
            index_path = dataset_dir / "stream_index.jsonl.gz"
            with index_path.open("wb") as file_handle:
                with gzip.GzipFile(filename="stream_index.jsonl", mode="wb", fileobj=file_handle, mtime=0) as zipped:
                    zipped.write(raw)
            manifest = {
                "schema": "project179-thulab-numeric-materialization-v1",
                "storage_layout": "sharded_v1",
                "binding_sha256": binding,
                "runtime_code_bindings": runtime_bindings,
                "selection": {
                    "chunk_byte_cap": 64,
                    "metadata_compressed_byte_cap": 1 << 20,
                },
                "payload_shards": [{
                    "shard_index": 0, "path": "payload/shard-000000.bin",
                    "bytes": len(shard_bytes), "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                }],
                "stream_index": {
                    "schema": "project179-thulab-stream-index-v1", "path": "stream_index.jsonl.gz",
                    "compression": "gzip", "record_encoding": "canonical_json_lines_utf8",
                    "record_count": 4, "entity_record_count": 1, "stream_record_count": 1,
                    "compressed_bytes": index_path.stat().st_size,
                    "compressed_sha256": contract_module.sha256_file(index_path),
                    "uncompressed_bytes": len(raw), "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                    "compressed_byte_cap": 1 << 20,
                },
                "selected_entity_dictionary_count": 1,
                "entity_dictionary_semantics": "metadata_only_typed_TAG_partition_provenance_not_numeric_payload",
                "totals": totals,
            }
            dataset_path = dataset_dir / "dataset_manifest.json"
            dataset_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                contract_module._validate_sharded_thulab_dataset(
                    "fixture", dataset_path, manifest
                ),
                (1, 0),
            )
            (dataset_dir / "unexpected.bin").write_bytes(b"x")
            with self.assertRaisesRegex(contract_module.ContractError, "exact file set"):
                contract_module._validate_sharded_thulab_dataset(
                    "fixture", dataset_path, manifest
                )


if __name__ == "__main__":
    unittest.main()
