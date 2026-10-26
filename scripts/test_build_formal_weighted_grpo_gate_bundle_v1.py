#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_formal_weighted_grpo_gate_bundle_v1 as builder
import formal_weighted_grpo_training_v1 as training
import operator_implementation_catalog
import weight_conditioned_operator_gate_v1 as operator_gate


def _runtime(**changes) -> builder.RuntimeIdentity:
    values = dict(
        source_fingerprint="1" * 64,
        action_registry_fingerprint="2" * 64,
        action_count=100,
        stop_action=0,
        max_depth=10,
        max_nodes=64,
        max_streams=32,
        special_categories=("concat", "split_segments", "stop"),
        edit_types=("delete", "insert", "parameter_change", "replace"),
        runtime_files=(("scripts/toy.py", "3" * 64),),
        measurement_protocol_schema=builder.experiment.FORMAL_MEASUREMENT_PROTOCOL_SCHEMA,
        measurement_contract=dict(builder.experiment.FORMAL_MEASUREMENT_CONTRACT),
    )
    values.update(changes)
    return builder.RuntimeIdentity(**values)


def _mechanism_ids() -> list[str]:
    return [f"M{index:03d}" for index in range(67)]


def _operator_report() -> dict:
    ids = _mechanism_ids()
    return {
        "schema": operator_gate.SCHEMA,
        "catalog_registry_version": operator_implementation_catalog.REGISTRY_VERSION,
        "catalog_sha256": operator_implementation_catalog.MANIFEST_SHA256,
        "pdf": {
            "sha256": builder.EXPECTED_PDF_SHA256,
            "matches_expected": True,
        },
        "counts": {
            "trace_entries": 102,
            "mechanisms": 67,
            "layers": dict(builder.EXPECTED_LAYER_COUNTS),
            "operator_groups": 30,
            "special_operations": 3,
        },
        "structural_gate_passed": True,
        "formal_training_gate_passed": True,
        "formal_training_blockers": [],
        "entries": [
            {
                "implementation_id": identifier,
                "layer": "mechanism",
                "selectable": True,
                "formal_evidence_ready": True,
            }
            for identifier in ids
        ],
    }


def _probes(runtime: builder.RuntimeIdentity) -> dict:
    rows = [
        {
            "implementation_id": identifier,
            "passed": True,
            "action_probes": [{"passed": True, "frame_bytes": 17}],
        }
        for identifier in _mechanism_ids()
    ]
    return {
        "schema": "weight-conditioned-mechanism-probes-v1",
        "registry_fingerprint": runtime.action_registry_fingerprint,
        "mechanism_count": 67,
        "passed_mechanism_count": 67,
        "failed_mechanism_count": 0,
        "mechanisms": rows,
    }


def _graph_report(runtime: builder.RuntimeIdentity) -> dict:
    return {
        "schema": "weight-conditioned-fixed-graph-actions-v1",
        "registry_fingerprint": runtime.action_registry_fingerprint,
        "fixed_policy_action_count": runtime.action_count,
        "stop_action": 0,
        "continuation_edit_action_space": {
            "type_action_ids": {name: index for index, name in enumerate(runtime.edit_types)},
            "max_completed_edits_per_block": 2,
            "protocol": (
                "edit_type -> edit_position -> fixed operator/parameter action; "
                "delete completes after edit_position"
            ),
            "replay_contract": (
                "after every completed edit the entire edited recipe is replayed "
                "from the original root data flows before it becomes a legal state"
            ),
        },
        "formal_training_ready": True,
        "formal_training_blockers": [],
        "measurement_protocol_schema": runtime.measurement_protocol_schema,
        "measurement_contract": runtime.measurement_contract,
        "compression_timing_boundary": builder.experiment.FORMAL_TIMING_BOUNDARY,
        "decompression_timing_boundary": builder.experiment.FORMAL_TIMING_BOUNDARY,
        "mechanism_coverage": {
            "mechanism_count": 67,
            "cataloged_mechanism_count": 67,
            "probe_passed_mechanism_count": 67,
            "exhaustive_bit_exact_probe_receipt": True,
            "omitted_selectable_mechanisms": [],
            "probe_failed_mechanisms": [],
            "policy_reachability_blockers": [],
            "upstream_blocked_mechanisms": [],
            "probe_registry_fingerprint": runtime.action_registry_fingerprint,
        },
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class FormalGateBundleBuilderTest(unittest.TestCase):
    def _fixture(self, root: Path):
        runtime = _runtime()
        operator_path = root / "operator.json"
        graph_path = root / "graph.json"
        probes_path = root / "probes.json"
        reference_path = root / "reference.json"
        prepared = root / "prepared"
        prepared.mkdir()
        _write(prepared / "dataset_manifest.json", {"dataset_id": "toy"})
        _write(operator_path, _operator_report())
        _write(graph_path, _graph_report(runtime))
        _write(probes_path, _probes(runtime))
        _write(reference_path, {
            "schema": "toy-reference", "references": [{}], "blocks": [{}],
        })
        return runtime, operator_path, graph_path, probes_path, reference_path, prepared

    def test_pass_writes_four_pinned_receipts_and_training_loader_accepts_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, operator_path, graph_path, probes_path, reference_path, prepared = (
                self._fixture(root)
            )
            provenance = {"toy": "4" * 64}
            with mock.patch.object(
                builder, "_validate_dataset_admission",
                return_value={"source_spec_gate": {"complete_cohort_trainable": True,
                                                    "blocked": 0},
                              "prepared_admission": {}},
            ), mock.patch.object(
                builder, "_validate_data",
                return_value=(provenance, {"complete_train_validation": True}, []),
            ), mock.patch.object(
                builder, "_validate_reference", return_value="5" * 64,
            ):
                result = builder.assemble(
                    operator_gate_path=operator_path,
                    graph_catalog_path=graph_path,
                    graph_probes_path=probes_path,
                    prepared=[prepared], reference_path=reference_path,
                    output_dir=root / "bundle", runtime=runtime,
                    live_reports=(
                        _operator_report(), _graph_report(runtime), _probes(runtime),
                    ),
                )
            bundle_path = Path(result["bundle"])
            bundle = json.loads(bundle_path.read_text())
            self.assertEqual(set(bundle["receipts"]), set(training.REQUIRED_GATE_ASSERTIONS))
            corpus = training.TrainingCorpus(
                train_blocks=(), validation_blocks=(),
                dataset_provenance_sha256=tuple(provenance.items()),
                calibration_sha256="5" * 64, corpus_sha256="6" * 64,
            )
            with mock.patch.object(
                builder.graph_task, "MEASUREMENT_PROTOCOL_SCHEMA",
                builder.experiment.FORMAL_MEASUREMENT_PROTOCOL_SCHEMA,
            ), mock.patch.object(
                builder.graph_task, "FORMAL_MEASUREMENT_CONTRACT",
                dict(builder.experiment.FORMAL_MEASUREMENT_CONTRACT), create=True,
            ):
                evidence = training._load_gate_bundle(bundle_path, corpus)
            self.assertEqual(evidence.source_fingerprint, runtime.source_fingerprint)
            self.assertEqual(
                evidence.action_registry_fingerprint,
                runtime.action_registry_fingerprint,
            )

            mechanism = bundle_path.parent / bundle["receipts"]["mechanisms"]["path"]
            mechanism.write_bytes(mechanism.read_bytes() + b" ")
            with self.assertRaises(training.TrainingAdmissionError):
                training._load_gate_bundle(bundle_path, corpus)

    def test_mechanism_gate_fails_without_one_current_cold_bit_exact_probe(self):
        runtime = _runtime()
        report = _probes(runtime)
        report["mechanisms"][0]["passed"] = False
        report["mechanisms"][0]["action_probes"] = [{"passed": False}]
        report["passed_mechanism_count"] = 66
        report["failed_mechanism_count"] = 1
        with self.assertRaisesRegex(builder.GateAssemblyError, "67|passing probe"):
            builder._validate_graph_evidence(
                _graph_report(runtime), report, runtime, _mechanism_ids()
            )

    def test_graph_gate_fails_for_wrong_specials_or_edit_semantics(self):
        runtime = _runtime(special_categories=("concat", "stop"))
        with self.assertRaisesRegex(builder.GateAssemblyError, "special"):
            builder._validate_graph_evidence(
                _graph_report(runtime), _probes(runtime), runtime, _mechanism_ids()
            )
        runtime = _runtime()
        report = _graph_report(runtime)
        report["continuation_edit_action_space"]["max_completed_edits_per_block"] = 3
        with self.assertRaisesRegex(builder.GateAssemblyError, "two"):
            builder._validate_graph_evidence(
                report, _probes(runtime), runtime, _mechanism_ids()
            )

    def test_data_gate_failure_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, operator_path, graph_path, probes_path, reference_path, prepared = (
                self._fixture(root)
            )
            with mock.patch.object(
                builder, "_validate_dataset_admission",
                return_value={"source_spec_gate": {"complete_cohort_trainable": True,
                                                    "blocked": 0},
                              "prepared_admission": {}},
            ), mock.patch.object(
                builder, "_validate_data",
                side_effect=builder.GateAssemblyError("incomplete timestamp/leakage audit"),
            ):
                with self.assertRaisesRegex(builder.GateAssemblyError, "timestamp/leakage"):
                    builder.assemble(
                        operator_gate_path=operator_path,
                        graph_catalog_path=graph_path,
                        graph_probes_path=probes_path,
                        prepared=[prepared], reference_path=reference_path,
                        output_dir=root / "bundle", runtime=runtime,
                        live_reports=(
                            _operator_report(), _graph_report(runtime), _probes(runtime),
                        ),
                    )
            self.assertFalse((root / "bundle").exists())

    def test_blocked_prepared_manifest_prevents_bundle_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, operator_path, graph_path, probes_path, reference_path, prepared = self._fixture(root)
            with mock.patch.object(
                builder.dataset_spec_gate, "validate_bundle",
                return_value={"complete_cohort_trainable": False, "approved": 5, "blocked": 11},
            ), mock.patch.object(
                builder.dataset_spec_gate, "validate_prepared_admissions",
                side_effect=builder.dataset_spec_gate.SpecGateError(
                    "FORMAL_DATASET_ADMISSION_BLOCKER: prepared artifact cannot admit blocked dataset deep-space-probes"
                ),
            ):
                with self.assertRaisesRegex(builder.GateAssemblyError, "blocked dataset"):
                    builder.assemble(
                        operator_gate_path=operator_path, graph_catalog_path=graph_path,
                        graph_probes_path=probes_path, prepared=[prepared],
                        reference_path=reference_path, output_dir=root / "bundle",
                        runtime=runtime,
                        live_reports=(_operator_report(), _graph_report(runtime), _probes(runtime)),
                    )
            self.assertFalse((root / "bundle").exists())

    def test_reference_gate_rejects_smoke_schema(self):
        with self.assertRaisesRegex(builder.GateAssemblyError, "smoke/non-formal"):
            builder._validate_reference(
                {"schema": "weighted-grpo-zlib6-reference-smoke-v1"}, []
            )

    def test_ready_boolean_cannot_override_hybrid_runtime_timing(self):
        runtime = _runtime(
            measurement_protocol_schema="typed-graph-timing-samples-v2",
            measurement_contract=None,
        )
        report = _graph_report(_runtime())
        self.assertTrue(report["formal_training_ready"])
        with self.assertRaisesRegex(builder.GateAssemblyError, "hybrid"):
            builder._validate_graph_evidence(report, _probes(runtime), runtime, _mechanism_ids())

    def test_native_contract_cannot_override_hybrid_catalog_boundary(self):
        runtime = _runtime()
        report = _graph_report(runtime)
        report["compression_timing_boundary"] = "Python scheduling + native child-process operators"
        with self.assertRaisesRegex(builder.GateAssemblyError, "timing_boundary"):
            builder._validate_graph_evidence(report, _probes(runtime), runtime, _mechanism_ids())

    def test_live_hybrid_schema_is_not_formally_admissible(self):
        with self.assertRaisesRegex(builder.experiment.ExperimentContractError, "hybrid"):
            builder.experiment._validate_current_runtime_measurement()

    def test_timestamp_gate_accepts_only_audited_coordinate_semantics(self):
        calendar = {
            "source_kind": "calendar_date_coordinate",
            "semantic_kind": "calendar_date_utc_midnight_encoding",
            "source_unit": "ms",
            "output_unit": "ms",
            "timezone": "UTC",
            "formats": ["integer_coordinate_from_tsfile"],
            "ambiguous_fold": None,
            "invalid_policy": "fail_no_unit_or_timezone_inference",
            "source_event_timezone_inferred": False,
        }
        builder._validate_timestamp_contract(calendar, "exchange_rate")
        civil = {
            "source_kind": "civil_time_coordinate",
            "semantic_kind": "portuguese_civil_time_coordinate_encoding",
            "source_unit": "ms",
            "output_unit": "ms",
            "timezone": "Europe/Lisbon source civil-time labels",
            "formats": ["integer_coordinate_from_tsfile"],
            "ambiguous_fold": None,
            "invalid_policy": "fail_no_unit_or_timezone_inference",
            "source_event_timezone_inferred": False,
            "event_utc_instant_claim": False,
            "event_time_mapping_is_injective": False,
        }
        builder._validate_timestamp_contract(civil, "electricity_load_2011_2014")
        changed = dict(civil)
        changed["event_utc_instant_claim"] = True
        with self.assertRaisesRegex(builder.GateAssemblyError, "audit flag"):
            builder._validate_timestamp_contract(changed, "electricity_load_2011_2014")
        changed = dict(civil)
        changed["semantic_kind"] = "unreviewed_coordinate"
        with self.assertRaisesRegex(builder.GateAssemblyError, "not audited"):
            builder._validate_timestamp_contract(changed, "electricity_load_2011_2014")

    def test_reference_gate_recomputes_native_metrics_and_group_medians(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = b"\x01\x00\x00\x00"
            bitmap = b"\x00"
            (root / "values.bin").write_bytes(values)
            (root / "bitmap.bin").write_bytes(bitmap)
            block = {
                "split": "train", "block_id": "b0", "dtype": "int32",
                "values_path": "values.bin", "missing_bitmap_path": "bitmap.bin",
                "original_numeric_bytes": 4,
            }
            provenance = "7" * 64
            manifests = [{
                "__root__": str(root), "dataset_id": "toy",
                "provenance_sha256": provenance, "blocks": [block],
            }]
            payload = values + bitmap
            native = {
                "schema": builder.calibration.BENCHMARK_SCHEMA,
                "level": 6, "raw_bytes": len(payload), "warmups": 1,
                "repeats": 3, "complete_frame_bytes": 20,
                "compression_median_ns": 40, "decompression_median_ns": 20,
                "timing_boundary": builder.NATIVE_REFERENCE_TIMING_BOUNDARY,
            }
            block_hash = builder._sha256_bytes(builder._canonical_json([["toy", "b0"]]))
            reference = {
                "schema": builder.calibration.SCHEMA,
                "formal_training_eligible": True,
                "calibration_scope": "complete_frozen_16_source_train_cohort",
                "objective_normalization": "E/(E+E0), D/(D+D0)",
                "reference_method": "native zlib level 6 complete ZR61 frame",
                "prepared_dataset_count": 16, "training_block_count": 1,
                "warmups": 1, "repeats": 3,
                "blocks": [{
                    "dataset_id": "toy", "provenance_sha256": provenance,
                    "block_id": "b0", "dtype": "int32",
                    "prepared_block_bytes": len(payload),
                    "block_size_bucket_bytes": 8,
                    "original_numeric_bytes": 4,
                    "calibration_payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "compression_ns_per_original_byte": 10.0,
                    "decompression_ns_per_original_byte": 5.0,
                    "native_receipt": native,
                }],
                "references": [{
                    "dtype": "int32", "block_size_bucket_bytes": 8,
                    "E0_compression_ns_per_original_byte": 10.0,
                    "D0_decompression_ns_per_original_byte": 5.0,
                    "training_block_count": 1, "block_ids_sha256": block_hash,
                }],
            }
            reference["calibration_sha256"] = builder._sha256_bytes(
                builder._canonical_json(reference)
            )
            self.assertEqual(
                builder._validate_reference(reference, manifests),
                reference["calibration_sha256"],
            )
            for invalid_boundary in (None, "Python process launch plus native calls"):
                native["timing_boundary"] = invalid_boundary
                basis = dict(reference)
                basis.pop("calibration_sha256")
                reference["calibration_sha256"] = builder._sha256_bytes(builder._canonical_json(basis))
                with self.assertRaisesRegex(builder.GateAssemblyError, "timing boundary"):
                    builder._validate_reference(reference, manifests)
            native["timing_boundary"] = builder.NATIVE_REFERENCE_TIMING_BOUNDARY
            reference["references"][0]["E0_compression_ns_per_original_byte"] = 9.0
            basis = dict(reference)
            basis.pop("calibration_sha256")
            reference["calibration_sha256"] = builder._sha256_bytes(
                builder._canonical_json(basis)
            )
            with self.assertRaisesRegex(builder.GateAssemblyError, "group median"):
                builder._validate_reference(reference, manifests)

    def test_stale_registry_fingerprint_is_rejected(self):
        runtime = _runtime()
        report = _graph_report(runtime)
        report["registry_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(builder.GateAssemblyError, "stale"):
            builder._validate_graph_evidence(
                report, _probes(runtime), runtime, _mechanism_ids()
            )


if __name__ == "__main__":
    unittest.main()
