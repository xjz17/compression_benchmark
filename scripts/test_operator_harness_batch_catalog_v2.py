"""Contract tests for the local-only operator harness-v2 batch catalog."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr

import operator_harness_batch_catalog_v2 as catalog


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class OperatorHarnessBatchCatalogV2Tests(unittest.TestCase):
    def test_exact102_and_exact97_batch_partition(self):
        document = catalog.load_batch_catalog()
        self.assertEqual(document["schema"], catalog.SCHEMA)
        self.assertEqual(document["counts"]["all_trace_entries"], 102)
        self.assertEqual(document["counts"]["eligible"], 97)
        self.assertEqual(document["counts"]["disabled_by_policy"], 5)
        self.assertEqual(document["batch_order"], list(catalog.BATCH_ORDER))
        self.assertEqual(
            document["counts"]["batches"],
            {
                "v1core9": 9,
                "ordinary_source_pipelines": 66,
                "source_exact_dispatcher": 5,
                "special_residual_deps": 17,
            },
        )
        batch_ids = [
            implementation_id
            for batch in document["batches"]
            for implementation_id in batch["implementation_ids"]
        ]
        self.assertEqual(len(batch_ids), 97)
        self.assertEqual(len(set(batch_ids)), 97)

    def test_all_rows_share_generic_required_shape(self):
        required = {
            "implementation_id", "name", "layer", "batch_id", "selection",
            "initial_kind", "initial_width", "input_view", "present_pipeline",
            "focal_step_index", "backend", "dispatcher", "source_refs",
        }
        rows = catalog.load_batch_catalog()["operators"]
        self.assertEqual(len(rows), 102)
        for row in rows:
            self.assertLessEqual(required, set(row), row["implementation_id"])
            self.assertFalse(row["selection"]["overridable"])

    def test_excluded_families_are_explicit_and_have_no_override(self):
        document = catalog.load_batch_catalog()
        by_id = {row["implementation_id"]: row for row in document["operators"]}
        expected = {"A106", "A107", "A108", "A109", "A110"}
        self.assertEqual(set(document["policy"]["disabled_ids"]), expected)
        self.assertFalse(document["policy"]["runtime_override_allowed"])
        self.assertEqual(document["policy"]["nncp_state"], "disabled_by_policy")
        for implementation_id in expected:
            row = by_id[implementation_id]
            self.assertIsNone(row["batch_id"])
            self.assertEqual(row["selection"]["state"], "disabled_by_policy")
            self.assertFalse(row["selection"]["selectable"])
            self.assertFalse(row["selection"]["overridable"])
        self.assertEqual(by_id["A106"]["selection"]["family"], "PPMd")
        self.assertEqual(by_id["A107"]["selection"]["family"], "DMC")
        self.assertEqual(by_id["A108"]["selection"]["family"], "PAQ8")
        self.assertEqual(by_id["A109"]["selection"]["family"], "CMIX/LSTM")
        self.assertEqual(by_id["A110"]["selection"]["family"], "NNCP")
        for implementation_id in expected:
            self.assertEqual(
                by_id[implementation_id]["dispatch_support"]["support_status"],
                "disabled_by_policy",
            )

    def test_negative_exclusion_policy_drift_fails_closed(self):
        missing_nncp = dict(catalog.DISABLED_POLICY)
        missing_nncp.pop("A110")
        with mock.patch.object(catalog, "DISABLED_POLICY", missing_nncp):
            with self.assertRaisesRegex(ValueError, "disabled policy must remain exact"):
                catalog.load_batch_catalog()
        extra = dict(catalog.DISABLED_POLICY)
        extra["A105"] = {"family": "unexpected", "reason": "test"}
        with mock.patch.object(catalog, "DISABLED_POLICY", extra):
            with self.assertRaisesRegex(ValueError, "disabled policy must remain exact"):
                catalog.load_batch_catalog()

    def test_pipeline_roles_and_parameters_are_preserved(self):
        document = catalog.load_batch_catalog()
        by_id = {row["implementation_id"]: row for row in document["operators"]}
        resolved = json.loads(catalog.RESOLVED_CATALOG.read_text())
        frozen = {
            row["implementation_id"]: row
            for row in resolved["resolved_implementations"]
        }
        for implementation_id, row in by_id.items():
            source = frozen[implementation_id]
            stripped = [
                {key: value for key, value in step.items() if key not in {"role", "step_index"}}
                for step in row["present_pipeline"]
            ]
            self.assertEqual(stripped, source["present_pipeline"], implementation_id)
            focal = row["focal_step_index"]
            for index, step in enumerate(row["present_pipeline"]):
                expected_role = "prefix" if index < focal else "focal" if index == focal else "suffix"
                self.assertEqual(step["role"], expected_role, implementation_id)
                self.assertEqual(step["step_index"], index, implementation_id)

    def test_fallback_wrapper_is_not_miscounted_as_an_operator(self):
        document = catalog.load_batch_catalog()
        self.assertEqual(
            document["domain_fallback_accounting"]["implementation_ids"],
            ["A003", "A006", "A007", "A008"],
        )
        self.assertFalse(document["domain_fallback_accounting"]["wrapper_counts_as_operator"])
        by_id = {row["implementation_id"]: row for row in document["operators"]}
        for implementation_id in ["A003", "A006", "A007", "A008"]:
            domain = by_id[implementation_id]["domain_execution"]
            self.assertEqual(
                domain["measurement_wrapper"],
                "operator102.lossless_domain_fallback.selector_raw.v1",
            )
            self.assertFalse(domain["wrapper_counts_as_operator"])
            self.assertEqual(
                domain["support_accounting"],
                "strict_target_and_raw_fallback_reported_separately",
            )

    def test_special_bridge_support_set_does_not_absorb_other_dispatchers(self):
        document = catalog.load_batch_catalog()
        special = document["special_bridge_contract"]
        self.assertEqual(special["supported_ids"], list(catalog.CPP_SPECIAL_BRIDGE_IDS))
        self.assertEqual(special["unsupported_frozen_dispatcher_ids"], [])
        self.assertTrue(special["support_set_is_exact"])
        by_id = {row["implementation_id"]: row for row in document["operators"]}
        self.assertEqual(
            {aid for aid, row in by_id.items() if row["dispatcher"] == catalog.SPECIAL_BRIDGE_DISPATCHER},
            set(catalog.CPP_SPECIAL_BRIDGE_IDS),
        )
        self.assertEqual(by_id["A087"]["dispatch_support"]["kind"], "cpp_general_residual_support_dependencies")
        self.assertEqual(by_id["A022"]["dispatch_support"]["kind"], "cpp_source_exact_dispatcher")
        self.assertEqual(
            special["separate_dispatchers"]["residual_dependency_ids"],
            ["A087", "A088", "A089", "A090", "A091", "A092", "A093"],
        )
        self.assertEqual(
            special["separate_dispatchers"]["source_exact_ids"],
            ["A022", "A027", "A043", "A084", "A085"],
        )

    def test_layer_and_composite_adapter_semantics_are_explicit(self):
        document = catalog.load_batch_catalog()
        by_id = {row["implementation_id"]: row for row in document["operators"]}
        self.assertEqual(document["counts"]["eligible_layers"], {
            "mechanism": 62, "profile": 7, "recipe": 14, "adapter": 14,
        })
        self.assertEqual(
            by_id["A048"]["execution_semantics"]["v1_registry_binding"],
            "harness_composite_adapter_over_existing_cpp_byte_plane_kernels",
        )
        self.assertEqual(
            by_id["A019"]["execution_semantics"]["pipeline"],
            "prefix_focal_suffix_composite",
        )
        adapter = next(row for row in document["operators"] if row["layer"] == "adapter")
        self.assertEqual(adapter["execution_semantics"]["catalog_layer"], "compatibility_adapter_trace")

    def test_deterministic_json_primitives_hashes_and_detached_results(self):
        first = catalog.load_batch_catalog()
        second = catalog.load_batch_catalog()
        self.assertEqual(first, second)
        payload = dict(first)
        expected_hash = payload.pop("catalog_sha256")
        self.assertEqual(hashlib.sha256(canonical_bytes(payload)).hexdigest(), expected_hash)
        for source in first["source_catalogs"]:
            path = catalog.ROOT / source["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])
        bridge = first["special_bridge_contract"]
        bridge_path = catalog.ROOT / bridge["cpp_source"]
        self.assertEqual(
            hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
            bridge["cpp_source_sha256"],
        )
        first["operators"][0]["name"] = "mutated"
        self.assertNotEqual(first, catalog.load_batch_catalog())
        self.assertEqual(second, catalog.load_batch_catalog())

    def test_checked_in_catalog_matches_generator(self):
        self.assertEqual(
            json.loads(catalog.DEFAULT_OUTPUT.read_text()),
            catalog.load_batch_catalog(),
        )

    def test_cli_writes_the_same_catalog_and_has_no_policy_override(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            self.assertEqual(catalog.main(["--out", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text()), catalog.load_batch_catalog())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            catalog.main(["--allow-nncp"])


if __name__ == "__main__":
    unittest.main()
