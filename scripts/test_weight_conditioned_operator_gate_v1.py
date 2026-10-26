import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import weight_conditioned_operator_gate_v1 as gate


class OperatorGateTests(unittest.TestCase):
    def test_pdf_resolution_prefers_snapshot_and_preserves_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled, desktop, explicit = (root / name for name in
                                          ("bundled.pdf", "desktop.pdf", "explicit.pdf"))
            bundled.write_bytes(b"bundled")
            desktop.write_bytes(b"desktop")
            with (mock.patch.object(gate, "DEFAULT_PDF", bundled),
                  mock.patch.object(gate, "DESKTOP_PDF", desktop),
                  mock.patch.dict(os.environ, {}, clear=True)):
                self.assertEqual(gate.resolve_pdf_path(), bundled)
                with mock.patch.dict(os.environ, {"WEIGHTED_GRPO_OPERATOR_PDF": str(explicit)}):
                    self.assertEqual(gate.resolve_pdf_path(), explicit)
                    self.assertEqual(gate.resolve_pdf_path(desktop), desktop)
                bundled.unlink()
                self.assertEqual(gate.resolve_pdf_path(), desktop)
                desktop.unlink()
                self.assertEqual(gate.resolve_pdf_path(), bundled)

    def test_pdf_hash_failure_does_not_fall_back_to_valid_desktop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled, desktop, missing = (root / name for name in
                                         ("bundled.pdf", "desktop.pdf", "missing.pdf"))
            bundled.write_bytes(b"wrong source")
            desktop.write_bytes(b"approved source")
            expected = hashlib.sha256(desktop.read_bytes()).hexdigest()
            with (mock.patch.object(gate, "DEFAULT_PDF", bundled),
                  mock.patch.object(gate, "DESKTOP_PDF", desktop),
                  mock.patch.object(gate, "EXPECTED_PDF_SHA256", expected),
                  mock.patch.dict(os.environ, {}, clear=True)):
                report = gate.build_report()
                self.assertEqual(report["pdf"]["path"], str(bundled))
                self.assertFalse(report["pdf"]["matches_expected"])
                self.assertFalse(report["structural_gate_passed"])
                with mock.patch.dict(os.environ, {"WEIGHTED_GRPO_OPERATOR_PDF": str(missing)}):
                    report = gate.build_report()
                    self.assertEqual(report["pdf"]["path"], str(missing))
                    self.assertFalse(report["pdf"]["matches_expected"])
                    self.assertFalse(report["structural_gate_passed"])

    def test_catalog_layers_and_pdf_identity_are_preserved(self):
        report = gate.build_report()
        self.assertEqual(report["counts"]["trace_entries"], 102)
        self.assertEqual(report["counts"]["mechanisms"], 67)
        self.assertEqual(report["counts"]["layers"], {
            "mechanism": 67, "profile": 7, "recipe": 14, "adapter": 14})
        self.assertTrue(report["pdf"]["matches_expected"])

    def test_direct_graph_bridges_are_not_counted_as_new_trace_entries(self):
        report = gate.build_report()
        by_id = {row["implementation_id"]: row for row in report["entries"]}
        self.assertEqual(by_id["A035"]["status"], "direct_graph_bridge_ready")
        self.assertEqual(by_id["A111"]["status"], "direct_graph_bridge_ready")
        self.assertEqual(len(by_id), 102)

    def test_only_cold_decoded_graph_probability_paths_are_eligible(self):
        report = gate.build_report()
        blockers = {row["implementation_id"] for row in report["formal_training_blockers"]}
        providers = gate._prediction_entropy().get("providers", {})
        provider_by_id = {
            "A106": "ppmd", "A107": "dmc", "A108": "paq", "A109": "cmix",
        }
        readiness = {
            implementation_id: providers.get(provider, {}).get("graph_eligible") is True
            for implementation_id, provider in provider_by_id.items()
        }
        for implementation_id, ready in readiness.items():
            self.assertEqual(implementation_id not in blockers, ready)
        cmix_ready = readiness["A109"]
        self.assertEqual("A109" not in blockers, cmix_ready)
        nncp_ready = gate._nncp_graph_entropy().get("graph_eligible") is True
        self.assertEqual("A110" not in blockers, nncp_ready)
        expected = ([implementation_id for implementation_id in
                     ("A106", "A107", "A108", "A109")
                     if readiness[implementation_id]]
                    + (["A110"] if nncp_ready else []))
        self.assertEqual(report["probability_training_scope"]["eligible_mechanisms"], expected)
        by_id = {row["implementation_id"]: row for row in report["entries"]}
        for implementation_id in ("A106", "A107", "A108"):
            self.assertEqual(by_id[implementation_id]["probability_evidence"][
                "independent_cold_decode_validated"], readiness[implementation_id])
            self.assertEqual(by_id[implementation_id]["probability_evidence"][
                "graph_entropy_integrated"], readiness[implementation_id])
        self.assertEqual(by_id["A109"]["selectable"], cmix_ready)
        self.assertTrue(report["structural_gate_passed"])
        self.assertFalse(report["formal_training_gate_passed"])

    def test_standalone_nncp_frame_alone_does_not_imply_graph_policy_eligibility(self):
        current = {
            "available": True,
            "validated_current_build": True,
            "reason": "test-only current validated standalone path",
        }
        graph = {"graph_eligible": False, "cold_decode_validated": False,
                 "reason": "test-only missing graph receipt"}
        with (mock.patch.object(gate, "_nncp", return_value=current),
              mock.patch.object(gate, "_nncp_graph_entropy", return_value=graph)):
            report = gate.build_report()
        row = next(item for item in report["entries"] if item["implementation_id"] == "A110")
        self.assertFalse(row["selectable"])
        self.assertTrue(row["probability_evidence"]["standalone_reversible_entropy_path"])
        self.assertTrue(row["probability_evidence"]["independent_cold_decode_validated"])
        self.assertFalse(row["probability_evidence"]["graph_entropy_integrated"])
        self.assertIn("unvalidated", row["status"])
        self.assertFalse(report["formal_training_gate_passed"])

    def test_current_nnp1_graph_receipt_makes_a110_eligible_as_complete_path(self):
        standalone = {"available": True, "validated_current_build": True,
                      "reason": "test-only current standalone path"}
        graph = {"graph_eligible": True, "cold_decode_validated": True,
                 "reason": "test-only current graph path"}
        with (mock.patch.object(gate, "_nncp", return_value=standalone),
              mock.patch.object(gate, "_nncp_graph_entropy", return_value=graph)):
            report = gate.build_report()
        row = next(item for item in report["entries"] if item["implementation_id"] == "A110")
        self.assertTrue(row["selectable"])
        self.assertEqual(row["probability_evidence"]["graph_path_role"],
                         "complete_nnp1_terminal_entropy_path")
        self.assertTrue(row["probability_evidence"]["graph_entropy_integrated"])
        self.assertNotIn("A110", {
            item["implementation_id"] for item in report["formal_training_blockers"]
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
