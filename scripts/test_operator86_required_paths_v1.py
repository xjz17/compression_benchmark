#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("operator86_required_paths_v1.py")
SPEC = importlib.util.spec_from_file_location("operator86_required_paths_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Operator86RequiredPathsV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = MODULE.build_method_requirements()
        cls.by_label = {row["label"]: row for row in cls.rows}

    def test_exactly_86_unique_labels_in_authoritative_groups(self) -> None:
        self.assertEqual(len(self.rows), 86)
        self.assertEqual(len(self.by_label), 86)
        self.assertEqual(sum(row["category"] == "specialist" for row in self.rows), 28)
        self.assertEqual(sum(row["category"] == "general" for row in self.rows), 58)
        self.assertEqual([row["ordinal"] for row in self.rows], list(range(1, 87)))
        for row in self.rows:
            self.assertEqual(row["current_authoritative_label_check"]["status"], "exact_match")
            self.assertRegex(
                row["current_authoritative_label_check"]["source_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_json_serializable_and_deterministic(self) -> None:
        first = json.dumps(self.rows, ensure_ascii=False, sort_keys=True)
        second = json.dumps(MODULE.build_method_requirements(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(first, second)

    def test_every_label_has_an_ordered_documentary_path_and_hashes(self) -> None:
        for row in self.rows:
            self.assertTrue(row["required_path"], row["label"])
            self.assertEqual(
                [stage["ordinal"] for stage in row["documentary_path"]],
                list(range(1, len(row["required_path"]) + 1)),
            )
            self.assertEqual(
                [stage["mechanism"] for stage in row["documentary_path"]],
                row["required_path"],
            )
            for digest in row["source_receipts"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_historical_a_ids_are_catalog_ids_but_not_graph_proof(self) -> None:
        for row in self.rows:
            for a_id in row["historical_a_ids"]:
                self.assertRegex(a_id, r"^A\d{3}$")
            self.assertFalse(row["claim_boundary"]["historical_a_id_is_6341_registration_proof"])
            for logical_type in ("int64", "float64"):
                requirement = row["requirements_by_logical_type"][logical_type]
                self.assertEqual(
                    requirement["policy_graph_registration"]["status"],
                    "gap_not_established_by_consumed_sources",
                )
                self.assertFalse(
                    requirement["direct_method_evidence"]["proves_decomposed_graph_path"]
                )

    def test_proxy_sequences_are_not_promoted_to_original_paths(self) -> None:
        for label in ("FSE", "Huff0", "Density", "FPCmp-R"):
            row = self.by_label[label]
            self.assertFalse(
                row["original_algorithm_requirement"][
                    "proxy_runtime_sequence_is_substitute"
                ]
            )
            for logical_type in ("int64", "float64"):
                requirement = row["requirements_by_logical_type"][logical_type]
                status = requirement[
                    "audited_runtime_sequence"
                ]["status"]
                self.assertEqual(status, "project_proxy_not_original_algorithm_decomposition")
                self.assertEqual(requirement["candidate_seed"]["ordered_runtime_actions"], [])
                self.assertTrue(
                    requirement["candidate_seed"][
                        "historical_comparison_runtime_actions"
                    ]
                )

    def test_huff0_proxy_is_separate_from_huff0_source_requirements(self) -> None:
        huff0 = self.by_label["Huff0"]
        self.assertEqual(huff0["required_path"], ["entropy_huffman"])
        extra = {
            item["requirement"]
            for item in huff0["original_algorithm_requirement"][
                "additional_required_exactness"
            ]
        }
        self.assertEqual(
            extra,
            {
                "huff0_histogram_and_table_build",
                "huff0_table_description",
                "huff0_single_or_four_stream_bitstream",
            },
        )
        historical = huff0["requirements_by_logical_type"]["int64"][
            "audited_runtime_sequence"
        ]
        self.assertEqual(
            historical["ordered_historical_a_ids"],
            ["A065", "A057", "A060", "A067"],
        )
        self.assertEqual(
            historical["status"],
            "project_proxy_not_original_algorithm_decomposition",
        )

    def test_zstd_preserves_runtime_sequence_and_exactness_gaps(self) -> None:
        zstd = self.by_label["Zstd"]
        self.assertEqual(
            zstd["required_path"],
            ["lz_match", "match_parse_policy", "literal_entropy", "entropy_fse"],
        )
        requirement = zstd["requirements_by_logical_type"]["int64"]
        self.assertEqual(
            requirement["audited_runtime_sequence"]["ordered_runtime_actions"],
            [
                "op.lz_match.zstd_lazy65535.cpp_reference_v1",
                "op.match_parse_policy.generic_packet.cpp_reference_v1",
                "op.literal_entropy.zstd_huffman.cpp_reference_v1",
                "op.entropy_fse.zstd_rans.cpp_reference_v1",
                "op.block_state_control.zstd_residual_core.cpp_v1",
            ],
        )
        exactness = {
            gap["requirement"]
            for gap in requirement["required_gaps"]
            if gap["kind"] == "method_exactness"
        }
        self.assertEqual(
            exactness,
            {
                "literal_huff0_table_and_stream",
                "ll_ml_of_sequence_generation",
                "ll_ml_of_fse_tables_and_bitstream",
                "repcode_history",
                "cross_block_and_frame_state",
            },
        )
        self.assertEqual(
            requirement["direct_method_evidence"]["status"],
            "historical_direct_baseline_implementation",
        )
        self.assertNotEqual(
            requirement["audited_runtime_sequence"]["status"],
            "historical_audited_executable_runtime_sequence",
        )

    def test_historical_audit_is_not_presented_as_current_refresh(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["historical_audit_snapshot"]["manifest_a_implementation_count_verbatim"],
                106,
            )
            for logical_type in ("int64", "float64"):
                requirement = row["requirements_by_logical_type"][logical_type]
                self.assertIn(
                    "historical",
                    requirement["audited_runtime_sequence"]["evidence_time_scope"],
                )
                self.assertEqual(
                    requirement["direct_method_evidence"][
                        "current_native_baseline86_refresh_status"
                    ],
                    "not_refreshed_by_this_builder",
                )

    def test_source_receipt_covers_requested_inputs(self) -> None:
        receipt = MODULE.build_source_receipt()
        self.assertEqual(set(receipt), set(MODULE.CORE_INPUTS))
        self.assertIn("scripts/primitive_operator_action_runtime.py", receipt)
        for value in receipt.values():
            self.assertTrue(value["exists"])
            self.assertGreater(value["bytes"], 0)
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]))


if __name__ == "__main__":
    unittest.main()
