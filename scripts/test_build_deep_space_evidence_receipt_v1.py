#!/usr/bin/env python3
"""Focused tests for the deep-space evidence receipt assembler."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_deep_space_evidence_receipt_v1 as receipt_builder  # noqa: E402
from rebuild_weighted_grpo_deep_space_v1 import DeepSpaceSpec  # noqa: E402


class DeepSpaceEvidenceReceiptTest(unittest.TestCase):
    def test_receipt_binds_logical_contract_and_both_verified_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            audit = root / "audit.json"
            canonical_root = root / "canonical"
            prepared_root = root / "prepared"
            canonical_root.mkdir()
            prepared_root.mkdir()
            spec.write_text('{"spec":1}\n', encoding="utf-8")
            audit.write_text(
                json.dumps({"audit": 1, "numeric_schema_sha256": "e" * 64}),
                encoding="utf-8",
            )
            contract_hash = "a" * 64
            canonical = {
                "preparation_spec_sha256": contract_hash,
                "provenance_sha256": "b" * 64,
                "totals": {
                    "blocks": 3,
                    "original_numeric_bytes": 24,
                },
            }
            prepared = {
                "preparation_spec_sha256": contract_hash,
                "provenance_sha256": "c" * 64,
                "selection": {
                    "source_rows": 5,
                    "selected_rows": 4,
                    "split": "per_entity_chronological_80_20_before_blocks",
                },
                "totals": {
                    "blocks": 4,
                    "original_numeric_bytes": 32,
                    "prepared_block_bytes": 40,
                },
            }
            (canonical_root / "canonical_manifest.json").write_text(
                json.dumps(canonical), encoding="utf-8"
            )
            (prepared_root / "dataset_manifest.json").write_text(
                json.dumps(prepared), encoding="utf-8"
            )
            output = root / "receipt.json"
            contract = DeepSpaceSpec(
                path=root / "source.parquet",
                expected_sha256="d" * 64,
                expected_bytes=1,
                block_byte_cap=1024,
                spec_sha256=contract_hash,
            )
            with (
                mock.patch.object(receipt_builder.deep_space, "load_spec", return_value=contract),
                mock.patch.object(
                    receipt_builder.deep_space,
                    "validate_audit_receipt",
                    return_value={
                        "rows": 5,
                        "timestamp_only_rows": 1,
                        "numeric_schema_sha256": "e" * 64,
                    },
                ),
                mock.patch.object(
                    receipt_builder.deep_space,
                    "verify_canonical",
                    return_value={
                        "rows": 5,
                        "timestamp_only_rows": 1,
                        "provenance_sha256": "b" * 64,
                    },
                ),
                mock.patch.object(
                    receipt_builder,
                    "verify_prepared_dataset",
                    return_value={"provenance_sha256": "c" * 64},
                ),
            ):
                receipt = receipt_builder.build_receipt(
                    spec_path=spec,
                    audit_path=audit,
                    canonical_root=canonical_root,
                    prepared_root=prepared_root,
                    output_path=output,
                    repository_root=root,
                )
            self.assertEqual(receipt["formal_spec"]["contract_sha256"], contract_hash)
            self.assertEqual(receipt["formal_spec"]["path"], "spec.json")
            self.assertEqual(
                receipt["verified_local_canonical_rebuild"]["provenance_sha256"],
                "b" * 64,
            )
            self.assertEqual(
                receipt["verified_local_prepared_artifact"]["provenance_sha256"],
                "c" * 64,
            )
            self.assertEqual(json.loads(output.read_text()), receipt)


if __name__ == "__main__":
    unittest.main()
