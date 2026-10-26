#!/usr/bin/env python3
"""Migration preserves real numeric payloads and rejects unproven inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import migrate_weighted_grpo_prepared_role_v1 as migration
import prepare_weighted_grpo_timeseries_v1 as preparation
import rebuild_weighted_grpo_deep_space_v1 as deep
from test_formal_prepared_role_wiring_v1 import materialize
import test_rebuild_weighted_grpo_deep_space_v1 as deep_fixtures


def save_manifest(root: Path, manifest: dict) -> bytes:
    manifest["provenance_sha256"] = hashlib.sha256(preparation._canonical_json(
        preparation._manifest_fingerprint_basis(manifest)
    )).hexdigest()
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    (root / "dataset_manifest.json").write_bytes(data)
    return data


class PreparedRoleMigrationTests(unittest.TestCase):
    def _legacy(self, root: Path):
        cohort = preparation.load_cohort_manifest()
        entry = next(item for item in cohort["datasets"]
                     if item["role"] == "train_validation")
        prepared = materialize(root / "input", entry, cohort, 0)
        manifest = json.loads((prepared / "dataset_manifest.json").read_bytes())
        manifest.pop("split_contract")
        for entity in manifest["entities"]:
            entity.pop("test_rows")
        original = save_manifest(prepared, manifest)
        return prepared, manifest, original

    def test_copy_migration_preserves_old_data_and_passes_strict_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, original, original_bytes = self._legacy(root)
            output = root / "migrated"
            receipt = migration.migrate_prepared(source, output)
            migrated = json.loads((output / "dataset_manifest.json").read_bytes())
            self.assertEqual((source / "dataset_manifest.json").read_bytes(), original_bytes)
            self.assertEqual((output / "migration/input_dataset_manifest.json").read_bytes(), original_bytes)
            self.assertEqual(receipt["input"]["provenance_sha256"], original["provenance_sha256"])
            self.assertNotEqual(receipt["input"]["provenance_sha256"], receipt["output"]["provenance_sha256"])
            self.assertEqual(migrated["blocks"], original["blocks"])
            self.assertEqual(migrated["selection"], original["selection"])
            for entity in migrated["entities"]:
                self.assertEqual(entity.pop("test_rows"), 0)
            migrated.pop("split_contract")
            migrated["provenance_sha256"] = original["provenance_sha256"]
            self.assertEqual(migrated, original)
            verified = preparation.verify_prepared_dataset(
                output, verify_source=True, expected_role="train_validation",
            )
            self.assertEqual(verified, receipt["strict_verification"])
            for row in receipt["payloads"]:
                self.assertEqual((source / row["path"]).read_bytes(),
                                 (output / row["path"]).read_bytes())
                self.assertNotEqual((source / row["path"]).stat().st_ino,
                                    (output / row["path"]).stat().st_ino)

    def test_explicit_hardlink_mode_preserves_all_payload_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _manifest, _original = self._legacy(root)
            output = root / "linked"
            receipt = migration.migrate_prepared(source, output, transfer="hardlink")
            for row in receipt["payloads"]:
                self.assertEqual((source / row["path"]).stat().st_ino,
                                 (output / row["path"]).stat().st_ino)
                self.assertEqual(preparation.sha256_file(output / row["path"]), row["sha256"])

    def test_changed_source_or_payload_is_rejected_without_output(self):
        for kind in ("source", "payload"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, manifest, _original = self._legacy(root)
                path = (Path(manifest["source"]["path"]) if kind == "source"
                        else source / manifest["blocks"][0]["values_path"])
                data = bytearray(path.read_bytes())
                data[0] ^= 1
                path.write_bytes(data)
                output = root / "rejected"
                with self.assertRaises(preparation.PreparationError):
                    migration.migrate_prepared(source, output)
                self.assertFalse(output.exists())

    def test_unfrozen_cohort_and_changed_boundary_are_rejected(self):
        for kind in ("cohort", "boundary"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, manifest, _original = self._legacy(root)
                if kind == "cohort":
                    manifest["cohort"]["assignment_sha256"] = "f" * 64
                else:
                    manifest["entities"][0]["train_rows"] -= 1
                    manifest["entities"][0]["validation_rows"] += 1
                save_manifest(source, manifest)
                output = root / "rejected"
                with self.assertRaises((migration.MigrationError, preparation.PreparationError)):
                    migration.migrate_prepared(source, output)
                self.assertFalse(output.exists())

    def test_existing_output_and_nested_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _manifest, _original = self._legacy(root)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(migration.MigrationError, "must not exist"):
                migration.migrate_prepared(source, existing)
            with self.assertRaisesRegex(migration.MigrationError, "non-nested"):
                migration.migrate_prepared(source, source / "nested")

    def test_atomic_publication_cannot_replace_an_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary, existing = root / "temporary", root / "existing"
            temporary.mkdir()
            existing.mkdir()
            (temporary / "value").write_bytes(b"x")
            with self.assertRaises(FileExistsError):
                migration._publish_new_directory(temporary, existing)
            self.assertTrue((temporary / "value").is_file())
            self.assertEqual(list(existing.iterdir()), [])

    def test_failed_strict_verification_never_publishes_or_changes_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _manifest, original_bytes = self._legacy(root)
            output = root / "rejected"
            verifier = preparation.verify_prepared_dataset

            def corrupt_before_strict(path, **kwargs):
                if kwargs.get("expected_role") == "train_validation":
                    manifest = json.loads((path / "dataset_manifest.json").read_bytes())
                    payload = path / manifest["blocks"][0]["values_path"]
                    payload.write_bytes(payload.read_bytes() + b"x")
                return verifier(path, **kwargs)

            with mock.patch.object(preparation, "verify_prepared_dataset", corrupt_before_strict):
                with self.assertRaises(preparation.PreparationError):
                    migration.migrate_prepared(source, output)
            self.assertFalse(output.exists())
            self.assertEqual((source / "dataset_manifest.json").read_bytes(), original_bytes)
            self.assertEqual(list(root.glob(".rejected*")), [])
            verifier(source, verify_source=True)

    def test_real_deep_space_prepared_parquet_and_canonical_context_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = deep_fixtures.DeepSpaceRebuildTest._write_fixture(root)
            spec = deep_fixtures.DeepSpaceRebuildTest._contract(parquet)
            canonical = root / "canonical"
            deep.rebuild_canonical(spec, canonical, enforce_production_shape=False)
            source = root / "prepared"
            before = deep.prepare_experiment_dataset(
                canonical, source, dataset_byte_cap=10 * (len(deep.VALUE_COLUMNS) + 1) * 8,
                block_byte_cap=41,
            )
            output = root / "migrated"
            receipt = migration.migrate_prepared(source, output)
            after = json.loads((output / "dataset_manifest.json").read_bytes())
            self.assertEqual(before["source"]["format"], "parquet")
            self.assertEqual(before["canonical_rebuild"], after["canonical_rebuild"])
            self.assertEqual(before["blocks"], after["blocks"])
            self.assertEqual(before["input_contract"], after["input_contract"])
            self.assertEqual(before["totals"], after["totals"])
            self.assertTrue(receipt["strict_verification"]["source_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
