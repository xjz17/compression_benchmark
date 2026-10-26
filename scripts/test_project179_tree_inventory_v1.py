#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("project179_tree_inventory_v1.py")
SPEC = importlib.util.spec_from_file_location("project179_tree_inventory_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Project179TreeInventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "source"
        self.root.mkdir()
        (self.root / "dataset-a").mkdir()
        (self.root / "dataset-a" / "数值.bin").write_bytes(b"\x00\x01\x02")
        (self.root / "top.json").write_text('{"ok": true}\n', encoding="utf-8")

    def create_manifest(self) -> Path:
        output = self.base / "inventory.json"
        result = MODULE.main(
            [
                "create",
                "--root",
                str(self.root),
                "--output",
                str(output),
                "--root-label",
                "fixture",
            ]
        )
        self.assertEqual(result, 0)
        return output

    def test_create_and_verify_relocated_copy(self) -> None:
        manifest_path = self.create_manifest()
        manifest = MODULE.load_inventory(manifest_path)
        self.assertEqual(manifest["file_count"], 2)
        self.assertNotIn(str(self.root), manifest_path.read_text(encoding="utf-8"))

        relocated = self.base / "relocated"
        relocated.mkdir()
        (relocated / "dataset-a").mkdir()
        (relocated / "dataset-a" / "数值.bin").write_bytes(b"\x00\x01\x02")
        (relocated / "top.json").write_text('{"ok": true}\n', encoding="utf-8")
        receipt_path = self.base / "receipt.json"
        result = MODULE.main(
            [
                "verify",
                "--root",
                str(relocated),
                "--manifest",
                str(manifest_path),
                "--receipt",
                str(receipt_path),
            ]
        )
        self.assertEqual(result, 0)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["tree_sha256"], manifest["tree_sha256"])

    def test_content_change_is_rejected(self) -> None:
        manifest = MODULE.load_inventory(self.create_manifest())
        (self.root / "top.json").write_text('{"ok": false}\n', encoding="utf-8")
        with self.assertRaisesRegex(MODULE.InventoryError, "content mismatch"):
            MODULE.verify_inventory(self.root, manifest)

    def test_missing_and_extra_files_are_rejected(self) -> None:
        manifest = MODULE.load_inventory(self.create_manifest())
        (self.root / "top.json").unlink()
        (self.root / "extra.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(MODULE.InventoryError, "file set mismatch") as context:
            MODULE.verify_inventory(self.root, manifest)
        self.assertIn("top.json", str(context.exception))
        self.assertIn("extra.bin", str(context.exception))

    def test_symlink_is_rejected(self) -> None:
        (self.root / "link").symlink_to("top.json")
        with self.assertRaisesRegex(MODULE.InventoryError, "symlink is forbidden"):
            MODULE.build_inventory(self.root, root_label="fixture")

    def test_manifest_cannot_be_inside_root(self) -> None:
        result = MODULE.main(
            [
                "create",
                "--root",
                str(self.root),
                "--output",
                str(self.root / "inventory.json"),
                "--root-label",
                "fixture",
            ]
        )
        self.assertEqual(result, 1)
        self.assertFalse((self.root / "inventory.json").exists())

    def test_manifest_tampering_is_rejected(self) -> None:
        manifest_path = self.create_manifest()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["entries"][0]["size_bytes"] += 1
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.InventoryError, "total_size_bytes"):
            MODULE.load_inventory(manifest_path)


if __name__ == "__main__":
    unittest.main()
