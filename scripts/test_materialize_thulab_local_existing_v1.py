#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("materialize_thulab_local_existing_v1.py")
SPEC = importlib.util.spec_from_file_location("materialize_thulab_local_existing_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MaterializeExistingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scope = self.root / "scope.json"
        self.receipts = self.root / "receipts"
        self.candidates = self.root / "candidates"
        self.target = self.root / "target"
        self.progress = self.root / "progress.json"
        self.receipts.mkdir()
        self.candidates.mkdir()
        self.datasets = [
            ("alpha", "THULab/alpha", "a" * 40),
            ("beta", "THULab/beta", "b" * 40),
        ]
        self._write_scope()
        self.alpha_files = {
            ".gitattributes": b"*.tsfile filter=lfs\n",
            "README.md": b"alpha\n",
            "data/train.tsfile": b"alpha-tsfile\n",
        }
        self.beta_files = {
            ".gitattributes": b"*.tsfile filter=lfs\n",
            "README.md": b"beta\n",
            "beta.tsfile": b"beta-tsfile\n",
        }
        self._write_receipt("alpha", self.alpha_files)
        self._write_receipt("beta", self.beta_files)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_scope(self) -> None:
        rows = [
            {"id": dataset_id, "repo_id": repo_id, "revision": revision}
            for dataset_id, repo_id, revision in self.datasets
        ]
        self.scope.write_text(
            json.dumps({"timestamp_dataset_count": len(rows), "timestamp_datasets": rows}),
            encoding="utf-8",
        )

    def _write_receipt(self, dataset_id: str, files: dict[str, bytes]) -> None:
        repo_id, revision = next((repo, rev) for did, repo, rev in self.datasets if did == dataset_id)
        rows = [
            {"path": path, "bytes": len(data), "sha256": digest(data)}
            for path, data in sorted(files.items())
        ]
        receipt = {
            "status": "complete",
            "dataset_id": dataset_id,
            "repo_id": repo_id,
            "revision": revision,
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "files": rows,
        }
        (self.receipts / f"{dataset_id}.json").write_text(json.dumps(receipt), encoding="utf-8")

    def _args(self, *, dry_run: bool = False):
        argv = [
            "--scope", str(self.scope),
            "--receipts-dir", str(self.receipts),
            "--target", str(self.target),
            "--candidate-root", str(self.candidates),
            "--progress-json", str(self.progress),
        ]
        if dry_run:
            argv.append("--dry-run")
        return MODULE.parse_args(argv)

    def _put(self, relative: str, data: bytes) -> Path:
        path = self.candidates / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_dry_run_finds_exact_repo_and_loose_tsfile_without_target_writes(self) -> None:
        for relative, data in self.alpha_files.items():
            self._put(f"run/raw/alpha/{relative}", data)
        loose = self._put("other/beta.tsfile", self.beta_files["beta.tsfile"])
        loose_before = loose.read_bytes()

        report, code = MODULE.materialize(self._args(dry_run=True))

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["source_files_found"], 4)
        self.assertEqual(report["counts"]["would_materialize"], 4)
        self.assertEqual(report["counts"]["missing_source"], 2)
        self.assertFalse(self.target.exists())
        self.assertEqual(loose.read_bytes(), loose_before)
        self.assertEqual(json.loads(self.progress.read_text())["schema"], MODULE.SCHEMA)

    def test_materializes_verified_files_and_is_idempotent(self) -> None:
        for relative, data in self.alpha_files.items():
            self._put(f"alpha/{relative}", data)

        first, code = MODULE.materialize(self._args())
        self.assertEqual(code, 0)
        self.assertEqual(first["source_files_found"], 3)
        self.assertEqual(first["counts"]["apfs_clone"] + first["counts"]["copy"], 3)
        for relative, data in self.alpha_files.items():
            self.assertEqual((self.target / "alpha" / relative).read_bytes(), data)

        second, code = MODULE.materialize(self._args())
        self.assertEqual(code, 0)
        self.assertEqual(second["counts"]["already_present"], 3)
        self.assertEqual(second["counts"]["missing_source"], 3)

    def test_different_destination_is_conflict_and_is_not_overwritten(self) -> None:
        source = self._put("alpha/data/train.tsfile", self.alpha_files["data/train.tsfile"])
        destination = self.target / "alpha/data/train.tsfile"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"different")

        report, code = MODULE.materialize(self._args())

        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "blocked_conflicts")
        self.assertEqual(report["counts"]["conflict"], 1)
        self.assertEqual(destination.read_bytes(), b"different")
        self.assertEqual(source.read_bytes(), self.alpha_files["data/train.tsfile"])

    def test_loose_metadata_is_not_accepted(self) -> None:
        self._put("unrelated/README.md", self.alpha_files["README.md"])
        report, code = MODULE.materialize(self._args(dry_run=True))
        self.assertEqual(code, 0)
        self.assertEqual(report["source_files_found"], 0)

    def test_rejects_unsafe_receipt_path(self) -> None:
        receipt_path = self.receipts / "alpha.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][0]["path"] = "../escape"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaises(MODULE.MaterializationError):
            MODULE.materialize(self._args(dry_run=True))

    def test_rejects_candidate_inside_target(self) -> None:
        inside = self.target / "candidate"
        inside.mkdir(parents=True)
        args = self._args(dry_run=True)
        args.candidate_root = [inside]
        with self.assertRaises(MODULE.MaterializationError):
            MODULE.materialize(args)

    def test_rejects_target_inside_candidate(self) -> None:
        args = self._args(dry_run=True)
        args.target = self.candidates / "new-target"
        with self.assertRaises(MODULE.MaterializationError):
            MODULE.materialize(args)

    def test_parent_symlink_is_conflict(self) -> None:
        source = self._put("alpha/data/train.tsfile", self.alpha_files["data/train.tsfile"])
        outside = self.root / "outside"
        outside.mkdir()
        self.target.mkdir()
        (self.target / "alpha").symlink_to(outside, target_is_directory=True)

        report, code = MODULE.materialize(self._args())

        self.assertEqual(code, 1)
        self.assertEqual(report["counts"]["conflict"], 3)
        self.assertFalse((outside / "data/train.tsfile").exists())
        self.assertEqual(source.read_bytes(), self.alpha_files["data/train.tsfile"])


if __name__ == "__main__":
    unittest.main()
