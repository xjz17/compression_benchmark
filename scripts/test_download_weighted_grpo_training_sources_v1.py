#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import download_weighted_grpo_training_sources_v1 as downloader  # noqa: E402


class FakeLfs:
    def __init__(self, payload: bytes):
        self.size = len(payload)
        self.sha256 = hashlib.sha256(payload).hexdigest()


class FakeEntry:
    def __init__(self, path: str, payload: bytes, *, lfs=True):
        self.path = path
        self.size = len(payload)
        self.lfs = FakeLfs(payload) if lfs else None
        digest = hashlib.sha1()
        digest.update(f"blob {len(payload)}\0".encode("ascii"))
        digest.update(payload)
        self.blob_id = digest.hexdigest()


class FakeFrozenEntry:
    def __init__(self, path: str, size: int, git_oid: str, lfs_sha256=None):
        self.path = path
        self.size = size
        self.blob_id = git_oid
        if lfs_sha256 is None:
            self.lfs = None
        else:
            self.lfs = type("FrozenLfs", (), {
                "size": size,
                "sha256": lfs_sha256,
            })()


class FakeApi:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def list_repo_tree(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return iter(self.entries[repo_id])


class FrozenSourceDownloaderTests(unittest.TestCase):
    def test_inventory_requires_exact_revision_and_lfs_digest(self):
        payload = b"tsfile-bytes"
        api = FakeApi({
            "THULab/toy": [FakeEntry("nested/a.tsfile", payload),
                            FakeEntry("README.md", b"ignored")],
        })
        inventory = downloader.resolve_inventory([{
            "dataset_id": "toy", "repo_id": "THULab/toy",
            "revision": "a" * 40,
        }], api=api)
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0].relative_path, Path("toy/nested/a.tsfile"))
        self.assertEqual(inventory[0].sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(api.calls[0][1]["revision"], "a" * 40)
        self.assertTrue(api.calls[0][1]["recursive"])

    def test_verify_file_checks_size_and_sha256(self):
        payload = b"123456"
        frozen = downloader.FrozenFile(
            "toy", "THULab/toy", "b" * 40, "a.tsfile",
            len(payload), hashlib.sha256(payload).hexdigest(),
            "c" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.tsfile"
            path.write_bytes(payload)
            self.assertTrue(downloader.verify_file(path, frozen))
            path.write_bytes(payload + b"x")
            self.assertFalse(downloader.verify_file(path, frozen))

    def test_non_lfs_file_is_verified_by_git_blob_oid(self):
        payload = b"ordinary-git-blob"
        entry = FakeEntry("a.tsfile", payload, lfs=False)
        api = FakeApi({"THULab/toy": [entry]})
        frozen = downloader.resolve_inventory([{
            "dataset_id": "toy", "repo_id": "THULab/toy",
            "revision": "d" * 40,
        }], api=api)[0]
        self.assertIsNone(frozen.sha256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.tsfile"
            path.write_bytes(payload)
            self.assertTrue(downloader.verify_file(path, frozen))

    def test_deep_space_uses_only_exact_upstream_parquet_overlay(self):
        api = FakeApi({
            downloader.deep_space.UPSTREAM_REPO: [
                FakeFrozenEntry(
                    downloader.deep_space.UPSTREAM_REPO_PATH,
                    downloader.deep_space.SOURCE_BYTES,
                    downloader.deep_space.SOURCE_GIT_OID,
                    downloader.deep_space.SOURCE_SHA256,
                ),
                FakeFrozenEntry(
                    downloader.deep_space.UPSTREAM_README_PATH,
                    downloader.deep_space.README_BYTES,
                    downloader.deep_space.README_GIT_OID,
                ),
                FakeEntry("invalid-mirror.tsfile", b"must-not-be-selected"),
            ],
        })
        inventory = downloader.resolve_inventory([{
            "dataset_id": downloader.deep_space.DATASET_ID,
            "repo_id": downloader.deep_space.UPSTREAM_REPO,
            "revision": downloader.deep_space.UPSTREAM_REVISION,
        }], api=api)
        self.assertEqual(
            {item.repo_path for item in inventory},
            {
                downloader.deep_space.UPSTREAM_REPO_PATH,
                downloader.deep_space.UPSTREAM_README_PATH,
            },
        )
        self.assertNotIn("invalid-mirror.tsfile", {
            item.repo_path for item in inventory
        })

    def test_revision_spec_rejects_unmarked_non_thulab_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({
                "schema": "weighted-grpo-training-source-revisions-v1",
                "sources": [{
                    "dataset_id": "deep-space-probes",
                    "repo_id": "juliensimon/deep-space-probes",
                    "revision": downloader.deep_space.UPSTREAM_REVISION,
                }],
            }), encoding="utf-8")
            with self.assertRaises(downloader.SourceDownloadError):
                downloader.load_revision_spec(path)

    def test_revision_spec_rejects_moving_or_duplicate_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({
                "schema": "weighted-grpo-training-source-revisions-v1",
                "sources": [
                    {"dataset_id": "toy", "repo_id": "THULab/toy",
                     "revision": "main"},
                ],
            }), encoding="utf-8")
            with self.assertRaises(downloader.SourceDownloadError):
                downloader.load_revision_spec(path)

    def test_verify_lock_is_path_safe_and_recomputes_every_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"verified-cohort-file"
            relative = Path("toy/a.tsfile")
            path = root / relative
            path.parent.mkdir()
            path.write_bytes(payload)
            basis = {
                "schema": downloader.SCHEMA,
                "source_count": 1,
                "file_count": 1,
                "total_bytes": len(payload),
                "files": [{
                    "path": str(relative), "size": len(payload),
                    "verified_sha256": hashlib.sha256(payload).hexdigest(),
                }],
            }
            basis["lock_sha256"] = hashlib.sha256(
                downloader._canonical_json(basis)
            ).hexdigest()
            (root / "training_source_lock.json").write_text(
                json.dumps(basis), encoding="utf-8"
            )
            result = downloader.verify_lock(root)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["file_count"], 1)
            path.write_bytes(payload + b"corrupt")
            with self.assertRaises(downloader.SourceDownloadError):
                downloader.verify_lock(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
