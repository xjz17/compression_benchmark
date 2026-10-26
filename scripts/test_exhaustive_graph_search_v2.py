"""Small independent oracles for exhaustive traversal, resume and merge choices."""
from dataclasses import dataclass, replace
import itertools
import os
import pickle
import sqlite3
import subprocess
import sys
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from exhaustive_graph_checkpoint_objects_v2 import CheckpointObjectError, ExternalObjectStore

import exhaustive_graph_search_v2 as search


@dataclass(frozen=True)
class ToyStream:
    kind: str
    width: int
    data: bytes


@dataclass(frozen=True)
class ToyState:
    streams: dict
    depths: dict
    codec_count: int = 0
    letters: str = ""
    max_depth: int = 2

    @classmethod
    def from_streams(cls, streams, *, max_depth, **kwargs):
        return cls({f"s{i}": stream for i, stream in enumerate(streams)},
                   {f"s{i}": 0 for i in range(len(streams))}, max_depth=max_depth)

    def apply(self, action, input_stream_ids, **kwargs):
        if action == "FAIL":
            raise ValueError("toy executable failure")
        if action == "INTERRUPT":
            raise KeyboardInterrupt("simulate kill before transaction commit")
        key = input_stream_ids[0]
        source = self.streams[key]
        # A/B remove exactly two/three zero bytes. Decoder obtains the count
        # from a one-byte operator record; this is an actual reversible toy
        # grammar, independent of the production enumeration implementation.
        removed = {"A": 2, "B": 3}[action]
        if not source.data.startswith(bytes(removed)):
            raise ValueError("toy grammar requires leading zero bytes")
        new_key = "s" + str(self.codec_count + 1)
        return ToyState({new_key: replace(source, data=source.data[removed:])},
                        {new_key: self.depths[key] + 1}, self.codec_count + 1,
                        self.letters + action, self.max_depth)

    def stop(self):
        if len(self.streams) != 1:
            raise ValueError("toy wire is deliberately unary")
        return b"T" + bytes([len(self.letters)]) + self.letters.encode() + next(iter(self.streams.values())).data


class ToyRuntime:
    Stream = ToyStream
    GraphState = ToyState

    @staticmethod
    def registry_inventory():
        return {"search_scope_complete": True, "scope_gaps": []}

    @staticmethod
    def classify_exception(exc):
        # Even this classification must prevent proof when execution failed.
        return "invalid_input"

    @staticmethod
    def restore_frame(frame):
        assert frame[:1] == b"T"
        n = frame[1]
        letters = frame[2:2+n].decode()
        restored = bytes(sum({"A": 2, "B": 3}[letter] for letter in letters)) + frame[2+n:]
        return {"s0": ToyStream("serial", 1, restored)}


def options(*names):
    return [search.CodecOption(name, name, None, {}, ("bytes",)) for name in names]


class ExhaustiveSearchTests(unittest.TestCase):
    raw = bytes(10) + b"nonzero tail"

    def engine(self, directory, *, names=("A", "B"), max_depth=2, domain=None, raw=None):
        return search.SearchEngine(Path(directory), ToyRuntime, options(*names),
                                   domain or {"toy_finite_grammar": "A/B removes 2/3 zero bytes"},
                                   [ToyStream("serial", 1, self.raw if raw is None else raw)],
                                   source_kind="bytes", max_depth=max_depth,
                                   layouts=False, concat=False)

    def test_exhaustive_result_equals_independent_cartesian_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            try:
                status = engine.run_slice(10)
                # Four presentations A:ports,A:wire,B:ports,B:wire at each
                # depth. No presentation or repeated implementation removed.
                self.assertEqual(status["attempts"], 4 + 4**2)
                self.assertEqual(status["states_evaluated"], 1 + 4 + 4**2)
                self.assertTrue(status["configured_space_exhausted"])
                self.assertTrue(status["global_optimal_proven"])
                candidates = []
                for depth in range(3):
                    for sequence in itertools.product(("A", "B"), repeat=depth):
                        removed = sum({"A": 2, "B": 3}[name] for name in sequence)
                        candidates.append((2 + depth + len(self.raw) - removed, sequence))
                expected_size, expected_path = min(candidates)
                incumbent = engine.meta("incumbent")
                self.assertEqual(incumbent["compressed_bytes"], expected_size)
                self.assertEqual(tuple(step["action"] for step in incumbent["recipe"]), expected_path)
                self.assertEqual(expected_path, ("B", "B"))
                engine.benchmark_incumbent(1, independent_cold=False)
                engine.export()
                self.assertEqual(json.loads((Path(directory)/"status.json").read_text())["compression_ratio"],
                                 expected_size / len(self.raw))
                self.assertEqual(ToyRuntime.restore_frame((Path(directory)/"incumbent.frame").read_bytes())["s0"].data,
                                 self.raw)
            finally:
                engine.close()

    def test_resume_reuses_cursor_without_skipping_or_recounting(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.engine(directory)
            partial = first.run_slice(1e-12)
            self.assertEqual(partial["attempts"], 1)
            self.assertFalse(partial["configured_space_exhausted"])
            first.close()
            second = self.engine(directory)
            try:
                complete = second.run_slice(10)
                self.assertEqual(complete["attempts"], 20)
                self.assertEqual(complete["valid"], 20)
                self.assertTrue(complete["global_optimal_proven"])
            finally:
                second.close()

    def test_uncommitted_interruption_does_not_advance_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory, names=("INTERRUPT",))
            try:
                with self.assertRaises(KeyboardInterrupt):
                    engine.run_slice(10)
                cursor = engine.db.execute("SELECT cursor FROM stack WHERE position=0").fetchone()[0]
                self.assertEqual(cursor, "0")
                self.assertEqual(engine.meta("stats")["transitions_attempted"], 0)
            finally:
                engine.close()

    def test_executable_rejection_permanently_blocks_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory, names=("A", "FAIL"), max_depth=1)
            try:
                status = engine.run_slice(10)
                self.assertTrue(status["configured_space_exhausted"])
                self.assertEqual(status["transitions_failed"], 2)
                self.assertFalse(status["global_optimal_proven"])
                self.assertEqual(status["failure_categories"], {"invalid_input": 2})
            finally:
                engine.close()

    def test_declared_scope_gaps_block_proof_even_after_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory, names=(), max_depth=0,
                                 domain={"layout_domain_gaps": ["unimplemented model"]})
            try:
                status = engine.run_slice(10)
                self.assertTrue(status["configured_space_exhausted"])
                self.assertFalse(status["search_scope_complete"])
                self.assertFalse(status["global_optimal_proven"])
            finally:
                engine.close()

    def test_empty_original_denominator_is_na(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory, names=(), max_depth=0, raw=b"")
            try:
                self.assertIsNone(engine.run_slice(10)["compression_ratio"])
                timing = engine.benchmark_incumbent(1, independent_cold=False)
                self.assertIsNone(timing["compression_ns_per_byte"])
                self.assertIsNone(timing["decompression_ns_per_byte"])
            finally:
                engine.close()

    def test_disk_pause_preserves_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            try:
                status = engine.run_slice(10, min_free_bytes=10**30)
                self.assertEqual(status["state"], "paused")
                self.assertEqual(status["attempts"], 0)
                self.assertFalse(status["configured_space_exhausted"])
                self.assertFalse(status["global_optimal_proven"])
            finally:
                engine.close()

    def test_all_compatible_ordered_subset_merges(self):
        state = ToyState({"a": ToyStream("numeric", 4, b"1234"),
                          "b": ToyStream("numeric", 4, b"5678"),
                          "c": ToyStream("numeric", 4, b"9012"),
                          "d": ToyStream("serial", 1, b"x")},
                         {"a": 10, "b": 10, "c": 10, "d": 10})
        space = search.TransitionSpace(state, [], 10, 18, layouts=False)
        actual = {tuple(space.at(i)["input_stream_ids"]) for i in range(space.count)}
        expected = set(itertools.permutations(("a", "b", "c"), 2)) | set(itertools.permutations(("a", "b", "c"), 3))
        self.assertEqual(actual, expected)
        self.assertEqual(space.count, 12)

    def test_per_path_depth_does_not_limit_total_branch_nodes(self):
        state = ToyState({"a": ToyStream("serial", 1, b"abc"),
                          "b": ToyStream("serial", 1, b"def")},
                         {"a": 10, "b": 9}, codec_count=25)
        space = search.TransitionSpace(state, options("A"), 10, 18, layouts=False, concat=False)
        self.assertEqual(space.count, 2)
        self.assertTrue(all(space.at(i)["input_stream_ids"] == ["b"] for i in range(space.count)))


class NativeProvenanceTests(unittest.TestCase):
    def test_changed_native_binary_refuses_resume_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'codec_cli'
            binary.write_bytes(b'original native program')
            (root / 'native_artifact_fingerprints.json').write_text(json.dumps({
                'artifacts': [{'path': 'codec_cli', 'sha256': search.sha(binary.read_bytes())}]}))
            with patch.object(search, 'ROOT', root):
                self.assertEqual(search.verified_native_artifacts()['status'], 'verified')
                binary.write_bytes(b'rebuilt native program')
                with self.assertRaisesRegex(ValueError, 'native artifact changed'):
                    search.verified_native_artifacts()


class ExternalCheckpointTests(unittest.TestCase):
    raw = ExhaustiveSearchTests.raw
    engine = ExhaustiveSearchTests.engine

    def test_sqlite_length_limit_reproduces_old_failure_and_v2_passes(self):
        raw = bytes(128 * 1024) + b"tail"
        limit = 16 * 1024
        # Both old representations independently exceed SQLite's value limit.
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE old_checkpoint(value BLOB)")
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, limit)
            old_state = ToyState.from_streams([ToyStream("serial", 1, raw)], max_depth=3)
            for value in (search.json_bytes({"frame": old_state.stop().hex()}),
                          pickle.dumps(old_state, protocol=5)):
                with self.assertRaises(sqlite3.DataError):
                    db.execute("INSERT INTO old_checkpoint VALUES (?)", (value,))
        real_connect = sqlite3.connect

        def limited_connect(*args, **kwargs):
            db = real_connect(*args, **kwargs)
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, limit)
            return db

        with tempfile.TemporaryDirectory() as directory, patch.object(search.sqlite3, "connect", limited_connect):
            engine = self.engine(directory, raw=raw, max_depth=3)
            try:
                engine.run_slice(1e-12)
                self.assertEqual(engine.meta("stats")["transitions_attempted"], 1)
                self.assertLess(engine.db.execute("SELECT max(length(value)) FROM metadata").fetchone()[0], limit)
                self.assertLess(engine.db.execute("SELECT max(length(object_ref)) FROM stack").fetchone()[0], limit)
                self.assertNotIn("frame", engine.meta("incumbent"))
                self.assertGreater(sum(p.stat().st_size for p in (Path(directory)/"objects").glob("*.bin")), limit)
            finally:
                engine.close()
            resumed = self.engine(directory, raw=raw, max_depth=3)
            try:
                status = resumed.run_slice(10)
                self.assertEqual(status["attempts"], 4 + 16 + 64)
                self.assertTrue(status["configured_space_exhausted"])
                self.assertEqual([step["action"] for step in resumed.meta("incumbent")["recipe"]], ["B"] * 3)
                timing = resumed.benchmark_incumbent(1, independent_cold=False)
                self.assertNotIn("timing_error", timing)
                self.assertEqual(timing["compression_ns_per_byte"], timing["compress_ns"] / len(raw))
                self.assertEqual(timing["decompression_ns_per_byte"], timing["decompress_ns"] / len(raw))
                resumed.export()
                frame = (Path(directory)/"incumbent.frame").read_bytes()
                self.assertEqual(ToyRuntime.restore_frame(frame)["s0"].data, raw)
                self.assertEqual(status["compression_ratio"], len(frame)/len(raw))
                # Normal reclamation leaves just the final incumbent on exhaustion.
                self.assertEqual(len(list((Path(directory)/"objects").glob("*.bin"))), 1)
            finally:
                resumed.close()

    def test_process_exit_before_and_after_commit_preserves_exact_frontier(self):
        # os._exit bypasses Python exception rollback and close(), exercising WAL
        # recovery and real orphan files, not just an orderly resume.
        for crash_point in ("after_object", "before_commit", "after_commit"):
            with self.subTest(crash_point=crash_point), tempfile.TemporaryDirectory() as directory:
                engine = self.engine(directory, max_depth=3)
                original_best = engine.meta("incumbent")
                engine.close()
                script = r'''
import os, sys
from pathlib import Path
import exhaustive_graph_search_v2 as search
from test_exhaustive_graph_search_v2 import ToyRuntime, ToyStream, options
engine = search.SearchEngine(Path(sys.argv[1]), ToyRuntime, options("A", "B"),
    {"toy_finite_grammar": "A/B removes 2/3 zero bytes"},
    [ToyStream("serial", 1, bytes(10) + b"nonzero tail")],
    source_kind="bytes", max_depth=3, layouts=False, concat=False)
point = sys.argv[2]
if point == "after_object":
    original = engine.objects.put_pickle
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(71)
    engine.objects.put_pickle = crash
elif point == "before_commit":
    original = engine.set_meta
    def crash(key, value):
        original(key, value)
        if key == "stats":
            os._exit(71)
    engine.set_meta = crash
else:
    def crash(*args, **kwargs):
        os._exit(71)
    engine.collect_garbage = crash
engine.run_slice(10)
raise AssertionError("crash hook was not reached")
'''
                process = subprocess.run([sys.executable, "-c", script, directory, crash_point],
                                         cwd=Path(__file__).parent, capture_output=True, text=True)
                self.assertEqual(process.returncode, 71, process.stderr)
                resumed = self.engine(directory, max_depth=3)
                try:
                    rows = resumed.db.execute("SELECT position,cursor FROM stack ORDER BY position").fetchall()
                    if crash_point == "after_commit":
                        self.assertEqual(rows, [(0, "1"), (1, "0")])
                        self.assertEqual(resumed.meta("stats")["transitions_attempted"], 1)
                        self.assertNotEqual(resumed.meta("incumbent"), original_best)
                    else:
                        self.assertEqual(rows, [(0, "0")])
                        self.assertEqual(resumed.meta("stats")["transitions_attempted"], 0)
                        self.assertEqual(resumed.meta("incumbent"), original_best)
                    self.assertEqual(len(list((Path(directory)/"objects").glob("*.bin"))),
                                     len({ref["sha256"] for ref in resumed.checkpoint_references()}))
                    status = resumed.run_slice(10)
                    self.assertEqual(status["attempts"], 84)
                    self.assertEqual(status["valid"], 84)
                    self.assertTrue(status["global_optimal_proven"])
                    self.assertEqual([s["action"] for s in resumed.meta("incumbent")["recipe"]], ["B"] * 3)
                finally:
                    resumed.close()

    def test_missing_corrupt_truncated_references_refuse_resume(self):
        for target in ("incumbent", "ancestor", "child"):
            for damage in ("missing", "corrupt", "truncated"):
                with self.subTest(target=target, damage=damage), tempfile.TemporaryDirectory() as directory:
                    engine = self.engine(directory)
                    engine.run_slice(1e-12)
                    if target == "incumbent":
                        reference = engine.meta("incumbent")["frame_ref"]
                    else:
                        position = 0 if target == "ancestor" else 1
                        reference = json.loads(engine.db.execute("SELECT object_ref FROM stack WHERE position=?", (position,)).fetchone()[0])
                    path = engine.objects.path(reference)
                    rows = engine.db.execute("SELECT * FROM stack ORDER BY position").fetchall()
                    engine.close()
                    data = path.read_bytes()
                    if damage == "missing":
                        path.unlink()
                    elif damage == "corrupt":
                        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
                    else:
                        path.write_bytes(data[:-1])
                    with self.assertRaises(CheckpointObjectError):
                        self.engine(directory)
                    with sqlite3.connect(Path(directory)/"checkpoint.sqlite3") as db:
                        self.assertEqual(db.execute("SELECT * FROM stack ORDER BY position").fetchall(), rows)
                    # A failed constructor must release the flock for recovery.
                    path.write_bytes(data)
                    resumed = self.engine(directory)
                    resumed.close()

    def test_checkpoint_io_failure_rolls_back_instead_of_recording_codec_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            try:
                before = engine.db.execute("SELECT * FROM stack").fetchall()
                best = engine.meta("incumbent")
                with patch.object(engine.objects, "put_bytes", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        engine.run_slice(10)
                self.assertEqual(engine.db.execute("SELECT * FROM stack").fetchall(), before)
                self.assertEqual(engine.meta("incumbent"), best)
                self.assertEqual(engine.meta("stats")["transitions_attempted"], 0)
                self.assertEqual(engine.db.execute("SELECT count(*) FROM errors").fetchone()[0], 0)
                engine.verify_checkpoint_objects()
                engine.collect_garbage()
                self.assertEqual(engine.run_slice(10)["attempts"], 20)
            finally:
                engine.close()

    def test_v1_checkpoint_rejected_without_relabelling(self):
        import exhaustive_graph_search_v1 as old
        with tempfile.TemporaryDirectory() as directory:
            engine = old.SearchEngine(Path(directory), ToyRuntime, options("A", "B"),
                                      {"toy_finite_grammar": "A/B removes 2/3 zero bytes"},
                                      [ToyStream("serial", 1, self.raw)], source_kind="bytes",
                                      max_depth=2, layouts=False, concat=False)
            original = engine.meta("config")
            engine.close()
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                self.engine(directory)
            with sqlite3.connect(Path(directory)/"checkpoint.sqlite3") as db:
                self.assertEqual(json.loads(db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()[0]), original)

    def test_resume_regenerates_initialization_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.close()
            for name in ("search_domain.json", "config.json"):
                (Path(directory)/name).unlink()
            resumed = self.engine(directory)
            try:
                self.assertEqual(json.loads((Path(directory)/"config.json").read_bytes()), resumed.config)
                self.assertEqual(json.loads((Path(directory)/"search_domain.json").read_bytes()), resumed.domain)
            finally:
                resumed.close()

    def test_cached_timing_requires_intact_frame_and_requested_cold_check(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            try:
                first = engine.benchmark_incumbent(1, independent_cold=False)
                self.assertFalse(first["independent_cold_decode_passed"])
                expected = {"s0": {"kind": "serial", "width": 1, "bytes": len(self.raw), "sha256": search.sha(self.raw)}}
                completed = subprocess.CompletedProcess([], 0, search.json_bytes(expected), b"")
                with patch.object(search.subprocess, "run", return_value=completed) as run:
                    second = engine.benchmark_incumbent(1, independent_cold=True)
                    self.assertTrue(second["independent_cold_decode_passed"])
                    run.assert_called_once()
                engine.objects.path(engine.meta("incumbent")["frame_ref"]).unlink()
                with self.assertRaises(CheckpointObjectError):
                    engine.benchmark_incumbent(1, independent_cold=True)
            finally:
                engine.close()


class ObjectStoreTests(unittest.TestCase):
    def test_export_is_an_independent_copy_and_corrupt_frame_refuses_export(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExternalObjectStore(Path(directory)/"objects")
            ref = store.put_bytes(b"complete frame" * 1024)
            destination = Path(directory)/"incumbent.frame"
            store.export(ref, destination)
            self.assertNotEqual(destination.stat().st_ino, store.path(ref).stat().st_ino)
            destination.write_bytes(b"user edited export")
            store.validate(ref)
            store.path(ref).write_bytes(b"damaged")
            with self.assertRaises(CheckpointObjectError):
                store.export(ref, destination)
            self.assertEqual(destination.read_bytes(), b"user edited export")

    def test_publication_fsyncs_file_then_directory_before_returning_reference(self):
        import exhaustive_graph_checkpoint_objects_v2 as objects
        with tempfile.TemporaryDirectory() as directory:
            store = ExternalObjectStore(Path(directory)/"objects")
            original_fsync = os.fsync
            calls = []
            def track(descriptor):
                calls.append("directory" if objects.stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
                original_fsync(descriptor)
            with patch.object(objects.os, "fsync", track):
                ref = store.put_bytes(b"durable")
            self.assertEqual(calls, ["file", "directory"])
            store.validate(ref)

    def test_orphan_collection_is_bounded_and_keeps_all_live_references(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExternalObjectStore(Path(directory)/"objects")
            refs = [store.put_bytes(bytes([i])) for i in range(8)]
            removed = store.collect(refs[:2], candidates=refs, max_scan=4)
            self.assertEqual(removed, 2)
            for ref in refs[:2]:
                store.validate(ref)
            store.collect(refs[:2])
            self.assertEqual(len(list(store.directory.glob("*.bin"))), 2)


if __name__ == "__main__":
    unittest.main()
