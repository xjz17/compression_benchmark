"""Small independent oracles for exhaustive traversal, resume and merge choices."""
from dataclasses import dataclass, replace
import itertools
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import exhaustive_graph_search_v1 as search


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


if __name__ == "__main__":
    unittest.main()
