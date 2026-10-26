#!/usr/bin/env python3
"""Resumable exhaustive DFS over a frozen, finite typed-graph configuration space.

No beam, candidate budget, repeat prohibition, or compression-ratio pruning is
used. Every reachable state is a complete STOP candidate. Leaving a leaf
unchanged while extending another implements independent early leaf stopping.
Only exact byte/bit round trips are admitted. Execution failures are recorded
and permanently prevent an optimality proof, even when an adapter labels them
as invalid input. The current runtime has documented coverage gaps, so this
program must not describe an incumbent as a proven global optimum.

SQLite is the authoritative checkpoint index. Immutable external objects hold
complete frames and DFS state/recipe pairs, avoiding SQLite value size limits.
Objects are fsynced before their references commit with the DFS cursor atomically; an interrupted, uncommitted native call is replayed after restart.
Time slices are soft: they end at a native transition boundary. No default
per-codec timeout changes the enumerated space.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
from typing import Any

from exhaustive_graph_checkpoint_objects_v2 import ExternalObjectStore, durable_mkdir, fsync_directory

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "exhaustive-graph-search-v2-external-objects"
RUNTIME_MODULE = "exhaustive_graph_runtime_v1"


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_native_artifacts() -> dict[str, Any]:
    """Prevent a resumed search from mixing changed native implementations."""
    manifest_path = ROOT / "native_artifact_fingerprints.json"
    if not manifest_path.is_file():
        return {"status": "missing", "scope_gap": "Native binary/build/dependency manifest is absent."}
    encoded = manifest_path.read_bytes()
    manifest = json.loads(encoded)
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise ValueError("native artifact manifest has no artifacts")
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = ROOT / path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != artifact["sha256"]:
            raise ValueError(f"native artifact changed since deployment: {path}; use a new run directory")
    return {"status": "verified", "manifest_sha256": sha(encoded), "manifest": manifest}


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def normalized(ns: int | float, raw_bytes: int):
    return ns / raw_bytes if raw_bytes else None


def permutation_count(n: int, k: int) -> int:
    return math.factorial(n) // math.factorial(n - k)


def ordered_subset_at(items: tuple[str, ...], rank: int) -> tuple[str, ...]:
    """Unrank all length >=2 ordered subsets, without materializing n! items."""
    n = len(items)
    for k in range(2, n + 1):
        count = permutation_count(n, k)
        if rank >= count:
            rank -= count
            continue
        remaining, output = list(items), []
        for position in range(k):
            block = permutation_count(n - position - 1, k - position - 1)
            index, rank = divmod(rank, block)
            output.append(remaining.pop(index))
        return tuple(output)
    raise IndexError("ordered subset rank outside domain")


@dataclass(frozen=True)
class CodecOption:
    action: int | str
    name: str
    io_variant: str | None
    parameters: dict[str, Any]
    input_kinds: tuple[str, ...]

    def record(self):
        return {"action": self.action, "name": self.name,
                "io_variant": self.io_variant, "parameters": self.parameters,
                "input_kinds": list(self.input_kinds)}


def frozen_domain() -> tuple[list[CodecOption], dict[str, Any]]:
    """Freeze registry/parameter declarations; never inherit old search masks."""
    import canonical127_registry as registry
    import current_action_parameters as parameters
    import primitive_operator_action_runtime as primitive
    import unified_action_runtime as unified

    options, catalog_errors = [], []
    for group in registry.GROUPS:
        name = group["name"]
        if name in primitive.RETIRED_IDENTITY_ACTION_NAMES:
            continue
        module = parameters.BY_NAME.get(name)
        candidates = module.parameter_candidates() if module else ({},)
        for variant in group["io_variants"]:
            try:
                accepted = tuple(unified.describe_policy_action(
                    group["policy_action_id"], io_variant=variant["name"]
                ).accepted_input_kinds)
            except Exception as exc:
                accepted = tuple(variant.get("input_kinds", ()))
                catalog_errors.append({"implementation": name, "variant": variant["name"],
                                       "error": f"{type(exc).__name__}: {exc}"})
            for candidate in candidates:
                options.append(CodecOption(group["policy_action_id"], name,
                                           variant["name"], dict(candidate), accepted))
    files = sorted(set(["benchmark/reference/actions/canonical127.json",
                        "scripts/exhaustive_graph_search_v2.py",
                        "scripts/exhaustive_graph_checkpoint_objects_v2.py",
                        "scripts/exhaustive_graph_runtime_v1.py",
                        "scripts/operator_taxonomy_v30_v4.py",
                        "scripts/primitive_operator_action_runtime.py",
                        "scripts/unified_action_runtime.py",
                        "scripts/action_semantic_ports.py",
                        "scripts/current_action_parameters.py",
                        "scripts/block_state_parameters.py",
                        "scripts/native_lz_parameters.py",
                        "scripts/native_window_parameters.py",
                        "cpp/src/exhaustive_graph_carrier_cli.cpp"]))
    # Runtime imports can expand beyond the historically selected module list.
    # Freeze the complete local Python implementation surface, not just adapters.
    files = sorted(set(files) | {
        str(path.relative_to(ROOT))
        for directory in (ROOT / "benchmark/reference/actions", ROOT / "benchmark/reference/methods",
                          ROOT / "analysis_outputs/operator_implementation_dedup_20260903_v1")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in (".json", ".csv")
    } | {
        str(path.relative_to(ROOT))
        for directory in (ROOT / "scripts", ROOT / "backend/myapp")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    })
    fingerprints = {name: sha((ROOT / name).read_bytes()) for name in files
                    if (ROOT / name).is_file()}
    domain = {
        "schema": SCHEMA,
        "registry_version": primitive.ACTION_REGISTRY_VERSION,
        "source_registry_version": registry.REGISTRY_VERSION,
        "graph_registry_version": "operator-taxonomy-30-v4",
        "active_policy_action_count": len(primitive.POLICY_ACTIONS),
        "reserved_policy_action_slots": primitive.STOP_ACTION_ID,
        "retired_policy_action_ids": [5],
        "io_variant_count": len(options),
        "codec_configurations": [option.record() for option in options],
        "output_presentations": ["ports", "wire"],
        "typed_views": {"width8_numeric_or_struct": ["bytes", "int64", "float64"],
                        "other": ["bytes"]},
        "layout_configurations": {
            "columnize": "transpose with current typed source width; serial bytes use aligned widths 1,2,4,8",
            "float_deconstruct": ["fp16", "bf16", "fp32"],
            "bitsplit": ["top8", "fp", "bf16"],
            "bitunpack": {"bits": list(range(1, 65)), "unit": "bits per unpacked value",
                          "source": "cpp/src/openzl_layout_stages.cpp::bitunpack native finite domain 1..64",
                          "input_constraint": "Serial byte stream; floor(total_bits/bits)*bits must leave fewer than eight unused bits"},
            "generic_bitsplit": "no default bit partition declared; not enumerated",
            "split_struct": "no default field partition declared; not enumerated",
        },
        "layout_domain_gaps": ["generic bitsplit requires an explicit bit partition",
                               "split-struct requires explicit field widths"],
        "special_operations": {
            "STOP": "Every reachable graph is scored, including zero operators.",
            "Concat": "Every compatible subset of at least two leaves in every order; zero operator depth.",
            "continuous_Split": "Outside this run: the requested branches arise from codec outputs.",
        },
        "parameter_scope": "Declared finite parameter_candidates, named layout modes and explicit source-width defaults only; no claim over arbitrary parameter values.",
        "pruning": "Only declared type/depth incompatibility; no numeric objective pruning or duplicate-state pruning.",
        "catalog_errors": catalog_errors, "source_fingerprints": fingerprints,
        "native_artifact_provenance": verified_native_artifacts(),
    }
    domain["sha256"] = sha(json_bytes(domain))
    return options, domain


class TransitionSpace:
    """Random-access lazy choices permit constant-size DFS resume cursors."""

    def __init__(self, state, options: list[CodecOption], max_depth: int,
                 max_point: int, *, layouts: bool = True, concat: bool = True):
        self.direct: list[dict[str, Any]] = []
        self.concat_groups: list[tuple[tuple[str, ...], int]] = []
        self.declared_type_exclusions = 0
        for key in sorted(state.streams):
            stream = state.streams[key]
            if state.depths[key] >= max_depth:
                continue
            views = ("bytes", "int64", "float64") if (
                stream.width == 8 and stream.kind in ("numeric", "struct")) else ("bytes",)
            for option in options:
                for view in views:
                    if view not in option.input_kinds:
                        self.declared_type_exclusions += 2
                        continue
                    for presentation in ("ports", "wire"):
                        self.direct.append({
                            "action": option.action, "input_stream_ids": [key],
                            "parameters": option.parameters, "input_view": view,
                            "io_variant": option.io_variant, "max_point": max_point,
                            "output_mode": presentation,
                        })
            if layouts:
                def layout(action, parameters):
                    self.direct.append({"action": action, "input_stream_ids": [key],
                                        "parameters": parameters, "output_mode": "ports"})
                widths = ((1, 2, 4, 8) if stream.kind == "serial" else (stream.width,))
                widths = tuple(width for width in widths if len(stream.data) % width == 0)
                for width in widths:
                    layout("columnize", {"mode": "transpose", "width": width})
                if stream.kind == "serial" or stream.width in (2, 4):
                    for fmt, width in (("fp16", 2), ("bf16", 2), ("fp32", 4)):
                        if ((stream.kind == "serial" or stream.width == width)
                                and len(stream.data) % width == 0):
                            layout("float_deconstruct", {"format": fmt})
                for width in widths:
                    if width not in (1, 2, 4, 8):
                        continue
                    layout("bitsplit", {"mode": "top8", "width": width})
                    if width in (2, 4, 8):
                        layout("bitsplit", {"mode": "fp", "width": width})
                    if width == 2:
                        layout("bitsplit", {"mode": "bf16", "width": 2})
                if stream.kind == "serial":
                    total_bits = len(stream.data) * 8
                    for bits in range(1, 65):
                        if ((total_bits // bits) * bits + 7) // 8 == len(stream.data):
                            layout("bitunpack", {"bits": bits})
        if concat:
            compatible = defaultdict(list)
            for key, stream in sorted(state.streams.items()):
                compatible[(stream.kind, stream.width)].append(key)
            for keys in compatible.values():
                if len(keys) >= 2:
                    count = sum(permutation_count(len(keys), k) for k in range(2, len(keys) + 1))
                    self.concat_groups.append((tuple(keys), count))
        self.count = len(self.direct) + sum(n for _, n in self.concat_groups)

    def at(self, index: int) -> dict[str, Any]:
        if index < 0:
            raise IndexError(index)
        if index < len(self.direct):
            return self.direct[index]
        index -= len(self.direct)
        for keys, count in self.concat_groups:
            if index < count:
                return {"action": "concat", "input_stream_ids": list(ordered_subset_at(keys, index)),
                        "parameters": {}, "output_mode": "ports"}
            index -= count
        raise IndexError(index)


class SearchEngine:
    def __init__(self, output: Path, runtime, options: list[CodecOption], domain: dict,
                 source_streams: list, *, source_kind: str, max_depth: int = 3,
                 max_point: int = 18, source_identity: dict | None = None,
                 layouts: bool = True, concat: bool = True):
        self.output, self.runtime = Path(output), runtime
        durable_mkdir(self.output)
        self._lock = (self.output / "worker.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("another worker owns this column checkpoint") from None
        self.options, self.domain = options, domain
        self.source_streams, self.source_kind = source_streams, source_kind
        self.max_depth, self.max_point = max_depth, max_point
        self.layouts, self.concat = layouts, concat
        self.originals = {f"s{i}": stream for i, stream in enumerate(source_streams)}
        self.raw_bytes = sum(len(s.data) for s in source_streams)
        self.config = {"schema": SCHEMA, "domain_sha256": domain.get("sha256", sha(json_bytes(domain))),
                       "source_identity": source_identity or {}, "source_kind": source_kind,
                       "source_streams": [{"kind": s.kind, "width": s.width,
                                           "bytes": len(s.data), "sha256": sha(s.data)}
                                          for s in source_streams],
                       "max_depth": max_depth, "max_point": max_point,
                       "layouts": layouts, "concat": concat}
        self.db = None
        try:
            self.db = sqlite3.connect(str(self.output / "checkpoint.sqlite3"), timeout=60)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS stack (position INTEGER PRIMARY KEY, object_ref BLOB NOT NULL,
                    cursor TEXT NOT NULL, option_count TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS errors (id INTEGER PRIMARY KEY, at_ns INTEGER NOT NULL,
                    position INTEGER, choice_index TEXT, category TEXT NOT NULL,
                    transition BLOB, message TEXT NOT NULL);
            """)
            self.objects = ExternalObjectStore(self.output / "objects")
            saved = self.meta("config")
            if saved is not None:
                if saved != self.config:
                    raise ValueError("checkpoint configuration/source/runtime fingerprint mismatch; use a new run directory")
                self.verify_checkpoint_objects()
            else:
                initial = self.new_root()
                frame = initial.stop()
                if runtime.restore_frame(frame) != self.originals:
                    raise ValueError("initial raw graph failed exact round trip")
                try:
                    inventory = runtime.registry_inventory()
                except Exception as exc:
                    inventory = {"search_scope_complete": False,
                                 "inventory_error": f"{type(exc).__name__}: {exc}"}
                with self.db:
                    self.set_meta("config", self.config)
                    self.set_meta("inventory", inventory)
                    self.set_meta("stats", {"transitions_attempted": 0, "transitions_succeeded": 0,
                                           "transitions_failed": 0, "states_evaluated": 1,
                                           "states_exhausted": 0, "search_elapsed_ns": 0,
                                           "failure_categories": {}, "slices": 0})
                    self.set_meta("created_unix_ns", time.time_ns())
                    self.set_meta("incumbent", {"frame_ref": self.objects.put_bytes(frame), "frame_sha256": sha(frame),
                                                "compressed_bytes": len(frame), "recipe": [],
                                                "codec_count": 0, "maximum_path_depth": 0,
                                                "discovery_transition": 0})
                    self.push(0, initial, [])
            # Derived sidecars may be absent after a crash immediately following
            # the first SQLite commit; regenerate them on validated resume too.
            atomic_write(self.output / "search_domain.json", json_bytes(domain) + b"\n")
            atomic_write(self.output / "config.json", json_bytes(self.config) + b"\n")
            self.collect_garbage()
        except BaseException:
            self.close()
            raise

    def new_root(self):
        return self.runtime.GraphState.from_streams(
            self.source_streams, max_depth=self.max_depth, max_point=self.max_point,
            source_views={f"s{i}": self.source_kind for i in range(len(self.source_streams))})

    def meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else default

    def set_meta(self, key: str, value: Any):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, json_bytes(value)))

    def space(self, state):
        return TransitionSpace(state, self.options, self.max_depth, self.max_point,
                               layouts=self.layouts, concat=self.concat)

    def push(self, position: int, state, recipe: list):
        count = self.space(state).count
        reference = self.objects.put_pickle((state, recipe))
        self.db.execute("INSERT INTO stack VALUES (?,?,?,?)",
                        (position, json_bytes(reference), "0", str(count)))

    def checkpoint_references(self):
        references = [json.loads(row[0]) for row in
                      self.db.execute("SELECT object_ref FROM stack")]
        best = self.meta("incumbent")
        if best is not None:
            references.append(best["frame_ref"])
        return references

    def verify_checkpoint_objects(self):
        # Validate every saved ancestor, including cursor-exhausted nodes, before
        # accepting a resumed checkpoint. Missing data never advances a cursor.
        for reference in self.checkpoint_references():
            self.objects.validate(reference)
        best = self.meta("incumbent")
        if (best["frame_ref"]["sha256"] != best["frame_sha256"] or
                best["frame_ref"]["bytes"] != best["compressed_bytes"]):
            raise ValueError("incumbent object reference disagrees with frame metadata")

    def collect_garbage(self, *, candidates=None, max_scan=256):
        # The worker flock excludes another publisher/collector. Never collect
        # during a transaction: rollback may still need the previous references.
        if self.db.in_transaction:
            raise RuntimeError("object collection requires a committed checkpoint")
        return self.objects.collect(self.checkpoint_references(), candidates=candidates,
                                    max_scan=max_scan)

    def check_frame(self, frame: bytes) -> None:
        if self.runtime.restore_frame(frame) != self.originals:
            raise ValueError("complete graph byte/bit round trip mismatch")

    def record_error(self, position, cursor, transition, exc):
        classify = getattr(self.runtime, "classify_exception", lambda error: "runtime_failure")
        try:
            category = str(classify(exc))
        except Exception:
            category = "runtime_failure"
        self.db.execute("INSERT INTO errors(at_ns,position,choice_index,category,transition,message) VALUES (?,?,?,?,?,?)",
                        (time.time_ns(), position, str(cursor), category, json_bytes(transition),
                         f"{type(exc).__name__}: {exc}"))
        return category

    def run_slice(self, seconds: float, *, min_free_bytes: int = 0) -> dict:
        start = time.monotonic_ns()
        deadline = start + int(seconds * 1e9)
        stats = self.meta("stats")
        pause = "time_slice"
        # A positive slice always makes at least one transition or stack pop.
        first = True
        while first or time.monotonic_ns() < deadline:
            first = False
            if min_free_bytes and shutil.disk_usage(self.output).free < min_free_bytes:
                pause = "disk_safety_watermark"
                break
            row = self.db.execute("SELECT position,object_ref,cursor,option_count FROM stack ORDER BY position DESC LIMIT 1").fetchone()
            if row is None:
                pause = "exhausted"
                break
            position, reference_blob, cursor_text, count_text = row
            reference = json.loads(reference_blob)
            cursor, count = int(cursor_text), int(count_text)
            if cursor == count:
                self.objects.validate(reference)
                with self.db:
                    self.db.execute("DELETE FROM stack WHERE position=?", (position,))
                    stats["states_exhausted"] += 1
                    self.set_meta("stats", stats)
                self.collect_garbage(candidates=[reference])
                continue
            state, recipe = self.objects.load_pickle(reference)
            choices = self.space(state)
            if choices.count != count:
                raise RuntimeError("resumed state's choice domain changed")
            transition = choices.at(cursor)
            next_state = frame = error = None
            try:
                next_state = state.apply(**transition)
                if transition["action"] == "concat" and len(next_state.streams) >= len(state.streams):
                    raise ValueError("zero-depth Concat must strictly reduce live leaf count; otherwise enumeration can be infinite")
                if any(depth > self.max_depth for depth in next_state.depths.values()):
                    raise ValueError("runtime exceeded configured root-to-leaf operator depth")
                frame = next_state.stop()
                self.check_frame(frame)
            except Exception as exc:
                error = exc
            retired_objects = []
            with self.db:
                self.db.execute("UPDATE stack SET cursor=? WHERE position=?", (str(cursor + 1), position))
                stats["transitions_attempted"] += 1
                if error is not None:
                    category = self.record_error(position, cursor, transition, error)
                    stats["transitions_failed"] += 1
                    failures = stats["failure_categories"]
                    failures[category] = failures.get(category, 0) + 1
                else:
                    stats["transitions_succeeded"] += 1
                    stats["states_evaluated"] += 1
                    next_recipe = recipe + [transition]
                    self.push(position + 1, next_state, next_recipe)
                    best = self.meta("incumbent")
                    # Stable first-encounter tie break; timing is not an objective.
                    if len(frame) < best["compressed_bytes"]:
                        retired_objects.append(best["frame_ref"])
                        self.set_meta("incumbent", {
                            "frame_ref": self.objects.put_bytes(frame), "frame_sha256": sha(frame),
                            "compressed_bytes": len(frame), "recipe": next_recipe,
                            "codec_count": next_state.codec_count,
                            "maximum_path_depth": max(next_state.depths.values(), default=0),
                            "discovery_transition": stats["transitions_attempted"],
                        })
                self.set_meta("stats", stats)
            if retired_objects:
                self.collect_garbage(candidates=retired_objects)
        with self.db:
            stats["search_elapsed_ns"] += time.monotonic_ns() - start
            stats["slices"] += 1
            self.set_meta("stats", stats)
            self.set_meta("pause_reason", pause)
            self.set_meta("updated_unix_ns", time.time_ns())
        return self.status()

    def replay(self, recipe: list):
        state = self.new_root()
        for transition in recipe:
            state = state.apply(**transition)
        return state.stop()

    def benchmark_incumbent(self, repeats: int = 3, *, independent_cold: bool = True):
        best = self.meta("incumbent")
        self.objects.validate(best["frame_ref"])
        if (best["frame_ref"]["sha256"] != best["frame_sha256"] or
                best["frame_ref"]["bytes"] != best["compressed_bytes"]):
            raise ValueError("incumbent object reference disagrees with frame metadata")
        prior = self.meta("timing")
        if (prior and not prior.get("timing_error") and prior.get("frame_sha256") == best["frame_sha256"]
                and prior.get("repeats") == repeats
                and (not independent_cold or prior.get("independent_cold_decode_passed") is True)):
            return prior
        frame = self.objects.read_bytes(best["frame_ref"])
        encode_times, decode_times = [], []
        try:
            for _ in range(repeats):
                started = time.perf_counter_ns()
                replay = self.replay(best["recipe"])
                encode_times.append(time.perf_counter_ns() - started)
                if replay != frame:
                    raise ValueError("incumbent replay changed complete frame bytes")
                started = time.perf_counter_ns()
                restored = self.runtime.restore_frame(frame)
                decode_times.append(time.perf_counter_ns() - started)
                if restored != self.originals:
                    raise ValueError("incumbent timing round trip mismatch")
            cold = False
            if independent_cold:
                command = [sys.executable, str(Path(__file__).resolve()), "--cold-decode"]
                process = subprocess.run(command, input=frame, capture_output=True, check=False)
                if process.returncode:
                    raise RuntimeError("independent decoder failed: " + process.stderr.decode(errors="replace"))
                actual = json.loads(process.stdout)
                expected = {k: {"kind": s.kind, "width": s.width, "bytes": len(s.data), "sha256": sha(s.data)}
                            for k, s in self.originals.items()}
                if actual != expected:
                    raise ValueError("independent decoder fingerprint mismatch")
                cold = True
            compress_ns, decompress_ns = statistics.median(encode_times), statistics.median(decode_times)
            timing = {"frame_sha256": best["frame_sha256"], "repeats": repeats,
                      "aggregation": "median", "compress_ns_samples": encode_times,
                      "decompress_ns_samples": decode_times,
                      "compress_ns": compress_ns, "decompress_ns": decompress_ns,
                      "compression_ns_per_byte": normalized(compress_ns, self.raw_bytes),
                      "decompression_ns_per_byte": normalized(decompress_ns, self.raw_bytes),
                      "independent_cold_decode_passed": cold,
                      "timing_scope": "Complete graph replay/restore plus frame creation/parsing in Python; includes every native CLI launch, IPC and allocation; excludes input-file I/O, search, validation comparisons, Python interpreter startup and independent verification process. Not isolated in-process C++ kernel time."}
        except Exception as exc:
            timing = {"frame_sha256": best["frame_sha256"], "repeats": repeats,
                      "timing_error": f"{type(exc).__name__}: {exc}",
                      "compression_ns_per_byte": None, "decompression_ns_per_byte": None,
                      "independent_cold_decode_passed": False}
            with self.db:
                self.record_error(None, None, {"benchmark_incumbent": best["frame_sha256"]}, exc)
                self.set_meta("verification_failed", True)
        with self.db:
            self.set_meta("timing", timing)
        return timing

    def status(self) -> dict:
        stats, inventory = self.meta("stats"), self.meta("inventory")
        rows = self.db.execute("SELECT cursor,option_count FROM stack").fetchall()
        exhausted = not rows
        scope_complete = (inventory.get("search_scope_complete") is True
                          and not self.domain.get("catalog_errors")
                          and not self.domain.get("layout_domain_gaps"))
        no_errors = not stats["transitions_failed"] and not self.meta("verification_failed", False)
        known_pending = sum(int(total) - int(cursor) for cursor, total in rows)
        rate = (stats["transitions_attempted"] * 1e9 / stats["search_elapsed_ns"]
                if stats["search_elapsed_ns"] else None)
        best = self.meta("incumbent")
        result = {
            "schema": SCHEMA, **stats, "search_scope_complete": scope_complete,
            "state": ("exhausted" if exhausted else "paused" if self.meta("pause_reason") == "disk_safety_watermark" else "searching"),
            "attempts": stats["transitions_attempted"], "valid": stats["transitions_succeeded"],
            "elapsed_search_seconds": stats["search_elapsed_ns"] / 1e9,
            "configured_space_exhausted": exhausted,
            "global_optimal_proven": exhausted and scope_complete and no_errors,
            "incumbent_label": "best verified complete frame found so far",
            "pause_reason": self.meta("pause_reason", "initialized"),
            "dfs_stack_states": len(rows), "known_pending_transitions": known_pending,
            "transitions_per_second_observed": rate,
            "eta_seconds": None,
            "eta_note": "Exact remaining tree size depends on outputs and legal transitions; known_pending_transitions counts only saved ancestors, not their undiscovered descendants. No finite completion ETA is established.",
            "source_original_numeric_bytes": self.raw_bytes,
            "incumbent_compressed_bytes": best["compressed_bytes"],
            "compression_ratio": normalized(best["compressed_bytes"], self.raw_bytes),
            "incumbent_codec_count": best["codec_count"],
            "incumbent_maximum_path_depth": best["maximum_path_depth"],
            "max_allowed_root_to_leaf_depth": self.max_depth,
            "incumbent_frame_sha256": best["frame_sha256"],
            "incumbent": {"compression_ratio": normalized(best["compressed_bytes"], self.raw_bytes),
                          "compressed_bytes": best["compressed_bytes"], "original_numeric_bytes": self.raw_bytes,
                          "path": best["recipe"], "codec_count": best["codec_count"],
                          "max_path_depth": best["maximum_path_depth"],
                          "frame_path": str((self.output / "incumbent.frame").resolve())},
            "layout_domain_gaps": self.domain.get("layout_domain_gaps", []),
            "runtime_inventory": inventory,
            "time_slice_semantics": "Soft slice at completed-transition boundary; a slow native call can overrun the slice. No default codec timeout.",
            "created_unix_ns": self.meta("created_unix_ns"),
            "updated_unix_ns": self.meta("updated_unix_ns", self.meta("created_unix_ns")),
        }
        timing = self.meta("timing")
        if timing and timing.get("frame_sha256") == best["frame_sha256"]:
            result["incumbent_timing"] = timing
            result["incumbent"].update(timing)
        return result

    def export(self):
        best = self.meta("incumbent")
        self.objects.export(best["frame_ref"], self.output / "incumbent.frame")
        atomic_write(self.output / "incumbent_recipe.json", json_bytes(best) + b"\n")
        atomic_write(self.output / "status.json", json_bytes(self.status()) + b"\n")

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--raw-input", "--raw", dest="input", type=Path)
    parser.add_argument("--kind", choices=("int64", "float64"))
    parser.add_argument("--max-point", type=int, default=18)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--out-dir", "--output", dest="out_dir", type=Path)
    parser.add_argument("--time-slice", type=float, default=30)
    parser.add_argument("--min-free-gb", type=float, default=2)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--column", default="")
    parser.add_argument("--describe-domain", nargs="?", const="-", metavar="OUTPUT_JSON")
    parser.add_argument("--cold-decode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    runtime = importlib.import_module(RUNTIME_MODULE)
    if args.describe_domain:
        _, domain = frozen_domain()
        domain["runtime_inventory"] = runtime.registry_inventory()
        if args.describe_domain == "-":
            print(json.dumps(domain, sort_keys=True))
        else:
            destination = Path(args.describe_domain)
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(destination, json_bytes(domain) + b"\n")
        return 0
    if args.cold_decode:
        restored = runtime.restore_frame(sys.stdin.buffer.read())
        print(json.dumps({k: {"kind": s.kind, "width": s.width,
                             "bytes": len(s.data), "sha256": sha(s.data)} for k, s in restored.items()}))
        return 0
    if args.input is None or args.out_dir is None or args.kind is None:
        parser.error("--input, --kind and --out-dir are required")
    if not 0 <= args.max_depth <= 10 or not 0 <= args.max_point <= 18:
        parser.error("--max-depth must be 0..10 and --max-point must be 0..18")
    if args.time_slice <= 0 or args.timing_repeats < 1 or args.min_free_gb < 0:
        parser.error("time slice/repeats must be positive; disk watermark must be nonnegative")
    raw = args.input.read_bytes()
    if len(raw) % 8:
        parser.error("input must be whole little-endian 64-bit numeric values")
    options, domain = frozen_domain()
    engine = SearchEngine(args.out_dir, runtime, options, domain,
                          [runtime.Stream("numeric", 8, raw)], source_kind=args.kind,
                          max_depth=args.max_depth, max_point=args.max_point,
                          source_identity={"path": str(args.input.resolve()),
                                           "dataset": args.dataset, "column": args.column})
    try:
        engine.export()
        engine.run_slice(args.time_slice, min_free_bytes=int(args.min_free_gb * 1024**3))
        engine.benchmark_incumbent(args.timing_repeats)
        engine.export()
        print(json.dumps(engine.status(), sort_keys=True))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
