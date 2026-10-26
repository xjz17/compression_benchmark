#!/usr/bin/env python3
"""Bounded native preflight on three tiny fixtures, never a dataset benchmark.

Run every declared codec configuration in both wire/ports presentations on
fixtures of its accepted input kinds. Probe the frozen layout modes separately.
A raw-fixture rejection is not evidence that a packet-dependent stage is broken.
The report preserves unresolved native rejections and every missing dependency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from unittest import mock

import exhaustive_graph_runtime_v1 as runtime
from exhaustive_graph_search_v1 import TransitionSpace, frozen_domain
import operator_taxonomy_v30_v4 as taxonomy
import unified_action_runtime as unified


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fixtures():
    values = [1, 1, 2, 3, 5, 8, 13, 21] * 4
    return [
        ("tiny_int64", "int64", runtime.Stream("numeric", 8, struct.pack("<32q", *values))),
        ("tiny_float64", "float64", runtime.Stream("numeric", 8,
            struct.pack("<32d", *([1.25, 1.25, 2.5, 3.75, 5.0, 8.125, 13.5, 21.75] * 4)))),
        ("tiny_bytes", "bytes", runtime.Stream("serial", 1,
            bytes([1, 1, 2, 3, 5, 8, 13, 21]) * 32)),
    ]


def error_category(exc, phase):
    if phase == "roundtrip":
        if isinstance(exc, (FileNotFoundError, ModuleNotFoundError)):
            return "runtime_missing"
        if isinstance(exc, subprocess.TimeoutExpired):
            return "native_timeout"
        return "roundtrip_failure"
    category = runtime.classify_exception(exc)
    if category in ("runtime_missing", "unsupported", "invalid_input"):
        return category
    if isinstance(exc, subprocess.TimeoutExpired):
        return "native_timeout"
    if isinstance(exc, unified.ActionNotExecutableError):
        return "unsupported"
    if isinstance(exc, (ValueError, struct.error)):
        # Native CLIs commonly reject incomplete predecessor packets using
        # ValueError. Their diagnostic alone cannot establish a runtime bug.
        return "native_rejection_unresolved"
    return "runtime_exception"


def probe(native_timeout_seconds=15.0):
    options, domain = frozen_domain()
    inputs = fixtures()
    attempts, exclusions = [], []
    coverage = defaultdict(lambda: {"attempts": 0, "passed": 0,
                                    "categories": Counter(), "presentations_passed": set(),
                                    "fixtures_passed": set(), "parameters_passed": set()})
    actual_run = subprocess.run
    native_executables = set()

    def bounded_run(*args, **kwargs):
        # All called adapters use subprocess.run; timeout handling in run()
        # terminates and waits for that native child before propagating.
        kwargs.setdefault("timeout", native_timeout_seconds)
        command = args[0] if args else kwargs.get("args")
        if isinstance(command, (tuple, list)) and command:
            native_executables.add(str(command[0]))
        return actual_run(*args, **kwargs)

    def execute(fixture_name, source_kind, stream, implementation, transition, family):
        state = runtime.GraphState.from_streams([stream], source_views=[source_kind],
                                               max_point=3, max_depth=1)
        start = time.monotonic_ns()
        record = {"fixture": fixture_name, "implementation": implementation,
                  "family": family, "transition": transition, "status": "passed"}
        phase = "forward"
        try:
            result = state.apply(**transition)
            frame = result.stop()
            phase = "roundtrip"
            restored = runtime.restore_frame(frame)
            if restored != state.streams:
                raise AssertionError("byte/type mismatch after complete-frame inverse")
            record.update(compressed_bytes=len(frame), original_bytes=len(stream.data),
                          compression_ratio=len(frame) / len(stream.data),
                          output_streams=len(result.streams),
                          exposure_gaps=list(result.exposure_gaps),
                          frame_sha256=digest(frame))
        except Exception as exc:
            record.update(status=error_category(exc, phase), phase=phase,
                          exception=type(exc).__name__, detail=str(exc)[:2000])
        record["probe_elapsed_ns"] = time.monotonic_ns() - start
        attempts.append(record)
        row = coverage[implementation]
        row["attempts"] += 1
        row["categories"][record["status"]] += 1
        if record["status"] == "passed":
            row["passed"] += 1
            row["presentations_passed"].add(transition.get("output_mode", "ports"))
            row["fixtures_passed"].add(fixture_name)
            row["parameters_passed"].add(json.dumps({
                "io_variant": transition.get("io_variant"),
                "parameters": transition.get("parameters", {})}, sort_keys=True))

    started = time.time_ns()
    with mock.patch.object(subprocess, "run", side_effect=bounded_run):
        for option in options:
            descriptor = taxonomy.describe_action(option.action)
            for fixture_name, source_kind, stream in inputs:
                if source_kind not in option.input_kinds:
                    exclusions.append({"fixture": fixture_name, "implementation": option.name,
                                       "io_variant": option.io_variant,
                                       "parameters": option.parameters,
                                       "reason": "fixture kind outside declared accepted_input_kinds"})
                    continue
                for mode in ("ports", "wire"):
                    execute(fixture_name, source_kind, stream, option.name, {
                        "action": option.action, "input_stream_ids": ["s0"],
                        "parameters": option.parameters, "io_variant": option.io_variant,
                        "input_view": source_kind, "max_point": 3,
                        "output_mode": mode}, descriptor.operator_id)
        for fixture_name, source_kind, stream in inputs:
            state = runtime.GraphState.from_streams([stream], source_views=[source_kind])
            choices = TransitionSpace(state, [], 1, 3, layouts=True, concat=False)
            for index in range(choices.count):
                transition = choices.at(index)
                mechanism = transition["action"]
                descriptor = next(d for d in taxonomy.CODEC_ACTIONS if d.mechanism == mechanism)
                execute(fixture_name, source_kind, stream, descriptor.name,
                        transition, descriptor.operator_id)
    statuses = []
    for descriptor in taxonomy.OPERATOR_ACTIONS:
        result = {"action_id": descriptor.action_id, "name": descriptor.name,
                  "kind": descriptor.kind, "family": descriptor.operator_id}
        if descriptor.kind == "model_provider":
            result.update(status="blocked_graph_entropy_not_connected", attempts=0,
                          passed=0, note="Probability-only provider; never counted as compressed output.")
        else:
            measured = coverage[descriptor.name]
            result.update({k: dict(v) if isinstance(v, Counter) else sorted(v) if isinstance(v, set) else v
                           for k, v in measured.items()})
            result["status"] = ("representative_roundtrip_passed" if measured["passed"] else
                                "unverified_on_raw_fixtures")
            result["note"] = "Passing tiny inputs does not establish global input applicability or complete semantic decomposition."
        statuses.append(result)
    categories = Counter(r["status"] for r in attempts)
    report = {
        "schema": "exhaustive-graph-bounded-preflight-v1", "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(), "native_subprocess_timeout_seconds": native_timeout_seconds,
        "runtime_version": runtime.RUNTIME_VERSION, "runtime_registry_sha256": runtime.registry_fingerprint(),
        "probe_source_sha256": digest(Path(__file__).read_bytes()),
        "frozen_domain_sha256": domain["sha256"],
        "source_fingerprints": domain["source_fingerprints"],
        "native_executables_used": [{"path": name, "exists": Path(name).is_file(),
            "sha256": digest(Path(name).read_bytes()) if Path(name).is_file() else None}
            for name in sorted(native_executables)],
        "scope": "Only three synthetic 32-value/256-byte finite fixtures; no live dataset or depth search.",
        "fixtures": [{"name": name, "input_kind": kind, "raw_bytes": len(s.data),
                       "sha256": digest(s.data), "stream_kind": s.kind, "stream_width": s.width}
                      for name, kind, s in inputs],
        "declared_codec_configurations": len(options),
        "codec_implementation_count": len(taxonomy.CODEC_ACTIONS),
        "probability_provider_count": len(taxonomy.MODEL_PROVIDERS),
        "codec_implementations_attempted": sum(r["kind"] == "codec" and r["attempts"] > 0 for r in statuses),
        "codec_implementations_with_representative_roundtrip": sum(r["kind"] == "codec" and r["passed"] > 0 for r in statuses),
        "attempt_count": len(attempts), "outcomes": dict(categories),
        "declared_type_exclusion_count": len(exclusions),
        "search_scope_complete": False, "global_optimality_proven": False,
        "proof_gaps": [*runtime.registry_inventory()["scope_gaps"],
                       *domain["layout_domain_gaps"],
                       "Bounded raw-fixture preflight does not explore predecessor packets or all reachable graph inputs.",
                       "Native raw-input rejections remain unresolved; they are neither silently excluded nor declared broken.",
                       "Any missing dependency, timeout, roundtrip failure or runtime exception must be resolved before coverage certification."],
        "operators": statuses, "attempts": attempts, "declared_type_exclusions": exclusions,
        "timing_note": "probe_elapsed_ns is diagnostic whole-probe latency, not compression/decompression benchmark time.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-timeout-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.native_timeout_seconds <= 0:
        parser.error("native timeout must be positive")
    report = probe(args.native_timeout_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "codec_implementation_count", "probability_provider_count",
        "codec_implementations_attempted", "codec_implementations_with_representative_roundtrip",
        "attempt_count", "outcomes", "search_scope_complete")}, ensure_ascii=False))
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
