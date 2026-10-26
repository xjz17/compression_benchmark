#!/usr/bin/env python3
"""New typed numeric namespace: ZigZag is int64 -> uint64 with exactly 8N bytes.

Never imports, modifies or aliases the legacy canonical206 runtime. Packing,
zlib and framing are separately named actions and are never automatically added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpp/src/pure_numeric_operator_v1.cpp"
CLI = ROOT / "cpp/build/pure_numeric_v1/pure_numeric_operator_v1"
REGISTRY_VERSION = "pure-numeric-v1"
FRAME_MAGIC = b"PNV1"
Kind = Literal["int64", "uint64", "bytes"]
ACTIONS = {
    "zigzag": {"name": "op.zigzag.int64_to_uint64.pure_v1", "input": "int64", "output": "uint64",
               "command": "zigzag", "namespace_local_id": 11, "legacy_id": 11,
               "legacy_semantics_preserved_separately": True},
    "compact_uint64": {"name": "op.pack.uint64_uniform_width.pure_v1", "input": "uint64", "output": "bytes",
                       "command": "compact-u64", "packet_header_bytes": 13},
    "zlib6": {"name": "op.compress.zlib_level6.pure_v1", "input": "bytes", "output": "bytes", "command": "zlib6"},
}


@dataclass(frozen=True)
class TypedData:
    kind: Kind
    payload: bytes
    count: int

    def __post_init__(self):
        if self.kind not in ("int64", "uint64", "bytes"):
            raise ValueError(f"unsupported kind {self.kind!r}")
        if not isinstance(self.count, int) or self.count < 0:
            raise ValueError("count must be a non-negative integer")
        object.__setattr__(self, "payload", bytes(self.payload))
        if len(self.payload) != self.count * (1 if self.kind == "bytes" else 8):
            raise ValueError("typed payload length/count mismatch")

    @classmethod
    def ints(cls, values: Sequence[int]):
        values = tuple(values)
        return cls("int64", struct.pack(f"<{len(values)}q", *values), len(values))

    @classmethod
    def uints(cls, values: Sequence[int]):
        values = tuple(values)
        return cls("uint64", struct.pack(f"<{len(values)}Q", *values), len(values))

    def values(self):
        if self.kind == "bytes":
            raise ValueError("bytes has no typed numeric values")
        return struct.unpack(f"<{self.count}{'q' if self.kind == 'int64' else 'Q'}", self.payload)


def build_cli() -> dict:
    """Explicit build into its own directory; old CMake/CLI files are untouched."""
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required")
    CLI.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", str(SOURCE), "-lz", "-o", str(CLI)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = {"registry_version": REGISTRY_VERSION, "source": str(SOURCE),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "binary": str(CLI), "binary_sha256": hashlib.sha256(CLI.read_bytes()).hexdigest(),
              "command": command}
    (CLI.parent / "build_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _run(command: str, payload: bytes) -> bytes:
    if not CLI.is_file():
        raise RuntimeError("build the NEW runtime explicitly: python3 scripts/pure_numeric_operator_runtime_v1.py build")
    result = subprocess.run([str(CLI), command], input=payload, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def run_action(action: str, source: TypedData, *, reverse: bool = False) -> TypedData:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    spec = ACTIONS[action]
    expected = spec["output"] if reverse else spec["input"]
    target = spec["input"] if reverse else spec["output"]
    if source.kind != expected:
        raise TypeError(f"{action} {'inverse' if reverse else 'forward'} requires {expected}, got {source.kind}")
    payload = _run(spec["command"] + ("-inverse" if reverse else "-forward"), source.payload)
    return TypedData(target, payload, len(payload) // (1 if target == "bytes" else 8))


def zigzag_forward(source: TypedData) -> TypedData:
    return run_action("zigzag", source)


def zigzag_inverse(source: TypedData) -> TypedData:
    return run_action("zigzag", source, reverse=True)


def run_pipeline(source: TypedData, actions: Sequence[str], *, reverse: bool = False):
    """Only the explicitly listed actions are executed; no implicit terminal."""
    current = source
    stages = []
    for action in reversed(actions) if reverse else actions:
        before = current
        current = run_action(action, before, reverse=reverse)
        stages.append({"action": action, "reverse": reverse, "input_kind": before.kind,
                       "output_kind": current.kind, "input_bytes": len(before.payload),
                       "output_bytes": len(current.payload)})
    return current, stages


def frame_wrap(source: TypedData, result: TypedData, actions: Sequence[str]) -> bytes:
    """Explicit outer storage frame, NOT part of ZigZag or compression."""
    meta = {"registry_version": REGISTRY_VERSION, "source_kind": source.kind, "source_count": source.count,
            "result_kind": result.kind, "result_count": result.count, "actions": list(actions),
            "source_sha256": hashlib.sha256(source.payload).hexdigest()}
    encoded = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode()
    return FRAME_MAGIC + struct.pack("<I", len(encoded)) + encoded + result.payload


def frame_unwrap(frame: bytes):
    if len(frame) < 8 or frame[:4] != FRAME_MAGIC:
        raise ValueError("invalid pure-numeric frame")
    size, = struct.unpack_from("<I", frame, 4)
    if size > len(frame) - 8:
        raise ValueError("truncated pure-numeric frame metadata")
    meta = json.loads(frame[8:8 + size])
    if meta.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("pure-numeric registry version mismatch")
    actions = meta.get("actions")
    if not isinstance(actions, list) or not all(isinstance(a, str) and a in ACTIONS for a in actions):
        raise ValueError("invalid frame actions")
    return meta, TypedData(meta["result_kind"], frame[8 + size:], meta["result_count"])


def decode_frame(frame: bytes) -> TypedData:
    meta, result = frame_unwrap(frame)
    restored, _ = run_pipeline(result, meta["actions"], reverse=True)
    if (restored.kind != meta["source_kind"] or restored.count != meta["source_count"] or
            hashlib.sha256(restored.payload).hexdigest() != meta["source_sha256"]):
        raise ValueError("frame source kind/count/checksum mismatch")
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("registry")
    transform = sub.add_parser("transform", help="single pure action; defaults to ZigZag, never adds a backend")
    transform.add_argument("--action", choices=ACTIONS, default="zigzag")
    transform.add_argument("--reverse", action="store_true")
    pipeline = sub.add_parser("pipeline", help="execute only the explicitly named actions")
    pipeline.add_argument("--actions", required=True, help="e.g. zigzag,compact_uint64,zlib6")
    pipeline.add_argument("--input-kind", choices=("int64", "uint64", "bytes"), required=True)
    pipeline.add_argument("--framed", action="store_true", help="explicitly add a separate PNV1 storage frame")
    decode = sub.add_parser("decode-frame")
    for child in (transform, pipeline, decode):
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_cli(), indent=2)); return
    if args.command == "registry":
        print(json.dumps({"registry_version": REGISTRY_VERSION, "independent_of_legacy_canonical206": True,
                          "actions": ACTIONS, "frame": {"magic": FRAME_MAGIC.decode(), "implicit": False}}, indent=2)); return
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; choose a new path or explicitly use --overwrite")
    raw = args.input.read_bytes()
    if args.command == "decode-frame":
        output = decode_frame(raw).payload
    else:
        if args.command == "transform":
            spec = ACTIONS[args.action]
            kind = spec["output"] if args.reverse else spec["input"]
            source = TypedData(kind, raw, len(raw) // (1 if kind == "bytes" else 8))
            output = run_action(args.action, source, reverse=args.reverse).payload
        else:
            source = TypedData(args.input_kind, raw, len(raw) // (1 if args.input_kind == "bytes" else 8))
            actions = args.actions.split(",")
            result, stages = run_pipeline(source, actions)
            output = frame_wrap(source, result, actions) if args.framed else result.payload
            print(json.dumps({"stages": stages, "frame_added": args.framed, "output_bytes": len(output)}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)


if __name__ == "__main__":
    main()
