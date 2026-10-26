"""Versioned multi-stream execution; legacy UAR1 remains a frozen codec boundary.

This is an execution API, not a trained graph policy. A frame stores live leaves
and reconstruction metadata, never a cached copy of the original payload.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from dataclasses import dataclass
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"serial": 1, "struct": 2, "numeric": 3}
OLS_HEADER = struct.Struct("<4sBBIQII")
OLS_STREAM = struct.Struct("<BIQ")
FRAME_MAGIC = b"TGF1"
GRAPH_REGISTRY_VERSIONS = ("operator-taxonomy-30-v1", "operator-taxonomy-30-v2", "operator-taxonomy-30-v3", "operator-taxonomy-30-v4", "operator-taxonomy-30-v5", "operator-taxonomy-30-v6", "operator-taxonomy-30-v7")
CURRENT_GRAPH_REGISTRY = GRAPH_REGISTRY_VERSIONS[-1]
ADJACENT_XOR_PORT_ACTION = "op.xor.previous_word.semantic_ports.cpp_v1"


@dataclass(frozen=True)
class Stream:
    kind: str
    width: int
    data: bytes

    def __post_init__(self):
        if self.kind not in KINDS or type(self.width) is not int or self.width < 1:
            raise ValueError("invalid stream type/width")
        if self.kind == "serial" and self.width != 1:
            raise ValueError("Serial requires width 1")
        if self.kind == "numeric" and self.width not in (1, 2, 4, 8):
            raise ValueError("Numeric requires width 1/2/4/8")
        if self.kind == "struct" and self.width > 65536:
            raise ValueError("Struct width exceeds native layout limit")
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("stream data must be bytes-like (not an integer size)")
        object.__setattr__(self, "data", bytes(self.data))
        if len(self.data) % self.width:
            raise ValueError("partial stream element")

    def spec(self):
        return {"kind": self.kind, "width": self.width, "size": len(self.data)}


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _unb64(data):
    return base64.b64decode(data, validate=True)


def pack_bundle(streams, *, op=0, input_kind="serial", input_width=1,
                input_total_bytes=None, side=b""):
    streams = tuple(streams)
    total = sum(len(s.data) for s in streams) if input_total_bytes is None else input_total_bytes
    return OLS_HEADER.pack(b"OLS1", op, KINDS[input_kind], input_width, total,
                           len(side), len(streams)) + side + b"".join(
        OLS_STREAM.pack(KINDS[s.kind], s.width, len(s.data)) + s.data for s in streams)


def unpack_bundle(data):
    data = bytes(data)
    if len(data) < OLS_HEADER.size:
        raise ValueError("truncated OLS1 header")
    magic, op, kind, width, total, side_len, n = OLS_HEADER.unpack_from(data)
    reverse_kinds = {v: k for k, v in KINDS.items()}
    if magic != b"OLS1" or op > 8 or kind not in reverse_kinds:
        raise ValueError("invalid OLS1 header")
    Stream(reverse_kinds[kind], width, b"")
    pos = OLS_HEADER.size
    if side_len > len(data) - pos:
        raise ValueError("truncated OLS1 metadata")
    side = data[pos:pos + side_len]
    pos += side_len
    streams = []
    if n > (len(data) - pos) // OLS_STREAM.size:
        raise ValueError("invalid OLS1 stream count")
    for _ in range(n):
        if pos + OLS_STREAM.size > len(data):
            raise ValueError("truncated OLS1 stream")
        sk, sw, size = OLS_STREAM.unpack_from(data, pos)
        pos += OLS_STREAM.size
        if sk not in reverse_kinds or size > len(data) - pos:
            raise ValueError("invalid OLS1 stream")
        streams.append(Stream(reverse_kinds[sk], sw, data[pos:pos + size]))
        pos += size
    if pos != len(data):
        raise ValueError("trailing OLS1 bytes")
    return streams, dict(op=op, input_kind=reverse_kinds[kind], input_width=width,
                         input_total_bytes=total, side=side)


def _native(command, data, *args):
    binary = Path(os.environ.get("OPENZL_LAYOUT_STAGES_CLI",
                                 ROOT / "cpp/build/openzl_layout_stages_cli"))
    if not binary.is_file():
        raise RuntimeError(f"build openzl_layout_stages_cli first: {binary}")
    result = subprocess.run([str(binary), command, *map(str, args)], input=data,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _csv(values):
    if not isinstance(values, (list, tuple)) or not values or any(type(v) is not int for v in values):
        raise ValueError("expected a nonempty integer list")
    return ",".join(map(str, values))


def _layout_args(action, streams, p):
    if action == "concat":
        if p:
            raise ValueError("Concat has no parameters")
        return ("concat",), pack_bundle(streams)
    if len(streams) != 1:
        raise ValueError("this action requires one input stream")
    s = streams[0]
    modes = {
        "columnize": {"mode", "width", "fields"},
        "float_deconstruct": {"format"},
        "bitsplit": {"mode", "width", "bits"},
        "bitunpack": {"bits"},
        "split_segments": {"mode", "sizes", "min_segment"},
    }
    if action not in modes or set(p) - modes[action]:
        raise ValueError("unknown layout action or parameter")
    if action == "columnize":
        allowed = {"mode", "width"} if p.get("mode", "transpose") == "transpose" else {"mode", "fields"}
    elif action == "bitsplit":
        allowed = {"mode", "width", "bits"} if p.get("mode", "generic") == "generic" else {"mode", "width"}
    elif action == "split_segments":
        allowed = {"mode", "sizes"} if p.get("mode", "continuous") == "continuous" else {"mode", "min_segment"}
    else:
        allowed = modes[action]
    if set(p) - allowed:
        raise ValueError("parameters include unused fields for the selected mode")
    width = p.get("width", s.width)
    if type(width) is not int or width < 1:
        raise ValueError("invalid element width")
    if s.kind != "serial" and width != s.width:
        raise ValueError("width parameter conflicts with typed stream")
    if action == "columnize":
        mode = p.get("mode", "transpose")
        if mode == "transpose":
            args = ("transpose", width)
        elif mode == "split-struct":
            fields = p["fields"]
            if s.kind != "serial" and sum(fields) != s.width:
                raise ValueError("fields do not match record width")
            args = ("split-struct", _csv(fields))
        else:
            raise ValueError("unknown columnize mode")
    elif action == "float_deconstruct":
        fmt = p.get("format", "fp32")
        if fmt not in ("fp16", "bf16", "fp32"):
            raise ValueError("unsupported FloatDeconstruct format")
        if s.kind != "serial" and s.width != (4 if fmt == "fp32" else 2):
            raise ValueError("floating-point bit width mismatch")
        args = ("float-deconstruct", fmt)
    elif action == "bitsplit":
        mode = p.get("mode", "generic")
        if mode == "generic":
            args = ("bitsplit", width, _csv(p["bits"]))
        elif mode in ("top8", "fp"):
            args = ("bitsplit-" + mode, width)
        elif mode == "bf16" and width == 2:
            args = ("bitsplit-bf16",)
        else:
            raise ValueError("unknown BitSplit mode or width")
    elif action == "bitunpack":
        if s.kind != "serial":
            raise ValueError("Bitunpack consumes a packed Serial bitstream")
        args = ("bitunpack", p["bits"])
    else:
        mode = p.get("mode", "continuous")
        if mode == "continuous":
            args = ("split-segments", s.kind, s.width, _csv(p["sizes"]))
        elif mode == "range":
            if s.kind != "numeric":
                raise ValueError("SplitByRange requires Numeric")
            args = ("split-by-range", s.width)
            if "min_segment" in p:
                args += (p["min_segment"],)
        else:
            raise ValueError("unknown segmentation mode")
    return args, s.data


@dataclass(frozen=True)
class GraphState:
    streams: dict[str, Stream]
    roots: dict[str, dict[str, Any]]
    steps: tuple[dict[str, Any], ...] = ()
    next_id: int = 0
    registry_version: str = CURRENT_GRAPH_REGISTRY

    def __post_init__(self):
        if self.registry_version not in GRAPH_REGISTRY_VERSIONS:
            raise ValueError("unknown graph action registry")

    @classmethod
    def from_streams(cls, streams, *, registry_version=CURRENT_GRAPH_REGISTRY):
        entries = {f"s{i}": s for i, s in enumerate(streams)}
        if not entries:
            raise ValueError("a graph needs at least one input stream (which may be empty)")
        return cls(entries, {k: v.spec() for k, v in entries.items()},
                   next_id=len(entries), registry_version=registry_version)

    def apply(self, action, input_stream_ids, *, parameters=None, input_view="bytes",
              io_variant=None):
        """Execute one graph edge. IDs choose leaves, not a rebuilt whole packet."""
        if type(action) is int:
            import unified_action_runtime as uar
            return uar.run_graph_action(self, action, input_stream_ids,
                registry_version=self.registry_version, parameters=parameters,
                input_view=input_view, io_variant=io_variant)
        ids = list(input_stream_ids)
        if not ids or len(set(ids)) != len(ids) or any(k not in self.streams for k in ids):
            raise ValueError("inputs must be distinct live stream IDs")
        selected = [self.streams[k] for k in ids]
        p = dict(parameters or {})
        layouts = {"columnize", "float_deconstruct", "bitsplit", "bitunpack",
                   "concat", "split_segments"}
        record = {"action": action, "inputs": ids,
                  "input_specs": [s.spec() for s in selected]}
        import exception_separation
        import nncp_entropy_codec
        import prediction_entropy_codec
        import residual_window_pack
        if action == ADJACENT_XOR_PORT_ACTION:
            if (len(selected) != 1 or selected[0].kind != "numeric"
                    or selected[0].width != 8 or input_view not in {"int64", "float64"}
                    or p):
                raise ValueError("adjacent-XOR ports require one 64-bit numeric data flow")
            import unified_action_runtime as uar
            from action_semantic_ports import to_ports
            descriptor = uar._graph_taxonomy(self.registry_version).describe_action(
                "op.xor.gorilla.cpp_v1"
            )
            result = uar.run_policy_action(
                descriptor.legacy_policy_action_id,
                uar.ActionData(input_view, selected[0].data, len(selected[0].data) // 8),
                registry_version=getattr(
                    uar._graph_taxonomy(self.registry_version),
                    "POLICY_REGISTRY_VERSION",
                    uar._graph_taxonomy(self.registry_version).LEGACY_REGISTRY_VERSION,
                ),
                io_variant="op.xor.gorilla.cpp_v1",
            )
            ports = to_ports(result)
            if [port.name for port in ports.streams] != ["first", "xor_residuals"]:
                raise ValueError("adjacent-XOR C++ stage did not expose semantic residual ports")
            outputs = [Stream("numeric", 8, port.payload) for port in ports.streams]
            record.update(
                kind="adjacent_xor_ports", input_view=input_view,
                port_specs=[{"name": port.name, "kind": port.kind, "count": port.count}
                            for port in ports.streams],
                port_metadata=ports.metadata, port_header=ports.header,
            )
        elif action == nncp_entropy_codec.ACTION:
            if len(selected) != 1:
                raise ValueError("NNP1 entropy coding requires one selected data flow")
            if p:
                raise ValueError("the complete A110 NNP1 path has no graph parameters")
            encoded = nncp_entropy_codec.forward(selected[0].data, action)
            outputs = [Stream("serial", 1, encoded)]
            record.update(
                kind="nncp_entropy", provider=nncp_entropy_codec.PROVIDER,
                profile=nncp_entropy_codec.PROFILE,
            )
        elif action in prediction_entropy_codec.ACTION_TO_PROVIDER:
            if len(selected) != 1:
                raise ValueError("prediction entropy coding requires one selected data flow")
            allowed = {"ppmd_order", "ppmd_memory_mib"} if action.endswith(".ppm.provider_v1") else (
                {"paq_level"} if action.endswith(".paq.provider_v1") else (
                    {"allow_large_memory"} if action.endswith(".cmix.provider_v1") else set()))
            if set(p) - allowed:
                raise ValueError("parameters do not belong to the selected probability provider")
            provider = prediction_entropy_codec.ACTION_TO_PROVIDER[action]
            encoded = prediction_entropy_codec.forward(selected[0].data, action, **p)
            outputs = [Stream("serial", 1, encoded)]
            record.update(kind="prediction_entropy", provider=provider)
        elif action == exception_separation.ALGORITHM_NAME:
            if len(selected) != 1 or selected[0].kind != "numeric" or selected[0].width != 4:
                raise ValueError("A111 requires one uint32 Numeric data flow")
            if set(p) != {"width"}:
                raise ValueError("A111 requires exactly the supplied low-bit width")
            packet = exception_separation.forward(selected[0].data, p["width"])
            ports = exception_separation.to_ports(packet)
            outputs = [Stream("numeric", 4, ports.low),
                       Stream("numeric", 4, ports.positions),
                       Stream("numeric", 4, ports.high)]
            record.update(kind="exception_separation", parameters={"width": ports.width},
                          count=ports.count,
                          port_names=["low", "positions", "high"])
        elif action in (residual_window_pack.ELF_PLUS_ACTION,
                         residual_window_pack.CHIMP_ACTION):
            if len(selected) != 1 or selected[0].kind != "numeric" or selected[0].width not in (4, 8):
                raise ValueError("residual window packing requires one 32/64-bit Numeric residual data flow")
            if p:
                raise ValueError("residual window profile is determined by action and input width")
            elf_plus = action == residual_window_pack.ELF_PLUS_ACTION
            encoded = residual_window_pack.forward(
                selected[0].data, word_bytes=selected[0].width, elf_plus=elf_plus)
            outputs = [Stream("serial", 1, encoded)]
            record.update(kind="residual_window_pack", word_bytes=selected[0].width,
                          elf_plus=elf_plus)
        elif action in layouts:
            args, payload = _layout_args(action, selected, p)
            outputs, meta = unpack_bundle(_native("encode", payload, *args))
            meta["side"] = _b64(meta["side"])
            # Only executed inverse metadata is stored. Caller parameters are
            # not copied into the frame as an unchecked, redundant side channel.
            record.update(kind="layout", metadata=meta)
        else:
            if len(selected) != 1:
                raise ValueError("legacy codec needs exactly one selected leaf")
            import unified_action_runtime as uar
            s = selected[0]
            if input_view == "bytes":
                data = uar.ActionData.bytes(s.data)
            elif input_view in ("int64", "float64") and s.width == 8:
                data = uar.ActionData(input_view, s.data, len(s.data) // 8)
            else:
                raise ValueError("input_view must be bytes or an explicit 64-bit bit view")
            taxonomy = uar._graph_taxonomy(self.registry_version)
            descriptor = taxonomy.describe_action(action)
            if isinstance(action, str) and action in descriptor.io_variants and io_variant is None:
                io_variant = action
            module = uar.action_parameters.module_for_name(action)
            if module is not None and parameters is None:
                historical = uar.action_parameters.parameters_from_name(action)
                if historical is not None: p = historical
            selector = descriptor.legacy_policy_action_id
            if selector is None: raise ValueError('codec has no versioned atomic selector')
            result = uar.run_policy_action(selector, data,
                registry_version=getattr(taxonomy, "POLICY_REGISTRY_VERSION", taxonomy.LEGACY_REGISTRY_VERSION),
                parameters=p or None, io_variant=io_variant)
            import minimum_for
            header, _ = uar._unpack_result(result)
            if header.get("io_variant") == minimum_for.PLAIN_VARIANT:
                if s.kind != "serial" and s.width != 8:
                    raise ValueError("minimum FOR requires explicit 64-bit input values")
                if self.registry_version not in ("operator-taxonomy-30-v2", "operator-taxonomy-30-v3", "operator-taxonomy-30-v4", "operator-taxonomy-30-v5", "operator-taxonomy-30-v6", "operator-taxonomy-30-v7"):
                    raise ValueError("minimum FOR leaf outputs require graph registry v2")
                from action_semantic_ports import to_ports
                ports = to_ports(result)
                if ([s.name for s in ports.streams] != ["minimum", "offsets"]
                        or ports.streams[0].kind not in ("int64", "uint64")
                        or ports.streams[0].count != 1
                        or ports.streams[1].kind != "uint64"):
                    raise ValueError("invalid minimum FOR output ports")
                outputs = [Stream("numeric", 8, port.payload) for port in ports.streams]
                record.update(action=minimum_for.ALGORITHM_NAME,
                    kind="minimum_for", input_view=input_view,
                    port_specs=[{"name": s.name, "kind": s.kind, "count": s.count}
                                for s in ports.streams],
                    port_metadata=ports.metadata, port_header=ports.header)
            elif header.get("io_variant") == uar.primitive.BUFF_PLANE_ACTION.name:
                from action_semantic_ports import to_ports
                ports = to_ports(result)
                outputs = [Stream("numeric", 8 if p.kind == "int64" else 1, p.payload) for p in ports.streams]
                record.update(kind="buff_byte_planes", input_view=input_view,
                    port_specs=[{"name":p.name,"kind":p.kind,"count":p.count} for p in ports.streams],
                    port_metadata=ports.metadata, port_header=ports.header)
            else:
                outputs = [Stream("serial", 1, result.payload)]
                record.update(kind="legacy", input_view=input_view)
        new_ids = [f"s{self.next_id + i}" for i in range(len(outputs))]
        record.update(outputs=new_ids, output_specs=[s.spec() for s in outputs])
        live = {k: s for k, s in self.streams.items() if k not in ids}
        live.update(zip(new_ids, outputs))
        return GraphState(live, dict(self.roots), self.steps + (record,),
                          self.next_id + len(outputs), self.registry_version)

    def stop(self):
        """STOP is a special operation; its complete frame includes every leaf."""
        header = {"schema": "typed-graph-frame-v1", "roots": self.roots,
                  "root_order": list(self.roots),
                  "registry_version": self.registry_version,
                  "steps": self.steps,
                  "leaves": [{"id": k, **s.spec()} for k, s in self.streams.items()]}
        raw = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        content = FRAME_MAGIC + struct.pack("<Q", len(raw)) + raw + b"".join(
            s.data for s in self.streams.values())
        return content + hashlib.sha256(content).digest()


def _validate_specs(streams, specs):
    if len(streams) != len(specs) or any(s.spec() != spec for s, spec in zip(streams, specs)):
        raise ValueError("graph stream type/length mismatch")


def restore_frame(frame, *, allow_large_memory=False):
    """Cold inverse; consumes only this frame and pinned native implementations."""
    if len(frame) < 44 or frame[:4] != FRAME_MAGIC:
        raise ValueError("invalid graph frame")
    if hashlib.sha256(frame[:-32]).digest() != frame[-32:]:
        raise ValueError("graph checksum mismatch")
    frame = frame[:-32]
    n = struct.unpack_from("<Q", frame, 4)[0]
    if n > len(frame) - 12:
        raise ValueError("truncated graph metadata")
    def unique_object(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("duplicate graph metadata key")
            obj[k] = v
        return obj
    header = json.loads(frame[12:12 + n], object_pairs_hook=unique_object)
    if not isinstance(header, dict) or set(header) != {"schema", "roots", "root_order", "registry_version", "steps", "leaves"}:
        raise ValueError("invalid graph header fields")
    if header.get("schema") != "typed-graph-frame-v1":
        raise ValueError("unknown graph schema")
    if header.get("registry_version") not in GRAPH_REGISTRY_VERSIONS:
        raise ValueError("unknown graph action registry")
    live = {}
    pos = 12 + n
    for leaf in header["leaves"]:
        if not isinstance(leaf, dict) or set(leaf) != {"id", "kind", "width", "size"}:
            raise ValueError("invalid graph leaf fields")
        size, key = leaf["size"], leaf["id"]
        if type(size) is not int or size < 0 or size > len(frame) - pos or key in live:
            raise ValueError("invalid or duplicate leaf")
        live[key] = Stream(leaf["kind"], leaf["width"], frame[pos:pos + size])
        pos += size
    if pos != len(frame):
        raise ValueError("trailing graph bytes")
    for record in reversed(header["steps"]):
        common = {"action", "inputs", "input_specs", "kind", "outputs", "output_specs"}
        if record.get("kind") == "layout":
            fields = {"metadata"}
        elif record.get("kind") in {"minimum_for", "buff_byte_planes", "adjacent_xor_ports"}:
            fields = {"input_view", "port_specs", "port_metadata", "port_header"}
        elif record.get("kind") == "exception_separation":
            fields = {"parameters", "count", "port_names"}
        elif record.get("kind") == "residual_window_pack":
            fields = {"word_bytes", "elf_plus"}
        elif record.get("kind") == "prediction_entropy":
            fields = {"provider"}
        elif record.get("kind") == "nncp_entropy":
            fields = {"provider", "profile"}
        else:
            fields = {"input_view"}
        if set(record) != common | fields:
            raise ValueError("invalid graph edge fields")
        inputs, outputs = record["inputs"], record["outputs"]
        if (len(set(outputs)) != len(outputs) or len(set(inputs)) != len(inputs)
                or any(k not in live for k in outputs) or any(k in live for k in inputs)):
            raise ValueError("invalid graph dependency or duplicate edge")
        selected = [live[k] for k in outputs]
        _validate_specs(selected, record["output_specs"])
        if record["kind"] == "layout":
            meta = dict(record["metadata"])
            valid_ops = {"columnize": {1, 2}, "float_deconstruct": {3}, "bitsplit": {4},
                         "bitunpack": {5}, "concat": {6}, "split_segments": {7, 8}}
            if meta["op"] not in valid_ops.get(record["action"], set()):
                raise ValueError("graph layout name/op mismatch")
            meta["side"] = _unb64(meta["side"])
            restored, _ = unpack_bundle(_native("decode-bundle", pack_bundle(selected, **meta)))
            # Native CLI receives raw bytes for unary actions. Restore the
            # original port's declared type without a numeric conversion.
            specs = record["input_specs"]
            if len(restored) != len(specs):
                raise ValueError("incorrect inverse arity")
            restored = [Stream(spec["kind"], spec["width"], s.data)
                        for s, spec in zip(restored, specs)]
        elif record["kind"] == "adjacent_xor_ports":
            import unified_action_runtime as uar
            from action_semantic_ports import ActionPorts, TypedStream, from_ports
            specs = record["port_specs"]
            if (record["action"] != ADJACENT_XOR_PORT_ACTION
                    or record["input_view"] not in {"int64", "float64"}
                    or len(selected) != 2 or len(record["input_specs"]) != 1
                    or [item.get("name") for item in specs] != ["first", "xor_residuals"]
                    or any(stream.kind != "numeric" or stream.width != 8 for stream in selected)):
                raise ValueError("invalid adjacent-XOR semantic-port edge")
            ports = ActionPorts(
                tuple(TypedStream(item["name"], item["kind"], stream.data, item["count"])
                      for item, stream in zip(specs, selected)),
                record["port_metadata"], record["port_header"],
            )
            packet = from_ports(ports)
            descriptor = uar._graph_taxonomy(header["registry_version"]).describe_action(
                "op.xor.gorilla.cpp_v1"
            )
            result = uar.run_policy_action(
                descriptor.legacy_policy_action_id, packet,
                registry_version=getattr(
                    uar._graph_taxonomy(header["registry_version"]),
                    "POLICY_REGISTRY_VERSION",
                    uar._graph_taxonomy(header["registry_version"]).LEGACY_REGISTRY_VERSION,
                ), reverse=True,
            )
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], result.payload)]
        elif record["kind"] == "minimum_for":
            import minimum_for
            import unified_action_runtime as uar
            from action_semantic_ports import ActionPorts, TypedStream
            if header["registry_version"] not in ("operator-taxonomy-30-v2", "operator-taxonomy-30-v3", "operator-taxonomy-30-v4", "operator-taxonomy-30-v5", "operator-taxonomy-30-v6", "operator-taxonomy-30-v7"):
                raise ValueError("minimum FOR leaf outputs require graph registry v2")
            specs = record["port_specs"]
            if (record["action"] not in (minimum_for.ALGORITHM_NAME, minimum_for.PLAIN_VARIANT)
                    or len(selected) != 2 or len(record["input_specs"]) != 1
                    or not isinstance(specs, list) or len(specs) != 2
                    or any(not isinstance(s, dict) or set(s) != {"name", "kind", "count"} for s in specs)
                    or specs[0]["name"] != "minimum" or specs[0]["count"] != 1
                    or specs[0]["kind"] not in ("int64", "uint64")
                    or specs[1]["name"] != "offsets" or specs[1]["kind"] != "uint64"
                    or any(s.kind != "numeric" or s.width != 8 for s in selected)
                    or record["input_view"] not in ("bytes", "int64")
                    or not isinstance(record["port_header"], dict)
                    or record["port_header"].get("io_variant") != minimum_for.PLAIN_VARIANT):
                raise ValueError("invalid minimum FOR edge or ports")
            ports = ActionPorts(tuple(TypedStream(p["name"], p["kind"], s.data, p["count"])
                                      for p, s in zip(specs, selected)),
                                record["port_metadata"], record["port_header"])
            result = uar.run_policy_action(minimum_for.ALGORITHM_NAME, ports,
                registry_version=uar.primitive.ACTION_REGISTRY_VERSION, reverse=True)
            expected_kind = "int64" if record["input_view"] == "int64" else "bytes"
            if result.kind != expected_kind:
                raise ValueError("minimum FOR source type mismatch")
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], result.payload)]
        elif record["kind"] == "buff_byte_planes":
            import unified_action_runtime as uar
            from action_semantic_ports import ActionPorts, TypedStream
            specs=record.get("port_specs")
            if (header["registry_version"] != "operator-taxonomy-30-v7"
                    or record["action"] != uar.primitive.BUFF_PLANE_ACTION.name
                    or len(record["input_specs"]) != 1 or not isinstance(specs,list)
                    or len(specs) != len(selected) or not specs
                    or specs[0] != {"name":"minimum","kind":"int64","count":1}
                    or any(p["kind"] != "uint8" or p["name"] != f"byte_plane{i}" for i,p in enumerate(specs[1:]))):
                raise ValueError("invalid BUFF plane edge")
            ports=ActionPorts(tuple(TypedStream(p["name"],p["kind"],x.data,p["count"]) for p,x in zip(specs,selected)),record["port_metadata"],record["port_header"])
            result=uar.run_policy_action(58,ports,registry_version="cpp-canonical-127-v4",reverse=True)
            spec=record["input_specs"][0]
            restored=[Stream(spec["kind"],spec["width"],result.payload)]
        elif record["kind"] == "exception_separation":
            import exception_separation
            if (record["action"] != exception_separation.ALGORITHM_NAME
                    or len(selected) != 3 or len(record["input_specs"]) != 1
                    or record["port_names"] != ["low", "positions", "high"]
                    or any(s.kind != "numeric" or s.width != 4 for s in selected)
                    or set(record["parameters"]) != {"width"}):
                raise ValueError("invalid A111 graph edge")
            packet = exception_separation.from_ports(exception_separation.Ports(
                record["parameters"]["width"], record["count"],
                selected[0].data, selected[1].data, selected[2].data))
            raw = exception_separation.inverse(packet)
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], raw)]
        elif record["kind"] == "residual_window_pack":
            import residual_window_pack
            expected = (residual_window_pack.ELF_PLUS_ACTION if record["elf_plus"]
                        else residual_window_pack.CHIMP_ACTION)
            if (record["action"] != expected or len(selected) != 1
                    or selected[0].kind != "serial" or len(record["input_specs"]) != 1
                    or record["word_bytes"] not in (4, 8)):
                raise ValueError("invalid residual-window graph edge")
            raw = residual_window_pack.inverse(
                selected[0].data, word_bytes=record["word_bytes"],
                elf_plus=record["elf_plus"])
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], raw)]
        elif record["kind"] == "prediction_entropy":
            import prediction_entropy_codec
            if (record["action"] not in prediction_entropy_codec.ACTION_TO_PROVIDER
                    or record["provider"] != prediction_entropy_codec.ACTION_TO_PROVIDER[record["action"]]
                    or len(selected) != 1 or selected[0].kind != "serial"
                    or len(record["input_specs"]) != 1):
                raise ValueError("invalid prediction entropy graph edge")
            raw = prediction_entropy_codec.inverse(
                selected[0].data, expected_action_or_provider=record["action"],
                allow_large_memory=allow_large_memory)
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], raw)]
        elif record["kind"] == "nncp_entropy":
            import nncp_entropy_codec
            if (record["action"] != nncp_entropy_codec.ACTION
                    or record["provider"] != nncp_entropy_codec.PROVIDER
                    or record["profile"] != nncp_entropy_codec.PROFILE
                    or len(selected) != 1 or selected[0].kind != "serial"
                    or len(record["input_specs"]) != 1):
                raise ValueError("invalid A110 NNP1 graph edge")
            raw = nncp_entropy_codec.inverse(
                selected[0].data, expected_action=record["action"]
            )
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], raw)]
        elif record["kind"] == "legacy":
            import unified_action_runtime as uar
            if len(selected) != 1 or len(record["input_specs"]) != 1:
                raise ValueError("invalid legacy codec arity")
            if record["input_view"] not in ("bytes", "int64", "float64"):
                raise ValueError("invalid legacy input view")
            action = record["action"]
            taxonomy = uar._graph_taxonomy(header['registry_version'])
            descriptor = taxonomy.describe_action(action)
            if descriptor.legacy_policy_action_id is None:
                raise ValueError('codec has no versioned atomic selector')
            result = uar.run_policy_action(descriptor.legacy_policy_action_id, uar.ActionData.bytes(selected[0].data),
                registry_version=getattr(taxonomy, "POLICY_REGISTRY_VERSION", taxonomy.LEGACY_REGISTRY_VERSION), reverse=True)
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], result.payload)]
        else:
            raise ValueError("unknown graph edge kind")
        _validate_specs(restored, record["input_specs"])
        if len(restored) != len(inputs):
            raise ValueError("invalid inverse input count")
        for k in outputs:
            del live[k]
        live.update(zip(inputs, restored))
    roots = header["roots"]
    order = header["root_order"]
    if set(live) != set(roots) or len(order) != len(set(order)) or set(order) != set(roots):
        raise ValueError("unresolved graph dependency")
    _validate_specs([live[k] for k in roots], list(roots.values()))
    return {k: live[k] for k in order}
