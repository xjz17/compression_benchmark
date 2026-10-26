"""Versioned reversible graph carrier for exhaustive native-operator experiments.

Python performs graph scheduling and serialization; codec/layout kernels use
the existing C++ adapters. This is not a claim of an all-C++ timing boundary.
The frame contains every live payload and all reconstruction metadata. Ordinary
operators increment root-to-leaf depth; Concat does not. No implicit merging is
performed. Probability-only providers remain explicit unsupported inventory.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from typing import Any

import action_semantic_ports as ports_api
import canonical127_registry as canonical
import current_action_parameters as action_parameters
import operator_taxonomy_v30_v4 as taxonomy
import typed_graph_runtime as layout
import unified_action_runtime as unified

Stream = layout.Stream
SourceStream = Stream
REGISTRY_VERSION = taxonomy.REGISTRY_VERSION
RUNTIME_VERSION = "exhaustive-graph-runtime-v1.1-native-reference"
FRAME_SCHEMA = "exhaustive-graph-frame-v1"
FRAME_MAGIC = b"EGF1"
_WIDTHS = {"int64": 8, "uint64": 8, "uint32": 4, "uint8": 1,
           "float64": 8, "bytes": 1}
_REFERENCE_PORT_VARIANTS = {
    name for name, spec in ports_api._VARIANT_LAYOUTS.items()
    if "reference.opa1" in spec.get("schemas", ())
}
_LAYOUTS = {"columnize", "float_deconstruct", "bitsplit", "bitunpack",
            "concat", "split_segments"}
_CARRIER_DEFAULT = Path(__file__).resolve().parents[1] / "cpp/build/exhaustive_graph_carrier_cli"


class GraphInputError(ValueError):
    """This action is outside its declared input or depth domain."""


class UnsupportedActionError(RuntimeError):
    """An inventory item is not connected; this is not an invalid input."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _b64(value):
    return base64.b64encode(value).decode("ascii")


def _unb64(value):
    return base64.b64decode(value, validate=True)


@lru_cache(maxsize=1)
def registry_fingerprint():
    return hashlib.sha256(_json({
        "graph_registry": REGISTRY_VERSION,
        "canonical_manifest": canonical.MANIFEST,
        "parameters": action_parameters.schemas(),
        "runtime": RUNTIME_VERSION,
    })).hexdigest()


def registry_inventory():
    records = []
    for descriptor in taxonomy.OPERATOR_ACTIONS:
        row = asdict(descriptor)
        if descriptor.kind == "model_provider":
            row.update(executable=False, selectable=False,
                       status="blocked_graph_entropy_not_connected")
        else:
            row.update(executable=None, selectable=None,
                       status="registered_requires_current_binary_and_input_validation")
        records.append(row)
    return {
        "runtime_version": RUNTIME_VERSION,
        "registry_version": REGISTRY_VERSION,
        "registry_sha256": registry_fingerprint(),
        "operator_count": len(records), "codec_count": len(taxonomy.CODEC_ACTIONS),
        "provider_count": len(taxonomy.MODEL_PROVIDERS),
        "operators": records,
        "special_operations": [asdict(d) for d in taxonomy.SPECIAL_ACTIONS],
        "search_scope_complete": False,
        "scope_gaps": [
            "Five probability providers have no graph entropy-coding edge.",
            "Unrecognized opaque outputs remain byte streams with explicit semantic-exposure gaps.",
        ],
        "depth_definition": "maximum ordinary-operator count along each root-to-leaf path; Concat adds zero",
        "continuous_split_search_scope": "not enumerated: request covers operator-produced streams and explicit Concat",
        "timing_boundary": "Python graph orchestration, framing, C++ child process startup/IPC and native execution; excludes input file reads",
        "reference_carrier": {"implementation": "cpp/src/exhaustive_graph_carrier_cli.cpp",
            "dependency": "native C zlib linked by C++ CLI; level 6; no Python zlib path",
            "executable": str(Path(os.environ.get("EXHAUSTIVE_GRAPH_CARRIER_CLI", _CARRIER_DEFAULT)))},
    }


def classify_exception(exc):
    if isinstance(exc, UnsupportedActionError):
        return "unsupported"
    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError)):
        return "runtime_missing"
    message = str(exc).lower()
    if any(s in message for s in ("build target", "build its", "missing c++", "build openzl",
                                 "no such file", "missing /", "build target", "not found")):
        return "runtime_missing"
    if isinstance(exc, (GraphInputError, unified.ActionTypeError,
                        unified.ActionSequenceError)):
        return "invalid_input"
    # Generic native ValueError may mean either a rejected schema or a broken
    # implementation. Conservatively do not use it to certify coverage.
    return "runtime_failure"


def _source_data(stream, view):
    if view == "bytes":
        return unified.ActionData.bytes(stream.data)
    if view in ("int64", "float64") and stream.width == 8:
        return unified.ActionData(view, stream.data, len(stream.data) // 8)
    raise GraphInputError("64-bit numeric input views require an explicit width-8 stream")


def _port_stream(port):
    return Stream("serial" if port.kind == "bytes" else "numeric",
                  _WIDTHS[port.kind], port.payload)


def _port_view(kind):
    return kind if kind in ("int64", "float64") else "bytes"


def _carrier(command, payload):
    binary = Path(os.environ.get("EXHAUSTIVE_GRAPH_CARRIER_CLI", _CARRIER_DEFAULT))
    if not binary.is_file():
        raise FileNotFoundError(f"missing C++ reference carrier {binary}")
    result = subprocess.run([str(binary), command], input=payload, capture_output=True)
    if result.returncode:
        raise ValueError("native reference carrier: " + result.stderr.decode(errors="replace").strip())
    return result.stdout


def _reference_to_ports(payload, header):
    """OPA1 schema parser, using C++ for zlib and compact-value decoding."""
    try:
        raw = _carrier("decode", payload)
    except ValueError:
        return None
    if not raw.startswith(b"OPA1"):
        return None
    check = ports_api._check
    view = ports_api._View(raw, "reference.opa1", header)
    check(view.literal(4) == b"OPA1", "wrong reference magic")
    check(view.saved_number("<B") == 1, "unsupported reference version")
    count = view.saved_number("<B")
    ids = list(view.literal(count))
    if not ids or ids[-1] not in ports_api._REFERENCE_KINDS:
        return None
    check(all(i < 29 for i in ids), "invalid reference action ID")
    view.meta["reference_action_ids"] = ids
    for n, action_id in enumerate(ids):
        tag = view.literal(12).rstrip(b"\0")
        if action_id in ports_api._REFERENCE_TAGS:
            check(tag == ports_api._REFERENCE_TAGS[action_id], "reference action/tag mismatch")
        arrays = view.saved_number("<H")
        extra_size = view.saved_number("<I")
        view.stream(f"record{n}.extra", "bytes", extra_size)
        for a in range(arrays):
            count = view.saved_number("<I")
            size = view.saved_number("<I")
            check(size == count * 8, "reference side-array size mismatch")
            names = {2: ["base"], 3: ["original_index"], 7: ["first"],
                     8: ["first"], 28: ["first", "references"]}.get(action_id, [])
            label = names[a] if a < len(names) else f"array{a}"
            view.stream(f"record{n}.{label}", "int64", count)
            if action_id == 28 and a == 1:
                view.meta["constraints"].append({"type": "range", "stream": f"record{n}.{label}", "min": 0, "max": 127})
    values = ports_api._compact("compact-decode", view.take(len(raw) - view.pos))
    view.meta["reference_metadata_segments"] = view.meta.pop("segments")
    main = ports_api._View(values, "reference.main", header)
    if ids[-1] == 14:
        check(len(values) % 16 == 0, "reference RLE record shape mismatch")
        main.interleaved([("counts", "uint64", 1), ("values", "int64", 1)], len(values) // 16)
        view.meta["constraints"].append({"type": "range", "stream": "counts", "min": 1, "max": 0xffffffff})
    else:
        main.stream("main", ports_api._REFERENCE_KINDS[ids[-1]], len(values) // 8)
    view.streams.extend(main.streams)
    view.meta["segments"] = main.meta["segments"]
    view.meta["encoding"] = "OPA1 metadata + C++ compact values + C++ native zlib level 6"
    view.meta["semantic_note"] = "All typed streams and side records remain independently compressible; uint64 bit patterns are preserved."
    return view.finish()


def _native_to_ports(result):
    header, payload = unified._unpack_result(result)
    variant = header["io_variant"]
    if not variant.endswith(".reference_v1"):
        return ports_api.to_ports(result)
    # Keep the old module untouched. Its reference branch uses Python zlib;
    # this wrapper performs that branch through the native carrier instead.
    if variant not in ports_api._OPAQUE_SEMANTICS:
        reference = _reference_to_ports(payload, header)
        if reference is not None:
            return reference
        parsed = (ports_api._container(payload, header, variant)
                  or ports_api._simple_packet(payload, header)
                  or ports_api._envelope(payload, header))
        if parsed is not None:
            return parsed
    view = ports_api._View(payload, "opaque_bytes", header)
    view.stream("encoded", "bytes", len(payload), variable=True)
    view.meta["semantic_note"] = ports_api._OPAQUE_SEMANTICS.get(variant,
        "Unknown output layout; complete bytes are retained without claiming semantic stream exposure.")
    return view.finish()


def _native_from_ports(ports):
    if ports.metadata.get("schema") != "reference.opa1":
        return ports_api.from_ports(ports)
    ports_api._validate(ports)
    by_name = {s.name: s for s in ports.streams}
    used = set()
    values = ports_api._rebuild_segments(ports.metadata["segments"], by_name, used)
    prefix = ports_api._rebuild_segments(ports.metadata["reference_metadata_segments"], by_name, used)
    ports_api._check(used == set(by_name), "unreferenced extra reference stream would be discarded")
    payload = _carrier("encode", prefix + ports_api._compact("compact-encode", values))
    return unified._pack_existing_result(ports.header, payload)


@dataclass(frozen=True)
class GraphState:
    streams: dict[str, Stream]
    roots: dict[str, dict[str, Any]]
    depths: dict[str, int]
    source_views: dict[str, str]
    steps: tuple[dict[str, Any], ...] = ()
    next_id: int = 0
    max_depth: int = 10
    max_point: int | None = None
    registry_version: str = REGISTRY_VERSION
    codec_count: int = 0
    exposure_gaps: tuple[str, ...] = ()

    @classmethod
    def from_streams(cls, streams, *, max_point=None, source_views=None,
                     max_depth=10, registry_version=REGISTRY_VERSION):
        values = tuple(streams)
        if not values or not all(isinstance(s, Stream) for s in values):
            raise GraphInputError("one or more typed input streams are required")
        if registry_version != REGISTRY_VERSION:
            raise GraphInputError("this independent runtime freezes graph registry v4")
        if type(max_depth) is not int or max_depth < 0:
            raise GraphInputError("max_depth must be a nonnegative integer")
        entries = {f"s{i}": s for i, s in enumerate(values)}
        if source_views is None:
            views = {k: "bytes" for k in entries}
        elif isinstance(source_views, dict):
            views = dict(source_views)
        else:
            views = dict(zip(entries, source_views))
        if set(views) != set(entries):
            raise GraphInputError("source_views must specify every input stream")
        for key, stream in entries.items():
            _source_data(stream, views[key])
        return cls(entries, {k: s.spec() for k, s in entries.items()},
                   {k: 0 for k in entries}, views, next_id=len(entries),
                   max_depth=max_depth, max_point=max_point,
                   registry_version=registry_version)

    def apply(self, action, input_stream_ids, *, parameters=None,
              input_view="bytes", io_variant=None, max_point=None,
              output_mode="ports"):
        ids = list(input_stream_ids)
        if not ids or len(ids) != len(set(ids)) or any(k not in self.streams for k in ids):
            raise GraphInputError("inputs must be distinct current live-stream IDs")
        descriptor = None
        if action not in _LAYOUTS:
            descriptor = taxonomy.describe_action(action)
            if descriptor.kind == "model_provider":
                raise UnsupportedActionError("probability provider has no graph entropy-coding edge")
            if descriptor.name == "stop":
                raise GraphInputError("STOP is state.stop(), with no selected streams")
            # Explicit frozen byte-shuffle I/O variants remain independently
            # replayable native wire actions; no silent change to transpose.
            legacy_shuffle = (descriptor.action_id == taxonomy.COLUMNIZE_ACTION_ID
                              and io_variant is not None)
            if (descriptor.operator_id == "split_columns" and not legacy_shuffle) or descriptor.kind == "special":
                action = descriptor.mechanism
        if output_mode not in ("ports", "wire"):
            raise GraphInputError("output_mode must be ports or wire")
        selected = [self.streams[k] for k in ids]
        special = action in ("concat", "split_segments")
        if action == "concat":
            if len(ids) < 2:
                raise GraphInputError("Concat requires at least two live streams")
            if len({(s.kind, s.width) for s in selected}) != 1:
                raise GraphInputError("Concat requires identical kind and width")
        elif len(ids) != 1:
            raise GraphInputError("ordinary operators consume one selected stream")
        depth = max(self.depths[k] for k in ids) + (0 if special else 1)
        if depth > self.max_depth:
            raise GraphInputError("root-to-leaf ordinary-operator depth exceeds max_depth")
        record = {"action": action if isinstance(action, str) else descriptor.name,
                  "inputs": ids, "input_specs": [s.spec() for s in selected],
                  "input_depths": [self.depths[k] for k in ids],
                  "input_views": [self.source_views[k] for k in ids],
                  "depth": depth, "ordinary_operator": not special}
        p = dict(parameters or {})
        gaps = []
        if action in _LAYOUTS:
            if io_variant is not None:
                raise GraphInputError("layout modes use parameters, not legacy io_variant")
            args, payload = layout._layout_args(action, selected, p)
            outputs, metadata = layout.unpack_bundle(layout._native("encode", payload, *args))
            metadata["side"] = _b64(metadata["side"])
            record.update(kind="layout", metadata=metadata, parameters=p)
            if action == "concat":
                # Native OLS1 exposes lengths as an auxiliary stream. Keep
                # those inverse boundaries in the paid frame metadata so
                # Concat produces exactly ONE active leaf and strictly
                # decreases live-stream count for every >=2-input merge.
                if len(outputs) != 2 or outputs[0].kind != "numeric" or outputs[0].width != 4:
                    raise UnsupportedActionError("unexpected native Concat boundary layout")
                record["concat_boundaries"] = {
                    **outputs[0].spec(), "data": _b64(outputs[0].data)}
                outputs = outputs[1:]
            views = ["bytes"] * len(outputs)
            if action in ("concat", "split_segments"):
                in_views = {self.source_views[k] for k in ids}
                if len(in_views) == 1:
                    views = [next(iter(in_views))] * len(outputs)
        else:
            descriptor = descriptor or taxonomy.describe_action(action)
            if descriptor.legacy_policy_action_id is None:
                raise UnsupportedActionError("operator has no native atomic selector")
            if io_variant is None and isinstance(action, str) and action in descriptor.io_variants:
                io_variant = action
            chosen_view = self.source_views[ids[0]] if input_view is None else input_view
            source = _source_data(selected[0], chosen_view)
            result = unified.run_policy_action(descriptor.legacy_policy_action_id, source,
                registry_version=canonical.REGISTRY_VERSION,
                parameters=p or None, io_variant=io_variant,
                max_point=self.max_point if max_point is None else max_point)
            header, _ = unified._unpack_result(result)
            variant = header["io_variant"]
            if output_mode == "wire":
                outputs = [Stream("serial", 1, result.payload)]
                views = ["bytes"]
                record.update(kind="wire", policy_action_id=descriptor.legacy_policy_action_id)
            else:
                ports = _native_to_ports(result)
                outputs = [_port_stream(port) for port in ports.streams]
                views = [_port_view(port.kind) for port in ports.streams]
                if not outputs:
                    raise UnsupportedActionError("semantic exposure produced no reconstructible port")
                record.update(kind="ports", policy_action_id=descriptor.legacy_policy_action_id,
                    port_specs=[{"name": p.name, "kind": p.kind, "count": p.count}
                                for p in ports.streams],
                    port_metadata=ports.metadata, port_header=ports.header)
                if (ports.metadata.get("schema") == "opaque_bytes"
                        and variant not in ports_api._OPAQUE_SEMANTICS):
                    gaps.append(f"opaque semantic output not declared as a terminal byte stream: {variant}")
            record.update(input_view=chosen_view, io_variant=variant,
                          parameters=header.get("action_parameters", {}),
                          max_point=self.max_point if max_point is None else max_point,
                          output_mode=output_mode, exposure_gaps=gaps)
        new_ids = [f"s{self.next_id + i}" for i in range(len(outputs))]
        record.update(outputs=new_ids, output_specs=[s.spec() for s in outputs],
                      output_views=views)
        live = {k: v for k, v in self.streams.items() if k not in ids}
        live.update(zip(new_ids, outputs))
        depths = {k: v for k, v in self.depths.items() if k not in ids}
        depths.update({k: depth for k in new_ids})
        live_views = {k: v for k, v in self.source_views.items() if k not in ids}
        live_views.update(zip(new_ids, views))
        return GraphState(live, dict(self.roots), depths, live_views,
            self.steps + (record,), self.next_id + len(outputs), self.max_depth,
            self.max_point, self.registry_version, self.codec_count + int(not special),
            tuple(dict.fromkeys((*self.exposure_gaps, *gaps))))

    def stop(self):
        header = {"schema": FRAME_SCHEMA, "runtime_version": RUNTIME_VERSION,
            "registry_version": self.registry_version, "registry_sha256": registry_fingerprint(),
            "roots": self.roots, "root_order": list(self.roots), "steps": self.steps,
            "max_depth": self.max_depth, "codec_count": self.codec_count,
            "exposure_gaps": self.exposure_gaps,
            "leaves": [{"id": k, **s.spec(), "depth": self.depths[k],
                        "view": self.source_views[k]} for k, s in self.streams.items()]}
        raw = _json(header)
        content = FRAME_MAGIC + struct.pack("<Q", len(raw)) + raw + b"".join(
            s.data for s in self.streams.values())
        return content + hashlib.sha256(content).digest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate frame metadata key")
        result[key] = value
    return result


def _validate_specs(streams, specs):
    if len(streams) != len(specs) or any(s.spec() != spec for s, spec in zip(streams, specs)):
        raise ValueError("frame stream type/length mismatch")


def restore_frame(frame):
    """Restore only from this complete frame and its frozen codec installation."""
    if len(frame) < 44 or frame[:4] != FRAME_MAGIC:
        raise ValueError("invalid exhaustive graph frame")
    if hashlib.sha256(frame[:-32]).digest() != frame[-32:]:
        raise ValueError("frame checksum mismatch")
    body = frame[:-32]
    size = struct.unpack_from("<Q", body, 4)[0]
    if size > len(body) - 12:
        raise ValueError("truncated frame metadata")
    header = json.loads(body[12:12 + size], object_pairs_hook=_unique)
    fields = {"schema", "runtime_version", "registry_version", "registry_sha256",
              "roots", "root_order", "steps", "max_depth", "codec_count", "exposure_gaps", "leaves"}
    if not isinstance(header, dict) or set(header) != fields:
        raise ValueError("invalid frame metadata fields")
    if (header["schema"] != FRAME_SCHEMA or header["runtime_version"] != RUNTIME_VERSION
            or header["registry_version"] != REGISTRY_VERSION
            or header["registry_sha256"] != registry_fingerprint()):
        raise ValueError("frame runtime/registry fingerprint mismatch")
    live, depths, views = {}, {}, {}
    pos = 12 + size
    for leaf in header["leaves"]:
        if set(leaf) != {"id", "kind", "width", "size", "depth", "view"}:
            raise ValueError("invalid frame leaf fields")
        n, key, depth = leaf["size"], leaf["id"], leaf["depth"]
        if (type(n) is not int or n < 0 or n > len(body) - pos or key in live
                or type(depth) is not int or not 0 <= depth <= header["max_depth"]):
            raise ValueError("invalid/duplicate frame leaf")
        live[key] = Stream(leaf["kind"], leaf["width"], body[pos:pos + n])
        depths[key], views[key] = depth, leaf["view"]
        pos += n
    if pos != len(body):
        raise ValueError("trailing frame bytes")
    counted = 0
    for record in reversed(header["steps"]):
        inputs, outputs = record["inputs"], record["outputs"]
        if (not inputs or not outputs or len(set(inputs)) != len(inputs)
                or len(set(outputs)) != len(outputs) or any(k not in live for k in outputs)
                or any(k in live for k in inputs)):
            raise ValueError("missing leaf, duplicate edge or invalid graph dependency")
        selected = [live[k] for k in outputs]
        _validate_specs(selected, record["output_specs"])
        special = record["action"] in ("concat", "split_segments")
        if (record["ordinary_operator"] != (not special)
                or len(record["input_depths"]) != len(inputs)
                or record["depth"] != max(record["input_depths"]) + int(not special)
                or any(depths[k] != record["depth"] for k in outputs)
                or [views[k] for k in outputs] != record["output_views"]):
            raise ValueError("invalid graph depth/view metadata")
        counted += int(not special)
        if record["kind"] == "layout":
            metadata = dict(record["metadata"])
            valid_ops = {"columnize": {1, 2}, "float_deconstruct": {3}, "bitsplit": {4},
                         "bitunpack": {5}, "concat": {6}, "split_segments": {7, 8}}
            if metadata["op"] not in valid_ops.get(record["action"], set()):
                raise ValueError("layout name/op mismatch")
            metadata["side"] = _unb64(metadata["side"])
            native_selected = selected
            if record["action"] == "concat":
                if len(inputs) < 2 or len(selected) != 1:
                    raise ValueError("Concat must reduce >=2 inputs to one active leaf")
                boundary = record["concat_boundaries"]
                boundary_stream = Stream(boundary["kind"], boundary["width"], _unb64(boundary["data"]))
                if boundary_stream.spec() != {k: boundary[k] for k in ("kind", "width", "size")}:
                    raise ValueError("Concat boundary size mismatch")
                native_selected = [boundary_stream, *selected]
            restored, _ = layout.unpack_bundle(layout._native(
                "decode-bundle", layout.pack_bundle(native_selected, **metadata)))
            if len(restored) != len(record["input_specs"]):
                raise ValueError("inverse layout arity mismatch")
            restored = [Stream(spec["kind"], spec["width"], stream.data)
                        for spec, stream in zip(record["input_specs"], restored)]
        elif record["kind"] in ("ports", "wire"):
            if len(inputs) != 1:
                raise ValueError("codec edge must have one original input")
            descriptor = taxonomy.describe_action(record["action"])
            if descriptor.legacy_policy_action_id != record["policy_action_id"]:
                raise ValueError("frame action/selector disagreement")
            if record["kind"] == "wire":
                if len(selected) != 1 or selected[0].kind != "serial":
                    raise ValueError("wire inverse requires one serial stream")
                encoded = unified.ActionData.bytes(selected[0].data)
            else:
                specs = record["port_specs"]
                if len(specs) != len(selected):
                    raise ValueError("port inverse arity mismatch")
                typed = tuple(ports_api.TypedStream(p["name"], p["kind"], s.data, p["count"])
                              for p, s in zip(specs, selected))
                encoded = _native_from_ports(ports_api.ActionPorts(
                    typed, record["port_metadata"], record["port_header"]))
            data = unified.run_policy_action(record["policy_action_id"], encoded,
                registry_version=canonical.REGISTRY_VERSION, reverse=True,
                io_variant=record["io_variant"], parameters=record["parameters"] or None,
                max_point=record["max_point"])
            if data.kind != record["input_view"]:
                raise ValueError("codec input-view reconstruction mismatch")
            spec = record["input_specs"][0]
            restored = [Stream(spec["kind"], spec["width"], data.payload)]
        else:
            raise ValueError("unknown graph edge kind")
        _validate_specs(restored, record["input_specs"])
        for key in outputs:
            del live[key], depths[key], views[key]
        live.update(zip(inputs, restored))
        depths.update(zip(inputs, record["input_depths"]))
        views.update(zip(inputs, record["input_views"]))
    roots, order = header["roots"], header["root_order"]
    if (set(live) != set(roots) or set(order) != set(roots)
            or len(order) != len(set(order)) or counted != header["codec_count"]
            or any(d != 0 for d in depths.values())):
        raise ValueError("unresolved graph roots or operator count")
    _validate_specs([live[k] for k in order], [roots[k] for k in order])
    return {k: live[k] for k in order}


def export_state(state):
    """JSON-safe durable search checkpoint; includes only live data and metadata."""
    result = asdict(state)
    result["streams"] = {k: {"kind": s.kind, "width": s.width, "data": _b64(s.data)}
                         for k, s in state.streams.items()}
    result["runtime_version"] = RUNTIME_VERSION
    result["registry_sha256"] = registry_fingerprint()
    return result


def import_state(record):
    value = dict(record)
    if (value.pop("runtime_version") != RUNTIME_VERSION
            or value.pop("registry_sha256") != registry_fingerprint()):
        raise ValueError("checkpoint runtime/registry mismatch")
    value["streams"] = {k: Stream(v["kind"], v["width"], _unb64(v["data"]))
                        for k, v in value["streams"].items()}
    value["steps"] = tuple(value["steps"])
    value["exposure_gaps"] = tuple(value["exposure_gaps"])
    return GraphState(**value)
