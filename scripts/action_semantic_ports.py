"""Rebuildable typed views of action results, separate from UAR transport.

This module parses formats; it does not implement a compression transform.
Reference compact values are decoded/encoded by the existing C++ CLI. Encoded
bitstreams remain byte streams. Unsigned values are never numerically cast to
signed integers. Metadata records layout and side information, not a hidden
copy of the original payload used in place of edited streams.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import subprocess
from typing import Any, TYPE_CHECKING
import zlib

if TYPE_CHECKING:
    import unified_action_runtime as unified


class PortSchemaError(ValueError):
    """A typed stream or serialized packet violates its declared shape."""


_WIDTHS = {"int64": 8, "uint64": 8, "uint32": 4, "uint8": 1,
           "float64": 8, "bytes": 1}
_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TypedStream:
    name: str
    kind: str
    payload: bytes
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PortSchemaError("stream name must be nonempty")
        if self.kind not in _WIDTHS:
            raise PortSchemaError(f"unsupported stream kind: {self.kind!r}")
        if type(self.count) is not int or self.count < 0:
            raise PortSchemaError("stream count must be a nonnegative integer")
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise PortSchemaError("stream payload must be bytes-like")
        object.__setattr__(self, "payload", bytes(self.payload))
        if len(self.payload) != self.count * _WIDTHS[self.kind]:
            raise PortSchemaError(f"{self.name}: {self.kind}[{self.count}] byte length mismatch")


@dataclass(frozen=True)
class ActionPorts:
    streams: tuple[TypedStream, ...]
    metadata: dict[str, Any]
    header: dict[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "streams", tuple(self.streams))
        if not all(isinstance(s, TypedStream) for s in self.streams):
            raise PortSchemaError("streams must contain TypedStream objects")
        if len({s.name for s in self.streams}) != len(self.streams):
            raise PortSchemaError("duplicate stream name")
        if not isinstance(self.metadata, dict):
            raise PortSchemaError("metadata must be an object")
        if self.header is not None and not isinstance(self.header, dict):
            raise PortSchemaError("header must be a UAR metadata object or None")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise PortSchemaError(message)


class _View:
    def __init__(self, payload: bytes, schema: str, header: dict | None):
        self.payload = payload
        self.pos = 0
        self.streams: list[TypedStream] = []
        self.meta: dict[str, Any] = {"schema": schema, "segments": [], "constraints": []}
        self.header = header

    def take(self, size: int) -> bytes:
        _check(size >= 0 and size <= len(self.payload) - self.pos, "truncated packet")
        raw = self.payload[self.pos:self.pos + size]
        self.pos += size
        return raw

    def number(self, fmt: str) -> int:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def literal(self, size: int) -> bytes:
        raw = self.take(size)
        self.meta["segments"].append({"literal": raw.hex()})
        return raw

    def saved_number(self, fmt: str) -> int:
        raw = self.literal(struct.calcsize(fmt))
        return struct.unpack(fmt, raw)[0]

    def stream(self, name: str, kind: str, count: int, *, variable: bool = False) -> None:
        raw = self.take(count * _WIDTHS[kind])
        self.streams.append(TypedStream(name, kind, raw, count))
        segment = {"stream": name, "kind": kind}
        if not variable:
            segment["count"] = count
        self.meta["segments"].append(segment)

    def interleaved(self, fields: list[tuple[str, str, int]], count: int) -> None:
        widths = [(_WIDTHS[kind] * lanes) for _, kind, lanes in fields]
        stride = sum(widths)
        raw = self.take(stride * count)
        descriptors = []
        offset = 0
        for (name, kind, lanes), width in zip(fields, widths):
            body = b"".join(raw[i * stride + offset:i * stride + offset + width]
                            for i in range(count))
            self.streams.append(TypedStream(name, kind, body, count * lanes))
            descriptors.append({"stream": name, "kind": kind, "lanes": lanes})
            offset += width
        self.meta["segments"].append({"interleaved": descriptors, "count": count})

    def finish(self) -> ActionPorts:
        _check(self.pos == len(self.payload), "unexpected trailing packet bytes")
        result = ActionPorts(tuple(self.streams), self.meta, deepcopy(self.header))
        _validate(result)
        return result


def _values(stream: TypedStream) -> tuple[int, ...]:
    formats = {"uint8": "B", "uint32": "I", "uint64": "Q", "int64": "q"}
    _check(stream.kind in formats, "integer constraint applied to noninteger stream")
    return tuple(x[0] for x in struct.iter_unpack("<" + formats[stream.kind], stream.payload))


def _validate(ports: ActionPorts) -> None:
    by_name = {s.name: s for s in ports.streams}
    for rule in ports.metadata.get("constraints", []):
        try:
            values = _values(by_name[rule["stream"]])
        except KeyError as exc:
            raise PortSchemaError(f"missing constrained stream: {exc.args[0]}") from exc
        if rule["type"] == "range":
            _check(all(rule["min"] <= x <= rule["max"] for x in values), "stream value outside schema range")
        elif rule["type"] == "permutation":
            _check(sorted(values) == list(range(rule["count"])), "invalid original-index permutation")
        elif rule["type"] == "sum":
            _check(sum(values) == rule["total"], "run counts do not match source count")
        elif rule["type"] == "pongo_flags":
            _check(all(x == 0 or (x & 0xF0) == 0x80 for x in values), "invalid Pongo eraser flags")
        else:
            raise PortSchemaError("unknown stream constraint")


def _rebuild_segments(segments: list[dict], by_name: dict[str, TypedStream], used: set[str]) -> bytes:
    out = bytearray()

    def get(desc: dict, count: int | None = None) -> TypedStream:
        name = desc["stream"]
        _check(name in by_name, f"missing stream: {name}")
        value = by_name[name]
        _check(value.kind == desc["kind"], f"{name}: stream type changed")
        expected = desc.get("count", count)
        _check(expected is None or value.count == expected, f"{name}: stream shape changed")
        used.add(name)
        return value

    for segment in segments:
        if "literal" in segment:
            out.extend(bytes.fromhex(segment["literal"]))
        elif "stream" in segment:
            out.extend(get(segment).payload)
        elif "interleaved" in segment:
            count = segment["count"]
            fields = [(get(d, count * d["lanes"]), d["lanes"] * _WIDTHS[d["kind"]])
                      for d in segment["interleaved"]]
            for i in range(count):
                for stream, width in fields:
                    out.extend(stream.payload[i * width:(i + 1) * width])
        elif "length_of" in segment:
            name = segment["length_of"]
            _check(name in by_name, f"missing stream: {name}")
            out.extend(struct.pack(segment["format"], len(by_name[name].payload)))
        elif "leco_blocks" in segment:
            spec = segment["leco_blocks"]
            params = get({"stream": "model_parameters", "kind": "float64", "count": 2 * len(spec["lengths"])})
            residuals = get({"stream": "residuals", "kind": spec["residual_kind"],
                             "count": sum(spec["lengths"])})
            out.extend(spec["magic"].encode("ascii"))
            out.extend(struct.pack("<I", sum(spec["lengths"])))
            parameter_pos = residual_pos = 0
            for length in spec["lengths"]:
                out.extend(struct.pack("<H", length))
                out.extend(params.payload[parameter_pos:parameter_pos + 16])
                parameter_pos += 16
                size = length * 8
                out.extend(residuals.payload[residual_pos:residual_pos + size])
                residual_pos += size
        else:
            raise PortSchemaError("invalid reconstruction segment")
    return bytes(out)


def _compact(command: str, payload: bytes) -> bytes:
    executable = _ROOT / "cpp/build/reference_operator_actions_cli"
    _check(executable.is_file(), "reference C++ CLI is required for compact typed views")
    result = subprocess.run([str(executable), command], input=payload, capture_output=True, check=False)
    _check(result.returncode == 0, "reference compact codec rejected payload: " + result.stderr.decode(errors="replace"))
    return result.stdout


_REFERENCE_KINDS = {2: "uint64", 3: "int64", 7: "int64", 8: "uint64", 11: "uint64", 14: "int64", 28: "uint64"}
_REFERENCE_TAGS = {2: b"for", 3: b"reorder", 7: b"delta", 8: b"xor", 11: b"zigzag", 14: b"runlength", 28: b"xorchimp"}


def _reference(payload: bytes, header: dict) -> ActionPorts | None:
    try:
        dec = zlib.decompressobj()
        raw = dec.decompress(payload) + dec.flush()
    except zlib.error:
        return None
    if not raw.startswith(b"OPA1"):
        return None
    _check(dec.eof and not dec.unused_data and not dec.unconsumed_tail, "invalid reference zlib carrier")
    v = _View(raw, "reference.opa1", header)
    _check(v.literal(4) == b"OPA1", "wrong reference magic")
    _check(v.saved_number("<B") == 1, "unsupported reference version")
    action_count = v.saved_number("<B")
    ids = list(v.literal(action_count))
    if not ids or ids[-1] not in _REFERENCE_KINDS:
        return None
    _check(all(i < 29 for i in ids), "invalid reference action ID")
    v.meta["reference_action_ids"] = ids
    for n, action_id in enumerate(ids):
        tag = v.literal(12).rstrip(b"\0")
        if action_id in _REFERENCE_TAGS:
            _check(tag == _REFERENCE_TAGS[action_id], "reference action/tag mismatch")
        arrays = v.saved_number("<H")
        extra_size = v.saved_number("<I")
        v.stream(f"record{n}.extra", "bytes", extra_size)
        for a in range(arrays):
            count = v.saved_number("<I")
            size = v.saved_number("<I")
            _check(size == count * 8, "reference side-array size mismatch")
            field_names = {2: ["base"], 3: ["original_index"], 7: ["first"],
                           8: ["first"], 28: ["first", "references"]}
            names = field_names.get(action_id, [])
            label = names[a] if a < len(names) else f"array{a}"
            v.stream(f"record{n}.{label}", "int64", count)
            if action_id == 28 and a == 1:
                v.meta["constraints"].append({"type": "range", "stream": f"record{n}.{label}", "min": 0, "max": 127})
    compact_raw = v.take(len(raw) - v.pos)
    values = _compact("compact-decode", compact_raw)
    v.meta["reference_metadata_segments"] = v.meta.pop("segments")
    main = _View(values, "reference.main", header)
    if ids[-1] == 14:
        _check(len(values) % 16 == 0, "reference RLE record shape mismatch")
        main.interleaved([("counts", "uint64", 1), ("values", "int64", 1)], len(values) // 16)
        v.meta["constraints"].append({"type": "range", "stream": "counts", "min": 1, "max": 0xffffffff})
    else:
        main.stream("main", _REFERENCE_KINDS[ids[-1]], len(values) // 8)
    v.streams.extend(main.streams)
    v.meta["segments"] = main.meta["segments"]
    v.meta["encoding"] = "OPA1 metadata + C++ compact values + zlib level 6"
    v.meta["semantic_note"] = "Typed values precede compact/zlib carrier encoding; side records are required for inverse. uint64 fields preserve their original bit patterns."
    return v.finish()


def _simple_packet(payload: bytes, header: dict) -> ActionPorts | None:
    magic = payload[:4]
    known = {b"S2S1", b"S8S1", b"DBS1", b"DBD1", b"DBZ1", b"SVD1", b"BDL1", b"BF11",
             b"RRO1", b"RSD1", b"RAR1", b"RFO1", b"GXI1", b"GXF1", b"CXI1", b"CXF1", b"PER1", b"PXR1",
             b"SPI1", b"SPF1", b"SZI1", b"SZF1", b"SHF1", b"RLB1",
             b"B6S1", b"B6T1", b"B6D1", b"B6F1", b"UVL1", b"BSC1", b"BFR1", b"BSF1"}
    if magic not in known:
        return None
    v = _View(payload, magic.decode(), header)
    v.literal(4)
    if magic in {b"S2S1", b"S8S1", b"DBS1", b"DBD1", b"DBZ1"}:
        if magic == b"S2S1":
            _check(v.saved_number("<B") == 1, "S2S1 requires floating scaling metadata")
        if magic.startswith(b"DB"):
            scale = v.saved_number("<i")
            _check(-1 <= scale <= 18, "invalid DBP scale")
            v.meta["scale"] = scale
        else:
            v.stream("scale", "uint32", 1)
            v.meta["constraints"].append({"type": "range", "stream": "scale", "min": 0, "max": 18})
        count = v.saved_number("<Q")
        if magic in {b"DBD1", b"DBZ1"}:
            v.stream("first", "int64", min(count, 1))
            v.stream("residuals", "uint64" if magic == b"DBZ1" else "int64", max(0, count - 1))
        else:
            v.stream("values", "int64", count)
    elif magic == b"SVD1":
        v.meta["scale"] = v.saved_number("<i")
        _check(-1 <= v.meta["scale"] <= 18, "invalid SVB scale")
        count = v.saved_number("<I")
        v.stream("low_deltas", "uint32", count)
        v.stream("high_deltas", "uint32", count)
        v.meta["semantic_note"] = "Two independent modulo-2^32 delta streams; never reinterpret as one int64 subtraction."
    elif magic in {b"BDL1", b"BF11"}:
        count = v.saved_number("<Q")
        v.meta["source_count"] = count
        v.stream("first", "int64", 1)
        if magic == b"BF11":
            v.stream("base", "int64", 1)
        residual_count = v.saved_number("<Q")
        _check(residual_count == max(0, count - 1), "BOSA residual shape mismatch")
        v.stream("offsets" if magic == b"BF11" else "deltas", "uint64" if magic == b"BF11" else "int64", residual_count)
    elif magic in {b"RRO1", b"RSD1", b"RAR1", b"RFO1"}:
        count = v.saved_number("<I")
        if magic == b"RRO1":
            v.stream("values", "int64", count)
            v.stream("original_index", "uint32", count)
        else:
            v.stream("first", "int64", 1)
            if magic == b"RFO1":
                v.stream("base", "uint64", 1)
            v.stream("original_index", "uint32", count)
            residual_kind = "int64" if magic == b"RSD1" else "uint64"
            v.stream("offsets" if magic == b"RFO1" else "residuals", residual_kind, max(0, count - 1))
        v.meta["constraints"].append({"type": "permutation", "stream": "original_index", "count": count})
    elif magic in {b"GXI1", b"GXF1"}:
        count = v.saved_number("<Q")
        v.stream("first", "uint64", min(count, 1))
        v.stream("xor_residuals", "uint64", max(0, count - 1))
        v.meta["word_kind"] = "float64_bits" if magic == b"GXF1" else "int64_bits"
    elif magic in {b"CXI1", b"CXF1"}:
        count = v.saved_number("<Q")
        v.stream("first", "uint64", 1)
        v.interleaved([("references", "uint8", 1), ("trailing_zeros", "uint8", 1), ("xor_residuals", "uint64", 1)], count)
        v.meta["constraints"].extend([{"type": "range", "stream": "references", "min": 0, "max": 127},
                                      {"type": "range", "stream": "trailing_zeros", "min": 0, "max": 64}])
        v.meta["word_kind"] = "float64_bits" if magic == b"CXF1" else "int64_bits"
        v.meta["semantic_note"] = "Nonempty records include the typed EOS record; first is retained even for the empty packet."
    elif magic in {b"PER1", b"PXR1"}:
        count = v.saved_number("<Q")
        v.stream("first", "uint64", min(count, 1))
        v.interleaved([("eraser_flags", "uint8", 1), ("xor_residuals" if magic == b"PXR1" else "targets", "uint64", 1)], max(0, count - 1))
        v.meta["constraints"].append({"type": "pongo_flags", "stream": "eraser_flags"})
        v.meta["semantic_note"] = "eraser_flags: bit7=reencoded, bits0..3=leading decimal zeros; literal flag must be zero."
    elif magic in {b"SPI1", b"SPF1", b"SZI1", b"SZF1"}:
        count = v.saved_number("<Q")
        v.interleaved([("anchors", "int64", 1), ("residuals", "uint64" if magic[:2] == b"SZ" else "int64", 8)], count // 9)
        v.stream("raw_tail", "int64", count % 9)
        v.meta["word_kind"] = "float64_bits" if magic in {b"SPF1", b"SZF1"} else "int64"
        v.meta["semantic_note"] = "Only 8 residuals in each complete 9-word block are transformed; anchors and raw tail bypass ZigZag. Floating anchors/tail are raw bits."
    elif magic == b"SHF1":
        width = v.saved_number("<B")
        _check(width in {1, 4, 8}, "typed shuffle views support element widths 1/4/8")
        size = v.saved_number("<Q")
        v.meta["element_width"] = width
        for lane in range(width):
            v.stream(f"byte_plane{lane}", "uint8", size // width)
        v.stream("raw_tail", "bytes", size % width)
    elif magic == b"RLB1":
        size = v.saved_number("<Q")
        _check((len(payload) - v.pos) % 5 == 0, "RLE record width mismatch")
        v.interleaved([("values", "uint8", 1), ("counts", "uint32", 1)], (len(payload) - v.pos) // 5)
        v.meta["constraints"].extend([{"type": "range", "stream": "counts", "min": 1, "max": 0xffffffff},
                                      {"type": "sum", "stream": "counts", "total": size}])
    elif magic in {b"BSC1", b"BFR1", b"BSF1"}:
        scaled = v.saved_number("<B")
        scale = v.saved_number("<B")
        count = v.saved_number("<I")
        _check(scaled in (0,1), "invalid BUFF scaling flag")
        v.meta.update(source_count=count, scaled=bool(scaled), scale=scale)
        if magic == b"BSC1":
            v.stream("values", "int64", count)
        else:
            v.stream("minimum", "int64", 1)
            if magic == b"BFR1":
                v.stream("offsets", "uint64", count)
                v.meta["semantic_note"] = "A013 FOR outputs minimum and unsigned offsets. Scale and raw/scaled kind bypass."
            else:
                width = v.saved_number("<B")
                size = v.saved_number("<I")
                _check(width <= 64 and size == ((width+7)//8)*count, "invalid BUFF plane width or length")
                for plane in range((width+7)//8):
                    v.stream(f"byte_plane{plane}", "uint8", count)
                v.meta.update(bit_width=width, plane_count=(width+7)//8,
                    semantic_note="A048 generates only byte planes, high plane first. minimum is an incoming bypass stream; FOR is a separate A013 action. Low padding bits are zero.")
    elif magic in {b"B6S1", b"B6T1", b"B6D1", b"B6F1"}:
        scaled = magic in {b"B6T1", b"B6F1"}
        signed = magic in {b"B6S1", b"B6T1"}
        if scaled:
            scale = v.saved_number("<i")
            _check(-1 <= scale <= 18, "invalid BOS64 scale")
            v.meta["scale"] = scale
            inner = v.literal(4)
            _check(inner == (b"B6S1" if signed else b"B6D1"), "invalid scaled BOS64 inner packet")
        count = v.saved_number("<Q")
        v.stream("first", "int64", 1)
        residual_count = v.saved_number("<Q")
        _check(residual_count == max(0, count - 1), "BOS64 residual shape mismatch")
        v.stream("residuals", "int64" if signed else "uint64", residual_count)
        v.meta.update(source_count=count, scaled=scaled,
                      semantic_note="Delta output is the first value plus signed residuals; the ZigZag stage changes only residuals to uint64.")
    elif magic == b"UVL1":
        count = v.saved_number("<Q")
        size = v.saved_number("<Q")
        _check(size == len(payload) - v.pos, "unsigned ULEB128 payload length mismatch")
        v.stream("encoded", "bytes", size, variable=True)
        v.meta.update(value_count=count,
                      semantic_note="One unsigned ULEB128 byte stream; count and payload length are restoration metadata.")
    return v.finish()


def _leco_packet(payload: bytes, header: dict) -> ActionPorts | None:
    magic = payload[:4]
    if magic not in {b"LRD1", b"LAR1"}:
        return None
    v = _View(payload, magic.decode(), header)
    v.take(4)
    count = v.number("<I")
    lengths = []
    parameters = bytearray()
    residuals = bytearray()
    total = 0
    while total < count:
        length = v.number("<H")
        _check(length and total + length <= count, "invalid LECO block length")
        lengths.append(length)
        parameters.extend(v.take(16))
        residuals.extend(v.take(length * 8))
        total += length
    _check(v.pos == len(payload), "unexpected trailing LECO bytes")
    residual_kind = "int64" if magic == b"LRD1" else "uint64"
    streams = (
        TypedStream("model_parameters", "float64", bytes(parameters), 2 * len(lengths)),
        TypedStream("residuals", residual_kind, bytes(residuals), count),
    )
    metadata = {
        "schema": magic.decode(),
        "segments": [{"leco_blocks": {
            "magic": magic.decode(), "lengths": lengths,
            "residual_kind": residual_kind,
        }}],
        "constraints": [],
        "value_count": count,
        "parameter_order": "intercept,slope per block",
        "semantic_note": "Two semantic streams: float64 model parameters and signed/unsigned residual values; block lengths remain framing metadata.",
    }
    result = ActionPorts(streams, metadata, deepcopy(header))
    _validate(result)
    return result


_ENVELOPES = {b"B1F1", b"B2F1", b"GZF1", b"ZLF1", b"XZF1", b"CRD2"}


def _envelope(payload: bytes, header: dict) -> ActionPorts | None:
    if payload[:4] not in _ENVELOPES:
        return None
    v = _View(payload, "magic_length", header)
    magic = v.literal(4)
    size = v.number("<Q")
    _check(size == len(payload) - v.pos, "envelope length mismatch")
    v.meta["segments"].append({"length_of": "payload", "format": "<Q"})
    v.stream("payload", "bytes", size, variable=True)
    v.meta["magic"] = magic.decode()
    v.meta["semantic_note"] = "Magic + uint64LE payload byte length; no compression."
    if magic == b"CRD2":
        _check(size >= 12 and payload[12:16] == b"CZL2", "CRD2 requires CZL2 payload")
    return v.finish()


def _container(payload: bytes, header: dict, variant: str) -> ActionPorts | None:
    kind = {"op.container_frame.gzip.cpp_v1": "gzip", "op.container_frame.zlib.cpp_v1": "zlib",
            "op.container_frame.xz.cpp_v1": "xz"}.get(variant)
    if kind is None:
        return None
    n = len(payload)
    if kind == "gzip":
        _check(n >= 18 and payload[:3] == b"\x1f\x8b\x08", "invalid gzip header")
        flags = payload[3]
        _check(flags & 0xe0 == 0, "reserved gzip flags")
        head, trailer = 10, 8
        if flags & 4:
            _check(head + 2 <= n - trailer, "truncated gzip extra length")
            head += 2 + struct.unpack_from("<H", payload, head)[0]
            _check(head <= n - trailer, "truncated gzip extra")
        for bit in (8, 16):
            if flags & bit:
                end = payload.find(b"\0", head, n - trailer)
                _check(end >= 0, "unterminated gzip text field")
                head = end + 1
        if flags & 2:
            head += 2
    elif kind == "zlib":
        _check(n >= 6, "truncated zlib stream")
        cmf, flags = payload[:2]
        _check(cmf & 15 == 8 and cmf >> 4 <= 7 and (cmf * 256 + flags) % 31 == 0, "invalid zlib header")
        head, trailer = (6 if flags & 32 else 2), 4
    else:
        _check(n >= 24 and payload[:6] == b"\xfd7zXZ\0" and payload[-2:] == b"YZ", "invalid XZ stream")
        head, trailer = 12, 12
    _check(head <= n - trailer, "container header overlaps trailer")
    v = _View(payload, "container_parts", header)
    v.stream("container_header", "bytes", head, variable=True)
    v.stream("core", "bytes", n - head - trailer, variable=True)
    v.stream("trailer", "bytes", trailer, variable=True)
    v.meta["container_kind"] = kind
    v.meta["semantic_note"] = "Existing container parts only; core remains encoded bytes; checksums are preserved, not recomputed or validated."
    return v.finish()


def _variant(header: dict) -> str:
    value = header.get("io_variant")
    if isinstance(value, str):
        return value
    actions = header.get("actions", [])
    return actions[-1] if isinstance(actions, list) and actions and isinstance(actions[-1], str) else ""


def _limb_wires(payload: bytes, header: dict) -> ActionPorts:
    _check(payload.startswith(b"LNS1"), "expected LNS1 wires packet")
    try:
        template = json.loads(payload[4:])
    except (ValueError, UnicodeDecodeError) as exc:
        raise PortSchemaError("invalid LNS1 JSON") from exc
    _check(isinstance(template, dict) and template.get("stage") == "wires"
           and isinstance(template.get("blocks"), list), "expected LNS1 wires blocks")
    streams = []
    bindings = []
    for block_index, block in enumerate(template["blocks"]):
        _check(isinstance(block, dict), "invalid LNS1 block")
        fields = [("raw", None)] if "raw" in block else [("wires", i) for i in range(len(block.get("wires", [])))]
        _check(fields or ("wires" in block and isinstance(block["wires"], list)), "LNS1 block lacks raw/wires")
        for field, lane in fields:
            text = block[field] if lane is None else block[field][lane]
            _check(isinstance(text, str), "LNS1 body must be hex text")
            try:
                body = bytes.fromhex(text)
            except ValueError as exc:
                raise PortSchemaError("invalid LNS1 hex body") from exc
            _check(body.hex() == text.lower(), "noncanonical LNS1 hex body")
            name = f"block{block_index}.raw" if lane is None else f"block{block_index}.wire{lane}"
            streams.append(TypedStream(name, "bytes", body, len(body)))
            bindings.append({"stream": name, "block": block_index, "field": field, "lane": lane})
            if lane is None:
                block[field] = None
            else:
                block[field][lane] = None
    return ActionPorts(tuple(streams), {"schema": "LNS1.wires", "template": template, "bindings": bindings,
        "semantic_note": "Each wire is still an encoded ctrl8 token bitstream; raw bypass blocks remain bytes. Existing block/lane boundaries and all numeric side metadata are retained."}, deepcopy(header))


_OPAQUE_SEMANTICS = {
    "op.trailing_zero_pack.gorilla.cpp_v1": "Encoded Gorilla control bits, XOR window payload and typed sentinel; numeric residuals are not directly exposed.",
    "op.trailing_zero_pack.pongo.cpp_v1": "Encoded Pongo eraser side fields and XOR window bit payload; requires its bitstream decoder.",
    "op.trailing_zero_pack.fpc.cpp_v1": "FPC block headers, packed predictor/length nibbles and variable-byte residual payload; remains encoded bytes.",
    "op.trailing_zero_pack.pfpc.cpp_v1": "pFPC block headers, packed predictor/length nibbles and variable-byte residual payload; remains encoded bytes.",
    "op.match_token_format.native_lz.ctrl8_offset16_len8.bias5": "Encoded ctrl8 tokens; inverse also requires native_lz_stage_context in the UAR header.",
}


def _raw_atomic_stage(payload: bytes, header: dict, variant: str) -> ActionPorts | None:
    if variant == "op.zigzag.signed64.raw.cpp_v2":
        _check(len(payload) % 8 == 0, "raw ZigZag output requires complete uint64 lanes")
        v = _View(payload, "raw_zigzag_uint64", header)
        v.stream("values", "uint64", len(payload) // 8)
        v.meta["semantic_note"] = "One uint64 stream; signed input and unsigned output use the standard 64-bit ZigZag bijection."
        return v.finish()
    if variant == "op.integer_code.uleb128.reference_values.cpp_v2":
        v = _View(payload, "raw_uleb128", header)
        v.stream("encoded", "bytes", len(payload), variable=True)
        v.meta["semantic_note"] = "One unsigned ULEB128 byte stream; input values are uint64 little-endian lanes."
        return v.finish()
    if variant == "op.integer_code.uleb128.dbp_residuals.cpp_v2":
        _check(payload and payload[0] in (ord("I"), ord("F")), "invalid DBP ULEB128 wire tag")
        v = _View(payload, "dbp_uleb128", header)
        tag = v.literal(1)
        v.meta["value_domain"] = "integer" if tag == b"I" else "scaled_float"
        if tag == b"F":
            scale = v.saved_number("<i")
            _check(-1 <= scale <= 18, "invalid DBP scale")
            v.meta["scale"] = scale
        if v.pos < len(payload):
            _check(len(payload) - v.pos >= 8, "truncated DBP first value")
            v.stream("first", "int64", 1)
        else:
            v.stream("first", "int64", 0)
        v.stream("encoded_residuals", "bytes", len(payload) - v.pos, variable=True)
        v.meta["semantic_note"] = "Two semantic streams: the unencoded first int64 value and unsigned ULEB128 residual bytes; scale is framing metadata."
        return v.finish()
    if variant == "op.integer_code.uleb128.snappy_raw_size.cpp_v2":
        _check(payload, "truncated Snappy length")
        pos = 0
        value = 0
        for shift in range(0, 35, 7):
            _check(pos < len(payload), "truncated Snappy length")
            byte = payload[pos]
            pos += 1
            value |= (byte & 0x7f) << shift
            if not byte & 0x80:
                break
        else:
            raise PortSchemaError("Snappy length exceeds uint32 ULEB128")
        _check(value <= 0xffffffff, "Snappy length exceeds uint32")
        v = _View(payload, "snappy_uleb128", header)
        v.stream("encoded_length", "bytes", pos)
        v.stream("token_body", "bytes", len(payload) - pos, variable=True)
        v.meta.update(raw_size=value,
                      semantic_note="Two semantic streams: generated unsigned ULEB128 raw length and the bypassed Snappy token body.")
        return v.finish()
    return None


# Old canonical206 variant names intentionally survive algorithm-level merging.
# This is an exposure contract, not a second action registry or execution route.
_VARIANT_LAYOUTS = {
    **{f"op.{family}.reference_v1": {"schemas": ["reference.opa1"], "exposure": "numeric"}
       for family in ("frame_of_reference", "value_reorder", "delta", "xor", "zigzag", "run_length", "xor.chimp_history128")},
    **{name: {"schemas": schemas, "exposure": "numeric"} for name, schemas in {
        "op.scaling.ts_2diff.cpp_v1": ["S2S1"],
        "op.scaling.simple8b.cpp_v1": ["S8S1"],
        "op.frame_of_reference.bos.cpp_v1": ["BF11"],
        "op.frame_of_reference.reger.cpp_v1": ["RFO1"],
        "op.value_reorder.reger.cpp_v1": ["RRO1"],
        "op.delta.stream_vbyte.cpp_v1": ["SVD1"],
        "op.delta.delta_binary_packed.cpp_v1": ["DBD1"],
        "op.delta.bos.cpp_v1": ["BDL1"],
        "op.xor.gorilla.cpp_v1": ["GXI1", "GXF1"],
        "op.xor.pongo.cpp_v1": ["PXR1"],
        "op.xor.chimp128.cpp_v1": ["CXI1", "CXF1"],
        "op.zigzag.delta_binary_packed.cpp_v1": ["DBZ1"],
        "op.zigzag.sprintz.cpp_v1": ["SZI1", "SZF1"],
        "op.run_length.bzip2_uint8.cpp_reference_v1": ["RLB1"],
        **{f"op.block_transform.blosc_shuffle.typesize{width}.cpp_reference_v1": ["SHF1"] for width in (1, 4, 8)},
    }.items()},
    **{name: {"schemas": ["opaque_bytes"], "exposure": "encoded_bytes"}
       for name in _OPAQUE_SEMANTICS},
    "op.regression_predictor.leco_linear.cpp_v2": {"schemas": ["LRD1"], "exposure": "numeric"},
    "op.zigzag.leco_residual.cpp_v2": {"schemas": ["LAR1"], "exposure": "numeric"},
    "op.delta.reger_signed_residual.cpp_v2": {"schemas": ["RSD1"], "exposure": "numeric"},
    "op.zigzag.reger_residual.cpp_v2": {"schemas": ["RAR1"], "exposure": "numeric"},
    "op.scaling.buff.cpp_v1": {"schemas":["BSC1"], "exposure":"numeric"},
    "op.frame_of_reference.buff.cpp_v2": {"schemas":["BFR1"], "exposure":"numeric"},
    "op.block_transform.buff_byte_plane_split.cpp_v2": {"schemas":["BSF1"], "exposure":"numeric"},
    "op.delta.bos64_signed_residual.cpp_v2": {"schemas": ["B6S1", "B6T1"], "exposure": "numeric"},
    "op.zigzag.bos64_residual.cpp_v2": {"schemas": ["B6D1", "B6F1"], "exposure": "numeric"},
    "op.zigzag.signed64.raw.cpp_v2": {"schemas": ["raw_zigzag_uint64"], "exposure": "numeric"},
    "op.integer_code.uleb128.reference_values.cpp_v2": {"schemas": ["raw_uleb128"], "exposure": "encoded_bytes"},
    "op.integer_code.uleb128.dbp_residuals.cpp_v2": {"schemas": ["dbp_uleb128"], "exposure": "structured_bytes"},
    "op.integer_code.uleb128.snappy_raw_size.cpp_v2": {"schemas": ["snappy_uleb128"], "exposure": "structured_bytes"},
    "op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1": {
        "schemas": ["LNS1.wires"], "exposure": "structured_bytes"},
    **{f"op.container_frame.{codec}.cpp_v1": {"schemas": ["container_parts"], "exposure": "structured_bytes"}
       for codec in ("gzip", "zlib", "xz")},
    **{name: {"schemas": ["magic_length"], "exposure": "structured_bytes", "magic": magic}
       for name, magic in {
        "op.block_state_control.blosc1.cpp_reference_v1": "B1F1",
        "op.block_state_control.blosc2.cpp_reference_v1": "B2F1",
        "op.container_frame.gzip.cpp_reference_v1": "GZF1",
        "op.container_frame.zlib.cpp_reference_v1": "ZLF1",
        "op.container_frame.xz.cpp_reference_v1": "XZF1",
        "op.container_frame.corad_lossless.cpp_v2": "CRD2",
    }.items()},
}


def _minimum_for(payload: bytes, header: dict) -> ActionPorts:
    v = _View(payload, "MFR1", header)
    _check(v.literal(4) == b"MFR1", "minimum FOR requires MFR1")
    kind = v.saved_number("<B")
    count = v.saved_number("<Q")
    _check(kind in (0, 1), "invalid minimum FOR numeric kind")
    _check(len(payload) == 21 + count * 8, "minimum FOR offsets length mismatch")
    v.stream("minimum", "int64" if kind == 0 else "uint64", 1)
    v.stream("offsets", "uint64", count)
    v.meta.update(value_count=count, input_domain="int64" if kind == 0 else "uint64", empty_minimum=0,
                  semantic_note="Two independent numeric streams; offsets = values - minimum; no residual-only precondition")
    return v.finish()


def _max_value_width(payload: bytes, header: dict) -> ActionPorts:
    v = _View(payload, "MVW1", header)
    _check(v.literal(4) == b"MVW1", "max-value-width requires MVW1")
    count = v.saved_number("<Q")
    width = v.saved_number("<B")
    _check(width <= 64, "invalid max-value-width bit width")
    expected = (count * width + 7) // 8
    _check(len(payload) == 13 + expected, "max-value-width packed length mismatch")
    v.meta.update(value_count=count, bit_width=width, input_domain="uint64 values; signed int64 inputs must be nonnegative", bit_order="MSB-first",
                  semantic_note="One packed byte stream; count and bit width remain side metadata. Values need not be residuals.")
    v.stream("packed", "bytes", expected)
    return v.finish()


def to_ports(data: "unified.ActionData") -> ActionPorts:
    """Inspect an actual forward result; no implicit cast or codec invocation.

    Reference compact carrier decoding is the one explicit C++ codec call.
    An arbitrary raw bytes value is opaque even when it starts with a known
    stage magic. Only a UAR result declares a stage output boundary.
    """
    if data.kind != "bytes" or not data.payload.startswith(b"UAR1"):
        return ActionPorts((TypedStream("main", data.kind, data.payload, data.count),),
                           {"schema": "raw", "raw_kind": data.kind, "raw_count": data.count}, None)
    raw = data.payload
    _check(len(raw) >= 8, "truncated UAR1 header")
    header_size = struct.unpack_from("<I", raw, 4)[0]
    _check(header_size <= len(raw) - 8, "truncated UAR1 metadata")
    try:
        header = json.loads(raw[8:8 + header_size])
    except (ValueError, UnicodeDecodeError) as exc:
        raise PortSchemaError("invalid UAR1 JSON metadata") from exc
    _check(isinstance(header, dict), "invalid UAR1 metadata object")
    payload = raw[8 + header_size:]
    variant = _variant(header)
    if variant == "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1":
        return _minimum_for(payload, header)
    if variant == "op.bitpack_int.max_value_width.uint64_le.cpp_v1":
        return _max_value_width(payload, header)
    if variant in {"op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1", "op.match_token_format.native_lz_limb_batch.ctrl8.python_v1"}:
        return _limb_wires(payload, header)
    result = _raw_atomic_stage(payload, header, variant)
    if result is not None:
        return result
    result = _leco_packet(payload, header)
    if result is not None:
        return result
    # These actions emit bitstreams: coincidental magic bytes are not a schema.
    if variant not in _OPAQUE_SEMANTICS:
        if variant.endswith(".reference_v1"):
            result = _reference(payload, header)
            if result is not None:
                return result
        result = _container(payload, header, variant)
        if result is not None:
            return result
        result = _simple_packet(payload, header)
        if result is not None:
            return result
        result = _envelope(payload, header)
        if result is not None:
            return result
    v = _View(payload, "opaque_bytes", header)
    v.stream("encoded", "bytes", len(payload), variable=True)
    v.meta["semantic_note"] = _OPAQUE_SEMANTICS.get(variant, "Unknown or encoded output layout; complete bytes are preserved without claiming a numeric stream.")
    return v.finish()


def from_ports(ports: ActionPorts) -> "unified.ActionData":
    """Rebuild a result from current streams and layout metadata.

    Stream edits are serialized. Header lineage/io_variant is preserved. A
    structurally valid edit may still violate a codec's semantic precondition;
    the codec inverse remains authoritative for that validation.
    """
    import unified_action_runtime as unified
    _validate(ports)
    by_name = {s.name: s for s in ports.streams}
    if ports.metadata.get("schema") == "raw":
        _check(ports.header is None and len(ports.streams) == 1, "raw data requires a single stream without UAR header")
        stream = ports.streams[0]
        _check(stream.name == "main" and stream.kind == ports.metadata.get("raw_kind"), "raw stream type changed")
        _check(stream.kind in {"int64", "float64", "bytes"}, "raw ActionData cannot carry this typed component directly")
        return unified.ActionData(stream.kind, stream.payload, stream.count)
    _check(isinstance(ports.header, dict), "stage ports require a UAR header")
    used: set[str] = set()
    try:
        if ports.metadata["schema"] == "LNS1.wires":
            template = deepcopy(ports.metadata["template"])
            for binding in ports.metadata["bindings"]:
                name = binding["stream"]
                _check(name in by_name and by_name[name].kind == "bytes", "LNS1 wire/raw port must be bytes")
                used.add(name)
                block = template["blocks"][binding["block"]]
                if binding["lane"] is None:
                    block[binding["field"]] = by_name[name].payload.hex()
                else:
                    block[binding["field"]][binding["lane"]] = by_name[name].payload.hex()
            payload = b"LNS1" + json.dumps(template, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
        else:
            payload = _rebuild_segments(ports.metadata["segments"], by_name, used)
        if ports.metadata["schema"] == "reference.opa1":
            prefix = _rebuild_segments(ports.metadata["reference_metadata_segments"], by_name, used)
            payload = zlib.compress(prefix + _compact("compact-encode", payload), 6)
    except (KeyError, TypeError, ValueError, struct.error) as exc:
        if isinstance(exc, PortSchemaError):
            raise
        raise PortSchemaError(f"invalid reconstruction metadata: {exc}") from exc
    _check(used == set(by_name), "unreferenced extra stream would be discarded")
    if ports.metadata["schema"] == "magic_length" and ports.metadata.get("magic") == "CRD2":
        inner = by_name["payload"].payload
        _check(len(inner) >= 12 and inner.startswith(b"CZL2"), "CRD2 requires a CZL2 packet")
    header_bytes = json.dumps(ports.header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return unified.ActionData.bytes(b"UAR1" + struct.pack("<I", len(header_bytes)) + header_bytes + payload)


def supported_layouts() -> dict[str, Any]:
    """Machine-queryable capabilities, distinguishing typed and opaque output."""
    return deepcopy({
        "schema_version": "action-semantic-ports-v1",
        "stream_kinds": dict(_WIDTHS),
        "byte_order": "little-endian for numeric streams; bytes retain wire order",
        "raw": ["int64", "float64", "bytes"],
        "typed_packet_magics": ["S2S1", "S8S1", "DBS1", "DBD1", "DBZ1", "SVD1", "BDL1", "BF11",
                                "RRO1", "RSD1", "RAR1", "RFO1", "LRD1", "LAR1",
                                "B6S1", "B6T1", "B6D1", "B6F1", "UVL1",
                                "GXI1", "GXF1", "CXI1", "CXF1", "PER1", "PXR1",
                                "SPI1", "SPF1", "SZI1", "SZF1", "SHF1", "RLB1"],
        "reference_action_ids": sorted(_REFERENCE_KINDS),
        "reference_codec_dependency": "cpp/build/reference_operator_actions_cli (compact decode/encode)",
        "container_parts_variants": ["op.container_frame.gzip.cpp_v1", "op.container_frame.zlib.cpp_v1", "op.container_frame.xz.cpp_v1"],
        "magic_length_envelopes": sorted(m.decode() for m in _ENVELOPES),
        "structured_byte_layouts": {"MVW1": "packed:bytes; uint64 value count and uint8 bit width retained in metadata",
                                    "LNS1.wires": "blockN.wireM:bytes, blockN.raw:bytes; template metadata retains side fields",
                                    "dbp_uleb128": "first:int64[0 or 1], encoded_residuals:bytes; tag and optional scale remain metadata",
                                    "snappy_uleb128": "encoded_length:bytes, token_body:bytes; decoded raw size is validation metadata",
                                    "container_parts": "container_header:bytes, core:bytes, trailer:bytes",
                                    "magic_length": "payload:bytes; magic and length field in layout metadata"},
        "numeric_layouts": {
            "MFR1": {"minimum": "int64[1] or uint64[1] according to input kind", "offsets": "uint64[N]", "metadata": "kind/count; empty minimum is zero"},
            "S2S1/S8S1": {"scale": "uint32[1]", "values": "int64[N]"},
            "DBS1": {"values": "int64[N]", "scale_metadata": "signed int32 (-1 or 0..18)"},
            "DBD1/DBZ1": {"first": "int64[0 or 1]", "residuals": "int64[N-1] / uint64[N-1]"},
            "SVD1": {"low_deltas": "uint32[N]", "high_deltas": "uint32[N]"},
            "BDL1/BF11": {"first": "uint64[1]", "base_if_FOR": "int64[1]", "deltas_or_offsets": "int64[N-1] / uint64[N-1]"},
            "RRO1": {"values": "int64[N]", "original_index": "uint32[N]"},
            "RSD1/RAR1/RFO1": {"first": "int64[1]", "base_if_FOR": "uint64[1]", "original_index": "uint32[N]", "residuals_or_offsets": "int64[N-1] before ZigZag; uint64[N-1] after"},
            "LRD1/LAR1": {"model_parameters": "float64[2B] intercept,slope", "residuals": "int64[N] before ZigZag; uint64[N] after", "metadata": "block lengths"},
            "B6S1/B6T1/B6D1/B6F1": {"first": "int64[1]", "residuals": "int64[N-1] before ZigZag; uint64[N-1] after", "metadata": "count and optional decimal scale"},
            "UVL1": {"encoded": "bytes[M]", "metadata": "uint64 value count and payload length"},
            "raw_zigzag_uint64": {"values": "uint64[N]"},
            "raw_uleb128": {"encoded": "bytes[M]"},
            "GXI1/GXF1": {"first": "uint64[0 or 1]", "xor_residuals": "uint64[N-1]"},
            "CXI1/CXF1": {"first": "uint64[1]", "references": "uint8[N]", "trailing_zeros": "uint8[N]", "xor_residuals": "uint64[N] (includes EOS when N>0)"},
            "PER1/PXR1": {"first": "uint64[0 or 1]", "eraser_flags": "uint8[N-1]", "targets_or_xor": "uint64[N-1]"},
            "SPI1/SPF1/SZI1/SZF1": {"anchors": "int64[B] bits", "residuals": "int64[8B] / uint64[8B]", "raw_tail": "int64[T] bits, 0<=T<9"},
            "SHF1": {"byte_plane0..W-1": "uint8[N]", "raw_tail": "bytes[T]"},
            "RLB1": {"values": "uint8[R]", "counts": "uint32[R]"},
            "reference.opa1": {"main": "int64/uint64[N] according to actual final reference transform", "side_records": "int64 arrays and bytes extra", "RLE_main": "values:int64[R], counts:uint64[R]"},
        },
        "opaque_variant_semantics": _OPAQUE_SEMANTICS,
        "document_merged_variant_layouts": _VARIANT_LAYOUTS,
        "unknown_layout": "single complete bytes stream",
    })
