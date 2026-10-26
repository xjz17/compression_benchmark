#!/usr/bin/env python3
"""General common-symbol splitting; frozen v1/v2 runtimes stay unchanged.

split_common selects exactly one full uint64 bit pattern: maximum frequency,
then smallest unsigned value on ties. It stores that value and a full-position
bit mask in charged metadata, and emits only the other uint64 values in order.
It does NOT predict, difference, pack, or entropy-code those remaining values.
Float bits are never interpreted as floating-point numbers by this primitive.
"""
from __future__ import annotations

import base64

import numpy as np

import atomic_transform_extensions_v2 as v2

VERSION = "atomic-transform-extensions-v3"
State = v2.State
U64 = v2.U64
MAX_BYTES = v2.MAX_BYTES
SUPPORTED_OPERATORS = v2.SUPPORTED_OPERATORS + ("split_common",)
INVERSE_FIELDS = dict(v2.INVERSE_FIELDS, split_common=("common_value", "mask_b64"))


def validate_specs(specs: list[dict]) -> str:
    if not isinstance(specs, (list, tuple)):
        raise ValueError("specs must be a list/tuple")
    kind = "u64"
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("op") not in SUPPORTED_OPERATORS:
            raise ValueError("unknown operator spec")
        if spec["op"] == "split_common":
            if kind != "u64" or set(spec) != {"op"}:
                raise ValueError("split_common requires u64 and has no extra parameters")
        else:
            prefix = [{"op": "compact", "width": 8}] if kind == "bytes" else []
            kind = v2.validate_specs(prefix + [spec])
    return kind


def apply_one(state: State, spec: dict) -> tuple[State, dict]:
    if not isinstance(spec, dict) or spec.get("op") not in SUPPORTED_OPERATORS:
        raise ValueError("unknown operator spec")
    if spec["op"] != "split_common":
        return v2.apply_one(state, spec)
    if set(spec) != {"op"} or state.kind != "u64":
        raise ValueError("split_common requires u64 and has no extra parameters")
    values = state.values()
    if len(values):
        palette, counts = np.unique(values, return_counts=True)
        # np.unique sorts ascending; argmax returns the first maximum.
        common = int(palette[int(np.argmax(counts))])
    else:
        common = 0
    matches = values == np.uint64(common)
    mask = np.packbits(matches, bitorder="little").tobytes()
    output = v2.v1._numeric(values[~matches])
    meta = {"op": "split_common", "in_kind": "u64", "in_count": state.count,
            "in_nbytes": len(state.payload), "out_kind": "u64", "out_count": output.count,
            "out_nbytes": len(output.payload), "common_value": common,
            "mask_b64": base64.b64encode(mask).decode("ascii")}
    return output, meta


def reverse_one(state: State, meta: dict) -> State:
    if not isinstance(meta, dict) or meta.get("op") not in SUPPORTED_OPERATORS:
        raise ValueError("unknown inverse operator record")
    if meta["op"] != "split_common":
        return v2.reverse_one(state, meta)
    allowed = {"op", "in_kind", "in_count", "in_nbytes", "out_kind", "out_count", "out_nbytes", "common_value", "mask_b64"}
    if set(meta) != allowed or meta.get("in_kind") != "u64" or state.kind != "u64":
        raise ValueError("invalid split_common inverse schema/type")
    v2.v1._check_record_state(state, meta, "out")
    count = v2.v1._integer(meta.get("in_count"), "in_count", 0, MAX_BYTES // 8)
    common = v2.v1._integer(meta.get("common_value"), "common_value", 0, (1 << 64) - 1)
    if meta.get("in_nbytes") != count * 8:
        raise ValueError("split_common input typed length mismatch")
    mask_raw = v2._decode(meta.get("mask_b64"), "common mask", element_width=1)
    if len(mask_raw) != (count + 7) // 8:
        raise ValueError("split_common mask byte length mismatch")
    bits = np.unpackbits(np.frombuffer(mask_raw, dtype=np.uint8), bitorder="little")
    if np.any(bits[count:]):
        raise ValueError("split_common mask padding must be zero")
    selected = bits[:count].astype(bool)
    matched_count = int(selected.sum())
    if (count and matched_count == 0) or (count == 0 and common != 0):
        raise ValueError("split_common mask/value contradicts nonempty/empty input")
    remaining = state.values()
    if len(remaining) != count - matched_count:
        raise ValueError("split_common mask and remaining count disagree")
    if np.any(remaining == np.uint64(common)):
        raise ValueError("split_common remaining stream contains the split value")
    restored = np.full(count, common, dtype=U64)
    restored[~selected] = remaining
    output = v2.v1._numeric(restored)
    v2.v1._check_record_state(output, meta, "in")
    return output


def apply_ops(raw: bytes, specs: list[dict]) -> tuple[bytes, list[dict]]:
    validate_specs(specs)
    state = State(raw, "u64")
    records = []
    for spec in specs:
        state, record = apply_one(state, spec)
        records.append(record)
    return state.payload, records


def reverse_ops(payload: bytes, metadata: list[dict]) -> bytes:
    if not isinstance(metadata, (list, tuple)) or any(not isinstance(record, dict) for record in metadata):
        raise ValueError("metadata must be a sequence of dicts")
    state = State(payload, metadata[-1].get("out_kind") if metadata else "u64")
    for record in reversed(metadata):
        state = reverse_one(state, record)
    if state.kind != "u64":
        raise ValueError("inverse did not restore initial u64 state")
    return state.payload
