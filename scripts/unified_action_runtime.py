#!/usr/bin/env python3
"""One typed invocation boundary for the corrected semantic action records.

The public data contract contains only three kinds: little-endian int64,
little-endian float64, and opaque bytes.  Every catalog record is addressable
through this module, but the runtime keeps its evidence boundary explicit:

* standalone actions can run by themselves;
* closed-sequence actions run only as a complete registered sequence;
* stagewise actions expose a real forward/inverse implementation and can also
  participate in a registered multi-stage chain;
* experimental-non-atomic helpers remain callable for forensic wire replay,
  but are excluded from the RL policy and all independent stage sequences;
* descriptor-only C++ ablation candidates raise ``ActionNotExecutableError``
  until a real forward/inverse stage implementation is connected.

This prevents a method-scoped ablation label from silently executing a
different reference implementation with the same operator-family name.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import primitive_operator_action_runtime as primitive  # noqa: E402
import native_lz_parameters as lz_parameters  # noqa: E402
import native_window_parameters as window_parameters
import current_action_parameters as action_parameters
import block_state_parameters as block_parameters
import max_value_bitpack as bitpack
import minimum_for as minimum
import optimal_rle_actions as optimal_rle


DataKind = Literal["int64", "float64", "bytes"]
ExecutionMode = Literal[
    "standalone",
    "stagewise",
    "experimental_non_atomic",
    "closed_sequence_only",
    "descriptor_only",
    "retired",
]
ActionRef = int | str

CATALOG_PATH = (
    ROOT
    / "analysis_outputs/operator_implementation_dedup_20260903_v1/implementations_194.csv"
)
WIRE_MAGIC = b"UAR1"


class ActionRuntimeError(ValueError):
    """Base error for the unified action boundary."""


class ActionNotExecutableError(ActionRuntimeError):
    """The catalog record has no honest forward/inverse implementation yet."""


class ActionTypeError(ActionRuntimeError):
    """The input kind does not match the selected action implementation."""


class ActionSequenceError(ActionRuntimeError):
    """The selected actions do not form a registered executable sequence."""


@dataclass(frozen=True)
class ActionData:
    """Transport-safe typed input or output for every action call."""

    kind: DataKind
    payload: bytes
    count: int

    def __post_init__(self) -> None:
        if self.kind not in {"int64", "float64", "bytes"}:
            raise ActionTypeError(f"unsupported data kind: {self.kind!r}")
        object.__setattr__(self, "payload", bytes(self.payload))
        if self.count < 0:
            raise ActionTypeError("count must be non-negative")
        expected = self.count if self.kind == "bytes" else self.count * 8
        if len(self.payload) != expected:
            raise ActionTypeError(
                f"{self.kind} count={self.count} requires {expected} bytes, "
                f"got {len(self.payload)}"
            )

    @classmethod
    def ints(cls, values: Sequence[int] | np.ndarray) -> "ActionData":
        array = np.asarray(values, dtype="<i8")
        return cls("int64", array.tobytes(order="C"), int(array.size))

    @classmethod
    def floats(cls, values: Sequence[float] | np.ndarray) -> "ActionData":
        array = np.asarray(values, dtype="<f8")
        return cls("float64", array.tobytes(order="C"), int(array.size))

    @classmethod
    def bytes(cls, value: bytes | bytearray | memoryview) -> "ActionData":
        payload = value.tobytes() if isinstance(value, memoryview) else bytes(value)
        return cls("bytes", payload, len(payload))

    def as_ints(self) -> np.ndarray:
        if self.kind != "int64":
            raise ActionTypeError(f"expected int64 data, got {self.kind}")
        return np.frombuffer(self.payload, dtype="<i8").copy()

    def as_floats(self) -> np.ndarray:
        if self.kind != "float64":
            raise ActionTypeError(f"expected float64 data, got {self.kind}")
        return np.frombuffer(self.payload, dtype="<f8").copy()


@dataclass(frozen=True)
class ActionDescriptor:
    action_id: int
    inventory_index: int
    name: str
    operator_id: str
    execution_mode: ExecutionMode
    accepted_input_kinds: tuple[DataKind, ...]
    output_kinds: tuple[DataKind, ...]
    implementation_source: str
    evidence: str
    source_methods: tuple[str, ...]
    legacy_registered_action_id: int | None
    unavailable_reason: str = ""


def _split_pipe(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _atomic_input_kinds(name: str) -> tuple[DataKind, ...]:
    if name in {
        "op.dictionary.project_int64.v1",
        "op.run_length.tsfile_int64.v1",
    }:
        return ("int64",)
    if name == "op.dictionary.project_float64.v1":
        return ("float64",)
    return ("bytes",)


_EXACT_STAGE_SOURCE_BY_NAME: dict[str, str] = {}
_REFERENCE_STAGE_SOURCE_BY_NAME: dict[str, str] = {}
_EXPERIMENTAL_STAGE_SOURCE_BY_NAME: dict[str, str] = {}
_EXACT_STAGE_METHODS_BY_NAME: dict[str, tuple[str, ...]] = {}


def _register_exact_stage_group(
    names: Sequence[str], source: str, methods: tuple[str, ...]
) -> None:
    for action_name in names:
        _EXACT_STAGE_METHODS_BY_NAME[action_name] = methods
        if action_name in primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES:
            _EXPERIMENTAL_STAGE_SOURCE_BY_NAME[action_name] = source
        else:
            _EXACT_STAGE_SOURCE_BY_NAME[action_name] = source


def _register_reference_stage_group(
    names: Sequence[str], source: str, methods: tuple[str, ...]
) -> None:
    for action_name in names:
        _EXACT_STAGE_METHODS_BY_NAME[action_name] = methods
        _REFERENCE_STAGE_SOURCE_BY_NAME[action_name] = source


for _names, _source, _methods in (
    (primitive.TS2DIFF_FLOAT_ACTIONS, "cpp/src/ts2diff_exact_stages.cpp", ("2DIFF",)),
    (primitive.SVB_INT_ACTIONS, "cpp/src/svb_exact_stages.cpp", ("SVB",)),
    (primitive.SIMPLE8B_FLOAT_ACTIONS, "cpp/src/simple8b_exact_stages.cpp", ("Simple8b",)),
    (primitive.RLBE_ACTIONS, "cpp/src/p04_exact_stages.cpp", ("RLBE",)),
    (primitive.SPRINTZ_ACTIONS, "cpp/src/p04_exact_stages.cpp", ("Sprintz",)),
    (primitive.DBP_FLOAT_ACTIONS, "cpp/src/dbp_exact_stages.cpp", ("DBP",)),
    ((primitive.ZIGZAG_RAW_ACTION.name,), "cpp/src/pure_numeric_operator_v1.cpp", ("Reference integer code",)),
    ((primitive.ULEB128_RAW_ACTION.name,), "cpp/src/unsigned_leb128_cli.cpp", ("Reference integer code",)),
    (primitive.GORILLA_ACTIONS, "cpp/src/gorilla_exact_stages.cpp", ("Gorilla",)),
    (
        primitive.CHIMP128_ACTIONS,
        "cpp/src/chimp128_exact_stages.cpp",
        ("Chimp", "Chimp128"),
    ),
    (primitive.PONGO_ACTIONS, "cpp/src/pongo_codec.cpp", ("Pongo",)),
    (primitive.SPDP_ACTIONS, "cpp/src/spdp_exact_stages.cpp", ("SPDP",)),
    (primitive.FPC_ACTIONS, "cpp/src/fpc_exact_stages.cpp", ("FPC",)),
    (primitive.PFPC_ACTIONS, "cpp/src/pfpc_exact_stages.cpp", ("PFPC",)),
    (primitive.BOS_BOSA_ACTIONS, "cpp/src/bos_exact_stages.cpp", ("BOS-source-BOSA",)),
    (primitive.BOS64_INT_ACTIONS, "cpp/src/bos_exact_stages.cpp", ("BOS",)),
    (tuple(action.name for action in primitive.FASTPFOR_COMPOSITE_HELPER_ACTIONS), "cpp/src/fastpfor_exact_stages.cpp", ("FastPFor-source-C++",)),
    (primitive.FASTPFOR_INT_ACTIONS, "cpp/src/fastpfor_exact_stages.cpp", ("FastPFor-source-C++",)),
    (primitive.ELF_ACTIONS, "cpp/src/elf_float_codec.cpp", ("Elf",)),
    (primitive.ELF_PLUS_ACTIONS, "cpp/src/elf_float_codec.cpp", ("Elf+",)),
    (primitive.CAMEL_ACTIONS, "cpp/src/camel_exact_stages.cpp", ("Camel",)),
    (primitive.SNAPPY_ACTIONS, "cpp/src/snappy_exact_stages.cpp", ("Snappy-source-C++",)),
    (primitive.LZ4_ACTIONS, "cpp/src/lz4_exact_stages.cpp", ("LZ4",)),
    (primitive.GZIP_CONTAINER_ACTIONS, "cpp/src/deflate_container_stages.cpp", ("gzip-container-only",)),
    (primitive.ZLIB_CONTAINER_ACTIONS, "cpp/src/deflate_container_stages.cpp", ("Zlib-container-only",)),
    (primitive.XZ_CONTAINER_ACTIONS, "cpp/src/xz_container_stage.cpp", ("XZ-container-only",)),
    (primitive.FPZIP_ACTIONS, "cpp/src/fpzip_exact_stages.cpp", ("FPZIP",)),
    (primitive.LECO_ACTIONS, "cpp/src/learning_codec_exact_stages.cpp", ("LECO",)),
    (primitive.BUFF_ACTIONS + primitive.BUFF_ATOMIC_ACTIONS[1:3], "cpp/src/learning_codec_exact_stages.cpp", ("BUFF-reference",)),
):
    _register_exact_stage_group(_names, _source, _methods)

for _names, _source, _methods in (
    (primitive.BOS64_LEGACY_INT_ACTIONS, "cpp/src/bos_exact_stages.cpp", ("BOS-frozen",)),
    (primitive.LECO_LEGACY_ACTIONS, "cpp/src/learning_codec_exact_stages.cpp", ("LECO-frozen",)),
    (tuple(action.name for action in primitive.REGER_LEGACY_REFERENCE_ACTIONS),
     "cpp/src/learning_codec_exact_stages.cpp", ("REGER-frozen",)),
    (primitive.SNAPPY_FUSED_ACTIONS, "cpp/src/snappy_exact_stages.cpp", ("Snappy-frozen",)),
):
    _register_exact_stage_group(_names, _source, _methods)

_register_reference_stage_group(
    tuple(action.name for action in primitive.ALP_REFERENCE_ACTIONS),
    "cpp/src/learning_codec_exact_stages.cpp",
    ("ALP-reference",),
)
_register_reference_stage_group(
    tuple(action.name for action in primitive.REGER_REFERENCE_ACTIONS),
    "cpp/src/learning_codec_exact_stages.cpp",
    ("REGER-reference",),
)
_register_exact_stage_group(
    tuple(action.name for action in primitive.LEARNING_CODEC_RESIDUAL_ACTIONS),
    "cpp/src/learning_codec_exact_stages.cpp",
    ("ALP", "REGER"),
)
_EXACT_STAGE_METHODS_BY_NAME[primitive.LEARNING_CODEC_RESIDUAL_ACTIONS[0].name] = ("ALP",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.LEARNING_CODEC_RESIDUAL_ACTIONS[1].name] = ("REGER",)
_register_reference_stage_group(
    tuple(action.name for action in primitive.GENERAL_CODEC_REFERENCE_ACTIONS),
    "cpp/src/general_codec_reference_stages.cpp",
    ("general-codec-reference",),
)
_register_exact_stage_group(
    tuple(action.name for action in primitive.GENERAL_CODEC_RESIDUAL_ACTIONS),
    "cpp/src/general_codec_residual_stages.cpp",
    ("Brotli", "Zstd", "Blosc", "Blosc2", "Bzip2", "gzip", "Zlib", "XZ"),
)
for _index, _method in enumerate(
    ("Brotli", "Zstd", "Blosc", "Blosc", "Blosc", "Blosc2", "Blosc2", "Blosc2", "Bzip2", "gzip", "Zlib", "XZ")
):
    _EXACT_STAGE_METHODS_BY_NAME[
        primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[_index].name
    ] = (_method,)
for _name in primitive.BROTLI_REFERENCE_ACTIONS[:2]:
    _EXACT_STAGE_METHODS_BY_NAME[_name] = ("Brotli-reference",)
for _name in primitive.ZSTD_REFERENCE_ACTIONS:
    _EXACT_STAGE_METHODS_BY_NAME[_name] = ("Zstd-reference",)
for _name in primitive._BLOSCLZ_SHARED_ACTIONS:
    _EXACT_STAGE_METHODS_BY_NAME[_name] = ("Blosc-reference", "Blosc2-reference")
for _name in primitive.BLOSC_SHUFFLE_ACTION_BY_KIND.values():
    _EXACT_STAGE_METHODS_BY_NAME[_name] = ("Blosc-reference", "Blosc2-reference")
_EXACT_STAGE_METHODS_BY_NAME[primitive.GENERAL_CODEC_REFERENCE_ACTIONS[12].name] = ("Blosc-reference",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.GENERAL_CODEC_REFERENCE_ACTIONS[13].name] = ("Blosc2-reference",)
for _name in primitive.BZIP2_REFERENCE_ACTIONS[:3]:
    _EXACT_STAGE_METHODS_BY_NAME[_name] = ("Bzip2-reference",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.GENERAL_CODEC_REFERENCE_ACTIONS[2].name] = (
    "Brotli-reference", "Bzip2-reference", "gzip-reference", "Zlib-reference"
)
_EXACT_STAGE_METHODS_BY_NAME[primitive.ZSTD_REFERENCE_ACTIONS[1]] = (
    "Zstd-reference", "Zlib-reference", "XZ-reference"
)
_EXACT_STAGE_METHODS_BY_NAME[primitive.DEFLATE_REFERENCE_LZ_ACTION] = (
    "gzip-reference", "Zlib-reference"
)
_EXACT_STAGE_METHODS_BY_NAME[primitive.XZ_REFERENCE_LZ_ACTION] = ("XZ-reference",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.REFERENCE_RANGE_ACTION] = ("XZ-reference",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.GZIP_REFERENCE_FRAME_ACTION] = ("gzip-reference",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.ZLIB_REFERENCE_FRAME_ACTION] = ("Zlib-reference",)
_EXACT_STAGE_METHODS_BY_NAME[primitive.XZ_REFERENCE_FRAME_ACTION] = ("XZ-reference",)

# The same exact action is reused across these source methods.
_EXACT_STAGE_METHODS_BY_NAME[primitive.DBP_EXACT_ACTIONS[0].name] = ("DBP", "SVB", "BOS")
_EXACT_STAGE_METHODS_BY_NAME[primitive.ELF_EXACT_ACTIONS[0].name] = ("Elf", "Elf+")


def _requested_stage_contract(name: str) -> tuple[tuple[DataKind, ...], tuple[DataKind, ...]]:
    if (
        name in primitive.GENERAL_CODEC_REFERENCE_ACTIONS_BY_NAME
        or name in primitive.GENERAL_CODEC_RESIDUAL_ACTIONS_BY_NAME
    ):
        return ("bytes",), ("bytes",)
    if name == primitive.LECO_ACTIONS[0]:
        return ("int64",), ("bytes", "int64")
    if name == primitive.LECO_LEGACY_PREDICTOR_ACTION.name:
        return ("int64",), ("bytes", "int64")
    if name in primitive.LECO_ACTIONS[1:]:
        return ("bytes",), ("bytes",)
    if name in {primitive.ALP_MAIN_ACTIONS[0], primitive.ALP_DICTIONARY_ACTIONS[0]}:
        return ("float64",), ("bytes", "float64")
    if name in primitive.ALP_MAIN_ACTIONS[1:]:
        return ("bytes",), ("bytes",)
    if name == primitive.REGER_ACTIONS[0]:
        return ("int64",), ("bytes", "int64")
    if name == primitive.REGER_LEGACY_PREDICTOR_ACTION.name:
        return ("bytes",), ("bytes",)
    if name in primitive.REGER_ACTIONS[1:]:
        return ("bytes",), ("bytes",)
    if name == primitive.BUFF_ACTIONS[0]:
        return ("float64",), ("bytes", "float64")
    if name in primitive.BUFF_ACTIONS[1:] + primitive.BUFF_ATOMIC_ACTIONS[1:3]:
        return ("bytes",), ("bytes",)
    if name == primitive.BOS_BOSA_ACTIONS[0]:
        return ("int64",), ("bytes", "int64")
    if name in primitive.BOS_BOSA_ACTIONS[1:]:
        return ("bytes",), ("bytes",)
    if name == primitive.BOS64_INT_ACTIONS[0]:
        return ("int64", "bytes"), ("bytes", "int64")
    if name == primitive.BOS64_LEGACY_DELTA_ACTION.name:
        return ("int64", "bytes"), ("bytes", "int64")
    if name in primitive.BOS64_INT_ACTIONS[1:]:
        return ("bytes",), ("bytes",)
    if name == primitive.ZIGZAG_RAW_ACTION.name:
        return ("int64",), ("bytes", "int64")
    if name == primitive.ULEB128_RAW_ACTION.name:
        return ("bytes",), ("bytes",)
    if name == primitive.ULEB128_DBP_ACTION.name:
        return ("bytes",), ("bytes",)
    if name == primitive.FASTPFOR_COMPOSITE_HELPER_ACTIONS[0].name:
        return ("int64", "bytes"), ("bytes", "int64")
    if name == primitive.FASTPFOR_INT_ACTIONS[0]:
        return ("int64", "bytes"), ("bytes", "int64")
    if name == primitive.FASTPFOR_INT_ACTIONS[1]:
        return ("bytes",), ("bytes",)
    if name == primitive.ELF_EXACT_ACTIONS[0].name:
        return ("float64",), ("bytes", "float64")
    if name in primitive.ELF_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name == primitive.CAMEL_ACTIONS[0]:
        return ("int64", "float64"), ("bytes", "int64", "float64")
    if name in primitive.CAMEL_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name in primitive.SNAPPY_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name in primitive.LZ4_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name in primitive.DEFLATE_CONTAINER_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name in primitive.XZ_CONTAINER_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name == primitive.FPZIP_ACTIONS[0]:
        return ("float64",), ("bytes", "float64")
    if name in primitive.FPZIP_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    if name == primitive.PONGO_ACTIONS[0]:
        return ("int64", "float64"), ("bytes", "int64", "float64")
    if name in primitive.PONGO_ACTIONS_BY_NAME:
        return ("bytes",), ("bytes",)
    raise KeyError(name)


def _load_registry(path: Path = CATALOG_PATH) -> tuple[ActionDescriptor, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 194:
        raise RuntimeError(f"expected 194 audited action records, got {len(rows)}")

    # The earlier ratio/name-only audit collapsed Gorilla into Chimp128 for two
    # operators.  Their forward state machines are not interchangeable, so a
    # callable registry must split those false deduplications back out.
    expanded_rows: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        aliases = _split_pipe(row["member_names"])
        gorilla_aliases = [name for name in aliases if name in primitive.GORILLA_ACTIONS]
        if gorilla_aliases and row["canonical_name"] not in primitive.GORILLA_ACTIONS:
            row["member_names"] = row["canonical_name"]
            row["source_methods"] = "Chimp128"
            row["implementation_ids"] = row["implementation_ids"].split("|")[0].strip()
            expanded_rows.append(row)
            clone = dict(source_row)
            clone["canonical_name"] = gorilla_aliases[0]
            clone["member_names"] = gorilla_aliases[0]
            clone["source_methods"] = "Gorilla"
            clone["implementation_ids"] = clone["implementation_ids"].split("|")[-1].strip()
            expanded_rows.append(clone)
        else:
            expanded_rows.append(row)
    rows = expanded_rows
    if len(rows) != 196:
        raise RuntimeError(f"expected 196 semantically distinct action records, got {len(rows)}")

    descriptors: list[ActionDescriptor] = []
    for action_id, row in enumerate(rows):
        domain = row["execution_domain"]
        name = row["canonical_name"]
        if name in primitive.RETIRED_IDENTITY_ACTION_NAMES:
            mode = "retired"
            inputs, outputs = ("int64",), ("int64",)
            reason = "identity Scaling is retained only for old-frame inverse compatibility"
        elif name in primitive.RETIRED_COMPOSITE_ACTION_NAMES:
            mode = "retired"
            try:
                inputs, outputs = _requested_stage_contract(name)
            except KeyError:
                inputs, outputs = ("int64",), ("bytes", "int64")
            reason = "historical composite/duplicate is retained only for old-frame inverse compatibility"
        elif name in primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES:
            mode = "experimental_non_atomic"
            inputs, outputs = _requested_stage_contract(name)
            reason = (
                "callable only for forensic wire replay; invokes a whole-codec "
                "core or extracts one and is excluded from the RL policy"
            )
        elif name in primitive.TS2DIFF_FLOAT_ACTIONS:
            mode = "stagewise"
            if name == primitive.TS2DIFF_FLOAT_ACTIONS[0]:
                inputs, outputs = ("float64",), ("bytes", "float64")
            elif name == primitive.TS2DIFF_FLOAT_ACTIONS[1]:
                inputs, outputs = ("int64", "bytes"), ("bytes", "int64")
            else:
                inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.SVB_INT_ACTIONS:
            mode = "stagewise"
            if name == primitive.SVB_INT_ACTIONS[0]:
                inputs, outputs = ("int64", "bytes"), ("bytes", "int64")
            else:
                inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.SIMPLE8B_FLOAT_ACTIONS:
            mode = "stagewise"
            if name == primitive.SIMPLE8B_FLOAT_ACTIONS[0]:
                inputs, outputs = ("float64",), ("bytes", "float64")
            else:
                inputs, outputs = ("int64", "bytes"), ("bytes", "int64")
            reason = ""
        elif name in primitive.P04_EXACT_ACTIONS_BY_NAME:
            mode = "stagewise"
            if name in {primitive.RLBE_ACTIONS[0], primitive.SPRINTZ_ACTIONS[0]}:
                inputs, outputs = ("int64", "float64"), ("bytes", "int64", "float64")
            else:
                inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.DBP_ACTIONS_BY_NAME:
            mode = "stagewise"
            if name == primitive.DBP_FLOAT_ACTIONS[0]:
                inputs, outputs = ("float64",), ("bytes", "float64")
            elif name == primitive.DBP_FLOAT_ACTIONS[1]:
                inputs, outputs = ("int64", "bytes"), ("bytes", "int64")
            else:
                inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.GORILLA_ACTIONS_BY_NAME:
            mode = "stagewise"
            if name == primitive.GORILLA_ACTIONS[0]:
                inputs, outputs = ("int64", "float64"), ("bytes", "int64", "float64")
            else:
                inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.CHIMP128_ACTIONS_BY_NAME:
            mode = "stagewise"
            if name == primitive.CHIMP128_ACTIONS[0]:
                inputs, outputs = ("int64", "float64"), ("bytes", "int64", "float64")
            else:
                inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.SPDP_ACTIONS_BY_NAME:
            mode = "stagewise"
            inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.FPC_ACTIONS_BY_NAME:
            mode = "stagewise"
            inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in primitive.PFPC_ACTIONS_BY_NAME:
            mode = "stagewise"
            inputs, outputs = ("bytes",), ("bytes",)
            reason = ""
        elif name in _EXACT_STAGE_SOURCE_BY_NAME or name in _REFERENCE_STAGE_SOURCE_BY_NAME:
            mode = "stagewise"
            inputs, outputs = _requested_stage_contract(name)
            reason = ""
        elif domain == "normalized_int64_reference":
            mode: ExecutionMode = "standalone"
            inputs: tuple[DataKind, ...] = ("int64",)
            outputs: tuple[DataKind, ...] = ("bytes", "int64")
            reason = ""
        elif domain == "exact_codec_byte_stage" and name in primitive.EXACT_ATOMIC_ACTION_BY_NAME:
            mode = "standalone"
            inputs = _atomic_input_kinds(name)
            outputs = ("bytes", *inputs)
            reason = ""
        elif name in primitive.EXACT_NATIVE_ACTION_BY_NAME:
            mode = "stagewise"
            inputs = ("bytes",)
            outputs = ("bytes",)
            reason = ""
        elif domain == "exact_codec_byte_stage":
            mode = "closed_sequence_only"
            inputs = ("bytes",)
            outputs = ("bytes",)
            reason = "requires its complete registered exact action sequence"
        elif domain == "method_scoped_cpp_candidate":
            mode = "descriptor_only"
            inputs = ()
            outputs = ()
            reason = (
                "method-scoped C++ ablation candidate has no independent "
                "forward/inverse stage implementation"
            )
        else:
            raise RuntimeError(f"unknown execution domain {domain!r} for {name}")

        registered = row["registered_action_id"].strip()
        descriptors.append(
            ActionDescriptor(
                action_id=action_id,
                inventory_index=int(row["implementation_index"]),
                name=name,
                operator_id=row["operator_id"],
                execution_mode=mode,
                accepted_input_kinds=inputs,
                output_kinds=outputs,
                implementation_source=(
                    (
                        "scripts.primitive_operator_action_runtime."
                        "encode_exact_native_lz_stage/decode_exact_native_lz_stage; "
                        f"kernel={row['implementation_sources']}"
                    )
                    if name in primitive.EXACT_NATIVE_ACTION_BY_NAME
                    else _EXACT_STAGE_SOURCE_BY_NAME.get(
                        name,
                        _REFERENCE_STAGE_SOURCE_BY_NAME.get(
                            name,
                            _EXPERIMENTAL_STAGE_SOURCE_BY_NAME.get(
                                name, row["implementation_sources"]
                            ),
                        ),
                    )
                ),
                evidence=(
                    "exact_stage_cpp"
                    if name in _EXACT_STAGE_SOURCE_BY_NAME
                    else (
                        "reference_stage_cpp"
                        if name in _REFERENCE_STAGE_SOURCE_BY_NAME
                        else (
                            "non_atomic_experimental"
                            if name in _EXPERIMENTAL_STAGE_SOURCE_BY_NAME
                            else row["evidence"]
                        )
                    )
                ),
                source_methods=_EXACT_STAGE_METHODS_BY_NAME.get(
                    name, _split_pipe(row["source_methods"])
                ),
                legacy_registered_action_id=int(registered) if registered else None,
                unavailable_reason=reason,
            )
        )

    append_specs = [
        primitive.ALL_CONCRETE_ACTION_BY_NAME[primitive.RLE_SCALING_ACTION],
        *(action for action in (
            *primitive.BOS_BOSA_EXACT_ACTIONS,
            *primitive.BOS64_EXACT_ACTIONS,
            primitive.BOS64_LEGACY_DELTA_ACTION,
            primitive.ZIGZAG_RAW_ACTION,
            primitive.ULEB128_RAW_ACTION,
            primitive.ULEB128_DBP_ACTION,
            *primitive.FASTPFOR_COMPOSITE_HELPER_ACTIONS,
            *primitive.FASTPFOR_EXACT_ACTIONS,
            *primitive.ELF_EXACT_ACTIONS,
            *primitive.CAMEL_EXACT_ACTIONS,
            *primitive.LECO_SOURCE_EXACT_ACTIONS,
            primitive.LECO_LEGACY_PREDICTOR_ACTION,
            *primitive.ALP_REFERENCE_ACTIONS,
            *primitive.REGER_REFERENCE_ACTIONS,
            primitive.REGER_LEGACY_PREDICTOR_ACTION,
            *primitive.LEARNING_CODEC_RESIDUAL_ACTIONS,
            *primitive.BUFF_SOURCE_EXACT_ACTIONS,
            *primitive.BUFF_CURRENT_EXACT_ACTIONS,
            *primitive.GENERAL_CODEC_REFERENCE_ACTIONS,
            *primitive.GENERAL_CODEC_RESIDUAL_ACTIONS,
            *primitive.SNAPPY_EXACT_ACTIONS,
            primitive.SNAPPY_LEGACY_INTEGER_ACTION,
            *primitive.LZ4_EXACT_ACTIONS,
        ) if action.name not in {item.name for item in descriptors}),
    ]
    for spec in append_specs:
        if spec.name == primitive.RLE_SCALING_ACTION:
            inputs, outputs = ("float64",), ("bytes", "float64")
            methods = ("RLE",)
            source = "cpp/src/rle_exact_stages.cpp"
        elif spec.name in primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES:
            inputs, outputs = _requested_stage_contract(spec.name)
            methods = _EXACT_STAGE_METHODS_BY_NAME.get(spec.name, ())
            source = spec.implementation_source
        elif spec.name in _REFERENCE_STAGE_SOURCE_BY_NAME:
            inputs, outputs = _requested_stage_contract(spec.name)
            methods = _EXACT_STAGE_METHODS_BY_NAME[spec.name]
            source = _REFERENCE_STAGE_SOURCE_BY_NAME[spec.name]
        else:
            inputs, outputs = _requested_stage_contract(spec.name)
            methods = _EXACT_STAGE_METHODS_BY_NAME[spec.name]
            source = _EXACT_STAGE_SOURCE_BY_NAME[spec.name]
        descriptors.append(
            ActionDescriptor(
                action_id=len(descriptors), inventory_index=-1,
                name=spec.name, operator_id=spec.operator_id,
                execution_mode=(
                    "retired" if spec.name in primitive.RETIRED_COMPOSITE_ACTION_NAMES
                    else "experimental_non_atomic"
                    if spec.name in primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES
                    else "stagewise"
                ), accepted_input_kinds=inputs,
                output_kinds=outputs, implementation_source=source,
                evidence=(
                    "non_atomic_experimental"
                    if spec.name in primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES
                    else (
                        "reference_stage_cpp_added_after_historical_194_inventory"
                        if spec.name in _REFERENCE_STAGE_SOURCE_BY_NAME
                        else "exact_stage_cpp_added_after_historical_194_inventory"
                    )
                ),
                source_methods=methods,
                legacy_registered_action_id=spec.action_id,
                unavailable_reason=(
                    "historical composite/duplicate retained only for old-frame inverse"
                    if spec.name in primitive.RETIRED_COMPOSITE_ACTION_NAMES
                    else "whole-codec extractor or combined helper excluded from policy"
                    if spec.name in primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES
                    else ""
                ),
            )
        )

    for spec in primitive.LEGACY_COMPATIBILITY_ACTIONS:
        numeric = primitive.legacy_numeric_stages
        inputs = (("int64", "float64") if spec.name in (numeric.SCALE, primitive.REGER_LEGACY_ACTIONS[0], primitive.BUFF_NATIVE_ACTIONS[0]) else
                  ("float64",) if spec.name == numeric.ALP_SCALE else ("bytes",))
        methods = (("Snappy",) if spec.name == primitive.SNAPPY_LEGACY_MATCH_ACTION else
                   ("REGER",) if spec.name in primitive.REGER_LEGACY_ACTIONS else
                   ("BUFF",) if spec.name in primitive.BUFF_NATIVE_ACTIONS else
                   ("Blosc2",) if spec.name in primitive.BLOSC2_LEGACY_ACTIONS else
                   ("FastPFor", "LECO", "ALP"))
        descriptors.append(ActionDescriptor(
            action_id=len(descriptors), inventory_index=-1, name=spec.name,
            operator_id=spec.operator_id, execution_mode="stagewise",
            accepted_input_kinds=inputs, output_kinds=tuple(dict.fromkeys(("bytes", *inputs))),
            implementation_source=spec.implementation_source,
            evidence=("version_pinned_non_atomic_binding_macro" if spec.name == primitive.BLOSC2_LEGACY_ACTIONS[0] else
                      "current_cpp_native_stage" if spec.name in primitive.BUFF_NATIVE_ACTIONS else
                      "legacy_benchmark_independent_stage"),
            source_methods=methods,
            legacy_registered_action_id=primitive.ALL_CONCRETE_ACTION_BY_NAME[spec.name].action_id,
        ))

    for spec in primitive.LOSSLESS64_ACTIONS:
        inputs = (("float64",) if spec.name == primitive.ELF64_ERASER_ACTION else
                  ("int64", "float64") if spec.name == primitive.CORAD_LOSSLESS_ACTIONS[0] else ("bytes",))
        descriptors.append(ActionDescriptor(
            action_id=len(descriptors), inventory_index=-1, name=spec.name,
            operator_id=spec.operator_id, execution_mode="stagewise",
            accepted_input_kinds=inputs, output_kinds=tuple(dict.fromkeys(("bytes", *inputs))),
            implementation_source=spec.implementation_source,
            evidence="versioned_lossless64_stage",
            source_methods=(("CORAD",) if spec.name in primitive.CORAD_LOSSLESS_ACTIONS else
                            ("Elf", "Elf+") if spec.name == primitive.ELF64_ERASER_ACTION else
                            ("Elf",) if spec.name == primitive.ELF64_ACTIONS[1] else ("Elf+",)),
            legacy_registered_action_id=primitive.ALL_CONCRETE_ACTION_BY_NAME[spec.name].action_id,
        ))

    if len({item.name for item in descriptors}) != len(descriptors):
        raise RuntimeError("the corrected action catalog contains duplicate canonical names")
    return tuple(descriptors)


ACTIONS: tuple[ActionDescriptor, ...] = _load_registry()
ACTION_BY_NAME = {action.name: action for action in ACTIONS}
COMPATIBILITY_ACTIONS = ACTIONS  # Historical descriptor IDs, not implementation count.
LEGACY_IO_VARIANT_ACTIONS = tuple(
    replace(
        ACTION_BY_NAME[primitive.LEGACY_206_CANONICAL_TO_LEGACY_NAME[spec.name]],
        name=spec.name, action_id=spec.action_id,
        implementation_source=spec.implementation_source,
        accepted_input_kinds=(("int64", "float64")
                              if spec.name == primitive.cpp_migration.DICTIONARY
                              else ACTION_BY_NAME[primitive.LEGACY_206_CANONICAL_TO_LEGACY_NAME[spec.name]].accepted_input_kinds),
        output_kinds=(("bytes", "int64", "float64")
                      if spec.name == primitive.cpp_migration.DICTIONARY
                      else ACTION_BY_NAME[primitive.LEGACY_206_CANONICAL_TO_LEGACY_NAME[spec.name]].output_kinds),
    )
    for spec in primitive.LEGACY_206_POLICY_ACTIONS
)
_LEGACY_IO_VARIANT_BY_NAME = {item.name: item for item in LEGACY_IO_VARIANT_ACTIONS}
_PARAMETER_DESCRIPTORS = {
    module.ALGORITHM_NAME: replace(
        next(item for item in LEGACY_IO_VARIANT_ACTIONS if module.parameters_from_name(item.name) is not None),
        name=module.ALGORITHM_NAME)
    for module in action_parameters.MODULES
}
_PLAIN_BITPACK_DESCRIPTOR = replace(
    _LEGACY_IO_VARIANT_BY_NAME[bitpack.SOURCE_VARIANTS[0]],
    name=bitpack.PLAIN_VARIANT, execution_mode="standalone",
    accepted_input_kinds=("int64", "bytes"), output_kinds=("bytes", "int64"),
    implementation_source=bitpack.SOURCE,
    evidence="One max(input)-width kernel; nonnegative int64 or raw uint64 LE; self-describing MVW1 output",
    source_methods=("TS_2DIFF", "LECO", "ALP", "REGER"), legacy_registered_action_id=None,
)
_PLAIN_FOR_DESCRIPTOR = replace(
    _LEGACY_IO_VARIANT_BY_NAME[minimum.SOURCE_VARIANTS[0]],
    name=minimum.PLAIN_VARIANT, execution_mode="standalone",
    accepted_input_kinds=("int64", "bytes"), output_kinds=("bytes", "int64"),
    implementation_source=minimum.SOURCE,
    evidence="Shared C++ minimum + uint64 offsets; signed int64 or raw uint64 LE, two typed output streams",
    source_methods=("reference", "TS_2DIFF", "BOS", "ALP", "REGER"), legacy_registered_action_id=None,
)
_RLE_DESCRIPTORS = {
    variant: replace(_LEGACY_IO_VARIANT_BY_NAME['op.run_length.reference_v1'],
        name=variant, execution_mode='standalone' if variant in (optimal_rle.WORDS_VARIANT, optimal_rle.BYTES_VARIANT) else 'stagewise',
        accepted_input_kinds=('int64', 'float64', 'bytes') if variant == optimal_rle.WORDS_VARIANT else ('bytes',),
        output_kinds=('bytes', 'int64', 'float64') if variant == optimal_rle.WORDS_VARIANT else ('bytes',),
        implementation_source=optimal_rle.SOURCE,
        evidence='Current URL1 byte-optimal core; minimum complete bytes only within declared native grammar',
        source_methods=('RLE', 'RLBE', 'Bzip2'), legacy_registered_action_id=None)
    for variant in optimal_rle.VARIANTS
}
IO_VARIANT_ACTIONS = tuple(
    replace(_LEGACY_IO_VARIANT_BY_NAME[optimal_rle.OLD_SPRINTZ_VARIANT], name=name,
            evidence='Current exact Sprintz packer; integer tail uses shared URL1 RLE',
            implementation_source='cpp/src/p04_exact_stages.cpp') if name == optimal_rle.SPRINTZ_VARIANT else
    _RLE_DESCRIPTORS[name] if name in _RLE_DESCRIPTORS else
    _PLAIN_FOR_DESCRIPTOR if name == minimum.PLAIN_VARIANT else
    _PLAIN_BITPACK_DESCRIPTOR if name == bitpack.PLAIN_VARIANT else
    _PARAMETER_DESCRIPTORS[name] if name in _PARAMETER_DESCRIPTORS else
    ACTION_BY_NAME[name] if name in ACTION_BY_NAME and name not in _LEGACY_IO_VARIANT_BY_NAME else
    replace(_LEGACY_IO_VARIANT_BY_NAME[primitive.algorithm_io.VARIANT_BY_NAME[name]["legacy_names"][0]], name=name)
    if name not in _LEGACY_IO_VARIANT_BY_NAME else _LEGACY_IO_VARIANT_BY_NAME[name]
    for name in dict.fromkeys(name for variants in primitive.POLICY_IO_VARIANTS.values() for name in variants)
)
IO_VARIANT_ACTION_BY_NAME = {item.name: item for item in IO_VARIANT_ACTIONS}
CANONICAL_ACTIONS = tuple(
    replace(
        IO_VARIANT_ACTION_BY_NAME[primitive.POLICY_IO_VARIANTS[spec.name][0]],
        name=spec.name, action_id=spec.action_id, operator_id=spec.operator_id,
        implementation_source=spec.implementation_source,
        source_methods=tuple(dict.fromkeys(method for variant in primitive.POLICY_IO_VARIANTS[spec.name]
                             for method in IO_VARIANT_ACTION_BY_NAME[variant].source_methods)),
        accepted_input_kinds=tuple(dict.fromkeys(
            kind for variant in primitive.POLICY_IO_VARIANTS[spec.name]
            for kind in IO_VARIANT_ACTION_BY_NAME[variant].accepted_input_kinds)),
        output_kinds=tuple(dict.fromkeys(
            kind for variant in primitive.POLICY_IO_VARIANTS[spec.name]
            for kind in IO_VARIANT_ACTION_BY_NAME[variant].output_kinds)),
    ) for spec in primitive.POLICY_ACTIONS
)
CANONICAL_ACTION_BY_NAME = {item.name: item for item in CANONICAL_ACTIONS}
# Name aliases remain readable for old UAR1 packets. Only CANONICAL_ACTIONS
# defines the new policy's selectable implementations.
ACTION_BY_NAME.update(IO_VARIANT_ACTION_BY_NAME)
ACTION_BY_NAME.update(_LEGACY_IO_VARIANT_BY_NAME)
ACTION_BY_NAME.update(CANONICAL_ACTION_BY_NAME)
for _old_algorithm, _current_algorithm in primitive.algorithm_io.PREVIOUS_ALGORITHM_TO_CURRENT.items():
    if _current_algorithm in CANONICAL_ACTION_BY_NAME:
        ACTION_BY_NAME.setdefault(_old_algorithm, CANONICAL_ACTION_BY_NAME[_current_algorithm])

_INDEPENDENT_STAGEWISE_SEQUENCES = frozenset(
    {
        primitive.legacy_numeric_stages.SCALED_ACTIONS,
        primitive.legacy_numeric_stages.ALP_ACTIONS,
        primitive.SNAPPY_LEGACY_ACTIONS,
        primitive.REGER_LEGACY_ACTIONS,
        primitive.BUFF_NATIVE_ACTIONS,
        primitive.BLOSC2_LEGACY_ACTIONS,
        primitive.ELF64_ACTIONS,
        primitive.ELF64_PLUS_ACTIONS,
        primitive.CORAD_LOSSLESS_ACTIONS,
        primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS,
        primitive.TS2DIFF_INT_ACTIONS,
        primitive.TS2DIFF_FLOAT_ACTIONS,
        primitive.SVB_INT_ACTIONS,
        primitive.SVB_FLOAT_ACTIONS,
        primitive.SIMPLE8B_FLOAT_ACTIONS,
        primitive.RLBE_ACTIONS,
        primitive.SPRINTZ_ACTIONS,
        primitive.REFERENCE_INTEGER_ACTIONS,
        primitive.DBP_INT_ACTIONS,
        primitive.DBP_FLOAT_ACTIONS,
        primitive.GORILLA_ACTIONS,
        primitive.CHIMP128_ACTIONS,
        primitive.PONGO_ACTIONS,
        primitive.SPDP_ACTIONS,
        primitive.FPC_ACTIONS,
        primitive.PFPC_ACTIONS,
        primitive.BOS_BOSA_ACTIONS,
        primitive.BOS64_INT_ACTIONS,
        primitive.BOS64_FLOAT_ACTIONS,
        primitive.FASTPFOR_INT_ACTIONS,
        primitive.FASTPFOR_FLOAT_ACTIONS,
        primitive.ELF_ACTIONS,
        primitive.ELF_PLUS_ACTIONS,
        primitive.CAMEL_ACTIONS,
        primitive.SNAPPY_ACTIONS,
        primitive.LZ4_ACTIONS,
        primitive.FPZIP_ACTIONS,
        primitive.LECO_ACTIONS,
        primitive.ALP_MAIN_ACTIONS,
        primitive.ALP_DICTIONARY_ACTIONS,
        primitive.REGER_ACTIONS,
        primitive.BUFF_ATOMIC_ACTIONS,
        primitive.BROTLI_REFERENCE_ACTIONS,
        primitive.ZSTD_REFERENCE_ACTIONS,
        primitive.BZIP2_REFERENCE_ACTIONS,
        primitive.GZIP_REFERENCE_ACTIONS,
        primitive.ZLIB_REFERENCE_ACTIONS,
        primitive.XZ_REFERENCE_ACTIONS,
        primitive.BROTLI_EXACT_ACTIONS,
        primitive.ZSTD_EXACT_ACTIONS,
        primitive.BZIP2_EXACT_ACTIONS,
        primitive.GZIP_EXACT_ACTIONS,
        primitive.ZLIB_EXACT_ACTIONS,
        primitive.XZ_EXACT_ACTIONS,
        *primitive.BLOSC_REFERENCE_ACTIONS_BY_KIND.values(),
        *primitive.BLOSC2_REFERENCE_ACTIONS_BY_KIND.values(),
        *primitive.BLOSC_EXACT_ACTIONS_BY_KIND.values(),
        *primitive.BLOSC2_EXACT_ACTIONS_BY_KIND.values(),
        *(
            tuple(
                action.name
                for action in primitive.exact_native_lz_action_pipeline(algorithm_key)
            )
            for algorithm_key in primitive.EXACT_NATIVE_METHOD_KEYS.values()
        ),
    }
)


# Normalize historical implementation aliases to their concrete I/O profiles.
# Algorithm names alone discard source adapters (e.g. REGER FOR -> reference
# FOR or LECO bitpack -> TS_2DIFF) and create non-executable grammar edges.
_INDEPENDENT_STAGEWISE_SEQUENCES = _INDEPENDENT_STAGEWISE_SEQUENCES | frozenset(
    tuple(primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(name, name) for name in sequence)
    for sequence in _INDEPENDENT_STAGEWISE_SEQUENCES
)


def describe_action(action: ActionRef) -> ActionDescriptor:
    if type(action) is int:
        if 0 <= action < len(ACTIONS):
            return ACTIONS[action]
        raise KeyError(f"unknown action id {action}")
    try:
        return ACTION_BY_NAME[action]
    except KeyError as exc:
        raise KeyError(f"unknown action {action!r}") from exc


def registry() -> dict[str, Any]:
    from action_semantic_ports import supported_layouts
    ports = supported_layouts()
    modes = Counter(action.execution_mode for action in CANONICAL_ACTIONS)
    return {
        "schema": "unified-typed-action-runtime-v7",
        "graph_extension": {
            "registry_version": "operator-taxonomy-30-v7",
            "registry_entrypoint": "unified_action_runtime.graph_registry",
            "execution_entrypoint": "unified_action_runtime.run_graph_action",
            "probability_entrypoint": "unified_action_runtime.prediction_model",
            "note": "Legacy single-stream IDs stay frozen; use the versioned graph namespace for 30 families and 3 special actions.",
        },
        "policy_registry_version": primitive.ACTION_REGISTRY_VERSION,
        "data_kinds": ["int64", "float64", "bytes"],
        "semantic_stream_kinds": ["int64", "uint64", "uint32", "uint8", "float64", "bytes"],
        "io_variant_count": len(IO_VARIANT_ACTIONS),
        "parameter_schemas": {**action_parameters.schemas(), primitive.BUFF_PLANE_ACTION.name: {"bit_width":{"type":"integer","minimum":0,"maximum":64}, "default_selection":"maximum offset width chosen before invoking the pure split", "restoration":"selected width is recorded; inverse never recomputes it"}},
        "forward_wire_kind": "bytes",
        "output_kinds_note": "descriptor output_kinds includes inverse restoration kinds; use semantic_ports for forward stream types",
        "semantic_ports": ports,
        "active_action_count": len(CANONICAL_ACTIONS),
        "frozen_policy_id_slots": len(primitive.FROZEN_POLICY_ACTIONS),
        "policy_action_slots_including_stop": primitive.STOP_ACTION_ID + 1,
        "policy_stop_action_id": primitive.STOP_ACTION_ID,
        "retired_policy_id_slots": sorted(
            set(primitive.FROZEN_POLICY_ACTION_BY_ID) - set(primitive.POLICY_ACTION_BY_ID)
        ),
        "implementation_directory": {
            "registry_version": primitive.implementation_catalog.REGISTRY_VERSION,
            "implementation_count": primitive.implementation_catalog.IMPLEMENTATION_COUNT,
            "count_is_runtime_executability": False,
            "runtime_action_recipes": primitive.implementation_catalog.inventory()["runtime_action_recipes"],
        },
        "compatibility_descriptor_count": len(COMPATIBILITY_ACTIONS),
        "compatibility_note": "old numeric descriptor IDs and action names are replay selectors, not additional implementations",
        "aliases": primitive.ACTION_ALIASES,
        "execution_modes": dict(sorted(modes.items())),
        "strict_boundary": (
            "descriptor_only records never fall back to another implementation"
        ),
        "actions": [{**asdict(action), "io_variants": [
            {**asdict(IO_VARIANT_ACTION_BY_NAME[variant]),
             "io_description": (
                 primitive.algorithm_io.VARIANT_BY_NAME[variant]["description"]
                 if variant in primitive.algorithm_io.VARIANT_BY_NAME
                 else primitive.ALL_CONCRETE_ACTION_BY_NAME[variant].implementation_scope
             ),
             "semantic_output": ports["document_merged_variant_layouts"].get(variant,
                 {"exposure": "inspect_actual_output", "note": "to_ports reports the actual schema; unknown layouts stay bytes"})}
            for variant in primitive.POLICY_IO_VARIANTS[action.name]]}
            for action in CANONICAL_ACTIONS],
    }


def _graph_taxonomy(registry_version: str):
    if registry_version == "operator-taxonomy-30-v1":
        import operator_taxonomy_v30 as taxonomy
    elif registry_version == "operator-taxonomy-30-v2":
        import operator_taxonomy_v30_v2 as taxonomy
    elif registry_version == "operator-taxonomy-30-v3":
        import operator_taxonomy_v30_v3 as taxonomy
    elif registry_version == "operator-taxonomy-30-v4":
        import operator_taxonomy_v30_v4 as taxonomy
    elif registry_version == "operator-taxonomy-30-v5":
        import operator_taxonomy_v30_v5 as taxonomy
    elif registry_version == "operator-taxonomy-30-v6":
        import operator_taxonomy_v30_v6 as taxonomy
    elif registry_version == "operator-taxonomy-30-v7":
        import operator_taxonomy_v30_v7 as taxonomy
    else:
        raise ActionRuntimeError("unknown graph registry version")
    return taxonomy


def graph_registry(registry_version: str = "operator-taxonomy-30-v7") -> dict[str, Any]:
    """30-family catalog, with probability-only availability kept explicit."""
    taxonomy = _graph_taxonomy(registry_version)
    import typed_graph_runtime as graph
    import prediction_probability as prediction
    import os
    layout_ready = Path(os.environ.get("OPENZL_LAYOUT_STAGES_CLI",
        ROOT / "cpp/build/openzl_layout_stages_cli")).is_file()
    capabilities = {a.name: layout_ready for a in taxonomy.CODEC_ACTIONS
                    if a.operator_id == "split_columns"}
    capabilities.update(concat=layout_ready, split_segments=layout_ready)
    prediction_ready = Path(os.environ.get("WEB_COMPRESSION_PREDICTION_DRIVER",
                                           prediction.DEFAULT_DRIVER)).is_file()
    from prediction_entropy_codec import availability as entropy_availability
    entropy_state = entropy_availability()
    for descriptor in taxonomy.MODEL_PROVIDERS:
        provider = descriptor.mechanism.lower()
        implementation = prediction.PROVIDERS["ppmd" if provider == "ppm" else provider]
        if provider == "nncp":
            from nncp_entropy_codec import availability as nncp_entropy_availability
            nncp_state = nncp_entropy_availability()
            available = bool(nncp_state["standalone"].get("available"))
            capabilities[descriptor.name] = {
                "available": available,
                "selectable": bool(nncp_state["graph_eligible"]),
                "status": implementation["status"],
                "reason": "Complete NNP1 terminal entropy path: " + nncp_state["reason"],
            }
            continue
        else:
            available = prediction_ready
        entropy_provider = "ppmd" if provider == "ppm" else provider
        entropy = entropy_state["providers"][entropy_provider]
        capabilities[descriptor.name] = {
            "available": available and entropy_state["binary_exists"],
            "selectable": entropy["graph_eligible"],
            "status": ("matched_entropy_cold_decode" if entropy["graph_eligible"]
                       else "probability_provider_entropy_path_unvalidated"),
            "reason": ("PEC1 connects causal prediction, arithmetic coding, complete framing and cold decode."
                       if entropy["graph_eligible"] else
                       "Probability API exists, but the current PEC1 binary lacks a matching cold-decode receipt."),
        }
    result = taxonomy.registry(capabilities)
    result["execution_boundary"] = {
        "graph_frame": "typed-graph-frame-v1", "stream_kinds": graph.KINDS,
        "prediction_api": "prediction_model(...): PPM/DMC/PAQ/cmix predict()->P(1), observe(bit); NNCP predict()->float32[256], observe(byte)",
        "matched_entropy_api": (
            "validated PEC1 PPMd/DMC/PAQ graph actions use causal prediction, fixed probability "
            "quantization, arithmetic coding and cold independent decode; A110 uses its distinct "
            "complete NNP1 terminal entropy frame when its graph receipt is current"
        ),
        "probability_traces_are_compressed_data": False,
        "existing_linear_training_environments_extended": False,
        "note": (
            "Executable graph API; full cmix remains unselectable until its explicit large-memory "
            "cold-decode probe passes. A110 is not relabeled as an atomic predictor edge: its graph "
            "action is the complete NNP1 path. Legacy trained policies/checkpoints are not resized "
            "or reinterpreted."
        ),
    }
    return result


def run_graph_action(state, action: ActionRef, input_stream_ids=(), *,
                     registry_version: str, parameters=None, input_view="bytes",
                     io_variant=None):
    """Versioned graph edge: select actual live columns/segments explicitly."""
    taxonomy = _graph_taxonomy(registry_version)
    from typed_graph_runtime import GraphState
    if not isinstance(state, GraphState):
        raise ActionTypeError("expected a GraphState")
    # A caller may begin a graph in either frozen version; never relabel an
    # already executed graph whose numeric step IDs belong to another version.
    if state.registry_version != registry_version:
        if state.steps:
            raise ActionRuntimeError("graph state registry mismatch")
        state = replace(state, registry_version=registry_version)
    descriptor = taxonomy.describe_action(action)
    if descriptor.kind == "retired":
        raise ActionNotExecutableError(
            "retired graph stage; migrate the old selector to its explicit replacement recipe"
        )
    if descriptor.kind == "model_provider":
        from nncp_entropy_codec import ACTION as NNCP_ENTROPY_ACTION
        from prediction_entropy_codec import ACTION_TO_PROVIDER
        if descriptor.name != NNCP_ENTROPY_ACTION and descriptor.name not in ACTION_TO_PROVIDER:
            raise ActionNotExecutableError(
                "This probability provider has no typed-graph entropy integration")
        if descriptor.name == NNCP_ENTROPY_ACTION:
            from build_nncp_probability import nncp_availability
            capability = nncp_availability()
            if not capability.get("available"):
                raise ActionNotExecutableError(capability.get(
                    "reason", "the complete NNP1 path is unavailable"
                ))
        if io_variant is not None:
            raise ActionTypeError("prediction entropy actions do not use legacy I/O profiles")
        return state.apply(descriptor.name, input_stream_ids, parameters=parameters)
    if descriptor.name == "stop":
        if input_stream_ids or parameters or io_variant is not None:
            raise ActionTypeError("STOP has no stream selection or codec parameters")
        return state.stop()
    if descriptor.operator_id == "split_columns" or descriptor.kind == "special":
        if io_variant is not None:
            raise ActionTypeError("graph layout variants use explicit parameters, not legacy I/O profiles")
        p = dict(parameters or {})
        if action == "split_by_range":
            if p.get("mode", "range") not in ("range", "split_by_range"):
                raise ActionTypeError("SplitByRange alias conflicts with segmentation mode")
            p["mode"] = "range"
        if descriptor.name == "split_segments" and p.get("mode") == "split_by_range":
            p["mode"] = "range"
        return state.apply(descriptor.mechanism, input_stream_ids, parameters=p)
    return state.apply(action if isinstance(action, str) else descriptor.name,
                       input_stream_ids, parameters=parameters, input_view=input_view,
                       io_variant=io_variant)


def prediction_model(action: ActionRef, *, registry_version: str, **options):
    """Select a genuine causal probability source, never a substitute codec."""
    taxonomy = _graph_taxonomy(registry_version)
    from prediction_probability import PredictionSession, NNCPSymbolSession
    descriptor = taxonomy.describe_action(action)
    if descriptor.kind != "model_provider":
        raise ActionTypeError("expected a prediction_probability operator")
    provider = descriptor.mechanism.lower()
    if provider == "nncp":
        return NNCPSymbolSession(**options)
    return PredictionSession("ppmd" if provider == "ppm" else provider, **options)


def describe_policy_action(action: ActionRef, io_variant: str | None = None) -> ActionDescriptor:
    """Describe an active action in the preserved numeric slot namespace."""
    if type(action) is int:
        spec = primitive.POLICY_ACTION_BY_ID.get(action)
        if spec is None:
            raise ActionRuntimeError("unknown policy action or STOP used as codec")
        name = spec.name
    else:
        name = primitive.ACTION_CANONICAL_NAMES.get(action, action)
        parameter_module = action_parameters.module_for_name(action)
        if parameter_module is not None:
            name = parameter_module.ALGORITHM_NAME
    if name not in CANONICAL_ACTION_BY_NAME:
        raise ActionNotExecutableError("not a selectable canonical policy implementation")
    if io_variant is not None:
        variant = _normalize_io_variant(io_variant)
        if variant not in primitive.POLICY_IO_VARIANTS[name]:
            raise ActionTypeError(f"I/O variant {io_variant!r} does not belong to {name}")
        return IO_VARIANT_ACTION_BY_NAME[variant]
    return CANONICAL_ACTION_BY_NAME[name]


def policy_io_variants(action: ActionRef) -> tuple[ActionDescriptor, ...]:
    descriptor = describe_policy_action(action)
    return tuple(IO_VARIANT_ACTION_BY_NAME[name]
                 for name in primitive.POLICY_IO_VARIANTS[descriptor.name])


def _normalize_io_variant(name: str) -> str:
    if not isinstance(name, str):
        raise ActionTypeError("I/O variant must be a registered name")
    parameter_module = action_parameters.module_for_name(name)
    variant = (action_parameters.canonical_io_variant(name) if parameter_module else
               primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(name, name))
    variant = primitive.algorithm_io.LEGACY_VARIANT_TO_CURRENT.get(variant, variant)
    if variant not in IO_VARIANT_ACTION_BY_NAME:
        raise ActionTypeError(f"unknown I/O variant: {name!r}")
    return variant


def _wire_data(data: Any) -> ActionData:
    if isinstance(data, ActionData):
        return data
    from action_semantic_ports import ActionPorts, from_ports
    if isinstance(data, ActionPorts):
        return from_ports(data)
    raise ActionTypeError("expected ActionData or ActionPorts")


# Input schema predicates only; this resolver never tries compression to pick
# the smallest result.  Wildcard byte layouts remain ambiguous without an
# explicit I/O profile (e.g. shuffle width or desired container magic).
_VARIANT_PACKET_MAGICS = {
    112: (b"BDL1",), 146: (b"RAR1",), 84: (b"DBS1",), 94: (b"DBS1",),
    91: (b"SPI1",), 95: (b"DBD1",), 102: (b"PER1",),
    98: (b"GXI1", b"GXF1"), 103: (b"PXR1",), 108: (b"FPR1",), 110: (b"PFR1",),
    59: (b"NLS1\x03",), 188: (b"LNS1",),
    133: (b"DFC1G",), 134: (b"DFC1Z",), 135: (b"XZC1",), 205: (b"CZL2",),
}


def resolve_policy_io_variant(action: ActionRef, data: Any,
                              io_variant: str | None = None) -> str:
    """Resolve an explicit profile or the sole compatible input schema.

    Different output formats with identical input kinds cannot be inferred.
    In that case the caller must select one of policy_io_variants(action).
    """
    algorithm = describe_policy_action(action).name
    if io_variant is None and action == minimum.PREVIOUS_WHOLE_NAME:
        old_source, old_upstream = _unwrap_stage_source(_wire_data(data))
        if old_source.kind == "int64":
            return minimum.SOURCE_VARIANTS[0]
        if old_source.kind == "bytes":
            for variant in minimum.SOURCE_VARIANTS[1:3]:
                magic = minimum.PRODUCER_VARIANTS[variant][1]
                if old_source.payload.startswith(magic): return variant
        raise ActionTypeError("historical whole-stream FOR requires its reference/BOS/REGER I/O profile")
    if io_variant is None and isinstance(action, str) and action not in CANONICAL_ACTION_BY_NAME:
        io_variant = action
    if io_variant is not None:
        variant = _normalize_io_variant(io_variant)
        if variant not in primitive.POLICY_IO_VARIANTS[algorithm]:
            raise ActionTypeError(f"I/O variant {variant!r} does not belong to {algorithm}")
        return variant
    source = _wire_data(data)
    original_source = source
    try:
        source, upstream = _unwrap_stage_source(source)
    except (ActionRuntimeError, ValueError, TypeError, KeyError):
        if algorithm != minimum.ALGORITHM_NAME:
            raise
        source, upstream = original_source, None
    if algorithm == block_parameters.DEFLATE:
        raise ActionTypeError("DEFLATE requires an explicit gzip_prefix, zlib_prefix, or corad_ctk2 io_variant")
    if algorithm == optimal_rle.ALGORITHM_NAME:
        producer = (upstream.get('io_variant') or upstream.get('actions', [None])[-1]) if upstream else None
        for variant, expected in optimal_rle.PRODUCERS.items():
            if source.kind == 'bytes' and producer == expected:
                return variant
        if source.kind in ('int64', 'float64'): return optimal_rle.WORDS_VARIANT
        if source.kind == 'bytes': return optimal_rle.BYTES_VARIANT
        raise ActionTypeError('RLE requires raw numeric words, bytes, or an explicit source adapter')
    if algorithm == "op.match_window_model.reference_v1":
        return algorithm
    if algorithm == minimum.ALGORITHM_NAME:
        producer = (upstream.get("io_variant") or upstream.get("actions", [None])[-1]) if upstream else None
        if source.kind == "bytes":
            if source.payload.startswith(b"BSC1") and producer == primitive.BUFF_ACTIONS[0]:
                return primitive.BUFF_FOR_ACTION.name
            for variant, (expected_producer, magic) in minimum.PRODUCER_VARIANTS.items():
                if producer == expected_producer and source.payload.startswith(magic):
                    return variant
        if original_source.kind in {"int64", "bytes"} and len(original_source.payload) % 8 == 0:
            return minimum.PLAIN_VARIANT
        raise ActionTypeError("minimum FOR requires signed int64, uint64 LE bytes, or a declared source packet; float64 is not a lossless numeric FOR input")
    if algorithm == bitpack.ALGORITHM_NAME:
        if source.kind == "int64":
            return bitpack.PLAIN_VARIANT
        if source.kind == "bytes":
            producer = (upstream.get("io_variant") or upstream.get("actions", [None])[-1]) if upstream else None
            for variant, magic in bitpack.INPUT_MAGICS.items():
                if producer == bitpack.PRODUCER_VARIANTS[variant] and source.payload.startswith(magic):
                    return variant
            if len(source.payload) % 8 == 0:
                return bitpack.PLAIN_VARIANT
        raise ActionTypeError("max-value-width requires nonnegative int64, uint64 LE bytes, or a declared source packet")
    candidates = []
    for variant in primitive.POLICY_IO_VARIANTS[algorithm]:
        descriptor = IO_VARIANT_ACTION_BY_NAME[variant]
        if source.kind not in descriptor.accepted_input_kinds:
            continue
        frozen_variant = primitive.algorithm_io.VARIANT_BY_NAME.get(variant)
        old_id = (frozen_variant or {}).get("legacy_206_policy_id")
        magics = _VARIANT_PACKET_MAGICS.get(old_id)
        if source.kind == "bytes" and magics and not source.payload.startswith(magics):
            continue
        candidates.append(variant)
    if len(candidates) != 1:
        detail = ", ".join(candidates) if candidates else "no compatible input schema"
        raise ActionTypeError(f"{algorithm}: specify io_variant ({detail})")
    return candidates[0]


def _resolve_actions(actions: Sequence[ActionRef]) -> tuple[ActionDescriptor, ...]:
    result = []
    for ref in actions:
        descriptor = describe_action(ref)
        name = descriptor.name
        if name != primitive.cpp_migration.DICTIONARY:
            name = primitive.CANONICAL_TO_LEGACY_NAME.get(name, name)
        result.append(ACTION_BY_NAME[name])
    return tuple(result)


def _require_known_actions(action_names: Sequence[str]) -> tuple[ActionDescriptor, ...]:
    if not action_names:
        raise ActionSequenceError("at least one action is required")
    try:
        return tuple(ACTION_BY_NAME[name] for name in action_names)
    except KeyError as exc:
        raise KeyError(f"unknown action {exc.args[0]!r}") from exc


def _pack_result(
    action_names: Sequence[str],
    source: ActionData,
    encoded: bytes,
    max_point: int | None,
    *,
    upstream_header: dict[str, Any] | None = None,
    stage_input_payload_count: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ActionData:
    metadata: dict[str, Any] = {
        "actions": list(action_names),
        "source_kind": source.kind,
        "source_count": source.count,
        "max_point": max_point,
    }
    if upstream_header is not None:
        metadata["upstream_result_header"] = upstream_header
    if stage_input_payload_count is not None:
        metadata["stage_input_payload_count"] = stage_input_payload_count
    if extra_metadata:
        metadata.update(extra_metadata)
    return _pack_existing_result(metadata, encoded)


def _pack_existing_result(header: dict[str, Any], encoded: bytes) -> ActionData:
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ActionData.bytes(
        WIRE_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + encoded
    )


def _unpack_result(data: ActionData) -> tuple[dict[str, Any], bytes]:
    if data.kind != "bytes":
        raise ActionTypeError("decode input must be bytes from a prior action call")
    raw = data.payload
    if len(raw) < 8 or raw[:4] != WIRE_MAGIC:
        raise ActionRuntimeError("invalid unified action result header")
    header_size = struct.unpack_from("<I", raw, 4)[0]
    if header_size > len(raw) - 8:
        raise ActionRuntimeError("truncated unified action result header")
    try:
        header = json.loads(raw[8 : 8 + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionRuntimeError("invalid unified action result metadata") from exc
    if not isinstance(header, dict):
        raise ActionRuntimeError("invalid unified action result metadata object")
    return header, raw[8 + header_size :]


def _root_source_kind(header: dict[str, Any]) -> DataKind:
    current = header
    while isinstance(current.get("upstream_result_header"), dict):
        current = current["upstream_result_header"]
    kind = current.get("source_kind")
    if kind not in {"int64", "float64", "bytes"}:
        raise ActionRuntimeError("invalid root source kind metadata")
    return kind


def _root_source_count(header: dict[str, Any]) -> int:
    current = header
    while isinstance(current.get("upstream_result_header"), dict):
        current = current["upstream_result_header"]
    count = current.get("source_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ActionRuntimeError("invalid root source count metadata")
    return count


def _inherited_max_point(header: dict[str, Any]) -> int | None:
    current: dict[str, Any] | None = header
    while current is not None:
        value = current.get("max_point")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        nested = current.get("upstream_result_header")
        current = nested if isinstance(nested, dict) else None
    return None


def _unwrap_stage_source(
    source: ActionData,
) -> tuple[ActionData, dict[str, Any] | None]:
    """Expose a prior stage's raw payload while retaining its transport header.

    The public API wraps every result in UAR1 metadata so a single action can be
    inverted without out-of-band type information.  A downstream stage must
    consume the encoded payload, not those transport bytes.  The nested header
    lets its inverse reconstruct the exact preceding ``ActionData`` value.
    """

    if source.kind != "bytes" or not source.payload.startswith(WIRE_MAGIC):
        return source, None
    header, payload = _unpack_result(source)
    stored = header.get("actions")
    if not isinstance(stored, list) or not stored:
        return source, None
    try:
        descriptors = _require_known_actions(tuple(stored))
    except (KeyError, ActionSequenceError):
        return source, None
    if not all(
        # A current atomic stage may consume the typed packet recorded by a
        # retired historical composite.  This is decode/migration lineage, not
        # permission to execute that retired descriptor forward.
        item.execution_mode in {"stagewise", "experimental_non_atomic", "retired"}
        or item.name in optimal_rle.VARIANTS
        for item in descriptors
    ):
        return source, None
    return ActionData.bytes(payload), header


def _has_optimal_bzip2_prefix(header):
    """Require declared Huffman -> current uint8 RLE lineage, not magic guessing."""
    if not isinstance(header, dict) or header.get('actions') != ['op.entropy_huffman.project_bytes.cpp_v1']:
        return False
    previous = header.get('upstream_result_header')
    return isinstance(previous, dict) and previous.get('actions') == [optimal_rle.BYTES_VARIANT]


def _encode(action_names: tuple[str, ...], source: ActionData, max_point: int | None) -> bytes:
    descriptors = _require_known_actions(action_names)
    unavailable = [item for item in descriptors if item.execution_mode == "descriptor_only"]
    if unavailable:
        item = unavailable[0]
        raise ActionNotExecutableError(f"{item.name}: {item.unavailable_reason}")

    if len(action_names) == 1 and action_names[0] in optimal_rle.VARIANTS:
        try:
            return optimal_rle.forward(source.payload, source.kind, action_names[0])
        except ValueError as exc:
            raise ActionTypeError(str(exc)) from exc
    if action_names == (optimal_rle.SPRINTZ_VARIANT,):
        if source.kind != 'bytes': raise ActionTypeError('Sprintz bitpack requires SZI1/SZF1 input')
        return primitive._run_p04_stage_cli('sprintz-bitpack-forward-current', source.payload)

    if action_names == (minimum.PLAIN_VARIANT,):
        if source.kind not in {"int64", "bytes"} or len(source.payload) % 8:
            raise ActionTypeError("minimum FOR requires signed int64 or 8-byte-aligned uint64 LE bytes")
        try:
            return primitive._cpp_action_run("minimum_for_cli", "encode-int64" if source.kind == "int64" else "encode-uint64", source.payload)
        except ValueError as exc:
            raise ActionTypeError(str(exc)) from exc
    if action_names == (bitpack.PLAIN_VARIANT,):
        if source.kind not in {"int64", "bytes"} or len(source.payload) % 8:
            raise ActionTypeError("max-value-width requires nonnegative int64 or 8-byte-aligned uint64 LE bytes")
        if source.kind == "int64" and np.any(source.as_ints() < 0):
            raise ActionTypeError("negative int64 requires a separate signed-to-unsigned transform before max-value-width packing")
        try:
            return primitive._cpp_action_run("max_value_bitpack_cli", "encode", source.payload)
        except ValueError as exc:
            raise ActionTypeError(str(exc)) from exc
    if action_names == (primitive.cpp_migration.DICTIONARY,):
        if source.kind not in ("int64", "float64"):
            raise ActionTypeError("words64 dictionary requires int64 or float64")
        return primitive._cpp_action_run("dictionary64_action_cli", "encode", source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.LOSSLESS64_ACTION_BY_NAME:
        name = action_names[0]
        if source.kind not in descriptors[0].accepted_input_kinds:
            raise ActionTypeError(f"{name} does not accept {source.kind}")
        if name in primitive.CORAD_LOSSLESS_ACTIONS:
            return primitive.corad_lossless_stages.forward(name, source.payload, source.kind)
        if name == primitive.ELF64_ERASER_ACTION:
            return primitive.elf64_exact_stages.eraser_forward(source.payload)
        return primitive.elf64_exact_stages.xor_forward(source.payload,
                   "Elf+" if name == primitive.ELF64_PLUS_ACTIONS[1] else "Elf")

    if len(action_names) == 1 and action_names[0] in primitive.LEGACY_COMPATIBILITY_ACTION_BY_NAME:
        name = action_names[0]
        if source.kind not in descriptors[0].accepted_input_kinds:
            raise ActionTypeError(f"{name} does not accept {source.kind}")
        if name == primitive.SNAPPY_LEGACY_MATCH_ACTION:
            return primitive.snappy_legacy_stages.lz_match_forward(source.payload)
        if name in primitive.BLOSC2_LEGACY_ACTIONS:
            function = (primitive.blosc2_legacy_stages.binding_core_forward
                        if name == primitive.BLOSC2_LEGACY_ACTIONS[0]
                        else primitive.blosc2_legacy_stages.frame_forward)
            return function(source.payload)
        if name in primitive.REGER_LEGACY_ACTIONS:
            function = (primitive.reger_legacy_stages.block_core_forward
                        if name == primitive.REGER_LEGACY_ACTIONS[0]
                        else primitive.reger_legacy_stages.frame_forward)
            return function(source.payload)
        if name in primitive.BUFF_NATIVE_ACTIONS:
            command = "word-order-forward" if name == primitive.BUFF_NATIVE_ACTIONS[0] else "frame-forward"
            return primitive._run_exact_stage_cli(primitive.BUFF_NATIVE_STAGE_CLI, "BUFF native", command, source.payload)
        return primitive.legacy_numeric_stages.forward(name, source.payload, source.kind, max_point)

    if action_names == (primitive.RLE_SCALING_ACTION,):
        if source.kind != "float64" or max_point is None:
            raise ActionTypeError(
                "TsFile-RLE scaling requires float64 input and max_point"
            )
        return primitive._run_rle_stage_cli(
            "scale-forward", source.payload, max_point
        )

    if action_names == ("op.run_length.tsfile_int64.v1",) and source.kind == "bytes":
        return primitive._run_rle_stage_cli("run-forward", source.payload)

    if all(name in primitive.ACTION_BY_NAME for name in action_names):
        if source.kind != "int64":
            raise ActionTypeError("reference/proxy actions require int64 input")
        return primitive.encode_operator_pipeline(source.as_ints(), action_names)

    if action_names == primitive.TS2DIFF_INT_ACTIONS:
        if source.kind != "int64":
            raise ActionTypeError("integer TS_2DIFF requires int64 input")
        return primitive.encode_exact_ts2diff_action_pipeline(source.payload, "int")

    if action_names == primitive.TS2DIFF_FLOAT_ACTIONS:
        if source.kind != "float64":
            raise ActionTypeError("floating TS_2DIFF requires float64 input")
        return primitive.encode_exact_ts2diff_action_pipeline(
            source.payload, "float", max_point=max_point
        )

    if action_names == primitive.SVB_INT_ACTIONS:
        if source.kind != "int64":
            raise ActionTypeError("SVB exact stage chain requires int64 input")
        return primitive.encode_exact_svb_action_pipeline(source.payload, "int")

    if action_names == primitive.SVB_FLOAT_ACTIONS:
        if source.kind != "float64":
            raise ActionTypeError("floating SVB exact stage chain requires float64")
        return primitive.encode_exact_svb_action_pipeline(
            source.payload, "float", max_point=max_point
        )

    if action_names == primitive.SIMPLE8B_INT_ACTIONS and source.kind == "int64":
        return primitive.encode_exact_simple8b_action_pipeline(source.payload, "int")

    if action_names == primitive.SIMPLE8B_FLOAT_ACTIONS:
        if source.kind != "float64":
            raise ActionTypeError("floating Simple8b requires float64 input")
        return primitive.encode_exact_simple8b_action_pipeline(
            source.payload, "float", max_point=max_point
        )

    if action_names in {primitive.RLBE_ACTIONS, primitive.SPRINTZ_ACTIONS}:
        if source.kind not in {"int64", "float64"}:
            raise ActionTypeError("RLBE/Sprintz exact stage chains require int64 or float64")
        method = "RLBE" if action_names == primitive.RLBE_ACTIONS else "Sprintz"
        logical_type = "int" if source.kind == "int64" else "float"
        return primitive.encode_exact_p04_action_pipeline(
            source.payload, method, logical_type
        )

    if len(action_names) == 1 and action_names[0] in primitive.P04_EXACT_ACTIONS_BY_NAME:
        name = action_names[0]
        first_commands = {
            primitive.RLBE_ACTIONS[0]: "rlbe-delta-forward",
            primitive.SPRINTZ_ACTIONS[0]: "sprintz-predictor-forward",
        }
        if name in first_commands:
            if source.kind not in {"int64", "float64"}:
                raise ActionTypeError(f"{name} requires int64 or float64 input")
            command = first_commands[name]
            if name == primitive.SPRINTZ_ACTIONS[0] and source.kind == "float64":
                command = "sprintz-predictor-forward-double"
            return primitive._run_p04_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError(f"{name} requires a preceding p04 stage packet")
        command = {
            primitive.RLBE_ACTIONS[1]: "rlbe-run-forward",
            primitive.SPRINTZ_ACTIONS[1]: "sprintz-zigzag-forward",
            primitive.SPRINTZ_ACTIONS[2]: "sprintz-bitpack-forward",
        }[name]
        return primitive._run_p04_stage_cli(command, source.payload)

    if action_names == (primitive.ZIGZAG_RAW_ACTION.name,):
        if source.kind != "int64":
            raise ActionTypeError("raw ZigZag requires int64 input")
        return primitive._run_pure_numeric_stage_cli("zigzag-forward", source.payload)

    if action_names == (primitive.ULEB128_RAW_ACTION.name,):
        if source.kind != "bytes" or len(source.payload) % 8:
            raise ActionTypeError("unsigned ULEB128 requires uint64 little-endian lanes")
        return primitive._run_unsigned_leb128_stage_cli("raw-forward", source.payload)

    if action_names == primitive.DBP_INT_ACTIONS:
        if source.kind != "int64":
            raise ActionTypeError("integer DBP exact stage chain requires int64")
        return primitive.encode_exact_dbp_action_pipeline(source.payload, "int")

    if action_names == primitive.DBP_FLOAT_ACTIONS:
        if source.kind != "float64":
            raise ActionTypeError("floating DBP exact stage chain requires float64")
        return primitive.encode_exact_dbp_action_pipeline(
            source.payload, "float", max_point=max_point
        )

    if len(action_names) == 1 and action_names[0] in primitive.DBP_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.DBP_FLOAT_ACTIONS[0]:
            if source.kind != "float64" or max_point is None:
                raise ActionTypeError("DBP scaling requires float64 input and max_point")
            return primitive._run_dbp_stage_cli(
                "scale-forward", source.payload, max_point
            )
        if name == primitive.DBP_FLOAT_ACTIONS[1]:
            command = {
                "int64": "delta-forward-int",
                "bytes": "delta-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("DBP delta requires int64 or scaling-packet bytes")
            return primitive._run_dbp_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError(f"{name} requires a preceding DBP stage packet")
        command = ("zigzag-forward" if name == primitive.DBP_FLOAT_ACTIONS[2]
                   else "varint-forward")
        return primitive._run_dbp_stage_cli(command, source.payload)

    if action_names == primitive.GORILLA_ACTIONS:
        if source.kind not in {"int64", "float64"}:
            raise ActionTypeError("Gorilla exact stage chain requires int64 or float64")
        logical_type = "int" if source.kind == "int64" else "float"
        return primitive.encode_exact_gorilla_action_pipeline(
            source.payload, logical_type
        )

    if len(action_names) == 1 and action_names[0] in primitive.GORILLA_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.GORILLA_ACTIONS[0]:
            command = {
                "int64": "xor-forward-int", "float64": "xor-forward-float"
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("Gorilla XOR requires int64 or float64")
            return primitive._run_gorilla_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("Gorilla pack requires an XOR-stage packet")
        return primitive._run_gorilla_stage_cli("pack-forward", source.payload)

    if action_names == primitive.CHIMP128_ACTIONS:
        if source.kind not in {"int64", "float64"}:
            raise ActionTypeError("Chimp128 exact stage chain requires int64 or float64")
        logical_type = "int" if source.kind == "int64" else "float"
        return primitive.encode_exact_chimp128_action_pipeline(
            source.payload, logical_type
        )

    if len(action_names) == 1 and action_names[0] in primitive.CHIMP128_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.CHIMP128_ACTIONS[0]:
            command = {
                "int64": "xor-forward-int", "float64": "xor-forward-float"
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("Chimp128 XOR requires int64 or float64")
            return primitive._run_chimp128_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("Chimp128 pack requires an XOR-stage packet")
        return primitive._run_chimp128_stage_cli("pack-forward", source.payload)

    if action_names == primitive.PONGO_ACTIONS:
        if source.kind not in {"int64", "float64"}:
            raise ActionTypeError("Pongo exact stage chain requires int64 or float64")
        return primitive.encode_exact_pongo_action_pipeline(source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.PONGO_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.PONGO_ACTIONS[0]:
            if source.kind not in {"int64", "float64"}:
                raise ActionTypeError("Pongo eraser requires int64 or float64 lanes")
            return primitive._run_pongo_stage_cli("eraser-forward", source.payload)
        if source.kind != "bytes":
            raise ActionTypeError(f"{name} requires the preceding Pongo stage packet")
        command = (
            "xor-forward"
            if name == primitive.PONGO_ACTIONS[1]
            else "pack-forward"
        )
        return primitive._run_pongo_stage_cli(command, source.payload)

    if action_names == primitive.SPDP_ACTIONS:
        if source.kind != "bytes":
            raise ActionTypeError("SPDP exact stage chain requires bytes input")
        return primitive.encode_exact_spdp_action_pipeline(source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.SPDP_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("SPDP stages require bytes input")
        command = {
            primitive.SPDP_ACTIONS[0]: "delta-forward",
            primitive.SPDP_ACTIONS[1]: "block-forward",
            primitive.SPDP_ACTIONS[2]: "entropy-forward",
        }[action_names[0]]
        return primitive._run_spdp_stage_cli(command, source.payload)

    if action_names == primitive.FPC_ACTIONS:
        if source.kind != "bytes":
            raise ActionTypeError("FPC exact stage chain requires bytes input")
        return primitive.encode_exact_fpc_action_pipeline(source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.FPC_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("FPC stages require bytes input")
        command = (
            "xor-forward"
            if action_names[0] == primitive.FPC_ACTIONS[0]
            else "pack-forward"
        )
        return primitive._run_fpc_stage_cli(command, source.payload)

    if action_names == primitive.PFPC_ACTIONS:
        if source.kind != "bytes":
            raise ActionTypeError("pFPC exact stage chain requires bytes input")
        return primitive.encode_exact_pfpc_action_pipeline(source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.PFPC_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("pFPC stages require bytes input")
        command = (
            "xor-forward"
            if action_names[0] == primitive.PFPC_ACTIONS[0]
            else "pack-forward"
        )
        return primitive._run_pfpc_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.BOS_ACTIONS_BY_NAME:
        name = action_names[0]
        bosa_commands = {
            primitive.BOS_BOSA_ACTIONS[0]: "delta-forward",
            primitive.BOS_BOSA_ACTIONS[1]: "for1-forward",
            primitive.BOS_BOSA_ACTIONS[2]: "outlier-forward",
            primitive.BOS_BOSA_ACTIONS[3]: "for2-forward",
            primitive.BOS_BOSA_ACTIONS[4]: "bitpack-forward",
        }
        if name in bosa_commands:
            if name == primitive.BOS_BOSA_ACTIONS[0]:
                if source.kind != "int64":
                    raise ActionTypeError("BOSA delta requires int64 input")
            elif source.kind != "bytes":
                raise ActionTypeError("BOSA stages after delta require stage-packet bytes")
            return primitive._run_bos_stage_cli(bosa_commands[name], source.payload)
        if name == primitive.BOS64_LEGACY_DELTA_ACTION.name:
            command = {
                "int64": "adapter-delta-forward",
                "bytes": "adapter-delta-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("legacy BOS64 delta requires int64 or scaling-packet bytes")
            return primitive._run_bos_stage_cli(command, source.payload)
        if name == primitive.BOS64_INT_ACTIONS[0]:
            command = {
                "int64": "adapter-plain-delta-forward",
                "bytes": "adapter-plain-delta-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("BOS64 delta requires int64 or scaling-packet bytes")
            return primitive._run_bos_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("BOS64 downstream stages require a stage packet")
        if name == primitive.BOS64_INT_ACTIONS[1]:
            command = ("adapter-zigzag-forward-scaled"
                       if source.payload.startswith(b"B6T1")
                       else "adapter-zigzag-forward")
            return primitive._run_bos_stage_cli(command, source.payload)
        command = (
            "adapter-wire-forward-scaled"
            if source.payload.startswith(b"B6F1")
            else "adapter-wire-forward-int"
        )
        return primitive._run_bos_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.FASTPFOR_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.FASTPFOR_COMPOSITE_HELPER_ACTIONS[0].name:
            command = {
                "int64": "adapter-forward-int",
                "bytes": "adapter-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError(
                    "FastPFor composite helper requires int64 or scaling-packet bytes"
                )
            return primitive._run_fastpfor_stage_cli(command, source.payload)
        if name == primitive.FASTPFOR_INT_ACTIONS[0]:
            command = {
                "int64": "pfor-forward-int",
                "bytes": "pfor-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("PFor requires int64 or scaling-packet bytes")
        else:
            if source.kind != "bytes":
                raise ActionTypeError("FastPFor VariableByte requires a PFor-stage packet")
            command = "variable-byte-forward"
        return primitive._run_fastpfor_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.ELF_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.ELF_EXACT_ACTIONS[0].name:
            if source.kind != "float64":
                raise ActionTypeError("Elf eraser requires float64 input")
            return primitive._run_elf_stage_cli("eraser-forward", source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("Elf XOR requires an eraser-stage packet")
        command = (
            "elf-xor-forward"
            if name == primitive.ELF_ACTIONS[1]
            else "elf-plus-xor-forward"
        )
        return primitive._run_elf_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.CAMEL_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.CAMEL_ACTIONS[0]:
            if source.kind not in {"int64", "float64"}:
                raise ActionTypeError("Camel eraser requires int64 or float64 input")
            command = "eraser-forward-int" if source.kind == "int64" else "eraser-forward"
            return primitive._run_camel_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("Camel downstream stages require packet bytes")
        command = "xor-forward" if name == primitive.CAMEL_ACTIONS[1] else "pack-forward"
        return primitive._run_camel_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.SNAPPY_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("Snappy stages require bytes input")
        command = {
            primitive.SNAPPY_ACTIONS[0]: "lz-forward",
            primitive.SNAPPY_ACTIONS[1]: "token-forward",
            primitive.SNAPPY_ACTIONS[2]: "integer-forward",
            primitive.SNAPPY_LEGACY_INTEGER_ACTION.name: "integer-forward",
        }[action_names[0]]
        return primitive._run_snappy_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.LZ4_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("LZ4 stages require bytes input")
        command = {
            primitive.LZ4_ACTIONS[0]: "lz-forward",
            primitive.LZ4_ACTIONS[1]: "token-forward",
            primitive.LZ4_ACTIONS[2]: "integer-forward",
            primitive.LZ4_ACTIONS[3]: "container-forward",
        }[action_names[0]]
        return primitive._run_lz4_stage_cli(command, source.payload)

    if (
        len(action_names) == 1
        and action_names[0] in primitive.DEFLATE_CONTAINER_ACTIONS_BY_NAME
    ):
        if source.kind != "bytes":
            raise ActionTypeError("DEFLATE container stages require bytes input")
        command = {
            primitive.GZIP_CONTAINER_ACTIONS[0]: "gzip-forward",
            primitive.ZLIB_CONTAINER_ACTIONS[0]: "zlib-forward",
        }[action_names[0]]
        return primitive._run_deflate_container_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.XZ_CONTAINER_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("XZ container stage requires bytes input")
        return primitive._run_xz_container_stage_cli("forward", source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.FPZIP_ACTIONS_BY_NAME:
        name = action_names[0]
        if name == primitive.FPZIP_ACTIONS[0]:
            if source.kind != "float64":
                raise ActionTypeError("FPZIP predictor requires float64 input")
            return primitive._run_fpzip_stage_cli("predictor-forward", source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("FPZIP range coder requires predictor packet bytes")
        return primitive._run_fpzip_stage_cli("range-forward", source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.LEARNING_CODEC_STAGE_ACTIONS_BY_NAME:
        name = action_names[0]
        first_stage_kinds = {
            primitive.LECO_ACTIONS[0]: "int64",
            primitive.ALP_MAIN_ACTIONS[0]: "float64",
            primitive.ALP_DICTIONARY_ACTIONS[0]: "float64",
            primitive.REGER_ACTIONS[0]: "int64",
            primitive.BUFF_ACTIONS[0]: "float64",
        }
        required_kind = first_stage_kinds.get(name, "bytes")
        if source.kind != required_kind:
            raise ActionTypeError(f"{name} requires {required_kind} input")
        if name == primitive.BUFF_PLANE_ACTION.name:
            return primitive.run_buff_byte_plane(source.payload)[0]
        command = {
            primitive.BUFF_FOR_ACTION.name: "buff-for-forward",
            primitive.LECO_ACTIONS[0]: "leco-regression-forward",
            primitive.LECO_ACTIONS[1]: "leco-zigzag-forward",
            primitive.LECO_ACTIONS[2]: "leco-bitpack-forward",
            primitive.LECO_LEGACY_PREDICTOR_ACTION.name: "leco-autoregression-forward",
            primitive.ALP_MAIN_ACTIONS[0]: "alp-scaling-forward",
            primitive.ALP_MAIN_ACTIONS[1]: "alp-for-forward",
            primitive.ALP_MAIN_ACTIONS[2]: "alp-bitpack-forward",
            primitive.ALP_MAIN_ACTIONS[3]: "alp-residual-core-forward",
            primitive.ALP_DICTIONARY_ACTIONS[0]: "alp-dictionary-forward",
            primitive.REGER_ACTIONS[0]: "reger-reorder-forward",
            primitive.REGER_ACTIONS[1]: "reger-delta-forward",
            primitive.REGER_ACTIONS[2]: "reger-zigzag-forward",
            primitive.REGER_ACTIONS[3]: "reger-for-forward",
            primitive.REGER_ACTIONS[4]: "reger-bitpack-forward",
            primitive.REGER_ACTIONS[5]: "reger-residual-core-forward",
            primitive.REGER_LEGACY_PREDICTOR_ACTION.name: "reger-autoregression-forward",
            primitive.BUFF_ACTIONS[0]: "buff-scaling-forward",
            primitive.BUFF_ACTIONS[1]: "buff-column-split-forward",
            primitive.BUFF_ACTIONS[2]: "buff-bitpack-forward",
        }[name]
        return primitive._run_learning_codec_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.GENERAL_CODEC_REFERENCE_ACTIONS_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("general-codec reference stages require bytes input")
        name = action_names[0]
        command = {
            primitive.BROTLI_REFERENCE_ACTIONS[0]: "dictionary-forward",
            primitive.BROTLI_REFERENCE_ACTIONS[1]: "brotli-lz-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[2].name: "huffman-forward",
            primitive.ZSTD_REFERENCE_ACTIONS[0]: "zstd-lz-forward",
            primitive.ZSTD_REFERENCE_ACTIONS[1]: "parse-forward",
            primitive.ZSTD_REFERENCE_ACTIONS[2]: "huffman-forward",
            primitive.ZSTD_REFERENCE_ACTIONS[3]: "rans-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[7].name: "shuffle1-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[8].name: "shuffle4-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[9].name: "shuffle8-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[10].name: "blosclz-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[11].name: "token-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[12].name: "blosc-frame-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[13].name: "blosc2-frame-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[14].name: "bwt-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[15].name: "mtf-forward",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[16].name: "rle-forward",
            primitive.DEFLATE_REFERENCE_LZ_ACTION: "deflate-lz-forward",
            primitive.XZ_REFERENCE_LZ_ACTION: "xz-lz-forward",
            primitive.REFERENCE_RANGE_ACTION: "range-forward",
            primitive.GZIP_REFERENCE_FRAME_ACTION: "gzip-frame-forward",
            primitive.ZLIB_REFERENCE_FRAME_ACTION: "zlib-frame-forward",
            primitive.XZ_REFERENCE_FRAME_ACTION: "xz-frame-forward",
        }[name]
        return primitive._run_general_codec_reference_stage_cli(command, source.payload)

    if (
        len(action_names) == 1
        and action_names[0] in primitive.GENERAL_CODEC_RESIDUAL_ACTIONS_BY_NAME
    ):
        if source.kind != "bytes":
            raise ActionTypeError("general-codec residual stages require bytes input")
        name = action_names[0]
        command = {
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[0].name: "brotli-forward",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[1].name: "zstd-forward",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[2].name: "blosc1-forward-1",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[3].name: "blosc1-forward-4",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[4].name: "blosc1-forward-8",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[5].name: "blosc2-forward-1",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[6].name: "blosc2-forward-4",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[7].name: "blosc2-forward-8",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[8].name: "bzip2-forward",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[9].name: "gzip-forward",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[10].name: "zlib-forward",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[11].name: "xz-forward",
        }[name]
        return primitive._run_general_codec_residual_stage_cli(command, source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.SIMPLE8B_FLOAT_ACTIONS:
        name = action_names[0]
        if name == primitive.SIMPLE8B_FLOAT_ACTIONS[0]:
            if source.kind != "float64" or max_point is None:
                raise ActionTypeError("Simple8b scaling requires float64 input and max_point")
            return primitive._run_simple8b_stage_cli(
                "scale-forward", source.payload, max_point
            )
        if source.kind == "int64":
            return primitive._run_simple8b_stage_cli(
                "bitpack-forward-int", source.payload
            )
        if source.kind == "bytes":
            return primitive._run_simple8b_stage_cli(
                "bitpack-forward-scaled", source.payload
            )
        raise ActionTypeError("Simple8b bitpack requires int64 or scaling-packet bytes")

    if len(action_names) == 1 and action_names[0] in primitive.SVB_INT_ACTIONS:
        name = action_names[0]
        if name == primitive.SVB_INT_ACTIONS[0]:
            command = {
                "int64": "delta-forward",
                "bytes": "delta-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("SVB delta requires int64 or scaling-packet bytes")
            return primitive._run_svb_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError("SVB integer-code stage requires delta-packet bytes")
        return primitive._run_svb_stage_cli("integer-code-forward", source.payload)

    if len(action_names) == 1 and action_names[0] in primitive.TS2DIFF_FLOAT_ACTIONS:
        name = action_names[0]
        if name == primitive.TS2DIFF_FLOAT_ACTIONS[0]:
            if source.kind != "float64" or max_point is None:
                raise ActionTypeError("TS_2DIFF scaling requires float64 input and max_point")
            return primitive._run_ts2diff_stage_cli("scale-forward", source.payload, max_point)
        if name == primitive.TS2DIFF_FLOAT_ACTIONS[1]:
            command = {
                "int64": "delta-forward-int",
                "bytes": "delta-forward-scaled",
            }.get(source.kind)
            if command is None:
                raise ActionTypeError("TS_2DIFF delta requires int64 or scaling-packet bytes")
            return primitive._run_ts2diff_stage_cli(command, source.payload)
        if source.kind != "bytes":
            raise ActionTypeError(f"{name} requires stage-packet bytes")
        command = (
            "for-forward"
            if name == primitive.TS2DIFF_FLOAT_ACTIONS[2]
            else "bitpack-forward"
        )
        return primitive._run_ts2diff_stage_cli(command, source.payload)

    if action_names == primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS:
        if source.kind != "float64":
            raise ActionTypeError("scaling -> int64 RLE requires float64 input")
        return primitive.encode_registered_exact_action_pipeline(
            source.payload, action_names, logical_type="float", max_point=max_point
        )

    if len(action_names) == 1 and action_names[0] in primitive.EXACT_ATOMIC_ACTION_BY_NAME:
        required = _atomic_input_kinds(action_names[0])
        if source.kind not in required:
            raise ActionTypeError(
                f"{action_names[0]} requires {required}, got {source.kind}"
            )
        logical_type = {"int64": "int", "float64": "float", "bytes": "bytes"}[
            source.kind
        ]
        return primitive.encode_registered_exact_action_pipeline(
            source.payload,
            action_names,
            logical_type=logical_type,
            max_point=max_point,
        )

    if len(action_names) == 1 and action_names[0] in primitive.EXACT_NATIVE_ACTION_BY_NAME:
        if source.kind != "bytes":
            raise ActionTypeError("native-LZ stages require bytes or a preceding stage packet")
        try:
            return primitive.encode_exact_native_lz_stage(
                action_names[0], source.payload
            )
        except ValueError as exc:
            raise ActionSequenceError(str(exc)) from exc

    if action_names in primitive._EXACT_NATIVE_PIPELINE_KEY:
        if source.kind != "bytes":
            raise ActionTypeError("native-LZ exact stage sequences require bytes input")
        return primitive.encode_registered_exact_action_pipeline(source.payload, action_names)

    raise ActionSequenceError(
        "actions are individually registered but do not form a legal executable sequence"
    )


def _decode(expected_names: tuple[str, ...], encoded: ActionData) -> ActionData:
    header, payload = _unpack_result(encoded)
    stored_names = tuple(header.get("actions", ()))
    if expected_names != stored_names:
        raise ActionSequenceError(
            f"decode action sequence mismatch: expected {expected_names}, stored {stored_names}"
        )
    _require_known_actions(stored_names)
    kind = header.get("source_kind")
    count = header.get("source_count")
    max_point = header.get("max_point")
    root_kind = _root_source_kind(header)
    stage_input_payload_count = header.get("stage_input_payload_count")
    if not isinstance(stage_input_payload_count, int):
        stage_input_payload_count = count
    upstream_header = header.get("upstream_result_header")
    if upstream_header is not None and not isinstance(upstream_header, dict):
        raise ActionRuntimeError("invalid upstream action result metadata")
    if kind not in {"int64", "float64", "bytes"} or not isinstance(count, int):
        raise ActionRuntimeError("invalid source type metadata")

    if len(stored_names) == 1 and stored_names[0] in optimal_rle.VARIANTS:
        if type(count) is not int or count < 0:
            raise ActionTypeError('invalid RLE source count')
        if stored_names[0] != optimal_rle.WORDS_VARIANT and kind != 'bytes':
            raise ActionTypeError('RLE byte adapter has invalid source kind')
        try:
            raw = optimal_rle.inverse(payload, stored_names[0])
        except ValueError as exc:
            raise ActionTypeError(str(exc)) from exc
        expected_bytes = stage_input_payload_count if upstream_header is not None else count * (8 if kind != 'bytes' else 1)
        if type(expected_bytes) is not int or len(raw) != expected_bytes:
            raise ActionTypeError('RLE decoded extent disagrees with source metadata')
        restored = ActionData(kind, raw, count) if upstream_header is None else ActionData.bytes(raw)
    elif stored_names == (optimal_rle.SPRINTZ_VARIANT,):
        if kind != 'bytes': raise ActionTypeError('Sprintz bitpack requires byte source metadata')
        command = 'sprintz-bitpack-inverse-double-current' if root_kind == 'float64' else 'sprintz-bitpack-inverse-current'
        restored = ActionData.bytes(primitive._run_p04_stage_cli(command, payload))
    elif stored_names == (minimum.PLAIN_VARIANT,):
        if kind not in {"int64", "bytes"} or type(count) is not int or count < 0:
            raise ActionTypeError("invalid stored minimum FOR input type/count")
        if len(payload) < 21 or payload[:4] != b"MFR1":
            raise ActionTypeError("invalid minimum FOR MFR1 header")
        expected_kind = 0 if kind == "int64" else 1
        expected_bytes = stage_input_payload_count if upstream_header is not None else (count*8 if kind == "int64" else count)
        encoded_count = struct.unpack_from("<Q", payload, 5)[0]
        if payload[4] != expected_kind or type(expected_bytes) is not int or expected_bytes < 0 or encoded_count*8 != expected_bytes:
            raise ActionTypeError("MFR1 type/count disagrees with recorded input")
        try:
            raw = primitive._cpp_action_run("minimum_for_cli", "decode", payload)
        except ValueError as exc:
            raise ActionTypeError(str(exc)) from exc
        restored = ActionData(kind, raw, count) if upstream_header is None else ActionData.bytes(raw)
    elif stored_names == (bitpack.PLAIN_VARIANT,):
        if kind not in {"int64", "bytes"} or type(count) is not int or count < 0:
            raise ActionTypeError("invalid stored max-value-width input type/count")
        if len(payload) < 13 or payload[:4] != b"MVW1":
            raise ActionTypeError("invalid max-value-width MVW1 header")
        expected_bytes = stage_input_payload_count if upstream_header is not None else (count*8 if kind == "int64" else count)
        encoded_count = struct.unpack_from("<Q", payload, 4)[0]
        if type(expected_bytes) is not int or expected_bytes < 0 or encoded_count*8 != expected_bytes:
            raise ActionTypeError("MVW1 count disagrees with recorded input size")
        try:
            raw = primitive._cpp_action_run("max_value_bitpack_cli", "decode", payload)
        except ValueError as exc:
            raise ActionTypeError(str(exc)) from exc
        if kind == "int64" and np.any(np.frombuffer(raw, dtype="<i8") < 0):
            raise ActionTypeError("decoded uint64 value violates the recorded nonnegative int64 input domain")
        restored = ActionData(kind, raw, count) if upstream_header is None else ActionData.bytes(raw)
    elif stored_names == (primitive.cpp_migration.DICTIONARY,):
        raw = primitive._cpp_action_run("dictionary64_action_cli", "decode", payload)
        restored = ActionData(kind, raw, count)
    elif len(stored_names) == 1 and stored_names[0] in primitive.LOSSLESS64_ACTION_BY_NAME:
        name = stored_names[0]
        if name in primitive.CORAD_LOSSLESS_ACTIONS:
            raw = primitive.corad_lossless_stages.inverse(name, payload)
        elif name == primitive.ELF64_ERASER_ACTION:
            raw = primitive.elf64_exact_stages.eraser_inverse(payload)
        else:
            raw = primitive.elf64_exact_stages.xor_inverse(payload,
                      "Elf+" if name == primitive.ELF64_PLUS_ACTIONS[1] else "Elf")
        restored = (ActionData(kind, raw, count)
                    if name in (primitive.ELF64_ERASER_ACTION, primitive.CORAD_LOSSLESS_ACTIONS[0])
                    else ActionData.bytes(raw))
    elif len(stored_names) == 1 and stored_names[0] in primitive.LEGACY_COMPATIBILITY_ACTION_BY_NAME:
        name = stored_names[0]
        if name == primitive.SNAPPY_LEGACY_MATCH_ACTION:
            raw = primitive.snappy_legacy_stages.lz_match_inverse(payload)
        elif name in primitive.BLOSC2_LEGACY_ACTIONS:
            function = (primitive.blosc2_legacy_stages.binding_core_inverse
                        if name == primitive.BLOSC2_LEGACY_ACTIONS[0]
                        else primitive.blosc2_legacy_stages.frame_inverse)
            raw = function(payload)
        elif name in primitive.REGER_LEGACY_ACTIONS:
            function = (primitive.reger_legacy_stages.block_core_inverse
                        if name == primitive.REGER_LEGACY_ACTIONS[0]
                        else primitive.reger_legacy_stages.frame_inverse)
            raw = function(payload)
        elif name in primitive.BUFF_NATIVE_ACTIONS:
            command = "word-order-inverse" if name == primitive.BUFF_NATIVE_ACTIONS[0] else "frame-inverse"
            raw = primitive._run_exact_stage_cli(primitive.BUFF_NATIVE_STAGE_CLI, "BUFF native", command, payload)
        else:
            raw = primitive.legacy_numeric_stages.inverse(
                name, payload, count=header.get("legacy_numeric_count", _root_source_count(header))
            )
        if name in (primitive.legacy_numeric_stages.SCALE, primitive.legacy_numeric_stages.ALP_SCALE,
                    primitive.REGER_LEGACY_ACTIONS[0], primitive.BUFF_NATIVE_ACTIONS[0]):
            restored = ActionData(kind, raw, count)
        else:
            restored = ActionData.bytes(raw)
    elif stored_names == (primitive.RLE_SCALING_ACTION,):
        restored = ActionData(
            "float64",
            primitive._run_rle_stage_cli("scale-inverse", payload),
            count,
        )
    elif all(name in primitive.LEGACY_PRIMITIVE_ACTION_BY_NAME for name in stored_names):
        values, decoded_names = primitive.decode_operator_pipeline(payload)
        if decoded_names != stored_names:
            raise ActionRuntimeError("primitive decoder returned a different action sequence")
        restored = ActionData.ints(values)
    elif stored_names == primitive.TS2DIFF_INT_ACTIONS:
        restored = ActionData(
            "int64",
            primitive.decode_exact_ts2diff_action_pipeline(payload, "int"),
            count,
        )
    elif stored_names == primitive.TS2DIFF_FLOAT_ACTIONS:
        restored = ActionData(
            "float64",
            primitive.decode_exact_ts2diff_action_pipeline(
                payload, "float", max_point=max_point
            ),
            count,
        )
    elif stored_names == primitive.SVB_INT_ACTIONS:
        restored = ActionData(
            "int64",
            primitive.decode_exact_svb_action_pipeline(payload, "int"),
            count,
        )
    elif stored_names == primitive.SVB_FLOAT_ACTIONS:
        restored = ActionData(
            "float64",
            primitive.decode_exact_svb_action_pipeline(payload, "float"),
            count,
        )
    elif stored_names == primitive.SIMPLE8B_INT_ACTIONS and kind == "int64":
        restored = ActionData(
            "int64",
            primitive.decode_exact_simple8b_action_pipeline(payload, "int"),
            count,
        )
    elif stored_names == primitive.SIMPLE8B_FLOAT_ACTIONS:
        restored = ActionData(
            "float64",
            primitive.decode_exact_simple8b_action_pipeline(
                payload, "float", max_point=max_point
            ),
            count,
        )
    elif stored_names in {primitive.RLBE_ACTIONS, primitive.SPRINTZ_ACTIONS}:
        method = "RLBE" if stored_names == primitive.RLBE_ACTIONS else "Sprintz"
        logical_type = "int" if kind == "int64" else "float"
        restored = ActionData(
            kind,
            primitive.decode_exact_p04_action_pipeline(payload, method, logical_type),
            count,
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.P04_EXACT_ACTIONS_BY_NAME:
        name = stored_names[0]
        first_commands = {
            primitive.RLBE_ACTIONS[0]: "rlbe-delta-inverse",
            primitive.SPRINTZ_ACTIONS[0]: "sprintz-predictor-inverse",
        }
        if name in first_commands:
            restored = ActionData(
                kind,
                primitive._run_p04_stage_cli(first_commands[name], payload),
                count,
            )
        else:
            command = {
                primitive.RLBE_ACTIONS[1]: "rlbe-run-inverse",
                primitive.SPRINTZ_ACTIONS[1]: "sprintz-zigzag-inverse",
                primitive.SPRINTZ_ACTIONS[2]: (
                    "sprintz-bitpack-inverse-double"
                    if root_kind == "float64"
                    else "sprintz-bitpack-inverse"
                ),
            }[name]
            restored = ActionData.bytes(primitive._run_p04_stage_cli(command, payload))
    elif stored_names == (primitive.ZIGZAG_RAW_ACTION.name,):
        restored = ActionData(
            "int64",
            primitive._run_pure_numeric_stage_cli("zigzag-inverse", payload),
            count,
        )
    elif stored_names == (primitive.ULEB128_RAW_ACTION.name,):
        restored = ActionData.bytes(
            primitive._run_unsigned_leb128_stage_cli("raw-inverse", payload)
        )
    elif stored_names == primitive.DBP_INT_ACTIONS:
        restored = ActionData(
            "int64", primitive.decode_exact_dbp_action_pipeline(payload, "int"), count
        )
    elif stored_names == primitive.DBP_FLOAT_ACTIONS:
        restored = ActionData(
            "float64", primitive.decode_exact_dbp_action_pipeline(payload, "float"), count
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.DBP_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.DBP_FLOAT_ACTIONS[0]:
            restored = ActionData(
                "float64", primitive._run_dbp_stage_cli("scale-inverse", payload), count
            )
        elif name == primitive.DBP_FLOAT_ACTIONS[1]:
            if kind == "int64":
                restored = ActionData(
                    "int64", primitive._run_dbp_stage_cli("delta-inverse-int", payload), count
                )
            elif kind == "bytes":
                restored = ActionData.bytes(
                    primitive._run_dbp_stage_cli("delta-inverse-scaled", payload)
                )
            else:
                raise ActionTypeError("invalid stored source kind for DBP delta")
        else:
            command = (
                "zigzag-inverse"
                if name == primitive.DBP_FLOAT_ACTIONS[2]
                else "varint-inverse"
            )
            restored = ActionData.bytes(
                primitive._run_dbp_stage_cli(command, payload)
            )
    elif stored_names == primitive.GORILLA_ACTIONS:
        logical_type = "int" if kind == "int64" else "float"
        restored = ActionData(
            kind,
            primitive.decode_exact_gorilla_action_pipeline(payload, logical_type),
            count,
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.GORILLA_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.GORILLA_ACTIONS[0]:
            restored = ActionData(
                kind, primitive._run_gorilla_stage_cli("xor-inverse", payload), count
            )
        else:
            command = (
                "pack-inverse-float"
                if root_kind == "float64" or header.get("io_input_magic") == b"GXF1".hex()
                else "pack-inverse-int"
            )
            restored = ActionData.bytes(
                primitive._run_gorilla_stage_cli(command, payload)
            )
    elif stored_names == primitive.CHIMP128_ACTIONS:
        logical_type = "int" if kind == "int64" else "float"
        restored = ActionData(
            kind,
            primitive.decode_exact_chimp128_action_pipeline(payload, logical_type),
            count,
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.CHIMP128_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.CHIMP128_ACTIONS[0]:
            restored = ActionData(
                kind, primitive._run_chimp128_stage_cli("xor-inverse", payload), count
            )
        else:
            command = (
                "pack-inverse-float"
                if root_kind == "float64" or header.get("io_input_magic") == b"CXF1".hex()
                else "pack-inverse-int"
            )
            restored = ActionData.bytes(
                primitive._run_chimp128_stage_cli(command, payload)
            )
    elif stored_names == primitive.PONGO_ACTIONS:
        restored = ActionData(
            kind, primitive.decode_exact_pongo_action_pipeline(payload), count
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.PONGO_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.PONGO_ACTIONS[0]:
            restored = ActionData(
                kind, primitive._run_pongo_stage_cli("eraser-inverse", payload), count
            )
        else:
            command = (
                "xor-inverse"
                if name == primitive.PONGO_ACTIONS[1]
                else "pack-inverse"
            )
            restored = ActionData.bytes(
                primitive._run_pongo_stage_cli(command, payload)
            )
    elif stored_names == primitive.SPDP_ACTIONS:
        restored = ActionData.bytes(
            primitive.decode_exact_spdp_action_pipeline(payload, count)
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.SPDP_ACTIONS_BY_NAME:
        name = stored_names[0]
        command = {
            primitive.SPDP_ACTIONS[0]: "delta-inverse",
            primitive.SPDP_ACTIONS[1]: "block-inverse",
            primitive.SPDP_ACTIONS[2]: "entropy-inverse",
        }[name]
        restored = ActionData.bytes(
            primitive._run_spdp_stage_cli(
                command, payload,
                stage_input_payload_count - 12
                if name == primitive.SPDP_ACTIONS[2]
                else None,
            )
        )
    elif stored_names == primitive.FPC_ACTIONS:
        restored = ActionData.bytes(
            primitive.decode_exact_fpc_action_pipeline(payload)
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.FPC_ACTIONS_BY_NAME:
        command = (
            "xor-inverse"
            if stored_names[0] == primitive.FPC_ACTIONS[0]
            else "pack-inverse"
        )
        restored = ActionData.bytes(primitive._run_fpc_stage_cli(command, payload))
    elif stored_names == primitive.PFPC_ACTIONS:
        restored = ActionData.bytes(
            primitive.decode_exact_pfpc_action_pipeline(payload)
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.PFPC_ACTIONS_BY_NAME:
        command = (
            "xor-inverse"
            if stored_names[0] == primitive.PFPC_ACTIONS[0]
            else "pack-inverse"
        )
        restored = ActionData.bytes(primitive._run_pfpc_stage_cli(command, payload))
    elif len(stored_names) == 1 and stored_names[0] in primitive.BOS_ACTIONS_BY_NAME:
        name = stored_names[0]
        bosa_commands = {
            primitive.BOS_BOSA_ACTIONS[0]: "delta-inverse",
            primitive.BOS_BOSA_ACTIONS[1]: "for1-inverse",
            primitive.BOS_BOSA_ACTIONS[2]: "outlier-inverse",
            primitive.BOS_BOSA_ACTIONS[3]: "for2-inverse",
            primitive.BOS_BOSA_ACTIONS[4]: "bitpack-inverse",
        }
        if name in bosa_commands:
            raw = primitive._run_bos_stage_cli(bosa_commands[name], payload)
            restored = (
                ActionData("int64", raw, count)
                if name == primitive.BOS_BOSA_ACTIONS[0]
                else ActionData.bytes(raw)
            )
        elif name == primitive.BOS64_LEGACY_DELTA_ACTION.name:
            if kind == "int64":
                restored = ActionData(
                    "int64",
                    primitive._run_bos_stage_cli("adapter-delta-inverse", payload),
                    count,
                )
            elif kind == "bytes":
                restored = ActionData.bytes(
                    primitive._run_bos_stage_cli(
                        "adapter-delta-inverse-scaled", payload
                    )
                )
            else:
                raise ActionTypeError("invalid stored source kind for BOS64 delta")
        elif name == primitive.BOS64_INT_ACTIONS[0]:
            if kind == "int64":
                restored = ActionData(
                    "int64",
                    primitive._run_bos_stage_cli("adapter-plain-delta-inverse", payload),
                    count,
                )
            elif kind == "bytes":
                restored = ActionData.bytes(
                    primitive._run_bos_stage_cli(
                        "adapter-plain-delta-inverse-scaled", payload
                    )
                )
            else:
                raise ActionTypeError("invalid stored source kind for BOS64 delta")
        elif name == primitive.BOS64_INT_ACTIONS[1]:
            command = ("adapter-zigzag-inverse-scaled"
                       if payload.startswith(b"B6F1")
                       else "adapter-zigzag-inverse")
            restored = ActionData.bytes(
                primitive._run_bos_stage_cli(command, payload)
            )
        else:
            command = (
                "adapter-wire-inverse-scaled"
                if payload.startswith(b"F")
                else "adapter-wire-inverse-int"
            )
            restored = ActionData.bytes(
                primitive._run_bos_stage_cli(command, payload)
            )
    elif len(stored_names) == 1 and stored_names[0] in primitive.FASTPFOR_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.FASTPFOR_COMPOSITE_HELPER_ACTIONS[0].name and kind == "int64":
            restored = ActionData(
                "int64",
                primitive._run_fastpfor_stage_cli(
                    "adapter-inverse-int", payload, count
                ),
                count,
            )
        elif name == primitive.FASTPFOR_COMPOSITE_HELPER_ACTIONS[0].name and kind == "bytes":
            restored = ActionData.bytes(
                primitive._run_fastpfor_stage_cli(
                    "adapter-inverse-scaled", payload,
                    _root_source_count(header),
                )
            )
        elif name == primitive.FASTPFOR_INT_ACTIONS[0] and kind == "int64":
            restored = ActionData(
                "int64",
                primitive._run_fastpfor_stage_cli("pfor-inverse-int", payload),
                count,
            )
        elif name == primitive.FASTPFOR_INT_ACTIONS[0] and kind == "bytes":
            restored = ActionData.bytes(
                primitive._run_fastpfor_stage_cli("pfor-inverse-scaled", payload)
            )
        elif name == primitive.FASTPFOR_INT_ACTIONS[1] and kind == "bytes":
            restored = ActionData.bytes(
                primitive._run_fastpfor_stage_cli(
                    "variable-byte-inverse", payload, _root_source_count(header)
                )
            )
        else:
            raise ActionTypeError("invalid stored source kind for FastPFor")
    elif len(stored_names) == 1 and stored_names[0] in primitive.ELF_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.ELF_EXACT_ACTIONS[0].name:
            restored = ActionData(
                "float64",
                primitive._run_elf_stage_cli("eraser-inverse", payload),
                count,
            )
        else:
            command = (
                "elf-xor-inverse"
                if name == primitive.ELF_ACTIONS[1]
                else "elf-plus-xor-inverse"
            )
            restored = ActionData.bytes(primitive._run_elf_stage_cli(command, payload))
    elif len(stored_names) == 1 and stored_names[0] in primitive.CAMEL_ACTIONS_BY_NAME:
        name = stored_names[0]
        if name == primitive.CAMEL_ACTIONS[0]:
            command = "eraser-inverse-int" if kind == "int64" else "eraser-inverse"
            restored = ActionData(
                kind,
                primitive._run_camel_stage_cli(command, payload),
                count,
            )
        else:
            command = "xor-inverse" if name == primitive.CAMEL_ACTIONS[1] else "pack-inverse"
            restored = ActionData.bytes(
                primitive._run_camel_stage_cli(command, payload)
            )
    elif len(stored_names) == 1 and stored_names[0] in primitive.SNAPPY_ACTIONS_BY_NAME:
        command = {
            primitive.SNAPPY_ACTIONS[0]: "lz-inverse",
            primitive.SNAPPY_ACTIONS[1]: "token-inverse",
            primitive.SNAPPY_ACTIONS[2]: "integer-inverse",
            primitive.SNAPPY_LEGACY_INTEGER_ACTION.name: "integer-inverse",
        }[stored_names[0]]
        restored = ActionData.bytes(primitive._run_snappy_stage_cli(command, payload))
    elif len(stored_names) == 1 and stored_names[0] in primitive.LZ4_ACTIONS_BY_NAME:
        command = {
            primitive.LZ4_ACTIONS[0]: "lz-inverse",
            primitive.LZ4_ACTIONS[1]: "token-inverse",
            primitive.LZ4_ACTIONS[2]: "integer-inverse",
            primitive.LZ4_ACTIONS[3]: "container-inverse",
        }[stored_names[0]]
        restored = ActionData.bytes(primitive._run_lz4_stage_cli(command, payload))
    elif (
        len(stored_names) == 1
        and stored_names[0] in primitive.DEFLATE_CONTAINER_ACTIONS_BY_NAME
    ):
        command = {
            primitive.GZIP_CONTAINER_ACTIONS[0]: "gzip-inverse",
            primitive.ZLIB_CONTAINER_ACTIONS[0]: "zlib-inverse",
        }[stored_names[0]]
        restored = ActionData.bytes(
            primitive._run_deflate_container_stage_cli(command, payload)
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.XZ_CONTAINER_ACTIONS_BY_NAME:
        restored = ActionData.bytes(
            primitive._run_xz_container_stage_cli("inverse", payload)
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.FPZIP_ACTIONS_BY_NAME:
        if stored_names[0] == primitive.FPZIP_ACTIONS[0]:
            restored = ActionData(
                "float64", primitive._run_fpzip_stage_cli("predictor-inverse", payload), count
            )
        else:
            restored = ActionData.bytes(
                primitive._run_fpzip_stage_cli("range-inverse", payload)
            )
    elif (
        len(stored_names) == 1
        and stored_names[0] in (
            set(primitive.LEARNING_CODEC_STAGE_ACTIONS_BY_NAME)
            | {primitive.LECO_LEGACY_PREDICTOR_ACTION.name,
               primitive.REGER_LEGACY_PREDICTOR_ACTION.name}
        )
    ):
        name = stored_names[0]
        command = {
            primitive.BUFF_FOR_ACTION.name: "buff-for-inverse",
            primitive.BUFF_PLANE_ACTION.name: "buff-byte-plane-inverse",
            primitive.LECO_ACTIONS[0]: "leco-regression-inverse",
            primitive.LECO_ACTIONS[1]: "leco-zigzag-inverse",
            primitive.LECO_ACTIONS[2]: "leco-bitpack-inverse",
            primitive.LECO_LEGACY_PREDICTOR_ACTION.name: "leco-autoregression-inverse",
            primitive.ALP_MAIN_ACTIONS[0]: "alp-scaling-inverse",
            primitive.ALP_MAIN_ACTIONS[1]: "alp-for-inverse",
            primitive.ALP_MAIN_ACTIONS[2]: "alp-bitpack-inverse",
            primitive.ALP_MAIN_ACTIONS[3]: "alp-residual-core-inverse",
            primitive.ALP_DICTIONARY_ACTIONS[0]: "alp-dictionary-inverse",
            primitive.REGER_ACTIONS[0]: "reger-reorder-inverse",
            primitive.REGER_ACTIONS[1]: "reger-delta-inverse",
            primitive.REGER_ACTIONS[2]: "reger-zigzag-inverse",
            primitive.REGER_ACTIONS[3]: "reger-for-inverse",
            primitive.REGER_ACTIONS[4]: "reger-bitpack-inverse",
            primitive.REGER_ACTIONS[5]: "reger-residual-core-inverse",
            primitive.REGER_LEGACY_PREDICTOR_ACTION.name: "reger-autoregression-inverse",
            primitive.BUFF_ACTIONS[0]: "buff-scaling-inverse",
            primitive.BUFF_ACTIONS[1]: "buff-column-split-inverse",
            primitive.BUFF_ACTIONS[2]: "buff-bitpack-inverse",
        }[name]
        raw = primitive._run_learning_codec_stage_cli(command, payload)
        restored_kind = {
            primitive.LECO_ACTIONS[0]: "int64",
            primitive.LECO_LEGACY_PREDICTOR_ACTION.name: "int64",
            primitive.ALP_MAIN_ACTIONS[0]: "float64",
            primitive.ALP_DICTIONARY_ACTIONS[0]: "float64",
            primitive.REGER_ACTIONS[0]: "int64",
            primitive.REGER_LEGACY_PREDICTOR_ACTION.name: "bytes",
            primitive.BUFF_ACTIONS[0]: "float64",
        }.get(name)
        restored = (
            ActionData(restored_kind, raw, count)
            if restored_kind is not None
            else ActionData.bytes(raw)
        )
    elif (
        len(stored_names) == 1
        and stored_names[0] in primitive.GENERAL_CODEC_REFERENCE_ACTIONS_BY_NAME
    ):
        name = stored_names[0]
        command = {
            primitive.BROTLI_REFERENCE_ACTIONS[0]: "dictionary-inverse",
            primitive.BROTLI_REFERENCE_ACTIONS[1]: "lz-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[2].name: "huffman-inverse",
            primitive.ZSTD_REFERENCE_ACTIONS[0]: "lz-inverse",
            primitive.ZSTD_REFERENCE_ACTIONS[1]: "parse-inverse",
            primitive.ZSTD_REFERENCE_ACTIONS[2]: "huffman-inverse",
            primitive.ZSTD_REFERENCE_ACTIONS[3]: "rans-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[7].name: "shuffle-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[8].name: "shuffle-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[9].name: "shuffle-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[10].name: "lz-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[11].name: "token-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[12].name: "blosc-frame-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[13].name: "blosc2-frame-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[14].name: "bwt-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[15].name: "mtf-inverse",
            primitive.GENERAL_CODEC_REFERENCE_ACTIONS[16].name: "rle-inverse",
            primitive.DEFLATE_REFERENCE_LZ_ACTION: "lz-inverse",
            primitive.XZ_REFERENCE_LZ_ACTION: "lz-inverse",
            primitive.REFERENCE_RANGE_ACTION: "range-inverse",
            primitive.GZIP_REFERENCE_FRAME_ACTION: "gzip-frame-inverse",
            primitive.ZLIB_REFERENCE_FRAME_ACTION: "zlib-frame-inverse",
            primitive.XZ_REFERENCE_FRAME_ACTION: "xz-frame-inverse",
        }[name]
        restored = ActionData.bytes(
            primitive._run_general_codec_reference_stage_cli(command, payload)
        )
    elif (
        len(stored_names) == 1
        and stored_names[0] in primitive.GENERAL_CODEC_RESIDUAL_ACTIONS_BY_NAME
    ):
        name = stored_names[0]
        command = {
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[0].name: "brotli-inverse",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[1].name: "zstd-inverse",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[2].name: "blosc1-inverse-1",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[3].name: "blosc1-inverse-4",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[4].name: "blosc1-inverse-8",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[5].name: "blosc2-inverse-1",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[6].name: "blosc2-inverse-4",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[7].name: "blosc2-inverse-8",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[8].name: "bzip2-inverse",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[9].name: "gzip-inverse",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[10].name: "zlib-inverse",
            primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[11].name: "xz-inverse",
        }[name]
        if name == primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[8].name:
            optimal = _has_optimal_bzip2_prefix(upstream_header)
            if header.get('rle_prefix_version') not in (None, 'URL1') or (header.get('rle_prefix_version') == 'URL1') != optimal:
                raise ActionSequenceError('Bzip2 RLE prefix metadata disagrees with recorded lineage')
            if optimal: command = 'bzip2-optimal-inverse'
        restored = ActionData.bytes(
            primitive._run_general_codec_residual_stage_cli(command, payload)
        )
    elif len(stored_names) == 1 and stored_names[0] in primitive.SIMPLE8B_FLOAT_ACTIONS:
        name = stored_names[0]
        if name == primitive.SIMPLE8B_FLOAT_ACTIONS[0]:
            restored = ActionData(
                "float64",
                primitive._run_simple8b_stage_cli("scale-inverse", payload),
                count,
            )
        elif kind == "bytes":
            restored = ActionData.bytes(
                primitive._run_simple8b_stage_cli("bitpack-inverse-scaled", payload)
            )
        else:
            raise ActionTypeError("invalid stored source kind for Simple8b bitpack")
    elif len(stored_names) == 1 and stored_names[0] in primitive.SVB_INT_ACTIONS:
        name = stored_names[0]
        if name == primitive.SVB_INT_ACTIONS[0]:
            if kind == "int64":
                restored = ActionData(
                    "int64",
                    primitive._run_svb_stage_cli("delta-inverse", payload),
                    count,
                )
            elif kind == "bytes":
                restored = ActionData.bytes(
                    primitive._run_svb_stage_cli("delta-inverse-scaled", payload)
                )
            else:
                raise ActionTypeError("invalid stored source kind for SVB delta")
        else:
            restored = ActionData.bytes(
                primitive._run_svb_stage_cli("integer-code-inverse", payload)
            )
    elif len(stored_names) == 1 and stored_names[0] in primitive.TS2DIFF_FLOAT_ACTIONS:
        name = stored_names[0]
        if name == primitive.TS2DIFF_FLOAT_ACTIONS[0]:
            raw = primitive._run_ts2diff_stage_cli("scale-inverse", payload)
            restored = ActionData("float64", raw, count)
        elif name == primitive.TS2DIFF_FLOAT_ACTIONS[1]:
            if kind == "int64":
                raw = primitive._run_ts2diff_stage_cli("delta-inverse-int", payload)
                restored = ActionData("int64", raw, count)
            elif kind == "bytes":
                raw = primitive._run_ts2diff_stage_cli("delta-inverse-scaled", payload)
                restored = ActionData.bytes(raw)
            else:
                raise ActionTypeError("invalid stored source kind for TS_2DIFF delta")
        elif name == primitive.TS2DIFF_FLOAT_ACTIONS[2]:
            restored = ActionData.bytes(
                primitive._run_ts2diff_stage_cli("for-inverse", payload)
            )
        else:
            command = (
                "bitpack-inverse-float"
                if root_kind == "float64"
                else "bitpack-inverse-int"
            )
            restored = ActionData.bytes(
                primitive._run_ts2diff_stage_cli(command, payload, max_point)
            )
    elif stored_names == primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS:
        restored = ActionData(
            "float64", primitive._run_rle_stage_cli("decompose", payload), count
        )
    elif stored_names == ("op.entropy_huffman.project_huffman_bytes.v1",):
        restored = ActionData.bytes(primitive.huffman_decode(payload))
    elif stored_names == ("op.entropy_range.project_arithmetic_bytes.v1",):
        restored = ActionData.bytes(primitive.arithmetic_decode(payload))
    elif stored_names == ("op.entropy_fse.project_rans_bytes.v1",):
        restored = ActionData.bytes(primitive.ans_decode(payload))
    elif stored_names == ("op.dictionary.project_int64.v1",):
        restored = ActionData.ints(primitive.decode_dictionary_int(payload))
    elif stored_names == ("op.dictionary.project_float64.v1",):
        restored = ActionData.floats(primitive.decode_dictionary_float(payload))
    elif stored_names == ("op.run_length.tsfile_int64.v1",):
        if upstream_header is not None and root_kind == "float64":
            restored = ActionData.bytes(
                primitive._run_rle_stage_cli("run-inverse", payload)
            )
        else:
            restored = ActionData(
                "int64",
                primitive._run_rle_int64_action_cli("decode", payload),
                count,
            )
    elif len(stored_names) == 1 and stored_names[0] in primitive.EXACT_NATIVE_ACTION_BY_NAME:
        context = header.get("native_lz_stage_context")
        if context is not None and not isinstance(context, dict):
            raise ActionRuntimeError("invalid native-LZ stage context metadata")
        try:
            restored = ActionData.bytes(
                primitive.decode_exact_native_lz_stage(
                    stored_names[0], payload, context=context
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionRuntimeError(str(exc)) from exc
    elif stored_names in primitive._EXACT_NATIVE_PIPELINE_KEY:
        algorithm_key = primitive._EXACT_NATIVE_PIPELINE_KEY[stored_names]
        restored = ActionData.bytes(
            primitive.decompress_native_lz_family(payload, algorithm_key, count)
        )
    else:
        raise ActionSequenceError("stored action sequence has no decoder")

    if upstream_header is not None:
        if restored.kind != "bytes":
            raise ActionRuntimeError(
                "a downstream stage inverse did not recover stage-packet bytes"
            )
        restored = _pack_existing_result(upstream_header, restored.payload)

    if restored.kind != kind or restored.count != count:
        raise ActionRuntimeError(
            f"decoded shape mismatch: expected {kind}[{count}], "
            f"got {restored.kind}[{restored.count}]"
        )
    if stored_names == primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS and max_point is None:
        raise ActionRuntimeError("float RLE result omitted max_point metadata")
    return restored


def run_actions(
    actions: Sequence[ActionRef],
    data: ActionData,
    *,
    reverse: bool = False,
    max_point: int | None = None,
    io_variants: Sequence[str | None] | None = None,
    parameters: Sequence[dict[str, int] | None] | dict[str, int] | None = None,
    _concrete: bool = False,
) -> ActionData:
    """Run one or more action names through the same typed API."""

    data = _wire_data(data)
    if not reverse:
        for selected in actions:
            name = selected if isinstance(selected, str) else None
            if name is None and type(selected) is int:
                descriptor = ACTIONS[selected] if 0 <= selected < len(ACTIONS) else None
                name = descriptor.name if descriptor is not None else None
            if primitive.ACTION_CANONICAL_NAMES.get(name, name) in primitive.RETIRED_IDENTITY_ACTION_NAMES:
                raise ActionNotExecutableError(
                    "identity Scaling is retired; omit the stage from new recipes"
                )
            if primitive.ACTION_CANONICAL_NAMES.get(name, name) in primitive.RETIRED_COMPOSITE_ACTION_NAMES:
                raise ActionNotExecutableError(
                    "this historical composite or duplicate implementation is retired; "
                    "migrate the versioned recipe to its current atomic actions"
                )
    if reverse and len(actions) > 1 and all(isinstance(a, str) for a in actions):
        recorded, _ = _unpack_result(data)
        if tuple(recorded.get('actions', ())) == tuple(actions):
            return _decode(tuple(actions), data)
    if isinstance(parameters, dict):
        if len(actions) != 1:
            raise ActionSequenceError("a parameter object requires exactly one action")
        parameters = (parameters,)
    if parameters is not None and (isinstance(parameters, (str, bytes)) or
                                    len(parameters) != len(actions)):
        raise ActionSequenceError("one parameter object is required per selected action")
    if not _concrete and (io_variants is not None or parameters is not None or any(
        isinstance(action, str) and action not in optimal_rle.OLD_TO_CURRENT and (action in primitive.algorithm_io.MERGED_NAMES or
            (action in primitive.algorithm_io.PREVIOUS_ALGORITHM_TO_CURRENT and action not in primitive.algorithm_io.VARIANT_BY_NAME) or
            action_parameters.module_for_name(action) is not None or action == primitive.BUFF_PLANE_ACTION.name)
        for action in actions
    )):
        variants = tuple(io_variants) if io_variants is not None else (None,) * len(actions)
        if len(variants) != len(actions):
            raise ActionSequenceError("one I/O variant is required per selected action")
        current = data
        parameter_sets = tuple(parameters) if parameters is not None else (None,) * len(actions)
        steps = tuple(zip(actions, variants, parameter_sets))
        for action, variant, selected_parameters in reversed(steps) if reverse else steps:
            current = run_policy_action(action, current,
                registry_version=primitive.ACTION_REGISTRY_VERSION,
                io_variant=variant, parameters=selected_parameters,
                reverse=reverse, max_point=max_point)
        return current
    descriptors = _resolve_actions(actions)
    names = tuple(action.name for action in descriptors)
    if reverse and len(names) > 1:
        stored_header, _ = _unpack_result(data)
        if tuple(stored_header.get('actions', ())) == names:
            return _decode(names, data)
    if len(names) > 1 and names in _INDEPENDENT_STAGEWISE_SEQUENCES:
        current = data
        ordered = reversed(names) if reverse else names
        for name in ordered:
            current = run_actions(
                (name,), current,
                reverse=reverse,
                max_point=max_point,
                _concrete=_concrete,
            )
        return current
    if reverse:
        return _decode(names, data)
    encode_source = data
    upstream_header: dict[str, Any] | None = None
    if len(descriptors) == 1 and (
        descriptors[0].execution_mode in {"stagewise", "experimental_non_atomic"}
        or descriptors[0].name in optimal_rle.VARIANTS
        or descriptors[0].name in {"op.run_length.tsfile_int64.v1", bitpack.PLAIN_VARIANT}
    ):
        encode_source, upstream_header = _unwrap_stage_source(data)
        if max_point is None and upstream_header is not None:
            max_point = _inherited_max_point(upstream_header)
    extra_metadata: dict[str, Any] | None = None
    if names == (primitive.legacy_numeric_stages.FRAME,):
        packet = primitive.legacy_numeric_stages._read(encode_source.payload, "wires")
        extra_metadata = {"legacy_numeric_count": packet["blocks"][0]["count"]}
    if (
        len(names) == 1
        and names[0] in primitive.EXACT_NATIVE_ACTION_BY_NAME
        and primitive.exact_native_lz_stage_role(names[0]) == "token_format"
    ):
        try:
            context = primitive.native_lz_stage_packet_context(encode_source.payload)
        except ValueError as exc:
            raise ActionSequenceError(str(exc)) from exc
        extra_metadata = {"native_lz_stage_context": context}
    if names == (primitive.GENERAL_CODEC_RESIDUAL_ACTIONS[8].name,) and _has_optimal_bzip2_prefix(upstream_header):
        encoded = primitive._run_general_codec_residual_stage_cli('bzip2-optimal-forward', encode_source.payload)
        extra_metadata = dict(extra_metadata or {}, rle_prefix_version='URL1')
    else:
        encoded = _encode(names, encode_source, max_point)
    return _pack_result(
        names,
        data,
        encoded,
        max_point,
        upstream_header=upstream_header,
        stage_input_payload_count=encode_source.count,
        extra_metadata=extra_metadata,
    )


def run_action(
    action: ActionRef,
    data: ActionData,
    *,
    reverse: bool = False,
    max_point: int | None = None,
    io_variant: str | None = None,
    parameters: dict[str, int] | None = None,
    output: Literal["wire", "streams"] = "wire",
) -> Any:
    """Convenience wrapper for a single action."""

    if action == 'op.run_length.value_count.cpp_v2':
        return run_policy_action(65, data, registry_version='cpp-canonical-129-v1',
                                 io_variant=io_variant, parameters=parameters,
                                 output=output, reverse=reverse, max_point=max_point)
    if action == optimal_rle.OLD_SPRINTZ_VARIANT and reverse:
        recorded, _ = _unpack_result(_wire_data(data))
        if recorded.get('actions') == [optimal_rle.OLD_SPRINTZ_VARIANT]:
            return _decode((optimal_rle.OLD_SPRINTZ_VARIANT,), _wire_data(data))

    if (io_variant is not None or parameters is not None or output != "wire" or
        isinstance(action, str) and action not in optimal_rle.OLD_TO_CURRENT and (action in primitive.algorithm_io.MERGED_NAMES or
            (action in primitive.algorithm_io.PREVIOUS_ALGORITHM_TO_CURRENT and action not in primitive.algorithm_io.VARIANT_BY_NAME) or
            action_parameters.module_for_name(action) is not None)):
        return run_policy_action(action, data, registry_version=primitive.ACTION_REGISTRY_VERSION,
                                 io_variant=io_variant, parameters=parameters, output=output,
                                 reverse=reverse, max_point=max_point)
    return run_actions((action,), data, reverse=reverse, max_point=max_point)


def _normalize_parameters(parameters: Any, parameter_module=lz_parameters, *, io_variant=None) -> dict[str, int]:
    try:
        return (parameter_module.normalize_parameters(parameters, io_variant=io_variant)
                if parameter_module in block_parameters.MODULES else parameter_module.normalize_parameters(parameters))
    except (TypeError, ValueError) as exc:
        raise ActionTypeError(str(exc)) from exc


def _requested_lz_parameters(action: ActionRef, io_variant: str | None,
                             parameters: dict[str, int] | None, parameter_module=lz_parameters) -> dict[str, int] | None:
    """Preserve exact historical bounds before canonicalizing action names."""
    selected = None if parameters is None else _normalize_parameters(parameters, parameter_module)
    for name in (action, io_variant):
        old = parameter_module.parameters_from_name(name) if isinstance(name, str) else None
        if old is not None:
            if selected is not None and selected != old:
                raise ActionSequenceError("parameters conflict with historical native-LZ selector")
            selected = old
    return selected


def _stored_lz_parameters(header: dict[str, Any], stage_name: str, parameter_module=lz_parameters) -> dict[str, int]:
    historical = parameter_module.parameters_from_name(stage_name)
    if "action_parameters" in header:
        stored = header["action_parameters"]
        if not isinstance(stored, dict) or set(stored) != set(parameter_module.parameter_candidates()[0]):
            raise ActionTypeError("native-LZ action_parameters do not match the stage parameter schema")
        selected = _normalize_parameters(stored, parameter_module)
        if historical is not None and selected != historical:
            raise ActionSequenceError("stored parameters disagree with historical stage name")
        return selected
    if (stage_name == parameter_module.ALGORITHM_NAME or historical is None or
            header.get("policy_registry_version") == primitive.ACTION_REGISTRY_VERSION):
        raise ActionTypeError("parameterized native-LZ result omitted action_parameters")
    return historical


def _run_lz_parameter_stage(source: ActionData, parameters: dict[str, int], *,
                            reverse: bool, max_point: int | None = None, parameter_module=lz_parameters) -> ActionData:
    """Invoke the one C++ matcher with scalar parameters, preserving NLS1 context."""
    exact_name = parameter_module.legacy_action_name(parameters)
    if reverse:
        header, payload = _unpack_result(source)
        if header.get("source_kind") != "bytes" or type(header.get("source_count")) is not int:
            raise ActionRuntimeError("native-LZ source metadata requires bytes and an integer count")
        upstream = header.get("upstream_result_header")
        if upstream is not None and not isinstance(upstream, dict):
            raise ActionRuntimeError("invalid upstream action result metadata")
        try:
            restored = ActionData.bytes(primitive.decode_exact_native_lz_stage(exact_name, payload))
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionRuntimeError(str(exc)) from exc
        if upstream is not None:
            restored = _pack_existing_result(upstream, restored.payload)
        if restored.count != header["source_count"]:
            raise ActionRuntimeError("decoded native-LZ source count mismatch")
        return restored
    stage_source, upstream = _unwrap_stage_source(source)
    if stage_source.kind != "bytes":
        raise ActionTypeError("native-LZ stages require byte input")
    try:
        encoded = primitive.encode_exact_native_lz_stage(exact_name, stage_source.payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionSequenceError(str(exc)) from exc
    return _pack_result((parameter_module.ALGORITHM_NAME,), source, encoded, max_point,
                        upstream_header=upstream, stage_input_payload_count=stage_source.count,
                        extra_metadata={"action_parameters": parameters})


def _resolve_block_parameters(parameter_module, action, io_variant, parameters):
    try:
        return parameter_module.resolve_parameters(action, io_variant, parameters)
    except (ValueError, TypeError) as exc:
        raise ActionTypeError(str(exc)) from exc


def _native_block_run(name, payload, parameters, io_variant, reverse=False):
    try:
        return block_parameters.run(name, payload, parameters, io_variant=io_variant, reverse=reverse)
    except (ValueError, TypeError) as exc:
        raise ActionSequenceError(str(exc)) from exc


def _run_block_parameter_stage(source, parameters, *, reverse, parameter_module, io_variant, max_point=None):
    name = parameter_module.ALGORITHM_NAME
    if reverse:
        header, payload = _unpack_result(source)
        if header.get("source_kind") != "bytes" or type(header.get("source_count")) is not int:
            raise ActionRuntimeError("block codec source metadata requires bytes and integer count")
        restored = ActionData.bytes(_native_block_run(name, payload, parameters, io_variant, reverse=True))
        upstream = header.get("upstream_result_header")
        if upstream is not None:
            if not isinstance(upstream, dict): raise ActionRuntimeError("invalid upstream metadata")
            restored = _pack_existing_result(upstream, restored.payload)
        if restored.count != header["source_count"]:raise ActionRuntimeError("decoded block source count mismatch")
        return restored
    stage_source, upstream = _unwrap_stage_source(source)
    if stage_source.kind != "bytes":raise ActionTypeError("block codec adapters require bytes input")
    encoded = _native_block_run(name, stage_source.payload, parameters, io_variant)
    return _pack_result((io_variant,), source, encoded, max_point, upstream_header=upstream,
                        stage_input_payload_count=stage_source.count, extra_metadata={"action_parameters": parameters})


def _frozen_rle_source(action_id, registry_version):
    """Read a historical identity without upgrading it to the current RLE."""
    import re
    if type(action_id) is not int: return None
    if registry_version in ('cpp-canonical-206-v1', 'cpp-canonical-225-v1', 'legacy-225-v1'):
        specs = primitive.LEGACY_206_POLICY_ACTIONS if registry_version == 'cpp-canonical-206-v1' else primitive.LEGACY_225_POLICY_ACTIONS
        if not 0 <= action_id < len(specs): return None
        name = specs[action_id].name
        variant = primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(name, name)
        return {'name': name, 'io_variants': [{'name': variant}]} if variant in optimal_rle.OLD_TO_CURRENT or variant == optimal_rle.OLD_SPRINTZ_VARIANT else None
    match = re.fullmatch(r'cpp-canonical-(\d+)-v1', registry_version)
    if not match or registry_version == primitive.ACTION_REGISTRY_VERSION: return None
    path = ROOT / f'benchmark/reference/actions/canonical{match.group(1)}.json'
    if not path.is_file(): return None
    groups = json.loads(path.read_text())['actions']
    if not 0 <= action_id < len(groups): return None
    group = groups[action_id]
    return group if group['name'] in optimal_rle.OLD_ALGORITHMS or any(v['name'] == optimal_rle.OLD_SPRINTZ_VARIANT for v in group['io_variants']) else None


def _frozen_identity_source(action, registry_version):
    """Resolve only the retired identity slot in an explicitly named old registry."""
    import re
    if isinstance(action, str):
        canonical = primitive.ACTION_CANONICAL_NAMES.get(action, action)
        return {"name": canonical, "io_variants": [{"name": canonical}]} if canonical in primitive.RETIRED_IDENTITY_ACTION_NAMES else None
    if type(action) is not int:
        return None
    if registry_version == primitive.ACTION_REGISTRY_VERSION:
        spec = primitive.FROZEN_POLICY_ACTION_BY_ID.get(action)
        if spec is None:
            return None
        name = spec.name
        group = primitive.algorithm_io.GROUP_BY_NAME[name]
    elif registry_version in ('cpp-canonical-206-v1', 'cpp-canonical-225-v1', 'legacy-225-v1'):
        specs = (primitive.LEGACY_206_POLICY_ACTIONS if registry_version == 'cpp-canonical-206-v1'
                 else primitive.LEGACY_225_POLICY_ACTIONS)
        if not 0 <= action < len(specs):
            return None
        name = primitive.ACTION_CANONICAL_NAMES.get(specs[action].name, specs[action].name)
        group = {"name": name, "io_variants": [{"name": specs[action].name}]}
    else:
        match = re.fullmatch(r'cpp-canonical-(\d+)-v1', registry_version)
        path = ROOT / f'benchmark/reference/actions/canonical{match.group(1)}.json' if match else None
        if path is None or not path.is_file():
            return None
        groups = json.loads(path.read_text())['actions']
        if not 0 <= action < len(groups):
            return None
        group = groups[action]
        name = primitive.ACTION_CANONICAL_NAMES.get(group['name'], group['name'])
    return group if name in primitive.RETIRED_IDENTITY_ACTION_NAMES else None


def _frozen_composite_source(action, registry_version):
    """Resolve retired fused/duplicate selectors only for decode or rejection."""
    if isinstance(action, str):
        canonical = primitive.ACTION_CANONICAL_NAMES.get(action, action)
        if canonical in primitive.RETIRED_COMPOSITE_ACTION_NAMES:
            return {"name": canonical, "io_variants": [{"name": action}]}
        return None
    if type(action) is not int:
        return None
    try:
        migrated = primitive.migrate_policy_action(action, registry_version)
    except (TypeError, ValueError):
        return None
    if migrated.get("migration_kind") != "replacement_recipe":
        return None
    name = migrated["source_policy_action_name"]
    return {"name": name, "io_variants": [{"name": name}]}


def _run_frozen_composite(action, version, group, data, *, reverse, io_variant,
                          parameters, output):
    if not reverse:
        raise ActionNotExecutableError(
            "historical composite/duplicate actions are decode-only; migrate the "
            "versioned recipe to its current atomic replacement"
        )
    if parameters not in (None, {}):
        raise ActionTypeError("historical composite stages have no parameters")
    if output not in ("wire", "streams"):
        raise ActionTypeError("output must be wire or streams")
    source = _wire_data(data)
    header, _ = _unpack_result(source)
    stored = header.get("actions")
    if not isinstance(stored, list) or len(stored) != 1:
        raise ActionSequenceError("historical composite inverse requires one recorded stage")
    stored_name = primitive.ACTION_CANONICAL_NAMES.get(stored[0], stored[0])
    if stored_name != group["name"]:
        raise ActionSequenceError("historical composite source does not match the selector")
    if io_variant is not None:
        selected = primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(io_variant, io_variant)
        if selected != stored_name and selected != stored[0]:
            raise ActionSequenceError("historical composite I/O profile mismatch")
    if "policy_registry_version" in header:
        if (header["policy_registry_version"] != version
                or type(action) is not int
                or header.get("policy_action_id") != action):
            raise ActionSequenceError("historical composite source version/ID mismatch")
    result = run_actions((stored[0],), source, reverse=True, _concrete=True)
    if output == "streams":
        from action_semantic_ports import to_ports
        return to_ports(result)
    return result


def _run_frozen_identity(action, version, group, data, *, reverse, io_variant,
                         parameters, output):
    if version == primitive.ACTION_REGISTRY_VERSION:
        raise ActionNotExecutableError(
            "policy slot 5 is retired in the current registry; use an explicit "
            "frozen registry version only to decode an old frame"
        )
    if not reverse:
        raise ActionNotExecutableError(
            "identity Scaling is retired; omit the stage from new recipes"
        )
    if parameters not in (None, {}) or io_variant not in (None, "op.scaling.reference_v1"):
        raise ActionTypeError("retired identity Scaling has no parameters or alternate I/O profile")
    if output not in ('wire', 'streams'):
        raise ActionTypeError('output must be wire or streams')
    source = _wire_data(data)
    header, _ = _unpack_result(source)
    stored = header.get('actions')
    if (not isinstance(stored, list) or len(stored) != 1
            or primitive.ACTION_CANONICAL_NAMES.get(stored[0], stored[0])
            not in primitive.RETIRED_IDENTITY_ACTION_NAMES):
        raise ActionSequenceError('retired identity inverse requires its recorded stage')
    if 'policy_registry_version' in header:
        if (header['policy_registry_version'] != version
                or type(action) is not int
                or header.get('policy_action_id') != action):
            raise ActionSequenceError('retired identity source version/ID mismatch')
    result = run_actions((stored[0],), source, reverse=True, _concrete=True)
    if output == 'streams':
        from action_semantic_ports import to_ports
        return to_ports(result)
    return result


def _historical_policy_selector(name, version, io_variant):
    """Resolve strings only inside their explicitly named frozen namespace."""
    import re
    if version in ('cpp-canonical-206-v1', 'cpp-canonical-225-v1', 'legacy-225-v1'):
        specs = primitive.LEGACY_206_POLICY_ACTIONS if version == 'cpp-canonical-206-v1' else primitive.LEGACY_225_POLICY_ACTIONS
        matches = [s for s in specs if s.name == name]
        if len(matches) != 1: raise ActionRuntimeError('name is not in the requested historical policy registry')
        return matches[0].action_id, io_variant
    if version == primitive.PREVIOUS_ACTION_REGISTRY_VERSION:
        groups = [{**g, "policy_action_id":g["action_id"], "io_variants":[{"name":v} for v in primitive.V3_POLICY_SNAPSHOT["io_variants"][g["name"]]]} for g in primitive.V3_POLICY_SNAPSHOT["actions"]]
        matches = [g for g in groups if g['name'] == name or any(v['name'] == name for v in g['io_variants'])]
        if len(matches) != 1: raise ActionRuntimeError('name is not in frozen v3')
        group=matches[0]
        if name != group['name']:
            if io_variant is not None and io_variant != name: raise ActionTypeError('historical name conflicts with I/O')
            io_variant=name
        return group['policy_action_id'],io_variant
    if version == "cpp-canonical-127-v2":
        groups = primitive.algorithm_io.GROUPS
        matches = [g for g in groups if g['name'] == name or any(v['name'] == name for v in g['io_variants'])]
        if len(matches) != 1: raise ActionRuntimeError('name is not in the requested historical policy registry')
        group = matches[0]
        if name != group['name']:
            if io_variant is not None and io_variant != name: raise ActionTypeError('historical name conflicts with requested I/O')
            io_variant = name
        return group['policy_action_id'], io_variant
    match = re.fullmatch(r'cpp-canonical-(\d+)-v1', version)
    path = ROOT / f'benchmark/reference/actions/canonical{match.group(1)}.json' if match else None
    if path is None or not path.is_file(): raise ActionRuntimeError('unknown historical policy registry')
    groups = json.loads(path.read_text())['actions']
    matches = [g for g in groups if g['name'] == name or any(v['name'] == name for v in g['io_variants'])]
    if len(matches) != 1: raise ActionRuntimeError('name is not in the requested historical policy registry')
    group = matches[0]
    if name != group['name']:
        if io_variant is not None and io_variant != name: raise ActionTypeError('historical name conflicts with requested I/O')
        io_variant = name
    return group['policy_action_id'], io_variant


def _run_frozen_rle(action_id, version, group, data, *, reverse, io_variant, parameters, output, max_point):
    if parameters not in (None, {}): raise ActionTypeError('historical RLE has no parameters')
    if output not in ('wire', 'streams'): raise ActionTypeError('output must be wire or streams')
    source = _wire_data(data)
    choices = tuple(v['name'] for v in group['io_variants'])
    variant = primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(io_variant, io_variant) if io_variant is not None else None
    if reverse:
        header, _ = _unpack_result(source)
        names = header.get('actions')
        if not isinstance(names, list) or len(names) != 1:
            raise ActionSequenceError('historical RLE inverse requires one recorded stage')
        stored = primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(names[0], names[0])
        if stored not in choices or variant is not None and variant != stored:
            raise ActionSequenceError('historical RLE I/O does not match source registry')
        if 'policy_registry_version' in header and (header['policy_registry_version'] != version or header.get('policy_action_id') != action_id):
            raise ActionSequenceError('historical RLE source version/ID mismatch')
        if header.get('io_variant', stored) != stored or 'action_parameters' in header:
            raise ActionSequenceError('invalid historical RLE metadata')
        if header.get('policy_algorithm', group['name']) != group['name']:
            raise ActionSequenceError('historical RLE algorithm metadata mismatch')
        result = run_actions((stored,), source, reverse=True, _concrete=True)
    else:
        if variant is None:
            if len(choices) == 1: variant = choices[0]
            else:
                kind = _unwrap_stage_source(source)[0].kind
                variant = 'op.run_length.reference_v1' if kind == 'int64' else 'op.run_length.bzip2_uint8.cpp_reference_v1' if kind == 'bytes' else None
        if variant not in choices: raise ActionTypeError('historical RLE requires its original I/O profile')
        result = run_actions((variant,), source, max_point=max_point, _concrete=True)
        header, payload = _unpack_result(result)
        stage_source = _unwrap_stage_source(source)[0]
        header.update(policy_registry_version=version, policy_action_id=action_id,
                      policy_algorithm=group['name'], io_variant=variant,
                      io_input_kind=stage_source.kind,
                      io_input_magic=stage_source.payload[:4].hex() if stage_source.kind == 'bytes' else None)
        result = _pack_existing_result(header, payload)
    if output == 'streams':
        from action_semantic_ports import to_ports
        return to_ports(result)
    return result


def run_policy_action(action: ActionRef, data: ActionData, *,
                      registry_version: str, reverse: bool = False,
                      max_point: int | None = None,
                      io_variant: str | None = None,
                      parameters: dict[str, int] | None = None,
                      output: Literal["wire", "streams"] = "wire") -> Any:
    """Run a versioned algorithm with its I/O profile and optional parameters.

    Native-LZ min/max are parameters of one matcher. Historical names preserve
    their original bounds. ``streams`` exposes rebuildable typed components.
    ``run_action(int)`` continues to use historical descriptor IDs.
    """
    if registry_version != primitive.ACTION_REGISTRY_VERSION and isinstance(action, str):
        action, io_variant = _historical_policy_selector(action, registry_version, io_variant)
    if (registry_version == primitive.ACTION_REGISTRY_VERSION and isinstance(action, str)
            and action in optimal_rle.OLD_ALGORITHMS | optimal_rle.OLD_TO_CURRENT.keys()):
        raise ActionTypeError('historical RLE name requires its frozen registry; current replacement uses byte_optimal name and profiles')
    frozen_identity = _frozen_identity_source(action, registry_version)
    if frozen_identity is not None:
        return _run_frozen_identity(
            action, registry_version, frozen_identity, data, reverse=reverse,
            io_variant=io_variant, parameters=parameters, output=output,
        )
    frozen_composite = _frozen_composite_source(action, registry_version)
    if frozen_composite is not None:
        return _run_frozen_composite(
            action, registry_version, frozen_composite, data, reverse=reverse,
            io_variant=io_variant, parameters=parameters, output=output,
        )
    frozen = _frozen_rle_source(action, registry_version)
    if frozen is not None:
        return _run_frozen_rle(action, registry_version, frozen, data, reverse=reverse,
            io_variant=io_variant, parameters=parameters, output=output, max_point=max_point)
    if reverse and registry_version == primitive.ACTION_REGISTRY_VERSION:
        header, _ = _unpack_result(_wire_data(data))
        old_version, old_id = header.get('policy_registry_version'), header.get('policy_action_id')
        frozen = _frozen_rle_source(old_id, old_version) if isinstance(old_version, str) else None
        if frozen is not None:
            expected_algorithm = primitive.ACTION_CANONICAL_NAMES.get(frozen['name'], frozen['name'])
            if describe_policy_action(action).name != expected_algorithm:
                raise ActionSequenceError('historical RLE frame requested through another algorithm')
            return _run_frozen_rle(old_id, old_version, frozen, data, reverse=True,
                io_variant=io_variant, parameters=parameters, output=output, max_point=max_point)
    if registry_version != primitive.ACTION_REGISTRY_VERSION:
        if type(action) is not int:
            raise ActionRuntimeError("historical policy invocation requires its numeric ID and registry version")
        if reverse and parameters is None:
            stored_header, _ = _unpack_result(_wire_data(data))
            # A reverse call without overrides uses the encoded parameter
            # record; upgrading its ID must not replace it with a new default.
            if isinstance(stored_header.get('action_parameters'), dict):
                parameters = stored_header['action_parameters']
        try:
            migrated = primitive.migrate_policy_action(action, registry_version, io_variant=io_variant, parameters=parameters)
        except ValueError as exc:
            raise ActionRuntimeError(str(exc)) from exc
        if migrated.get("migration_kind") == "replacement_recipe":
            raise ActionNotExecutableError(
                "historical selector expands to multiple current actions; migrate the recipe before execution"
            )
        action, io_variant, parameters = migrated["policy_action_id"], migrated["io_variant"], migrated["parameters"]
        registry_version = primitive.ACTION_REGISTRY_VERSION
    if output not in {"wire", "streams"}:
        raise ActionTypeError("output must be 'wire' or 'streams'")
    algorithm = describe_policy_action(action)
    if algorithm.name == primitive.BUFF_PLANE_ACTION.name:
        return _run_buff_plane_policy(algorithm, data, reverse=reverse, registry_version=registry_version,
            parameters=parameters, io_variant=io_variant, output=output, max_point=max_point)
    parameter_module = action_parameters.BY_NAME.get(algorithm.name)
    is_block = parameter_module in block_parameters.MODULES
    is_lz = parameter_module is not None
    if parameters not in (None, {}) and not is_lz:
        raise ActionTypeError("this algorithm does not declare parameters")
    requested_parameters = (_requested_lz_parameters(action, io_variant, parameters, parameter_module) if is_lz and not is_block else
        _resolve_block_parameters(parameter_module, action, io_variant, parameters) if is_block and (parameters is not None or parameter_module.parameters_from_name(action) is not None or parameter_module.parameters_from_name(io_variant) is not None) else None)
    source = _wire_data(data)
    if reverse:
        header, _ = _unpack_result(source)
        stored_names = header.get("actions")
        if not isinstance(stored_names, list) or len(stored_names) != 1:
            raise ActionSequenceError("policy inverse requires exactly one recorded I/O stage")
        recorded_variant = _normalize_io_variant(stored_names[0])
        variant = _normalize_io_variant(header.get("io_variant", recorded_variant))
        if variant != recorded_variant or variant not in primitive.POLICY_IO_VARIANTS[algorithm.name]:
            raise ActionSequenceError("stored I/O variant does not match the recorded stage")
        stored_version = header.get("policy_registry_version")
        migrated = None
        stored_id = header.get("policy_action_id")
        if "policy_registry_version" in header and stored_version != registry_version:
            if not isinstance(stored_version, str) or type(stored_id) is not int:
                raise ActionRuntimeError("stored policy registry metadata is invalid")
            try:
                migrated = primitive.migrate_policy_action(stored_id, registry_version=stored_version,
                    io_variant=header.get("io_variant"), parameters=header.get("action_parameters"))
            except (TypeError, ValueError) as exc:
                raise ActionRuntimeError(str(exc)) from exc
            if migrated["policy_action_id"] != algorithm.action_id:
                raise ActionSequenceError("stored policy ID does not match the requested inverse")
        elif "policy_action_id" in header and (type(stored_id) is not int or stored_id != algorithm.action_id):
            raise ActionSequenceError("stored policy ID does not match the requested inverse")
        stored_algorithm = header.get("policy_algorithm")
        if "policy_algorithm" in header:
            try:
                stored_algorithm = describe_policy_action(stored_algorithm).name
            except (KeyError, TypeError, ValueError) as exc:
                raise ActionSequenceError("invalid stored policy algorithm") from exc
            if stored_algorithm != algorithm.name:
                raise ActionSequenceError("stored policy algorithm does not match the requested inverse")
        if io_variant is not None and _normalize_io_variant(io_variant) != variant:
            raise ActionSequenceError("requested inverse I/O variant differs from the encoded profile")
        if (isinstance(action, str) and action not in CANONICAL_ACTION_BY_NAME and
            action != minimum.PREVIOUS_WHOLE_NAME and _normalize_io_variant(action) != variant):
            raise ActionSequenceError("requested historical profile differs from the encoded profile")
        if is_lz:
            selected_parameters = _stored_lz_parameters(header, stored_names[0], parameter_module)
            historical_algorithm_parameters = parameter_module.parameters_from_name(header.get("policy_algorithm"))
            if (historical_algorithm_parameters is not None and
                    historical_algorithm_parameters != selected_parameters):
                raise ActionSequenceError("stored algorithm selector disagrees with native-LZ parameters")
            if migrated is not None and migrated["parameters"] != selected_parameters:
                raise ActionSequenceError("historical policy ID disagrees with native-LZ parameters")
            if requested_parameters is not None and selected_parameters != requested_parameters:
                raise ActionSequenceError("requested parameters differ from encoded native-LZ parameters")
            result = (_run_block_parameter_stage(source, selected_parameters, reverse=True, parameter_module=parameter_module, io_variant=variant)
                      if is_block else _run_lz_parameter_stage(source, selected_parameters, reverse=True, parameter_module=parameter_module))
        else:
            if "action_parameters" in header:
                raise ActionTypeError("non-parameterized algorithm has unexpected action_parameters")
            result = run_actions((variant,), source, reverse=True, max_point=max_point, _concrete=True)
    else:
        variant = resolve_policy_io_variant(action, source, io_variant=io_variant)
        if is_lz:
            selected_parameters = (_normalize_parameters(requested_parameters, parameter_module, io_variant=variant) if is_block else _normalize_parameters(requested_parameters, parameter_module))
            result = (_run_block_parameter_stage(source, selected_parameters, reverse=False, parameter_module=parameter_module, io_variant=variant, max_point=max_point)
                      if is_block else _run_lz_parameter_stage(source, selected_parameters, reverse=False, max_point=max_point, parameter_module=parameter_module))
        else:
            result = run_actions((variant,), source, max_point=max_point, _concrete=True)
        header, payload = _unpack_result(result)
        stage_source = source if variant == minimum.PLAIN_VARIANT else _unwrap_stage_source(source)[0]
        header.update(policy_registry_version=registry_version,
                      policy_algorithm=algorithm.name, policy_action_id=algorithm.action_id,
                      io_variant=variant, io_input_kind=stage_source.kind,
                      io_input_magic=stage_source.payload[:4].hex() if stage_source.kind == "bytes" else None)
        result = _pack_existing_result(header, payload)
    if output == "streams":
        from action_semantic_ports import to_ports
        return to_ports(result)
    return result


def _run_buff_plane_policy(algorithm, data, *, reverse, registry_version, parameters, io_variant, output, max_point):
    name = primitive.BUFF_PLANE_ACTION.name
    if io_variant not in (None, name): raise ActionTypeError("byte-plane I/O profile mismatch")
    try: settings = primitive.normalize_buff_plane_parameters(parameters)
    except ValueError as exc: raise ActionTypeError(str(exc)) from exc
    source = _wire_data(data)
    if reverse:
        header,payload = _unpack_result(source)
        if (header.get("actions") != [name] or header.get("policy_registry_version") != registry_version
                or header.get("policy_action_id") != algorithm.action_id or header.get("io_variant") != name):
            raise ActionSequenceError("byte-plane result registry/action/profile mismatch")
        selected = header.get("action_parameters")
        if not isinstance(selected,dict) or set(selected) != {"bit_width"}:
            raise ActionTypeError("byte-plane result omitted explicit width")
        try: primitive.normalize_buff_plane_parameters(selected)
        except ValueError as exc: raise ActionTypeError(str(exc)) from exc
        if settings and settings != selected: raise ActionSequenceError("requested width differs from stored width")
        if len(payload)<19 or payload[18] != selected["bit_width"]:
            raise ActionSequenceError("byte-plane packet width disagrees with parameter")
        raw=primitive._run_learning_codec_stage_cli("buff-byte-plane-inverse",payload)
        upstream=header.get("upstream_result_header")
        result=_pack_existing_result(upstream,raw) if isinstance(upstream,dict) else ActionData.bytes(raw)
        if result.count != header.get("source_count"): raise ActionRuntimeError("byte-plane source count mismatch")
    else:
        stage_source,upstream=_unwrap_stage_source(source)
        if stage_source.kind != "bytes" or stage_source.payload[:4] != b"BFR1":
            raise ActionTypeError("byte-plane split requires BFR1 uint64 offsets from explicit FOR")
        try: encoded,width=primitive.run_buff_byte_plane(stage_source.payload,settings.get("bit_width"))
        except ValueError as exc: raise ActionTypeError(str(exc)) from exc
        result=_pack_result((name,),source,encoded,max_point,upstream_header=upstream,
            stage_input_payload_count=stage_source.count,extra_metadata={
                "policy_registry_version":registry_version,"policy_action_id":algorithm.action_id,
                "policy_algorithm":name,"io_variant":name,"action_parameters":{"bit_width":width},
                "io_input_kind":"bytes","io_input_magic":b"BFR1".hex()})
    if output == "streams":
        from action_semantic_ports import to_ports
        return to_ports(result)
    return result


def self_check() -> dict[str, Any]:
    if tuple(action.action_id for action in ACTIONS) != tuple(range(len(ACTIONS))):
        raise AssertionError("unified action IDs must be dense over all records")
    modes = Counter(action.execution_mode for action in ACTIONS)
    if modes != {
        "standalone": 33,
        "stagewise": (169 + len(primitive.LEGACY_COMPATIBILITY_ACTIONS)
                      + len(primitive.LOSSLESS64_ACTIONS)
                      + len(primitive.CURRENT_ZIGZAG_SPLIT_EXACT_ACTIONS)
                      + len(primitive.BUFF_CURRENT_EXACT_ACTIONS)
                      - (len(primitive.RETIRED_COMPOSITE_ACTION_NAMES) - 1)),
        "experimental_non_atomic": 1,
        "descriptor_only": 42,
        "retired": 1 + len(primitive.RETIRED_COMPOSITE_ACTION_NAMES),
    }:
        raise AssertionError(f"unexpected execution-mode split: {modes}")

    int_probe = ActionData.ints([-9, -9, -3, 0, 1, 1, 17, 17, 17])
    float_probe = ActionData.floats([-3.5, -3.5, 0.0, 1.25, 1.25, 9.0])
    byte_probe = ActionData.bytes(bytes(range(64)) * 4)
    standalone_round_trips = 0
    for action in ACTIONS:
        if action.execution_mode != "standalone":
            continue
        source = {
            "int64": int_probe,
            "float64": float_probe,
            "bytes": byte_probe,
        }[action.accepted_input_kinds[0]]
        encoded = run_action(action.name, source)
        decoded = run_action(action.name, encoded, reverse=True)
        if decoded != source:
            raise AssertionError(f"standalone round trip failed: {action.name}")
        standalone_round_trips += 1

    native_sequences = 0
    native_action_round_trips = 0
    covered_native_actions: set[str] = set()
    for algorithm_key in primitive.EXACT_NATIVE_METHOD_KEYS.values():
        names = tuple(
            action.name for action in primitive.exact_native_lz_action_pipeline(algorithm_key)
        )
        encoded = byte_probe
        for name in names:
            stage_input = encoded
            encoded = run_action(name, stage_input)
            if run_action(name, encoded, reverse=True) != stage_input:
                raise AssertionError(f"native-LZ action round trip failed: {name}")
            if name not in covered_native_actions:
                native_action_round_trips += 1
            covered_native_actions.add(name)
        _header, raw_wire = _unpack_result(encoded)
        expected_wire = primitive.encode_exact_native_lz_pipeline(
            byte_probe.payload, algorithm_key
        )
        if raw_wire != expected_wire:
            raise AssertionError(f"native-LZ staged wire changed: {algorithm_key}")
        if run_actions(names, encoded, reverse=True) != byte_probe:
            raise AssertionError(f"native-LZ stage sequence failed: {algorithm_key}")
        native_sequences += 1
    expected_native = set(primitive.EXACT_NATIVE_ACTION_BY_NAME)
    if covered_native_actions != expected_native:
        raise AssertionError("some native-LZ stage actions were not independently checked")

    float_rle = primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS
    encoded = run_actions(float_rle, float_probe, max_point=2)
    if run_actions(float_rle, encoded, reverse=True) != float_probe:
        raise AssertionError("typed float scaling -> int64 RLE round trip failed")

    encoded = run_actions(primitive.TS2DIFF_INT_ACTIONS, int_probe)
    if run_actions(primitive.TS2DIFF_INT_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed integer TS_2DIFF stage chain failed")
    encoded = run_actions(
        primitive.TS2DIFF_FLOAT_ACTIONS, float_probe, max_point=2
    )
    if run_actions(
        primitive.TS2DIFF_FLOAT_ACTIONS, encoded, reverse=True
    ) != float_probe:
        raise AssertionError("typed floating TS_2DIFF stage chain failed")

    delta_encoded = run_action(primitive.TS2DIFF_FLOAT_ACTIONS[1], int_probe)
    if run_action(
        primitive.TS2DIFF_FLOAT_ACTIONS[1], delta_encoded, reverse=True
    ) != int_probe:
        raise AssertionError("standalone TS_2DIFF delta stage failed")

    encoded = run_actions(primitive.SVB_INT_ACTIONS, int_probe)
    if run_actions(primitive.SVB_INT_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed SVB exact stage chain failed")
    encoded = run_actions(primitive.SVB_FLOAT_ACTIONS, float_probe, max_point=2)
    if run_actions(primitive.SVB_FLOAT_ACTIONS, encoded, reverse=True) != float_probe:
        raise AssertionError("typed floating SVB exact stage chain failed")
    svb_delta = run_action(primitive.SVB_INT_ACTIONS[0], int_probe)
    if run_action(primitive.SVB_INT_ACTIONS[0], svb_delta, reverse=True) != int_probe:
        raise AssertionError("standalone SVB delta stage failed")

    encoded = run_actions(primitive.SIMPLE8B_INT_ACTIONS, int_probe)
    if run_actions(primitive.SIMPLE8B_INT_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed integer Simple8b exact action failed")
    encoded = run_actions(
        primitive.SIMPLE8B_FLOAT_ACTIONS, float_probe, max_point=2
    )
    if run_actions(
        primitive.SIMPLE8B_FLOAT_ACTIONS, encoded, reverse=True
    ) != float_probe:
        raise AssertionError("typed floating Simple8b exact stage chain failed")

    for method, names in (
        ("RLBE", primitive.RLBE_ACTIONS),
        ("Sprintz", primitive.SPRINTZ_ACTIONS),
    ):
        for source in (int_probe, float_probe):
            encoded = run_actions(names, source)
            if run_actions(names, encoded, reverse=True) != source:
                raise AssertionError(f"typed {method} exact stage chain failed")
        first = run_action(names[0], int_probe)
        if run_action(names[0], first, reverse=True) != int_probe:
            raise AssertionError(f"standalone {method} first stage failed")

    encoded = run_actions(primitive.DBP_INT_ACTIONS, int_probe)
    if run_actions(primitive.DBP_INT_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed integer DBP exact stage chain failed")
    encoded = run_actions(primitive.DBP_FLOAT_ACTIONS, float_probe, max_point=2)
    if run_actions(primitive.DBP_FLOAT_ACTIONS, encoded, reverse=True) != float_probe:
        raise AssertionError("typed floating DBP exact stage chain failed")
    dbp_delta = run_action(primitive.DBP_FLOAT_ACTIONS[1], int_probe)
    if run_action(primitive.DBP_FLOAT_ACTIONS[1], dbp_delta, reverse=True) != int_probe:
        raise AssertionError("standalone DBP delta stage failed")

    for source in (int_probe, float_probe):
        encoded = run_actions(primitive.GORILLA_ACTIONS, source)
        if run_actions(primitive.GORILLA_ACTIONS, encoded, reverse=True) != source:
            raise AssertionError("typed Gorilla exact stage chain failed")
    gorilla_xor = run_action(primitive.GORILLA_ACTIONS[0], float_probe)
    if run_action(primitive.GORILLA_ACTIONS[0], gorilla_xor, reverse=True) != float_probe:
        raise AssertionError("standalone Gorilla XOR stage failed")

    for source in (int_probe, float_probe):
        encoded = run_actions(primitive.CHIMP128_ACTIONS, source)
        if run_actions(primitive.CHIMP128_ACTIONS, encoded, reverse=True) != source:
            raise AssertionError("typed Chimp128 exact stage chain failed")

    for source in (int_probe, float_probe):
        encoded = run_actions(primitive.PONGO_ACTIONS, source)
        if run_actions(primitive.PONGO_ACTIONS, encoded, reverse=True) != source:
            raise AssertionError("typed Pongo exact stage chain failed")

    encoded = run_actions(primitive.SPDP_ACTIONS, byte_probe)
    if run_actions(primitive.SPDP_ACTIONS, encoded, reverse=True) != byte_probe:
        raise AssertionError("typed SPDP exact stage chain failed")
    spdp_delta_packet = primitive._run_spdp_stage_cli(
        "delta-forward", byte_probe.payload
    )
    spdp_delta = run_action(primitive.SPDP_ACTIONS[0], byte_probe)
    if run_action(primitive.SPDP_ACTIONS[0], spdp_delta, reverse=True) != byte_probe:
        raise AssertionError("standalone SPDP delta stage failed")
    spdp_block_source = ActionData.bytes(spdp_delta_packet)
    spdp_block = run_action(primitive.SPDP_ACTIONS[1], spdp_block_source)
    if run_action(primitive.SPDP_ACTIONS[1], spdp_block, reverse=True) != spdp_block_source:
        raise AssertionError("standalone SPDP block stage failed")
    spdp_block_packet = primitive._run_spdp_stage_cli(
        "block-forward", spdp_delta_packet
    )
    spdp_entropy_source = ActionData.bytes(spdp_block_packet)
    spdp_entropy = run_action(primitive.SPDP_ACTIONS[2], spdp_entropy_source)
    if run_action(primitive.SPDP_ACTIONS[2], spdp_entropy, reverse=True) != spdp_entropy_source:
        raise AssertionError("standalone SPDP entropy stage failed")

    encoded = run_actions(primitive.FPC_ACTIONS, byte_probe)
    if run_actions(primitive.FPC_ACTIONS, encoded, reverse=True) != byte_probe:
        raise AssertionError("typed FPC exact stage chain failed")
    fpc_packet = primitive._run_fpc_stage_cli("xor-forward", byte_probe.payload)
    fpc_xor = run_action(primitive.FPC_ACTIONS[0], byte_probe)
    if run_action(primitive.FPC_ACTIONS[0], fpc_xor, reverse=True) != byte_probe:
        raise AssertionError("standalone FPC predictor/XOR stage failed")
    fpc_pack_source = ActionData.bytes(fpc_packet)
    fpc_pack = run_action(primitive.FPC_ACTIONS[1], fpc_pack_source)
    if run_action(primitive.FPC_ACTIONS[1], fpc_pack, reverse=True) != fpc_pack_source:
        raise AssertionError("standalone FPC pack stage failed")

    encoded = run_actions(primitive.PFPC_ACTIONS, byte_probe)
    if run_actions(primitive.PFPC_ACTIONS, encoded, reverse=True) != byte_probe:
        raise AssertionError("typed pFPC exact stage chain failed")
    pfpc_packet = primitive._run_pfpc_stage_cli("xor-forward", byte_probe.payload)
    pfpc_xor = run_action(primitive.PFPC_ACTIONS[0], byte_probe)
    if run_action(primitive.PFPC_ACTIONS[0], pfpc_xor, reverse=True) != byte_probe:
        raise AssertionError("standalone pFPC predictor/XOR stage failed")
    pfpc_pack_source = ActionData.bytes(pfpc_packet)
    pfpc_pack = run_action(primitive.PFPC_ACTIONS[1], pfpc_pack_source)
    if run_action(primitive.PFPC_ACTIONS[1], pfpc_pack, reverse=True) != pfpc_pack_source:
        raise AssertionError("standalone pFPC pack stage failed")

    encoded = run_actions(primitive.BOS_BOSA_ACTIONS, int_probe)
    if run_actions(primitive.BOS_BOSA_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed full-BOSA exact stage chain failed")
    encoded = run_actions(primitive.BOS64_INT_ACTIONS, int_probe)
    if run_actions(primitive.BOS64_INT_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed integer BOS64 exact stage chain failed")
    encoded = run_actions(primitive.BOS64_FLOAT_ACTIONS, float_probe, max_point=2)
    if run_actions(primitive.BOS64_FLOAT_ACTIONS, encoded, reverse=True) != float_probe:
        raise AssertionError("typed floating BOS64 exact stage chain failed")

    encoded = run_actions(primitive.FASTPFOR_INT_ACTIONS, int_probe)
    if run_actions(primitive.FASTPFOR_INT_ACTIONS, encoded, reverse=True) != int_probe:
        raise AssertionError("typed integer FastPFor exact stage chain failed")
    encoded = run_actions(
        primitive.FASTPFOR_FLOAT_ACTIONS, float_probe, max_point=2
    )
    if run_actions(
        primitive.FASTPFOR_FLOAT_ACTIONS, encoded, reverse=True
    ) != float_probe:
        raise AssertionError("typed floating FastPFor exact stage chain failed")

    for names in (primitive.ELF_ACTIONS, primitive.ELF_PLUS_ACTIONS):
        encoded = run_actions(names, float_probe)
        if run_actions(names, encoded, reverse=True) != float_probe:
            raise AssertionError(f"typed Elf-family exact stage chain failed: {names}")

    for source in (int_probe, float_probe):
        encoded = run_actions(primitive.CAMEL_ACTIONS, source)
        if run_actions(primitive.CAMEL_ACTIONS, encoded, reverse=True) != source:
            raise AssertionError("typed Camel exact stage chain failed")

    # Container stages accept a typed packet from the preceding raw-DEFLATE
    # stage.  Use fixed valid public wires to verify the standalone boundary.
    for names, inverse_command in (
        (primitive.GZIP_CONTAINER_ACTIONS, "gzip-inverse"),
        (primitive.ZLIB_CONTAINER_ACTIONS, "zlib-inverse"),
    ):
        wire = (
            bytes.fromhex("1f8b080000000000000303000000000000000000")
            if names == primitive.GZIP_CONTAINER_ACTIONS
            else bytes.fromhex("78da030000000001")
        )
        packet = ActionData.bytes(
            primitive._run_deflate_container_stage_cli(inverse_command, wire)
        )
        encoded = run_actions(names, packet)
        if run_actions(names, encoded, reverse=True) != packet:
            raise AssertionError(f"DEFLATE container stage failed: {names}")

    xz_wire = bytes.fromhex(
        "fd377a585a000004e6d6b446000000001cdf44211fb6f37d"
        "010000000004595a"
    )
    xz_packet = ActionData.bytes(
        primitive._run_xz_container_stage_cli("inverse", xz_wire)
    )
    xz_encoded = run_actions(primitive.XZ_CONTAINER_ACTIONS, xz_packet)
    if run_actions(
        primitive.XZ_CONTAINER_ACTIONS, xz_encoded, reverse=True
    ) != xz_packet:
        raise AssertionError("XZ container stage failed")

    encoded = run_actions(primitive.FPZIP_ACTIONS, float_probe)
    if run_actions(primitive.FPZIP_ACTIONS, encoded, reverse=True) != float_probe:
        raise AssertionError("FPZIP exact stage chain failed")

    specialist_stage_chains = (
        ("LECO", primitive.LECO_ACTIONS, int_probe),
        ("ALP-exact", primitive.ALP_MAIN_ACTIONS, float_probe),
        ("ALP-RD-dictionary", primitive.ALP_DICTIONARY_ACTIONS, float_probe),
        ("REGER-exact", primitive.REGER_ACTIONS, int_probe),
        ("BUFF-reference", primitive.BUFF_ATOMIC_ACTIONS, float_probe),
    )
    for label, names, source in specialist_stage_chains:
        encoded = run_actions(names, source)
        if run_actions(names, encoded, reverse=True) != source:
            raise AssertionError(f"{label} stage chain failed")

    general_reference_chains = [
        ("Brotli-reference", primitive.BROTLI_REFERENCE_ACTIONS),
        ("Zstd-reference", primitive.ZSTD_REFERENCE_ACTIONS),
        ("Bzip2-reference", primitive.BZIP2_REFERENCE_ACTIONS),
        ("gzip-reference", primitive.GZIP_REFERENCE_ACTIONS),
        ("Zlib-reference", primitive.ZLIB_REFERENCE_ACTIONS),
        ("XZ-reference", primitive.XZ_REFERENCE_ACTIONS),
        *(
            (f"Blosc-reference-{kind}", names)
            for kind, names in primitive.BLOSC_REFERENCE_ACTIONS_BY_KIND.items()
        ),
        *(
            (f"Blosc2-reference-{kind}", names)
            for kind, names in primitive.BLOSC2_REFERENCE_ACTIONS_BY_KIND.items()
        ),
    ]
    for label, names in general_reference_chains:
        encoded = run_actions(names, byte_probe)
        if run_actions(names, encoded, reverse=True) != byte_probe:
            raise AssertionError(f"{label} stage chain failed")

    general_exact_chains = [
        ("Brotli", primitive.BROTLI_EXACT_ACTIONS),
        ("Zstd", primitive.ZSTD_EXACT_ACTIONS),
        ("Bzip2", primitive.BZIP2_EXACT_ACTIONS),
        ("gzip", primitive.GZIP_EXACT_ACTIONS),
        ("Zlib", primitive.ZLIB_EXACT_ACTIONS),
        ("XZ", primitive.XZ_EXACT_ACTIONS),
        *(
            (f"Blosc-{kind}", names)
            for kind, names in primitive.BLOSC_EXACT_ACTIONS_BY_KIND.items()
        ),
        *(
            (f"Blosc2-{kind}", names)
            for kind, names in primitive.BLOSC2_EXACT_ACTIONS_BY_KIND.items()
        ),
    ]
    for label, names in general_exact_chains:
        encoded = run_actions(names, byte_probe)
        if run_actions(names, encoded, reverse=True) != byte_probe:
            raise AssertionError(f"{label} exact residual chain failed")

    descriptor_rejections = 0
    for action in ACTIONS:
        if action.execution_mode != "descriptor_only":
            continue
        try:
            run_action(action.name, byte_probe)
        except ActionNotExecutableError:
            descriptor_rejections += 1
        else:
            raise AssertionError(f"descriptor-only action executed silently: {action.name}")

    return {
        "status": "PASS",
        "active_action_count": len(CANONICAL_ACTIONS),
        "policy_action_slots_including_stop": primitive.STOP_ACTION_ID + 1,
        "policy_stop_action_id": primitive.STOP_ACTION_ID,
        "compatibility_descriptor_count": len(ACTIONS),
        "compatibility_execution_modes": dict(sorted(modes.items())),
        "standalone_round_trips": standalone_round_trips,
        "native_lz_method_round_trips": native_sequences,
        "native_lz_independent_action_round_trips": native_action_round_trips,
        "native_lz_action_names_covered": len(covered_native_actions),
        "float_scaling_rle_round_trips": 1,
        "rle_scaling_exact_stage_actions": 1,
        "ts2diff_exact_stage_actions": 4,
        "ts2diff_exact_chain_round_trips": 2,
        "svb_exact_stage_actions": 2,
        "svb_exact_chain_round_trips": 2,
        "simple8b_exact_stage_actions": 2,
        "simple8b_exact_chain_round_trips": 2,
        "rlbe_exact_stage_actions": 2,
        "rlbe_exact_chain_round_trips": 2,
        "sprintz_exact_stage_actions": 3,
        "sprintz_exact_chain_round_trips": 2,
        "dbp_exact_stage_actions": 4,
        "dbp_exact_chain_round_trips": 2,
        "gorilla_exact_stage_actions": 2,
        "gorilla_exact_chain_round_trips": 2,
        "chimp128_exact_stage_actions": 2,
        "chimp128_exact_chain_round_trips": 2,
        "pongo_exact_stage_actions": 3,
        "pongo_exact_chain_round_trips": 2,
        "spdp_exact_stage_actions": 3,
        "spdp_exact_chain_round_trips": 1,
        "fpc_exact_stage_actions": 2,
        "fpc_exact_chain_round_trips": 1,
        "pfpc_exact_stage_actions": 2,
        "pfpc_exact_chain_round_trips": 1,
        "bos_policy_stage_actions": 7,
        "bos_independent_chain_round_trips": 3,
        "fastpfor_policy_stage_actions": 2,
        "fastpfor_exact_chain_round_trips": 2,
        "elf_family_exact_stage_actions": 3,
        "elf_family_exact_chain_round_trips": 2,
        "camel_policy_stage_actions": 3,
        "camel_exact_chain_round_trips": 2,
        "snappy_policy_stage_actions": len(primitive.SNAPPY_ACTIONS),
        "lz4_policy_stage_actions": len(primitive.LZ4_ACTIONS),
        "deflate_container_exact_stage_actions": 2,
        "deflate_container_exact_stage_round_trips": 2,
        "xz_container_exact_stage_actions": 1,
        "xz_container_exact_stage_round_trips": 1,
        "fpzip_exact_stage_actions": 2,
        "fpzip_exact_chain_round_trips": 1,
        "leco_exact_stage_actions": len(primitive.LECO_ACTIONS),
        "leco_exact_chain_round_trips": 1,
        "alp_exact_stage_actions": len(primitive.ALP_MAIN_ACTIONS),
        "alp_reference_chain_round_trips": 2,
        "reger_exact_stage_actions": len(primitive.REGER_ACTIONS),
        "reger_exact_chain_round_trips": 1,
        "buff_exact_stage_actions": len(primitive.BUFF_ATOMIC_ACTIONS),
        "buff_exact_chain_round_trips": 1,
        "general_reference_stage_actions": len(primitive.GENERAL_CODEC_REFERENCE_ACTIONS),
        "general_reference_chain_round_trips": len(general_reference_chains),
        "general_exact_residual_stage_actions": len(primitive.GENERAL_CODEC_RESIDUAL_ACTIONS),
        "general_exact_residual_chain_round_trips": len(general_exact_chains),
        "descriptor_only_rejections": descriptor_rejections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="store_true", help="print the corrected action registry")
    parser.add_argument("--describe", help="print one action descriptor")
    parser.add_argument("--self-check", action="store_true", help="run typed round-trip checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.registry:
        print(json.dumps(registry(), ensure_ascii=False, indent=2))
    if args.describe:
        action: ActionRef = int(args.describe) if args.describe.isdecimal() else args.describe
        print(json.dumps(asdict(describe_action(action)), ensure_ascii=False, indent=2))
    if args.self_check or not (args.registry or args.describe):
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
