#!/usr/bin/env python3
"""Probe every production fixed mechanism configuration through native NGF1.

This is an acceptance driver, not a substitute catalog.  It preserves every
production ActionSpec from the frozen catalog, constructs only the predecessor
recipe needed to reach that spec, asks ``native_pipeline_v1_server`` to replay
and evaluate the fixed recipe, then launches ``native_graph_request_cli`` as a
fresh process to decode the returned complete NGF1 frame.

The 193 identities intentionally ignore target slot, as the production catalog
duplicates each fixed configuration across active-flow slots.  The default
experiment profile executes 191 identities and records the two user-excluded
CMIX/NNCP identities separately without treating them as skips or failures.
A035 is executed at slot 1 after the production adjacent-XOR semantic-port
action because its physical residual is the second output; both the canonical
slot-0 ActionSpec and the actually executed homologous ActionSpec are retained
in the receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import native_experiment_scope_v1 as experiment_scope


SCHEMA = "native-fixed-configurations-acceptance-v1"
REGISTRY = "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd"
PROBABILITY_IMPLEMENTATIONS = {"A106", "A107", "A108", "A109", "A110"}

# Each selector is (runtime_action, input_view, io_variant, max_point).  None in
# the last two positions is an exact None requirement, not a wildcard.
PREFIXES: dict[str, tuple[tuple[str, str, str | None, int | None], ...]] = {
    "op.xor.camel.cpp_v1": (("op.eraser.camel.cpp_v1", "float64", "op.eraser.camel.cpp_v1", None),),
    "op.xor.pongo.cpp_v1": (("op.eraser.pongo.cpp_v1", "float64", "op.eraser.pongo.cpp_v1", None),),
    "op.frame_of_reference.bos.cpp_v1": (("op.delta.bos.cpp_v1", "int64", "op.delta.bos.cpp_v1", None),),
    "op.frame_of_reference.buff.cpp_v2": (("op.scaling.buff.cpp_v1", "float64", "op.scaling.buff.cpp_v1", None),),
    "op.outlier_separation.bos.cpp_v1": (
        ("op.delta.bos.cpp_v1", "int64", "op.delta.bos.cpp_v1", None),
        ("op.frame_of_reference.bos.cpp_v1", "bytes", "op.frame_of_reference.bos.cpp_v1", None),
    ),
    "op.trailing_zero_pack.gorilla.cpp_v1": (("op.xor.gorilla.cpp_v1", "float64", "op.xor.gorilla.cpp_v1", None),),
    "op.trailing_zero_pack.pongo.cpp_v1": (
        ("op.eraser.pongo.cpp_v1", "float64", "op.eraser.pongo.cpp_v1", None),
        ("op.xor.pongo.cpp_v1", "bytes", "op.xor.pongo.cpp_v1", None),
    ),
    "op.trailing_zero_pack.chimp128.cpp_v1": (("op.xor.chimp128.cpp_v1", "float64", "op.xor.chimp128.cpp_v1", None),),
    "op.trailing_zero_pack.fpc.cpp_v1": (("op.xor.fpc.cpp_v1", "bytes", "op.xor.fpc.cpp_v1", None),),
    "op.trailing_zero_pack.pfpc.cpp_v1": (("op.xor.pfpc.cpp_v1", "bytes", "op.xor.pfpc.cpp_v1", None),),
    "op.dictionary.corad_tricklet_xor.cpp_v2": (("op.scaling.corad_normalize_exact.cpp_v2", "int64", "op.scaling.corad_normalize_exact.cpp_v2", None),),
    "op.block_transform.buff_byte_plane_split.cpp_v2": (
        ("op.scaling.buff.cpp_v1", "float64", "op.scaling.buff.cpp_v1", None),
        ("op.frame_of_reference.buff.cpp_v2", "bytes", "op.frame_of_reference.buff.cpp_v2", None),
    ),
    "op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1": (
        ("op.scaling.legacy_exact_decimal.cpp_v1", "int64", "op.scaling.legacy_exact_decimal.cpp_v1", None),
    ),
    # The native GraphTask inserts window4096 as the cataloged internal adapter.
    "op.lz_match.native_lz.aligned8.min3.max18": (),
    "op.match_parse_policy.native_lz.aligned8_greedy": (
        ("op.lz_match.native_lz.aligned8.min3.max18", "bytes", None, None),
    ),
    "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3": (
        ("op.lz_match.native_lz.aligned8.min3.max18", "bytes", None, None),
        ("op.match_parse_policy.native_lz.aligned8_greedy", "bytes", "op.match_parse_policy.native_lz.aligned8_greedy", None),
    ),
    "op.match_token_format.snappy.cpp_v1": (("op.lz_match.snappy_snap_1_1_1.cpp_v1", "bytes", "op.lz_match.snappy_snap_1_1_1.cpp_v1", None),),
    "op.match_token_format.lz4.cpp_v1": (("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", "bytes", "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", None),),
    "op.match_token_format.blosclz.ctrl8.cpp_reference_v1": (
        ("op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1", "bytes", "op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1", None),
    ),
    "op.integer_code.stream_vbyte.cpp_v1": (("op.delta.stream_vbyte.cpp_v1", "int64", "op.delta.stream_vbyte.cpp_v1", None),),
    "op.integer_code.fastpfor_variablebyte.cpp_v1": (("op.bitpack_int.fastpfor_pfor.cpp_v1", "int64", "op.bitpack_int.fastpfor_pfor.cpp_v1", None),),
    "op.integer_code.lz4.cpp_v1": (
        ("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", "bytes", "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", None),
        ("op.match_token_format.lz4.cpp_v1", "bytes", "op.match_token_format.lz4.cpp_v1", None),
    ),
    "op.bitstream_pack.camel.cpp_v1": (
        ("op.eraser.camel.cpp_v1", "float64", "op.eraser.camel.cpp_v1", None),
        ("op.xor.camel.cpp_v1", "bytes", "op.xor.camel.cpp_v1", None),
    ),
    "op.entropy_range.fpzip.cpp_v1": (("op.delta.fpzip.cpp_v1", "float64", "op.delta.fpzip.cpp_v1", None),),
    "op.entropy_special.spdp.cpp_v1": (
        ("op.delta.spdp.cpp_v1", "bytes", "op.delta.spdp.cpp_v1", None),
    ),
}

# These production specs accept opaque bytes but the bytes have a specific
# typed carrier.  The target ActionSpec remains unchanged; only its fixture gets
# the production predecessor that creates that carrier.
CARRIER_PREFIXES: dict[int, tuple[tuple[str, str, str | None, int | None], ...]] = {
    98: (("op.scaling.delta_binary_packed.cpp_v1", "float64", "op.scaling.delta_binary_packed.cpp_v1", 2),),
    102: (("op.scaling.exact_decimal.cpp_v2", "float64", "op.scaling.simple8b.cpp_v1", 2),),
    108: (("op.scaling.delta_binary_packed.cpp_v1", "float64", "op.scaling.delta_binary_packed.cpp_v1", 2),),
    110: (("op.scaling.exact_decimal.cpp_v2", "float64", "op.scaling.ts_2diff.cpp_v1", 2),),
    119: (("op.eraser.pongo.cpp_v1", "float64", "op.eraser.pongo.cpp_v1", None),),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256(raw)}


def configuration_basis(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "implementation_id", "runtime_action", "input_view", "parameter_mode",
        "parameter_value", "parameter_items", "max_point", "io_variant",
    )}


def configuration_key(row: dict[str, Any]) -> str:
    return sha256(canonical(configuration_basis(row)))


def mechanism_rows(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in actions
            if row["category"] == "mechanism" and row["target_slot"] == 0]
    if len(rows) != 193 or len({configuration_key(row) for row in rows}) != 193:
        raise ValueError("production catalog does not contain exactly 193 unique slot-0 mechanism configurations")
    return rows


def find_action(actions: list[dict[str, Any]], selector: tuple[str, str, str | None, int | None],
                *, slot: int = 0) -> dict[str, Any]:
    runtime, view, io_variant, max_point = selector
    found = [row for row in actions if row["category"] == "mechanism"
             and row["target_slot"] == slot and row["runtime_action"] == runtime
             and row["input_view"] == view and row["io_variant"] == io_variant
             and row["max_point"] == max_point]
    if len(found) != 1:
        raise ValueError(f"prefix selector is not unique: {selector!r} slot={slot}: {len(found)} rows")
    return found[0]


def homolog_at_slot(actions: list[dict[str, Any]], base: dict[str, Any], slot: int) -> dict[str, Any]:
    basis = configuration_basis(base)
    found = [row for row in actions if row["category"] == "mechanism"
             and row["target_slot"] == slot and configuration_basis(row) == basis]
    if len(found) != 1:
        raise ValueError(f"configuration {base['key']} has {len(found)} homologs at slot {slot}")
    return found[0]


def actual_parameters(spec: dict[str, Any], physical_width: int) -> dict[str, Any]:
    result = {str(key): value for key, value in spec["parameter_items"]}
    if spec["io_variant"] is not None and spec["io_variant"] != spec["runtime_action"]:
        result["io_variant"] = spec["io_variant"]
    if spec["max_point"] is not None:
        result["max_point"] = spec["max_point"]
    mode = spec["parameter_mode"]
    if mode == "fp32":
        result["format"] = "fp32"
    elif mode == "top8":
        result.update(mode="top8", width=physical_width)
    elif mode == "bits":
        result["bits"] = spec["parameter_value"]
    elif mode == "width":
        result["width"] = spec["parameter_value"]
    elif mode != "none":
        raise ValueError(f"unsupported mechanism parameter mode in acceptance probe: {mode}")
    return result


def root(kind: str, width: int, logical_view: str, data: bytes) -> dict[str, Any]:
    return {"kind": kind, "width": width, "data_hex": data.hex(),
            "logical_view": logical_view, "original_numeric_payload": True}


def fixture(view: str, implementation: str, *, width: int | None = None,
            probability: bool = False) -> dict[str, Any]:
    if implementation in {"A006", "A007", "A008"}:
        return root("numeric", 8, "float64", struct.pack("<16d", *([0.0] * 16)))
    if implementation == "A035":
        words = (0, 1, 1, 0x101, 0x10101, 0x100010101, 0x7fffffffffffffff)
        return root("numeric", 8, "int64", struct.pack("<7q", *words))
    if implementation == "A103":
        words = (0x00000000, 0x80000000, 0x3f800000, 0x7fc01234, 0x00000001)
        return root("numeric", 4, "float32", struct.pack("<5I", *words))
    if implementation == "A104":
        if width not in {1, 2, 4, 8}:
            raise ValueError("A104 fixture requires physical width 1, 2, 4, or 8")
        return root("numeric", width, "bytes", bytes(range(64)))
    if implementation == "A111":
        words = (0, 1, 2, 7, 255, 1025, 0xffffffff)
        return root("numeric", 4, "uint32", struct.pack("<7I", *words))
    if view == "int64":
        return root("numeric", 8, "int64", struct.pack("<64q", *range(64)))
    if view == "float64":
        values = tuple(float(index) / 4.0 for index in range(64))
        return root("numeric", 8, "float64", struct.pack("<64d", *values))
    if view == "float32":
        return root("numeric", 4, "float32", struct.pack("<16f", *(float(i) for i in range(16))))
    if view == "uint32":
        return root("numeric", 4, "uint32", struct.pack("<16I", *range(16)))
    size = 1 if implementation == "A109" and probability else 512
    raw = (b"ABCDEFGH" * ((size + 7) // 8))[:size]
    return root("serial", 1, "bytes", raw)


def process_limit(gib: int | None):
    if gib is None or not hasattr(resource, "RLIMIT_AS"):
        return None
    limit = gib * (1 << 30)

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return apply


def run_jsonl(server: Path, catalog: Path, requests: list[dict[str, Any]], *,
              timeout: float, address_space_gib: int | None,
              experiment_profile: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in requests)
    command = [str(server), "--catalog", str(catalog)]
    if experiment_profile is not None:
        command.extend(("--experiment-profile", str(experiment_profile)))
    command.append("serve")
    completed = subprocess.run(
        command, input=text,
        text=True, capture_output=True, timeout=timeout, check=False,
        preexec_fn=process_limit(address_space_gib),
    )
    if completed.returncode:
        raise RuntimeError(f"native server exited {completed.returncode}: {completed.stderr.strip()}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != len(requests):
        raise RuntimeError(f"native server returned {len(lines)} JSON lines for {len(requests)} requests; stderr={completed.stderr.strip()}")
    return [json.loads(line) for line in lines], completed.stderr


def cold_decode(graph_cli: Path, frame: bytes, expected_root: dict[str, Any], *,
                timeout: float, address_space_gib: int | None) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as request_file:
        json.dump({"schema": "native-graph-request-v1", "frame_hex": frame.hex()}, request_file)
        request_file.flush()
        completed = subprocess.run(
            [str(graph_cli), "decode", request_file.name], text=True,
            capture_output=True, timeout=timeout, check=False,
            preexec_fn=process_limit(address_space_gib),
        )
    if completed.returncode:
        raise RuntimeError(f"independent cold decoder exited {completed.returncode}: {completed.stderr.strip()}")
    decoded = json.loads(completed.stdout)
    if decoded.get("original_numeric_bytes") != len(bytes.fromhex(expected_root["data_hex"])):
        raise RuntimeError("cold decoder original numeric denominator mismatch")
    if decoded.get("roots") != [expected_root]:
        raise RuntimeError("cold decoder root bytes/type/width/logical view mismatch")
    return decoded


def make_request(config: dict[str, Any], executed: dict[str, Any], prefixes: list[dict[str, Any]],
                 root_row: dict[str, Any], *, allow_large_memory: bool) -> dict[str, Any]:
    recipe = [row["action_id"] for row in prefixes] + [executed["action_id"], 0]
    return {
        "cmd": "reset", "request_id": 1,
        "task_id": "fixed-config-" + configuration_key(config)[:16],
        "roots": [root_row], "weights": [1.0, 0.0, 0.0],
        "compression_reference_ns_per_byte": 1.0,
        "decompression_reference_ns_per_byte": 1.0,
        "original_numeric_bytes": len(bytes.fromhex(root_row["data_hex"])),
        "starting_recipe_action_ids": recipe, "block_index": 0,
        "continuation_edit_mode": False, "timing_warmups": 0,
        "timing_repetitions": 1, "allow_large_memory": allow_large_memory,
        "require_complete_native_catalog": False,
    }


def probe_one(server: Path, catalog: Path, graph_cli: Path, actions: list[dict[str, Any]],
              config: dict[str, Any], *, physical_width: int | None,
              include_cmix: bool, timeout: float, probability_timeout: float,
              cmix_timeout: float, cmix_address_space_gib: int,
              experiment_profile: Path | None = None,
              expected_experiment_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    key = configuration_key(config)
    record: dict[str, Any] = {
        "configuration_key": key,
        "production_action_spec": config,
        "configuration_basis": configuration_basis(config),
        "passed": False,
    }
    if config["implementation_id"] == "A109" and not include_cmix:
        record.update(status="skipped_resource_gate", attempted=False,
                      reason="cmix requires --include-cmix and a bounded production run")
        return record

    selectors = CARRIER_PREFIXES.get(config["action_id"], PREFIXES.get(config["runtime_action"], ()))
    if config["implementation_id"] == "A035":
        selectors = (("op.xor.previous_word.semantic_ports.cpp_v1", "int64", None, None),)
    prefixes = [find_action(actions, selector) for selector in selectors]
    executed = homolog_at_slot(actions, config, 1) if config["implementation_id"] == "A035" else config
    first_spec = prefixes[0] if prefixes else config
    first_view = (first_spec["required_logical_views"][0]
                  if first_spec["required_logical_views"] else first_spec["input_view"])
    root_row = fixture(first_view, config["implementation_id"], width=physical_width,
                       probability=config["implementation_id"] in PROBABILITY_IMPLEMENTATIONS)
    record.update(
        attempted=True, status="running", executed_action_spec=executed,
        prefix_action_specs=prefixes, root=root_row,
        actual_parameters=actual_parameters(executed, root_row["width"]),
        executed_recipe_action_ids=[row["action_id"] for row in prefixes] + [executed["action_id"]],
    )
    is_probability = config["implementation_id"] in PROBABILITY_IMPLEMENTATIONS
    is_cmix = config["implementation_id"] == "A109"
    selected_timeout = cmix_timeout if is_cmix else probability_timeout if is_probability else timeout
    address_limit = cmix_address_space_gib if is_cmix else None
    reset = make_request(config, executed, prefixes, root_row, allow_large_memory=is_cmix)
    evaluate = {"cmd": "evaluate_terminal", "request_id": 2,
                "task_handle": "t1", "state_handle": "s1", "include_frame_hex": True}
    try:
        responses, stderr = run_jsonl(server, catalog, [reset, evaluate],
                                      timeout=selected_timeout, address_space_gib=address_limit,
                                      experiment_profile=experiment_profile)
        record["server_stderr"] = stderr or None
        if not responses[0].get("ok"):
            raise RuntimeError("reset/replay failed: " + str(responses[0].get("error")))
        if expected_experiment_scope is not None:
            experiment_scope.verify_native(
                responses[0].get("catalog", {}), expected_experiment_scope)
        if not responses[1].get("ok"):
            raise RuntimeError("terminal evaluation failed: " + str(responses[1].get("error")))
        terminal = responses[1]
        frame = bytes.fromhex(terminal["frame_hex"])
        if frame[:4] != b"NGF1":
            raise RuntimeError("terminal result is not a complete NGF1 frame")
        if terminal["recipe_action_ids"] != record["executed_recipe_action_ids"]:
            raise RuntimeError("native server evaluated a different recipe")
        if terminal["frame_sha256"] != sha256(frame) or terminal["frame_bytes"] != len(frame):
            raise RuntimeError("complete frame size/hash receipt mismatch")
        if terminal["original_numeric_bytes"] != len(bytes.fromhex(root_row["data_hex"])):
            raise RuntimeError("terminal original numeric denominator mismatch")
        if len(terminal["encode_samples_ns"]) != 1 or len(terminal["decode_samples_ns"]) != 1:
            raise RuntimeError("terminal evaluation did not return one complete-frame timing sample")
        cold_decode(graph_cli, frame, root_row, timeout=selected_timeout,
                    address_space_gib=address_limit)
        record.update(
            passed=True, status="passed", frame_bytes=len(frame),
            frame_sha256=sha256(frame), original_numeric_bytes=terminal["original_numeric_bytes"],
            compression_ratio=terminal["compression_ratio"],
            encode_ns=terminal["encode_samples_ns"][0],
            decode_ns=terminal["decode_samples_ns"][0],
            in_process_roundtrip=True, independent_cold_decode=True,
            cold_decoder_model_or_encoder_state_supplied=False,
        )
    except subprocess.TimeoutExpired as error:
        record.update(status="failed", failure_phase="bounded_execution",
                      error_type=type(error).__name__, error=str(error))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        record.update(status="failed", failure_phase="native_roundtrip_or_cold_decode",
                      error_type=type(error).__name__, error=str(error))
    return record


def parse_ids(values: Iterable[str]) -> set[int]:
    result: set[int] = set()
    for value in values:
        for item in value.split(","):
            if item.strip(): result.add(int(item))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--experiment-profile", type=Path,
                        default=experiment_scope.DEFAULT_PROFILE,
                        help="frozen action-scope profile (defaults to CMIX/NNCP exclusion)")
    parser.add_argument("--graph-cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-action-id", action="append", default=[],
                        help="comma-separated canonical slot-0 IDs; repeatable")
    parser.add_argument("--include-cmix", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--probability-timeout", type=float, default=300.0)
    parser.add_argument("--cmix-timeout", type=float, default=600.0)
    parser.add_argument("--cmix-address-space-gib", type=int, default=8)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    if args.include_cmix:
        parser.error("--include-cmix is incompatible with the frozen CMIX/NNCP exclusion profile")
    for path in (args.server, args.catalog, args.graph_cli, args.experiment_profile):
        if not path.is_file(): parser.error(f"missing file: {path}")
    if args.cmix_address_space_gib < 6:
        parser.error("--cmix-address-space-gib must be at least 6 GiB for the frozen provider")

    document = json.loads(args.catalog.read_text())
    if document.get("schema") != "native-pipeline-v1-frozen-complete-action-catalog":
        parser.error("unexpected catalog schema")
    payload = document["registry_payload"]
    if document.get("registry_fingerprint") != REGISTRY or payload.get("actions") is None:
        parser.error("catalog registry identity mismatch")
    actions = payload["actions"]
    all_configs = mechanism_rows(actions)
    try:
        expected_scope = experiment_scope.load(args.catalog, args.experiment_profile)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid experiment profile: {error}")
    excluded_ids = set(expected_scope["excluded_action_ids"])
    excluded_configs = [row for row in all_configs if row["action_id"] in excluded_ids]
    configs = [row for row in all_configs if row["action_id"] not in excluded_ids]
    if len(excluded_configs) != expected_scope["excluded_fixed_configuration_count"]:
        parser.error("experiment profile fixed-configuration exclusion count mismatch")
    if len(configs) != expected_scope["enabled_fixed_configuration_count"]:
        parser.error("experiment profile enabled fixed-configuration count mismatch")
    selected = parse_ids(args.only_action_id)
    if selected:
        unknown = selected - {row["action_id"] for row in all_configs}
        if unknown: parser.error(f"--only-action-id contains noncanonical IDs: {sorted(unknown)}")
        forbidden = selected & excluded_ids
        if forbidden:
            parser.error(
                "--only-action-id selects user-excluded CMIX/NNCP IDs: "
                f"{sorted(forbidden)}")
        configs = [row for row in configs if row["action_id"] in selected]

    records: list[dict[str, Any]] = []
    for index, config in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {config['action_id']} {config['key']}",
              file=sys.stderr, flush=True)
        if config["implementation_id"] == "A104":
            variants = [probe_one(
                args.server.resolve(), args.catalog.resolve(), args.graph_cli.resolve(),
                actions, config, physical_width=width, include_cmix=args.include_cmix,
                timeout=args.timeout, probability_timeout=args.probability_timeout,
                cmix_timeout=args.cmix_timeout,
                cmix_address_space_gib=args.cmix_address_space_gib,
                experiment_profile=args.experiment_profile.resolve(),
                expected_experiment_scope=expected_scope,
            ) for width in (1, 2, 4, 8)]
            # width=1 has no lower byte plane and is deliberately masked by the
            # production GraphTask.  Preserve that observed rejection, while
            # the valid 2/4/8 modes determine implementation acceptance.
            primary = dict(next(row for row in variants if row["root"]["width"] == 4))
            primary["a104_width_mode_matrix"] = variants
            primary["a104_single_byte_expected_masked"] = not variants[0]["passed"]
            primary["passed"] = all(row["passed"] for row in variants[1:])
            primary["status"] = "passed" if primary["passed"] else "failed"
            records.append(primary)
        else:
            records.append(probe_one(
                args.server.resolve(), args.catalog.resolve(), args.graph_cli.resolve(),
                actions, config, physical_width=None, include_cmix=args.include_cmix,
                timeout=args.timeout, probability_timeout=args.probability_timeout,
                cmix_timeout=args.cmix_timeout,
                cmix_address_space_gib=args.cmix_address_space_gib,
                experiment_profile=args.experiment_profile.resolve(),
                expected_experiment_scope=expected_scope,
            ))

    attempted = [row for row in records if row.get("attempted")]
    failed = [row for row in attempted if not row["passed"]]
    excluded_records = [{
        "configuration_key": configuration_key(row),
        "production_action_spec": row,
        "configuration_basis": configuration_basis(row),
        "attempted": False,
        "passed": None,
        "status": "excluded_by_experiment_profile",
        "reason": "user excluded CMIX/NNCP from this experiment because they are too slow",
        "experiment_profile_sha256": expected_scope["profile_sha256"],
    } for row in excluded_configs]
    receipt = {
        "schema": SCHEMA,
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "registry_fingerprint": REGISTRY,
        "scope": "production fixed mechanism configurations; C++ in-process complete NGF1 encode/decode plus independent native cold decode",
        "configuration_identity": "193 unique production ActionSpec tuples at canonical target_slot=0; 191 enabled and 2 separately excluded without renumbering IDs",
        "timing_scope": "one correctness sample from C++ whole fixed-recipe graph and complete NGF1 frame; driver I/O and cold-process startup excluded",
        "host": {"hostname": platform.node(), "platform": platform.platform(),
                 "python": platform.python_version(), "pid": os.getpid()},
        "limits": {"ordinary_timeout_seconds": args.timeout,
                   "probability_timeout_seconds": args.probability_timeout,
                   "cmix_timeout_seconds": args.cmix_timeout,
                   "cmix_address_space_gib": args.cmix_address_space_gib,
                   "cmix_explicitly_enabled": False,
                   "timing_warmups": 0, "timing_repetitions": 1},
        "artifacts": {"server": file_identity(args.server),
                      "graph_cli": file_identity(args.graph_cli),
                      "catalog": file_identity(args.catalog),
                      "experiment_profile": file_identity(args.experiment_profile),
                      "probe": file_identity(Path(__file__))},
        "experiment_scope": expected_scope,
        "counts": {"catalog_fixed_configuration_count": 193,
                   "enabled_fixed_configuration_count": 191,
                   "excluded_fixed_configuration_count": len(excluded_records),
                   "selected_enabled_configuration_count": len(records),
                   "attempted_configuration_count": len(attempted),
                   "passed_configuration_count": sum(row["passed"] for row in attempted),
                   "failed_configuration_count": len(failed),
                   "skipped_configuration_count": 0},
        "all_selected_configurations_passed": len(failed) == 0 and len(attempted) == len(records),
        "all_selected_enabled_configurations_passed": len(failed) == 0 and len(attempted) == len(records),
        "failed_configuration_keys": [row["configuration_key"] for row in failed],
        "skipped_configuration_keys": [],
        "excluded_configuration_keys": [row["configuration_key"] for row in excluded_records],
        "excluded_records": excluded_records,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(receipt, indent=2, sort_keys=True,
                                       ensure_ascii=False, allow_nan=False).encode() + b"\n")
    print(json.dumps(receipt["counts"], sort_keys=True), file=sys.stderr)
    return 0 if args.allow_failures or receipt["all_selected_configurations_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
