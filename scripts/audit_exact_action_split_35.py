#!/usr/bin/env python3
"""Emit the source-level split contract for the 35 unresolved benchmark codecs.

This audit never upgrades an omission label into an implementation.  A row is
``exact_executable`` only when both forward and inverse stage symbols exist and
the stage is connected to the typed action runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import primitive_operator_action_runtime as primitive  # noqa: E402
import unified_action_runtime as unified  # noqa: E402


EXPECTED_METHODS = (
    "Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "BOS", "REGER",
    "SVB", "FastPFor", "Simple8b", "RLE", "RLBE", "Sprintz", "DBP",
    "ALP", "LECO", "Camel", "Elf", "Elf+", "FPC", "PFPC", "SPDP",
    "CORAD", "BUFF", "FPZIP", "gzip", "Brotli", "Zstd", "LZ4", "XZ",
    "Snappy", "Blosc", "Blosc2", "Bzip2", "Zlib",
)

# concrete_actions() already retains both position-specific BOS FOR records.
# Do not inject conceptual stages that are absent from the selected BOSA source.
PIPELINE_OCCURRENCE_OVERRIDES: dict[str, tuple[str, ...]] = {}

TS2DIFF_BINDINGS = {
    "scaling": (
        "float64", "bytes",
        "exact_actions::ts2diff::scaling_forward",
        "exact_actions::ts2diff::scaling_inverse",
    ),
    "delta": (
        "int64|bytes", "bytes",
        "exact_actions::ts2diff::delta_forward_int64/delta_forward_scaled",
        "exact_actions::ts2diff::delta_inverse_int64/delta_inverse_scaled",
    ),
    "frame_of_reference": (
        "bytes", "bytes",
        "exact_actions::ts2diff::frame_of_reference_forward",
        "exact_actions::ts2diff::frame_of_reference_inverse",
    ),
    "bitpack_int": (
        "bytes", "bytes",
        "exact_actions::ts2diff::bitpack_forward",
        "exact_actions::ts2diff::bitpack_inverse",
    ),
}

SVB_BINDINGS = {
    "delta": (
        "int64", "bytes",
        "exact_actions::svb::delta_forward",
        "exact_actions::svb::delta_inverse",
    ),
    "integer_code": (
        "bytes", "bytes",
        "exact_actions::svb::integer_code_forward",
        "exact_actions::svb::integer_code_inverse",
    ),
}

SIMPLE8B_BINDINGS = {
    "scaling": (
        "float64", "bytes",
        "exact_actions::simple8b::scaling_forward",
        "exact_actions::simple8b::scaling_inverse",
    ),
    "bitpack_int": (
        "int64|bytes", "bytes",
        "exact_actions::simple8b::bitpack_forward_int64/bitpack_forward_scaled",
        "exact_actions::simple8b::bitpack_inverse_int64/bitpack_inverse_scaled",
    ),
}

RLBE_BINDINGS = {
    "delta": (
        "int64|float64", "bytes",
        "exact_actions::p04::rlbe_delta_forward",
        "exact_actions::p04::rlbe_delta_inverse",
    ),
    "run_length": (
        "bytes", "bytes",
        "exact_actions::p04::rlbe_run_length_forward",
        "exact_actions::p04::rlbe_run_length_inverse",
    ),
}

SPRINTZ_BINDINGS = {
    # The 28-family taxonomy retains the historical `delta` label, while the
    # actual implementation is Sprintz LongFIRE adaptive prediction.
    "delta": (
        "int64|float64", "bytes",
        "exact_actions::p04::sprintz_predictor_forward",
        "exact_actions::p04::sprintz_predictor_inverse",
    ),
    "zigzag": (
        "bytes", "bytes",
        "exact_actions::p04::sprintz_zigzag_forward",
        "exact_actions::p04::sprintz_zigzag_inverse",
    ),
    "bitpack_int": (
        "bytes", "bytes",
        "exact_actions::p04::sprintz_bitpack_forward",
        "exact_actions::p04::sprintz_bitpack_inverse",
    ),
}

DBP_BINDINGS = {
    "scaling": (
        "float64", "bytes",
        "exact_actions::dbp::scaling_forward",
        "exact_actions::dbp::scaling_inverse",
    ),
    "delta": (
        "int64|bytes", "bytes",
        "exact_actions::dbp::delta_forward_int64/delta_forward_scaled",
        "exact_actions::dbp::delta_inverse_int64/delta_inverse_scaled",
    ),
    "zigzag": (
        "bytes", "bytes",
        "exact_actions::dbp::zigzag_forward",
        "exact_actions::dbp::zigzag_inverse",
    ),
    "bitpack_int": (
        "bytes", "bytes",
        "exact_actions::dbp::varint_forward",
        "exact_actions::dbp::varint_inverse",
    ),
}

GORILLA_BINDINGS = {
    "xor": (
        "int64|float64", "bytes",
        "exact_actions::gorilla::xor_forward_int64/xor_forward_double_bits",
        "exact_actions::gorilla::xor_inverse",
    ),
    "trailing_zero_pack": (
        "bytes", "bytes",
        "exact_actions::gorilla::trailing_pack_forward",
        "exact_actions::gorilla::trailing_pack_inverse_int64/trailing_pack_inverse_double",
    ),
}

CHIMP128_BINDINGS = {
    "xor": (
        "int64|float64", "bytes",
        "exact_actions::chimp128::xor_forward_int64/xor_forward_double_bits",
        "exact_actions::chimp128::xor_inverse",
    ),
    "trailing_zero_pack": (
        "bytes", "bytes",
        "exact_actions::chimp128::trailing_pack_forward",
        "exact_actions::chimp128::trailing_pack_inverse_int64/trailing_pack_inverse_double",
    ),
}

PONGO_BINDINGS = {
    "eraser": (
        "int64|float64", "bytes", "exact_actions::pongo::eraser_forward",
        "exact_actions::pongo::eraser_inverse",
    ),
    "xor": (
        "bytes", "bytes", "exact_actions::pongo::xor_forward",
        "exact_actions::pongo::xor_inverse",
    ),
    "trailing_zero_pack": (
        "bytes", "bytes", "exact_actions::pongo::trailing_pack_forward",
        "exact_actions::pongo::trailing_pack_inverse",
    ),
}

SPDP_BINDINGS = {
    "delta": (
        "bytes", "bytes", "exact_actions::spdp::delta_forward",
        "exact_actions::spdp::delta_inverse",
    ),
    "block_transform": (
        "bytes", "bytes", "exact_actions::spdp::block_transform_forward",
        "exact_actions::spdp::block_transform_inverse",
    ),
    "entropy_special": (
        "bytes", "bytes", "exact_actions::spdp::entropy_forward",
        "exact_actions::spdp::entropy_inverse",
    ),
}

FPC_BINDINGS = {
    "xor": (
        "bytes", "bytes", "exact_actions::fpc::predictor_xor_forward",
        "exact_actions::fpc::predictor_xor_inverse",
    ),
    "trailing_zero_pack": (
        "bytes", "bytes", "exact_actions::fpc::trailing_pack_forward",
        "exact_actions::fpc::trailing_pack_inverse",
    ),
}

PFPC_BINDINGS = {
    "xor": (
        "bytes", "bytes", "exact_actions::pfpc::predictor_xor_forward",
        "exact_actions::pfpc::predictor_xor_inverse",
    ),
    "trailing_zero_pack": (
        "bytes", "bytes", "exact_actions::pfpc::trailing_pack_forward",
        "exact_actions::pfpc::trailing_pack_inverse",
    ),
}

BOS_BINDINGS_BY_ACTION = {
    primitive.BOS_BOSA_ACTIONS[0]: (
        "int64", "bytes", "exact_actions::bos::delta_forward",
        "exact_actions::bos::delta_inverse",
    ),
    primitive.BOS_BOSA_ACTIONS[1]: (
        "bytes", "bytes", "exact_actions::bos::first_for_forward",
        "exact_actions::bos::first_for_inverse",
    ),
    primitive.BOS_BOSA_ACTIONS[2]: (
        "bytes", "bytes", "exact_actions::bos::outlier_forward",
        "exact_actions::bos::outlier_inverse",
    ),
    primitive.BOS_BOSA_ACTIONS[3]: (
        "bytes", "bytes", "exact_actions::bos::second_for_forward",
        "exact_actions::bos::second_for_inverse",
    ),
    primitive.BOS_BOSA_ACTIONS[4]: (
        "bytes", "bytes", "exact_actions::bos::bitpack_forward",
        "exact_actions::bos::bitpack_inverse",
    ),
}

FASTPFOR_BINDINGS = {
    "bitpack_int": (
        "int64|bytes", "bytes", "exact_actions::fastpfor::pfor_stage_forward_*",
        "exact_actions::fastpfor::pfor_stage_inverse_*",
    ),
    "integer_code": (
        "bytes", "bytes", "exact_actions::fastpfor::variable_byte_forward",
        "exact_actions::fastpfor::variable_byte_inverse",
    ),
}

ELF_BINDINGS = {
    "eraser": (
        "float64", "bytes", "exact_actions::elf::eraser_forward_double_bits",
        "exact_actions::elf::eraser_inverse_double_bits",
    ),
    "xor": (
        "bytes", "bytes", "exact_actions::elf::elf_xor_forward",
        "exact_actions::elf::elf_xor_inverse",
    ),
}

ELF_PLUS_BINDINGS = {
    "eraser": ELF_BINDINGS["eraser"],
    "xor": (
        "bytes", "bytes", "exact_actions::elf::elf_plus_xor_forward",
        "exact_actions::elf::elf_plus_xor_inverse",
    ),
}

CAMEL_BINDINGS = {
    "eraser": (
        "int64|float64", "bytes",
        "exact_actions::camel::eraser_forward_int64_numeric/eraser_forward",
        "exact_actions::camel::eraser_inverse_int64_numeric/eraser_inverse",
    ),
    "xor": (
        "bytes", "bytes", "exact_actions::camel::xor_forward",
        "exact_actions::camel::xor_inverse",
    ),
    "bitstream_pack": (
        "bytes", "bytes", "exact_actions::camel::bitstream_pack_forward",
        "exact_actions::camel::bitstream_pack_inverse",
    ),
}

SNAPPY_BINDINGS = {
    "lz_match": (
        "bytes", "bytes", "exact_actions::snappy::lz_match_forward",
        "exact_actions::snappy::lz_match_inverse",
    ),
    "match_token_format": (
        "bytes", "bytes", "exact_actions::snappy::token_format_forward",
        "exact_actions::snappy::token_format_inverse",
    ),
    "integer_code": (
        "bytes", "bytes", "exact_actions::snappy::integer_code_forward",
        "exact_actions::snappy::integer_code_inverse",
    ),
}

LZ4_BINDINGS = {
    "lz_match": (
        "bytes", "bytes", "exact_actions::lz4::lz_match_forward",
        "exact_actions::lz4::lz_match_inverse",
    ),
    "match_token_format": (
        "bytes", "bytes", "exact_actions::lz4::token_format_forward",
        "exact_actions::lz4::token_format_inverse",
    ),
    "integer_code": (
        "bytes", "bytes", "exact_actions::lz4::integer_code_forward",
        "exact_actions::lz4::integer_code_inverse",
    ),
    "container_frame": (
        "bytes", "bytes", "exact_actions::lz4::container_forward",
        "exact_actions::lz4::container_inverse",
    ),
}

FPZIP_BINDINGS = {
    "delta": (
        "float64", "bytes", "exact_actions::fpzip::predictor_forward",
        "exact_actions::fpzip::predictor_inverse",
    ),
    "entropy_range": (
        "bytes", "bytes", "exact_actions::fpzip::range_forward",
        "exact_actions::fpzip::range_inverse",
    ),
}

METHOD_BLOCKERS = {
    "Chimp": "",
    "Pongo": "",
    "BOS": "",
    "REGER": "value reorder, regression residuals, and packed exceptions are not exposed as reversible public stages",
    "SVB": "integer core is split exactly; float scaling and benchmark type/scale framing are not yet connected",
    "FastPFor": "source C++ is split into FastPFor<8> prefix and VariableByte tail; current benchmark adapter separately uses a Zstd-limb surrogate",
    "Simple8b": "selector choice and packed-word emission are private fused functions",
    "ALP": "factor/exponent selection, exception stream, fallback, and FastPFor payload need source extraction",
    "LECO": "current benchmark path is a FastPFor surrogate; regression predictor stage is not connected",
    "Camel": "",
    "Elf": "exact on the float32-representable supported domain",
    "Elf+": "exact on the float32-representable supported domain",
    "FPC": "",
    "PFPC": "",
    "SPDP": "",
    "CORAD": "current benchmark implementation has no lossless decoder to the original column",
    "BUFF": "current precision-bounded kernel has no exact float64 reconstruction",
    "FPZIP": "",
    "gzip": "RFC 1952 container is split exactly; DEFLATE match parser and Huffman stages remain fused",
    "Brotli": "upstream dictionary/LZ/context-Huffman intermediate streams are private",
    "Zstd": "public sequence API combines matching and parse policy; replay is reversible but is not an atomic split and is not always byte/size-identical",
    "LZ4": "token, integer-length, and container stages are independent; lz_match currently extracts records by invoking the whole LZ4 compressor",
    "XZ": "XZ stream header/footer is split exactly; block headers, LZMA2 parser, range coder, and index remain fused",
    "Snappy": "token and integer-code stages are independent, but lz_match extracts records by invoking whole Google Snappy; the benchmark also uses cramjam/snap",
    "Blosc": "shuffle/block control and BloscLZ token state are fused; frozen lossless level is unresolved",
    "Blosc2": "filter/block control and BloscLZ token state are fused; frozen lossless level is unresolved",
    "Bzip2": "BWT, MTF/RLE, Huffman tables, and block framing need export from the patched source",
    "Zlib": "RFC 1950 container is split exactly; DEFLATE parser and Huffman stages remain fused",
}


def load_catalog(executable: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        [str(executable)], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "catalog CLI failed")
    return list(csv.DictReader(completed.stdout.splitlines()))


def row_contract(row: dict[str, str], order: int) -> dict[str, Any]:
    method = row["method"]
    operator_id = row["operator_id"]
    if method in {"Chimp", "Chimp128"}:
        input_kind, output_kind, forward, inverse = CHIMP128_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Pongo":
        input_kind, output_kind, forward, inverse = PONGO_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "SPDP":
        input_kind, output_kind, forward, inverse = SPDP_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "FPC":
        input_kind, output_kind, forward, inverse = FPC_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "PFPC":
        input_kind, output_kind, forward, inverse = PFPC_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Gorilla":
        input_kind, output_kind, forward, inverse = GORILLA_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "2DIFF":
        input_kind, output_kind, forward, inverse = TS2DIFF_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "SVB":
        input_kind, output_kind, forward, inverse = SVB_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Simple8b":
        input_kind, output_kind, forward, inverse = SIMPLE8B_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "RLBE":
        input_kind, output_kind, forward, inverse = RLBE_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Sprintz":
        input_kind, output_kind, forward, inverse = SPRINTZ_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "DBP":
        input_kind, output_kind, forward, inverse = DBP_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "BOS":
        input_kind, output_kind, forward, inverse = BOS_BINDINGS_BY_ACTION[
            row["action_name"]
        ]
        status = "exact_executable"
        blocker = ""
    elif method == "FastPFor":
        input_kind, output_kind, forward, inverse = FASTPFOR_BINDINGS[operator_id]
        status = "exact_executable_source_variant"
        blocker = ""
    elif method == "Camel":
        input_kind, output_kind, forward, inverse = CAMEL_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Elf":
        input_kind, output_kind, forward, inverse = ELF_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Elf+":
        input_kind, output_kind, forward, inverse = ELF_PLUS_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method == "Snappy":
        input_kind, output_kind, forward, inverse = SNAPPY_BINDINGS[operator_id]
        status = (
            "non_atomic_experimental_unregistered"
            if row["evidence"] == "non_atomic_experimental"
            else "exact_executable_source_variant"
        )
        blocker = METHOD_BLOCKERS[method]
    elif method == "LZ4":
        input_kind, output_kind, forward, inverse = LZ4_BINDINGS[operator_id]
        status = (
            "non_atomic_experimental_unregistered"
            if row["evidence"] == "non_atomic_experimental"
            else "exact_executable"
        )
        blocker = METHOD_BLOCKERS[method] if status.startswith("non_atomic") else ""
    elif method == "FPZIP":
        input_kind, output_kind, forward, inverse = FPZIP_BINDINGS[operator_id]
        status = "exact_executable"
        blocker = ""
    elif method in {"gzip", "Zlib"} and operator_id == "container_frame":
        input_kind = output_kind = "bytes"
        prefix = "gzip" if method == "gzip" else "zlib"
        forward = f"exact_actions::deflate_container::{prefix}_forward"
        inverse = f"exact_actions::deflate_container::{prefix}_inverse"
        status = "exact_executable"
        blocker = ""
    elif method == "XZ" and operator_id == "container_frame":
        input_kind = output_kind = "bytes"
        forward = "exact_actions::xz_container::forward"
        inverse = "exact_actions::xz_container::inverse"
        status = "exact_executable"
        blocker = ""
    elif method == "RLE":
        input_kind, output_kind = "int64", "bytes"
        forward = "tsfile::p02::encode_rle_int64"
        inverse = "tsfile::p02::decode_rle_int64"
        status = "exact_executable"
        blocker = ""
    else:
        input_kind = output_kind = "unwired"
        forward = inverse = ""
        status = "not_executable_stagewise"
        blocker = METHOD_BLOCKERS[method]
    return {
        "method": method,
        "kind": "specialist" if method in EXPECTED_METHODS[:25] else "general",
        "selected_level": row["selected_level"],
        "stage_order": order,
        "operator_id": operator_id,
        "action_name": row["action_name"],
        "input_kind": input_kind,
        "output_kind": output_kind,
        "forward_symbol": forward,
        "inverse_symbol": inverse,
        "implementation_source": row["implementation_source"],
        "evidence": row["evidence"],
        "status": status,
        "blocker": blocker,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-cli",
        type=Path,
        default=ROOT / "cpp/build/remaining_codec_action_catalog_cli",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog_cli)
    if len(catalog) != 102:
        raise RuntimeError(f"expected 102 corrected stage records, got {len(catalog)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in catalog:
        grouped[row["method"]].append(row)
    if tuple(grouped) != EXPECTED_METHODS:
        raise RuntimeError(f"35-codec target drift: {tuple(grouped)!r}")

    contracts = [
        row_contract(row, order)
        for method in EXPECTED_METHODS
        for order, row in enumerate(grouped[method], 1)
    ]
    registered_names = {action.name for action in primitive.POLICY_ACTIONS}
    exact_contracts = [
        row for row in contracts if row["status"].startswith("exact_executable")
    ]
    missing_registered = [
        row["action_name"] for row in exact_contracts
        if row["action_name"] not in registered_names
    ]
    if missing_registered:
        raise RuntimeError(
            "exact contract rows are not callable through POLICY_ACTIONS: "
            + ", ".join(missing_registered)
        )
    summaries: list[dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        rows = [row for row in contracts if row["method"] == method]
        exact = sum(row["status"].startswith("exact_executable") for row in rows)
        if method in {"Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "Simple8b", "RLBE", "Sprintz", "DBP", "SPDP", "FPC", "PFPC", "BOS", "FastPFor", "Camel", "Elf", "Elf+", "FPZIP"}:
            status = "exact_stage_chain_supported_domain"
        elif method == "Snappy":
            status = "partial_source_variant_match_extractor_unregistered"
        elif method == "LZ4":
            status = "wire_replay_exact_match_extractor_unregistered"
        elif method in {"gzip", "Zlib", "XZ"}:
            status = "partial_exact_container_only"
        elif method == "RLE":
            status = "exact_typed_chain_integer_rle"
        elif method == "SVB":
            status = "exact_stage_chain_supported_domain"
        else:
            status = "requires_source_stage_extraction"
        summaries.append(
            {
                "method": method,
                "kind": rows[0]["kind"],
                "selected_level": rows[0]["selected_level"],
                "stage_occurrences": len(
                    PIPELINE_OCCURRENCE_OVERRIDES.get(
                        method, tuple(row["operator_id"] for row in rows)
                    )
                ),
                "unique_action_candidates": len(rows),
                "ordered_operators": " -> ".join(
                    PIPELINE_OCCURRENCE_OVERRIDES.get(
                        method, tuple(row["operator_id"] for row in rows)
                    )
                ),
                "exact_executable_stages": exact,
                "unwired_stages": len(rows) - exact,
                "status": status,
                "blocker": "" if exact == len(rows) else METHOD_BLOCKERS[method],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "split_contract_35.csv", contracts)
    write_csv(args.output_dir / "codec_summary_35.csv", summaries)
    modes = Counter(item.execution_mode for item in unified.ACTIONS)
    exact_action_represented = [
        row["method"] for row in summaries if row["exact_executable_stages"] > 0
    ]
    partial_exact_core = [
        row["method"]
        for row in summaries
        if 0 < row["exact_executable_stages"] < row["unique_action_candidates"]
    ]
    remaining_source_extraction = [
        row["method"] for row in summaries if row["unwired_stages"] > 0
    ]
    payload = {
        "schema": "exact-action-split-35-v1",
        "target_codecs": len(summaries),
        "specialists": sum(row["kind"] == "specialist" for row in summaries),
        "generals": sum(row["kind"] == "general" for row in summaries),
        "method_scoped_stage_records": len(contracts),
        "method_scoped_unique_action_names": len(
            {row["action_name"] for row in contracts}
        ),
        "pipeline_stage_occurrences": sum(
            row["stage_occurrences"] for row in summaries
        ),
        "exact_executable_stage_records_in_target": sum(
            row["status"].startswith("exact_executable") for row in contracts
        ),
        "exact_executable_unique_actions_in_target": len(
            {row["action_name"] for row in exact_contracts}
        ),
        "exact_contract_actions_missing_from_policy_registry": [],
        "exact_action_represented_target_codecs": exact_action_represented,
        "partial_exact_core_codecs": partial_exact_core,
        "remaining_source_extraction_codecs": remaining_source_extraction,
        "registered_trainable_actions": len(primitive.POLICY_ACTIONS),
        "callable_action_implementation_records": len(
            primitive.ALL_CONCRETE_ACTION_IMPLEMENTATIONS
        ),
        "policy_excluded_non_atomic_actions": sorted(
            primitive.POLICY_EXCLUDED_NON_ATOMIC_ACTION_NAMES
        ),
        "union_inventory_records": len(unified.ACTIONS),
        "new_exact_actions_outside_corrected_method_scoped_102": [
            primitive.RLE_SCALING_ACTION
        ],
        "unified_execution_modes": dict(sorted(modes.items())),
        "identity_boundary": (
            "whole-codec wrappers, identity stages, omission labels, and same-family proxies "
            "are never promoted to exact executable actions"
        ),
    }
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 35 个未通过 codec 的逐 stage 拆分状态",
        "",
        f"- 目标集合已校正为 **{payload['specialists']} specialist + {payload['generals']} general = 35**；RLE 取代已通过的 ANS。",
        f"- 修正后共列出 **{len(contracts)}** 个 method-scoped stage；当前有 **{payload['exact_executable_stage_records_in_target']}** 个已具备真实 forward/inverse。",
        "- 当前 benchmark 的 Chimp/Chimp128 共用同一 TSFile 实现与两段 action；BOS、Pongo、Camel 与 FPZIP 具备完整独立链，Elf/Elf+ 在有限域内具备独立链。source C++ FastPFor 已拆成 PFor prefix + VariableByte tail；同名 Python benchmark 仍是 Zstd-limb surrogate。LZ4/Snappy 的 match helper 仍调用 whole codec；gzip/Zlib/XZ 目前只完成容器边界。",
        f"- 仍需 source extraction 的 codec：**{len(payload['remaining_source_extraction_codecs'])}**。",
        f"- 当前训练 action registry：**{payload['registered_trainable_actions']}**；纠正后 union 为 **{payload['union_inventory_records']}** 条，执行边界为 `{payload['unified_execution_modes']}`。",
        "",
        "| codec | 类型 | level | 有序 stage | 已实现/总数 | 状态 |",
        "|---|---|---|---|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['kind']} | {row['selected_level']} | "
            f"{row['ordered_operators']} | {row['exact_executable_stages']}/{row['unique_action_candidates']} | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "逐 stage 的 typed I/O、真实符号、源码和 blocker 见 `split_contract_35.csv`。",
            "",
        ]
    )
    (args.output_dir / "SPLIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
