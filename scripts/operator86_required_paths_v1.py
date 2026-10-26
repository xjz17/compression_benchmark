#!/usr/bin/env python3
"""Build evidence-bounded required paths for the fixed benchmark-86 labels.

This module is deliberately a catalog reader.  It does not import the operator
runtime, mutate a registry, or claim that a baseline codec wrapper is an
independently schedulable graph.  In particular, the following are separate:

* the Methods-page mechanism path (documentary context),
* the audited historical A-ID/runtime-action sequence, and
* registration in the current policy graph (not established by these inputs).

``build_method_requirements`` is the integration API.  Its return value is a
JSON-serializable list in the authoritative 28-specialist + 58-general order.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_INPUTS = (
    "scripts/primitive_operator_action_runtime.py",
    "docs/operator_102/handbook_content.py",
    "docs/operator_102/current_handbook.json",
    "docs/operator_102/source_inputs/methods_86_audit.json",
    "benchmark/reference/methods/method_operator_pipelines.json",
    "benchmark/reference/methods/method_metadata.json",
    "benchmark/reference/methods/encoding_operators.json",
    "benchmark/reference/methods/byte_codec_operators.json",
)

# Labels whose benchmark spelling intentionally differs from the Methods-page
# key.  All other labels are resolved by punctuation-insensitive matching.
PIPELINE_ALIASES = {
    "Chimp": "CHIMP",
    "Gorilla": "GORILLA",
    "2DIFF": "TS_2DIFF",
    "SVB": "Stream VByte",
    "FastPFor": "fastpfor",
    "Simple8b": "simple8b",
    "Sprintz": "SPRINTZ",
    "Dict.": "col_dictionary",
    "DBP": "DELTA_BINARY_PACKED",
    "Camel": "camel",
    "Elf": "elf",
    "Elf+": "elf_plus",
    "FPC": "fpc",
    "PFPC": "pfpc",
    "SPDP": "spdp",
    "CORAD": "corad_tricklet",
    "BUFF": "buff",
    "FPCmp-R": "FPCompress DPratio",
    "FPCmp-S": "FPCompress DPspeed",
    "Brotli": "brotli",
    "Zstd": "zstd",
    "Snappy": "snappy",
    "Libdeflate": "libdeflate",
    "NRV": "UCL NRV",
}

# These mechanism families need decoder-visible history or framing continuity.
# This is a conservative contract marker, not a claim about an implementation.
STATEFUL_MECHANISMS = frozenset(
    {
        "eraser",
        "scaling",
        "frame_of_reference",
        "value_reorder",
        "regression_predictor",
        "delta",
        "xor",
        "trailing_zero_pack",
        "dictionary",
        "block_transform",
        "symbol_recode",
        "run_length",
        "lz_match",
        "match_parse_policy",
        "match_window_model",
        "match_token_format",
        "integer_code",
        "bitstream_pack",
        "entropy_huffman",
        "entropy_range",
        "entropy_fse",
        "entropy_special",
        "literal_entropy",
        "block_state_control",
        "container_frame",
    }
)

# The coarse Methods path omits details required for byte-exact Zstd.  Listing
# them as gaps prevents A077/A080 or an upstream zstd wrapper from being treated
# as proof of a decomposed Zstd graph.
METHOD_EXACTNESS_GAPS: dict[str, list[dict[str, str]]] = {
    "Zstd": [
        {
            "requirement": "literal_huff0_table_and_stream",
            "detail": "Real Zstd literal Huff0 table description/reuse and literal stream coding are not established by the documentary literal_entropy stage.",
        },
        {
            "requirement": "ll_ml_of_sequence_generation",
            "detail": "Literal-length, match-length and offset-code sequence generation with Zstd extra-bit rules needs an independently typed path.",
        },
        {
            "requirement": "ll_ml_of_fse_tables_and_bitstream",
            "detail": "The three Zstd sequence alphabets require their actual FSE modes/tables, state initialization and reverse bitstream order; project rANS is not evidence for this.",
        },
        {
            "requirement": "repcode_history",
            "detail": "Zstd repeat-offset update rules must be decoder-visible state and must cross sequences exactly.",
        },
        {
            "requirement": "cross_block_and_frame_state",
            "detail": "Block modes, table reuse, dictionary/window state, checksums and frame/block serialization must link across blocks.",
        },
    ],
    "Huff0": [
        {
            "requirement": "huff0_histogram_and_table_build",
            "detail": "The original Huff0 path requires its symbol histogram, code-length/table construction and supported table-reuse decisions; a native-LZ profile is unrelated evidence.",
        },
        {
            "requirement": "huff0_table_description",
            "detail": "The actual Huff0 CTable/DTable description and decoder reconstruction format must be represented as typed, decoder-visible metadata.",
        },
        {
            "requirement": "huff0_single_or_four_stream_bitstream",
            "detail": "Huff0 single-/four-stream symbol coding, stream lengths, bit order and terminal-state checks require source-exact stages and inverse contracts.",
        },
    ],
    "FSE": [
        {
            "requirement": "fse_normalized_counts_and_tables",
            "detail": "The original FSE path requires normalized counts plus CTable/DTable construction or table-description reuse; a native-LZ profile is unrelated evidence.",
        },
        {
            "requirement": "fse_state_machine_and_reverse_bitstream",
            "detail": "FSE encoder/decoder state transitions, interleaving, extra bits, reverse bitstream order and terminal-state checks need source-exact typed stages.",
        },
    ],
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "exists": False, "bytes": None, "sha256": None}
    return {
        "path": relative_path,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_source_receipt(repo_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return fresh hashes for every stable input consumed by this generator."""

    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    return {path: _receipt(root, path) for path in CORE_INPUTS}


def _literal_assignments(path: Path, names: set[str]) -> dict[str, Any]:
    """Read literal data from handbook_content.py without importing it."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                result[target.id] = ast.literal_eval(node.value)
    missing = names - set(result)
    if missing:
        raise ValueError(f"handbook_content.py is missing literal assignments: {sorted(missing)}")
    return result


def _normalize_label(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _resolve_named_key(label: str, mapping: dict[str, Any]) -> str:
    if label in PIPELINE_ALIASES:
        key = PIPELINE_ALIASES[label]
        if key not in mapping:
            raise KeyError(f"pipeline alias for {label!r} points to absent key {key!r}")
        return key
    matches = [key for key in mapping if _normalize_label(key) == _normalize_label(label)]
    if len(matches) != 1:
        raise KeyError(f"expected one Methods-page key for {label!r}, found {matches}")
    return matches[0]


def _variant_path(pipeline: Any, logical_type: str) -> list[str]:
    if isinstance(pipeline, list):
        return list(pipeline)
    if not isinstance(pipeline, dict):
        raise TypeError(f"pipeline must be a list or dict, got {type(pipeline).__name__}")
    key = "int" if logical_type == "int64" else "float"
    if key in pipeline:
        return list(pipeline[key])
    if "shared_byte_chain" in pipeline:
        return list(pipeline["shared_byte_chain"])
    raise KeyError(f"pipeline has no {key!r} variant: {sorted(pipeline)}")


_SOURCE_REF_RE = re.compile(r"^(.*?)(?::(\d+(?:-\d+)?))?$")


def _source_evidence(root: Path, source_ref: str) -> dict[str, Any]:
    match = _SOURCE_REF_RE.fullmatch(source_ref)
    if match is None:
        return {"source_ref": source_ref, "path": source_ref, "locator": None, "sha256": None}
    source_path, locator = match.groups()
    path = Path(source_path)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            return {
                "source_ref": source_ref,
                "path": source_path,
                "locator": locator,
                "sha256": None,
                "note": "outside repository; not hashed",
            }
    else:
        relative = path
    resolved = root / relative
    return {
        "source_ref": source_ref,
        "path": relative.as_posix(),
        "locator": locator,
        "sha256": _sha256(resolved) if resolved.is_file() else None,
        "exists": resolved.is_file(),
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _a_details(
    root: Path,
    a_ids: list[str],
    implementation_by_id: dict[str, dict[str, Any]],
    examples: dict[str, Any],
    overrides: dict[str, Any],
    handbook_content_sha: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for a_id in _unique(a_ids):
        implementation = implementation_by_id.get(a_id)
        if implementation is None:
            result.append(
                {
                    "a_id": a_id,
                    "catalog_status": "gap_missing_from_current_102_handbook",
                    "source_evidence": [],
                }
            )
            continue
        result.append(
            {
                "a_id": a_id,
                "name": implementation["name"],
                "classification": implementation["classification"],
                "catalog_status": "present_in_historical_102_catalog",
                "source_action_names": list(implementation.get("source_action_names", [])),
                "composed_source_action_names": list(
                    implementation.get("composed_source_action_names", [])
                ),
                "source_evidence": [
                    _source_evidence(root, source_file)
                    for source_file in implementation.get("source_files", [])
                ],
                "handbook_content": {
                    "example_present": a_id in examples,
                    "override_present": a_id in overrides,
                    "source": "docs/operator_102/handbook_content.py",
                    "source_sha256": handbook_content_sha,
                },
                "claim_boundary": (
                    "Catalog membership and source association only; this does not prove "
                    "registration in the current policy graph or equivalence to a named upstream codec."
                ),
            }
        )
    return result


def _runtime_sequence_status(grade: str, applicable: bool) -> str:
    if not applicable:
        return "historical_logical_type_gap"
    if grade == "real_executable_action_chain":
        return "historical_audited_executable_runtime_sequence"
    if "method label/profile proxy" in grade:
        return "project_proxy_not_original_algorithm_decomposition"
    if "pending_atomic_rewire" in grade:
        return "executable_but_fused_atomic_rewire_gap"
    if "composite" in grade or "residual" in grade:
        return "executable_but_contains_composite_or_residual_gap"
    if "reference_or_proxy" in grade:
        return "reference_or_proxy_prefix_gap"
    return "audit_grade_requires_review"


def _direct_method_status(adapter: dict[str, Any]) -> str:
    implementation = str(adapter.get("implementation", ""))
    status = str(adapter.get("status", ""))
    if "excluded" in status or implementation in {"", "none"}:
        return "historical_unavailable_or_excluded"
    if "project_native_reference" in implementation:
        return "historical_project_reference_proxy"
    if implementation.startswith("upstream_") or implementation.startswith("stdlib_"):
        return "historical_direct_baseline_implementation"
    return "historical_project_method_adapter"


def _stage_mapping(stage: dict[str, Any]) -> dict[str, Any]:
    a_ids = [
        entry["a_id"] for entry in stage.get("ordered_a_implementations", [])
    ]
    return {
        "runtime_action": stage["runtime_action"],
        "canonical_action": stage.get("canonical_action"),
        "runtime_kind": stage.get("runtime_kind"),
        "mapping_status": stage.get("mapping_status"),
        "historical_a_ids": a_ids,
    }


def _requirements_for_variant(
    root: Path,
    method: str,
    logical_type: str,
    variant: dict[str, Any],
    documentary_path: list[str],
) -> dict[str, Any]:
    dispatch = variant["strongest_exact_action_dispatch"]
    coarse = variant["coarse_declaration_for_context_only"]
    adapter = variant["benchmark_adapter"]
    runtime_status = _runtime_sequence_status(
        str(dispatch.get("coverage_grade", "")), bool(dispatch.get("applicable"))
    )

    gaps: list[dict[str, Any]] = []
    if not dispatch.get("applicable"):
        gaps.append(
            {
                "kind": "logical_type_dispatch",
                "status": "gap",
                "detail": f"No exact audited dispatch is available for {logical_type}.",
            }
        )
    if runtime_status != "historical_audited_executable_runtime_sequence":
        gaps.append(
            {
                "kind": "original_algorithm_decomposition",
                "status": "gap",
                "detail": runtime_status,
            }
        )
    gaps.append(
        {
            "kind": "typed_ports",
            "status": "gap",
            "mechanisms": list(documentary_path),
            "detail": "The Methods-page family path does not specify byte-exact typed input/output ports between every original-algorithm stage.",
        }
    )
    stateful = [stage for stage in documentary_path if stage in STATEFUL_MECHANISMS]
    if stateful:
        gaps.append(
            {
                "kind": "state_continuity",
                "status": "gap",
                "mechanisms": stateful,
                "detail": "Decoder-visible initialization, reset and cross-block/chunk continuity are not established by the family list.",
            }
        )
    for extra in METHOD_EXACTNESS_GAPS.get(method, []):
        gaps.append({"kind": "method_exactness", "status": "gap", **extra})
    gaps.append(
        {
            "kind": "policy_graph_registration",
            "status": "gap",
            "detail": "The consumed sources do not prove that this complete ordered path is registered as a legal path in the current 6341-action policy graph.",
        }
    )

    runtime_actions = list(dispatch.get("ordered_runtime_actions", []))
    stage_mappings = [_stage_mapping(stage) for stage in dispatch.get("ordered_stage_to_a_mapping", [])]
    direct_evidence = [_source_evidence(root, ref) for ref in adapter.get("source_refs", [])]
    dispatch_evidence = [
        _source_evidence(root, ref) for ref in dispatch.get("dispatch_source_refs", [])
    ]
    return {
        "logical_type": logical_type,
        "benchmark_input_contract": variant.get("benchmark_input_contract"),
        "required_mechanism_path": list(documentary_path),
        "audited_runtime_sequence": {
            "status": runtime_status,
            "coverage_grade_verbatim": dispatch.get("coverage_grade"),
            "topology": dispatch.get("topology"),
            "selected_variant_key": dispatch.get("selected_variant_key"),
            "required_parameter_note": dispatch.get("required_parameter_note"),
            "ordered_runtime_actions": runtime_actions,
            "ordered_stage_mapping": stage_mappings,
            "ordered_historical_a_ids": list(dispatch.get("ordered_a_ids", [])),
            "coverage_notes": list(dispatch.get("coverage_notes", [])),
            "source_evidence": dispatch_evidence,
            "evidence_time_scope": "historical methods_86_audit snapshot; fresh file hashes do not revalidate behavior against the current runtime",
            "claim_boundary": (
                "This is historical audit evidence for the audited dispatcher snapshot.  "
                "It is not a refresh of current code behavior and is not, by itself, proof "
                "of upstream-algorithm decomposition or policy-graph registration."
            ),
        },
        "documentary_coarse_sequence": {
            "status": "context_only_not_true_implementation_chain",
            "ordered_actions": list(coarse.get("ordered_actions", [])),
            "ordered_historical_a_ids": list(coarse.get("ordered_a_ids", [])),
            "must_not_be_used_as_true_implementation_chain": True,
            "source_evidence": [
                _source_evidence(root, ref) for ref in coarse.get("source_refs", [])
            ],
        },
        "direct_method_evidence": {
            "status": _direct_method_status(adapter),
            "implementation": adapter.get("implementation"),
            "adapter_status_verbatim": adapter.get("status"),
            "fixed_exact_dispatch_key": adapter.get("fixed_exact_dispatch_key"),
            "input_to_codec": adapter.get("input_to_codec"),
            "note": adapter.get("note"),
            "source_evidence": direct_evidence,
            "evidence_time_scope": "historical methods_86_audit snapshot",
            "current_native_baseline86_refresh_status": "not_refreshed_by_this_builder",
            "proves_decomposed_graph_path": False,
        },
        "policy_graph_registration": {
            "status": "gap_not_established_by_consumed_sources",
            "registered_action_ids": [],
            "complete_path_registered": False,
        },
        "required_gaps": gaps,
        "candidate_seed": {
            "considered": True,
            "status": "required_but_blocked_on_explicit_gaps",
            "ordered_runtime_actions": [],
            "historical_comparison_runtime_actions": runtime_actions,
            "required_original_mechanisms": list(documentary_path),
            "note": "Historical dispatcher actions are retained for comparison and later compilation work, but are not admitted as an original-algorithm seed while gaps remain.",
            "composition_with_102": "prefix/suffix composition allowed only after typed-port, state, inverse and graph-registration checks pass",
        },
    }


def build_method_requirements(repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the 86 evidence-bounded method requirements in frozen order."""

    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    receipts = build_source_receipt(root)
    missing = [path for path, receipt in receipts.items() if not receipt["exists"]]
    if missing:
        raise FileNotFoundError(f"required catalog inputs are missing: {missing}")

    audit = _json(root / "docs/operator_102/source_inputs/methods_86_audit.json")
    pipelines_doc = _json(
        root / "benchmark/reference/methods/method_operator_pipelines.json"
    )
    metadata_doc = _json(root / "benchmark/reference/methods/method_metadata.json")
    encoding_doc = _json(root / "benchmark/reference/methods/encoding_operators.json")
    byte_doc = _json(root / "benchmark/reference/methods/byte_codec_operators.json")
    handbook = _json(root / "docs/operator_102/current_handbook.json")
    handbook_literals = _literal_assignments(
        root / "docs/operator_102/handbook_content.py", {"EXAMPLES", "OVERRIDES"}
    )
    current_runtime_catalog = _literal_assignments(
        root / "scripts/primitive_operator_action_runtime.py",
        {"BENCHMARK_86_SPECIALISTS", "BENCHMARK_86_GENERALS"},
    )

    if audit.get("authoritative_order") != [row["method"] for row in audit["methods"]]:
        raise ValueError("methods_86_audit authoritative order differs from its method rows")
    if len(audit["authoritative_order"]) != 86 or len(set(audit["authoritative_order"])) != 86:
        raise ValueError("methods_86_audit must contain exactly 86 unique authoritative labels")
    current_authoritative_order = list(
        current_runtime_catalog["BENCHMARK_86_SPECIALISTS"]
        + current_runtime_catalog["BENCHMARK_86_GENERALS"]
    )
    if current_authoritative_order != audit["authoritative_order"]:
        raise ValueError(
            "historical methods_86_audit labels/order differ from the current "
            "BENCHMARK_86_SPECIALISTS + BENCHMARK_86_GENERALS AST constants"
        )

    pipeline_map = pipelines_doc["pipelines"]
    metadata_map = {row["name"]: row for row in metadata_doc["methods"]}
    operator_map = {
        row["id"]: (row, "benchmark/reference/methods/encoding_operators.json")
        for row in encoding_doc["operators"]
    }
    operator_map.update(
        {
            row["id"]: (row, "benchmark/reference/methods/byte_codec_operators.json")
            for row in byte_doc["operators"]
        }
    )
    implementation_by_id = {row["id"]: row for row in handbook["implementations"]}
    handbook_content_sha = receipts["docs/operator_102/handbook_content.py"]["sha256"]
    audit_sha = receipts["docs/operator_102/source_inputs/methods_86_audit.json"]["sha256"]
    pipeline_sha = receipts[
        "benchmark/reference/methods/method_operator_pipelines.json"
    ]["sha256"]

    result: list[dict[str, Any]] = []
    for audited_method in audit["methods"]:
        label = audited_method["method"]
        pipeline_key = _resolve_named_key(label, pipeline_map)
        pipeline = pipeline_map[pipeline_key]
        paths_by_type = {
            logical_type: _variant_path(pipeline, logical_type)
            for logical_type in ("int64", "float64")
        }
        if paths_by_type["int64"] != paths_by_type["float64"]:
            raise ValueError(f"{label} has divergent int64/float64 documentary paths")
        required_path = paths_by_type["int64"]
        unknown = [stage for stage in required_path if stage not in operator_map]
        if unknown:
            raise KeyError(f"{label} documentary path contains unknown mechanisms: {unknown}")

        documentary_path = []
        for ordinal, mechanism in enumerate(required_path, 1):
            descriptor, source_path = operator_map[mechanism]
            documentary_path.append(
                {
                    "ordinal": ordinal,
                    "mechanism": mechanism,
                    "name": descriptor["name"],
                    "description": descriptor["description"],
                    "evidence": {
                        "kind": "documentary_mechanism_family",
                        "path": source_path,
                        "source_sha256": receipts[source_path]["sha256"],
                        "json_pointer": f"/operators/{mechanism}",
                    },
                    "typed_port_contract": "gap_not_specified_by_documentary_family",
                    "state_continuity_contract": (
                        "gap_required" if mechanism in STATEFUL_MECHANISMS else "not_established"
                    ),
                }
            )

        variant_requirements = {
            logical_type: _requirements_for_variant(
                root,
                label,
                logical_type,
                audited_method["variants"][logical_type],
                required_path,
            )
            for logical_type in ("int64", "float64")
        }
        all_a_ids = _unique(
            [
                a_id
                for logical_type in ("int64", "float64")
                for a_id in variant_requirements[logical_type]["audited_runtime_sequence"][
                    "ordered_historical_a_ids"
                ]
            ]
            + [
                a_id
                for logical_type in ("int64", "float64")
                for a_id in variant_requirements[logical_type]["documentary_coarse_sequence"][
                    "ordered_historical_a_ids"
                ]
            ]
        )

        try:
            metadata_key = _resolve_named_key(label, metadata_map)
            metadata = dict(metadata_map[metadata_key])
        except KeyError:
            metadata = None

        result.append(
            {
                "ordinal": audited_method["ordinal"],
                "label": label,
                "category": audited_method["benchmark_group"],
                "considered_as_required_candidate": True,
                "methods_page_key": pipeline_key,
                "metadata": metadata,
                "required_path": list(required_path),
                "documentary_path": documentary_path,
                "original_algorithm_requirement": {
                    "status": "required_with_unresolved_exactness_and_graph_gaps",
                    "ordered_documentary_mechanisms": list(required_path),
                    "additional_required_exactness": list(
                        METHOD_EXACTNESS_GAPS.get(label, [])
                    ),
                    "proxy_runtime_sequence_is_substitute": False,
                    "direct_method_wrapper_is_substitute": False,
                },
                "requirements_by_logical_type": variant_requirements,
                "historical_a_ids": all_a_ids,
                "historical_a_evidence": _a_details(
                    root,
                    all_a_ids,
                    implementation_by_id,
                    handbook_literals["EXAMPLES"],
                    handbook_literals["OVERRIDES"],
                    handbook_content_sha,
                ),
                "source_receipts": {
                    "current_authoritative_runtime_catalog_sha256": receipts[
                        "scripts/primitive_operator_action_runtime.py"
                    ]["sha256"],
                    "methods_86_audit_sha256": audit_sha,
                    "method_operator_pipelines_sha256": pipeline_sha,
                    "handbook_content_sha256": handbook_content_sha,
                    "current_handbook_sha256": receipts[
                        "docs/operator_102/current_handbook.json"
                    ]["sha256"],
                },
                "historical_audit_snapshot": {
                    "date": audit.get("date"),
                    "schema": audit.get("schema"),
                    "manifest_a_implementation_count_verbatim": audit.get("counts", {}).get(
                        "current_manifest_a_implementations"
                    ),
                    "note": "The audit predates the current 102 catalog. Fresh source hashes below are provenance only and do not refresh its behavioral claims.",
                },
                "current_authoritative_label_check": {
                    "status": "exact_match",
                    "source": "scripts/primitive_operator_action_runtime.py",
                    "specialist_constant": "BENCHMARK_86_SPECIALISTS",
                    "general_constant": "BENCHMARK_86_GENERALS",
                    "source_sha256": receipts[
                        "scripts/primitive_operator_action_runtime.py"
                    ]["sha256"],
                },
                "claim_boundary": {
                    "documentary_path_is_executable_proof": False,
                    "direct_baseline_is_decomposed_graph_proof": False,
                    "historical_a_id_is_6341_registration_proof": False,
                    "full_graph_admission": "gap until typed/state/inverse/source-exactness and current graph-registration evidence pass",
                },
            }
        )

    if [row["label"] for row in result] != audit["authoritative_order"]:
        raise AssertionError("generated method order drifted from authoritative order")
    return result


def build_document(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    methods = build_method_requirements(root)
    return {
        "schema": "operator86-required-paths-v1",
        "method_count": len(methods),
        "claim_boundary": (
            "All 86 labels are considered required candidates.  A documentary family path, "
            "historical A-ID sequence, direct baseline, or exact dispatcher does not by itself "
            "prove a complete path is registered in the current policy graph."
        ),
        "source_receipt": build_source_receipt(root),
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    payload = build_document(args.repo_root)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=None if args.compact else 2,
    ) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
