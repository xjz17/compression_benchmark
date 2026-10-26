#!/usr/bin/env python3
"""Prepare operator-tree stress inputs and runner plans (metadata only).

The generated input specification is consumed by the C++ stress input builder.
The generated plans are consumed by the C++ batch runner.  This module never
generates payload bytes, invokes an operator, performs timing, or contacts a remote benchmark host.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


INPUT_SPEC_SCHEMA = "operator-tree-stress-input-spec-v3"
INPUTS_SCHEMA = "operator-tree-stress-inputs-v3"
PLAN_SCHEMA = "operator-harness-batch-plan-v2"
SUITE_ID = "operator_tree_stress_v3"
EXCLUDED_IDS = ("A106", "A107", "A108", "A109", "A110")
TARGET_REAL_COUNTS = (4096, 8192)
MAX_CASES = 1000

# These three cases remain source-anchored strict-domain outcomes after the
# native empty-input fixes. Keep the mapping
# exact: an expected label is planning metadata, while the summarizer still
# requires the independently emitted native reason code and message.
EXPECTED_DOMAIN_REJECTIONS = {
    ("A004", "boundary_int64_extrema_17"): {
        "expected_reason_category": "STRICT_INPUT_DOMAIN_REJECTED",
        "expected_reason_code": "CAMEL_INT64_NOT_EXACTLY_REPRESENTABLE",
        "expected_reason": "Camel integer is not exactly representable as float64",
    },
    ("A004", "boundary_int64_tail_4097"): {
        "expected_reason_category": "STRICT_INPUT_DOMAIN_REJECTED",
        "expected_reason_code": "CAMEL_INT64_NOT_EXACTLY_REPRESENTABLE",
        "expected_reason": "Camel integer is not exactly representable as float64",
    },
    ("A038", "boundary_float64_special_18"): {
        "expected_reason_category": "STRICT_INPUT_DOMAIN_REJECTED",
        "expected_reason_code": "GORILLA_RESERVED_MARKER_INPUT",
        "expected_reason": "Gorilla input contains reserved end-marker bit pattern",
    },
}


def _load(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parameter(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _safe_token(value: str) -> str:
    result = "".join(c if c.isalnum() else "_" for c in value)
    return result.strip("_")[:48] or "sample"


def _write_immutable_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_text() == rendered,
                 f"refusing to replace changed metadata artifact {path}")
        return
    with path.open("x") as output:
        output.write(rendered)


def _manifest_rows(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and value, "SOURCE_MANIFESTS must be a nonempty array")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in value:
        _require(isinstance(entry, dict), "source manifest entry must be an object")
        sample, manifest = entry.get("sample"), entry.get("manifest")
        _require(isinstance(sample, dict) and isinstance(manifest, dict),
                 "source manifest entry lacks sample/manifest")
        _require(manifest.get("schema") == "native-baseline86-scaled-column-segment-manifest-v1",
                 "unexpected source manifest schema")
        _require(sample.get("column_identity") == manifest.get("column_identity"),
                 "sample/manifest column identity mismatch")
        key = (str(sample.get("dataset_id")), str(sample.get("column_id")))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"sample": sample, "manifest": manifest})
    return rows


def _candidate_blocks(rows: list[dict[str, Any]], dtype: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        sample, manifest = row["sample"], row["manifest"]
        if sample.get("dtype") != dtype:
            continue
        width = sample.get("source_byte_width")
        _require(type(width) is int and width in (4, 8), "unsupported real source width")
        blocks = manifest.get("blocks")
        _require(isinstance(blocks, list) and blocks, "source manifest has no blocks")
        # Keep all accepted block records.  Some real float32 columns store only
        # hundreds of values per block; the C++ builder can then concatenate
        # consecutive verified value segments into one logical column slice.
        indexes = range(len(blocks))
        for index in indexes:
            block = blocks[index]
            if not isinstance(block, dict):
                continue
            length = block.get("values_length")
            valid = block.get("valid_count")
            if type(length) is not int or type(valid) is not int or length != valid * width:
                continue
            if block.get("dtype") != dtype or block.get("values_path") is None:
                continue
            if not isinstance(block.get("values_sha256"), str) or len(block["values_sha256"]) != 64:
                continue
            candidates.append({"sample": sample, "manifest": manifest,
                               "block": block, "block_index": index,
                               "available_count": valid})
    candidates.sort(key=lambda item: (
        str(item["sample"]["dataset_id"]), str(item["sample"]["column_id"]),
        item["block_index"]))
    return candidates


def _choose_real_blocks(candidates: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    _require(candidates, "no accepted real blocks for dtype")
    selected: list[tuple[int, dict[str, Any]]] = []
    used: set[tuple[str, int]] = set()
    next_index: dict[str, int] = {}
    for target in TARGET_REAL_COUNTS:
        fitting = [item for item in candidates if item["available_count"] >= target]
        if fitting:
            choices = fitting
            actual = target
            choice = next((item for item in choices
                           if (str(item["sample"]["column_id"]), item["block_index"]) not in used),
                          choices[0])
            used.add((str(choice["sample"]["column_id"]), choice["block_index"]))
            choice = {**choice, "segments": [choice]}
            selected.append((actual, choice))
            continue

        # No individual accepted block is large enough.  Preserve the manifest
        # block order and concatenate only its valid-value payload segments.  Each
        # full source block is independently hash checked by the C++ builder.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in candidates:
            grouped.setdefault(str(item["sample"]["column_id"]), []).append(item)
        viable = [(column_id, group) for column_id, group in sorted(grouped.items())
                  if sum(item["available_count"] for item in group) >= target]
        _require(viable, f"accepted blocks contain fewer than {target} values in total")
        column_id, group = viable[0]
        group.sort(key=lambda item: item["block_index"])
        start = next_index.get(column_id, 0)
        if sum(item["available_count"] for item in group[start:]) < target:
            start = 0
        segments: list[dict[str, Any]] = []
        accumulated = 0
        position = start
        while position < len(group) and accumulated < target:
            segment = group[position]
            take = min(segment["available_count"], target - accumulated)
            segments.append({**segment, "take_count": take})
            accumulated += take
            position += 1
        _require(accumulated == target, "failed to form logical accepted-block sequence")
        next_index[column_id] = position
        selected.append((target, {**segments[0], "segments": segments,
                                  "logical_concatenation": True}))
    return selected


def _real_slice(dtype: str, requested: int, actual: int,
                chosen: Mapping[str, Any]) -> dict[str, Any]:
    sample = chosen["sample"]
    block = chosen["block"]
    width = sample["source_byte_width"]
    block_id = str(block.get("block_id", f"block-{chosen['block_index']}"))
    sample_id = (f"real_{_safe_token(dtype)}_{actual}v_"
                 f"{_safe_token(sample['dataset_id'])}_{str(sample['column_id'])[:12]}_"
                 f"b{chosen['block_index']:06d}")
    segments = chosen.get("segments", [chosen])
    source_segments = []
    for segment in segments:
        segment_block = segment["block"]
        take_count = segment.get("take_count", actual)
        source_segments.append({
            "source_values_path": segment_block["values_path"],
            "source_block_offset": segment_block["values_offset"],
            "source_block_bytes": segment_block["values_length"],
            "source_block_sha256": segment_block["values_sha256"],
            "slice_offset_in_source_block_bytes": 0,
            "element_count": take_count,
            "source_block_id": segment_block.get("block_id"),
            "source_block_index": segment["block_index"],
            "source_block_valid_count": segment_block["valid_count"],
            "source_block_missing_count": segment_block.get("missing_count"),
        })
    result = {
        "sample_id": sample_id,
        "category": "large_real",
        "case_class": "large_real",
        "dtype": dtype,
        "source_byte_width": width,
        "element_count": actual,
        "requested_element_count": requested,
        "requested_size_satisfied": actual == requested,
        "source_block_id": block_id,
        "source_block_index": chosen["block_index"],
        "source_block_valid_count": block["valid_count"],
        "source_block_missing_count": block.get("missing_count"),
        "source_manifest": sample["source_manifest"],
        "source_manifest_sha256": sample["source_manifest_sha256"],
        "dataset_id": sample["dataset_id"],
        "column_id": sample["column_id"],
        "column_identity": deepcopy(sample["column_identity"]),
        "source_segments": source_segments,
        "source_layout": ("logical_valid_value_sequence_across_accepted_blocks"
                          if len(source_segments) > 1 else
                          "single_accepted_block_contiguous_slice"),
        "source_boundaries_preserved": True,
        "provenance_policy": (
            "each full accepted block hash then aligned contiguous value segment; "
            "multi-block samples concatenate segments in manifest order"
        ),
    }
    if len(source_segments) == 1:
        result.update(source_segments[0])
    return result


def _fixture(sample_id: str, fixture: str, dtype: str, width: int,
             count: int, family: str, **extra: Any) -> dict[str, Any]:
    result = {
        "sample_id": sample_id,
        "fixture": fixture,
        "dtype": dtype,
        "source_byte_width": width,
        "element_count": count,
        "boundary_family": family,
    }
    result.update(extra)
    return result


def _boundary_fixtures() -> list[dict[str, Any]]:
    fixtures = [
        _fixture("boundary_bytes_empty", "empty", "bytes", 1, 0, "empty"),
        _fixture("boundary_bytes_singleton_ff", "bytes_singleton_ff", "bytes", 1, 1, "singleton"),
        _fixture("boundary_bytes_patterns_17", "bytes_pattern_cycle", "bytes", 1, 17, "byte_patterns"),
        _fixture("boundary_bytes_tail_8193", "bytes_tail_cycle", "bytes", 1, 8193, "non_multiple_tail"),

        _fixture("boundary_int64_empty", "empty", "int64", 8, 0, "empty"),
        _fixture("boundary_int64_singleton_min", "int64_singleton_min", "int64", 8, 1, "singleton"),
        _fixture("boundary_int64_extrema_17", "int64_extrema_cycle", "int64", 8, 17, "integer_extrema"),
        _fixture("boundary_int64_tail_4097", "int64_tail_cycle", "int64", 8, 4097, "non_multiple_tail"),

        _fixture("boundary_float64_empty", "empty", "float64", 8, 0, "empty"),
        _fixture("boundary_float64_singleton_negzero", "float64_singleton_negative_zero", "float64", 8, 1, "singleton_signed_zero"),
        _fixture("boundary_float64_finite_17", "float64_finite_extrema_cycle", "float64", 8, 17, "finite_extrema_subnormal_signed_zero"),
        _fixture("boundary_float64_special_18", "float64_special_cycle", "float64", 8, 18, "nonfinite_nan_payloads"),
        _fixture("boundary_float64_tail_4097", "float64_tail_cycle", "float64", 8, 4097, "non_multiple_tail"),

        _fixture("boundary_float32_empty", "empty", "float32", 4, 0, "empty"),
        _fixture("boundary_float32_singleton_negzero", "float32_singleton_negative_zero", "float32", 4, 1, "singleton_signed_zero"),
        _fixture("boundary_float32_finite_17", "float32_finite_extrema_cycle", "float32", 4, 17, "finite_extrema_subnormal_signed_zero"),
        _fixture("boundary_float32_special_18", "float32_special_cycle", "float32", 4, 18, "nonfinite_nan_payloads"),
        _fixture("boundary_float32_tail_4097", "float32_tail_cycle", "float32", 4, 4097, "non_multiple_tail"),

        _fixture("boundary_uint32_empty", "empty", "uint32", 4, 0, "empty"),
        _fixture("boundary_uint32_patterns_17", "uint32_bit_patterns_cycle", "uint32", 4, 17, "unsigned_bit_patterns"),
        _fixture("boundary_uint32_tail_4097", "uint32_bit_patterns_cycle", "uint32", 4, 4097, "non_multiple_tail"),

        _fixture("boundary_uint64_empty", "empty", "uint64", 8, 0, "empty"),
        _fixture("boundary_uint64_singleton_max", "uint64_singleton_max", "uint64", 8, 1, "singleton"),
        _fixture("boundary_uint64_patterns_17", "uint64_bit_patterns_cycle", "uint64", 8, 17, "unsigned_bit_patterns"),
        _fixture("boundary_uint64_tail_4097", "uint64_bit_patterns_cycle", "uint64", 8, 4097, "non_multiple_tail"),
    ]
    for bits in range(65):
        fixtures.append(_fixture(
            f"boundary_a048_b{bits:02d}_257", "uint64_bounded_cycle", "uint64", 8,
            257, "a048_bit_width_domain", bit_width=bits,
            operator_parameter_overrides={"bit_width": str(bits)},
        ))
    return fixtures


def build_input_spec(source_manifests: Path, output: Path,
                     boundary_only: bool = False) -> dict[str, Any]:
    value, raw = _load(source_manifests)
    rows = _manifest_rows(value)
    real: list[dict[str, Any]] = []
    coverage: dict[str, list[dict[str, Any]]] = {}
    if not boundary_only:
        for dtype in ("int64", "float64", "float32"):
            chosen = _choose_real_blocks(_candidate_blocks(rows, dtype))
            coverage[dtype] = []
            for requested, (actual, candidate) in zip(TARGET_REAL_COUNTS, chosen):
                item = _real_slice(dtype, requested, actual, candidate)
                real.append(item)
                coverage[dtype].append({
                    "requested_element_count": requested,
                    "actual_element_count": actual,
                    "satisfied": requested == actual,
                    "sample_id": item["sample_id"],
                })
    spec = {
        "schema": INPUT_SPEC_SCHEMA,
        "suite_id": SUITE_ID,
        "terminology": "operator_tree",
        "case_scope": "boundary-only" if boundary_only else "all",
        "source_manifests_path": str(source_manifests.resolve()),
        "source_manifests_sha256": _sha(raw),
        "real_size_coverage": coverage,
        "real_slices": real,
        "boundary_fixtures": _boundary_fixtures(),
        "excluded_ids": list(EXCLUDED_IDS),
    }
    _write_immutable_json(output, spec)
    return spec


def _operator_rows(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = catalog.get("operators")
    _require(isinstance(rows, list) and len(rows) == 102, "catalog must contain 102 operators")
    ids = [row.get("implementation_id") for row in rows]
    _require(len(set(ids)) == 102, "catalog implementation IDs are not unique")
    selectable = [row for row in rows if row.get("selection", {}).get("selectable")]
    _require(len(selectable) == 97, "catalog must expose exactly 97 eligible operators")
    _require(set(ids) >= set(EXCLUDED_IDS), "catalog lacks excluded IDs")
    _require(not set(EXCLUDED_IDS) & {row["implementation_id"] for row in selectable},
             "excluded ID became selectable")
    return selectable


def _real_samples(samples: list[dict[str, Any]], dtype: str) -> list[dict[str, Any]]:
    choices = [s for s in samples if s.get("case_class") == "large_real" and s.get("dtype") == dtype]
    choices.sort(key=lambda s: (s.get("requested_element_count", 0), s["sample_id"]))
    _require(len(choices) >= 2, f"need two large real samples for {dtype}")
    return choices[:2]


def _fixture_samples(samples: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    choices = [s for s in samples if s.get("case_class") == "boundary" and
               str(s.get("sample_id", "")).startswith(prefix)]
    choices.sort(key=lambda s: s["sample_id"])
    _require(choices, f"no boundary samples for prefix {prefix}")
    return choices


def _sample_selection(row: Mapping[str, Any], samples: list[dict[str, Any]],
                      include_real: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kind, view = row["initial_kind"], row["input_view"]
    if row["implementation_id"] == "A048" and row.get("present_pipeline", [{}])[0].get("action") == "harness.a048.byte_planes.v1":
        real = _real_samples(samples, "int64") if include_real else []
        boundary = _fixture_samples(samples, "boundary_a048_")
        boundary += [next(s for s in samples if s["sample_id"] == "boundary_uint64_patterns_17")]
        return real, boundary
    if kind == "serial":
        real = ([_real_samples(samples, "int64")[0], _real_samples(samples, "float64")[1]]
                if include_real else [])
        return real, _fixture_samples(samples, "boundary_bytes_")
    if view == "int64":
        return (_real_samples(samples, "int64") if include_real else []), _fixture_samples(samples, "boundary_int64_")
    if view in ("float64", "physical_word"):
        return (_real_samples(samples, "float64") if include_real else []), _fixture_samples(samples, "boundary_float64_")
    if view == "float32":
        return (_real_samples(samples, "float32") if include_real else []), _fixture_samples(samples, "boundary_float32_")
    if view == "uint32":
        return (_real_samples(samples, "float32") if include_real else []), _fixture_samples(samples, "boundary_uint32_")
    if view == "uint64":
        return (_real_samples(samples, "int64") if include_real else []), _fixture_samples(samples, "boundary_uint64_")
    raise ValueError(f"{row['implementation_id']}: unsupported input view {view!r}")


def _recipe(row: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    pipeline = deepcopy(row["present_pipeline"])
    _require(isinstance(pipeline, list) and pipeline, "operator pipeline is empty")
    overrides = sample.get("operator_parameter_overrides")
    if overrides is not None:
        _require(row["implementation_id"] == "A048", "parameter override is restricted to A048")
        focal = row["focal_step_index"]
        _require(pipeline[focal]["action"] == "harness.a048.byte_planes.v1",
                 "A048 parameter override targets unexpected stage")
        bits = overrides.get("bit_width")
        _require(isinstance(bits, str) and bits.isdigit() and 0 <= int(bits) <= 64,
                 "A048 bit_width is outside 0..64")
        pipeline[focal]["parameters"] = {**pipeline[focal].get("parameters", {}),
                                          "bit_width": bits}
    steps = []
    for index, stage in enumerate(pipeline):
        steps.append({
            "kind": "stage",
            "inputs": [0 if index == 0 else index + 1],
            "stage": {
                "action": stage["action"],
                "profile": stage["profile"],
                "parameters": {key: _parameter(value)
                               for key, value in stage.get("parameters", {}).items()},
            },
        })
    steps.append({"kind": "stop"})
    return {"policy_registry": "operator-harness-v2", "steps": steps}


def _domain_probe(sample: Mapping[str, Any]) -> bool:
    return sample.get("boundary_family") in {
        "nonfinite_nan_payloads", "finite_extrema_subnormal_signed_zero",
        "integer_extrema", "empty",
    }


def prepare_plans(catalog_path: Path, inputs_path: Path, output: Path,
                  runtime_root: str, repeats: int,
                  case_scope: str = "all") -> dict[str, Any]:
    catalog, catalog_raw = _load(catalog_path)
    inputs, inputs_raw = _load(inputs_path)
    _require(isinstance(catalog, dict), "catalog must be an object")
    _require(isinstance(inputs, dict) and inputs.get("schema") == INPUTS_SCHEMA,
             "wrong stress inputs schema")
    _require(repeats == 5, "stress-v3 acceptance requires exactly five timed repeats")
    _require(case_scope in ("all", "boundary-only"), "invalid case scope")
    rows = _operator_rows(catalog)
    samples = inputs.get("samples")
    _require(isinstance(samples, list) and samples, "stress inputs contain no samples")
    sample_ids = [s.get("sample_id") for s in samples]
    _require(len(sample_ids) == len(set(sample_ids)), "duplicate stress sample ID")
    root = runtime_root.rstrip("/")
    catalog_hash = _sha(catalog_raw)
    cases_by_batch: dict[str, list[dict[str, Any]]] = {
        batch: [] for batch in catalog["batch_order"]
    }
    for row in rows:
        real, boundary = _sample_selection(row, samples, case_scope == "all")
        selected = real + boundary
        _require(case_scope == "boundary-only" or len(real) == 2,
                 f"{row['implementation_id']}: real sample count differs from two")
        planned_path = ("wrapper_allowed" if row.get("domain_execution", {}).get("measurement_wrapper")
                        else "strict_required")
        for sample in selected:
            sample_id = sample["sample_id"]
            case_id = f"{row['implementation_id']}__stress_v3__{sample_id}"
            expected_rejection = EXPECTED_DOMAIN_REJECTIONS.get(
                (row["implementation_id"], sample_id)
            )
            request = {
                "schema": "operator-harness-request-v2",
                "command": "run",
                "request_id": case_id,
                "implementation_ids": [row["implementation_id"]],
                "catalog_path": root + "/source/benchmark/reference/actions/operator_harness_tree_catalog_v3.json",
                "catalog_sha256": catalog_hash,
                "input_path": root + "/inputs/" + sample_id + ".bin",
                "initial_kind": row["initial_kind"],
                "initial_width": row["initial_width"],
                "input_view": row["input_view"],
                "dtype": sample["dtype"],
                "element_count": sample["element_count"],
                "source_byte_width": sample["source_byte_width"],
                "original_numeric_payload": sample.get("case_class") == "large_real",
                "recipe": _recipe(row, sample),
                "warmups": 1,
                "repeats": repeats,
            }
            case = {
                "case_id": case_id,
                "implementation_id": row["implementation_id"],
                "batch_id": row["batch_id"],
                "sample_id": sample_id,
                "sample_sha256": sample["sample_sha256"],
                "suite": SUITE_ID,
                "case_scope": case_scope,
                "case_class": sample["case_class"],
                "category": sample["case_class"],
                "boundary_family": sample.get("boundary_family"),
                "expected_outcome": (
                    "domain_rejected" if expected_rejection else "roundtrip"
                ),
                "planned_path": planned_path,
                "expected_execution_path": planned_path,
                "domain_probe": planned_path == "wrapper_allowed" and _domain_probe(sample),
                "request": request,
                "provenance": deepcopy(sample),
                "focal_step_index": row["focal_step_index"],
                "performance_scope": (
                    "whole registered operator tree including prefixes, NGF1 and source schema; "
                    "shared-host SCHED_IDLE functional timing, not performance ranking"
                ),
            }
            if expected_rejection:
                case.update(expected_rejection)
            cases_by_batch[row["batch_id"]].append(case)
    all_cases = [case for batch in catalog["batch_order"] for case in cases_by_batch[batch]]
    _require(len({case["implementation_id"] for case in all_cases}) == 97,
             "stress plan does not cover 97 eligible operators")
    _require(len(all_cases) <= MAX_CASES, f"stress plan exceeds {MAX_CASES} cases")
    _require(not set(EXCLUDED_IDS) & {case["implementation_id"] for case in all_cases},
             "stress plan includes excluded operator")
    _require(all(case["category"] == case["case_class"] for case in all_cases),
             "case category/class mismatch")
    _require(all(case["provenance"]["sample_sha256"] == case["sample_sha256"]
                 for case in all_cases),
             "case/provenance sample digest mismatch")
    planned_rejections = {
        (case["implementation_id"], case["sample_id"])
        for case in all_cases if case["expected_outcome"] == "domain_rejected"
    }
    _require(planned_rejections == set(EXPECTED_DOMAIN_REJECTIONS),
             "planned domain-rejection cases differ from the native preflight contract")
    _require(all(case["expected_outcome"] == "roundtrip" or
                 (case["implementation_id"], case["sample_id"])
                 in EXPECTED_DOMAIN_REJECTIONS
                 for case in all_cases),
             "unexpected non-roundtrip planning outcome")
    output.mkdir(parents=True, exist_ok=True)
    batch_files: list[str] = []
    for batch in catalog["batch_order"]:
        plan = {
            "schema": PLAN_SCHEMA,
            "suite": SUITE_ID,
            "case_scope": case_scope,
            "terminology": "operator_tree",
            "batch_id": batch,
            "catalog_file_sha256": catalog_hash,
            "inputs_file_sha256": _sha(inputs_raw),
            "excluded_ids": list(EXCLUDED_IDS),
            "cases": cases_by_batch[batch],
        }
        target = output / f"{batch}.json"
        _write_immutable_json(target, plan)
        batch_files.append(target.name)
    by_class: dict[str, int] = {}
    for case in all_cases:
        by_class[case["case_class"]] = by_class.get(case["case_class"], 0) + 1
    index = {
        "schema": "operator-tree-stress-plan-index-v3",
        "suite": SUITE_ID,
        "case_scope": case_scope,
        "terminology": "operator_tree",
        "catalog_path": str(catalog_path),
        "catalog_file_sha256": catalog_hash,
        "inputs_path": str(inputs_path),
        "inputs_file_sha256": _sha(inputs_raw),
        "operator_count": 97,
        "case_count": len(all_cases),
        "case_class_counts": by_class,
        "planned_path_counts": {
            path: sum(case["planned_path"] == path for case in all_cases)
            for path in ("strict_required", "wrapper_allowed")
        },
        "expected_outcome_counts": {
            outcome: sum(case["expected_outcome"] == outcome for case in all_cases)
            for outcome in ("roundtrip", "domain_rejected")
        },
        "excluded_ids": list(EXCLUDED_IDS),
        "batch_order": list(catalog["batch_order"]),
        "batch_files": batch_files,
        "case_ids": [case["case_id"] for case in all_cases],
    }
    _write_immutable_json(output / "INDEX.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("input-spec", help="derive C++ builder spec from accepted manifests")
    spec.add_argument("--source-manifests", required=True, type=Path)
    spec.add_argument("--output", required=True, type=Path)
    spec.add_argument("--boundary-only", action="store_true")
    plan = commands.add_parser("plan", help="build C++ batch-runner plans")
    plan.add_argument("--catalog", required=True, type=Path)
    plan.add_argument("--inputs", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--runtime-root", required=True)
    plan.add_argument("--repeats", type=int, default=5)
    plan.add_argument("--case-scope", choices=("all", "boundary-only"), default="all")
    args = parser.parse_args()
    if args.command == "input-spec":
        result = build_input_spec(args.source_manifests, args.output, args.boundary_only)
        summary = {
            "schema": result["schema"],
            "real_slice_count": len(result["real_slices"]),
            "boundary_fixture_count": len(result["boundary_fixtures"]),
            "real_size_coverage": result["real_size_coverage"],
        }
    else:
        result = prepare_plans(args.catalog, args.inputs, args.output,
                               args.runtime_root, args.repeats, args.case_scope)
        summary = result
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
