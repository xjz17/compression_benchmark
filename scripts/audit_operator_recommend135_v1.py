#!/usr/bin/env python3
"""Independent, metadata-only audit for an operator-recommend135-v1 run.

The run root is treated as read-only.  This program writes only
``RUN_ROOT/audit/AUDIT.json`` and ``RUN_ROOT/audit/FILES_SHA256.json``.
It never executes a codec, model, or benchmark kernel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, cast


SCHEMA = "operator-recommend135-independent-audit-v1"
FILES_SCHEMA = "operator-recommend135-files-sha256-v1"
TARGETS = ("space", "encode", "decode", "balanced")
TOP_KS = (1, 3, 5)
TIE_TOLERANCE = 1e-12
DISTANCE_FLOOR = 1e-12
EXPECTED_CANDIDATES = 24
EXPECTED_DATASETS = 135
EXPECTED_SAMPLES = 360
EXPECTED_GAPS = 225
EXPECTED_GROUPS = 123
FORBIDDEN_COMPONENTS = {f"A{i}" for i in range(106, 111)}
FORBIDDEN_ACTION_IDS = set(range(106, 111))
FORBIDDEN_ACTION_TOKENS = ("nncp", "lstm", "ppmd", ".ppm.", ".dmc.",
                           ".paq.", "paq8", ".cmix.")
TIMING_KEYS = ("encode_cpu_ns", "decode_cpu_ns", "encode_wall_ns",
               "decode_wall_ns")
STRICT_METRIC_KEYS = ("frame_bytes",) + TIMING_KEYS


class AuditProblem(RuntimeError):
    """A structural error which prevents further checks in one section."""


class Auditor:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.findings: list[dict[str, Any]] = []
        self.checks = 0
        self.suppressed_findings = 0

    def check(self, condition: bool, code: str, message: str,
              **evidence: Any) -> bool:
        self.checks += 1
        if condition:
            return True
        if len(self.findings) < 2000:
            finding: dict[str, Any] = {"code": code, "message": message}
            if evidence:
                finding["evidence"] = evidence
            self.findings.append(finding)
        else:
            self.suppressed_findings += 1
        return False

    def require(self, condition: bool, code: str, message: str,
                **evidence: Any) -> None:
        if not self.check(condition, code, message, **evidence):
            raise AuditProblem(message)

    def section_error(self, section: str, error: BaseException) -> None:
        self.check(False, f"{section}_audit_exception",
                   f"{section} audit could not complete: {error}",
                   exception_type=type(error).__name__)


class HashCache:
    def __init__(self) -> None:
        self._cache: dict[Path, tuple[int, int, str]] = {}

    def sha256(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        cached = self._cache.get(resolved)
        signature = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == signature:
            return cached[2]
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._cache[resolved] = (signature[0], signature[1], value)
        return value


def read_json(path: Path) -> Any:
    with path.open("rb") as source:
        return json.load(source)


def json_bytes(value: Any) -> bytes:
    # nlohmann::json uses an ordered object map and compact separators for dump().
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_number(value: Any, context: str) -> float:
    if not is_number(value) or not math.isfinite(float(value)):
        raise AuditProblem(f"{context} must be a finite number")
    return float(value)


def string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value):
        raise AuditProblem(f"{context} must be an array of non-empty strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def same_float(left: Any, right: Any) -> bool:
    if not is_number(left) or not is_number(right):
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def tied(left: float, right: float) -> bool:
    return abs(left - right) <= TIE_TOLERANCE * max(1.0, abs(left), abs(right))


def check_float(audit: Auditor, actual: Any, expected: float, code: str,
                context: str) -> None:
    audit.check(same_float(actual, expected), code,
                f"{context} differs from independent recomputation",
                actual=actual, expected=expected)


def resolve_artifact(audit: Auditor, value: Any, context: str,
                     relative_to: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditProblem(f"{context} is not a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditProblem(f"{context} does not exist: {path}") from error
    audit.check(resolved.is_relative_to(audit.run_root),
                "artifact_outside_run_root",
                f"{context} resolves outside the new run root",
                path=str(resolved), run_root=str(audit.run_root))
    return resolved


def find_named(run_root: Path, name: str) -> list[Path]:
    result: list[Path] = []
    for directory, directories, files in os.walk(run_root, followlinks=False):
        here = Path(directory)
        directories[:] = [item for item in directories
                          if item != "audit" and not (here / item).is_symlink()]
        if name in files:
            result.append((here / name).resolve())
    return sorted(result)


def select_named(audit: Auditor, name: str, explicit: Path | None,
                 required: bool = True) -> Path | None:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else audit.run_root / explicit
        if path.is_dir():
            path /= name
        audit.require(path.is_file(), "explicit_artifact_missing",
                      f"explicit {name} path is not a file", path=str(path))
        return path.resolve()
    direct = audit.run_root / name
    if direct.is_file():
        return direct.resolve()
    matches = find_named(audit.run_root, name)
    if len(matches) == 1:
        return matches[0]
    if required:
        audit.require(False, "artifact_discovery_failed",
                      f"expected exactly one {name} below run root",
                      matches=[str(item) for item in matches])
    return None


def atomic_json(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True,
                      ensure_ascii=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def upper_median(values: Iterable[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise AuditProblem("cannot compute a median from no timing samples")
    return ordered[len(ordered) // 2]


def audit_inputs(audit: Auditor, hashes: HashCache,
                 inputs_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    inputs = read_json(inputs_path)
    audit.check(inputs.get("schema") == "operator-recommend135-inputs-v1",
                "inputs_schema", "INPUTS.json has the wrong schema",
                actual=inputs.get("schema"))
    samples = inputs.get("samples")
    if not isinstance(samples, list):
        raise AuditProblem("INPUTS.json samples is not an array")
    by_id: dict[str, dict[str, Any]] = {}
    datasets: set[str] = set()
    files: set[Path] = set()
    bytes_hashed = 0
    dtype_width = {"int64": 8, "float64": 8, "float32": 4}
    for index, sample in enumerate(samples):
        context = f"samples[{index}]"
        if not isinstance(sample, dict):
            audit.check(False, "sample_not_object", f"{context} is not an object")
            continue
        sample_id = sample.get("sample_id")
        dataset = sample.get("dataset_id")
        dtype = sample.get("dtype")
        audit.check(isinstance(sample_id, str) and bool(sample_id),
                    "sample_id_invalid", f"{context} has no valid sample_id")
        audit.check(isinstance(dataset, str) and bool(dataset),
                    "dataset_id_invalid", f"{context} has no valid dataset_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        audit.check(sample_id not in by_id, "duplicate_sample_id",
                    "INPUTS.json contains a duplicate sample_id",
                    sample_id=sample_id)
        by_id[sample_id] = sample
        if isinstance(dataset, str) and dataset:
            datasets.add(dataset)
        width = dtype_width.get(dtype) if isinstance(dtype, str) else None
        audit.check(width is not None, "input_dtype_out_of_scope",
                    "sample dtype is outside the frozen three-dtype scope",
                    sample_id=sample_id, dtype=dtype)
        element_count = sample.get("element_count")
        for part_name in ("head", "tail"):
            part = sample.get(part_name)
            if not isinstance(part, dict):
                audit.check(False, "input_part_missing",
                            f"{context}.{part_name} is not an object")
                continue
            for payload_key, sha_key in (("values_path", "values_sha256"),
                                         ("missing_path", "missing_sha256")):
                try:
                    path = resolve_artifact(audit, part.get(payload_key),
                                            f"{sample_id}.{part_name}.{payload_key}",
                                            inputs_path.parent)
                    files.add(path)
                    actual_sha = hashes.sha256(path)
                    bytes_hashed += path.stat().st_size
                    audit.check(actual_sha == part.get(sha_key),
                                "input_payload_sha_mismatch",
                                "sample payload SHA differs from INPUTS.json",
                                sample_id=sample_id, part=part_name,
                                payload=payload_key, expected=part.get(sha_key),
                                actual=actual_sha, path=str(path))
                except (AuditProblem, OSError) as error:
                    audit.check(False, "input_payload_unreadable", str(error),
                                sample_id=sample_id, part=part_name,
                                payload=payload_key)
            if width is not None:
                try:
                    values_path = resolve_artifact(
                        audit, part.get("values_path"),
                        f"{sample_id}.{part_name}.values_path", inputs_path.parent)
                    missing_path = resolve_artifact(
                        audit, part.get("missing_path"),
                        f"{sample_id}.{part_name}.missing_path", inputs_path.parent)
                    valid_count = part.get("valid_count")
                    row_count = part.get("row_count")
                    audit.check(isinstance(valid_count, int) and valid_count >= 0,
                                "valid_count_invalid", "part valid_count is invalid",
                                sample_id=sample_id, part=part_name,
                                value=valid_count)
                    audit.check(isinstance(row_count, int) and row_count >= 0,
                                "row_count_invalid", "part row_count is invalid",
                                sample_id=sample_id, part=part_name,
                                value=row_count)
                    if isinstance(valid_count, int) and valid_count >= 0:
                        audit.check(values_path.stat().st_size == valid_count * width,
                                    "input_values_size_mismatch",
                                    "values byte length differs from valid_count*dtype width",
                                    sample_id=sample_id, part=part_name)
                        audit.check(valid_count == element_count,
                                    "input_element_count_mismatch",
                                    "part valid_count differs from sample element_count",
                                    sample_id=sample_id, part=part_name,
                                    valid_count=valid_count,
                                    element_count=element_count)
                    if isinstance(row_count, int) and row_count >= 0:
                        audit.check(missing_path.stat().st_size ==
                                    row_count // 8 + (row_count % 8 != 0),
                                    "input_mask_size_mismatch",
                                    "missing byte length differs from row_count",
                                    sample_id=sample_id, part=part_name)
                except (AuditProblem, OSError) as error:
                    audit.check(False, "input_shape_unreadable", str(error),
                                sample_id=sample_id, part=part_name)
        head = sample.get("head", {})
        tail = sample.get("tail", {})
        if isinstance(head, dict) and isinstance(tail, dict):
            head_end = head.get("row_end_exclusive")
            tail_start = tail.get("row_start")
            if isinstance(head_end, int) and isinstance(tail_start, int):
                audit.check(head_end <= tail_start, "input_windows_overlap",
                            "head and tail row ranges overlap",
                            sample_id=sample_id, head_end=head_end,
                            tail_start=tail_start)
    audit.check(len(datasets) == EXPECTED_DATASETS, "dataset_coverage",
                "INPUTS.json does not cover exactly 135 datasets",
                actual=len(datasets), expected=EXPECTED_DATASETS)
    audit.check(len(samples) == EXPECTED_SAMPLES, "input_sample_count",
                "INPUTS.json does not contain the frozen 360 samples",
                actual=len(samples), expected=EXPECTED_SAMPLES)
    declared_datasets = inputs.get("dataset_ids")
    audit.check(declared_datasets == sorted(datasets), "input_dataset_ids",
                "INPUTS dataset_ids differ from the observed sample datasets")
    gaps = inputs.get("gaps")
    gap_reason_counts: Counter[str] = Counter()
    gap_pairs: set[tuple[str, str]] = set()
    if not isinstance(gaps, list):
        audit.check(False, "input_gaps", "INPUTS gaps is not an array")
        gaps = []
    for index, gap in enumerate(gaps):
        if not isinstance(gap, dict):
            audit.check(False, "input_gap_entry", "INPUTS gap is not an object",
                        index=index)
            continue
        raw_pair = (gap.get("dataset_id"), gap.get("dtype"))
        valid_pair = all(isinstance(item, str) and item for item in raw_pair)
        audit.check(valid_pair,
                    "input_gap_identity", "INPUTS gap has an invalid identity",
                    index=index, pair=raw_pair)
        if not valid_pair:
            continue
        pair = cast(tuple[str, str], raw_pair)
        audit.check(pair not in gap_pairs, "input_gap_duplicate",
                    "INPUTS contains a duplicate dataset/dtype gap", pair=pair)
        gap_pairs.add(pair)
        reason = gap.get("reason")
        gap_reason_counts[str(reason)] += 1
        audit.check(reason == "dtype_absent", "input_non_absent_dtype_gap",
                    "frozen inputs contain a gap for a present but too-short dtype",
                    index=index, pair=pair, reason=reason)
        audit.check(gap.get("longest_same_entity_valid_count") == 0,
                    "input_absent_dtype_count",
                    "dtype_absent gap has a nonzero longest sequence", index=index)
    audit.check(len(gaps) == EXPECTED_GAPS, "input_gap_count",
                "INPUTS does not contain the frozen 225 dtype-absent gaps",
                actual=len(gaps), expected=EXPECTED_GAPS)
    sequence_scales: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        if isinstance(sample, dict):
            dataset = sample.get("dataset_id")
            dtype = sample.get("dtype")
            scale = sample.get("scale")
            if all(isinstance(item, str) for item in (dataset, dtype, scale)):
                sequence_scales[(cast(str, dataset), cast(str, dtype))].add(
                    cast(str, scale))
            else:
                audit.check(False, "input_sequence_identity",
                            "sample dataset/dtype/scale is invalid")
    audit.check(all(scales == {"primary", "u64"}
                    for scales in sequence_scales.values()),
                "input_scale_pairs",
                "each selected dataset/dtype sequence must have primary and u64 samples",
                bad_pairs=[list(pair) for pair, scales in sequence_scales.items()
                           if scales != {"primary", "u64"}][:20])
    audit.check(set(sequence_scales).isdisjoint(gap_pairs),
                "input_gap_sample_overlap",
                "a dataset/dtype is both sampled and declared as a gap")
    expected_pairs = {(dataset, dtype) for dataset in datasets
                      for dtype in dtype_width}
    audit.check(set(sequence_scales) | gap_pairs == expected_pairs,
                "input_dataset_dtype_partition",
                "samples and gaps do not partition all 135x3 dataset/dtype pairs")
    coverage = inputs.get("coverage")
    if not isinstance(coverage, dict):
        audit.check(False, "input_coverage", "INPUTS coverage is not an object")
        coverage = {}
    audit.check(coverage.get("plan_dataset_count") == EXPECTED_DATASETS and
                coverage.get("covered_dataset_count") == EXPECTED_DATASETS and
                coverage.get("missing_dataset_ids") == [] and
                coverage.get("sample_count") == len(samples) and
                coverage.get("gap_count") == len(gaps),
                "input_coverage_receipt",
                "INPUTS coverage receipt differs from independently counted content",
                actual=coverage)
    manifest_view = dict(inputs)
    declared_manifest_sha = manifest_view.pop("manifest_sha256", None)
    audit.check(declared_manifest_sha == canonical_json_sha256(manifest_view),
                "inputs_manifest_sha",
                "INPUTS manifest_sha256 differs from its canonical pre-hash object",
                actual=declared_manifest_sha,
                expected=canonical_json_sha256(manifest_view))
    return ({"path": str(inputs_path),
             "sha256": hashes.sha256(inputs_path),
             "sample_count": len(samples),
             "unique_sample_ids": len(by_id),
             "dataset_count": len(datasets),
             "payload_file_count": len(files),
             "payload_bytes_hashed": bytes_hashed,
             "gap_count": len(gaps),
             "gap_reason_counts": dict(sorted(gap_reason_counts.items())),
             "selected_dataset_dtype_count": len(sequence_scales),
             "datasets": sorted(datasets)}, by_id)


def nested_actions(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "action" and isinstance(item, str):
                yield item
            yield from nested_actions(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_actions(item)


def nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_strings(item)
    elif isinstance(value, str):
        yield value


def audit_catalog(audit: Auditor, hashes: HashCache,
                  catalog_path: Path) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    catalog = read_json(catalog_path)
    audit.check(catalog.get("schema") ==
                "operator-recommend135-candidate-catalog-v1",
                "catalog_schema", "CANDIDATES.json has the wrong schema")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list):
        raise AuditProblem("catalog candidates is not an array")
    ids: list[str] = []
    kinds: dict[str, str] = {}
    forbidden_components: list[dict[str, Any]] = []
    forbidden_action_ids: list[dict[str, Any]] = []
    forbidden_actions: list[dict[str, Any]] = []
    ordinals: list[int] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            audit.check(False, "catalog_candidate_not_object",
                        "catalog contains a non-object candidate")
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            audit.check(False, "catalog_candidate_id_invalid",
                        "catalog candidate_id is invalid")
            continue
        ids.append(candidate_id)
        kind = candidate.get("kind")
        audit.check(isinstance(kind, str), "catalog_candidate_kind",
                    "catalog candidate kind is invalid", candidate_id=candidate_id)
        kinds[candidate_id] = kind if isinstance(kind, str) else ""
        ordinal = candidate.get("candidate_ordinal")
        if isinstance(ordinal, int):
            ordinals.append(ordinal)
        for component in candidate.get("component_ids", []):
            lower = str(component).lower()
            if component in FORBIDDEN_COMPONENTS or any(
                    token.strip(".") in lower for token in
                    ("nncp", "lstm", "ppmd", "dmc", "paq8", "cmix")):
                forbidden_components.append(
                    {"candidate_id": candidate_id, "component": component})
        for action_id in candidate.get("action_ids", []):
            if action_id in FORBIDDEN_ACTION_IDS:
                forbidden_action_ids.append(
                    {"candidate_id": candidate_id, "action_id": action_id})
        for action in nested_actions(candidate.get("recipe")):
            lower = action.lower()
            if any(token in lower for token in FORBIDDEN_ACTION_TOKENS):
                forbidden_actions.append(
                    {"candidate_id": candidate_id, "action": action})
        for text in nested_strings(candidate):
            lower = text.lower()
            if (any(token in lower for token in FORBIDDEN_ACTION_TOKENS) or
                    any(component.lower() in lower
                        for component in FORBIDDEN_COMPONENTS)):
                hit = {"candidate_id": candidate_id, "text": text}
                if hit not in forbidden_actions:
                    forbidden_actions.append(hit)
        audit.check(candidate.get("allows_harness_raw_fallback") is False,
                    "catalog_raw_fallback_allowed",
                    "candidate does not fail closed on raw fallback",
                    candidate_id=candidate_id)
    audit.check(len(ids) == EXPECTED_CANDIDATES, "candidate_count",
                "catalog does not contain exactly 24 candidates",
                actual=len(ids), expected=EXPECTED_CANDIDATES)
    audit.check(len(ids) == len(set(ids)), "duplicate_candidate_id",
                "catalog contains duplicate candidate IDs")
    audit.check(sorted(ordinals) == list(range(EXPECTED_CANDIDATES)),
                "candidate_ordinals", "catalog ordinals are not exactly 0..23",
                actual=sorted(ordinals))
    audit.check(not forbidden_components, "forbidden_component",
                "catalog contains forbidden components",
                hits=forbidden_components)
    audit.check(not forbidden_action_ids, "forbidden_action_id",
                "catalog contains forbidden action IDs",
                hits=forbidden_action_ids)
    audit.check(not forbidden_actions, "forbidden_action",
                "catalog contains forbidden action families",
                hits=forbidden_actions)
    required_kinds = {"raw", "algorithm", "operator", "operator_tree",
                      "algorithm_combination"}
    audit.check(required_kinds.issubset(set(kinds.values())),
                "candidate_kind_coverage",
                "catalog lacks a required candidate kind",
                required=sorted(required_kinds), actual=sorted(set(kinds.values())))
    canonical_sha = canonical_json_sha256(catalog)
    return ({"path": str(catalog_path),
             "raw_sha256": hashes.sha256(catalog_path),
             "canonical_sha256": canonical_sha,
             "candidate_count": len(ids),
             "kind_counts": dict(sorted(Counter(kinds.values()).items())),
             "forbidden_component_hits": forbidden_components,
             "forbidden_action_id_hits": forbidden_action_ids,
             "forbidden_action_hits": forbidden_actions}, ids, kinds)


def strict_cost(row: dict[str, Any], candidate: str) -> dict[str, float] | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or candidate not in metrics:
        return None
    metric = metrics[candidate]
    if not isinstance(metric, dict) or metric.get("status") != "strict_verified":
        return None
    input_bytes = finite_number(row.get("input_bytes"), "row.input_bytes")
    if input_bytes <= 0:
        raise AuditProblem("row.input_bytes must be positive")
    frame = finite_number(metric.get("frame_bytes"), "metric.frame_bytes")
    enc = finite_number(metric.get("encode_cpu_ns"), "metric.encode_cpu_ns")
    dec = finite_number(metric.get("decode_cpu_ns"), "metric.decode_cpu_ns")
    enc_wall = finite_number(metric.get("encode_wall_ns"),
                             "metric.encode_wall_ns")
    dec_wall = finite_number(metric.get("decode_wall_ns"),
                             "metric.decode_wall_ns")
    space = frame / input_bytes
    encode = max(enc, 1.0) / input_bytes
    decode = max(dec, 1.0) / input_bytes
    encode_wall = max(enc_wall, 1.0) / input_bytes
    decode_wall = max(dec_wall, 1.0) / input_bytes
    return {"space": space, "encode": encode, "decode": decode,
            "balanced": math.exp((math.log(space) + math.log(encode) +
                                  math.log(decode)) / 3.0),
            "encode_wall": encode_wall, "decode_wall": decode_wall,
            "balanced_wall": math.exp((math.log(space) +
                                       math.log(encode_wall) +
                                       math.log(decode_wall)) / 3.0)}


def target_cost(cost: dict[str, float], target: str, wall: bool = False) -> float:
    if not wall or target == "space":
        return cost[target]
    return cost[f"{target}_wall"]


def choose_baseline(training_rows: list[dict[str, Any]], candidates: list[str],
                    target: str) -> dict[str, Any] | None:
    best: tuple[float, str] | None = None
    for candidate in candidates:
        costs: list[float] = []
        for row in training_rows:
            value = strict_cost(row, candidate)
            if value is None:
                costs = []
                break
            costs.append(target_cost(value, target))
        if len(costs) != len(training_rows):
            continue
        mean = sum(math.log(item) for item in costs) / len(costs)
        if (best is None or mean < best[0] - TIE_TOLERANCE or
                (abs(mean - best[0]) <= TIE_TOLERANCE and candidate < best[1])):
            best = (mean, candidate)
    if best is None:
        return None
    return {"candidate_id": best[1], "training_rows": len(training_rows),
            "training_mean_log_cost": best[0]}


def labels_for_row(row: dict[str, Any], candidates: list[str]) -> dict[str, dict[str, float]]:
    eligible = set(string_list(row.get("eligible_candidates"),
                               "row.eligible_candidates"))
    costs = {candidate: strict_cost(row, candidate) for candidate in candidates
             if candidate in eligible}
    strict_cost_map: dict[str, dict[str, float]] = {
        candidate: value for candidate, value in costs.items()
        if value is not None}
    if not strict_cost_map:
        return {}
    minima = {target: min(target_cost(value, target)
                          for value in strict_cost_map.values())
              for target in TARGETS}
    return {candidate: {target: math.log(target_cost(value, target) /
                                         minima[target])
                        for target in TARGETS}
            for candidate, value in strict_cost_map.items()}


def normalization(training_rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    vectors = [[finite_number(item, "row.features") for item in row["features"]]
               for row in training_rows]
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise AuditProblem("training feature vectors have inconsistent lengths")
    means = [sum(vector[i] for vector in vectors) / len(vectors)
             for i in range(dimensions)]
    scales: list[float] = []
    for i, mean in enumerate(means):
        variance = sum((vector[i] - mean) ** 2 for vector in vectors) / len(vectors)
        scales.append(math.sqrt(variance) if variance > 0 else 1.0)
    return means, scales


def compare_number_array(audit: Auditor, actual: Any, expected: list[float],
                         code: str, context: str) -> None:
    if not isinstance(actual, list):
        audit.check(False, code, f"{context} is not an array")
        return
    audit.check(len(actual) == len(expected), code,
                f"{context} has the wrong length",
                actual=len(actual), expected=len(expected))
    for index, (left, right) in enumerate(zip(actual, expected)):
        check_float(audit, left, right, code, f"{context}[{index}]")


def recompute_prediction(training_rows: list[dict[str, Any]],
                         test_row: dict[str, Any], candidates: list[str],
                         means: list[float], scales: list[float]) -> dict[str, Any]:
    points: list[tuple[str, list[float], dict[str, dict[str, float]]]] = []
    for row in training_rows:
        vector = [finite_number(item, "training features") for item in row["features"]]
        normalized = [(item - means[i]) / scales[i]
                      for i, item in enumerate(vector)]
        points.append((row["sample_id"], normalized, labels_for_row(row, candidates)))
    features = [finite_number(item, "test features") for item in test_row["features"]]
    query = [(item - means[i]) / scales[i] for i, item in enumerate(features)]
    point_distances: list[tuple[float, str, dict[str, dict[str, float]]]] = []
    for sample_id, vector, labels in points:
        distance = math.sqrt(sum((query[i] - vector[i]) ** 2
                                 for i in range(len(query))))
        point_distances.append((distance, sample_id, labels))
    eligible_input = string_list(test_row.get("eligible_candidates"),
                                 "test row eligible_candidates")
    eligible = [item for item in eligible_input if item in set(candidates)]
    unscored: set[str] = {item for item in eligible_input if item not in set(candidates)}
    targets: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        ranking: list[dict[str, Any]] = []
        for candidate in eligible:
            neighbors = [(distance, sample_id, labels[candidate][target])
                         for distance, sample_id, labels in point_distances
                         if candidate in labels and target in labels[candidate]]
            neighbors.sort(key=lambda item: (item[0], item[1]))
            if not neighbors:
                unscored.add(candidate)
                continue
            selected = neighbors[:5]
            exact = [item for item in selected if item[0] <= DISTANCE_FLOOR]
            if exact:
                predicted = sum(item[2] for item in exact) / len(exact)
            else:
                weights = [1.0 / item[0] for item in selected]
                predicted = sum(weight * item[2]
                                for weight, item in zip(weights, selected)) / sum(weights)
            ranking.append({"candidate_id": candidate,
                            "predicted_log_regret": predicted,
                            "neighbor_count": len(selected),
                            "exact_match_count": len(exact)})
        ranking.sort(key=lambda item: (item["predicted_log_regret"],
                                       item["candidate_id"]))
        targets[target] = ranking
    return {"targets": targets, "unscored_candidates": sorted(unscored)}


def pareto_sets(costs: dict[str, dict[str, float]]) -> tuple[set[str], set[str]]:
    dominated: set[str] = set()
    for candidate, current in costs.items():
        for other_id, other in costs.items():
            if candidate == other_id:
                continue
            no_worse = True
            strictly_better = False
            for target in ("space", "encode", "decode"):
                if other[target] > current[target] and not tied(other[target], current[target]):
                    no_worse = False
                    break
                if other[target] < current[target] and not tied(other[target], current[target]):
                    strictly_better = True
            if no_worse and strictly_better:
                dominated.add(candidate)
                break
    return set(costs) - dominated, dominated


def audit_prediction_scores(audit: Auditor, row: dict[str, Any],
                            prediction_record: dict[str, Any],
                            recomputed_prediction: dict[str, Any],
                            baselines: dict[str, Any], context: str) -> None:
    prediction = prediction_record.get("prediction")
    scores = prediction_record.get("scores")
    if not isinstance(prediction, dict) or not isinstance(scores, dict):
        raise AuditProblem(f"{context} lacks prediction/scores objects")
    audit.check(prediction.get("eligible_candidates") ==
                row.get("eligible_candidates"), "prediction_eligibility",
                f"{context} eligible candidates differ from the row")
    audit.check(prediction.get("unscored_candidates") ==
                recomputed_prediction["unscored_candidates"],
                "prediction_unscored", f"{context} unscored set differs")
    eligible = string_list(row.get("eligible_candidates"),
                           f"{context}.eligible_candidates")
    raw_true_costs = {candidate: strict_cost(row, candidate)
                      for candidate in eligible}
    true_costs: dict[str, dict[str, float]] = {
        candidate: value for candidate, value in raw_true_costs.items()
        if value is not None}
    pareto, dominated = pareto_sets(true_costs)
    actual_pareto = prediction_record.get("pareto")
    audit.check(isinstance(actual_pareto, dict) and
                set(actual_pareto.get("candidates", [])) == pareto and
                set(actual_pareto.get("dominated_candidates", [])) == dominated,
                "prediction_pareto_truth",
                f"{context} Pareto truth differs from independent recomputation")
    for target in TARGETS:
        target_prediction = prediction.get("targets", {}).get(target, {})
        ranking = target_prediction.get("ranking")
        expected_ranking = recomputed_prediction["targets"][target]
        if not isinstance(ranking, list):
            audit.check(False, "prediction_ranking_missing",
                        f"{context}.{target} ranking is missing")
            continue
        audit.check(len(ranking) == len(expected_ranking),
                    "prediction_ranking_length",
                    f"{context}.{target} ranking length differs from training-only recomputation",
                    actual=len(ranking), expected=len(expected_ranking))
        for index, (actual, expected) in enumerate(zip(ranking, expected_ranking)):
            audit.check(actual.get("candidate_id") == expected["candidate_id"],
                        "prediction_ranking_candidate",
                        f"{context}.{target} ranking candidate differs",
                        index=index, actual=actual.get("candidate_id"),
                        expected=expected["candidate_id"])
            check_float(audit, actual.get("predicted_log_regret"),
                        expected["predicted_log_regret"],
                        "prediction_ranking_score",
                        f"{context}.{target}.ranking[{index}]")
            audit.check(actual.get("neighbor_count") == expected["neighbor_count"] and
                        actual.get("exact_match_count") == expected["exact_match_count"],
                        "prediction_neighbor_counts",
                        f"{context}.{target} neighbor counts differ", index=index)
        ranking_ids = [item.get("candidate_id") for item in ranking]
        for k in TOP_KS:
            expected_top = ranking_ids[:min(k, len(ranking_ids))]
            actual_top = target_prediction.get(f"top{k}")
            audit.check(actual_top == expected_top, "prediction_topk_prefix",
                        f"{context}.{target}.top{k} is not the ranking prefix",
                        actual=actual_top, expected=expected_top)
        score = scores.get(target)
        if not isinstance(score, dict):
            audit.check(False, "score_missing", f"{context}.{target} score is missing")
            continue
        if not true_costs:
            audit.check(score.get("oracle") is None, "unexpected_oracle",
                        f"{context}.{target} has an oracle without strict costs")
            continue
        oracle = min(target_cost(value, target) for value in true_costs.values())
        wall_oracle = min(target_cost(value, target, True)
                          for value in true_costs.values())
        oracle_ids = {candidate for candidate, value in true_costs.items()
                      if tied(target_cost(value, target), oracle)}
        wall_oracle_ids = {candidate for candidate, value in true_costs.items()
                           if tied(target_cost(value, target, True), wall_oracle)}
        actual_oracle = score.get("oracle")
        actual_wall_oracle = score.get("wall_oracle")
        if not isinstance(actual_oracle, dict) or not isinstance(actual_wall_oracle, dict):
            audit.check(False, "oracle_missing", f"{context}.{target} oracle is missing")
            continue
        check_float(audit, actual_oracle.get("cost"), oracle,
                    "oracle_cost", f"{context}.{target}.oracle")
        check_float(audit, actual_wall_oracle.get("cost"), wall_oracle,
                    "wall_oracle_cost", f"{context}.{target}.wall_oracle")
        audit.check(set(actual_oracle.get("candidates", [])) == oracle_ids,
                    "oracle_candidates", f"{context}.{target} oracle candidates differ")
        audit.check(set(actual_wall_oracle.get("candidates", [])) == wall_oracle_ids,
                    "wall_oracle_candidates",
                    f"{context}.{target} wall oracle candidates differ")
        topk_scores = score.get("topk", {})
        for k in TOP_KS:
            predicted = target_prediction.get(f"top{k}", [])
            top = topk_scores.get(str(k))
            if not isinstance(top, dict):
                audit.check(False, "topk_score_missing",
                            f"{context}.{target}.top{k} score is missing")
                continue
            available = [true_costs[item] for item in predicted if item in true_costs]
            expected_regret = (min(target_cost(item, target) for item in available) /
                               oracle) if available else None
            expected_wall_regret = (min(target_cost(item, target, True)
                                        for item in available) /
                                    wall_oracle) if available else None
            audit.check(top.get("candidates") == predicted,
                        "topk_score_candidates",
                        f"{context}.{target}.top{k} candidate set differs")
            audit.check(top.get("tie_aware_hit") ==
                        bool(set(predicted) & oracle_ids), "topk_tie_hit",
                        f"{context}.{target}.top{k} tie-aware hit differs")
            audit.check(top.get("wall_tie_aware_hit") ==
                        bool(set(predicted) & wall_oracle_ids),
                        "topk_wall_tie_hit",
                        f"{context}.{target}.top{k} wall tie-aware hit differs")
            if expected_regret is None:
                audit.check(top.get("oracle_best_within_topk_multiplicative_regret")
                            is None, "topk_regret",
                            f"{context}.{target}.top{k} regret should be null")
            else:
                check_float(audit,
                            top.get("oracle_best_within_topk_multiplicative_regret"),
                            expected_regret, "topk_regret",
                            f"{context}.{target}.top{k} regret")
            if expected_wall_regret is None:
                audit.check(top.get(
                    "wall_oracle_best_within_topk_multiplicative_regret") is None,
                    "topk_wall_regret",
                    f"{context}.{target}.top{k} wall regret should be null")
            else:
                check_float(audit, top.get(
                    "wall_oracle_best_within_topk_multiplicative_regret"),
                    expected_wall_regret, "topk_wall_regret",
                    f"{context}.{target}.top{k} wall regret")
            expected_dominated = sum(item in dominated for item in predicted)
            expected_pareto = (sum(item in pareto for item in predicted) / len(pareto)
                               if pareto else 0.0)
            audit.check(top.get("dominated_count") == expected_dominated,
                        "topk_dominated", f"{context}.{target}.top{k} dominated count differs")
            check_float(audit, top.get("pareto_recall"), expected_pareto,
                        "topk_pareto_recall",
                        f"{context}.{target}.top{k} Pareto recall")
            expected_semantics = ("online_top1_selection_no_probe" if k == 1 else
                                  "offline_set_quality_no_online_probe")
            audit.check(top.get("semantics") == expected_semantics,
                        "topk_semantics",
                        f"{context}.{target}.top{k} semantics differs")
            if k == 1:
                audit.check(top.get("online_selected_multiplicative_regret") ==
                            top.get("oracle_best_within_topk_multiplicative_regret") and
                            top.get("wall_online_selected_multiplicative_regret") ==
                            top.get("wall_oracle_best_within_topk_multiplicative_regret"),
                            "top1_online_regret_alias",
                            f"{context}.{target}.top1 online regret alias differs")
        for baseline_name in ("fixed_algorithm", "fixed_candidate"):
            expected_choice = baselines[target][baseline_name]
            actual_baseline = score.get(f"{baseline_name}_baseline")
            if expected_choice is None:
                audit.check(actual_baseline is None, "baseline_score_unexpected",
                            f"{context}.{target} has an unexpected {baseline_name} score")
            elif isinstance(actual_baseline, dict):
                audit.check(actual_baseline.get("candidate_id") ==
                            expected_choice["candidate_id"],
                            "baseline_score_candidate",
                            f"{context}.{target} {baseline_name} candidate differs")
                candidate = expected_choice["candidate_id"]
                cost = true_costs.get(candidate)
                expected_available = cost is not None
                audit.check(actual_baseline.get("available") is expected_available,
                            "baseline_score_available",
                            f"{context}.{target} {baseline_name} availability differs")
                if cost is not None:
                    expected_cost = target_cost(cost, target)
                    expected_wall_cost = target_cost(cost, target, True)
                    check_float(audit, actual_baseline.get("cost"), expected_cost,
                                "baseline_score_cost",
                                f"{context}.{target}.{baseline_name}.cost")
                    check_float(audit,
                                actual_baseline.get("multiplicative_regret"),
                                expected_cost / oracle, "baseline_score_regret",
                                f"{context}.{target}.{baseline_name}.regret")
                    check_float(audit, actual_baseline.get("wall_cost"),
                                expected_wall_cost, "baseline_score_wall_cost",
                                f"{context}.{target}.{baseline_name}.wall_cost")
                    check_float(audit,
                                actual_baseline.get("wall_multiplicative_regret"),
                                expected_wall_cost / wall_oracle,
                                "baseline_score_wall_regret",
                                f"{context}.{target}.{baseline_name}.wall_regret")
            else:
                audit.check(False, "baseline_score_missing",
                            f"{context}.{target} {baseline_name} score is missing")


def audit_groups(audit: Auditor, hashes: HashCache,
                 samples: dict[str, dict[str, Any]],
                 bindings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    groups_path = select_named(audit, "GROUPS.json", None)
    if groups_path is None:
        raise AuditProblem("GROUPS.json was not found")
    groups = read_json(groups_path)
    mapping = groups.get("dataset_to_group")
    if not isinstance(mapping, dict) or any(
            not isinstance(dataset, str) or not isinstance(group, str)
            for dataset, group in mapping.items()):
        raise AuditProblem("GROUPS dataset_to_group is not a string map")
    datasets = {sample.get("dataset_id") for sample in samples.values()}
    audit.check(set(mapping) == datasets, "groups_dataset_coverage",
                "GROUPS dataset map differs from INPUTS datasets",
                missing=sorted(datasets - set(mapping)),
                extra=sorted(set(mapping) - datasets))
    group_ids = set(mapping.values())
    audit.check(len(group_ids) == EXPECTED_GROUPS, "groups_count",
                "GROUPS does not contain the frozen 123 source groups",
                actual=len(group_ids), expected=EXPECTED_GROUPS)
    audit.check(groups.get("group_count") == len(group_ids),
                "groups_declared_count",
                "GROUPS group_count differs from dataset_to_group")
    audit.check(groups.get("input_file_sha256") == bindings.get("input_sha256"),
                "groups_input_binding",
                "GROUPS input SHA differs from the campaign INPUTS binding")
    audit.check(groups.get("finalization_timing") ==
                "before any candidate performance measurement",
                "groups_finalization_timing",
                "GROUPS does not declare pre-measurement finalization")
    group_rows = groups.get("groups")
    if not isinstance(group_rows, list):
        audit.check(False, "groups_rows", "GROUPS groups is not an array")
        group_rows = []
    from_rows: dict[str, str] = {}
    for index, group in enumerate(group_rows):
        if not isinstance(group, dict) or not isinstance(group.get("group_id"), str):
            audit.check(False, "groups_row_identity", "GROUPS row is invalid",
                        index=index)
            continue
        members = group.get("datasets")
        if not isinstance(members, list) or any(
                not isinstance(item, str) for item in members):
            audit.check(False, "groups_row_members", "GROUPS members are invalid",
                        index=index)
            continue
        for dataset in members:
            audit.check(dataset not in from_rows, "groups_dataset_duplicate",
                        "a dataset appears in multiple GROUPS rows",
                        dataset_id=dataset)
            from_rows[dataset] = group["group_id"]
    audit.check(from_rows == mapping, "groups_rows_mapping",
                "GROUPS rows do not reconstruct dataset_to_group")
    groups_sha = hashes.sha256(groups_path)
    audit.check(groups_sha == bindings.get("groups_sha256"),
                "campaign_groups_binding",
                "campaign GROUPS SHA differs from GROUPS.json",
                actual=groups_sha, expected=bindings.get("groups_sha256"))
    return ({"path": str(groups_path), "sha256": groups_sha,
             "dataset_count": len(mapping), "group_count": len(group_ids),
             "duplicate_merge_count": len(groups.get("sample_duplicate_merges", []))
             if isinstance(groups.get("sample_duplicate_merges"), list) else None},
            mapping)


def audit_campaign(audit: Auditor, hashes: HashCache, campaign_root: Path,
                   inputs_path: Path, samples: dict[str, dict[str, Any]],
                   candidate_ids: list[str], candidate_kinds: dict[str, str],
                   catalog_canonical_sha: str,
                   group_mapping: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_index_path = campaign_root / "CASE_INDEX.json"
    rows_path = campaign_root / "ROWS.json"
    bindings_path = campaign_root / "BINDINGS.json"
    complete_path = campaign_root / "CAMPAIGN_COMPLETE.json"
    for path in (case_index_path, rows_path, bindings_path, complete_path):
        audit.require(path.is_file(), "campaign_file_missing",
                      "campaign output file is missing", path=str(path))
    index = read_json(case_index_path)
    rows = read_json(rows_path)
    bindings = read_json(bindings_path)
    complete = read_json(complete_path)
    if not isinstance(index, list) or not isinstance(rows, list):
        raise AuditProblem("CASE_INDEX.json and ROWS.json must be arrays")
    audit.check(bindings.get("input_sha256") == hashes.sha256(inputs_path),
                "campaign_input_binding", "campaign input SHA is not bound to INPUTS.json")
    audit.check(bindings.get("candidate_catalog_sha256") == catalog_canonical_sha,
                "campaign_catalog_binding", "campaign catalog SHA differs")
    expected_cases = len(samples) * len(candidate_ids)
    audit.check(len(index) == expected_cases, "case_count",
                "CASE_INDEX does not contain samples*24 cases",
                actual=len(index), expected=expected_cases,
                samples=len(samples), candidates=len(candidate_ids))
    by_case: dict[tuple[str, str], dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    strict_by_candidate: Counter[str] = Counter()
    timing_sample_counts: Counter[int] = Counter()
    expected_case_set = {(sample_id, candidate_id) for sample_id in samples
                         for candidate_id in candidate_ids}
    for position, entry in enumerate(index):
        if not isinstance(entry, dict):
            audit.check(False, "case_index_entry", "CASE_INDEX entry is not an object",
                        position=position)
            continue
        key = (entry.get("sample_id"), entry.get("candidate_id"))
        if not all(isinstance(item, str) and item for item in key):
            audit.check(False, "case_index_identity",
                        "CASE_INDEX entry has an invalid sample/candidate identity",
                        position=position, key=key)
            continue
        key = cast(tuple[str, str], key)
        audit.check(key not in by_case, "duplicate_case",
                    "CASE_INDEX contains a duplicate sample/candidate pair", key=key)
        try:
            result_path = resolve_artifact(audit, entry.get("result_path"),
                                           f"CASE_INDEX[{position}].result_path",
                                           campaign_root)
            actual_result_sha = hashes.sha256(result_path)
            audit.check(actual_result_sha == entry.get("sha256"),
                        "result_index_sha", "RESULT raw SHA differs from CASE_INDEX",
                        key=key, actual=actual_result_sha,
                        expected=entry.get("sha256"))
            sha_path = result_path.with_name("RESULT.sha256")
            audit.check(sha_path.is_file(), "result_sha_file_missing",
                        "RESULT.sha256 is missing", key=key, path=str(sha_path))
            if sha_path.is_file():
                tokens = sha_path.read_text(encoding="ascii").split()
                audit.check(tokens == [actual_result_sha], "result_sha_file",
                            "RESULT.sha256 does not contain the raw RESULT hash",
                            key=key, tokens=tokens)
            result = read_json(result_path)
        except (AuditProblem, OSError, json.JSONDecodeError) as error:
            audit.check(False, "result_unreadable", str(error), key=key)
            continue
        by_case[key] = result
        audit.check(result.get("sample_id") == key[0] and
                    result.get("candidate_id") == key[1], "result_identity",
                    "RESULT identity differs from CASE_INDEX", key=key)
        audit.check(result.get("dataset_id") == samples.get(key[0], {}).get("dataset_id"),
                    "result_dataset", "RESULT dataset differs from INPUTS", key=key)
        audit.check(result.get("kind") == candidate_kinds.get(key[1]),
                    "result_kind", "RESULT kind differs from catalog", key=key)
        status = result.get("status")
        audit.check(status == entry.get("status"), "result_status_index",
                    "RESULT status differs from CASE_INDEX", key=key)
        audit.check(status in {"strict_verified", "domain_rejected", "failed"},
                    "result_status_unknown", "RESULT has an unknown status",
                    key=key, status=status)
        status_counts[str(status)] += 1
        audit.check(result.get("bindings") == bindings, "result_bindings",
                    "RESULT does not carry the campaign bindings", key=key)
        sample = samples.get(key[0])
        if sample is not None:
            audit.check(result.get("source") == sample.get("tail"),
                        "result_source", "RESULT tail source differs from INPUTS",
                        key=key)
            sample_path = result_path.parent.parent / "SAMPLE.json"
            audit.check(sample_path.is_file(), "sample_receipt_missing",
                        "SAMPLE.json is missing", key=key)
            if sample_path.is_file():
                sample_sha = hashes.sha256(sample_path)
                audit.check(sample_sha == result.get("sample_sha256"),
                            "sample_receipt_sha", "RESULT sample SHA differs",
                            key=key)
                try:
                    audit.check(read_json(sample_path) == sample,
                                "sample_receipt_content",
                                "SAMPLE.json differs from INPUTS sample", key=key)
                except (OSError, json.JSONDecodeError) as error:
                    audit.check(False, "sample_receipt_unreadable", str(error), key=key)
            measure_process_path = result_path.parent.parent / "MEASURE_PROCESS.json"
            try:
                audit.check(result.get("measurement_process") ==
                            read_json(measure_process_path),
                            "result_measurement_process_copy",
                            "RESULT measurement process differs from sample receipt",
                            key=key)
            except (OSError, json.JSONDecodeError) as error:
                audit.check(False, "measurement_process_unreadable", str(error),
                            key=key)
        if status == "failed":
            audit.check(False, "failed_case", "campaign contains a failed case", key=key)
        if status == "domain_rejected":
            measurement = result.get("measurement")
            audit.check(isinstance(measurement, dict) and
                        measurement.get("status") == "domain_rejected",
                        "domain_status_binding",
                        "domain-rejected RESULT lacks a domain measurement", key=key)
        if status != "strict_verified":
            continue
        strict_by_candidate[str(key[1])] += 1
        measurement = result.get("measurement")
        if not isinstance(measurement, dict):
            audit.check(False, "strict_measurement_missing",
                        "strict RESULT lacks its measurement", key=key)
            continue
        audit.check(measurement.get("status") == "measured",
                    "strict_measurement_status",
                    "strict RESULT measurement is not measured", key=key)
        measurement_audit = measurement.get("audit")
        audit.check(isinstance(measurement_audit, dict) and
                    measurement_audit.get("execution_status") == "strict_ready" and
                    measurement_audit.get("fallback_used") is False,
                    "strict_runtime_audit",
                    "strict RESULT lacks strict_ready/no-fallback runtime evidence",
                    key=key)
        raw_timings = measurement.get("raw_timings")
        if not isinstance(raw_timings, list):
            audit.check(False, "raw_timings_missing",
                        "strict measurement lacks raw_timings", key=key)
        else:
            timing_sample_counts[len(raw_timings)] += 1
            audit.check(len(raw_timings) >= 5, "raw_timing_count",
                        "strict measurement has fewer than five timing samples",
                        key=key, actual=len(raw_timings))
            repeats = [timing.get("repeat") for timing in raw_timings
                       if isinstance(timing, dict)]
            audit.check(repeats == list(range(len(raw_timings))),
                        "raw_timing_repeats",
                        "raw timing repeats are not contiguous", key=key,
                        actual=repeats)
            for timing_key in TIMING_KEYS:
                values = [timing.get(timing_key) for timing in raw_timings
                          if isinstance(timing, dict)]
                valid = (len(values) == len(raw_timings) and
                         all(isinstance(item, int) and not isinstance(item, bool) and
                             item >= 0 for item in values))
                audit.check(valid, "raw_timing_value",
                            "raw timing contains an invalid value",
                            key=key, timing_key=timing_key)
                if valid:
                    integer_values = cast(list[int], values)
                    audit.check(measurement.get(timing_key) ==
                                upper_median(integer_values),
                                "timing_median",
                                "stored timing is not the upper median of raw samples",
                                key=key, timing_key=timing_key,
                                actual=measurement.get(timing_key),
                                expected=upper_median(integer_values))
        try:
            frame_path = resolve_artifact(audit, measurement.get("frame_path"),
                                          f"{key}.frame_path", result_path.parent)
            frame_sha = hashes.sha256(frame_path)
            audit.check(frame_sha == measurement.get("frame_sha256"),
                        "frame_sha", "strict frame SHA differs from measurement",
                        key=key, actual=frame_sha,
                        expected=measurement.get("frame_sha256"))
            audit.check(frame_path.stat().st_size == measurement.get("frame_bytes"),
                        "frame_size", "strict frame size differs", key=key)
        except (AuditProblem, OSError) as error:
            audit.check(False, "frame_unreadable", str(error), key=key)
            continue
        process = result.get("measurement_process")
        decode_process = result.get("independent_decode_process")
        verification = result.get("independent_verification")
        audit.check(isinstance(process, dict) and process.get("exit_code") == 0 and
                    process.get("timeout") is False,
                    "measurement_process_status",
                    "strict RESULT measurement process did not exit cleanly", key=key)
        audit.check(isinstance(decode_process, dict) and
                    decode_process.get("exit_code") == 0 and
                    decode_process.get("timeout") is False,
                    "decode_process_status",
                    "strict RESULT decode process did not exit cleanly", key=key)
        if not isinstance(decode_process, dict) or not isinstance(verification, dict):
            continue
        expected_decode_argv = ["decode", measurement.get("frame_path"),
                                str(result_path.parent / "decoded")]
        audit.check(decode_process.get("argv") == expected_decode_argv,
                    "decode_frame_only_argv",
                    "independent decoder was not invoked with only frame and output paths",
                    key=key, actual=decode_process.get("argv"),
                    expected=expected_decode_argv)
        decoded_root = result_path.parent / "decoded"
        decode_path = decoded_root / "DECODE.json"
        try:
            decode = read_json(decode_path)
            values_path = decoded_root / "values.bin"
            missing_path = decoded_root / "missing.bin"
            values_sha = hashes.sha256(values_path)
            missing_sha = hashes.sha256(missing_path)
        except (OSError, json.JSONDecodeError) as error:
            audit.check(False, "decoded_artifact_unreadable", str(error), key=key)
            continue
        audit.check(verification.get("verified") is True,
                    "independent_verification_flag",
                    "strict RESULT independent verification is false", key=key)
        verification_checks = verification.get("checks")
        audit.check(isinstance(verification_checks, dict) and
                    bool(verification_checks) and
                    all(value is True for value in verification_checks.values()),
                    "independent_verification_checks",
                    "one or more independent decode checks is not true", key=key)
        audit.check(verification.get("decode") == decode,
                    "decode_receipt_copy",
                    "RESULT does not contain the exact DECODE receipt", key=key)
        audit.check(decode.get("decode_pid") == decode_process.get("pid"),
                    "decode_pid_binding", "DECODE PID differs from child PID", key=key)
        if isinstance(process, dict):
            audit.check(decode.get("decode_pid") != process.get("pid"),
                        "decode_process_isolation",
                        "measurement and independent decode reuse a PID", key=key)
            measurement_pid = process.get("pid")
            decode_pid = decode_process.get("pid")
            audit.check(isinstance(measurement_pid, int) and
                        isinstance(decode_pid, int) and
                        measurement_pid > 0 and decode_pid > 0 and
                        process.get("parent_pid") == decode_process.get("parent_pid"),
                        "child_process_parent_binding",
                        "measurement/decode child receipts do not share a valid campaign parent",
                        key=key)
        audit.check(decode.get("frame_sha256") == measurement.get("frame_sha256"),
                    "decode_frame_binding",
                    "DECODE frame SHA differs from measured frame", key=key)
        audit.check(decode.get("input_contract") ==
                    "frame bytes only; no source input, candidate ID or inverse sidecar",
                    "decode_input_contract",
                    "DECODE receipt does not declare the frozen frame-only contract",
                    key=key)
        tail = samples[key[0]]["tail"]
        audit.check(values_sha == decode.get("values_sha256") ==
                    tail.get("values_sha256"), "decoded_values_sha",
                    "decoded values are not bound to DECODE and INPUTS", key=key)
        audit.check(missing_sha == decode.get("missing_sha256") ==
                    tail.get("missing_sha256"), "decoded_missing_sha",
                    "decoded missing bitmap is not bound to DECODE and INPUTS", key=key)
        audit.check(decode.get("dtype") == samples[key[0]].get("dtype"),
                    "decoded_dtype", "DECODE dtype differs from INPUTS", key=key)
        audit.check(decode.get("row_count") == tail.get("row_count"),
                    "decoded_row_count", "DECODE row_count differs from INPUTS", key=key)
    audit.check(set(by_case) == expected_case_set, "case_cartesian_coverage",
                "CASE_INDEX is not the full sample*candidate Cartesian product",
                missing_count=len(expected_case_set - set(by_case)),
                extra_count=len(set(by_case) - expected_case_set))
    for candidate_id in candidate_ids:
        audit.check(strict_by_candidate[candidate_id] > 0,
                    "candidate_never_strict",
                    "candidate has no strict-verified case",
                    candidate_id=candidate_id)
    audit.check(status_counts["failed"] == 0, "campaign_failures",
                "campaign contains failed cases", count=status_counts["failed"])
    audit.check(len(rows) == len(samples), "rows_count",
                "ROWS count differs from INPUTS sample count",
                actual=len(rows), expected=len(samples))
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            audit.check(False, "row_identity", "ROWS contains an invalid row")
            continue
        sample_id = row["sample_id"]
        audit.check(sample_id not in rows_by_id, "duplicate_row",
                    "ROWS contains a duplicate sample", sample_id=sample_id)
        rows_by_id[sample_id] = row
        sample = samples.get(sample_id)
        audit.check(sample is not None, "row_unknown_sample",
                    "ROWS contains a sample absent from INPUTS", sample_id=sample_id)
        if sample is None:
            continue
        for key_name in ("dataset_id", "dtype", "scale"):
            audit.check(row.get(key_name) == sample.get(key_name),
                        "row_sample_metadata", "ROWS metadata differs from INPUTS",
                        sample_id=sample_id, field=key_name)
        sample_dataset = sample.get("dataset_id")
        expected_group = (group_mapping.get(sample_dataset)
                          if isinstance(sample_dataset, str) else None)
        audit.check(row.get("group_id") == expected_group,
                    "row_group_mapping", "ROWS group differs from GROUPS",
                    sample_id=sample_id, actual=row.get("group_id"),
                    expected=expected_group)
        try:
            tail = sample["tail"]
            values_path = resolve_artifact(
                audit, tail.get("values_path"), f"{sample_id}.tail.values_path",
                inputs_path.parent)
            missing_path = resolve_artifact(
                audit, tail.get("missing_path"), f"{sample_id}.tail.missing_path",
                inputs_path.parent)
            audit.check(row.get("input_bytes") ==
                        values_path.stat().st_size + missing_path.stat().st_size,
                        "row_input_bytes",
                        "ROWS input_bytes differs from tail values+mask bytes",
                        sample_id=sample_id)
        except (AuditProblem, OSError) as error:
            audit.check(False, "row_input_bytes_unreadable", str(error),
                        sample_id=sample_id)
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            audit.check(False, "row_metrics", "row metrics is not an object",
                        sample_id=sample_id)
            continue
        audit.check(set(metrics) == set(candidate_ids), "row_metric_coverage",
                    "row metrics do not cover the candidate catalog",
                    sample_id=sample_id)
        for candidate_id in candidate_ids:
            metric = metrics.get(candidate_id)
            result = by_case.get((sample_id, candidate_id))
            if not isinstance(metric, dict) or result is None:
                continue
            audit.check(metric.get("status") == result.get("status"),
                        "row_metric_status", "row metric status differs from RESULT",
                        sample_id=sample_id, candidate_id=candidate_id)
            audit.check(metric.get("kind") == candidate_kinds[candidate_id],
                        "row_metric_kind", "row metric kind differs from catalog",
                        sample_id=sample_id, candidate_id=candidate_id)
            if result.get("status") == "strict_verified":
                measurement = result.get("measurement", {})
                for metric_key in STRICT_METRIC_KEYS:
                    audit.check(metric.get(metric_key) == measurement.get(metric_key),
                                "row_metric_value",
                                "row metric differs from RESULT measurement",
                                sample_id=sample_id, candidate_id=candidate_id,
                                field=metric_key)
        sample_root = campaign_root / "samples" / sample_id
        measurements_path = sample_root / "MEASUREMENTS.json"
        if measurements_path.is_file():
            try:
                measured = read_json(measurements_path)
                for field in ("features", "feature_cpu_ns", "feature_wall_ns",
                              "eligibility_cpu_ns", "eligibility_wall_ns",
                              "eligibility_provenance", "catalog_sha256",
                              "input_bytes", "eligible_candidates"):
                    audit.check(row.get(field) == measured.get(field),
                                "row_measurement_field",
                                "ROWS field differs from MEASUREMENTS",
                                sample_id=sample_id, field=field)
                process_path = sample_root / "MEASURE_PROCESS.json"
                process_receipt = read_json(process_path)
                audit.check(measured.get("measurement_pid") ==
                            process_receipt.get("pid"),
                            "measurement_pid_binding",
                            "MEASUREMENTS PID differs from MEASURE_PROCESS",
                            sample_id=sample_id)
                audit.check(process_receipt.get("exit_code") == 0 and
                            process_receipt.get("timeout") is False,
                            "sample_measurement_process",
                            "sample measurement process did not exit cleanly",
                            sample_id=sample_id)
                audit.check(measured.get("catalog_sha256") == catalog_canonical_sha,
                            "measurement_catalog_binding",
                            "MEASUREMENTS catalog SHA differs from campaign catalog",
                            sample_id=sample_id)
            except (OSError, json.JSONDecodeError) as error:
                audit.check(False, "measurements_unreadable", str(error),
                            sample_id=sample_id)
        else:
            audit.check(False, "measurements_missing",
                        "sample MEASUREMENTS.json is missing", sample_id=sample_id)
    audit.check(set(rows_by_id) == set(samples), "row_sample_coverage",
                "ROWS sample IDs differ from INPUTS",
                missing=sorted(set(samples) - set(rows_by_id))[:20],
                extra=sorted(set(rows_by_id) - set(samples))[:20])
    audit.check(complete.get("index_sha256") == hashes.sha256(case_index_path),
                "complete_index_sha", "CAMPAIGN_COMPLETE index SHA differs")
    audit.check(complete.get("rows_sha256") == hashes.sha256(rows_path),
                "complete_rows_sha", "CAMPAIGN_COMPLETE rows SHA differs")
    audit.check(complete.get("samples") == len(samples) and
                complete.get("cases") == expected_cases,
                "complete_counts", "CAMPAIGN_COMPLETE counts differ")
    audit.check(complete.get("strict_verified") == status_counts["strict_verified"] and
                complete.get("domain_rejected") == status_counts["domain_rejected"] and
                complete.get("failed") == status_counts["failed"],
                "complete_status_counts", "CAMPAIGN_COMPLETE status counts differ")
    audit.check(complete.get("bindings") == bindings,
                "complete_bindings",
                "CAMPAIGN_COMPLETE bindings differ from BINDINGS.json")
    audit.check(complete.get("status") == "completed" and
                complete.get("lossless_bypass") == 0,
                "complete_status",
                "CAMPAIGN_COMPLETE is not a completed zero-bypass campaign")
    return ({"root": str(campaign_root),
             "case_index_sha256": hashes.sha256(case_index_path),
             "rows_sha256": hashes.sha256(rows_path),
             "sample_count": len(samples), "row_count": len(rows),
             "case_count": len(index), "expected_case_count": expected_cases,
             "status_counts": dict(sorted(status_counts.items())),
             "strict_cases_by_candidate": dict(sorted(strict_by_candidate.items())),
             "raw_timing_sample_counts": {
                 str(key): value for key, value in sorted(timing_sample_counts.items())}},
            rows)


def compare_baseline(audit: Auditor, actual: Any, expected: dict[str, Any] | None,
                     context: str) -> None:
    if expected is None:
        audit.check(actual is None, "baseline_selection",
                    f"{context} should be null")
        return
    if not isinstance(actual, dict):
        audit.check(False, "baseline_selection", f"{context} is missing")
        return
    audit.check(actual.get("candidate_id") == expected["candidate_id"] and
                actual.get("training_rows") == expected["training_rows"],
                "baseline_selection", f"{context} candidate/count differs",
                actual=actual, expected=expected)
    check_float(audit, actual.get("training_mean_log_cost"),
                expected["training_mean_log_cost"], "baseline_selection",
                f"{context}.training_mean_log_cost")


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGETS:
        scored = [prediction for prediction in predictions
                  if prediction["scores"][target].get("oracle") is not None]
        ranked_rows = sum(bool(prediction["prediction"]["targets"][target]
                               .get("ranking", []))
                          for prediction in predictions)
        target_summary: dict[str, Any] = {
            "coverage": {"total_rows": len(predictions),
                         "scored_rows": len(scored),
                         "unscored_rows": len(predictions) - len(scored),
                         "ranked_rows": ranked_rows,
                         "unranked_rows": len(predictions) - ranked_rows}}
        for k in TOP_KS:
            rows = len(scored)
            hits = wall_hits = dominated_count = 0
            pareto_recall_sum = 0.0
            regrets: list[float] = []
            wall_regrets: list[float] = []
            for prediction in scored:
                top = prediction["scores"][target]["topk"][str(k)]
                hits += int(top["tie_aware_hit"])
                wall_hits += int(top["wall_tie_aware_hit"])
                dominated_count += int(top["dominated_count"])
                pareto_recall_sum += finite_number(
                    top["pareto_recall"], "summary pareto_recall")
                regret = top.get(
                    "oracle_best_within_topk_multiplicative_regret")
                if is_number(regret) and float(regret) > 0 and math.isfinite(float(regret)):
                    regrets.append(float(regret))
                wall_regret = top.get(
                    "wall_oracle_best_within_topk_multiplicative_regret")
                if (is_number(wall_regret) and float(wall_regret) > 0 and
                        math.isfinite(float(wall_regret))):
                    wall_regrets.append(float(wall_regret))
            top_summary: dict[str, Any] = {
                "rows_with_oracle": rows,
                "unscored_rows": len(predictions) - rows,
                "tie_aware_hits": hits,
                "tie_aware_hit_rate": hits / rows if rows else 0.0,
                "wall_tie_aware_hits": wall_hits,
                "wall_tie_aware_hit_rate": wall_hits / rows if rows else 0.0,
                "geometric_mean_oracle_best_within_topk_multiplicative_regret":
                    math.exp(sum(math.log(item) for item in regrets) / len(regrets))
                    if regrets else None,
                "wall_geometric_mean_oracle_best_within_topk_multiplicative_regret":
                    math.exp(sum(math.log(item) for item in wall_regrets) /
                             len(wall_regrets)) if wall_regrets else None,
                "pareto_recall_mean": pareto_recall_sum / rows if rows else 0.0,
                "dominated_count": dominated_count}
            if k == 1:
                top_summary[
                    "geometric_mean_online_selected_multiplicative_regret"] = (
                        top_summary[
                            "geometric_mean_oracle_best_within_topk_multiplicative_regret"])
                top_summary[
                    "wall_geometric_mean_online_selected_multiplicative_regret"] = (
                        top_summary[
                            "wall_geometric_mean_oracle_best_within_topk_multiplicative_regret"])
            target_summary[f"top{k}"] = top_summary
        result[target] = target_summary
    return result


def compare_json_numbers(audit: Auditor, actual: Any, expected: Any,
                         code: str, context: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            audit.check(False, code, f"{context} is not an object")
            return
        audit.check(set(actual) == set(expected), code,
                    f"{context} has different fields",
                    actual=sorted(actual), expected=sorted(expected))
        for key, value in expected.items():
            compare_json_numbers(audit, actual.get(key), value, code,
                                 f"{context}.{key}")
    elif is_number(expected):
        check_float(audit, actual, float(expected), code, context)
    else:
        audit.check(actual == expected, code,
                    f"{context} differs from independent recomputation",
                    actual=actual, expected=expected)


def audit_evaluation(audit: Auditor, hashes: HashCache, evaluation_path: Path,
                     rows: list[dict[str, Any]], candidate_ids: list[str],
                     candidate_kinds: dict[str, str],
                     catalog_sha: str, rows_sha: str,
                     source_bindings: dict[str, Any]) -> dict[str, Any]:
    evaluation = read_json(evaluation_path)
    audit.check(evaluation.get("schema") == "operator-recommend135-evaluation-v1",
                "evaluation_schema", "EVALUATION.json has the wrong schema")
    scope = evaluation.get("candidate_scope", {})
    audit.check(scope.get("candidate_ids") == candidate_ids,
                "evaluation_candidate_scope",
                "evaluation candidate IDs differ from catalog")
    audit.check(scope.get("catalog_sha256") == catalog_sha,
                "evaluation_catalog_sha", "evaluation catalog SHA differs")
    audit.check(evaluation.get("training_rows_sha256") == rows_sha,
                "evaluation_rows_binding",
                "EVALUATION training_rows_sha256 differs from raw ROWS.json",
                actual=evaluation.get("training_rows_sha256"), expected=rows_sha)
    audit.check(evaluation.get("source_bindings") == source_bindings,
                "evaluation_source_bindings",
                "EVALUATION source_bindings differ from campaign bindings")
    rows_by_sample = {row["sample_id"]: row for row in rows
                      if isinstance(row, dict) and isinstance(row.get("sample_id"), str)}
    dataset_group: dict[str, str] = {}
    for row in rows:
        dataset = row.get("dataset_id")
        group = row.get("group_id")
        if isinstance(dataset, str) and isinstance(group, str):
            audit.check(dataset not in dataset_group or dataset_group[dataset] == group,
                        "dataset_group_split",
                        "one dataset appears in multiple source groups",
                        dataset_id=dataset)
            dataset_group[dataset] = group
    groups = sorted(set(dataset_group.values()))
    folds = evaluation.get("folds")
    if not isinstance(folds, list):
        raise AuditProblem("evaluation folds is not an array")
    audit.check(len(folds) == len(groups), "fold_count",
                "evaluation fold count differs from source group count",
                actual=len(folds), expected=len(groups))
    folds_by_group = {fold.get("held_out_group_id"): fold for fold in folds
                      if isinstance(fold, dict)}
    audit.check(set(folds_by_group) == set(groups), "fold_group_coverage",
                "evaluation does not have exactly one fold per group")
    prediction_count = 0
    all_predictions: list[dict[str, Any]] = []
    for held_out in groups:
        fold = folds_by_group.get(held_out)
        if not isinstance(fold, dict):
            continue
        training_rows = [row for row in rows if row.get("group_id") != held_out]
        test_rows = [row for row in rows if row.get("group_id") == held_out]
        training_groups = sorted({row["group_id"] for row in training_rows})
        training_datasets = sorted({row["dataset_id"] for row in training_rows})
        test_datasets = sorted({row["dataset_id"] for row in test_rows})
        audit.check(held_out not in set(fold.get("training_group_ids", [])),
                    "heldout_group_leakage",
                    "held-out group appears in training_group_ids",
                    held_out_group=held_out)
        audit.check(fold.get("training_group_ids") == training_groups,
                    "training_group_ids",
                    "fold training_group_ids differ from ROWS",
                    held_out_group=held_out)
        audit.check(fold.get("training_dataset_ids") == training_datasets,
                    "training_dataset_ids",
                    "fold training_dataset_ids differ from ROWS",
                    held_out_group=held_out)
        audit.check(fold.get("test_dataset_ids") == test_datasets,
                    "test_dataset_ids", "fold test_dataset_ids differ from ROWS",
                    held_out_group=held_out)
        audit.check(not set(training_datasets) & set(test_datasets),
                    "heldout_dataset_leakage",
                    "test datasets appear in fold training datasets",
                    held_out_group=held_out,
                    overlap=sorted(set(training_datasets) & set(test_datasets)))
        audit.check(fold.get("training_row_count") == len(training_rows) and
                    fold.get("test_row_count") == len(test_rows),
                    "fold_row_counts", "fold row counts differ from ROWS",
                    held_out_group=held_out)
        audit.check(fold.get("candidate_ids") == candidate_ids and
                    fold.get("catalog_sha256") == catalog_sha,
                    "fold_candidate_scope", "fold candidate scope differs",
                    held_out_group=held_out)
        means, scales = normalization(training_rows)
        fold_normalization = fold.get("normalization", {})
        compare_number_array(audit, fold_normalization.get("mean"), means,
                             "fold_normalization", f"fold[{held_out}].mean")
        compare_number_array(audit, fold_normalization.get("scale"), scales,
                             "fold_normalization", f"fold[{held_out}].scale")
        algorithms = [candidate for candidate in candidate_ids
                      if candidate_kinds.get(candidate, "").lower() == "algorithm"]
        baselines: dict[str, dict[str, Any]] = {}
        for target in TARGETS:
            baselines[target] = {
                "fixed_algorithm": choose_baseline(training_rows, algorithms, target),
                "fixed_candidate": choose_baseline(training_rows, candidate_ids, target)}
            actual = fold.get("baselines", {}).get(target, {})
            compare_baseline(audit, actual.get("fixed_algorithm"),
                             baselines[target]["fixed_algorithm"],
                             f"fold[{held_out}].{target}.fixed_algorithm")
            compare_baseline(audit, actual.get("fixed_candidate"),
                             baselines[target]["fixed_candidate"],
                             f"fold[{held_out}].{target}.fixed_candidate")
        predictions = fold.get("predictions")
        if not isinstance(predictions, list):
            audit.check(False, "fold_predictions", "fold predictions is not an array",
                        held_out_group=held_out)
            continue
        prediction_count += len(predictions)
        all_predictions.extend(item for item in predictions if isinstance(item, dict))
        by_sample = {prediction["sample_id"]: prediction
                     for prediction in predictions
                     if isinstance(prediction, dict) and
                     isinstance(prediction.get("sample_id"), str)}
        expected_test_ids = {row["sample_id"] for row in test_rows}
        audit.check(set(by_sample) == expected_test_ids, "fold_prediction_coverage",
                    "fold predictions differ from held-out rows",
                    held_out_group=held_out,
                    missing=sorted(expected_test_ids - set(by_sample))[:20],
                    extra=sorted(set(by_sample) - expected_test_ids)[:20])
        for sample_id in sorted(expected_test_ids):
            record = by_sample.get(sample_id)
            row = rows_by_sample[sample_id]
            if not isinstance(record, dict):
                continue
            audit.check(record.get("group_id") == held_out and
                        record.get("dataset_id") == row.get("dataset_id"),
                        "prediction_identity",
                        "prediction identity differs from held-out row",
                        sample_id=sample_id)
            expected_prediction = recompute_prediction(
                training_rows, row, candidate_ids, means, scales)
            audit_prediction_scores(audit, row, record, expected_prediction,
                                    baselines, f"fold[{held_out}].{sample_id}")
        try:
            expected_summary = summarize_predictions(
                [item for item in predictions if isinstance(item, dict)])
            compare_json_numbers(audit, fold.get("summary"), expected_summary,
                                 "fold_summary",
                                 f"fold[{held_out}].summary")
        except (AuditProblem, KeyError, TypeError) as error:
            audit.check(False, "fold_summary_unreadable", str(error),
                        held_out_group=held_out)
    audit.check(prediction_count == len(rows), "evaluation_prediction_count",
                "each row must be predicted exactly once",
                actual=prediction_count, expected=len(rows))
    try:
        compare_json_numbers(audit, evaluation.get("summary"),
                             summarize_predictions(all_predictions),
                             "evaluation_summary", "evaluation.summary")
    except (AuditProblem, KeyError, TypeError) as error:
        audit.check(False, "evaluation_summary_unreadable", str(error))
    model_path = evaluation_path.parent / "MODEL.json"
    audit.check(model_path.is_file(), "model_file_missing",
                "MODEL.json is missing next to EVALUATION.json",
                path=str(model_path))
    model_summary: dict[str, Any] | None = None
    if model_path.is_file():
        try:
            model = read_json(model_path)
            audit.check(model.get("schema") == "operator-recommend135-model-v1",
                        "model_schema", "MODEL.json has the wrong schema")
            audit.check(model.get("training_rows_sha256") == rows_sha,
                        "model_rows_binding",
                        "MODEL training_rows_sha256 differs from raw ROWS.json")
            audit.check(model.get("source_bindings") == source_bindings,
                        "model_source_bindings",
                        "MODEL source_bindings differ from campaign bindings")
            audit.check(model.get("candidate_ids") == candidate_ids,
                        "model_candidate_scope",
                        "MODEL candidate IDs differ from the catalog")
            audit.check(model.get("candidate_catalog_sha256") == catalog_sha,
                        "model_catalog_binding",
                        "MODEL catalog SHA differs from the catalog")
            training = model.get("training")
            expected_training_groups = sorted(
                {row["group_id"] for row in rows})
            expected_training_datasets = sorted(
                {row["dataset_id"] for row in rows})
            audit.check(isinstance(training, dict) and
                        training.get("row_count") == len(rows) and
                        training.get("group_ids") == expected_training_groups and
                        training.get("dataset_ids") == expected_training_datasets,
                        "model_training_scope",
                        "MODEL training scope differs from all production ROWS")
            points = model.get("points")
            point_ids = ([point.get("sample_id") for point in points]
                         if isinstance(points, list) and
                         all(isinstance(point, dict) for point in points) else [])
            audit.check(len(point_ids) == len(rows) and
                        set(point_ids) == set(rows_by_sample),
                        "model_training_points",
                        "MODEL points do not cover every production row exactly once")
            full_means, full_scales = normalization(rows)
            model_normalization = model.get("normalization")
            if not isinstance(model_normalization, dict):
                model_normalization = {}
            compare_number_array(audit, model_normalization.get("mean"),
                                 full_means, "model_normalization", "MODEL.mean")
            compare_number_array(audit, model_normalization.get("scale"),
                                 full_scales, "model_normalization", "MODEL.scale")
            if isinstance(points, list):
                for point in points:
                    if not isinstance(point, dict) or not isinstance(
                            point.get("sample_id"), str):
                        continue
                    source_row = rows_by_sample.get(point["sample_id"])
                    if source_row is None:
                        continue
                    expected_normalized = [
                        (finite_number(value, "model source feature") - full_means[i]) /
                        full_scales[i]
                        for i, value in enumerate(source_row["features"])]
                    compare_number_array(
                        audit, point.get("normalized_features"), expected_normalized,
                        "model_point_features",
                        f"MODEL.points[{point['sample_id']}].normalized_features")
                    compare_json_numbers(
                        audit, point.get("labels"),
                        labels_for_row(source_row, candidate_ids),
                        "model_point_labels",
                        f"MODEL.points[{point['sample_id']}].labels")
            model_summary = {"path": str(model_path),
                             "sha256": hashes.sha256(model_path)}
        except (OSError, json.JSONDecodeError) as error:
            audit.check(False, "model_unreadable", str(error))
    return {"path": str(evaluation_path),
            "sha256": hashes.sha256(evaluation_path),
            "fold_count": len(folds),
            "source_group_count": len(groups),
            "prediction_count": prediction_count,
            "row_count": len(rows),
            "training_rows_sha256": rows_sha,
            "model": model_summary,
            "training_only_prediction_recomputed": True,
            "oracle_topk_regret_recomputed": True,
            "fixed_baselines_recomputed": True}


def audit_build(audit: Auditor, hashes: HashCache,
                bindings: dict[str, Any]) -> dict[str, Any]:
    receipts = find_named(audit.run_root, "BUILD_RECEIPT.json")
    if len(receipts) != 1:
        audit.check(False, "build_receipt_discovery",
                    "expected exactly one BUILD_RECEIPT.json in new run root",
                    matches=[str(item) for item in receipts])
        return {"receipt_count": len(receipts)}
    path = receipts[0]
    receipt = read_json(path)
    audit.check(receipt.get("schema") == "operator-recommend135-build-v1" and
                receipt.get("status") == "built-and-tested",
                "build_receipt_status", "build receipt is not built-and-tested")
    audit.check(receipt.get("NNCP") == "OFF", "build_nncp",
                "build receipt does not bind NNCP=OFF")
    audit.check(set(receipt.get("excluded_action_ids", [])) ==
                FORBIDDEN_ACTION_IDS, "build_excluded_ids",
                "build receipt does not exclude A106-A110 IDs")
    tests = receipt.get("tests", {})
    audit.check(isinstance(tests, dict) and bool(tests) and all(
        isinstance(value, dict) and value.get("returncode") == 0
        for value in tests.values()), "build_tests",
        "one or more build receipt tests did not pass")
    cli_sha = receipt.get("binaries", {}).get("operator_recommend135_cli")
    audit.check(cli_sha == bindings.get("executable_sha256"),
                "campaign_executable_binding",
                "campaign executable SHA differs from build receipt",
                build=cli_sha, campaign=bindings.get("executable_sha256"))
    binary_matches = []
    binaries = receipt.get("binaries")
    if not isinstance(binaries, dict):
        audit.check(False, "build_binaries", "build receipt binaries is not an object")
        binaries = {}
    for name, expected_sha in binaries.items():
        candidate = path.parent / name
        audit.check(candidate.is_file() and not candidate.is_symlink(),
                    "build_binary_missing",
                    "receipt-bound build binary is missing", path=str(candidate))
        if candidate.is_file() and not candidate.is_symlink():
            actual_sha = hashes.sha256(candidate)
            audit.check(actual_sha == expected_sha, "build_binary_sha",
                        "build binary SHA differs from receipt", path=str(candidate),
                        actual=actual_sha, expected=expected_sha)
            if name == "operator_recommend135_cli" and actual_sha == cli_sha:
                binary_matches.append(str(candidate.resolve()))
    audit.check(bool(binary_matches), "built_binary_missing",
                "no CLI binary in run root matches BUILD_RECEIPT",
                expected_sha256=cli_sha)
    source_hashes = receipt.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        audit.check(False, "build_source_hashes",
                    "build receipt has no source_sha256 map")
        source_hashes = {}
    verified_sources = 0
    for source_name, expected_sha in source_hashes.items():
        try:
            source_path = resolve_artifact(audit, source_name,
                                           "BUILD_RECEIPT source", path.parent)
            actual_sha = hashes.sha256(source_path)
            audit.check(actual_sha == expected_sha, "build_source_sha",
                        "build source SHA differs from receipt",
                        path=str(source_path), actual=actual_sha,
                        expected=expected_sha)
            verified_sources += 1
        except (AuditProblem, OSError) as error:
            audit.check(False, "build_source_unreadable", str(error),
                        path=str(source_name))
    maps = receipt.get("maps_sha256")
    if not isinstance(maps, dict) or not maps:
        audit.check(False, "build_map_hashes", "build receipt has no maps_sha256 map")
        maps = {}
    verified_maps = 0
    for map_name, expected_sha in maps.items():
        map_path = path.parent / map_name
        audit.check(map_path.is_file() and not map_path.is_symlink(),
                    "build_map_missing", "receipt-bound link map is missing",
                    path=str(map_path))
        if map_path.is_file() and not map_path.is_symlink():
            actual_sha = hashes.sha256(map_path)
            audit.check(actual_sha == expected_sha, "build_map_sha",
                        "link map SHA differs from receipt", path=str(map_path),
                        actual=actual_sha, expected=expected_sha)
            verified_maps += 1
    return {"path": str(path), "sha256": hashes.sha256(path),
            "status": receipt.get("status"), "NNCP": receipt.get("NNCP"),
            "test_count": len(tests) if isinstance(tests, dict) else 0,
            "verified_source_count": verified_sources,
            "verified_map_count": verified_maps,
            "matching_cli_binaries": binary_matches}


def classify_file(relative: Path) -> str:
    lower_name = relative.name.lower()
    if (lower_name.endswith(".log") or lower_name in
            {"progress.json", "phase.json"} or ".tmp." in lower_name or
            lower_name.endswith(".partial")):
        return "logs_progress_transient"
    parts = {part.lower() for part in relative.parts}
    if parts & {"source", "sources", "build", "build-local", "cpp", "scripts"}:
        return "source_build"
    return "final"


def files_manifest(run_root: Path, audit_path: Path,
                   hashes: HashCache) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    symlinks: list[dict[str, Any]] = []
    for directory, directories, files in os.walk(run_root, followlinks=False):
        here = Path(directory)
        relative_here = here.relative_to(run_root)
        kept_directories = []
        for name in directories:
            path = here / name
            if relative_here == Path(".") and name == "audit":
                continue
            if path.is_symlink():
                symlinks.append({"path": str(path.relative_to(run_root)),
                                 "target": os.readlink(path), "kind": "directory"})
            else:
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in files:
            path = here / name
            relative = path.relative_to(run_root)
            if path.is_symlink():
                symlinks.append({"path": str(relative),
                                 "target": os.readlink(path), "kind": "file"})
                continue
            if not path.is_file():
                continue
            category = classify_file(relative)
            categories[category].append(
                {"path": str(relative), "bytes": path.stat().st_size,
                 "sha256": hashes.sha256(path)})
    categories["final"].append(
        {"path": str(audit_path.relative_to(run_root)),
         "bytes": audit_path.stat().st_size, "sha256": hashes.sha256(audit_path)})
    for entries in categories.values():
        entries.sort(key=lambda item: item["path"])
    return {"schema": FILES_SCHEMA, "status": "complete",
            "run_root": str(run_root),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_excludes": ["audit/FILES_SHA256.json",
                                  "audit temporary files"],
            "counts": {category: len(entries)
                       for category, entries in sorted(categories.items())},
            "files": dict(sorted(categories.items())),
            "symlinks_not_followed": sorted(symlinks,
                                             key=lambda item: item["path"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path,
                        help="new recommend135 run root; read-only except run_root/audit")
    parser.add_argument("--inputs", type=Path,
                        help="INPUTS.json or its directory, if discovery is ambiguous")
    parser.add_argument("--campaign", type=Path,
                        help="campaign directory or CASE_INDEX.json")
    parser.add_argument("--evaluation", type=Path,
                        help="EVALUATION.json or its directory, if discovery is ambiguous")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve(strict=True)
    if not run_root.is_dir():
        raise SystemExit(f"run root is not a directory: {run_root}")
    audit = Auditor(run_root)
    hashes = HashCache()
    audit_root = run_root / "audit"
    if audit_root.exists() and (not audit_root.is_dir() or audit_root.is_symlink()):
        raise SystemExit(f"audit output is not a real directory: {audit_root}")
    audit_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run_root": str(run_root),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_contract": "hash and metadata audit only; no codec/model kernel executed",
    }
    samples: dict[str, dict[str, Any]] = {}
    candidate_ids: list[str] = []
    candidate_kinds: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    inputs_path: Path | None = None
    catalog_path: Path | None = None
    campaign_root: Path | None = None
    catalog_sha = ""
    group_mapping: dict[str, str] = {}
    source_bindings: dict[str, Any] = {}
    try:
        inputs_path = select_named(audit, "INPUTS.json", args.inputs)
        if inputs_path is not None:
            report["inputs"], samples = audit_inputs(audit, hashes, inputs_path)
    except Exception as error:
        audit.section_error("inputs", error)
    try:
        campaign_selection = args.campaign
        if campaign_selection is not None:
            selected = campaign_selection if campaign_selection.is_absolute() else run_root / campaign_selection
            campaign_root = selected if selected.is_dir() else selected.parent
        else:
            index_path = select_named(audit, "CASE_INDEX.json", None)
            campaign_root = index_path.parent if index_path is not None else None
        if campaign_root is None:
            raise AuditProblem("campaign root was not found")
        catalog_path = campaign_root / "CANDIDATES.json"
        audit.require(catalog_path.is_file(), "catalog_missing",
                      "campaign CANDIDATES.json is missing", path=str(catalog_path))
        report["catalog"], candidate_ids, candidate_kinds = audit_catalog(
            audit, hashes, catalog_path)
        catalog_sha = report["catalog"]["canonical_sha256"]
    except Exception as error:
        audit.section_error("catalog", error)
    try:
        if not (campaign_root and inputs_path and samples and candidate_ids):
            raise AuditProblem("campaign prerequisites are incomplete")
        source_bindings = read_json(campaign_root / "BINDINGS.json")
        report["groups"], group_mapping = audit_groups(
            audit, hashes, samples, source_bindings)
        report["campaign"], rows = audit_campaign(
            audit, hashes, campaign_root, inputs_path, samples,
            candidate_ids, candidate_kinds, catalog_sha, group_mapping)
    except Exception as error:
        audit.section_error("campaign", error)
    try:
        evaluation_path = select_named(audit, "EVALUATION.json", args.evaluation)
        if (evaluation_path is None or campaign_root is None or not rows or
                not candidate_ids):
            raise AuditProblem("evaluation prerequisites are incomplete")
        report["evaluation"] = audit_evaluation(
            audit, hashes, evaluation_path, rows, candidate_ids,
            candidate_kinds, catalog_sha,
            hashes.sha256(campaign_root / "ROWS.json"), source_bindings)
    except Exception as error:
        audit.section_error("evaluation", error)
    try:
        if campaign_root is None:
            raise AuditProblem("campaign root is unavailable")
        bindings = read_json(campaign_root / "BINDINGS.json")
        report["build"] = audit_build(audit, hashes, bindings)
    except Exception as error:
        audit.section_error("build", error)
    report["checks_performed"] = audit.checks
    report["finding_count"] = len(audit.findings) + audit.suppressed_findings
    report["findings"] = audit.findings
    if audit.suppressed_findings:
        report["suppressed_findings"] = audit.suppressed_findings
    report["status"] = "pass" if not audit.findings and not audit.suppressed_findings else "fail"
    audit_path = audit_root / "AUDIT.json"
    atomic_json(audit_path, report)
    manifest = files_manifest(run_root, audit_path, hashes)
    atomic_json(audit_root / "FILES_SHA256.json", manifest)
    print(json.dumps({"status": report["status"],
                      "checks": report["checks_performed"],
                      "findings": report["finding_count"],
                      "audit": str(audit_path),
                      "files": str(audit_root / "FILES_SHA256.json")},
                     sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
