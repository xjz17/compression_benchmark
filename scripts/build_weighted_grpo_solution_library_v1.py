#!/usr/bin/env python3
"""Build the immutable training-only starting-recipe library for weighted GRPO.

The builder accepts only verified prepared TRAIN blocks and complete typed-graph
measurement receipts.  Every retained frame is hash checked, restored in this
process, and restored again by a fresh Python interpreter.  Validation/test
rows, stale registries, proxy metrics, and recipes without actual complete
frames fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import calibrate_weighted_grpo_references_v1 as calibration
import prepare_weighted_grpo_timeseries_v1 as preparation
import typed_graph_runtime as graph_runtime
import weight_conditioned_graph_task_v1 as graph_task
import weighted_grpo_graph_task_factory_v1 as factory


RECEIPT_SCHEMA = "weighted-grpo-training-recipe-measurements-v1"
LIBRARY_SCHEMA = factory.SOLUTION_LIBRARY_SCHEMA
LIBRARY_FORMAT = factory.SOLUTION_LIBRARY_FORMAT


class SolutionLibraryBuildError(RuntimeError):
    """The training evidence cannot produce a leakage-free solution library."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise SolutionLibraryBuildError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SolutionLibraryBuildError(f"{label} must be a non-empty string")
    return value


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SolutionLibraryBuildError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SolutionLibraryBuildError(f"{label} must be a JSON object")
    return value, raw


def _safe_prepared_child(root: Path, relative: Any, label: str) -> Path:
    text = _required_text(relative, label)
    path = (root / text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise SolutionLibraryBuildError(f"{label} leaves the prepared root") from error
    if not path.is_file():
        raise SolutionLibraryBuildError(f"missing prepared payload: {path}")
    return path


def _load_prepared(
    paths: Sequence[Path],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise SolutionLibraryBuildError("at least one prepared training dataset is required")
    blocks: dict[tuple[str, str], dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for supplied in paths:
        root = supplied.resolve() if supplied.is_dir() else supplied.resolve().parent
        manifest_path = root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise SolutionLibraryBuildError(f"missing prepared manifest: {manifest_path}")
        try:
            verification = preparation.verify_prepared_dataset(root, verify_source=True)
        except preparation.PreparationError as error:
            raise SolutionLibraryBuildError(
                f"prepared dataset verification failed: {manifest_path}"
            ) from error
        manifest, raw = _load_json(manifest_path, "prepared manifest")
        dataset_id = _required_text(manifest.get("dataset_id"), "prepared dataset_id")
        cohort = manifest.get("cohort")
        if not isinstance(cohort, Mapping) or cohort.get("role") != "train_validation":
            raise SolutionLibraryBuildError(
                f"prepared dataset {dataset_id!r} is not a frozen train_validation source"
            )
        if verification.get("dataset_id") != dataset_id:
            raise SolutionLibraryBuildError("prepared verification identity mismatch")
        manifest_provenance = _valid_sha(
            manifest.get("provenance_sha256"), "prepared provenance_sha256"
        )
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise SolutionLibraryBuildError("prepared manifest source must be an object")
        source_sha = _valid_sha(source.get("sha256"), "prepared source SHA-256")
        rows = manifest.get("blocks")
        if not isinstance(rows, list):
            raise SolutionLibraryBuildError("prepared manifest blocks must be a list")
        for index, block in enumerate(rows):
            if not isinstance(block, Mapping):
                raise SolutionLibraryBuildError("prepared block must be an object")
            block_id = _required_text(block.get("block_id"), "prepared block_id")
            key = (dataset_id, block_id)
            if key in blocks:
                raise SolutionLibraryBuildError(f"duplicate prepared block identity: {key}")
            row = dict(block)
            row["_root"] = root
            row["_dataset_id"] = dataset_id
            row["_dataset_provenance_sha256"] = manifest_provenance
            row["_manifest_path"] = manifest_path
            blocks[key] = row
        provenance.append({
            "dataset_id": dataset_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_bytes(raw),
            "provenance_sha256": manifest_provenance,
            "source_sha256": source_sha,
            "verified_block_count": len(rows),
        })
    return blocks, sorted(provenance, key=lambda row: row["dataset_id"])


def _load_reference(path: Path) -> tuple[dict[tuple[str, int], tuple[float, float]], dict[str, Any]]:
    document, raw = _load_json(path.resolve(), "frozen zlib6 reference table")
    if document.get("schema") != calibration.SCHEMA or document.get("formal_training_eligible") is not True:
        raise SolutionLibraryBuildError("solution library requires the formal zlib6 reference table")
    expected = _valid_sha(document.get("calibration_sha256"), "calibration_sha256")
    basis = {key: value for key, value in document.items() if key != "calibration_sha256"}
    if _sha256_bytes(_canonical_json(basis)) != expected:
        raise SolutionLibraryBuildError("zlib6 reference-table content fingerprint mismatch")
    references: dict[tuple[str, int], tuple[float, float]] = {}
    for row in document.get("references", ()):
        if not isinstance(row, Mapping):
            raise SolutionLibraryBuildError("invalid zlib6 reference group")
        try:
            key = (str(row["dtype"]), int(row["block_size_bucket_bytes"]))
            values = (
                float(row["E0_compression_ns_per_original_byte"]),
                float(row["D0_decompression_ns_per_original_byte"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SolutionLibraryBuildError("invalid zlib6 reference group") from error
        if key in references or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise SolutionLibraryBuildError("duplicate or invalid zlib6 reference group")
        references[key] = values
    if not references:
        raise SolutionLibraryBuildError("zlib6 reference table has no groups")
    provenance = {
        "path": str(path.resolve()),
        "file_sha256": _sha256_bytes(raw),
        "calibration_sha256": expected,
        "objective_normalization": document.get("objective_normalization"),
    }
    return references, provenance


def _cold_validate_frame(
    frame: bytes,
    *,
    values: bytes,
    bitmap: bytes,
    width: int,
) -> None:
    expected = {
        "values_sha256": _sha256_bytes(values),
        "bitmap_sha256": _sha256_bytes(bitmap),
        "width": width,
    }
    script = r'''
import hashlib, json, sys
from typed_graph_runtime import restore_frame
expected = json.loads(sys.argv[1])
roots = restore_frame(sys.stdin.buffer.read())
if list(roots) != ["s0", "s1"]:
    raise SystemExit("unexpected root order")
values, bitmap = roots["s0"], roots["s1"]
if values.kind != "numeric" or values.width != expected["width"]:
    raise SystemExit("wrong numeric root")
if bitmap.kind != "serial" or bitmap.width != 1:
    raise SystemExit("wrong bitmap root")
if hashlib.sha256(values.data).hexdigest() != expected["values_sha256"]:
    raise SystemExit("numeric root mismatch")
if hashlib.sha256(bitmap.data).hexdigest() != expected["bitmap_sha256"]:
    raise SystemExit("bitmap root mismatch")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(expected, sort_keys=True)],
        input=frame, capture_output=True, cwd=Path(__file__).resolve().parent,
        check=False, timeout=120,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise SolutionLibraryBuildError(f"independent cold decode failed: {message}")


def _validate_restored(
    frame: bytes, values: bytes, bitmap: bytes, width: int,
    recipe: tuple[int, ...],
) -> None:
    if len(frame) < 44 or frame[:4] != graph_runtime.FRAME_MAGIC:
        raise SolutionLibraryBuildError("complete frame is not a typed graph frame")
    header_size = struct.unpack_from("<Q", frame, 4)[0]
    try:
        header = json.loads(frame[12:12 + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SolutionLibraryBuildError("complete frame graph header is invalid") from error
    expected_actions = [graph_task.ACTION_CATALOG[action].runtime_action for action in recipe]
    actual_actions = [row.get("action") for row in header.get("steps", ())]
    if (header.get("registry_version") != graph_runtime.CURRENT_GRAPH_REGISTRY
            or actual_actions != expected_actions):
        raise SolutionLibraryBuildError(
            "complete frame embedded graph does not match recipe_action_ids"
        )
    try:
        roots = graph_runtime.restore_frame(frame)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise SolutionLibraryBuildError("complete frame cannot be restored") from error
    if list(roots) != ["s0", "s1"]:
        raise SolutionLibraryBuildError("complete frame has unexpected root identities/order")
    if (roots["s0"].kind != "numeric" or roots["s0"].width != width
            or roots["s0"].data != values):
        raise SolutionLibraryBuildError("complete frame does not bit-exactly restore values")
    if (roots["s1"].kind != "serial" or roots["s1"].width != 1
            or roots["s1"].data != bitmap):
        raise SolutionLibraryBuildError("complete frame does not restore the missing bitmap")
    _cold_validate_frame(frame, values=values, bitmap=bitmap, width=width)


def _measurement_digest(document: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json({
        key: value for key, value in document.items() if key != "measurement_sha256"
    }))


def _load_measurements(
    paths: Sequence[Path],
    blocks: Mapping[tuple[str, str], Mapping[str, Any]],
    references: Mapping[tuple[str, int], tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise SolutionLibraryBuildError("at least one measured-recipe receipt is required")
    retained: list[dict[str, Any]] = []
    receipt_provenance: list[dict[str, Any]] = []
    for path in paths:
        collection, raw = _load_json(path.resolve(), "measured-recipe receipt")
        if collection.get("schema") != RECEIPT_SCHEMA:
            raise SolutionLibraryBuildError("unsupported measured-recipe receipt schema")
        if collection.get("source_scope") != "training_only":
            raise SolutionLibraryBuildError("measured recipes must declare training_only scope")
        if collection.get("action_registry_fingerprint") != graph_task.REGISTRY_FINGERPRINT:
            raise SolutionLibraryBuildError("measured recipe uses a stale action registry")
        rows = collection.get("measurements")
        if not isinstance(rows, list) or not rows:
            raise SolutionLibraryBuildError("measured-recipe receipt has no measurements")
        for index, row in enumerate(rows):
            label = f"{path.name}.measurements[{index}]"
            if not isinstance(row, Mapping):
                raise SolutionLibraryBuildError(f"{label} must be an object")
            expected_digest = _valid_sha(row.get("measurement_sha256"), f"{label} digest")
            if _measurement_digest(row) != expected_digest:
                raise SolutionLibraryBuildError(f"{label} content fingerprint mismatch")
            if row.get("status") != "passed" or row.get("source_split") != "train":
                raise SolutionLibraryBuildError(
                    f"{label} is not a passed TRAIN-only measurement"
                )
            dataset_id = _required_text(row.get("dataset_id"), f"{label}.dataset_id")
            block_id = _required_text(row.get("block_id"), f"{label}.block_id")
            try:
                block = blocks[(dataset_id, block_id)]
            except KeyError as error:
                raise SolutionLibraryBuildError(
                    f"{label} does not name a verified prepared training block"
                ) from error
            if block.get("split") != "train":
                raise SolutionLibraryBuildError(f"{label} leaks a validation/test block")
            if row.get("dataset_provenance_sha256") != block.get("_dataset_provenance_sha256"):
                raise SolutionLibraryBuildError(f"{label} prepared provenance mismatch")
            dtype = _required_text(row.get("dtype"), f"{label}.dtype")
            if dtype != block.get("dtype"):
                raise SolutionLibraryBuildError(f"{label} dtype mismatch")
            root = Path(block["_root"])
            values_path = _safe_prepared_child(root, block.get("values_path"), "values_path")
            bitmap_path = _safe_prepared_child(
                root, block.get("missing_bitmap_path"), "missing_bitmap_path"
            )
            values = values_path.read_bytes()
            bitmap = bitmap_path.read_bytes()
            values_sha = _sha256_bytes(values)
            bitmap_sha = _sha256_bytes(bitmap)
            if (values_sha != _valid_sha(block.get("values_sha256"), "prepared values SHA-256")
                    or values_sha != row.get("values_sha256")):
                raise SolutionLibraryBuildError(f"{label} values fingerprint mismatch")
            if (bitmap_sha != _valid_sha(
                    block.get("missing_bitmap_sha256"), "prepared bitmap SHA-256"
                    ) or bitmap_sha != row.get("missing_bitmap_sha256")):
                raise SolutionLibraryBuildError(f"{label} bitmap fingerprint mismatch")
            try:
                recipe = factory._recipe_tuple(
                    row.get("recipe_action_ids"), f"{label}.recipe_action_ids"
                )
            except factory.ProductionTaskFactoryError as error:
                raise SolutionLibraryBuildError(str(error)) from error
            if not recipe:
                raise SolutionLibraryBuildError(
                    f"{label} is STOP-only; an actually measured operator recipe is required"
                )
            operator_graph = row.get("operator_graph")
            if (not isinstance(operator_graph, Mapping)
                    or operator_graph.get("registry_fingerprint") != graph_task.REGISTRY_FINGERPRINT
                    or operator_graph.get("typed_graph_registry") != graph_runtime.CURRENT_GRAPH_REGISTRY
                    or operator_graph.get("recipe_action_ids") != list(recipe)):
                raise SolutionLibraryBuildError(f"{label} operator graph/recipe mismatch")
            if (operator_graph.get("independent_cold_validation") is not True
                    or operator_graph.get("complete_frame_includes_all_root_flows") is not True):
                raise SolutionLibraryBuildError(
                    f"{label} lacks cold-decode/complete-frame evidence"
                )
            frame_text = _required_text(row.get("complete_frame_path"), f"{label}.frame path")
            frame_path = Path(frame_text).expanduser()
            if not frame_path.is_absolute():
                frame_path = (path.resolve().parent / frame_path).resolve()
            if not frame_path.is_file():
                raise SolutionLibraryBuildError(f"{label} complete frame is missing")
            frame = frame_path.read_bytes()
            if (_sha256_bytes(frame) != _valid_sha(
                    row.get("complete_frame_sha256"), f"{label}.frame SHA-256")
                    or len(frame) != int(row.get("complete_frame_bytes", -1))):
                raise SolutionLibraryBuildError(f"{label} complete frame hash/size mismatch")
            raw_bytes = int(block.get("original_numeric_bytes", -1))
            if raw_bytes <= 0 or raw_bytes != int(row.get("original_numeric_bytes", -1)):
                raise SolutionLibraryBuildError(f"{label} B_raw mismatch")
            compression_ns = int(row.get("compression_time_ns", -1))
            decompression_ns = int(row.get("decompression_time_ns", -1))
            if compression_ns <= 0 or decompression_ns <= 0:
                raise SolutionLibraryBuildError(f"{label} timings must be measured and positive")
            rho = len(frame) / raw_bytes
            e_value = compression_ns / raw_bytes
            d_value = decompression_ns / raw_bytes
            named_metrics = (
                ("rho_complete_bytes_per_original_numeric_byte", rho),
                ("E_compression_ns_per_original_numeric_byte", e_value),
                ("D_decompression_ns_per_original_numeric_byte", d_value),
            )
            for field, expected_value in named_metrics:
                try:
                    supplied = float(row[field])
                except (KeyError, TypeError, ValueError) as error:
                    raise SolutionLibraryBuildError(f"{label} missing metric {field}") from error
                if (not math.isfinite(supplied)
                        or not math.isclose(supplied, expected_value, rel_tol=1.0e-12, abs_tol=1.0e-12)):
                    raise SolutionLibraryBuildError(f"{label} metric {field} mismatch")
            prepared_bytes = len(values) + len(bitmap)
            bucket_bytes = calibration.block_size_bucket(prepared_bytes)
            if (dtype, bucket_bytes) not in references:
                raise SolutionLibraryBuildError(f"{label} lacks a frozen E0/D0 group")
            width = int(preparation.SUPPORTED_DTYPES[dtype][0])
            _validate_restored(frame, values, bitmap, width, recipe)
            bucket_id, vector = factory.solution_library_feature_bucket(dtype, values, bitmap)
            retained.append({
                "dataset_id": dataset_id,
                "block_id": block_id,
                "source_block_id": f"{dataset_id}:{block_id}",
                "dtype": dtype,
                "recipe_action_ids": list(recipe),
                "bucket_id": bucket_id,
                "feature_vector": list(vector),
                "rho": rho,
                "E": e_value,
                "D": d_value,
                "prepared_block_bytes": prepared_bytes,
                "reference_bucket_bytes": bucket_bytes,
                "measurement_sha256": expected_digest,
                "complete_frame_sha256": _sha256_bytes(frame),
            })
        receipt_provenance.append({
            "path": str(path.resolve()),
            "file_sha256": _sha256_bytes(raw),
            "measurement_count": len(rows),
        })
    return retained, sorted(receipt_provenance, key=lambda row: row["path"])


def _fixed_433_objective(
    measurement: Mapping[str, Any],
    references: Mapping[tuple[str, int], tuple[float, float]],
) -> float:
    e0, d0 = references[(
        str(measurement["dtype"]), int(measurement["reference_bucket_bytes"])
    )]
    rho = float(measurement["rho"])
    e_value = float(measurement["E"])
    d_value = float(measurement["D"])
    return 0.4 * rho + 0.3 * e_value / (e_value + e0) + 0.3 * d_value / (d_value + d0)


def build_library(
    prepared: Sequence[Path],
    receipts: Sequence[Path],
    reference_table: Path,
) -> dict[str, Any]:
    blocks, prepared_provenance = _load_prepared(prepared)
    references, reference_provenance = _load_reference(reference_table)
    measurements, receipt_provenance = _load_measurements(receipts, blocks, references)
    if not measurements:
        raise SolutionLibraryBuildError("no eligible measured TRAIN recipe remains")

    grouped: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for row in measurements:
        key = (str(row["dtype"]), tuple(int(action) for action in row["recipe_action_ids"]))
        grouped.setdefault(key, []).append(row)

    recipes: list[dict[str, Any]] = []
    for (dtype, action_ids), members in sorted(grouped.items()):
        recipe_id = "recipe-" + _sha256_bytes(_canonical_json({
            "dtype": dtype, "recipe_action_ids": list(action_ids),
        }))[:24]
        by_bucket: dict[str, list[dict[str, Any]]] = {}
        for member in members:
            by_bucket.setdefault(str(member["bucket_id"]), []).append(member)
        evidence = []
        for bucket_id, bucket_members in sorted(by_bucket.items()):
            vectors = {tuple(row["feature_vector"]) for row in bucket_members}
            if len(vectors) != 1:
                raise SolutionLibraryBuildError("feature bucket contains inconsistent vectors")
            evidence.append({
                "bucket_id": bucket_id,
                "feature_vector": list(next(iter(vectors))),
                "measurements": [{
                    "dataset_id": row["dataset_id"],
                    "block_id": row["source_block_id"],
                    "rho_complete_bytes_per_original_numeric_byte": row["rho"],
                    "E_compression_ns_per_original_numeric_byte": row["E"],
                    "D_decompression_ns_per_original_numeric_byte": row["D"],
                    "measurement_sha256": row["measurement_sha256"],
                    "complete_frame_sha256": row["complete_frame_sha256"],
                } for row in sorted(
                    bucket_members,
                    key=lambda item: (item["source_block_id"], item["measurement_sha256"]),
                )],
            })
        recipes.append({
            "recipe_id": recipe_id,
            "dtype": dtype,
            "recipe_action_ids": list(action_ids),
            "source_split": "train",
            "source_block_ids": sorted({row["source_block_id"] for row in members}),
            "bucket_evidence": evidence,
        })

    defaults: dict[str, str] = {}
    for dtype in sorted({recipe["dtype"] for recipe in recipes}):
        candidates = []
        for recipe in recipes:
            if recipe["dtype"] != dtype:
                continue
            recipe_members = grouped[(dtype, tuple(recipe["recipe_action_ids"]))]
            score = float(statistics.median(
                _fixed_433_objective(row, references) for row in recipe_members
            ))
            candidates.append((score, tuple(recipe["recipe_action_ids"]), recipe["recipe_id"]))
        if not candidates:
            raise SolutionLibraryBuildError(f"no eligible measured recipe for dtype {dtype!r}")
        defaults[dtype] = min(candidates)[2]

    document: dict[str, Any] = {
        "schema": LIBRARY_SCHEMA,
        "library_format": LIBRARY_FORMAT,
        "source_scope": "training_only",
        "feature_bucket_schema": factory.FEATURE_BUCKET_SCHEMA,
        "selection_contract": {
            "current_block_only": True,
            "future_blocks_visible": False,
            "validation_or_test_labels_consumed": False,
            "bucket_rule": "exact feature bucket, else minimum L1 quantised-feature distance",
            "ranking": "median exact J over measured TRAIN complete-frame receipts",
            "objective": "a*rho+b*E/(E+E0)+c*D/(D+D0)",
            "weights": "any finite nonnegative simplex weight, including the fixed six",
            "tie_break": ["J", "recipe_action_ids", "recipe_id"],
        },
        "action_registry_fingerprint": graph_task.REGISTRY_FINGERPRINT,
        "typed_graph_registry": graph_runtime.CURRENT_GRAPH_REGISTRY,
        "prepared_training_sources": prepared_provenance,
        "measurement_receipts": receipt_provenance,
        "reference_table": reference_provenance,
        "recipes": recipes,
        "default_recipe_by_dtype": defaults,
        "counts": {
            "unique_dtype_recipe_pairs": len(recipes),
            "eligible_measurements": len(measurements),
            "feature_buckets": sum(len(recipe["bucket_evidence"]) for recipe in recipes),
        },
    }
    document["library_content_sha256"] = _sha256_bytes(_canonical_json(document))
    return document


def write_immutable(path: Path, document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SolutionLibraryBuildError(
                "refusing to overwrite a different immutable solution-library manifest"
            )
        return _sha256_bytes(payload)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return _sha256_bytes(payload)


def validate_library_manifest(
    path: Path, *, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    """Validate a frozen manifest without constructing or measuring a task."""
    loaded = factory.ProductionTaskFactory(
        path,
        expected_library_sha256=expected_file_sha256,
        independent_cold_validation=False,
        require_formal_training_ready=False,
    )
    if loaded.library_format != LIBRARY_FORMAT:
        raise SolutionLibraryBuildError(
            "validator requires a measured feature-bucket solution library"
        )
    return {
        "schema": LIBRARY_SCHEMA,
        "library_format": loaded.library_format,
        "path": str(loaded.solution_library_path),
        "file_sha256": loaded.solution_library_sha256,
        "action_registry_fingerprint": loaded.action_registry_fingerprint,
        "recipe_count": len(loaded._recipes),
        "dtype_count": len(loaded._defaults),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", action="append", type=Path)
    parser.add_argument("--receipt", action="append", type=Path)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-manifest", type=Path)
    parser.add_argument("--expected-file-sha256")
    args = parser.parse_args(argv)
    if args.validate_manifest is not None:
        if any((args.prepared, args.receipt, args.reference_table, args.output)):
            parser.error("--validate-manifest cannot be combined with build inputs")
        print(json.dumps(validate_library_manifest(
            args.validate_manifest,
            expected_file_sha256=args.expected_file_sha256,
        ), sort_keys=True))
        return 0
    if not args.prepared or not args.receipt or args.reference_table is None or args.output is None:
        parser.error(
            "build mode requires --prepared, --receipt, --reference-table, and --output"
        )
    document = build_library(args.prepared, args.receipt, args.reference_table)
    file_sha = write_immutable(args.output, document)
    print(json.dumps({
        "schema": LIBRARY_SCHEMA,
        "output": str(args.output.expanduser().resolve()),
        "file_sha256": file_sha,
        "library_content_sha256": document["library_content_sha256"],
        "counts": document["counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
