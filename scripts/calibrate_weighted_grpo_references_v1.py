#!/usr/bin/env python3
"""Freeze zlib6 E0/D0 references from prepared training blocks.

The C++ helper measures complete ZR61 frame encode/decode in-process.  Python
only verifies prepared-data receipts, supplies ``values || missing-bitmap`` to
the helper, groups observations by original dtype and power-of-two block-size
bucket, and writes the immutable median table used by the weighted objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import prepare_weighted_grpo_timeseries_v1 as preparation


SCHEMA = "weighted-grpo-zlib6-reference-table-v1"
SMOKE_SCHEMA = "weighted-grpo-zlib6-reference-smoke-v1"
BENCHMARK_SCHEMA = "zlib6-reference-benchmark-v1"


class CalibrationError(RuntimeError):
    """A prepared block or native reference violated the calibration contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def block_size_bucket(byte_count: int) -> int:
    """Return the next power of two, capped by the prepared 1 MiB contract."""
    if isinstance(byte_count, bool) or not 1 <= byte_count <= preparation.MAX_BLOCK_BYTES:
        raise ValueError("prepared block byte count must be in 1..1MiB")
    return 1 << (byte_count - 1).bit_length()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CalibrationError(f"prepared path leaves its dataset directory: {relative}") from error
    if not candidate.is_file():
        raise CalibrationError(f"missing prepared block payload: {candidate}")
    return candidate


def _run_native(
    cli: Path, payload: bytes, *, warmups: int, repeats: int
) -> dict[str, Any]:
    if not cli.is_file():
        raise CalibrationError(f"missing native zlib6 reference CLI: {cli}")
    result = subprocess.run(
        [str(cli), "benchmark", "--warmups", str(warmups), "--repeats", str(repeats)],
        input=payload, capture_output=True, check=False, timeout=120,
    )
    if result.returncode:
        raise CalibrationError(
            "native zlib6 reference failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CalibrationError("native zlib6 reference returned invalid JSON") from error
    if record.get("schema") != BENCHMARK_SCHEMA:
        raise CalibrationError("native zlib6 reference schema mismatch")
    if (record.get("level") != 6 or record.get("raw_bytes") != len(payload)
            or record.get("warmups") != warmups or record.get("repeats") != repeats):
        raise CalibrationError("native zlib6 reference receipt mismatch")
    for field in ("compression_median_ns", "decompression_median_ns"):
        if isinstance(record.get(field), bool) or int(record.get(field, -1)) < 0:
            raise CalibrationError(f"invalid native timing field: {field}")
    return record


def _load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    root = path.resolve() if path.is_dir() else path.resolve().parent
    manifest_path = root / "dataset_manifest.json" if path.is_dir() else path.resolve()
    verification = preparation.verify_prepared_dataset(root, verify_source=True)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if verification["dataset_id"] != manifest.get("dataset_id"):
        raise CalibrationError("prepared verification dataset identity mismatch")
    return root, manifest


def calibrate(
    prepared: Sequence[Path],
    *,
    cli: Path,
    warmups: int = 3,
    repeats: int = 15,
    require_complete_cohort: bool = True,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not prepared:
        raise ValueError("at least one prepared training dataset is required")
    if warmups < 1 or repeats < 1 or repeats % 2 == 0:
        raise ValueError("warmups must be positive and repeats must be positive odd")
    started = time.monotonic()
    loaded = []
    for path in prepared:
        if progress is not None:
            progress({"event": "verify_prepared_source", "path": str(path)})
        loaded.append(_load_manifest(path))
    preparation.validate_prepared_cohort_manifests(
        [manifest for _root, manifest in loaded],
        preparation.load_cohort_manifest(),
        require_complete_train=require_complete_cohort,
    )
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    total_blocks = sum(
        block["split"] == "train" and int(block["original_numeric_bytes"]) > 0
        for _root, manifest in loaded for block in manifest["blocks"]
    )
    measured_bytes = 0
    last_progress = time.monotonic()
    if progress is not None:
        progress({"event": "calibration_start", "training_blocks": total_blocks,
                  "datasets": len(loaded), "formal_training_eligible": require_complete_cohort})
    for root, manifest in loaded:
        for block in manifest["blocks"]:
            if block["split"] != "train" or int(block["original_numeric_bytes"]) <= 0:
                continue
            values_path = _safe_child(root, str(block["values_path"]))
            bitmap_path = _safe_child(root, str(block["missing_bitmap_path"]))
            values = values_path.read_bytes()
            bitmap = bitmap_path.read_bytes()
            if _sha256(values) != block["values_sha256"]:
                raise CalibrationError(f"value payload hash drift: {block['block_id']}")
            if _sha256(bitmap) != block["missing_bitmap_sha256"]:
                raise CalibrationError(f"missing bitmap hash drift: {block['block_id']}")
            payload = values + bitmap
            receipt = _run_native(cli, payload, warmups=warmups, repeats=repeats)
            denominator = int(block["original_numeric_bytes"])
            row = {
                "dataset_id": manifest["dataset_id"],
                "provenance_sha256": manifest["provenance_sha256"],
                "block_id": block["block_id"],
                "dtype": block["dtype"],
                "prepared_block_bytes": len(payload),
                "block_size_bucket_bytes": block_size_bucket(len(payload)),
                "original_numeric_bytes": denominator,
                "calibration_payload_sha256": _sha256(payload),
                "compression_ns_per_original_byte": (
                    int(receipt["compression_median_ns"]) / denominator
                ),
                "decompression_ns_per_original_byte": (
                    int(receipt["decompression_median_ns"]) / denominator
                ),
                "complete_frame_bytes": int(receipt["complete_frame_bytes"]),
                "native_receipt": receipt,
            }
            rows.append(row)
            measured_bytes += denominator
            groups.setdefault(
                (str(block["dtype"]), row["block_size_bucket_bytes"]), []
            ).append(row)
            current = time.monotonic()
            if progress is not None and (
                len(rows) % 1000 == 0 or current - last_progress >= 30
                or len(rows) == total_blocks
            ):
                progress({"event": "calibration_progress", "measured_blocks": len(rows),
                          "training_blocks": total_blocks, "dataset_id": manifest["dataset_id"],
                          "original_numeric_bytes": measured_bytes,
                          "elapsed_seconds": current - started})
                last_progress = current
    if not rows:
        raise CalibrationError("no non-empty training blocks were available for calibration")
    references = []
    for (dtype, bucket), members in sorted(groups.items()):
        encode_values = [row["compression_ns_per_original_byte"] for row in members]
        decode_values = [row["decompression_ns_per_original_byte"] for row in members]
        e0 = float(statistics.median(encode_values))
        d0 = float(statistics.median(decode_values))
        if not all(math.isfinite(value) and value > 0.0 for value in (e0, d0)):
            raise CalibrationError("calibration references must be finite and positive")
        references.append({
            "dtype": dtype,
            "block_size_bucket_bytes": bucket,
            "E0_compression_ns_per_original_byte": e0,
            "D0_decompression_ns_per_original_byte": d0,
            "training_block_count": len(members),
            "block_ids_sha256": _sha256(_canonical_json(sorted(
                (row["dataset_id"], row["block_id"]) for row in members
            ))),
        })
    cli_version = subprocess.run(
        [str(cli), "version"], input=b"", capture_output=True, check=False, timeout=120,
    )
    if cli_version.returncode:
        raise CalibrationError("cannot read native zlib6 reference version")
    document = {
        "schema": SCHEMA if require_complete_cohort else SMOKE_SCHEMA,
        "formal_training_eligible": require_complete_cohort,
        "calibration_scope": (
            "complete_frozen_16_source_train_cohort"
            if require_complete_cohort else "non_formal_smoke_subset"
        ),
        "objective_normalization": "E/(E+E0), D/(D+D0)",
        "reference_method": "native zlib level 6 complete ZR61 frame",
        "reference_role": "time scale only; operator recipes are not required to use zlib6",
        "grouping": "original dtype and next-power-of-two prepared block bytes up to 1MiB",
        "warmups": warmups,
        "repeats": repeats,
        "native_cli": str(cli.resolve()),
        "native_cli_version": cli_version.stdout.decode("utf-8", errors="strict").strip(),
        "prepared_dataset_count": len(loaded),
        "training_block_count": len(rows),
        "references": references,
        "blocks": rows,
    }
    document["calibration_sha256"] = _sha256(_canonical_json(document))
    return document


def write_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", action="append", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument(
        "--smoke-subset", action="store_true",
        help=(
            "allow an incomplete frozen-cohort subset and emit a non-formal "
            "schema that formal training/experiment loaders reject"
        ),
    )
    args = parser.parse_args(argv)
    document = calibrate(
        args.prepared, cli=args.cli, warmups=args.warmups, repeats=args.repeats,
        require_complete_cohort=not args.smoke_subset,
        progress=lambda event: print(json.dumps(event, sort_keys=True), flush=True),
    )
    write_atomic(args.output, document)
    print(json.dumps({
        "schema": document["schema"],
        "calibration_sha256": document["calibration_sha256"],
        "training_block_count": document["training_block_count"],
        "reference_group_count": len(document["references"]),
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
