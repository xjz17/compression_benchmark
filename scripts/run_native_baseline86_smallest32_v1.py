#!/usr/bin/env python3
"""Run the frozen native baseline-86 catalog on the 16 test datasets.

This is an orchestration and aggregation layer only.  Compression, complete
NGF1 framing, encode timing, and decode timing remain inside the C++
``native_baseline86_cli`` process.  The process is pinned to one Linux CPU and
all conventional thread-pool environment variables are fixed to one.

A manifest is resumable only after its complete JSONL has passed structural
and metric validation and a fingerprinted checkpoint has been atomically
written.  Partial output and stale checkpoints are never reused.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence

import native_prepared_cohort_v1 as data


SCHEMA = "native-baseline86-smallest32-campaign-v1"
CHECKPOINT_SCHEMA = "native-baseline86-manifest-checkpoint-v1"
COLUMN_SCHEMA = "native-baseline86-column-summary-v1"
FRAME_INDEX_SCHEMA = "native-baseline86-complete-frame-index-v1"
ALLOWED_STATUSES = {"ok", "N/A", "unsupported", "failed", "invalid_input"}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "BLOSC_NTHREADS": "1",
}


class CampaignError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
    )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_cpus(value: str) -> tuple[int, ...]:
    try:
        cpus = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise CampaignError("--cpus must be a comma-separated integer list") from error
    require(cpus and all(cpu >= 0 for cpu in cpus), "CPU list must be nonempty and nonnegative")
    require(len(cpus) == len(set(cpus)), "CPU list contains duplicates")
    return cpus


def bind_cpu_pool(cpus: Sequence[int]) -> None:
    require(sys.platform.startswith("linux"), "formal timing requires Linux CPU affinity")
    require(hasattr(os, "sched_setaffinity"), "os.sched_setaffinity is unavailable")
    try:
        os.sched_setaffinity(0, set(cpus))
    except OSError as error:
        raise CampaignError(f"cannot bind campaign to CPUs {list(cpus)}: {error}") from error
    require(os.sched_getaffinity(0) == set(cpus), "requested CPU affinity did not take effect")


def pin_child(cpu: int):
    def apply() -> None:
        os.sched_setaffinity(0, {cpu})
    return apply


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(THREAD_ENV)
    return environment


def read_inventory(cli: Path, *, cpu: int) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        [str(cli), "inventory"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=child_environment(), check=False,
        preexec_fn=pin_child(cpu),
    )
    require(completed.returncode == 0, f"inventory failed: {completed.stderr.strip()}")
    try:
        inventory = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise CampaignError("inventory is not valid JSON") from error
    methods = inventory.get("methods")
    require(inventory.get("schema") == "native-baseline86-inventory-v1", "wrong inventory schema")
    require(inventory.get("catalog_labels") == 86 and isinstance(methods, list) and len(methods) == 86,
            "inventory must retain exactly 86 labels")
    labels = [row.get("label") for row in methods]
    require(all(isinstance(x, str) and x for x in labels) and len(set(labels)) == 86,
            "inventory labels are missing or duplicated")
    linked = sum(row.get("linked") is True for row in methods)
    primary = sum(row.get("linked") is True and row.get("primary_comparison") is True for row in methods)
    unsupported = sum(row.get("linked") is False for row in methods)
    proxy = sum(row.get("linked") is True and row.get("primary_comparison") is False for row in methods)
    require((linked, primary, unsupported, proxy) ==
            (inventory.get("linked"), inventory.get("primary_comparison"),
             inventory.get("unsupported"), inventory.get("proxy_or_alias")),
            "inventory summary does not match method records")
    return inventory, sha256_bytes(canonical(inventory))


def selected_manifests(corpus: data.Corpus) -> list[dict[str, Any]]:
    require(corpus.role == "test_candidate", "only test_candidate corpus is allowed")
    grouped: dict[Path, list[data.Block]] = collections.defaultdict(list)
    for block in corpus.split("test"):
        grouped[block.manifest_path.resolve()].append(block)
    require(len(grouped) == 16, f"expected 16 test manifests, found {len(grouped)}")
    result = []
    for path, blocks in grouped.items():
        datasets = {block.dataset_id for block in blocks}
        require(len(datasets) == 1, "one manifest resolved to multiple dataset IDs")
        # Recheck every prepared payload before the manifest is admitted to the
        # C++ process.  The C++ process independently verifies the same hashes.
        for block in blocks:
            block.payload()
        manifest = json.loads(path.read_bytes())
        require(manifest.get("cohort", {}).get("role") == "test_candidate",
                "manifest role drift")
        require({row.get("split") for row in manifest.get("blocks", [])} == {"test"},
                "test manifest contains another split")
        result.append({
            "dataset_id": next(iter(datasets)),
            "domain": blocks[0].domain,
            "path": path,
            "sha256": sha256_file(path),
            "provenance_sha256": blocks[0].provenance_sha256,
            "block_count": len(blocks),
            "original_numeric_bytes": sum(block.record["original_numeric_bytes"] for block in blocks),
            "blocks": {block.record["block_id"]: block.record for block in blocks},
        })
    result.sort(key=lambda row: row["dataset_id"])
    require(len({row["dataset_id"] for row in result}) == 16, "duplicate test dataset ID")
    return result


def identity_for_manifest(
    manifest: dict[str, Any], *, corpus_fingerprint: str, cli_sha256: str,
    inventory_sha256: str, runner_sha256: str, warmups: int, repeats: int, cpu: int,
    frame_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset_id": manifest["dataset_id"],
        "manifest_path": str(manifest["path"]),
        "manifest_sha256": manifest["sha256"],
        "provenance_sha256": manifest["provenance_sha256"],
        "corpus_fingerprint": corpus_fingerprint,
        "cli_sha256": cli_sha256,
        "inventory_sha256": inventory_sha256,
        "runner_sha256": runner_sha256,
        "block_count": manifest["block_count"],
        "warmups": warmups,
        "repeats": repeats,
        "cpu": cpu,
        "frame_dir": str(frame_dir.resolve()),
        "frame_storage": "content_addressed_ngf1_after_timing",
        "thread_environment": THREAD_ENV,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"invalid JSONL at {path}:{line_number}") from error
            require(isinstance(value, dict), f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def verify_complete_frame(row: dict[str, Any], frame_dir: Path) -> dict[str, Any]:
    digest = row.get("complete_frame_sha256")
    relative = row.get("complete_frame_relative_path")
    size = row.get("complete_frame_bytes")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(character in "0123456789abcdef" for character in digest),
            "complete frame SHA-256 is missing or malformed")
    expected = f"{digest[:2]}/{digest}.ngf1"
    require(relative == expected and row.get("complete_frame_storage") == "content_addressed_ngf1",
            "complete frame is not stored at its content-addressed relative path")
    root = frame_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CampaignError("complete frame path escapes frame directory") from error
    require(path.is_file(), f"complete frame is missing: {relative}")
    require(type(size) is int and size >= 0 and path.stat().st_size == size,
            f"complete frame size differs: {relative}")
    require(sha256_file(path) == digest, f"complete frame SHA-256 differs: {relative}")
    return {"relative_path": relative, "sha256": digest, "bytes": size}


def validate_manifest_rows(
    rows: Sequence[dict[str, Any]], manifest: dict[str, Any],
    inventory: dict[str, Any], *, repeats: int, frame_dir: Path,
) -> dict[str, int]:
    methods = {row["label"]: row for row in inventory["methods"]}
    blocks = manifest["blocks"]
    expected = {(block_id, method) for block_id in blocks for method in methods}
    require(len(rows) == len(expected),
            f"{manifest['dataset_id']}: expected {len(expected)} rows, got {len(rows)}")
    seen: set[tuple[str, str]] = set()
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        require(row.get("schema") == "native-baseline86-result-v1", "wrong result schema")
        key = (row.get("block_id"), row.get("method"))
        require(key in expected and key not in seen, f"unexpected or duplicate result pair {key}")
        seen.add(key)
        method = methods[key[1]]
        block = blocks[key[0]]
        require(row.get("dataset_id") == manifest["dataset_id"], "result dataset drift")
        require(Path(row.get("manifest", "")).resolve() == manifest["path"], "result manifest drift")
        require(row.get("dtype") == block["dtype"] and row.get("row_count") == block["row_count"],
                "result block shape drift")
        require(row.get("implementation_class") == method["implementation_class"] and
                row.get("primary_comparison") is method["primary_comparison"] and
                row.get("configuration") == method["configuration"],
                "result implementation identity drift")
        status = row.get("status")
        require(status in ALLOWED_STATUSES, f"unknown result status {status!r}")
        counts[status] += 1
        if status != "invalid_input":
            require(row.get("original_numeric_bytes") == block["original_numeric_bytes"],
                    "result B_raw differs from prepared block")
        if status in {"ok", "N/A"}:
            b_raw = block["original_numeric_bytes"]
            require((status == "N/A") == (b_raw == 0), "N/A must mean zero B_raw")
            require(isinstance(row.get("complete_frame_bytes"), int) and
                    row["complete_frame_bytes"] >= 0, "missing complete frame size")
            for sample_name in ("compression_ns_samples", "decompression_ns_samples"):
                samples = row.get(sample_name)
                require(isinstance(samples, list) and len(samples) == repeats and
                        all(type(x) is int and x >= 0 for x in samples),
                        f"{sample_name} does not contain {repeats} timings")
            require(isinstance(row.get("compression_ns"), (int, float)) and
                    isinstance(row.get("decompression_ns"), (int, float)),
                    "block median timing is missing")
            verify_complete_frame(row, frame_dir)
            if status == "ok":
                require(b_raw > 0 and close(row["compression_ratio"],
                        row["complete_frame_bytes"] / b_raw), "block ratio mismatch")
                require(close(row["compression_ns_per_original_numeric_byte"],
                              row["compression_ns"] / b_raw), "block compression timing mismatch")
                require(close(row["decompression_ns_per_original_numeric_byte"],
                              row["decompression_ns"] / b_raw), "block decompression timing mismatch")
            else:
                for name in ("compression_ratio", "compression_ns_per_original_numeric_byte",
                             "decompression_ns_per_original_numeric_byte"):
                    require(row.get(name) is None, f"N/A row has fabricated {name}")
        else:
            require(isinstance(row.get("reason"), str) and row["reason"],
                    f"{status} row lacks a reason")
    require(seen == expected, "manifest result matrix is incomplete")
    return {status: counts.get(status, 0) for status in sorted(ALLOWED_STATUSES)}


def checkpoint_reusable(
    checkpoint_path: Path, raw_path: Path, identity: dict[str, Any],
    manifest: dict[str, Any], inventory: dict[str, Any], *, repeats: int,
    frame_dir: Path,
) -> tuple[bool, list[dict[str, Any]] | None]:
    if not checkpoint_path.is_file() or not raw_path.is_file():
        return False, None
    try:
        checkpoint = json.loads(checkpoint_path.read_bytes())
        if (checkpoint.get("schema") != CHECKPOINT_SCHEMA or
                checkpoint.get("status") != "verified" or
                checkpoint.get("identity") != identity or
                checkpoint.get("raw_jsonl_sha256") != sha256_file(raw_path)):
            return False, None
        rows = load_jsonl(raw_path)
        counts = validate_manifest_rows(
            rows, manifest, inventory, repeats=repeats, frame_dir=frame_dir)
        if checkpoint.get("status_counts") != counts:
            return False, None
        frames = sorted(
            {tuple(verify_complete_frame(row, frame_dir).items())
             for row in rows if row.get("status") in {"ok", "N/A"}}
        )
        if checkpoint.get("complete_frames") != [dict(value) for value in frames]:
            return False, None
        return True, rows
    except (OSError, json.JSONDecodeError, CampaignError, KeyError, TypeError):
        return False, None


def run_one_manifest(
    cli: Path, manifest: dict[str, Any], raw_path: Path, checkpoint_path: Path,
    identity: dict[str, Any], inventory: dict[str, Any], *, warmups: int,
    repeats: int, timeout_seconds: int, cpu: int, frame_dir: Path,
) -> tuple[list[dict[str, Any]], bool]:
    reusable, rows = checkpoint_reusable(
        checkpoint_path, raw_path, identity, manifest, inventory, repeats=repeats,
        frame_dir=frame_dir,
    )
    if reusable:
        return rows or [], True
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{manifest['dataset_id']}.", suffix=".jsonl.partial",
        dir=raw_path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
    command = [str(cli), "benchmark-manifest", str(manifest["path"]),
               str(temporary), str(warmups), str(repeats),
               "--frame-dir", str(frame_dir.resolve())]
    started = utc_now()
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=child_environment(), timeout=(timeout_seconds or None), check=False,
            preexec_fn=pin_child(cpu),
        )
    except subprocess.TimeoutExpired as error:
        raise CampaignError(f"{manifest['dataset_id']}: native CLI timeout") from error
    if completed.returncode != 0:
        raise CampaignError(
            f"{manifest['dataset_id']}: native CLI exited {completed.returncode}: "
            f"{completed.stderr.strip()} (partial output: {temporary})"
        )
    try:
        rows = load_jsonl(temporary)
        counts = validate_manifest_rows(
            rows, manifest, inventory, repeats=repeats, frame_dir=frame_dir)
    except Exception:
        raise CampaignError(
            f"{manifest['dataset_id']}: result verification failed; partial output retained at {temporary}"
        )
    temporary.replace(raw_path)
    frames = sorted(
        {tuple(verify_complete_frame(row, frame_dir).items())
         for row in rows if row.get("status") in {"ok", "N/A"}}
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "verified",
        "started_at": started,
        "completed_at": utc_now(),
        "identity": identity,
        "raw_jsonl": str(raw_path),
        "raw_jsonl_sha256": sha256_file(raw_path),
        "rows": len(rows),
        "status_counts": counts,
        "complete_frames": [dict(value) for value in frames],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    atomic_json(checkpoint_path, checkpoint)
    return rows, False


def column_status(statuses: collections.Counter[str]) -> str:
    present = {name for name, count in statuses.items() if count}
    if present == {"ok"}:
        return "ok"
    if present == {"N/A"}:
        return "N/A"
    if present == {"unsupported"}:
        return "unsupported"
    if "failed" in present or "invalid_input" in present:
        return "failed" if len(present) == 1 else "partial_failed"
    if "N/A" in present:
        return "partial_N/A"
    if "unsupported" in present:
        return "partial_unsupported"
    raise CampaignError(f"cannot summarize statuses {sorted(present)}")


def summarize_columns(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = (row.get("dataset_id"), row.get("domain"), row.get("column"),
               row.get("source_column"), row.get("dtype"), row.get("method"))
        grouped[key].append(row)
    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        statuses = collections.Counter(row["status"] for row in group)
        status = column_status(statuses)
        ok_rows = [row for row in group if row["status"] == "ok"]
        framed_rows = [row for row in group if row["status"] in {"ok", "N/A"}]
        b_raw_expected = sum(
            (row["original_numeric_bytes"] if isinstance(row.get("original_numeric_bytes"), int)
             else row.get("declared_original_numeric_bytes", 0)) for row in group
        )
        observed_b_raw = sum(row["original_numeric_bytes"] for row in framed_rows)
        observed_frame_bytes = sum(row["complete_frame_bytes"] for row in framed_rows)
        observed_compression_ns = sum(row["compression_ns"] for row in framed_rows)
        observed_decompression_ns = sum(row["decompression_ns"] for row in framed_rows)
        admitted = status == "ok" and b_raw_expected > 0
        implementation_rows = {(row.get("implementation_class"), row.get("implementation"),
                                row.get("configuration"), row.get("primary_comparison")) for row in group}
        require(len(implementation_rows) == 1, "one method changed implementation within a column")
        implementation_class, implementation, configuration, primary = next(iter(implementation_rows))
        summary = {
            "schema": COLUMN_SCHEMA,
            "dataset_id": key[0], "domain": key[1], "column": key[2],
            "source_column": key[3], "dtype": key[4], "method": key[5],
            "implementation_class": implementation_class,
            "implementation": implementation,
            "configuration": configuration,
            "primary_comparison": primary,
            "status": status,
            "main_metrics_admitted": admitted,
            "block_count": len(group),
            "entity_count": len({row.get("entity_id") for row in group}),
            "status_counts": {name: statuses.get(name, 0) for name in sorted(ALLOWED_STATUSES)},
            # Stable compatibility fields consumed by the final comparison join.
            "original_numeric_bytes": b_raw_expected,
            "frame_bytes": observed_frame_bytes if admitted else None,
            "compression_time_ns": observed_compression_ns if admitted else None,
            "decompression_time_ns": observed_decompression_ns if admitted else None,
            "compression_ns_per_byte": observed_compression_ns / b_raw_expected if admitted else None,
            "decompression_ns_per_byte": observed_decompression_ns / b_raw_expected if admitted else None,
            # Explicitly named contract fields retained for standalone audit.
            "original_numeric_bytes_sum": b_raw_expected,
            "complete_frame_bytes_sum": observed_frame_bytes if admitted else None,
            "compression_ns_sum_of_block_medians": observed_compression_ns if admitted else None,
            "decompression_ns_sum_of_block_medians": observed_decompression_ns if admitted else None,
            "compression_ratio": observed_frame_bytes / b_raw_expected if admitted else None,
            "compression_ns_per_original_numeric_byte": observed_compression_ns / b_raw_expected if admitted else None,
            "decompression_ns_per_original_numeric_byte": observed_decompression_ns / b_raw_expected if admitted else None,
            "diagnostic_observed_B_raw": observed_b_raw,
            "diagnostic_observed_complete_frame_bytes": observed_frame_bytes,
            "failure_reasons": sorted({row.get("reason") for row in group if row.get("reason")}),
            "aggregation": (
                "sum complete frame bytes and sum each block's median nanoseconds, divided by "
                "sum original valid numeric bytes; main metrics only when every block is status=ok"
            ),
        }
        summaries.append(summary)
    return summaries


CSV_FIELDS = [
    "dataset_id", "domain", "column", "source_column", "dtype", "method",
    "implementation_class", "implementation", "configuration", "primary_comparison",
    "status", "main_metrics_admitted", "block_count", "entity_count",
    "original_numeric_bytes", "frame_bytes", "compression_time_ns",
    "decompression_time_ns", "compression_ratio", "compression_ns_per_byte",
    "decompression_ns_per_byte", "original_numeric_bytes_sum", "complete_frame_bytes_sum",
    "compression_ns_sum_of_block_medians", "decompression_ns_sum_of_block_medians",
    "compression_ns_per_original_numeric_byte", "decompression_ns_per_original_numeric_byte",
    "diagnostic_observed_B_raw",
    "diagnostic_observed_complete_frame_bytes", "status_counts", "failure_reasons",
]


def write_column_outputs(output: Path, summaries: Sequence[dict[str, Any]]) -> None:
    jsonl = b"".join(canonical(row) + b"\n" for row in summaries)
    atomic_bytes(output / "columns.jsonl", jsonl)
    atomic_json(output / "columns.json", {"schema": COLUMN_SCHEMA, "columns": list(summaries)})
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for value in summaries:
            row = dict(value)
            row["status_counts"] = json.dumps(row["status_counts"], sort_keys=True)
            row["failure_reasons"] = json.dumps(row["failure_reasons"], ensure_ascii=False)
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / "columns.csv")


def write_frame_index(output: Path, frame_dir: Path,
                      rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") not in {"ok", "N/A"}:
            continue
        record = verify_complete_frame(row, frame_dir)
        previous = unique.setdefault(record["relative_path"], record)
        require(previous == record, "one content-addressed frame path changed identity")
    frames = [
        {"path": str(Path("frames") / value["relative_path"]),
         "sha256": value["sha256"], "bytes": value["bytes"]}
        for value in sorted(unique.values(), key=lambda item: item["relative_path"])
    ]
    document = {
        "schema": FRAME_INDEX_SCHEMA,
        "storage": "content_addressed_ngf1_after_timing",
        "frame_root": "frames",
        "frame_count": len(frames),
        "frame_bytes": sum(row["bytes"] for row in frames),
        "frames": frames,
    }
    atomic_json(output / "frames.json", document)
    return document


def preflight(
    *, cohort_path: Path, materialized: Path, cli: Path, cpus: Sequence[int],
    warmups: int, repeats: int,
) -> tuple[data.Corpus, list[dict[str, Any]], dict[str, Any], str, str]:
    require(warmups == 1, "formal baseline protocol requires warmups=1")
    require(repeats == 5, "formal baseline protocol requires repeats=5")
    require(cli.is_file() and os.access(cli, os.X_OK), "baseline CLI is not executable")
    bind_cpu_pool(cpus)
    corpus = data.load(cohort_path, materialized, role="test_candidate")
    manifests = selected_manifests(corpus)
    inventory, inventory_sha = read_inventory(cli, cpu=cpus[0])
    cli_sha = sha256_file(cli)
    return corpus, manifests, inventory, inventory_sha, cli_sha


def campaign(arguments: argparse.Namespace) -> dict[str, Any]:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cpus = parse_cpus(arguments.cpus)
    corpus, manifests, inventory, inventory_sha, cli_sha = preflight(
        cohort_path=arguments.cohort.resolve(), materialized=arguments.materialized.resolve(),
        cli=arguments.cli.resolve(), cpus=cpus,
        warmups=arguments.warmups, repeats=arguments.repeats,
    )
    inventory_path = output / "inventory.json"
    atomic_json(inventory_path, inventory)
    runner_sha = sha256_file(Path(__file__).resolve())
    manifest_cpus = {row["dataset_id"]: cpus[index % len(cpus)]
                     for index, row in enumerate(manifests)}
    identities = {
        manifest["dataset_id"]: identity_for_manifest(
            manifest, corpus_fingerprint=corpus.fingerprint, cli_sha256=cli_sha,
            inventory_sha256=inventory_sha, runner_sha256=runner_sha,
            warmups=arguments.warmups,
            repeats=arguments.repeats, cpu=manifest_cpus[manifest["dataset_id"]],
            frame_dir=frame_dir,
        ) for manifest in manifests
    }
    plan = {
        "schema": SCHEMA,
        "mode": "preflight" if arguments.preflight_only else "execute",
        "created_at": utc_now(),
        "corpus": corpus.summary(),
        "cli": str(arguments.cli.resolve()),
        "cli_sha256": cli_sha,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": runner_sha,
        "inventory_sha256": inventory_sha,
        "cpus": list(cpus),
        "execution": "manifests run serially; each C++ child is bound to its assigned single CPU",
        "thread_environment": THREAD_ENV,
        "warmups": arguments.warmups,
        "repeats": arguments.repeats,
        "frame_storage": {"root": str(frame_dir.resolve()),
                          "format": "content_addressed_ngf1_after_timing"},
        "manifest_count": len(manifests),
        "manifests": [identities[row["dataset_id"]] for row in manifests],
    }
    atomic_json(output / "plan.json", plan)
    if arguments.preflight_only:
        return plan

    all_rows: list[dict[str, Any]] = []
    progress = []
    for index, manifest in enumerate(manifests, 1):
        dataset_id = manifest["dataset_id"]
        cpu = manifest_cpus[dataset_id]
        raw_path = output / "raw" / f"{dataset_id}.jsonl"
        checkpoint_path = output / "checkpoints" / f"{dataset_id}.json"
        print(json.dumps({"event": "baseline86_manifest_start", "index": index,
                          "total": len(manifests), "dataset_id": dataset_id,
                          "cpu": cpu}), flush=True)
        rows, resumed = run_one_manifest(
            arguments.cli.resolve(), manifest, raw_path, checkpoint_path,
            identities[dataset_id], inventory, warmups=arguments.warmups,
            repeats=arguments.repeats,
            timeout_seconds=arguments.manifest_timeout_seconds, cpu=cpu,
            frame_dir=frame_dir,
        )
        all_rows.extend(rows)
        item = {"dataset_id": dataset_id, "status": "verified", "resumed": resumed,
                "raw_jsonl": str(raw_path), "raw_jsonl_sha256": sha256_file(raw_path),
                "checkpoint": str(checkpoint_path), "rows": len(rows), "cpu": cpu}
        progress.append(item)
        atomic_json(output / "run_state.json", {
            "schema": SCHEMA, "status": "running", "updated_at": utc_now(),
            "completed_manifests": progress,
        })
        print(json.dumps({"event": "baseline86_manifest_verified", **item}), flush=True)

    summaries = summarize_columns(all_rows)
    write_column_outputs(output, summaries)
    frame_index = write_frame_index(output, frame_dir, all_rows)
    raw_counts = collections.Counter(row["status"] for row in all_rows)
    column_counts = collections.Counter(row["status"] for row in summaries)
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "completed_at": utc_now(),
        "claim_boundary": (
            "The receipt covers the 16 freshly verified test_candidate manifests and all 86 "
            "labels. Only primary_comparison=true, status=ok column rows are eligible for the "
            "same-label primary comparison. Unsupported, proxy/alias, N/A, and failed rows remain explicit."
        ),
        "timing_contract": {
            "implementation": "C++ in-process complete NGF1 encode/decode",
            "available_cpu_affinity": list(cpus), "per_process_cpu_affinity": 1,
            "manifest_execution": "serial", "threads": 1,
            "warmups": arguments.warmups, "repeats": arguments.repeats,
            "block_aggregate": "median elapsed ns",
            "column_aggregate": "sum block medians / sum original numeric bytes",
            "source_file_io_timed": False,
            "complete_frame_file_io_timed": False,
        },
        "metric_contract": {
            "compression_ratio": "sum complete frame bytes / sum original valid numeric bytes",
            "compression_time": "sum per-block median compression ns / sum original valid numeric bytes",
            "decompression_time": "sum per-block median decompression ns / sum original valid numeric bytes",
            "direction": "all lower is better",
            "partial_column_policy": "no main metrics unless every block is status=ok",
        },
        "corpus": corpus.summary(),
        "cli": str(arguments.cli.resolve()), "cli_sha256": cli_sha,
        "runner": str(Path(__file__).resolve()), "runner_sha256": runner_sha,
        "inventory": str(inventory_path), "inventory_sha256": inventory_sha,
        "manifest_count": len(manifests), "manifest_receipts": progress,
        "raw_row_count": len(all_rows),
        "raw_status_counts": {name: raw_counts.get(name, 0) for name in sorted(ALLOWED_STATUSES)},
        "column_row_count": len(summaries),
        "column_status_counts": dict(sorted(column_counts.items())),
        "complete_frame_count": frame_index["frame_count"],
        "complete_frame_bytes_on_disk": frame_index["frame_bytes"],
        "outputs": {
            "columns_jsonl": {"path": str(output / "columns.jsonl"),
                              "sha256": sha256_file(output / "columns.jsonl")},
            "columns_json": {"path": str(output / "columns.json"),
                             "sha256": sha256_file(output / "columns.json")},
            "columns_csv": {"path": str(output / "columns.csv"),
                             "sha256": sha256_file(output / "columns.csv")},
            "frames_json": {"path": str(output / "frames.json"),
                            "sha256": sha256_file(output / "frames.json")},
        },
    }
    atomic_json(output / "receipt.json", receipt)
    atomic_json(output / "run_state.json", {
        "schema": SCHEMA, "status": "complete", "updated_at": utc_now(),
        "completed_manifests": progress, "receipt": str(output / "receipt.json"),
    })
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cohort", required=True, type=Path)
    result.add_argument("--materialized", required=True, type=Path)
    result.add_argument("--cli", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--cpus", required=True,
                        help="comma-separated allowed CPUs; manifests run serially and each child uses one CPU")
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--manifest-timeout-seconds", type=int, default=0,
                        help="0 disables the per-manifest timeout")
    result.add_argument("--preflight-only", action="store_true",
                        help="validate the test corpus/runtime and write plan.json without compression")
    return result


def main() -> None:
    arguments = parser().parse_args()
    require(arguments.manifest_timeout_seconds >= 0, "timeout must be nonnegative")
    result = campaign(arguments)
    print(json.dumps({"event": "baseline86_campaign_complete",
                      "mode": result.get("mode", "execute"),
                      "status": result.get("status", "preflight"),
                      "output": str(arguments.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
