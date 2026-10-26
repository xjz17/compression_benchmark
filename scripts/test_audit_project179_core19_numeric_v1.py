#!/usr/bin/env python3
"""Fail-closed tests for the independent Core19 cold auditor."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import audit_project179_core19_numeric_v1 as auditor_module
from audit_project179_core19_numeric_v1 import (
    AuditError,
    audit,
    file_binding,
    main,
    parse_timestamp_us,
)
from prepare_project179_core19_numeric_v1 import prepare_scope


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_us(text: str) -> int:
    value = datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = value - epoch
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


class BackupZstdDescriptorTest(unittest.TestCase):
    def make_executable(self, root: Path, name: str, source: str) -> Path:
        path = root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o700)
        return path

    def test_zstd_receives_same_open_descriptor_on_stdin_without_dev_fd_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backup = root / "backup.csv.zst"
            original = b"event_time,value\n2026-01-02 03:04:05,7\n"
            backup.write_bytes(original)
            zstd = self.make_executable(
                root,
                "zstd-reject-dev-fd",
                "#!/bin/sh\n"
                "if [ \"$#\" -ne 1 ] || [ \"$1\" != -dc ]; then\n"
                "  echo 'Warning : /dev/fd/3 is a symbolic link, ignoring' >&2\n"
                "  exit 41\n"
                "fi\n"
                "exec /bin/cat\n",
            )
            descriptor = os.open(backup, os.O_RDONLY)
            try:
                replacement = root / "replacement.csv.zst"
                replacement.write_bytes(b"wrong_header\nwrong_value\n")
                os.replace(replacement, backup)
                with auditor_module.open_backup_csv_descriptor(
                    descriptor, backup, zstd
                ) as handle:
                    rows = list(csv.DictReader(handle))
            finally:
                os.close(descriptor)
            self.assertEqual(rows, [{"event_time": "2026-01-02 03:04:05", "value": "7"}])

    @unittest.skipUnless(shutil.which("zstd"), "zstd executable is unavailable")
    def test_real_zstd_compressed_backup_is_parsed_from_stdin(self) -> None:
        zstd = Path(shutil.which("zstd") or "").resolve()
        payload = b"event_time,value\n2026-01-02 03:04:05,7\n"
        with tempfile.TemporaryDirectory() as temporary:
            backup = Path(temporary).resolve() / "backup.csv.zst"
            with backup.open("wb") as output:
                subprocess.run(
                    [str(zstd), "-q", "-c"],
                    input=payload,
                    stdout=output,
                    check=True,
                )
            descriptor = os.open(backup, os.O_RDONLY)
            try:
                with auditor_module.open_backup_csv_descriptor(
                    descriptor, backup, zstd
                ) as handle:
                    rows = list(csv.DictReader(handle))
            finally:
                os.close(descriptor)
            self.assertEqual(rows, [{"event_time": "2026-01-02 03:04:05", "value": "7"}])

    def test_empty_stdout_reports_zstd_stderr_instead_of_backup_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backup = root / "backup.csv.zst"
            backup.write_bytes(b"not-a-zstd-frame")
            zstd = self.make_executable(
                root,
                "zstd-fail",
                "#!/bin/sh\n"
                "/bin/cat >/dev/null\n"
                "echo 'synthetic stdin decompression failure' >&2\n"
                "exit 37\n",
            )
            entry = {
                "output_name": "event_time",
                "output_source_index": 0,
                "source_columns": ["event_time"],
                "formats": ["%Y-%m-%d %H:%M:%S"],
                "timezone_rule": "assume_utc_coordinate",
                "event_utc_instant_claim": False,
            }
            with self.assertRaises(AuditError) as caught:
                auditor_module.recompute_backup_timestamps(
                    "D00", [entry], backup, zstd
                )
            self.assertEqual(caught.exception.code, "backup_read")
            self.assertEqual(caught.exception.details["exit_code"], 37)
            self.assertIn(
                "synthetic stdin decompression failure",
                caught.exception.details["stderr"],
            )

    def test_partial_header_nonzero_reports_zstd_failure_not_missing_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backup = root / "backup.csv.zst"
            backup.write_bytes(b"not-a-zstd-frame")
            zstd = self.make_executable(
                root,
                "zstd-partial-fail",
                "#!/bin/sh\n"
                "/bin/cat >/dev/null\n"
                "echo 'wrong_header'\n"
                "echo 'synthetic partial-header decompression failure' >&2\n"
                "exit 37\n",
            )
            entry = {
                "output_name": "event_time",
                "output_source_index": 0,
                "source_columns": ["event_time"],
                "formats": ["%Y-%m-%d %H:%M:%S"],
                "timezone_rule": "assume_utc_coordinate",
                "event_utc_instant_claim": False,
            }
            with self.assertRaises(AuditError) as caught:
                auditor_module.recompute_backup_timestamps(
                    "D00", [entry], backup, zstd
                )
            self.assertEqual(caught.exception.code, "backup_read")
            self.assertEqual(caught.exception.details["exit_code"], 37)
            self.assertIn(
                "synthetic partial-header decompression failure",
                caught.exception.details["stderr"],
            )


class Core19AuditFixture:
    def __init__(self, root: Path) -> None:
        root = root.resolve()
        self.root = root
        self.scope = root / "scope.json"
        self.policy = root / "policy.json"
        self.producer = root / "producer.py"
        self.active = root / "active"
        self.backup = root / "backup"
        self.output = root / "output"
        self.zstd = root / "zstd"
        self.datasets = [f"D{index:02d}" for index in range(19)]
        self.active.mkdir()
        self.backup.mkdir()
        self.output.mkdir()
        self.producer.write_text("# frozen fixture producer\n", encoding="utf-8")
        self.zstd.write_text("#!/bin/sh\nexec /bin/cat\n", encoding="utf-8")
        self.zstd.chmod(0o700)
        self._build_scope_policy()
        self._build_datasets()
        self._build_active_top()
        self._build_output_top()

    def _build_scope_policy(self) -> None:
        write_json(
            self.scope,
            {
                "schema": "web-compression-project-dataset-scope-v1",
                "scope_id": "fixture-core19",
                "core_csv_dataset_count": 19,
                "core_csv_datasets": self.datasets,
                "core_csv_column_cap": {
                    "limit_bytes": 1 << 30,
                    "unit": "bytes",
                    "label": "1 GiB",
                    "selection": "longest whole-column prefix",
                },
                "timestamp_dataset_count": 160,
                "timestamp_datasets": [
                    {
                        "id": f"T{index:03d}",
                        "repo_id": f"THULab/T{index:03d}",
                        "revision": f"{index:040x}",
                        "project_role": "real_timestamp_dataset",
                    }
                    for index in range(160)
                ],
                "total_dataset_count": 179,
            },
        )
        policies = {dataset: {"timestamps": []} for dataset in self.datasets}
        policies["D00"] = {
            "timestamps": [
                {
                    "output_name": "event_time",
                    "output_source_index": 0,
                    "source": "backup_string",
                    "source_columns": ["event_time"],
                    "formats": [
                        "%Y-%m-%d %H:%M:%S.%f",
                        "%Y-%m-%d %H:%M:%S",
                    ],
                    "timezone_rule": "assume_utc_coordinate",
                    "event_utc_instant_claim": False,
                },
                {
                    "output_name": "time_only",
                    "output_source_index": 2,
                    "source": "backup_string",
                    "source_columns": ["time_only"],
                    "format": "%H:%M",
                    "timezone_rule": "drop_time_only",
                    "event_utc_instant_claim": False,
                    "action": "drop",
                    "blocker_reason": "No authoritative date pairing or rollover rule.",
                },
            ]
        }
        write_json(
            self.policy,
            {
                "schema": "project179-core19-utc-policy-v2",
                "scope_id": "fixture-core19",
                "target_timestamp": {
                    "dtype": "int64",
                    "unit": "unix_epoch_microseconds",
                    "byte_order": "little",
                    "range": "signed_int64",
                    "submicrosecond_policy": "reject",
                    "missing_policy": "drop_missing_per_column",
                    "invalid_nonmissing_policy": "block_column",
                },
                "policy_counts": {
                    "timestamp_entries": 2,
                    "convertible_if_sources_validate": 1,
                    "exact_utc_instant_claim_true": 0,
                    "coordinate_only_instant_claim_false": 1,
                    "time_only_dropped": 1,
                },
                "datasets": policies,
            },
        )

    @staticmethod
    def split_spec(chunk_bytes: int) -> dict:
        return {
            "train_fraction": "0.1",
            "validation_fraction": "0.1",
            "test_fraction": "0.8",
            "rounding": "floor train and validation counts; test receives remainder",
            "train_chunking": "at least 3 near-equal chunks when train rows >= 3",
            "validation_test_chunking": "minimum number of near-equal max-bounded chunks",
            "max_chunk_bytes": chunk_bytes,
            "value_width_bytes": 8,
        }

    def _write_output_column(
        self,
        dataset: str,
        *,
        name: str,
        source_index: int,
        dtype: str,
        payload: bytes,
        source_kind: str,
        timestamp: bool = False,
    ) -> dict:
        dataset_dir = self.output / "datasets" / dataset
        raw = dataset_dir / f"column_{source_index:04d}.{dtype}.bin"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(payload)
        chunk = dataset_dir / "chunks" / f"column_{source_index:04d}" / "test-00000.bin"
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_bytes(payload)
        row = {
            "dataset": dataset,
            "column_name": name,
            "source_index": source_index,
            "dtype": dtype,
            "source_kind": source_kind,
            "status": "ready",
            "raw_path": str(raw.resolve()),
            "raw_bytes": len(payload),
            "raw_sha256": sha256(raw),
            "retained_rows": len(payload) // 8,
            "removed_missing_count": 0,
            "split_rows": {"train": 0, "validation": 0, "test": len(payload) // 8},
            "chunks": [
                {
                    "split": "test",
                    "index": 0,
                    "rows": len(payload) // 8,
                    "bytes": len(payload),
                    "path": str(chunk.resolve()),
                    "sha256": sha256(chunk),
                    "source_start_row": 0,
                    "source_end_row_exclusive": len(payload) // 8,
                }
            ],
        }
        if timestamp:
            row.update(
                timestamp_unit="unix_epoch_microseconds",
                timezone_rule="assume_utc_coordinate",
                event_utc_instant_claim=False,
            )
        return row

    def _build_datasets(self) -> None:
        policy_sha = sha256(self.policy)
        producer_sha = sha256(self.producer)
        chunk_bytes = 64
        for dataset_index, dataset in enumerate(self.datasets):
            prepared_dir = self.root / "prepared" / dataset
            prepared_dir.mkdir(parents=True)
            source_index = 1 if dataset == "D00" else 0
            source_raw = prepared_dir / "value.int64.bin"
            source_raw.write_bytes(struct.pack("<qq", dataset_index + 1, dataset_index + 2))
            prepared_column = {
                "key": f"{dataset}__value",
                "column_name": "value",
                "source_index": 0,
                "dtype": "int64",
                "status": "ready",
                "raw_path": str(source_raw.resolve()),
                "raw_bytes": source_raw.stat().st_size,
                "raw_sha256": sha256(source_raw),
                "retained_rows": 2,
                "removed_nan_count": 0,
            }
            prepared_manifest = prepared_dir / "manifest.json"
            write_json(prepared_manifest, {"dataset": dataset, "columns": [prepared_column]})

            metadata_path = self.backup / f"{dataset}_column_info.csv"
            with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                writer = __import__("csv").writer(handle)
                writer.writerow(["column_index", "column_name", "logical_type", "max_point_number"])
                if dataset == "D00":
                    writer.writerow([0, "event_time", "string", 0])
                writer.writerow([source_index, "value", "integer", 0])
                if dataset == "D00":
                    writer.writerow([2, "time_only", "time", 0])

            active_manifest = self.active / "datasets" / dataset / "manifest.json"
            write_json(
                active_manifest,
                {
                    "dataset": dataset,
                    "identity": {
                        "prepared_dataset_manifest_path": str(prepared_manifest.resolve()),
                        "prepared_dataset_manifest_sha256": sha256(prepared_manifest),
                        "ready_raw_columns": [
                            {
                                "key": prepared_column["key"],
                                "path": str(source_raw.resolve()),
                                "bytes": source_raw.stat().st_size,
                                "sha256": sha256(source_raw),
                            }
                        ],
                        "upstream_metadata_path": str(metadata_path.resolve()),
                    },
                },
            )

            backup_path = None
            if dataset == "D00":
                backup_path = self.backup / "D00_removed_or_timestamp_columns.csv"
                with backup_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = __import__("csv").writer(handle)
                    writer.writerow(["event_time", "time_only"])
                    writer.writerow(["2020-01-01 00:00:00.000001", "00:00"])
                    writer.writerow(["2020-01-01 00:00:01", "00:01"])

            policy_entry = load_json(self.policy)["datasets"][dataset]
            identity = {
                "version": "fixture-v1",
                "dataset": dataset,
                "limit_bytes": 1 << 30,
                "chunk_bytes": chunk_bytes,
                "producer_sha256": producer_sha,
                "split_spec": self.split_spec(chunk_bytes),
                "policy_sha256": policy_sha,
                "dataset_policy": policy_entry,
                "active_manifest_path": str(active_manifest.resolve()),
                "active_manifest_sha256": sha256(active_manifest),
                "prepared_manifest_path": str(prepared_manifest.resolve()),
                "prepared_manifest_sha256": sha256(prepared_manifest),
                "backup": None if backup_path is None else file_binding(backup_path),
                "metadata": file_binding(metadata_path),
            }
            columns = []
            timestamp_receipts = []
            if dataset == "D00":
                (self.output / "datasets" / dataset / ".staging").mkdir(parents=True, exist_ok=True)
                values = [
                    timestamp_us("2020-01-01 00:00:00.000001"),
                    timestamp_us("2020-01-01 00:00:01.000000"),
                ]
                timestamp_payload = struct.pack("<qq", *values)
                timestamp_column = self._write_output_column(
                    dataset,
                    name="event_time",
                    source_index=0,
                    dtype="int64",
                    payload=timestamp_payload,
                    source_kind="backup_string",
                    timestamp=True,
                )
                columns.append(timestamp_column)
                # The producer writes intentional drops before convertible
                # backup timestamps.  The auditor must validate identities as
                # a set rather than accidentally depending on policy order.
                timestamp_receipts.append(
                    {
                        "dataset": dataset,
                        "column_name": "time_only",
                        "source_index": 2,
                        "dtype": "int64",
                        "timestamp_unit": "unix_epoch_microseconds",
                        "timezone_rule": "drop_time_only",
                        "event_utc_instant_claim": False,
                        "source_kind": "backup_string",
                        "source_columns": ["time_only"],
                        "format": "%H:%M",
                        "status": "dropped_time_only",
                        "blocker_reason": "No authoritative date pairing or rollover rule.",
                    }
                )
                timestamp_receipts.append(
                    {
                        "dataset": dataset,
                        "column_name": "event_time",
                        "source_index": 0,
                        "dtype": "int64",
                        "timestamp_unit": "unix_epoch_microseconds",
                        "timezone_rule": "assume_utc_coordinate",
                        "event_utc_instant_claim": False,
                        "source_kind": "backup_string",
                        "source_columns": ["event_time"],
                        "formats": [
                            "%Y-%m-%d %H:%M:%S.%f",
                            "%Y-%m-%d %H:%M:%S",
                        ],
                        "status": "candidate_ready",
                        "retained_rows": 2,
                        "removed_missing_count": 0,
                        "source_rows": 2,
                        "raw_bytes": len(timestamp_payload),
                        "raw_sha256": hashlib.sha256(timestamp_payload).hexdigest(),
                        "backup_path": str(backup_path.resolve()),
                        "backup_bytes": backup_path.stat().st_size,
                        "backup_sha256": sha256(backup_path),
                    }
                )
            numeric_column = self._write_output_column(
                dataset,
                name="value",
                source_index=source_index,
                dtype="int64",
                payload=source_raw.read_bytes(),
                source_kind="prepared_numeric",
            )
            columns.append(numeric_column)
            for active_index, column in enumerate(columns):
                column["active_index"] = active_index
            counts = {
                "timestamp_admitted": 1 if dataset == "D00" else 0,
                "timestamp_selected": 1 if dataset == "D00" else 0,
                "timestamp_blocked": 0,
                "time_only_dropped": 1 if dataset == "D00" else 0,
                "remaining_strings_dropped": 0,
                "numeric_candidates": len(columns),
                "numeric_selected": len(columns),
                "chunks": len(columns),
                "chunk_bytes": sum(column["raw_bytes"] for column in columns),
            }
            dataset_manifest = self.output / "datasets" / dataset / "manifest.json"
            write_json(
                dataset_manifest,
                {
                    "schema": "project179-core19-numeric-utc-dataset-v1",
                    "version": "fixture-v1",
                    "status": "complete",
                    "dataset": dataset,
                    "identity": identity,
                    "selection": {
                        "kind": "longest_whole_numeric_column_prefix_after_timestamp_conversion",
                        "limit_bytes": 1 << 30,
                        "selected_bytes": sum(column["raw_bytes"] for column in columns),
                        "candidate_columns": len(columns),
                        "selected_columns": len(columns),
                        "selected_source_indices": [column["source_index"] for column in columns],
                        "excluded_by_cap_count": 0,
                    },
                    "columns": columns,
                    "excluded_numeric_columns": [],
                    "timestamp_receipts": timestamp_receipts,
                    "string_receipts": [],
                    "counts": counts,
                    "prepared_unix_seconds": 0.0,
                },
            )

    def _build_active_top(self) -> None:
        active_top = self.active / "manifest.json"
        derived_scope = {
            "schema": "core19-1g-column-prefix-derived-scope-v1",
            "scope_id": "fixture-core19",
            "scope_path": str(self.scope.resolve()),
            "scope_sha256": sha256(self.scope),
            "source_top_manifest_path": str((self.root / "immutable-source.json").resolve()),
            "source_top_manifest_sha256": "0" * 64,
            "excluded_datasets": ["poisson_timestamps"],
            "trimmed_dataset_count": 0,
            "trimmed_datasets": [],
            "deletion_semantics": "fixture",
        }
        selection = {
            dataset: {
                "source_columns": 1,
                "selected_columns": 1,
                "selected_source_indices": [0],
                "selected_bytes": 16,
            }
            for dataset in self.datasets
        }
        totals = {
            "blocking_source_numeric_failures": 0,
            "chunks": 19,
            "provenance_only_nonready_entries": 0,
            "raw_bytes": 19 * 16,
            "ready_columns": 19,
            "retained_rows": 19 * 2,
        }
        active_entries = [
            {
                "dataset": dataset,
                "manifest_path": str((self.active / "datasets" / dataset / "manifest.json").resolve()),
                "manifest_sha256": sha256(self.active / "datasets" / dataset / "manifest.json"),
                "totals": {
                    "blocking_source_numeric_failures": 0,
                    "chunks": 1,
                    "provenance_only_nonready_entries": 0,
                    "raw_bytes": 16,
                    "ready_columns": 1,
                    "retained_rows": 2,
                },
            }
            for dataset in self.datasets
        ]
        write_json(
            active_top,
            {
                "schema": "fixture-active-top",
                "datasets": active_entries,
                "totals": totals,
                "derived_scope": derived_scope,
            },
        )
        write_json(self.active / "PREVIEW_PROTOCOL.json", {"derived_scope": derived_scope})
        write_json(
            self.active / "SOURCE_CONTRACT.json",
            {
                "schema": "core19-1g-column-prefix-source-contract-v1",
                "status": "complete",
                "active_manifest_path": str(active_top.resolve()),
                "active_manifest_sha256": sha256(active_top),
                "dataset_count": 19,
                "datasets": self.datasets,
                "selection": selection,
                "totals": totals,
                "dataset_manifest_hashes": {
                    dataset: sha256(self.active / "datasets" / dataset / "manifest.json")
                    for dataset in self.datasets
                },
                "derived_scope": derived_scope,
            },
        )

    def _build_output_top(self) -> None:
        entries = []
        totals = {
            "timestamp_admitted": 0,
            "timestamp_selected": 0,
            "timestamp_blocked": 0,
            "time_only_dropped": 0,
            "remaining_strings_dropped": 0,
            "numeric_candidates": 0,
            "numeric_selected": 0,
            "chunks": 0,
            "chunk_bytes": 0,
        }
        for dataset in self.datasets:
            path = self.output / "datasets" / dataset / "manifest.json"
            manifest = load_json(path)
            entries.append(
                {
                    "dataset": dataset,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "status": "complete",
                    "counts": manifest["counts"],
                }
            )
            for key in totals:
                totals[key] += int(manifest["counts"][key])
        write_json(
            self.output / "manifest.json",
            {
                "schema": "project179-core19-numeric-utc-scope-v1",
                "version": "fixture-v1",
                "status": "complete",
                "dataset_count": 19,
                "datasets": self.datasets,
                "policy_path": str(self.policy.resolve()),
                "policy_sha256": sha256(self.policy),
                "active_root": str(self.active.resolve()),
                "backup_root": str(self.backup.resolve()),
                "limit_bytes": 1 << 30,
                "chunk_bytes": 64,
                "totals": totals,
                "dataset_manifests": entries,
                "producer": {
                    "path": str(self.producer.resolve()),
                    "sha256": sha256(self.producer),
                },
                "completed_unix_seconds": 0.0,
            },
        )

    def refresh_top(self) -> None:
        top = load_json(self.output / "manifest.json")
        totals = {key: 0 for key in top["totals"]}
        for entry in top["dataset_manifests"]:
            path = Path(entry["path"])
            manifest = load_json(path)
            entry["sha256"] = sha256(path)
            entry["status"] = manifest["status"]
            entry["counts"] = manifest["counts"]
            for key in totals:
                totals[key] += int(manifest["counts"][key])
        top["totals"] = totals
        write_json(self.output / "manifest.json", top)

    def run(self):
        return audit(
            scope_path=self.scope,
            policy_path=self.policy,
            producer_path=self.producer,
            active_root=self.active,
            backup_root=self.backup,
            output_root=self.output,
            zstd_path=self.zstd,
        )


class Project179Core19ColdAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Core19AuditFixture(Path(self.temporary.name))
        authority_values = {
            "EXPECTED_SCOPE_SHA256": sha256(self.fixture.scope),
            "EXPECTED_POLICY_SHA256": sha256(self.fixture.policy),
            "EXPECTED_PRODUCER_SHA256": sha256(self.fixture.producer),
            "EXPECTED_ZSTD_PATH": str(self.fixture.zstd),
            "EXPECTED_ZSTD_SHA256": sha256(self.fixture.zstd),
        }
        self.authority_patchers = [
            mock.patch.object(auditor_module, name, value)
            for name, value in authority_values.items()
        ]
        for patcher in self.authority_patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.authority_patchers):
            patcher.stop()
        self.temporary.cleanup()

    def assertAuditCode(self, code: str) -> None:
        with self.assertRaises(AuditError) as caught:
            self.fixture.run()
        self.assertEqual(caught.exception.code, code)

    def test_valid_tree_passes_without_writing_output(self) -> None:
        before = {
            str(path.relative_to(self.fixture.output)): sha256(path)
            for path in self.fixture.output.rglob("*")
            if path.is_file()
        }
        receipt = self.fixture.run()
        after = {
            str(path.relative_to(self.fixture.output)): sha256(path)
            for path in self.fixture.output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["dataset_count"], 19)
        self.assertEqual(receipt["totals"]["timestamp_blocked"], 0)
        self.assertTrue(receipt["release_checks"]["all_source_output_and_chunk_hashes_streamed"])
        self.assertEqual(before, after)

    def test_coordinated_dataset_and_top_manifest_tamper_fails(self) -> None:
        path = self.fixture.output / "datasets/D01/manifest.json"
        manifest = load_json(path)
        manifest["columns"][0]["source_index"] = 99
        manifest["selection"]["selected_source_indices"] = [99]
        write_json(path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("prefix_selection")

    def test_logical_boolean_payload_fails_even_when_receipts_are_rehashed(self) -> None:
        metadata = self.fixture.backup / "D01_column_info.csv"
        text = metadata.read_text(encoding="utf-8").replace(",integer,", ",BOOLEAN,")
        metadata.write_text(text, encoding="utf-8")
        path = self.fixture.output / "datasets/D01/manifest.json"
        manifest = load_json(path)
        manifest["identity"]["metadata"] = file_binding(metadata)
        write_json(path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("logical_type")

    def test_scope_with_poisson_is_rejected_before_output_trust(self) -> None:
        scope = load_json(self.fixture.scope)
        scope["core_csv_datasets"][-1] = "poisson_timestamps"
        write_json(self.fixture.scope, scope)
        with mock.patch.object(auditor_module, "EXPECTED_SCOPE_SHA256", sha256(self.fixture.scope)):
            self.assertAuditCode("retired_dataset")

    def test_nonlongest_prefix_fails_after_coordinated_receipt_update(self) -> None:
        path = self.fixture.output / "datasets/D00/manifest.json"
        manifest = load_json(path)
        removed = manifest["columns"].pop()
        excluded = {key: value for key, value in removed.items() if key not in {"raw_path", "chunks", "split_rows", "active_index"}}
        excluded["status"] = "excluded_by_1g_whole_column_prefix"
        manifest["excluded_numeric_columns"] = [excluded]
        manifest["selection"].update(
            selected_bytes=manifest["columns"][0]["raw_bytes"],
            selected_columns=1,
            selected_source_indices=[0],
            excluded_by_cap_count=1,
        )
        manifest["counts"]["numeric_selected"] = 1
        manifest["counts"]["chunks"] = 1
        manifest["counts"]["chunk_bytes"] = manifest["columns"][0]["raw_bytes"]
        write_json(path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("prefix_selection")

    def test_chunk_hash_mismatch_fails(self) -> None:
        path = self.fixture.output / "datasets/D01/chunks/column_0000/test-00000.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 0xFF
        path.write_bytes(payload)
        self.assertAuditCode("chunk_hash")

    def test_chunk_raw_mismatch_fails_even_if_chunk_receipt_is_rehashed(self) -> None:
        dataset_path = self.fixture.output / "datasets/D01/manifest.json"
        manifest = load_json(dataset_path)
        chunk = Path(manifest["columns"][0]["chunks"][0]["path"])
        payload = bytearray(chunk.read_bytes())
        payload[-1] ^= 0xFF
        chunk.write_bytes(payload)
        manifest["columns"][0]["chunks"][0]["sha256"] = sha256(chunk)
        write_json(dataset_path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("chunk_raw_mismatch")

    def test_raw_hash_mismatch_fails(self) -> None:
        path = self.fixture.output / "datasets/D01/column_0000.int64.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 0x80
        path.write_bytes(payload)
        self.assertAuditCode("raw_hash")

    def test_extra_output_file_fails(self) -> None:
        (self.fixture.output / "datasets/D01/unreceipted.bin").write_bytes(b"extra")
        self.assertAuditCode("output_file_set")

    def test_incomplete_top_fails(self) -> None:
        path = self.fixture.output / "manifest.json"
        top = load_json(path)
        top["status"] = "running"
        write_json(path, top)
        self.assertAuditCode("top_incomplete")

    def test_timestamp_payload_cannot_be_laundered_by_rehashing_receipts(self) -> None:
        dataset_path = self.fixture.output / "datasets/D00/manifest.json"
        manifest = load_json(dataset_path)
        column = manifest["columns"][0]
        raw = Path(column["raw_path"])
        chunk = Path(column["chunks"][0]["path"])
        corrupt = struct.pack("<qq", 1, 2)
        raw.write_bytes(corrupt)
        chunk.write_bytes(corrupt)
        forged = hashlib.sha256(corrupt).hexdigest()
        column["raw_sha256"] = forged
        column["chunks"][0]["sha256"] = forged
        timestamp_receipt = next(
            row for row in manifest["timestamp_receipts"]
            if row["column_name"] == "event_time"
        )
        timestamp_receipt["raw_sha256"] = forged
        write_json(dataset_path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("timestamp_receipts")

    def test_receipt_inside_output_root_is_refused_before_audit(self) -> None:
        receipt = self.fixture.output / "audit.json"
        argv = [
            "audit_project179_core19_numeric_v1.py",
            "--scope", str(self.fixture.scope),
            "--policy", str(self.fixture.policy),
            "--producer", str(self.fixture.producer),
            "--active-root", str(self.fixture.active),
            "--backup-root", str(self.fixture.backup),
            "--output-root", str(self.fixture.output),
            "--zstd", str(self.fixture.zstd),
            "--receipt", str(receipt),
        ]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            code = main()
        self.assertEqual(code, 2)
        self.assertFalse(receipt.exists())
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "unsafe_receipt_path")

    def test_explicit_external_receipt_is_written_and_bound(self) -> None:
        receipt = self.fixture.root / "control" / "audit.json"
        receipt.parent.mkdir()
        argv = [
            "audit_project179_core19_numeric_v1.py",
            "--scope", str(self.fixture.scope),
            "--policy", str(self.fixture.policy),
            "--producer", str(self.fixture.producer),
            "--active-root", str(self.fixture.active),
            "--backup-root", str(self.fixture.backup),
            "--output-root", str(self.fixture.output),
            "--zstd", str(self.fixture.zstd),
            "--receipt", str(receipt),
        ]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            code = main()
        self.assertEqual(code, 0)
        written = load_json(receipt)
        reported = json.loads(output.getvalue())
        self.assertEqual(written["status"], "PASS")
        self.assertEqual(written["dataset_count"], 19)
        self.assertEqual(reported["written_receipt"]["sha256"], sha256(receipt))

    def test_real_producer_output_passes_auditor(self) -> None:
        shutil.rmtree(self.fixture.output)
        self.fixture.output.mkdir()
        producer = Path(__file__).with_name("prepare_project179_core19_numeric_v1.py").resolve()
        produced = prepare_scope(
            self.fixture.active,
            self.fixture.backup,
            self.fixture.output,
            self.fixture.policy,
            limit_bytes=1 << 30,
            chunk_bytes=64,
        )
        with mock.patch.object(auditor_module, "EXPECTED_PRODUCER_SHA256", sha256(producer)):
            receipt = audit(
                scope_path=self.fixture.scope,
                policy_path=self.fixture.policy,
                producer_path=producer,
                active_root=self.fixture.active,
                backup_root=self.fixture.backup,
                output_root=self.fixture.output,
                zstd_path=self.fixture.zstd,
            )
        self.assertEqual(produced["status"], "complete")
        self.assertEqual(receipt["status"], "PASS")

    def test_existing_receipt_is_never_clobbered(self) -> None:
        receipt = self.fixture.root / "audit.json"
        receipt.write_bytes(b"sentinel\n")
        argv = [
            "audit_project179_core19_numeric_v1.py",
            "--scope", str(self.fixture.scope),
            "--policy", str(self.fixture.policy),
            "--producer", str(self.fixture.producer),
            "--active-root", str(self.fixture.active),
            "--backup-root", str(self.fixture.backup),
            "--output-root", str(self.fixture.output),
            "--zstd", str(self.fixture.zstd),
            "--receipt", str(receipt),
        ]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            code = main()
        self.assertEqual(code, 2)
        self.assertEqual(receipt.read_bytes(), b"sentinel\n")
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "receipt_exists")

    def test_receipt_cannot_alias_control_input(self) -> None:
        original = self.fixture.policy.read_bytes()
        argv = [
            "audit_project179_core19_numeric_v1.py",
            "--scope", str(self.fixture.scope),
            "--policy", str(self.fixture.policy),
            "--producer", str(self.fixture.producer),
            "--active-root", str(self.fixture.active),
            "--backup-root", str(self.fixture.backup),
            "--output-root", str(self.fixture.output),
            "--zstd", str(self.fixture.zstd),
            "--receipt", str(self.fixture.policy),
        ]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            code = main()
        self.assertEqual(code, 2)
        self.assertEqual(self.fixture.policy.read_bytes(), original)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "unsafe_receipt_path")

    def test_relative_zstd_path_is_rejected(self) -> None:
        with self.assertRaises(AuditError) as caught:
            audit(
                scope_path=self.fixture.scope,
                policy_path=self.fixture.policy,
                producer_path=self.fixture.producer,
                active_root=self.fixture.active,
                backup_root=self.fixture.backup,
                output_root=self.fixture.output,
                zstd_path=Path("zstd"),
            )
        self.assertEqual(caught.exception.code, "zstd_path")

    def test_explicit_zstd_ignores_path_hijack(self) -> None:
        csv_path = self.fixture.backup / "D00_removed_or_timestamp_columns.csv"
        zst_path = csv_path.with_suffix(".csv.zst")
        csv_path.rename(zst_path)
        binding = file_binding(zst_path)
        manifest_path = self.fixture.output / "datasets/D00/manifest.json"
        manifest = load_json(manifest_path)
        manifest["identity"]["backup"] = binding
        for row in manifest["timestamp_receipts"]:
            if row.get("status") == "candidate_ready":
                row["backup_path"] = binding["path"]
                row["backup_bytes"] = binding["bytes"]
                row["backup_sha256"] = binding["sha256"]
        write_json(manifest_path, manifest)
        self.fixture.refresh_top()
        malicious = self.fixture.root / "malicious-bin"
        malicious.mkdir()
        marker = self.fixture.root / "PATH_WAS_USED"
        fake = malicious / "zstd"
        fake.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n", encoding="utf-8")
        fake.chmod(0o700)
        with mock.patch.dict(os.environ, {"PATH": str(malicious)}):
            self.assertEqual(self.fixture.run()["status"], "PASS")
        self.assertFalse(marker.exists())

    def test_prepared_nonready_numeric_candidate_fails(self) -> None:
        prepared_path = self.fixture.root / "prepared/D01/manifest.json"
        prepared = load_json(prepared_path)
        prepared["columns"].append(
            {
                "key": "D01__omitted",
                "column_name": "omitted",
                "source_index": 1,
                "dtype": "int64",
                "status": "failed_parse",
            }
        )
        write_json(prepared_path, prepared)
        active_path = self.fixture.active / "datasets/D01/manifest.json"
        active = load_json(active_path)
        active["identity"]["prepared_dataset_manifest_sha256"] = sha256(prepared_path)
        write_json(active_path, active)
        output_path = self.fixture.output / "datasets/D01/manifest.json"
        output = load_json(output_path)
        output["identity"]["prepared_manifest_sha256"] = sha256(prepared_path)
        output["identity"]["active_manifest_sha256"] = sha256(active_path)
        write_json(output_path, output)
        self.fixture._build_active_top()
        self.fixture.refresh_top()
        self.assertAuditCode("numeric_nonready")

    def test_metadata_numeric_candidate_omission_fails(self) -> None:
        metadata = self.fixture.backup / "D01_column_info.csv"
        with metadata.open("a", encoding="utf-8", newline="") as handle:
            handle.write("1,missing_numeric,float,0\n")
        manifest_path = self.fixture.output / "datasets/D01/manifest.json"
        manifest = load_json(manifest_path)
        manifest["identity"]["metadata"] = file_binding(metadata)
        write_json(manifest_path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("numeric_candidate_missing")

    def test_drop_timestamp_source_column_must_exist(self) -> None:
        backup = self.fixture.backup / "D00_removed_or_timestamp_columns.csv"
        backup.write_text(
            "event_time\n2020-01-01 00:00:00.000001\n2020-01-01 00:00:01.000002\n",
            encoding="utf-8",
        )
        binding = file_binding(backup)
        manifest_path = self.fixture.output / "datasets/D00/manifest.json"
        manifest = load_json(manifest_path)
        manifest["identity"]["backup"] = binding
        for row in manifest["timestamp_receipts"]:
            if row.get("status") == "candidate_ready":
                row["backup_bytes"] = binding["bytes"]
                row["backup_sha256"] = binding["sha256"]
        write_json(manifest_path, manifest)
        self.fixture.refresh_top()
        self.assertAuditCode("backup_schema")

    def test_utc_midnight_coordinate_rejects_nonmidnight(self) -> None:
        entry = {
            "format": "%Y-%m-%d %H:%M:%S",
            "timezone_rule": "utc_midnight_coordinate",
        }
        with self.assertRaises(AuditError) as caught:
            parse_timestamp_us(entry, ["2020-01-01 00:00:01"])
        self.assertEqual(caught.exception.code, "timestamp_midnight")

    def test_output_symlink_hardlink_special_and_extra_directory_fail(self) -> None:
        raw = self.fixture.output / "datasets/D01/column_0000.int64.bin"
        raw_link = self.fixture.output / "datasets/D01/raw-link.bin"
        raw_link.symlink_to(raw)
        self.assertAuditCode("symlink_rejected")
        raw_link.unlink()

        hard_link = self.fixture.output / "datasets/D01/raw-hard.bin"
        os.link(raw, hard_link)
        self.assertAuditCode("hardlink_rejected")
        hard_link.unlink()

        fifo = self.fixture.output / "datasets/D01/fifo"
        os.mkfifo(fifo)
        self.assertAuditCode("special_node_rejected")
        fifo.unlink()

        extra = self.fixture.output / "datasets/D01/empty"
        extra.mkdir()
        self.assertAuditCode("output_directory_set")

    def test_raw_unresolved_symlink_input_is_rejected(self) -> None:
        original = self.fixture.scope.read_bytes()
        target = self.fixture.root / "scope-real.json"
        target.write_bytes(original)
        self.fixture.scope.unlink()
        self.fixture.scope.symlink_to(target)
        self.assertAuditCode("symlink_rejected")

    def test_source_contract_must_bind_scope_hash_and_path(self) -> None:
        contract_path = self.fixture.active / "SOURCE_CONTRACT.json"
        contract = load_json(contract_path)
        contract["derived_scope"]["scope_sha256"] = "0" * 64
        write_json(contract_path, contract)
        self.assertAuditCode("active_scope_binding")

    def test_unpinned_self_consistent_fake_scope_is_rejected(self) -> None:
        official_scope_sha256 = "52d3891e8efe653a15d1f1535e7fa922302e6e97cdc797639caa3d289573444f"
        with mock.patch.object(auditor_module, "EXPECTED_SCOPE_SHA256", official_scope_sha256):
            self.assertAuditCode("authority_scope_hash")

    def test_explicit_fake_decoder_is_rejected_by_pinned_hash(self) -> None:
        official_zstd_sha256 = "d304445daa7e6429293dc02035063b7993fb6a489ee90d8851bff497952836dc"
        with mock.patch.object(auditor_module, "EXPECTED_ZSTD_SHA256", official_zstd_sha256):
            self.assertAuditCode("authority_zstd_hash")

    def test_output_raw_mutation_after_first_hash_is_detected_at_end(self) -> None:
        target = self.fixture.output / "datasets/D01/column_0000.int64.bin"
        original_binding = auditor_module.stable_file_binding
        injected = False

        def mutate_after_binding(path, *, tee=None):
            nonlocal injected
            result = original_binding(path, tee=tee)
            if Path(path) == target and not injected:
                injected = True
                payload = bytearray(target.read_bytes())
                payload[0] ^= 0x01
                target.write_bytes(payload)
            return result

        with mock.patch.object(auditor_module, "stable_file_binding", side_effect=mutate_after_binding):
            self.assertAuditCode("file_changed")
        self.assertTrue(injected)

    def test_output_chunk_aba_replacement_after_first_hash_is_detected(self) -> None:
        target = self.fixture.output / "datasets/D01/chunks/column_0000/test-00000.bin"
        original_binding = auditor_module.stable_file_binding
        injected = False

        def replace_after_binding(path, *, tee=None):
            nonlocal injected
            result = original_binding(path, tee=tee)
            if Path(path) == target and not injected:
                injected = True
                replacement = target.with_name("replacement.bin")
                replacement.write_bytes(target.read_bytes())
                os.replace(replacement, target)
            return result

        with mock.patch.object(auditor_module, "stable_file_binding", side_effect=replace_after_binding):
            self.assertAuditCode("file_changed")
        self.assertTrue(injected)

    def test_metadata_aba_after_single_fd_parse_is_detected_at_end(self) -> None:
        target = self.fixture.backup / "D01_column_info.csv"
        original_reader = auditor_module.read_verified_bytes_stable
        injected = False

        def replace_after_read(path, role):
            nonlocal injected
            result = original_reader(path, role)
            if Path(path) == target and role == "column metadata" and not injected:
                injected = True
                replacement = target.with_name("metadata-replacement.csv")
                replacement.write_bytes(target.read_bytes())
                os.replace(replacement, target)
            return result

        with mock.patch.object(auditor_module, "read_verified_bytes_stable", side_effect=replace_after_read):
            self.assertAuditCode("file_changed")
        self.assertTrue(injected)

    def test_backup_transient_replace_and_restore_aba_is_detected(self) -> None:
        target = self.fixture.backup / "D00_removed_or_timestamp_columns.csv"
        original_open = auditor_module.open_backup_csv_descriptor
        injected = False

        @contextmanager
        def transient_replace(descriptor, path, zstd_path):
            nonlocal injected
            if Path(path) == target and not injected:
                injected = True
                saved = target.with_name("saved-original.csv")
                os.replace(target, saved)
                target.write_text("event_time,time_only\n1999-01-01 00:00:00.000000,00:00\n", encoding="utf-8")
                os.replace(saved, target)
            with original_open(descriptor, path, zstd_path) as handle:
                yield handle

        with mock.patch.object(auditor_module, "open_backup_csv_descriptor", transient_replace):
            self.assertAuditCode("file_changed")
        self.assertTrue(injected)

    def test_late_file_after_final_payload_binding_is_rejected_by_rescan(self) -> None:
        target = self.fixture.output / "datasets/D18/chunks/column_0000/test-00000.bin"
        late = self.fixture.output / "late-extra.bin"
        original_binding = auditor_module.stable_file_binding
        calls = 0
        injected = False

        def create_after_second_binding(path, *, tee=None):
            nonlocal calls, injected
            result = original_binding(path, tee=tee)
            if Path(path) == target:
                calls += 1
                if calls == 2:
                    late.write_bytes(b"late")
                    injected = True
            return result

        with mock.patch.object(auditor_module, "stable_file_binding", side_effect=create_after_second_binding):
            self.assertAuditCode("output_tree_changed")
        self.assertTrue(injected)


if __name__ == "__main__":
    unittest.main()
