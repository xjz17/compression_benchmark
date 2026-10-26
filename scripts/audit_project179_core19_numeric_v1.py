#!/usr/bin/env python3
"""Cold, independent verifier for the Core19 numeric/UTC-us derived tree.

The verifier deliberately does not import the producer.  It rebuilds the exact
Core19 identity and every candidate column from the authoritative scope,
policy, active dataset manifests, prepared manifests, metadata and preserved
timestamp sources.  It then streams every selected raw file and chunk, checks
the complete receipt tree, and emits a machine-readable PASS receipt on stdout.

No file is written unless ``--receipt`` is supplied.  A receipt is never
allowed inside the audited output tree.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
from typing import Any, Iterator, Mapping, TextIO


AUDIT_SCHEMA = "project179-core19-numeric-utc-cold-audit-v1"
EXPECTED_SCOPE_SCHEMA = "web-compression-project-dataset-scope-v1"
EXPECTED_POLICY_SCHEMA = "project179-core19-utc-policy-v2"
EXPECTED_TOP_SCHEMA = "project179-core19-numeric-utc-scope-v1"
EXPECTED_DATASET_SCHEMA = "project179-core19-numeric-utc-dataset-v1"
EXPECTED_SCOPE_SHA256 = "52d3891e8efe653a15d1f1535e7fa922302e6e97cdc797639caa3d289573444f"
EXPECTED_POLICY_SHA256 = "98d6101e88848ee99c62c3d158a9f3105ed68142f4ad7b88d3d115573f996f1e"
EXPECTED_PRODUCER_SHA256 = "d1ee66b6c42feb24eaecb4992de852e1d35844e634c66c44d50aec0cd6e48a20"
EXPECTED_ZSTD_PATH = "/usr/bin/zstd"
EXPECTED_ZSTD_SHA256 = "d304445daa7e6429293dc02035063b7993fb6a489ee90d8851bff497952836dc"
LIMIT_BYTES = 1 << 30
VALUE_BYTES = 8
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
ALLOWED_DTYPES = frozenset({"int64", "float64"})
ALLOWED_NUMERIC_LOGICAL_TYPES = frozenset({"integer", "float"})
RECEIPTED_NONNUMERIC_TYPES = frozenset(
    {"string", "timestamp", "datetime", "date", "time"}
)
MISSING = frozenset(
    {"", "nan", "+nan", "-nan", "na", "n/a", "null", "none", "<na>", "''"}
)
COUNT_KEYS = (
    "timestamp_admitted",
    "timestamp_selected",
    "timestamp_blocked",
    "time_only_dropped",
    "remaining_strings_dropped",
    "numeric_candidates",
    "numeric_selected",
    "chunks",
    "chunk_bytes",
)

# Large real-world CSV cells must not be rejected by the interpreter's small
# platform default.  This is an input-format requirement, not a best effort.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:  # pragma: no cover - only relevant to narrow C longs
    csv.field_size_limit((1 << 31) - 1)


class AuditError(RuntimeError):
    """A fail-closed audit error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def fail(code: str, message: str, **details: Any) -> None:
    raise AuditError(code, message, **details)


def require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        fail(code, message, **details)


def absolute_unresolved(path: Path | str) -> Path:
    """Return an absolute lexical path without following any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def reject_symlink_components(
    path: Path | str, role: str, *, allow_missing_leaf: bool = False
) -> Path:
    """lstat every component of the caller-supplied path before resolve/use."""

    raw = absolute_unresolved(path)
    current = Path(raw.anchor)
    parts = raw.parts[1:]
    for position, part in enumerate(parts):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and position == len(parts) - 1:
                return raw
            fail("missing_file", f"missing {role}", path=str(current))
        except OSError as exc:
            fail("path_lstat", f"cannot lstat {role}", path=str(current), error=str(exc))
        require(
            not stat.S_ISLNK(info.st_mode),
            "symlink_rejected",
            f"{role} path contains a symlink",
            path=str(current),
        )
    return raw


@contextmanager
def open_regular_fd(path: Path | str, role: str) -> Iterator[tuple[int, os.stat_result]]:
    raw = reject_symlink_components(path, role)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(raw, flags)
    except FileNotFoundError:
        fail("missing_file", f"missing {role}", path=str(raw))
    except OSError as exc:
        fail("read_error", f"cannot open {role}", path=str(raw), error=str(exc))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "special_node_rejected", f"{role} must be a regular file", path=str(raw))
        require(before.st_nlink == 1, "hardlink_rejected", f"{role} must not be hard-linked", path=str(raw), link_count=before.st_nlink)
        yield descriptor, before
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "file_changed",
            f"{role} changed while it was being audited",
            path=str(raw),
        )
    finally:
        os.close(descriptor)


def require_directory(path: Path | str, role: str) -> Path:
    raw = reject_symlink_components(path, role)
    try:
        info = os.lstat(raw)
    except OSError as exc:
        fail("missing_directory", f"missing {role}", path=str(raw), error=str(exc))
    require(stat.S_ISDIR(info.st_mode), "special_node_rejected", f"{role} must be a directory", path=str(raw))
    return raw


def read_verified_bytes(path: Path | str, role: str) -> tuple[bytes, dict[str, Any]]:
    payload, stable = read_verified_bytes_stable(path, role)
    return payload, public_binding(stable)


def public_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(binding["path"]),
        "bytes": int(binding["bytes"]),
        "sha256": str(binding["sha256"]),
    }


def read_verified_bytes_stable(path: Path | str, role: str) -> tuple[bytes, dict[str, Any]]:
    raw = absolute_unresolved(path)
    digest = hashlib.sha256()
    blocks: list[bytes] = []
    with open_regular_fd(raw, role) as (descriptor, before):
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
    return b"".join(blocks), {
        "path": str(raw),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
        "stat_dev": before.st_dev,
        "stat_ino": before.st_ino,
    }


def load_object_binding(path: Path | str, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload, binding = read_verified_bytes(path, role)
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail("invalid_json", f"cannot read {role}", path=str(path), error=str(exc))
    require(isinstance(value, dict), "invalid_json", f"{role} must be a JSON object", path=str(path))
    return value, binding


def load_object(path: Path, role: str) -> dict[str, Any]:
    return load_object_binding(path, role)[0]


def stable_file_binding(path: Path, *, tee: Any | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    raw = absolute_unresolved(path)
    with open_regular_fd(raw, "required binary file") as (descriptor, before):
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
            if tee is not None:
                tee.update(block)
    return {
        "path": str(raw),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "stat_dev": before.st_dev,
        "stat_ino": before.st_ino,
    }


def stream_sha256(path: Path, *, tee: Any | None = None) -> tuple[int, str]:
    binding = stable_file_binding(path, tee=tee)
    return int(binding["bytes"]), str(binding["sha256"])


def file_binding(path: Path) -> dict[str, Any]:
    return public_binding(stable_file_binding(path))


def require_regular_file(path: Path, role: str) -> None:
    with open_regular_fd(path, role):
        pass


def same_path(left: Any, right: Path) -> bool:
    try:
        return absolute_unresolved(Path(str(left))) == absolute_unresolved(right)
    except (OSError, RuntimeError, ValueError):
        return False


def inside(path: Path, root: Path) -> bool:
    try:
        absolute_unresolved(path).relative_to(absolute_unresolved(root))
        return True
    except ValueError:
        return False


def atomic_json_no_clobber(path: Path, value: Any) -> None:
    """Publish JSON atomically without ever replacing an existing pathname."""

    path = reject_symlink_components(path, "receipt", allow_missing_leaf=True)
    require_directory(path.parent, "receipt parent")
    require(not path.exists(), "receipt_exists", "refusing to replace an existing receipt", path=str(path))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        # link(2) is an atomic no-replace publication on the same filesystem.
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        fail("receipt_exists", "refusing to replace an existing receipt", path=str(path))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_scope_and_policy(
    scope: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[list[str], int]:
    require(scope.get("schema") == EXPECTED_SCOPE_SCHEMA, "scope_schema", "unsupported scope schema")
    datasets = list(map(str, scope.get("core_csv_datasets", [])))
    require(
        int(scope.get("core_csv_dataset_count", -1)) == 19,
        "scope_count",
        "scope count must be exactly 19",
    )
    require(len(datasets) == 19 and len(set(datasets)) == 19, "scope_identity", "scope must contain 19 distinct ordered dataset identities")
    require("poisson_timestamps" not in datasets, "retired_dataset", "poisson_timestamps is forbidden")
    timestamp_datasets = scope.get("timestamp_datasets")
    require(isinstance(timestamp_datasets, list), "scope_timestamp_set", "scope timestamp_datasets must be a list")
    timestamp_ids: list[str] = []
    timestamp_repos: list[str] = []
    for row in timestamp_datasets:
        require(isinstance(row, dict), "scope_timestamp_set", "timestamp dataset entries must be objects")
        identity = str(row.get("id", ""))
        repository = str(row.get("repo_id", ""))
        require(identity and repository, "scope_timestamp_set", "timestamp dataset identity is incomplete")
        timestamp_ids.append(identity)
        timestamp_repos.append(repository)
    require(
        int(scope.get("timestamp_dataset_count", -1)) == 160
        and len(timestamp_ids) == 160
        and len(set(timestamp_ids)) == 160
        and len(set(timestamp_repos)) == 160,
        "scope_timestamp_set",
        "scope must contain exactly 160 distinct timestamp dataset identities",
    )
    require(
        int(scope.get("total_dataset_count", -1)) == 179,
        "scope_total",
        "scope total must be exactly 19 + 160 = 179",
    )
    timestamp_aliases = set(timestamp_ids) | set(timestamp_repos) | {
        value.rsplit("/", 1)[-1] for value in timestamp_repos
    }
    require(not (set(datasets) & timestamp_aliases), "scope_overlap", "core and timestamp dataset identities must be disjoint")
    require(
        "poisson_timestamps" not in timestamp_aliases,
        "retired_dataset",
        "poisson_timestamps is forbidden from the full project scope",
    )
    cap = scope.get("core_csv_column_cap", {})
    limit_bytes = int(cap.get("limit_bytes", -1))
    require(limit_bytes == LIMIT_BYTES, "scope_limit", "Core19 byte limit must be exactly 1 GiB", actual=limit_bytes)

    require(policy.get("schema") == EXPECTED_POLICY_SCHEMA, "policy_schema", "unsupported policy schema")
    require(policy.get("scope_id") == scope.get("scope_id"), "scope_policy_id", "scope_id differs between scope and policy")
    policy_datasets = policy.get("datasets")
    require(isinstance(policy_datasets, dict), "policy_datasets", "policy datasets must be an object")
    require(list(policy_datasets) == datasets, "scope_policy_identity", "policy dataset order or identity differs from authoritative scope")
    require("poisson_timestamps" not in policy_datasets, "retired_dataset", "policy contains poisson_timestamps")
    target = policy.get("target_timestamp", {})
    require(
        target.get("dtype") == "int64"
        and target.get("unit") == "unix_epoch_microseconds"
        and target.get("byte_order") == "little"
        and target.get("range") == "signed_int64",
        "timestamp_target",
        "policy target must be little-endian signed int64 Unix microseconds",
    )

    entries: list[Mapping[str, Any]] = []
    for dataset in datasets:
        dataset_policy = policy_datasets[dataset]
        timestamps = dataset_policy.get("timestamps")
        require(isinstance(timestamps, list), "timestamp_policy", "timestamps must be a list", dataset=dataset)
        output_names: set[str] = set()
        output_indices: set[int] = set()
        for entry in timestamps:
            name = str(entry.get("output_name", ""))
            try:
                index = int(entry["output_source_index"])
            except (KeyError, TypeError, ValueError):
                fail("timestamp_policy", "timestamp output index is invalid", dataset=dataset, output_name=name)
            require(name and name not in output_names, "timestamp_policy", "duplicate or empty timestamp output name", dataset=dataset, output_name=name)
            require(index not in output_indices, "timestamp_policy", "duplicate timestamp output index", dataset=dataset, source_index=index)
            output_names.add(name)
            output_indices.add(index)
            source = entry.get("source")
            rule = entry.get("timezone_rule")
            if source == "backup_string":
                timestamp_formats(entry)
            if entry.get("action") == "drop":
                require(rule == "drop_time_only", "timestamp_policy", "only time-only timestamps may be dropped", dataset=dataset, output_name=name)
                require(
                    source == "backup_string"
                    and bool(entry.get("source_columns"))
                    and bool(entry.get("blocker_reason")),
                    "timestamp_policy",
                    "dropped time-only definition is incomplete",
                    dataset=dataset,
                    output_name=name,
                )
            elif source == "backup_string":
                require(bool(entry.get("source_columns")), "timestamp_policy", "backup timestamp definition is incomplete", dataset=dataset, output_name=name)
                require(rule in {"exact_offset", "utc_z", "assume_utc_coordinate", "utc_midnight_coordinate"}, "timestamp_policy", "unsupported backup timezone rule", dataset=dataset, output_name=name)
            elif source == "prepared_numeric":
                require(
                    bool(entry.get("source_column"))
                    and entry.get("source_unit") in {"unix_epoch_seconds", "unix_epoch_milliseconds", "unix_epoch_microseconds"}
                    and rule == "unix_epoch",
                    "timestamp_policy",
                    "numeric timestamp definition is incomplete",
                    dataset=dataset,
                    output_name=name,
                )
            else:
                fail("timestamp_policy", "unsupported timestamp source", dataset=dataset, output_name=name, source=source)
            instant = bool(entry.get("event_utc_instant_claim"))
            require(not instant or rule in {"exact_offset", "utc_z", "unix_epoch"}, "timestamp_policy", "unproved UTC instant claim", dataset=dataset, output_name=name)
            entries.append(entry)

    expected_counts = {
        "timestamp_entries": len(entries),
        "convertible_if_sources_validate": sum(e.get("action") != "drop" for e in entries),
        "exact_utc_instant_claim_true": sum(bool(e.get("event_utc_instant_claim")) for e in entries),
        "coordinate_only_instant_claim_false": sum(e.get("action") != "drop" and not bool(e.get("event_utc_instant_claim")) for e in entries),
        "time_only_dropped": sum(e.get("action") == "drop" for e in entries),
    }
    counts = policy.get("policy_counts", {})
    for key, expected in expected_counts.items():
        require(int(counts.get(key, -1)) == expected, "policy_counts", "policy_counts drift", key=key, expected=expected, actual=counts.get(key))
    if "telco_entries_blocked_when_preserved_backup_is_absent" in counts:
        expected_telco = sum(
            e.get("source") == "backup_string" and e.get("action") != "drop"
            for e in policy_datasets.get("Telco", {}).get("timestamps", [])
        )
        require(
            int(counts["telco_entries_blocked_when_preserved_backup_is_absent"]) == expected_telco,
            "policy_counts",
            "Telco backup-source policy count drift",
        )
    return datasets, limit_bytes


def read_metadata_binding(
    path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    try:
        payload, stable_binding = read_verified_bytes_stable(path, "column metadata")
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline="")))
    except (OSError, UnicodeError, csv.Error) as exc:
        fail("metadata_read", "cannot read column metadata", path=str(path), error=str(exc))
    try:
        indices = [int(row["column_index"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        fail("metadata_schema", "invalid metadata column_index", path=str(path), error=str(exc))
    require(len(indices) == len(set(indices)), "metadata_index", "duplicate metadata source index", path=str(path))
    return rows, public_binding(stable_binding), stable_binding


def read_metadata(path: Path) -> list[dict[str, str]]:
    return read_metadata_binding(path)[0]


def source_index(name: str, metadata_by_name: Mapping[str, list[dict[str, str]]], fallback: int) -> int:
    generic = re.fullmatch(r"column_(\d+)", name)
    if generic:
        return int(generic.group(1))
    matches = metadata_by_name.get(name, [])
    require(len(matches) == 1, "metadata_mapping", "named prepared column must map to exactly one metadata row", column_name=name, matches=len(matches))
    return int(matches[0]["column_index"])


def epoch_us(value: datetime) -> int:
    require(value.tzinfo is not None, "timestamp_timezone", "timestamp is timezone-coordinate naive")
    utc = value.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc - epoch
    result = ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds
    require(I64_MIN <= result <= I64_MAX, "timestamp_overflow", "timestamp is outside signed int64 microseconds")
    return result


def timestamp_formats(entry: Mapping[str, Any]) -> tuple[str, ...]:
    has_format = "format" in entry
    has_formats = "formats" in entry
    require(
        has_format != has_formats,
        "timestamp_policy",
        "timestamp policy must define exactly one of format or formats",
    )
    if has_format:
        value = entry.get("format")
        require(
            isinstance(value, str) and bool(value),
            "timestamp_policy",
            "timestamp format must be a nonempty string",
        )
        return (value,)
    values = entry.get("formats")
    require(
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and bool(value) for value in values)
        and len(values) == len(set(values)),
        "timestamp_policy",
        "timestamp formats must be a nonempty unique string list",
    )
    return tuple(values)


def timestamp_format_receipt(entry: Mapping[str, Any]) -> dict[str, Any]:
    formats = timestamp_formats(entry)
    if "format" in entry:
        return {"format": formats[0]}
    return {"formats": list(formats)}


def parse_timestamp_us(entry: Mapping[str, Any], cells: list[str]) -> int:
    text = str(entry.get("join_with", "")).join(cell.strip() for cell in cells)
    errors: list[str] = []
    for timestamp_format in timestamp_formats(entry):
        try:
            value = datetime.strptime(text, timestamp_format)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        fail(
            "timestamp_parse",
            "invalid nonmissing timestamp",
            value=cells,
            errors=errors,
        )
    rule = str(entry["timezone_rule"])
    if rule == "exact_offset":
        require(value.tzinfo is not None, "timestamp_timezone", "exact_offset parsed an offset-naive value", value=cells)
    elif rule in {"utc_z", "assume_utc_coordinate", "utc_midnight_coordinate"}:
        require(value.tzinfo is None, "timestamp_timezone", "coordinate timestamp unexpectedly contains an offset", value=cells)
        if rule == "utc_midnight_coordinate":
            require(
                (value.hour, value.minute, value.second, value.microsecond) == (0, 0, 0, 0),
                "timestamp_midnight",
                "utc_midnight_coordinate must parse to exact midnight",
                value=cells,
            )
        value = value.replace(tzinfo=timezone.utc)
    else:
        fail("timestamp_policy", "unsupported string timestamp timezone rule", rule=rule)
    return epoch_us(value)


def hash_open_descriptor(descriptor: int) -> tuple[int, str]:
    """Hash a seekable open file description without allocating its contents."""

    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 8 << 20)
        if not block:
            break
        size += len(block)
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return size, digest.hexdigest()


@contextmanager
def open_backup_csv_descriptor(
    descriptor: int, path: Path, zstd_path: Path
) -> Iterator[TextIO]:
    """Parse backup CSV from an already-open descriptor; never reopen by path."""

    os.lseek(descriptor, 0, os.SEEK_SET)
    if path.suffix != ".zst":
        duplicate = os.dup(descriptor)
        wrapper = io.TextIOWrapper(os.fdopen(duplicate, "rb"), encoding="utf-8-sig", newline="")
        try:
            yield wrapper
        except (OSError, UnicodeError, csv.Error) as exc:
            fail("backup_read", "cannot read preserved backup CSV", path=str(path), error=str(exc))
        finally:
            wrapper.close()
        return
    # Feed zstd from the exact open file description that was already checked
    # and hashed above. Passing /dev/fd/N asks zstd to reopen a symlink path;
    # current server builds intentionally refuse that pathname. Popen's integer
    # stdin form dup2s this descriptor onto the child's fd 0 without a pathname
    # lookup, while preserving the shared open-file offset.
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        process = subprocess.Popen(
            [str(zstd_path), "-dc"],
            stdin=descriptor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": ""},
        )
    except OSError as exc:
        fail("missing_dependency", "explicit zstd executable could not be launched", path=str(zstd_path), error=str(exc))
    assert process.stdout is not None
    wrapper = io.TextIOWrapper(process.stdout, encoding="utf-8-sig", newline="")

    def finish() -> tuple[int, str]:
        if not wrapper.closed:
            wrapper.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        if process.stderr is not None:
            process.stderr.close()
        return process.wait(), stderr.strip()

    try:
        yield wrapper
    except AuditError as exc:
        # A failing decoder may emit no CSV or a partial/misleading header.
        # Always reap it before accepting any downstream schema/content error;
        # only a successful decoder is allowed to preserve the original error.
        code, stderr = finish()
        require(code == 0, "backup_read", "zstd decompression failed", path=str(path), stderr=stderr, exit_code=code)
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        code, stderr = finish()
        if code != 0:
            fail("backup_read", "zstd decompression failed", path=str(path), stderr=stderr, exit_code=code)
        fail("backup_read", "cannot read preserved backup CSV", path=str(path), error=str(exc))
    else:
        code, stderr = finish()
        require(code == 0, "backup_read", "zstd decompression failed", path=str(path), stderr=stderr, exit_code=code)
    finally:
        if not wrapper.closed:
            wrapper.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
        if process.stderr is not None and not process.stderr.closed:
            process.stderr.close()


def find_backup(backup_root: Path, dataset: str) -> Path | None:
    candidates: list[Path] = []
    for name in (
        f"{dataset}_removed_or_timestamp_columns.csv.zst",
        f"{dataset}_removed_or_timestamp_columns.csv",
    ):
        path = backup_root / name
        try:
            os.lstat(path)
        except FileNotFoundError:
            continue
        candidates.append(path)
    require(len(candidates) <= 1, "backup_ambiguous", "multiple preserved backup candidates exist", dataset=dataset, paths=list(map(str, candidates)))
    if not candidates:
        return None
    require_regular_file(candidates[0], "preserved backup CSV")
    return absolute_unresolved(candidates[0])


def find_metadata(backup_root: Path, dataset: str, active: Mapping[str, Any]) -> Path:
    local = backup_root / f"{dataset}_column_info.csv"
    try:
        os.lstat(local)
    except FileNotFoundError:
        pass
    else:
        require_regular_file(local, "column metadata")
        return absolute_unresolved(local)
    upstream = active.get("identity", {}).get("upstream_metadata_path")
    require(bool(upstream), "metadata_missing", "dataset has no column metadata", dataset=dataset)
    path = absolute_unresolved(Path(str(upstream)))
    require_regular_file(path, "upstream column metadata")
    return path


def compare_fields(
    actual: Mapping[str, Any], expected: Mapping[str, Any], fields: tuple[str, ...], code: str, **details: Any
) -> None:
    for field in fields:
        require(actual.get(field) == expected.get(field), code, f"field mismatch: {field}", field=field, expected=expected.get(field), actual=actual.get(field), **details)


def recompute_backup_timestamps(
    dataset: str,
    entries: list[Mapping[str, Any]],
    backup: Path,
    zstd_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("action") == "drop":
            continue
        states.append(
            {
                "entry": entry,
                "digest": hashlib.sha256(),
                "retained_rows": 0,
                "removed_missing_count": 0,
                "source_rows": 0,
            }
        )
    with open_regular_fd(backup, "preserved backup CSV") as (descriptor, before):
        backup_bytes, backup_sha256 = hash_open_descriptor(descriptor)
        stable_binding = {
            "path": str(absolute_unresolved(backup)),
            "bytes": backup_bytes,
            "sha256": backup_sha256,
            "stat_dev": before.st_dev,
            "stat_ino": before.st_ino,
        }
        backup_binding = public_binding(stable_binding)
        with open_backup_csv_descriptor(descriptor, backup, zstd_path) as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, "backup_schema", "backup CSV has no header", dataset=dataset)
            required = {str(column) for entry in entries for column in entry["source_columns"]}
            duplicates = sorted(column for column in required if reader.fieldnames.count(column) > 1)
            missing = sorted(column for column in required if column not in reader.fieldnames)
            require(not duplicates, "backup_schema", "duplicate required timestamp column", dataset=dataset, columns=duplicates)
            require(not missing, "backup_schema", "missing required timestamp column", dataset=dataset, columns=missing)
            pack = struct.Struct("<q")
            for row_number, row in enumerate(reader, 1):
                for state in states:
                    state["source_rows"] = row_number
                    entry = state["entry"]
                    cells = [(row.get(str(name)) or "").strip() for name in entry["source_columns"]]
                    if any(cell.lower() in MISSING for cell in cells):
                        state["removed_missing_count"] += 1
                        continue
                    try:
                        value = parse_timestamp_us(entry, cells)
                    except AuditError as exc:
                        exc.details.update(dataset=dataset, output_name=entry["output_name"], source_row=row_number)
                        raise
                    state["digest"].update(pack.pack(value))
                    state["retained_rows"] += 1
    results: list[dict[str, Any]] = []
    for state in states:
        entry = state["entry"]
        rows = int(state["retained_rows"])
        results.append(
            {
                "dataset": dataset,
                "column_name": str(entry["output_name"]),
                "source_index": int(entry["output_source_index"]),
                "dtype": "int64",
                "timestamp_unit": "unix_epoch_microseconds",
                "timezone_rule": str(entry["timezone_rule"]),
                "event_utc_instant_claim": bool(entry["event_utc_instant_claim"]),
                "source_kind": "backup_string",
                "source_columns": list(entry["source_columns"]),
                "status": "candidate_ready",
                "retained_rows": rows,
                "removed_missing_count": int(state["removed_missing_count"]),
                "source_rows": int(state["source_rows"]),
                "raw_bytes": rows * VALUE_BYTES,
                "raw_sha256": state["digest"].hexdigest(),
                "backup_path": str(absolute_unresolved(backup)),
                "backup_bytes": int(backup_binding["bytes"]),
                "backup_sha256": str(backup_binding["sha256"]),
                "candidate_kind": "timestamp",
            } | timestamp_format_receipt(entry)
        )
    return results, backup_binding, stable_binding


def recompute_numeric_timestamp(
    dataset: str, entry: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    dtype = str(source["dtype"])
    require(dtype in ALLOWED_DTYPES, "timestamp_source_dtype", "numeric timestamp source has an unsupported dtype", dataset=dataset, column=entry["source_column"], dtype=dtype)
    source_path = absolute_unresolved(Path(str(source["raw_path"])))
    source_bytes = int(source["raw_bytes"])
    require(source_bytes % VALUE_BYTES == 0, "timestamp_source_alignment", "numeric timestamp source is not 8-byte aligned", dataset=dataset, path=str(source_path))
    multiplier = {
        "unix_epoch_seconds": 1_000_000,
        "unix_epoch_milliseconds": 1_000,
        "unix_epoch_microseconds": 1,
    }[str(entry["source_unit"])]
    unpack = struct.Struct("<q" if dtype == "int64" else "<d")
    pack = struct.Struct("<q")
    digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    rows = 0
    with open_regular_fd(source_path, "numeric timestamp source") as (descriptor, before):
        require(before.st_size == source_bytes, "timestamp_source_hash", "numeric timestamp source size differs from manifest", dataset=dataset, path=str(source_path))
        while True:
            block = os.read(descriptor, VALUE_BYTES)
            if not block:
                break
            source_digest.update(block)
            require(len(block) == VALUE_BYTES, "timestamp_source_alignment", "partial numeric timestamp value", dataset=dataset, path=str(source_path))
            value = unpack.unpack(block)[0]
            if dtype == "float64":
                require(math.isfinite(value), "timestamp_value", "nonfinite numeric timestamp", dataset=dataset, row=rows + 1)
                numerator, denominator = value.as_integer_ratio()
                scaled = numerator * multiplier
                require(scaled % denominator == 0, "timestamp_value", "numeric timestamp is not exactly representable as microseconds", dataset=dataset, row=rows + 1)
                result = scaled // denominator
            else:
                result = int(value) * multiplier
            require(I64_MIN <= result <= I64_MAX, "timestamp_overflow", "numeric timestamp overflows signed int64 microseconds", dataset=dataset, row=rows + 1)
            digest.update(pack.pack(result))
            rows += 1
    require(source_digest.hexdigest() == str(source["raw_sha256"]), "timestamp_source_hash", "numeric timestamp source hash differs from manifest", dataset=dataset, path=str(source_path))
    return {
        "dataset": dataset,
        "column_name": str(entry["output_name"]),
        "source_index": int(entry["output_source_index"]),
        "dtype": "int64",
        "timestamp_unit": "unix_epoch_microseconds",
        "timezone_rule": "unix_epoch",
        "event_utc_instant_claim": bool(entry["event_utc_instant_claim"]),
        "source_kind": "prepared_numeric",
        "source_column": str(entry["source_column"]),
        "source_unit": str(entry["source_unit"]),
        "status": "candidate_ready",
        "retained_rows": rows,
        "removed_missing_count": 0,
        "raw_bytes": rows * VALUE_BYTES,
        "raw_sha256": digest.hexdigest(),
        "source_path": str(source_path),
        "source_bytes": source_bytes,
        "source_sha256": str(source["raw_sha256"]),
        "candidate_kind": "timestamp",
    }


def split_row_counts(rows: int) -> dict[str, int]:
    train = rows // 10
    validation = rows // 10
    return {"train": train, "validation": validation, "test": rows - train - validation}


def chunk_row_counts(rows: int, chunk_bytes: int, *, minimum_chunks: int = 1) -> list[int]:
    if rows == 0:
        return []
    maximum_rows = max(1, chunk_bytes // VALUE_BYTES)
    count = max(minimum_chunks, (rows + maximum_rows - 1) // maximum_rows)
    count = min(rows, count)
    quotient, remainder = divmod(rows, count)
    return [quotient + (index < remainder) for index in range(count)]


def candidate_projection(row: Mapping[str, Any]) -> tuple[Any, ...]:
    fields = (
        "dataset",
        "column_name",
        "source_index",
        "dtype",
        "source_kind",
        "raw_bytes",
        "raw_sha256",
        "retained_rows",
        "removed_missing_count",
        "timestamp_unit",
        "timezone_rule",
        "event_utc_instant_claim",
    )
    return tuple(row.get(field) for field in fields)


def audit_dataset(
    dataset: str,
    *,
    policy_entry: Mapping[str, Any],
    policy_sha256: str,
    producer_sha256: str,
    active_root: Path,
    backup_root: Path,
    output_root: Path,
    expected_version: str,
    limit_bytes: int,
    chunk_bytes: int,
    zstd_path: Path,
) -> tuple[dict[str, Any], dict[str, int], set[Path]]:
    dataset_dir = absolute_unresolved(output_root / "datasets" / dataset)
    require_directory(dataset_dir, "output dataset directory")
    manifest_path = dataset_dir / "manifest.json"
    manifest, manifest_binding = load_object_binding(manifest_path, "output dataset manifest")
    require(manifest.get("schema") == EXPECTED_DATASET_SCHEMA, "dataset_schema", "unsupported output dataset schema", dataset=dataset)
    require(manifest.get("version") == expected_version, "dataset_version", "dataset version differs from top manifest", dataset=dataset)
    require(manifest.get("status") == "complete", "dataset_incomplete", "dataset is not strictly complete", dataset=dataset, status=manifest.get("status"))
    require(manifest.get("dataset") == dataset, "dataset_identity", "dataset manifest identity mismatch", dataset=dataset)
    identity = manifest.get("identity")
    require(isinstance(identity, dict), "dataset_identity", "dataset identity is absent", dataset=dataset)

    active_path = absolute_unresolved(active_root / "datasets" / dataset / "manifest.json")
    require(inside(active_path, active_root), "source_containment", "active dataset manifest escapes active root", dataset=dataset)
    active, active_binding = load_object_binding(active_path, "active dataset manifest")
    prepared_value = active.get("identity", {}).get("prepared_dataset_manifest_path")
    prepared_expected_sha = str(active.get("identity", {}).get("prepared_dataset_manifest_sha256", ""))
    require(bool(prepared_value) and bool(prepared_expected_sha), "prepared_identity", "active manifest lacks prepared-manifest identity", dataset=dataset)
    prepared_path = Path(str(prepared_value))
    if not prepared_path.is_absolute():
        prepared_path = absolute_unresolved(active_path.parent / prepared_path)
    else:
        prepared_path = absolute_unresolved(prepared_path)
    require(not inside(prepared_path, output_root), "source_containment", "prepared manifest must not be inside output root", dataset=dataset)
    prepared, prepared_binding = load_object_binding(prepared_path, "prepared dataset manifest")
    require(prepared_binding["sha256"] == prepared_expected_sha, "prepared_manifest_hash", "prepared manifest differs from active identity", dataset=dataset)

    metadata_path = find_metadata(backup_root, dataset, active)
    require(inside(metadata_path, backup_root), "source_containment", "metadata must be contained by backup root", dataset=dataset, path=str(metadata_path))
    metadata, metadata_binding, metadata_stable_binding = read_metadata_binding(metadata_path)
    metadata_by_name: dict[str, list[dict[str, str]]] = {}
    for row in metadata:
        metadata_by_name.setdefault(str(row.get("column_name", "")), []).append(row)

    backup = find_backup(backup_root, dataset)
    if backup is not None:
        require(inside(backup, backup_root), "source_containment", "backup must be contained by backup root", dataset=dataset, path=str(backup))
    timestamps = list(policy_entry.get("timestamps", []))
    string_entries = [entry for entry in timestamps if entry.get("source") == "backup_string"]
    backup_timestamp_candidates: list[dict[str, Any]] = []
    backup_stable_binding: dict[str, Any] | None = None
    if string_entries:
        require(backup is not None, "timestamp_source_missing", "preserved timestamp backup is absent", dataset=dataset)
        backup_timestamp_candidates, backup_binding, backup_stable_binding = recompute_backup_timestamps(
            dataset, string_entries, backup, zstd_path
        )
    elif backup is None:
        backup_binding = None
    else:
        backup_stable_binding = stable_file_binding(backup)
        backup_binding = public_binding(backup_stable_binding)
    expected_identity = {
        "version": manifest.get("version"),
        "dataset": dataset,
        "limit_bytes": limit_bytes,
        "chunk_bytes": chunk_bytes,
        "producer_sha256": producer_sha256,
        "policy_sha256": policy_sha256,
        "dataset_policy": policy_entry,
        "active_manifest_path": str(active_path),
        "active_manifest_sha256": active_binding["sha256"],
        "prepared_manifest_path": str(prepared_path),
        "prepared_manifest_sha256": prepared_binding["sha256"],
        "backup": backup_binding,
        "metadata": metadata_binding,
    }
    for field, expected in expected_identity.items():
        require(identity.get(field) == expected, "dataset_identity", "dataset identity field mismatch", dataset=dataset, field=field, expected=expected, actual=identity.get(field))
    expected_split_spec = {
        "train_fraction": "0.1",
        "validation_fraction": "0.1",
        "test_fraction": "0.8",
        "rounding": "floor train and validation counts; test receives remainder",
        "train_chunking": "at least 3 near-equal chunks when train rows >= 3",
        "validation_test_chunking": "minimum number of near-equal max-bounded chunks",
        "max_chunk_bytes": chunk_bytes,
        "value_width_bytes": VALUE_BYTES,
    }
    require(identity.get("split_spec") == expected_split_spec, "split_spec", "dataset split/chunk identity differs", dataset=dataset)

    frozen_rows = active.get("identity", {}).get("ready_raw_columns", [])
    require(isinstance(frozen_rows, list), "prepared_identity", "active ready_raw_columns must be a list", dataset=dataset)
    frozen = {str(row["key"]): row for row in frozen_rows}
    require(len(frozen) == len(frozen_rows), "prepared_identity", "duplicate frozen prepared key", dataset=dataset)
    prepared_columns = prepared.get("columns")
    require(isinstance(prepared_columns, list), "prepared_identity", "prepared columns must be a list", dataset=dataset)
    numeric_metadata_names = {
        str(row.get("column_name", ""))
        for row in metadata
        if str(row.get("logical_type", "")).strip().lower() in ALLOWED_NUMERIC_LOGICAL_TYPES
    }
    prepared_names = [str(row.get("column_name", "")) for row in prepared_columns if isinstance(row, dict)]
    require(len(prepared_names) == len(prepared_columns), "prepared_identity", "prepared columns must be objects", dataset=dataset)
    require(len(prepared_names) == len(set(prepared_names)), "prepared_identity", "prepared column names must be unique", dataset=dataset)
    missing_numeric = sorted(numeric_metadata_names - set(prepared_names))
    require(not missing_numeric, "numeric_candidate_missing", "metadata numeric candidates are omitted from prepared manifest", dataset=dataset, columns=missing_numeric)
    prepared_numeric_names: set[str] = set()
    for row in prepared_columns:
        name = str(row.get("column_name", ""))
        dtype = str(row.get("dtype", ""))
        logical_numeric = name in numeric_metadata_names or dtype in ALLOWED_DTYPES
        if logical_numeric:
            prepared_numeric_names.add(name)
            require(row.get("status") == "ready", "numeric_nonready", "numeric prepared candidate is not ready", dataset=dataset, column_name=name, status=row.get("status"))
            require(dtype in ALLOWED_DTYPES, "payload_dtype", "numeric prepared candidate has unsupported dtype", dataset=dataset, column_name=name, dtype=dtype)
    ready = [dict(row) for row in prepared_columns if row.get("status") == "ready"]
    require(bool(ready), "prepared_empty", "prepared manifest has no ready columns", dataset=dataset)
    ready_by_key = {str(row["key"]): row for row in ready}
    require(len(ready_by_key) == len(ready) and set(ready_by_key) == set(frozen), "prepared_identity", "active and prepared ready-column inventories differ", dataset=dataset)
    ready_by_name: dict[str, dict[str, Any]] = {}
    source_bindings: dict[str, dict[str, Any]] = {}
    for row in ready:
        key = str(row["key"])
        name = str(row["column_name"])
        require(name not in ready_by_name, "prepared_identity", "duplicate prepared column name", dataset=dataset, column_name=name)
        ready_by_name[name] = row
        frozen_row = frozen[key]
        source_path = absolute_unresolved(Path(str(row["raw_path"])))
        require(inside(source_path, prepared_path.parent), "source_containment", "prepared raw column escapes prepared dataset directory", dataset=dataset, column_name=name, path=str(source_path))
        require(not inside(source_path, output_root), "source_containment", "prepared raw source overlaps output root", dataset=dataset, column_name=name)
        require(
            int(row["raw_bytes"]) == int(frozen_row["bytes"])
            and str(row["raw_sha256"]) == str(frozen_row["sha256"])
            and same_path(frozen_row["path"], source_path),
            "prepared_identity",
            "prepared column differs from frozen active identity",
            dataset=dataset,
            column_name=name,
        )
        source_binding = file_binding(source_path)
        require(
            source_binding["bytes"] == int(row["raw_bytes"])
            and source_binding["sha256"] == str(row["raw_sha256"]),
            "prepared_raw_hash",
            "prepared raw bytes differ from their frozen manifest",
            dataset=dataset,
            column_name=name,
        )
        source_bindings[name] = source_binding

    replaced = {str(name) for entry in timestamps for name in entry.get("replaces_prepared_columns", [])}
    require(replaced <= set(ready_by_name), "timestamp_replacement", "timestamp replacement names are absent from prepared inputs", dataset=dataset, missing=sorted(replaced - set(ready_by_name)))

    candidates: list[dict[str, Any]] = []
    for row in ready:
        name = str(row["column_name"])
        if name in replaced:
            continue
        dtype = str(row["dtype"])
        require(dtype in ALLOWED_DTYPES, "payload_dtype", "prepared payload dtype is not int64/float64", dataset=dataset, column_name=name, dtype=dtype)
        matches = metadata_by_name.get(name, [])
        require(len(matches) == 1, "metadata_mapping", "prepared payload must map to exactly one metadata row", dataset=dataset, column_name=name, matches=len(matches))
        logical_type = str(matches[0].get("logical_type", "")).strip().lower()
        require(
            logical_type in ALLOWED_NUMERIC_LOGICAL_TYPES,
            "logical_type",
            "non-numeric logical column was admitted as prepared payload",
            dataset=dataset,
            column_name=name,
            logical_type=logical_type,
        )
        rows = int(row["retained_rows"])
        require(int(row["raw_bytes"]) == rows * VALUE_BYTES, "prepared_row_bytes", "prepared raw byte count differs from rows*8", dataset=dataset, column_name=name)
        candidates.append(
            {
                "dataset": dataset,
                "column_name": name,
                "source_index": source_index(name, metadata_by_name, int(row["source_index"])),
                "dtype": dtype,
                "source_kind": "prepared_numeric",
                "status": "candidate_ready",
                "raw_bytes": int(row["raw_bytes"]),
                "raw_sha256": str(row["raw_sha256"]),
                "retained_rows": rows,
                "removed_missing_count": int(row.get("removed_nan_count", 0)),
                "candidate_kind": "prepared_numeric",
            }
        )

    candidates.extend(backup_timestamp_candidates)
    for entry in timestamps:
        if entry.get("source") == "prepared_numeric" and entry.get("action") != "drop":
            source_name = str(entry["source_column"])
            require(source_name in ready_by_name, "timestamp_source_missing", "prepared numeric timestamp source is absent", dataset=dataset, column_name=source_name)
            candidates.append(recompute_numeric_timestamp(dataset, entry, ready_by_name[source_name]))

    timestamp_receipts = manifest.get("timestamp_receipts")
    require(isinstance(timestamp_receipts, list), "timestamp_receipts", "timestamp receipts must be a list", dataset=dataset)
    expected_receipts: list[dict[str, Any]] = []
    candidate_timestamp_by_key = {
        (row["column_name"], int(row["source_index"])): row
        for row in candidates
        if row.get("candidate_kind") == "timestamp"
    }
    for entry in timestamps:
        key = (str(entry["output_name"]), int(entry["output_source_index"]))
        if entry.get("action") == "drop":
            expected_receipts.append(
                {
                    "dataset": dataset,
                    "column_name": key[0],
                    "source_index": key[1],
                    "dtype": "int64",
                    "timestamp_unit": "unix_epoch_microseconds",
                    "timezone_rule": str(entry["timezone_rule"]),
                    "event_utc_instant_claim": bool(entry["event_utc_instant_claim"]),
                    "source_kind": str(entry["source"]),
                    "source_columns": list(entry["source_columns"]),
                    "status": "dropped_time_only",
                    "blocker_reason": str(entry["blocker_reason"]),
                } | timestamp_format_receipt(entry)
            )
        else:
            require(key in candidate_timestamp_by_key, "timestamp_candidate", "timestamp candidate was not independently reconstructed", dataset=dataset, output_name=key[0])
            expected_receipts.append(candidate_timestamp_by_key[key])
    require(len(timestamp_receipts) == len(expected_receipts), "timestamp_receipts", "timestamp receipt count differs from policy", dataset=dataset)
    actual_receipts_by_key = {
        (str(row.get("column_name", "")), int(row.get("source_index", -1))): row
        for row in timestamp_receipts
    }
    expected_receipts_by_key = {
        (str(row["column_name"]), int(row["source_index"])): row
        for row in expected_receipts
    }
    require(
        len(actual_receipts_by_key) == len(timestamp_receipts)
        and set(actual_receipts_by_key) == set(expected_receipts_by_key),
        "timestamp_receipts",
        "timestamp receipt identity set differs from policy",
        dataset=dataset,
    )
    receipt_fields = (
        "dataset", "column_name", "source_index", "dtype", "timestamp_unit",
        "timezone_rule", "event_utc_instant_claim", "source_kind", "source_columns",
        "format", "formats", "source_column", "source_unit", "status", "retained_rows",
        "removed_missing_count", "source_rows", "raw_bytes", "raw_sha256",
        "backup_path", "backup_bytes", "backup_sha256", "source_path",
        "source_bytes", "source_sha256", "blocker_reason",
    )
    for key, expected in expected_receipts_by_key.items():
        actual = actual_receipts_by_key[key]
        compare_fields(actual, expected, receipt_fields, "timestamp_receipts", dataset=dataset, output_name=expected["column_name"])
        require(not str(actual.get("status", "")).startswith("blocked_"), "timestamp_blocked", "blocked timestamp receipt is not releasable", dataset=dataset, output_name=actual.get("column_name"))

    candidates.sort(key=lambda row: (int(row["source_index"]), str(row["column_name"])))
    candidate_indices = [int(row["source_index"]) for row in candidates]
    candidate_names = [str(row["column_name"]) for row in candidates]
    require(len(candidate_indices) == len(set(candidate_indices)), "source_index_collision", "candidate source indices are not unique", dataset=dataset)
    require(len(candidate_names) == len(set(candidate_names)), "column_name_collision", "candidate output names are not unique", dataset=dataset)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    selected_bytes = 0
    capped = False
    for row in candidates:
        size = int(row["raw_bytes"])
        if capped or selected_bytes + size > limit_bytes:
            capped = True
            excluded.append(row)
        else:
            selected.append(row)
            selected_bytes += size
    require(selected_bytes <= limit_bytes, "prefix_limit", "selected prefix exceeds 1 GiB", dataset=dataset)

    actual_columns = manifest.get("columns")
    actual_excluded = manifest.get("excluded_numeric_columns")
    require(isinstance(actual_columns, list) and isinstance(actual_excluded, list), "dataset_columns", "dataset columns/exclusions must be lists", dataset=dataset)
    require(len(actual_columns) == len(selected), "prefix_selection", "selected column count differs from independently rebuilt prefix", dataset=dataset)
    require(len(actual_excluded) == len(excluded), "prefix_selection", "excluded column count differs from independently rebuilt prefix", dataset=dataset)
    for actual, expected in zip(actual_columns, selected):
        require(candidate_projection(actual) == candidate_projection(expected), "prefix_selection", "selected candidate differs from independently rebuilt prefix", dataset=dataset, expected=[expected["source_index"], expected["column_name"]], actual=[actual.get("source_index"), actual.get("column_name")])
    for actual, expected in zip(actual_excluded, excluded):
        require(candidate_projection(actual) == candidate_projection(expected), "prefix_selection", "excluded candidate differs from independently rebuilt prefix", dataset=dataset, expected=[expected["source_index"], expected["column_name"]], actual=[actual.get("source_index"), actual.get("column_name")])
        require(actual.get("status") == "excluded_by_1g_whole_column_prefix", "prefix_selection", "excluded candidate status is invalid", dataset=dataset, column_name=actual.get("column_name"))
        require("raw_path" not in actual, "prefix_selection", "excluded receipt must not expose an output raw path", dataset=dataset, column_name=actual.get("column_name"))

    selection = manifest.get("selection", {})
    expected_selection = {
        "kind": "longest_whole_numeric_column_prefix_after_timestamp_conversion",
        "limit_bytes": limit_bytes,
        "selected_bytes": selected_bytes,
        "candidate_columns": len(candidates),
        "selected_columns": len(selected),
        "selected_source_indices": [int(row["source_index"]) for row in selected],
        "excluded_by_cap_count": len(excluded),
    }
    require(selection == expected_selection, "prefix_selection", "selection receipt differs from independently rebuilt prefix", dataset=dataset)

    expected_files: set[Path] = {absolute_unresolved(manifest_path)}
    output_stable_bindings: list[dict[str, Any]] = []
    chunk_total = 0
    chunk_bytes_total = 0
    for active_index, (column, candidate) in enumerate(zip(actual_columns, selected)):
        require(column.get("status") == "ready" and int(column.get("active_index", -1)) == active_index, "column_identity", "selected column status/index is invalid", dataset=dataset, column_name=column.get("column_name"))
        raw_path = absolute_unresolved(dataset_dir / f"column_{int(candidate['source_index']):04d}.{candidate['dtype']}.bin")
        require(same_path(column.get("raw_path"), raw_path), "column_path", "selected raw path is not canonical", dataset=dataset, column_name=candidate["column_name"])
        require(inside(raw_path, dataset_dir), "column_path", "selected raw path escapes its dataset directory", dataset=dataset, path=str(raw_path))
        require_regular_file(raw_path, "selected raw column")
        raw_stable_binding = stable_file_binding(raw_path)
        raw_size = int(raw_stable_binding["bytes"])
        raw_sha = str(raw_stable_binding["sha256"])
        require(raw_size == int(candidate["raw_bytes"]) and raw_sha == str(candidate["raw_sha256"]), "raw_hash", "selected raw bytes/hash differ from independently rebuilt candidate", dataset=dataset, column_name=candidate["column_name"])
        output_stable_bindings.append(raw_stable_binding)
        expected_files.add(raw_path)

        rows = int(candidate["retained_rows"])
        splits = split_row_counts(rows)
        require(column.get("split_rows") == splits, "chunk_rows", "split row receipt is invalid", dataset=dataset, column_name=candidate["column_name"])
        expected_chunk_rows: list[tuple[str, int, int]] = []
        for split in ("train", "validation", "test"):
            minimum = 3 if split == "train" and splits[split] >= 3 else 1
            for index, count in enumerate(chunk_row_counts(splits[split], chunk_bytes, minimum_chunks=minimum)):
                expected_chunk_rows.append((split, index, count))
        chunks = column.get("chunks")
        require(isinstance(chunks, list) and len(chunks) == len(expected_chunk_rows), "chunk_count", "chunk count differs from independent split", dataset=dataset, column_name=candidate["column_name"])
        concat = hashlib.sha256()
        source_start = 0
        for chunk, (split, index, rows_in_chunk) in zip(chunks, expected_chunk_rows):
            chunk_path = absolute_unresolved(dataset_dir / "chunks" / f"column_{int(candidate['source_index']):04d}" / f"{split}-{index:05d}.bin")
            expected_chunk = {
                "split": split,
                "index": index,
                "rows": rows_in_chunk,
                "bytes": rows_in_chunk * VALUE_BYTES,
                "path": str(chunk_path),
                "source_start_row": source_start,
                "source_end_row_exclusive": source_start + rows_in_chunk,
            }
            compare_fields(chunk, expected_chunk, tuple(expected_chunk), "chunk_receipt", dataset=dataset, column_name=candidate["column_name"])
            require(inside(chunk_path, dataset_dir), "chunk_path", "chunk path escapes its dataset directory", dataset=dataset, path=str(chunk_path))
            require_regular_file(chunk_path, "chunk file")
            chunk_stable_binding = stable_file_binding(chunk_path, tee=concat)
            chunk_size = int(chunk_stable_binding["bytes"])
            chunk_sha = str(chunk_stable_binding["sha256"])
            require(chunk_size == expected_chunk["bytes"] and chunk_sha == str(chunk.get("sha256", "")), "chunk_hash", "chunk bytes/hash mismatch", dataset=dataset, column_name=candidate["column_name"], split=split, index=index)
            output_stable_bindings.append(chunk_stable_binding)
            expected_files.add(chunk_path)
            source_start += rows_in_chunk
            chunk_total += 1
            chunk_bytes_total += chunk_size
        require(source_start == rows and concat.hexdigest() == raw_sha, "chunk_raw_mismatch", "ordered chunks do not reconstruct the raw column", dataset=dataset, column_name=candidate["column_name"])

    consumed = {str(name) for entry in timestamps for name in entry.get("source_columns", [])}
    expected_string_receipts = []
    for row in metadata:
        logical_type = str(row.get("logical_type", "")).strip().lower()
        name = str(row.get("column_name", ""))
        if logical_type in RECEIPTED_NONNUMERIC_TYPES and name not in consumed:
            expected_string_receipts.append(
                {
                    "column_name": name,
                    "source_index": int(row["column_index"]),
                    "status": "dropped_remaining_string",
                    "logical_type": row.get("logical_type"),
                }
            )
    require(manifest.get("string_receipts") == expected_string_receipts, "string_receipts", "remaining string receipt inventory differs from metadata", dataset=dataset)

    timestamp_admitted = sum(row.get("candidate_kind") == "timestamp" for row in candidates)
    timestamp_selected = sum(row.get("candidate_kind") == "timestamp" for row in selected)
    time_only_dropped = sum(entry.get("action") == "drop" for entry in timestamps)
    counts = {
        "timestamp_admitted": timestamp_admitted,
        "timestamp_selected": timestamp_selected,
        "timestamp_blocked": 0,
        "time_only_dropped": time_only_dropped,
        "remaining_strings_dropped": len(expected_string_receipts),
        "numeric_candidates": len(candidates),
        "numeric_selected": len(selected),
        "chunks": chunk_total,
        "chunk_bytes": chunk_bytes_total,
    }
    require(manifest.get("counts") == counts, "dataset_counts", "dataset counts differ from independent totals", dataset=dataset, expected=counts, actual=manifest.get("counts"))

    summary = {
        "dataset": dataset,
        "status": "PASS",
        "output_manifest": manifest_binding,
        "active_manifest": active_binding,
        "prepared_manifest": prepared_binding,
        "prepared_sources": [source_bindings[name] for name in sorted(source_bindings)],
        "metadata": metadata_binding,
        "metadata_stable_binding": metadata_stable_binding,
        "output_stable_bindings": output_stable_bindings,
        "backup": backup_binding,
        "backup_stable_binding": backup_stable_binding,
        "prepared_numeric_columns": len(prepared_numeric_names),
        "prepared_ready_columns": len(ready),
        "candidate_columns": len(candidates),
        "selected_columns": len(selected),
        "excluded_columns": len(excluded),
        "selected_bytes": selected_bytes,
        "selected_source_indices": [int(row["source_index"]) for row in selected],
        "timestamp_admitted": timestamp_admitted,
        "timestamp_selected": timestamp_selected,
        "timestamp_blocked": 0,
        "chunks": chunk_total,
        "chunk_bytes": chunk_bytes_total,
    }
    return summary, counts, expected_files


def scan_output_tree(output_root: Path) -> tuple[set[Path], set[Path]]:
    """Return an exact no-follow snapshot while rejecting unsafe node types."""

    actual_files: set[Path] = set()
    actual_directories: set[Path] = {output_root}
    pending = [output_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                path = absolute_unresolved(entry.path)
                require(not stat.S_ISLNK(info.st_mode), "symlink_rejected", "output tree contains a symlink", path=str(path))
                if stat.S_ISDIR(info.st_mode):
                    actual_directories.add(path)
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    require(info.st_nlink == 1, "hardlink_rejected", "output tree contains a hard-linked file", path=str(path), link_count=info.st_nlink)
                    actual_files.add(path)
                else:
                    fail("special_node_rejected", "output tree contains a special filesystem node", path=str(path))
    return actual_files, actual_directories


def require_exact_output_tree(
    actual_files: set[Path],
    actual_directories: set[Path],
    expected_files: set[Path],
    expected_directories: set[Path],
) -> None:
    missing_files = sorted(map(str, expected_files - actual_files))
    extra_files = sorted(map(str, actual_files - expected_files))
    require(not missing_files and not extra_files, "output_file_set", "output eligible file set differs from manifests", missing=missing_files, extra=extra_files)
    require(actual_directories == expected_directories, "output_directory_set", "output directory set differs from exact manifested tree", missing=sorted(map(str, expected_directories - actual_directories)), extra=sorted(map(str, actual_directories - expected_directories)))


def audit(
    *,
    scope_path: Path | str,
    policy_path: Path | str,
    producer_path: Path | str,
    active_root: Path | str,
    backup_root: Path | str,
    output_root: Path | str,
    zstd_path: Path | str,
) -> dict[str, Any]:
    scope_path = reject_symlink_components(scope_path, "scope")
    policy_path = reject_symlink_components(policy_path, "policy")
    producer_path = reject_symlink_components(producer_path, "producer")
    active_root = require_directory(active_root, "active root")
    backup_root = require_directory(backup_root, "backup root")
    output_root = require_directory(output_root, "output root")
    zstd_argument = Path(zstd_path)
    require(zstd_argument.is_absolute(), "zstd_path", "zstd executable path must be explicit and absolute", path=str(zstd_argument))
    zstd_path = reject_symlink_components(zstd_argument, "zstd executable")
    require_regular_file(zstd_path, "zstd executable")
    require(os.access(zstd_path, os.X_OK), "zstd_path", "zstd executable is not executable", path=str(zstd_path))
    roots = (("active", active_root), ("backup", backup_root), ("output", output_root))
    for index, (left_name, left) in enumerate(roots):
        for right_name, right in roots[index + 1 :]:
            require(
                not inside(left, right) and not inside(right, left),
                "root_overlap",
                "active, backup, and output roots must be disjoint",
                left=left_name,
                right=right_name,
            )

    scope, scope_binding = load_object_binding(scope_path, "authoritative scope")
    policy, policy_binding = load_object_binding(policy_path, "Core19 UTC policy")
    require(scope_binding["sha256"] == EXPECTED_SCOPE_SHA256, "authority_scope_hash", "scope is not the pinned project179 authority", expected=EXPECTED_SCOPE_SHA256, actual=scope_binding["sha256"])
    require(policy_binding["sha256"] == EXPECTED_POLICY_SHA256, "authority_policy_hash", "policy is not the pinned Core19 UTC policy", expected=EXPECTED_POLICY_SHA256, actual=policy_binding["sha256"])
    datasets, limit_bytes = validate_scope_and_policy(scope, policy)
    producer_binding = file_binding(producer_path)
    auditor_binding = file_binding(absolute_unresolved(Path(__file__)))
    zstd_binding = file_binding(zstd_path)
    require(producer_binding["sha256"] == EXPECTED_PRODUCER_SHA256, "authority_producer_hash", "producer is not the pinned formal producer", expected=EXPECTED_PRODUCER_SHA256, actual=producer_binding["sha256"])
    require(str(zstd_path) == EXPECTED_ZSTD_PATH, "authority_zstd_path", "zstd executable path is not the pinned 166 path", expected=EXPECTED_ZSTD_PATH, actual=str(zstd_path))
    require(zstd_binding["sha256"] == EXPECTED_ZSTD_SHA256, "authority_zstd_hash", "zstd executable is not the pinned 166 binary", expected=EXPECTED_ZSTD_SHA256, actual=zstd_binding["sha256"])
    control_bindings = [scope_binding, policy_binding, producer_binding, auditor_binding, zstd_binding]
    control_paths = [Path(str(row["path"])) for row in control_bindings]
    require(len(set(control_paths)) == len(control_paths), "input_alias", "control inputs and executable paths must be distinct")
    for control_path in control_paths:
        require(
            not any(inside(control_path, root) for _name, root in roots),
            "input_overlap",
            "control inputs, executables, and auditor must be outside all source/output roots",
            path=str(control_path),
        )

    source_contract_path = active_root / "SOURCE_CONTRACT.json"
    source_top_path = active_root / "manifest.json"
    source_contract, source_contract_binding = load_object_binding(source_contract_path, "active source contract")
    source_top, source_top_binding = load_object_binding(source_top_path, "active top manifest")
    require(source_contract.get("status") == "complete", "active_contract", "active source contract is incomplete")
    require(source_contract.get("schema") == "core19-1g-column-prefix-source-contract-v1", "active_contract", "unsupported SOURCE_CONTRACT schema")
    require(int(source_contract.get("dataset_count", -1)) == 19 and source_contract.get("datasets") == datasets, "active_contract", "active source contract identity differs from authoritative scope")
    derived_scope = source_contract.get("derived_scope", {})
    require(isinstance(derived_scope, dict), "active_contract", "SOURCE_CONTRACT derived scope is absent")
    require(derived_scope.get("excluded_datasets") == ["poisson_timestamps"], "active_contract", "active source contract must exclude exactly poisson_timestamps")
    require(
        derived_scope.get("scope_id") == scope.get("scope_id")
        and same_path(derived_scope.get("scope_path"), scope_path)
        and derived_scope.get("scope_sha256") == scope_binding["sha256"],
        "active_scope_binding",
        "SOURCE_CONTRACT derived scope does not bind the explicit authoritative scope",
    )
    source_dataset_hashes = source_contract.get("dataset_manifest_hashes")
    require(
        isinstance(source_dataset_hashes, dict)
        and set(source_dataset_hashes) == set(datasets),
        "active_contract",
        "active source contract dataset-manifest hash inventory differs from Core19",
    )
    require(
        same_path(source_contract.get("active_manifest_path"), source_top_path)
        and source_contract.get("active_manifest_sha256") == source_top_binding["sha256"],
        "active_contract",
        "active top manifest binding differs from SOURCE_CONTRACT",
    )
    contract_selection = source_contract.get("selection")
    require(isinstance(contract_selection, dict) and set(contract_selection) == set(datasets), "active_contract", "SOURCE_CONTRACT selection inventory differs from Core19")
    contract_totals = source_contract.get("totals")
    require(isinstance(contract_totals, dict), "active_contract", "SOURCE_CONTRACT totals are absent")
    require(int(contract_totals.get("blocking_source_numeric_failures", -1)) == 0, "numeric_nonready", "SOURCE_CONTRACT reports blocking numeric failures")
    require(int(contract_totals.get("provenance_only_nonready_entries", -1)) == 0, "numeric_nonready", "SOURCE_CONTRACT reports omitted/nonready numeric entries")
    require(source_top.get("totals") == contract_totals, "active_top", "active top totals differ from SOURCE_CONTRACT")
    source_top_derived = source_top.get("derived_scope") or source_top.get("identity", {}).get("derived_scope")
    require(source_top_derived == derived_scope, "active_top", "active top derived scope differs from SOURCE_CONTRACT")
    source_top_entries = source_top.get("datasets")
    require(isinstance(source_top_entries, list), "active_top", "active top dataset inventory must be a list")
    require([row.get("dataset") for row in source_top_entries if isinstance(row, dict)] == datasets, "active_top", "active top dataset order/identity differs from Core19")
    source_top_by_dataset = {str(row["dataset"]): row for row in source_top_entries}

    active_expected_top = {"SOURCE_CONTRACT.json", "PREVIEW_PROTOCOL.json", "manifest.json", "datasets"}
    require({entry.name for entry in os.scandir(active_root)} == active_expected_top, "active_tree_set", "active root contains unexpected or missing entries")
    require_regular_file(active_root / "PREVIEW_PROTOCOL.json", "active preview protocol")
    require_directory(active_root / "datasets", "active datasets root")
    require({entry.name for entry in os.scandir(active_root / "datasets")} == set(datasets), "active_tree_set", "active dataset directory set differs from Core19")
    for dataset in datasets:
        dataset_source_dir = require_directory(active_root / "datasets" / dataset, "active dataset directory")
        require({entry.name for entry in os.scandir(dataset_source_dir)} == {"manifest.json"}, "active_tree_set", "active dataset directory contains unexpected entries", dataset=dataset)

    top_path = output_root / "manifest.json"
    top, top_binding = load_object_binding(top_path, "output top manifest")
    require(top.get("schema") == EXPECTED_TOP_SCHEMA, "top_schema", "unsupported output top schema")
    require(top.get("status") == "complete", "top_incomplete", "output top manifest is not complete", status=top.get("status"))
    require(int(top.get("dataset_count", -1)) == 19 and top.get("datasets") == datasets, "top_identity", "output top dataset identity/order differs from scope")
    require(top.get("policy_sha256") == policy_binding["sha256"] and same_path(top.get("policy_path"), policy_path), "top_policy", "output top policy binding differs from explicit policy")
    require(same_path(top.get("active_root"), active_root) and same_path(top.get("backup_root"), backup_root), "top_roots", "output top source roots differ from explicit roots")
    require(int(top.get("limit_bytes", -1)) == limit_bytes, "top_limit", "output top byte limit differs from scope")
    chunk_bytes = int(top.get("chunk_bytes", 0))
    require(chunk_bytes >= VALUE_BYTES and chunk_bytes % VALUE_BYTES == 0, "top_chunk_bytes", "chunk byte limit must be a positive multiple of 8")
    require(top.get("producer", {}).get("sha256") == producer_binding["sha256"] and same_path(top.get("producer", {}).get("path"), producer_path), "top_producer", "output top producer binding differs from explicit producer")

    top_entries = top.get("dataset_manifests")
    require(isinstance(top_entries, list) and [row.get("dataset") for row in top_entries] == datasets, "top_dataset_receipts", "top dataset receipt order/identity differs from scope")
    output_datasets_root = require_directory(output_root / "datasets", "output datasets directory")
    actual_dataset_entries = {entry.name for entry in os.scandir(output_datasets_root)}
    require(actual_dataset_entries == set(datasets), "output_dataset_set", "output dataset directory set differs from exact Core19", expected=sorted(datasets), actual=sorted(actual_dataset_entries))
    require({entry.name for entry in os.scandir(output_root)} == {"manifest.json", "datasets"}, "output_top_set", "output root contains unexpected top-level entries")

    summaries: list[dict[str, Any]] = []
    totals = {key: 0 for key in COUNT_KEYS}
    expected_files: set[Path] = {absolute_unresolved(top_path)}
    top_by_dataset = {str(row["dataset"]): row for row in top_entries}
    require(len(top_by_dataset) == 19, "top_dataset_receipts", "top dataset receipts contain duplicate identities")
    policy_datasets = policy["datasets"]
    for dataset in datasets:
        summary, counts, files = audit_dataset(
            dataset,
            policy_entry=policy_datasets[dataset],
            policy_sha256=policy_binding["sha256"],
            producer_sha256=producer_binding["sha256"],
            active_root=active_root,
            backup_root=backup_root,
            output_root=output_root,
            expected_version=str(top.get("version", "")),
            limit_bytes=limit_bytes,
            chunk_bytes=chunk_bytes,
            zstd_path=zstd_path,
        )
        summaries.append(summary)
        expected_files.update(files)
        require(
            source_dataset_hashes.get(dataset) == summary["active_manifest"]["sha256"],
            "active_contract",
            "active dataset manifest hash differs from SOURCE_CONTRACT",
            dataset=dataset,
        )
        active_entry = source_top_by_dataset[dataset]
        require(
            same_path(active_entry.get("manifest_path"), active_root / "datasets" / dataset / "manifest.json")
            and active_entry.get("manifest_sha256") == summary["active_manifest"]["sha256"],
            "active_top",
            "active top dataset manifest binding is invalid",
            dataset=dataset,
        )
        selection_entry = contract_selection[dataset]
        require(isinstance(selection_entry, dict), "active_contract", "SOURCE_CONTRACT dataset selection must be an object", dataset=dataset)
        source_columns = int(selection_entry.get("source_columns", -1))
        selected_columns = int(selection_entry.get("selected_columns", -1))
        selected_indices = selection_entry.get("selected_source_indices")
        require(
            source_columns == int(summary["prepared_numeric_columns"])
            and 0 <= selected_columns <= source_columns
            and isinstance(selected_indices, list)
            and len(selected_indices) == selected_columns
            and len(set(map(int, selected_indices))) == selected_columns,
            "active_contract",
            "SOURCE_CONTRACT selection does not account for every prepared numeric candidate",
            dataset=dataset,
        )
        active_totals = active_entry.get("totals")
        require(isinstance(active_totals, dict), "active_top", "active top dataset totals are absent", dataset=dataset)
        require(int(active_totals.get("blocking_source_numeric_failures", -1)) == 0, "numeric_nonready", "active top reports blocking numeric failures", dataset=dataset)
        require(int(active_totals.get("provenance_only_nonready_entries", -1)) == 0, "numeric_nonready", "active top reports omitted/nonready numeric entries", dataset=dataset)
        for key in COUNT_KEYS:
            totals[key] += int(counts[key])
        top_entry = top_by_dataset[dataset]
        manifest_binding = summary["output_manifest"]
        expected_top_entry = {
            "dataset": dataset,
            "path": manifest_binding["path"],
            "sha256": manifest_binding["sha256"],
            "status": "complete",
            "counts": counts,
        }
        require(top_entry == expected_top_entry, "top_dataset_receipts", "top dataset receipt differs from audited dataset manifest", dataset=dataset)
    require(top.get("totals") == totals, "top_totals", "top totals differ from independent dataset totals", expected=totals, actual=top.get("totals"))

    expected_directories = {output_root}
    for path in expected_files:
        parent = path.parent
        while inside(parent, output_root):
            expected_directories.add(parent)
            if parent == output_root:
                break
            parent = parent.parent
    for summary in summaries:
        if int(summary["timestamp_admitted"]) > 0:
            expected_directories.add(output_root / "datasets" / str(summary["dataset"]) / ".staging")
    first_tree_files, first_tree_directories = scan_output_tree(output_root)
    require_exact_output_tree(first_tree_files, first_tree_directories, expected_files, expected_directories)

    stable_bindings = control_bindings + [source_contract_binding, source_top_binding, top_binding]
    for summary in summaries:
        stable_bindings.extend(
            binding
            for binding in (
                summary["output_manifest"],
                summary["active_manifest"],
                summary["prepared_manifest"],
                summary["metadata"],
                summary["backup"],
            )
            if binding is not None
        )
        stable_bindings.extend(summary["prepared_sources"])
    for binding in stable_bindings:
        require(
            file_binding(Path(str(binding["path"]))) == binding,
            "file_changed",
            "a bound input or manifest changed during the audit",
            path=binding["path"],
        )
    stable_stat_bindings = [summary["metadata_stable_binding"] for summary in summaries]
    for summary in summaries:
        stable_stat_bindings.extend(summary["output_stable_bindings"])
        if summary["backup_stable_binding"] is not None:
            stable_stat_bindings.append(summary["backup_stable_binding"])
    for binding in stable_stat_bindings:
        require(
            stable_file_binding(Path(str(binding["path"]))) == binding,
            "file_changed",
            "an output payload or metadata file changed after its first verified read",
            path=binding["path"],
        )
    final_tree_files, final_tree_directories = scan_output_tree(output_root)
    require(
        final_tree_files == first_tree_files and final_tree_directories == first_tree_directories,
        "output_tree_changed",
        "output tree changed after the first exact scan",
        first_files=sorted(map(str, first_tree_files)),
        final_files=sorted(map(str, final_tree_files)),
        first_directories=sorted(map(str, first_tree_directories)),
        final_directories=sorted(map(str, final_tree_directories)),
    )
    require_exact_output_tree(final_tree_files, final_tree_directories, expected_files, expected_directories)

    receipt = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "mode": "cold_read_only",
        "scope_id": scope.get("scope_id"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_bindings": {
            "scope": scope_binding,
            "policy": policy_binding,
            "producer": producer_binding,
            "auditor": auditor_binding,
            "zstd": zstd_binding,
            "active_source_contract": source_contract_binding,
            "active_top_manifest": source_top_binding,
        },
        "output_binding": {
            "root": str(output_root),
            "top_manifest": top_binding,
            "eligible_file_count": len(expected_files),
        },
        "dataset_count": 19,
        "datasets": summaries,
        "totals": totals,
        "release_checks": {
            "exact_ordered_core19": True,
            "pinned_scope_policy_producer_and_zstd": True,
            "poisson_timestamps_rejected": True,
            "logical_payload_integer_float_or_explicit_timestamp_only": True,
            "timestamp_signed_int64_unix_microseconds": True,
            "timestamp_blocked": 0,
            "longest_whole_column_prefix_per_dataset": True,
            "limit_bytes_per_dataset": limit_bytes,
            "all_source_output_and_chunk_hashes_streamed": True,
            "ordered_chunks_reconstruct_raw": True,
            "strict_output_eligible_file_set": True,
            "output_payload_bindings_stable_through_audit_end": True,
            "metadata_single_fd_hash_parse_and_final_recheck": True,
            "backup_single_fd_raw_hash_csv_parse_and_final_recheck": True,
            "final_output_tree_rescan_after_all_stable_bindings": True,
        },
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zstd", type=Path, required=True, help="Absolute path to a frozen zstd executable")
    parser.add_argument("--receipt", type=Path, help="Optional independent receipt path outside output-root")
    args = parser.parse_args()
    output_root = absolute_unresolved(args.output_root)
    if args.receipt is not None:
        receipt_path = absolute_unresolved(args.receipt)
        protected_roots = [absolute_unresolved(path) for path in (args.active_root, args.backup_root, args.output_root)]
        protected_files = [
            absolute_unresolved(path)
            for path in (args.scope, args.policy, args.producer, args.zstd, Path(__file__))
        ]
        if any(inside(receipt_path, root) for root in protected_roots) or receipt_path in protected_files:
            print(json.dumps({"schema": AUDIT_SCHEMA, "status": "FAIL", "error": {"code": "unsafe_receipt_path", "message": "receipt path must not overlap any input, source, output, control, executable, or auditor path", "path": str(receipt_path)}}, sort_keys=True))
            return 2
        try:
            reject_symlink_components(receipt_path, "receipt", allow_missing_leaf=True)
            if os.path.lexists(receipt_path):
                print(json.dumps({"schema": AUDIT_SCHEMA, "status": "FAIL", "error": {"code": "receipt_exists", "message": "refusing to replace an existing receipt", "path": str(receipt_path)}}, sort_keys=True))
                return 2
        except AuditError as exc:
            print(json.dumps({"schema": AUDIT_SCHEMA, "status": "FAIL", "error": {"code": exc.code, "message": exc.message, "details": exc.details}}, sort_keys=True))
            return 2
    try:
        result = audit(
            scope_path=args.scope,
            policy_path=args.policy,
            producer_path=args.producer,
            active_root=args.active_root,
            backup_root=args.backup_root,
            output_root=args.output_root,
            zstd_path=args.zstd,
        )
        if args.receipt is not None:
            receipt_path = absolute_unresolved(args.receipt)
            atomic_json_no_clobber(receipt_path, result)
            result = result | {
                "written_receipt": file_binding(receipt_path),
            }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except AuditError as exc:
        print(
            json.dumps(
                {
                    "schema": AUDIT_SCHEMA,
                    "status": "FAIL",
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": AUDIT_SCHEMA,
                    "status": "FAIL",
                    "error": {
                        "code": "unexpected_audit_error",
                        "message": "unexpected exception while auditing; fail closed",
                        "details": {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
