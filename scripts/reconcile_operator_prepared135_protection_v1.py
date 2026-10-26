#!/usr/bin/env python3
"""Accept one frozen prepared135 protection failure caused only by kworker affinity.

This is a read-only reconciliation of the completed measurement evidence and
the live identities of the two kernel threads named by PROTECTION_POST.json.
It does not rerun a measurement, alter a process, replace DONE.json, or rewrite
the original failure/protection receipts.  On success it creates only
RECOVERY_ACCEPTANCE.json under the supplied run root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
from typing import Any, Callable


SCHEMA = "operator-prepared135-protection-recovery-acceptance-v1"
EXPECTED_COLUMNS = 3519
EXPECTED_DATASETS = 135
EXPECTED_STAGES = (
    "BUILD",
    "REPLAY360",
    "FIT_DEVELOPMENT",
    "WITNESS_COLUMNS",
    "COLUMNS3519",
)
EXPECTED_SOURCE_FILES = 20
EXPECTED_PROTECTED_FILES = 3817
EXPECTED_CATALOG_SHA256 = (
    "568f2422b415d2869712a4a0e5f8edb37205555d92ecfe3fc0b36946c1d56326"
)
PF_KTHREAD = 0x00200000
EXPECTED_KERNEL_FLAGS = 69238880
TWO_GIB = 2 * 1024 * 1024 * 1024
DEFAULT_SCHED_HEADER = Path(
    "/usr/src/linux-headers-6.8.0-147-generic/include/linux/sched.h"
)
EXPECTED_KERNEL_CHANGES = {
    (12, "131", "kworker/R-mm_pe"): ([61], [63]),
    (898, "955", "kworker/R-dm-ca"): ([61], [6]),
}
EXPECTED_EXCLUDED_ACTION_IDS = [106, 107, 108, 109, 110]


class ReconciliationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path, label: str) -> Any:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconciliationError(f"cannot read {label} {path}: {error}") from error


def mapping_of_hashes(value: Any, label: str) -> dict[str, str]:
    require(isinstance(value, dict), f"{label} must be an object")
    result: dict[str, str] = {}
    for path, digest in value.items():
        require(isinstance(path, str) and path, f"{label} contains an invalid path")
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"{label} has an invalid SHA-256 for {path}",
        )
        result[path] = digest
    return result


def verify_hashes(expected: dict[str, str], label: str) -> None:
    mismatches = []
    for path_text, expected_sha in expected.items():
        path = Path(path_text)
        if not path.is_file():
            mismatches.append({"path": path_text, "reason": "missing"})
            continue
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            mismatches.append(
                {
                    "path": path_text,
                    "reason": "sha256_mismatch",
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )
    require(not mismatches, f"{label} changed: {mismatches}")


def parse_status(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing live process status: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key] = value.strip()
    return result


def parse_stat(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing live process stat: {path}")
    value = path.read_text(encoding="utf-8").strip()
    close = value.rfind(")")
    require(close > 0, f"malformed live process stat: {path}")
    fields = value[close + 2 :].split()
    require(len(fields) > 19, f"truncated live process stat: {path}")
    return {
        "state": fields[0],
        "ppid": int(fields[1]),
        "flags": int(fields[6]),
        "start_ticks": fields[19],
    }


def read_pf_kthread_definition(header: Path) -> dict[str, Any]:
    require(header.is_file(), f"missing Linux scheduler header: {header}")
    header_bytes = header.read_bytes()
    text = header_bytes.decode("utf-8")
    matches = [
        line.strip()
        for line in text.splitlines()
        if re.fullmatch(r"\s*#define\s+PF_KTHREAD\s+0x00200000(?:\s+.*)?", line)
    ]
    require(
        len(matches) == 1,
        f"Linux scheduler header does not define PF_KTHREAD exactly once as 0x00200000: {header}",
    )
    return {
        "path": str(header),
        "sha256": hashlib.sha256(header_bytes).hexdigest(),
        "definition": matches[0],
        "value_hex": f"0x{PF_KTHREAD:08x}",
    }


def probe_live_process(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    directory = proc_root / str(pid)
    first_stat = parse_stat(directory / "stat")
    status = parse_status(directory / "status")
    comm = (directory / "comm").read_text(encoding="utf-8").rstrip("\n")
    cmdline = (directory / "cmdline").read_bytes()
    exe_path = directory / "exe"
    try:
        exe_target: str | None = os.readlink(exe_path)
    except FileNotFoundError:
        exe_target = None
    except OSError as error:
        raise ReconciliationError(f"cannot inspect live process executable {exe_path}: {error}") from error
    second_stat = parse_stat(directory / "stat")
    require(
        all(first_stat[key] == second_stat[key] for key in ("ppid", "flags", "start_ticks")),
        f"live process identity changed during inspection: {pid}",
    )
    try:
        affinity = sorted(os.sched_getaffinity(pid))
        scheduler = os.sched_getscheduler(pid)
        nice = os.getpriority(os.PRIO_PROCESS, pid)
    except (ProcessLookupError, PermissionError, OSError) as error:
        raise ReconciliationError(f"cannot inspect live process scheduling for PID {pid}: {error}") from error
    return {
        "pid": pid,
        "start_ticks": first_stat["start_ticks"],
        "comm": comm,
        "state_before": first_stat["state"],
        "state_after": second_stat["state"],
        "ppid": first_stat["ppid"],
        "flags": first_stat["flags"],
        "flags_hex": f"0x{first_stat['flags']:08x}",
        "pf_kthread_set": bool(first_stat["flags"] & PF_KTHREAD),
        "status_name": status.get("Name"),
        "status_kthread": status.get("Kthread"),
        "status_ppid": status.get("PPid"),
        "cmdline_bytes": len(cmdline),
        "exe_absent": exe_target is None,
        "exe_target": exe_target,
        "affinity": affinity,
        "scheduler": scheduler,
        "nice": nice,
    }


def execution_profile() -> dict[str, Any]:
    profile = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "scheduler": os.sched_getscheduler(0),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "ionice": subprocess.check_output(
            ["ionice", "-p", str(os.getpid())], text=True
        ).strip(),
        "address_space_limit_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
    }
    require(profile["hostname"] == "timecho-20", "reconciliation must run on server 192")
    require(profile["affinity"] == [38], f"reconciliation CPU affinity mismatch: {profile}")
    require(profile["scheduler"] == os.SCHED_IDLE, f"reconciliation scheduler mismatch: {profile}")
    require(profile["nice"] == 19, f"reconciliation nice mismatch: {profile}")
    require(profile["ionice"] == "idle", f"reconciliation I/O priority mismatch: {profile}")
    require(
        profile["address_space_limit_bytes"] == [TWO_GIB, TWO_GIB],
        f"reconciliation address-space limit mismatch: {profile}",
    )
    return profile


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }


def verify_kernel_changes(
    changes: Any,
    observed_jobs: Any,
    process_probe: Callable[[int], dict[str, Any]],
    *,
    allow_affinity_refresh: bool = False,
) -> list[dict[str, Any]]:
    require(isinstance(changes, list), "PROTECTION_POST.settings_changed must be an array")
    if allow_affinity_refresh:
        require(
            0 < len(changes) <= len(EXPECTED_KERNEL_CHANGES),
            "kernel affinity refresh must contain one or two bound identities",
        )
    else:
        require(
            len(changes) == len(EXPECTED_KERNEL_CHANGES),
            "unexpected settings_changed count",
        )
    require(isinstance(observed_jobs, list), "PROTECTION_PRE.observed_jobs must be an array")
    observed = {
        (job.get("pid"), str(job.get("start_ticks")), job.get("comm")): job
        for job in observed_jobs
        if isinstance(job, dict)
    }
    evidence = []
    seen: set[tuple[int, str, str]] = set()
    for position, change in enumerate(changes):
        require(isinstance(change, dict), f"settings_changed[{position}] is not an object")
        require(set(change) == {"before", "after"}, f"settings_changed[{position}] has unexpected fields")
        before = change["before"]
        after = change["after"]
        require(
            isinstance(before, dict) and isinstance(after, dict),
            f"settings_changed[{position}] before/after must be objects",
        )
        identity = (before.get("pid"), str(before.get("start_ticks")), before.get("comm"))
        require(identity in EXPECTED_KERNEL_CHANGES, f"unexpected changed process identity: {identity}")
        require(identity not in seen, f"duplicate changed process identity: {identity}")
        seen.add(identity)
        require(observed.get(identity) == before, f"raw before evidence differs from PROTECTION_PRE for {identity}")
        require(
            (after.get("pid"), str(after.get("start_ticks")), after.get("comm")) == identity,
            f"raw process identity changed for {identity}",
        )
        require(
            changed_fields(before, after) == {"affinity"},
            f"raw settings change is not affinity-only for {identity}: {changed_fields(before, after)}",
        )
        expected_before, expected_after = EXPECTED_KERNEL_CHANGES[identity]
        require(before.get("affinity") == expected_before, f"raw before affinity differs for {identity}")
        if allow_affinity_refresh:
            refreshed = after.get("affinity")
            require(
                isinstance(refreshed, list)
                and refreshed
                and all(isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0 for cpu in refreshed)
                and len(refreshed) == len(set(refreshed)),
                f"raw refreshed affinity is invalid for {identity}",
            )
        else:
            require(
                after.get("affinity") == expected_after,
                f"raw affinity transition differs from the reviewed evidence for {identity}",
            )
        require(
            before.get("scheduler") == after.get("scheduler") == 0
            and before.get("nice") == after.get("nice") == -20,
            f"raw scheduler/nice differs from the reviewed kernel-thread evidence for {identity}",
        )

        current = process_probe(identity[0])
        require(current.get("pid") == identity[0], f"live PID mismatch for {identity}")
        require(str(current.get("start_ticks")) == identity[1], f"live start_ticks mismatch for {identity}")
        require(current.get("comm") == identity[2] and identity[2].startswith("kworker/"), f"live comm is not the reviewed kworker for {identity}")
        require(current.get("flags") == EXPECTED_KERNEL_FLAGS, f"live flags changed for {identity}")
        require(current.get("pf_kthread_set") is True, f"PF_KTHREAD is absent for {identity}")
        require(current.get("ppid") == 2, f"live PPid is not 2 for {identity}")
        require(current.get("status_name") == identity[2], f"live status Name mismatch for {identity}")
        require(current.get("status_kthread") == "1", f"live status Kthread is not 1 for {identity}")
        require(current.get("status_ppid") == "2", f"live status PPid is not 2 for {identity}")
        require(current.get("cmdline_bytes") == 0, f"live cmdline is not empty for {identity}")
        require(current.get("exe_absent") is True and current.get("exe_target") is None, f"live exe exists for {identity}")
        require(
            current.get("scheduler") == before["scheduler"]
            and current.get("nice") == before["nice"],
            f"live scheduler/nice changed for {identity}",
        )
        evidence.append(
            {
                "identity": {
                    "pid": identity[0],
                    "start_ticks": identity[1],
                    "comm": identity[2],
                },
                "raw_before": before,
                "raw_after": after,
                "changed_fields": ["affinity"],
                "live": current,
                "classification": (
                    "bound_kernel_thread_affinity_refresh_only"
                    if allow_affinity_refresh
                    else "kernel_thread_affinity_migration_only"
                ),
                "affinity_refresh": allow_affinity_refresh,
                "accepted_exception": True,
            }
        )
    if allow_affinity_refresh:
        require(
            seen.issubset(set(EXPECTED_KERNEL_CHANGES)),
            "kernel affinity refresh contains an unbound identity",
        )
    else:
        require(seen == set(EXPECTED_KERNEL_CHANGES), "reviewed kernel-thread changes are incomplete")
    evidence.sort(key=lambda item: item["identity"]["pid"])
    return evidence


def validate_build_bindings(
    root: Path,
    source_files: dict[str, str],
    complete: dict[str, Any],
    bindings: dict[str, Any],
    build: dict[str, Any],
) -> dict[str, Any]:
    require(
        complete.get("bindings") == bindings,
        "columns3519 COMPLETE/BINDINGS mismatch",
    )
    require(
        build.get("schema") == "operator-prepared135-build-v1"
        and build.get("status") == "built-and-tested",
        "prepared build receipt is not built-and-tested",
    )
    require(
        build.get("NNCP") == "OFF"
        and build.get("excluded_action_ids") == EXPECTED_EXCLUDED_ACTION_IDS
        and bindings.get("NNCP") == "OFF"
        and bindings.get("excluded_action_ids") == EXPECTED_EXCLUDED_ACTION_IDS,
        "prepared build/run exclusion policy mismatch",
    )
    binary_path = root / "build/prepared_columns"
    require(binary_path.is_file(), f"missing prepared_columns binary: {binary_path}")
    binary_sha = sha256(binary_path)
    build_binary_sha = build.get("binaries", {}).get("prepared_columns")
    require(
        binary_sha == build_binary_sha == bindings.get("binary_sha256"),
        "prepared_columns binary/build/run binding mismatch",
    )
    build_sources = mapping_of_hashes(build.get("source_sha256"), "BUILD_RECEIPT.source_sha256")
    build_sources_after = mapping_of_hashes(
        build.get("source_after_sha256"), "BUILD_RECEIPT.source_after_sha256"
    )
    require(build_sources_after == build_sources, "prepared build source before/after differs")
    require(
        all(source_files.get(path) == digest for path, digest in build_sources.items()),
        "prepared build source closure is not contained in SOURCE_FREEZE",
    )
    require(
        build.get("frozen_runtime_before") == build.get("frozen_runtime_after"),
        "prepared build frozen runtime before/after differs",
    )
    require(
        bindings.get("catalog_sha256") == EXPECTED_CATALOG_SHA256,
        "prepared run catalog binding mismatch",
    )
    return {
        "prepared_columns_path": str(binary_path),
        "prepared_columns_sha256": binary_sha,
        "build_source_files": len(build_sources),
        "build_source_is_subset_of_source_freeze": True,
        "build_source_before_after_equal": True,
        "frozen_runtime_before_after_equal": True,
        "complete_bindings_equal": True,
        "NNCP": "OFF",
        "excluded_action_ids": EXPECTED_EXCLUDED_ACTION_IDS,
    }


def reconcile_protection_reports(
    before: dict[str, Any],
    raw_report: dict[str, Any],
    acceptance: dict[str, Any],
    *,
    process_probe: Callable[[int], dict[str, Any]] | None = None,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Recheck the raw failure as kernel-affinity-only without writing it."""
    require(
        acceptance.get("schema") == SCHEMA and acceptance.get("status") == "pass",
        "invalid recovery acceptance schema/status",
    )
    require(
        raw_report.get("status") == "fail"
        and raw_report.get("checked_immutable_files") == EXPECTED_PROTECTED_FILES
        and raw_report.get("mismatch") == [],
        "raw protection report is not the reviewed hash-clean failure",
    )
    if process_probe is None:
        process_probe = lambda pid: probe_live_process(pid, Path(proc_root))
    current = verify_kernel_changes(
        raw_report.get("settings_changed"),
        before.get("observed_jobs"),
        process_probe,
        allow_affinity_refresh=True,
    )
    recorded = acceptance.get("kernel_affinity_exceptions")
    require(isinstance(recorded, list), "acceptance lacks kernel affinity evidence")
    require(
        len(recorded) == len(EXPECTED_KERNEL_CHANGES),
        "acceptance does not contain both initially bound kernel identities",
    )
    recorded_by_identity: dict[tuple[int, str, str], dict[str, Any]] = {}
    for item in recorded:
        require(isinstance(item, dict), "acceptance kernel evidence is not an object")
        identity_value = item.get("identity", {})
        identity = (
            identity_value.get("pid"),
            str(identity_value.get("start_ticks")),
            identity_value.get("comm"),
        )
        require(
            identity in EXPECTED_KERNEL_CHANGES and identity not in recorded_by_identity,
            f"acceptance contains an unexpected or duplicate kernel identity: {identity}",
        )
        expected_before, expected_after = EXPECTED_KERNEL_CHANGES[identity]
        raw_before = item.get("raw_before", {})
        raw_after = item.get("raw_after", {})
        require(
            raw_before.get("affinity") == expected_before
            and raw_after.get("affinity") == expected_after
            and changed_fields(raw_before, raw_after) == {"affinity"}
            and item.get("changed_fields") == ["affinity"]
            and item.get("classification")
            == "kernel_thread_affinity_migration_only"
            and item.get("affinity_refresh", False) is False
            and item.get("accepted_exception") is True,
            f"acceptance no longer binds the initial reviewed transition for {identity}",
        )
        recorded_by_identity[identity] = item
    require(
        set(recorded_by_identity) == set(EXPECTED_KERNEL_CHANGES),
        "acceptance initial kernel identities are incomplete",
    )
    refreshed_identities = {
        (
            item["identity"]["pid"],
            str(item["identity"]["start_ticks"]),
            item["identity"]["comm"],
        )
        for item in current
    }
    require(
        refreshed_identities.issubset(set(recorded_by_identity)),
        "raw recheck contains a kernel identity absent from acceptance",
    )
    gate = acceptance.get("gate_reason", {})
    require(
        gate.get("allowed_changed_field") == "affinity"
        and gate.get("userspace_process_exception_allowed") is False
        and gate.get("other_settings_exception_allowed") is False
        and gate.get("file_hash_exception_allowed") is False,
        "acceptance weakens the strict recovery gate",
    )
    return {
        "status": "pass",
        "raw_status": "fail",
        "raw_file_mismatches": 0,
        "accepted_kernel_affinity_changes": len(current),
        "classification": "kernel-affinity-refresh",
        "kernel_affinity_refresh": True,
        "initial_acceptance_transitions_preserved": True,
        "refreshed_identity_subset": len(current) < len(recorded),
        "live_kernel_evidence": current,
    }


def reconcile(
    root: Path,
    *,
    proc_root: Path = Path("/proc"),
    sched_header: Path = DEFAULT_SCHED_HEADER,
    enforce_execution_profile: bool = True,
    process_probe: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), f"run root is not a directory: {root}")
    output = root / "RECOVERY_ACCEPTANCE.json"
    temporary = root / "RECOVERY_ACCEPTANCE.json.tmp"
    require(not output.exists() and not temporary.exists(), "recovery acceptance already exists")
    require(not (root / "DONE.json").exists(), "DONE.json already exists; reconciliation is not applicable")

    paths = {
        "source_freeze": root / "SOURCE_FREEZE.json",
        "protection_pre": root / "PROTECTION_PRE.json",
        "protection_post": root / "PROTECTION_POST.json",
        "failed": root / "FAILED.json",
        "stages": root / "STAGES.json",
        "columns_complete": root / "columns3519/COMPLETE.json",
        "columns_bindings": root / "columns3519/BINDINGS.json",
        "build_receipt": root / "build/BUILD_RECEIPT.json",
    }
    values = {name: read_json(path, name) for name, path in paths.items()}
    evidence_hashes = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }

    stages = values["stages"]
    require(isinstance(stages, list) and len(stages) == len(EXPECTED_STAGES), "STAGES must contain exactly five stages")
    require(
        tuple(stage.get("stage") for stage in stages if isinstance(stage, dict)) == EXPECTED_STAGES,
        "STAGES names/order differ from the frozen coordinator",
    )
    require(
        all(isinstance(stage, dict) and stage.get("returncode") == 0 for stage in stages),
        "a measurement stage did not return zero",
    )
    failed = values["failed"]
    require(isinstance(failed, dict) and failed.get("status") == "failed", "FAILED.json is not a failure receipt")
    require(failed.get("error") == "preservation postcheck failed", "coordinator failed for a reason other than preservation postcheck")
    require(failed.get("stages") == stages, "FAILED/STAGES evidence differs")

    complete = values["columns_complete"]
    require(
        isinstance(complete, dict)
        and complete.get("schema") == "operator-prepared135-columns-complete-v1"
        and complete.get("status") == "complete"
        and complete.get("completed") == EXPECTED_COLUMNS
        and complete.get("expected") == EXPECTED_COLUMNS
        and complete.get("failed") == 0
        and complete.get("limited") is False,
        "columns3519 is not a full 3,519-column zero-failure run",
    )
    require(
        isinstance(complete.get("index_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", complete["index_sha256"]) is not None,
        "columns3519 COMPLETE lacks a valid INDEX binding",
    )

    source_freeze = values["source_freeze"]
    require(isinstance(source_freeze, dict), "SOURCE_FREEZE must be an object")
    source_files = mapping_of_hashes(source_freeze.get("files"), "SOURCE_FREEZE.files")
    require(len(source_files) == EXPECTED_SOURCE_FILES, "SOURCE_FREEZE does not contain exactly 20 files")
    verify_hashes(source_files, "frozen measurement source")

    bindings = values["columns_bindings"]
    build = values["build_receipt"]
    require(isinstance(bindings, dict), "columns3519 BINDINGS must be an object")
    require(isinstance(build, dict), "BUILD_RECEIPT must be an object")
    build_binding = validate_build_bindings(
        root, source_files, complete, bindings, build
    )

    pre = values["protection_pre"]
    require(isinstance(pre, dict), "PROTECTION_PRE must be an object")
    immutable = mapping_of_hashes(pre.get("immutable_files"), "PROTECTION_PRE.immutable_files")
    manifests = mapping_of_hashes(pre.get("source_manifests"), "PROTECTION_PRE.source_manifests")
    protected = {**immutable, **manifests}
    require(
        len(protected) == EXPECTED_PROTECTED_FILES
        and len(immutable) + len(manifests) == EXPECTED_PROTECTED_FILES,
        "PROTECTION_PRE protected closure is not exactly 3,817 unique files",
    )
    verify_hashes(protected, "protected immutable/source-manifest closure")

    post = values["protection_post"]
    require(isinstance(post, dict) and post.get("status") == "fail", "PROTECTION_POST is not the reviewed raw failure")
    require(post.get("checked_immutable_files") == EXPECTED_PROTECTED_FILES, "PROTECTION_POST checked-file count mismatch")
    require(post.get("mismatch") == [], "PROTECTION_POST contains a file mismatch")

    header_evidence = read_pf_kthread_definition(Path(sched_header))
    if process_probe is None:
        process_probe = lambda pid: probe_live_process(pid, Path(proc_root))
    kernel_evidence = verify_kernel_changes(
        post.get("settings_changed"), pre.get("observed_jobs"), process_probe
    )
    profile = execution_profile() if enforce_execution_profile else {
        "enforced": False,
        "reason": "fixture-only injection; production CLI always enforces the idle profile",
    }

    receipt = {
        "schema": SCHEMA,
        "status": "pass",
        "accepted_at": now(),
        "root": str(root),
        "execution_profile": profile,
        "original_evidence": evidence_hashes,
        "campaign": {
            "stage_count": len(stages),
            "stage_names": list(EXPECTED_STAGES),
            "all_stage_returncodes_zero": True,
            "columns_completed": complete["completed"],
            "columns_expected": complete["expected"],
            "columns_failed": complete["failed"],
            "limited": complete["limited"],
            "expected_datasets": EXPECTED_DATASETS,
            "dataset_count_verification": "performed later by the frozen COLUMNS/INDEX finalizer validation",
            "index_sha256": complete["index_sha256"],
        },
        "protection": {
            "source_freeze_files": len(source_files),
            "protected_files": len(protected),
            "immutable_files": len(immutable),
            "source_manifests": len(manifests),
            "source_hash_mismatches": 0,
            "protected_hash_mismatches": 0,
            "raw_post_status": post["status"],
            "raw_post_mismatch": post["mismatch"],
            "raw_settings_changed_count": len(kernel_evidence),
        },
        "build_binding": build_binding,
        "pf_kthread_definition": header_evidence,
        "kernel_affinity_exceptions": kernel_evidence,
        "gate_reason": {
            "original_failure": "preservation postcheck failed",
            "reconciled_classification": "the only settings changes were Linux kernel kworker affinity migrations",
            "allowed_changed_field": "affinity",
            "userspace_process_exception_allowed": False,
            "other_settings_exception_allowed": False,
            "file_hash_exception_allowed": False,
        },
        "scope": {
            "classification_only": True,
            "measurements_rerun": False,
            "models_modified": False,
            "results_modified": False,
            "processes_modified": False,
            "original_evidence_overwritten": False,
            "done_written": False,
            "accepts_only_the_two_bound_kernel_thread_affinity_changes": True,
            "does_not_replace_full_finalizer_validation": True,
        },
    }
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            json.dump(receipt, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(output)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    print(json.dumps({"status": "pass", "path": str(output), "sha256": sha256(output)}))
    return receipt


def validate_acceptance(
    root: Path,
    receipt_path: Path | None = None,
    *,
    proc_root: Path = Path("/proc"),
    sched_header: Path = DEFAULT_SCHED_HEADER,
    process_probe: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate an existing acceptance and recheck its live kernel identities."""
    root = Path(root).resolve(strict=True)
    receipt_path = (
        root / "RECOVERY_ACCEPTANCE.json"
        if receipt_path is None
        else Path(receipt_path).resolve(strict=True)
    )
    acceptance = read_json(receipt_path, "recovery acceptance")
    require(
        isinstance(acceptance, dict)
        and acceptance.get("schema") == SCHEMA
        and acceptance.get("status") == "pass"
        and acceptance.get("root") == str(root),
        "recovery acceptance identity/schema/status mismatch",
    )
    expected_paths = {
        "source_freeze": root / "SOURCE_FREEZE.json",
        "protection_pre": root / "PROTECTION_PRE.json",
        "protection_post": root / "PROTECTION_POST.json",
        "failed": root / "FAILED.json",
        "stages": root / "STAGES.json",
        "columns_complete": root / "columns3519/COMPLETE.json",
        "columns_bindings": root / "columns3519/BINDINGS.json",
        "build_receipt": root / "build/BUILD_RECEIPT.json",
    }
    original = acceptance.get("original_evidence")
    require(
        isinstance(original, dict) and set(original) == set(expected_paths),
        "acceptance original-evidence set mismatch",
    )
    for name, path in expected_paths.items():
        binding = original.get(name)
        require(
            isinstance(binding, dict)
            and binding.get("path") == str(path)
            and binding.get("sha256") == sha256(path),
            f"acceptance original evidence changed: {name}",
        )

    source_freeze = read_json(expected_paths["source_freeze"], "SOURCE_FREEZE")
    source_files = mapping_of_hashes(source_freeze.get("files"), "SOURCE_FREEZE.files")
    require(len(source_files) == EXPECTED_SOURCE_FILES, "acceptance source-file count mismatch")
    verify_hashes(source_files, "accepted frozen measurement source")
    before = read_json(expected_paths["protection_pre"], "PROTECTION_PRE")
    raw_report = read_json(expected_paths["protection_post"], "PROTECTION_POST")
    immutable = mapping_of_hashes(before.get("immutable_files"), "PROTECTION_PRE.immutable_files")
    manifests = mapping_of_hashes(before.get("source_manifests"), "PROTECTION_PRE.source_manifests")
    protected = {**immutable, **manifests}
    require(
        len(protected) == EXPECTED_PROTECTED_FILES
        and len(immutable) + len(manifests) == EXPECTED_PROTECTED_FILES,
        "acceptance protected-file count mismatch",
    )
    verify_hashes(protected, "accepted protected closure")

    stages = read_json(expected_paths["stages"], "STAGES")
    failed = read_json(expected_paths["failed"], "FAILED")
    complete = read_json(expected_paths["columns_complete"], "columns COMPLETE")
    bindings = read_json(expected_paths["columns_bindings"], "columns BINDINGS")
    build = read_json(expected_paths["build_receipt"], "BUILD_RECEIPT")
    require(
        isinstance(stages, list)
        and len(stages) == len(EXPECTED_STAGES)
        and tuple(stage.get("stage") for stage in stages) == EXPECTED_STAGES
        and all(stage.get("returncode") == 0 for stage in stages),
        "accepted stages no longer satisfy the five-stage zero-return contract",
    )
    require(
        failed.get("status") == "failed"
        and failed.get("error") == "preservation postcheck failed"
        and failed.get("stages") == stages,
        "accepted coordinator failure is not preservation-only",
    )
    require(
        complete.get("schema") == "operator-prepared135-columns-complete-v1"
        and complete.get("status") == "complete"
        and complete.get("completed") == EXPECTED_COLUMNS
        and complete.get("expected") == EXPECTED_COLUMNS
        and complete.get("failed") == 0
        and complete.get("limited") is False,
        "accepted columns COMPLETE is not full zero-failure evidence",
    )
    build_binding = validate_build_bindings(
        root, source_files, complete, bindings, build
    )
    require(
        acceptance.get("build_binding") == build_binding,
        "acceptance build binding differs from current frozen evidence",
    )
    header = read_pf_kthread_definition(Path(sched_header))
    require(
        acceptance.get("pf_kthread_definition") == header,
        "acceptance PF_KTHREAD definition binding changed",
    )
    reconciled = reconcile_protection_reports(
        before,
        raw_report,
        acceptance,
        process_probe=process_probe,
        proc_root=proc_root,
    )
    scope = acceptance.get("scope", {})
    for field in (
        "measurements_rerun",
        "models_modified",
        "results_modified",
        "processes_modified",
        "original_evidence_overwritten",
        "done_written",
    ):
        require(scope.get(field) is False, f"acceptance scope does not keep {field}=false")
    require(
        scope.get("classification_only") is True
        and scope.get("accepts_only_the_two_bound_kernel_thread_affinity_changes") is True,
        "acceptance scope is not classification-only",
    )
    return {
        "status": "pass",
        "schema": "operator-prepared135-protection-recovery-validation-v1",
        "acceptance_path": str(receipt_path),
        "acceptance_sha256": sha256(receipt_path),
        "source_files_verified": len(source_files),
        "protected_files_verified": len(protected),
        "campaign_columns_verified": complete["completed"],
        "build_binding": build_binding,
        "kernel_reconciliation": reconciled,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="failed prepared135 run root")
    parser.add_argument(
        "--sched-header",
        type=Path,
        default=DEFAULT_SCHED_HEADER,
        help="Linux sched.h used to bind the PF_KTHREAD definition",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        options = parse_args(sys.argv[1:] if argv is None else argv)
        reconcile(options.root, sched_header=options.sched_header)
        return 0
    except (ReconciliationError, OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
