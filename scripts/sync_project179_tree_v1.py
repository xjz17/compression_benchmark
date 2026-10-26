#!/usr/bin/env python3
"""Safely stream one verified regular-file tree between SSH hosts.

The data path is always ``source ssh stdout -> destination ssh stdin``.  Data
files are never staged on the machine running this program.  A pre-existing
relocatable inventory is read into memory, verified on the source, installed
with the verifier in a destination control directory, and used to verify every
top-level component.  "Exact" here means the set of regular-file relative paths
and every byte payload; empty directories, permissions, timestamps, ownership,
and hard-link topology are deliberately outside the contract.  Only a final
exact regular-file verification permits a no-replace atomic rename of the
staging directory to the requested final directory.

Authentication is deliberately outside this program.  It supports ordinary
non-interactive SSH configuration and existing OpenSSH control sockets, but it
has no password option and disables password and keyboard-interactive prompts.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SCHEMA = "project179-tree-sync-v1"
INVENTORY_NAME = "tree_inventory.json"
VERIFIER_NAME = "project179_tree_inventory_v1.py"
ORCHESTRATOR_NAME = "sync_project179_tree_v1.py"
STATE_NAME = "sync_state_v1.json"
VERIFY_RECEIPT_NAME = "tree_verification_receipt.json"
INSTALL_RECEIPT_NAME = "tree_install_receipt.json"
TREE_SEMANTICS = (
    "exact regular-file relative paths and byte payloads; directory metadata, "
    "empty directories, ownership, permissions, timestamps, and hard-link topology excluded"
)
MACHINE_ID_RE = re.compile(r"[0-9a-f]{32}\Z")

ENDPOINT_RE = re.compile(
    r"(?:(?P<user>[A-Za-z0-9_][A-Za-z0-9_.-]{0,63})@)?"
    r"(?P<host>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,251}[A-Za-z0-9])?)\Z"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# A safe prefix must be a project-specific path below (not equal to) these
# broad/system locations.  Requiring at least three path components also
# rejects paths such as /data and /data/project.
DANGEROUS_PREFIXES = {
    "/",
    "/bin",
    "/boot",
    "/data",
    "/dev",
    "/etc",
    "/home",
    "/lib",
    "/lib64",
    "/opt",
    "/proc",
    "/root",
    "/run",
    "/sbin",
    "/srv",
    "/sys",
    "/tmp",
    "/usr",
    "/var",
}
PROTECTED_DESTINATION_PREFIXES = (
    "/data/original_datasets",
    "/data/tsfile_datasets",
)


class SyncError(RuntimeError):
    """Raised when the transfer contract cannot be satisfied safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_inventory_module():
    script = Path(__file__).resolve().with_name(VERIFIER_NAME)
    if not script.is_file() or script.is_symlink():
        raise SyncError(
            f"inventory verifier must be a regular non-symlink file: {script}"
        )
    spec = importlib.util.spec_from_file_location(
        "project179_tree_inventory_v1", script
    )
    if spec is None or spec.loader is None:
        raise SyncError(f"cannot load inventory verifier: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_remote_path(raw: str, *, field: str) -> str:
    if (
        not isinstance(raw, str)
        or not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
    ):
        raise SyncError(f"{field} is not a valid remote path")
    path = PurePosixPath(raw)
    if not path.is_absolute() or raw != path.as_posix():
        raise SyncError(f"{field} must be a canonical absolute POSIX path: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SyncError(f"{field} contains an unsafe path component: {raw!r}")
    return raw


def _safe_prefix(raw: str, *, field: str) -> str:
    value = _canonical_remote_path(raw, field=field)
    path = PurePosixPath(value)
    if value in DANGEROUS_PREFIXES or len(path.parts) < 4:
        raise SyncError(f"{field} is too broad or dangerous: {value!r}")
    return value


def _reject_protected_destination_path(raw: str, *, field: str) -> str:
    """Reject a destination write path at or below a protected data tree."""
    value = _canonical_remote_path(raw, field=field)
    # POSIX permits an implementation-defined leading ``//``.  Treat it as
    # the ordinary root for this guard so it cannot alias a protected tree.
    path = PurePosixPath("/" + value.lstrip("/"))
    for protected in PROTECTED_DESTINATION_PREFIXES:
        try:
            path.relative_to(PurePosixPath(protected))
        except ValueError:
            continue
        raise SyncError(
            f"{field} is at or below protected destination prefix: {protected!r}"
        )
    return value


def _strict_descendant(child: str, parent: str, *, field: str) -> None:
    child_path = PurePosixPath(child)
    parent_path = PurePosixPath(parent)
    try:
        relative = child_path.relative_to(parent_path)
    except ValueError as exc:
        raise SyncError(f"{field} must be below its explicit safe prefix") from exc
    if not relative.parts:
        raise SyncError(f"{field} must be a strict descendant of its safe prefix")


def _paths_overlap(left: str, right: str) -> bool:
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    try:
        left_path.relative_to(right_path)
        return True
    except ValueError:
        pass
    try:
        right_path.relative_to(left_path)
        return True
    except ValueError:
        return False


def _validate_endpoint(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not ENDPOINT_RE.fullmatch(value):
        raise SyncError(f"{field} must be a plain SSH alias, host, or user@host")
    if value.startswith("-") or "@-" in value:
        raise SyncError(f"{field} must not begin with an option marker")
    return value


def _validate_control_socket(path: Path | None, *, field: str) -> Path | None:
    if path is None:
        return None
    if not path.is_absolute():
        raise SyncError(f"{field} must be an absolute local path")
    try:
        details = path.lstat()
    except OSError as exc:
        raise SyncError(f"{field} does not exist: {path}") from exc
    if not stat.S_ISSOCK(details.st_mode):
        raise SyncError(f"{field} is not a Unix-domain socket: {path}")
    if details.st_uid != os.geteuid():
        raise SyncError(f"{field} must be owned by the current local user: {path}")
    if stat.S_IMODE(details.st_mode) & 0o077:
        raise SyncError(f"{field} must not grant group or other permissions: {path}")
    parent = path.parent
    try:
        parent_details = parent.lstat()
    except OSError as exc:
        raise SyncError(f"{field} parent does not exist: {parent}") from exc
    if stat.S_ISLNK(parent_details.st_mode) or not stat.S_ISDIR(parent_details.st_mode):
        raise SyncError(f"{field} parent must be a plain directory: {parent}")
    if (
        parent_details.st_uid != os.geteuid()
        or stat.S_IMODE(parent_details.st_mode) & 0o022
    ):
        raise SyncError(
            f"{field} parent must be owned by the user and not group/world writable"
        )
    return path


def _validate_expected_identity(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not MACHINE_ID_RE.fullmatch(value):
        raise SyncError(f"{field} must be the 32-character lowercase /etc/machine-id")
    return value


def _validate_top_component(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise SyncError("invalid top-level component")
    path = PurePosixPath(value)
    if len(path.parts) != 1 or path.parts[0] in {".", ".."} or value != path.as_posix():
        raise SyncError(f"component is not one canonical top-level name: {value!r}")
    return value


@dataclass(frozen=True)
class SyncConfig:
    source_host: str
    destination_host: str
    source_safe_prefix: str
    source_root: str
    source_inventory: str
    source_verifier: str
    destination_safe_prefix: str
    destination_stage: str
    destination_final: str
    destination_control: str
    source_identity: str
    destination_identity: str
    source_control_socket: Path | None = None
    destination_control_socket: Path | None = None

    def validated(self) -> "SyncConfig":
        _validate_endpoint(self.source_host, field="source host")
        _validate_endpoint(self.destination_host, field="destination host")
        source_identity = _validate_expected_identity(
            self.source_identity, field="source identity"
        )
        destination_identity = _validate_expected_identity(
            self.destination_identity, field="destination identity"
        )
        if source_identity == destination_identity:
            raise SyncError("source and destination identities must differ")
        source_prefix = _safe_prefix(
            self.source_safe_prefix, field="source safe prefix"
        )
        destination_prefix = _reject_protected_destination_path(
            self.destination_safe_prefix, field="destination safe prefix"
        )
        destination_prefix = _safe_prefix(
            destination_prefix, field="destination safe prefix"
        )
        for field, value in (
            ("source root", self.source_root),
            ("source inventory", self.source_inventory),
            ("source verifier", self.source_verifier),
        ):
            canonical = _canonical_remote_path(value, field=field)
            _strict_descendant(canonical, source_prefix, field=field)
        if _paths_overlap(self.source_root, self.source_inventory) or _paths_overlap(
            self.source_root, self.source_verifier
        ):
            raise SyncError(
                "source inventory and verifier must be outside the data root"
            )
        for field, value in (
            ("destination stage", self.destination_stage),
            ("destination final", self.destination_final),
            ("destination control", self.destination_control),
        ):
            canonical = _reject_protected_destination_path(value, field=field)
            _strict_descendant(canonical, destination_prefix, field=field)
        if self.destination_stage == self.destination_final:
            raise SyncError("destination stage and final paths must differ")
        if _paths_overlap(self.destination_stage, self.destination_final):
            raise SyncError("destination stage and final paths must not overlap")
        if _paths_overlap(
            self.destination_control, self.destination_stage
        ) or _paths_overlap(self.destination_control, self.destination_final):
            raise SyncError("destination control must be outside both data roots")
        _validate_control_socket(
            self.source_control_socket, field="source control socket"
        )
        _validate_control_socket(
            self.destination_control_socket, field="destination control socket"
        )
        if (
            self.source_control_socket is not None
            and self.destination_control_socket is not None
            and self.source_control_socket.resolve(strict=True)
            == self.destination_control_socket.resolve(strict=True)
        ):
            raise SyncError("source and destination control sockets must differ")
        return self


def _ssh_argv(host: str, control_socket: Path | None, remote_command: str) -> list[str]:
    argv = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ControlMaster=no",
    ]
    if control_socket is not None:
        argv.extend(["-S", str(control_socket)])
    argv.extend([host, remote_command])
    return argv


def _remote_command(arguments: Sequence[str]) -> str:
    if not arguments:
        raise SyncError("empty remote command")
    return shlex.join(list(arguments))


class SubprocessRunner:
    """Minimal injectable process layer; never invokes a local shell."""

    def run(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        check: bool = True,
    ):
        result = subprocess.run(
            list(argv),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
            raise SyncError(f"command failed with exit {result.returncode}: {stderr}")
        return result

    def capture(self, argv: Sequence[str]) -> bytes:
        return self.run(argv, check=True).stdout

    def write(self, argv: Sequence[str], payload: bytes) -> None:
        self.run(argv, input_bytes=payload, check=True)

    def pipe(self, source_argv: Sequence[str], destination_argv: Sequence[str]) -> None:
        source = subprocess.Popen(list(source_argv), stdout=subprocess.PIPE)
        assert source.stdout is not None
        destination = subprocess.Popen(list(destination_argv), stdin=source.stdout)
        source.stdout.close()
        destination_status = destination.wait()
        source_status = source.wait()
        if source_status != 0 or destination_status != 0:
            raise SyncError(
                "tar stream failed: "
                f"source ssh exit={source_status}, destination ssh exit={destination_status}"
            )


REMOTE_PATH_BOOTSTRAP = r"""
import json, os, stat, sys

dangerous = {"/", "/bin", "/boot", "/data", "/dev", "/etc", "/home", "/lib",
             "/lib64", "/opt", "/proc", "/root", "/run", "/sbin", "/srv",
             "/sys", "/tmp", "/usr", "/var"}
protected = ("/data/original_datasets", "/data/tsfile_datasets")
prefix, stage, final, control = sys.argv[1:]

def fail(message):
    raise SystemExit(message)

def canonical(value):
    return value.startswith("/") and os.path.normpath(value) == value and "\x00" not in value

for value in (prefix, stage, final, control):
    if not canonical(value): fail("non-canonical remote path")
    if any(os.path.commonpath((value, parent)) == parent for parent in protected):
        fail("target path is at or below a protected destination prefix")
if prefix in dangerous or len([p for p in prefix.split("/") if p]) < 3:
    fail("dangerous safe prefix")

def below(value, parent):
    return os.path.commonpath((value, parent)) == parent and value != parent

if not all(below(value, prefix) for value in (stage, final, control)):
    fail("target path escapes safe prefix")
if any(os.path.commonpath((left, right)) in (left, right)
       for left, right in ((stage, final), (control, stage), (control, final))):
    fail("target paths overlap")

def reject_link_ancestors(path):
    current = "/"
    for part in [p for p in path.split("/") if p]:
        current = os.path.join(current, part)
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            fail("symlink ancestor is forbidden: " + current)

for value in (prefix, stage, final, control): reject_link_ancestors(value)
prefix_parent = os.path.dirname(prefix)
if not os.path.isdir(prefix_parent) or stat.S_ISLNK(os.lstat(prefix_parent).st_mode):
    fail("safe-prefix parent must already be a plain directory")
if not os.path.lexists(prefix): os.mkdir(prefix, 0o700)
reject_link_ancestors(prefix)
def require_private_directory(path, label):
    details = os.lstat(path)
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        fail(label + " is not a plain directory")
    if details.st_uid != os.geteuid(): fail(label + " is not owned by the remote user")
    if stat.S_IMODE(details.st_mode) & 0o022: fail(label + " is group/world writable")
require_private_directory(prefix, "safe prefix")
os.makedirs(control, mode=0o700, exist_ok=True)
reject_link_ancestors(control)
require_private_directory(control, "control directory")
print(json.dumps({"status": "bootstrapped", "control": control}))
""".strip()


REMOTE_IDENTITY_PROBE = r"""
import os, re

paths = ("/etc/machine-id", "/var/lib/dbus/machine-id")
for path in paths:
    try:
        with open(path, "r", encoding="ascii") as handle:
            value = handle.read().strip().lower()
    except OSError:
        continue
    if re.fullmatch(r"[0-9a-f]{32}", value):
        print(value)
        raise SystemExit(0)
raise SystemExit("no valid Linux machine-id")
""".strip()


REMOTE_GNU_TAR_PROBE = r"""
import subprocess

result = subprocess.run(
    ["tar", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
)
first_line = result.stdout.splitlines()[0] if result.stdout else b""
if result.returncode != 0 or b"GNU tar" not in first_line:
    raise SystemExit("GNU tar is required")
print(first_line.decode("utf-8", errors="replace"))
""".strip()


REMOTE_RENAMEAT2_PROBE = r"""
import ctypes, sys

if not sys.platform.startswith("linux"):
    raise SystemExit("Linux renameat2 is required")
libc = ctypes.CDLL(None, use_errno=True)
if getattr(libc, "renameat2", None) is None:
    raise SystemExit("libc renameat2 is unavailable")
print("renameat2 RENAME_NOREPLACE available")
""".strip()


REMOTE_SOURCE_PREFLIGHT = r"""
import json, os, stat, sys

prefix, root, inventory, verifier = sys.argv[1:]
dangerous = {"/", "/bin", "/boot", "/data", "/dev", "/etc", "/home", "/lib",
             "/lib64", "/opt", "/proc", "/root", "/run", "/sbin", "/srv",
             "/sys", "/tmp", "/usr", "/var"}

def fail(message): raise SystemExit(message)
def canonical(value): return value.startswith("/") and os.path.normpath(value) == value and "\x00" not in value
if any(not canonical(v) for v in (prefix, root, inventory, verifier)): fail("non-canonical source path")
if prefix in dangerous or len([p for p in prefix.split("/") if p]) < 3: fail("dangerous source prefix")
if any(os.path.commonpath((v, prefix)) != prefix or v == prefix for v in (root, inventory, verifier)):
    fail("source path escapes safe prefix")

def reject_link_ancestors(path):
    current = "/"
    for part in [p for p in path.split("/") if p]:
        current = os.path.join(current, part)
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            fail("source symlink ancestor is forbidden: " + current)

for value in (root, inventory, verifier): reject_link_ancestors(value)
if not os.path.isdir(root) or stat.S_ISLNK(os.lstat(root).st_mode): fail("source root is not a plain directory")
for value in (inventory, verifier):
    if not os.path.isfile(value) or stat.S_ISLNK(os.lstat(value).st_mode): fail("source control file is not regular")
for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
    for name in dirs:
        mode = os.lstat(os.path.join(current, name)).st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode): fail("unsafe source directory entry")
    for name in files:
        mode = os.lstat(os.path.join(current, name)).st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode): fail("unsafe source file entry")
print(json.dumps({"status": "source_preflight_ok"}))
""".strip()


REMOTE_ATOMIC_WRITE = r"""
import hashlib, json, os, stat, sys, tempfile

target, expected_sha = sys.argv[1:]
parent = os.path.dirname(target)
if not os.path.isabs(target) or os.path.normpath(target) != target: raise SystemExit("unsafe write path")
protected = ("/data/original_datasets", "/data/tsfile_datasets")
if any(os.path.commonpath((target, value)) == value for value in protected):
    raise SystemExit("write path is at or below a protected destination prefix")
current = "/"
for part in [p for p in target.split("/") if p]:
    current = os.path.join(current, part)
    if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
        raise SystemExit("symlink write ancestor is forbidden")
if not os.path.isdir(parent) or stat.S_ISLNK(os.lstat(parent).st_mode): raise SystemExit("unsafe write parent")
if os.path.lexists(target) and (stat.S_ISLNK(os.lstat(target).st_mode) or not stat.S_ISREG(os.lstat(target).st_mode)):
    raise SystemExit("write target is not a regular file")
payload = sys.stdin.buffer.read()
actual_sha = hashlib.sha256(payload).hexdigest()
if actual_sha != expected_sha: raise SystemExit("stdin sha256 mismatch")
if os.path.exists(target):
    with open(target, "rb") as handle:
        existing_sha = hashlib.sha256(handle.read()).hexdigest()
    if existing_sha != expected_sha:
        raise SystemExit("refusing to replace a different control file")
    print(json.dumps({"status": "already_current", "sha256": actual_sha, "target": target}))
    raise SystemExit(0)
fd, temporary = tempfile.mkstemp(prefix=".incoming.", dir=parent)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary): os.unlink(temporary)
print(json.dumps({"status": "written", "sha256": actual_sha, "target": target}))
""".strip()


def _bootstrap_destination(config: SyncConfig, runner: SubprocessRunner) -> None:
    command = _remote_command(
        [
            "python3",
            "-c",
            REMOTE_PATH_BOOTSTRAP,
            config.destination_safe_prefix,
            config.destination_stage,
            config.destination_final,
            config.destination_control,
        ]
    )
    runner.run(
        _ssh_argv(config.destination_host, config.destination_control_socket, command)
    )


def _source_preflight(config: SyncConfig, runner: SubprocessRunner) -> None:
    command = _remote_command(
        [
            "python3",
            "-c",
            REMOTE_SOURCE_PREFLIGHT,
            config.source_safe_prefix,
            config.source_root,
            config.source_inventory,
            config.source_verifier,
        ]
    )
    runner.run(_ssh_argv(config.source_host, config.source_control_socket, command))


def _probe_remote_identity(
    host: str,
    control_socket: Path | None,
    expected_identity: str,
    *,
    field: str,
    runner: SubprocessRunner,
) -> None:
    command = _remote_command(["python3", "-c", REMOTE_IDENTITY_PROBE])
    output = runner.capture(_ssh_argv(host, control_socket, command))
    try:
        actual_identity = output.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SyncError(f"{field} identity probe did not return ASCII") from exc
    if not MACHINE_ID_RE.fullmatch(actual_identity):
        raise SyncError(f"{field} identity probe returned an invalid machine-id")
    if actual_identity != expected_identity:
        raise SyncError(
            f"{field} identity mismatch: expected {expected_identity}, got {actual_identity}"
        )


def _probe_gnu_tar(
    host: str,
    control_socket: Path | None,
    *,
    field: str,
    runner: SubprocessRunner,
) -> None:
    command = _remote_command(["python3", "-c", REMOTE_GNU_TAR_PROBE])
    output = runner.capture(_ssh_argv(host, control_socket, command))
    lines = output.splitlines()
    if not lines or b"GNU tar" not in lines[0]:
        raise SyncError(f"{field} GNU tar capability probe returned an invalid receipt")


def _probe_renameat2(
    host: str,
    control_socket: Path | None,
    *,
    runner: SubprocessRunner,
) -> None:
    command = _remote_command(["python3", "-c", REMOTE_RENAMEAT2_PROBE])
    output = runner.capture(_ssh_argv(host, control_socket, command))
    if output.strip() != b"renameat2 RENAME_NOREPLACE available":
        raise SyncError(
            "destination renameat2 capability probe returned an invalid receipt"
        )


def _preflight_transport(config: SyncConfig, runner: SubprocessRunner) -> None:
    _probe_remote_identity(
        config.source_host,
        config.source_control_socket,
        config.source_identity,
        field="source",
        runner=runner,
    )
    _probe_remote_identity(
        config.destination_host,
        config.destination_control_socket,
        config.destination_identity,
        field="destination",
        runner=runner,
    )
    _probe_gnu_tar(
        config.source_host,
        config.source_control_socket,
        field="source",
        runner=runner,
    )
    _probe_gnu_tar(
        config.destination_host,
        config.destination_control_socket,
        field="destination",
        runner=runner,
    )
    _probe_renameat2(
        config.destination_host,
        config.destination_control_socket,
        runner=runner,
    )


def _remote_atomic_write(
    config: SyncConfig, runner: SubprocessRunner, target: str, payload: bytes
) -> None:
    _reject_protected_destination_path(target, field="remote control file")
    _strict_descendant(target, config.destination_control, field="remote control file")
    command = _remote_command(
        ["python3", "-c", REMOTE_ATOMIC_WRITE, target, _sha256_bytes(payload)]
    )
    runner.write(
        _ssh_argv(config.destination_host, config.destination_control_socket, command),
        payload,
    )


def _load_inventory_bytes(payload: bytes) -> dict[str, object]:
    inventory_module = _load_inventory_module()
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyncError(f"source inventory is not valid UTF-8 JSON: {exc}") from exc
    try:
        manifest = inventory_module._validated_manifest(raw)
    except inventory_module.InventoryError as exc:
        raise SyncError(f"source inventory is invalid: {exc}") from exc
    if not manifest["entries"]:
        raise SyncError("refusing to synchronize an empty tree")
    return manifest


def _top_components(manifest: Mapping[str, object]) -> list[str]:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    components = sorted(
        {
            _validate_top_component(str(entry["path"]).split("/", 1)[0])
            for entry in entries
        }
    )
    return components


def _source_verify_command(config: SyncConfig) -> list[str]:
    remote = _remote_command(
        [
            "python3",
            config.source_verifier,
            "verify",
            "--root",
            config.source_root,
            "--manifest",
            config.source_inventory,
        ]
    )
    return _ssh_argv(config.source_host, config.source_control_socket, remote)


def _assert_source_inventory_sha256(
    config: SyncConfig, runner: SubprocessRunner, expected_sha256: str
) -> None:
    remote = _remote_command(["sha256sum", "--", config.source_inventory])
    output = runner.capture(
        _ssh_argv(config.source_host, config.source_control_socket, remote)
    )
    try:
        actual_sha256 = output.decode("ascii").split(None, 1)[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise SyncError("cannot parse source inventory sha256") from exc
    if actual_sha256 != expected_sha256:
        raise SyncError("source inventory changed during synchronization")


def _destination_script(config: SyncConfig) -> str:
    return str(PurePosixPath(config.destination_control) / ORCHESTRATOR_NAME)


def _destination_inventory(config: SyncConfig) -> str:
    return str(PurePosixPath(config.destination_control) / INVENTORY_NAME)


def _component_command(config: SyncConfig, component: str) -> list[str]:
    remote = _remote_command(
        [
            "python3",
            _destination_script(config),
            "verify-component",
            "--root",
            config.destination_stage,
            "--manifest",
            _destination_inventory(config),
            "--component",
            component,
            "--control",
            config.destination_control,
        ]
    )
    return _ssh_argv(config.destination_host, config.destination_control_socket, remote)


def _assert_stage_safe_command(config: SyncConfig) -> list[str]:
    remote = _remote_command(
        [
            "python3",
            _destination_script(config),
            "assert-stage-safe",
            "--safe-prefix",
            config.destination_safe_prefix,
            "--stage",
            config.destination_stage,
            "--final",
            config.destination_final,
            "--control",
            config.destination_control,
        ]
    )
    return _ssh_argv(config.destination_host, config.destination_control_socket, remote)


def _stream_component(
    config: SyncConfig, component: str, runner: SubprocessRunner
) -> None:
    source_remote = _remote_command(
        [
            "env",
            "-u",
            "TAR_OPTIONS",
            "tar",
            "--create",
            "--file=-",
            "--directory",
            config.source_root,
            "--",
            component,
        ]
    )
    destination_remote = _remote_command(
        [
            "env",
            "-u",
            "TAR_OPTIONS",
            "tar",
            "--extract",
            "--file=-",
            "--directory",
            config.destination_stage,
            "--no-same-owner",
            "--no-same-permissions",
            "--no-overwrite-dir",
        ]
    )
    runner.pipe(
        _ssh_argv(config.source_host, config.source_control_socket, source_remote),
        _ssh_argv(
            config.destination_host,
            config.destination_control_socket,
            destination_remote,
        ),
    )


def _initialize_remote_control(
    config: SyncConfig,
    runner: SubprocessRunner,
    inventory_bytes: bytes,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    local_verifier = Path(__file__).resolve().with_name(VERIFIER_NAME)
    local_orchestrator = Path(__file__).resolve()
    for path in (local_verifier, local_orchestrator):
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise SyncError(
                f"local control program is not a plain regular file: {path}"
            )
    verifier_bytes = local_verifier.read_bytes()
    orchestrator_bytes = local_orchestrator.read_bytes()
    control = PurePosixPath(config.destination_control)
    _remote_atomic_write(config, runner, str(control / INVENTORY_NAME), inventory_bytes)
    _remote_atomic_write(config, runner, str(control / VERIFIER_NAME), verifier_bytes)
    _remote_atomic_write(
        config, runner, str(control / ORCHESTRATOR_NAME), orchestrator_bytes
    )

    state = {
        "schema": SCHEMA,
        "source_host": config.source_host,
        "source_identity": config.source_identity,
        "source_safe_prefix": config.source_safe_prefix,
        "source_root": config.source_root,
        "source_inventory": config.source_inventory,
        "source_verifier": config.source_verifier,
        "destination_host": config.destination_host,
        "destination_identity": config.destination_identity,
        "destination_safe_prefix": config.destination_safe_prefix,
        "destination_stage": config.destination_stage,
        "destination_final": config.destination_final,
        "destination_control": config.destination_control,
        "inventory_sha256": _sha256_bytes(inventory_bytes),
        "tree_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "verifier_sha256": _sha256_bytes(verifier_bytes),
        "orchestrator_sha256": _sha256_bytes(orchestrator_bytes),
        "tree_semantics": TREE_SEMANTICS,
    }
    state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    remote = _remote_command(
        [
            "python3",
            _destination_script(config),
            "init-destination",
            "--safe-prefix",
            config.destination_safe_prefix,
            "--stage",
            config.destination_stage,
            "--final",
            config.destination_final,
            "--control",
            config.destination_control,
            "--state-json",
            state_json,
        ]
    )
    result = runner.run(
        _ssh_argv(config.destination_host, config.destination_control_socket, remote)
    )
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyncError("destination initialization returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") not in {
        "destination_initialized",
        "installed",
    }:
        raise SyncError("destination initialization returned an unexpected status")
    return payload


def synchronize(
    config: SyncConfig, *, runner: SubprocessRunner | None = None
) -> dict[str, object]:
    config.validated()
    active_runner = runner or SubprocessRunner()
    _preflight_transport(config, active_runner)
    _source_preflight(config, active_runner)

    cat_inventory = _remote_command(["cat", "--", config.source_inventory])
    inventory_bytes = active_runner.capture(
        _ssh_argv(config.source_host, config.source_control_socket, cat_inventory)
    )
    manifest = _load_inventory_bytes(inventory_bytes)
    inventory_sha256 = _sha256_bytes(inventory_bytes)
    _source_preflight(config, active_runner)
    _assert_source_inventory_sha256(config, active_runner, inventory_sha256)
    active_runner.run(_source_verify_command(config))

    _bootstrap_destination(config, active_runner)
    initialization = _initialize_remote_control(
        config, active_runner, inventory_bytes, manifest
    )
    if initialization["status"] == "installed":
        return {
            **initialization,
            "tree_semantics": TREE_SEMANTICS,
            "source_identity": config.source_identity,
            "destination_identity": config.destination_identity,
            "transferred_components": [],
            "resumed_components": _top_components(manifest),
            "destination_control": config.destination_control,
        }

    transferred: list[str] = []
    resumed: list[str] = []
    for component in _top_components(manifest):
        verification = active_runner.run(
            _component_command(config, component), check=False
        )
        if verification.returncode == 0:
            resumed.append(component)
            continue
        active_runner.run(_assert_stage_safe_command(config))
        _stream_component(config, component, active_runner)
        active_runner.run(_component_command(config, component), check=True)
        transferred.append(component)

    # Revalidate the immutable-source assumption after all streams and before
    # permitting target promotion.
    _probe_remote_identity(
        config.source_host,
        config.source_control_socket,
        config.source_identity,
        field="source",
        runner=active_runner,
    )
    _probe_remote_identity(
        config.destination_host,
        config.destination_control_socket,
        config.destination_identity,
        field="destination",
        runner=active_runner,
    )
    _source_preflight(config, active_runner)
    _assert_source_inventory_sha256(config, active_runner, inventory_sha256)
    active_runner.run(_source_verify_command(config))
    install_remote = _remote_command(
        [
            "python3",
            _destination_script(config),
            "verify-and-install",
            "--safe-prefix",
            config.destination_safe_prefix,
            "--stage",
            config.destination_stage,
            "--final",
            config.destination_final,
            "--control",
            config.destination_control,
        ]
    )
    active_runner.run(
        _ssh_argv(
            config.destination_host, config.destination_control_socket, install_remote
        )
    )
    return {
        "status": "installed",
        "schema": SCHEMA,
        "tree_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "tree_semantics": TREE_SEMANTICS,
        "source_identity": config.source_identity,
        "destination_identity": config.destination_identity,
        "transferred_components": transferred,
        "resumed_components": resumed,
        "destination_final": config.destination_final,
        "destination_control": config.destination_control,
    }


def _assert_no_symlink_tree(root: Path) -> None:
    if not os.path.lexists(root):
        return
    root_mode = root.lstat().st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise SyncError(f"tree root is not a plain directory: {root}")
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in dirnames:
            child = current_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SyncError(f"unsafe directory node: {child}")
        for name in filenames:
            child = current_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise SyncError(f"unsafe file node: {child}")


def _assert_no_symlink_ancestors(path: Path) -> None:
    if not path.is_absolute():
        raise SyncError(f"path must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise SyncError(f"symlink path ancestor is forbidden: {current}")


def _resolved_strict_child(path: Path, parent: Path, *, field: str) -> None:
    parent_resolved = parent.resolve(strict=True)
    path_resolved = path.resolve(strict=False)
    try:
        relative = path_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise SyncError(f"{field} is outside control directory") from exc
    if not relative.parts:
        raise SyncError(f"{field} must be below control directory")


def _component_entries(
    manifest: Mapping[str, object], component: str
) -> list[dict[str, object]]:
    _validate_top_component(component)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    selected = [
        dict(entry)
        for entry in entries
        if str(entry["path"]).split("/", 1)[0] == component
    ]
    if not selected:
        raise SyncError(f"component is absent from inventory: {component!r}")
    return selected


def _component_digest(entries: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        record = [entry["path"], entry["size_bytes"], entry["sha256"]]
        digest.update(
            (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _verify_component(
    root: Path, manifest_path: Path, component: str, control: Path
) -> dict[str, object]:
    _reject_protected_destination_path(root.as_posix(), field="destination stage")
    _reject_protected_destination_path(
        control.as_posix(), field="destination control"
    )
    _assert_no_symlink_ancestors(root)
    _assert_no_symlink_ancestors(manifest_path)
    _assert_no_symlink_ancestors(control)
    root = root.resolve(strict=True)
    control = control.resolve(strict=True)
    _resolved_strict_child(manifest_path, control, field="inventory manifest")
    if manifest_path.resolve(strict=True) != (control / INVENTORY_NAME).resolve(
        strict=True
    ):
        raise SyncError("component verification must use the bound inventory manifest")
    _assert_owned_private_directory(control, field="control directory")
    state = _load_state(control)
    if state.get("destination_control") != str(control):
        raise SyncError("sync state does not bind the component control directory")
    if state.get("destination_stage") != str(root):
        raise SyncError("sync state does not bind the component staging directory")
    _assert_bound_control_programs(control, state)
    manifest, inventory_sha = _load_bound_manifest(control, state)
    inventory_module = _load_inventory_module()
    entries = _component_entries(manifest, component)
    component_path = root / component

    expected_paths = [str(entry["path"]) for entry in entries]
    expected_by_path = {str(entry["path"]): entry for entry in entries}
    if not os.path.lexists(component_path):
        raise SyncError(f"component is incomplete: {component!r}")
    mode = component_path.lstat().st_mode
    actual_paths: list[str]
    if stat.S_ISREG(mode):
        if expected_paths != [component]:
            raise SyncError(f"component type does not match inventory: {component!r}")
        actual_paths = [component]
    elif stat.S_ISDIR(mode):
        if component in expected_by_path:
            raise SyncError(f"component type does not match inventory: {component!r}")
        actual_paths = sorted(
            f"{component}/{relative}"
            for relative, _ in inventory_module._iter_regular_files(component_path)
        )
    else:
        raise SyncError(f"component is a symlink or special object: {component!r}")
    if actual_paths != expected_paths:
        raise SyncError(
            f"component file set is incomplete or has extras: {component!r}"
        )

    actual_entries: list[dict[str, object]] = []
    for relative in expected_paths:
        size_bytes, sha256 = inventory_module._hash_stable_regular_file(root / relative)
        expected = expected_by_path[relative]
        if size_bytes != expected["size_bytes"] or sha256 != expected["sha256"]:
            raise SyncError(f"component content mismatch: {relative!r}")
        actual_entries.append(
            {"path": relative, "size_bytes": size_bytes, "sha256": sha256}
        )

    receipt = {
        "status": "component_verified",
        "schema": SCHEMA,
        "component": component,
        "file_count": len(actual_entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in actual_entries),
        "component_sha256": _component_digest(actual_entries),
        "tree_sha256": manifest["tree_sha256"],
        "tree_semantics": TREE_SEMANTICS,
        "inventory_manifest_sha256": inventory_sha,
        "verified_at_utc": _utc_now(),
    }
    receipt_name = f"{hashlib.sha256(component.encode('utf-8')).hexdigest()}.json"
    receipt_path = control / "components" / receipt_name
    _resolved_strict_child(receipt_path, control, field="component receipt")
    inventory_module._write_json_atomic(receipt_path, receipt)
    return receipt


def _validate_local_install_paths(
    safe_prefix: Path, stage: Path, final: Path, control: Path
) -> None:
    prefix_text = _reject_protected_destination_path(
        safe_prefix.as_posix(), field="safe prefix"
    )
    prefix_text = _safe_prefix(prefix_text, field="safe prefix")
    for field, path in (("stage", stage), ("final", final), ("control", control)):
        value = _reject_protected_destination_path(path.as_posix(), field=field)
        _strict_descendant(value, prefix_text, field=field)
    if _paths_overlap(stage.as_posix(), final.as_posix()):
        raise SyncError("stage and final paths overlap")
    if _paths_overlap(control.as_posix(), stage.as_posix()) or _paths_overlap(
        control.as_posix(), final.as_posix()
    ):
        raise SyncError("control overlaps a data root")


def _assert_owned_private_directory(path: Path, *, field: str) -> None:
    try:
        details = path.lstat()
    except OSError as exc:
        raise SyncError(f"{field} does not exist: {path}") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise SyncError(f"{field} is not a plain directory: {path}")
    if details.st_uid != os.geteuid():
        raise SyncError(f"{field} is not owned by the remote user: {path}")
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise SyncError(f"{field} is group/world writable: {path}")


def _assert_owned_private_regular_file(path: Path, *, field: str) -> None:
    try:
        details = path.lstat()
    except OSError as exc:
        raise SyncError(f"{field} does not exist: {path}") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise SyncError(f"{field} is not a plain regular file: {path}")
    if details.st_uid != os.geteuid():
        raise SyncError(f"{field} is not owned by the remote user: {path}")
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise SyncError(f"{field} is group/world writable: {path}")


def _read_stable_regular_file(path: Path, *, field: str) -> bytes:
    _assert_owned_private_regular_file(path, field=field)
    before = path.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identities = [
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        for value in (before, opened_before, opened_after, after)
    ]
    if len(set(identities)) != 1:
        raise SyncError(f"{field} changed while being read: {path}")
    return b"".join(chunks)


def _validate_state_payload(state: object) -> dict[str, object]:
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise SyncError("sync state has the wrong schema")
    for key in (
        "inventory_sha256",
        "tree_sha256",
        "verifier_sha256",
        "orchestrator_sha256",
    ):
        if not isinstance(state.get(key), str) or not SHA256_RE.fullmatch(state[key]):
            raise SyncError(f"invalid state digest: {key}")
    for key in ("source_identity", "destination_identity"):
        _validate_expected_identity(state.get(key), field=f"state {key}")
    for key in (
        "destination_safe_prefix",
        "destination_stage",
        "destination_final",
        "destination_control",
    ):
        _reject_protected_destination_path(state.get(key), field=f"state {key}")
    if state.get("tree_semantics") != TREE_SEMANTICS:
        raise SyncError("sync state has unexpected tree semantics")
    return dict(state)


def _load_state(control: Path) -> dict[str, object]:
    state_path = control / STATE_NAME
    try:
        payload = json.loads(
            _read_stable_regular_file(state_path, field="sync state").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyncError(f"cannot read existing sync state: {exc}") from exc
    return _validate_state_payload(payload)


def _assert_state_destination_paths(
    state: Mapping[str, object], args: argparse.Namespace
) -> None:
    expected = {
        "destination_safe_prefix": str(args.safe_prefix),
        "destination_stage": str(args.stage),
        "destination_final": str(args.final),
        "destination_control": str(args.control),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SyncError(f"sync state does not bind {key}")


def _load_bound_manifest(
    control: Path, state: Mapping[str, object]
) -> tuple[dict[str, object], str]:
    manifest_path = control / INVENTORY_NAME
    manifest_bytes = _read_stable_regular_file(
        manifest_path, field="inventory manifest"
    )
    inventory_sha = _sha256_bytes(manifest_bytes)
    if state.get("inventory_sha256") != inventory_sha:
        raise SyncError("sync state does not bind the installed inventory")
    manifest = _load_inventory_bytes(manifest_bytes)
    if state.get("tree_sha256") != manifest.get("tree_sha256"):
        raise SyncError("sync state does not bind the inventory tree digest")
    return manifest, inventory_sha


def _assert_bound_control_programs(control: Path, state: Mapping[str, object]) -> None:
    for name, key in (
        (VERIFIER_NAME, "verifier_sha256"),
        (ORCHESTRATOR_NAME, "orchestrator_sha256"),
    ):
        payload = _read_stable_regular_file(control / name, field="control program")
        if _sha256_bytes(payload) != state.get(key):
            raise SyncError(f"sync state does not bind control program: {name}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace_linux(source: Path, destination: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise SyncError("atomic no-replace installation requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SyncError("libc renameat2 is unavailable; refusing unsafe fallback")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise SyncError(f"destination final appeared concurrently: {destination}")
        raise SyncError(
            f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error_number)}"
        )


def _write_verification_receipt(
    root: Path,
    control: Path,
    manifest: Mapping[str, object],
    inventory_sha: str,
) -> dict[str, object]:
    _reject_protected_destination_path(root.as_posix(), field="verified root")
    _reject_protected_destination_path(
        control.as_posix(), field="verification receipt control"
    )
    inventory_module = _load_inventory_module()
    receipt = dict(inventory_module.verify_inventory(root, manifest))
    receipt["inventory_manifest_sha256"] = inventory_sha
    receipt["tree_semantics"] = TREE_SEMANTICS
    receipt["verified_root"] = str(root)
    inventory_module._write_json_atomic(control / VERIFY_RECEIPT_NAME, receipt)
    _fsync_directory(control)
    return receipt


def _write_install_receipt(
    final: Path,
    control: Path,
    receipt: Mapping[str, object],
    inventory_sha: str,
    state: Mapping[str, object],
    *,
    recovered: bool,
) -> dict[str, object]:
    _reject_protected_destination_path(final.as_posix(), field="installed final")
    _reject_protected_destination_path(
        control.as_posix(), field="install receipt control"
    )
    install_receipt = {
        "status": "installed",
        "schema": SCHEMA,
        "destination_final": str(final),
        "file_count": receipt["file_count"],
        "total_size_bytes": receipt["total_size_bytes"],
        "tree_sha256": receipt["tree_sha256"],
        "tree_semantics": TREE_SEMANTICS,
        "source_identity": state["source_identity"],
        "destination_identity": state["destination_identity"],
        "inventory_manifest_sha256": inventory_sha,
        "verification_receipt": str(control / VERIFY_RECEIPT_NAME),
        "recovered_after_promotion": recovered,
        "installed_at_utc": _utc_now(),
    }
    inventory_module = _load_inventory_module()
    inventory_module._write_json_atomic(control / INSTALL_RECEIPT_NAME, install_receipt)
    _fsync_directory(control)
    return install_receipt


def _recover_installed_final(
    args: argparse.Namespace,
    manifest: Mapping[str, object],
    inventory_sha: str,
    state: Mapping[str, object],
) -> dict[str, object]:
    if os.path.lexists(args.stage):
        raise SyncError(
            "destination stage and final both exist; refusing ambiguous recovery"
        )
    _assert_no_symlink_tree(args.final)
    _assert_owned_private_directory(args.final, field="destination final")
    receipt = _write_verification_receipt(
        args.final, args.control, manifest, inventory_sha
    )
    return _write_install_receipt(
        args.final,
        args.control,
        receipt,
        inventory_sha,
        state,
        recovered=True,
    )


def _init_destination(args: argparse.Namespace) -> dict[str, object]:
    safe_prefix = args.safe_prefix
    stage = args.stage
    final = args.final
    control = args.control
    _validate_local_install_paths(safe_prefix, stage, final, control)
    for path in (safe_prefix, stage, final, control):
        _assert_no_symlink_ancestors(path)
    _assert_owned_private_directory(safe_prefix, field="safe prefix")
    _assert_owned_private_directory(control, field="control directory")
    _assert_no_symlink_tree(control)
    try:
        expected_state = _validate_state_payload(json.loads(args.state_json))
    except json.JSONDecodeError as exc:
        raise SyncError(f"invalid state JSON: {exc}") from exc
    _assert_state_destination_paths(expected_state, args)

    inventory = control / INVENTORY_NAME
    verifier = control / VERIFIER_NAME
    orchestrator = control / ORCHESTRATOR_NAME
    expected_files = {
        inventory: expected_state["inventory_sha256"],
        verifier: expected_state["verifier_sha256"],
        orchestrator: expected_state["orchestrator_sha256"],
    }
    for path, expected_sha in expected_files.items():
        payload = _read_stable_regular_file(path, field="control file")
        if _sha256_bytes(payload) != expected_sha:
            raise SyncError(f"control file hash mismatch: {path}")

    state_path = control / STATE_NAME
    if os.path.lexists(state_path):
        actual_state = _load_state(control)
        if actual_state != expected_state:
            raise SyncError(
                "existing staging directory belongs to a different sync state"
            )
        if os.path.lexists(final):
            manifest, inventory_sha = _load_bound_manifest(control, actual_state)
            return _recover_installed_final(args, manifest, inventory_sha, actual_state)
        if not os.path.lexists(stage):
            raise SyncError("state exists but its bound staging directory is missing")
        _assert_no_symlink_tree(stage)
        _assert_owned_private_directory(stage, field="destination stage")
    else:
        if os.path.lexists(stage):
            raise SyncError("unbound destination stage already exists")
        if os.path.lexists(final):
            raise SyncError("destination final exists without matching sync state")
        try:
            stage.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise SyncError("destination stage appeared concurrently") from exc
        inventory_module = _load_inventory_module()
        try:
            inventory_module._write_json_atomic(state_path, expected_state)
            _fsync_directory(control)
            _fsync_directory(stage.parent)
        except BaseException:
            try:
                stage.rmdir()
            except OSError:
                pass
            raise
    return {"status": "destination_initialized", "stage": str(stage)}


def _assert_stage_safe(args: argparse.Namespace) -> dict[str, object]:
    _validate_local_install_paths(
        args.safe_prefix, args.stage, args.final, args.control
    )
    for path in (args.safe_prefix, args.stage, args.final, args.control):
        _assert_no_symlink_ancestors(path)
    _assert_owned_private_directory(args.safe_prefix, field="safe prefix")
    _assert_owned_private_directory(args.control, field="control directory")
    if os.path.lexists(args.final):
        raise SyncError(f"destination final already exists: {args.final}")
    _assert_no_symlink_tree(args.control)
    _assert_no_symlink_tree(args.stage)
    if not args.stage.is_dir():
        raise SyncError("destination stage does not exist")
    _assert_owned_private_directory(args.stage, field="destination stage")
    state = _load_state(args.control)
    _assert_state_destination_paths(state, args)
    _assert_bound_control_programs(args.control, state)
    _load_bound_manifest(args.control, state)
    return {"status": "stage_safe", "stage": str(args.stage)}


def _verify_and_install(args: argparse.Namespace) -> dict[str, object]:
    _validate_local_install_paths(
        args.safe_prefix, args.stage, args.final, args.control
    )
    for path in (args.safe_prefix, args.stage, args.final, args.control):
        _assert_no_symlink_ancestors(path)
    _assert_owned_private_directory(args.safe_prefix, field="safe prefix")
    _assert_owned_private_directory(args.control, field="control directory")
    _assert_no_symlink_tree(args.control)
    state = _load_state(args.control)
    _assert_state_destination_paths(state, args)
    _assert_bound_control_programs(args.control, state)
    manifest, inventory_sha = _load_bound_manifest(args.control, state)
    if os.path.lexists(args.final):
        return _recover_installed_final(args, manifest, inventory_sha, state)
    if not os.path.lexists(args.stage):
        raise SyncError("destination stage is missing and final is not installed")
    _assert_no_symlink_tree(args.stage)
    _assert_owned_private_directory(args.stage, field="destination stage")
    receipt = _write_verification_receipt(
        args.stage, args.control, manifest, inventory_sha
    )
    _assert_no_symlink_tree(args.stage)
    _rename_noreplace_linux(args.stage, args.final)
    for parent in {args.stage.parent, args.final.parent}:
        _fsync_directory(parent)
    return _write_install_receipt(
        args.final,
        args.control,
        receipt,
        inventory_sha,
        state,
        recovered=False,
    )


def _run_sync(args: argparse.Namespace) -> dict[str, object]:
    config = SyncConfig(
        source_host=args.source_host,
        destination_host=args.destination_host,
        source_safe_prefix=args.source_safe_prefix,
        source_root=args.source_root,
        source_inventory=args.source_inventory,
        source_verifier=args.source_verifier,
        destination_safe_prefix=args.destination_safe_prefix,
        destination_stage=args.destination_stage,
        destination_final=args.destination_final,
        destination_control=args.destination_control,
        source_identity=args.source_identity,
        destination_identity=args.destination_identity,
        source_control_socket=args.source_control_socket,
        destination_control_socket=args.destination_control_socket,
    )
    return synchronize(config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="run the two-host verified stream")
    sync.add_argument("--source-host", required=True)
    sync.add_argument("--destination-host", required=True)
    sync.add_argument(
        "--source-identity",
        required=True,
        help="expected normalized lowercase /etc/machine-id for the source",
    )
    sync.add_argument(
        "--destination-identity",
        required=True,
        help="expected normalized lowercase /etc/machine-id for the destination",
    )
    sync.add_argument("--source-safe-prefix", required=True)
    sync.add_argument("--source-root", required=True)
    sync.add_argument("--source-inventory", required=True)
    sync.add_argument("--source-verifier", required=True)
    sync.add_argument("--destination-safe-prefix", required=True)
    sync.add_argument("--destination-stage", required=True)
    sync.add_argument("--destination-final", required=True)
    sync.add_argument("--destination-control", required=True)
    sync.add_argument("--source-control-socket", type=Path)
    sync.add_argument("--destination-control-socket", type=Path)
    sync.set_defaults(handler=_run_sync)

    component = subparsers.add_parser("verify-component", help=argparse.SUPPRESS)
    component.add_argument("--root", type=Path, required=True)
    component.add_argument("--manifest", type=Path, required=True)
    component.add_argument("--component", required=True)
    component.add_argument("--control", type=Path, required=True)
    component.set_defaults(
        handler=lambda args: _verify_component(
            args.root, args.manifest, args.component, args.control
        )
    )

    for name, handler in (
        ("init-destination", _init_destination),
        ("assert-stage-safe", _assert_stage_safe),
        ("verify-and-install", _verify_and_install),
    ):
        subparser = subparsers.add_parser(name, help=argparse.SUPPRESS)
        subparser.add_argument("--safe-prefix", type=Path, required=True)
        subparser.add_argument("--stage", type=Path, required=True)
        subparser.add_argument("--final", type=Path, required=True)
        subparser.add_argument("--control", type=Path, required=True)
        if name == "init-destination":
            subparser.add_argument("--state-json", required=True)
        subparser.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (SyncError, OSError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
