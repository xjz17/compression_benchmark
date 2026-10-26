#!/usr/bin/env python3
"""Create and verify a relocatable, exact file-tree inventory.

The inventory contains POSIX relative paths, byte sizes, and SHA-256 hashes.
Absolute source paths are deliberately excluded so the same manifest can verify
an independently located copy.  Symlinks and non-regular filesystem objects are
rejected: a successful verification therefore means that the regular-file set
is exact (no missing or extra files) and every byte payload matches.

Keep the manifest outside the inventoried root.  This avoids self-reference and
makes it possible to install an incoming tree atomically after verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence


SCHEMA = "project179-relocatable-tree-inventory-v1"
BUFFER_BYTES = 8 * 1024 * 1024


class InventoryError(RuntimeError):
    """Raised when a tree or manifest violates the inventory contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_relative_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise InventoryError(f"invalid relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix():
        raise InventoryError(f"non-canonical relative path: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise InventoryError(f"unsafe relative path: {raw!r}")
    return raw


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_regular_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield every regular file while rejecting links and special objects."""
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            child = current_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise InventoryError(f"symlink directory is forbidden: {child.relative_to(root)}")
            if not stat.S_ISDIR(mode):
                raise InventoryError(f"non-directory tree node is forbidden: {child.relative_to(root)}")
        for name in filenames:
            child = current_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise InventoryError(f"symlink is forbidden: {child.relative_to(root)}")
            if not stat.S_ISREG(mode):
                raise InventoryError(f"non-regular file is forbidden: {child.relative_to(root)}")
            relative = child.relative_to(root).as_posix()
            yield _safe_relative_path(relative), child


def _hash_stable_regular_file(path: Path) -> tuple[int, str]:
    """Hash one file and reject replacement/modification during the read."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise InventoryError(f"not a regular file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise InventoryError(f"not a regular file after open: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, BUFFER_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened_before = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
        opened_before.st_mtime_ns,
    )
    identity_opened_after = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if not (
        identity_before
        == identity_opened_before
        == identity_opened_after
        == identity_after
    ):
        raise InventoryError(f"file changed while hashing: {path}")
    return opened_after.st_size, digest.hexdigest()


def _entry_line(entry: Mapping[str, object]) -> bytes:
    record = [entry["path"], entry["size_bytes"], entry["sha256"]]
    return (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _tree_digest(entries: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(_entry_line(entry))
    return digest.hexdigest()


def build_inventory(root: Path, *, root_label: str) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise InventoryError(f"root is not a directory: {root}")
    if not root_label.strip():
        raise InventoryError("root label must not be empty")

    discovered = sorted(_iter_regular_files(root), key=lambda item: item[0])
    entries: list[dict[str, object]] = []
    for relative, path in discovered:
        size_bytes, sha256 = _hash_stable_regular_file(path)
        entries.append({"path": relative, "size_bytes": size_bytes, "sha256": sha256})

    return {
        "schema": SCHEMA,
        "generated_at_utc": _utc_now(),
        "root_label": root_label,
        "path_semantics": "POSIX relative to the verified root; no symlinks",
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "tree_sha256": _tree_digest(entries),
        "entries": entries,
    }


def _validated_manifest(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise InventoryError(f"manifest schema must be {SCHEMA!r}")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise InventoryError("manifest entries must be a list")

    entries: list[dict[str, object]] = []
    last_path: str | None = None
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict) or set(raw) != {"path", "size_bytes", "sha256"}:
            raise InventoryError(f"invalid entry at index {index}")
        relative = _safe_relative_path(raw["path"])
        size_bytes = raw["size_bytes"]
        sha256 = raw["sha256"]
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise InventoryError(f"invalid size for {relative!r}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise InventoryError(f"invalid sha256 for {relative!r}")
        if last_path is not None and relative <= last_path:
            raise InventoryError("manifest paths must be unique and strictly sorted")
        last_path = relative
        entries.append({"path": relative, "size_bytes": size_bytes, "sha256": sha256})

    expected_count = payload.get("file_count")
    expected_bytes = payload.get("total_size_bytes")
    expected_tree_sha256 = payload.get("tree_sha256")
    if expected_count != len(entries):
        raise InventoryError("manifest file_count does not match entries")
    if expected_bytes != sum(int(entry["size_bytes"]) for entry in entries):
        raise InventoryError("manifest total_size_bytes does not match entries")
    if expected_tree_sha256 != _tree_digest(entries):
        raise InventoryError("manifest tree_sha256 does not match entries")

    validated = dict(payload)
    validated["entries"] = entries
    return validated


def load_inventory(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read manifest {path}: {exc}") from exc
    return _validated_manifest(payload)


def verify_inventory(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise InventoryError(f"root is not a directory: {root}")

    expected_entries = manifest["entries"]
    assert isinstance(expected_entries, list)
    expected_by_path = {str(entry["path"]): entry for entry in expected_entries}
    actual_paths = sorted(relative for relative, _ in _iter_regular_files(root))
    expected_paths = list(expected_by_path)

    missing = sorted(set(expected_paths) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected_paths))
    if missing or extra:
        preview = {
            "missing_count": len(missing),
            "missing_preview": missing[:20],
            "extra_count": len(extra),
            "extra_preview": extra[:20],
        }
        raise InventoryError(f"file set mismatch: {json.dumps(preview, ensure_ascii=False)}")

    mismatches: list[dict[str, object]] = []
    total_size_bytes = 0
    actual_entries: list[dict[str, object]] = []
    for relative in expected_paths:
        size_bytes, sha256 = _hash_stable_regular_file(root / Path(relative))
        total_size_bytes += size_bytes
        expected = expected_by_path[relative]
        actual_entries.append({"path": relative, "size_bytes": size_bytes, "sha256": sha256})
        if size_bytes != expected["size_bytes"] or sha256 != expected["sha256"]:
            mismatches.append(
                {
                    "path": relative,
                    "expected_size_bytes": expected["size_bytes"],
                    "actual_size_bytes": size_bytes,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": sha256,
                }
            )
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise InventoryError(
            f"content mismatch ({len(mismatches)} previewed): "
            f"{json.dumps(mismatches, ensure_ascii=False)}"
        )

    actual_tree_sha256 = _tree_digest(actual_entries)
    if actual_tree_sha256 != manifest["tree_sha256"]:
        raise InventoryError("verified entries produced an unexpected tree_sha256")
    return {
        "status": "verified",
        "schema": SCHEMA,
        "root_label": manifest.get("root_label"),
        "file_count": len(actual_entries),
        "total_size_bytes": total_size_bytes,
        "tree_sha256": actual_tree_sha256,
        "verified_at_utc": _utc_now(),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _create_command(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if _path_is_within(output, root):
        raise InventoryError("manifest output must be outside the inventoried root")
    inventory = build_inventory(root, root_label=args.root_label)
    _write_json_atomic(output, inventory)
    return {
        "status": "created",
        "manifest": str(output),
        "file_count": inventory["file_count"],
        "total_size_bytes": inventory["total_size_bytes"],
        "tree_sha256": inventory["tree_sha256"],
    }


def _verify_command(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_inventory(args.manifest)
    receipt = verify_inventory(args.root, manifest)
    if args.receipt is not None:
        root = args.root.resolve(strict=True)
        receipt_path = args.receipt.resolve(strict=False)
        if _path_is_within(receipt_path, root):
            raise InventoryError("verification receipt must be outside the verified root")
        receipt = dict(receipt)
        receipt["inventory_manifest_sha256"] = hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest()
        _write_json_atomic(receipt_path, receipt)
        receipt["receipt"] = str(receipt_path)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="hash a completed source tree")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--root-label", required=True)
    create.set_defaults(handler=_create_command)

    verify = subparsers.add_parser("verify", help="verify an exact relocated copy")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--receipt", type=Path)
    verify.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (InventoryError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
