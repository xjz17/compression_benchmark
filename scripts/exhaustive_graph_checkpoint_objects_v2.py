"""Durable external checkpoint objects; never part of a compression frame.

The caller must hold its exclusive checkpoint worker lock. Publication writes
and fsyncs a unique temporary file, renames it, then fsyncs the directory before
returning a reference. SQLite may only commit references returned this way.
Pickles are local trusted checkpoint data, validated before deserialization.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import re
import stat
import uuid

CHUNK_BYTES = 1024 * 1024
OBJECT_SCHEMA = "exhaustive-checkpoint-object-v2"


class CheckpointObjectError(RuntimeError):
    """A checkpoint object cannot be trusted; the search must stop."""


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(directory: Path) -> None:
    directory = Path(directory)
    missing, ancestor = [], directory
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    directory.mkdir(parents=True, exist_ok=True)
    for path in reversed(missing):
        fsync_directory(path.parent)
    fsync_directory(directory.parent)


class _HashingWriter:
    def __init__(self, handle):
        self.handle, self.digest, self.size = handle, hashlib.sha256(), 0

    def write(self, value):
        view = memoryview(value)
        for offset in range(0, len(view), CHUNK_BYTES):
            block = view[offset:offset + CHUNK_BYTES]
            count = self.handle.write(block)
            if count != len(block):
                raise OSError("short checkpoint object write")
            self.digest.update(block)
            self.size += count
        return len(view)


class ExternalObjectStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        durable_mkdir(self.directory)

    def path(self, reference: dict) -> Path:
        if (not isinstance(reference, dict) or
                reference.get("schema") != OBJECT_SCHEMA or
                not isinstance(reference.get("sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) or
                type(reference.get("bytes")) is not int or reference["bytes"] < 0):
            raise CheckpointObjectError("invalid checkpoint object reference")
        return self.directory / (reference["sha256"] + ".bin")

    def _publish(self, write) -> dict:
        temporary = self.directory / (".writing-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as handle:
                writer = _HashingWriter(handle)
                write(writer)
                handle.flush()
                os.fsync(handle.fileno())
            reference = {"schema": OBJECT_SCHEMA, "sha256": writer.digest.hexdigest(),
                         "bytes": writer.size}
            destination = self.path(reference)
            if destination.exists():
                # Do not silently repair corruption or replace an object that
                # may still be referenced by the committed checkpoint.
                self.validate(reference)
                temporary.unlink()
            else:
                os.replace(temporary, destination)
            fsync_directory(self.directory)
            return reference
        finally:
            temporary.unlink(missing_ok=True)

    def put_bytes(self, value: bytes) -> dict:
        return self._publish(lambda writer: writer.write(value))

    def put_pickle(self, value) -> dict:
        # pickle.dump streams large bytes directly to disk; pickle.dumps would
        # allocate another complete multi-hundred-MB state in memory.
        return self._publish(lambda writer: pickle.dump(value, writer, protocol=5))

    def _verified_handle(self, reference: dict):
        path = self.path(reference)
        handle = None
        try:
            if path.is_symlink():
                raise CheckpointObjectError(f"checkpoint object is a symlink: {path}")
            handle = path.open("rb")
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != reference["bytes"]:
                raise CheckpointObjectError(f"checkpoint object size/type mismatch: {path}")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
                digest.update(block)
            if digest.hexdigest() != reference["sha256"]:
                raise CheckpointObjectError(f"checkpoint object checksum mismatch: {path}")
            handle.seek(0)
            return handle
        except BaseException as exc:
            if handle is not None:
                handle.close()
            if isinstance(exc, OSError):
                raise CheckpointObjectError(f"checkpoint object unavailable: {path}: {exc}") from exc
            raise

    def validate(self, reference: dict) -> None:
        with self._verified_handle(reference):
            pass

    def read_bytes(self, reference: dict) -> bytes:
        with self._verified_handle(reference) as handle:
            return handle.read()

    def load_pickle(self, reference: dict):
        with self._verified_handle(reference) as handle:
            try:
                value = pickle.load(handle)
                if handle.read(1):
                    raise ValueError("trailing data after checkpoint state")
                return value
            except Exception as exc:
                raise CheckpointObjectError("invalid checkpoint state pickle") from exc

    def export(self, reference: dict, destination: Path) -> None:
        # A copy, never a hard link: the user may replace or edit the export.
        destination = Path(destination)
        temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with self._verified_handle(reference) as source, temporary.open("xb") as target:
                digest = hashlib.sha256()
                for block in iter(lambda: source.read(CHUNK_BYTES), b""):
                    target.write(block)
                    digest.update(block)
                if digest.hexdigest() != reference["sha256"]:
                    raise CheckpointObjectError("checkpoint frame changed during export")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def collect(self, live_references, *, candidates=None, max_scan=256) -> int:
        """Bounded lazy GC under the exclusive worker lock, after SQL commit.

        The normal path only examines retired references; startup additionally
        removes orphan objects/temporary files from interrupted publication.
        Unknown files are left alone. Failure to reclaim may leak disk space,
        but can never invalidate a live reference.
        """
        live = {self.path(reference).name for reference in live_references}
        removed = 0
        entries = os.scandir(self.directory) if candidates is None else None
        # Bound deletion work, not the first directory entries: a prefix of
        # live objects must never permanently hide crash-created orphans.
        names = ((entry.name for entry in entries) if entries is not None else
                 (self.path(reference).name for reference in candidates[:max_scan]))
        try:
            for name in names:
                if removed >= max_scan:
                    break
                if name in live:
                    continue
                if not (re.fullmatch(r"[0-9a-f]{64}\.bin", name) or
                        re.fullmatch(r"\.writing-[0-9a-f]{32}", name)):
                    continue
                try:
                    (self.directory / name).unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    # Committed checkpoint remains valid even on GC I/O failure.
                    continue
        finally:
            if entries is not None:
                entries.close()
        if removed:
            fsync_directory(self.directory)
        return removed
