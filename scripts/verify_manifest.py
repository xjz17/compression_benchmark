#!/usr/bin/env python3
"""Verify every tracked, non-gitlink path in MANIFEST.sha256."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tracked_paths() -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=ROOT
    )
    paths: set[str] = set()
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if mode == b"160000" or relative == MANIFEST.relative_to(ROOT).as_posix():
            continue
        paths.add(relative)
    return paths


def main() -> int:
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        sha256, relative = line.split("  ", 1)
        expected[relative] = sha256

    actual_paths = tracked_paths()
    problems: list[str] = []
    if actual_paths != set(expected):
        for relative in sorted(actual_paths - set(expected)):
            problems.append(f"unmanifested file: {relative}")
        for relative in sorted(set(expected) - actual_paths):
            problems.append(f"missing file: {relative}")
    for relative, expected_sha in sorted(expected.items()):
        path = ROOT / relative
        if path.is_file() and digest(path) != expected_sha:
            problems.append(f"digest mismatch: {relative}")
    if problems:
        print("FAIL: manifest verification")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"PASS: manifest verified ({len(expected)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
