#!/usr/bin/env python3
"""Verify the manifest in a source checkout or extracted review archive."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tracked_paths(root: Path) -> set[str]:
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        checkout = Path(top).resolve() == root
    except (OSError, subprocess.CalledProcessError):
        checkout = False
    if not checkout:
        return {p.relative_to(root).as_posix() for p in root.rglob("*")
                if ".git" not in p.relative_to(root).parts
                and (p.is_file() or p.is_symlink())
                and p.relative_to(root).as_posix() != "MANIFEST.sha256"}
    output = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=root
    )
    paths: set[str] = set()
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if mode == b"160000" or relative == "MANIFEST.sha256":
            continue
        paths.add(relative)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sha256, relative = line.split("  ", 1)
        if Path(relative).is_absolute() or ".." in Path(relative).parts or relative in expected:
            print("FAIL: unsafe or duplicate manifest path")
            return 1
        expected[relative] = sha256

    actual_paths = tracked_paths(root)
    problems: list[str] = []
    if actual_paths != set(expected):
        for relative in sorted(actual_paths - set(expected)):
            problems.append(f"unmanifested file: {relative}")
        for relative in sorted(set(expected) - actual_paths):
            problems.append(f"missing file: {relative}")
    for relative, expected_sha in sorted(expected.items()):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            problems.append(f"missing or non-regular file: {relative}")
        elif digest(path) != expected_sha:
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
