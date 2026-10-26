#!/usr/bin/env python3
"""Regenerate MANIFEST.sha256 for every tracked, non-gitlink artifact file."""

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


def tracked_files() -> list[Path]:
    """Return tracked blobs/symlinks, excluding gitlinks and this manifest."""

    output = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=ROOT
    )
    files: list[Path] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        if mode == b"160000" or relative == MANIFEST.relative_to(ROOT):
            continue
        files.append(ROOT / relative)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> int:
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in tracked_files()
    ]
    MANIFEST.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {MANIFEST.name}: {len(lines)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
