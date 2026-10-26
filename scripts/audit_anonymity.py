#!/usr/bin/env python3
"""Fail closed on author-identity, private-infrastructure, and secret leaks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Strings are split so this file does not trigger its own audit.
FORBIDDEN_FRAGMENTS = tuple(
    value.encode()
    for value in (
        "xj" + "z17",
        "jz" + "xiao",
        "xiaojin" + "zhao",
        "THU" + "Lab",
        "web_" + "compression",
        "/Users/" + "xiao" + "jin" + "zhao",
        "/home/user/" + "xjz",
        "/data/user/" + "new_ai_compress",
        "timecho" + "-20",
        "208.68." + "39.109",
        "111.203." + "204.134",
        "server " + "166",
        "server " + "192",
    )
)
SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
PRIVATE_INFRA_PATTERNS = (
    re.compile(
        rb"\b(?:server|host|queue|timing|deployment|runs?)\s*[-_:]?\s*(?:166|192)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rb"\b(?:166|192)\s*(?:server|host|queue|timing|deployment|run|linux|bos)\b",
        re.IGNORECASE,
    ),
    re.compile(rb"\bon(?:166|192)\b", re.IGNORECASE),
    re.compile(rb"(?:166|192)(?:\xe6\x9c\x8d\xe5\x8a\xa1\xe5\x99\xa8|\xe9\xaa\x8c\xe6\x94\xb6|\xe9\x83\xa8\xe7\xbd\xb2|\xe9\xaa\x8c\xe8\xaf\x81)"),
)
FORBIDDEN_SUFFIXES = {".o", ".a", ".so", ".dylib", ".dll", ".exe", ".pyc", ".log"}
FORBIDDEN_NAMES = {".DS_Store", "CMakeCache.txt"}
FORBIDDEN_DIRS = {"__pycache__", "CMakeFiles", "node_modules", "results", "logs"}
MAX_ANONYMOUS_GITHUB_FILE_BYTES = 8 * 1024 * 1024


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def tracked_files() -> list[tuple[bytes, Path]]:
    output = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "--stage", "-z"]
    )
    files: list[tuple[bytes, Path]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        if mode == b"160000":
            continue
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        files.append((mode, ROOT / relative))
    return sorted(files, key=lambda item: item[1].relative_to(ROOT).as_posix())


def main() -> int:
    problems: list[str] = []
    checked = 0
    for mode, path in tracked_files():
        relative = path.relative_to(ROOT)
        if mode == b"120000" or path.is_symlink():
            problems.append(f"symbolic link: {relative}")
            continue
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            problems.append(f"generated/output directory: {relative}")
            continue
        if not path.is_file():
            problems.append(f"missing tracked file: {relative}")
            continue
        checked += 1
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden generated file: {relative}")
        if path.stat().st_size > MAX_ANONYMOUS_GITHUB_FILE_BYTES:
            problems.append(f"file exceeds Anonymous GitHub 8 MiB limit: {relative}")
        payload = path.read_bytes()
        lowered = payload.lower()
        for fragment in FORBIDDEN_FRAGMENTS:
            if fragment.lower() in lowered:
                problems.append(f"identity/infrastructure fragment in {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                problems.append(f"possible secret in {relative}")
        for pattern in PRIVATE_INFRA_PATTERNS:
            if pattern.search(payload):
                problems.append(f"private host alias in {relative}")

    if (ROOT / ".git").exists():
        try:
            if git_output("rev-list", "--count", "HEAD") != "1":
                problems.append("Git history must contain exactly one root commit")
            if git_output("rev-list", "--parents", "-n", "1", "HEAD").count(" "):
                problems.append("anonymous commit must not have a parent")
            identities = git_output(
                "log", "-1", "--format=%an <%ae>%n%cn <%ce>"
            ).splitlines()
            expected = "Anonymous Authors <anonymous@invalid>"
            if identities != [expected, expected]:
                problems.append(f"non-neutral Git identity: {identities!r}")
        except (OSError, subprocess.CalledProcessError) as exc:
            problems.append(f"unable to verify Git metadata: {exc}")

    if problems:
        print("FAIL: anonymity audit")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1
    print(f"PASS: anonymity audit ({checked} files; Git metadata verified when present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
