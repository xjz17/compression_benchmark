#!/usr/bin/env python3
"""Audit an artifact without publishing the authors' private search terms.

Optional author-specific terms belong in a UTF-8 file outside the artifact.
Every tracked file, including this scanner, is inspected. Extracted archives
are supported without requiring or claiming verification of Git metadata.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
PERSONAL_PATH_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9])(?:[A-Z]:[\\/][A-Za-z0-9_. -]{1,64}[\\/]|/(?:Users|home)/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
PRIVATE_HOST_PATTERN = re.compile(
    rb"\b(?:server|host|deployment)\s*[-_:]?\s*\d{2,3}\b(?![\d.])", re.IGNORECASE
)
THIRD_PARTY_PREFIXES = ("external/", "benchmark/reference/upstream/", "patches/vendor/")
FORBIDDEN_SUFFIXES = {".o", ".a", ".so", ".dylib", ".dll", ".exe", ".pyc", ".log"}
FORBIDDEN_NAMES = {".DS_Store", "CMakeCache.txt"}
FORBIDDEN_DIRS = {"__pycache__", "CMakeFiles", "node_modules", "results", "logs"}
MAX_FILE_BYTES = 8 * 1024 * 1024
NEUTRAL_IDENTITY = "Anonymous Authors <anonymous@invalid>"


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def is_checkout(root: Path) -> bool:
    try:
        return Path(git_output(root, "rev-parse", "--show-toplevel")).resolve() == root
    except (OSError, subprocess.CalledProcessError):
        return False


def artifact_files(root: Path, checkout: bool) -> list[tuple[str, Path]]:
    if not checkout:
        return [("", p) for p in sorted(root.rglob("*"))
                if ".git" not in p.relative_to(root).parts and (p.is_file() or p.is_symlink())]
    output = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
    files = []
    for record in output.split(b"\0"):
        if record:
            metadata, raw_path = record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0].decode("ascii")
            if mode != "160000":
                files.append((mode, root / raw_path.decode("utf-8", errors="surrogateescape")))
    return sorted(files, key=lambda item: item[1].relative_to(root).as_posix())


def literal_strings(payload: bytes, suffix: str) -> list[bytes]:
    """Also inspect constant Python concatenations without executing the code."""
    if suffix != ".py":
        return []
    try:
        tree = ast.parse(payload)
    except (SyntaxError, ValueError, UnicodeError):
        return []

    def folded(node: ast.AST) -> str | bytes | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = folded(node.left), folded(node.right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, bytes) and isinstance(right, bytes):
                return left + right
        return None

    values = []
    for node in ast.walk(tree):
        value = folded(node)
        if value is not None:
            values.append(value.encode("utf-8") if isinstance(value, str) else value)
    return values


def content_problems(name: str, payload: bytes, private_terms: tuple[bytes, ...]) -> list[str]:
    problems = []
    candidates = [payload, name.encode("utf-8"), *literal_strings(payload, Path(name).suffix)]
    for candidate in candidates:
        if any(term in candidate.lower() for term in private_terms):
            problems.append(f"private identity/infrastructure term in {name}")
        if any(pattern.search(candidate) for pattern in SECRET_PATTERNS):
            problems.append(f"possible secret in {name}")
        if not name.startswith(THIRD_PARTY_PREFIXES):
            if PERSONAL_PATH_PATTERN.search(candidate):
                problems.append(f"personal absolute path in {name}")
            if PRIVATE_HOST_PATTERN.search(candidate):
                problems.append(f"private host alias in {name}")
    return problems


def historical_content_problems(root: Path, private_terms: tuple[bytes, ...]) -> list[str]:
    problems = []
    seen = set()
    for commit in git_output(root, "rev-list", "HEAD").splitlines():
        message = git_output(root, "show", "-s", "--format=%B", commit).encode("utf-8")
        for problem in content_problems("commit-message", message, private_terms):
            problems.append(f"historical commit {commit[:12]}: {problem}")
        tree = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-z", commit])
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            if kind != b"blob":
                continue
            name = raw_path.decode("utf-8", errors="surrogateescape")
            if (name, oid) in seen:
                continue
            seen.add((name, oid))
            payload = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid.decode()])
            for problem in content_problems(name, payload, private_terms):
                problems.append(f"historical commit {commit[:12]}: {problem}")
            if mode == b"120000":
                problems.append(f"historical commit {commit[:12]}: symbolic link: {name}")
    return problems


def audit(root: Path, private_terms: tuple[bytes, ...], check_history: bool,
          check_git_identities: bool = False) -> tuple[list[str], int, bool]:
    root = root.resolve()
    checkout = is_checkout(root)
    problems: list[str] = []
    checked = 0
    for mode, path in artifact_files(root, checkout):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if mode == "120000" or path.is_symlink():
            problems.append(f"symbolic link: {name}")
            continue
        if not path.is_file():
            problems.append(f"missing tracked file: {name}")
            continue
        checked += 1
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            problems.append(f"generated/output directory: {name}")
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden generated file: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"file exceeds Anonymous GitHub 8 MiB limit: {name}")
        problems.extend(content_problems(name, path.read_bytes(), private_terms))
    if check_history or check_git_identities:
        if not checkout:
            problems.append("Git checks requested but no artifact checkout is available")
        else:
            identities = git_output(root, "log", "HEAD", "--format=%an <%ae>%n%cn <%ce>").splitlines()
            if not identities or any(value != NEUTRAL_IDENTITY for value in identities):
                problems.append("non-neutral author or committer in artifact history")
            if check_history:
                problems.extend(historical_content_problems(root, private_terms))
    return sorted(set(problems)), checked, checkout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--private-terms", type=Path,
                        help="external UTF-8 file, one literal private term per line; never publish it")
    parser.add_argument("--check-history", action="store_true",
                        help="inspect every reachable commit's content and author/committer identities")
    parser.add_argument("--check-git-identities", action="store_true",
                        help="check only author/committer metadata, without certifying historical content")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("artifact root must be an existing directory")
    private_terms: tuple[bytes, ...] = ()
    if args.private_terms:
        config = args.private_terms.resolve()
        if config.is_relative_to(root):
            parser.error("private terms must be stored outside the artifact root")
        private_terms = tuple(line.strip().encode("utf-8").lower()
                              for line in config.read_text(encoding="utf-8").splitlines()
                              if line.strip() and not line.lstrip().startswith("#"))
        if not private_terms:
            parser.error("private terms file must contain at least one term")
    problems, checked, _ = audit(root, private_terms, args.check_history, args.check_git_identities)
    if problems:
        print("FAIL: anonymity audit")
        for problem in problems:
            print(f"- {problem}")
        return 1
    git_status = ("content and identities verified" if args.check_history else
                  "identities only verified" if args.check_git_identities else "not checked")
    terms_status = "included" if private_terms else "not supplied"
    print(f"PASS: content audit ({checked} files; private terms {terms_status}; Git history {git_status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
