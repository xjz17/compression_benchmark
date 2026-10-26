#!/usr/bin/env python3
"""Download the frozen weighted-GRPO source cohort with object verification.

The downloader resolves every ``.tsfile`` at each exact BenchmarkHost revision.  The
deep-space-probes member is the single explicit exception: it resolves the
exact upstream Parquet and README overlay and never resolves the invalid
BenchmarkHost TsFile mirror.  Every object is size/hash checked, resumes into a
``.part`` file, and is published only after verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from huggingface_hub import HfApi

import rebuild_weighted_grpo_deep_space_v1 as deep_space


SCHEMA = "weighted-grpo-training-source-lock-v1"
DEFAULT_SPEC = Path(__file__).with_name(
    "weighted_grpo_training_source_revisions_v1.json"
)


class SourceDownloadError(RuntimeError):
    """A frozen source could not be resolved, downloaded, or verified."""


@dataclass(frozen=True)
class FrozenFile:
    dataset_id: str
    repo_id: str
    revision: str
    repo_path: str
    size: int
    sha256: str | None
    git_oid: str

    @property
    def relative_path(self) -> Path:
        return Path(self.dataset_id) / Path(*PurePosixPath(self.repo_path).parts)

    @property
    def url(self) -> str:
        path = "/".join(quote(part, safe="") for part in PurePosixPath(
            self.repo_path
        ).parts)
        return (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/"
            f"{self.revision}/{path}?download=true"
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_revision_spec(path: Path) -> list[dict[str, str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceDownloadError(f"cannot read source revision spec: {path}") from error
    if (not isinstance(document, Mapping)
            or document.get("schema") != "weighted-grpo-training-source-revisions-v1"):
        raise SourceDownloadError("unsupported source revision spec")
    rows = document.get("sources")
    if not isinstance(rows, list) or not rows:
        raise SourceDownloadError("source revision spec has no sources")
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceDownloadError("source revision row is not an object")
        dataset_id = str(row.get("dataset_id", ""))
        repo_id = str(row.get("repo_id", ""))
        revision = str(row.get("revision", ""))
        repository_allowed = repo_id.startswith("BenchmarkHost/") or (
            dataset_id == deep_space.DATASET_ID
            and repo_id == deep_space.UPSTREAM_REPO
            and row.get("source_overlay")
            == "exact_upstream_parquet_replaces_invalid_benchmarkhost_tsfile_mirror"
            and row.get("rejected_mirror_repo_id") == "BenchmarkHost/deep-space-probes"
            and row.get("rejected_mirror_revision")
            == "2ce6a7c5b0e22c3b6aa19459e4fa16a433f7bc8c"
        )
        if (not dataset_id or dataset_id in seen or not repository_allowed
                or len(revision) != 40
                or any(char not in "0123456789abcdef" for char in revision)):
            raise SourceDownloadError(f"invalid frozen source row: {row}")
        seen.add(dataset_id)
        output.append({
            "dataset_id": dataset_id, "repo_id": repo_id,
            "revision": revision,
        })
    return output


def resolve_inventory(
    sources: Sequence[Mapping[str, str]], api: HfApi | None = None
) -> list[FrozenFile]:
    client = api or HfApi()
    files: list[FrozenFile] = []
    for source in sources:
        dataset_id = source["dataset_id"]
        repo_id = source["repo_id"]
        revision = source["revision"]
        matches = []
        is_deep_space_overlay = (
            dataset_id == deep_space.DATASET_ID
            and repo_id == deep_space.UPSTREAM_REPO
            and revision == deep_space.UPSTREAM_REVISION
        )
        selected_paths = (
            {deep_space.UPSTREAM_REPO_PATH, deep_space.UPSTREAM_README_PATH}
            if is_deep_space_overlay
            else None
        )
        for entry in client.list_repo_tree(
            repo_id, repo_type="dataset", revision=revision,
            recursive=True, expand=True,
        ):
            path = str(getattr(entry, "path", ""))
            if (selected_paths is not None and path in selected_paths) or (
                selected_paths is None and path.lower().endswith(".tsfile")
            ):
                matches.append(entry)
        if selected_paths is not None and {str(entry.path) for entry in matches} != selected_paths:
            raise SourceDownloadError(
                f"deep-space exact-upstream overlay inventory changed at {repo_id}@{revision}"
            )
        if not matches:
            raise SourceDownloadError(f"no admitted source object found at {repo_id}@{revision}")
        for entry in sorted(matches, key=lambda item: str(item.path)):
            lfs = getattr(entry, "lfs", None)
            size = int(getattr(entry, "size", 0) or 0)
            git_oid = str(
                getattr(entry, "blob_id", None)
                or getattr(entry, "oid", "")
                or ""
            )
            if (len(git_oid) != 40
                    or any(char not in "0123456789abcdef" for char in git_oid)):
                raise SourceDownloadError(f"{repo_id}:{entry.path} lacks a Git OID")
            digest: str | None = None
            if lfs is not None:
                digest = str(getattr(lfs, "sha256", "") or "")
                lfs_size = int(getattr(lfs, "size", 0) or 0)
                if (size <= 0 or size != lfs_size or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    raise SourceDownloadError(
                        f"{repo_id}:{entry.path} has invalid LFS size/SHA-256"
                    )
            elif size <= 0:
                raise SourceDownloadError(f"{repo_id}:{entry.path} has invalid size")
            if is_deep_space_overlay:
                expected = {
                    deep_space.UPSTREAM_REPO_PATH: (
                        deep_space.SOURCE_BYTES,
                        deep_space.SOURCE_SHA256,
                        deep_space.SOURCE_GIT_OID,
                    ),
                    deep_space.UPSTREAM_README_PATH: (
                        deep_space.README_BYTES,
                        None,
                        deep_space.README_GIT_OID,
                    ),
                }[str(entry.path)]
                if (size, digest, git_oid) != expected:
                    raise SourceDownloadError(
                        f"deep-space exact-upstream object identity changed: {entry.path}"
                    )
            files.append(FrozenFile(
                dataset_id=dataset_id, repo_id=repo_id, revision=revision,
                repo_path=str(entry.path), size=size, sha256=digest,
                git_oid=git_oid,
            ))
    if len({item.relative_path for item in files}) != len(files):
        raise SourceDownloadError("resolved source inventory has duplicate destinations")
    return files


def verify_file(path: Path, frozen: FrozenFile) -> bool:
    if not path.is_file() or path.stat().st_size != frozen.size:
        return False
    if frozen.sha256 is not None:
        return sha256_file(path) == frozen.sha256
    return git_blob_sha1_file(path) == frozen.git_oid


def _aria2_command(aria2c: str, frozen: FrozenFile, part: Path) -> list[str]:
    return [
        aria2c, "--continue=true", "--allow-overwrite=true",
        "--auto-file-renaming=false", "--file-allocation=none",
        "--max-connection-per-server=4", "--split=4",
        "--min-split-size=1M", "--max-tries=0", "--retry-wait=3",
        "--connect-timeout=20", "--timeout=60", "--summary-interval=30",
        "--console-log-level=warn", f"--dir={part.parent}",
        f"--out={part.name}", frozen.url,
    ]


def download_one(root: Path, frozen: FrozenFile, aria2c: str) -> dict[str, Any]:
    final = root / frozen.relative_path
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    if verify_file(final, frozen):
        return {**asdict(frozen), "path": str(frozen.relative_path),
                "verified_sha256": sha256_file(final),
                "status": "already_verified"}
    if final.exists():
        raise SourceDownloadError(
            f"existing final file does not match frozen digest: {final}"
        )
    completed = subprocess.run(
        _aria2_command(aria2c, frozen, part), check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-12:])
        raise SourceDownloadError(
            f"aria2 failed for {frozen.relative_path}: rc={completed.returncode}\n{tail}"
        )
    if not verify_file(part, frozen):
        actual_size = part.stat().st_size if part.exists() else None
        actual_sha = sha256_file(part) if part.exists() else None
        actual_git_oid = git_blob_sha1_file(part) if part.exists() else None
        raise SourceDownloadError(
            f"download verification failed for {frozen.relative_path}: "
            f"size={actual_size}, sha256={actual_sha}, git_oid={actual_git_oid}"
        )
    verified_sha256 = sha256_file(part)
    os.replace(part, final)
    control = part.with_name(part.name + ".aria2")
    if control.exists():
        control.unlink()
    return {**asdict(frozen), "path": str(frozen.relative_path),
            "verified_sha256": verified_sha256,
            "status": "downloaded_and_verified"}


def write_lock_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify_lock(root: Path) -> dict[str, Any]:
    """Verify a copied cohort without network access or moving-branch lookup."""
    resolved_root = root.resolve()
    lock_path = resolved_root / "training_source_lock.json"
    try:
        document = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceDownloadError(f"cannot read source lock: {lock_path}") from error
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise SourceDownloadError("unsupported training source lock")
    expected_lock = str(document.get("lock_sha256", ""))
    basis = {key: value for key, value in document.items() if key != "lock_sha256"}
    actual_lock = hashlib.sha256(_canonical_json(basis)).hexdigest()
    if expected_lock != actual_lock:
        raise SourceDownloadError("training source lock digest mismatch")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise SourceDownloadError("training source lock has no files")
    checked: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceDownloadError("training source lock file row is invalid")
        relative = Path(str(row.get("path", "")))
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise SourceDownloadError("locked source path leaves cohort root") from error
        expected_size = int(row.get("size", -1))
        expected_sha = str(row.get("verified_sha256", ""))
        if (expected_size <= 0 or len(expected_sha) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha)):
            raise SourceDownloadError(f"invalid lock row for {relative}")
        if not candidate.is_file() or candidate.stat().st_size != expected_size:
            raise SourceDownloadError(f"locked source size mismatch: {relative}")
        actual_sha = sha256_file(candidate)
        if actual_sha != expected_sha:
            raise SourceDownloadError(f"locked source SHA-256 mismatch: {relative}")
        checked.append({
            "path": str(relative), "bytes": expected_size,
            "sha256": actual_sha,
        })
    actual_total = sum(row["bytes"] for row in checked)
    if actual_total != int(document.get("total_bytes", -1)):
        raise SourceDownloadError("training source lock total byte count mismatch")
    if len(checked) != int(document.get("file_count", -1)):
        raise SourceDownloadError("training source lock file count mismatch")
    return {
        "schema": "weighted-grpo-training-source-lock-verification-v1",
        "status": "verified",
        "root": str(resolved_root),
        "source_lock_path": str(lock_path),
        "source_lock_document_sha256": sha256_file(lock_path),
        "lock_sha256": actual_lock,
        "source_count": int(document.get("source_count", -1)),
        "file_count": len(checked),
        "total_bytes": actual_total,
        "files": checked,
    }


def run(
    spec: Path, root: Path, *, workers: int = 3, aria2c: str | None = None
) -> dict[str, Any]:
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    executable = aria2c or shutil.which("aria2c")
    if not executable:
        raise SourceDownloadError("aria2c is required for resumable downloads")
    sources = load_revision_spec(spec)
    inventory = resolve_inventory(sources)
    root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(download_one, root, frozen, executable): frozen
            for frozen in inventory
        }
        for future in as_completed(future_map):
            receipt = future.result()
            receipts.append(receipt)
            print(json.dumps({
                "dataset_id": receipt["dataset_id"],
                "path": receipt["path"], "size": receipt["size"],
                "status": receipt["status"],
            }, sort_keys=True), flush=True)
    receipts.sort(key=lambda row: (row["dataset_id"], row["repo_path"]))
    basis: dict[str, Any] = {
        "schema": SCHEMA,
        "spec_path": str(spec.resolve()),
        "spec_sha256": sha256_file(spec),
        "root": str(root.resolve()),
        "source_count": len(sources),
        "file_count": len(receipts),
        "total_bytes": sum(int(row["size"]) for row in receipts),
        "files": receipts,
    }
    basis["lock_sha256"] = hashlib.sha256(_canonical_json(basis)).hexdigest()
    write_lock_atomic(root / "training_source_lock.json", basis)
    return basis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_lock(args.root)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run(args.spec, args.root, workers=args.workers)
    print(json.dumps({
        "schema": result["schema"], "source_count": result["source_count"],
        "file_count": result["file_count"], "total_bytes": result["total_bytes"],
        "lock_sha256": result["lock_sha256"],
        "lock": str((args.root / "training_source_lock.json").resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
