#!/usr/bin/env python3
"""Prepare and optionally publish BenchmarkHost dataset-card benchmark badges.

The default command is offline: it validates the frozen card manifest and
creates one byte-preserving README patch per dataset.  ``--publish`` uses the
normal Hugging Face SDK credential lookup and creates a one-file commit only
when the repository HEAD still equals the manifest's pinned revision.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "analysis_outputs/dataset_badges_20261009/cards_manifest.json"
EXPECTED_MANIFEST_SHA256 = "54289c6abf63378bf98fe5339bf814e0a15f1c69eb9072cfb72c58117a2e846f"
EXPECTED_REPOSITORIES = 151
EXPECTED_MEASURED = 146
EXPECTED_CATALOG_ONLY = 5
EXPECTED_CATALOG_ONLY_KEYS = {
    "app_flow",
    "ethereum_signals_sample",
    "github_top_developers",
    "tiktok_trending_hashtags_music",
    "youtube_trending_2019_2020",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
DATASET_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
BADGE_START = b"<!-- BENCHMARKHOST_COMPRESSION_BENCHMARK_BADGE_START -->"
BADGE_END = b"<!-- BENCHMARKHOST_COMPRESSION_BENCHMARK_BADGE_END -->"
FORBIDDEN_ADDITIONS = (b"benchmark:official", b"benchmark:eval-yaml")
COMMIT_MESSAGE = "Link dataset card to BenchmarkHost Compression Benchmark"


class BadgePublishError(RuntimeError):
    """A validation or publication gate failed closed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_regular_bytes(path: Path, role: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise BadgePublishError(f"cannot stat {role}: {path}: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise BadgePublishError(f"{role} must be a regular non-symlink file: {path}")
    binding = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BadgePublishError(f"cannot read {role}: {path}: {exc}") from exc
    after = path.stat()
    if binding != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise BadgePublishError(f"{role} changed while being read: {path}")
    return payload


def resolve_repo_relative_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BadgePublishError(f"{role} must be a non-empty repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or any(part in {"", ".", ".."} for part in pure.parts):
        raise BadgePublishError(f"unsafe {role}: {value!r}")
    path = (REPO_ROOT / pure.as_posix()).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise BadgePublishError(f"{role} escapes the repository: {value!r}") from exc
    return path


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = read_regular_bytes(path, "card manifest")
    manifest_sha256 = sha256_bytes(payload)
    if path.resolve() == DEFAULT_MANIFEST.resolve() and manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise BadgePublishError(
            f"default manifest SHA-256 mismatch: expected {EXPECTED_MANIFEST_SHA256}, found {manifest_sha256}"
        )
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadgePublishError(f"card manifest is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise BadgePublishError("card manifest must have schema_version=1 and status=PASS")
    rows = manifest.get("rows")
    counts = manifest.get("counts")
    if not isinstance(rows, list) or not isinstance(counts, dict):
        raise BadgePublishError("card manifest rows/counts are missing")
    expected_counts = {
        "repositories": EXPECTED_REPOSITORIES,
        "measured": EXPECTED_MEASURED,
        "catalog_only": EXPECTED_CATALOG_ONLY,
        "http_200": EXPECTED_REPOSITORIES,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise BadgePublishError(f"manifest count {key!r} must be {expected}, found {counts.get(key)!r}")
    if len(rows) != EXPECTED_REPOSITORIES:
        raise BadgePublishError(f"manifest must contain {EXPECTED_REPOSITORIES} rows")
    return manifest, manifest_sha256


def validate_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_keys: set[str] = set()
    repo_ids: set[str] = set()
    readme_paths: set[str] = set()
    measured = 0
    catalog_only: set[str] = set()
    for index, raw in enumerate(manifest["rows"]):
        if not isinstance(raw, dict):
            raise BadgePublishError(f"manifest row {index} is not an object")
        dataset_key = raw.get("dataset_key")
        repo_id = raw.get("repo_id")
        revision = raw.get("revision")
        dataset_url = raw.get("dataset_url")
        status = raw.get("measurement_status")
        readme_path = raw.get("readme_path")
        readme_sha256 = raw.get("readme_sha256")
        if not isinstance(dataset_key, str) or not DATASET_KEY_RE.fullmatch(dataset_key):
            raise BadgePublishError(f"row {index} has unsafe dataset_key")
        if repo_id != f"BenchmarkHost/{dataset_key}":
            raise BadgePublishError(f"row {index} repo_id does not exactly match dataset_key")
        if dataset_url != f"https://huggingface.co/datasets/{repo_id}":
            raise BadgePublishError(f"row {index} dataset_url does not exactly match repo_id")
        if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
            raise BadgePublishError(f"row {index} has an invalid pinned revision")
        if status not in {"measured", "catalog_only"}:
            raise BadgePublishError(f"row {index} has an invalid measurement_status")
        if raw.get("readme_http_status") != 200:
            raise BadgePublishError(f"row {index} README was not captured with HTTP 200")
        if not isinstance(readme_path, str):
            raise BadgePublishError(f"row {index} has no README snapshot path")
        if not isinstance(readme_sha256, str) or not SHA256_RE.fullmatch(readme_sha256):
            raise BadgePublishError(f"row {index} has an invalid README SHA-256")
        if dataset_key in dataset_keys or repo_id in repo_ids or readme_path in readme_paths:
            raise BadgePublishError(f"manifest identity is not unique at row {index}")
        dataset_keys.add(dataset_key)
        repo_ids.add(repo_id)
        readme_paths.add(readme_path)
        if status == "measured":
            measured += 1
        else:
            catalog_only.add(dataset_key)
        rows.append(raw)
    if measured != EXPECTED_MEASURED or catalog_only != EXPECTED_CATALOG_ONLY_KEYS:
        raise BadgePublishError("manifest measured/catalog-only partition does not match the frozen scope")
    return sorted(rows, key=lambda row: row["dataset_key"].encode("utf-8"))


def newline_for(payload: bytes) -> bytes:
    first_lf = payload.find(b"\n")
    if first_lf > 0 and payload[first_lf - 1 : first_lf + 1] == b"\r\n":
        return b"\r\n"
    return b"\n"


def frontmatter_end(payload: bytes) -> int:
    offset = 3 if payload.startswith(b"\xef\xbb\xbf") else 0
    first_lf = payload.find(b"\n", offset)
    if first_lf < 0 or payload[offset:first_lf].rstrip(b"\r") != b"---":
        return offset
    cursor = first_lf + 1
    while cursor < len(payload):
        line_end = payload.find(b"\n", cursor)
        if line_end < 0:
            line_end = len(payload)
            next_cursor = line_end
        else:
            next_cursor = line_end + 1
        if payload[cursor:line_end].rstrip(b"\r") in {b"---", b"..."}:
            return next_cursor
        cursor = next_cursor
    raise BadgePublishError("README starts YAML front matter but has no closing delimiter")


def strip_managed_badge(payload: bytes) -> tuple[bytes, bool]:
    starts = payload.count(BADGE_START)
    ends = payload.count(BADGE_END)
    if starts == 0 and ends == 0:
        return payload, False
    if starts != 1 or ends != 1:
        raise BadgePublishError("README has malformed or duplicate managed badge markers")
    start = payload.index(BADGE_START)
    end = payload.index(BADGE_END)
    if end < start:
        raise BadgePublishError("README managed badge markers are reversed")
    end += len(BADGE_END)
    newline = newline_for(payload)
    if payload[end : end + len(newline) * 2] != newline * 2:
        raise BadgePublishError("existing managed badge does not use the stable two-newline boundary")
    end += len(newline) * 2
    return payload[:start] + payload[end:], True


def benchmark_url(dataset_key: str) -> str:
    encoded = quote(dataset_key, safe="")
    return (
        "https://benchmarkhost-compressionbenchmark.static.hf.space/"
        f"index.html?scope=169&dataset={encoded}#dataset"
    )


def badge_block(dataset_key: str, measurement_status: str, newline: bytes) -> bytes:
    target = benchmark_url(dataset_key)
    if measurement_status == "measured":
        note = "Compare lossless compression ratio, compression time, and decompression time."
    else:
        note = "Listed in the benchmark catalog; measurements are not yet available."
    lines = [
        BADGE_START.decode("ascii"),
        (
            "[🏆 ![BenchmarkHost Compression Benchmark]"
            "(https://img.shields.io/badge/BenchmarkHost-Compression_Benchmark-EAB308?style=flat&labelColor=334155)]"
            f"({target})"
        ),
        f"<sub>{note}</sub>",
        BADGE_END.decode("ascii"),
        "",
        "",
    ]
    return newline.join(line.encode("utf-8") for line in lines)


def transform_readme(before: bytes, dataset_key: str, measurement_status: str) -> bytes:
    clean, _ = strip_managed_badge(before)
    insertion = frontmatter_end(clean)
    newline = newline_for(clean)
    after = clean[:insertion] + badge_block(dataset_key, measurement_status, newline) + clean[insertion:]
    if after.count(BADGE_START) != 1 or after.count(BADGE_END) != 1:
        raise BadgePublishError("prepared README does not contain exactly one managed badge")
    roundtrip, present = strip_managed_badge(after)
    if not present or roundtrip != clean:
        raise BadgePublishError("managed badge removal did not restore the exact README baseline")
    if transform_readme_once(after, dataset_key, measurement_status) != after:
        raise BadgePublishError("README transformation is not idempotent")
    for forbidden in FORBIDDEN_ADDITIONS:
        if after.count(forbidden) != before.count(forbidden):
            raise BadgePublishError(f"prepared README changes reserved Hub tag {forbidden.decode()!r}")
    return after


def transform_readme_once(before: bytes, dataset_key: str, measurement_status: str) -> bytes:
    clean, _ = strip_managed_badge(before)
    insertion = frontmatter_end(clean)
    newline = newline_for(clean)
    return clean[:insertion] + badge_block(dataset_key, measurement_status, newline) + clean[insertion:]


def exact_diff(before: bytes, after: bytes, repo_id: str) -> bytes:
    return b"".join(
        difflib.diff_bytes(
            difflib.unified_diff,
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{repo_id}/README.md".encode("utf-8"),
            tofile=f"b/{repo_id}/README.md".encode("utf-8"),
            lineterm=b"\n",
        )
    )


def prepare(manifest_path: Path, artifact_root: Path) -> tuple[dict[str, Any], Path]:
    manifest, manifest_sha256 = load_manifest(manifest_path)
    rows = validate_rows(manifest)
    prepared_root = artifact_root / "prepared_cards"
    plan_rows: list[dict[str, Any]] = []
    marker_replacements = 0
    for row in rows:
        baseline_path = resolve_repo_relative_path(row["readme_path"], "README snapshot")
        before = read_regular_bytes(baseline_path, f"README snapshot for {row['repo_id']}")
        if len(before) != row.get("readme_bytes") or sha256_bytes(before) != row["readme_sha256"]:
            raise BadgePublishError(f"README snapshot identity mismatch for {row['repo_id']}")
        _, had_marker = strip_managed_badge(before)
        marker_replacements += int(had_marker)
        after = transform_readme(before, row["dataset_key"], row["measurement_status"])
        if sha256_bytes(read_regular_bytes(baseline_path, "README snapshot after transform")) != row["readme_sha256"]:
            raise BadgePublishError(f"README snapshot changed during preparation for {row['repo_id']}")
        card_root = prepared_root / row["dataset_key"]
        readme_output = card_root / "README.md"
        diff_output = card_root / "README.diff"
        difference = exact_diff(before, after, row["repo_id"])
        if not difference:
            raise BadgePublishError(f"prepared README has no diff for {row['repo_id']}")
        atomic_write(readme_output, after)
        atomic_write(diff_output, difference)
        plan_rows.append(
            {
                "dataset_key": row["dataset_key"],
                "repo_id": row["repo_id"],
                "dataset_url": row["dataset_url"],
                "measurement_status": row["measurement_status"],
                "parent_commit": row["revision"],
                "source_readme_path": str(baseline_path.relative_to(REPO_ROOT)),
                "source_readme_sha256": row["readme_sha256"],
                "prepared_readme_path": str(readme_output.relative_to(REPO_ROOT)),
                "prepared_readme_sha256": sha256_bytes(after),
                "prepared_readme_bytes": len(after),
                "diff_path": str(diff_output.relative_to(REPO_ROOT)),
                "diff_sha256": sha256_bytes(difference),
                "benchmark_url": benchmark_url(row["dataset_key"]),
                "commit_operation": {"action": "add_or_update", "path_in_repo": "README.md"},
                "commit_message": COMMIT_MESSAGE,
            }
        )
    plan = {
        "schema_version": 1,
        "status": "PREPARED",
        "generated_at": manifest.get("generated_at"),
        "source_manifest_path": str(manifest_path.resolve().relative_to(REPO_ROOT)),
        "source_manifest_sha256": manifest_sha256,
        "counts": {
            "repositories": len(plan_rows),
            "measured": sum(row["measurement_status"] == "measured" for row in plan_rows),
            "catalog_only": sum(row["measurement_status"] == "catalog_only" for row in plan_rows),
            "existing_managed_badges_replaced": marker_replacements,
            "prepared": len(plan_rows),
        },
        "safety": {
            "default_mode_writes_remote": False,
            "publish_requires_flag": "--publish",
            "remote_operation_count_per_repo": 1,
            "remote_path": "README.md",
            "parent_commit_pinned": True,
            "reserved_benchmark_tags_added": False,
            "eval_yaml_added": False,
            "baseline_snapshots_modified": False,
        },
        "rows": plan_rows,
    }
    plan_path = artifact_root / "publication_plan.json"
    atomic_write(plan_path, canonical_json_bytes(plan))
    review_lines = [
        "# BenchmarkHost dataset-card benchmark badge review",
        "",
        f"Status: PREPARED ({len(plan_rows)} repositories; no remote writes)",
        "",
        f"- Source manifest SHA-256: `{manifest_sha256}`",
        f"- Measured datasets: {plan['counts']['measured']}",
        f"- Catalog-only datasets: {plan['counts']['catalog_only']}",
        f"- Existing managed badge blocks replaced: {marker_replacements}",
        "- Every source README hash and byte count matched the frozen manifest.",
        "- Removing the managed block restores every baseline README byte-for-byte, including YAML front matter.",
        "- Reapplying the transformation is byte-identical (idempotent).",
        "- The prepared change adds no `benchmark:official`, `benchmark:eval-yaml`, or `eval.yaml`.",
        "- Publication is limited to one `README.md` operation with a pinned parent commit per repository.",
        "",
        "## Catalog-only datasets",
        "",
    ]
    review_lines.extend(f"- `{key}` — measurements are not yet available" for key in sorted(EXPECTED_CATALOG_ONLY_KEYS))
    review_lines.extend(
        [
            "",
            "## Managed block",
            "",
            "The block is bounded by `BENCHMARKHOST_COMPRESSION_BENCHMARK_BADGE_START` and ",
            "`BENCHMARKHOST_COMPRESSION_BENCHMARK_BADGE_END`. The trophy and badge link directly to the dataset view.",
            "",
        ]
    )
    atomic_write(artifact_root / "review.md", "\n".join(review_lines).encode("utf-8"))
    return plan, plan_path


def exception_receipt(exc: BaseException) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    category = "authentication_or_authorization" if status in {401, 403} else "sdk_or_network_error"
    return {"category": category, "type": type(exc).__name__, "http_status": status}


def matching_verified_receipt(
    path: Path, row: dict[str, Any], manifest_sha256: str
) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    matches = (
        receipt.get("status") == "VERIFIED"
        and receipt.get("repo_id") == row["repo_id"]
        and receipt.get("source_manifest_sha256") == manifest_sha256
        and receipt.get("parent_commit") == row["parent_commit"]
        and receipt.get("prepared_readme_sha256") == row["prepared_readme_sha256"]
        and isinstance(receipt.get("published_commit"), str)
        and REVISION_RE.fullmatch(receipt["published_commit"]) is not None
    )
    return receipt if matches else None


def remote_readme_sha256(
    hf_hub_download: Any, repo_id: str, revision: str
) -> str:
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="README.md",
            revision=revision,
        )
    )
    try:
        resolved_path = cached_path.resolve(strict=True)
    except OSError as exc:
        raise BadgePublishError(
            f"cannot resolve cached immutable README for {repo_id}: {exc}"
        ) from exc
    return sha256_bytes(
        read_regular_bytes(resolved_path, f"immutable remote README for {repo_id}")
    )


def publish(plan: dict[str, Any], artifact_root: Path) -> dict[str, int]:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as exc:
        raise BadgePublishError("huggingface_hub is required for --publish") from exc
    api = HfApi()
    receipt_root = artifact_root / "publish_receipts"
    summary = {
        "verified": 0,
        "recovered": 0,
        "skipped_verified": 0,
        "drift_refused": 0,
        "failed": 0,
    }
    for row in plan["rows"]:
        receipt_path = receipt_root / f"{row['dataset_key']}.json"
        base_receipt = {
            "schema_version": 1,
            "repo_id": row["repo_id"],
            "dataset_key": row["dataset_key"],
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "parent_commit": row["parent_commit"],
            "prepared_readme_sha256": row["prepared_readme_sha256"],
        }
        try:
            prepared_path = resolve_repo_relative_path(row["prepared_readme_path"], "prepared README")
            prepared = read_regular_bytes(prepared_path, f"prepared README for {row['repo_id']}")
            if sha256_bytes(prepared) != row["prepared_readme_sha256"]:
                raise BadgePublishError("prepared README hash no longer matches the publication plan")
            prior_receipt = matching_verified_receipt(
                receipt_path, row, plan["source_manifest_sha256"]
            )
            if prior_receipt is not None:
                published_commit = prior_receipt["published_commit"]
                immutable = api.repo_info(
                    row["repo_id"], repo_type="dataset", revision=published_commit
                ).sha
                immutable_readme_sha256 = remote_readme_sha256(
                    hf_hub_download, row["repo_id"], published_commit
                )
                current_head = api.repo_info(
                    row["repo_id"], repo_type="dataset", revision="main"
                ).sha
                current_readme_sha256 = remote_readme_sha256(
                    hf_hub_download, row["repo_id"], current_head
                )
                if (
                    immutable == published_commit
                    and immutable_readme_sha256 == row["prepared_readme_sha256"]
                    and current_readme_sha256 == row["prepared_readme_sha256"]
                ):
                    receipt = {
                        **base_receipt,
                        "status": "VERIFIED",
                        "published_commit": published_commit,
                        "immutable_revision": immutable,
                        "current_head": current_head,
                        "remote_readme_sha256": current_readme_sha256,
                        "verification_mode": "fresh_resume",
                    }
                    atomic_write(receipt_path, canonical_json_bytes(receipt))
                    summary["skipped_verified"] += 1
                    continue
            head_before = api.repo_info(row["repo_id"], repo_type="dataset", revision="main").sha
            if head_before != row["parent_commit"]:
                observed_readme_sha256 = remote_readme_sha256(
                    hf_hub_download, row["repo_id"], head_before
                )
                if observed_readme_sha256 == row["prepared_readme_sha256"]:
                    receipt = {
                        **base_receipt,
                        "status": "VERIFIED",
                        "published_commit": head_before,
                        "immutable_revision": head_before,
                        "current_head": head_before,
                        "remote_readme_sha256": observed_readme_sha256,
                        "verification_mode": "recovered_current_state",
                    }
                    atomic_write(receipt_path, canonical_json_bytes(receipt))
                    summary["recovered"] += 1
                    continue
                receipt = {
                    **base_receipt,
                    "status": "DRIFT_REFUSED",
                    "observed_head": head_before,
                    "observed_readme_sha256": observed_readme_sha256,
                }
                atomic_write(receipt_path, canonical_json_bytes(receipt))
                summary["drift_refused"] += 1
                continue
            result = api.create_commit(
                repo_id=row["repo_id"],
                repo_type="dataset",
                revision="main",
                parent_commit=row["parent_commit"],
                commit_message=row["commit_message"],
                operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=prepared)],
            )
            commit = getattr(result, "oid", None)
            if not isinstance(commit, str) or not REVISION_RE.fullmatch(commit):
                raise BadgePublishError("Hugging Face did not return an immutable 40-hex commit")
            immutable = api.repo_info(row["repo_id"], repo_type="dataset", revision=commit).sha
            remote_sha256 = remote_readme_sha256(
                hf_hub_download, row["repo_id"], commit
            )
            head_after = api.repo_info(row["repo_id"], repo_type="dataset", revision="main").sha
            if immutable != commit or head_after != commit or remote_sha256 != row["prepared_readme_sha256"]:
                raise BadgePublishError("published commit, immutable README, or current HEAD failed verification")
            receipt = {
                **base_receipt,
                "status": "VERIFIED",
                "published_commit": commit,
                "immutable_revision": immutable,
                "current_head": head_after,
                "remote_readme_sha256": remote_sha256,
                "verification_mode": "new_commit",
            }
            atomic_write(receipt_path, canonical_json_bytes(receipt))
            summary["verified"] += 1
        except Exception as exc:
            receipt = {**base_receipt, "status": "FAILED", "error": exception_receipt(exc)}
            atomic_write(receipt_path, canonical_json_bytes(receipt))
            summary["failed"] += 1
    atomic_write(
        artifact_root / "publish_summary.json",
        canonical_json_bytes(
            {
                "schema_version": 1,
                "status": "PASS" if summary["failed"] == 0 and summary["drift_refused"] == 0 else "INCOMPLETE",
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "counts": summary,
            }
        ),
    )
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Output directory (default: the manifest directory)",
    )
    parser.add_argument("--publish", action="store_true", help="Publish pinned one-file README commits")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = args.manifest.resolve()
    artifact_root = (args.artifact_root or manifest_path.parent).resolve()
    try:
        artifact_root.relative_to(REPO_ROOT)
    except ValueError:
        print("ERROR BadgePublishError: artifact root must stay inside the repository", file=sys.stderr)
        return 2
    try:
        plan, plan_path = prepare(manifest_path, artifact_root)
        print(
            json.dumps(
                {
                    "status": "PREPARED",
                    "repositories": len(plan["rows"]),
                    "publication_plan": str(plan_path),
                    "remote_writes": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        if not args.publish:
            return 0
        summary = publish(plan, artifact_root)
        print(json.dumps({"status": "PUBLISH_COMPLETE", "counts": summary}, sort_keys=True))
        return 0 if summary["failed"] == 0 and summary["drift_refused"] == 0 else 3
    except Exception as exc:
        if isinstance(exc, BadgePublishError):
            print(f"ERROR BadgePublishError: {exc}", file=sys.stderr)
        else:
            safe = exception_receipt(exc)
            print(f"ERROR {safe['type']} http_status={safe['http_status']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
