#!/usr/bin/env python3
"""Scan benchmark/reference/papers/*.pdf for GitHub URLs.

Extracts text with pypdf (first N pages per file by default). Normalizes URLs to
``owner/repo`` roots, skips obvious non-implementation links, and compares
against URLs already listed in ``scripts/init_compression_submodules.sh``
(external clones are **not** committed in this repo).

The output CSV (``benchmark/reference/papers_github_from_pdf.csv``) is only a
**first pass**. Always double-check each important repo against the paper’s
**artifact / code availability / supplementary** section before adding a clone
path. See ``benchmark/reference/external_compression_integration.md``.

Usage::

  python3 scripts/ingest_github_from_paper_pdfs.py --dry-run
  python3 scripts/ingest_github_from_paper_pdfs.py --suggest-clones

Optional: ``pip install pypdf`` in the repo venv.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS = ROOT / "benchmark" / "reference" / "papers"
DEFAULT_MANIFEST = ROOT / "benchmark" / "reference" / "papers_github_from_pdf.csv"

# Whole-repo submodules we do not vendor (see compression-research-import skill).
SKIP_REPO_ROOTS_LC: set[tuple[str, str]] = {
    ("duckdb", "duckdb"),
    ("cwida", "duckdb"),
    ("apache", "arrow"),
    ("tensorflow", "tensorflow"),
    ("pytorch", "pytorch"),
    # Typical citation dependencies / non–compression-algorithm repos in survey PDFs
    ("facebook", "rocksdb"),
    ("google", "leveldb"),
    ("influxdata", "influxdb2-sample-data"),
    ("nvidia", "apex"),
    ("nvidia", "dali"),
    ("tensorflow", "tpu"),
    ("marbl", "canu"),
    ("tkarras", "progressive_growing_of_gans"),
    ("cloudflare", "jpegtran"),
    ("mpeggroup", "mpeg-pcc-tmc13"),
    ("mpeggroup", "mpeg-pcc-tmc2"),
    ("google", "snappy"),
    # Already vendored under different canonical remote for the benchmark
    ("szcompressor", "sz"),
}

GH_REF = re.compile(
    r"(?:https?://)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:/[\w ./%-]*)?",
    re.IGNORECASE,
)
CODE_CTX = re.compile(
    r"(artifact|software|source code|code repository|github\.com|implementation|"
    r"open[- ]source|available at|project page|repository)",
    re.IGNORECASE,
)


def load_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise SystemExit("Install pypdf:  python3 -m pip install pypdf") from e
    return PdfReader


def extract_text_pdf(path: Path, max_pages: int) -> str:
    PdfReader = load_pypdf()
    try:
        r = PdfReader(str(path))
    except Exception as e:
        return f"\n<<pdf_read_error:{e}>>\n"
    parts: list[str] = []
    n = min(len(r.pages), max_pages)
    for i in range(n):
        try:
            t = r.pages[i].extract_text() or ""
        except Exception:
            t = ""
        parts.append(t)
    return "\n".join(parts)


def normalize_root(owner: str, repo: str) -> tuple[str, str] | None:
    owner, repo = owner.strip().strip("."), repo.strip().strip(".")
    if not owner or not repo:
        return None
    low = owner.lower()
    if low in {"topics", "collections", "enterprise", "sponsors", "settings"}:
        return None
    if repo.lower() in {".github", owner.lower()}:
        return None
    return owner, repo


def find_github_urls(text: str) -> list[tuple[str, str, bool]]:
    """Return list of (owner, repo, near_code_keyword)."""
    out: list[tuple[str, str, bool]] = []
    low = text.lower()
    for m in GH_REF.finditer(text):
        raw = normalize_root(m.group(1), m.group(2))
        if not raw:
            continue
        if (raw[0].lower(), raw[1].lower()) in SKIP_REPO_ROOTS_LC:
            continue
        start = max(0, m.start() - 120)
        ctx = text[start : m.end() + 120]
        flagged = CODE_CTX.search(ctx) is not None
        out.append((*raw, flagged))
    return out


def parse_init_compression_clone_urls() -> set[tuple[str, str]]:
    """Lowercase (owner, repo) for each GitHub URL in ``init_compression_submodules.sh``."""
    sh = ROOT / "scripts" / "init_compression_submodules.sh"
    if not sh.is_file():
        return set()
    raw = sh.read_text(encoding="utf-8", errors="replace")
    out: set[tuple[str, str]] = set()
    for m in re.finditer(r"https://github\.com/([^/]+)/([^/.]+)\.git", raw, re.I):
        out.add((m.group(1).lower(), m.group(2).lower()))
    return out


def parse_init_compression_paths() -> set[str]:
    """Basenames of clone target paths in ``init_compression_submodules.sh``."""
    sh = ROOT / "scripts" / "init_compression_submodules.sh"
    if not sh.is_file():
        return set()
    out: set[str] = set()
    for line in sh.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r'"https://github\.com/[^"]+\|([^"]+)"', line)
        if m:
            out.add(Path(m.group(1).strip()).name)
    return out


def suggest_path(owner: str, repo: str, used: set[str]) -> str:
    base = repo.replace(".", "-")
    if base in used:
        base = f"{owner}-{repo}".replace(".", "-")
    while base in used:
        base += "_1"
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS)
    ap.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Extract text from the first N pages (implementation links are usually up front).",
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--suggest-clones",
        action="store_true",
        help="Print suggested git clone lines and PAIR entries for init_compression_submodules.sh.",
    )
    ap.add_argument(
        "--add-submodules",
        action="store_true",
        help=argparse.SUPPRESS,  # deprecated alias
    )
    ap.add_argument(
        "--include-all-urls",
        action="store_true",
        help="Do not require URL to appear near a code-related keyword (more false positives).",
    )
    args = ap.parse_args()

    existing = parse_init_compression_clone_urls()
    pdfs = sorted(args.papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs under {args.papers_dir}", file=sys.stderr)
        sys.exit(1)

    # owner/repo -> {best_pdf, near_code}
    best: dict[tuple[str, str], tuple[str, bool]] = {}
    rows: list[dict[str, str]] = []

    for pdf in pdfs:
        text = extract_text_pdf(pdf, args.max_pages)
        found = find_github_urls(text)
        seen_this_pdf: set[tuple[str, str]] = set()
        for owner, repo, near in found:
            key = (owner.lower(), repo.lower())
            if key in seen_this_pdf:
                continue
            seen_this_pdf.add(key)
            rows.append(
                {
                    "pdf": pdf.name,
                    "owner": owner,
                    "repo": repo,
                    "near_code_keyword": str(near),
                    "url": f"https://github.com/{owner}/{repo}",
                }
            )
            use = near or args.include_all_urls
            if not use:
                continue
            if key in SKIP_REPO_ROOTS_LC:
                continue
            prev = best.get(key)
            if prev is None or (near and not prev[1]):
                best[key] = (pdf.name, near)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pdf", "owner", "repo", "near_code_keyword", "url"],
        )
        w.writeheader()
        w.writerows(rows)

    to_add: list[tuple[str, str, str]] = []  # owner, repo, from_pdf
    for key, (pdf_name, _) in sorted(best.items(), key=lambda x: x[0]):
        if key in existing:
            continue
        owner, repo = key
        # canonical casing from first row
        for r in rows:
            if (
                r["owner"].lower() == owner
                and r["repo"].lower() == repo
            ):
                to_add.append((r["owner"], r["repo"], pdf_name))
                break

    print(f"PDFs scanned: {len(pdfs)}  unique (owner,repo) with code context: {len(best)}")
    print(f"Already listed in scripts/init_compression_submodules.sh: {len(existing)}")
    print(f"New candidates: {len(to_add)}")
    for o, r, p in to_add:
        print(f"  https://github.com/{o}/{r}.git  (from {p})")

    if args.dry_run or not (args.suggest_clones or args.add_submodules):
        if not (args.suggest_clones or args.add_submodules):
            print("\nPass --suggest-clones to print clone / init-script hints.")
        sys.exit(0)

    used_paths = parse_init_compression_paths()

    for owner, repo, _ in to_add:
        rel = suggest_path(owner, repo, used_paths)
        used_paths.add(rel)
        url = f"https://github.com/{owner}/{repo}.git"
        path = f"external/compression/{rel}"
        pair = f'  "{url}|{path}"'
        print(f"+ git clone --depth 1 {url} {path}")
        print(f"  add to PAIRS in scripts/init_compression_submodules.sh: {pair}", flush=True)


if __name__ == "__main__":
    if not (ROOT / ".git").exists():
        print("Run from clone root.", file=sys.stderr)
        sys.exit(1)
    main()
