#!/usr/bin/env python3
"""
Filter ``benchmark/reference/papers/_download_manifest.csv``:

- Keep rows whose **Title** matches any of (case-insensitive):
  - ``floating`` / ``float `` / ``float-`` / ``float,`` / ``float.`` / ``float)`` (covers "floating-point", "float ")
  - ``integer`` / ``integers``
  - ``time series`` / ``time-series`` / ``timeseries``

- Drop rows whose Title suggests an algorithm **already integrated** in
  ``frontend/src/utils/benchmarkAlgorithmCatalog.js`` (heuristic: tokens + regex).

Outputs:

- ``manifest_float_int_ts_not_integrated.csv`` — full filtered list + ``pdf_tier`` + ``url_has_pdf_hint``
- ``manifest_float_int_ts_pdf_ready.csv`` — subset where manifest reports ``ok`` / ``skip_exists`` (PDF present or skipped as existing)

Re-run after changing the manifest or catalog:

    python3 scripts/filter_papers_manifest_int_float_ts.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark/reference/papers/_download_manifest.csv"
OUT = ROOT / "benchmark/reference/papers/manifest_float_int_ts_not_integrated.csv"
OUT_PDF_READY = ROOT / "benchmark/reference/papers/manifest_float_int_ts_pdf_ready.csv"


def _pdf_tier(status: str) -> str:
    s = (status or "").strip()
    if s in ("ok", "skip_exists"):
        return "ready"
    if s == "no_pdf_url" or not s:
        return "no_url"
    if s.startswith("fail:"):
        return "blocked"
    return "other"


def _url_pdf_hint(url: str) -> str:
    u = (url or "").strip().lower()
    if "arxiv.org/pdf" in u or u.endswith(".pdf"):
        return "1"
    return "0"

# Sync with benchmarkAlgorithmCatalog.js (JAVA_INT + JAVA_FLOAT + PYTHON_FLOAT_EXTRA + fpzip insert).
# Heuristic: exclude titles that name these systems (word-boundary where needed).
_INTEGRATED_SUBSTRINGS = [
    "buff_float",
    "buff ",
    "brotli",
    "gzip",
    "lz4",
    "lzma",
    "snappy",
    "zstd",
    "fpzip",
    "blosc",
    "blosc_blosclz",
    "blosc2",
    "fastpfor",
    "chimp_float",
    "chimp_int",
    "chimp:",
    "chimp ",
    "gorilla",
    "sprintz",
    "ts_2diff",
    "ts-2diff",
    "col_dictionary",
    "col_bss",
    "DELTA_BINARY_PACKED",
    "delta_binary_packed",
    "fastpfor_float",
    "simple8b_float",
    "fastpfor_int",
    "simple8b_int",
    "leco_float",
    "alp_float",
    "elf_on_chimp",
    "elf_plus",
    "camel_float",
    "simpiece",
    "sim-piece",
    "sz_float",
    "sz3_float",
    "sz3:",
    "sz3 ",
    "serf_float",
    "ext_zfp",
    "bitshuffle",
    "learn-to-compress",
    "learn to compress",
    "mix-piece",
    "mixpiece",
    "iotdb",
    "tsfile",
    "fcbench",  # benchmark survey paper tied to integrated ecosystem
]

_INTEGRATED_RES = [
    re.compile(r"\bchimp\b", re.I),
    re.compile(r"\bgorilla\b", re.I),
    re.compile(r"\bsprintz\b", re.I),
    re.compile(r"\bcamel\b", re.I),
    re.compile(r"\bleco\b", re.I),
    re.compile(r"\balp\b", re.I),
    re.compile(r"\belf\b", re.I),
    re.compile(r"\bserf\b", re.I),
    re.compile(r"\bbuff\b", re.I),
    re.compile(r"\bsz3\b", re.I),
    re.compile(r"\bsz2\b", re.I),
    re.compile(r"\bfpzip\b", re.I),
    re.compile(r"\bzfp\b", re.I),
    re.compile(r"\bbrotli\b", re.I),
    re.compile(r"\bgzip\b", re.I),
    re.compile(r"\blz4\b", re.I),
    re.compile(r"\blzma\b", re.I),
    re.compile(r"\bzstd\b", re.I),
    re.compile(r"\bsnappy\b", re.I),
    re.compile(r"\bndzip\b", re.I),
    re.compile(r"\bnvcomp\b", re.I),
    re.compile(r"\bfastlanes\b", re.I),
    re.compile(r"\bfastpfor\b", re.I),
    re.compile(r"\bsimple8b\b", re.I),
    re.compile(r"\bfsst\b", re.I),
    re.compile(r"\bbtrblocks\b", re.I),
    # "SZ" as compressor name (not substring of other words)
    re.compile(r"\bSZ\b"),
    re.compile(r"\(SZ\)"),
]


def _title_matches_topic(title: str) -> tuple[bool, str]:
    t = title or ""
    tl = t.lower()
    hits: list[str] = []
    if "floating" in tl or re.search(r"\bfloat[\s\-.,;:\)]", tl):
        hits.append("floating")
    if "integer" in tl or "integers" in tl:
        hits.append("integer")
    if "time series" in tl or "time-series" in tl or "timeseries" in tl:
        hits.append("time_series")
    if not hits:
        return False, ""
    return True, "|".join(hits)


def _title_suggests_integrated(title: str) -> bool:
    if not title:
        return False
    tl = title.lower()
    for s in _INTEGRATED_SUBSTRINGS:
        if s.lower() in tl:
            return True
    for rx in _INTEGRATED_RES:
        if rx.search(title):
            return True
    return False


def main() -> None:
    rows_out: list[dict[str, str]] = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = (row.get("Title") or "").strip()
            ok, matched = _title_matches_topic(title)
            if not ok:
                continue
            if _title_suggests_integrated(title):
                continue
            st = row.get("status", "")
            tier = _pdf_tier(st)
            url = row.get("url", "")
            rows_out.append(
                {
                    "excel_row": row.get("excel_row", ""),
                    "File": row.get("File", ""),
                    "Title": title,
                    "matched_terms": matched,
                    "pdf_tier": tier,
                    "url_has_pdf_hint": _url_pdf_hint(url),
                    "url": url,
                    "dest": row.get("dest", ""),
                    "status": st,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "excel_row",
        "File",
        "Title",
        "matched_terms",
        "pdf_tier",
        "url_has_pdf_hint",
        "url",
        "dest",
        "status",
    ]
    pdf_ready = [r for r in rows_out if r["pdf_tier"] == "ready"]

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {OUT.relative_to(ROOT)}")

    with OUT_PDF_READY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(pdf_ready)
    print(f"Wrote {len(pdf_ready)} rows to {OUT_PDF_READY.relative_to(ROOT)} (pdf_tier=ready)")


if __name__ == "__main__":
    main()
