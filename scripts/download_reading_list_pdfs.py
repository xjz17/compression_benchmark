#!/usr/bin/env python3
"""Download open PDFs for reading-list papers into benchmark/reference/papers/.

Uses the Semantic Scholar + OpenAlex payload stored in
``benchmark/reference/.reading_list_api_cache.json`` (same keys as
``fetch_reading_list_code_hints.py``), with optional overrides from
``reading_list_code_audit.csv`` (``s2_open_pdf``, ``s2_arxiv``).

Does not commit PDFs; add ``benchmark/reference/papers/`` to version control only if you intend to vendor them.

Examples::

  python3 scripts/download_reading_list_pdfs.py
  python3 scripts/download_reading_list_pdfs.py --dry-run
  python3 scripts/download_reading_list_pdfs.py --dest /path/to/papers

Environment::

  S2_API_KEY — optional; refresh Semantic Scholar metadata (``--ss-refresh``).

  UNPAYWALL_EMAIL — **strongly recommended** for paywalled DOIs; queries
  `api.unpaywall.org` for a legal open-access PDF when available.

Many ACM / IEEE proceedings are **closed access**; OpenAlex has no ``pdf_url``,
Unpaywall often returns nothing, and ``dl.acm.org/.../pdf`` may respond **403**
without a browser session or institutional subscription. Those rows will stay
``no_pdf_url`` / ``fail:http_403`` in the manifest; fetch those copies through
your library or publisher portal.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = ROOT / "benchmark" / "reference" / "reading list - compress.xlsx"
DEFAULT_CACHE = ROOT / "benchmark" / "reference" / ".reading_list_api_cache.json"
DEFAULT_AUDIT = ROOT / "benchmark" / "reference" / "reading_list_code_audit.csv"
DEFAULT_DEST = ROOT / "benchmark" / "reference" / "papers"
DEFAULT_RESOLVE_CACHE = ROOT / "benchmark" / "reference" / ".reading_list_pdf_resolve_cache.json"

UA = "web_compression-reading-list-pdfs/1.0 (mailto:https://github.com/xiaojinzhao/web_compression)"
UA_BROWSER = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def cache_key(prefix: str, title: str, file_key: str) -> str:
    h = hashlib.sha256(f"{prefix}|{title}|{file_key}".encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{h}"


def read_excel_titles(xlsx: Path) -> list[dict[str, str]]:
    import openpyxl  # type: ignore

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    if "paper" not in wb.sheetnames:
        wb.close()
        raise SystemExit(f"No 'paper' sheet in {xlsx}")
    ws = wb["paper"]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    col = {name: i for i, name in enumerate(header) if name}

    def get(row: tuple[Any, ...], name: str) -> str:
        i = col.get(name)
        if i is None or i >= len(row):
            return ""
        v = row[i]
        return str(v).strip() if v is not None and str(v).strip() != "None" else ""

    out: list[dict[str, str]] = []
    for r_i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        title = get(row, "Title")
        if not title:
            continue
        out.append(
            {
                "excel_row": str(r_i),
                "File": get(row, "File"),
                "Title": title,
                "Name": get(row, "Name"),
            }
        )
    wb.close()
    return out


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_audit_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    m: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t, k = row.get("Title") or "", row.get("File") or ""
            if t:
                m[(t, k)] = row
    return m


def _oarxiv_pdf(arxiv: str) -> str | None:
    if not arxiv or not arxiv.strip():
        return None
    a = arxiv.strip()
    a = re.sub(r"^arxiv:", "", a, flags=re.I)
    a = re.sub(r"^https?://arxiv\.org/abs/", "", a, flags=re.I)
    a = re.sub(r"^https?://arxiv\.org/pdf/", "", a, flags=re.I)
    a = a.removesuffix(".pdf")
    if "/" in a:
        a = a.split("/")[-1]
    # keep version suffix for pdf link; arxiv accepts .../pdf/1234.5678v2.pdf
    if not re.match(r"^[\d.]+(v\d+)?$", a) and not re.match(r"^[\w.-]+/\d{7}$", a):
        # old-style e.g. cs/0112017
        if not re.match(r"^[a-z-]+/\d{7}(v\d+)?$", a, re.I):
            return None
    return f"https://arxiv.org/pdf/{a}.pdf"


def _pdf_from_ss_hit(hit: dict[str, Any] | None) -> str | None:
    if not hit:
        return None
    oa = hit.get("openAccessPdf") or {}
    u = (oa.get("url") or "").strip()
    if u:
        return u
    ext = hit.get("externalIds") or {}
    return _oarxiv_pdf(ext.get("ArXiv") or "")


def _norm_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def titles_compatible(expected_title: str, hit_title: str, min_ratio: float = 0.52) -> bool:
    if not expected_title or not hit_title:
        return False
    a, b = _norm_title(expected_title), _norm_title(hit_title)
    if len(a) < 6 or len(b) < 6:
        return a == b
    return difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio


def _pdf_from_oa_hit(hit: dict[str, Any] | None, *, expect_title: str) -> str | None:
    if not hit:
        return None
    ht = (hit.get("display_name") or hit.get("title") or "").strip()
    if ht and not titles_compatible(expect_title, ht):
        return None
    loc = hit.get("primary_location") or {}
    u = (loc.get("pdf_url") or "").strip()
    if u and _suspicious_pdf_host(u, expect_title):
        return None
    if u:
        return u
    oa = hit.get("open_access") or {}
    ou = (oa.get("oa_url") or "").strip()
    if ou.lower().endswith(".pdf"):
        if not _suspicious_pdf_host(ou, expect_title):
            return ou
    return None


def _suspicious_pdf_host(url: str, expect_title: str) -> bool:
    low = url.lower()
    if "cran.r-project.org" in low:
        et = expect_title.lower()
        return "introduction to r" not in et and "r intro" not in et and "manuals" not in et
    return False


def openalex_search_first(
    title: str,
    mailto: str,
    *,
    sleep_s: float,
) -> tuple[dict[str, Any] | None, str | None]:
    if not title.strip():
        return None, "empty_title"
    params = urllib.parse.urlencode({"search": title.strip(), "per_page": 1})
    url = f"https://api.openalex.org/works?{params}"
    headers = {"User-Agent": f"{UA_BROWSER} ({mailto})"}
    time.sleep(sleep_s)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return None, f"openalex_http_{e.code}"
    except urllib.error.URLError as e:
        return None, f"openalex_url_{e!s}"[:120]
    results = data.get("results") or []
    if not results:
        return None, "openalex_empty"
    hit = results[0]
    rel = float(hit.get("relevance_score") or 0.0)
    disp = (hit.get("display_name") or hit.get("title") or "").strip()
    ok_title = titles_compatible(title, disp) if disp else False
    if not ok_title and rel < 85.0:
        return None, f"openalex_low_confidence score={rel:.1f}"
    return hit, None


def unpaywall_pdf_url(doi: str, email: str) -> str | None:
    """https://unpaywall.org/products/api"""
    if not doi or not email:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = urllib.parse.quote(d, safe="")
    url = f"https://api.unpaywall.org/v2/{d}?email={urllib.parse.quote(email)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None
    loc = data.get("best_oa_location") or {}
    u = (loc.get("url_for_pdf") or "").strip()
    if u:
        return u
    u2 = (loc.get("url") or "").strip()
    return u2 if u2.lower().endswith(".pdf") else None


def load_resolve_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_resolve_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(s: str, max_len: int = 48) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s.-]", "", s, flags=re.UNICODE)
    s = s.replace(" ", "_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_-.")
    return s or "untitled"


def _dest_name(excel_row: str, file_key: str, title: str, paper_id: str) -> str:
    f = _slug(file_key or "ref", 24)
    t = _slug(title, 40)
    key = paper_id[:12] if paper_id else ""
    base = f"r{excel_row}_{f}_{t}"
    if key:
        base += f"_{key}"
    return base + ".pdf"


def fetch_ss_paper(paper_id: str, api_key: str | None) -> dict[str, Any] | None:
    fields = "openAccessPdf,externalIds,title"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    headers = {"User-Agent": UA}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None


def download_url(url: str, dest: Path, sleep_after: float) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    hdrs: dict[str, str] = {
        "User-Agent": UA_BROWSER,
        "Accept": "application/pdf,*/*",
    }
    if "dl.acm.org" in url.lower():
        hdrs["Referer"] = url
        m = re.search(r"10\.\d+/[^\s?#]+", url)
        if m:
            hdrs["Referer"] = f"https://dl.acm.org/doi/{m.group(0)}"
    req = urllib.request.Request(
        url,
        headers=hdrs,
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            body = resp.read()
    except urllib.error.HTTPError as e:
        time.sleep(sleep_after)
        return False, f"http_{e.code}"
    except urllib.error.URLError as e:
        time.sleep(sleep_after)
        return False, f"url_{e!s}"[:200]
    time.sleep(sleep_after)

    if len(body) < 500:
        return False, f"too_small_{len(body)}"
    if body[:4] != b"%PDF" and "pdf" not in ctype:
        return False, f"not_pdf_ctype={ctype!r}"
    dest.write_bytes(body)
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--sleep", type=float, default=1.5, help="Seconds after each download.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--ss-refresh",
        action="store_true",
        help="If no URL found, call S2 paper API when s2_paper_id is present (needs quota).",
    )
    ap.add_argument(
        "--resolve-openalex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When cache/audit lack a PDF URL, query OpenAlex by title (polite rate limit).",
    )
    ap.add_argument(
        "--resolve-cache",
        type=Path,
        default=DEFAULT_RESOLVE_CACHE,
        help="JSON cache for live OpenAlex lookups (speeds re-runs).",
    )
    ap.add_argument("--mailto-openalex", default="mailto:https://github.com/xiaojinzhao/web_compression")
    ap.add_argument("--openalex-sleep", type=float, default=0.4)
    ap.add_argument("--max-papers", type=int, default=0, help="Process only first N papers (0 = all).")
    args = ap.parse_args()
    api_key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    uw_email = os.environ.get("UNPAYWALL_EMAIL", "").strip()

    papers = read_excel_titles(args.xlsx)
    if args.max_papers:
        papers = papers[: args.max_papers]
    cache = load_cache(args.cache)
    audit = load_audit_csv(args.audit_csv)
    resolve_cache = load_resolve_cache(args.resolve_cache)

    manifest_path = args.dest / "_download_manifest.csv"
    manifest_rows: list[dict[str, str]] = []

    ok = fail = skip = 0
    for rec in papers:
        title = rec["Title"]
        fk = rec["File"]
        er = rec["excel_row"]
        key_ss = cache_key("ss", title, fk)
        key_oa = cache_key("oa", title, fk)

        hit_ss = (cache.get(key_ss) or {}).get("hit")
        hit_oa = (cache.get(key_oa) or {}).get("hit")

        au = audit.get((title, fk), {})
        pid = (au.get("s2_paper_id") or "").strip()
        pdf_url: str | None = None
        src = ""
        live_hit: dict[str, Any] | None = None

        cand = (au.get("s2_open_pdf") or "").strip()
        if cand:
            pdf_url, src = cand, "audit_csv_s2_open_pdf"

        if not pdf_url:
            pdf_url = _pdf_from_ss_hit(hit_ss if isinstance(hit_ss, dict) else None)
            if pdf_url:
                src = "cache_ss_open_access"
        if not pdf_url:
            ax = (au.get("s2_arxiv") or "").strip()
            pdf_url = _oarxiv_pdf(ax)
            if pdf_url:
                src = "audit_csv_arxiv"
        if not pdf_url and isinstance(hit_ss, dict):
            pdf_url = _oarxiv_pdf((hit_ss.get("externalIds") or {}).get("ArXiv") or "")
            if pdf_url:
                src = "cache_ss_arxiv"
        if not pdf_url:
            pdf_url = _pdf_from_oa_hit(
                hit_oa if isinstance(hit_oa, dict) else None,
                expect_title=title,
            )
            if pdf_url:
                src = "cache_openalex_pdf_url"

        rkey = cache_key("resolve", title, fk)
        if not pdf_url and args.resolve_openalex:
            if rkey in resolve_cache and resolve_cache[rkey].get("hit"):
                live_hit = resolve_cache[rkey]["hit"]
            else:
                live_hit, oa_err = openalex_search_first(
                    title,
                    args.mailto_openalex,
                    sleep_s=args.openalex_sleep,
                )
                resolve_cache[rkey] = {"hit": live_hit, "error": oa_err, "ts": time.time()}
                save_resolve_cache(args.resolve_cache, resolve_cache)
            if live_hit:
                pdf_url = _pdf_from_oa_hit(live_hit, expect_title=title)
                if pdf_url:
                    src = "live_openalex"

        if not pdf_url and uw_email:
            doi_try = (au.get("openalex_doi") or au.get("s2_doi") or "").strip()
            if not doi_try and live_hit:
                ids = live_hit.get("ids") or {}
                doi_try = (ids.get("doi") or "").strip()
                doi_try = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi_try, flags=re.I)
            if doi_try:
                pdf_url = unpaywall_pdf_url(doi_try, uw_email)
                time.sleep(0.25)
                if pdf_url and _suspicious_pdf_host(pdf_url, title):
                    pdf_url = None
                elif pdf_url:
                    src = "unpaywall"

        if not pdf_url and args.ss_refresh and pid:
            fresh = fetch_ss_paper(pid, api_key)
            time.sleep(2.0 if not api_key else 1.0)
            if fresh:
                pdf_url = _pdf_from_ss_hit(fresh)
                if pdf_url:
                    src = "s2_api_refresh"

        out_name = _dest_name(er, fk, title, pid or (hit_ss or {}).get("paperId", "") if isinstance(hit_ss, dict) else "")
        dest = args.dest / out_name

        rec_out = {
            "excel_row": er,
            "File": fk,
            "Title": title[:120],
            "source": src,
            "url": (pdf_url or "")[:500],
            "dest": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
            "status": "",
        }

        if not pdf_url:
            rec_out["status"] = "no_pdf_url"
            manifest_rows.append(rec_out)
            fail += 1
            print(f"[fail:no_url] row {er} {title[:60]}…", flush=True)
            continue

        if dest.is_file() and dest.stat().st_size > 1024:
            rec_out["status"] = "skip_exists"
            manifest_rows.append(rec_out)
            skip += 1
            print(f"[skip] {dest.name}", flush=True)
            continue

        if args.dry_run:
            rec_out["status"] = "dry_run"
            manifest_rows.append(rec_out)
            print(f"[dry-run] {src} -> {dest.name}\n  {pdf_url[:100]}…", flush=True)
            continue

        good, err = download_url(pdf_url, dest, args.sleep)
        if good:
            rec_out["status"] = "ok"
            ok += 1
            print(f"[ok] {dest.name}", flush=True)
        else:
            rec_out["status"] = f"fail:{err}"
            fail += 1
            print(f"[fail:{err}] row {er} {title[:50]}…", flush=True)
        manifest_rows.append(rec_out)

    args.dest.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["excel_row", "File", "Title", "source", "url", "dest", "status"],
        )
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Done. ok={ok} skip={skip} fail/no_url={fail} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    if not (ROOT / "benchmark" / "reference").is_dir():
        print("Run from repo root.", file=sys.stderr)
        sys.exit(1)
    main()
