"""
Resolve benchmark CSV / TsFile paths and enumerate catalog datasets.

Regen **read** lookup for master CSVs (``resolve_benchmark_dataset_csv``,
``iter_catalog_benchmark_datasets``, etc.), in order:

1. ``COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT`` (when configured)
2. ``COMPRESSION_BENCHMARK_REGEN_LOOKUP_DATASET_CSV_ROOT`` (when configured)
3. ``backend/dataset_csv/``

CSV **export/write** (``dataset_csv_root``, used by ``regenerate_dataset_csv.py``) still
uses only (1) or the in-repo fallback, never the read-only regeneration lookup.

TsFile bundles: the configured external root first, else ``backend/dataset_tsfile/``.

Catalog sidecars (``column_info``, compression reports, charts, etc.) stay under
``backend/dataset/<slug>/``.

Override via ``COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT`` /
``COMPRESSION_BENCHMARK_EXTERNAL_DATASET_TSFILE_ROOT`` (mirrors ``myapp.dataset_bundle``).
"""

from __future__ import annotations

import os
from pathlib import Path

_EXTERNAL_DATASET_CSV_ROOT_ENV = "COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT"
_REGEN_LOOKUP_DATASET_CSV_ROOT_ENV = "COMPRESSION_BENCHMARK_REGEN_LOOKUP_DATASET_CSV_ROOT"
_EXTERNAL_DATASET_TSFILE_ROOT_ENV = "COMPRESSION_BENCHMARK_EXTERNAL_DATASET_TSFILE_ROOT"


def builtin_dataset_csv_root(backend: Path) -> Path:
    return backend / "dataset_csv"


def external_dataset_csv_root() -> Path | None:
    raw = os.environ.get(_EXTERNAL_DATASET_CSV_ROOT_ENV, "").strip()
    if not raw:
        return None
    path = Path(os.path.expanduser(raw))
    return path if path.is_dir() else None


def dataset_csv_root(backend: Path) -> Path:
    """Write target for ``regenerate_dataset_csv.py`` (external when available, else in-repo)."""
    ext = external_dataset_csv_root()
    return ext if ext else builtin_dataset_csv_root(backend)


def regen_lookup_dataset_csv_roots() -> list[Path]:
    """Read-only regen lookup roots (excluding export/write targets)."""
    raw = os.environ.get(_REGEN_LOOKUP_DATASET_CSV_ROOT_ENV, "").strip()
    if not raw:
        return []
    path = Path(os.path.expanduser(raw))
    return [path] if path.is_dir() else []


def iter_dataset_csv_roots(backend: Path) -> list[Path]:
    """Ordered roots for regen read lookup; first existing ``{slug}.csv`` wins."""
    roots: list[Path] = []
    ext = external_dataset_csv_root()
    if ext is not None:
        roots.append(ext)
    for path in regen_lookup_dataset_csv_roots():
        if path not in roots:
            roots.append(path)
    builtin = builtin_dataset_csv_root(backend)
    if builtin not in roots:
        roots.append(builtin)
    return roots


def describe_dataset_csv_lookup(backend: Path) -> str:
    return " or ".join(str(p) for p in iter_dataset_csv_roots(backend))


def resolve_benchmark_dataset_csv(backend: Path, slug: str, fname: str) -> Path | None:
    for root in iter_dataset_csv_roots(backend):
        path = root / fname
        if path.is_file():
            return path
    return None


def iter_master_csv_paths(backend: Path) -> list[tuple[str, Path]]:
    """All ``(slug, csv_path)`` from external + in-repo roots (external wins duplicate slugs)."""
    seen: set[str] = set()
    pairs: list[tuple[str, Path]] = []
    for root in iter_dataset_csv_roots(backend):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.csv")):
            slug = path.stem
            if slug in seen:
                continue
            seen.add(slug)
            pairs.append((slug, path))
    return sorted(pairs, key=lambda item: item[0].lower())


def builtin_dataset_tsfile_root(backend: Path) -> Path:
    return backend / "dataset_tsfile"


def external_dataset_tsfile_root() -> Path | None:
    raw = os.environ.get(_EXTERNAL_DATASET_TSFILE_ROOT_ENV, "").strip()
    if not raw:
        return None
    path = Path(os.path.expanduser(raw))
    return path if path.is_dir() else None


def dataset_tsfile_root(backend: Path) -> Path:
    """Preferred TsFile bundle root (external when available, else in-repo)."""
    ext = external_dataset_tsfile_root()
    return ext if ext else builtin_dataset_tsfile_root(backend)


def iter_dataset_tsfile_roots(backend: Path) -> list[Path]:
    roots: list[Path] = []
    ext = external_dataset_tsfile_root()
    if ext is not None:
        roots.append(ext)
    builtin = builtin_dataset_tsfile_root(backend)
    if builtin not in roots:
        roots.append(builtin)
    return roots


def describe_dataset_tsfile_lookup(backend: Path) -> str:
    return " or ".join(str(p) for p in iter_dataset_tsfile_roots(backend))


def resolve_tsfile_bundle_dir(backend: Path, slug: str) -> Path | None:
    for root in iter_dataset_tsfile_roots(backend):
        path = root / slug
        if path.is_dir():
            return path
    return None


def tsfile_bundle_dir(backend: Path, slug: str) -> Path:
    """Existing bundle dir, else primary write location under preferred TsFile root."""
    resolved = resolve_tsfile_bundle_dir(backend, slug)
    if resolved is not None:
        return resolved
    return dataset_tsfile_root(backend) / slug


def iter_catalog_benchmark_datasets(backend: Path) -> list[tuple[str, Path]]:
    """
    List catalog datasets: ``backend/dataset/<slug>/`` with ``{slug}_column_info.csv``
    and master CSV resolvable via :func:`resolve_benchmark_dataset_csv`.
    Returns ``(slug, csv_path)`` sorted by slug (case-insensitive).
    """
    info_root = backend / "dataset"
    if not info_root.is_dir():
        return []
    pairs: list[tuple[str, Path]] = []
    for child in sorted(info_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        slug = child.name
        col_info = child / f"{slug}_column_info.csv"
        if not col_info.is_file():
            continue
        csv_path = resolve_benchmark_dataset_csv(backend, slug, f"{slug}.csv")
        if csv_path is None:
            continue
        pairs.append((slug, csv_path))
    return pairs
