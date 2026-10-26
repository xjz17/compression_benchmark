"""Major dataset categories (aligned with benchmark/reference/datasets/dataset_metadata.json)."""

from __future__ import annotations

DATASET_MAJOR_CATEGORY_OTHERS = 'Others'

DATASET_MAJOR_CATEGORY_GEO = 'Geoscience & Space'

DATASET_MAJOR_CATEGORY_CLOUD = 'Cloud & Infrastructure'

DATASET_MAJOR_CATEGORY_CHOICES = [
    ('Climate & Weather', 'Climate & Weather'),
    (DATASET_MAJOR_CATEGORY_CLOUD, DATASET_MAJOR_CATEGORY_CLOUD),
    ('Energy & Power', 'Energy & Power'),
    ('Finance & Economics', 'Finance & Economics'),
    (DATASET_MAJOR_CATEGORY_GEO, DATASET_MAJOR_CATEGORY_GEO),
    ('Transport & Mobility', 'Transport & Mobility'),
    (DATASET_MAJOR_CATEGORY_OTHERS, 'Others'),
]

DATASET_MAJOR_CATEGORY_VALUES = frozenset(c[0] for c in DATASET_MAJOR_CATEGORY_CHOICES)

# Merged 2026-06: Geospatial & Ecology + Space & Geoscience -> Geoscience & Space
_LEGACY_MAJOR_CATEGORY_ALIASES = {
    'Geospatial & Ecology': DATASET_MAJOR_CATEGORY_GEO,
    'Space & Geoscience': DATASET_MAJOR_CATEGORY_GEO,
    'Geospatial & Geoscience': DATASET_MAJOR_CATEGORY_GEO,
    'Geospatial & Space': DATASET_MAJOR_CATEGORY_GEO,
}


def normalize_major_category(value: str | None) -> str:
    text = str(value or '').strip()
    text = _LEGACY_MAJOR_CATEGORY_ALIASES.get(text, text)
    if text in DATASET_MAJOR_CATEGORY_VALUES:
        return text
    return DATASET_MAJOR_CATEGORY_OTHERS
