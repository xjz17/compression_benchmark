"""Load method metadata and encoding-operator pipeline hints for the Methods page."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

from .benchmark_loss import is_known_lossy_algorithm
from .lzbench_codec_catalog import LZBENCH_CODEC_ENTRIES, entry_algorithm_keys
from .active_method_policy import is_active_algorithm_key

_METHOD_METADATA_FILENAME = 'method_metadata.json'
_METHOD_PIPELINES_FILENAME = 'method_operator_pipelines.json'


def _methods_reference_dir() -> Path:
    return Path(settings.BASE_DIR).resolve().parent / 'benchmark' / 'reference' / 'methods'


def method_metadata_path() -> Path:
    return _methods_reference_dir() / _METHOD_METADATA_FILENAME


def method_operator_pipelines_path() -> Path:
    return _methods_reference_dir() / _METHOD_PIPELINES_FILENAME


def _normalize_method_match_token(text: str) -> str:
    import re

    return re.sub(r'[^a-z0-9]+', '', str(text or '').strip().lower())


def _build_lzbench_method_mappings() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    explicit: dict[str, tuple[str, ...]] = {}
    prefixes: dict[str, str] = {}
    for entry in LZBENCH_CODEC_ENTRIES:
        keys = tuple(entry_algorithm_keys(entry))
        if entry.presets is not None:
            explicit[entry.display_name] = keys
        else:
            prefixes[entry.display_name] = f'{entry.codec_id.replace("-", "_")}_'
    return explicit, prefixes


_LZBENCH_METHOD_EXPLICIT, _LZBENCH_METHOD_PREFIXES = _build_lzbench_method_mappings()

_LZO1_FAMILY_ALGORITHM_KEYS = (
    'lzo1', 'lzo1a', 'lzo1b', 'lzo1c', 'lzo1f',
    'lzo1x_1', 'lzo1x_112', 'lzo1x_223', 'lzo1x_334', 'lzo1x_445',
    'lzo1x_555', 'lzo1x_666', 'lzo1x_777', 'lzo1x_888', 'lzo1x_999',
    'lzo1y', 'lzo1z',
)
_SLZ_FAMILY_ALGORITHM_KEYS = tuple(
    f'{codec}_{level}'
    for codec in ('slz_deflate', 'slz_gzip', 'slz_zlib')
    for level in range(1, 4)
)
_UCL_NRV_FAMILY_ALGORITHM_KEYS = (
    *(f'ucl_nrv2b_{level}' for level in range(1, 10)),
    'ucl_nrv2d',
    'ucl_nrv2e',
)
_LZSSE_FAMILY_ALGORITHM_KEYS = (
    'lzsse2',
    'lzsse4',
    'lzsse4fast',
    'lzsse8',
    'lzsse8fast',
)

_METHOD_NAME_ALGORITHM_KEYS: dict[str, tuple[str, ...]] = {
    'gzip': ('gz',),
    'buff': ('buff_kernel_native',),
    'libdeflate': tuple(f'libdeflate_{level}' for level in range(1, 13)),
    'zlib': tuple(f'zlib_{level}' for level in range(1, 10)),
    'zlib-ng': tuple(f'zlib_ng_{level}' for level in range(1, 10)),
    'Grouping': ('grouping',),
    'Chimp128': ('CHIMP128',),
    'elf': ('elf_on_chimp',),
    'fpc': ('fpc',),
    'pfpc': ('pfpc',),
    'MOST': ('most_minimal',),
    'LZ4': ('lz4-framed',),
    'snappy': ('snappy-framed',),
    'Stream VByte': ('streamvbyte_delta1',),
    'FSE': ('fse',),
    'Huff0': ('huff0',),
    'BOS': ('ts_2diff_bos',),
    'REGER': ('reger',),
    'DELTA_BINARY_PACKED': ('DELTA_BINARY_PACKED',),
    'FPZIP': ('fpzip',),
    'Blosc': ('blosc_blosclz',),
    'Blosc2': ('blosc2_blosclz',),
    'Blosc BloscLZ': ('blosc_blosclz',),
    'Blosc2 BloscLZ': ('blosc2_blosclz',),
    'BLOSC': ('blosc_blosclz',),
    'BLOSC2': ('blosc2_blosclz',),
    'ZFP': ('ext_zfp_zfpy',),
    'FPCompress DPratio': ('fpcompress_dpratio',),
    'FPCompress DPspeed': ('fpcompress_dpspeed',),
    'LZO1': _LZO1_FAMILY_ALGORITHM_KEYS,
    'LZSSE': _LZSSE_FAMILY_ALGORITHM_KEYS,
    'SLZ': _SLZ_FAMILY_ALGORITHM_KEYS,
    'UCL NRV': _UCL_NRV_FAMILY_ALGORITHM_KEYS,
    **_LZBENCH_METHOD_EXPLICIT,
}

_METHOD_NAME_KEY_PREFIXES: dict[str, str] = {
    'zstd': 'zstd_level_',
    'XZ': 'xz_level_',
    'SZ': 'sz_',
    'SZ3': 'sz3_',
    'Machete': 'machete_rel_',
    'SimPiece': 'simpiece_',
    'spdp': 'spdp_level_',
    **_LZBENCH_METHOD_PREFIXES,
}


@lru_cache(maxsize=1)
def default_method_catalog_algorithm_keys() -> tuple[str, ...]:
    """Return report keys known to the Methods page, including core and lzbench rows."""
    from .lzbench_codec_catalog import all_catalog_algorithm_keys
    from .native_lz_family import NATIVE_LZ_FAMILY_ALGORITHMS

    core_keys: list[str] = [
        'brotli',
        'gz',
        'lz4-framed',
        'snappy-framed',
        'ans',
        'fse',
        'huffman',
        'huff0',
        'arithmetic',
        'CHIMP',
        'CHIMP128',
        'Pongo',
        'GORILLA',
        'RLE',
        'RLBE',
        'SPRINTZ',
        'TS_2DIFF',
        'ts_2diff_bos',
        'reger',
        'col_dictionary',
        'grouping',
        'buff_kernel_native',
        'col_bss_lz4',
        'DELTA_BINARY_PACKED',
        'streamvbyte_delta1',
        'fastpfor',
        'fpc',
        'pfpc',
        'simple8b',
        'spdp',
        *(f'spdp_level_{i}' for i in range(10)),
        'most_minimal',
        'corad_tricklet',
        'leco',
        'alp',
        'elf_on_chimp',
        'elf_plus',
        'camel',
        'fpzip',
        'blosc_blosclz',
        'blosc2_blosclz',
        'ext_zfp_zfpy',
        'ext_zfp_tol_1e-1',
        'ext_zfp_tol_1e-2',
        'ext_zfp_tol_1e-3',
        'fpcompress_dpratio',
        'fpcompress_dpspeed',
        *(f'xz_level_{i}' for i in range(10)),
        *(f'zstd_level_{i}' for i in range(1, 10)),
        *NATIVE_LZ_FAMILY_ALGORITHMS,
    ]
    return tuple(
        key
        for key in dict.fromkeys([*core_keys, *all_catalog_algorithm_keys()])
        if is_active_algorithm_key(key)
    )


def algorithm_keys_for_method_name(method_name: str, *, catalog: tuple[str, ...] | None = None) -> list[str]:
    name = str(method_name or '').strip()
    if not name:
        return []
    if catalog is None:
        catalog = default_method_catalog_algorithm_keys()
    catalog_set = set(catalog)

    explicit = _METHOD_NAME_ALGORITHM_KEYS.get(name)
    if explicit:
        return [key for key in explicit if key in catalog_set]

    prefix = _METHOD_NAME_KEY_PREFIXES.get(name)
    if prefix:
        if prefix in set(_LZBENCH_METHOD_PREFIXES.values()):
            return [key for key in catalog if key.startswith(prefix) and key[len(prefix):].isdigit()]
        return [key for key in catalog if key.startswith(prefix)]

    norm_method = _normalize_method_match_token(name)
    matched: list[str] = []
    for key in catalog:
        if _normalize_method_match_token(key) == norm_method:
            matched.append(key)
    return matched


PipelineStep = str | dict[str, list[str]]
Pipeline = list[PipelineStep]
TypedPipeline = dict[str, Pipeline]


def _normalize_pipeline_step(raw: object) -> PipelineStep | None:
    if isinstance(raw, str):
        operator_id = raw.strip()
        return operator_id or None
    if isinstance(raw, dict):
        any_of = raw.get('any_of')
        if isinstance(any_of, list):
            operator_ids = [str(item).strip() for item in any_of if str(item).strip()]
            if operator_ids:
                return {'any_of': operator_ids}
    return None


def _normalize_pipeline_steps(raw: object) -> Pipeline:
    if not isinstance(raw, list):
        return []
    steps: Pipeline = []
    for item in raw:
        step = _normalize_pipeline_step(item)
        if step is not None:
            steps.append(step)
    return steps


def _operator_ids_from_step(step: object) -> list[str]:
    if isinstance(step, str):
        operator_id = step.strip()
        return [operator_id] if operator_id else []
    if isinstance(step, dict):
        any_of = step.get('any_of')
        if isinstance(any_of, list):
            return [str(item).strip() for item in any_of if str(item).strip()]
    return []


def _flatten_operator_pipeline(pipeline: object) -> list[str]:
    if isinstance(pipeline, list):
        ops: list[str] = []
        for step in pipeline:
            ops.extend(_operator_ids_from_step(step))
        return ops
    if isinstance(pipeline, dict):
        ops: list[str] = []
        for steps in pipeline.values():
            if isinstance(steps, list):
                ops.extend(_flatten_operator_pipeline(steps))
        return ops
    return []


@lru_cache(maxsize=1)
def _column_operator_ids() -> frozenset[str]:
    from .encoding_operator_catalog import load_encoding_operators

    return frozenset(row['id'] for row in load_encoding_operators())


@lru_cache(maxsize=1)
def _byte_codec_operator_ids() -> frozenset[str]:
    from .encoding_operator_catalog import load_byte_codec_operators

    return frozenset(row['id'] for row in load_byte_codec_operators())


def compression_families_for_method_name(method_name: str) -> frozenset[str]:
    """Return ``value_based`` and/or ``byte_based`` when the method pipeline maps to operators."""
    pipeline = load_method_operator_pipelines().get(str(method_name or '').strip())
    operator_ids = _flatten_operator_pipeline(pipeline)
    families: set[str] = set()
    column_ids = _column_operator_ids()
    byte_ids = _byte_codec_operator_ids()
    if any(op_id in column_ids for op_id in operator_ids):
        families.add('value_based')
    if any(op_id in byte_ids for op_id in operator_ids):
        families.add('byte_based')
    return frozenset(families)


def method_name_for_algorithm_key(
    algorithm_key: str,
    *,
    catalog_keys: tuple[str, ...] | None = None,
) -> str | None:
    key = str(algorithm_key or '').strip()
    if not key:
        return None
    if key.startswith('ext_zfp_tol_'):
        return None
    pipelines = load_method_operator_pipelines()
    for method_name in pipelines:
        if key in algorithm_keys_for_method_name(method_name, catalog=catalog_keys):
            return method_name
    label = None
    try:
        from .home_leaderboard_boxplot_data import boxplot_group_label

        label = boxplot_group_label(key)
    except Exception:
        label = None
    if label:
        norm_label = _normalize_method_match_token(label)
        for method_name in pipelines:
            if _normalize_method_match_token(method_name) == norm_label:
                return method_name
    norm_key = _normalize_method_match_token(key)
    for method_name in pipelines:
        if _normalize_method_match_token(method_name) == norm_key:
            return method_name
    return None


def compression_families_for_algorithm_key(
    algorithm_key: str,
    *,
    catalog_keys: tuple[str, ...] | None = None,
) -> frozenset[str]:
    method_name = method_name_for_algorithm_key(algorithm_key, catalog_keys=catalog_keys)
    if not method_name:
        return frozenset()
    return compression_families_for_method_name(method_name)


_LOSSY_METHOD_NAMES = frozenset({'MOST', 'Machete', 'SimPiece', 'SZ', 'SZ3'})


def compression_mode_for_method_name(method_name: str) -> str:
    name = str(method_name or '').strip()
    if name in _LOSSY_METHOD_NAMES:
        return 'lossy'
    keys = algorithm_keys_for_method_name(name)
    if keys and any(is_known_lossy_algorithm(key) for key in keys):
        return 'lossy'
    return 'lossless'


def operator_steps_from_pipeline(
    pipeline: object,
    *,
    valid_operator_ids: frozenset[str] | None = None,
) -> dict[str, int | str]:
    """Convert an ordered operator list to ``{operator_id: step_number}`` (1-based).

    Repeated operators keep every occurrence as a comma-separated string, e.g.
    ``{"frame_of_reference": "2,4"}``.
    """
    if not isinstance(pipeline, list):
        return {}
    steps: dict[str, int | str] = {}
    for idx, raw in enumerate(pipeline):
        for operator_id in _operator_ids_from_step(raw):
            if valid_operator_ids is not None and operator_id not in valid_operator_ids:
                continue
            if operator_id not in steps:
                steps[operator_id] = idx + 1
            else:
                steps[operator_id] = f'{steps[operator_id]},{idx + 1}'
    return steps


def resolve_method_operator_steps(
    method_name: str,
    *,
    row_pipeline: object = None,
    valid_operator_ids: frozenset[str] | None = None,
    data_type: str = '',
) -> dict[str, int | str]:
    """Resolve pipeline for one method: inline JSON field, then overlay file."""
    dtype = str(data_type or '').strip().lower()

    def _resolve(raw: object) -> dict[str, int | str]:
        if raw is None:
            return {}
        if isinstance(raw, dict) and dtype in ('int', 'float') and dtype in raw:
            return operator_steps_from_pipeline(
                raw.get(dtype),
                valid_operator_ids=valid_operator_ids,
            )
        if isinstance(raw, dict) and all(isinstance(v, int) for v in raw.values()):
            out: dict[str, int] = {}
            for key, value in raw.items():
                operator_id = str(key or '').strip()
                if not operator_id:
                    continue
                if valid_operator_ids is not None and operator_id not in valid_operator_ids:
                    continue
                try:
                    out[operator_id] = int(value)
                except (TypeError, ValueError):
                    continue
            return out
        return operator_steps_from_pipeline(raw, valid_operator_ids=valid_operator_ids)

    if row_pipeline is not None:
        steps = _resolve(row_pipeline)
        if steps:
            return steps

    overlay = load_method_operator_pipelines().get(str(method_name or '').strip())
    return _resolve(overlay)


@lru_cache(maxsize=1)
def load_method_operator_pipelines() -> dict[str, Pipeline | TypedPipeline]:
    path = method_operator_pipelines_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        logger.exception('Failed to parse method operator pipelines: %s', path)
        return {}
    pipelines = payload.get('pipelines')
    if not isinstance(pipelines, dict):
        return {}
    out: dict[str, Pipeline | TypedPipeline] = {}
    for name, pipeline in pipelines.items():
        key = str(name or '').strip()
        if not key:
            continue
        if isinstance(pipeline, list):
            out[key] = _normalize_pipeline_steps(pipeline)
        elif isinstance(pipeline, dict):
            typed: TypedPipeline = {}
            for dtype, steps in pipeline.items():
                dtype_key = str(dtype or '').strip().lower()
                if not dtype_key or not isinstance(steps, list):
                    continue
                typed[dtype_key] = _normalize_pipeline_steps(steps)
            if typed:
                out[key] = typed
    return out
