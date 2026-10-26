"""Read/write precomputed Home leaderboard boxplot data under ``backend/dataset_result/``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings

from .dataset_catalog_result_store import RESULT_DIRNAME, manifest_path

BOXPLOT_RESULT_VERSION = 4
HOME_LEADERBOARD_BOXPLOTS_FILENAME = 'home_leaderboard_boxplots.json'

METRIC_KEYS = (
    'compression_rate',
    'compression_time',
    'decompression_time',
    'balanced_score',
)


def _is_hidden_home_chart_label(label: object) -> bool:
    from .home_leaderboard_boxplot_data import is_home_leaderboard_label_enabled

    text = str(label or '').strip()
    return not text or not is_home_leaderboard_label_enabled(text)


def _is_hidden_home_chart_algorithm_key(algorithm_key: object) -> bool:
    from .report_filters import is_hidden_report_algorithm

    key = str(algorithm_key or '').strip()
    return bool(key) and is_hidden_report_algorithm(key)


def _is_removed_bos_subcolumn_level_row(algorithm: object, family: object) -> bool:
    text = str(algorithm or '').strip()
    fam = str(family or '').strip()
    if fam == 'OSS':
        return True
    if fam not in {'BOS', 'Sub-column'}:
        return False
    compact = text.lower().replace('-', ' ').replace('_', ' ')
    return 'level' in compact or compact.startswith('bos l') or compact.startswith('sub column l')


def _filtered_rank_groups(rank_groups: object) -> list[dict[str, Any]]:
    if not isinstance(rank_groups, list):
        return []
    out: list[dict[str, Any]] = []
    for group in rank_groups:
        if not isinstance(group, dict):
            continue
        algorithms = [
            str(name)
            for name in group.get('algorithms') or []
            if not _is_hidden_home_chart_label(name)
        ]
        if len(algorithms) < 2:
            continue
        next_group = dict(group)
        next_group['algorithms'] = algorithms
        out.append(next_group)
    return out


def boxplot_result_path() -> Path:
    return Path(settings.BASE_DIR) / RESULT_DIRNAME / HOME_LEADERBOARD_BOXPLOTS_FILENAME


def _is_number_list(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return False
    return True


def _validate_pareto_chart(pareto: object) -> list[str]:
    errors: list[str] = []
    if pareto is None:
        errors.append('missing pareto_chart')
        return errors
    if not isinstance(pareto, dict):
        return ['pareto_chart must be a dict']
    if not isinstance(pareto.get('title'), str):
        errors.append('pareto_chart.title must be a string')
    if not isinstance(pareto.get('x_label'), str):
        errors.append('pareto_chart.x_label must be a string')
    for field in ('x_min', 'x_max'):
        if field in pareto and pareto.get(field) is not None:
            val = pareto.get(field)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f'pareto_chart.{field} must be a number or null')
    if not isinstance(pareto.get('y_label'), str):
        errors.append('pareto_chart.y_label must be a string')
    points = pareto.get('points')
    if not isinstance(points, list):
        errors.append('pareto_chart.points must be a list')
        return errors
    for idx, point in enumerate(points):
        prefix = f'pareto_chart.points[{idx}]'
        if not isinstance(point, dict):
            errors.append(f'{prefix} must be a dict')
            continue
        if not isinstance(point.get('algorithm'), str):
            errors.append(f'{prefix}.algorithm must be a string')
        for field in ('rate', 'compression_time'):
            val = point.get(field)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f'{prefix}.{field} must be a number')
        if 'pareto' in point and not isinstance(point.get('pareto'), bool):
            errors.append(f'{prefix}.pareto must be a boolean')
        if 'family' in point and not isinstance(point.get('family'), str):
            errors.append(f'{prefix}.family must be a string')
        if 'level_order' in point and point.get('level_order') is not None:
            val = point.get('level_order')
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f'{prefix}.level_order must be a number or null')
    frontier = pareto.get('frontier')
    if frontier is not None:
        if not isinstance(frontier, list) or not all(_is_number_list(row) for row in frontier):
            errors.append('pareto_chart.frontier must be a list of [x, y] pairs')
    level_series = pareto.get('level_series')
    if level_series is not None:
        if not isinstance(level_series, list):
            errors.append('pareto_chart.level_series must be a list')
        else:
            for idx, row in enumerate(level_series):
                prefix = f'pareto_chart.level_series[{idx}]'
                if not isinstance(row, dict):
                    errors.append(f'{prefix} must be a dict')
                    continue
                if not isinstance(row.get('family'), str):
                    errors.append(f'{prefix}.family must be a string')
                pts = row.get('points')
                if not isinstance(pts, list) or len(pts) < 2:
                    errors.append(f'{prefix}.points must be a list with at least 2 entries')
                    continue
                for pidx, pt in enumerate(pts):
                    pt_prefix = f'{prefix}.points[{pidx}]'
                    if not isinstance(pt, dict):
                        errors.append(f'{pt_prefix} must be a dict')
                        continue
                    if not isinstance(pt.get('algorithm'), str):
                        errors.append(f'{pt_prefix}.algorithm must be a string')
                    for field in ('rate', 'compression_time', 'level_order'):
                        val = pt.get(field)
                        if not isinstance(val, (int, float)) or isinstance(val, bool):
                            errors.append(f'{pt_prefix}.{field} must be a number')
    return errors


def validate_boxplot_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ['payload must be a dict']

    groups = payload.get('groups')
    if not isinstance(groups, dict):
        return ['groups must be a dict']

    for metric_key in METRIC_KEYS:
        group = groups.get(metric_key)
        if group is None:
            errors.append(f'missing groups.{metric_key}')
            continue
        if not isinstance(group, dict):
            errors.append(f'groups.{metric_key} must be a dict')
            continue
        if not isinstance(group.get('title'), str):
            errors.append(f'groups.{metric_key}.title must be a string')
        charts = group.get('charts')
        if not isinstance(charts, dict):
            errors.append(f'groups.{metric_key}.charts must be a dict')
            continue
        for chart_key, chart in charts.items():
            prefix = f'groups.{metric_key}.charts.{chart_key}'
            if not isinstance(chart, dict):
                errors.append(f'{prefix} must be a dict')
                continue
            algorithms = chart.get('algorithms')
            values = chart.get('values')
            if not isinstance(algorithms, list) or not all(isinstance(a, str) for a in algorithms):
                errors.append(f'{prefix}.algorithms must be a string list')
            if not isinstance(values, list) or not all(_is_number_list(row) for row in values):
                errors.append(f'{prefix}.values must be a list of number lists')
            elif isinstance(algorithms, list) and isinstance(values, list):
                if len(algorithms) != len(values):
                    errors.append(
                        f'{prefix}: algorithms and values length mismatch '
                        f'({len(algorithms)} vs {len(values)})',
                    )
            if not isinstance(chart.get('y_label'), str):
                errors.append(f'{prefix}.y_label must be a string')
            if 'log_y' in chart and not isinstance(chart.get('log_y'), bool):
                errors.append(f'{prefix}.log_y must be a boolean')
            if 'chart_type' in chart and not isinstance(chart.get('chart_type'), str):
                errors.append(f'{prefix}.chart_type must be a string')
            for field in ('critical_difference', 'sample_count'):
                if field in chart and chart.get(field) is not None:
                    val = chart.get(field)
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        errors.append(f'{prefix}.{field} must be a number or null')
            if 'rank_groups' in chart and chart.get('rank_groups') is not None:
                if not isinstance(chart.get('rank_groups'), list):
                    errors.append(f'{prefix}.rank_groups must be a list')

    errors.extend(_validate_pareto_chart(payload.get('pareto_chart')))
    return errors


def load_precomputed_home_leaderboard_boxplots() -> dict[str, Any] | None:
    path = boxplot_result_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    meta = data.get('meta')
    groups = data.get('groups')
    if not isinstance(meta, dict) or not isinstance(groups, dict):
        return None
    if int(meta.get('version', 0) or 0) != BOXPLOT_RESULT_VERSION:
        return None
    if validate_boxplot_payload({'groups': groups, 'pareto_chart': data.get('pareto_chart')}):
        return None
    return data


def write_precomputed_home_leaderboard_boxplots(
    *,
    groups: dict[str, Any],
    pareto_chart: dict[str, Any],
    meta_extra: dict[str, Any] | None = None,
) -> Path:
    body = {'groups': groups, 'pareto_chart': pareto_chart}
    errors = validate_boxplot_payload(body)
    if errors:
        raise ValueError(
            'Refusing to write invalid boxplot payload: ' + '; '.join(errors),
        )

    root = Path(settings.BASE_DIR) / RESULT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    meta = {
        'version': BOXPLOT_RESULT_VERSION,
        'generated_at': generated_at,
    }
    if meta_extra:
        meta.update(meta_extra)

    envelope = {
        'meta': meta,
        'groups': groups,
        'pareto_chart': pareto_chart,
    }
    path = boxplot_result_path()
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    _touch_manifest_boxplot_file(generated_at=generated_at)
    return path


def _touch_manifest_boxplot_file(*, generated_at: str) -> None:
    """Add boxplot file entry to ``manifest.json`` when present."""
    path = manifest_path()
    if not path.is_file():
        return
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        return
    files = manifest.get('files')
    if not isinstance(files, dict):
        files = {}
    files['home_leaderboard_boxplots'] = HOME_LEADERBOARD_BOXPLOTS_FILENAME
    manifest['files'] = files
    manifest['boxplot_generated_at'] = generated_at
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def boxplot_groups_for_api(
    payload: dict[str, Any] | None,
    *,
    hide_lossy: bool = False,
) -> dict[str, Any]:
    """Convert stored ``groups`` into the list-shaped API consumed by Home."""
    if not payload or not isinstance(payload.get('groups'), dict):
        return {}

    out: dict[str, Any] = {}
    for metric_key in METRIC_KEYS:
        group = payload['groups'].get(metric_key)
        if not isinstance(group, dict):
            continue
        charts_obj = group.get('charts')
        if not isinstance(charts_obj, dict):
            continue
        charts_list = []
        for chart_key, chart in charts_obj.items():
            if hide_lossy and str(chart_key).startswith('lossy'):
                continue
            if not isinstance(chart, dict):
                continue
            algorithms = list(chart.get('algorithms') or [])
            values = chart.get('values') or []
            filtered_algorithms: list[str] = []
            filtered_values: list[Any] = []
            for index, algorithm in enumerate(algorithms):
                if _is_hidden_home_chart_label(algorithm):
                    continue
                filtered_algorithms.append(str(algorithm))
                if isinstance(values, list) and index < len(values):
                    filtered_values.append(values[index])
            out_chart = {
                'key': str(chart.get('key') or chart_key),
                'title': str(chart.get('title') or chart_key),
                'y_label': str(chart.get('y_label') or ''),
                'log_y': bool(chart.get('log_y')),
                'chart_type': str(chart.get('chart_type') or 'boxplot'),
                'algorithms': filtered_algorithms,
                'values': filtered_values,
            }
            for field in ('critical_difference', 'sample_count', 'rank_groups'):
                if field in chart:
                    out_chart[field] = (
                        _filtered_rank_groups(chart.get(field))
                        if field == 'rank_groups'
                        else chart.get(field)
                    )
            charts_list.append(out_chart)
        out[metric_key] = {
            'title': str(group.get('title') or metric_key),
            'charts': charts_list,
        }
    return out


def pareto_chart_for_api(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload or not isinstance(payload.get('pareto_chart'), dict):
        return {}
    chart = payload['pareto_chart']
    points = []
    hidden_frontier_coords: set[tuple[float, float]] = set()
    for row in chart.get('points') or []:
        if not isinstance(row, dict):
            continue
        try:
            rate = float(row['rate'])
            compression_time = float(row['compression_time'])
        except (KeyError, TypeError, ValueError):
            continue
        algorithm = str(row.get('algorithm') or '')
        family = str(row.get('family') or '') or None
        if (
            _is_hidden_home_chart_algorithm_key(row.get('algorithm_key'))
            or _is_hidden_home_chart_label(algorithm)
            or (family is not None and _is_hidden_home_chart_label(family))
            or _is_removed_bos_subcolumn_level_row(algorithm, family)
        ):
            hidden_frontier_coords.add((rate, compression_time))
            continue
        points.append(
            {
                'algorithm': algorithm,
                'rate': rate,
                'compression_time': compression_time,
                'pareto': bool(row.get('pareto')),
                'family': family,
                'level_order': (
                    float(row['level_order'])
                    if row.get('level_order') is not None
                    else None
                ),
            },
        )
    level_series = []
    for series in chart.get('level_series') or []:
        if not isinstance(series, dict):
            continue
        family = str(series.get('family') or '')
        if _is_hidden_home_chart_label(family) or family in {'BOS', 'Sub-column', 'OSS'}:
            continue
        pts = []
        for pt in series.get('points') or []:
            if not isinstance(pt, dict):
                continue
            try:
                algorithm = str(pt.get('algorithm') or '')
                if _is_hidden_home_chart_label(algorithm):
                    continue
                pts.append(
                    {
                        'algorithm': algorithm,
                        'rate': float(pt['rate']),
                        'compression_time': float(pt['compression_time']),
                        'level_order': float(pt['level_order']),
                    },
                )
            except (KeyError, TypeError, ValueError):
                continue
        if len(pts) >= 2 and family:
            level_series.append({'family': family, 'points': pts})
    frontier = []
    for pair in chart.get('frontier') or []:
        if isinstance(pair, list) and len(pair) >= 2:
            point = (float(pair[0]), float(pair[1]))
            if point in hidden_frontier_coords:
                continue
            frontier.append([point[0], point[1]])
    return {
        'title': str(chart.get('title') or 'Pareto Frontier'),
        'x_label': str(chart.get('x_label') or 'Compression rate'),
        'x_min': chart.get('x_min') if isinstance(chart.get('x_min'), (int, float)) else None,
        'x_max': chart.get('x_max') if isinstance(chart.get('x_max'), (int, float)) else None,
        'y_label': str(chart.get('y_label') or 'Compression time (ns/point)'),
        'frontier_style': str(chart.get('frontier_style') or ''),
        'points': points,
        'level_series': level_series,
        'frontier': frontier,
    }
