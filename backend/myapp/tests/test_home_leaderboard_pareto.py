from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from django.test import RequestFactory

from myapp.home_leaderboard_boxplot_data import (
    _build_pareto_level_series,
    best_metrics_from_report,
    build_home_leaderboard_boxplot_groups,
    compute_balanced_score_metrics,
    compute_pareto_chart,
    boxplot_group_label,
    iter_catalog_benchmark_datasets,
    metrics_for_dataset,
    pareto_level_sort_key,
    pareto_point_label,
)
from myapp.paper_table3_leaderboard import (
    paper_table3_display_rows,
    paper_table3_eligible_dataset_keys,
    paper_table3_static_rows,
)
from myapp.lzbench_codec_catalog import default_lzbench_level_selection
from myapp.benchmark_boxplot_result_store import boxplot_groups_for_api, pareto_chart_for_api
from myapp.report_filters import parse_report_filter_params
from myapp.views import BenchmarkBoxplotManifestView, StaticDataView, _staticdata_items_from_metric_rows


class HomeLeaderboardParetoTests(unittest.TestCase):
    def test_boxplot_groups_for_api_hides_all_lossy_charts(self) -> None:
        payload = {
            'groups': {
                'compression_rate': {
                    'title': 'Compression rate',
                    'charts': {
                        'lossless': {'key': 'lossless', 'title': 'Lossless', 'y_label': 'rate', 'algorithms': [], 'values': []},
                        'lossy': {'key': 'lossy', 'title': 'Lossy', 'y_label': 'rate', 'algorithms': [], 'values': []},
                    },
                },
                'compression_time': {'title': 'Compression time', 'charts': {}},
                'decompression_time': {'title': 'Decompression time', 'charts': {}},
                'balanced_score': {
                    'title': 'Balanced score',
                    'charts': {
                        'lossless_score': {'key': 'lossless_score', 'title': 'Lossless score', 'y_label': 'score', 'algorithms': [], 'values': []},
                        'lossless_rank': {'key': 'lossless_rank', 'title': 'Lossless rank', 'y_label': 'rank', 'algorithms': [], 'values': []},
                        'lossy_score': {'key': 'lossy_score', 'title': 'Lossy score', 'y_label': 'score', 'algorithms': [], 'values': []},
                        'lossy_rank': {'key': 'lossy_rank', 'title': 'Lossy rank', 'y_label': 'rank', 'algorithms': [], 'values': []},
                    },
                },
            },
        }

        groups = boxplot_groups_for_api(payload, hide_lossy=True)

        self.assertEqual(
            [chart['key'] for chart in groups['compression_rate']['charts']],
            ['lossless'],
        )
        self.assertEqual(
            [chart['key'] for chart in groups['balanced_score']['charts']],
            ['lossless_score', 'lossless_rank'],
        )

    def test_boxplot_manifest_api_keeps_base_subcolumn_and_filters_cached_level_payloads(self) -> None:
        payload = {
            'groups': {
                'compression_rate': {
                    'title': 'Compression rate',
                    'charts': {
                        'lossless': {
                            'key': 'lossless',
                            'title': 'Lossless',
                            'y_label': 'rate',
                            'algorithms': ['BOS', 'Sub-column', 'ZSTD'],
                            'values': [[0.4], [0.3], [0.5]],
                        },
                    },
                },
                'compression_time': {'title': 'Compression time', 'charts': {}},
                'decompression_time': {'title': 'Decompression time', 'charts': {}},
                'balanced_score': {'title': 'Balanced score', 'charts': {}},
            },
            'pareto_chart': {
                'title': 'Pareto',
                'x_label': 'rate',
                'y_label': 'time',
                'points': [
                    {'algorithm': 'BOS', 'algorithm_key': 'ts_2diff_bos', 'family': 'BOS', 'rate': 0.4, 'compression_time': 80.0},
                    {'algorithm': 'Sub-column', 'algorithm_key': 'subcolumn', 'family': 'Sub-column', 'rate': 0.3, 'compression_time': 20.0},
                ],
                'level_series': [
                    {
                        'family': 'Sub-column',
                        'points': [
                            {'algorithm': 'Sub-column (level 0)', 'rate': 0.2, 'compression_time': 10.0, 'level_order': 0},
                            {'algorithm': 'Sub-column (level 1)', 'rate': 0.3, 'compression_time': 20.0, 'level_order': 1},
                        ],
                    },
                ],
            },
        }

        groups = boxplot_groups_for_api(payload)
        pareto = pareto_chart_for_api(payload)

        chart = groups['compression_rate']['charts'][0]
        self.assertEqual(chart['algorithms'], ['BOS', 'Sub-column', 'ZSTD'])
        self.assertEqual(chart['values'], [[0.4], [0.3], [0.5]])
        self.assertEqual([point['algorithm'] for point in pareto['points']], ['BOS', 'Sub-column'])
        self.assertEqual(pareto['level_series'], [])

    def test_boxplot_manifest_forwards_leaderboard_filter_scope(self) -> None:
        captured = {}
        groups_obj = {
            'compression_rate': {'title': 'Compression rate', 'charts': {}},
            'compression_time': {'title': 'Compression time', 'charts': {}},
            'decompression_time': {'title': 'Decompression time', 'charts': {}},
            'balanced_score': {'title': 'Balanced score', 'charts': {}},
        }
        pareto_chart = {
            'title': 'Pareto Frontier (Compression Rate vs Time)',
            'x_label': 'Compression rate',
            'x_min': 0.0,
            'x_max': 0.9,
            'y_label': 'Compression time (ns/point)',
            'points': [],
            'level_series': [],
            'frontier': [],
        }

        def fake_build(_backend, **kwargs):
            captured.update(kwargs)
            return groups_obj, pareto_chart, 2

        request = RequestFactory().get(
            '/api/benchmark-boxplots-manifest/',
            {
                'data_dimension': 'single_column',
                'dataset': 'demo',
                'column_index': '3',
                'data_type': 'float',
                'implement_language': 'cpp',
                'float_precision': 'fixed',
            },
        )
        with patch(
            'myapp.benchmark_boxplot_result_store.load_precomputed_home_leaderboard_boxplots',
            return_value={'groups': groups_obj, 'pareto_chart': pareto_chart, 'meta': {}},
        ), patch(
            'myapp.home_leaderboard_boxplot_data.build_home_leaderboard_boxplot_groups',
            side_effect=fake_build,
        ):
            response = BenchmarkBoxplotManifestView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured['dataset_keys'], ['demo'])
        self.assertEqual(captured['selected_column_index'], 3)
        self.assertEqual(captured['implement_language'], 'cpp')
        self.assertEqual(captured['data_type'], 'float')
        self.assertEqual(captured['float_precision'], 'fixed')

    def test_paper_table3_eligible_dataset_keys_requires_complete_support(self) -> None:
        backend = Path('/tmp/paper-table3')
        d1 = pd.DataFrame([{'algorithm': 'd1'}])
        d2 = pd.DataFrame([{'algorithm': 'd2'}])
        good_metrics = {f'M{i}': (0.10 + i * 0.01, 1.0, 1.0) for i in range(20)}
        bad_metrics = {f'M{i}': (0.20 + i * 0.01, 1.0, 1.0) for i in range(19)}

        def fake_best_metrics(report, **_kwargs):
            key = str(report.iloc[0]['algorithm'])
            return (good_metrics if key == 'd1' else bad_metrics), {}

        with patch(
            'myapp.paper_table3_leaderboard.read_benchmark_report',
            side_effect=lambda _bundle_dir, dataset_key, **_kwargs: d1 if dataset_key == 'd1' else d2,
        ), patch(
            'myapp.paper_table3_leaderboard.best_metrics_from_report',
            side_effect=fake_best_metrics,
        ):
            keys = paper_table3_eligible_dataset_keys(
                backend,
                data_type='overall',
                float_precision='all',
                implement_language='cpp',
                dataset_keys=['d1', 'd2'],
            )

        self.assertEqual(keys, ['d1'])

    def test_paper_table3_static_rows_drop_excluded_families_and_average_metrics(self) -> None:
        backend = Path('/tmp/paper-table3')
        d1 = pd.DataFrame([{'algorithm': 'd1'}])
        d2 = pd.DataFrame([{'algorithm': 'd2'}])

        def metrics_for_dataset(base_rate: float) -> dict[str, tuple[float, float, float]]:
            metrics = {
                'REGER': (base_rate, 10.0, 20.0),
                'BOS': (base_rate + 0.20, 20.0, 30.0),
                'CORAD': (0.01, 1.0, 1.0),
            }
            for i in range(18):
                metrics[f'F{i}'] = (base_rate + 0.01 * (i + 1), 5.0 + i, 6.0 + i)
            return metrics

        def fake_best_metrics(report, **_kwargs):
            key = str(report.iloc[0]['algorithm'])
            return (metrics_for_dataset(0.10) if key == 'd1' else metrics_for_dataset(0.20)), {}

        with patch(
            'myapp.paper_table3_leaderboard.paper_table3_eligible_dataset_keys',
            return_value=['d1', 'd2'],
        ), patch(
            'myapp.paper_table3_leaderboard.read_benchmark_report',
            side_effect=lambda _bundle_dir, dataset_key, **_kwargs: d1 if dataset_key == 'd1' else d2,
        ), patch(
            'myapp.paper_table3_leaderboard.best_metrics_from_report',
            side_effect=fake_best_metrics,
        ), patch(
            'myapp.paper_table3_leaderboard._collect_column_rank_samples',
            return_value=({'REGER': [1.0, 1.0], 'BOS': [2.0, 2.0]}, 2),
        ):
            rows = paper_table3_static_rows(
                backend,
                data_type='overall',
                float_precision='all',
                implement_language='cpp',
                dataset_keys=['d1', 'd2'],
            )

        by_algorithm = {row['method']: row for row in rows}
        self.assertNotIn('CORAD', by_algorithm)
        self.assertIn('REGER', by_algorithm)
        self.assertIn('BOS', by_algorithm)
        self.assertAlmostEqual(by_algorithm['REGER']['average_rate'], 0.15)
        self.assertEqual(by_algorithm['REGER']['dataset_count'], 2)

    def test_staticdata_paper_table3_mode_uses_paper_rows(self) -> None:
        request = RequestFactory().get(
            '/api/staticdata/',
            {
                'data_dimension': 'all_datasets',
                'data_type': 'overall',
                'implement_language': 'java',
                'leaderboard_mode': 'paper_table3',
            },
        )
        with patch('myapp.views._all_dataset_keys_for_ui', return_value=['d1', 'd2']), patch(
            'myapp.paper_table3_leaderboard.paper_table3_display_rows',
            return_value=[
                {
                    'method': 'REGER',
                    'rank': 1.0,
                    'average_rate': 0.1,
                    'average_compression_ns': 1.0,
                    'average_decompression_ns': 2.0,
                    'median_balance_score': 0.2,
                    'dataset_count': 2,
                },
            ],
        ):
            response = StaticDataView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode('utf-8'))
        self.assertEqual([row['algorithm'] for row in payload['items']], ['REGER'])
        self.assertAlmostEqual(payload['items'][0]['compression_rate'], 0.1)
        self.assertAlmostEqual(payload['items'][0]['average_rank'], 1.0)

    def test_paper_table3_display_rows_uses_paper_snapshot(self) -> None:
        backend = Path('/tmp/paper-table3-snapshot')
        overall = paper_table3_display_rows(
            backend,
            data_type='overall',
            float_precision='all',
            implement_language='cpp',
            dataset_keys=['d7', 'd8'],
        )

        self.assertEqual(len(overall), 20)
        self.assertEqual(overall[0]['method'], 'BOS')
        self.assertAlmostEqual(overall[0]['average_rate'], 0.2933)
        self.assertAlmostEqual(overall[0]['average_compression_ns'], 12.6)
        self.assertAlmostEqual(overall[0]['average_decompression_ns'], 11.3)
        self.assertAlmostEqual(overall[0]['median_balance_score'], 0.1203)
        self.assertAlmostEqual(overall[0]['rank'], 9.2)

        integer = paper_table3_display_rows(
            backend,
            data_type='int',
            float_precision='all',
            implement_language='cpp',
            dataset_keys=['d7', 'd8'],
        )
        self.assertEqual(
            [row['method'] for row in integer],
            [row['method'] for row in overall],
        )
        self.assertAlmostEqual(integer[0]['average_rate'], 0.1590)
        self.assertAlmostEqual(integer[0]['average_compression_ns'], 7.8)
        self.assertAlmostEqual(integer[0]['average_decompression_ns'], 5.1)
        self.assertAlmostEqual(integer[0]['median_balance_score'], 0.0575)
        self.assertAlmostEqual(integer[0]['rank'], 5.9)

    def test_boxplot_manifest_forwards_paper_table3_mode(self) -> None:
        captured = {}
        groups_obj = {
            'compression_rate': {'title': 'Compression rate', 'charts': {}},
            'compression_time': {'title': 'Compression time', 'charts': {}},
            'decompression_time': {'title': 'Decompression time', 'charts': {}},
            'balanced_score': {'title': 'Balanced score', 'charts': {}},
        }
        pareto_chart = {
            'title': 'Pareto Frontier (Compression Rate vs Time)',
            'x_label': 'Compression rate',
            'x_min': 0.0,
            'x_max': 0.9,
            'y_label': 'Compression time (ns/point)',
            'points': [],
            'level_series': [],
            'frontier': [],
        }

        def fake_build(_backend, **kwargs):
            captured.update(kwargs)
            return groups_obj, pareto_chart, 2

        request = RequestFactory().get(
            '/api/benchmark-boxplots-manifest/',
            {
                'data_dimension': 'all_datasets',
                'data_type': 'overall',
                'implement_language': 'java',
                'leaderboard_mode': 'paper_table3',
            },
        )
        with patch(
            'myapp.benchmark_boxplot_result_store.load_precomputed_home_leaderboard_boxplots',
            return_value={'groups': groups_obj, 'pareto_chart': pareto_chart, 'meta': {}},
        ), patch(
            'myapp.home_leaderboard_boxplot_data.build_home_leaderboard_boxplot_groups',
            side_effect=fake_build,
        ):
            response = BenchmarkBoxplotManifestView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured['leaderboard_mode'], 'paper_table3')
        self.assertEqual(captured['implement_language'], 'cpp')
        self.assertIsInstance(captured['dataset_keys'], list)

    def test_boxplot_manifest_all_datasets_uses_timestamped_compress_scope(self) -> None:
        captured = {}
        groups_obj = {
            'compression_rate': {'title': 'Compression rate', 'charts': {}},
            'compression_time': {'title': 'Compression time', 'charts': {}},
            'decompression_time': {'title': 'Decompression time', 'charts': {}},
            'balanced_score': {'title': 'Balanced score', 'charts': {}},
        }
        pareto_chart = {
            'title': 'Pareto Frontier (Compression Rate vs Time)',
            'x_label': 'Compression rate',
            'x_min': 0.0,
            'x_max': 0.9,
            'y_label': 'Compression time (ns/point)',
            'points': [],
            'level_series': [],
            'frontier': [],
        }

        def fake_build(_backend, **kwargs):
            captured.update(kwargs)
            return groups_obj, pareto_chart, 2

        request = RequestFactory().get(
            '/api/benchmark-boxplots-manifest/',
            {
                'data_dimension': 'all_datasets',
                'data_type': 'float',
                'implement_language': 'java',
            },
        )
        with patch(
            'myapp.benchmark_boxplot_result_store.load_precomputed_home_leaderboard_boxplots',
            return_value={'groups': groups_obj, 'pareto_chart': pareto_chart, 'meta': {}},
        ), patch(
            'myapp.home_leaderboard_boxplot_data.build_home_leaderboard_boxplot_groups',
            side_effect=fake_build,
        ):
            response = BenchmarkBoxplotManifestView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured['dataset_keys']), 19)
        self.assertIn('Air-Quality', captured['dataset_keys'])
        self.assertIn('YaleLanguages', captured['dataset_keys'])
        self.assertNotIn('poisson_timestamps', captured['dataset_keys'])
        self.assertNotIn('AI4I-Maintenance', captured['dataset_keys'])

    def test_default_level_filters_use_best_compression_levels(self) -> None:
        params = parse_report_filter_params({})

        self.assertEqual(params['xz_level'], 9)
        self.assertEqual(params['lzma_level'], 9)
        self.assertEqual(params['zstd_level'], 9)
        self.assertEqual(params['chimp_history'], 128)
        self.assertEqual(params['spdp_level'], 9)
        self.assertEqual(params['zfp_tol_eps'], 2)

    def test_default_lzbench_levels_use_best_compression_levels(self) -> None:
        defaults = default_lzbench_level_selection()

        self.assertEqual(defaults['brieflz'], 9)
        self.assertEqual(defaults['bsc'], 'bsc6')
        self.assertEqual(defaults['csc'], 5)
        self.assertEqual(defaults['lizard'], 49)
        self.assertEqual(defaults['lz4fast'], 1)
        self.assertEqual(defaults['lz4hc'], 12)
        self.assertEqual(defaults['lzo1_variant'], 'lzo1z')
        self.assertEqual(defaults['slz_variant'], 'slz_zlib_3')
        self.assertEqual(defaults['ucl_nrv_variant'], 'ucl_nrv2e')
        self.assertEqual(defaults['zlib'], 9)

    def test_staticdata_balanced_score_uses_weighted_normalized_times(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 10,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'B',
                    'originalSize': 100,
                    'compressedSize': 20,
                    'compressionTime': 20,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 30,
                    'compressionTime': 30,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'B',
                    'originalSize': 100,
                    'compressedSize': 60,
                    'compressionTime': 60,
                    'decompressionTime': 1,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows)
        by_algorithm = {row['algorithm']: row for row in items}

        self.assertAlmostEqual(by_algorithm['A']['balanced_score'], 0.08)
        self.assertAlmostEqual(by_algorithm['B']['balanced_score'], 0.46)

    def test_staticdata_dataset_rows_use_boxplot_family_best_representatives(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'zstd_level_1',
                    'originalSize': 100,
                    'compressedSize': 40,
                    'compressionTime': 10,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'zstd_level_3',
                    'originalSize': 100,
                    'compressedSize': 20,
                    'compressionTime': 100,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'ts_2diff_bos',
                    'originalSize': 100,
                    'compressedSize': 30,
                    'compressionTime': 30,
                    'decompressionTime': 1,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows)
        by_algorithm = {row['algorithm']: row for row in items}

        self.assertEqual(set(by_algorithm), {'ZSTD', 'BOS'})
        self.assertAlmostEqual(by_algorithm['ZSTD']['compression_rate'], 0.2)
        self.assertAlmostEqual(by_algorithm['ZSTD']['compression_time'], 8.0)

    def test_staticdata_dataset_balanced_score_uses_same_family_labels_as_boxplots(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'ppmd8_9',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 10,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'ts_2diff_bos',
                    'originalSize': 100,
                    'compressedSize': 20,
                    'compressionTime': 20,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'ppmd8_9',
                    'originalSize': 100,
                    'compressedSize': 60,
                    'compressionTime': 60,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'ts_2diff_bos',
                    'originalSize': 100,
                    'compressedSize': 30,
                    'compressionTime': 30,
                    'decompressionTime': 1,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows)
        by_algorithm = {row['algorithm']: row for row in items}

        self.assertEqual(set(by_algorithm), {'PPMD8', 'BOS'})
        self.assertAlmostEqual(by_algorithm['PPMD8']['balanced_score'], 0.29)
        self.assertAlmostEqual(by_algorithm['BOS']['balanced_score'], 0.25)

    def test_staticdata_hides_lossy_family_labels_after_grouping(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'machete_rel_1e_5',
                    'originalSize': 100,
                    'compressedSize': 5,
                    'compressionTime': 10,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'most_minimal',
                    'originalSize': 100,
                    'compressedSize': 5,
                    'compressionTime': 10,
                    'decompressionTime': 1,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'ts_2diff_bos',
                    'originalSize': 100,
                    'compressedSize': 20,
                    'compressionTime': 20,
                    'decompressionTime': 1,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows)

        self.assertEqual([row['algorithm'] for row in items], ['BOS'])

    def test_staticdata_balanced_score_without_dataset_key_uses_final_rows(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 10,
                    'decompressionTime': 1,
                },
                {
                    'algorithm': 'B',
                    'originalSize': 100,
                    'compressedSize': 20,
                    'compressionTime': 40,
                    'decompressionTime': 1,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows)
        by_algorithm = {row['algorithm']: row for row in items}

        self.assertAlmostEqual(by_algorithm['A']['balanced_score'], 0.04)
        self.assertAlmostEqual(by_algorithm['B']['balanced_score'], 0.38)

    def test_staticdata_metric_modes_can_rank_by_median_and_average(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 100,
                    'decompressionTime': 1000,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 90,
                    'compressionTime': 300,
                    'decompressionTime': 3000,
                },
                {
                    'dataset_key': 'd1',
                    'algorithm': 'B',
                    'originalSize': 100,
                    'compressedSize': 40,
                    'compressionTime': 800,
                    'decompressionTime': 2000,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'B',
                    'originalSize': 100,
                    'compressedSize': 60,
                    'compressionTime': 1200,
                    'decompressionTime': 4000,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(
            rows,
            rate_mode='median',
            compression_time_mode='average',
            decompression_time_mode='median',
        )
        by_algorithm = {row['algorithm']: row for row in items}

        self.assertAlmostEqual(by_algorithm['A']['compression_rate'], 0.5)
        self.assertAlmostEqual(by_algorithm['B']['compression_rate'], 0.5)
        self.assertAlmostEqual(by_algorithm['A']['compression_time'], 16.0)
        self.assertAlmostEqual(by_algorithm['B']['compression_time'], 80.0)
        self.assertAlmostEqual(by_algorithm['A']['decompression_time'], 160.0)
        self.assertAlmostEqual(by_algorithm['B']['decompression_time'], 240.0)
        self.assertEqual(by_algorithm['A']['space_cost'], 100)

    def test_staticdata_average_rate_without_dataset_key_uses_rows(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 100,
                    'decompressionTime': 1000,
                },
                {
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 50,
                    'compressionTime': 300,
                    'decompressionTime': 3000,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(
            rows,
            rate_mode='average',
            compression_time_mode='average',
            decompression_time_mode='median',
            dataset_original_sizes={'__dataset__': 1000},
        )

        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['compression_rate'], 0.3)
        self.assertAlmostEqual(items[0]['compression_time'], 16.0)
        self.assertAlmostEqual(items[0]['decompression_time'], 160.0)

    def test_staticdata_average_rate_uses_arithmetic_mean(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 10,
                    'decompressionTime': 10,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'A',
                    'originalSize': 900,
                    'compressedSize': 450,
                    'compressionTime': 20,
                    'decompressionTime': 20,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows, rate_mode='average')

        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['compression_rate'], 0.3)

    def test_staticdata_overall_rate_uses_weighted_ratio(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    'dataset_key': 'd1',
                    'algorithm': 'A',
                    'originalSize': 100,
                    'compressedSize': 10,
                    'compressionTime': 10,
                    'decompressionTime': 10,
                },
                {
                    'dataset_key': 'd2',
                    'algorithm': 'A',
                    'originalSize': 900,
                    'compressedSize': 450,
                    'compressionTime': 20,
                    'decompressionTime': 20,
                },
            ],
        )

        items = _staticdata_items_from_metric_rows(rows, rate_mode='overall')

        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['compression_rate'], 460 / 1000)

    def test_staticdata_overall_data_type_combines_int_and_float_reports(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'demo'
            bundle.mkdir(parents=True)
            (bundle / 'demo_column_info.csv').write_text(
                '\n'.join([
                    'column_index,column_name,logical_type,column_csv_size_bytes',
                    '0,i,integer,100',
                    '1,f,double,200',
                    '',
                ]),
                encoding='utf-8',
            )
            report_header = 'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime\n'
            (bundle / 'demo_int_java_compression_result.csv').write_text(
                report_header + '0,A,100,25,100,10\n0,B,100,50,200,20\n',
                encoding='utf-8',
            )
            (bundle / 'demo_float_java_compression_result.csv').write_text(
                report_header + '1,A,200,50,300,30\n1,B,200,100,400,40\n',
                encoding='utf-8',
            )

            request = RequestFactory().get(
                '/api/staticdata/',
                {
                    'data_dimension': 'single_dataset',
                    'dataset': 'demo',
                    'data_type': 'overall',
                    'implement_language': 'java',
                },
            )
            with patch('myapp.views._DATASET_ROOT', tmp), patch(
                'myapp.views._dataset_bundle_dir',
                return_value=str(bundle),
            ):
                response = StaticDataView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode('utf-8'))
        by_algorithm = {row['algorithm']: row for row in payload['items']}
        self.assertEqual(set(by_algorithm), {'A', 'B'})
        self.assertAlmostEqual(by_algorithm['A']['compression_rate'], 75 / 300)
        self.assertAlmostEqual(by_algorithm['B']['compression_rate'], 150 / 300)
        self.assertEqual(by_algorithm['A']['space_cost'], 75)
        self.assertAlmostEqual(by_algorithm['A']['compression_time'], 400 / 37.5)

    def test_pareto_level_sort_key_for_lzbench_and_core_levels(self) -> None:
        self.assertEqual(pareto_level_sort_key('brieflz_1'), 1.0)
        self.assertEqual(pareto_level_sort_key('brieflz_9'), 9.0)
        self.assertEqual(pareto_level_sort_key('zstd_level_3'), 3.0)
        self.assertEqual(pareto_level_sort_key('xz_level_9'), 9.0)
        self.assertIsNone(pareto_level_sort_key('brotli'))

    def test_pareto_point_label_includes_level(self) -> None:
        self.assertEqual(pareto_point_label('brieflz_3'), 'BriefLZ (level 3)')
        self.assertEqual(pareto_point_label('zstd_level_5'), 'ZSTD (level 5)')

    def test_compute_pareto_chart_builds_level_series(self) -> None:
        family_rates = {'Brotli': [0.55]}
        family_comp = {'Brotli': [80.0]}
        level_rates = {
            'brieflz_1': [0.5],
            'brieflz_3': [0.45],
            'brieflz_9': [0.4],
        }
        level_comp = {
            'brieflz_1': [100.0],
            'brieflz_3': [200.0],
            'brieflz_9': [500.0],
        }
        chart = compute_pareto_chart(
            family_rates,
            family_comp,
            level_rates,
            level_comp,
            include_lossy=False,
        )
        self.assertEqual(len(chart['points']), 4)
        families = {p['family'] for p in chart['points']}
        self.assertIn('BriefLZ', families)
        self.assertIn('Brotli', families)
        level_series = chart['level_series']
        self.assertEqual(len(level_series), 1)
        self.assertEqual(level_series[0]['family'], 'BriefLZ')
        self.assertEqual(len(level_series[0]['points']), 3)

    def test_compute_pareto_chart_family_point_for_delta(self) -> None:
        family_rates = {
            'DELTA_BINARY_PACKED': [0.30, 0.32],
            'BOS': [0.31],
        }
        family_comp = {
            'DELTA_BINARY_PACKED': [50.0, 55.0],
            'BOS': [90.0],
        }
        chart = compute_pareto_chart(
            family_rates,
            family_comp,
            {},
            {},
            include_lossy=False,
        )
        delta_pts = [p for p in chart['points'] if p['family'] == 'DELTA_BINARY_PACKED']
        self.assertEqual(len(delta_pts), 1)
        self.assertAlmostEqual(float(delta_pts[0]['rate']), 0.31)
        self.assertAlmostEqual(float(delta_pts[0]['compression_time']), 52.5)

    def test_boxplot_group_label_abbreviates_delta_binary_packed(self) -> None:
        self.assertEqual(boxplot_group_label('DELTA_BINARY_PACKED'), 'DBP')
        self.assertEqual(pareto_point_label('DELTA_BINARY_PACKED'), 'DBP')

    def test_compute_pareto_chart_frontier_uses_only_real_points(self) -> None:
        chart = compute_pareto_chart(
            {
                'Compact': [0.20],
                'Middle': [0.40],
                'Fast': [0.80],
            },
            {
                'Compact': [100.0],
                'Middle': [50.0],
                'Fast': [20.0],
            },
            {},
            {},
            include_lossy=False,
        )

        self.assertEqual(chart['x_min'], 0.0)
        self.assertEqual(
            chart['frontier'],
            [
                [0.20, 100.0],
                [0.40, 50.0],
                [0.80, 20.0],
            ],
        )
        self.assertEqual(chart['frontier_style'], 'actual_polyline')

    def test_build_pareto_level_series_skips_single_level_families(self) -> None:
        points = [
            {
                'algorithm': 'Brotli',
                'family': 'Brotli',
                'level_order': None,
                'rate': 0.5,
                'compression_time': 10.0,
            },
            {
                'algorithm': 'ZSTD (level 1)',
                'family': 'ZSTD',
                'level_order': 1.0,
                'rate': 0.4,
                'compression_time': 20.0,
            },
        ]
        self.assertEqual(_build_pareto_level_series(points), [])

    def test_compute_balanced_score_metrics_uses_weighted_normalized_times_and_ranks(self) -> None:
        datasets = [
            {
                'Fast': (0.50, 10.0, 0.0),
                'Balanced': (0.25, 20.0, 0.0),
                'Compact': (0.10, 100.0, 0.0),
            },
            {
                'Fast': (0.60, 15.0, 0.0),
                'Balanced': (0.30, 30.0, 0.0),
                'Compact': (0.15, 90.0, 0.0),
            },
        ]

        scores, ranks = compute_balanced_score_metrics(datasets)

        self.assertIn('Balanced', scores)
        self.assertIn('Balanced', ranks)
        self.assertEqual(len(scores['Balanced']), 2)
        self.assertAlmostEqual(scores['Balanced'][0], 0.30 * ((20.0 - 10.0) / (100.0 - 10.0)) + 0.40 * 0.25)
        self.assertEqual(ranks['Fast'], [2.0, 2.0])
        self.assertEqual(ranks['Balanced'], [1.0, 1.0])
        self.assertEqual(ranks['Compact'], [3.0, 3.0])

    def test_best_metrics_keeps_base_rows_and_ignores_removed_level_algorithms(self) -> None:
        report = pd.DataFrame(
            [
                {
                    'column_index': 0,
                    'algorithm': 'ts_2diff_bos',
                    'originalSize': 800,
                    'compressedSize': 320,
                    'compressionTime': 800,
                    'decompressionTime': 100,
                },
                {
                    'column_index': 0,
                    'algorithm': 'ts_2diff_bos_l0',
                    'originalSize': 800,
                    'compressedSize': 80,
                    'compressionTime': 800,
                    'decompressionTime': 100,
                },
                {
                    'column_index': 0,
                    'algorithm': 'subcolumn',
                    'originalSize': 800,
                    'compressedSize': 240,
                    'compressionTime': 800,
                    'decompressionTime': 100,
                },
                {
                    'column_index': 0,
                    'algorithm': 'subcolumn_l0',
                    'originalSize': 800,
                    'compressedSize': 40,
                    'compressionTime': 800,
                    'decompressionTime': 100,
                },
                {
                    'column_index': 0,
                    'algorithm': 'oss_l0',
                    'originalSize': 800,
                    'compressedSize': 20,
                    'compressionTime': 800,
                    'decompressionTime': 100,
                },
            ],
        )

        best, per_algo = best_metrics_from_report(report, file_size=800)

        self.assertAlmostEqual(best['BOS'][0], 0.4)
        self.assertAlmostEqual(best['Sub-column'][0], 0.3)
        self.assertNotIn('OSS', best)
        self.assertNotIn('ts_2diff_bos_l0', per_algo)
        self.assertIn('subcolumn', per_algo)
        self.assertNotIn('subcolumn_l0', per_algo)
        self.assertNotIn('oss_l0', per_algo)

    def test_compute_pareto_chart_keeps_base_subcolumn_and_ignores_level_algorithms(self) -> None:
        chart = compute_pareto_chart(
            {
                'BOS': [0.4],
                'Sub-column': [0.3],
            },
            {
                'BOS': [80.0],
                'Sub-column': [20.0],
            },
            {
                'subcolumn': [0.3],
                'subcolumn_l0': [0.2],
            },
            {
                'subcolumn': [20.0],
                'subcolumn_l0': [10.0],
            },
            include_lossy=False,
        )

        self.assertEqual([point['algorithm'] for point in chart['points']], ['BOS', 'Sub-column'])
        self.assertEqual(chart['level_series'], [])

    def test_build_groups_adds_critical_difference_rank_chart(self) -> None:
        with TemporaryDirectory() as tmp:
            backend = Path(tmp)
            bundle = backend / 'dataset' / 'demo'
            bundle.mkdir(parents=True)
            (bundle / 'demo_column_info.csv').write_text(
                'column_name,logical_type,column_csv_size_bytes\nx,double,100\n',
                encoding='utf-8',
            )
            (bundle / 'demo_float_java_compression_result.csv').write_text(
                '\n'.join([
                    'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime',
                    '0,A,800,80,800,100',
                    '0,B,800,160,1600,100',
                    '',
                ]),
                encoding='utf-8',
            )
            with patch('myapp.home_leaderboard_boxplot_data.resolve_master_csv_path', return_value=None):
                groups, _pareto, _datasets_used = build_home_leaderboard_boxplot_groups(backend)

        chart = groups['balanced_score']['charts']['lossless_critical_difference']
        score_chart = groups['balanced_score']['charts']['lossless_score']

        self.assertEqual(score_chart['chart_type'], 'bar')
        self.assertTrue(score_chart['log_y'])
        self.assertEqual(score_chart['algorithms'], ['A', 'B'])
        self.assertAlmostEqual(score_chart['values'][0][0], 0.32)
        self.assertAlmostEqual(score_chart['values'][1][0], 0.94)

        self.assertEqual(chart['chart_type'], 'critical_difference')
        self.assertEqual(chart['title'], 'Critical difference diagram')
        self.assertEqual(chart['algorithms'], ['A', 'B'])
        self.assertEqual(chart['values'], [[1.0], [2.0]])
        self.assertEqual(chart['sample_count'], 1)
        self.assertIn('critical_difference', chart)

    def test_build_groups_respects_dataset_language_and_column_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            backend = Path(tmp)
            demo = backend / 'dataset' / 'demo'
            other = backend / 'dataset' / 'other'
            demo.mkdir(parents=True)
            other.mkdir(parents=True)
            column_info = '\n'.join([
                'column_index,column_name,logical_type,column_csv_size_bytes',
                '0,a,double,100',
                '1,b,double,100',
                '',
            ])
            (demo / 'demo_column_info.csv').write_text(column_info, encoding='utf-8')
            (other / 'other_column_info.csv').write_text(column_info, encoding='utf-8')
            header = 'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime\n'
            (demo / 'demo_float_java_compression_result.csv').write_text(
                header + '1,A,800,10,800,80\n1,B,800,20,1600,160\n',
                encoding='utf-8',
            )
            (demo / 'demo_float_cpp_compression_result.csv').write_text(
                header + '0,A,800,90,900,90\n1,A,800,80,800,80\n1,B,800,40,1600,160\n',
                encoding='utf-8',
            )
            (other / 'other_float_cpp_compression_result.csv').write_text(
                header + '1,A,800,5,100,10\n1,B,800,5,100,10\n',
                encoding='utf-8',
            )
            with patch('myapp.home_leaderboard_boxplot_data.resolve_master_csv_path', return_value=None):
                groups, pareto, datasets_used = build_home_leaderboard_boxplot_groups(
                    backend,
                    implement_language='cpp',
                    data_type='float',
                    dataset_keys=['demo'],
                    selected_column_index=1,
                )

        chart = groups['compression_rate']['charts']['lossless']
        self.assertEqual(datasets_used, 1)
        self.assertEqual(chart['algorithms'], ['A', 'B'])
        self.assertEqual(chart['values'], [[0.8], [0.4]])
        self.assertEqual(pareto['x_min'], 0.0)

    def test_best_metrics_from_report_uses_selected_lzbench_level(self) -> None:
        report = pd.DataFrame(
            [
                {
                    'column_index': 0,
                    'algorithm': 'kanzi_5',
                    'originalSize': 800,
                    'compressedSize': 160,
                    'compressionTime': 800,
                    'decompressionTime': 100,
                },
                {
                    'column_index': 0,
                    'algorithm': 'kanzi_9',
                    'originalSize': 800,
                    'compressedSize': 80,
                    'compressionTime': 800000,
                    'decompressionTime': 100,
                },
                {
                    'column_index': 0,
                    'algorithm': 'zstd_level_3',
                    'originalSize': 800,
                    'compressedSize': 200,
                    'compressionTime': 1000,
                    'decompressionTime': 100,
                },
            ],
        )

        best, _per_algo = best_metrics_from_report(
            report,
            file_size=800,
            implement_language='java',
            lzbench_levels={'kanzi': 5},
            zstd_level=3,
        )

        self.assertIn('Kanzi', best)
        rate, comp_ns, _decomp_ns = best['Kanzi']
        self.assertAlmostEqual(rate, 0.2)
        self.assertAlmostEqual(comp_ns, 8.0)

    def test_best_metrics_from_report_does_not_resanitize_overall_report(self) -> None:
        report = pd.DataFrame(
            [
                {
                    'column_index': 0,
                    'algorithm': 'DELTA_BINARY_PACKED',
                    'originalSize': 400,
                    'compressedSize': 100,
                    'compressionTime': 400,
                    'decompressionTime': 80,
                },
                {
                    'column_index': 0,
                    'algorithm': 'DELTA_BINARY_PACKED',
                    'originalSize': 800,
                    'compressedSize': 200,
                    'compressionTime': 800,
                    'decompressionTime': 160,
                },
            ],
        )

        best, per_algo = best_metrics_from_report(
            report,
            file_size=0,
            implement_language='cpp',
            use_report_original_size_for_rate=True,
            report_is_sanitized=True,
        )

        self.assertAlmostEqual(best['DBP'][0], 0.25)
        self.assertAlmostEqual(per_algo['DELTA_BINARY_PACKED'][0], 0.25)

    def test_metrics_for_dataset_keeps_overall_report_without_numeric_file_size(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            (bundle / 'demo_column_info.csv').write_text(
                'column_name,logical_type,column_csv_size_bytes\n'
                'published,timestamp,800\n',
                encoding='utf-8',
            )
            (bundle / 'demo_int_cpp_compression_result.csv').write_text(
                'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime\n'
                '0,DELTA_BINARY_PACKED,800,240,800,160\n',
                encoding='utf-8',
            )

            best, per_algo = metrics_for_dataset(
                'demo',
                bundle,
                implement_language='cpp',
                data_type='overall',
            )

        self.assertAlmostEqual(best['DBP'][0], 0.3)
        self.assertIn('DELTA_BINARY_PACKED', per_algo)

    def test_iter_catalog_benchmark_datasets_accepts_local_precomputed_reports(self) -> None:
        with TemporaryDirectory() as tmp:
            backend = Path(tmp)
            bundle = backend / 'dataset' / 'demo'
            bundle.mkdir(parents=True)
            (bundle / 'demo_column_info.csv').write_text('column_name,logical_type\nx,double\n', encoding='utf-8')
            (bundle / 'demo_float_java_compression_result.csv').write_text(
                'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime\n',
                encoding='utf-8',
            )
            with patch('myapp.home_leaderboard_boxplot_data.resolve_master_csv_path', return_value=None):
                pairs = iter_catalog_benchmark_datasets(backend)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], 'demo')


if __name__ == '__main__':
    unittest.main()
