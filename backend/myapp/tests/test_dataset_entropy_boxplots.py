from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from myapp.dataset_correlation_heatmaps import (
    _all_catalog_operator_ids,
    _build_operator_entropy_change_boxplots,
    _method_operator_ids,
    build_dataset_correlation_heatmaps,
    load_correlation_heatmap_context,
)
from myapp.dataset_operators import (
    SUPPORTED_OPERATORS,
    apply_scaling_series,
    operator_statistic_csv_path,
)
from myapp.encoding_operator_catalog import load_method_page_operators
from myapp.views import (
    _build_overall_operator_step_byte_entropy_chart,
    _build_overall_operator_step_markov3_entropy_rate_chart,
    _merge_entropy_boxplot_payloads,
    _merge_heatmap_blocks,
    _overall_catalog_bundle_keys,
)


class DatasetEntropyBoxplotTests(unittest.TestCase):
    def test_operator_entropy_boxplots_include_before_after_medians(self) -> None:
        charts = _build_operator_entropy_change_boxplots(
            operator_ids=['delta', 'rle'],
            operator_labels=['Delta', 'RLE'],
            operator_entropy={
                'rle': {
                    'byte_entropy_before': [8.0, 6.0, 4.0],
                    'byte_entropy_after': [4.0, 3.0, 2.0],
                    'byte_entropy_change': [-4.0, -3.0, -2.0],
                    'value_entropy_before': [5.0, 4.0, 3.0],
                    'value_entropy_after': [3.0, 2.0, 1.0],
                    'value_entropy_change': [-2.0, -2.0, -2.0],
                },
                'delta': {
                    'byte_entropy_before': [8.0, 6.0, 4.0],
                    'byte_entropy_after': [7.0, 5.0, 3.0],
                    'byte_entropy_change': [-1.0, -1.0, -1.0],
                    'value_entropy_before': [5.0, 4.0, 3.0],
                    'value_entropy_after': [4.0, 3.0, 2.0],
                    'value_entropy_change': [-1.0, -1.0, -1.0],
                },
            },
        )

        byte_chart = charts['byte_entropy_change']
        self.assertEqual(byte_chart['algorithms'], ['RLE', 'Delta'])
        self.assertEqual(byte_chart['values'], [[-4.0, -3.0, -2.0], [-1.0, -1.0, -1.0]])
        self.assertEqual(byte_chart['before_medians'], [6.0, 6.0])
        self.assertEqual(byte_chart['after_medians'], [3.0, 5.0])

    def test_all_algorithm_heatmap_combines_value_and_byte_based_algorithms(self) -> None:
        payload = build_dataset_correlation_heatmaps(
            bundle_dir=Path(tempfile.mkdtemp()),
            dataset_key='unit',
            file_key='unit',
            column_info_path='unit_column_info.csv',
            column_indices=[0, 1, 2],
            column_sizes={0: 100.0, 1: 100.0, 2: 100.0},
            report_cache={
                'int': pd.DataFrame(
                    [
                        {'column_index': 0, 'algorithm': 'TS_2DIFF', 'compressedSize': 10},
                        {'column_index': 1, 'algorithm': 'TS_2DIFF', 'compressedSize': 20},
                        {'column_index': 2, 'algorithm': 'TS_2DIFF', 'compressedSize': 30},
                        {'column_index': 0, 'algorithm': 'Bzip2', 'compressedSize': 30},
                        {'column_index': 1, 'algorithm': 'Bzip2', 'compressedSize': 20},
                        {'column_index': 2, 'algorithm': 'Bzip2', 'compressedSize': 10},
                    ]
                ),
            },
            baseline_stat_rows=[
                {
                    'column_index': 0,
                    'mean': 1,
                    'standard_deviation': 1,
                    'spread': 10,
                    'byte_entropy': 1,
                    'value_entropy': 2,
                    'repeat': 0.1,
                    'sortedness': 0.9,
                },
                {
                    'column_index': 1,
                    'mean': 2,
                    'standard_deviation': 2,
                    'spread': 20,
                    'byte_entropy': 2,
                    'value_entropy': 3,
                    'repeat': 0.2,
                    'sortedness': 0.5,
                },
                {
                    'column_index': 2,
                    'mean': 4,
                    'standard_deviation': 3,
                    'spread': 30,
                    'byte_entropy': 4,
                    'value_entropy': 5,
                    'repeat': 0.3,
                    'sortedness': 0.1,
                },
            ],
            operator_ids=[],
        )

        self.assertIn('algorithm_heatmap', payload)
        self.assertEqual(payload['algorithm_heatmap']['title'], 'Compression algorithms vs column features')
        self.assertEqual(payload['algorithm_heatmap']['y_labels'], ['Bzip2', 'TS_2DIFF'])

    def test_operator_heatmap_uses_full_operator_catalog_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            baseline_rows = [
                {'column_index': 0, 'mean': 1, 'standard_deviation': 1, 'spread': 10, 'byte_entropy': 1, 'value_entropy': 2, 'repeat': 0.1, 'sortedness': 0.9},
                {'column_index': 1, 'mean': 2, 'standard_deviation': 2, 'spread': 20, 'byte_entropy': 2, 'value_entropy': 3, 'repeat': 0.2, 'sortedness': 0.5},
                {'column_index': 2, 'mean': 4, 'standard_deviation': 3, 'spread': 30, 'byte_entropy': 4, 'value_entropy': 5, 'repeat': 0.3, 'sortedness': 0.1},
            ]
            for operator_id, after_values in {
                'delta': [1.5, 2.25, 3.0],
                'lz_match': [1.2, 3.0, 3.5],
                'run_length': [0.5, 2.0, 5.0],
            }.items():
                op_dir = bundle_dir / 'operators' / operator_id
                op_dir.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            'column_index': idx,
                            'byte_entropy': value,
                            'value_entropy': value + 1,
                        }
                        for idx, value in enumerate(after_values)
                    ]
                ).to_csv(op_dir / 'unit_numeric_statistic.csv', index=False)

            payload = build_dataset_correlation_heatmaps(
                bundle_dir=bundle_dir,
                dataset_key='unit',
                file_key='unit',
                column_info_path='unit_column_info.csv',
                column_indices=[0, 1, 2],
                column_sizes={0: 100.0, 1: 100.0, 2: 100.0},
                report_cache={},
                baseline_stat_rows=baseline_rows,
                operator_ids=['delta', 'lz_match', 'run_length'],
            )

        self.assertEqual(
            payload['operator_heatmap']['y_labels'],
            ['delta', 'lz matching', 'run length'],
        )
        self.assertEqual(len(payload['operator_heatmap']['values']), 3)

    def test_operator_entropy_charts_are_segmented_by_numeric_column_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            float_precision_dir = bundle_dir / 'float_decimal_places'
            float_precision_dir.mkdir()
            pd.DataFrame(
                [{'decimal_places': 2, 'count': 9, 'fraction': 0.9}]
            ).to_csv(float_precision_dir / 'column_1.csv', index=False)
            pd.DataFrame(
                [
                    {'decimal_places': 1, 'count': 5, 'fraction': 0.5},
                    {'decimal_places': 2, 'count': 5, 'fraction': 0.5},
                ]
            ).to_csv(float_precision_dir / 'column_2.csv', index=False)
            column_info = bundle_dir / 'unit_column_info.csv'
            pd.DataFrame(
                [
                    {'column_index': 0, 'column_name': 'a', 'logical_type': 'integer', 'max_point_number': 0, 'column_csv_size_bytes': 100},
                    {'column_index': 1, 'column_name': 'b', 'logical_type': 'float', 'max_point_number': 2, 'column_csv_size_bytes': 100},
                    {'column_index': 2, 'column_name': 'c', 'logical_type': 'double', 'max_point_number': 0, 'column_csv_size_bytes': 100},
                ]
            ).to_csv(column_info, index=False)
            baseline_rows = [
                {'column_index': 0, 'mean': 1, 'standard_deviation': 1, 'spread': 10, 'byte_entropy': 10, 'value_entropy': 2, 'repeat': 0.1, 'sortedness': 0.9},
                {'column_index': 1, 'mean': 2, 'standard_deviation': 2, 'spread': 20, 'byte_entropy': 20, 'value_entropy': 3, 'repeat': 0.2, 'sortedness': 0.5},
                {'column_index': 2, 'mean': 4, 'standard_deviation': 3, 'spread': 30, 'byte_entropy': 30, 'value_entropy': 5, 'repeat': 0.3, 'sortedness': 0.1},
            ]
            op_dir = bundle_dir / 'operators' / 'delta'
            op_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {'column_index': 0, 'byte_entropy': 8, 'value_entropy': 1},
                    {'column_index': 1, 'byte_entropy': 15, 'value_entropy': 2},
                    {'column_index': 2, 'byte_entropy': 25, 'value_entropy': 4},
                ]
            ).to_csv(op_dir / 'unit_numeric_statistic.csv', index=False)

            payload = build_dataset_correlation_heatmaps(
                bundle_dir=bundle_dir,
                dataset_key='unit',
                file_key='unit',
                column_info_path=str(column_info),
                column_indices=[0, 1, 2],
                column_sizes={0: 100.0, 1: 100.0, 2: 100.0},
                report_cache={},
                baseline_stat_rows=baseline_rows,
                operator_ids=['delta'],
            )

        segments = {
            row['segment_key']: row
            for row in payload['operator_entropy_change_boxplots']['byte_entropy_change']['segments']
        }
        self.assertEqual(segments['int']['values'], [[-2.0]])
        self.assertEqual(segments['int']['before_medians'], [10.0])
        self.assertEqual(segments['int']['after_medians'], [8.0])
        self.assertEqual(segments['fixed_float']['values'], [[-5.0]])
        self.assertEqual(segments['non_fixed_float']['values'], [[-5.0]])

    def test_overall_heatmap_merge_uses_median_per_operator_feature(self) -> None:
        merged = _merge_heatmap_blocks(
            [
                {
                    'operator_heatmap': {
                        'title': 'Operator heatmap',
                        'description': '',
                        'x_labels': ['repeat', 'spread'],
                        'y_labels': ['delta'],
                        'values': [[1.0, 3.0]],
                    },
                },
                {
                    'operator_heatmap': {
                        'title': 'Operator heatmap',
                        'description': '',
                        'x_labels': ['repeat', 'spread'],
                        'y_labels': ['delta', 'rle'],
                        'values': [[5.0, None], [2.0, 4.0]],
                    },
                },
            ],
            'operator_heatmap',
        )

        self.assertEqual(merged['y_labels'], ['delta', 'rle'])
        self.assertEqual(merged['values'], [[3.0, 3.0], [2.0, 4.0]])

    def test_overall_entropy_boxplot_merge_preserves_operator_segments(self) -> None:
        merged = _merge_entropy_boxplot_payloads(
            [
                {
                    'operator_entropy_change_boxplots': {
                        'byte_entropy_change': {
                            'title': 'Byte entropy change',
                            'description': '',
                            'algorithms': ['Delta'],
                            'values': [[-2.0, -4.0]],
                            'before_medians': [8.0],
                            'after_medians': [5.0],
                            'y_label': 'bits',
                            'log_y': False,
                            'segments': [
                                {
                                    'segment_key': 'int',
                                    'segment_label': 'Integer columns',
                                    'column_count': 2,
                                    'algorithms': ['Delta'],
                                    'values': [[-2.0]],
                                    'before_medians': [8.0],
                                    'after_medians': [6.0],
                                },
                            ],
                        },
                    },
                },
                {
                    'operator_entropy_change_boxplots': {
                        'byte_entropy_change': {
                            'title': 'Byte entropy change',
                            'description': '',
                            'algorithms': ['Delta', 'RLE'],
                            'values': [[-6.0], [-1.0]],
                            'before_medians': [10.0, 7.0],
                            'after_medians': [4.0, 6.0],
                            'y_label': 'bits',
                            'log_y': False,
                            'segments': [
                                {
                                    'segment_key': 'int',
                                    'segment_label': 'Integer columns',
                                    'column_count': 3,
                                    'algorithms': ['Delta'],
                                    'values': [[-6.0]],
                                    'before_medians': [10.0],
                                    'after_medians': [4.0],
                                },
                                {
                                    'segment_key': 'float',
                                    'segment_label': 'Float columns',
                                    'column_count': 1,
                                    'algorithms': ['RLE'],
                                    'values': [[-1.0]],
                                    'before_medians': [7.0],
                                    'after_medians': [6.0],
                                },
                            ],
                        },
                    },
                },
            ],
        )['byte_entropy_change']

        self.assertEqual(merged['algorithms'], ['Delta', 'RLE'])
        self.assertEqual(merged['values'], [[-2.0, -4.0, -6.0], [-1.0]])
        self.assertEqual(merged['before_medians'], [9.0, 7.0])
        self.assertEqual(merged['after_medians'], [4.5, 6.0])
        segments = {row['segment_key']: row for row in merged['segments']}
        self.assertEqual(segments['int']['column_count'], 5)
        self.assertEqual(segments['int']['before_medians'], [9.0])
        self.assertEqual(segments['int']['after_medians'], [5.0])

    def test_overall_operator_step_byte_entropy_chart_aligns_method_operator_catalog(self) -> None:
        chart = _build_overall_operator_step_byte_entropy_chart(
            [
                {
                    'dataset_key': 'ds1',
                    'data_type': 'int',
                    'algorithm': 'alg_a',
                    'operator_id': 'delta',
                    'before_byte_entropy': '8.0',
                    'after_byte_entropy': '6.0',
                    'byte_entropy_change': '-2.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds2',
                    'data_type': 'float',
                    'algorithm': 'alg_b',
                    'operator_id': 'delta',
                    'before_byte_entropy': '9.0',
                    'after_byte_entropy': '4.0',
                    'byte_entropy_change': '-3.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds2',
                    'data_type': 'float',
                    'algorithm': 'alg_b',
                    'operator_id': 'rle',
                    'before_byte_entropy': '7.0',
                    'after_byte_entropy': '6.0',
                    'byte_entropy_change': '-1.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds2',
                    'data_type': 'float',
                    'algorithm': 'alg_b',
                    'operator_id': 'run_length',
                    'before_byte_entropy': '9.0',
                    'after_byte_entropy': '3.0',
                    'byte_entropy_change': '-6.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds2',
                    'data_type': 'float',
                    'algorithm': 'alg_b',
                    'operator_id': 'subcolumn',
                    'before_byte_entropy': '9.0',
                    'after_byte_entropy': '2.0',
                    'byte_entropy_change': '-7.0',
                    'before_count': '10',
                    'after_count': '10',
                },
            ],
            display_names={
                'delta': 'Delta',
                'rle': 'run length',
                'run_length': 'run length',
                'lz_match': 'LZ match',
            },
            operator_ids=['rle', 'run_length', 'lz_match', 'delta'],
        )

        assert chart is not None
        self.assertEqual(chart['chart_type'], 'bar')
        self.assertEqual(chart['title'], 'Overall byte entropy reduction by operator')
        self.assertEqual(chart['operator_ids'], ['run_length', 'delta', 'lz_match'])
        self.assertEqual(chart['algorithms'], ['run length', 'Delta', 'LZ match'])
        self.assertEqual(chart['mean_changes'], [3.5, 2.5, None])
        self.assertEqual(chart['sample_counts'], [1, 2, 0])
        self.assertNotIn('before_medians', chart)
        self.assertNotIn('after_medians', chart)
        self.assertNotIn('values', chart)
        self.assertNotIn('mean_reductions', chart)
        self.assertNotIn('segments', chart)

    def test_overall_operator_step_markov3_entropy_rate_chart_aligns_method_operator_catalog(self) -> None:
        chart = _build_overall_operator_step_markov3_entropy_rate_chart(
            [
                {
                    'dataset_key': 'ds1',
                    'data_type': 'int',
                    'algorithm': 'alg_a',
                    'operator_id': 'delta',
                    'before_byte_entropy': '8.0',
                    'after_byte_entropy': '6.0',
                    'byte_entropy_change': '-2.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds2',
                    'data_type': 'float',
                    'algorithm': 'alg_b',
                    'operator_id': 'rle',
                    'before_byte_entropy': '9.0',
                    'after_byte_entropy': '3.0',
                    'byte_entropy_change': '-6.0',
                    'before_count': '10',
                    'after_count': '10',
                },
            ],
            display_names={
                'delta': 'Delta',
                'rle': 'run length',
                'run_length': 'run length',
            },
            operator_ids=['rle', 'run_length', 'delta'],
        )

        assert chart is not None
        self.assertEqual(chart['title'], 'Overall Markov-3 byte entropy-rate reduction by operator')
        self.assertEqual(chart['operator_ids'], ['run_length', 'delta'])
        self.assertEqual(chart['algorithms'], ['run length', 'Delta'])
        self.assertEqual(chart['mean_changes'], [6.0, 2.0])
        self.assertEqual(chart['sample_counts'], [1, 1])

    def test_overall_operator_step_byte_entropy_chart_uses_dataset_equal_aggregation(self) -> None:
        chart = _build_overall_operator_step_byte_entropy_chart(
            [
                {
                    'dataset_key': 'ds1',
                    'data_type': 'int',
                    'algorithm': 'alg_a',
                    'operator_id': 'delta',
                    'before_byte_entropy': '8.0',
                    'after_byte_entropy': '7.0',
                    'byte_entropy_change': '-1.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds1',
                    'data_type': 'int',
                    'algorithm': 'alg_a',
                    'operator_id': 'delta',
                    'before_byte_entropy': '8.0',
                    'after_byte_entropy': '5.0',
                    'byte_entropy_change': '-3.0',
                    'before_count': '10',
                    'after_count': '10',
                },
                {
                    'dataset_key': 'ds2',
                    'data_type': 'int',
                    'algorithm': 'alg_b',
                    'operator_id': 'delta',
                    'before_byte_entropy': '8.0',
                    'after_byte_entropy': '3.0',
                    'byte_entropy_change': '-5.0',
                    'before_count': '10',
                    'after_count': '10',
                },
            ],
            display_names={'delta': 'Delta'},
            operator_ids=['delta'],
        )

        assert chart is not None
        self.assertEqual(chart['operator_ids'], ['delta'])
        self.assertEqual(chart['algorithms'], ['Delta'])
        self.assertEqual(chart['mean_changes'], [3.5])
        self.assertEqual(chart['sample_counts'], [2])

    def test_overall_entropy_segment_sorts_by_displayed_arrow_delta(self) -> None:
        merged = _merge_entropy_boxplot_payloads(
            [
                {
                    'operator_entropy_change_boxplots': {
                        'byte_entropy_change': {
                            'title': 'Byte entropy change',
                            'description': '',
                            'algorithms': ['A', 'B'],
                            'values': [[-10.0], [-1.0]],
                            'before_medians': [100.0, 1.0],
                            'after_medians': [90.0, 0.0],
                            'y_label': 'bits',
                            'log_y': False,
                            'segments': [
                                {
                                    'segment_key': 'int',
                                    'segment_label': 'Integer columns',
                                    'column_count': 2,
                                    'algorithms': ['A', 'B'],
                                    'values': [[-10.0], [-1.0]],
                                    'before_medians': [100.0, 1.0],
                                    'after_medians': [90.0, 0.0],
                                },
                            ],
                        },
                    },
                },
            ],
        )['byte_entropy_change']

        segment = merged['segments'][0]
        self.assertEqual(segment['before_medians'], [50.5, 50.5])
        self.assertEqual(segment['algorithms'], ['B', 'A'])

    def test_overall_catalog_bundle_keys_prefers_entropy_ready_scope_without_focused_markers(self) -> None:
        class StubPath:
            def __init__(self, exists: bool) -> None:
                self.exists = exists

            def is_file(self) -> bool:
                return self.exists

        with (
            patch('myapp.views._focused_original_dataset_keys', return_value=set()),
            patch('myapp.views._all_dataset_keys_for_ui', return_value=['wide', 'ready']),
            patch('myapp.views._built_in_dataset_catalog_ready', return_value=True),
            patch('myapp.views.catalog_has_statistics_bundle', return_value=True),
            patch(
                'myapp.views._operator_step_entropy_csv_path',
                side_effect=lambda key: StubPath(key == 'ready'),
            ),
        ):
            self.assertEqual(_overall_catalog_bundle_keys(), ['ready'])

    def test_every_method_page_operator_can_back_dataset_operator_statistics(self) -> None:
        catalog_ids = {row['id'] for row in load_method_page_operators()}
        self.assertFalse(catalog_ids - set(SUPPORTED_OPERATORS))

    def test_dataset_entropy_operator_set_matches_methods_visible_count(self) -> None:
        ids = list(_method_operator_ids())

        self.assertEqual(len(ids), 28)
        self.assertNotIn('rd', ids)
        self.assertNotIn('subcolumn', ids)
        self.assertNotIn('ease', ids)
        self.assertEqual(_all_catalog_operator_ids(Path(tempfile.mkdtemp())), ids)

    def test_scaling_operator_handles_high_decimal_precision_as_float64(self) -> None:
        out = apply_scaling_series(
            pd.Series([1.234567890123456, 2.345678901234567], dtype=float),
            as_float=True,
            max_point_number=24,
        )

        self.assertEqual(len(out), 2)
        self.assertTrue(all(pd.notna(out)))

    def test_correlation_context_materializes_missing_operator_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / 'bundle'
            bundle_dir.mkdir()
            master_csv = root / 'unit.csv'
            master_csv.write_text(
                'a,b,c\n'
                '1,2,4\n'
                '2,4,8\n'
                '3,8,16\n'
                '4,16,32\n',
                encoding='utf-8',
            )
            column_info = bundle_dir / 'unit_column_info.csv'
            pd.DataFrame(
                [
                    {
                        'column_index': 0,
                        'column_name': 'a',
                        'logical_type': 'integer',
                        'max_point_number': 0,
                        'column_csv_size_bytes': 100,
                    },
                    {
                        'column_index': 1,
                        'column_name': 'b',
                        'logical_type': 'integer',
                        'max_point_number': 0,
                        'column_csv_size_bytes': 100,
                    },
                    {
                        'column_index': 2,
                        'column_name': 'c',
                        'logical_type': 'integer',
                        'max_point_number': 0,
                        'column_csv_size_bytes': 100,
                    },
                ]
            ).to_csv(column_info, index=False)
            pd.DataFrame(
                [
                    {'column_index': 0, 'mean': 2.5, 'standard_deviation': 1.118, 'spread': 3, 'byte_entropy': 1, 'value_entropy': 2, 'repeat': 0.0, 'sortedness': 1.0},
                    {'column_index': 1, 'mean': 7.5, 'standard_deviation': 5.408, 'spread': 14, 'byte_entropy': 2, 'value_entropy': 3, 'repeat': 0.0, 'sortedness': 1.0},
                    {'column_index': 2, 'mean': 15.0, 'standard_deviation': 10.817, 'spread': 28, 'byte_entropy': 3, 'value_entropy': 4, 'repeat': 0.0, 'sortedness': 1.0},
                ]
            ).to_csv(bundle_dir / 'unit_numeric_statistic.csv', index=False)

            delta_stat = operator_statistic_csv_path(bundle_dir, 'unit', 'delta')
            self.assertFalse(delta_stat.exists())

            payload = load_correlation_heatmap_context(
                bundle_dir=bundle_dir,
                dataset_key='unit',
                file_key='unit',
                column_info_path=str(column_info),
                master_csv=str(master_csv),
            )

            self.assertIsNotNone(payload)
            self.assertTrue(delta_stat.exists())
            self.assertIn('delta', payload['operator_heatmap']['y_labels'])

    def test_correlation_context_can_skip_operator_statistic_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / 'bundle'
            bundle_dir.mkdir()
            master_csv = root / 'unit.csv'
            master_csv.write_text(
                'a,b,c\n'
                '1,2,4\n'
                '2,4,8\n'
                '3,8,16\n'
                '4,16,32\n',
                encoding='utf-8',
            )
            column_info = bundle_dir / 'unit_column_info.csv'
            pd.DataFrame(
                [
                    {'column_index': 0, 'column_name': 'a', 'logical_type': 'integer', 'max_point_number': 0, 'column_csv_size_bytes': 100},
                    {'column_index': 1, 'column_name': 'b', 'logical_type': 'integer', 'max_point_number': 0, 'column_csv_size_bytes': 100},
                    {'column_index': 2, 'column_name': 'c', 'logical_type': 'integer', 'max_point_number': 0, 'column_csv_size_bytes': 100},
                ]
            ).to_csv(column_info, index=False)
            pd.DataFrame(
                [
                    {'column_index': 0, 'mean': 2.5, 'standard_deviation': 1.118, 'spread': 3, 'byte_entropy': 1, 'value_entropy': 2, 'repeat': 0.0, 'sortedness': 1.0},
                    {'column_index': 1, 'mean': 7.5, 'standard_deviation': 5.408, 'spread': 14, 'byte_entropy': 2, 'value_entropy': 3, 'repeat': 0.0, 'sortedness': 1.0},
                    {'column_index': 2, 'mean': 15.0, 'standard_deviation': 10.817, 'spread': 28, 'byte_entropy': 3, 'value_entropy': 4, 'repeat': 0.0, 'sortedness': 1.0},
                ]
            ).to_csv(bundle_dir / 'unit_numeric_statistic.csv', index=False)

            delta_stat = operator_statistic_csv_path(bundle_dir, 'unit', 'delta')
            payload = load_correlation_heatmap_context(
                bundle_dir=bundle_dir,
                dataset_key='unit',
                file_key='unit',
                column_info_path=str(column_info),
                master_csv=str(master_csv),
                materialize_missing_operator_assets=False,
            )

            self.assertIsNotNone(payload)
            self.assertFalse(delta_stat.exists())


if __name__ == '__main__':
    unittest.main()
