from __future__ import annotations

import django
import tempfile
import unittest
from django.apps import apps
from django.test import Client
from pathlib import Path
from unittest.mock import patch

from myapp.dataset_catalog_compression_rate import (
    best_available_column_compression_metrics,
    clear_report_lookup_cache,
)
from myapp.dataset_catalog_result_store import (
    _merge_structural_dtype_distributions,
    ledger_public_compress_summary_path,
    ledger_catalog_summary_path,
    load_precomputed_catalog_summary,
)
from myapp.views import _focused_original_dataset_keys


class DatasetCatalogLedgerSummaryTests(unittest.TestCase):
    def test_structural_dtype_cache_replaces_stale_timestamp_counts(self) -> None:
        summary = {
            'dataset_count': 20,
            'dtype_column_distribution': [
                {'category': 'timestamp', 'count': 11, 'value': 11},
            ],
            'dtype_overall_compression_rate_distribution': [
                {'category': 'timestamp', 'compression_rate': 0.1, 'value': 0.1},
            ],
        }
        structural_source = {
            'dataset_count': 20,
            'dtype_column_distribution': [
                {'category': 'timestamp', 'count': 20, 'value': 20},
            ],
        }

        merged = _merge_structural_dtype_distributions(
            summary,
            structural_source=structural_source,
        )

        self.assertEqual(merged['dtype_column_distribution'][0]['count'], 20)
        self.assertEqual(
            merged['dtype_overall_compression_rate_distribution'],
            summary['dtype_overall_compression_rate_distribution'],
        )

    def test_ledger_summary_is_available_for_dataset_overview_cache(self) -> None:
        self.assertTrue(ledger_public_compress_summary_path().is_file())
        self.assertTrue(ledger_catalog_summary_path().is_file())

        summary = load_precomputed_catalog_summary()
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary['dataset_count'], 19)
        self.assertEqual(len(summary['available_categories']), 4)

        counts = {
            row['category']: int(row['count'])
            for row in summary['category_distribution']
        }
        self.assertEqual(sum(counts.values()), 19)
        self.assertGreater(counts['Finance & Economics'], 0)
        self.assertGreater(counts['Climate & Weather'], 0)

        total_size = float(summary['total_size_bytes'])
        total_tsfile = float(summary['total_tsfile_size_bytes'])
        total_compressed = float(summary['total_compressed_size_bytes'])
        overall_rate = float(summary['overall_compression_rate'])
        self.assertGreater(total_size, 0)
        self.assertGreater(total_tsfile, 0)
        self.assertEqual(total_tsfile, total_compressed)
        self.assertLess(total_compressed, total_size)
        self.assertGreater(overall_rate, 0)
        self.assertLess(overall_rate, 1)

    def test_best_available_column_metrics_uses_lossless_per_column_minimum(self) -> None:
        clear_report_lookup_cache()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            column_info = root / 'demo_column_info.csv'
            int_report = root / 'demo_int_java_compression_result.csv'
            float_report = root / 'demo_float_java_compression_result.csv'
            column_info.write_text(
                '\n'.join(
                    [
                        'column_index,column_name,logical_type,max_point_number,column_csv_size_bytes',
                        '0,a,integer,0,100',
                        '1,b,integer,0,200',
                        '2,c,float,0,300',
                    ]
                ),
                encoding='utf-8',
            )
            int_report.write_text(
                '\n'.join(
                    [
                        'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime,isLossy,lossRate',
                        '0,algo_a,100,80,0,0,0,0',
                        '0,algo_b,100,30,0,0,0,0',
                        '0,sz_foo,100,1,0,0,1,0.1',
                        '1,algo_a,200,100,0,0,0,0',
                        '1,ts_2diff_bos_l2,200,1,0,0,0,0',
                    ]
                ),
                encoding='utf-8',
            )
            float_report.write_text(
                '\n'.join(
                    [
                        'column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime,isLossy,lossRate',
                        '2,algo_float,300,60,0,0,1,0.1',
                    ]
                ),
                encoding='utf-8',
            )

            def fake_report_path(dataset_key, dtype, *, user_record=None):
                return str(int_report if dtype == 'int' else float_report)

            with patch(
                'myapp.dataset_catalog_compression_rate._java_report_path',
                side_effect=fake_report_path,
            ):
                metrics = best_available_column_compression_metrics(
                    column_info_path=str(column_info),
                    dataset_key='demo',
                    user_record=None,
                )

            self.assertEqual(metrics['used_column_count'], 2)
            self.assertEqual(metrics['missing_column_count'], 1)
            self.assertEqual(metrics['compressed_size_bytes'], 130.0)
            self.assertEqual(metrics['original_size_bytes'], 300.0)
            self.assertAlmostEqual(metrics['compression_rate_sum'], 0.8)
            self.assertAlmostEqual(metrics['average_compression_rate'], 0.4)
        clear_report_lookup_cache()

    def test_dataset_summary_api_returns_focused_original_scope(self) -> None:
        if not apps.ready:
            django.setup()
        focused = _focused_original_dataset_keys()
        self.assertEqual(len(focused), 19)
        self.assertNotIn('poisson_timestamps', focused)
        self.assertNotIn('AI4I-Maintenance', focused)
        self.assertIn('Air-Quality', focused)
        self.assertIn('YaleLanguages', focused)

        response = Client().get('/api/datasets/', {'part': 'summary'})
        self.assertEqual(response.status_code, 200)
        summary = response.json()['summary']
        self.assertEqual(summary['dataset_count'], 19)
        self.assertIn('total_compressed_size_bytes', summary)
        self.assertEqual(
            summary['total_compressed_size_bytes'],
            summary['total_tsfile_size_bytes'],
        )

        dtype_counts = {
            row['category']: int(row['count'])
            for row in summary['dtype_column_distribution']
        }
        self.assertEqual(set(dtype_counts), {'int', 'float', 'timestamp'})
        self.assertGreater(dtype_counts['int'], 0)
        self.assertGreater(dtype_counts['float'], 0)
        self.assertEqual(dtype_counts['timestamp'], 19)

        for key, value_field in (
            ('dtype_data_point_distribution', 'data_points'),
            ('dtype_csv_size_distribution', 'csv_size_bytes'),
            ('dtype_tsfile_size_distribution', 'tsfile_size_bytes'),
        ):
            rows = {
                row['category']: float(row[value_field])
                for row in summary[key]
            }
            self.assertEqual(set(rows), {'int', 'float', 'timestamp'})
            self.assertGreater(rows['timestamp'], 0)

        rate_dtypes = {
            row['category']
            for row in summary['dtype_overall_compression_rate_distribution']
        }
        self.assertEqual(rate_dtypes, {'int', 'float', 'timestamp'})
        average_rate_dtypes = {
            row['category']
            for row in summary['dtype_average_compression_rate_distribution']
        }
        self.assertEqual(average_rate_dtypes, {'int', 'float', 'timestamp'})

        items_response = Client().get('/api/datasets/', {'part': 'items'})
        self.assertEqual(items_response.status_code, 200)
        items = items_response.json()['items']
        self.assertEqual({row['key'] for row in items}, focused)
        self.assertNotIn('poisson_timestamps', {row['key'] for row in items})
        air_quality = next(row for row in items if row['key'] == 'Air-Quality')
        self.assertIn('float', air_quality['benchmark_dtypes'])
