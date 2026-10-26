from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from myapp.views import _operator_step_algorithm_options


class OperatorStepEntropyOptionsTests(unittest.TestCase):
    def test_options_use_method_label_without_parentheses(self) -> None:
        rows = [
            {
                'algorithm': 'alp',
                'method_name': 'ALP',
                'data_type': 'float',
                'result_file': 'float_report.csv',
            },
            {
                'algorithm': 'CHIMP',
                'method_name': 'CHIMP',
                'data_type': 'float',
                'result_file': 'float_report.csv',
            },
        ]

        options = _operator_step_algorithm_options(rows)

        self.assertEqual(
            [(row['algorithm'], row['label']) for row in options],
            [('alp', 'ALP'), ('CHIMP', 'CHIMP')],
        )

    def test_level_family_keeps_only_best_compression_rate_level(self) -> None:
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            with (bundle_dir / 'float_report.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        'column_index',
                        'algorithm',
                        'originalSize',
                        'compressedSize',
                        'compressionTime',
                        'decompressionTime',
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            'column_index': '0',
                            'algorithm': 'zstd_level_1',
                            'originalSize': '100',
                            'compressedSize': '40',
                            'compressionTime': '1',
                            'decompressionTime': '1',
                        },
                        {
                            'column_index': '0',
                            'algorithm': 'zstd_level_9',
                            'originalSize': '100',
                            'compressedSize': '25',
                            'compressionTime': '1',
                            'decompressionTime': '1',
                        },
                        {
                            'column_index': '0',
                            'algorithm': 'brieflz_1',
                            'originalSize': '100',
                            'compressedSize': '50',
                            'compressionTime': '1',
                            'decompressionTime': '1',
                        },
                        {
                            'column_index': '0',
                            'algorithm': 'brieflz_9',
                            'originalSize': '100',
                            'compressedSize': '30',
                            'compressionTime': '1',
                            'decompressionTime': '1',
                        },
                    ]
                )
            rows = [
                {
                    'algorithm': 'zstd_level_1',
                    'method_name': 'zstd',
                    'data_type': 'float',
                    'result_file': 'float_report.csv',
                },
                {
                    'algorithm': 'zstd_level_9',
                    'method_name': 'zstd',
                    'data_type': 'float',
                    'result_file': 'float_report.csv',
                },
                {
                    'algorithm': 'brieflz_1',
                    'method_name': 'BriefLZ',
                    'data_type': 'float',
                    'result_file': 'float_report.csv',
                },
                {
                    'algorithm': 'brieflz_9',
                    'method_name': 'BriefLZ',
                    'data_type': 'float',
                    'result_file': 'float_report.csv',
                },
                {
                    'algorithm': 'elf_on_chimp',
                    'method_name': 'elf',
                    'data_type': 'float',
                    'result_file': 'float_report.csv',
                },
                {
                    'algorithm': 'elf_plus',
                    'method_name': 'elf',
                    'data_type': 'float',
                    'result_file': 'float_report.csv',
                },
            ]

            options = _operator_step_algorithm_options(rows, bundle_dir=bundle_dir)

        self.assertEqual(
            [(row['algorithm'], row['label']) for row in options],
            [
                ('brieflz_9', 'BriefLZ'),
                ('elf_on_chimp', 'elf'),
                ('elf_plus', 'elf'),
                ('zstd_level_9', 'zstd'),
            ],
        )


if __name__ == '__main__':
    unittest.main()
