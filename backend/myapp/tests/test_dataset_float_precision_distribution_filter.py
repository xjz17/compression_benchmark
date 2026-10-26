from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from myapp.dataset_float_precision_distribution import (
    filter_report_dataframe_for_float_precision,
    reger_float_precision_report_path,
)


class FloatPrecisionReportFilterTests(unittest.TestCase):
    def test_reger_uses_explicit_precision_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "sample"
            bundle.mkdir()
            column_info = bundle / "sample_column_info.csv"
            with column_info.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("column_index", "column_name", "logical_type"),
                )
                writer.writeheader()
                writer.writerow({"column_index": 1, "column_name": "fixed", "logical_type": "float"})
                writer.writerow({"column_index": 2, "column_name": "varying", "logical_type": "double"})
            decimal_dir = bundle / "float_decimal_places"
            decimal_dir.mkdir()
            pd.DataFrame([{"decimal_places": 2, "fraction": 0.9}]).to_csv(
                decimal_dir / "column_1.csv", index=False
            )
            pd.DataFrame([{"decimal_places": 2, "fraction": 0.5}]).to_csv(
                decimal_dir / "column_2.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "precision_scope": "fixed",
                        "column_index": -1,
                        "algorithm": "reger",
                        "originalSize": 800,
                        "compressedSize": 120,
                        "compressionTime": 10,
                        "decompressionTime": 4,
                        "isLossy": 0,
                        "lossRate": 0,
                    },
                    {
                        "precision_scope": "non_fixed",
                        "column_index": -1,
                        "algorithm": "reger",
                        "originalSize": 1600,
                        "compressedSize": 640,
                        "compressionTime": 20,
                        "decompressionTime": 8,
                        "isLossy": 0,
                        "lossRate": 0,
                    },
                ]
            ).to_csv(reger_float_precision_report_path(str(bundle)), index=False)
            report = pd.DataFrame(
                [
                    {"column_index": 1, "algorithm": "bos", "originalSize": 800, "compressedSize": 200},
                    {"column_index": 2, "algorithm": "bos", "originalSize": 1600, "compressedSize": 700},
                    {"column_index": -1, "algorithm": "reger", "originalSize": 2400, "compressedSize": 700},
                ]
            )

            fixed = filter_report_dataframe_for_float_precision(
                report,
                column_info_path=str(column_info),
                bundle_dir=str(bundle),
                float_precision="fixed",
            )
            self.assertEqual(set(fixed["algorithm"]), {"bos", "reger"})
            reger = fixed[fixed["algorithm"] == "reger"].iloc[0]
            self.assertEqual(int(reger["originalSize"]), 800)
            self.assertEqual(int(reger["compressedSize"]), 120)

            non_fixed = filter_report_dataframe_for_float_precision(
                report,
                column_info_path=str(column_info),
                bundle_dir=str(bundle),
                float_precision="non_fixed",
            )
            self.assertEqual(set(non_fixed["algorithm"]), {"bos", "reger"})
            reger = non_fixed[non_fixed["algorithm"] == "reger"].iloc[0]
            self.assertEqual(int(reger["originalSize"]), 1600)
            self.assertEqual(int(reger["compressedSize"]), 640)

            unfiltered = filter_report_dataframe_for_float_precision(
                report,
                column_info_path=str(column_info),
                bundle_dir=str(bundle),
                float_precision="all",
            )
            pd.testing.assert_frame_equal(unfiltered, report)


if __name__ == "__main__":
    unittest.main()
