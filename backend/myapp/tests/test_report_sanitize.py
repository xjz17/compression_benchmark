"""Tests for compression report algorithm sanitization."""

from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from myapp.lzbench_codec_catalog import (
    canonical_report_algorithm_key,
    is_lzbench_report_row_algorithm,
    is_lzbench_selector_garbage_algorithm,
)
from myapp.report_filters import filter_compression_report_dataframe, sanitize_compression_report_dataframe
from myapp.views import _filter_staticdata_items_hide_lossy


class ReportSanitizeTests(SimpleTestCase):
    def test_staticdata_lossy_filter_drops_family_labels(self):
        items = [
            {"algorithm": "SimPiece"},
            {"algorithm": "Sim-Piece"},
            {"algorithm": "simpiece_rel_range_2"},
            {"algorithm": "SZ"},
            {"algorithm": "SZ3"},
            {"algorithm": "BOS"},
            {"algorithm": "Simple8b"},
        ]

        out = _filter_staticdata_items_hide_lossy(items)
        self.assertEqual([row["algorithm"] for row in out], ["BOS", "Simple8b"])

    def test_drop_lzbench_selector_garbage(self):
        self.assertTrue(is_lzbench_selector_garbage_algorithm("lzbench selector brieflz"))
        self.assertIsNone(canonical_report_algorithm_key("lzbench selector brieflz"))

    def test_map_raw_brieflz_stdout_name(self):
        self.assertEqual(canonical_report_algorithm_key("brieflz 1.3.0 -5"), "brieflz_5")
        self.assertTrue(is_lzbench_report_row_algorithm("brieflz 1.3.0 -5"))

    def test_map_new_lzbench_stdout_names(self):
        self.assertEqual(canonical_report_algorithm_key("csc 2016-10-13 -5"), "csc_5")
        self.assertEqual(canonical_report_algorithm_key("lz4fast 1.10.0 -17"), "lz4fast_17")
        self.assertEqual(canonical_report_algorithm_key("lz4 1.10.0 --fast -17"), "lz4fast_17")
        self.assertEqual(canonical_report_algorithm_key("lz4hc 1.10.0 -12"), "lz4hc_12")
        self.assertEqual(canonical_report_algorithm_key("slz_deflate 1.2.1 -3"), "slz_deflate_3")
        self.assertEqual(canonical_report_algorithm_key("slz_zlib 1.2.1 -3"), "slz_zlib_3")
        self.assertEqual(canonical_report_algorithm_key("zlib 1.3.1 -9"), "zlib_9")

    def test_map_chimp128_aliases(self):
        self.assertEqual(canonical_report_algorithm_key("Chimp128"), "CHIMP128")
        self.assertEqual(canonical_report_algorithm_key("chimp128"), "CHIMP128")
        self.assertEqual(canonical_report_algorithm_key("chimp_128"), "CHIMP128")

    def test_map_legacy_lzma_rows_to_xz(self):
        self.assertEqual(canonical_report_algorithm_key("lzma_level_9"), "xz_level_9")
        self.assertEqual(canonical_report_algorithm_key("lzma"), "xz")

    def test_xz_filter_accepts_legacy_lzma_rows(self):
        df = pd.DataFrame(
            [
                {
                    "column_index": 1,
                    "algorithm": "lzma_level_1",
                    "originalSize": 100,
                    "compressedSize": 70,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
                {
                    "column_index": 1,
                    "algorithm": "xz_level_9",
                    "originalSize": 100,
                    "compressedSize": 40,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
            ]
        )

        default_out = filter_compression_report_dataframe(df, implement_language="python")
        self.assertEqual(default_out["algorithm"].tolist(), ["xz_level_9"])

        level_1 = filter_compression_report_dataframe(df, implement_language="python", xz_level=1)
        self.assertEqual(level_1["algorithm"].tolist(), ["xz_level_1"])

    def test_chimp_default_history_prefers_128_with_legacy_fallback(self):
        df = pd.DataFrame(
            [
                {
                    "column_index": 1,
                    "algorithm": "CHIMP",
                    "originalSize": 100,
                    "compressedSize": 60,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
                {
                    "column_index": 1,
                    "algorithm": "Chimp128",
                    "originalSize": 100,
                    "compressedSize": 50,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
            ]
        )

        default_out = filter_compression_report_dataframe(df, implement_language="python")
        self.assertEqual(default_out["algorithm"].tolist(), ["CHIMP128"])

        history_1 = filter_compression_report_dataframe(df, implement_language="python", chimp_history=1)
        self.assertEqual(history_1["algorithm"].tolist(), ["CHIMP"])

        legacy_only = filter_compression_report_dataframe(df.iloc[:1], implement_language="python")
        self.assertEqual(legacy_only["algorithm"].tolist(), ["CHIMP"])

    def test_parameter_variant_families_default_to_latest_variant(self):
        def row(algorithm):
            return {
                "column_index": 1,
                "algorithm": algorithm,
                "originalSize": 100,
                "compressedSize": 50,
                "compressionTime": 1,
                "decompressionTime": 1,
            }

        df = pd.DataFrame(
            [
                row("lzo1a"),
                row("lzo1z"),
                row("slz_deflate_3"),
                row("slz_zlib_3"),
                row("ucl_nrv2b_9"),
                row("ucl_nrv2e"),
            ]
        )

        default_out = filter_compression_report_dataframe(df, implement_language="python")
        self.assertEqual(default_out["algorithm"].tolist(), ["lzo1z", "slz_zlib_3", "ucl_nrv2e"])

        selected = filter_compression_report_dataframe(
            df,
            implement_language="python",
            lzbench_levels={
                "lzo1_variant": "lzo1a",
                "slz_variant": "slz_deflate_3",
                "ucl_nrv_variant": "ucl_nrv2b_9",
            },
        )
        self.assertEqual(selected["algorithm"].tolist(), ["lzo1a", "slz_deflate_3", "ucl_nrv2b_9"])

    def test_sanitize_dedupes_canonical_over_raw(self):
        df = pd.DataFrame(
            [
                {
                    "column_index": 1,
                    "algorithm": "brieflz 1.3.0 -1",
                    "originalSize": 13137,
                    "compressedSize": 2723,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
                {
                    "column_index": 1,
                    "algorithm": "brieflz_1",
                    "originalSize": 1858176,
                    "compressedSize": 900000,
                    "compressionTime": 2,
                    "decompressionTime": 2,
                },
            ]
        )
        out = sanitize_compression_report_dataframe(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["algorithm"], "brieflz_1")
        self.assertEqual(int(out.iloc[0]["originalSize"]), 1858176)

    def test_sanitize_drops_selector_placeholder(self):
        df = pd.DataFrame(
            [
                {
                    "column_index": 1,
                    "algorithm": "lzbench selector brieflz",
                    "originalSize": 23,
                    "compressedSize": 46,
                    "compressionTime": 69,
                    "decompressionTime": 92,
                },
                {
                    "column_index": 1,
                    "algorithm": "TS_2DIFF",
                    "originalSize": 80000,
                    "compressedSize": 1872,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
            ]
        )
        out = sanitize_compression_report_dataframe(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["algorithm"], "TS_2DIFF")

    def test_sanitize_drops_legacy_per_column_reger_rows(self):
        df = pd.DataFrame(
            [
                {
                    "column_index": 3,
                    "algorithm": "reger",
                    "originalSize": 100,
                    "compressedSize": 90,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
                {
                    "column_index": -1,
                    "algorithm": "reger",
                    "originalSize": 300,
                    "compressedSize": 60,
                    "compressionTime": 2,
                    "decompressionTime": 2,
                },
                {
                    "column_index": 3,
                    "algorithm": "TS_2DIFF",
                    "originalSize": 100,
                    "compressedSize": 30,
                    "compressionTime": 1,
                    "decompressionTime": 1,
                },
            ]
        )
        out = sanitize_compression_report_dataframe(df)
        self.assertEqual(out["algorithm"].tolist(), ["reger", "TS_2DIFF"])
        self.assertEqual(out[out["algorithm"] == "reger"]["column_index"].tolist(), [-1])
