from __future__ import annotations

import os

from django.test import SimpleTestCase

from myapp.native_lz_family import (
    NATIVE_LZ_FAMILY_ALGORITHMS,
    bench_native_lz_family_bytes,
    compress_native_lz_family,
    decompress_native_lz_family,
)
from myapp.active_method_policy import is_active_algorithm_key
from myapp.lzbench_codec_catalog import all_catalog_algorithm_keys
from myapp.method_catalog import (
    algorithm_keys_for_method_name,
    default_method_catalog_algorithm_keys,
    method_name_for_algorithm_key,
)


class NativeLzFamilyTests(SimpleTestCase):
    def test_catalog_keeps_only_base_bos_and_subcolumn_rows(self) -> None:
        removed_level_keys = {
            *(f"ts_2diff_bos_l{level}" for level in range(6)),
            *(f"subcolumn_l{level}" for level in range(6)),
            *(f"oss_l{level}" for level in range(6)),
        }

        self.assertEqual(algorithm_keys_for_method_name("BOS"), ["ts_2diff_bos"])
        self.assertTrue(removed_level_keys.isdisjoint(default_method_catalog_algorithm_keys()))

    def test_lzsse_levels_share_one_method_family(self) -> None:
        levels = ["lzsse2", "lzsse4", "lzsse4fast", "lzsse8", "lzsse8fast"]

        self.assertEqual(algorithm_keys_for_method_name("LZSSE"), levels)
        for level in levels:
            with self.subTest(level=level):
                self.assertEqual(method_name_for_algorithm_key(level), "LZSSE")

    def test_all_missing_smart_grid_algorithms_have_native_presets(self) -> None:
        expected = {
            "density",
            "gipfeli",
            "lzmat",
            "lzo1",
            "lzo1a",
            "lzo1b",
            "lzo1c",
            "lzo1f",
            "lzo1y",
            "lzo1z",
            "lzo2a",
            "lzrw",
            "lzsse2",
            "lzsse4",
            "lzsse4fast",
            "lzsse8",
            "lzsse8fast",
            "lzvn",
            "pithy",
            "shrinker",
            "ucl_nrv2d",
            "ucl_nrv2e",
            "wflz",
            "xpack",
            "yalz77",
            "yappy",
        }

        self.assertTrue(expected.issubset(set(NATIVE_LZ_FAMILY_ALGORITHMS)))
        self.assertNotIn("glza", NATIVE_LZ_FAMILY_ALGORITHMS)

    def test_generic_and_lzbench_report_rows_have_native_presets(self) -> None:
        expected = {
            "brotli",
            "gz",
            "lz4-framed",
            "snappy-framed",
            "blosc_blosclz",
            "blosc2_blosclz",
            "fpzip",
            "ext_zfp_zfpy",
            "fse",
            "huff0",
            "xz_level_0",
            "xz_level_9",
            "zstd_level_1",
            "zstd_level_9",
            "brieflz_1",
            "bsc6",
            "bzip2_9",
            "csc_5",
            "lz4fast_17",
            "lz4hc_12",
            "lzo1x_999",
            "slz_deflate_3",
            "slz_zlib_3",
            "tamp_15",
            "zlib_9",
            "zlib_ng_9",
        }

        native = set(NATIVE_LZ_FAMILY_ALGORITHMS)

        self.assertTrue(expected.issubset(native))
        active_catalog = {key for key in all_catalog_algorithm_keys() if is_active_algorithm_key(key)}
        self.assertTrue(active_catalog.issubset(native))

    def test_round_trip_and_stable_sizes_for_representative_presets(self) -> None:
        payload = (b"alpha-beta-alpha-beta-" * 12) + bytes(range(64)) + (b"xyzxyzxyz" * 7)
        expected_sizes = {
            "lzrw": 146,
            "gipfeli": 98,
            "lzsse4fast": 113,
            "xpack": 94,
        }

        for algorithm, expected_size in expected_sizes.items():
            with self.subTest(algorithm=algorithm):
                compressed = compress_native_lz_family(payload, algorithm)
                self.assertEqual(len(compressed), expected_size)
                self.assertEqual(decompress_native_lz_family(compressed, algorithm, len(payload)), payload)

    def test_retired_glza_is_rejected_by_active_encode_api(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper164-active85-glza-retired-v1"):
            compress_native_lz_family(b"historical-glza" * 16, "glza")

    def test_representative_generated_rows_round_trip(self) -> None:
        payload = (
            b"alpha-alpha-alpha;"
            + bytes(range(64))
            + b"bravo-charlie-delta-" * 11
            + bytes(range(63, 0, -1))
        )
        for algorithm in (
            "gz",
            "brotli",
            "xz_level_9",
            "zstd_level_9",
            "fpzip",
            "ext_zfp_zfpy",
            "fse",
            "huff0",
            "brieflz_1",
            "bzip2_9",
            "csc_5",
            "zlib_9",
            "lz4fast_17",
            "lzo1x_999",
        ):
            with self.subTest(algorithm=algorithm):
                compressed = compress_native_lz_family(payload, algorithm)
                self.assertEqual(decompress_native_lz_family(compressed, algorithm, len(payload)), payload)

    def test_fast_path_round_trips_large_payload(self) -> None:
        payload = (b"0123456789abcdef" * 2048) + (b"sensor-value;" * 1024)
        old = os.environ.get("COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES")
        os.environ["COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES"] = "1"
        try:
            compressed = compress_native_lz_family(payload, "zlib_8")
        finally:
            if old is None:
                os.environ.pop("COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES", None)
            else:
                os.environ["COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES"] = old

        self.assertLess(len(compressed), len(payload))
        self.assertEqual(decompress_native_lz_family(compressed, "zlib_8", len(payload)), payload)

    def test_every_native_preset_round_trips_repetitive_and_mixed_payloads(self) -> None:
        payloads = [
            b"",
            b"abc" * 128,
            (b"0123456789abcdef" * 40) + bytes(range(127)) + (b"abcdef" * 31),
        ]

        for algorithm in NATIVE_LZ_FAMILY_ALGORITHMS:
            for payload in payloads:
                with self.subTest(algorithm=algorithm, payload_size=len(payload)):
                    compressed = compress_native_lz_family(payload, algorithm)
                    self.assertEqual(
                        decompress_native_lz_family(compressed, algorithm, len(payload)),
                        payload,
                    )

    def test_column_benchmark_accumulates_lossless_sizes(self) -> None:
        columns = [b"aaaaabbbbbccccc" * 20, b"0123456789" * 30]

        result = bench_native_lz_family_bytes(columns, "lzrw")

        self.assertEqual(result[0], sum(len(c) for c in columns))
        self.assertGreater(result[1], 0)
        self.assertEqual(result[4], 0.0)
        self.assertGreater(result[5], 0.0)

    def test_methods_catalog_maps_native_lz_method_names_to_report_keys(self) -> None:
        expected = {
            "Density": "density",
            "Gipfeli": "gipfeli",
            "LZMAT": "lzmat",
            "LZO1": "lzo1",
            "LZO2A": "lzo2a",
            "LZRW": "lzrw",
            "LZSSE2": "lzsse2",
            "LZSSE4": "lzsse4",
            "LZSSE4 Fast": "lzsse4fast",
            "LZSSE8": "lzsse8",
            "LZSSE8 Fast": "lzsse8fast",
            "LZVN": "lzvn",
            "Pithy": "pithy",
            "Shrinker": "shrinker",
            "SLZ": "slz_zlib_3",
            "UCL NRV": "ucl_nrv2e",
            "WFLZ": "wflz",
            "XPACK": "xpack",
            "YALZ77": "yalz77",
            "Yappy": "yappy",
        }

        for method_name, algorithm_key in expected.items():
            with self.subTest(method_name=method_name):
                self.assertIn(algorithm_key, algorithm_keys_for_method_name(method_name))
        self.assertEqual(algorithm_keys_for_method_name("GLZA"), [])
        self.assertEqual(
            algorithm_keys_for_method_name("GLZA", catalog=("glza",)),
            ["glza"],
        )
        for variant in ("lzo1a", "lzo1b", "lzo1c", "lzo1f", "lzo1x_999", "lzo1y", "lzo1z"):
            with self.subTest(method_name="LZO1", variant=variant):
                self.assertIn(variant, algorithm_keys_for_method_name("LZO1"))
        for variant in ("ucl_nrv2b_9", "ucl_nrv2d", "ucl_nrv2e"):
            with self.subTest(method_name="UCL NRV", variant=variant):
                self.assertIn(variant, algorithm_keys_for_method_name("UCL NRV"))

    def test_methods_catalog_maps_expanded_report_rows_to_method_names(self) -> None:
        expected = {
            "xz_level_9": "XZ",
            "zstd_level_9": "zstd",
            "fse": "FSE",
            "huff0": "Huff0",
            "zlib_9": "zlib",
            "zlib_ng_9": "zlib-ng",
            "csc_5": "CSC",
            "lz4fast_17": "LZ4 Fast",
            "lz4hc_12": "LZ4HC",
            "bzip2_9": "Bzip2",
            "lzo1x_999": "LZO1",
            "slz_zlib_3": "SLZ",
            "ucl_nrv2e": "UCL NRV",
            "tamp_15": "TAMP",
        }

        for algorithm_key, method_name in expected.items():
            with self.subTest(algorithm_key=algorithm_key):
                self.assertEqual(method_name_for_algorithm_key(algorithm_key), method_name)
