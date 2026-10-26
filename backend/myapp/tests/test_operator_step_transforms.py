from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from myapp.dataset_operators import SUPPORTED_OPERATORS, _ColumnCatalogMeta
from myapp.column_info_loader import catalog_dtype_from_column_info_row
from myapp.dataset_catalog_assets import _read_catalog_master_csv


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_operator_step_byte_entropy.py"
PIPELINES_PATH = REPO_ROOT / "benchmark" / "reference" / "methods" / "method_operator_pipelines.json"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("operator_step_entropy_generator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_operator_ids():
    payload = json.loads(PIPELINES_PATH.read_text(encoding="utf-8"))
    operators = set()

    def collect(step):
        if isinstance(step, str) and step:
            operators.add(step)
        elif isinstance(step, dict):
            for item in step.get("any_of", []):
                collect(item)

    for pipeline in payload["pipelines"].values():
        if isinstance(pipeline, list):
            for step in pipeline:
                collect(step)
        elif isinstance(pipeline, dict):
            for typed_pipeline in pipeline.values():
                if isinstance(typed_pipeline, list):
                    for step in typed_pipeline:
                        collect(step)
    return operators


class OperatorStepTransformTests(unittest.TestCase):
    def test_medium_posts_published_is_parsed_as_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "medium_posts.csv"
            source.write_text(
                "published,domain,url\n"
                "2012-08-13 22:54:53.510Z,medium.com,https://example.test\n",
                encoding="utf-8",
            )

            frame = _read_catalog_master_csv("medium_posts", str(source))

        self.assertEqual(int(frame.loc[0, "published"]), 1344898493510000)
        self.assertEqual(
            catalog_dtype_from_column_info_row(
                {"column_name": "published", "logical_type": "string"}
            ),
            "timestamp",
        )

    def test_all_pipeline_operators_are_supported(self) -> None:
        generator = load_generator_module()

        missing = sorted(
            generator.OPERATOR_ALIASES.get(operator_id, operator_id)
            for operator_id in pipeline_operator_ids()
            if generator.OPERATOR_ALIASES.get(operator_id, operator_id) not in SUPPORTED_OPERATORS
        )

        self.assertEqual(missing, [])

    def test_entropy_for_empty_input_is_unknown_not_zero(self) -> None:
        generator = load_generator_module()

        entropy, count = generator._byte_entropy(pd.Series([], dtype=float))

        self.assertIsNone(entropy)
        self.assertEqual(count, 0)

    def test_regression_predictor_matches_reger_integer_residuals(self) -> None:
        generator = load_generator_module()
        meta = _ColumnCatalogMeta(logical_type="float", max_point_number=1)

        transformed = generator._fast_apply_operator_to_series(
            pd.Series([0.0, 0.1, 0.4, 0.9], dtype=float),
            "regression_predictor",
            meta=meta,
        )

        self.assertEqual(transformed.tolist(), [0.0, 1.0, 0.0])

    def test_previously_unsupported_pipeline_generates_ok_nonzero_entropy_steps(self) -> None:
        generator = load_generator_module()
        series = pd.Series(
            [
                11.0, 12.0, 13.0, 11.0, 12.0, 13.0, 99.0, 100.0,
                11.0, 12.0, 13.0, 101.0, 102.0, 11.0, 12.0, 13.0,
            ],
            dtype=float,
        )
        rows = generator._build_step_rows_for_pair(
            dataset_key="unit",
            source_csv=Path("unit.csv"),
            result_file=generator.CompressionResultFile(
                path=Path("unit_float_python_compression_result.csv"),
                data_type="float",
                language="python",
            ),
            algorithm="zstd_level_9",
            method_name="zstd",
            column_info=generator.ColumnInfo(
                index=0,
                name="value",
                logical_type="float",
                max_point_number=3,
            ),
            source_series=series,
            pipeline=[
                "lz_match",
                "match_parse_policy",
                "literal_entropy",
                "entropy_fse",
                "container_frame",
                "block_transform",
                "symbol_recode",
                "bitstream_pack",
                "entropy_special",
            ],
        )

        self.assertEqual({row["operator_status"] for row in rows}, {"ok"})
        self.assertTrue(all(row["before_byte_entropy"] for row in rows))
        self.assertTrue(all(row["after_byte_entropy"] for row in rows))
        self.assertTrue(all(row["byte_entropy_change"] for row in rows))
        self.assertFalse(
            any(abs(float(row["byte_entropy_change"])) < 1e-12 for row in rows),
            rows,
        )

    def test_bos_second_for_rebases_only_the_normal_stream(self) -> None:
        generator = load_generator_module()
        series = pd.Series(
            [0.0, *[float(100 + (index % 8)) for index in range(80)], 10000.0],
            dtype=float,
        )

        rows = generator._build_step_rows_for_pair(
            dataset_key="unit",
            source_csv=Path("unit.csv"),
            result_file=generator.CompressionResultFile(
                path=Path("unit_int_cpp_compression_result.csv"),
                data_type="int",
                language="cpp",
            ),
            algorithm="ts_2diff_bos",
            method_name="BOS",
            column_info=generator.ColumnInfo(
                index=0,
                name="value",
                logical_type="int",
                max_point_number=0,
            ),
            source_series=series,
            pipeline=["outlier_separation", "frame_of_reference"],
        )

        self.assertEqual([row["operator_status"] for row in rows], ["ok", "ok"])
        self.assertNotEqual(rows[1]["before_byte_entropy"], rows[1]["after_byte_entropy"])
        self.assertTrue(rows[1]["byte_entropy_change"])

    def test_dataset_operator_byte_codec_transform_is_available(self) -> None:
        from myapp.dataset_operators import _apply_operator_to_series

        out = _apply_operator_to_series(
            pd.Series([1.0, 2.0, 1.0, 2.0, 3.0], dtype=float),
            "lz_match",
            meta=_ColumnCatalogMeta(logical_type="float", max_point_number=0),
        )

        self.assertGreater(len(out), 0)
        self.assertNotEqual(out.tolist(), [1.0, 2.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
