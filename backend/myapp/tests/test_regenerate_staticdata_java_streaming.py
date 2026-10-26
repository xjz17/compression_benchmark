from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "regenerate_staticdata_java.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("regenerate_staticdata_java", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_materialized_column_java_bench_uses_existing_column_csv(tmp_path, monkeypatch):
    module = _load_module()
    input_csv = tmp_path / "unit_float_col2.csv"
    input_csv.write_text("value\n1.5\n2.5\n", encoding="utf-8")
    column_info = tmp_path / "unit_column_info.csv"
    column_info.write_text(
        "column_index,column_name,logical_type,max_point_number\n"
        "2,value,float,3\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run_java_benchmark(
        input_path,
        data_type,
        output_path,
        *,
        column_info=None,
        column_index=None,
        algorithms=None,
    ):
        calls.append(
            {
                "input_path": Path(input_path),
                "data_type": data_type,
                "column_info": Path(column_info),
                "column_index": column_index,
                "algorithms": algorithms,
            }
        )
        with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "column_index",
                    "algorithm",
                    "originalSize",
                    "compressedSize",
                    "compressionTime",
                    "decompressionTime",
                    "isLossy",
                    "lossRate",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "column_index": "2",
                    "algorithm": "TS_2DIFF",
                    "originalSize": "16",
                    "compressedSize": "8",
                    "compressionTime": "3",
                    "decompressionTime": "2",
                    "isLossy": "0",
                    "lossRate": "0",
                }
            )

    monkeypatch.setattr(module, "run_java_benchmark", fake_run_java_benchmark)

    rows = module._run_materialized_column_java_bench(
        tmp_csv_path=input_csv,
        slug="unit",
        data_type="float",
        column_index=2,
        column_info_path=column_info,
        algorithms=["TS_2DIFF"],
    )

    assert calls == [
        {
            "input_path": input_csv,
            "data_type": "float",
            "column_info": column_info,
            "column_index": 2,
            "algorithms": ["TS_2DIFF"],
        }
    ]
    assert rows == [
        {
            "column_index": 2,
            "algorithm": "TS_2DIFF",
            "originalSize": 16,
            "compressedSize": 8,
            "compressionTime": 3,
            "decompressionTime": 2,
            "isLossy": 0,
            "lossRate": 0.0,
        }
    ]
    assert input_csv.exists()


def test_persist_report_rows_keeps_prior_columns_and_upserts_exact_key(tmp_path, monkeypatch):
    module = _load_module()
    report = tmp_path / "unit_float_java_compression_result.csv"
    fields = [
        "column_index",
        "algorithm",
        "originalSize",
        "compressedSize",
        "compressionTime",
        "decompressionTime",
        "isLossy",
        "lossRate",
    ]

    def row(column_index, algorithm, compressed_size):
        return {
            "column_index": column_index,
            "algorithm": algorithm,
            "originalSize": 16,
            "compressedSize": compressed_size,
            "compressionTime": 3,
            "decompressionTime": 2,
            "isLossy": 0,
            "lossRate": 0,
        }

    monkeypatch.delenv("WEB_COMPRESSION_STATICDATA_REGEN", raising=False)
    module._persist_report_rows(report, [row(1, "zstd_level_7", 8)])
    module._persist_report_rows(
        report,
        [row(2, "zstd_level_7", 7), row(1, "zstd_level_8", 6)],
    )
    module._persist_report_rows(report, [row(2, "zstd_level_7", 5)])

    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == fields
    assert len(rows) == 3
    by_key = {(int(item["column_index"]), item["algorithm"]): item for item in rows}
    assert (1, "zstd_level_7") in by_key
    assert (1, "zstd_level_8") in by_key
    assert by_key[(2, "zstd_level_7")]["compressedSize"] == "5"


def test_persist_report_rows_only_rewrites_when_explicitly_requested(tmp_path, monkeypatch):
    module = _load_module()
    report = tmp_path / "unit_int_java_compression_result.csv"
    old = {
        "column_index": 1,
        "algorithm": "old",
        "originalSize": 8,
        "compressedSize": 4,
        "compressionTime": 1,
        "decompressionTime": 1,
        "isLossy": 0,
        "lossRate": 0,
    }
    new = {**old, "column_index": 2, "algorithm": "new"}
    monkeypatch.delenv("WEB_COMPRESSION_STATICDATA_REGEN", raising=False)
    module._persist_report_rows(report, [old])
    monkeypatch.setenv("WEB_COMPRESSION_STATICDATA_REGEN", "1")
    module._persist_report_rows(report, [new])

    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["column_index"], row["algorithm"]) for row in rows] == [("2", "new")]
