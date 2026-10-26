#!/usr/bin/env python3
"""Minimal end-to-end check for the resume-safe Python chunk runner."""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "backfill_python_chunked_staticdata.py"


class ChunkedBackfillTest(unittest.TestCase):
    def test_1000_int_values_are_lossless_and_summed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            data_root = work / "cleaned"
            info_root = work / "catalog"
            report_root = work / "reports"
            state_dir = work / "state"
            data_root.mkdir()
            bundle = info_root / "Tiny"
            bundle.mkdir(parents=True)
            with (data_root / "Tiny.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["value", "all_null"])
                writer.writerows(([i % 31, ""] for i in range(1000)))
            with (bundle / "Tiny_column_info.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["column_index", "column_name", "logical_type", "max_point_number"])
                writer.writerow([0, "value", "integer", 0])
                writer.writerow([1, "all_null", "integer", 0])
            tasks = work / "tasks.tsv"
            tasks.write_text(
                "dataset\tdtype\tcolumn_index\tcolumn_name\talgorithm\n"
                "Tiny\tint\t7\tvalue\tTS_2DIFF\n"
                "Tiny\tint\t8\tall_null\tnever_run_a\n"
                "Tiny\tint\t8\tall_null\tnever_run_b\n",
                encoding="utf-8",
            )
            command = [
                sys.executable, str(SCRIPT), "--tasks", str(tasks),
                "--dataset-root", str(data_root), "--column-info-root", str(info_root),
                "--report-root", str(report_root), "--state-dir", str(state_dir),
                "--report-suffix", ".selftest", "--patch-rows", "100",
                "--initial-segment-rows", "500", "--timeout-sec", "120",
            ]
            try:
                subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=180)
            except subprocess.CalledProcessError as exc:
                self.fail(
                    f"chunk runner exited {exc.returncode}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
                )
            # A restart must reuse committed segments rather than benchmark them again.
            subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=180)

            report = report_root / "Tiny" / "Tiny_int_python_compression_result.csv.selftest"
            with report.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["algorithm"], "TS_2DIFF")
            self.assertEqual(int(rows[0]["column_index"]), 7)
            self.assertEqual(int(rows[0]["originalSize"]), 8000)
            self.assertGreater(int(rows[0]["compressedSize"]), 0)
            self.assertEqual(int(rows[0]["isLossy"]), 0)
            self.assertEqual(float(rows[0]["lossRate"]), 0.0)
            with sqlite3.connect(state_dir / "state.sqlite3") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM segments WHERE status='complete'").fetchone()[0], 2)
                counts = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
                self.assertEqual(counts, {"complete": 1, "skipped_unmapped_or_empty": 2})
            status = json.loads((state_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["summary"], {"complete": 1, "skipped_unmapped_or_empty": 2})


if __name__ == "__main__":
    unittest.main()
