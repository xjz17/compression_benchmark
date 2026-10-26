from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts" / "dbp_float64_python_cli.py"
if str(CLI.parent) not in sys.path:
    sys.path.insert(0, str(CLI.parent))

from dbp_float64_python_cli import _iter_manifest  # noqa: E402
from backend.myapp.dbp_float64_codec import encode  # noqa: E402


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class DBPFloat64CLITest(unittest.TestCase):
    def test_manifest_parser_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = self._manifest_row(
                "a", 8, 0, 0, struct.pack("<d", 1.0), 0, b"\0", 1, 1, 0
            )
            manifest = root / "blocks.tsv"
            manifest.write_text(good + "\nmalformed\n", encoding="utf-8")
            rows = _iter_manifest(manifest)
            self.assertIs(iter(rows), rows)
            self.assertEqual(next(rows).aggregate_id, "a")
            with self.assertRaisesRegex(ValueError, "expected 15 fields"):
                next(rows)

    def test_bench_manifest_validates_and_aggregates_both_widths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f64 = struct.pack("<2d", 1.25, -2.5)
            f32 = struct.pack("<2f", 1.25, -2.5)
            values_blob = b"PV" + f64 + f32
            missing_blob = b"M\x02\x00\x0f"
            (root / "values.bin").write_bytes(values_blob)
            (root / "missing.bin").write_bytes(missing_blob)
            (root / "aggregates.jsonl").write_text("{}\n", encoding="utf-8")

            rows = [
                self._manifest_row("a", 8, 2, 2, f64, 1, b"\x02", 3, 2, 1),
                self._manifest_row("b", 4, 2, 2 + len(f64), f32, 2, b"\x00", 2, 2, 0),
                self._manifest_row(
                    "c", 4, 2, len(values_blob), b"", 3, b"\x0f", 4, 0, 4
                ),
            ]
            manifest = root / "blocks.tsv"
            manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
            output = root / "results.jsonl"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "bench-manifest",
                    "--manifest",
                    str(manifest),
                    "--aggregates",
                    str(root / "aggregates.jsonl"),
                    "--output",
                    str(output),
                    "--repeats",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["aggregate_id"] for row in results], ["a", "b", "c"])
            expected_keys = {
                "status",
                "aggregate_id",
                "width",
                "blocks",
                "rows",
                "missing_count",
                "validity_bytes",
                "header_bytes",
                "valid_count",
                "original_bytes",
                "compressed_bytes",
                "encode_ns",
                "decode_ns",
                "scaled_blocks",
                "raw_blocks",
                "verified_values",
                "frame_sequence_sha256",
            }
            self.assertTrue(all(set(row) == expected_keys for row in results))
            self.assertEqual(
                {
                    key: results[0][key]
                    for key in (
                        "status",
                        "width",
                        "blocks",
                        "rows",
                        "missing_count",
                        "validity_bytes",
                        "header_bytes",
                        "valid_count",
                        "original_bytes",
                        "scaled_blocks",
                        "raw_blocks",
                        "verified_values",
                    )
                },
                {
                    "status": "ok",
                    "width": 8,
                    "blocks": 1,
                    "rows": 3,
                    "missing_count": 1,
                    "validity_bytes": 1,
                    "header_bytes": 10,
                    "valid_count": 2,
                    "original_bytes": 16,
                    "scaled_blocks": 1,
                    "raw_blocks": 0,
                    "verified_values": 2,
                },
            )
            self.assertEqual(results[1]["width"], 4)
            self.assertEqual(results[1]["verified_values"], 2)
            self.assertEqual(results[2]["status"], "n_a")
            self.assertEqual(results[2]["compressed_bytes"], 10)
            self.assertEqual(results[2]["raw_blocks"], 1)
            self.assertEqual(results[2]["verified_values"], 0)
            expected_frame = encode([1.25, -2.5], max_point=2)
            self.assertEqual(
                results[0]["frame_sequence_sha256"],
                sha256(struct.pack("<I", len(expected_frame)) + expected_frame),
            )
            self.assertTrue(
                all(len(row["frame_sequence_sha256"]) == 64 for row in results)
            )
            self.assertTrue(all(row["encode_ns"] > 0 for row in results))
            self.assertTrue(all(row["decode_ns"] > 0 for row in results))

    def test_bench_manifest_fails_on_segment_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = struct.pack("<d", 1.0)
            (root / "values.bin").write_bytes(raw)
            (root / "missing.bin").write_bytes(b"\0")
            (root / "aggregates.jsonl").write_text("{}\n", encoding="utf-8")
            row = self._manifest_row("a", 8, 0, 0, raw, 0, b"\0", 1, 1, 0)
            fields = row.split("\t")
            fields[6] = "00" * 32
            (root / "blocks.tsv").write_text("\t".join(fields) + "\n", encoding="utf-8")
            output = root / "results.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "bench-manifest",
                    "--manifest",
                    str(root / "blocks.tsv"),
                    "--aggregates",
                    str(root / "aggregates.jsonl"),
                    "--output",
                    str(output),
                    "--repeats",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("values SHA mismatch", completed.stderr)
            self.assertFalse(output.exists())

    @staticmethod
    def _manifest_row(
        aggregate_id: str,
        width: int,
        max_point: int,
        values_offset: int,
        values: bytes,
        missing_offset: int,
        missing: bytes,
        row_count: int,
        valid_count: int,
        missing_count: int,
    ) -> str:
        return "\t".join(
            [
                aggregate_id,
                str(width),
                str(max_point),
                "values.bin",
                str(values_offset),
                str(len(values)),
                sha256(values),
                "missing.bin",
                str(missing_offset),
                str(len(missing)),
                sha256(missing),
                "11" * 32,
                str(row_count),
                str(valid_count),
                str(missing_count),
            ]
        )


if __name__ == "__main__":
    unittest.main()
