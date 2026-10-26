#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "source" / "runner.py"
BINARY = REPO / "cpp/native_baseline86_v1/build-v4-parity/native_baseline86_cli"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunnerIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(BINARY.exists(), "local frozen baseline CLI is absent")
    def test_zstd_real_frame_and_cold_carrier_inverse(self) -> None:
        with tempfile.TemporaryDirectory(prefix="joint-runner-test-") as temp:
            directory = Path(temp)
            o2r1 = directory / "block.o2r1"
            payload = b"O2R1" + bytes(range(251)) * 7
            o2r1.write_bytes(payload)
            source_sha = hashlib.sha256(b"source-column-fixture").hexdigest()
            inverse = directory / "operator-inverse.json"
            inverse.write_text(json.dumps({
                "status": "passed",
                "bit_exact": True,
                "applicable": True,
                "strict": True,
                "fallback_used": False,
                "operator_id": "A013",
                "source_column_sha256": source_sha,
                "o2r1_sha256": hashlib.sha256(payload).hexdigest(),
            }))
            output = directory / "result"
            command = [
                sys.executable, str(RUNNER), "measure-one",
                "--algorithm", "Zstd",
                "--binary", str(BINARY),
                "--expected-binary-sha256", digest(BINARY),
                "--algorithm-profile", "level3_single_thread",
                "--operator-id", "A013",
                "--operator-catalog-sha256", "a" * 64,
                "--operator-inverse-receipt", str(inverse),
                "--o2r1", str(o2r1),
                "--source-column-sha256", source_sha,
                "--source-numeric-bytes", "4096",
                "--dataset-id", "fixture",
                "--column-id", "fixture/value",
                "--output", str(output),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "RESULT.json").read_text())
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["bit_exact"])
            self.assertEqual(result["binding"]["carrier_kind"], "direct_u8")
            self.assertGreater(result["joint_complete_frame_bytes"], 0)
            self.assertEqual(
                len(result["algorithm_measurement"]["compression_ns_samples"]), 3
            )
            self.assertEqual(
                len(result["algorithm_measurement"]["decompression_ns_samples"]), 3
            )


if __name__ == "__main__":
    unittest.main()
