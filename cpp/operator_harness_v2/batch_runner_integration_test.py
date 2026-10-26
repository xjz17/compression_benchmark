#!/usr/bin/env python3
"""Failure-isolation/resume contract for the native harness-v2 coordinator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = Path(
    os.environ.get("OPERATOR_HARNESS_V2_BUILD_DIR", str(HERE / "build-local"))
).expanduser().resolve()
DRIVER = BUILD / "operator_harness_v2_cli"
RUNNER = BUILD / "operator_harness_v2_batch_runner"
CATALOG = ROOT / "benchmark/reference/actions/operator_harness_batch_catalog_v2.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def a013_request(sample: Path, catalog_sha: str) -> dict:
    return {
        "schema": "operator-harness-request-v2",
        "command": "run",
        "request_id": "replaced-by-runner",
        "implementation_ids": ["A013"],
        "catalog_path": str(CATALOG),
        "catalog_sha256": catalog_sha,
        "input_path": str(sample),
        "initial_kind": "numeric",
        "initial_width": 8,
        "input_view": "int64",
        "dtype": "int64",
        "element_count": 5,
        "source_byte_width": 8,
        "original_numeric_payload": True,
        "recipe": {
            "policy_registry": "operator-harness-v2",
            "steps": [
                {
                    "kind": "stage",
                    "inputs": [0],
                    "stage": {
                        "action": (
                            "op.frame_of_reference.minimum."
                            "int64_uint64_le.cpp_v1"
                        ),
                        "profile": "int64",
                        "parameters": {},
                    },
                },
                {"kind": "stop"},
            ],
        },
        "warmups": 0,
        "repeats": 1,
    }


class BatchRunnerIntegrationTest(unittest.TestCase):
    def test_catalog_cannot_swap_an_excluded_row_into_the_eligible_97(self) -> None:
        self.assertTrue(DRIVER.is_file(), f"missing built driver: {DRIVER}")
        self.assertTrue(RUNNER.is_file(), f"missing built runner: {RUNNER}")
        with tempfile.TemporaryDirectory(prefix="operator-harness-v2-policy-") as tmp:
            directory = Path(tmp)
            sample = directory / "sample.bin"
            sample.write_bytes(struct.pack("<5q", -3, 0, 7, 7, 1))

            catalog = json.loads(CATALOG.read_text())
            rows = {row["implementation_id"]: row for row in catalog["operators"]}
            rows["A001"]["selection"]["selectable"] = False
            rows["A001"]["batch_id"] = None
            rows["A106"]["selection"]["selectable"] = True
            rows["A106"]["batch_id"] = "ordinary_source_pipelines"
            catalog.pop("catalog_sha256")
            canonical = json.dumps(
                catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
            catalog["catalog_sha256"] = hashlib.sha256(canonical).hexdigest()
            swapped_catalog = directory / "swapped-catalog.json"
            swapped_catalog.write_text(
                json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )

            request = a013_request(sample, sha256(swapped_catalog))
            request["catalog_path"] = str(swapped_catalog)
            plan = directory / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema": "operator-harness-batch-plan-v2",
                        "cases": [
                            {
                                "case_id": "swapped-exclusion-policy",
                                "implementation_id": "A013",
                                "batch_id": "integration",
                                "sample_id": "five-int64",
                                "sample_sha256": sha256(sample),
                                "request": request,
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
            output = directory / "results"
            run = subprocess.run(
                [str(RUNNER), str(DRIVER), str(plan), str(output), "30000", "2048"],
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            receipt = json.loads(next((output / "cases").glob("*/RESULT.json")).read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertIn(
                "nonselectable rows differ from A106-A110 policy",
                receipt["native_run"]["error"],
            )

    def test_good_bad_good_exclusions_and_resume(self) -> None:
        self.assertTrue(DRIVER.is_file(), f"missing built driver: {DRIVER}")
        self.assertTrue(RUNNER.is_file(), f"missing built runner: {RUNNER}")
        with tempfile.TemporaryDirectory(prefix="operator-harness-v2-batch-") as tmp:
            directory = Path(tmp)
            sample = directory / "sample.bin"
            sample.write_bytes(struct.pack("<5q", -3, 0, 7, 7, 1))
            sample_sha = sha256(sample)
            catalog_sha = sha256(CATALOG)

            def case(case_id: str, implementation_id: str, request: dict) -> dict:
                request = json.loads(json.dumps(request))
                request["implementation_ids"] = [implementation_id]
                return {
                    "case_id": case_id,
                    "implementation_id": implementation_id,
                    "batch_id": "integration",
                    "sample_id": "five-int64",
                    "sample_sha256": sample_sha,
                    "request": request,
                }

            good = a013_request(sample, catalog_sha)
            bad = json.loads(json.dumps(good))
            bad["recipe"]["steps"][0]["stage"]["action"] = (
                "op.zigzag.signed64.raw.cpp_v2"
            )
            cases = [
                case("01-good-before", "A013", good),
                case("02-bad-binding", "A013", bad),
                case("03-good-after", "A013", good),
            ]
            for index, implementation_id in enumerate(
                ("A106", "A107", "A108", "A109", "A110"), start=4
            ):
                cases.append(
                    case(f"{index:02d}-excluded-{implementation_id}", implementation_id, good)
                )

            plan = directory / "plan.json"
            plan.write_text(
                json.dumps(
                    {"schema": "operator-harness-batch-plan-v2", "cases": cases},
                    indent=2,
                )
                + "\n"
            )
            output = directory / "results"
            command = [
                str(RUNNER),
                str(DRIVER),
                str(plan),
                str(output),
                "30000",
                "2048",
            ]
            first = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary = json.loads((output / "SUMMARY.json").read_text())
            self.assertEqual(summary["case_count"], 8)
            self.assertEqual(summary["status_counts"], {"disabled": 5, "failed": 1, "ok": 2})
            self.assertEqual(
                [(item["case_id"], item["status"]) for item in summary["cases"]],
                [
                    ("01-good-before", "ok"),
                    ("02-bad-binding", "failed"),
                    ("03-good-after", "ok"),
                    ("04-excluded-A106", "disabled"),
                    ("05-excluded-A107", "disabled"),
                    ("06-excluded-A108", "disabled"),
                    ("07-excluded-A109", "disabled"),
                    ("08-excluded-A110", "disabled"),
                ],
            )

            receipts = sorted((output / "cases").glob("*/RESULT.json"))
            self.assertEqual(len(receipts), 8)
            receipt_bytes = {path: path.read_bytes() for path in receipts}
            records = [json.loads(value) for value in receipt_bytes.values()]
            failed = next(record for record in records if record["case_id"] == "02-bad-binding")
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["native_run"]["reason_code"], "REQUEST_OR_EXECUTION_FAILED")
            self.assertIn("catalog binding", failed["native_run"]["error"])
            for record in (item for item in records if item["status"] == "ok"):
                self.assertGreater(record["native_run_pid"], 0)
                self.assertGreater(record["native_decode_pid"], 0)
                self.assertNotEqual(
                    record["native_run_pid"], record["native_decode_pid"]
                )
                frame = Path(record["frame_path"])
                restored = Path(record["output_path"])
                self.assertEqual(sha256(frame), record["frame_sha256"])
                self.assertEqual(sha256(restored), sample_sha)

            second = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                json.loads((output / "SUMMARY.json").read_text())["status_counts"],
                {"disabled": 5, "failed": 1, "ok": 2},
            )
            self.assertEqual(
                {path: path.read_bytes() for path in receipts}, receipt_bytes
            )

    @unittest.skipUnless(platform.system() == "Linux", "Linux process-group contract")
    def test_timeout_kills_only_the_owned_child_process_group(self) -> None:
        self.assertTrue(RUNNER.is_file(), f"missing built runner: {RUNNER}")
        with tempfile.TemporaryDirectory(prefix="operator-harness-v2-timeout-") as tmp:
            directory = Path(tmp)
            sample = directory / "sample.bin"
            sample.write_bytes(struct.pack("<q", 1))
            descendant_pid_path = directory / "descendant.pid"
            driver = directory / "sleeping-driver.py"
            driver.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, subprocess, sys, time\n"
                "request=json.loads(sys.stdin.readline())\n"
                "child=subprocess.Popen(['sleep','30'])\n"
                "pathlib.Path(request['descendant_pid_path']).write_text(str(child.pid))\n"
                "time.sleep(30)\n"
            )
            driver.chmod(0o700)
            request = {
                "implementation_ids": ["A013"],
                "input_path": str(sample),
                "descendant_pid_path": str(descendant_pid_path),
            }
            plan = directory / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema": "operator-harness-batch-plan-v2",
                        "cases": [
                            {
                                "case_id": "timeout-owned-group",
                                "implementation_id": "A013",
                                "batch_id": "integration",
                                "sample_id": "one-int64",
                                "sample_sha256": sha256(sample),
                                "request": request,
                            }
                        ],
                    }
                )
                + "\n"
            )
            output = directory / "results"
            run = subprocess.run(
                [str(RUNNER), str(driver), str(plan), str(output), "200", "256"],
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads((output / "SUMMARY.json").read_text())
            self.assertEqual(summary["status_counts"], {"timeout": 1})
            self.assertTrue(descendant_pid_path.is_file())
            descendant_pid = int(descendant_pid_path.read_text())
            deadline = time.monotonic() + 2
            while Path(f"/proc/{descendant_pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(
                Path(f"/proc/{descendant_pid}").exists(),
                "descendant survived the owned process-group timeout kill",
            )


if __name__ == "__main__":
    unittest.main()
