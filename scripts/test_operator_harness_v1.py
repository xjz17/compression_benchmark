"""Runner durability tests plus opt-in real C++ graph integration.

The tiny fake driver is a protocol test double; its numbers are never native
performance evidence. Real integration uses OPERATOR_HARNESS_DRIVER.
"""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import operator_harness_v1 as harness
from operator_harness_suite_v1 import representative_suite


FAKE_DRIVER = r'''
import json,sys
for line in sys.stdin:
    req=json.loads(line)
    out={"schema":"operator-harness-response-v1","command":req["command"],"case_id":req["case_id"]}
    if req["command"]=="run":
        if req["case_id"].startswith("native_failure"):
            out.update(status="failed",error="deliberate native failure")
        else:
            frame=json.dumps(req["roots"],sort_keys=True).encode()
            out.update(status="ok",bitexact=True,frame_hex=frame.hex(),frame_bytes=len(frame),
              original_numeric_bytes=sum(len(bytes.fromhex(r["data_hex"])) for r in req["roots"] if r["original_numeric_payload"]),
              encode_ns=[123]*req["repeats"],decode_ns=[456]*req["repeats"],
              timing_boundary="TEST DOUBLE ONLY",dispatcher="fake_protocol_test")
    else:
        roots=json.loads(bytes.fromhex(req["frame_hex"]))
        out.update(status="ok",roots=roots,decode_ns=[42],
          original_numeric_bytes=sum(len(bytes.fromhex(r["data_hex"])) for r in roots if r["original_numeric_payload"]))
    print(json.dumps(out),flush=True)
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.driver = self.root / "fake_driver"
        self.driver.write_text("#!" + sys.executable + "\n" + FAKE_DRIVER)
        self.driver.chmod(0o755)
        self.suite = representative_suite()
        self.suite["cases"] = self.suite["cases"][:2]

    def tearDown(self):
        self.temporary.cleanup()

    def run_suite(self, suite=None, output=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return harness.run_suite(suite or self.suite, self.driver,
                                     output or self.root / "results", repeats=2, **kwargs)

    def test_independent_success_failure_success_and_durable_resume(self):
        good1, good2 = copy.deepcopy(self.suite["cases"])
        failure = copy.deepcopy(good1)
        failure["case_id"] = "native_failure_between_successes"
        suite = {**self.suite, "cases": [good1, failure, good2]}
        summary = self.run_suite(suite)
        self.assertEqual(summary["status_counts"], {"ok": 2, "failed": 1})
        self.assertFalse(summary["all_cases_passed"])
        receipts = [Path(row["result_path"]) for row in summary["results"]]
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in receipts]
        event_bytes = (self.root / "results/events.jsonl").read_bytes()
        resumed = self.run_suite(suite)
        self.assertEqual(summary, resumed)
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns) for path in receipts])
        self.assertEqual(event_bytes, (self.root / "results/events.jsonl").read_bytes())
        self.assertTrue(json.loads(receipts[0].read_text())["cold_decode_verified"])
        operator_receipts = sorted((self.root / "results/operators").glob("A*/*.json"))
        self.assertEqual(len(operator_receipts), 3)
        for path in operator_receipts:
            record = json.loads(path.read_text())
            self.assertTrue(record["independent_single_operator_case"])
            self.assertEqual(record["timing_scope"], "complete fixed-recipe graph; not attributed per-kernel time")

    def test_resume_repairs_only_missing_operator_evidence(self):
        summary = self.run_suite()
        case_receipts = [Path(row["result_path"]) for row in summary["results"]]
        before = [path.read_bytes() for path in case_receipts]
        evidence = next((self.root / "results/operators").glob("A*/*.json"))
        evidence_bytes = evidence.read_bytes()
        evidence.unlink()
        self.run_suite()
        self.assertEqual(evidence.read_bytes(), evidence_bytes)
        self.assertEqual([path.read_bytes() for path in case_receipts], before)

    def test_changed_run_contract_cannot_overwrite_existing_results(self):
        summary = self.run_suite()
        first = Path(summary["results"][0]["result_path"])
        original = first.read_bytes()
        changed = copy.deepcopy(self.suite)
        changed["cases"][0]["roots"][0]["data_hex"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "immutable run identity differs"):
            self.run_suite(changed)
        self.assertEqual(original, first.read_bytes())

    def test_resume_rejects_changed_frame_and_changed_receipt(self):
        summary = self.run_suite()
        receipt = Path(summary["results"][0]["result_path"])
        frame = receipt.parent / "frame.ngf"
        original_frame = frame.read_bytes()
        frame.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "frame missing or changed"):
            self.run_suite()
        frame.write_bytes(original_frame)
        record = json.loads(receipt.read_text())
        record["metrics"]["complete_frame_bytes"] += 1
        receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            self.run_suite()

    def test_nncp_is_disabled_before_any_process_is_started(self):
        disabled = copy.deepcopy(self.suite)
        disabled["cases"] = [disabled["cases"][0]]
        disabled["cases"][0]["implementation_ids"] = ["A110"]
        disabled["cases"][0]["enable_nncp"] = True
        with patch.object(harness, "NativeDriver", side_effect=AssertionError("must not launch")):
            summary = self.run_suite(disabled)
        self.assertEqual(summary["status_counts"], {"disabled": 1})
        self.assertEqual(summary["independently_verified_implementation_ids"], [])

    def test_transport_deadline_stops_only_its_local_process(self):
        sleeping = self.root / "sleeping_driver"
        sleeping.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(20)\n")
        sleeping.chmod(0o755)
        with harness.NativeDriver(sleeping, 0.05, self.root / "stderr.log") as driver:
            with self.assertRaises(TimeoutError):
                driver.request({"schema": harness.REQUEST_SCHEMA, "command": "run", "case_id": "deadline"})
            process = driver.process
        self.assertIsNotNone(process.poll())

    def test_malformed_root_is_its_own_failure_and_sibling_commits(self):
        bad = copy.deepcopy(self.suite["cases"][0])
        bad["case_id"] = "bad_alignment"
        bad["roots"][0]["data_hex"] = "01"
        summary = self.run_suite({**self.suite, "cases": [bad, self.suite["cases"][1]]})
        self.assertEqual(summary["status_counts"], {"failed": 1, "ok": 1})

    def test_malformed_or_unknown_ids_cannot_block_siblings_or_create_fake_ids(self):
        malformed = copy.deepcopy(self.suite["cases"][0])
        malformed.update(case_id="null_ids", implementation_ids=None)
        unknown = copy.deepcopy(malformed)
        unknown.update(case_id="unknown_ids", implementation_ids=["A999"])
        summary = self.run_suite({**self.suite, "cases": [malformed, unknown, self.suite["cases"][1]]})
        self.assertEqual(summary["status_counts"], {"failed": 2, "ok": 1})
        self.assertFalse((self.root / "results/operators/A999").exists())
        with redirect_stdout(io.StringIO()):
            resumed = self.run_suite({**self.suite, "cases": [malformed, unknown, self.suite["cases"][1]]})
        self.assertEqual(summary, resumed)

    def test_duplicate_case_ids_rejected_before_run_directory(self):
        cases = [self.suite["cases"][0], self.suite["cases"][0]]
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            self.run_suite({**self.suite, "cases": cases})
        self.assertFalse((self.root / "results/RUN.json").exists())

    def test_output_refuses_existing_campaign_or_workspace_directory(self):
        existing = self.root / "unrelated_campaign"
        existing.mkdir()
        marker = existing / "old_result.json"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "nonempty directory"):
            self.run_suite(output=existing)
        self.assertEqual(marker.read_text(), "preserve me")
        self.assertEqual({entry.name for entry in existing.iterdir()}, {"old_result.json"})
        with self.assertRaisesRegex(ValueError, "workspace or its ancestor"):
            self.run_suite(output=harness.ROOT)

    def test_timing_arrays_and_byte_accounting_must_match(self):
        request = {"repeats": 2, "roots": [{"data_hex": "00", "original_numeric_payload": True}]}
        response = {"status": "ok", "bitexact": True, "frame_hex": "01", "frame_bytes": 2,
                    "original_numeric_bytes": 1, "timing_boundary": "test", "encode_ns": [1, 2], "decode_ns": [3, 4]}
        with self.assertRaisesRegex(harness.DriverError, "accounting"):
            harness._verify_run(response, request)
        response["frame_bytes"] = 1
        response["encode_ns"] = [1]
        with self.assertRaisesRegex(harness.DriverError, "encode"):
            harness._verify_run(response, request)


@unittest.skipUnless(os.environ.get("OPERATOR_HARNESS_DRIVER"), "set OPERATOR_HARNESS_DRIVER for real native integration")
class NativeIntegrationTests(unittest.TestCase):
    def test_representative_graphs_cold_decode_and_native_samples(self):
        binary = Path(os.environ["OPERATOR_HARNESS_DRIVER"])
        with tempfile.TemporaryDirectory() as temporary:
            with redirect_stdout(io.StringIO()):
                summary = harness.run_suite(representative_suite(), binary, Path(temporary), warmups=0, repeats=2)
            self.assertTrue(summary["all_cases_passed"], summary)
            self.assertEqual(summary["case_count"], 30)
            self.assertEqual(summary["independently_verified_implementation_ids"],
                             ["A013", "A020", "A023", "A031", "A046", "A048", "A077", "A080", "A111"])
            for row in summary["results"]:
                receipt = json.loads(Path(row["result_path"]).read_text())
                self.assertTrue(receipt["cold_decode_verified"])
                self.assertEqual(len(receipt["metrics"]["encode"]["samples_ns"]), 2)


if __name__ == "__main__":
    unittest.main()
