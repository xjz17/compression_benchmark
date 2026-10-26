#!/usr/bin/env python3
"""Fail-closed tests for complete-schema and DMI compatibility launch proof."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import gate_project179_thulab_csv_preflight_v1 as gate


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode()


class GateTests(unittest.TestCase):
    def schema_fixture(self):
        ids = [gate.DMI, *[f"fixture-{i}" for i in range(159)]]
        rows = []
        for i, dataset_id in enumerate(ids):
            if i == 0:
                sources = [{"source_path": p, "source_sha256": s["sha256"],
                            "schema_sha256": s["schema_sha256"], "source_action": "materialize",
                            "table_name": "dmi"} for p, s in gate.DMI_SOURCES.items()]
            else:
                sources = [{"source_path": f"{j}.tsfile", "source_sha256": "a" * 64,
                            "schema_sha256": "b" * 64, "source_action": "materialize",
                            "table_name": "fixture"} for j in range(1364 if i == 1 else 1)]
            rows.append({"dataset_id": dataset_id, "status": "passed", "binding_sha256": "c" * 64,
                         "source_file_count": len(sources), "sources": sources,
                         "schema_sha256": hashlib.sha256(encode(sources)[:-1]).hexdigest()})
        return ids, {"status": "passed", "error_count": 0, "errors": [], "dataset_count": 160,
                     "source_count": 1524, "dataset_ids": ids, "datasets": rows,
                     "scope_sha256": "1" * 64, "timestamp_policy_file_sha256": "2" * 64,
                     "exporter_sha256": "3" * 64, "tsfile_version": "2.1.5"}

    def test_exact_full_schema_closure(self):
        ids, report = self.schema_fixture()
        gate.check_schema(report, ids, "1" * 64, "2" * 64, "3" * 64)

    def test_schema_errors_missing_source_hashes_and_wrong_dmi_fail(self):
        ids, report = self.schema_fixture()
        mutations = [lambda r: r.update(status="failed"), lambda r: r.update(error_count=1),
                     lambda r: r.update(source_count=1523), lambda r: r.update(tsfile_version="2.1.4"),
                     lambda r: r["datasets"].pop(),
                     lambda r: r["datasets"][0]["sources"][0].update(source_sha256="d" * 64),
                     lambda r: r["datasets"][1].update(binding_sha256="bad"),
                     lambda r: r.update(exporter_sha256="4" * 64)]
        for mutate in mutations:
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                gate.check_schema(changed, ids, "1" * 64, "2" * 64, "3" * 64)

    def dmi_fixture(self, root):
        directory = root / gate.DMI
        directory.mkdir()
        files = []
        for name in ("data.csv", "series_map.csv", "columns.csv"):
            path = directory / name
            path.write_bytes((
                "column,table_name,source_name,source_type,output_dtype,conversion\n"
                "c0000,dmi_aarhus_data,reference_time,INT64,int64,policy_source_bound_integer_epoch_to_int64_us\n"
                "c0001,dmi_aarhus_training_matrix,reference_time,STRING,int64,policy_temporal_string_to_int64_us\n"
            ).encode() if name == "columns.csv" else b"fixture\n")
            files.append({"path": name, "bytes_decimal_20": f"{path.stat().st_size:020d}",
                          "sha256": gate.digest(path)})
        files.append({"path": "dataset_manifest.json", "bytes_decimal_20": "0" * 20, "sha256": None})
        receipt = {"status": "complete", "dataset_id": gate.DMI, "scope_sha256": "1" * 64,
                   "timestamp_policy_file_sha256": "2" * 64, "exporter_sha256": "3" * 64,
                   "emitted_rows_decimal_20": f"{43910:020d}",
                   "lost_rows_due_to_byte_cap_decimal_20": "0" * 20,
                   "row_exception_filtered_rows": 0, "files": files,
                   "source_tsfiles": [{"path": p, "sha256": s["sha256"]}
                                      for p, s in gate.DMI_SOURCES.items()]}
        path = directory / "dataset_manifest.json"
        files[-1]["bytes_decimal_20"] = f"{len(encode(receipt)):020d}"
        path.write_bytes(encode(receipt))
        report = {"status": "passed", "dataset_count": 1, "datasets": [{"dataset_id": gate.DMI}],
                  "pilot_source_rescans": [{"dataset_id": gate.DMI, "status": "passed",
                                           "source_files": 2, "source_rows": 43910,
                                           "compared_output_rows": 43910,
                                           "temporal_unit_audits": [{
                                               "source_path": p, "source_sha256": s["sha256"],
                                               "source_schema_sha256": s["schema_sha256"],
                                               "source_rows": s["rows"], "reference_time_source_type": s["source_type"],
                                               "reference_time_source_category": s["source_category"],
                                               "conversion": s["conversion"], "reference_time_nulls": 0,
                                               "lead_time_hours_nulls": 0, "int64_us_overflow_count": 0,
                                               "lead_relation_mismatch_count": 0,
                                               **{k: s[k] for k in ("reference_epoch_s_min", "reference_epoch_s_max",
                                                                   "time_epoch_s_min", "time_epoch_s_max")}
                                           } for p, s in gate.DMI_SOURCES.items()]}]}
        return report, receipt

    def test_dmi_complete_rows_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report, _ = self.dmi_fixture(root)
            evidence = gate.check_dmi(report, root, "1" * 64, "2" * 64, "3" * 64)
            self.assertEqual(evidence["output_rows"], 43910)
            self.assertEqual(len(evidence["output_hashes"]), 4)
            (root / gate.DMI / "data.csv").write_bytes(b"tampered\n")
            with self.assertRaises(ValueError):
                gate.check_dmi(report, root, "1" * 64, "2" * 64, "3" * 64)

    def test_dmi_partial_scan_or_new_row_loss_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report, receipt = self.dmi_fixture(root)
            changed = copy.deepcopy(report)
            changed["pilot_source_rescans"][0]["compared_output_rows"] -= 1
            with self.assertRaises(ValueError):
                gate.check_dmi(changed, root, "1" * 64, "2" * 64, "3" * 64)
            receipt["row_exception_filtered_rows"] = 1
            (root / gate.DMI / "dataset_manifest.json").write_bytes(encode(receipt))
            with self.assertRaises(ValueError):
                gate.check_dmi(report, root, "1" * 64, "2" * 64, "3" * 64)

    def test_no_symlink_input(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source"
            path.write_bytes(b"{}"); link = Path(temp) / "link"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                gate.load(link)

    def test_dmi_unit_audit_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report, _ = self.dmi_fixture(root)
            for key, value in (("reference_time_nulls", 1), ("int64_us_overflow_count", 1),
                               ("lead_relation_mismatch_count", 1), ("reference_epoch_s_max", 0),
                               ("conversion", "identity")):
                changed = copy.deepcopy(report)
                changed["pilot_source_rescans"][0]["temporal_unit_audits"][0][key] = value
                with self.assertRaises(ValueError):
                    gate.check_dmi(changed, root, "1" * 64, "2" * 64, "3" * 64)


if __name__ == "__main__":
    unittest.main()
