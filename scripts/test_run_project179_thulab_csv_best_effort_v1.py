#!/usr/bin/env python3
"""Tests for the isolated all-160 best-effort CSV attempt coordinator."""

from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_project179_thulab_csv_best_effort_v1 as coordinator  # noqa: E402


def _write(path: Path, payload: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


class BestEffortCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        for name in ("collections", "validations", "attempts"):
            (self.output / name).mkdir()
        self.receipts = self.root / "receipts"
        self.receipts.mkdir()
        self.mirror = self.root / "mirror"
        self.mirror.mkdir()
        self.scope = _write(self.root / "scope.json")
        self.policy = _write(self.root / "policy.json")
        self.exceptions = _write(self.root / "exceptions.json")
        self.exporter = _write(self.root / "exporter.py")
        self.validator = _write(self.root / "validator.py")
        self.lock_proof = _write(self.receipts / "LOCK_PROOF.json", "{}\n")
        self.config = coordinator.RunConfig(
            run_id="test-run",
            scope=self.scope,
            mirror=self.mirror,
            timestamp_policies=self.policy,
            row_exceptions=self.exceptions,
            output_root=self.output,
            receipts=self.receipts,
            lock_proof=self.lock_proof,
            python=Path(sys.executable),
            exporter_path=self.exporter,
            validator_path=self.validator,
            expected_exporter_sha256=coordinator._sha256(self.exporter),
            expected_validator_sha256=coordinator._sha256(self.validator),
            dataset_byte_cap=1_000_000_000,
            min_free_temp_bytes=100_000_000_000,
            temp_space_factor=64.0,
            batch_rows=128,
            insert_batch_rows=64,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def spec(dataset_id: str, index: int, source_bytes: int = 1) -> coordinator.DatasetSpec:
        return coordinator.DatasetSpec(
            dataset_id=dataset_id,
            repo_id=f"repo/{dataset_id}",
            revision="a" * 40,
            binding_sha256="b" * 64,
            source_file_count=1,
            source_bytes=source_bytes,
            source_files=((f"{dataset_id}.tsfile", "c" * 64),),
            attempt_index=index,
        )

    def test_commands_are_one_dataset_isolated_and_macro_is_audited(self) -> None:
        ordinary = self.spec("ordinary", 0)
        export = coordinator._export_command(self.config, ordinary)
        self.assertEqual(export[export.index("--dataset") + 1], "ordinary")
        self.assertEqual(export[export.index("--workers") + 1], "1")
        self.assertEqual(export[export.index("--temp-space-factor") + 1], "64")
        self.assertNotIn("--require-all-160", export)

        for dataset_id in sorted(coordinator.PILOT_RESCAN_DATASETS):
            validate = coordinator._validator_command(
                self.config, self.spec(dataset_id, 0)
            )
            self.assertIn("--schema-metadata-pass", validate)
            self.assertEqual(validate[validate.index("--workers") + 1], "1")
            self.assertEqual(
                validate[validate.index("--pilot-source-rescan") + 1], dataset_id
            )
            self.assertNotIn("--require-all-160", validate)

    def test_scheduler_continues_after_failure_and_returns_frozen_order(self) -> None:
        specs = [self.spec(f"d{index}", index) for index in range(6)]
        seen: list[str] = []
        active = 0
        maximum = 0
        guard = threading.Lock()

        def attempt(config, spec, reservation, registry):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
                seen.append(spec.dataset_id)
            time.sleep(0.01)
            with guard:
                active -= 1
            return {
                "dataset_id": spec.dataset_id,
                "status": "failed" if spec.dataset_id == "d2" else "success",
            }

        results = coordinator._schedule_attempts(
            self.config,
            specs,
            workers=3,
            registry=coordinator._ProcessRegistry(),
            disk_free=lambda _path: 10**15,
            attempt_worker=attempt,
        )
        self.assertEqual([item["dataset_id"] for item in results], [s.dataset_id for s in specs])
        self.assertEqual(sorted(seen), sorted(s.dataset_id for s in specs))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertGreater(maximum, 1)
        self.assertLessEqual(maximum, 3)
        self.assertEqual(results[2]["status"], "failed")

    def test_active_reservations_bound_concurrency(self) -> None:
        specs = [self.spec(f"d{index}", index) for index in range(3)]
        active = 0
        maximum = 0
        guard = threading.Lock()

        def attempt(config, spec, reservation, registry):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return {"dataset_id": spec.dataset_id, "status": "success"}

        coordinator._schedule_attempts(
            self.config,
            specs,
            workers=3,
            registry=coordinator._ProcessRegistry(),
            # 100 GB reserve + one 2 GB reservation fits; two do not.
            disk_free=lambda _path: 103_000_000_000,
            attempt_worker=attempt,
        )
        self.assertEqual(maximum, 1)

    def test_idle_disk_block_is_explicit_failure_without_launch(self) -> None:
        specs = [self.spec("d0", 0)]
        launched = []

        def attempt(*args):
            launched.append(True)
            raise AssertionError("unsafe launch")

        results = coordinator._schedule_attempts(
            self.config,
            specs,
            workers=1,
            registry=coordinator._ProcessRegistry(),
            disk_free=lambda _path: 100_000_000_000,
            attempt_worker=attempt,
        )
        self.assertFalse(launched)
        self.assertEqual(results[0]["stage"], "admission")
        self.assertEqual(
            results[0]["failure"]["error_type"],
            "insufficient_free_temporary_space",
        )
        self.assertEqual(set(results[0]["evidence"]), {
            "attempt", "export_stdout", "export_stderr",
            "validation_stdout", "validation_stderr",
        })

    def test_schema_preflight_failure_is_one_dataset_failure_not_global(self) -> None:
        specs = [self.spec("bad", 0), self.spec("good", 1)]
        launched: list[str] = []

        def attempt(config, spec, reservation, registry):
            launched.append(spec.dataset_id)
            return {"dataset_id": spec.dataset_id, "status": "success"}

        results = coordinator._schedule_attempts(
            self.config,
            specs,
            workers=2,
            registry=coordinator._ProcessRegistry(),
            preflight_failures={
                "bad": {
                    "dataset_id": "bad",
                    "error_type": "ValueError",
                    "error_message": "schema mismatch",
                }
            },
            disk_free=lambda _path: 10**15,
            attempt_worker=attempt,
        )
        self.assertEqual(launched, ["good"])
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["stage"], "export")
        self.assertEqual(results[0]["failure"]["error_type"], "schema_preflight_failed")
        self.assertEqual(results[1]["status"], "success")

    def test_one_dataset_success_publishes_collection_validation_and_five_refs(self) -> None:
        spec = self.spec("ordinary", 0)

        def run(command, *, stdout_path, stderr_path, dataset_id, stage, registry):
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            if stage == "export":
                collection = self.output / "collections" / dataset_id
                (collection / dataset_id).mkdir(parents=True)
                _write(
                    collection / "CSV_EXPORT_MANIFEST.json",
                    json.dumps({
                        "status": "complete", "dataset_count": 1,
                        "datasets": [{"dataset_id": dataset_id}],
                    }) + "\n",
                )
                _write(collection / dataset_id / "dataset_manifest.json", "{}\n")
            else:
                stdout_path.write_text(
                    json.dumps({
                        "status": "passed", "dataset_count": 1,
                        "datasets": [{"dataset_id": dataset_id}],
                        "pilot_source_rescans": [], "row_exception_audits": [],
                    }) + "\n",
                    encoding="utf-8",
                )
            return 0

        with mock.patch.object(coordinator, "_run_logged_command", side_effect=run):
            result = coordinator._attempt_dataset(
                self.config, spec, 2_000_000_000, coordinator._ProcessRegistry()
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stage"], "complete")
        self.assertEqual(result["collection"]["path"], "collections/ordinary")
        self.assertEqual(result["validation"]["path"], "validations/ordinary.json")
        self.assertEqual(set(result["evidence"]), {
            "attempt", "export_stdout", "export_stderr",
            "validation_stdout", "validation_stderr",
        })
        for reference in result["evidence"].values():
            self.assertEqual(coordinator._sha256(self.output / reference["path"]), reference["sha256"])

    def test_signal_returncode_is_preserved_as_signal_failure(self) -> None:
        spec = self.spec("crash", 0)

        def run(command, *, stdout_path, stderr_path, dataset_id, stage, registry):
            stdout_path.write_text(
                json.dumps({"event": "source_start", "source_path": "x.tsfile"}) + "\n",
                encoding="utf-8",
            )
            stderr_path.write_text("native crash\n", encoding="utf-8")
            return -11

        with mock.patch.object(coordinator, "_run_logged_command", side_effect=run):
            result = coordinator._attempt_dataset(
                self.config, spec, 2_000_000_000, coordinator._ProcessRegistry()
            )
        self.assertEqual(result["export_process"], {"returncode": -11, "signal": 11})
        self.assertEqual(result["failure"]["error_type"], "subprocess_signal")
        self.assertEqual(result["failure"]["last_source_event"]["source_path"], "x.tsfile")
        self.assertTrue(result["failure"]["evidence"])

    def test_validation_signal_never_inherits_export_source_attribution(self) -> None:
        spec = self.spec("validation-crash", 0)
        attempt_dir = self.output / "attempts" / spec.dataset_id
        attempt_dir.mkdir()
        result = coordinator._base_result(self.config, spec, 2_000_000_000)
        result.update(status="running", stage="validation", started_at_utc="now")
        _write(
            attempt_dir / "export.stdout.log",
            json.dumps({"event": "source_start", "source_path": "export.tsfile"}) + "\n",
        )
        _write(attempt_dir / "export.stderr.log", "")
        _write(attempt_dir / "validation.stdout.log", "")
        _write(attempt_dir / "validation.stderr.log", "native validation crash\n")
        failed = coordinator._failed_result(
            self.config,
            spec,
            result,
            stage="validation",
            returncode=-11,
            message=None,
        )
        self.assertEqual(failed["failure"]["native_crash_source"], "not_identified")
        self.assertEqual(
            failed["failure"]["known_source_files"],
            [{"path": "validation-crash.tsfile", "sha256": "c" * 64}],
        )

    def test_lock_proof_binds_live_parent_fd_and_writer_directory(self) -> None:
        mirror_lock = _write(self.mirror / ".sync.lock", "")
        writer_lock = self.root / "writer.lock"
        writer_lock.mkdir()
        proc_root = self.root / "proc"
        pid = os.getppid()
        fd_dir = proc_root / str(pid) / "fd"
        fdinfo_dir = proc_root / str(pid) / "fdinfo"
        fd_dir.mkdir(parents=True)
        fdinfo_dir.mkdir(parents=True)
        (fd_dir / "9").symlink_to(mirror_lock)
        fdinfo = "lock:\t1: FLOCK ADVISORY WRITE 0 00:00:0 0 EOF\n"
        _write(fdinfo_dir / "9", fdinfo)
        proof = {
            "schema": "project179-thulab-csv-best-effort-lock-proof-v1",
            "run_id": self.config.run_id,
            "parent_pid": pid,
            "writer_lock": str(writer_lock),
            "mirror_lock": str(mirror_lock),
            "fd_number": 9,
            "fd9_target": str(mirror_lock),
            "fd9_fdinfo": fdinfo,
            "created_at": "2026-10-01T00:00:00Z",
        }
        self.lock_proof.write_text(json.dumps(proof) + "\n", encoding="utf-8")
        result = coordinator._verify_lock_proof(self.config, proc_root=proc_root)
        self.assertEqual(result["parent_pid"], pid)
        self.assertEqual(result["sha256"], coordinator._sha256(self.lock_proof))

    def test_runtime_symlink_is_validated_without_resolving_invocation(self) -> None:
        runtime = self.root / "venv" / "bin" / "python"
        runtime.parent.mkdir(parents=True)
        runtime.symlink_to(Path(sys.executable))
        self.assertEqual(
            coordinator._require_executable(runtime, "runtime"),
            runtime,
        )

    def test_process_registry_terminates_only_live_children(self) -> None:
        live = mock.Mock()
        live.poll.return_value = None
        dead = mock.Mock()
        dead.poll.return_value = 0
        registry = coordinator._ProcessRegistry()
        registry.add("live", "export", live)
        registry.add("dead", "validation", dead)
        registry.terminate_all()
        live.terminate.assert_called_once_with()
        dead.terminate.assert_not_called()

    def test_real_subprocess_returncodes_cover_zero_nonzero_and_signal(self) -> None:
        cases = (
            ("import sys; sys.exit(0)", 0),
            ("import sys; sys.exit(7)", 7),
            (
                "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
                -15,
            ),
        )
        registry = coordinator._ProcessRegistry()
        for index, (program, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                stdout = self.root / f"real-{index}.stdout.log"
                stderr = self.root / f"real-{index}.stderr.log"
                returncode = coordinator._run_logged_command(
                    [sys.executable, "-c", program],
                    stdout_path=stdout,
                    stderr_path=stderr,
                    dataset_id=f"real-{index}",
                    stage="export",
                    registry=registry,
                )
                self.assertEqual(returncode, expected)
                self.assertTrue(stdout.is_file())
                self.assertTrue(stderr.is_file())
        self.assertEqual(registry._processes, {})

    def test_real_fake_exporter_then_validator_reaches_success(self) -> None:
        spec = self.spec("real-e2e", 0)
        fake_exporter = self.root / "fake_exporter.py"
        fake_validator = self.root / "fake_validator.py"
        fake_exporter.write_text(
            """\
import json
from pathlib import Path
import sys
output = Path(sys.argv[1])
dataset_id = sys.argv[2]
(output / dataset_id).mkdir(parents=True)
(output / 'CSV_EXPORT_MANIFEST.json').write_text(json.dumps({
    'status': 'complete', 'dataset_count': 1,
    'datasets': [{'dataset_id': dataset_id}],
}) + '\\n', encoding='utf-8')
(output / dataset_id / 'dataset_manifest.json').write_text('{}\\n', encoding='utf-8')
print(json.dumps({'status': 'complete', 'dataset_id': dataset_id}), flush=True)
""",
            encoding="utf-8",
        )
        fake_validator.write_text(
            """\
import json
import sys
dataset_id = sys.argv[1]
print(json.dumps({
    'status': 'passed', 'dataset_count': 1,
    'datasets': [{'dataset_id': dataset_id}],
    'pilot_source_rescans': [], 'row_exception_audits': [],
}), flush=True)
""",
            encoding="utf-8",
        )
        with (
            mock.patch.object(
                coordinator,
                "_export_command",
                return_value=[
                    sys.executable,
                    str(fake_exporter),
                    str(self.output / "collections" / spec.dataset_id),
                    spec.dataset_id,
                ],
            ),
            mock.patch.object(
                coordinator,
                "_validator_command",
                return_value=[sys.executable, str(fake_validator), spec.dataset_id],
            ),
        ):
            result = coordinator._attempt_dataset(
                self.config,
                spec,
                2_000_000_000,
                coordinator._ProcessRegistry(),
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["export_process"], {"returncode": 0, "signal": None})
        self.assertEqual(result["validation_process"], {"returncode": 0, "signal": None})
        self.assertEqual(result["validation"]["path"], "validations/real-e2e.json")
        self.assertIn('"status": "passed"', (
            self.output / result["evidence"]["validation_stdout"]["path"]
        ).read_text(encoding="utf-8"))

    def test_global_loader_never_enters_native_schema_sdk(self) -> None:
        source = inspect.getsource(coordinator._load_frozen_specs)
        self.assertNotIn("_load_tsfile_module", source)
        self.assertNotIn("_preflight_bindings", source)
        self.assertIn("bind_mirror", source)
        _write(self.mirror / coordinator.RAW_MANIFEST_NAME, "{}\n")
        rows = tuple(
            SimpleNamespace(
                dataset_id=f"d{index:03d}",
                repo_id=f"repo/d{index:03d}",
                revision="a" * 40,
            )
            for index in range(160)
        )
        bindings = tuple(
            SimpleNamespace(
                row=row,
                binding_sha256="b" * 64,
                source_files=tuple(
                    (
                        f"{row.dataset_id}/{source_index}.tsfile",
                        self.mirror / "x",
                        1,
                        "c" * 64,
                    )
                    for source_index in range(1365 if index == 0 else 1)
                ),
            )
            for index, row in enumerate(rows)
        )
        with (
            mock.patch.object(coordinator, "_verify_payload_pins"),
            mock.patch.object(
                coordinator.exporter.policy_core,
                "load_scope",
                return_value=({"scope_id": "scope"}, rows, "1" * 64),
            ),
            mock.patch.object(
                coordinator.exporter.policy_core,
                "load_policies",
                return_value=({}, "2" * 64),
            ),
            mock.patch.object(coordinator.exporter, "_check_frozen_policy_closure"),
            mock.patch.object(
                coordinator.exporter,
                "_load_row_exception_policy",
                return_value=SimpleNamespace(file_sha256="3" * 64),
            ),
            mock.patch.object(
                coordinator.exporter.policy_core,
                "bind_mirror",
                return_value=bindings,
            ),
            mock.patch.object(coordinator.exporter, "_bind_row_exception_to_bindings"),
            mock.patch.object(
                coordinator.exporter,
                "_load_tsfile_module",
                side_effect=AssertionError("native SDK entered coordinator"),
            ) as load_native,
            mock.patch.object(
                coordinator.exporter,
                "_preflight_bindings",
                side_effect=AssertionError("native schema preflight entered coordinator"),
            ) as preflight,
        ):
            specs, contract, failures = coordinator._load_frozen_specs(self.config)
        self.assertEqual(len(specs), 160)
        self.assertEqual(contract["source_count"], 1524)
        self.assertEqual(failures, {})
        load_native.assert_not_called()
        preflight.assert_not_called()

    def test_existing_attempt_directory_is_never_reused(self) -> None:
        spec = self.spec("already", 0)
        (self.output / "attempts" / spec.dataset_id).mkdir()
        with mock.patch.object(coordinator, "_run_logged_command") as run:
            with self.assertRaises(FileExistsError):
                coordinator._attempt_dataset(
                    self.config, spec, 2_000_000_000, coordinator._ProcessRegistry()
                )
        run.assert_not_called()

    def test_final_document_is_exact160_and_lists_failures_without_success_claim(self) -> None:
        specs = [self.spec(f"d{index:03d}", index) for index in range(160)]
        results = []
        for spec in specs:
            if spec.attempt_index == 7:
                results.append({
                    "dataset_id": spec.dataset_id,
                    "status": "failed",
                    "stage": "validation",
                    "failure": {
                        "error_type": "subprocess_nonzero",
                        "error_message": "bad row",
                        "evidence": [{"path": "x", "sha256": "a" * 64, "bytes": 1}],
                    },
                })
            else:
                results.append({
                    "dataset_id": spec.dataset_id,
                    "status": "success",
                    "stage": "complete",
                    "failure": None,
                })
        final = coordinator._build_final_document(
            self.config,
            specs,
            {
                "scope_sha256": "1" * 64,
                "timestamp_policy_file_sha256": "2" * 64,
                "row_exception_policy_file_sha256": "3" * 64,
                "raw_manifest_sha256": "4" * 64,
                "source_count": 1524,
            },
            {"sha256": "5" * 64},
            results,
            workers=10,
        )
        self.assertEqual(final["attempted_dataset_count"], 160)
        self.assertEqual(final["success_count"], 159)
        self.assertEqual(final["failure_count"], 1)
        self.assertEqual(final["failure_dataset_ids"], ["d007"])
        self.assertEqual(final["failures"][0]["error_message"], "bad row")
        self.assertEqual(
            final["completion_claim"],
            "exactly_once_attempts_not_formal_160_success",
        )


if __name__ == "__main__":
    unittest.main()
