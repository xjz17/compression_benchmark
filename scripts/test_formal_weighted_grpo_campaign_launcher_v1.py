#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import formal_weighted_grpo_campaign_launcher_v1 as launcher


class FormalCampaignLauncherTests(unittest.TestCase):
    def _spec(self, root: Path, *, prepared_count: int = 16) -> Path:
        repository = root / "repo"
        repository.mkdir()
        scripts = repository / "scripts"
        scripts.mkdir()
        campaign = root / "campaign"
        prepared = [str(root / f"prepared_{index:02d}")
                    for index in range(prepared_count)]
        true = str(Path("/usr/bin/true").resolve())

        def command(name: str, protocol: str) -> dict[str, object]:
            return {
                "protocol": protocol,
                "argv": [true],
                "cwd": str(repository),
                "expected_outputs": [str(campaign / (name + ".json"))],
            }

        document = {
            "schema": launcher.SPEC_SCHEMA,
            "repository_root": str(repository),
            "campaign_root": str(campaign),
            "prepared": prepared,
            "test_prepared": [str(root / "test_prepared")],
            "reference_table": str(root / "reference.json"),
            "gate_bundle": str(root / "gate_bundle.json"),
            "outputs": {
                "measurement_receipt": str(campaign / "seed_measurements.json"),
                "measurement_payload_dir": str(campaign / "seed_payloads"),
                "solution_library_manifest": str(campaign / "solution_library.json"),
                "training_output": str(campaign / "training"),
                "payload_audit_receipt": str(campaign / "payload_audit.json"),
                "final_detail_output": str(campaign / "formal_details.json"),
                "final_summary_output": str(campaign / "formal_summary.json"),
            },
            "task_factory": launcher.TASK_FACTORY,
            "max_measurement_workers": 4,
            "per_seed_budget_seconds": 57600,
            "seeds": [0, 1, 2],
            "weights": [list(row) for row in launcher.FORMAL_WEIGHTS],
            "commands": {
                "frozen_six_weight_test": command(
                    "test", "frozen_six_weight_test_v1"),
                "beam30_comparison": command(
                    "beam30", "beam30_shared_budget_v1"),
                "baseline86_native_comparison": command(
                    "baseline86", "native_baseline86_v1"),
                "ablations": [
                    {**command("ablation_no_pruning",
                               "no_target_pruning_ranking_v1"),
                     "id": "no_target_pruning_ranking"},
                    {**command("ablation_no_reuse", "no_block_reuse_v1"),
                     "id": "no_block_reuse"},
                ],
            },
        }
        path = root / "campaign_spec.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_spec_rejects_reduced_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._spec(Path(directory), prepared_count=15)
            with self.assertRaisesRegex(
                launcher.CampaignAdmissionError, "exactly 16"
            ):
                launcher.load_spec(path)

    def test_cuda_device_and_worker_count_reach_training_and_frozen_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._spec(Path(directory))
            document = json.loads(path.read_text())
            document.update(policy_device="cuda:0", policy_inference_device="cuda:0",
                            max_measurement_workers=2)
            path.write_text(json.dumps(document))
            spec = launcher.load_spec(path)
            stages = {stage.name: stage for stage in launcher._generated_stages(spec)}
            train = stages["three_seed_training"].argv
            self.assertEqual(train[train.index("--policy-device") + 1], "cuda:0")
            self.assertEqual(train[train.index("--measurement-workers") + 1], "2")
            env = dict(stages["frozen_six_weight_test"].environment)
            self.assertEqual(env[launcher.POLICY_INFERENCE_DEVICE_ENV], "cuda:0")
            document["policy_inference_device"] = "cuda:1"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(launcher.CampaignAdmissionError, "same single CUDA"):
                launcher.load_spec(path)

    def test_spec_freezes_budget_workers_weights_and_factory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root)
            document = json.loads(path.read_text())
            document["weights"][-1] = [0.3, 0.3, 0.4]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(
                launcher.CampaignAdmissionError, "exactly weights"
            ):
                launcher.load_spec(path)

    def test_plan_orders_gate_verification_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = launcher.load_spec(self._spec(Path(directory)))
            admission = {
                "dataset_count": 16,
                "corpus_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
            }
            plan = launcher.build_plan(spec, admission)
            self.assertEqual(plan["stage_order"][:5], [
                "prepared_data_verification",
                "formal_zlib6_calibration",
                "training_seed_measurement",
                "training_solution_library",
                "three_seed_training",
            ])
            self.assertEqual(plan["weights"], [
                list(row) for row in launcher.FORMAL_WEIGHTS
            ])
            self.assertEqual(plan["max_measurement_workers"], 4)
            self.assertEqual(plan["test_prepared_manifests"], [
                str(spec.test_prepared[0])
            ])
            self.assertEqual(plan["stage_order"][-2:], [
                "payload_artifact_audit", "final_result_aggregation",
            ])
            training_stage = next(
                row for row in plan["stages"]
                if row["name"] == "three_seed_training"
            )
            self.assertIn("--admission-bundle", training_stage["argv"])
            self.assertEqual(
                training_stage["environment"]
                ["WEIGHTED_GRPO_SOLUTION_LIBRARY_MANIFEST"],
                str(spec.solution_library_manifest),
            )
            self.assertEqual(
                plan["solution_library_manifest"],
                str(spec.solution_library_manifest),
            )
            for name in (
                "frozen_six_weight_test", "beam30_comparison",
                "ablations/no_target_pruning_ranking", "ablations/no_block_reuse",
            ):
                stage = next(row for row in plan["stages"] if row["name"] == name)
                self.assertEqual(
                    stage["environment"][launcher.SOLUTION_LIBRARY_ENV],
                    str(spec.solution_library_manifest),
                )

    def test_stage_resume_requires_unchanged_output_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            output = campaign / "result.txt"
            writer = root / "writer.py"
            writer.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "Path(sys.argv[1]).write_text('stable', encoding='utf-8')\n",
                encoding="utf-8",
            )
            stage = launcher.CommandStage(
                "sample", (str(Path(sys.executable).resolve()), str(writer), str(output)),
                root, (output,),
            )
            first = launcher._run_stage(
                stage, campaign_root=campaign, resume=False,
                max_measurement_workers=4,
            )
            self.assertEqual(first["status"], "complete")
            second = launcher._run_stage(
                stage, campaign_root=campaign, resume=True,
                max_measurement_workers=4,
            )
            self.assertEqual(second["status"], "resumed_complete")
            output.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(
                launcher.CampaignExecutionError, "changed after receipt"
            ):
                launcher._run_stage(
                    stage, campaign_root=campaign, resume=True,
                    max_measurement_workers=4,
                )

    def test_failed_stage_receipt_is_rerunnable_with_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            output = campaign / "result.txt"
            writer = root / "writer.py"
            writer.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "Path(sys.argv[1]).write_text('ok', encoding='utf-8')\n",
                encoding="utf-8",
            )
            stage = launcher.CommandStage(
                "retry", (str(Path(sys.executable).resolve()), str(writer), str(output)),
                root, (output,),
            )
            receipt = campaign / "stage_receipts" / "retry.json"
            launcher._atomic_json(receipt, {
                "schema": launcher.STAGE_RECEIPT_SCHEMA,
                "stage": "retry",
                "status": "failed",
                "stage_fingerprint": stage.fingerprint,
            })
            result = launcher._run_stage(
                stage, campaign_root=campaign, resume=True,
                max_measurement_workers=4,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(output.read_text(), "ok")

    def test_command_environment_rejects_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root)
            document = json.loads(path.read_text())
            document["commands"]["frozen_six_weight_test"]["environment"] = {
                "ACCESS_TOKEN": "not-allowed"
            }
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(
                launcher.CampaignAdmissionError, "credentials"
            ):
                launcher.load_spec(path)

    def test_command_environment_rejects_solution_library_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._spec(root)
            document = json.loads(path.read_text())
            document["commands"]["beam30_comparison"]["environment"] = {
                launcher.SOLUTION_LIBRARY_ENV: "/wrong/library.json"
            }
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(
                launcher.CampaignAdmissionError, "launcher-owned"
            ):
                launcher.load_spec(path)

    def test_resume_rejects_solution_library_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            library = root / "library.json"
            library.write_bytes(b"library-v1")
            output = campaign / "result.txt"
            writer = root / "writer.py"
            writer.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "Path(sys.argv[1]).write_text('stable', encoding='utf-8')\n",
                encoding="utf-8",
            )
            stage = launcher.CommandStage(
                "frozen_six_weight_test",
                (str(Path(sys.executable).resolve()), str(writer), str(output)),
                root, (output,),
                ((launcher.SOLUTION_LIBRARY_ENV, str(library.resolve())),),
            )
            first_binding = {
                "path": str(library.resolve()),
                "file_sha256": launcher._sha256_file(library),
                "action_registry_fingerprint": "registry-v1",
            }
            launcher._run_stage(
                stage, campaign_root=campaign, resume=False,
                max_measurement_workers=4,
                solution_library_binding=first_binding,
            )
            library.write_bytes(b"library-v2")
            changed_binding = {
                **first_binding,
                "file_sha256": launcher._sha256_file(library),
            }
            with self.assertRaisesRegex(
                launcher.CampaignExecutionError, "changed after receipt"
            ):
                launcher._run_stage(
                    stage, campaign_root=campaign, resume=True,
                    max_measurement_workers=4,
                    solution_library_binding=changed_binding,
                )

    def test_stage_rejects_conflicting_inherited_library_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library.json"
            library.write_bytes(b"library")
            output = root / "campaign" / "result.txt"
            stage = launcher.CommandStage(
                "beam30_comparison", (str(Path("/usr/bin/true").resolve()),),
                root, (output,),
                ((launcher.SOLUTION_LIBRARY_ENV, str(library.resolve())),),
            )
            binding = {
                "path": str(library.resolve()),
                "file_sha256": launcher._sha256_file(library),
                "action_registry_fingerprint": "registry-v1",
            }
            with mock.patch.dict(os.environ, {
                launcher.SOLUTION_LIBRARY_SHA256_ENV: "0" * 64,
            }, clear=False):
                with self.assertRaisesRegex(
                    launcher.CampaignExecutionError, "conflicting inherited"
                ):
                    launcher._run_stage(
                        stage, campaign_root=root / "campaign", resume=False,
                        max_measurement_workers=4,
                        solution_library_binding=binding,
                    )


if __name__ == "__main__":
    unittest.main()
