#!/usr/bin/env python3
"""Recovery ordering tests for the native smallest32 campaign."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_native_smallest32_campaign_v1 as campaign


class CampaignRecoveryTests(unittest.TestCase):
    def running_state(self,root: Path,seed: int=0) -> None:
        directory=root/"seed_supervision"
        directory.mkdir(parents=True)
        (directory/f"seed_{seed}.json").write_text(json.dumps({"status":"running"}))

    def test_owned_recovery_precedes_strict_fresh_gpu_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.running_state(root)
            events=[]
            argv=["python","supervisor.py","--seed","0","--","python","train.py"]
            def command(phase,values):
                events.append(("recover",phase,list(values)))
            def snapshot(gpu_uuid,workers):
                events.append(("snapshot",gpu_uuid,workers))
                return {"gpu_uuid":gpu_uuid,"workers":workers}
            with mock.patch.object(campaign,"resource_snapshot",side_effect=snapshot):
                result=campaign.recover_owned_orphan(
                    root,0,argv,gpu_uuid="GPU-test",workers=2,command=command)
            self.assertEqual([event[0] for event in events],["recover","snapshot"])
            recovery=events[0][2]
            self.assertLess(recovery.index("--recover-only"),recovery.index("--"))
            self.assertEqual(result,{"gpu_uuid":"GPU-test","workers":2})
            self.assertTrue((root/"seed_0_resource_after_recovery.json").is_file())

    def test_unknown_gpu_process_rejection_is_not_relaxed_after_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.running_state(root)
            commands=[]
            argv=["python","supervisor.py","--","python","train.py"]
            with mock.patch.object(
                campaign,"resource_snapshot",side_effect=ValueError("selected GPU has a compute process")):
                with self.assertRaisesRegex(ValueError,"compute process"):
                    campaign.recover_owned_orphan(
                        root,0,argv,gpu_uuid="GPU-test",workers=2,
                        command=lambda phase,values: commands.append((phase,values)))
            self.assertEqual(len(commands),1)
            self.assertIn("--recover-only",commands[0][1])
            self.assertFalse((root/"seed_0_resource_after_recovery.json").exists())

    def test_nonrunning_state_never_invokes_recovery_or_gpu_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            state=root/"seed_supervision";state.mkdir()
            (state/"seed_0.json").write_text(json.dumps({"status":"complete"}))
            with mock.patch.object(campaign,"resource_snapshot") as snapshot:
                result=campaign.recover_owned_orphan(
                    root,0,["python","supervisor.py","--","python","train.py"],
                    gpu_uuid="GPU-test",workers=2,
                    command=lambda *_: self.fail("unexpected recovery command"))
            self.assertIsNone(result);snapshot.assert_not_called()


if __name__=="__main__":
    unittest.main()
