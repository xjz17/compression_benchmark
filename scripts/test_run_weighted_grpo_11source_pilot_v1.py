#!/usr/bin/env python3
"""Boundaries and process lifecycle tests; no native mechanism probes required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import run_weighted_grpo_11source_pilot_v1 as pilot
import weight_conditioned_grpo_v1 as grpo


def echo(value):
    return value


def fail():
    raise ValueError("expected worker failure")


def descendant_hang(path):
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    Path(path).write_text(str(child.pid))
    time.sleep(30)


def training_hang(path, deadline):
    os.setsid()
    pilot.run_isolated(descendant_hang, (path,), 30)


def running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class PilotBoundaryTests(unittest.TestCase):
    def test_isolated_success_and_failure(self):
        self.assertEqual(pilot.run_isolated(echo, ({"x": 3},), 1), ("ok", {"x": 3}))
        status, result = pilot.run_isolated(fail, (), 1)
        self.assertEqual(status, "error")
        self.assertEqual(result["type"], "ValueError")

    def test_group_timeout_kills_native_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"pid"
            status, _ = pilot.run_isolated(descendant_hang, (str(path),), .3)
            self.assertEqual(status, "timeout")
            self.assertFalse(running(int(path.read_text())))

    @unittest.skipUnless(sys.platform == "linux", "production deadline uses Linux /proc")
    def test_hard_training_deadline_kills_nested_process_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"pid"
            result = pilot.supervise_training(training_hang, (str(path),), .4)
            self.assertTrue(result["deadline_enforced"])
            self.assertLess(result["elapsed_seconds"], 2)
            self.assertFalse(running(int(path.read_text())))

    def test_full_block_and_bitmap_hashes_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values=np.arange(128,dtype="<i8").tobytes()
            bitmap=bytes(16)
            (root/"values").write_bytes(values)
            (root/"bitmap").write_bytes(bitmap)
            record={"manifest_path":str(root/"dataset_manifest.json"), "block":{
                "values_path":"values","missing_bitmap_path":"bitmap",
                "values_sha256":pilot.sha(values),"missing_bitmap_sha256":pilot.sha(bitmap),
                "original_numeric_bytes":len(values),"prepared_block_bytes":len(values)+len(bitmap)}}
            self.assertEqual(pilot.read_block(record),(values,bitmap))
            (root/"values").write_bytes(values[:512])
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                pilot.read_block(record)

    def test_seed_selection_uses_training_measurements_and_weights(self):
        record={"block":{"dtype":"int64","prepared_block_bytes":1024},"E0":1.,"D0":1.}
        library={"measurements":[
            {"dtype":"int64","block_size_bucket_bytes":1024,"rho":.1,"E_ns_per_byte":10.,"D_ns_per_byte":10.,"recipe_action_ids":[92,100]},
            {"dtype":"int64","block_size_bucket_bytes":1024,"rho":1.,"E_ns_per_byte":.1,"D_ns_per_byte":.1,"recipe_action_ids":[]}]}
        self.assertEqual(pilot.seed_recipe(record,(1,0,0),library),(92,100))
        self.assertEqual(pilot.seed_recipe(record,(0,1,0),library),())

    def test_persistent_adam_survives_two_updates_and_checkpoint(self):
        identity=grpo.PolicyIdentity("test",(("x","hash"),),(("x","root"),),2,0,2,2,(4,4))
        policy=grpo.TwoHiddenMLP(identity,np.random.default_rng(3))
        config=grpo.GRPOConfig(hidden_sizes=(4,4),groups_per_iteration=2)
        optimizer=grpo.Adam(policy.params,config.learning_rate)
        episodes=[]
        for i in range(16):
            episodes.append({"features":np.ones((1,7)),"legal":np.ones((1,2),dtype=bool),
                             "actions":np.asarray([i%2]),"old_logp":np.asarray([-np.log(2)]),"return":float(i%8)})
        batch=pilot.pack_batch(episodes)
        before=pilot.parameter_hash(policy)
        grpo._update(policy,optimizer,batch,config)
        first=[v.copy() for v in optimizer.first]
        grpo._update(policy,optimizer,batch,config)
        self.assertEqual(optimizer.steps,8)
        self.assertNotEqual(before,pilot.parameter_hash(policy))
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(first,optimizer.first)))
        with tempfile.TemporaryDirectory() as tmp:
            pilot._checkpoint(Path(tmp),policy,optimizer,config,3,2,"final_policy.json",{})
            loaded=grpo.load_checkpoint(Path(tmp)/"final_policy.json")
            self.assertEqual(loaded.completed_iterations,2)
            self.assertEqual(int(np.load(Path(tmp)/"optimizer.npz")["steps"]),8)

    def test_advantages_do_not_mix_different_weight_groups(self):
        groups=[]
        for reward in list(range(8))+list(range(100,108)):
            groups.append({"features":np.ones((1,7)),"legal":np.ones((1,2),dtype=bool),
                           "actions":np.asarray([0]),"old_logp":np.asarray([0.]),"return":reward})
        batch=pilot.pack_batch(groups)
        np.testing.assert_allclose(batch["advantages"][:8],batch["advantages"][8:])

    def test_frozen_document_hash_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"doc.json"
            path.write_text("{}")
            pilot.checked_document({"path":str(path),"sha256":pilot.sha(b"{}")})
            with self.assertRaisesRegex(ValueError,"changed"):
                pilot.checked_document({"path":str(path),"sha256":"0"*64})


if __name__=="__main__":
    unittest.main()
