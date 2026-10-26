import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import supervise_weighted_grpo_seed_v1 as supervisor


@unittest.skipUnless(hasattr(os, "waitid"), "target Linux supervisor requires waitid/WNOWAIT")
class SeedSupervisorTests(unittest.TestCase):
    def test_early_leader_exit_cleans_descendants_before_reaping(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                worker = root / "early.py"
                worker.write_text("import subprocess,sys\n"
                    "subprocess.Popen([sys.executable,'-c',\"import time,pathlib;time.sleep(.4);pathlib.Path('escaped').write_text('bad')\"])\n"
                    f"sys.exit({exit_code})\n")
                result = supervisor.supervise(run_root=root, seed=0, cwd=root,
                    command=[sys.executable,str(worker)], budget_seconds=2)
                self.assertEqual(result["status"], "complete" if exit_code == 0 else "failed")
                self.assertEqual(result["attempts"][0]["returncode"],exit_code)
                time.sleep(.5)
                self.assertFalse((root/"escaped").exists())

    def test_deadline_kills_only_owned_group_and_keeps_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = subprocess.Popen([sys.executable, "-c",
                "import pathlib,time;time.sleep(.6);pathlib.Path('outside_ok').write_text('ok')"],
                cwd=root, start_new_session=True)
            worker = root / "worker.py"
            worker.write_text("""
import os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path('checkpoint.json').write_text('{"completed_iterations": 7}')
subprocess.Popen([sys.executable,'-c',"import pathlib,time;time.sleep(1);pathlib.Path('child_escaped').write_text('bad')"])
pathlib.Path('checkpoint.json.partial').write_text('{incomplete')
time.sleep(5)
""")
            result = supervisor.supervise(run_root=root, seed=0, cwd=root,
                command=[sys.executable, str(worker)], budget_seconds=.4, grace_seconds=.1)
            self.assertEqual(result["status"], "hard_deadline_reached")
            self.assertEqual(json.loads((root/"checkpoint.json").read_text())["completed_iterations"],7)
            outside.wait(timeout=3)
            time.sleep(.65)
            self.assertTrue((root/"outside_ok").exists())
            self.assertFalse((root/"child_escaped").exists())
            self.assertLess(result["attempts"][0]["elapsed_wall_ns"], 900_000_000)

    def test_resume_subtracts_consumed_budget_and_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            worker=root/"worker.py"
            worker.write_text("import pathlib,time\np=pathlib.Path('once')\n"
                              "s=2 if p.exists() else .12\np.write_text('1')\ntime.sleep(s)\n")
            args=dict(run_root=root,seed=1,cwd=root,command=[sys.executable,str(worker)],
                      budget_seconds=.5,grace_seconds=.05)
            first=supervisor.supervise(**args)
            second=supervisor.supervise(**args)
            remaining=second["attempts"][1]["remaining_budget_ns_at_launch"]
            self.assertEqual(remaining,500_000_000-first["consumed_wall_ns"])
            self.assertLess(remaining,400_000_000)
            self.assertEqual(second["status"],"hard_deadline_reached")
            with self.assertRaisesRegex(RuntimeError,"identity mismatch"):
                supervisor.supervise(**{**args,"budget_seconds":1.})

    def test_lost_supervisor_conservatively_exhausts_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=dict(run_root=root,seed=2,cwd=root,command=[sys.executable,"-c","pass"],
                      budget_seconds=.5)
            supervisor.supervise(**args)
            ledger=root/"seed_supervision/seed_2.json"
            state=json.loads(ledger.read_text())
            state.update(status="running",active_attempt={"pid":999999999,"process_identity":None})
            ledger.write_text(json.dumps(state))
            result=supervisor.supervise(**args)
            self.assertEqual(result["status"],"abandoned_attempt_budget_exhausted")
            self.assertEqual(result["consumed_wall_ns"],500_000_000)
            self.assertEqual(len(result["attempts"]),1)

    def test_recover_only_kills_exact_ledger_owned_group_without_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=dict(run_root=root,seed=0,cwd=root,
                      command=[sys.executable,"-c","pass"],budget_seconds=.5)
            supervisor.supervise(**args)
            orphan=subprocess.Popen(
                [sys.executable,"-c","import time;time.sleep(30)"],
                cwd=root,start_new_session=True)
            try:
                ledger=root/"seed_supervision/seed_0.json"
                state=json.loads(ledger.read_text())
                state.update(status="running",active_attempt={
                    "pid":orphan.pid,"pgid":orphan.pid,
                    "process_identity":supervisor._linux_process_identity(orphan.pid)})
                ledger.write_text(json.dumps(state))
                result=supervisor.supervise(**args,recover_only=True)
                orphan.wait(timeout=3)
                self.assertEqual(result["status"],"abandoned_attempt_budget_exhausted")
                self.assertEqual(len(result["attempts"]),1)
                self.assertTrue(result["orphan_recovery"]["exact_owned_group_verified"])
                self.assertTrue(result["orphan_recovery"]["sigkill_sent_to_owned_group"])
                self.assertEqual(result["orphan_recovery"]["mode"],"recover_only")
            finally:
                if orphan.poll() is None:
                    os.killpg(orphan.pid,signal.SIGKILL);orphan.wait()

    def test_recover_only_cannot_clean_while_supervisor_lock_is_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=dict(run_root=root,seed=1,cwd=root,
                      command=[sys.executable,"-c","pass"],budget_seconds=.5)
            supervisor.supervise(**args)
            orphan=subprocess.Popen(
                [sys.executable,"-c","import time;time.sleep(30)"],
                cwd=root,start_new_session=True)
            lock_path=root/"seed_supervision/seed_1.lock"
            ledger=root/"seed_supervision/seed_1.json"
            try:
                state=json.loads(ledger.read_text())
                state.update(status="running",active_attempt={
                    "pid":orphan.pid,"pgid":orphan.pid,
                    "process_identity":supervisor._linux_process_identity(orphan.pid)})
                ledger.write_text(json.dumps(state))
                with lock_path.open("a+") as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with self.assertRaisesRegex(RuntimeError,"active supervisor"):
                        supervisor.supervise(**args,recover_only=True)
                    self.assertIsNone(orphan.poll())
                    unchanged=json.loads(ledger.read_text())
                    self.assertEqual(unchanged["status"],"running")
                    fcntl.flock(lock,fcntl.LOCK_UN)
            finally:
                if orphan.poll() is None:
                    os.killpg(orphan.pid,signal.SIGKILL);orphan.wait()


if __name__ == "__main__":
    unittest.main()
