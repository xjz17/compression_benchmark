#!/usr/bin/env python3
"""Admission and paired-metric regression tests; no benchmark claims."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_prepared_cohort_v1 as data
import train_native_smallest32_v1 as training
from native_grpo_task_v1 import NativeGraphTask
import weight_conditioned_grpo_v1 as grpo

GATES=('all_fixed_configuration_roundtrips','independent_cold_decode','native_policy_parity',
       'native_feature_parity','typed_graph_and_corruption','probability_state_isolation',
       'baseline86_inventory','training_bridge_and_budget')

class AdmissionTests(unittest.TestCase):
    def test_infeasible_cost_has_no_fake_metrics_and_dominates_bounded_J(self):
        failed=grpo.TerminalFailure({'recipe_action_ids':[1]},64,1<<30,'range not representable')
        for weight in grpo.FIXED_WEIGHTS:
            bound=weight[0]*(1<<30)/64+weight[1]+weight[2]
            self.assertGreater(grpo.objective_value(failed,weight),bound)
        self.assertFalse(hasattr(failed,'compressed_payload'))
        self.assertFalse(hasattr(failed,'metrics'))

    def test_receipt_or_runtime_change_revokes_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);evidence=root/'receipt';evidence.write_text('verified')
            binary=root/'binary';binary.write_bytes(b'cpp-v1')
            scope=training.experiment_scope.load()
            doc=dict(schema='native-smallest32-training-admission-v2',passed=True,
                action_registry_fingerprint=training.REGISTRY,action_count=6341,
                catalog_fixed_configuration_count=193,fixed_configuration_count=191,experiment_scope=scope,
                gates={k:True for k in GATES},receipts={k:dict(path=str(evidence),sha256=data.digest(evidence.read_bytes())) for k in GATES},
                frozen_files={str(binary):data.digest(binary.read_bytes()),scope['profile_path']:scope['profile_sha256']})
            path=root/'admission.json';path.write_text(json.dumps(doc));training.admission(path)
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source/binary changed'):training.admission(path)
            binary.write_bytes(b'cpp-v1');evidence.write_text('replaced')
            with self.assertRaisesRegex(ValueError,'receipt changed'):training.admission(path)

    def test_missing_configuration_gate_cannot_admit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json'
            doc=dict(schema='native-smallest32-training-admission-v2',passed=True,
                action_registry_fingerprint=training.REGISTRY,action_count=6341,
                catalog_fixed_configuration_count=193,fixed_configuration_count=190,
                gates={k:True for k in GATES},receipts={},frozen_files={})
            p.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'inventory drift'):training.admission(p)

if __name__=='__main__':unittest.main()
