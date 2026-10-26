#!/usr/bin/env python3
"""Policy logits and STOP must not share the stable concrete-ID namespace."""
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified


def main():
    policy = primitive.POLICY_ACTIONS
    assert [action.action_id for action in policy] == [
        action_id for action_id in range(primitive.STOP_ACTION_ID)
        if action_id not in {5, 18, 35, 87, 90}
    ]
    assert primitive.FROZEN_POLICY_ACTION_BY_ID[5].name == "op.scaling.reference_v1"
    assert 5 not in primitive.POLICY_ACTION_BY_ID
    assert primitive.STOP_ACTION_ID not in primitive.POLICY_ACTION_BY_ID
    for action in policy:
        concrete = primitive.ALL_CONCRETE_ACTION_BY_NAME[
            primitive.CANONICAL_TO_LEGACY_NAME[action.name]]
        assert primitive.POLICY_TO_CONCRETE_ACTION_ID[action.action_id] == concrete.action_id
        assert primitive.POLICY_ACTION_BY_ID[action.action_id].name == action.name
        assert unified.describe_action(action.name).name == action.name
        assert action.implementation_source.startswith("cpp/")
    assert len(policy) == 122
    assert len(primitive.FROZEN_POLICY_ACTIONS) == 127
    assert primitive.STOP_ACTION_ID == 127
    assert sum(map(len, primitive.POLICY_IO_VARIANTS.values())) == 170
    assert len(primitive.LEGACY_182_POLICY_ID_TO_CURRENT) == 183
    assert primitive.LEGACY_182_POLICY_ID_TO_CURRENT[182] == primitive.STOP_ACTION_ID
    assert len(primitive.LEGACY_206_POLICY_ID_TO_CURRENT) == 207
    assert primitive.LEGACY_206_POLICY_ID_TO_CURRENT[206] == primitive.STOP_ACTION_ID
    assert len(primitive.LEGACY_225_POLICY_ID_TO_CURRENT) == 226
    assert primitive.LEGACY_225_POLICY_ID_TO_CURRENT[225] == primitive.STOP_ACTION_ID
    print({"status": "PASS", "policy_actions": len(policy),
           "policy_stop_id": primitive.STOP_ACTION_ID,
           "stable_concrete_ids": len(primitive.ALL_CONCRETE_ACTION_IMPLEMENTATIONS)})


if __name__ == "__main__":
    main()
