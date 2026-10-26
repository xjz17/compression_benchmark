#!/usr/bin/env python3
"""Small dependency-free regression check for the unified action boundary."""

from __future__ import annotations

from unified_action_runtime import (
    ACTIONS,
    ActionData,
    ActionNotExecutableError,
    ActionTypeError,
    describe_action,
    run_action,
    run_actions,
    self_check,
    _unpack_result,
)
import primitive_operator_action_runtime as primitive


def assert_stepwise(
    names: tuple[str, ...],
    source: ActionData,
    expected_wire: bytes,
    *,
    max_point: int | None = None,
) -> None:
    current = source
    for name in names:
        current = run_action(name, current, max_point=max_point)
    _, raw_wire = _unpack_result(current)
    assert raw_wire == expected_wire
    for name in reversed(names):
        current = run_action(name, current, reverse=True)
    assert current == source


def main() -> None:
    result = self_check()
    assert result["status"] == "PASS"
    assert result["active_action_count"] == 122
    assert result["policy_action_slots_including_stop"] == 128
    assert result["policy_stop_action_id"] == 127
    assert result["compatibility_execution_modes"] == {
        "descriptor_only": 42,
        "experimental_non_atomic": 1,
        "stagewise": 196,
        "standalone": 33,
        "retired": 8,
    }
    assert result["native_lz_method_round_trips"] == 47
    assert result["native_lz_independent_action_round_trips"] == 60
    assert result["native_lz_action_names_covered"] == 60
    assert result["ts2diff_exact_stage_actions"] == 4
    assert result["svb_exact_stage_actions"] == 2
    assert result["simple8b_exact_stage_actions"] == 2
    assert result["rlbe_exact_stage_actions"] == 2
    assert result["sprintz_exact_stage_actions"] == 3
    assert result["dbp_exact_stage_actions"] == 4
    assert result["gorilla_exact_stage_actions"] == 2
    assert result["chimp128_exact_stage_actions"] == 2
    assert result["pongo_exact_stage_actions"] == 3
    assert result["spdp_exact_stage_actions"] == 3
    assert result["fpc_exact_stage_actions"] == 2
    assert result["pfpc_exact_stage_actions"] == 2
    assert result["rle_scaling_exact_stage_actions"] == 1
    assert result["bos_policy_stage_actions"] == 7
    assert result["bos_independent_chain_round_trips"] == 3
    assert result["fastpfor_policy_stage_actions"] == 2
    assert result["fastpfor_exact_chain_round_trips"] == 2
    assert result["elf_family_exact_stage_actions"] == 3
    assert result["camel_policy_stage_actions"] == 3
    assert result["camel_exact_chain_round_trips"] == 2
    assert result["snappy_policy_stage_actions"] == 3
    assert result["lz4_policy_stage_actions"] == 4
    assert result["deflate_container_exact_stage_actions"] == 2
    assert result["xz_container_exact_stage_actions"] == 1
    assert result["fpzip_exact_stage_actions"] == 2
    assert result["leco_exact_stage_actions"] == 3
    assert result["alp_exact_stage_actions"] == 4
    assert result["reger_exact_stage_actions"] == 6
    assert result["buff_exact_stage_actions"] == 4
    assert result["general_reference_stage_actions"] == 23
    assert result["general_reference_chain_round_trips"] == 14
    assert result["general_exact_residual_stage_actions"] == 12
    assert result["general_exact_residual_chain_round_trips"] == 14

    entropy = "op.entropy_fse.project_rans_bytes.v1"
    source = ActionData.bytes(b"typed-action" * 32)
    encoded = run_action(entropy, source)
    assert encoded.kind == "bytes"
    assert run_action(entropy, encoded, reverse=True) == source
    entropy_id = describe_action(entropy).action_id
    encoded_by_id = run_action(entropy_id, source)
    assert run_action(entropy_id, encoded_by_id, reverse=True) == source

    cpp_only = next(action for action in ACTIONS if action.execution_mode == "descriptor_only")
    try:
        run_action(cpp_only.name, source)
    except ActionNotExecutableError:
        pass
    else:
        raise AssertionError("descriptor-only C++ candidate must not execute a proxy")

    try:
        run_action("op.delta.reference_v1", source)
    except ActionTypeError:
        pass
    else:
        raise AssertionError("int64 action accepted bytes input")

    native_actions = [
        action for action in ACTIONS
        if action.name in primitive.EXACT_NATIVE_ACTION_BY_NAME
    ]
    assert len(native_actions) == 60
    assert all(action.execution_mode == "stagewise" for action in native_actions)

    assert describe_action(cpp_only.name).action_id == cpp_only.action_id

    int_probe = ActionData.ints([-9, -9, -3, 0, 1, 1, 17, 17, 17])
    float_probe = ActionData.floats([-3.5, -3.5, 0.0, 1.25, 1.25, 9.0])
    byte_probe = ActionData.bytes(bytes(range(64)) * 4)
    try:
        run_action("op.run_length.tsfile_int64.v1", float_probe)
    except ActionTypeError:
        pass
    else:
        raise AssertionError("integer-only RLE action accepted direct float64 input")
    stepwise_cases = (
        (
            primitive.RLBE_ACTIONS,
            int_probe,
            primitive.encode_exact_p04_action_pipeline(int_probe.payload, "RLBE", "int"),
            None,
        ),
        (
            primitive.SPRINTZ_ACTIONS,
            float_probe,
            primitive.encode_exact_p04_action_pipeline(float_probe.payload, "Sprintz", "float"),
            None,
        ),
        (
            primitive.DBP_FLOAT_ACTIONS,
            float_probe,
            primitive.encode_exact_dbp_action_pipeline(float_probe.payload, "float", max_point=2),
            2,
        ),
        (
            primitive.GORILLA_ACTIONS,
            float_probe,
            primitive.encode_exact_gorilla_action_pipeline(float_probe.payload, "float"),
            None,
        ),
        (
            primitive.CHIMP128_ACTIONS,
            float_probe,
            primitive.encode_exact_chimp128_action_pipeline(float_probe.payload, "float"),
            None,
        ),
        (
            primitive.SPDP_ACTIONS,
            byte_probe,
            primitive.encode_exact_spdp_action_pipeline(byte_probe.payload),
            None,
        ),
        (
            primitive.FPC_ACTIONS,
            byte_probe,
            primitive.encode_exact_fpc_action_pipeline(byte_probe.payload),
            None,
        ),
        (
            primitive.PFPC_ACTIONS,
            byte_probe,
            primitive.encode_exact_pfpc_action_pipeline(byte_probe.payload),
            None,
        ),
        (
            primitive.BOS64_INT_ACTIONS,
            int_probe,
            primitive.encode_exact_bos64_action_pipeline(int_probe.payload, "int"),
            None,
        ),
        (
            primitive.BOS64_FLOAT_ACTIONS,
            float_probe,
            primitive.encode_exact_bos64_action_pipeline(
                float_probe.payload, "float", max_point=2
            ),
            2,
        ),
        (
            primitive.FASTPFOR_INT_ACTIONS,
            int_probe,
            primitive.encode_exact_fastpfor_action_pipeline(int_probe.payload, "int"),
            None,
        ),
        (
            primitive.FASTPFOR_FLOAT_ACTIONS,
            float_probe,
            primitive.encode_exact_fastpfor_action_pipeline(
                float_probe.payload, "float", max_point=2
            ),
            2,
        ),
        (
            primitive.FLOAT_SCALING_THEN_INT_RLE_ACTIONS,
            float_probe,
            primitive.encode_exact_atomic_method(
                float_probe.payload, "RLE", "float", max_point=2
            ),
            2,
        ),
        (
            primitive.LECO_ACTIONS,
            int_probe,
            primitive.encode_exact_leco_action_pipeline(int_probe.payload),
            None,
        ),
        (
            primitive.ALP_MAIN_ACTIONS,
            float_probe,
            primitive.encode_exact_alp_action_pipeline(float_probe.payload),
            None,
        ),
        (
            primitive.REGER_ACTIONS,
            int_probe,
            primitive.encode_exact_reger_action_pipeline(int_probe.payload),
            None,
        ),
        (
            primitive.BUFF_ATOMIC_ACTIONS,
            float_probe,
            primitive.encode_exact_buff_action_pipeline(float_probe.payload),
            None,
        ),
    )
    for names, case_source, expected, max_point in stepwise_cases:
        assert_stepwise(names, case_source, expected, max_point=max_point)

    alp_dictionary = run_actions(primitive.ALP_DICTIONARY_ACTIONS, float_probe)
    assert run_actions(
        primitive.ALP_DICTIONARY_ACTIONS, alp_dictionary, reverse=True
    ) == float_probe

    for names in (
        primitive.BROTLI_REFERENCE_ACTIONS,
        primitive.ZSTD_REFERENCE_ACTIONS,
        primitive.BZIP2_REFERENCE_ACTIONS,
        primitive.GZIP_REFERENCE_ACTIONS,
        primitive.ZLIB_REFERENCE_ACTIONS,
        primitive.XZ_REFERENCE_ACTIONS,
        *primitive.BLOSC_REFERENCE_ACTIONS_BY_KIND.values(),
        *primitive.BLOSC2_REFERENCE_ACTIONS_BY_KIND.values(),
    ):
        encoded = run_actions(names, byte_probe)
        assert run_actions(names, encoded, reverse=True) == byte_probe

    print(result)


if __name__ == "__main__":
    main()
