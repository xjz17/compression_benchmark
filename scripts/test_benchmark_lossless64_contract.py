#!/usr/bin/env python3
"""Regression checks for the user-selected NaN-free binary64 benchmark contract."""
import struct

import benchmark_86_combinations as benchmark
import primitive_operator_action_runtime as primitive


def main():
    floating = benchmark.Column("probe", 0, "float", "float", 5,
                                (float("nan"), -0.0, 1.2345678912345, 2.0))
    assert floating.source_count == 4 and floating.removed_nan_count == 1
    assert len(floating.values) == 3
    assert floating.raw[:8] == struct.pack("<d", -0.0)
    integer = benchmark.Column("probe", 1, "int", "integer", 0,
                               ("9223372036854775807", "NaN", "9007199254740993"))
    assert integer.values == (2**63 - 1, 2**53 + 1)
    assert integer.removed_nan_count == 1
    empty = benchmark.Column("probe", 2, "empty", "float", 0, (float("nan"),))
    assert empty.values == () and empty.raw == b""
    try:
        benchmark.run_column(empty)
    except ValueError:
        pass
    else:
        raise AssertionError("all-NaN columns must not be measured")
    assert benchmark.specialist_artifact(floating, "Camel")[0] is None
    for method, col in (("Elf", floating), ("Elf+", floating), ("CORAD", floating),
                        ("CORAD", integer), ("BUFF", floating), ("BUFF", integer)):
        artifact, reason = benchmark.specialist_artifact(col, method)
        assert artifact is not None, (method, reason)
        assert artifact.decode(artifact.payload) == col.raw, method
    wire = primitive.encode_registered_exact_action_pipeline(
        integer.raw, primitive.CORAD_LOSSLESS_ACTIONS, logical_type="int")
    assert primitive.corad_lossless_stages.decode(wire, "int64") == integer.raw
    print({"status": "PASS", "contract": benchmark.BENCHMARK_CONTRACT,
           "nan_filtered": True, "large_int64_exact": True,
           "elf_elfplus_corad_buff_bitwise_roundtrip": True,
           "camel_excluded": True})


if __name__ == "__main__":
    main()
