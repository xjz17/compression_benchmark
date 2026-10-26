#!/usr/bin/env python3
"""Check real transformed-data stages against the untouched legacy benchmark."""
import struct

import audit_86_composability_realdata as external
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified


def main():
    columns, _ = external.load_external_columns(128)
    Column = external.benchmark.Column
    for index, values in enumerate(((), (0,), (-2**63, 2**63-1), (1,)*1025)):
        columns.append(Column("boundary", index, str(index), "integer", 0, values))
    for index, values in enumerate(((), (0.0, -0.0), (1.25,)*1025,
                                    (float('nan'), float('inf'), -float('inf')))):
        columns.append(Column("boundary", index, str(index), "float", 2, values))
    cases, unavailable = 0, 0
    for column in columns:
        for method in ("FastPFor", "LECO", "ALP"):
            original, _ = external.benchmark.specialist_artifact(column, method)
            if original is None:
                unavailable += 1
                continue
            kind = "int64" if column.logical_type == "integer" else "float64"
            key = "int" if kind == "int64" else "float"
            names = tuple(primitive.benchmark_86_legacy_action_pipelines()[method][key])
            source = unified.ActionData(kind, column.raw, len(column.values))
            current = source
            for name in names:
                previous = current
                current = unified.run_action(name, current, max_point=column.max_point)
                assert unified.run_action(name, current, reverse=True) == previous, name
            assert unified._unpack_result(current)[1] == original.payload, (method, column.name)
            assert unified.run_actions(names, current, reverse=True) == source
            cases += 1
    # A standalone framing call must also retain its inverse count without a
    # nested UAR1 history from preceding stages.
    module = primitive.legacy_numeric_stages
    packet = struct.pack('<qq', -9, 17)
    for action in module.SCALED_ACTIONS[:-1]:
        packet = module.forward(action, packet, 'int64')
    data = unified.ActionData.bytes(packet)
    result = unified.run_action(module.FRAME, data)
    assert unified.run_action(module.FRAME, result, reverse=True) == data
    print({'status': 'PASS', 'exact_legacy_numeric_cases': cases,
           'original_unavailable_cases': unavailable})


if __name__ == '__main__':
    main()
