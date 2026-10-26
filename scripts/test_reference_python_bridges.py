#!/usr/bin/env python3
"""Check thin plotting/OPC1 bridges against frozen pre-migration OPA1 states.

No historical Python forward/inverse kernel is loaded. OPC1 expected bytes
are the frozen OPA1 body with only the documented public magic replaced.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import zlib

import numpy as np

import generate_air_operator_pair_heatmap as heatmap
import measure_selected_codec_cost as codec

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "analysis_outputs/reference_operator_cpp_migration_v1/python_goldens.json"


def main() -> None:
    golden = json.loads(GOLDENS.read_text())
    checks = {"state_wires": 0, "state_inverses": 0, "opc1_wires": 0,
              "opc1_inverses": 0, "compact": 0, "single_step_inverse": 0,
              "legacy_negative_rejections": 0, "retired_identity_rejections": 0,
              "malformed_rejections": 0}
    failures = []
    for case_index, case in enumerate(golden["cases"]):
        raw = base64.b64decode(case["raw"])
        values = np.frombuffer(raw, dtype="<i8")
        compact = heatmap.compact_int_payload(values)
        assert codec.parse_compact_payload(compact, 0).tobytes() == raw
        checks["compact"] += 1
        for record in case["records"]:
            if "op.xor.chimp_history128.reference_v1" in record["actions"]:
                continue  # The historical plotting API exposes only 28 families.
            pipeline = tuple(name[3:-len(".reference_v1")] for name in record["actions"])
            if pipeline == ("scaling",):
                try:
                    heatmap.apply_operator(heatmap.initial_state(values), "scaling")
                except ValueError as error:
                    assert "unsupported operator: scaling" in str(error)
                    checks["retired_identity_rejections"] += 1
                    continue
                failures.append([case["label"], pipeline, "retired identity forward was accepted"])
                continue
            try:
                state = heatmap.initial_state(values)
                for operator in pipeline:
                    state = heatmap.apply_operator(state, operator)
                if "wire" not in record:
                    raise AssertionError("accepted an original forward failure")
                original = zlib.decompress(base64.b64decode(record["wire"]))
                expected_state = original[6 + len(pipeline):]
                assert state.serialized() == expected_state
                checks["state_wires"] += 1
                if record.get("lossless"):
                    assert state.restore().tobytes() == raw
                    checks["state_inverses"] += 1
                else:
                    try:
                        state.restore()
                    except RuntimeError:
                        checks["legacy_negative_rejections"] += 1
                    else:
                        raise AssertionError("accepted an original inverse failure")
                if pipeline:
                    expected_wire = zlib.compress(b"OPC1" + original[4:], level=6)
                    wire = codec.encode_stream(values, pipeline, heatmap)
                    assert wire == expected_wire
                    checks["opc1_wires"] += 1
                    if record.get("lossless"):
                        restored, recovered = codec.decode_stream(wire)
                        assert restored.tobytes() == raw and recovered == pipeline
                        checks["opc1_inverses"] += 1
                if (len(pipeline) == 1 and record.get("lossless")
                        and case_index in (0, 8, 16, len(golden["cases"]) - 1)):
                    (tag, arrays, extra), offset = codec.parse_meta(expected_state, 0)
                    final = codec.parse_compact_payload(expected_state, offset)
                    assert final.tobytes() == state.values.tobytes()
                    assert codec.invert_step(final, tag, arrays, extra).tobytes() == raw
                    checks["single_step_inverse"] += 1
            except (RuntimeError, OverflowError) as error:
                if "encode_error" in record:
                    checks["legacy_negative_rejections"] += 1
                else:
                    failures.append([case["label"], pipeline, str(error)])
            except Exception as error:
                failures.append([case["label"], pipeline, str(error)])
    for compact in (b"", b"I0", b"I0\x01\x00\x00\x00",
                    b"ZZ\x00\x00\x00\x00", b"I8\x01\x00\x00\x00",
                    b"U1\x00\x00\x00\x00\x00"):
        try:
            codec.parse_compact_payload(compact, 0)
        except codec.CodecError:
            checks["malformed_rejections"] += 1
        else:
            failures.append(["malformed", compact.hex()])
    print(json.dumps({"status": "FAIL" if failures else "PASS", **checks,
                      "failures": failures[:20], "failure_count": len(failures)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
