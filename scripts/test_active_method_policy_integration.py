#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_active_combinations as combinations  # noqa: E402
import active_primitive_operator_runtime as primitive  # noqa: E402
import physical_operator_runtime_v2 as physical  # noqa: E402
from myapp.active_method_policy import active_method_policy_summary  # noqa: E402


def main() -> None:
    policy = active_method_policy_summary()
    assert policy["policy_path"] == "benchmark/reference/methods/active_method_policy_v1.json"
    assert policy["policy_sha256"] == "f48677a529247a165898bce319f3e20ef6ba9c305552676344f3ead39d0bfc5d"
    active_methods = combinations.ACTIVE_SPECIALIST_METHODS + combinations.ACTIVE_GENERAL_METHODS
    assert len(active_methods) == policy["active_method_count"] == 85
    assert "GLZA" not in active_methods
    assert len(primitive.ACTIVE_BENCHMARK_METHODS) == 85
    assert "GLZA" not in primitive.ACTIVE_BENCHMARK_METHODS
    try:
        primitive.require_active_method("GLZA")
    except ValueError as exc:
        assert policy["policy_id"] in str(exc)
    else:
        raise AssertionError("active primitive API accepted GLZA")
    assert all(spec.codec_key != "glza" for spec in physical.COMPRESSORS)
    retired = [spec for spec in physical.RETIRED_CATALOG_COMPRESSORS if spec.codec_key == "glza"]
    assert len(retired) == 1
    assert physical.ACTION_BY_ID[retired[0].wire_id].codec_key == "glza"
    assert retired[0].name not in physical.ACTION_BY_NAME
    binding = combinations.active_method_policy_binding()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        combinations._write_policy_provenance(output, binding)
        sidecar = json.loads((output / "ACTIVE_METHOD_POLICY.json").read_text())
        assert sidecar == binding
        (output / "manifest.json").write_text('{"schema": 1}\n', encoding="utf-8")
        combinations._bind_completed_manifest(output, binding)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["active_method_policy"] == binding
    print("active method policy integration: PASS")


if __name__ == "__main__":
    main()
