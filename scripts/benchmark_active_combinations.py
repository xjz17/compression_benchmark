#!/usr/bin/env python3
"""Active-method entry point for the historical benchmark-86 composition driver.

The imported driver remains the frozen 86-method replay surface.  This wrapper
derives the current 85-method execution scope from the versioned retirement
policy before any column is scheduled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import benchmark_86_combinations as historical  # noqa: E402
from myapp.active_method_policy import (  # noqa: E402
    active_method_policy_binding,
    active_method_policy_summary,
    is_active_method_name,
)


ACTIVE_SPECIALIST_METHODS = tuple(
    method for method in historical.SPECIALIST_METHODS if is_active_method_name(method)
)
ACTIVE_GENERAL_METHODS = tuple(
    method for method in historical.GENERAL_METHODS if is_active_method_name(method)
)


def _output_path(argv: list[str]) -> Path:
    try:
        index = argv.index("--output")
        return Path(argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("active benchmark requires --output for policy provenance") from error


def _write_policy_provenance(output: Path, binding: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "ACTIVE_METHOD_POLICY.json").write_text(
        json.dumps(binding, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _bind_column_results(binding: dict) -> None:
    historical_run_column = historical.run_column

    def run_column_with_policy(col):
        result = historical_run_column(col)
        result["active_method_policy"] = binding
        return result

    historical.run_column = run_column_with_policy


def _bind_completed_manifest(output: Path, binding: dict) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("active benchmark completed without manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_method_policy"] = binding
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    policy = active_method_policy_summary()
    binding = active_method_policy_binding()
    active_count = len(ACTIVE_SPECIALIST_METHODS) + len(ACTIVE_GENERAL_METHODS)
    if active_count != policy["active_method_count"]:
        raise RuntimeError(
            f"active composition scope mismatch: {active_count} != "
            f"{policy['active_method_count']}"
        )
    historical.SPECIALIST_METHODS = ACTIVE_SPECIALIST_METHODS
    historical.GENERAL_METHODS = ACTIVE_GENERAL_METHODS
    output = _output_path(sys.argv[1:])
    _write_policy_provenance(output, binding)
    _bind_column_results(binding)
    status = historical.main()
    if status == 0:
        _bind_completed_manifest(output, binding)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
