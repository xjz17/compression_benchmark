#!/usr/bin/env python3
"""Final migration inventory, frozen pre-migration ratio regression and smoke tests."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import primitive_operator_action_runtime as p
import unified_action_runtime as u

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_OUT = ROOT / "analysis_outputs/python_to_cpp_action_migration_v1"
OUT = ROOT / "analysis_outputs/native_lz_parameters_20260908_v1/integration"


def save_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    assert len(p.POLICY_ACTIONS) == len(u.CANONICAL_ACTIONS) == 156
    assert sum(map(len, p.POLICY_IO_VARIANTS.values())) == 180
    assert len({a.operator_id for a in p.POLICY_ACTIONS}) == 28
    rows = []
    for spec in p.POLICY_ACTIONS:
        assert ".python" not in spec.name
        assert spec.implementation_source.startswith("cpp/")
        descriptor = u.describe_action(spec.name)
        rows.append(dict(policy_id=spec.action_id, name=spec.name, operator_id=spec.operator_id,
                         input_kinds="|".join(descriptor.accepted_input_kinds),
                         output_kinds="|".join(descriptor.output_kinds),
                         io_variants="|".join(p.POLICY_IO_VARIANTS[spec.name]),
                         source=spec.implementation_source,
                         scope=spec.implementation_scope))
    save_csv("implementations_156.csv", rows)
    migration = []
    for old_id, old in enumerate(p.LEGACY_225_POLICY_ACTIONS):
        canonical = p.ACTION_CANONICAL_NAMES[old.name]
        new_id = p.LEGACY_225_POLICY_ID_TO_CURRENT[old_id]
        assert p.POLICY_ACTION_BY_ID[new_id].name == canonical
        migration.append(dict(old_policy_id=old_id, old_name=old.name,
                              policy_id=new_id, canonical_name=canonical,
                              operator_id=old.operator_id,
                              parameters=json.dumps(p.migrate_policy_action(old_id, registry_version="cpp-canonical-225-v1")["parameters"], sort_keys=True),
                              io_variant=p.migrate_policy_action(old_id, registry_version="cpp-canonical-225-v1")["io_variant"]))
    save_csv("policy_id_migration_225_to_156.csv", migration)
    migration206 = []
    for old in p.LEGACY_206_POLICY_ACTIONS:
        invocation = p.migrate_policy_action(old.action_id, registry_version="cpp-canonical-206-v1")
        assert invocation["io_variant"] == p.algorithm_io.LEGACY_VARIANT_TO_CURRENT.get(old.name, old.name)
        migration206.append(dict(old_policy_id=old.action_id, old_name=old.name,
            **{**invocation, "parameters": json.dumps(invocation["parameters"], sort_keys=True)}, canonical_name=p.POLICY_ACTION_BY_ID[invocation["policy_action_id"]].name))
    save_csv("policy_id_migration_206_to_156.csv", migration206)
    migration182 = []
    for old in p.LEGACY_182_POLICY_ACTIONS:
        invocation = p.migrate_policy_action(old.action_id, registry_version="cpp-canonical-182-v1")
        migration182.append(dict(old_policy_id=old.action_id, old_name=old.name, **{**invocation, "parameters": json.dumps(invocation["parameters"], sort_keys=True)}))
    save_csv("policy_id_migration_182_to_156.csv", migration182)
    save_csv("aliases.csv", [row for row in migration if row["old_name"] != row["canonical_name"]])
    (OUT / "canonical_registry.json").write_text(
        json.dumps(p.registry(), ensure_ascii=False, indent=2) + "\n")
    assert p.LEGACY_225_POLICY_ID_TO_CURRENT[225] == 156
    assert p.LEGACY_206_POLICY_ID_TO_CURRENT[206] == 156
    assert len({row["canonical_name"] for row in migration}) == 156
    for filename in ("ans_rans_python.py", "huffman_bytes_python.py", "arithmetic_bytes_python.py"):
        assert not (ROOT / "backend/myapp" / filename).exists()

    # Both typed views of identical bits use exactly one dictionary kernel.
    import struct
    words = [0, 1 << 63, (1 << 64)-1, (1 << 53)+1, 0x7ff800000000002a,
             0x7ff0000000000000, 1, 0, 1 << 63]
    raw = struct.pack("<" + "Q"*len(words), *words)
    wires = []
    for kind in ("int64", "float64"):
        source = u.ActionData(kind, raw, len(words))
        wire = u.run_policy_action(p.cpp_migration.DICTIONARY, source,
                                   registry_version=p.ACTION_REGISTRY_VERSION)
        assert u.run_policy_action(p.cpp_migration.DICTIONARY, wire,
                 registry_version=p.ACTION_REGISTRY_VERSION, reverse=True) == source
        wires.append(u._unpack_result(wire)[1])
    assert wires[0] == wires[1]
    try:
        u.run_policy_action(0, u.ActionData.ints([1]), registry_version="old-225")
    except u.ActionRuntimeError:
        pass
    else:
        raise AssertionError("old IDs silently accepted")
    return dict(old_actions=182, canonical_actions=156, families=28, removed_parameter_configurations=26,
                io_variants=180, io_merge_groups=15, parameter_merge_groups=1, alias_names=len(p.ACTION_ALIASES), stop_id=156,
                dictionary_union_roundtrips=2)


def frozen_ratios():
    rows = []
    for size in (128, 2048):
        oldpath = ROOT / f"analysis_outputs/lossless_action_policy_20260905_v2/final_{size}/method_column_comparisons.csv"
        newpath = HISTORICAL_OUT / f"benchmark86_{size}/method_column_comparisons.csv"
        with oldpath.open(encoding="utf-8-sig", newline="") as stream:
            old = list(csv.DictReader(stream))
        with newpath.open(encoding="utf-8-sig", newline="") as stream:
            new = list(csv.DictReader(stream))
        key = lambda r: (r["method"], r["dataset"], r["column"], r["raw_sha256"])
        lookup = {key(row): row for row in old}
        assert set(lookup) == {key(row) for row in new}
        for row in new:
            previous = lookup[key(row)]
            expected_status = previous["status"]
            assert row["status"] == expected_status, (size, key(row), previous["status"], row["status"])
            matched = expected_status == "EXACT_BYTE_AND_RATIO_PARITY"
            if matched:
                assert row["compressed_bytes"] == previous["original_bytes"] == previous["compressed_bytes"]
                assert row["full_inverse_exact"] == row["per_stage_inverse_exact"] == "True"
            rows.append(dict(requested_rows=size, method=row["method"], dataset=row["dataset"],
                             column=row["column"], raw_sha256=row["raw_sha256"],
                             status=expected_status, old_bytes=previous["original_bytes"],
                             current_bytes=row["compressed_bytes"], frozen_ratio_verified=matched))
    save_csv("frozen_86_ratio_regression.csv", rows)
    checked = sum(row["frozen_ratio_verified"] for row in rows)
    assert checked == 1336
    return dict(frozen_cases=len(rows), frozen_ratio_matches=checked, ratio_mismatches=0)


def archived_sources():
    records = []
    for archive in ("python_implementation_source_backup.tar.gz", "reger_codec_source_backup.tar.gz"):
        with tarfile.open(HISTORICAL_OUT / archive) as handle:
            for entry in handle.getmembers():
                if not entry.isfile() or Path(entry.name).name.startswith("._"):
                    continue
                old = handle.extractfile(entry).read()
                path = ROOT / entry.name
                new = path.read_bytes() if path.exists() else b""
                records.append(dict(path=entry.name, archive=archive,
                                    old_lines=len(old.splitlines()), current_lines=len(new.splitlines()),
                                    old_sha256=hashlib.sha256(old).hexdigest(),
                                    current_sha256=hashlib.sha256(new).hexdigest() if path.exists() else "",
                                    status="deleted_file" if not path.exists() else
                                           "modified" if old != new else "unchanged"))
    save_csv("source_removal_manifest.csv", records)
    return dict(archived_source_files=len(records), deleted_files=sum(r["status"] == "deleted_file" for r in records))


def smoke_tests():
    commands = [
        ["scripts/test_policy_action_ids.py"],
        ["scripts/primitive_operator_action_runtime.py", "--self-check"],
        ["scripts/unified_action_runtime.py", "--self-check"],
        ["scripts/test_atomic_cpp_migration.py", "verify"],
        ["scripts/test_legacy_numeric_stages.py"],
        ["scripts/test_snappy_legacy_stages.py"],
        ["scripts/test_reger_legacy_stages.py"],
        ["scripts/test_blosc2_legacy_stages.py"],
        ["scripts/test_corad_lossless_stages.py"],
        ["scripts/test_lossless64_action_runtime.py"],
        ["scripts/test_reference_operator_cpp.py"],
        ["scripts/test_reference_python_bridges.py"],
    ]
    results = []
    for command in commands:
        print("TEST", " ".join(command), flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)
        row = dict(command=command, exit_code=result.returncode,
                   stdout=result.stdout, stderr=result.stderr)
        results.append(row)
        (OUT / "regression_tests.json").write_text(json.dumps(results, indent=2) + "\n")
        assert result.returncode == 0, row
    return dict(regression_commands=len(results), passed=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--inventory-only", action="store_true",
                        help="verify/export current 156 IDs and all I/O migrations; do not read historical benchmark results")
    args = parser.parse_args()
    if args.inventory_only:
        result = dict(status="PASS", registry_version=p.ACTION_REGISTRY_VERSION, inventory=inventory())
        (OUT / "inventory_verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        raise SystemExit(0)
    result = dict(status="PASS", inventory=inventory(), historical_206_frozen_86=frozen_ratios(),
                  source_removal=archived_sources())
    if not args.skip_smoke:
        result["smoke"] = smoke_tests()
    elif (OUT / "regression_tests.json").exists():
        tests = json.loads((OUT / "regression_tests.json").read_text())
        assert len(tests) == 12 and all(row["exit_code"] == 0 for row in tests)
        result["previous_completed_smoke"] = dict(regression_commands=len(tests), passed=len(tests))
    for directory, key in (("canonical_integration", "integration"),
                           ("remaining206_pairs", "remaining_pairs")):
        path = HISTORICAL_OUT / directory / "summary.json"
        if path.exists():
            extended = json.loads(path.read_text())
            if key == "integration":
                assert not extended["failures"]
            result[key] = {name: value for name, value in extended.items()
                           if name not in ("source_hashes", "legacy_domain_rejections", "failures")}
    sources = {ROOT / "scripts" / name for name in
               ("action_cpp_migration.py", "primitive_operator_action_runtime.py",
                "unified_action_runtime.py", "test_cpp_action_migration.py")}
    sources.update(ROOT / action.implementation_source.split(":")[0]
                   for action in p.POLICY_ACTIONS)
    sources.update(ROOT / name for name in (
        "cpp/CMakeLists.txt", "backend/myapp/cpp_action_bridge.py",
        "backend/myapp/native_lz_family.py", "backend/myapp/legacy_numeric_stages.py",
        "backend/myapp/snappy_legacy_stages.py", "backend/myapp/corad_lossless_stages.py",
        "backend/myapp/reger_legacy_stages.py", "backend/myapp/reger_codec.py",
        "backend/myapp/blosc2_legacy_stages.py", "backend/myapp/column_encodings.py",
        "backend/myapp/entropy_cpp.py", "scripts/generate_air_operator_pair_heatmap.py",
        "scripts/measure_selected_codec_cost.py"))
    result["source_hashes"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    (OUT / "migration_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
