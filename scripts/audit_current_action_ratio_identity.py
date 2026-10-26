#!/usr/bin/env python3
"""Same-family action size audit, not an implementation-deduplication command.

Compare only equal typed inputs, public packet context, and max_point.  Walk
registered sequences to obtain valid intermediate packets, then cross-apply
actions within a family.  Codec/stage payload and UAR1 transport sizes are
reported separately.  Equal measured sizes never imply universal equivalence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import audit_86_composability_realdata as realdata
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified

ROOT = Path(__file__).resolve().parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def dump_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def make_cases(row_limits):
    cases, manifests = [], []
    for limit in row_limits:
        columns, manifest = realdata.load_external_columns(limit)
        manifests.extend(dict(item, requested_rows=limit) for item in manifest)
        for column in columns:
            kind = "int64" if column.logical_type == "integer" else "float64"
            label = f"{column.dataset}/{column.name}/rows{limit}"
            typed = unified.ActionData(kind, column.raw, len(column.values))
            cases.append((label, typed, column.max_point))
    # ponytail: four bounded fixtures cover repeated, smooth, and float32-exact
    # inputs; broader fuzzing belongs in a separate runtime-validation task.
    cases.extend([
        ("synthetic/int_constant", unified.ActionData.ints([7] * 128), 0),
        ("synthetic/int_ramp", unified.ActionData.ints(list(range(128))), 0),
        ("synthetic/float32_exact", unified.ActionData.floats(
            [-3.5, -3.5, 0.0, 1.25, 1.25, 9.0] * 24), 2),
        ("synthetic/byte_repeat", unified.ActionData.bytes(bytes(range(64)) * 64), 0),
    ])
    return cases, manifests


def pair_result(left, right, measurements):
    """Pure aggregation, also exercised by self_check below."""
    common = sorted(set(left) & set(right))
    common = [key for key in common
              if measurements[left[key]]["kernel_input_id"] ==
              measurements[right[key]]["kernel_input_id"]]
    equal = [key for key in common if measurements[left[key]]["payload_bytes"] ==
             measurements[right[key]]["payload_bytes"]]
    wire_equal = [key for key in common if measurements[left[key]]["payload_sha256"] ==
                  measurements[right[key]]["payload_sha256"]]
    different = [key for key in common if key not in equal]
    return common, equal, wire_equal, different


def self_check():
    records = [dict(kernel_input_id="x", payload_bytes=n, payload_sha256=h)
               for n, h in [(3, "a"), (3, "b"), (4, "c")]]
    assert pair_result({"s": 0}, {"s": 1}, records) == (["s"], ["s"], [], [])
    assert pair_result({"s": 0}, {"s": 2}, records) == (["s"], [], [], ["s"])
    records[2]["kernel_input_id"] = "other"
    assert pair_result({"s": 0}, {"s": 2}, records) == ([], [], [], [])


def audit(output, row_limits):
    started = time.monotonic()
    specs = list(primitive.POLICY_ACTIONS)
    descriptors = {s.name: unified.describe_action(s.name) for s in specs}
    families = defaultdict(list)
    for spec in specs:
        families[spec.operator_id].append(spec.name)
    excluded = set(primitive.CAMEL_ACTIONS)
    cases, manifest = make_cases(row_limits)
    states, pools, attempts, measurements = {}, defaultdict(set), {}, []
    trace_errors, trace_success, good = [], Counter(), defaultdict(dict)
    case_rows = []

    def state_id(data, point, label, prefix):
        digest = sha(json.dumps([data.kind, data.count, point]).encode() + data.payload)
        if digest not in states:
            states[digest] = (data, point, label, prefix)
        return digest

    def kernel_input(name, data):
        descriptor = descriptors[name]
        if descriptor.execution_mode == "stagewise" or name == "op.run_length.tsfile_int64.v1":
            return unified._unwrap_stage_source(data)[0]
        return data

    def execute(name, sid):
        data, point, label, prefix = states[sid]
        source = kernel_input(name, data)
        row = dict(action=name, operator_id=descriptors[name].operator_id,
                   input_id=sid, case=label, prefix=" | ".join(prefix),
                   input_kind=data.kind, kernel_kind=source.kind, max_point=point,
                   input_bytes=len(data.payload), kernel_input_bytes=len(source.payload),
                   kernel_input_id=sha(source.kind.encode() + source.payload),
                   payload_bytes="", payload_sha256="", transport_bytes="",
                   input_to_payload_ratio="", roundtrip=False, status="", error="")
        encoded = None
        try:
            encoded = unified.run_action(name, data, max_point=point)
            _, payload = unified._unpack_result(encoded)
            row.update(payload_bytes=len(payload), payload_sha256=sha(payload),
                       transport_bytes=len(encoded.payload),
                       input_to_payload_ratio=(len(source.payload) / len(payload) if payload else "inf"))
            if unified.run_action(name, encoded, reverse=True) != data:
                raise AssertionError("inverse did not reproduce the exact incoming ActionData")
            row.update(roundtrip=True, status="ok")
        except Exception as error:
            row.update(status=("roundtrip_failure" if isinstance(error, AssertionError)
                               else "rejected_or_execution_error"),
                       error=f"{type(error).__name__}: {error}"[:700])
        index = len(measurements)
        measurements.append(row)
        attempts[name, sid] = index
        if row["status"] == "ok":
            good[name][sid] = index
            return encoded
        return None

    sequences = sorted({
        tuple(primitive.ACTION_CANONICAL_NAMES.get(name, name) for name in sequence)
        for sequence in unified._INDEPENDENT_STAGEWISE_SEQUENCES
    })
    for case_index, (label, typed, point) in enumerate(cases):
        print(f"trace {case_index + 1}/{len(cases)}: {label}", flush=True)
        roots = [typed] if typed.kind == "bytes" else [typed, unified.ActionData.bytes(typed.payload)]
        for root in roots:
            sid = state_id(root, point, label, ())
            case_rows.append(dict(case=label, kind=root.kind, count=root.count,
                                  max_point=point, payload_sha256=sha(root.payload), input_id=sid))
            for family, names in families.items():
                if any(root.kind in descriptors[name].accepted_input_kinds for name in names):
                    pools[family].add(sid)
        # Cache only this case's traced prefixes. Cross-probe measurements store
        # hashes/lengths, not every generated output buffer.
        prefix_cache = {}
        for sequence in sequences:
            if excluded.intersection(sequence):
                continue
            first = descriptors[sequence[0]]
            root = next((r for r in roots if r.kind in first.accepted_input_kinds), None)
            if root is None:
                continue
            current = root
            for step, name in enumerate(sequence):
                sid = state_id(current, point, label, sequence[:step])
                pools[descriptors[name].operator_id].add(sid)
                key = (name, sid)
                if key not in prefix_cache:
                    prefix_cache[key] = execute(name, sid)
                result = prefix_cache[key]
                if result is None:
                    row = measurements[attempts[key]]
                    trace_errors.append(dict(case=label, sequence=" | ".join(sequence),
                                             failed_action=name, status=row["status"], error=row["error"]))
                    break
                trace_success[name] += 1
                current = result

    for family, names in families.items():
        print(f"cross {family}: {len(names)} actions x {len(pools[family])} inputs", flush=True)
        for sid in sorted(pools[family]):
            data = states[sid][0]
            for name in names:
                if name in excluded or (name, sid) in attempts:
                    continue
                source = kernel_input(name, data)
                if source.kind not in descriptors[name].accepted_input_kinds:
                    continue
                execute(name, sid)

    pairs, family_rows = [], []
    for family, names in families.items():
        family_pairs = []
        for a, b in itertools.combinations(names, 2):
            common, equal, wire_equal, different = pair_result(good[a], good[b], measurements)
            if {a, b} & excluded:
                status = "excluded_camel"
            elif different:
                status = "different_size_witness"
            elif common:
                status = "same_payload_all_common" if len(wire_equal) == len(common) else "same_size_all_common"
            elif not set(descriptors[a].accepted_input_kinds) & set(descriptors[b].accepted_input_kinds):
                status = "declared_input_kinds_disjoint"
            else:
                status = "no_common_valid_input"
            witness = different[0] if different else common[0] if common else ""
            pair = dict(operator_id=family, action_a=a, action_b=b, status=status,
                        common_inputs=len(common), same_size_inputs=len(equal),
                        same_payload_inputs=len(wire_equal), different_size_inputs=len(different),
                        only_a_valid=len(set(good[a]) - set(good[b])),
                        only_b_valid=len(set(good[b]) - set(good[a])), witness_input_id=witness,
                        a_payload_bytes=measurements[good[a][witness]]["payload_bytes"] if witness else "",
                        b_payload_bytes=measurements[good[b][witness]]["payload_bytes"] if witness else "")
            pairs.append(pair)
            family_pairs.append(pair)
        counts = Counter(p["status"] for p in family_pairs)
        family_rows.append(dict(operator_id=family, actions=len(names), pairs=len(family_pairs),
                                actions_roundtripped=sum(bool(good[name]) for name in names),
                                different_size_witness=counts["different_size_witness"],
                                same_payload_all_common=counts["same_payload_all_common"],
                                same_size_all_common=counts["same_size_all_common"],
                                declared_input_kinds_disjoint=counts["declared_input_kinds_disjoint"],
                                no_common_valid_input=counts["no_common_valid_input"],
                                excluded_camel=counts["excluded_camel"]))

    assert len(pairs) == sum(len(v) * (len(v) - 1) // 2 for v in families.values())
    assert len({(p["action_a"], p["action_b"]) for p in pairs}) == len(pairs)
    action_rows = [dict(policy_action_id=s.action_id, action=s.name, operator_id=s.operator_id,
                        execution_mode=descriptors[s.name].execution_mode,
                        input_kinds="|".join(descriptors[s.name].accepted_input_kinds),
                        successful_inputs=len(good[s.name]), sequence_visits=trace_success[s.name],
                        excluded=s.name in excluded) for s in specs]
    state_rows = [dict(input_id=sid, kind=data.kind, count=data.count, max_point=point,
                       bytes=len(data.payload), sha256=sha(data.payload), case=label,
                       prefix=" | ".join(prefix))
                  for sid, (data, point, label, prefix) in states.items()]
    summary = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                   registry_actions=len(specs), families=len(families), total_pairs=len(pairs),
                   pair_status=dict(Counter(p["status"] for p in pairs)),
                   measurement_status=dict(Counter(m["status"] for m in measurements)),
                   actions_roundtripped=sum(bool(good[s.name]) for s in specs),
                   untested_actions=[s.name for s in specs if not good[s.name]],
                   excluded_actions=sorted(excluded), source_cases=len(cases),
                   case_kinds=dict(Counter(c[1].kind for c in cases)),
                   row_limits=row_limits, registered_sequences=len(sequences),
                   trace_failures=len(trace_errors), python=sys.executable,
                   platform=platform.platform(), elapsed_seconds=time.monotonic() - started,
                   source_hashes=realdata._exact_stage_artifact_hashes(),
                   audit_script_sha256=sha(Path(__file__).read_bytes()),
                   boundaries=[
                       "Same family = canonical operator_id; no actions are merged by this audit.",
                       "Common means identical public ActionData, max_point, and consumed kernel bytes/kind.",
                       "All numeric source NaNs removed; Camel actions explicitly excluded by user policy.",
                       "Two real-prefix lengths are not independent datasets; generated data are excluded from the current scope.",
                       "Same-size/all-common and same-payload/all-common are bounded observations, not universal identities.",
                       "Payload excludes UAR1 transport; transport_bytes includes it. Required context is not claimed free.",
                       "Rejected cross-packets are applicability evidence, not proof of a broken action.",
                       "No common valid input is inconclusive, not a distinct-compression-ratio claim.",
                   ])
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in [("pairs.csv", pairs), ("families.csv", family_rows),
                           ("actions.csv", action_rows), ("measurements.csv", measurements),
                           ("inputs.csv", state_rows), ("cases.csv", case_rows),
                           ("trace_failures.csv", trace_errors)]:
        dump_csv(output / filename, rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_hashes"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 2048])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory; existing results are never overwritten")
    self_check()
    original_run = subprocess.run

    def bounded_run(*args, **kwargs):
        kwargs.setdefault("timeout", 15)
        return original_run(*args, **kwargs)

    # Limit malformed cross-packet CLI probes without changing production code.
    with patch("subprocess.run", bounded_run):
        audit(args.output, args.rows)


if __name__ == "__main__":
    main()
