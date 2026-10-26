#!/usr/bin/env python3
"""Supplement the frozen ratio audit with BOS64's correct integer mode.

This does not update the original pair table or modify production code.
Compare max_point=None with the earlier max_point=0 diagnostic control.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import audit_current_action_ratio_identity as original_audit
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "analysis_outputs/action_ratio_identity_225_20260905_v1"
DEFAULT_OUTPUT = ROOT / "analysis_outputs/bos64_integer_ratio_supplement_20260905_v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing results: {output}")
    frozen_paths = [FROZEN / name for name in ("summary.json", "pairs.csv", "cases.csv")]
    frozen_hashes = {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in frozen_paths}
    original_case_rows = {row["case"]: row for row in csv_rows(FROZEN / "cases.csv")
                          if row["kind"] == "int64"}
    cases, manifests = original_audit.make_cases([128, 2048])
    cases = [(label, source) for label, source, _ in cases if source.kind == "int64"]
    if len(cases) != 10 or set(original_case_rows) != {label for label, _ in cases}:
        raise AssertionError("expected exactly the frozen audit's ten integer source cases")
    sequence = primitive.BOS64_INT_ACTIONS
    records, stages = [], []
    benchmark = original_audit.realdata.benchmark

    for label, source in cases:
        old = original_case_rows[label]
        if sha(source.payload) != old["payload_sha256"] or source.count != int(old["count"]):
            raise AssertionError(f"frozen source input changed: {label}")
        canonical = primitive.encode_exact_bos64_action_pipeline(source.payload, "int")
        canonical_roundtrip = primitive.decode_exact_bos64_action_pipeline(canonical, "int") == source.payload
        # This Python benchmark adapter is independent of the two C++ stage calls.
        column = benchmark.Column("supplement", 0, label, "integer", 0,
                                  tuple(int(v) for v in source.as_ints()))
        reference, reason = benchmark.specialist_artifact(column, "BOS")
        if reference is None:
            raise AssertionError(f"original BOS benchmark unavailable: {reason}")
        reference_roundtrip = benchmark._values_equal(column, reference.decode(reference.payload))

        for point in (None, 0):
            mode = "None_integer" if point is None else "0_scaled_control"
            row = dict(case=label, mode=mode, max_point_json=json.dumps(point),
                       input_kind=source.kind, input_count=source.count,
                       raw_bytes=len(source.payload), raw_sha256=sha(source.payload),
                       source_matches_frozen=True, status="", failed_action="", error="",
                       forward_steps=0, exact_stage_inverses=0, full_chain_inverse=False,
                       payload_bytes="", transport_bytes="", payload_sha256="",
                       canonical_bytes=len(canonical), canonical_roundtrip=canonical_roundtrip,
                       canonical_payload_identical=False,
                       benchmark_implementation=reference.implementation,
                       benchmark_bytes=len(reference.payload), benchmark_roundtrip=reference_roundtrip,
                       benchmark_payload_identical=False)
            current = source
            for index, name in enumerate(sequence):
                stage = dict(case=label, mode=mode, step=index + 1, action=name,
                             input_kind=current.kind, input_bytes=len(current.payload),
                             input_sha256=sha(current.payload), payload_bytes="",
                             transport_bytes="", forward_ok=False, inverse_exact=False,
                             error="")
                try:
                    encoded = unified.run_action(name, current, max_point=point)
                    payload = unified._unpack_result(encoded)[1]
                    stage.update(forward_ok=True, payload_bytes=len(payload),
                                 transport_bytes=len(encoded.payload))
                    row["forward_steps"] += 1
                    if unified.run_action(name, encoded, reverse=True) != current:
                        raise AssertionError("stage inverse changed its incoming ActionData")
                    stage["inverse_exact"] = True
                    row["exact_stage_inverses"] += 1
                    current = encoded
                except Exception as error:
                    stage["error"] = f"{type(error).__name__}: {error}"
                    row.update(status="forward_error" if not stage["forward_ok"] else "inverse_error",
                               failed_action=name, error=stage["error"])
                    stages.append(stage)
                    break
                stages.append(stage)
            else:
                payload = unified._unpack_result(current)[1]
                row.update(full_chain_inverse=(unified.run_actions(sequence, current, reverse=True) == source),
                           payload_bytes=len(payload), transport_bytes=len(current.payload),
                           payload_sha256=sha(payload), canonical_payload_identical=(payload == canonical),
                           benchmark_payload_identical=(payload == reference.payload), status="ok")
            records.append(row)

    correct = [row for row in records if row["mode"] == "None_integer"]
    control = [row for row in records if row["mode"] == "0_scaled_control"]
    correct_passes = sum(row["status"] == "ok" and row["exact_stage_inverses"] == 2
                         and row["full_chain_inverse"] and row["canonical_roundtrip"]
                         and row["benchmark_roundtrip"] and row["canonical_payload_identical"]
                         and row["benchmark_payload_identical"] for row in correct)
    control_failures = sum(row["status"] == "forward_error" and row["forward_steps"] == 1
                           and row["exact_stage_inverses"] == 1
                           and row["failed_action"] == sequence[1]
                           and "wrong BOS64 scaled delta packet type" in row["error"] for row in control)
    frozen_unchanged = all(sha((ROOT / name).read_bytes()) == digest
                           for name, digest in frozen_hashes.items())
    source_paths = [Path(__file__), Path(original_audit.__file__),
                    Path(primitive.__file__), Path(unified.__file__),
                    ROOT / "scripts/benchmark_86_combinations.py", ROOT / "cpp/src/bos_exact_stages.cpp"]
    summary = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                   scope="ten integer inputs from the frozen current-action ratio audit",
                   source_cases=len(cases), real_integer_slices=8, synthetic_integer_cases=2,
                   actions=list(sequence), none_mode_case_passes=correct_passes,
                   none_mode_stage_inverse_passes=sum(row["exact_stage_inverses"] for row in correct),
                   none_mode_full_chain_passes=sum(row["full_chain_inverse"] for row in correct),
                   canonical_payload_identical=sum(row["canonical_payload_identical"] for row in correct),
                   independent_benchmark_payload_identical=sum(row["benchmark_payload_identical"] for row in correct),
                   point0_control_expected_forward_rejections=control_failures,
                   frozen_audit_files_unchanged=frozen_unchanged,
                   frozen_audit_hashes=frozen_hashes, python=sys.executable,
                   source_hashes={str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in source_paths},
                   boundaries=[
                       "This supplements a harness mode omission; no production or original audit files were changed.",
                       "max_point=None selects the integer BOS64 path; 0 selects the scaled mode in the current runtime.",
                       "Original 2036-pair statistics remain frozen; these ten cases are not inserted into that pair table.",
                       "Canonical C++ composition shares the same stage kernels; Python benchmark parity is checked separately.",
                       "Payload and UAR1 transport byte counts remain separate.",
                       "This is a bounded ten-input check, not universal implementation equivalence or a deduplication result.",
                   ])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "case_comparisons.csv", records)
    write_csv(output / "stage_roundtrips.csv", stages)
    (output / "data_manifest.json").write_text(json.dumps(
        [m for m in manifests if m["logical_type"] == "integer"], ensure_ascii=False, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if correct_passes != 10 or control_failures != 10 or not frozen_unchanged:
        raise AssertionError(f"supplement validation failed; inspect {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = audit(args.output.resolve())
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"source_hashes", "frozen_audit_hashes"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
