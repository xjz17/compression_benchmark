#!/usr/bin/env python3
"""Diagnose native-LZ stage-size collisions without editing the action registry.

The window/match packets bind parameters; greedy parse performs the search.
Final codec bytes use the existing registered context (preset and raw length),
not an added header invented by this diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random

import primitive_operator_action_runtime as primitive

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis_outputs/native_lz_ratio_equivalence_20260905_v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chain(raw: bytes, window: str, match: str, parse: str, tokens: dict[int, str]):
    encode = primitive.encode_exact_native_lz_stage
    decode = primitive.decode_exact_native_lz_stage
    window_packet = encode(window, raw)
    match_packet = encode(match, window_packet)
    parse_packet = encode(parse, match_packet)
    minimum, _maximum = primitive._native_match_bounds_from_action(match)
    token = tokens[minimum]
    wire = encode(token, parse_packet)
    context = primitive.native_lz_stage_packet_context(parse_packet)
    assert decode(window, window_packet) == raw
    assert decode(match, match_packet) == window_packet
    assert decode(parse, parse_packet) == match_packet
    restored_parse = decode(token, wire, context=context)
    assert restored_parse == parse_packet
    restored = decode(window, decode(match, decode(parse, restored_parse)))
    assert restored == raw
    return window_packet, match_packet, parse_packet, wire


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    actions = sorted(primitive.EXACT_NATIVE_ACTION_BY_NAME)
    role = primitive.exact_native_lz_stage_role
    windows = [name for name in actions if role(name) == "window"]
    matches = [name for name in actions if role(name) == "match"]
    parse_action = next(name for name in actions if role(name) == "parse")
    tokens = {primitive._native_bias_from_action(name): name for name in actions if role(name) == "token_format"}
    assert len(windows) == 14 and len(matches) == 42
    fixed_window = "op.match_window_model.native_lz.window65535"
    fixed_match = "op.lz_match.native_lz.aligned8.min3.max258"
    assert fixed_window in windows and fixed_match in matches
    far = random.Random(1729).randbytes(16392)
    fixtures = {
        "zeros1024": bytes(1024),
        "match_repeat4096": bytes(range(64)) * 64,
        "window_far_repeat16648": far + far[:256],
    }

    measurements, saved = [], {}
    for family, family_actions in (("window", windows), ("match", matches)):
        for fixture, raw in fixtures.items():
            stage_packets = []
            for name in family_actions:
                window = name if family == "window" else fixed_window
                match = name if family == "match" else fixed_match
                window_packet, match_packet, parse_packet, wire = chain(raw, window, match, parse_action, tokens)
                stage = window_packet if family == "window" else match_packet
                stage_packets.append(stage)
                saved[family, name, fixture] = (parse_packet, wire)
                measurements.append({
                    "family": family, "action_name": name, "fixture": fixture,
                    "raw_bytes": len(raw), "stage_input_bytes": len(raw) if family == "window" else len(window_packet),
                    "stage_output_bytes": len(stage), "stage_output_sha256": sha(stage),
                    "final_wire_bytes": len(wire), "final_wire_sha256": sha(wire),
                    "fixed_other_action": fixed_match if family == "window" else fixed_window,
                    "four_stage_inverses": True, "complete_reverse": True,
                })
            assert len({len(packet) for packet in stage_packets}) == 1
            assert len(set(stage_packets)) == len(family_actions)
            assert all(len(packet) == len(raw) + primitive._NATIVE_STAGE_PACKET_HEADER.size for packet in stage_packets)

    classes: dict[tuple[int, int], list[str]] = {}
    for name in matches:
        minimum, maximum = primitive._native_match_bounds_from_action(name)
        effective = (max(8, (minimum + 7) // 8 * 8), maximum // 8 * 8)
        classes.setdefault(effective, []).append(name)
    assert len(classes) == 19
    config_rows = []
    pair_rows = []
    for (effective_min, effective_max), members in sorted(classes.items()):
        for name in members:
            minimum, maximum = primitive._native_match_bounds_from_action(name)
            config_rows.append({
                "action_name": name, "registered_min": minimum, "registered_max": maximum,
                "effective_min": effective_min, "effective_max": effective_max,
                "class_size": len(members),
                **{fixture + "_final_bytes": len(saved["match", name, fixture][1]) for fixture in fixtures},
            })
        for left, right in itertools.combinations(members, 2):
            same_tokens = same_lengths = same_wires = True
            wire_equal_by_fixture = []
            for fixture in fixtures:
                lp, lw = saved["match", left, fixture]
                rp, rw = saved["match", right, fixture]
                same_tokens &= (primitive._unpack_native_stage_packet(lp)[-1] ==
                                primitive._unpack_native_stage_packet(rp)[-1])
                same_lengths &= len(lw) == len(rw)
                same_wires &= lw == rw
                wire_equal_by_fixture.append(lw == rw)
            assert same_tokens and same_lengths
            pair_rows.append({
                "left_action": left, "right_action": right,
                "effective_min": effective_min, "effective_max": effective_max,
                "same_token_sequence_all_fixtures": same_tokens,
                "same_final_length_all_fixtures": same_lengths,
                "same_final_wire_all_fixtures": same_wires,
                "wire_equal_by_fixture": json.dumps(wire_equal_by_fixture),
            })
    assert len(pair_rows) == 61

    def counterexample(family: str, left: str, right: str, fixture: str) -> dict:
        rows = [row for row in measurements if row["family"] == family and row["fixture"] == fixture
                and row["action_name"] in (left, right)]
        assert len(rows) == 2
        assert rows[0]["stage_output_bytes"] == rows[1]["stage_output_bytes"]
        assert rows[0]["stage_output_sha256"] != rows[1]["stage_output_sha256"]
        assert rows[0]["final_wire_bytes"] != rows[1]["final_wire_bytes"]
        return {"fixture": fixture, "left": next(row for row in rows if row["action_name"] == left),
                "right": next(row for row in rows if row["action_name"] == right)}

    window_counterexample = counterexample(
        "window", "op.match_window_model.native_lz.window16384",
        "op.match_window_model.native_lz.window19456", "window_far_repeat16648")
    match_counterexample = counterexample(
        "match", "op.lz_match.native_lz.aligned8.min3.max18", fixed_match, "match_repeat4096")
    summary = {
        "status": "PASS", "window_actions": len(windows), "match_actions": len(matches),
        "fixtures": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in fixtures.items()},
        "stage_header_bytes": primitive._NATIVE_STAGE_PACKET_HEADER.size,
        "chains_checked": len(measurements), "single_stage_inverse_checks": len(measurements) * 4,
        "complete_reverse_checks": len(measurements),
        "window_distinct_packets_but_one_size_per_fixture": True,
        "match_distinct_packets_but_one_size_per_fixture": True,
        "same_stage_size_pairs": len(windows) * (len(windows) - 1) // 2 + len(matches) * (len(matches) - 1) // 2,
        "effective_match_search_classes": len(classes), "within_class_pairs": len(pair_rows),
        "within_class_final_wire_equal_all_fixtures_pairs": sum(row["same_final_wire_all_fixtures"] for row in pair_rows),
        "within_class_same_length_but_wire_diff_pairs": sum(not row["same_final_wire_all_fixtures"] for row in pair_rows),
        "configuration_redundancy_for_search_under_fixed_window": len(matches) - len(classes),
        "window_counterexample": window_counterexample, "match_counterexample": match_counterexample,
        "no_registry_changes": True,
        "scope": "All 14/42 stages checked; 19 search classes follow aligned-8 source semantics under identical window. Equal stage length does not prove whole implementation equivalence. Different min values remain token-format biases, so aliases cannot simply be deleted.",
        "source_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in (
            Path(__file__).resolve(), ROOT / "scripts/primitive_operator_action_runtime.py",
            ROOT / "backend/myapp/native_lz_family.py")},
    }
    csv_write(output / "stage_and_terminal_measurements.csv", measurements)
    csv_write(output / "match_configuration_classes.csv", config_rows)
    csv_write(output / "within_class_pairs.csv", pair_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    wc, mc = window_counterexample, match_counterexample
    report = f"""# Native-LZ ratio-equivalence diagnostic

Reproduce: `python3 scripts/audit_native_lz_ratio_equivalence.py`.

Checked 14 window actions and 42 match actions on 3 deterministic fixtures.
All {len(measurements)} chains, {len(measurements) * 4} individual stage inverses,
and {len(measurements)} complete reverse operations passed.

Every window output is raw-size + 25 bytes; every match output keeps that size.
All 14 window packets differ, and all 42 match packets differ, on each fixture.
The resulting 952 same-stage-size pairs cannot establish implementation identity.

| Counterexample | Same stage bytes | Left final bytes | Right final bytes |
| --- | ---: | ---: | ---: |
| window16384 / window19456; Random(1729).randbytes(16392) + first 256 bytes | {wc['left']['stage_output_bytes']} | {wc['left']['final_wire_bytes']} | {wc['right']['final_wire_bytes']} |
| min3.max18 / min3.max258; bytes(range(64))*64 | {mc['left']['stage_output_bytes']} | {mc['left']['final_wire_bytes']} | {mc['right']['final_wire_bytes']} |

The actual aligned-8 matcher rounds match length down to a multiple of 8.
For current min=3/4/5, the effective minimum is 8; effective max is floor(max/8)*8.
The 42 configurations therefore form 19 search classes at a fixed window,
with 23 redundant search configurations and 61 within-class pairs.
All 61 pairs produced identical semantic tokens and final lengths on all fixtures;
{summary['within_class_same_length_but_wire_diff_pairs']} pairs had different final wire bytes on at least one fixture,
because the original minimum remains the token-format length bias.

This diagnostic does not merge IDs or claim that all same-size stages are equal.
The final native wire inverse uses the existing preset and raw-length context;
no additional diagnostic header is counted as codec bytes.
The source-level search equivalence does not make intermediate packets or all
downstream interfaces interchangeable: original min/max metadata is preserved.

See `match_configuration_classes.csv` for all 42 configurations,
`within_class_pairs.csv` for all 61 pairs, and `stage_and_terminal_measurements.csv`
for all 168 complete measurements and wire hashes. `summary.json` records source
and fixture hashes. No action registry was modified.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "status", "window_actions", "match_actions", "chains_checked", "single_stage_inverse_checks",
        "effective_match_search_classes", "within_class_pairs", "within_class_same_length_but_wire_diff_pairs")}, indent=2))
    print(f"Window final bytes: {wc['left']['final_wire_bytes']} vs {wc['right']['final_wire_bytes']}")
    print(f"Match final bytes: {mc['left']['final_wire_bytes']} vs {mc['right']['final_wire_bytes']}")
    print(output)


if __name__ == "__main__":
    main()
