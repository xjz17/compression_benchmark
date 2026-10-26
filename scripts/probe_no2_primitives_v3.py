#!/usr/bin/env python3
"""Bounded nonrepeated ACF2 NO2 probe; no shared runtime/codec modifications.

This control complements, but does not include, the new split_common primitive.
All reported candidates are full-file encodes followed by byte-exact decoding.
Entropy estimates allocate measurements only and are never compressed results.
"""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from canonical206_data import load_smallest_two
from atomic_transform_extensions_v2 import apply_ops
from atomic_composition_frame_v2 import encode_prepared, decode_frame, inspect_frame, make_body
from run_atomic_composition_research_v2 import entropy_estimates

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis_outputs/primitive_composition_research_20260906_v1"
OUT = RUN / "probe_no2_v3"
LIMIT = 1000
TARGET = 7571
TERMINALS = ["bit16", "bit12", "bit8", "range1", "sparse1"]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def op(name, **parameters):
    return {"op": name, **parameters}


def key(specs, terminal):
    defaults = {"delta": {"lag": 1, "order": 1}, "xor": {"lag": 1},
                "compact": {"width": "auto"}, "bitpack": {"bits": "auto"},
                "dictionary": {"max_entries": 4096}, "shuffle": {"width": 8},
                "bittranspose": {"width": 8}, "block_for": {"block_size": 128},
                "block_bitpack": {"block_size": 128}}
    normalized = [{**defaults.get(s["op"], {}), **s} for s in specs]
    return json.dumps([normalized, terminal], sort_keys=True, separators=(",", ":"))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    c = next(c for c in load_smallest_two(ROOT) if c["dataset"] == "Air-Quality" and c["numeric_index"] == 8)
    raw = c["raw"]
    assert sha(raw) == "38b536c6fdb6b28703b200d42fa74917caf24641df35fbb0e7cb73e745c5511c"
    frozen = [ROOT / "scripts" / n for n in ["atomic_transform_runtime_v1.py", "atomic_transform_extensions_v2.py",
              "atomic_composition_frame_v1.py", "atomic_composition_frame_v2.py"]]
    before = {str(p): sha(p.read_bytes()) for p in frozen}
    excluded = set()
    snapshots = {}
    for run in ["round1", "round2_smoke", "round2"]:
        path = RUN / run / "traces/Air-Quality__08.jsonl"
        blob = path.read_bytes()
        snapshots[run] = {"bytes": len(blob), "sha256": sha(blob)}
        for line in blob.splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok" and r.get("family") == "pure":
                excluded.add(key(r["specs"], r["terminals"]))
    starts = [[], [op("rle")], [op("dictionary")], [op("zigzag"), op("dictionary")],
              [op("delta"), op("zigzag"), op("rle")],
              [op("delta"), op("zigzag"), op("sparsezero")],
              [op("dictionary"), op("rle")], [op("rle"), op("dictionary")]]
    predictors = [[]] + [[op("delta", lag=lag), op("zigzag")] for lag in [1, 3, 6, 12, 24, 48]]
    predictors += [[op("xor", lag=lag)] for lag in [1, 3, 6, 12, 24]]
    packers = [[op("compact")], [op("bitpack")]] + [[op("block_bitpack", block_size=b)] for b in [16, 64, 256]]
    reorders = [[], [op("shuffle", width=2)], [op("bittranspose", width=1)],
                [op("bittranspose", width=2)], [op("bittranspose", width=4)]]
    pool, body_seen = [], set()
    counters = collections.Counter()
    for start, predictor, packer, reorder in itertools.product(starts, predictors, packers, reorders):
        specs = start + predictor + packer + reorder
        if len(specs) + 1 > 10:
            continue
        if all(key(specs, [terminal]) in excluded for terminal in TERMINALS):
            continue
        counters["representation_proposals"] += 1
        try:
            payload, metadata = apply_ops(raw, specs)
            body = make_body(payload, metadata)
            body_hash = sha(body)
            if body_hash in body_seen:
                counters["duplicate_full_body_skipped"] += 1
                continue
            body_seen.add(body_hash)
            estimates = entropy_estimates(body)
            pool.append((min(estimates), specs, payload, metadata, body_hash))
        except ValueError:
            counters["representation_rejected"] += 1
    pool.sort(key=lambda row: row[0])
    dump(OUT / "manifest.json", {"input": {k: v for k, v in c.items() if k != "raw"},
         "limit": LIMIT, "target_bytes_strictly_below": TARGET, "frame": "ACF2", "split_common_included": False,
         "runtime_hashes": before, "prior_snapshot_hashes": snapshots, "prior_candidate_keys_excluded": len(excluded),
         "unique_full_bodies_screened": len(pool), "screening_is_not_measurement": True})
    best = None
    with (OUT / "candidates.jsonl").open("x") as trace:
        for estimate, specs, payload, metadata, body_hash in pool:
            for terminal in TERMINALS:
                candidate_key = key(specs, [terminal])
                if candidate_key in excluded:
                    counters["previous_candidate_skipped"] += 1
                    continue
                excluded.add(candidate_key)
                counters["full_file_attempts"] += 1
                record = {"specs": specs, "terminals": [terminal], "depth": len(specs) + 1,
                          "body_sha256": body_hash, "screening_estimate_not_result": estimate}
                t0 = time.perf_counter()
                try:
                    frame = encode_prepared(raw, c["kind"], specs, [terminal], payload, metadata)
                    encode_s = time.perf_counter() - t0
                    assert decode_frame(frame) == raw
                    record.update(status="ok", bytes=len(frame), ratio=len(raw) / len(frame), encode_sec=encode_s,
                                  full_roundtrip_sec=time.perf_counter() - t0, frame_sha256=sha(frame))
                    counters["full_file_verified"] += 1
                    if best is None or len(frame) < best["bytes"]:
                        best = {**record, "metadata": metadata, "inspection": inspect_frame(frame)}
                        (OUT / "best.acf2").write_bytes(frame)
                        dump(OUT / "best.json", best)
                except (ValueError, RuntimeError) as exc:
                    record.update(status="rejected", reason=str(exc))
                    counters["full_file_rejected"] += 1
                trace.write(json.dumps(record, separators=(",", ":")) + "\n")
                if counters["full_file_attempts"] % 100 == 0:
                    trace.flush()
                    print(json.dumps({"attempts": counters["full_file_attempts"], "best_bytes": best["bytes"] if best else None,
                                      "elapsed_s": time.perf_counter() - started}), flush=True)
                if counters["full_file_attempts"] >= LIMIT:
                    break
            if counters["full_file_attempts"] >= LIMIT:
                break
    after = {str(p): sha(p.read_bytes()) for p in frozen}
    assert before == after
    summary = {"status": "COMPLETE", "counters": dict(counters), "best": best,
               "target_bytes": TARGET, "strict_win": best is not None and best["bytes"] < TARGET,
               "frozen_sources_unchanged": True, "elapsed_s": time.perf_counter() - started,
               "not_global_optimum": True, "split_common_included": False,
               "split_common_followup": "delegated_to_parent_ACF3_not_measured_by_this_probe"}
    dump(OUT / "SUMMARY.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "best"}), flush=True)


if __name__ == "__main__":
    main()
