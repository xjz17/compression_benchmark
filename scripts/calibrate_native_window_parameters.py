#!/usr/bin/env python3
"""Finite native-LZ window calibration using production parse/pack C++ stages.

Reuses the previous sixty small numeric calibration blocks and adds one 64KiB
block per column plus a 128KiB block for each longer Metro-Traffic column. This
is default calibration, not a held-out evaluation or joint window/min/max fit.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import time

import calibrate_native_lz_parameters as lz

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis_outputs/native_window_parameters_20260908_v1"
DEFAULTS = ROOT / "benchmark/reference/actions/native_window_defaults.json"
PREVIOUS = ROOT / "analysis_outputs/native_lz_parameters_20260908_v1/sample_manifest.json"
REGISTRY = ROOT / "benchmark/reference/actions/canonical156.json"

HARNESS = r'''
#include "native_lz_action_stages.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
using namespace native_lz_action;
using Clock = std::chrono::steady_clock;
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
void put(Bytes& b,std::uint64_t v,unsigned n) {
  for (unsigned i=0;i<n;++i) b.push_back(std::uint8_t(v>>(i*8)));
}
std::uint64_t get(const Bytes& b,std::size_t& p,unsigned n) {
  require(p<=b.size() && n<=b.size()-p,"truncated calibration frame");
  std::uint64_t v=0; for (unsigned i=0;i<n;++i) v|=std::uint64_t(b[p++])<<(8*i);
  return v;
}
Bytes encode(const Bytes& raw,unsigned window) {
  const auto wire=pack(parse(raw,window,8,256),3);
  Bytes frame{'N','L','C','1'};
  put(frame,window,4); put(frame,8,4); put(frame,256,4); put(frame,3,4);
  put(frame,raw.size(),8); put(frame,wire.size(),8);
  frame.insert(frame.end(),wire.begin(),wire.end()); return frame;
}
Bytes decode(const Bytes& frame) {
  require(frame.size()>=36 && std::string(frame.begin(),frame.begin()+4)=="NLC1",
          "invalid calibration frame magic");
  std::size_t pos=4;
  const auto window=get(frame,pos,4), minimum=get(frame,pos,4), maximum=get(frame,pos,4);
  const auto bias=get(frame,pos,4), count=get(frame,pos,8), size=get(frame,pos,8);
  require(window>=1 && window<=65535 && minimum==8 && maximum==256 && bias==3,
          "invalid calibration parameters");
  require(size==frame.size()-pos,"calibration payload size mismatch");
  Bytes wire(frame.begin()+pos,frame.end());
  return expand(unpack(wire,bias,count),count);
}
std::int64_t ns(Clock::time_point a,Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}
int main(int argc,char** argv) {
  try {
    require(argc==4,"usage: harness samples.tsv candidates.tsv repetitions");
    const unsigned repeats=std::stoul(argv[3]); require(repeats>=3 && repeats%2,"odd repeats >=3 required");
    struct Sample { std::string id; Bytes raw; };
    std::vector<Sample> samples; std::ifstream manifest(argv[1]); std::string id,path;
    while (manifest>>id>>path) {
      std::ifstream in(path,std::ios::binary); require(bool(in),"sample open failed");
      Bytes raw((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
      require(!raw.empty(),"empty sample"); samples.push_back({id,std::move(raw)});
    }
    require(!samples.empty(),"empty sample manifest");
    std::vector<unsigned> candidates; std::ifstream params(argv[2]); unsigned window;
    while (params>>window) candidates.push_back(window);
    require(candidates.size()==14,"expected 14 distinct historical window candidates");
    for (int i=0;i<3;++i) require(decode(encode(samples[0].raw,32768))==samples[0].raw,"warmup failed");
    std::cout<<"sample_id,window_bytes,raw_bytes,wire_bytes,frame_bytes,encode_ns_median,decode_ns_median,round_trips\n";
    for (const auto& s:samples) for (auto w:candidates) {
      std::vector<std::int64_t> enc,dec; std::size_t frame_size=0;
      for (unsigned r=0;r<repeats;++r) {
        const auto begin=Clock::now(); const auto frame=encode(s.raw,w); const auto middle=Clock::now();
        const auto restored=decode(frame); const auto end=Clock::now();
        require(restored==s.raw,"bit-exact round trip failed");
        require(frame_size==0 || frame_size==frame.size(),"non-deterministic size"); frame_size=frame.size();
        enc.push_back(ns(begin,middle)); dec.push_back(ns(middle,end));
      }
      std::sort(enc.begin(),enc.end()); std::sort(dec.begin(),dec.end());
      std::cout<<s.id<<','<<w<<','<<s.raw.size()<<','<<frame_size-36<<','<<frame_size<<','
               <<enc[repeats/2]<<','<<dec[repeats/2]<<','<<repeats<<'\n';
    }
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    lz.write_json(path, value)


def collect_samples(out: Path) -> tuple[list[dict], list[dict]]:
    """Copy verified existing blocks, then select larger blocks independently."""
    prior = json.loads(PREVIOUS.read_text())
    samples = []
    sample_dir = out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in prior["samples"]:
        raw = Path(old["sample_path"]).read_bytes()
        if sha(raw) != old["sha256"] or len(raw) != old["raw_bytes"]:
            raise ValueError("previous sample changed")
        sample = dict(old)
        destination = sample_dir / f"{old['sample_id']}.bin"
        destination.write_bytes(raw)
        sample.update(sample_path=str(destination), cohort="previous_small_5456B")
        samples.append(sample)
    assert len(samples) == 60
    for dataset, fields in lz.FIELDS.items():
        path = ROOT / "orignial_datasets" / f"{dataset}.csv"
        source_hash = sha(path.read_bytes())
        old_source = next(x for x in prior["datasets"] if x["dataset"] == dataset)
        if source_hash != old_source["sha256"]:
            raise ValueError("source CSV differs from previous calibration")
        columns = {column: [] for column in fields}
        skipped = {column: 0 for column in fields}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle)):
                for column, kind in fields.items():
                    value_text = row[column].strip()
                    if not value_text or value_text.lower() in {"nan", "null", "none"}:
                        skipped[column] += 1
                        continue
                    if kind == "int64":
                        number = Decimal(value_text)
                        if not number.is_finite():
                            skipped[column] += 1
                            continue
                        if number != number.to_integral_value() or not -(1 << 63) <= number < (1 << 63):
                            raise ValueError(f"invalid int64 {dataset}/{column}/{row_number}")
                        packed = struct.pack("<q", int(number))
                    else:
                        number = float(value_text)
                        if not math.isfinite(number):
                            skipped[column] += 1
                            continue
                        packed = struct.pack("<d", number)
                    columns[column].append((row_number, packed))
        for column, kind in fields.items():
            values = columns[column]
            selections = [("front64KiB", 0, 8192)]
            if dataset == "Metro-Traffic":
                selections.append(("back128KiB", len(values) - 16384, 16384))
            for cohort, start, count in selections:
                if start < 0 or start + count > len(values):
                    raise ValueError(f"insufficient large-block values: {dataset}/{column}")
                chosen = values[start:start + count]
                raw = b"".join(value for _, value in chosen)
                sample_id = f"s{len(samples):03d}"
                destination = sample_dir / f"{sample_id}.bin"
                destination.write_bytes(raw)
                samples.append({"sample_id": sample_id, "dataset": dataset, "column": column,
                    "kind": kind, "byte_order": "little", "cohort": cohort, "region": cohort,
                    "source_path": str(path), "source_sha256": source_hash,
                    "valid_value_range_half_open": [start, start + count],
                    "csv_data_rows_zero_based": [row for row, _ in chosen],
                    "source_valid_values": len(values), "source_skipped_missing_or_nonfinite": skipped[column],
                    "sample_path": str(destination), "raw_bytes": len(raw), "sha256": sha(raw)})
    assert len(samples) == 86
    return samples, prior["datasets"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--defaults", type=Path, default=DEFAULTS)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 3 or args.repetitions % 2 == 0:
        parser.error("--repetitions must be odd and >=3")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SUMMARY.json").exists():
        parser.error("output already contains completed calibration; choose a new directory")
    registry_raw = REGISTRY.read_bytes()
    windows = sorted(set(map(int, re.findall(rb"op.match_window_model.native_lz.window(\d+)", registry_raw))))
    assert len(windows) == 14 and max(windows) == 65535
    started = time.monotonic()
    samples, sources = collect_samples(out)
    cohorts = sorted({s["cohort"] for s in samples})
    provenance = {"purpose": "finite default calibration only; not held-out performance evaluation",
        "datasets": sources, "samples": samples, "columns": 20, "blocks": len(samples),
        "total_raw_bytes": sum(s["raw_bytes"] for s in samples),
        "sampling": "Previous 60 front/middle/back 5456B blocks; one front 64KiB block for all 20 columns; one back 128KiB block for 6 Metro-Traffic columns. Blocks overlap previous small samples and are independent codec frames.",
        "conversion": "Explicit int64/float64 schema, little endian; skip/count missing/nonfinite values; preserve finite -0.0.",
        "previous_sample_manifest": str(PREVIOUS), "previous_sample_manifest_sha256": sha(PREVIOUS.read_bytes()),
        "cohorts": {c: {"blocks": sum(s["cohort"] == c for s in samples),
            "raw_bytes": sum(s["raw_bytes"] for s in samples if s["cohort"] == c)} for c in cohorts}}
    write_json(out / "sample_manifest.json", provenance)
    (out / "samples.tsv").write_text("".join(f"{s['sample_id']}\t{s['sample_path']}\n" for s in samples))
    candidates = windows.copy()
    random.Random(20260908).shuffle(candidates)
    (out / "candidates.tsv").write_text("".join(f"{w}\n" for w in candidates))
    cpp = out / "calibration_harness.cpp"
    cpp.write_text(HARNESS)
    snapshot_root = out / "source_snapshot"
    source_hashes, source_snapshots = {}, {}
    for source_path in (ROOT / "cpp/src/native_lz_action_stages.cpp",
                        ROOT / "cpp/src/native_lz_action_stages.hpp", Path(__file__), Path(lz.__file__), REGISTRY):
        relative = str(source_path.relative_to(ROOT))
        raw = source_path.read_bytes()
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_hashes[relative], source_snapshots[relative] = sha(raw), str(target)
    harness_snapshot = snapshot_root / "generated/calibration_harness.cpp"
    harness_snapshot.parent.mkdir(parents=True, exist_ok=True)
    harness_snapshot.write_bytes(cpp.read_bytes())
    source_hashes["generated/calibration_harness.cpp"] = sha(harness_snapshot.read_bytes())
    source_snapshots["generated/calibration_harness.cpp"] = str(harness_snapshot)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("C++ compiler not found")
    executable = out / "calibration_harness"
    command = [compiler, "-O3", "-std=c++17", "-I", str(snapshot_root / "cpp/src"), str(harness_snapshot),
        str(snapshot_root / "cpp/src/native_lz_action_stages.cpp"), "-o", str(executable)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = [str(executable), str(out / "samples.tsv"), str(out / "candidates.tsv"), str(args.repetitions)]
    with (out / "sample_results.csv").open("w") as handle:
        subprocess.run(run, stdout=handle, check=True, text=True)
    with (out / "sample_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(samples) * len(windows)
    sample_cohorts = {s["sample_id"]: s["cohort"] for s in samples}
    aggregates, seen = {}, set()
    for row in rows:
        window = int(row["window_bytes"])
        identity = (row["sample_id"], window)
        assert identity not in seen
        seen.add(identity)
        item = aggregates.setdefault(window, {"window_bytes": window, "raw_bytes": 0, "wire_bytes": 0,
            "frame_bytes": 0, "encode_ns_sum_of_block_medians": 0, "decode_ns_sum_of_block_medians": 0,
            "round_trips": 0, "blocks": 0, **{f"{c}_frame_bytes": 0 for c in cohorts}})
        for field in ("raw_bytes", "wire_bytes", "frame_bytes", "round_trips"):
            item[field] += int(row[field])
        item["blocks"] += 1
        item["encode_ns_sum_of_block_medians"] += int(row["encode_ns_median"])
        item["decode_ns_sum_of_block_medians"] += int(row["decode_ns_median"])
        item[f"{sample_cohorts[row['sample_id']]}_frame_bytes"] += int(row["frame_bytes"])
    ranked = sorted(aggregates.values(), key=lambda item: (item["frame_bytes"], item["window_bytes"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
        item["compressed_fraction"] = item["frame_bytes"] / item["raw_bytes"]
        item["compression_ratio_raw_over_frame"] = item["raw_bytes"] / item["frame_bytes"]
        item["encode_mib_per_second"] = item["raw_bytes"] / (1024 * 1024) / (item["encode_ns_sum_of_block_medians"] / 1e9)
        item["decode_mib_per_second"] = item["raw_bytes"] / (1024 * 1024) / (item["decode_ns_sum_of_block_medians"] / 1e9)
    with (out / "candidate_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    selected = ranked[0]
    manifest_hash = sha((out / "sample_manifest.json").read_bytes())
    results_hash = sha((out / "candidate_results.csv").read_bytes())
    criterion = {"objective": "minimize sum of complete calibration frame bytes over all 86 blocks",
        "tie_break": "ascending window_bytes; choose smallest window among equal byte totals",
        "timing_used_for_selection": False, "frame_header_bytes": 36,
        "frame_layout": "NLC1 + uint32le window/search_min/search_max/length_bias + uint64le raw_size/payload_size + ctrl8 payload"}
    domain = {"window_bytes": windows, "candidate_count": len(windows), "valid_integer_range": [1, 65535],
        "candidate_source": "all 14 distinct historical native window variants, already including 65535"}
    validation = {"candidate_sample_pairs": len(rows),
        "bit_exact_round_trips": sum(int(r["round_trips"]) for r in rows), "all_passed": True}
    calibration = {"sample_count": len(samples), "candidate_count": len(windows),
        "raw_bytes": selected["raw_bytes"], "frame_bytes": selected["frame_bytes"],
        "round_trips": validation["bit_exact_round_trips"], "cohorts": provenance["cohorts"], "held_out": False}
    summary = {"schema": "native-lz-window-calibration-v1", "selected": selected, "criterion": criterion,
        "search_domain": domain, "fixed_parameters": {"min_match": 8, "max_match": 256, "length_bias": 3,
            "parser": "aligned8_greedy"}, "sample_manifest": str(out / "sample_manifest.json"),
        "sample_manifest_sha256": manifest_hash, "sample_results_sha256": sha((out / "sample_results.csv").read_bytes()),
        "candidate_results_sha256": results_hash, "source_sha256": source_hashes, "source_snapshots": source_snapshots,
        "calibration_provenance": {"datasets": list(lz.FIELDS), "columns": 20, "blocks": len(samples),
            "raw_bytes": selected["raw_bytes"], "held_out": False}, "calibration": calibration, "validation": validation,
        "timing": {"repetitions": args.repetitions,
            "statistic": "sum of per-block median process-internal steady_clock nanoseconds",
            "candidate_order_seed": 20260908, "warmups": "3 first-block round trips with window32768",
            "encoder_includes": "production parse + pack + complete frame construction",
            "decoder_includes": "frame parsing + production unpack + expand; unpack also validates by expansion",
            "excludes": "CLI/process startup, compilation, file I/O, post-timer round-trip equality check",
            "shared_host": "other work may run concurrently; observational timing is not used for selection and does not establish speed rankings"},
        "platform": platform.platform(), "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "compile_command": command, "run_command": run, "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Best only among 14 tested windows on these calibration blocks with min8/max256/bias3 fixed.",
            "Not a joint window/min/max optimization: the existing min/max default was calibrated separately at window32768.",
            "Calibration samples overlap; bytes are summed over independently encoded sample frames, not unique dataset coverage.",
            "Window stage alone only records context; this benchmark includes downstream production parse and pack.",
            "No held-out, full pipeline, other-parser, entropy-coder or universal optimum claim.",
            "The fixed 36-byte NLC1 calibration frame is independently decodable, not the UAR runtime envelope."]}
    write_json(out / "SUMMARY.json", summary)
    defaults = {"schema": "native-lz-window-defaults-v1", "parameters": {"window_bytes": selected["window_bytes"]},
        "fixed_parameters": summary["fixed_parameters"], "search_domain": domain, "criterion": criterion,
        "calibration": calibration, "calibration_provenance": {**summary["calibration_provenance"],
            "manifest": str(out / "sample_manifest.json"), "manifest_sha256": manifest_hash,
            "summary": str(out / "SUMMARY.json"), "source_sha256": source_hashes, "source_snapshots": source_snapshots},
        "results_hash": results_hash, "results_hash_algorithm": "sha256", "results_file": str(out / "candidate_results.csv"),
        "selected_total_frame_bytes": selected["frame_bytes"], "selected_total_raw_bytes": selected["raw_bytes"]}
    args.defaults.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.defaults, defaults)
    table = "\n".join(f"| {x['window_bytes']} | {x['frame_bytes']:,} | {x['compression_ratio_raw_over_frame']:.4f} | {x['encode_ns_sum_of_block_medians']/1e6:.3f} | {x['decode_ns_sum_of_block_medians']/1e6:.3f} |" for x in ranked)
    report = f"""# Native-LZ window 参数有限校准

默认选择：`window_bytes={selected['window_bytes']}`。固定 aligned8 greedy、min_match=8、max_match=256、length_bias=3；14个历史候选已包含65535，不重复计为第15个候选。

按全部86个校准块的完整NLC1帧总字节选择；同样大小时选择较小window，时间不参与选择。**只说明这些候选、样本与固定下游算法上的最小字节数，不是保留集结果或任意数据上的最优。**既有min/max默认在window32768上另行校准，两个默认的组合不等于联合最优。

沿用60个5456B数值块（327,360B），另加20个前部64KiB块和6个Metro-Traffic尾部128KiB块，总计{selected['raw_bytes']:,}B。样本有重叠，各自独立编码；完整原始行号、类型、数据集与样本SHA-256均在sample_manifest.json。原60块过小，不能区分较大窗口，因此补入较大块。

全部{len(rows):,}个候选/样本组合各执行{args.repetitions}次，共{validation['bit_exact_round_trips']:,}次逐位无损往返通过。每块计入36B NLC1头：window/min/max/bias/raw_size/payload_size；这不是UAR运行时封装。Match Window阶段本身只写上下文，本次测的是后续production parse+pack及完整帧。

| window字节 | 完整帧总字节 | 原始/压缩 | 压缩ms | 解压ms |
|---:|---:|---:|---:|---:|
{table}

时间为C++进程内每块3次测量的中位数之和，编码含parse、pack与帧构造；解码含帧解析、unpack和expand，unpack也做展开验证。排除进程启动、编译、文件IO和计时后相等检查。共享机器上允许其他工作并行，时间只作观测，不证明速度排序。

candidate_results.csv含14个候选与各大小样本组的字节统计；sample_results.csv逐块记录；SUMMARY.json包含编译、来源哈希与限制。source_snapshot冻结实际编译C++源码、生成harness、运行脚本与历史注册表。默认文件：{args.defaults}。
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected": defaults["parameters"], **calibration,
        "report": str(out / "REPORT.md"), "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
