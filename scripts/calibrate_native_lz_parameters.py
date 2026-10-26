#!/usr/bin/env python3
"""Select native-LZ defaults on bounded, explicitly non-held-out numeric samples.

No registry/runtime import is required: the harness links the production C++
aligned-eight parser and ctrl8 codec directly. The compressed-size objective
includes a self-describing 36-byte calibration header on every sampled block.
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
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis_outputs/native_lz_parameters_20260908_v1"
DEFAULTS = ROOT / "benchmark/reference/actions/native_lz_defaults.json"
MINIMA = (8, 16, 24, 32, 64)
MAXIMA = tuple(range(8, 257, 8))
CANDIDATES = tuple((minimum, maximum) for minimum in MINIMA for maximum in MAXIMA
                   if minimum <= maximum)
FIELDS = {
    "Air-Quality": {
        "timestamp": "int64", "CO(GT)": "float64", "PT08.S1(CO)": "int64",
        "NMHC(GT)": "int64", "C6H6(GT)": "float64", "PT08.S2(NMHC)": "int64",
        "NOx(GT)": "int64", "PT08.S3(NOx)": "int64", "NO2(GT)": "int64",
        "PT08.S4(NO2)": "int64", "PT08.S5(O3)": "int64", "T": "float64",
        "RH": "float64", "AH": "float64",
    },
    "Metro-Traffic": {
        "temp": "float64", "rain_1h": "float64", "snow_1h": "float64",
        "clouds_all": "int64", "date_time": "int64", "traffic_volume": "int64",
    },
}

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
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void put(Bytes& b, std::uint64_t v, unsigned n) {
  for (unsigned i=0;i<n;++i) b.push_back(std::uint8_t(v>>(i*8)));
}
std::uint64_t get(const Bytes& b, std::size_t& p, unsigned n) {
  require(p<=b.size() && n<=b.size()-p,"truncated calibration frame");
  std::uint64_t v=0; for (unsigned i=0;i<n;++i) v|=std::uint64_t(b[p++])<<(8*i);
  return v;
}
Bytes encode(const Bytes& raw, unsigned minimum, unsigned maximum) {
  const auto wire=pack(parse(raw,32768,minimum,maximum),3);
  Bytes frame{'N','L','C','1'};
  put(frame,32768,4); put(frame,minimum,4); put(frame,maximum,4); put(frame,3,4);
  put(frame,raw.size(),8); put(frame,wire.size(),8);
  frame.insert(frame.end(),wire.begin(),wire.end()); return frame;
}
Bytes decode(const Bytes& frame) {
  require(frame.size()>=36 && std::string(frame.begin(),frame.begin()+4)=="NLC1",
          "invalid calibration frame magic");
  std::size_t pos=4;
  const auto window=get(frame,pos,4), minimum=get(frame,pos,4), maximum=get(frame,pos,4);
  const auto bias=get(frame,pos,4), count=get(frame,pos,8), size=get(frame,pos,8);
  require(window==32768 && minimum>=8 && minimum%8==0 && maximum>=minimum &&
          maximum<=256 && maximum%8==0 && bias==3,"invalid calibration parameters");
  require(size==frame.size()-pos,"calibration payload size mismatch");
  Bytes wire(frame.begin()+pos,frame.end());
  return expand(unpack(wire,bias,count),count);
}
std::int64_t ns(Clock::time_point a, Clock::time_point b) {
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
    std::vector<std::pair<unsigned,unsigned>> candidates; std::ifstream params(argv[2]);
    unsigned minimum,maximum; while (params>>minimum>>maximum) candidates.emplace_back(minimum,maximum);
    require(candidates.size()==147,"expected 147 candidates");
    for (int i=0;i<3;++i) require(decode(encode(samples[0].raw,8,256))==samples[0].raw,"warmup failed");
    std::cout<<"sample_id,min_match,max_match,raw_bytes,wire_bytes,frame_bytes,encode_ns_median,decode_ns_median,round_trips\n";
    for (const auto& s:samples) for (auto [lo,hi]:candidates) {
      std::vector<std::int64_t> enc,dec; std::size_t frame_size=0;
      for (unsigned r=0;r<repeats;++r) {
        const auto begin=Clock::now(); const auto frame=encode(s.raw,lo,hi); const auto middle=Clock::now();
        const auto restored=decode(frame); const auto end=Clock::now();
        require(restored==s.raw,"bit-exact round trip failed");
        require(frame_size==0 || frame_size==frame.size(),"non-deterministic size"); frame_size=frame.size();
        enc.push_back(ns(begin,middle)); dec.push_back(ns(middle,end));
      }
      std::sort(enc.begin(),enc.end()); std::sort(dec.begin(),dec.end());
      std::cout<<s.id<<','<<lo<<','<<hi<<','<<s.raw.size()<<','<<frame_size-36<<','<<frame_size<<','
               <<enc[repeats/2]<<','<<dec[repeats/2]<<','<<repeats<<'\n';
    }
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_samples(out: Path) -> tuple[list[dict], list[dict]]:
    samples, sources = [], []
    sample_dir = out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # 682 float/int64 values per block: 3 * 682 * 8 = 16,368 bytes per column.
    block_values = 682
    for dataset, fields in FIELDS.items():
        path = ROOT / "orignial_datasets" / f"{dataset}.csv"
        source = path.read_bytes()
        columns = {column: [] for column in fields}
        skipped = {column: 0 for column in fields}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader):
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
                            raise ValueError(f"invalid int64: {dataset}/{column}/{row_number}")
                        packed = struct.pack("<q", int(number))
                    else:
                        value = float(value_text)
                        if not math.isfinite(value):
                            skipped[column] += 1
                            continue
                        packed = struct.pack("<d", value)
                    columns[column].append((row_number, packed))
        sources.append({"dataset": dataset, "path": str(path), "sha256": sha(source),
                        "csv_bytes": len(source), "selected_columns": list(fields)})
        for column, kind in fields.items():
            values = columns[column]
            if len(values) < block_values * 3:
                raise ValueError(f"too few calibration values: {dataset}/{column}")
            starts = (0, (len(values) - block_values) // 2, len(values) - block_values)
            for region, start in zip(("front", "middle", "back"), starts):
                chosen = values[start:start + block_values]
                raw = b"".join(value for _, value in chosen)
                sample_id = f"s{len(samples):03d}"
                payload_path = sample_dir / f"{sample_id}.bin"
                payload_path.write_bytes(raw)
                samples.append({
                    "sample_id": sample_id, "dataset": dataset, "column": column, "kind": kind,
                    "byte_order": "little", "region": region, "source_path": str(path),
                    "source_sha256": sha(source), "valid_value_range_half_open": [start, start + block_values],
                    "csv_data_rows_zero_based": [row for row, _ in chosen],
                    "source_valid_values": len(values), "source_skipped_missing_or_nonfinite": skipped[column],
                    "sample_path": str(payload_path), "raw_bytes": len(raw), "sha256": sha(raw),
                })
    return samples, sources


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
        parser.error("output already has a completed calibration; choose a new directory")
    assert len(CANDIDATES) == 147
    started = time.monotonic()
    samples, sources = collect_samples(out)
    provenance = {
        "purpose": "default parameter calibration only; not held-out performance evaluation",
        "datasets": sources, "samples": samples, "columns": sum(map(len, FIELDS.values())),
        "blocks": len(samples), "total_raw_bytes": sum(s["raw_bytes"] for s in samples),
        "sampling": "20 predeclared numeric columns; independent front/middle/back 682-value blocks; <=16KiB per column",
        "conversion": "Explicit int64/float64 column schema, little endian; skip and count missing/nonfinite values; preserve finite -0.0",
    }
    write_json(out / "sample_manifest.json", provenance)
    (out / "samples.tsv").write_text("".join(f"{s['sample_id']}\t{s['sample_path']}\n" for s in samples))
    candidates = list(CANDIDATES)
    random.Random(20260908).shuffle(candidates)
    (out / "candidates.tsv").write_text("".join(f"{lo}\t{hi}\n" for lo, hi in candidates))
    cpp = out / "calibration_harness.cpp"
    cpp.write_text(HARNESS)
    # Compile only the frozen copies: other agents may continue editing the
    # workspace while this bounded calibration runs.
    snapshot_root = out / "source_snapshot"
    source_hashes, source_snapshots = {}, {}
    for source_path in (ROOT / "cpp/src/native_lz_action_stages.cpp",
                        ROOT / "cpp/src/native_lz_action_stages.hpp", Path(__file__)):
        relative = str(source_path.relative_to(ROOT))
        raw = source_path.read_bytes()
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_hashes[relative] = sha(raw)
        source_snapshots[relative] = str(target)
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
    if len(rows) != len(samples) * len(CANDIDATES):
        raise AssertionError("missing candidate/sample measurements")
    seen = set()
    aggregates = {}
    for row in rows:
        key = (int(row["min_match"]), int(row["max_match"]))
        identity = (row["sample_id"], key)
        if identity in seen:
            raise AssertionError("duplicate candidate/sample measurement")
        seen.add(identity)
        item = aggregates.setdefault(key, {"min_match": key[0], "max_match": key[1], "raw_bytes": 0,
                "wire_bytes": 0, "frame_bytes": 0, "encode_ns_sum_of_block_medians": 0,
                "decode_ns_sum_of_block_medians": 0, "round_trips": 0, "blocks": 0})
        for field in ("raw_bytes", "wire_bytes", "frame_bytes", "round_trips"):
            item[field] += int(row[field])
        item["blocks"] += 1
        item["encode_ns_sum_of_block_medians"] += int(row["encode_ns_median"])
        item["decode_ns_sum_of_block_medians"] += int(row["decode_ns_median"])
    ranked = sorted(aggregates.values(), key=lambda item: (item["frame_bytes"], item["min_match"], item["max_match"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
        item["compressed_fraction"] = item["frame_bytes"] / item["raw_bytes"]
        item["compression_ratio_raw_over_frame"] = item["raw_bytes"] / item["frame_bytes"]
        item["encode_mib_per_second"] = item["raw_bytes"] / (1024 * 1024) / (item["encode_ns_sum_of_block_medians"] / 1e9)
        item["decode_mib_per_second"] = item["raw_bytes"] / (1024 * 1024) / (item["decode_ns_sum_of_block_medians"] / 1e9)
    with (out / "candidate_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader(); writer.writerows(ranked)
    selected = ranked[0]
    results_hash = sha((out / "candidate_results.csv").read_bytes())
    manifest_hash = sha((out / "sample_manifest.json").read_bytes())
    criterion = {"objective": "minimize sum of complete calibration frame bytes over all 60 blocks",
                 "tie_break": "ascending (min_match, max_match)", "timing_used_for_selection": False,
                 "frame_header_bytes": 36,
                 "frame_layout": "NLC1 + uint32le window/search_min/search_max/length_bias + uint64le raw_size/payload_size + ctrl8 payload"}
    domain = {"min_match": list(MINIMA), "max_match": list(MAXIMA), "constraint": "min_match <= max_match",
              "candidate_count": 147, "step_bytes": 8}
    summary = {
        "schema": "native-lz-parameter-calibration-v1", "selected": selected,
        "criterion": criterion, "search_domain": domain,
        "fixed_parameters": {"window": 32768, "length_bias": 3, "parser": "aligned8_greedy"},
        "sample_manifest": str(out / "sample_manifest.json"), "sample_manifest_sha256": manifest_hash,
        "sample_results_sha256": sha((out / "sample_results.csv").read_bytes()),
        "candidate_results_sha256": results_hash, "source_sha256": source_hashes,
        "source_snapshots": source_snapshots,
        "calibration_provenance": {"datasets": list(FIELDS), "columns": provenance["columns"],
             "blocks": len(samples), "raw_bytes": provenance["total_raw_bytes"], "held_out": False},
        "validation": {"candidate_sample_pairs": len(rows), "bit_exact_round_trips": sum(int(r["round_trips"]) for r in rows),
                       "all_passed": True},
        "timing": {"repetitions": args.repetitions, "statistic": "sum of per-block median process-internal steady_clock nanoseconds",
             "candidate_order_seed": 20260908, "warmups": "3 first-block round trips with min8/max256",
             "encoder_includes": "production parse + pack + complete frame construction",
             "decoder_includes": "frame parsing + production unpack + expand; unpack also validates by expansion",
             "excludes": "CLI/process startup, compilation, file I/O, post-timer round-trip equality check",
             "scope": "serial local calibration harness, not full search/RL runtime or held-out throughput",
             "shared_host": "other work may run concurrently; timing is observational and does not establish a speed ranking"},
        "platform": platform.platform(), "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "compile_command": command, "run_command": run, "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Best only among the 147 tested pairs on these calibration blocks.",
             "No held-out performance claim; selected data used to choose the defaults.",
             "No downstream transform/entropy stage; no optimum claim for other data, windows, parsers or complete pipelines.",
             "The fixed 36-byte calibration header is independently decodable, not the UAR runtime envelope."],
    }
    write_json(out / "SUMMARY.json", summary)
    defaults = {"schema": "native-lz-defaults-v1", "parameters": {"min_match": selected["min_match"], "max_match": selected["max_match"]},
                "fixed_parameters": summary["fixed_parameters"], "search_domain": domain, "criterion": criterion,
                "calibration_provenance": {**summary["calibration_provenance"], "manifest": str(out / "sample_manifest.json"),
                    "manifest_sha256": manifest_hash, "summary": str(out / "SUMMARY.json"),
                    "source_sha256": source_hashes, "source_snapshots": source_snapshots}, "results_hash": results_hash,
                "results_hash_algorithm": "sha256", "results_file": str(out / "candidate_results.csv"),
                "selected_total_frame_bytes": selected["frame_bytes"], "selected_total_raw_bytes": selected["raw_bytes"]}
    args.defaults.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.defaults, defaults)
    top = "\n".join(f"| {x['min_match']} | {x['max_match']} | {x['frame_bytes']:,} | {x['compression_ratio_raw_over_frame']:.4f} | {x['encode_ns_sum_of_block_medians']/1e6:.3f} | {x['decode_ns_sum_of_block_medians']/1e6:.3f} |" for x in ranked[:10])
    report = f"""# Native-LZ 默认参数有限校准

选择结果：`min_match={selected['min_match']}, max_match={selected['max_match']}`。固定 window=32768、aligned8 greedy、length_bias=3；搜索 min 与编码长度 bias 独立。

在147个预定候选中，选择60个校准块完整帧总字节最少者；相同字节数按(min_match,max_match)升序。**该结论只针对本次校准集，不是保留测试集结论或普遍最优结论。**

Air-Quality的14列及Metro-Traffic的6列，每列取前/中/后各682个数值（每列16,368字节，3块独立编码），共{provenance['total_raw_bytes']:,}字节。字段类型、原文件SHA-256、原始行号与范围、样本SHA-256详见sample_manifest.json；缺失或非有限值跳过并计数。

全部{len(rows):,}个候选/样本组合分别完成{args.repetitions}次逐位往返，共{summary['validation']['bit_exact_round_trips']:,}次通过。大小计入每块36字节NLC1头，保存window、search_min、search_max、length_bias、raw_size和payload_size。

| min | max | 完整帧总字节 | 原始/压缩 | 压缩ms | 解压ms |
|---:|---:|---:|---:|---:|---:|
{top}

时间为C++进程内逐块中位数之和，包含parse/pack及固定帧，解压包含unpack和expand（unpack本身也做展开验证）；排除进程启动、编译、文件I/O和计时后的相等检查。时间未用于默认选择。本机允许其他工作并行，时间仅为观测，不能据此断言速度排序。其数值仅描述该校准harness，不能当作完整runtime、下游熵编码或新数据的吞吐量。

完整147候选：candidate_results.csv；逐块结果：sample_results.csv；摘要和来源哈希：SUMMARY.json。默认参数文件：{args.defaults}。

实际编译使用source_snapshot/下冻结的C++源码及harness；执行脚本也已快照。SUMMARY.json记录每份快照及SHA-256，避免校准过程中工作区继续修改导致来源混淆。
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected": defaults["parameters"], "raw_bytes": selected["raw_bytes"],
          "frame_bytes": selected["frame_bytes"], "round_trips": summary["validation"]["bit_exact_round_trips"],
          "report": str(out / "REPORT.md"), "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
