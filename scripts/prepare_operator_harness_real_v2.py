#!/usr/bin/env python3
"""Prepare metadata only. All graph execution, timing and commits are C++."""
import argparse
import hashlib
import json
from pathlib import Path


def parameter(value):
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def prepare(catalog_path, samples_path, output, runtime_root):
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    samples = json.loads(samples_path.read_bytes())["samples"]
    for sample in samples:
        payload = bytes.fromhex(sample["data_hex"])
        if len(payload) != sample["sample_bytes"] or len(payload) != sample["source_byte_width"] * sample["element_count"]:
            raise ValueError("invalid sample byte accounting: " + sample["sample_id"])
        if hashlib.sha256(payload).hexdigest() != sample["sample_sha256"]:
            raise ValueError("invalid sample digest: " + sample["sample_id"])
        mirror = samples_path.parent / "inputs" / (sample["sample_id"] + ".bin")
        if mirror.exists() and mirror.read_bytes() != payload:
            raise ValueError("sample mirror differs from extraction receipt: " + sample["sample_id"])
        for key in ("source_manifest_sha256", "source_block_sha256"):
            if len(sample[key]) != 64 or any(c not in "0123456789abcdef" for c in sample[key]):
                raise ValueError("missing source provenance digest")
        if not sample["original_order_preserved"] or sample["sample_offset_in_source_block_bytes"] % sample["source_byte_width"]:
            raise ValueError("sample is not an aligned original-order slice")
    by_id = {s["sample_id"]: s for s in samples}
    groups = {
        "int64": ["rating_d81efc8fc641_prefix", "Telco_09f769686a74_prefix"],
        "float64": ["rating_bb5612e4b49b_prefix", "weather_data_bavaria_02f0f0dc1fd0_prefix", "Telco_1af1878bc567_prefix"],
        "float32": ["temperature_rain_9b0c1414af51_prefix", "temperature_rain_9b0c1414af51_middle"],
        "bytes": ["rating_d81efc8fc641_prefix", "weather_data_bavaria_02f0f0dc1fd0_prefix"],
    }
    runtime_root = str(runtime_root).rstrip("/")
    file_hash = hashlib.sha256(catalog_bytes).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for batch_id in catalog["batch_order"]:
        cases = []
        for row in catalog["operators"]:
            if not row["selection"]["selectable"] or row["batch_id"] != batch_id:
                continue
            view = row["input_view"]
            group = "bytes" if row["initial_kind"] == "serial" else view
            if group == "physical_word":
                group = "float64"
            if group == "uint32":
                group = "float32"
            for sample_id in groups[group]:
                sample = by_id[sample_id]
                stages = row["present_pipeline"]
                steps = [{"kind": "stage", "inputs": [0 if i == 0 else i + 1],
                          "stage": {"action": s["action"], "profile": s["profile"],
                                    "parameters": {k: parameter(v) for k, v in s["parameters"].items()}}}
                         for i, s in enumerate(stages)] + [{"kind": "stop"}]
                case_id = row["implementation_id"] + "__" + sample_id
                request = {
                    "schema": "operator-harness-request-v2", "command": "run", "request_id": case_id,
                    "implementation_ids": [row["implementation_id"]],
                    "catalog_path": runtime_root + "/source/benchmark/reference/actions/operator_harness_batch_catalog_v2.json",
                    "catalog_sha256": file_hash,
                    "input_path": runtime_root + "/inputs/" + sample_id + ".bin",
                    "initial_kind": row["initial_kind"], "initial_width": row["initial_width"],
                    "input_view": view, "dtype": sample["dtype"], "element_count": sample["element_count"],
                    "source_byte_width": sample["source_byte_width"], "original_numeric_payload": True,
                    "recipe": {"policy_registry": "operator-harness-v2", "steps": steps},
                    "warmups": 1, "repeats": 5,
                }
                cases.append({"case_id": case_id, "implementation_id": row["implementation_id"],
                              "batch_id": batch_id, "sample_id": sample_id,
                              "sample_sha256": sample["sample_sha256"], "request": request,
                              "provenance": {k: v for k, v in sample.items() if k != "data_hex"},
                              "focal_step_index": row["focal_step_index"],
                              "performance_scope": "whole registered graph including prefixes, NGF1 and source schema; shared-host SCHED_IDLE timing"})
        plan = {"schema": "operator-harness-batch-plan-v2", "batch_id": batch_id,
                "catalog_file_sha256": file_hash, "cases": cases}
        (output / (batch_id + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
        all_cases.extend(cases)
    manifest = {"schema": "operator-harness-real-plan-index-v2", "catalog_file_sha256": file_hash,
                "case_count": len(all_cases), "operator_count": len({x["implementation_id"] for x in all_cases}),
                "excluded_ids": ["A106", "A107", "A108", "A109", "A110"],
                "batch_order": catalog["batch_order"],
                "case_ids": [x["case_id"] for x in all_cases]}
    (output / "INDEX.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=Path("benchmark/reference/actions/operator_harness_batch_catalog_v2.json"))
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--runtime-root", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.catalog, a.samples, a.output, a.runtime_root)))
