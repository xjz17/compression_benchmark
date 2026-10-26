"""Build a layered current view without rewriting the frozen 102-v1 catalog."""
from pathlib import Path
import hashlib
import json

from operator_catalog_layers import VERSION, CANDIDATES, accounting, validate_layers

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmark/reference/actions/operator_implementations_102.json"
OUT = ROOT / "benchmark/reference/actions/operator_implementations_layered_v1.json"


def build():
    raw = BASE.read_bytes()
    data = json.loads(raw)
    data.update(registry_version=VERSION, date="2026-09-17",
                previous_catalog_version=data["registry_version"],
                previous_catalog_sha256=hashlib.sha256(raw).hexdigest(),
                scope="102 historical trace entries partitioned into mechanisms, profiles, recipes and compatibility adapters; mechanism count is not an irreducibility or executable-action claim.")
    for item in data["implementations"]:
        item["accounting"] = accounting(item["implementation_id"])
    by_id = {item["implementation_id"]: item for item in data["implementations"]}
    by_id["A022"].update(name="Sprintz nine-value block packing and tail recipe",
        difference="Each complete group contains one anchor and eight residuals. A020 packs the residuals; the tail uses the selected current/legacy RLE or Gorilla profile.")
    by_id["A056"].update(name="Threshold-four int64 RLE compatibility proxy",
        difference="Runs of length >=4 use repeat records; shorter runs use literal records. This is neither LZ matching nor A055 optimal RLE.")
    by_id["A073"].update(name="VariableByte Tail Coding after FastPFor Prefix",
        difference="Consume the PFS1 packet, preserve the already encoded low/high pfor_words, and VariableByte-code each raw_tail before appending it. The final group has its high bit set; the coded tail is padded to uint32 words. This entry does not perform the preceding PFor block encoding.",
        source_refs=["cpp/src/fastpfor_exact_stages.cpp"])
    for key in ("A001", "A047", "A049", "A051", "A066", "A104"):
        by_id[key]["source_refs"].append("cpp/src/operator_layout_primitives.hpp")
    for key in ("A041", "A042", "A043", "A053"):
        by_id[key]["source_refs"].append("cpp/src/dictionary_recode.hpp")
    by_id["A111"].update(classification="standalone_exception_separation",
        execution_status="independent_cpp_api_cli_and_three_stream_bridge; separate_A020_recipe; not_registered_in_graph_policy",
        source_refs=["cpp/src/exception_separation.hpp", "cpp/src/exception_separation_cli.cpp", "scripts/exception_separation.py"],
        parameters_or_profiles=["supplied b in [0,32]", "uint32 numeric input", "EHS1 independent streams", "EHP1 explicit A020 recipe"],
        difference="Given b, split uint32 values into low[N], strictly increasing exception positions[E], and nonzero high[E]. Restore by low[pos] OR (high<<b). Independent API and CLI exist; the original FastPFor fused path remains separately identified.")
    by_id["A110"].update(name="NNCP 3.3 Transformer byte probability, stream1 profile",
        classification="probability_provider_platform_gated",
        execution_status="native_c_model_cpp_cli_symbol_session; matched_entropy_cold_decode; Linux_x86_64_or_explicit_Docker; not_graph_entropy_integrated",
        source_refs=["cpp/include/nncp_probability.h", "cpp/src/nncp_probability_bridge.c", "cpp/src/nncp_probability_cli.cpp", "scripts/prediction_probability.py", "scripts/build_nncp_probability.py"],
        parameters_or_profiles=["transformer-stream1-v1", "batch_size=1", "seg_len=1", "vocabulary=256 raw bytes", "seed=123", "float32 CPU", "session_limit=4096 symbols"],
        difference="Original NNCP 3.3 Transformer, gradient and optimizer through the original LibNC dependency. predict returns 256 float32 next-byte probabilities before observe(byte); NNP1 matches upstream arithmetic symbol coding. Streaming schedule is explicit and differs from the upstream default file profile.")
    data["layer_counts"] = validate_layers(data["implementations"])
    data["mechanism_count"] = data["layer_counts"]["mechanism"]
    data["trace_entry_count"] = len(data["implementations"])
    data["consolidation_candidates_not_applied"] = CANDIDATES
    data["count_change"] = {
        "previous_trace_entries": 102, "current_trace_entries": 102,
        "current_layer_counts": data["layer_counts"],
        "rule": "A IDs preserved for traceability; non-mechanism layers excluded from mechanism accounting, not deleted from compatibility decoding.",
    }
    data["pending_runtime_work"] = [
        line for line in data["pending_runtime_work"]
        if "A111 is an independent semantic definition" not in line and "NNCP" not in line
    ]
    data["pending_runtime_work"].append("A111 standalone API/CLI and EHP1 recipe do not migrate the original FastPFor method or add a graph policy action.")
    data["pending_runtime_work"].append("NNCP stream1 standalone API and NNP1 require original Linux x86-64 LibNC or explicit local Docker; no default NNCP benchmark-equivalence or graph entropy integration claim.")
    data["revision_note"] = "2026-09-17 layered accounting and shared kernels; preserve prior source crosswalk, old packet formats and 86 method labels. See per-entry interface status and revision validation."
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "layers": data["layer_counts"], "trace_entries": data["trace_entry_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    build()
