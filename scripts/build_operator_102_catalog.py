"""Build the versioned source-equivalence catalog and explicit recipe crosswalk."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmark/reference/actions/operator_implementations_106.json"
OUT = ROOT / "benchmark/reference/actions/operator_implementations_102.json"


def build():
    source = BASE.read_bytes()
    data = json.loads(source)
    old = {x["implementation_id"]: x for x in data["implementations"]}
    data.update(registry_version="operator-implementations-102-v1", date="2026-09-16",
                count=102, previous_catalog_version=data["registry_version"],
                previous_catalog_sha256=hashlib.sha256(source).hexdigest(),
                count_change={"before": 106, "after": 102, "removed": ["A024", "A025", "A034", "A036"]})
    recipes = {
        "A024": {"kind": "branched_pack", "replacement_implementation_ids": ["A020"],
                 "upstream": ["A019", "A013"],
                 "branches": {"normal": {"a_id": "A020", "width": "supplied_b", "bit_order": "msb"},
                              "exception_positions": {"a_id": "A020", "width": 32, "bit_order": "lsb"},
                              "exception_values": {"a_id": "A020", "width": 64, "bit_order": "lsb"}},
                 "framing": "exception position/value pairs remain interleaved before packed normal body",
                 "bypass": ["original_count", "first", "source_base", "range_base", "normal_count", "exception_count"]},
        "A025": {"kind": "parameter_profile", "replacement_implementation_ids": ["A020"],
                 "parameters": {"minimum_width": 1, "wide_cutoff": 56, "wide_width": 64, "bit_order": "msb"},
                 "bypass": ["original_count", "first", "scale"]},
        "A034": {"kind": "sequence", "replacement_implementation_ids": ["A031", "A039"],
                 "parameters": {"word_bits": [32, 64], "history_index_bits": 0,
                                "significant_length_bits": {"32": 5, "64": 6}},
                 "bypass": ["eraser_tag", "beta", "chunk_count", "first_word", "EOS"]},
        "A036": {"kind": "record_dispatch", "replacement_implementation_ids": ["A031", "A028"],
                 "branches": {"first": "anchor/reset", "literal": "A031", "decimal": "A028"},
                 "state_rule": "Process original record order; update raw_bits and integer state after every record. Never split into independent per-tag histories.",
                 "bypass": ["record_tag", "sign", "decimal_count", "decimal_value", "chunk_count"]},
    }
    for retired, target in {"A024": "A020", "A025": "A020", "A034": "A039", "A036": "A031"}.items():
        record = old[retired]
        old[target]["source_entry_ids"].extend(record["source_entry_ids"])
        old[target].setdefault("composed_source_action_names", []).extend(record["source_action_names"])
        data["retired"].append({"implementation_id": retired, "name": record["name"],
                                "reason": "explicit composition or parameter profile; not another implementation",
                                **recipes[retired]})
    old["A013"]["family_numbers"] = [3]
    old["A013"]["difference"] = "Minimum FOR only: minimum plus unsigned offsets. All source profiles use the same core."
    old["A020"].update(name="Parameterized fixed-width bit packing", classification="shared_core",
        parameters_or_profiles=["supplied width or maximum-width selection", "minimum width", "allowed widths / wide cutoff", "MSB/LSB order", "byte or word alignment"],
        difference="pack_fixed consumes values and width; maximum_width/allowed_width select a width. BOS normal, position and exception-value streams share this packer.")
    old["A020"]["source_refs"] = ["cpp/src/max_value_bitpack.hpp", "cpp/src/bos_exact_stages.cpp"]
    old["A031"]["source_refs"].append("cpp/src/adjacent_residual.hpp")
    old["A028"]["source_refs"].append("cpp/src/adjacent_residual.hpp")
    old["A028"].setdefault("composed_source_action_names", []).append("op.xor.camel.cpp_v1")
    old["A035"].update(name="Elf+ residual window packing", family_numbers=[10], classification="core",
        difference="Consumes precomputed XOR residuals. Reuse requires equal rounded leading and sufficient trailing zeros; new windows omit the known low one and use short/long length headers.",
        source_refs=["cpp/src/residual_window_pack.hpp", "cpp/src/residual_window_pack_cli.cpp", "cpp/src/elf_float_codec.cpp", "cpp/src/elf_double_codec.cpp"],
        source_action_names=[], composed_source_action_names=old["A035"]["source_action_names"],
        execution_status="standalone_cpp_cli_and_shared_core; standalone_graph_action_not_registered")
    old["A039"].update(name="Parameterized Chimp residual window packing", classification="shared_core",
        difference="Four control cases pack existing residuals. History index width zero represents adjacent-XOR input; width seven represents a 128-slot reference.",
        parameters_or_profiles=["word_bits=32/64", "history_index_bits=0/7", "leading bucket table", "trailing threshold", "significant length field width"],
        source_refs=["cpp/src/residual_window_pack.hpp", "cpp/src/chimp128_exact_stages.cpp", "cpp/src/elf_float_codec.cpp", "cpp/src/elf_double_codec.cpp"])
    data["implementations"] = [v for k, v in old.items() if k not in recipes]
    data["review_recipes"] = recipes
    action_recipes = {}
    for key, r in recipes.items():
        for action in old[key]["source_action_names"]:
            action_recipes[action] = {"recipe": r, "execution": "cpp_composition_adapter", "independent_graph_branches": False}
    for action in old["A035"]["composed_source_action_names"]:
        action_recipes[action] = {"recipe": {"kind": "sequence", "replacement_implementation_ids": ["A031", "A035"]},
                                  "execution": "cpp_composition_adapter", "independent_graph_branches": False}
    data["runtime_action_recipes"] = action_recipes
    data["revision_note"] = "Two source reviews applied: FOR classification; BOS packing profiles; adjacent XOR plus residual window packing; Camel ordered record dispatch. Wire IDs unchanged because adapter I/O and bytes are preserved."
    data["pending_runtime_work"] += ["Residual-window cores have native C++ APIs and a standalone CLI; separate graph residual/window actions are not registered."]
    data["source_entry_accounting"] = {"unique_active_source_entries": 131,
        "rule": "Primary ownership is unique; runtime_action_recipes records one-to-many semantic composition."}
    assert len(data["implementations"]) == 102
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    build()
