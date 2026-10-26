#include "remaining_codec_action_catalog.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main() {
  const auto pipelines = remaining_codec_actions::pipelines();
  const auto actions = remaining_codec_actions::concrete_actions();
  if (pipelines.size() != 35) throw std::runtime_error("expected 35 codec pipelines");
  if (actions.size() != 104) throw std::runtime_error("expected 104 corrected method-scoped actions");

  const std::set<std::string> expected_methods = {
      "Chimp", "Chimp128", "Pongo", "Gorilla", "2DIFF", "BOS", "REGER",
      "SVB", "FastPFor", "Simple8b", "RLE", "RLBE", "Sprintz", "DBP", "ALP",
      "LECO", "Camel", "Elf", "Elf+", "FPC", "PFPC", "SPDP", "CORAD",
      "BUFF", "FPZIP", "gzip", "Brotli", "Zstd", "LZ4", "XZ", "Snappy",
      "Blosc", "Blosc2", "Bzip2", "Zlib"};
  std::set<std::string> actual_methods;
  std::size_t specialist_count = 0;
  std::size_t general_count = 0;
  for (const auto& pipeline : pipelines) {
    actual_methods.emplace(pipeline.method);
    specialist_count += pipeline.kind == "specialist" ? 1 : 0;
    general_count += pipeline.kind == "general" ? 1 : 0;
  }
  if (actual_methods != expected_methods || specialist_count != 25 || general_count != 10) {
    throw std::runtime_error("35-codec target set mismatch");
  }

  std::set<std::string> names;
  std::size_t shared_action_occurrences = 0;
  std::size_t measurable = 0;
  std::size_t standalone_composable = 0;
  std::size_t codec_ablation = 0;
  std::size_t exact_stage = 0;
  std::size_t reference_stage = 0;
  std::size_t atomic = 0;
  std::size_t blocked = 0;
  std::size_t proxy = 0;
  std::size_t source_unwired = 0;
  std::size_t non_atomic = 0;
  for (const auto& action : actions) {
    if (!names.insert(action.name).second) {
      const bool shared_elf_eraser =
          action.name == "op.eraser.elf.cpp_v1" && action.method == "Elf+";
      const bool shared_chimp_stage =
          action.method == "Chimp128" &&
          (action.name == "op.xor.chimp128.cpp_v1" ||
           action.name == "op.trailing_zero_pack.chimp128.cpp_v1");
      if (!shared_elf_eraser && !shared_chimp_stage) {
        throw std::runtime_error("unexpected duplicate action name: " + action.name);
      }
      ++shared_action_occurrences;
    }
    if (action.name.find("catalog.specialist") != std::string::npos ||
        action.name.find("catalog.general") != std::string::npos) {
      throw std::runtime_error("whole-codec catalog action leaked into primitive registry");
    }
    if (action.evidence == remaining_codec_actions::Evidence::codec_ablation_cpp ||
        action.evidence == remaining_codec_actions::Evidence::exact_stage_cpp ||
        action.evidence == remaining_codec_actions::Evidence::reference_stage_cpp ||
        action.evidence == remaining_codec_actions::Evidence::atomic_cpp) {
      ++measurable;
    }
    codec_ablation +=
        action.evidence == remaining_codec_actions::Evidence::codec_ablation_cpp ? 1 : 0;
    exact_stage +=
        action.evidence == remaining_codec_actions::Evidence::exact_stage_cpp ? 1 : 0;
    reference_stage +=
        action.evidence == remaining_codec_actions::Evidence::reference_stage_cpp ? 1 : 0;
    atomic += action.evidence == remaining_codec_actions::Evidence::atomic_cpp ? 1 : 0;
    blocked += action.evidence ==
                       remaining_codec_actions::Evidence::blocked_no_lossless_decoder
                   ? 1
                   : 0;
    proxy += action.evidence == remaining_codec_actions::Evidence::proxy_only ? 1 : 0;
    source_unwired +=
        action.evidence == remaining_codec_actions::Evidence::source_hook_unwired ? 1 : 0;
    non_atomic +=
        action.evidence == remaining_codec_actions::Evidence::non_atomic_experimental ? 1 : 0;
    standalone_composable += action.standalone_composable ? 1 : 0;
  }
  if (names.size() != 101 || shared_action_occurrences != 3) {
    throw std::runtime_error("expected 104 method-scoped records and 101 concrete action names");
  }
  if (measurable != 88) {
    throw std::runtime_error("expected 88 measurable C++ stage candidates");
  }
  if (standalone_composable != 71) {
    throw std::runtime_error(
        "expected exact Pongo/Camel/Chimp128/Gorilla/2DIFF/SVB/Simple8b/RLBE/Sprintz/DBP/SPDP/FPC/PFPC stages plus integer RLE");
  }
  if (codec_ablation != 17 || exact_stage != 62 || reference_stage != 8 ||
      atomic != 1 || blocked != 4 || proxy != 3 ||
      source_unwired != 7 || non_atomic != 2) {
    throw std::runtime_error("action evidence distribution mismatch");
  }
  std::cout << "pipelines=" << pipelines.size() << " action_candidates=" << actions.size()
            << " measurable=" << measurable << " standalone_composable="
            << standalone_composable << " blocked_or_proxy="
            << actions.size() - measurable << '\n';
  return 0;
}
