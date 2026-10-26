#include "remaining_codec_action_catalog.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <initializer_list>
#include <stdexcept>
#include <unordered_set>

namespace remaining_codec_actions {
namespace {

using E = Evidence;

constexpr std::array kXorTrailing = {std::string_view("xor"),
                                     std::string_view("trailing_zero_pack")};
constexpr std::array kEraserXor = {std::string_view("eraser"),
                                   std::string_view("xor")};
constexpr std::array kCamel = {std::string_view("eraser"),
                               std::string_view("xor"),
                               std::string_view("bitstream_pack")};
constexpr std::array kEraserXorTrailing = {std::string_view("eraser"),
                                           std::string_view("xor"),
                                           std::string_view("trailing_zero_pack")};
constexpr std::array kScalingDeltaForBitpack = {
    std::string_view("scaling"), std::string_view("delta"),
    std::string_view("frame_of_reference"), std::string_view("bitpack_int")};
constexpr std::array kBos = {
    std::string_view("delta"),
    std::string_view("frame_of_reference"), std::string_view("outlier_separation"),
    std::string_view("frame_of_reference"), std::string_view("bitpack_int")};
constexpr std::array kReger = {
    std::string_view("value_reorder"), std::string_view("regression_predictor"),
    std::string_view("frame_of_reference"), std::string_view("bitpack_int")};
constexpr std::array kDeltaInteger = {std::string_view("delta"),
                                      std::string_view("integer_code")};
constexpr std::array kFastPfor = {std::string_view("bitpack_int"),
                                  std::string_view("integer_code")};
constexpr std::array kScalingBitpack = {std::string_view("scaling"),
                                        std::string_view("bitpack_int")};
constexpr std::array kBuff = {std::string_view("scaling"),
                              std::string_view("block_transform"),
                              std::string_view("bitpack_int")};
constexpr std::array kDeltaRunLength = {std::string_view("delta"),
                                        std::string_view("run_length")};
constexpr std::array kSprintz = {std::string_view("delta"), std::string_view("zigzag"),
                                 std::string_view("bitpack_int")};
constexpr std::array kDbp = {std::string_view("scaling"), std::string_view("delta"),
                             std::string_view("zigzag"),
                             std::string_view("bitpack_int")};
constexpr std::array kAlp = {std::string_view("scaling"),
                             std::string_view("frame_of_reference"),
                             std::string_view("bitpack_int")};
constexpr std::array kLeco = {std::string_view("regression_predictor"),
                              std::string_view("bitpack_int")};
constexpr std::array kSpdp = {std::string_view("delta"),
                              std::string_view("block_transform"),
                              std::string_view("entropy_special")};
constexpr std::array kFpzip = {std::string_view("delta"),
                               std::string_view("entropy_range")};
constexpr std::array kGzip = {std::string_view("lz_match"),
                              std::string_view("entropy_huffman"),
                              std::string_view("container_frame")};
constexpr std::array kBrotli = {std::string_view("dictionary"),
                                std::string_view("lz_match"),
                                std::string_view("entropy_huffman")};
constexpr std::array kZstd = {std::string_view("lz_match"),
                              std::string_view("match_parse_policy"),
                              std::string_view("literal_entropy"),
                              std::string_view("entropy_fse")};
constexpr std::array kLz4 = {std::string_view("lz_match"),
                             std::string_view("match_token_format"),
                             std::string_view("integer_code"),
                             std::string_view("container_frame")};
constexpr std::array kXz = {std::string_view("lz_match"),
                            std::string_view("match_parse_policy"),
                            std::string_view("entropy_range"),
                            std::string_view("container_frame")};
constexpr std::array kSnappy = {std::string_view("lz_match"),
                                std::string_view("match_token_format"),
                                std::string_view("integer_code")};
// RLE is intentionally integer-only.  Floating-point columns must first pass
// through an explicit, separately selected scaling action; the RLE action
// itself never accepts or silently rounds floating-point values.
constexpr std::array kRle = {std::string_view("run_length")};
constexpr std::array kBlosc = {std::string_view("block_transform"),
                               std::string_view("lz_match"),
                               std::string_view("match_token_format"),
                               std::string_view("block_state_control")};
constexpr std::array kBzip2 = {std::string_view("block_transform"),
                               std::string_view("symbol_recode"),
                               std::string_view("run_length"),
                               std::string_view("entropy_huffman")};
constexpr std::array kZlib = {std::string_view("lz_match"),
                              std::string_view("match_parse_policy"),
                              std::string_view("entropy_huffman"),
                              std::string_view("container_frame")};

const std::array<PipelineSpec, 35> kPipelines = {{
    // The current 86-codec benchmark routes both public labels through the
    // same TSFile Chimp encoder. Keep two method occurrences but one concrete
    // pair of action identities.
    {"Chimp", "Chimp128", "specialist", "current", "chimp128_exact_stages.cpp", E::exact_stage_cpp, kXorTrailing},
    {"Chimp128", "Chimp128", "specialist", "current", "chimp128_exact_stages.cpp", E::exact_stage_cpp, kXorTrailing},
    {"Pongo", "Pongo", "specialist", "current", "pongo_codec.cpp", E::exact_stage_cpp, kEraserXorTrailing},
    {"Gorilla", "GORILLA", "specialist", "current", "gorilla_exact_stages.cpp", E::exact_stage_cpp, kXorTrailing},
    {"2DIFF", "TS_2DIFF", "specialist", "current", "ts2diff_exact_stages.cpp", E::exact_stage_cpp, kScalingDeltaForBitpack},
    {"BOS", "BOS", "specialist", "source_BOSA", "bos_exact_stages.cpp", E::exact_stage_cpp, kBos},
    {"REGER", "REGER", "specialist", "reference_one_series", "learning_codec_exact_stages.cpp", E::reference_stage_cpp, kReger},
    {"SVB", "Stream VByte", "specialist", "delta1", "svb_exact_stages.cpp", E::exact_stage_cpp, kDeltaInteger},
    {"FastPFor", "fastpfor", "specialist", "source_C++", "fastpfor_exact_stages.cpp", E::exact_stage_cpp, kFastPfor},
    {"Simple8b", "simple8b", "specialist", "current", "simple8b_exact_stages.cpp", E::exact_stage_cpp, kScalingBitpack},
    {"RLE", "RLE", "specialist", "integer_only", "tsfile_encoding/tsfile_encode_p02.cpp + tsfile_encoding/tsfile_decode_p02.cpp", E::atomic_cpp, kRle},
    {"RLBE", "RLBE", "specialist", "current", "p04_exact_stages.cpp", E::exact_stage_cpp, kDeltaRunLength},
    {"Sprintz", "SPRINTZ", "specialist", "current", "p04_exact_stages.cpp", E::exact_stage_cpp, kSprintz},
    {"DBP", "DELTA_BINARY_PACKED", "specialist", "current", "dbp_exact_stages.cpp", E::exact_stage_cpp, kDbp},
    {"ALP", "ALP", "specialist", "reference_main_plus_rd", "learning_codec_exact_stages.cpp", E::reference_stage_cpp, kAlp},
    {"LECO", "LECO", "specialist", "current", "learning_codec_exact_stages.cpp", E::exact_stage_cpp, kLeco},
    {"Camel", "camel", "specialist", "current", "camel_exact_stages.cpp", E::exact_stage_cpp, kCamel},
    {"Elf", "elf", "specialist", "current", "elf_float_codec.cpp", E::exact_stage_cpp, kEraserXor},
    {"Elf+", "elf_plus", "specialist", "current", "elf_float_codec.cpp", E::exact_stage_cpp, kEraserXor},
    {"FPC", "fpc", "specialist", "current", "fpc_exact_stages.cpp", E::exact_stage_cpp, kXorTrailing},
    {"PFPC", "pfpc", "specialist", "current", "pfpc_exact_stages.cpp", E::exact_stage_cpp, kXorTrailing},
    {"SPDP", "spdp", "specialist", "level_5", "spdp_exact_stages.cpp", E::exact_stage_cpp, kSpdp},
    {"CORAD", "corad_tricklet", "specialist", "current", "corad_tricklet_float_codec.cpp", E::blocked_no_lossless_decoder, kScalingDeltaForBitpack},
    {"BUFF", "buff", "specialist", "current", "learning_codec_exact_stages.cpp", E::exact_stage_cpp, kBuff},
    {"FPZIP", "FPZIP", "specialist", "precision_0", "fpzip_exact_stages.cpp", E::exact_stage_cpp, kFpzip},
    {"gzip", "gzip", "general", "level_6", "native_streams.cpp", E::codec_ablation_cpp, kGzip},
    {"Brotli", "brotli", "general", "quality_6", "brotli upstream encoder", E::proxy_only, kBrotli},
    {"Zstd", "zstd", "general", "level_9", "zstd_exact_stages.cpp", E::source_hook_unwired, kZstd},
    {"LZ4", "LZ4", "general", "default", "lz4_exact_stages.cpp", E::exact_stage_cpp, kLz4},
    {"XZ", "XZ", "general", "preset_5", "lzbench/lz/xz source hook", E::source_hook_unwired, kXz},
    {"Snappy", "snappy", "general", "source_google_snappy", "snappy_exact_stages.cpp", E::exact_stage_cpp, kSnappy},
    {"Blosc", "Blosc", "general", "blosclz_5", "blosc_codec.cpp", E::codec_ablation_cpp, kBlosc},
    {"Blosc2", "Blosc2", "general", "blosclz_5", "blosc2_codec.cpp", E::codec_ablation_cpp, kBlosc},
    {"Bzip2", "Bzip2", "general", "level_4", "project-patched bzip2 source + bzip2_operator_ablation.cpp", E::codec_ablation_cpp, kBzip2},
    {"Zlib", "zlib", "general", "level_9", "native_streams.cpp", E::codec_ablation_cpp, kZlib},
}};

std::string slug(std::string_view text) {
  std::string out;
  for (unsigned char ch : text) {
    if (std::isalnum(ch)) out.push_back(static_cast<char>(std::tolower(ch)));
    else if (out.empty() || out.back() != '_') out.push_back('_');
  }
  while (!out.empty() && out.back() == '_') out.pop_back();
  return out;
}

}  // namespace

std::span<const PipelineSpec> pipelines() { return kPipelines; }

std::vector<ConcreteAction> concrete_actions() {
  std::vector<ConcreteAction> actions;
  for (const auto& pipeline : kPipelines) {
    std::unordered_set<std::string_view> seen;
    for (const auto operator_id : pipeline.operators) {
      const bool repeated = !seen.insert(operator_id).second;
      if (repeated && pipeline.method != "BOS") continue;
      const bool bos_post_outlier_for =
          repeated && pipeline.method == "BOS" && operator_id == "frame_of_reference";
      const std::string suffix = bos_post_outlier_for ? ".post_outlier" : "";
      const std::string implementation_suffix =
          bos_post_outlier_for ? "_post_outlier" : "";
      const std::string implementation_id = slug(pipeline.canonical_method) + "_" +
          std::string(operator_id) + implementation_suffix + "_cpp_v1";
      auto evidence = pipeline.evidence;
      std::string implementation_source(pipeline.implementation_source);
      if ((pipeline.method == "gzip" || pipeline.method == "Zlib" ||
           pipeline.method == "XZ") &&
          operator_id == "container_frame") {
        evidence = E::exact_stage_cpp;
        implementation_source = pipeline.method == "XZ"
                                    ? "xz_container_stage.cpp"
                                    : "deflate_container_stages.cpp";
      }
      if ((pipeline.method == "Snappy" && operator_id == "lz_match") ||
          (pipeline.method == "LZ4" && operator_id == "lz_match")) {
        evidence = E::non_atomic_experimental;
      }
      std::string action_name =
          "op." + std::string(operator_id) + "." + slug(pipeline.canonical_method) +
          suffix + ".cpp_v1";
      std::string resolved_implementation_id = implementation_id;
      if (pipeline.method == "FastPFor" && operator_id == "bitpack_int") {
        action_name = "op.bitpack_int.fastpfor_pfor.cpp_v1";
        resolved_implementation_id =
            "lemire_fastpfor8_prefix_int64_limbs_cpp_v1";
      } else if (pipeline.method == "FastPFor" &&
                 operator_id == "integer_code") {
        action_name = "op.integer_code.fastpfor_variablebyte.cpp_v1";
        resolved_implementation_id =
            "lemire_variablebyte_tail_int64_limbs_cpp_v1";
      } else if (pipeline.method == "Elf+" && operator_id == "eraser") {
        // Elf and Elf+ call the same byte-identical eraser implementation.
        // Keep two method-scoped occurrences but one concrete action identity.
        action_name = "op.eraser.elf.cpp_v1";
        resolved_implementation_id = "elf32_shared_eraser_cpp_v1";
      } else if (pipeline.method == "RLE") {
        // Match the actual integer-only runtime registration exactly.
        action_name = "op.run_length.tsfile_int64.v1";
        resolved_implementation_id = "tsfile_rle_int64_v1";
      } else if (pipeline.method == "LECO" && operator_id == "regression_predictor") {
        resolved_implementation_id = "leco_linear_model_residual_cpp_v1";
      } else if (pipeline.method == "LECO" && operator_id == "bitpack_int") {
        resolved_implementation_id = "leco_residual_bitpack_cpp_v1";
      } else if (pipeline.method == "ALP" && operator_id == "scaling") {
        resolved_implementation_id = "alp_exact_decimal_scaling_cpp_v1";
      } else if (pipeline.method == "ALP" && operator_id == "frame_of_reference") {
        resolved_implementation_id = "alp_for_normalization_cpp_v1";
      } else if (pipeline.method == "ALP" && operator_id == "bitpack_int") {
        resolved_implementation_id = "alp_for_bitpack_cpp_v1";
      } else if (pipeline.method == "REGER" && operator_id == "value_reorder") {
        resolved_implementation_id = "reger_stable_value_reorder_cpp_v1";
      } else if (pipeline.method == "REGER" && operator_id == "regression_predictor") {
        resolved_implementation_id = "reger_ar1_residual_cpp_v1";
      } else if (pipeline.method == "REGER" && operator_id == "frame_of_reference") {
        resolved_implementation_id = "reger_residual_for_cpp_v1";
      } else if (pipeline.method == "REGER" && operator_id == "bitpack_int") {
        resolved_implementation_id = "reger_residual_bitpack_cpp_v1";
      } else if (pipeline.method == "BUFF" && operator_id == "scaling") {
        resolved_implementation_id = "buff_exact_decimal_scaling_cpp_v1";
      } else if (pipeline.method == "BUFF" && operator_id == "block_transform") {
        action_name = "op.block_transform.buff_column_split.cpp_v1";
        resolved_implementation_id = "buff_byte_column_split_cpp_v1";
      } else if (pipeline.method == "BUFF" && operator_id == "bitpack_int") {
        resolved_implementation_id = "buff_fixed_width_bitpack_cpp_v1";
      }
      actions.push_back({
          action_name,
          std::string(operator_id), resolved_implementation_id,
          implementation_source, std::string(pipeline.method),
          std::string(pipeline.selected_level), evidence,
          evidence == Evidence::atomic_cpp || evidence == Evidence::exact_stage_cpp ||
              evidence == Evidence::reference_stage_cpp,
      });
    }
    if (pipeline.method == "ALP") {
      actions.push_back({
          "op.dictionary.alp_rd.cpp_v1", "dictionary",
          "alp_rd_dictionary_cpp_v1", "learning_codec_exact_stages.cpp",
          "ALP", "reference_rd_alternative", Evidence::reference_stage_cpp, true,
      });
    }
  }
  return actions;
}

std::string_view evidence_name(Evidence evidence) {
  switch (evidence) {
    case Evidence::codec_ablation_cpp: return "codec_ablation_cpp";
    case Evidence::exact_stage_cpp: return "exact_stage_cpp";
    case Evidence::reference_stage_cpp: return "reference_stage_cpp";
    case Evidence::atomic_cpp: return "atomic_cpp";
    case Evidence::source_hook_unwired: return "source_hook_unwired";
    case Evidence::proxy_only: return "proxy_only";
    case Evidence::non_atomic_experimental: return "non_atomic_experimental";
    case Evidence::blocked_no_lossless_decoder: return "blocked_no_lossless_decoder";
  }
  throw std::logic_error("unknown action evidence");
}

}  // namespace remaining_codec_actions
