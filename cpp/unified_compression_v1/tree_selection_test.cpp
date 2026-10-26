#include "api.hpp"
#include "selection.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace uc = unified_compression_v1;
using Json = nlohmann::json;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <class F>
void rejected(F&& operation, std::string_view expected_code = {}) {
  try {
    operation();
  } catch (const uc::Error& error) {
    if (!expected_code.empty()) require(error.code() == expected_code, error.what());
    return;
  } catch (const std::exception&) {
    require(expected_code.empty(), "rejection did not use the expected unified error");
    return;
  }
  throw std::runtime_error("invalid request unexpectedly succeeded");
}

std::uint64_t get_le(std::span<const std::uint8_t> bytes, std::size_t offset,
                     unsigned width) {
  require(offset + width <= bytes.size(), "test fixture read exceeds frame");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < width; ++i)
    value |= std::uint64_t(bytes[offset + i]) << (8U * i);
  return value;
}

void refresh_ucc1_digest(uc::Bytes& frame) {
  require(frame.size() >= 32, "test fixture is not a complete UCC1 frame");
  std::array<unsigned char, 32> digest{};
  unsigned digest_size = 0;
  require(EVP_Digest(frame.data(), frame.size() - digest.size(), digest.data(),
                     &digest_size, EVP_sha256(), nullptr) == 1 &&
              digest_size == digest.size(),
          "test fixture SHA-256 failed");
  std::copy(digest.begin(), digest.end(), frame.end() - digest.size());
}

uc::Bytes rewrite_ucc1_method(uc::Bytes frame, std::string_view replacement) {
  require(frame.size() >= 52 && std::equal(frame.begin(), frame.begin() + 4, "UCC1"),
          "test fixture is not UCC1");
  const auto name_size = get_le(frame, 8, 4);
  require(name_size == replacement.size(), "replacement route width mismatch");
  std::copy(replacement.begin(), replacement.end(), frame.begin() + 20);
  refresh_ucc1_digest(frame);
  return frame;
}

uc::Column ordinary_column(std::string_view dtype) {
  if (dtype == "int32") {
    const std::vector<std::int32_t> values{17, -4, 17, 0, 99, -100, 7, 7};
    return uc::make_column(std::span<const std::int32_t>(values));
  }
  if (dtype == "int64") {
    const std::vector<std::int64_t> values{17, -4, 17, 0, 99, -100, 7, 7};
    return uc::make_column(std::span<const std::int64_t>(values));
  }
  if (dtype == "float32") {
    const std::vector<float> values{1.25F, -3.5F, 1.25F, 0.0F,
                                    16.0F, 9.75F, -2.0F, 4.5F};
    return uc::make_column(std::span<const float>(values));
  }
  if (dtype == "float64") {
    const std::vector<double> values{1.25, -3.5, 1.25, 0.0,
                                     16.0, 9.75, -2.0, 4.5};
    return uc::make_column(std::span<const double>(values));
  }
  throw std::runtime_error("unknown test dtype");
}

uc::Column nullable_i64() {
  const std::vector<std::int64_t> compact_values{
      std::numeric_limits<std::int64_t>::min(), -1, 0, 17,
      std::numeric_limits<std::int64_t>::max()};
  auto column = uc::make_column(std::span<const std::int64_t>(compact_values));
  column.row_count = 7;
  column.missing = {0x22};  // Rows 1 and 5 are NULL; five compact values remain.
  return column;
}

uc::Column edge_f32() {
  const std::vector<float> values{
      0.0F, -0.0F, std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(UINT32_C(0x7fc01234)),
      std::bit_cast<float>(UINT32_C(0xffc05678))};
  return uc::make_column(std::span<const float>(values));
}

uc::Column edge_f64() {
  const std::vector<double> values{
      0.0, -0.0, std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::bit_cast<double>(UINT64_C(0x7ff8000000001234)),
      std::bit_cast<double>(UINT64_C(0xfff8000000005678))};
  return uc::make_column(std::span<const double>(values));
}

void require_result(const uc::Registry& registry,
                    const uc::CompressionResult& result,
                    const uc::Column& source,
                    std::size_t samples = 1) {
  require(result.bit_exact, "result lacks bit-exact verification");
  require(result.compressed_bytes == result.compressed_data.size(),
          "complete frame size is not charged");
  require(result.original_numeric_bytes == source.values.size(),
          "numeric source byte accounting mismatch");
  require(result.compression_samples_ns.size() == samples &&
              result.decompression_samples_ns.size() == samples,
          "measurement sample count mismatch");
  if (source.values.empty()) {
    require(!result.compression_ratio, "zero-byte numeric input must have null ratio");
  } else {
    require(result.compression_ratio &&
                *result.compression_ratio ==
                    static_cast<double>(result.compressed_bytes) /
                        static_cast<double>(source.values.size()),
            "ratio does not charge the complete frame");
  }
  require(registry.decompress(result.compressed_data) == source,
          "frame-only decode changed source bits or schema");
  require(result.execution_audit.is_object(), "execution audit is missing");
  require(result.execution_audit.value("strict", false),
          "successful ranked execution was not strict");
  require(!result.execution_audit.value("fallback_used", true),
          "successful ranked execution used a fallback");
}

Json minimal_tree() {
  return {
      {"schema", "ucc-operator-tree-v1"},
      {"source", {{"root_kind", "numeric"}}},
      {"nodes", Json::array({
                    {{"id", "z"},
                     {"op", "zstd3"},
                     {"inputs", Json::array({"values"})},
                     {"parameters", Json::object()}},
                })},
  };
}

Json branching_tree() {
  return {
      {"schema", "ucc-operator-tree-v1"},
      {"source", {{"root_kind", "numeric"}}},
      {"nodes", Json::array({
                    {{"id", "dictionary"},
                     {"op", "dictionary"},
                     {"inputs", Json::array({"values"})},
                     {"parameters", Json::object()}},
                    {{"id", "dictionary_values"},
                     {"op", "zstd3"},
                     {"inputs", Json::array({"dictionary:0"})},
                     {"parameters", Json::object()}},
                    {{"id", "dictionary_ids"},
                     {"op", "lz4"},
                     {"inputs", Json::array({"dictionary:1"})},
                     {"parameters", Json::object()}},
                })},
  };
}

void test_registry_and_fixed30(uc::Registry& registry) {
  require(registry.methods().size() == 218,
          "registry must contain 86 algorithms, 102 operators and 30 combinations");
  std::size_t combinations = 0;
  std::set<std::string> combination_names;
  for (const auto& method : registry.methods()) {
    if (method.kind != "combo") continue;
    ++combinations;
    require(method.name.starts_with("combo_v1:"), "combo canonical name is not namespaced");
    require(combination_names.insert(method.name).second, "duplicate combo canonical name");
    require(!method.dtypes.empty(), "combo has no declared source dtype");
    for (const auto& dtype : method.dtypes) {
      const auto source = ordinary_column(dtype);
      const auto result = registry.compress(source, method.name);
      require(result.method == method.name, "combo result route changed");
      require_result(registry, result, source);
    }
  }
  require(combinations == 30, "fixed combination catalog is incomplete");
  require(combination_names.contains("combo_v1:zstd3") &&
              combination_names.contains("combo_v1:lz4"),
          "direct combo baselines are missing");

  const auto nullable = nullable_i64();
  require_result(registry, registry.compress(nullable, "combo_v1:columnize_zstd3"),
                 nullable);
  const uc::Column empty{"float64", 0, {}, {}};
  const uc::Column all_null{"int64", 5, {}, {0x1f}};
  require_result(registry, registry.compress(empty, "combo_v1:zstd3"), empty);
  require_result(registry, registry.compress(all_null, "combo_v1:lz4"), all_null);
  require_result(registry,
                 registry.compress(empty, "combo_v1:a048_planes_zstd3"), empty);

  rejected([&] {
    registry.compress(ordinary_column("float32"), "combo_v1:for_bitpack_zstd3");
  }, "domain_rejected");
  rejected([&] {
    registry.compress(ordinary_column("int64"),
                      "combo_v1:float_deconstruct_zstd3");
  }, "domain_rejected");
}

void test_explicit_trees(uc::Registry& registry) {
  const auto contract = registry.tree_contract();
  require(contract.value("schema", "") == "ucc-operator-tree-v1" &&
              contract.at("security").at("strict_raw_fallback_forbidden") == true,
          "public tree contract does not declare strict execution");
  const auto excluded = contract.at("security").at("excluded_families");
  require(std::find(excluded.begin(), excluded.end(), "NNCP") != excluded.end() &&
              std::find(excluded.begin(), excluded.end(), "PPMd") != excluded.end(),
          "public tree contract omits disabled families");

  const auto expected_id =
      registry.compress_tree(ordinary_column("int64"), minimal_tree()).method;
  require(expected_id.starts_with("tree_v1:") && expected_id.size() == 72,
          "tree identifier must bind a SHA-256 digest");

  for (const auto& source : {edge_f32(), edge_f64(), nullable_i64()}) {
    const auto result = registry.compress_tree(source, minimal_tree(), {1, 2});
    require(result.method == expected_id, "explicit tree result route is not canonical");
    require_result(registry, result, source, 2);
    require(result.execution_audit.value("tree_sha256", "").size() == 64,
            "explicit tree audit lacks its canonical digest");
  }

  const auto branch_source = ordinary_column("int64");
  const auto branch = registry.compress_tree(branch_source, branching_tree());
  require(branch.method.starts_with("tree_v1:") && branch.method != expected_id,
          "branching tree route does not uniquely bind its recipe");
  require_result(registry, branch, branch_source);
  require(branch.compressed_data !=
              registry.compress_tree(branch_source, minimal_tree()).compressed_data,
          "distinct trees unexpectedly produced an identical complete route/frame");

  auto invalid = minimal_tree();
  invalid["unexpected"] = true;
  rejected([&] { registry.compress_tree(branch_source, invalid); });

  invalid = minimal_tree();
  invalid["nodes"][0]["op"] = "unknown_transform";
  rejected([&] { registry.compress_tree(branch_source, invalid); });

  for (const auto* forbidden : {"nncp", "ppmd", "dmc", "paq8", "cmix", "lstm"}) {
    invalid = minimal_tree();
    invalid["nodes"][0]["op"] = forbidden;
    rejected([&] { registry.compress_tree(branch_source, invalid); });
  }

  invalid = minimal_tree();
  invalid["nodes"].push_back({{"id", "second"},
                               {"op", "lz4"},
                               {"inputs", Json::array({"values"})},
                               {"parameters", Json::object()}});
  rejected([&] { registry.compress_tree(branch_source, invalid); });

  invalid = minimal_tree();
  invalid["nodes"][0]["inputs"] = Json::array({"later:0"});
  rejected([&] { registry.compress_tree(branch_source, invalid); });

  invalid = minimal_tree();
  invalid["nodes"][0]["parameters"] = {{"level", 3}};
  rejected([&] { registry.compress_tree(branch_source, invalid); });

  invalid = minimal_tree();
  invalid["nodes"][0]["op"] = "bitpack";
  invalid["nodes"][0]["parameters"] = {
      {"alignment_bytes", 1}, {"bit_order", "msb"},
      {"minimum_width", 0}, {"wide_cutoff", 64}};
  rejected([&] { registry.compress_tree(edge_f64(), invalid); });
}

void test_routes_and_bindings(uc::Registry& registry) {
  const auto source = ordinary_column("int64");
  const auto first = registry.compress(source, "combo_v1:columnize_zstd3_lz4");
  auto forged = rewrite_ucc1_method(first.compressed_data,
                                    "combo_v1:columnize_lz4_zstd3");
  rejected([&] { registry.decompress(forged); });

  const auto small_tree = registry.compress_tree(source, minimal_tree());
  const auto branch_tree = registry.compress_tree(source, branching_tree());
  require(small_tree.method.size() == branch_tree.method.size(),
          "tree routes must have a fixed digest width");
  forged = rewrite_ucc1_method(branch_tree.compressed_data, small_tree.method);
  rejected([&] { registry.decompress(forged); });

  forged = branch_tree.compressed_data;
  const auto name_size = get_le(forged, 8, 4);
  const auto payload_offset = 20 + static_cast<std::size_t>(name_size);
  require(payload_offset + 32 < forged.size(), "tree frame has no native payload");
  forged[payload_offset + 17] ^= 0x40;
  refresh_ucc1_digest(forged);
  rejected([&] { registry.decompress(forged); });
}

void test_existing_routes(uc::Registry& registry) {
  const auto source = nullable_i64();
  const auto algorithm = registry.compress(source, "algorithm:Zstd", {1, 2});
  require(algorithm.method == "algorithm:Zstd", "algorithm route changed");
  require_result(registry, algorithm, source, 2);

  const auto op = registry.compress(ordinary_column("int64"), "operator:A013");
  require(op.method == "operator:A013", "operator route changed");
  require_result(registry, op, ordinary_column("int64"));

  rejected([&] { registry.compress(source, "algorithm:NNCP"); }, "method_disabled");
  for (const auto* family : {"PPMd", "DMC", "PAQ8", "CMIX", "LSTM"})
    rejected([&] { registry.compress(source, family); }, "method_disabled");

  struct ConservativeAuditCase {
    const char* method;
    const char* dtype;
    const char* candidate_id;
  };
  constexpr ConservativeAuditCase conservative_cases[] = {
      {"algorithm:FastLZ", "int64", "fastlz-unverified"},
      {"algorithm:FPCmp-R", "float64", "fpcmp-r-unverified"},
      {"algorithm:FPCmp-S", "float64", "fpcmp-s-unverified"},
  };
  for (const auto& audit_case : conservative_cases) {
    const auto method = std::find_if(
        registry.methods().begin(), registry.methods().end(),
        [&](const auto& candidate) { return candidate.name == audit_case.method; });
    if (method == registry.methods().end() || !method->enabled || !method->available)
      continue;
    require(std::find(method->dtypes.begin(), method->dtypes.end(), audit_case.dtype) !=
                method->dtypes.end(),
            "conservative audit fixture does not use a declared dtype");

    const auto audit_source = ordinary_column(audit_case.dtype);
    const auto measured = registry.compress(audit_source, method->name);
    require(measured.bit_exact &&
                registry.decompress(measured.compressed_data) == audit_source,
            "conservative compatibility path is not bit exact");
    require(measured.execution_audit.is_object() &&
                !measured.execution_audit.value("strict", true) &&
                measured.execution_audit.contains("fallback_used") &&
                measured.execution_audit.at("fallback_used").is_null() &&
                measured.execution_audit.contains("raw_fallback_stage_count") &&
                measured.execution_audit.at("raw_fallback_stage_count").is_null() &&
                measured.execution_audit.value("strict_status", "") ==
                    "adapter_bypass_unverified",
            "legacy algorithm path made an unsupported no-fallback claim");

    uc::SelectionOptions options;
    options.budget.max_trials = 1;
    options.candidates = {uc::Candidate::method_candidate(
        audit_case.candidate_id, method->name)};
    const auto selection = uc::select(registry, audit_source, options);
    require(selection.feedback.size() == 1 &&
                selection.feedback.front().status == "non_strict" &&
                !selection.feedback.front().strict &&
                !selection.selected_candidate && !selection.selected_result,
            "selector ranked a bit-exact path whose fallback status is unknown");
  }
}

void test_budgeted_selection(uc::Registry& registry) {
  const auto source = ordinary_column("int64");

  const auto fixed = uc::fixed30_candidates();
  require(fixed.size() == 30, "selector fixed scope is not the unified fixed30");
  std::set<std::string> fixed_ids;
  for (const auto& candidate : fixed) {
    require(fixed_ids.insert(candidate.id).second, "selector fixed30 has duplicate IDs");
    require(candidate.method && !candidate.tree,
            "fixed30 candidate does not route through the unified method registry");
  }

  uc::SelectionOptions options;
  options.candidate_scope = "integration-budget-zero";
  options.budget.max_trials = 0;
  std::size_t proposer_calls = 0;
  options.proposer = [&](const std::vector<uc::TrialFeedback>&)
      -> std::optional<uc::Candidate> {
    ++proposer_calls;
    return uc::Candidate::method_candidate("zstd", "combo_v1:zstd3");
  };
  auto selected = uc::select(registry, source, options);
  require(proposer_calls == 0, "zero budget invoked the proposer");
  require(selected.feedback.empty() && !selected.selected_candidate &&
              !selected.selected_result,
          "zero budget executed or selected a candidate");
  require(selected.stop_reason == "trial_budget_exhausted",
          "zero budget stop reason changed");
  require(!selected.costs.final_recompression_performed,
          "zero budget reported final recompression");

  options = {};
  options.candidate_scope = "integration-infeasible";
  options.budget.max_trials = 3;
  options.constraints.max_compressed_bytes = 1;
  options.candidates = {
      uc::Candidate::method_candidate("zstd", "combo_v1:zstd3"),
      uc::Candidate::method_candidate("lz4", "combo_v1:lz4")};
  selected = uc::select(registry, source, options);
  require(selected.feedback.size() == 2, "infeasible candidates were not both measured");
  require(!selected.selected_candidate && !selected.selected_result,
          "constraint-violating candidate was selected");
  require(std::all_of(selected.feedback.begin(), selected.feedback.end(),
                      [](const auto& trial) {
                        return trial.status == "ok" && trial.strict &&
                               trial.bit_exact && !trial.feasible &&
                               trial.compressed_bytes.has_value();
                      }),
          "infeasible feedback lost successful harness measurements");

  options = {};
  options.candidate_scope = "integration-feedback";
  options.budget.max_trials = 4;
  proposer_calls = 0;
  options.proposer = [&](const std::vector<uc::TrialFeedback>& feedback)
      -> std::optional<uc::Candidate> {
    ++proposer_calls;
    if (feedback.empty())
      return uc::Candidate::method_candidate("first", "combo_v1:zstd3");
    if (feedback.size() == 1) {
      require(feedback.front().status == "ok" && feedback.front().strict &&
                  feedback.front().compressed_bytes.has_value(),
              "proposer did not receive complete first-trial feedback");
      return uc::Candidate::method_candidate("second", "combo_v1:lz4");
    }
    require(feedback.size() == 2, "proposer feedback history is incomplete");
    return std::nullopt;
  };
  selected = uc::select(registry, source, options);
  require(proposer_calls == 3 && selected.feedback.size() == 2,
          "proposer/feedback loop did not terminate after two measurements");
  require(selected.stop_reason == "proposer_exhausted",
          "proposer exhaustion was not reported");
  require(selected.selected_candidate && selected.selected_result,
          "feasible proposal loop produced no winner");
  const auto smallest = std::min_element(
      selected.feedback.begin(), selected.feedback.end(),
      [](const auto& left, const auto& right) {
        return *left.compressed_bytes < *right.compressed_bytes;
      });
  require(selected.selected_candidate->id == smallest->candidate.id &&
              selected.selected_result->compressed_bytes == *smallest->compressed_bytes,
          "byte objective did not return the measured minimum");
  require(!selected.costs.final_recompression_performed,
          "selector recompressed instead of returning its measured winner");
  const auto report = uc::selection_report(selected);
  require(report.value("schema", "") ==
              "unified-compression-selection-report-v1" &&
              report.value("trial_count", 0U) == 2 &&
              report.at("selected").at("compressed_bytes") ==
                  selected.selected_result->compressed_bytes &&
              !report.value("compressed_binary_included", true),
          "selection report lost feedback, winner metrics, or binary policy");

  options = {};
  options.candidate_scope = "integration-errors";
  options.budget.max_trials = 3;
  options.candidates = {
      uc::Candidate::method_candidate("broken", "combo_v1:does-not-exist"),
      uc::Candidate::method_candidate("valid", "combo_v1:zstd3")};
  selected = uc::select(registry, source, options);
  require(selected.feedback.size() == 2 &&
              selected.feedback[0].status == "error" &&
              selected.feedback[1].status == "ok",
          "candidate failure was not isolated from the next proposal");
  require(selected.selected_candidate && selected.selected_candidate->id == "valid",
          "valid candidate after an error was not eligible for selection");

  options = {};
  options.candidate_scope = "integration-unsupported";
  options.budget.max_trials = 2;
  options.candidates = {
      uc::Candidate::method_candidate("wrong-dtype",
                                      "combo_v1:for_bitpack_zstd3")};
  selected = uc::select(registry, ordinary_column("float32"), options);
  require(selected.feedback.size() == 1 &&
              selected.feedback.front().status == "unsupported" &&
              selected.feedback.front().error_code == "domain_rejected",
          "declared input-domain rejection was not reported as unsupported");
  require(!selected.selected_candidate && !selected.selected_result,
          "unsupported candidate was selected");

  options = {};
  options.candidate_scope = "integration-duplicate";
  options.budget.max_trials = 3;
  options.proposer = [](const std::vector<uc::TrialFeedback>& feedback)
      -> std::optional<uc::Candidate> {
    if (feedback.size() < 2)
      return uc::Candidate::method_candidate("same", "combo_v1:zstd3");
    return std::nullopt;
  };
  selected = uc::select(registry, source, options);
  require(selected.feedback.size() == 2 &&
              selected.feedback[0].status == "ok" &&
              selected.feedback[1].status == "duplicate",
          "duplicate proposal did not consume one visible trial");

  options = {};
  options.candidate_scope = "integration-wall-zero";
  options.budget.max_trials = 8;
  options.budget.soft_wall_time_ns = 0;
  proposer_calls = 0;
  options.proposer = [&](const std::vector<uc::TrialFeedback>&)
      -> std::optional<uc::Candidate> {
    ++proposer_calls;
    return uc::Candidate::method_candidate("unreachable", "combo_v1:zstd3");
  };
  selected = uc::select(registry, source, options);
  require(proposer_calls == 0 && selected.feedback.empty(),
          "expired wall budget admitted a proposal");
  require(selected.stop_reason == "soft_wall_budget_reached",
          "expired wall budget stop reason changed");

  options = {};
  options.candidate_scope = "integration-explicit-tree";
  options.budget.max_trials = 1;
  options.candidates = {
      uc::Candidate::tree_candidate("tree", branching_tree())};
  selected = uc::select(registry, source, options);
  require(selected.feedback.size() == 1 && selected.feedback[0].status == "ok" &&
              selected.feedback[0].execution_audit.value("strict", false),
          "selector did not feed an explicit tree through the harness");
  require(selected.selected_result &&
              selected.selected_result->method.starts_with("tree_v1:"),
          "selector lost the explicit tree route/binding");
  require(registry.decompress(selected.selected_result->compressed_data) == source,
          "selected tree result is not independently decodable");

  const auto parsed = uc::selection_options_from_json({
      {"objective", "compressed_bytes"},
      {"candidate_scope", "json-explicit-tree"},
      {"budget", {{"max_trials", 1U}}},
      {"measurement", {{"warmups", 0U}, {"repeats", 1U}}},
      {"candidates",
       Json::array({{{"id", "tree-from-json"},
                     {"tree", branching_tree()},
                     {"metadata", {{"proposed_by", "test-recommender"}}}}})},
  });
  require(parsed.candidates.size() == 1 && parsed.candidates.front().tree &&
              parsed.candidate_scope == "json-explicit-tree",
          "JSON selection request did not preserve an explicit tree candidate");
  const auto parsed_result = uc::select(registry, source, parsed);
  require(parsed_result.selected_candidate && parsed_result.selected_result &&
              parsed_result.selected_candidate->id == "tree-from-json",
          "JSON explicit-tree selection did not execute and select its candidate");
  rejected([&] {
    uc::selection_options_from_json({{"budget", {{"max_trials", 1U}}},
                                     {"unknown", true}});
  });
}

}  // namespace

int main() {
  try {
    uc::Registry registry;
    test_registry_and_fixed30(registry);
    test_explicit_trees(registry);
    test_routes_and_bindings(registry);
    test_existing_routes(registry);
    test_budgeted_selection(registry);
    std::cout << "unified tree integration: fixed30/tree/selection/binding/compatibility checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
