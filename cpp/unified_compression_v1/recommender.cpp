#include "recommender.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace unified_compression_v1 {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kCatalogSha256 =
    "568f2422b415d2869712a4a0e5f8edb37205555d92ecfe3fc0b36946c1d56326";

constexpr std::array<std::string_view, 24> kModelCandidates{
    "raw",
    "zstd1",
    "zstd3",
    "zstd9",
    "lz4frame",
    "zlib6",
    "huffman_a077",
    "rans_a080",
    "columnize_zstd3",
    "columnize_lz4",
    "columnize_zlib6",
    "xor_zstd3",
    "xor_lz4",
    "xor_zlib6",
    "float_deconstruct_zstd3",
    "float_deconstruct_lz4",
    "float_deconstruct_zstd3_lz4",
    "for_bitpack_zstd3",
    "for_bitpack_lz4",
    "a048_planes_zstd3",
    "a048_planes_lz4",
    "a048_planes_zstd3_lz4",
    "split_zstd3_lz4",
    "split_lz4_zstd3",
};

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("prepared recommender: " + message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::uint64_t elapsed_ns(Clock::time_point begin) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin)
          .count());
}

std::string lower_sha(std::string value, std::string_view context) {
  require(value.size() == 64, std::string(context) + " must be a SHA-256 hex string");
  for (auto& ch : value) {
    require(std::isxdigit(static_cast<unsigned char>(ch)) != 0,
            std::string(context) + " must be a SHA-256 hex string");
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

std::vector<std::string> strings(const Json& value, std::string_view context) {
  require(value.is_array(), std::string(context) + " must be an array");
  std::vector<std::string> result;
  std::set<std::string> unique;
  for (const auto& item : value) {
    require(item.is_string() && !item.get_ref<const std::string&>().empty(),
            std::string(context) + " must contain non-empty strings");
    auto text = item.get<std::string>();
    require(unique.insert(text).second,
            std::string(context) + " contains a duplicate");
    result.push_back(std::move(text));
  }
  return result;
}

std::string target_for(SelectionObjective objective) {
  switch (objective) {
    case SelectionObjective::compressed_bytes: return "space";
    case SelectionObjective::compression_time: return "encode";
    case SelectionObjective::decompression_time: return "decode";
    case SelectionObjective::balanced: return "balanced";
  }
  fail("unknown selection objective");
}

std::optional<std::string> mapped_method(std::string_view old_id) {
  if (old_id == "zstd3") return "combo_v1:zstd3";
  if (old_id == "lz4frame") return "combo_v1:lz4";
  static const std::set<std::string_view> intersection{
      "columnize_zstd3",
      "columnize_lz4",
      "xor_zstd3",
      "xor_lz4",
      "float_deconstruct_zstd3",
      "float_deconstruct_lz4",
      "float_deconstruct_zstd3_lz4",
      "for_bitpack_zstd3",
      "for_bitpack_lz4",
      "a048_planes_zstd3",
      "a048_planes_lz4",
      "a048_planes_zstd3_lz4",
  };
  if (intersection.contains(old_id))
    return "combo_v1:" + std::string(old_id);
  return std::nullopt;
}

bool model_candidate_eligible(std::string_view id, const Column& column) {
  const bool nonempty = !column.values.empty();
  const bool word64 = column.dtype == "int64" || column.dtype == "float64";
  if (id == "huffman_a077" || id == "rans_a080") return nonempty;
  if (id.starts_with("xor_")) return word64;
  if (id.starts_with("float_deconstruct_"))
    return column.dtype == "float32" && nonempty;
  if (id.starts_with("for_bitpack_"))
    return column.dtype == "int64" && nonempty;
  if (id.starts_with("a048_planes_")) return word64;
  return true;
}

std::vector<std::string> eligible_fixed30(const Registry& registry,
                                          const Column& column) {
  std::set<std::string> fixed_names;
  for (const auto& candidate : fixed30_candidates())
    fixed_names.insert(*candidate.method);
  std::vector<std::string> result;
  for (const auto& method : registry.methods()) {
    if (!fixed_names.contains(method.name)) continue;
    const bool dtype_ok = std::find(method.dtypes.begin(), method.dtypes.end(),
                                    column.dtype) != method.dtypes.end();
    const bool requires_nonempty =
        method.contract.value("requires_nonempty_values", false);
    if (method.kind == "combo" && method.enabled && method.available && dtype_ok &&
        (!requires_nonempty || !column.values.empty()))
      result.push_back(method.name);
  }
  return result;
}

}  // namespace

struct PreparedRecommender::Impl {
  Json model;
  Json split;
  operator_recommend135_v1::PreparedModel prepared_model;
  std::string split_sha256;
  bool split_file_sha_verified = false;
  std::vector<std::string> training_datasets;
  std::vector<std::string> confirm_datasets;

  Impl(Json model_input, Json split_input, std::string expected_split_sha256)
      : model(std::move(model_input)),
        split(std::move(split_input)),
        prepared_model(model) {
    require(model.is_object() &&
                model.value("schema", "") == "operator-recommend135-model-v1",
            "unsupported frozen model schema");
    require(split.is_object() &&
                split.value("schema", "") ==
                    "operator-selection-source-group-split-v1",
            "unsupported frozen source-group split schema");
    require(model.contains("candidate_catalog_sha256") &&
                model.at("candidate_catalog_sha256").is_string() &&
                lower_sha(model.at("candidate_catalog_sha256").get<std::string>(),
                          "model candidate catalog SHA-256") == kCatalogSha256,
            "frozen model candidate catalog SHA-256 mismatch");

    const auto candidate_ids = strings(model.at("candidate_ids"),
                                       "model.candidate_ids");
    require(candidate_ids.size() == kModelCandidates.size(),
            "frozen model candidate scope must contain exactly 24 entries");
    for (std::size_t index = 0; index < candidate_ids.size(); ++index)
      require(candidate_ids[index] == kModelCandidates[index],
              "frozen model candidate scope or order changed");

    require(model.contains("formal_protocol") &&
                model.at("formal_protocol").is_object(),
            "model lacks formal_protocol");
    const auto& formal = model.at("formal_protocol");
    require(formal.contains("confirm_labels_in_fit") &&
                formal.at("confirm_labels_in_fit").is_boolean() &&
                !formal.at("confirm_labels_in_fit").get<bool>(),
            "confirm labels must not enter frozen model fitting");
    require(formal.contains("split_sha256") &&
                formal.at("split_sha256").is_string(),
            "model lacks frozen split SHA-256");
    split_sha256 = lower_sha(formal.at("split_sha256").get<std::string>(),
                             "model split SHA-256");
    if (!expected_split_sha256.empty()) {
      split_file_sha_verified = true;
      require(lower_sha(std::move(expected_split_sha256),
                        "provided split file SHA-256") == split_sha256,
              "provided split file SHA-256 differs from the frozen model");
    }

    require(split.contains("confirm_dataset_ids"),
            "split lacks confirm_dataset_ids");
    confirm_datasets =
        strings(split.at("confirm_dataset_ids"), "split.confirm_dataset_ids");
    require(!confirm_datasets.empty(), "split confirm group is empty");
    const std::set<std::string> confirm_set(confirm_datasets.begin(),
                                            confirm_datasets.end());

    require(model.contains("training") && model.at("training").is_object() &&
                model.at("training").contains("dataset_ids"),
            "model lacks training.dataset_ids");
    training_datasets = strings(model.at("training").at("dataset_ids"),
                                "model.training.dataset_ids");
    const auto formal_training = strings(formal.at("training_dataset_ids"),
                                         "formal_protocol.training_dataset_ids");
    require(training_datasets == formal_training,
            "declared model training dataset lists disagree");
    for (const auto& dataset : training_datasets) {
      require(!confirm_set.contains(dataset),
              "confirm dataset appears in frozen model training: " + dataset);
      require(split.contains("dataset_assignments") &&
                  split.at("dataset_assignments").is_object() &&
                  split.at("dataset_assignments").contains(dataset) &&
                  split.at("dataset_assignments").at(dataset).is_object() &&
                  split.at("dataset_assignments")
                          .at(dataset)
                          .value("split", "") == "development",
              "training dataset is not assigned to development: " + dataset);
    }
    for (const auto& dataset : confirm_datasets) {
      require(split.at("dataset_assignments").contains(dataset) &&
                  split.at("dataset_assignments")
                          .at(dataset)
                          .value("split", "") == "confirm",
              "confirm_dataset_ids disagrees with split assignment: " + dataset);
    }
  }
};

PreparedRecommender::PreparedRecommender(
    const Json& frozen_model, const Json& frozen_split,
    std::string expected_split_sha256)
    : impl_(std::make_shared<const Impl>(frozen_model, frozen_split,
                                         std::move(expected_split_sha256))) {}

RecommendationResult PreparedRecommender::recommend(
    const Registry& registry, const Column& column,
    SelectionOptions options) const {
  require(!options.proposer && options.candidates.empty(),
          "model mode does not accept caller candidates or proposer callbacks");
  validate_column(column);
  const auto total_begin = Clock::now();
  const auto prediction_begin = Clock::now();

  std::vector<std::string> model_eligible;
  for (const auto id : kModelCandidates)
    if (model_candidate_eligible(id, column))
      model_eligible.emplace_back(id);
  const auto features = operator_recommend135_v1::extract_features(
      column.values, column.dtype, column.missing, column.row_count);
  Json prediction = impl_->prepared_model.predict(features, model_eligible);

  const auto fixed_eligible = eligible_fixed30(registry, column);
  const std::set<std::string> fixed_eligible_set(fixed_eligible.begin(),
                                                 fixed_eligible.end());
  const auto target = target_for(options.objective);
  const auto& ranking = prediction.at("targets").at(target).at("ranking");
  require(ranking.is_array(), "prepared prediction lacks target ranking");
  std::vector<std::string> seed_order;
  std::set<std::string> seeded;
  Json mapped = Json::array();
  Json ignored = Json::array();
  std::size_t rank = 0;
  for (const auto& entry : ranking) {
    ++rank;
    const auto old_id = entry.at("candidate_id").get<std::string>();
    const auto method = mapped_method(old_id);
    if (!method) {
      ignored.push_back({{"model_candidate_id", old_id},
                         {"model_rank", rank},
                         {"reason", "outside_measured_fixed30_intersection"}});
      continue;
    }
    if (!fixed_eligible_set.contains(*method)) {
      ignored.push_back({{"model_candidate_id", old_id},
                         {"mapped_method", *method},
                         {"model_rank", rank},
                         {"reason", "mapped_method_not_eligible_for_input"}});
      continue;
    }
    mapped.push_back({{"model_candidate_id", old_id},
                      {"fixed30_method", *method},
                      {"model_rank", rank},
                      {"predicted_log_regret",
                       entry.at("predicted_log_regret")}});
    if (seeded.insert(*method).second) seed_order.push_back(*method);
  }
  const auto prediction_wall_time_ns = elapsed_ns(prediction_begin);

  if (options.budget.soft_wall_time_ns) {
    const auto original = *options.budget.soft_wall_time_ns;
    options.budget.soft_wall_time_ns =
        prediction_wall_time_ns >= original
            ? 0
            : original - prediction_wall_time_ns;
  }
  options.candidate_scope = "fixed30:model24_rank_transport";
  options.proposer = [seed_order, fixed_eligible](
                         const std::vector<TrialFeedback>& feedback) {
    return propose_fixed30(feedback, std::span<const std::string>(seed_order),
                           std::span<const std::string>(fixed_eligible));
  };

  RecommendationResult result;
  result.selection = select(registry, column, options);
  result.prediction = std::move(prediction);
  result.prediction_wall_time_ns = prediction_wall_time_ns;
  result.total_wall_time_ns = elapsed_ns(total_begin);
  result.provenance = {
      {"schema", "prepared-recommender-provenance-v1"},
      {"adapter", "operator-recommend135-model24-to-measured-fixed30-v1"},
      {"model_schema", impl_->model.at("schema")},
      {"model_candidate_catalog_sha256", kCatalogSha256},
      {"model_candidate_count", kModelCandidates.size()},
      {"model_candidate_ids", impl_->model.at("candidate_ids")},
      {"fixed30_candidate_count", fixed30_candidates().size()},
      {"objective_target", target},
      {"source_group_split_sha256", impl_->split_sha256},
      {"split_file_sha256_verified", impl_->split_file_sha_verified},
      {"confirm_labels_in_fit", false},
      {"training_dataset_count", impl_->training_datasets.size()},
      {"confirm_dataset_count", impl_->confirm_datasets.size()},
      {"model_eligible_candidates", model_eligible},
      {"fixed30_eligible_methods", fixed_eligible},
      {"mapped_rank_entries", std::move(mapped)},
      {"ignored_rank_entries", std::move(ignored)},
      {"rank_transport_only", true},
      {"model_fitted_to_fixed30", false},
      {"fit_or_update_performed", false},
      {"joint_optimum_claimed", false},
      {"mandatory_baselines_measured_first", true},
      {"prediction_wall_time_ns", prediction_wall_time_ns},
      {"total_wall_time_ns", result.total_wall_time_ns}};
  return result;
}

}  // namespace unified_compression_v1
