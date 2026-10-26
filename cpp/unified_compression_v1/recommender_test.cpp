#include "recommender.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace uc = unified_compression_v1;
using Json = nlohmann::json;

namespace {

constexpr std::string_view kSplitSha256 =
    "d0545713f3d56aed5d9934814ca4144ab9cc8e968af032a3cffb4741cb4676a4";

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

Json read_json(const char* path) {
  std::ifstream input(path);
  require(static_cast<bool>(input), std::string("cannot read JSON fixture: ") + path);
  Json result;
  input >> result;
  return result;
}

template <class F>
void rejected(F&& operation) {
  try {
    operation();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("invalid frozen recommender fixture unexpectedly succeeded");
}

uc::Column source_column() {
  const std::vector<std::int64_t> values{
      -100, -7, 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
  return uc::make_column(std::span<const std::int64_t>(values));
}

std::set<std::string> string_set(const Json& array) {
  std::set<std::string> result;
  for (const auto& value : array) result.insert(value.get<std::string>());
  return result;
}

void test_frozen_provenance(const Json& model, const Json& split) {
  require(model.at("formal_protocol").at("confirm_labels_in_fit") == false,
          "fixture is not the development-only frozen model");
  const auto training = string_set(model.at("training").at("dataset_ids"));
  const auto formal_training =
      string_set(model.at("formal_protocol").at("training_dataset_ids"));
  const auto confirm = string_set(split.at("confirm_dataset_ids"));
  require(training == formal_training && !training.empty() && !confirm.empty(),
          "frozen model training declarations disagree");
  for (const auto& dataset : training)
    require(!confirm.contains(dataset),
            "confirm dataset appears in the frozen training fixture");
}

void test_constructor_rejections(const Json& model, const Json& split) {
  auto invalid = model;
  invalid["formal_protocol"]["confirm_labels_in_fit"] = true;
  rejected([&] { uc::PreparedRecommender recommender(invalid, split,
                                                      std::string(kSplitSha256)); });

  const auto confirm_dataset = split.at("confirm_dataset_ids").at(0).get<std::string>();
  invalid = model;
  invalid["formal_protocol"]["confirm_labels_in_fit"] = false;
  invalid["training"]["dataset_ids"].push_back(confirm_dataset);
  invalid["formal_protocol"]["training_dataset_ids"].push_back(confirm_dataset);
  rejected([&] { uc::PreparedRecommender recommender(invalid, split,
                                                      std::string(kSplitSha256)); });

  invalid = model;
  invalid["formal_protocol"]["confirm_labels_in_fit"] = false;
  invalid["training"]["dataset_ids"].push_back("not-in-frozen-split");
  invalid["formal_protocol"]["training_dataset_ids"].push_back(
      "not-in-frozen-split");
  rejected([&] { uc::PreparedRecommender recommender(invalid, split,
                                                      std::string(kSplitSha256)); });

  rejected([&] {
    uc::PreparedRecommender recommender(model, split, std::string(64, '0'));
  });
}

void test_measured_rank_transport(const Json& model, const Json& split) {
  uc::Registry registry;
  uc::PreparedRecommender recommender(model, split, std::string(kSplitSha256));
  uc::SelectionOptions options;
  options.objective = uc::SelectionObjective::compressed_bytes;
  options.budget.max_trials = 2;
  options.measurement = {0, 1};
  const auto result = recommender.recommend(registry, source_column(), options);

  require(result.selection.candidate_scope ==
              "fixed30:model24_rank_transport" &&
              result.selection.feedback.size() == 2,
          "prepared recommendation did not use the bounded fixed30 feedback loop");
  require(result.selection.feedback[0].candidate.id == "combo_v1:zstd3" &&
              result.selection.feedback[1].candidate.id == "combo_v1:lz4",
          "mandatory measured baselines were not proposed first");
  require(std::all_of(result.selection.feedback.begin(),
                      result.selection.feedback.end(), [](const auto& feedback) {
                        return feedback.status == "ok" && feedback.strict &&
                               feedback.bit_exact && feedback.feasible;
                      }),
          "prepared recommendation accepted a non-strict baseline trial");
  require(result.selection.selected_candidate && result.selection.selected_result &&
              registry.decompress(result.selection.selected_result->compressed_data) ==
                  source_column(),
          "prepared recommendation did not return a reusable measured winner");
  require(!result.selection.costs.final_recompression_performed,
          "prepared recommendation recompressed its winner");

  const auto& provenance = result.provenance;
  require(provenance.value("confirm_labels_in_fit", true) == false &&
              provenance.value("rank_transport_only", false) &&
              provenance.value("model_fitted_to_fixed30", true) == false &&
              provenance.value("fit_or_update_performed", true) == false &&
              provenance.value("joint_optimum_claimed", true) == false &&
              provenance.value("split_file_sha256_verified", false),
          "prepared recommender provenance overstates training or joint scope");
  require(provenance.at("model_candidate_count") == 24 &&
              provenance.at("fixed30_candidate_count") == 30 &&
              !provenance.at("mapped_rank_entries").empty() &&
              !provenance.at("ignored_rank_entries").empty(),
          "old24-to-fixed30 rank transport was not disclosed");
  require(result.prediction.contains("targets") &&
              result.prediction.at("targets").contains("space"),
          "prepared model prediction is missing the requested target");
  require(result.total_wall_time_ns >= result.prediction_wall_time_ns,
          "recommendation total time excludes prediction");
}

void test_prediction_charged_to_soft_budget(const Json& model,
                                            const Json& split) {
  uc::Registry registry;
  uc::PreparedRecommender recommender(model, split, std::string(kSplitSha256));
  uc::SelectionOptions options;
  options.budget.max_trials = 8;
  options.budget.soft_wall_time_ns = 0;
  const auto result = recommender.recommend(registry, source_column(), options);
  require(result.selection.feedback.empty() &&
              !result.selection.selected_candidate &&
              !result.selection.selected_result,
          "expired end-to-end soft budget admitted a harness trial");
  require(result.selection.stop_reason == "soft_wall_budget_reached",
          "expired prediction-inclusive budget has the wrong stop reason");
  require(result.total_wall_time_ns >= result.prediction_wall_time_ns &&
              result.provenance.at("prediction_wall_time_ns") ==
                  result.prediction_wall_time_ns,
          "prediction cost is absent from recommendation accounting");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: unified_recommender_test MODEL_confirm.json split_config.json");
    const auto model = read_json(argv[1]);
    const auto split = read_json(argv[2]);
    test_frozen_provenance(model, split);
    test_constructor_rejections(model, split);
    test_measured_rank_transport(model, split);
    test_prediction_charged_to_soft_budget(model, split);
    std::cout << "prepared recommender: provenance/rank-transport/budget checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
