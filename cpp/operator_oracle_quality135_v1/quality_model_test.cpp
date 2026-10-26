#include "quality_model.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using operator_oracle_quality135_v1::CandidateCosts;
using operator_oracle_quality135_v1::Domain;
using operator_oracle_quality135_v1::FitConfig;
using operator_oracle_quality135_v1::Objective;
using operator_oracle_quality135_v1::QualityModel;
using operator_oracle_quality135_v1::TrainingRow;
using operator_oracle_quality135_v1::fit;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Function>
void require_throws(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

CandidateCosts candidate(std::string id, double space,
                         std::optional<double> encode = std::nullopt,
                         std::optional<double> decode = std::nullopt,
                         std::optional<double> balanced = std::nullopt,
                         bool eligible = true) {
  return {std::move(id), eligible, space, encode, decode, balanced};
}

TrainingRow row(std::string sample, std::string dataset, std::string group,
                double weight, std::vector<double> features,
                std::vector<CandidateCosts> candidates) {
  return {std::move(sample), std::move(dataset), std::move(group), weight,
          std::move(features), std::move(candidates)};
}

FitConfig config(std::size_t k = 2, double ridge_weight = 0.5) {
  FitConfig result;
  result.ridge_lambda = 1.0;
  result.knn_k = k;
  result.ridge_weight = ridge_weight;
  result.center_log_cost_per_row = true;
  result.feature_names = {"f0", "f1"};
  result.candidate_catalog_sha256 = std::string(64, 'a');
  return result;
}

std::vector<TrainingRow> basic_rows() {
  return {
      row("s0", "d0", "g0", 1.0, {0.0, 0.0},
          {candidate("a", 1.0, 1.0, 3.0, 2.0),
           candidate("b", 4.0, 4.0, 1.0, 2.0),
           candidate("never", 8.0, std::nullopt, std::nullopt,
                     std::nullopt, false)}),
      row("s1", "d0", "g0", 1.0, {0.1, 0.0},
          {candidate("a", 1.0, 1.0, 3.0, 2.0),
           candidate("b", 4.0, 4.0, 1.0, 2.0),
           candidate("never", 8.0, std::nullopt, std::nullopt,
                     std::nullopt, false)}),
      row("s2", "d1", "g1", 1.0, {10.0, 1.0},
          {candidate("a", 4.0, 4.0, 1.0, 2.0),
           candidate("b", 1.0, 1.0, 3.0, 2.0),
           candidate("never", 8.0, std::nullopt, std::nullopt,
                     std::nullopt, false)}),
      row("s3", "d1", "g1", 1.0, {10.1, 1.0},
          {candidate("a", 4.0, 4.0, 1.0, 2.0),
           candidate("b", 1.0, 1.0, 3.0, 2.0),
           candidate("never", 8.0, std::nullopt, std::nullopt,
                     std::nullopt, false)}),
  };
}

void test_fit_rank_reload_and_stable_order() {
  const QualityModel model =
      fit(basic_rows(), Domain::Algorithm, Objective::Space, config());
  require(model.feature_count() == 2, "feature count mismatch");
  require(model.candidate_ids() ==
              std::vector<std::string>({"a", "b", "never"}),
          "candidate IDs must be canonical and complete");

  const auto prediction = model.rank({0.0, 0.0}, {"b", "a", "new", "a",
                                                        "never"});
  require(prediction.ranking.size() == 2,
          "all trained eligible candidates must be ranked");
  require(prediction.ranking.front().candidate_id == "a",
          "nearby lower-cost candidate must rank first");
  require(prediction.unscored_candidates ==
              std::vector<std::string>({"never", "new"}),
          "unknown and unsupported candidates must be explicit and sorted");
  const auto reused = model.rank_with_scoring(
      {0.0, 0.0}, {"b", "a", "new", "a", "never"}, 2, 0.5);
  require(reused.ranking.front().candidate_id ==
              prediction.ranking.front().candidate_id,
          "explicit scoring options must reproduce stored scoring options");
  const auto ridge_model =
      fit(basic_rows(), Domain::Algorithm, Objective::Space, config(0, 1.0));
  const auto cached_grid_score = ridge_model.rank_with_scoring(
      {0.0, 0.0}, {"b", "a", "new", "a", "never"}, 2, 0.5);
  require(cached_grid_score.ranking.size() == prediction.ranking.size(),
          "cached grid scoring shape mismatch");
  for (std::size_t i = 0; i < prediction.ranking.size(); ++i) {
    require(cached_grid_score.ranking[i].candidate_id ==
                    prediction.ranking[i].candidate_id &&
                cached_grid_score.ranking[i].predicted_log_cost ==
                    prediction.ranking[i].predicted_log_cost,
            "cached grid scoring must equal a direct fit");
  }

  const auto serialized = model.to_json();
  require(serialized.at("training").at("identifiers_used_as_features") ==
              false,
          "model must record that identifiers are not features");
  require(serialized.at("training").at("row_weight_contract") ==
              "positive_development_only_input",
          "row-weight contract is missing");
  const QualityModel reloaded = QualityModel::from_json(serialized);
  const auto replay =
      reloaded.rank({0.0, 0.0}, {"b", "a", "new", "never"});
  require(replay.ranking.size() == prediction.ranking.size(),
          "reload ranking shape mismatch");
  for (std::size_t i = 0; i < replay.ranking.size(); ++i) {
    require(replay.ranking[i].candidate_id ==
                    prediction.ranking[i].candidate_id &&
                replay.ranking[i].predicted_log_cost ==
                    prediction.ranking[i].predicted_log_cost &&
                replay.ranking[i].ridge_log_cost ==
                    prediction.ranking[i].ridge_log_cost &&
                replay.ranking[i].knn_log_cost ==
                    prediction.ranking[i].knn_log_cost,
            "serialize/reload must preserve exact predictions");
  }

  auto ties = basic_rows();
  for (auto& training_row : ties) {
    training_row.candidates = {candidate("z", 1.0), candidate("a", 1.0)};
  }
  const auto tie_model =
      fit(ties, Domain::Algorithm, Objective::Space, config(0, 1.0));
  const auto tie_prediction = tie_model.rank({2.0, 0.0}, {"z", "a"});
  require(tie_prediction.ranking[0].candidate_id == "a" &&
              tie_prediction.ranking[1].candidate_id == "z",
          "exact score ties must use candidate ID order");
}

void test_dataset_equal_weights_affect_fit_and_knn() {
  std::vector<TrainingRow> rows{
      row("heavy", "d0", "g0", 9.0, {0.0, 0.0},
          {candidate("a", 1.0), candidate("b", 4.0)}),
      row("light", "d1", "g1", 1.0, {0.0, 0.0},
          {candidate("a", 4.0), candidate("b", 1.0)}),
  };
  const auto model =
      fit(rows, Domain::Algorithm, Objective::Space, config(2, 0.0));
  const auto prediction = model.rank({0.0, 0.0}, {"a", "b"});
  require(prediction.ranking.front().candidate_id == "a",
          "kNN exact-match aggregation must honor row weights");
  const auto serialized = model.to_json();
  require(serialized.at("normalization").at("mean").at(0) == 0.0,
          "weighted normalization mismatch");
  require(serialized.at("points").at(0).at("row_weight") == 9.0,
          "serialized row weight mismatch");
}

void test_missing_times_and_domain_boundaries() {
  std::vector<TrainingRow> rows{
      row("s0", "d0", "g0", 1.0, {0.0, 0.0},
          {candidate("timed", 2.0, 1.0),
           candidate("space_only", 1.0)}),
      row("s1", "d1", "g1", 1.0, {1.0, 0.0},
          {candidate("timed", 1.0, 1.0),
           candidate("space_only", 2.0)}),
  };
  const auto encode_model =
      fit(rows, Domain::Algorithm, Objective::Encode, config());
  const auto prediction =
      encode_model.rank({0.0, 0.0}, {"space_only", "timed"});
  require(prediction.ranking.size() == 1 &&
              prediction.ranking[0].candidate_id == "timed" &&
              prediction.unscored_candidates ==
                  std::vector<std::string>({"space_only"}),
          "missing times must remain unsupported, not imputed");

  require_throws(
      [&] {
        fit(rows, Domain::OperatorZstd3, Objective::Encode, config());
      },
      "operator timing objective must be rejected");
  require_throws(
      [&] {
        fit(rows, Domain::ObservedPipeline, Objective::Balanced, config());
      },
      "observed-pipeline timing objective must be rejected");

  std::vector<TrainingRow> observed{
      row("p0", "d0", "g0", 1.0, {0.0, 0.0},
          {candidate("algorithm:Zstd3", 2.0),
           candidate("operator_zstd3:A048", 1.0)}),
      row("p1", "d1", "g1", 1.0, {1.0, 0.0},
          {candidate("algorithm:Zstd3", 1.0),
           candidate("operator_zstd3:A048", 2.0)}),
  };
  const auto observed_model = fit(observed, Domain::ObservedPipeline,
                                  Objective::Space, config());
  require(observed_model.candidate_ids() ==
              std::vector<std::string>({"algorithm:Zstd3",
                                        "operator_zstd3:A048"}),
          "observed pipelines must remain explicit measured slots");
}

void test_invalid_inputs_and_serialized_support() {
  auto rows = basic_rows();
  rows[0].features[0] = std::numeric_limits<double>::quiet_NaN();
  require_throws(
      [&] { fit(rows, Domain::Algorithm, Objective::Space, config()); },
      "non-finite training features must fail");

  const auto model =
      fit(basic_rows(), Domain::Algorithm, Objective::Space, config());
  require_throws(
      [&] {
        model.rank({std::numeric_limits<double>::infinity(), 0.0}, {"a"});
      },
      "non-finite prediction features must fail");
  require_throws([&] { model.rank({0.0}, {"a"}); },
                 "wrong prediction shape must fail");
  require_throws([&] { model.rank({0.0, 0.0}, {""}); },
                 "empty eligible candidate ID must fail");
  require_throws(
      [&] { model.rank_with_scoring({0.0, 0.0}, {"a"}, 0, 0.5); },
      "invalid scoring override must fail");

  auto corrupted = model.to_json();
  corrupted["candidates"][0]["support_rows"] = 999;
  require_throws([&] { QualityModel::from_json(corrupted); },
                 "reload must verify support counts");

  auto bad_weight = basic_rows();
  bad_weight[0].weight = 0.0;
  require_throws(
      [&] {
        fit(bad_weight, Domain::Algorithm, Objective::Space, config());
      },
      "non-positive dataset weight must fail");
}

}  // namespace

int main() {
  try {
    test_fit_rank_reload_and_stable_order();
    test_dataset_equal_weights_affect_fit_and_knn();
    test_missing_times_and_domain_boundaries();
    test_invalid_inputs_and_serialized_support();
    std::cout << "quality_model_test: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "quality_model_test: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
