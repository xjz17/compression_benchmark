#include "model.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using operator_recommend135_v1::evaluate_leave_group_out;
using operator_recommend135_v1::extract_features;
using operator_recommend135_v1::feature_names;
using operator_recommend135_v1::fit_model;
using operator_recommend135_v1::PreparedModel;
using operator_recommend135_v1::predict;

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

std::size_t feature_index(const std::string& name) {
  const auto names = feature_names();
  const auto it = std::find(names.begin(), names.end(), name);
  require(it != names.end(), "missing feature name: " + name);
  return static_cast<std::size_t>(it - names.begin());
}

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t bits) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(bits >> shift));
  }
}

Json metric(double frame, double encode, double decode,
            std::string status = "strict_verified",
            std::string kind = "recipe") {
  return Json{{"status", std::move(status)},
              {"frame_bytes", frame},
              {"encode_cpu_ns", encode},
              {"decode_cpu_ns", decode},
              {"encode_wall_ns", encode + 1.0},
              {"decode_wall_ns", decode + 1.0},
              {"kind", std::move(kind)}};
}

std::vector<double> fv(double first, double second) {
  std::vector<double> result(feature_names().size(), 0.0);
  result.at(0) = first;
  result.at(1) = second;
  return result;
}

Json row(std::string sample, std::string dataset, std::string group,
         std::string scale, std::vector<double> features, Json metrics,
         Json eligible = Json()) {
  if (eligible.is_null()) {
    eligible = Json::array();
    for (const auto& candidate : metrics.items()) {
      eligible.push_back(candidate.key());
    }
  }
  Json result{{"sample_id", std::move(sample)},
              {"dataset_id", std::move(dataset)},
              {"group_id", std::move(group)},
              {"scale", std::move(scale)},
              {"dtype", "f32"},
              {"features", std::move(features)},
              {"input_bytes", 100.0},
              {"feature_cpu_ns", 17},
              {"feature_wall_ns", 23},
              {"eligibility_cpu_ns", 5},
              {"eligibility_wall_ns", 7},
              {"eligibility_provenance",
               "frozen runtime dtype/shape rule on head, before any tail encode"},
              {"catalog_sha256", std::string(64, 'a')},
              {"metrics", std::move(metrics)}};
  result["eligible_candidates"] = std::move(eligible);
  return result;
}

Json basic_rows() {
  Json rows = Json::array();
  rows.push_back(row("a-small", "dataset-a", "source-a", "small", fv(0, 0),
                     {{"a", metric(100, 400, 400, "strict_verified", "algorithm")},
                      {"b", metric(200, 200, 200)},
                      {"c", metric(300, 100, 300)},
                      {"d", metric(400, 400, 400)}}));
  rows.push_back(row("a-large", "dataset-a", "source-a", "large", fv(0.1, 0),
                     {{"a", metric(101, 390, 410, "strict_verified", "algorithm")},
                      {"b", metric(201, 210, 190)},
                      {"c", metric(301, 110, 290)},
                      {"d", metric(401, 410, 390)}}));
  rows.push_back(row("b-small", "dataset-b", "source-b", "small", fv(1, 1),
                     {{"a", metric(180, 300, 300, "strict_verified", "algorithm")},
                      {"b", metric(100, 100, 100)},
                      {"c", metric(300, 200, 300)},
                      {"d", metric(500, 500, 500)}}));
  rows.push_back(row("b-large", "dataset-b", "source-b", "large", fv(1.1, 1),
                     {{"a", metric(181, 310, 290, "strict_verified", "algorithm")},
                      {"b", metric(101, 110, 90)},
                      {"c", metric(301, 210, 290)},
                      {"d", metric(501, 510, 490)}}));
  rows.push_back(row("c-small", "dataset-c", "source-c", "small", fv(2, 2),
                     {{"a", metric(100, 300, 300, "strict_verified", "algorithm")},
                      {"b", metric(200, 200, 200)},
                      {"c", metric(300, 100, 300)},
                      {"d", metric(400, 400, 400)}},
                     Json::array({"a", "b", "c", "d"})));
  rows.push_back(row("c-large", "dataset-c", "source-c", "large", fv(2.1, 2),
                     {{"a", metric(100, 300, 300, "strict_verified", "algorithm")},
                      {"b", metric(100, 300, 300)},
                      {"c", metric(300, 100, 300)},
                      {"d", metric(400, 400, 400)}},
                     Json::array({"a", "b", "c", "d"})));
  return rows;
}

Json options() {
  return Json{{"candidate_ids", Json::array({"a", "b", "c", "d"})},
              {"catalog_sha256", std::string(64, 'a')},
              {"fixed_algorithm_candidates", Json::array({"a"})}};
}

Json legacy_predict_targets(const Json& model,
                            const std::vector<double>& features,
                            const std::vector<std::string>& eligible_input) {
  constexpr std::size_t k = 5;
  constexpr double distance_floor = 1e-12;
  static constexpr const char* targets[] = {"space", "encode", "decode",
                                             "balanced"};
  const auto means =
      model.at("normalization").at("mean").get<std::vector<double>>();
  const auto scales =
      model.at("normalization").at("scale").get<std::vector<double>>();
  std::vector<double> normalized(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) {
    normalized[i] = (features[i] - means[i]) / scales[i];
  }
  const auto model_candidates =
      model.at("candidate_ids").get<std::vector<std::string>>();
  const std::set<std::string> model_candidate_set(model_candidates.begin(),
                                                   model_candidates.end());
  std::vector<std::string> eligible;
  std::set<std::string> seen;
  for (const auto& candidate : eligible_input) {
    if (seen.insert(candidate).second && model_candidate_set.contains(candidate)) {
      eligible.push_back(candidate);
    }
  }

  struct Neighbor {
    double distance;
    double label;
    std::string sample_id;
  };
  Json result = Json::object();
  for (const auto* target : targets) {
    Json ranking = Json::array();
    for (const auto& candidate : eligible) {
      std::vector<Neighbor> neighbors;
      for (const auto& point : model.at("points")) {
        if (!point.at("labels").contains(candidate) ||
            !point.at("labels").at(candidate).contains(target)) {
          continue;
        }
        const auto point_features =
            point.at("normalized_features").get<std::vector<double>>();
        double squared = 0;
        for (std::size_t i = 0; i < normalized.size(); ++i) {
          const double difference = normalized[i] - point_features[i];
          squared += difference * difference;
        }
        neighbors.push_back(
            {std::sqrt(squared),
             point.at("labels").at(candidate).at(target).get<double>(),
             point.at("sample_id").get<std::string>()});
      }
      std::sort(neighbors.begin(), neighbors.end(),
                [](const Neighbor& left, const Neighbor& right) {
                  if (left.distance != right.distance) {
                    return left.distance < right.distance;
                  }
                  return left.sample_id < right.sample_id;
                });
      if (neighbors.empty()) {
        continue;
      }
      const auto count = std::min(k, neighbors.size());
      double weighted_sum = 0;
      double weight_sum = 0;
      std::size_t exact_count = 0;
      for (std::size_t i = 0; i < count; ++i) {
        if (neighbors[i].distance <= distance_floor) {
          weighted_sum += neighbors[i].label;
          ++exact_count;
        }
      }
      if (exact_count != 0) {
        weighted_sum /= static_cast<double>(exact_count);
      } else {
        weighted_sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
          const double weight = 1.0 / neighbors[i].distance;
          weighted_sum += weight * neighbors[i].label;
          weight_sum += weight;
        }
        weighted_sum /= weight_sum;
      }
      ranking.push_back({{"candidate_id", candidate},
                         {"predicted_log_regret", weighted_sum},
                         {"neighbor_count", count},
                         {"exact_match_count", exact_count}});
    }
    std::sort(ranking.begin(), ranking.end(), [](const Json& left,
                                                  const Json& right) {
      const auto left_score = left.at("predicted_log_regret").get<double>();
      const auto right_score = right.at("predicted_log_regret").get<double>();
      return left_score != right_score
                 ? left_score < right_score
                 : left.at("candidate_id").get<std::string>() <
                       right.at("candidate_id").get<std::string>();
    });
    auto top = [&](std::size_t count) {
      Json ids = Json::array();
      for (std::size_t i = 0; i < std::min(count, ranking.size()); ++i) {
        ids.push_back(ranking.at(i).at("candidate_id"));
      }
      return ids;
    };
    result[target] = {{"ranking", ranking},
                      {"top1", top(1)},
                      {"top3", top(3)},
                      {"top5", top(5)}};
  }
  return result;
}

void test_raw_bit_features_and_shape_checks() {
  std::vector<std::uint8_t> values;
  append_u32(values, 0x00000000U);
  append_u32(values, 0x80000000U);
  append_u32(values, 0x7fc01234U);
  append_u32(values, 0x7f800000U);
  const std::vector<std::uint8_t> missing{0b00000100};

  const auto features = extract_features(values, "f32", missing, 5);
  require(features.size() == feature_names().size(),
          "feature vector must match feature_names");
  for (double value : features) {
    require(std::isfinite(value), "feature extraction emitted non-finite value");
  }
  require(std::abs(features[feature_index("finite_fraction")] - 0.5) < 1e-12,
          "finite fraction must be computed from original f32 exponent bits");
  require(std::abs(features[feature_index("nonfinite_fraction")] - 0.5) < 1e-12,
          "non-finite fraction must preserve NaN/Inf bit classes");
  require(std::abs(features[feature_index("distinct_fraction")] - 1.0) < 1e-12,
          "+0 and -0 must remain distinct raw bit patterns");
  require(std::abs(features[feature_index("missing_fraction")] - 0.2) < 1e-12,
          "missing fraction mismatch");

  require_throws(
      [&] { extract_features(std::span<const std::uint8_t>(values).first(15),
                             "f32", missing, 5); },
      "wrong byte shape must fail");
  require_throws(
      [&] { extract_features(values, "f32", {}, 5); },
      "row/value mismatch without a missing bitmap must fail");
  require_throws(
      [&] { extract_features(values, "f128", missing, 5); },
      "unsupported dtype must fail");

  std::vector<std::uint8_t> prefix_a;
  std::vector<std::uint8_t> prefix_b;
  prefix_a.reserve(2000 * 4);
  prefix_b.reserve(2000 * 4);
  for (std::uint32_t i = 0; i < 2000; ++i) {
    const std::uint32_t shared = i < 1024 ? (i * 2654435761U) : 0U;
    append_u32(prefix_a, shared);
    append_u32(prefix_b, i < 1024 ? shared : 0xffffffffU - i);
  }
  const auto capped_a = extract_features(prefix_a, "uint32", {}, 2000);
  const auto capped_b = extract_features(prefix_b, "uint32", {}, 2000);
  require(capped_a == capped_b,
          "features must ignore values beyond the fixed 1024-word prefix");
  require(capped_a[feature_index("scanned_word_count")] == 1024.0,
          "actual scanned word count must be recorded");
  require(capped_a[feature_index("scanned_value_bytes")] == 4096.0,
          "actual scanned byte count must be recorded");
}

void test_fit_predict_round_trip_and_validation() {
  Json training = Json::array({
      row("p0", "d0", "g0", "small", fv(0, 0),
          {{"a", metric(100, 300, 300, "strict_verified", "algorithm")},
           {"b", metric(200, 100, 100)}}),
      row("p1", "d1", "g1", "small", fv(10, 10),
          {{"a", metric(200, 100, 100, "strict_verified", "algorithm")},
           {"b", metric(100, 300, 300)}}),
  });
  const Json model = fit_model(training, options());
  require(model.at("schema") == "operator-recommend135-model-v1",
          "model schema mismatch");
  require(model.at("algorithm").at("k") == 5, "KNN k must be fixed at five");
  require(model.at("feature_extraction").at("max_words") == 1024 &&
              model.at("feature_extraction").at("max_value_bytes") == 8192,
          "serialized model must record the bounded feature scan contract");
  require(model.at("candidate_catalog_sha256") == std::string(64, 'a'),
          "serialized model must bind the frozen candidate catalog");
  require(model.at("normalization").at("mean").at(0) == 5.0 &&
              model.at("normalization").at("mean").at(1) == 5.0,
          "fit normalization mismatch");
  require(model.at("training").at("group_ids") ==
              Json::array({"g0", "g1"}),
          "training group provenance missing");

  const Json prediction = predict(model, fv(0, 0), {"a", "b"});
  require(prediction.at("targets").at("space").at("top1").at(0) == "a",
          "serialized model must independently rank a new feature vector");
  require(prediction.at("targets").at("space").at("top5").size() == 2,
          "top-k output must gracefully handle fewer than five candidates");
  require(prediction.at("online_cost").at("model_cpu_ns").get<double>() >= 0,
          "prediction must return online CPU cost");

  Json bad_training = training;
  bad_training[0]["features"][0] =
      std::numeric_limits<double>::infinity();
  require_throws([&] { fit_model(bad_training, options()); },
                 "fit must reject non-finite input features");
  require_throws(
      [&] { predict(model, {0}, {"a"}); },
      "predict must reject a feature vector with the wrong shape");
  require_throws(
      [&] {
        auto features = fv(0, 0);
        features[0] = std::numeric_limits<double>::quiet_NaN();
        predict(model, features, {"a"});
      },
      "predict must reject non-finite input features");
  Json bad_options = options();
  bad_options["k"] = 3;
  require_throws([&] { fit_model(training, bad_options); },
                 "the fixed k=5 contract must reject tuning");
}

void test_single_distance_sort_matches_legacy_predictor() {
  Json model = fit_model(basic_rows(), options());
  // Exercise sparse per-target labels: the legacy implementation built and
  // sorted a separate neighbor vector for each candidate and target.
  model["points"][0]["labels"]["b"].erase("decode");
  model["points"][1]["labels"].erase("c");
  model["points"][2]["labels"]["a"].erase("space");
  model["points"][3]["labels"]["d"].erase("balanced");
  const auto features = fv(0.55, 0.5);
  const std::vector<std::string> eligible{"d", "b", "a", "c", "b"};
  const auto expected = legacy_predict_targets(model, features, eligible);
  const auto actual = predict(model, features, eligible);
  require(actual.at("targets") == expected,
          "one global distance sort must preserve legacy rankings and scores");
}

void test_prepared_model_matches_reference_with_sparse_labels_and_ties() {
  Json model = fit_model(basic_rows(), options());
  model["points"][0]["labels"]["b"].erase("decode");
  model["points"][1]["labels"].erase("c");
  model["points"][2]["labels"]["a"].erase("space");
  model["points"][3]["labels"]["d"].erase("balanced");

  const auto features = fv(0.55, 0.5);
  const std::vector<std::string> eligible{"unknown", "d", "b", "a", "c",
                                          "b"};
  const Json reference = predict(model, features, eligible);
  const Json prepared = PreparedModel(model).predict(features, eligible);
  require(prepared.at("targets") == reference.at("targets"),
          "prepared model must preserve sparse-label rankings and scores");
  require(prepared.at("serving") == reference.at("serving"),
          "prepared model must preserve serving selections");
  require(prepared.at("eligible_candidates") ==
              reference.at("eligible_candidates") &&
              prepared.at("unscored_candidates") ==
                  reference.at("unscored_candidates"),
          "prepared model must preserve eligible and unscored reporting");

  const std::array<const char*, 6> sample_ids = {"z", "a", "e", "c", "b", "d"};
  const std::array<double, 6> labels = {100.0, 1.0, 5.0, 3.0, 2.0, 4.0};
  for (std::size_t i = 0; i < sample_ids.size(); ++i) {
    model["points"][i]["sample_id"] = sample_ids[i];
    model["points"][i]["normalized_features"] =
        std::vector<double>(feature_names().size(), 0.0);
    for (const char* target : {"space", "encode", "decode", "balanced"}) {
      model["points"][i]["labels"]["a"][target] = labels[i];
    }
  }
  const auto means =
      model.at("normalization").at("mean").get<std::vector<double>>();
  const Json tied_reference = predict(model, means, {"a"});
  const Json tied_prepared = PreparedModel(model).predict(means, {"a"});
  require(tied_prepared.at("targets") == tied_reference.at("targets"),
          "prepared model must preserve sample-id ordering for tied distances");
  require(tied_prepared.at("targets")
              .at("space")
              .at("ranking")
              .at(0)
              .at("predicted_log_regret") == 3.0,
          "tied exact matches must use the first five sample IDs");
}

void test_prepared_model_eagerly_rejects_malformed_points() {
  const Json valid = fit_model(basic_rows(), options());

  Json nonfinite_label = valid;
  nonfinite_label["points"][0]["labels"]["a"]["space"] =
      std::numeric_limits<double>::infinity();
  require_throws([&] { PreparedModel prepared(nonfinite_label); },
                 "prepared construction must validate every stored label");

  Json wrong_point_shape = valid;
  wrong_point_shape["points"][0]["normalized_features"].erase(0);
  require_throws([&] { PreparedModel prepared(wrong_point_shape); },
                 "prepared construction must validate every point shape");

  Json bad_scale = valid;
  bad_scale["normalization"]["scale"][0] = 0.0;
  require_throws([&] { PreparedModel prepared(bad_scale); },
                 "prepared construction must reject a non-positive scale");
}

void test_leave_group_out_no_leakage_and_group_integrity() {
  const Json rows = basic_rows();
  const Json evaluation = evaluate_leave_group_out(rows, options());
  require(evaluation.at("schema") == "operator-recommend135-evaluation-v1",
          "evaluation schema mismatch");
  require(evaluation.at("folds").size() == 3,
          "leave-group-out must produce one fold per source group");

  const auto& c_fold = evaluation.at("folds").at(2);
  require(c_fold.at("held_out_group_id") == "source-c",
          "fold order must be deterministic");
  require(c_fold.at("training_group_ids") ==
              Json::array({"source-a", "source-b"}),
          "held-out group leaked into training provenance");
  require(c_fold.at("predictions").size() == 2,
          "all scales from one dataset must be held out together");
  require(std::abs(c_fold.at("normalization").at("mean").at(0).get<double>() -
                       0.55) < 1e-12 &&
              std::abs(c_fold.at("normalization").at("mean").at(1).get<double>() -
                       0.5) < 1e-12,
          "fold standardization must use training rows only");
  require(c_fold.at("offline_fit_cpu_ns").get<double>() >= 0,
          "offline training time must be reported separately");

  Json perturbed = rows;
  for (auto& candidate : perturbed[4]["metrics"].items()) {
    candidate.value()["status"] = "domain_rejected";
    candidate.value()["frame_bytes"] = 999999999.0;
  }
  for (auto& candidate : perturbed[5]["metrics"].items()) {
    candidate.value()["status"] = "failed";
    candidate.value()["frame_bytes"] = 0.000001;
  }
  const Json changed = evaluate_leave_group_out(perturbed, options());
  for (std::size_t i = 0; i < 2; ++i) {
    require(c_fold.at("predictions").at(i).at("prediction").at("targets") ==
                changed.at("folds").at(2).at("predictions").at(i).at(
                    "prediction").at("targets"),
            "held-out truth/status changed a fold prediction");
    require(c_fold.at("baselines") == changed.at("folds").at(2).at("baselines"),
            "held-out truth changed a training-selected baseline");
  }

  Json missing_eligibility = rows;
  missing_eligibility[4].erase("eligible_candidates");
  require_throws(
      [&] { evaluate_leave_group_out(missing_eligibility, options()); },
      "evaluation must require eligibility fixed before tail measurement");

  Json missing_feature_time = rows;
  missing_feature_time[4].erase("feature_cpu_ns");
  require_throws(
      [&] { evaluate_leave_group_out(missing_feature_time, options()); },
      "missing feature timing must not silently become zero");

  Json split = rows;
  split[1]["group_id"] = "source-b";
  require_throws([&] { evaluate_leave_group_out(split, options()); },
                 "one dataset split across source groups must fail");
}

void test_ties_pareto_and_strict_only_scoring() {
  const Json evaluation = evaluate_leave_group_out(basic_rows(), options());
  const auto& scored = evaluation.at("folds").at(2).at("predictions").at(1);
  const auto& space = scored.at("scores").at("space");
  require(space.at("oracle").at("candidates") == Json::array({"a", "b"}),
          "equal strict costs must both be oracle ties");
  require(space.at("topk").at("1").at("tie_aware_hit").is_boolean(),
          "tie-aware TopK result missing");
  require(scored.at("pareto").at("candidates").size() == 3,
          "true Pareto frontier mismatch");
  require(scored.at("pareto").at("dominated_candidates") == Json::array({"d"}),
          "dominated candidate accounting mismatch");
  require(space.at("topk").at("3").contains("pareto_recall"),
          "TopK Pareto recall missing");
  require(space.contains("wall_oracle") &&
              space.at("topk").at("3").contains(
                  "wall_oracle_best_within_topk_multiplicative_regret"),
          "steady-wall scoring must be preserved separately");
  require(space.at("topk").at("1").contains(
              "online_selected_multiplicative_regret") &&
              !space.at("topk").at("3").contains(
                  "online_selected_multiplicative_regret"),
          "only Top1 may be presented as the online selected outcome");
  require(scored.at("prediction").at("online_cost").at("probe_used") ==
              false &&
              scored.at("prediction")
                      .at("online_cost")
                      .at("probe_cost_cpu_ns") == 0,
          "online serving must explicitly use no candidate probe");
  require(evaluation.at("summary")
                  .at("space")
                  .at("coverage")
                  .contains("unscored_rows"),
          "evaluation summary must expose scored/unscored coverage");

  Json rows = basic_rows();
  rows[5]["metrics"]["a"] = metric(1, 1, 1, "wrapper_verified", "algorithm");
  const Json wrapper_eval = evaluate_leave_group_out(rows, options());
  const auto& wrapper_space = wrapper_eval.at("folds").at(2).at("predictions").at(1)
                                  .at("scores").at("space");
  require(wrapper_space.at("oracle").at("candidates") == Json::array({"b"}),
          "wrapper-verified metrics must not enter strict primary scoring");

  const auto& cost = scored.at("decision_cost");
  require(cost.at("online_decision_cpu_ns").get<double>() >= 22 &&
              cost.at("eligibility_provenance") ==
                  "frozen runtime dtype/shape rule on head, before any tail encode",
          "eligibility, feature and model CPU costs must compose online");
  require(space.contains("selected_total_encode_cpu_ns"),
          "report must compose decision and selected encode cost");

  Json combination_rows = basic_rows();
  for (auto& candidate_row : combination_rows) {
    candidate_row["metrics"]["d"] =
        metric(1, 1, 1, "strict_verified", "algorithm_combination");
  }
  Json inferred_options = options();
  inferred_options.erase("fixed_algorithm_candidates");
  const Json inferred =
      evaluate_leave_group_out(combination_rows, inferred_options);
  require(inferred.at("folds")
                  .at(2)
                  .at("baselines")
                  .at("space")
                  .at("fixed_algorithm")
                  .at("candidate_id") == "a",
          "algorithm baseline must require exact kind=algorithm");
}

}  // namespace

int main() {
  try {
    test_raw_bit_features_and_shape_checks();
    test_fit_predict_round_trip_and_validation();
    test_single_distance_sort_matches_legacy_predictor();
    test_prepared_model_matches_reference_with_sparse_labels_and_ties();
    test_prepared_model_eagerly_rejects_malformed_points();
    test_leave_group_out_no_leakage_and_group_integrity();
    test_ties_pareto_and_strict_only_scoring();
    std::cout << "model_test: all checks passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "model_test: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
