#include "model.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using operator_recommend135_v1::feature_names;
using operator_recommend135_v1::fit_model;
using operator_recommend135_v1::predict;

constexpr std::array<const char*, 4> kTargets = {"space", "encode",
                                                 "decode", "balanced"};
constexpr const char* kCatalogSha =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> features(double first, double second) {
  std::vector<double> result(feature_names().size(), 0.0);
  result.at(0) = first;
  result.at(1) = second;
  return result;
}

Json metric(double frame, double encode, double decode,
            const std::string& kind) {
  return {{"status", "strict_verified"},
          {"frame_bytes", frame},
          {"encode_cpu_ns", encode},
          {"decode_cpu_ns", decode},
          {"encode_wall_ns", encode + 1.0},
          {"decode_wall_ns", decode + 1.0},
          {"kind", kind}};
}

Json row(const std::string& sample, const std::string& group, double first,
         double second, double shift) {
  Json metrics = {
      {"raw", metric(80 + shift, 10 + shift, 10 + shift, "raw")},
      {"algorithm_a",
       metric(100 + shift, 80 - shift, 50 + shift, "algorithm")},
      {"operator_a",
       metric(90 + shift, 60 + shift, 70 - shift, "operator")},
      {"tree_a",
       metric(70 + shift, 90 - shift, 60 + shift, "operator_tree")},
      {"combination_a",
       metric(60 + shift, 70 + shift, 90 - shift,
              "algorithm_combination")}};
  return {{"sample_id", sample},
          {"dataset_id", "dataset-" + sample},
          {"group_id", group},
          {"scale", "u64"},
          {"dtype", "float64"},
          {"features", features(first, second)},
          {"input_bytes", 512},
          {"feature_cpu_ns", 10},
          {"feature_wall_ns", 11},
          {"eligibility_cpu_ns", 12},
          {"eligibility_wall_ns", 13},
          {"eligibility_provenance",
           "frozen runtime dtype/shape rule on head, before any tail encode"},
          {"catalog_sha256", kCatalogSha},
          {"eligible_candidates",
           Json::array({"raw", "algorithm_a", "operator_a", "tree_a",
                        "combination_a"})},
          {"metrics", std::move(metrics)}};
}

Json filtered_ranking(const Json& ranking) {
  Json result = Json::array();
  for (const auto& item : ranking) {
    if (item.at("candidate_id") != "raw") result.push_back(item);
  }
  return result;
}

void test_remove_raw_is_exact_rank_filter_with_sparse_labels() {
  Json rows = Json::array(
      {row("s0", "g0", 0.0, 0.0, 0.0),
       row("s1", "g1", 0.2, 0.1, 2.0),
       row("s2", "g2", 0.5, 0.4, 4.0),
       row("s3", "g3", 0.8, 0.7, 6.0),
       row("s4", "g4", 1.0, 1.0, 8.0),
       row("s5", "g5", 1.2, 1.1, 10.0)});
  const std::vector<std::string> all_candidates = {
      "raw", "algorithm_a", "operator_a", "tree_a", "combination_a"};
  const Json options{{"candidate_ids", all_candidates},
                     {"catalog_sha256", kCatalogSha}};
  Json model = fit_model(rows, options);

  // Sparse labels force candidate/target-specific neighbor subsets. Removing
  // raw must still be only a ranking filter, with no changed neighbor scores.
  model["points"][0]["labels"]["operator_a"].erase("decode");
  model["points"][1]["labels"]["tree_a"].erase("space");
  model["points"][2]["labels"]["combination_a"].erase("balanced");
  model["points"][3]["labels"]["algorithm_a"].erase("encode");
  model["points"][4]["labels"].erase("operator_a");

  const auto query = features(0.61, 0.55);
  const Json before = model;
  const Json full = predict(model, query, all_candidates);
  const std::vector<std::string> compression_candidates = {
      "algorithm_a", "operator_a", "tree_a", "combination_a"};

  for (std::size_t repeat = 0; repeat < 5; ++repeat) {
    const Json compression = predict(model, query, compression_candidates);
    for (const char* target : kTargets) {
      const Json expected =
          filtered_ranking(full.at("targets").at(target).at("ranking"));
      const Json& actual =
          compression.at("targets").at(target).at("ranking");
      require(actual.size() == compression_candidates.size(),
              std::string(target) + " lost a scored compression candidate");
      require(actual == expected,
              std::string(target) +
                  " candidate order or predicted score changed after raw "
                  "eligibility was removed");
      for (std::size_t index = 0; index < actual.size(); ++index) {
        require(actual.at(index).at("candidate_id") ==
                    expected.at(index).at("candidate_id") &&
                    actual.at(index).at("predicted_log_regret") ==
                    expected.at(index).at("predicted_log_regret") &&
                    actual.at(index).at("neighbor_count") ==
                    expected.at(index).at("neighbor_count") &&
                    actual.at(index).at("exact_match_count") ==
                    expected.at(index).at("exact_match_count"),
                std::string(target) +
                    " per-candidate prediction fields differ after filtering");
      }
    }
  }
  require(model == before,
          "compression-only prediction must not refit or mutate the model");
}

}  // namespace

int main() {
  try {
    test_remove_raw_is_exact_rank_filter_with_sparse_labels();
    std::cout << "compression_serve_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "compression_serve_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
