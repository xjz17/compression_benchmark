#include "campaign.hpp"
#include "common.hpp"
#include "model.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace operator_recommend135_v1 {
namespace {

namespace d = detail;
using Json = nlohmann::json;

constexpr std::array<const char*, 4> kTargets = {"space", "encode",
                                                 "decode", "balanced"};
constexpr std::array<const char*, 4> kCompressionKinds = {
    "operator", "algorithm", "operator_tree", "algorithm_combination"};

std::vector<std::string> catalog_candidate_ids(const Json& catalog) {
  d::require(catalog.contains("candidates") &&
                 catalog.at("candidates").is_array(),
             "candidate catalog is missing candidates");
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto& candidate : catalog.at("candidates")) {
    const auto id = candidate.at("candidate_id").get<std::string>();
    d::require(!id.empty() && seen.insert(id).second,
               "candidate catalog contains an empty or duplicate ID");
    result.push_back(id);
  }
  return result;
}

std::map<std::string, std::string> catalog_candidate_kinds(
    const Json& catalog) {
  std::map<std::string, std::string> result;
  for (const auto& candidate : catalog.at("candidates")) {
    result.emplace(candidate.at("candidate_id").get<std::string>(),
                   candidate.at("kind").get<std::string>());
  }
  return result;
}

void require_model_contract(const Json& model, const Json& catalog) {
  const auto catalog_sha256 = d::sha(catalog.dump());
  d::require(model.at("candidate_catalog_sha256").is_string() &&
                 model.at("candidate_catalog_sha256").get<std::string>() ==
                     catalog_sha256,
             "model/runtime catalog SHA mismatch");
  d::require(model.at("feature_names") == Json(feature_names()),
             "model/runtime feature_names mismatch");
}

std::vector<std::string> without_raw(
    const std::vector<std::string>& eligible_candidates) {
  std::vector<std::string> result;
  result.reserve(eligible_candidates.size());
  std::copy_if(eligible_candidates.begin(), eligible_candidates.end(),
               std::back_inserter(result),
               [](const std::string& id) { return id != "raw"; });
  return result;
}

Json ranking_without_raw(const Json& ranking) {
  d::require(ranking.is_array(), "frozen ranking must be an array");
  Json result = Json::array();
  for (const auto& item : ranking) {
    if (item.at("candidate_id") != "raw") result.push_back(item);
  }
  return result;
}

void attach_compression_profile(Json& prediction, const Json& catalog) {
  const auto kinds = catalog_candidate_kinds(catalog);
  prediction["profile"] = "compression_only_no_raw";
  for (const auto& id : prediction.at("eligible_candidates")) {
    d::require(id.get<std::string>() != "raw",
               "compression-only prediction retained raw eligibility");
  }

  Json first_by_target = Json::object();
  for (const char* target : kTargets) {
    Json first = Json::object();
    for (const char* kind : kCompressionKinds) first[kind] = nullptr;
    std::size_t rank = 0;
    for (const auto& item :
         prediction.at("targets").at(target).at("ranking")) {
      ++rank;
      const auto id = item.at("candidate_id").get<std::string>();
      d::require(id != "raw", "compression-only ranking retained raw");
      const auto found = kinds.find(id);
      d::require(found != kinds.end(),
                 "prediction contains a candidate outside the catalog");
      const auto& kind = found->second;
      if (first.contains(kind) && first.at(kind).is_null()) {
        Json selected = item;
        selected["kind"] = kind;
        selected["rank"] = rank;
        first[kind] = std::move(selected);
      }
    }
    prediction["targets"][target]["first_by_kind"] = first;
    first_by_target[target] = std::move(first);
  }
  prediction["first_predicted_by_kind"] = std::move(first_by_target);
}

std::uint64_t median(std::vector<std::uint64_t> values) {
  d::require(!values.empty(), "cannot take the median of no timings");
  std::sort(values.begin(), values.end());
  return values.at(values.size() / 2);
}

double nonnegative_number(const Json& object, const char* key) {
  d::require(object.contains(key) && object.at(key).is_number(),
             std::string("missing measured cost: ") + key);
  const double result = object.at(key).get<double>();
  d::require(result >= 0.0, std::string("negative measured cost: ") + key);
  return result;
}

Json serve_one(const std::filesystem::path& model_path,
               const std::filesystem::path& sample_path) {
  const Json model = d::read_json(model_path);
  const Json sample = d::read_json(sample_path);
  const Block block =
      load_input_part(sample, sample.contains("head") ? "head" : "tail");

  const auto feature_cpu_start = d::cpu_ns();
  const auto feature_wall_start = d::wall_ns();
  const auto features =
      extract_features(block.values, block.dtype, block.missing, block.row_count);
  const auto feature_wall_ns = d::wall_ns() - feature_wall_start;
  const auto feature_cpu_ns = d::cpu_ns() - feature_cpu_start;

  const auto eligibility_cpu_start = d::cpu_ns();
  const auto eligibility_wall_start = d::wall_ns();
  const Json catalog = candidate_catalog();
  std::vector<std::string> full_eligible;
  for (const auto& candidate : catalog.at("candidates")) {
    const auto id = candidate.at("candidate_id").get<std::string>();
    if (candidate_eligible(id, block)) full_eligible.push_back(id);
  }
  const auto eligibility_wall_ns = d::wall_ns() - eligibility_wall_start;
  const auto eligibility_cpu_ns = d::cpu_ns() - eligibility_cpu_start;

  require_model_contract(model, catalog);
  const auto filter_cpu_start = d::cpu_ns();
  const auto filter_wall_start = d::wall_ns();
  const auto eligible = without_raw(full_eligible);
  const auto filter_wall_ns = d::wall_ns() - filter_wall_start;
  const auto filter_cpu_ns = d::cpu_ns() - filter_cpu_start;
  d::require(!eligible.empty(),
             "no compression candidate is eligible for this input");

  const auto model_cpu_start = d::cpu_ns();
  const auto model_wall_start = d::wall_ns();
  Json result = predict(model, features, eligible);
  const auto model_wall_ns = d::wall_ns() - model_wall_start;
  const auto model_cpu_ns = d::cpu_ns() - model_cpu_start;
  attach_compression_profile(result, catalog);

  result["sample_id"] = sample.value("sample_id", std::string("new-input"));
  result["dtype"] = block.dtype;
  result["row_count"] = block.row_count;
  result["input_bytes"] = block.values.size() + block.missing.size();
  result["features"] = features;
  result["full_runtime_eligible_candidates"] = full_eligible;
  result["decision_cost"] = {
      {"feature_cpu_ns", feature_cpu_ns},
      {"feature_wall_ns", feature_wall_ns},
      {"eligibility_cpu_ns", eligibility_cpu_ns},
      {"eligibility_wall_ns", eligibility_wall_ns},
      {"eligibility_provenance",
       "frozen runtime dtype/shape rule on head before compression-only filter"},
      {"filter_cpu_ns", filter_cpu_ns},
      {"filter_wall_ns", filter_wall_ns},
      {"filter_contract", "construct eligible list with candidate_id != raw"},
      {"model_cpu_ns", model_cpu_ns},
      {"model_wall_ns", model_wall_ns},
      {"model_timing_contract", "outer clock around complete predict call"},
      {"online_decision_cpu_ns",
       feature_cpu_ns + eligibility_cpu_ns + filter_cpu_ns + model_cpu_ns},
      {"online_decision_wall_ns",
       feature_wall_ns + eligibility_wall_ns + filter_wall_ns + model_wall_ns},
      {"probe_used", false},
      {"probe_cost_cpu_ns", 0},
      {"probe_cost_wall_ns", 0}};
  return result;
}

Json replay(const std::filesystem::path& rows_path,
            const std::filesystem::path& evaluation_path) {
  const auto rows_bytes = d::read_bytes(rows_path);
  const auto evaluation_bytes = d::read_bytes(evaluation_path);
  const Json rows = Json::parse(rows_bytes.begin(), rows_bytes.end());
  const Json evaluation =
      Json::parse(evaluation_bytes.begin(), evaluation_bytes.end());
  d::require(rows.is_array(), "replay rows must be an array");
  d::require(evaluation.value("schema", std::string()) ==
                 "operator-recommend135-evaluation-v1",
             "unsupported evaluation schema");

  const Json catalog = candidate_catalog();
  const auto catalog_sha256 = d::sha(catalog.dump());
  const auto catalog_ids = catalog_candidate_ids(catalog);
  d::require(catalog_ids.size() == 24,
             "replay requires the frozen 24-candidate catalog");
  const auto& scope = evaluation.at("candidate_scope");
  d::require(scope.at("candidate_ids").get<std::vector<std::string>>() ==
                     catalog_ids &&
                 scope.at("catalog_sha256").get<std::string>() ==
                     catalog_sha256,
             "evaluation candidate scope differs from the runtime catalog");
  if (evaluation.contains("training_rows_sha256")) {
    d::require(evaluation.at("training_rows_sha256").get<std::string>() ==
                   d::sha(rows_bytes),
               "evaluation training_rows_sha256 mismatch");
  }

  std::map<std::string, const Json*> rows_by_sample;
  for (const auto& row : rows) {
    const auto sample_id = row.at("sample_id").get<std::string>();
    d::require(rows_by_sample.emplace(sample_id, &row).second,
               "duplicate sample_id in replay rows");
  }

  const Json options{{"candidate_ids", catalog_ids},
                     {"catalog_sha256", catalog_sha256}};
  Json output_predictions = Json::array();
  Json offline_fits = Json::array();
  std::size_t matched_count = 0;
  std::set<std::string> seen_folds;
  std::set<std::string> matched_samples;

  for (const auto& frozen_fold : evaluation.at("folds")) {
    const auto held_out =
        frozen_fold.at("held_out_group_id").get<std::string>();
    d::require(seen_folds.insert(held_out).second,
               "evaluation repeats a held-out group");
    Json training_rows = Json::array();
    std::set<std::string> training_groups;
    for (const auto& row : rows) {
      const auto group = row.at("group_id").get<std::string>();
      if (group != held_out) {
        training_rows.push_back(row);
        training_groups.insert(group);
      }
    }
    d::require(training_rows.size() ==
                   frozen_fold.at("training_row_count").get<std::size_t>(),
               "reconstructed fold training row count mismatch");
    d::require(std::vector<std::string>(training_groups.begin(),
                                       training_groups.end()) ==
                   frozen_fold.at("training_group_ids")
                       .get<std::vector<std::string>>(),
               "reconstructed fold training groups mismatch");

    const Json model = fit_model(training_rows, options);
    require_model_contract(model, catalog);
    d::require(model.at("candidate_ids").get<std::vector<std::string>>() ==
                   catalog_ids,
               "replay fit changed the frozen candidate scope");
    offline_fits.push_back(
        {{"held_out_group_id", held_out},
         {"training_row_count", training_rows.size()},
         {"training_group_ids",
          std::vector<std::string>(training_groups.begin(),
                                   training_groups.end())},
         {"offline_fit_cpu_ns", model.at("offline_fit_cpu_ns")},
         {"offline_fit_wall_ns", model.at("offline_fit_wall_ns")}});

    for (const auto& frozen : frozen_fold.at("predictions")) {
      const auto sample_id = frozen.at("sample_id").get<std::string>();
      const auto found = rows_by_sample.find(sample_id);
      d::require(found != rows_by_sample.end(),
                 "evaluation sample is absent from replay rows");
      const Json& test_row = *found->second;
      d::require(test_row.at("group_id") == held_out &&
                     frozen.at("group_id") == held_out,
                 "evaluation sample is in the wrong held-out group");
      d::require(frozen.at("eligible_candidates") ==
                     test_row.at("eligible_candidates"),
                 "evaluation eligibility differs from the source row");
      d::require(matched_samples.insert(sample_id).second,
                 "evaluation repeats a held-out sample");

      const auto filter_cpu_start = d::cpu_ns();
      const auto filter_wall_start = d::wall_ns();
      const auto eligible = without_raw(
          test_row.at("eligible_candidates").get<std::vector<std::string>>());
      const auto filter_wall_ns = d::wall_ns() - filter_wall_start;
      const auto filter_cpu_ns = d::cpu_ns() - filter_cpu_start;
      d::require(!eligible.empty(),
                 "held-out row has no eligible compression candidate");

      const auto features = test_row.at("features").get<std::vector<double>>();
      Json raw_timings = Json::array();
      std::vector<std::uint64_t> model_cpu_samples;
      std::vector<std::uint64_t> model_wall_samples;
      Json representative;
      for (std::size_t repeat = 0; repeat < 5; ++repeat) {
        const auto cpu_start = d::cpu_ns();
        const auto wall_start = d::wall_ns();
        Json current = predict(model, features, eligible);
        const auto wall_ns = d::wall_ns() - wall_start;
        const auto cpu_ns = d::cpu_ns() - cpu_start;
        model_cpu_samples.push_back(cpu_ns);
        model_wall_samples.push_back(wall_ns);
        raw_timings.push_back({{"repeat", repeat},
                               {"model_cpu_ns", cpu_ns},
                               {"model_wall_ns", wall_ns},
                               {"internal_online_cost",
                                current.at("online_cost")}});
        for (const char* target : kTargets) {
          const Json expected = ranking_without_raw(
              frozen.at("prediction").at("targets").at(target).at("ranking"));
          d::require(current.at("targets").at(target).at("ranking") ==
                         expected,
                     std::string("replay ranking differs from frozen ") +
                         target + " ranking after removing raw");
        }
        if (repeat == 0) representative = std::move(current);
      }
      attach_compression_profile(representative, catalog);
      const auto model_cpu_ns = median(model_cpu_samples);
      const auto model_wall_ns = median(model_wall_samples);
      const auto feature_cpu_ns = nonnegative_number(test_row, "feature_cpu_ns");
      const auto feature_wall_ns =
          nonnegative_number(test_row, "feature_wall_ns");
      const auto eligibility_cpu_ns =
          nonnegative_number(test_row, "eligibility_cpu_ns");
      const auto eligibility_wall_ns =
          nonnegative_number(test_row, "eligibility_wall_ns");
      ++matched_count;
      output_predictions.push_back(
          {{"sample_id", sample_id},
           {"group_id", held_out},
           {"filter_cpu_ns", filter_cpu_ns},
           {"filter_wall_ns", filter_wall_ns},
           {"model_cpu_ns", model_cpu_ns},
           {"model_wall_ns", model_wall_ns},
           {"raw_timings", std::move(raw_timings)},
           {"rankings_match_frozen", true},
           {"decision_cost",
            {{"feature_cpu_ns", feature_cpu_ns},
             {"feature_wall_ns", feature_wall_ns},
             {"eligibility_cpu_ns", eligibility_cpu_ns},
             {"eligibility_wall_ns", eligibility_wall_ns},
             {"eligibility_provenance",
              test_row.value("eligibility_provenance",
                             std::string("fixture_unspecified"))},
             {"filter_cpu_ns", filter_cpu_ns},
             {"filter_wall_ns", filter_wall_ns},
             {"model_cpu_ns", model_cpu_ns},
             {"model_wall_ns", model_wall_ns},
             {"model_timing_contract",
              "median of five outer-clock complete predict calls"},
             {"online_decision_cpu_ns",
              feature_cpu_ns + eligibility_cpu_ns + filter_cpu_ns +
                  model_cpu_ns},
             {"online_decision_wall_ns",
              feature_wall_ns + eligibility_wall_ns + filter_wall_ns +
                  model_wall_ns},
             {"probe_used", false},
             {"probe_cost_cpu_ns", 0},
             {"probe_cost_wall_ns", 0}}},
           {"prediction", std::move(representative)}});
    }
  }
  d::require(matched_samples.size() == rows.size(),
             "evaluation does not contain every replay row exactly once");

  return {{"schema", "operator-recommend135-compression-replay-v1"},
          {"profile", "compression_only_no_raw"},
          {"source_rows_sha256", d::sha(rows_bytes)},
          {"source_evaluation_sha256", d::sha(evaluation_bytes)},
          {"catalog_sha256", catalog_sha256},
          {"predictions", std::move(output_predictions)},
          {"offline_fits", std::move(offline_fits)},
          {"fold_count", evaluation.at("folds").size()},
          {"matched_count", matched_count},
          {"status", "matched_frozen_rankings"}};
}

}  // namespace
}  // namespace operator_recommend135_v1

int main(int argc, char** argv) {
  namespace ors = operator_recommend135_v1;
  namespace d = operator_recommend135_v1::detail;
  try {
    nlohmann::json result;
    std::filesystem::path output;
    if (argc == 4) {
      output = argv[3];
      result = ors::serve_one(argv[1], argv[2]);
    } else if (argc == 5 && std::string(argv[1]) == "--replay") {
      output = argv[4];
      result = ors::replay(argv[2], argv[3]);
    } else {
      throw std::runtime_error(
          "usage: compression_serve MODEL SAMPLE OUTPUT | "
          "compression_serve --replay ROWS EVALUATION OUTPUT");
    }
    d::write_json(output, result);
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << nlohmann::json({{"status", "failed"},
                                 {"error", error.what()}})
                     .dump()
              << '\n';
    return 1;
  }
}
