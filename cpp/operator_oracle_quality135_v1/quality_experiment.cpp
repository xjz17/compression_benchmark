#include "quality_model.hpp"

#include "../operator_recommend135_v1/common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace q = operator_oracle_quality135_v1;
namespace d = operator_recommend135_v1::detail;
using Json = nlohmann::json;

namespace {

constexpr std::size_t kExpectedDevelopmentRows = 3433;
constexpr std::size_t kFoldCount = 5;
constexpr double kUnsupportedPenalty = 1000.0;
constexpr std::string_view kRequiredTimeContract =
    "server192-current135-valid-value-w1-r3-t1-v1";

const std::vector<std::string>& expected_feature_names() {
  static const std::vector<std::string> names = {
      "log1p_row_count",          "log1p_valid_count",
      "scanned_word_count",       "scanned_value_bytes",
      "dtype_bit_width",          "dtype_is_float",
      "dtype_is_signed",          "missing_fraction",
      "byte_entropy",             "distinct_fraction",
      "run_fraction",             "value_zero_fraction",
      "finite_fraction",          "nonfinite_fraction",
      "delta_zero_fraction",      "delta_bw_1_8_fraction",
      "delta_bw_9_16_fraction",   "delta_bw_17_32_fraction",
      "delta_bw_33_64_fraction",  "xor_zero_fraction",
      "xor_bw_1_8_fraction",      "xor_bw_9_16_fraction",
      "xor_bw_17_32_fraction",    "xor_bw_33_64_fraction"};
  return names;
}

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("operator_oracle_quality135_v1 experiment: " +
                           message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::string text(const Json& value, const char* key,
                 const std::string& context) {
  require(value.contains(key) && value.at(key).is_string() &&
              !value.at(key).get_ref<const std::string&>().empty(),
          context + " missing non-empty string " + key);
  return value.at(key).get<std::string>();
}

std::size_t unsigned_integer(const Json& value, const char* key,
                             const std::string& context) {
  require(value.contains(key) &&
              (value.at(key).is_number_unsigned() ||
               (value.at(key).is_number_integer() &&
                value.at(key).get<std::int64_t>() >= 0)),
          context + " missing nonnegative integer " + key);
  return value.at(key).get<std::size_t>();
}

std::optional<double> optional_positive(const Json& value, const char* key,
                                        const std::string& context) {
  if (!value.contains(key) || value.at(key).is_null()) return std::nullopt;
  require(value.at(key).is_number(), context + " nonnumeric " + key);
  const double result = value.at(key).get<double>();
  require(std::isfinite(result) && result > 0.0,
          context + " nonpositive/nonfinite " + key);
  return result;
}

std::vector<std::string> strings(const Json& value,
                                 const std::string& context) {
  require(value.is_array(), context + " must be an array");
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto& item : value) {
    require(item.is_string() && !item.get_ref<const std::string&>().empty(),
            context + " contains a non-string/empty item");
    auto candidate = item.get<std::string>();
    require(seen.insert(candidate).second, context + " contains duplicates");
    result.push_back(std::move(candidate));
  }
  return result;
}

std::vector<double> finite_features(const Json& value,
                                    const std::string& context) {
  require(value.is_array(), context + " must be an array");
  auto result = value.get<std::vector<double>>();
  require(result.size() == expected_feature_names().size(),
          context + " must contain exactly 24 values");
  require(std::all_of(result.begin(), result.end(),
                      [](double x) { return std::isfinite(x); }),
          context + " contains a non-finite value");
  return result;
}

std::vector<Json> read_jsonl(const fs::path& path) {
  std::ifstream source(path);
  require(bool(source), "cannot read JSONL " + path.string());
  std::vector<Json> result;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(source, line)) {
    ++line_number;
    if (line.empty()) continue;
    try {
      result.push_back(Json::parse(line));
    } catch (const std::exception& error) {
      fail(path.string() + ":" + std::to_string(line_number) + ": " +
           error.what());
    }
  }
  require(source.eof(), "error while reading JSONL " + path.string());
  require(!result.empty(), "empty JSONL " + path.string());
  return result;
}

std::string key(const std::string& dataset, const std::string& sample) {
  return dataset + "\n" + sample;
}

struct FeatureRow {
  std::string dataset_id;
  std::string sample_id;
  std::string dtype;
  std::vector<double> features;
};

struct RawCandidate {
  q::CandidateCosts costs;
  std::string timing_contract;
};

struct CostRow {
  std::string sample_id;
  std::string dataset_id;
  std::string dtype;
  std::string source_group;
  std::string legacy_group;
  std::size_t dev_fold = 0;
  std::map<q::Domain, std::vector<RawCandidate>> domains;
};

struct Task {
  std::string id;
  q::Domain domain;
  q::Objective objective;
};

const std::array<Task, 6> kTasks = {
    Task{"algorithm_space", q::Domain::Algorithm, q::Objective::Space},
    Task{"algorithm_encode", q::Domain::Algorithm, q::Objective::Encode},
    Task{"algorithm_decode", q::Domain::Algorithm, q::Objective::Decode},
    Task{"algorithm_balanced", q::Domain::Algorithm, q::Objective::Balanced},
    Task{"operator_zstd3_space", q::Domain::OperatorZstd3,
         q::Objective::Space},
    Task{"observed_pipeline_space", q::Domain::ObservedPipeline,
         q::Objective::Space}};

bool is_time_task(const Task& task) {
  return task.domain == q::Domain::Algorithm &&
         task.objective != q::Objective::Space;
}

const std::optional<double>& selected_cost(const q::CandidateCosts& costs,
                                           q::Objective objective) {
  switch (objective) {
    case q::Objective::Space:
      return costs.space_cost;
    case q::Objective::Encode:
      return costs.encode_cost;
    case q::Objective::Decode:
      return costs.decode_cost;
    case q::Objective::Balanced:
      return costs.balanced_cost;
  }
  fail("unknown objective");
}

struct Protocol {
  bool require_frozen_current135 = true;
  std::set<std::string> confirm_legacy_groups;
  std::string algorithm_time_contract;
  std::set<std::string> allowed_time_candidate_ids;
  fs::path features_receipt_path;
};

Protocol parse_protocol(const fs::path& path) {
  const auto value = d::read_json(path);
  require(value.is_object(), "protocol must be an object");
  require(value.value("schema", std::string()) ==
              "operator-oracle-quality135-protocol-v1",
          "unexpected protocol schema");
  require(unsigned_integer(value, "dev_fold_count", "protocol") ==
              kFoldCount,
          "protocol must declare five development folds");
  require(value.contains("confirm_legacy_groups"),
          "protocol lacks confirm_legacy_groups");
  Protocol result;
  for (const auto& group :
       strings(value.at("confirm_legacy_groups"),
               "protocol.confirm_legacy_groups")) {
    result.confirm_legacy_groups.insert(group);
  }
  result.algorithm_time_contract =
      text(value, "algorithm_time_contract", "protocol");
  require(result.algorithm_time_contract == kRequiredTimeContract,
          "protocol algorithm_time_contract does not match the frozen "
          "server192 contract");
  if (value.contains("allowed_time_candidate_ids")) {
    for (const auto& candidate :
         strings(value.at("allowed_time_candidate_ids"),
                 "protocol.allowed_time_candidate_ids")) {
      result.allowed_time_candidate_ids.insert(candidate);
    }
  }
  result.require_frozen_current135 =
      value.value("require_frozen_current135", true);
  const fs::path receipt(
      text(value, "features_receipt_path", "protocol"));
  result.features_receipt_path =
      (receipt.is_absolute() ? receipt : path.parent_path() / receipt)
          .lexically_normal();
  return result;
}

std::map<std::string, FeatureRow> load_features(
    const fs::path& path, const Json* receipt = nullptr,
    bool require_frozen = false) {
  std::map<std::string, FeatureRow> result;
  std::optional<std::string> contract;
  std::optional<std::string> names_sha;
  std::optional<std::string> columns_sha;
  std::optional<std::string> oracle_sha;
  const auto expected_names_sha = d::sha(Json(expected_feature_names()).dump());
  for (const auto& value : read_jsonl(path)) {
    require(value.is_object() &&
                value.value("schema", std::string()) ==
                    "operator-oracle-quality135-feature-row-v1",
            "unexpected feature row schema");
    require(value.value("oracle_labels_read", true) == false,
            "feature row is not label-isolated");
    const auto dataset = text(value, "dataset_id", "feature row");
    const auto sample = text(value, "column_id", "feature row");
    const auto dtype = text(value, "input_dtype", "feature row");
    const auto row_contract = text(value, "feature_contract", "feature row");
    const auto row_names_sha =
        text(value, "feature_names_sha256", "feature row");
    const auto row_columns_sha =
        text(value, "columns_sha256", "feature row");
    const auto row_oracle_sha =
        text(value, "oracle_reference_jsonl_sha256", "feature row");
    require(row_names_sha == expected_names_sha,
            "feature_names_sha256 does not match current 24 features");
    if (contract) {
      require(*contract == row_contract, "feature contracts are mixed");
      require(*names_sha == row_names_sha, "feature-name hashes are mixed");
      require(*columns_sha == row_columns_sha, "column hashes are mixed");
      require(*oracle_sha == row_oracle_sha, "oracle hashes are mixed");
    } else {
      contract = row_contract;
      names_sha = row_names_sha;
      columns_sha = row_columns_sha;
      oracle_sha = row_oracle_sha;
    }
    FeatureRow row{dataset, sample, dtype,
                   finite_features(value.at("features"), "features")};
    require(result.emplace(key(dataset, sample), std::move(row)).second,
            "duplicate feature identity");
  }
  if (receipt != nullptr) {
    require(receipt->is_object() &&
                receipt->value("schema", std::string()) ==
                    "operator-oracle-quality135-features-receipt-v1" &&
                receipt->value("status", std::string()) == "complete",
            "unexpected features receipt");
    require(fs::absolute(path).lexically_normal() ==
                fs::path(text(*receipt, "output_path", "features receipt"))
                    .lexically_normal(),
            "features receipt output_path mismatch");
    require(d::file_sha(path) ==
                text(*receipt, "output_sha256", "features receipt"),
            "features output SHA256 mismatch");
    require(*columns_sha ==
                text(*receipt, "columns_sha256", "features receipt") &&
                *oracle_sha == text(*receipt,
                                    "oracle_reference_jsonl_sha256",
                                    "features receipt") &&
                *names_sha == text(*receipt, "feature_names_sha256",
                                    "features receipt"),
            "features rows disagree with receipt hashes");
    require(receipt->at("feature_names") == Json(expected_feature_names()),
            "features receipt names mismatch");
    require(unsigned_integer(*receipt, "column_count", "features receipt") ==
                result.size(),
            "features receipt column count mismatch");
    if (require_frozen) {
      require(result.size() == 3519 &&
                  unsigned_integer(*receipt, "dataset_count",
                                   "features receipt") == 135,
              "features receipt is outside frozen current135 scope");
    }
  }
  return result;
}

RawCandidate parse_candidate(const Json& value, const std::string& context) {
  require(value.is_object(), context + " candidate is not an object");
  RawCandidate result;
  result.costs.candidate_id = text(value, "candidate_id", context);
  require(value.contains("eligible") && value.at("eligible").is_boolean(),
          context + " lacks boolean eligible");
  result.costs.eligible = value.at("eligible").get<bool>();
  result.costs.space_cost = optional_positive(value, "space_cost", context);
  result.costs.encode_cost = optional_positive(value, "encode_cost", context);
  result.costs.decode_cost = optional_positive(value, "decode_cost", context);
  result.costs.balanced_cost =
      optional_positive(value, "balanced_cost", context);
  if (value.contains("timing_contract") &&
      value.at("timing_contract").is_string()) {
    result.timing_contract = value.at("timing_contract").get<std::string>();
  }
  return result;
}

std::vector<CostRow> load_costs(
    const fs::path& path, const Protocol& protocol,
    std::map<q::Domain, std::vector<std::string>>& declared_candidates) {
  std::vector<CostRow> result;
  std::map<std::string, std::size_t> source_fold;
  std::ifstream source(path);
  require(bool(source), "cannot read development costs " + path.string());
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(source, line)) {
    ++line_number;
    if (line.empty()) continue;
    Json value;
    try {
      value = Json::parse(line);
    } catch (const std::exception& error) {
      fail(path.string() + ":" + std::to_string(line_number) + ": " +
           error.what());
    }
    require(value.is_object(), "cost row must be an object");
    require(text(value, "split", "cost row") == "development",
            "tune accepts development rows only");
    CostRow row;
    row.sample_id = text(value, "sample_id", "cost row");
    row.dataset_id = text(value, "dataset_id", "cost row");
    row.dtype = text(value, "dtype", "cost row");
    row.source_group = text(value, "source_group", "cost row");
    row.legacy_group = text(value, "legacy_group", "cost row");
    row.dev_fold = unsigned_integer(value, "dev_fold", "cost row");
    require(row.dev_fold < kFoldCount, "dev_fold is outside [0,4]");
    const auto source_it = source_fold.find(row.source_group);
    if (source_it == source_fold.end()) {
      source_fold.emplace(row.source_group, row.dev_fold);
    } else {
      require(source_it->second == row.dev_fold,
              "one source_group crosses official development folds");
    }
    require(value.contains("domains") && value.at("domains").is_object(),
            "cost row lacks domains");
    for (const auto domain : {q::Domain::Algorithm, q::Domain::OperatorZstd3,
                              q::Domain::ObservedPipeline}) {
      const auto name = std::string(q::domain_name(domain));
      const auto& input = value.at("domains").at(name);
      require(input.is_array() && !input.empty(),
              "domain " + name + " must be a non-empty array");
      std::vector<RawCandidate> candidates;
      std::set<std::string> ids;
      for (std::size_t i = 0; i < input.size(); ++i) {
        auto candidate = parse_candidate(
            input.at(i), "domain " + name + " row " + std::to_string(i));
        require(ids.insert(candidate.costs.candidate_id).second,
                "duplicate candidate in domain " + name);
        candidates.push_back(std::move(candidate));
      }
      std::vector<std::string> current(ids.begin(), ids.end());
      if (!declared_candidates.contains(domain)) {
        declared_candidates.emplace(domain, current);
      } else {
        require(declared_candidates.at(domain) == current,
                "declared candidate set changes across rows for " + name);
      }
      row.domains.emplace(domain, std::move(candidates));
    }
    result.push_back(std::move(row));
  }
  require(source.eof(), "error while reading development costs");
  require(!result.empty(), "development costs JSONL is empty");
  if (protocol.require_frozen_current135) {
    require(result.size() == kExpectedDevelopmentRows,
            "frozen development row count differs from 3433");
    require(declared_candidates.at(q::Domain::Algorithm).size() == 84,
            "algorithm training scope must contain 84 candidates");
    require(declared_candidates.at(q::Domain::OperatorZstd3).size() == 97,
            "operator scope must contain 97 candidates");
    require(declared_candidates.at(q::Domain::ObservedPipeline).size() == 181,
            "observed-pipeline scope must contain 181 measured slots");
  }
  return result;
}

const RawCandidate* find_candidate(const CostRow& row, q::Domain domain,
                                   const std::string& candidate_id) {
  const auto& candidates = row.domains.at(domain);
  const auto it = std::find_if(candidates.begin(), candidates.end(),
                               [&](const RawCandidate& candidate) {
                                 return candidate.costs.candidate_id ==
                                        candidate_id;
                               });
  return it == candidates.end() ? nullptr : &*it;
}

bool usable(const RawCandidate& candidate, const Task& task,
            const Protocol& protocol) {
  if (!candidate.costs.eligible ||
      !selected_cost(candidate.costs, task.objective)) {
    return false;
  }
  if (is_time_task(task)) {
    if (candidate.timing_contract != protocol.algorithm_time_contract) {
      return false;
    }
    if (!protocol.allowed_time_candidate_ids.empty() &&
        !protocol.allowed_time_candidate_ids.contains(
            candidate.costs.candidate_id)) {
      return false;
    }
  }
  return true;
}

using DTypePools = std::map<std::string, std::vector<std::string>>;

DTypePools build_dtype_pools(
    const std::vector<CostRow>& costs, const std::vector<std::size_t>& indexes,
    const Task& task, const Protocol& protocol,
    const std::vector<std::string>& declared) {
  std::map<std::string, std::vector<std::size_t>> by_dtype;
  for (const auto index : indexes) by_dtype[costs[index].dtype].push_back(index);
  DTypePools result;
  for (const auto& [dtype, rows] : by_dtype) {
    auto& pool = result[dtype];
    for (const auto& candidate_id : declared) {
      const bool complete = std::all_of(
          rows.begin(), rows.end(), [&](std::size_t index) {
            const auto* candidate =
                find_candidate(costs[index], task.domain, candidate_id);
            return candidate != nullptr && usable(*candidate, task, protocol);
          });
      if (complete) pool.push_back(candidate_id);
    }
    require(!pool.empty(), "empty conservative candidate pool for task " +
                               task.id + " dtype " + dtype);
  }
  return result;
}

std::vector<q::TrainingRow> make_training_rows(
    const std::vector<CostRow>& costs, const std::vector<std::size_t>& indexes,
    const std::map<std::string, FeatureRow>& features, const Task& task,
    const Protocol& protocol) {
  std::map<std::string, std::size_t> dataset_columns;
  for (const auto index : indexes) ++dataset_columns[costs[index].dataset_id];
  std::vector<q::TrainingRow> result;
  result.reserve(indexes.size());
  for (const auto index : indexes) {
    const auto& source = costs[index];
    const auto feature_it =
        features.find(key(source.dataset_id, source.sample_id));
    require(feature_it != features.end(), "cost row lacks joined features");
    require(feature_it->second.dtype == source.dtype,
            "cost/feature dtype mismatch");
    q::TrainingRow row;
    row.sample_id = source.dataset_id + ":" + source.sample_id;
    row.dataset_id = source.dataset_id;
    row.source_group = source.source_group;
    row.weight = 1.0 /
                 static_cast<double>(dataset_columns.at(source.dataset_id));
    row.features = feature_it->second.features;
    for (const auto& candidate : source.domains.at(task.domain)) {
      auto copy = candidate.costs;
      if (!usable(candidate, task, protocol)) copy.eligible = false;
      row.candidates.push_back(std::move(copy));
    }
    result.push_back(std::move(row));
  }
  return result;
}

std::vector<std::size_t> training_indexes(
    const std::vector<CostRow>& costs, std::optional<std::size_t> heldout_fold,
    const Protocol& protocol) {
  std::set<std::string> excluded = protocol.confirm_legacy_groups;
  if (heldout_fold) {
    for (const auto& row : costs) {
      if (row.dev_fold == *heldout_fold) excluded.insert(row.legacy_group);
    }
  }
  std::vector<std::size_t> result;
  for (std::size_t i = 0; i < costs.size(); ++i) {
    if (heldout_fold && costs[i].dev_fold == *heldout_fold) continue;
    if (excluded.contains(costs[i].legacy_group)) continue;
    result.push_back(i);
  }
  require(!result.empty(), "legacy purge removed every training row");
  return result;
}

std::vector<std::size_t> validation_indexes(
    const std::vector<CostRow>& costs, std::size_t fold) {
  std::vector<std::size_t> result;
  for (std::size_t i = 0; i < costs.size(); ++i)
    if (costs[i].dev_fold == fold) result.push_back(i);
  require(!result.empty(), "official development fold is empty");
  return result;
}

std::map<std::string, std::string> constant_best(
    const std::vector<CostRow>& costs, const std::vector<std::size_t>& indexes,
    const Task& task, const Protocol& protocol, const DTypePools& pools) {
  std::map<std::string, std::string> result;
  for (const auto& [dtype, pool] : pools) {
    double best_score = std::numeric_limits<double>::infinity();
    std::string best;
    for (const auto& candidate_id : pool) {
      std::map<std::string, std::pair<double, std::size_t>> datasets;
      for (const auto index : indexes) {
        const auto& row = costs[index];
        if (row.dtype != dtype) continue;
        const auto* candidate =
            find_candidate(row, task.domain, candidate_id);
        require(candidate != nullptr && usable(*candidate, task, protocol),
                "constant baseline candidate violates training pool");
        auto& aggregate = datasets[row.dataset_id];
        aggregate.first +=
            std::log(*selected_cost(candidate->costs, task.objective));
        ++aggregate.second;
      }
      double score = 0.0;
      for (const auto& [dataset, aggregate] : datasets) {
        (void)dataset;
        score += aggregate.first / static_cast<double>(aggregate.second);
      }
      score /= static_cast<double>(datasets.size());
      if (score < best_score || (score == best_score && candidate_id < best)) {
        best_score = score;
        best = candidate_id;
      }
    }
    require(!best.empty(), "failed to choose constant baseline");
    result[dtype] = best;
  }
  return result;
}

struct PolicyConfig {
  std::string id;
  std::string kind;
  q::FitConfig model;
  std::string fixed_candidate;
};

std::vector<PolicyConfig> policy_grid(const Task& task) {
  std::vector<PolicyConfig> result;
  for (double lambda : {0.1, 1.0, 10.0}) {
    q::FitConfig ridge;
    ridge.ridge_lambda = lambda;
    ridge.knn_k = 0;
    ridge.ridge_weight = 1.0;
    ridge.center_log_cost_per_row = true;
    ridge.feature_names = expected_feature_names();
    result.push_back({"ridge_l" + std::to_string(lambda), "model", ridge,
                      {}});
    for (std::size_t k : {std::size_t{8}, std::size_t{24}}) {
      q::FitConfig blend = ridge;
      blend.knn_k = k;
      blend.ridge_weight = 0.5;
      result.push_back({"blend_l" + std::to_string(lambda) + "_k" +
                            std::to_string(k),
                        "model", blend, {}});
    }
  }
  // Pure kNN ignores ridge coefficients, so lambda is fixed once to avoid
  // duplicate grid points.
  for (std::size_t k : {std::size_t{8}, std::size_t{24}}) {
    q::FitConfig knn;
    knn.ridge_lambda = 1.0;
    knn.knn_k = k;
    knn.ridge_weight = 0.0;
    knn.center_log_cost_per_row = true;
    knn.feature_names = expected_feature_names();
    result.push_back(
        {"knn_k" + std::to_string(k), "model", knn, {}});
  }
  result.push_back({"constant_best_per_dtype", "constant", {}, {}});
  if (task.domain == q::Domain::Algorithm) {
    result.push_back({"fixed_LZ4", "fixed", {}, "LZ4"});
    result.push_back({"fixed_Zstd3", "fixed", {}, "Zstd"});
  }
  return result;
}

Json policy_json(const PolicyConfig& policy) {
  Json result{{"id", policy.id}, {"kind", policy.kind}};
  if (policy.kind == "model") {
    result["ridge_lambda"] = policy.model.ridge_lambda;
    result["knn_k"] = policy.model.knn_k;
    result["ridge_weight"] = policy.model.ridge_weight;
    result["center_log_cost_per_row"] =
        policy.model.center_log_cost_per_row;
  }
  if (policy.kind == "fixed") {
    result["fixed_candidate_id"] = policy.fixed_candidate;
  }
  return result;
}

double finite_number(const Json& value, const char* key,
                     const std::string& context) {
  require(value.contains(key) && value.at(key).is_number(),
          context + " missing numeric " + key);
  const double result = value.at(key).get<double>();
  require(std::isfinite(result), context + " non-finite " + key);
  return result;
}

PolicyConfig parse_policy(const Json& value, const std::string& context) {
  require(value.is_object(), context + " must be an object");
  PolicyConfig result;
  result.id = text(value, "id", context);
  result.kind = text(value, "kind", context);
  if (result.kind == "model") {
    result.model.ridge_lambda = finite_number(value, "ridge_lambda", context);
    result.model.knn_k = unsigned_integer(value, "knn_k", context);
    result.model.ridge_weight = finite_number(value, "ridge_weight", context);
    require(value.contains("center_log_cost_per_row") &&
                value.at("center_log_cost_per_row").is_boolean(),
            context + " missing boolean center_log_cost_per_row");
    result.model.center_log_cost_per_row =
        value.at("center_log_cost_per_row").get<bool>();
    result.model.feature_names = expected_feature_names();
    require(result.model.ridge_lambda > 0.0,
            context + " ridge_lambda must be positive");
    require(result.model.ridge_weight >= 0.0 &&
                result.model.ridge_weight <= 1.0,
            context + " ridge_weight must be in [0,1]");
    require(result.model.knn_k != 0 || result.model.ridge_weight == 1.0,
            context + " knn_k=0 requires ridge_weight=1");
  } else if (result.kind == "fixed") {
    result.fixed_candidate =
        text(value, "fixed_candidate_id", context);
  } else {
    require(result.kind == "constant",
            context + " has unsupported kind " + result.kind);
  }
  require(policy_json(result) == value,
          context + " is not a canonical selected-policy object");
  return result;
}

struct Selection {
  std::optional<std::string> candidate_id;
  std::optional<double> predicted_log_cost;
  std::vector<q::RankedCandidate> ranking;
};

Selection select(const PolicyConfig& policy, const q::QualityModel* model,
                 const std::map<std::string, std::string>& constants,
                 const DTypePools& pools, const FeatureRow& features) {
  const auto pool_it = pools.find(features.dtype);
  require(pool_it != pools.end(), "query dtype has no training-derived pool");
  if (policy.kind == "model") {
    require(model != nullptr, "model policy lacks fitted model");
    auto ranking =
        model
            ->rank_with_scoring(features.features, pool_it->second,
                                policy.model.knn_k,
                                policy.model.ridge_weight)
            .ranking;
    require(!ranking.empty(), "model returned an empty conservative ranking");
    return {ranking.front().candidate_id,
            ranking.front().predicted_log_cost, std::move(ranking)};
  }
  std::optional<std::string> selected;
  if (policy.kind == "constant") {
    const auto constant_it = constants.find(features.dtype);
    require(constant_it != constants.end(),
            "constant policy lacks query dtype");
    selected = constant_it->second;
  } else if (policy.kind == "fixed") {
    if (std::find(pool_it->second.begin(), pool_it->second.end(),
                  policy.fixed_candidate) != pool_it->second.end()) {
      selected = policy.fixed_candidate;
    }
  } else {
    fail("unknown policy kind " + policy.kind);
  }
  return {selected, std::nullopt, {}};
}

double oracle_minimum(const CostRow& row, const Task& task,
                      const Protocol& protocol) {
  double result = std::numeric_limits<double>::infinity();
  for (const auto& candidate : row.domains.at(task.domain)) {
    if (usable(candidate, task, protocol)) {
      result = std::min(
          result, *selected_cost(candidate.costs, task.objective));
    }
  }
  require(std::isfinite(result), "validation row has no allowed oracle cost");
  return result;
}

double observed_loss(const CostRow& row, const Task& task,
                     const Protocol& protocol,
                     const std::optional<std::string>& selected,
                     bool& supported) {
  const double oracle = oracle_minimum(row, task, protocol);
  if (!selected) {
    supported = false;
    return std::log(kUnsupportedPenalty);
  }
  const auto* candidate = find_candidate(row, task.domain, *selected);
  if (candidate == nullptr || !usable(*candidate, task, protocol)) {
    supported = false;
    return std::log(kUnsupportedPenalty);
  }
  supported = true;
  return std::log(*selected_cost(candidate->costs, task.objective) / oracle);
}

double dataset_equal_loss(const std::vector<Json>& rows) {
  std::map<std::string, std::pair<double, std::size_t>> datasets;
  for (const auto& row : rows) {
    auto& value = datasets[row.at("dataset_id").get<std::string>()];
    value.first += row.at("loss_log_regret").get<double>();
    ++value.second;
  }
  double total = 0.0;
  for (const auto& [dataset, value] : datasets) {
    (void)dataset;
    total += value.first / static_cast<double>(value.second);
  }
  return total / static_cast<double>(datasets.size());
}

struct CVResult {
  double loss = std::numeric_limits<double>::infinity();
  std::size_t unsupported = 0;
  std::vector<Json> rows;
};

using CVModelCache =
    std::map<std::pair<std::size_t, double>, q::QualityModel>;

CVResult evaluate_policy(
    const std::vector<CostRow>& costs,
    const std::map<std::string, FeatureRow>& features, const Task& task,
    const Protocol& protocol, const std::vector<std::string>& declared,
    const PolicyConfig& policy, bool keep_full_ranking,
    CVModelCache& model_cache) {
  CVResult result;
  for (std::size_t fold = 0; fold < kFoldCount; ++fold) {
    const auto train_indexes = training_indexes(costs, fold, protocol);
    const auto valid_indexes = validation_indexes(costs, fold);
    const auto pools =
        build_dtype_pools(costs, train_indexes, task, protocol, declared);
    const auto constants =
        constant_best(costs, train_indexes, task, protocol, pools);
    std::optional<q::QualityModel> model;
    if (policy.kind == "model") {
      // lambda has no effect on a pure-kNN score. Canonicalizing it to 1.0
      // leaves only three fitted models per fold for the bounded grid.
      const double fit_lambda = policy.model.ridge_weight == 0.0
                                    ? 1.0
                                    : policy.model.ridge_lambda;
      const auto cache_key = std::make_pair(fold, fit_lambda);
      auto cached = model_cache.find(cache_key);
      if (cached == model_cache.end()) {
        auto fit_config = policy.model;
        fit_config.ridge_lambda = fit_lambda;
        fit_config.knn_k = 0;
        fit_config.ridge_weight = 1.0;
        fit_config.candidate_catalog_sha256 =
            d::sha(Json(declared).dump());
        cached = model_cache
                     .emplace(
                         cache_key,
                         q::fit(make_training_rows(costs, train_indexes,
                                                   features, task, protocol),
                                task.domain, task.objective, fit_config))
                     .first;
      }
      model = cached->second;
    }
    for (const auto index : valid_indexes) {
      const auto& row = costs[index];
      const auto feature_it =
          features.find(key(row.dataset_id, row.sample_id));
      require(feature_it != features.end(), "validation row lacks features");
      auto selection = select(policy, model ? &*model : nullptr, constants,
                              pools, feature_it->second);
      bool supported = false;
      const double loss =
          observed_loss(row, task, protocol, selection.candidate_id, supported);
      if (!supported) ++result.unsupported;
      Json output{{"schema", "operator-oracle-quality135-oof-row-v1"},
                  {"task_id", task.id},
                  {"policy_id", policy.id},
                  {"fold", fold},
                  {"sample_id", row.sample_id},
                  {"dataset_id", row.dataset_id},
                  {"dtype", row.dtype},
                  {"source_group", row.source_group},
                  {"legacy_group", row.legacy_group},
                  {"selected_candidate_id",
                   selection.candidate_id ? Json(*selection.candidate_id)
                                          : Json(nullptr)},
                  {"predicted_log_cost",
                   selection.predicted_log_cost
                       ? Json(*selection.predicted_log_cost)
                       : Json(nullptr)},
                  {"selected_supported_on_validation_row", supported},
                  {"loss_log_regret", loss},
                  {"unsupported_penalty", supported ? Json(nullptr)
                                                      : Json(kUnsupportedPenalty)},
                  {"query_costs_used_for_selection", false}};
      if (keep_full_ranking) {
        Json ranking_ids = Json::array();
        Json ranking_scores = Json::array();
        if (policy.kind == "model") {
          for (const auto& item : selection.ranking) {
            ranking_ids.push_back(item.candidate_id);
            ranking_scores.push_back(item.predicted_log_cost);
          }
        } else {
          const auto& pool = pools.at(row.dtype);
          if (selection.candidate_id) {
            ranking_ids.push_back(*selection.candidate_id);
            ranking_scores.push_back(nullptr);
          }
          for (const auto& candidate : pool) {
            if (!selection.candidate_id || candidate != *selection.candidate_id) {
              ranking_ids.push_back(candidate);
              ranking_scores.push_back(nullptr);
            }
          }
        }
        output["ranking_candidate_ids"] = std::move(ranking_ids);
        output["ranking_predicted_log_costs"] = std::move(ranking_scores);
        output["candidate_pool"] = pools.at(row.dtype);
      }
      result.rows.push_back(std::move(output));
    }
  }
  require(result.rows.size() == costs.size(),
          "OOF coverage does not match development rows");
  result.loss = dataset_equal_loss(result.rows);
  return result;
}

std::string jsonl_text(const std::vector<Json>& rows) {
  std::string result;
  for (const auto& row : rows) {
    result += row.dump();
    result.push_back('\n');
  }
  return result;
}

void write_new_text(const fs::path& path, const std::string& value) {
  require(!fs::exists(path), "refusing to overwrite " + path.string());
  d::write_text(path, value);
}

void write_new_json(const fs::path& path, const Json& value) {
  write_new_text(path, value.dump(2) + "\n");
}

Json pools_json(const DTypePools& pools) {
  Json result = Json::object();
  for (const auto& [dtype, candidates] : pools) result[dtype] = candidates;
  return result;
}

Json identity_receipt(const std::vector<std::string>& identities) {
  auto canonical = identities;
  std::sort(canonical.begin(), canonical.end());
  require(std::adjacent_find(canonical.begin(), canonical.end()) ==
              canonical.end(),
          "identity receipt contains duplicates");
  return {{"count", canonical.size()},
          {"canonical_json_sha256", d::sha(Json(canonical).dump())}};
}

Json task_training_audit(
    const std::vector<CostRow>& costs, const Task& task,
    const Protocol& protocol, const std::vector<std::string>& declared) {
  Json folds = Json::array();
  for (std::size_t fold = 0; fold < kFoldCount; ++fold) {
    const auto train = training_indexes(costs, fold, protocol);
    const auto validation = validation_indexes(costs, fold);
    std::vector<std::string> train_samples;
    std::vector<std::string> validation_samples;
    std::set<std::string> train_sources;
    std::set<std::string> train_legacy;
    std::set<std::string> heldout_sources;
    std::set<std::string> heldout_legacy;
    for (const auto index : train) {
      train_samples.push_back(costs[index].dataset_id + ":" +
                              costs[index].sample_id);
      train_sources.insert(costs[index].source_group);
      train_legacy.insert(costs[index].legacy_group);
    }
    for (const auto index : validation) {
      validation_samples.push_back(costs[index].dataset_id + ":" +
                                   costs[index].sample_id);
      heldout_sources.insert(costs[index].source_group);
      heldout_legacy.insert(costs[index].legacy_group);
    }
    for (const auto& row : costs) {
      if (heldout_sources.contains(row.source_group))
        require(row.dev_fold == fold,
                "heldout source group leaks across official folds");
    }
    for (const auto index : train) {
      require(!heldout_legacy.contains(costs[index].legacy_group) &&
                  !protocol.confirm_legacy_groups.contains(
                      costs[index].legacy_group),
              "legacy alias leaked into fold training");
    }
    folds.push_back(
        {{"fold", fold},
         {"training_sample_ids", identity_receipt(train_samples)},
         {"validation_sample_ids", identity_receipt(validation_samples)},
         {"training_source_groups",
          std::vector<std::string>(train_sources.begin(), train_sources.end())},
         {"training_legacy_groups",
          std::vector<std::string>(train_legacy.begin(), train_legacy.end())},
         {"excluded_heldout_source_groups",
          std::vector<std::string>(heldout_sources.begin(),
                                   heldout_sources.end())},
         {"excluded_heldout_legacy_groups",
          std::vector<std::string>(heldout_legacy.begin(),
                                   heldout_legacy.end())},
         {"excluded_confirm_legacy_groups",
          std::vector<std::string>(protocol.confirm_legacy_groups.begin(),
                                   protocol.confirm_legacy_groups.end())},
         {"dtype_candidate_pools",
          pools_json(build_dtype_pools(costs, train, task, protocol,
                                       declared))}});
  }
  const auto final_train = training_indexes(costs, std::nullopt, protocol);
  std::vector<std::string> final_samples;
  for (const auto index : final_train)
    final_samples.push_back(costs[index].dataset_id + ":" +
                            costs[index].sample_id);
  return {{"task_id", task.id},
          {"domain", q::domain_name(task.domain)},
          {"objective", q::objective_name(task.objective)},
          {"folds", std::move(folds)},
          {"final_training_sample_ids", identity_receipt(final_samples)},
          {"final_dtype_candidate_pools",
           pools_json(build_dtype_pools(costs, final_train, task, protocol,
                                        declared))}};
}

void tune(const fs::path& costs_path, const fs::path& features_path,
          const fs::path& protocol_path, const fs::path& output_dir) {
  require(!fs::exists(output_dir),
          "output directory already exists: " + output_dir.string());
  require(fs::create_directories(output_dir / "cv_configs") &&
              fs::create_directories(output_dir / "models"),
          "cannot create output directory");
  std::cout << "[quality] loading protocol/features/development costs\n";
  const auto protocol = parse_protocol(protocol_path);
  const auto features_receipt = d::read_json(protocol.features_receipt_path);
  const auto features = load_features(features_path, &features_receipt,
                                      protocol.require_frozen_current135);
  std::map<q::Domain, std::vector<std::string>> declared;
  const auto costs = load_costs(costs_path, protocol, declared);
  for (const auto& row : costs) {
    require(features.contains(key(row.dataset_id, row.sample_id)),
            "development cost row lacks feature row");
  }

  Json task_summaries = Json::array();
  Json index_tasks = Json::array();
  Json training_audit_tasks = Json::array();
  std::string chosen_ranks;
  for (const auto& task : kTasks) {
    std::cout << "[quality] tuning " << task.id << std::endl;
    const auto policies = policy_grid(task);
    training_audit_tasks.push_back(task_training_audit(
        costs, task, protocol, declared.at(task.domain)));
    Json policy_summaries = Json::array();
    std::optional<PolicyConfig> best_policy;
    double best_loss = std::numeric_limits<double>::infinity();
    CVModelCache model_cache;
    for (const auto& policy : policies) {
      std::cout << "[quality]   CV " << policy.id << std::endl;
      auto result = evaluate_policy(costs, features, task, protocol,
                                    declared.at(task.domain), policy, false,
                                    model_cache);
      const auto filename = task.id + "__" + policy.id + ".jsonl";
      write_new_text(output_dir / "cv_configs" / filename,
                     jsonl_text(result.rows));
      policy_summaries.push_back(
          {{"policy", policy_json(policy)},
           {"dataset_equal_mean_log_regret", result.loss},
           {"unsupported_validation_rows", result.unsupported},
           {"oof_rows", result.rows.size()},
           {"oof_path", "cv_configs/" + filename},
           {"oof_sha256", d::file_sha(output_dir / "cv_configs" / filename)}});
      if (!best_policy || result.loss < best_loss ||
          (result.loss == best_loss && policy.id < best_policy->id)) {
        best_policy = policy;
        best_loss = result.loss;
      }
    }
    require(best_policy.has_value(), "no policy selected for " + task.id);
    std::cout << "[quality]   selected " << best_policy->id << " loss="
              << best_loss << std::endl;
    auto chosen = evaluate_policy(costs, features, task, protocol,
                                  declared.at(task.domain), *best_policy, true,
                                  model_cache);
    chosen_ranks += jsonl_text(chosen.rows);

    const auto final_indexes = training_indexes(costs, std::nullopt, protocol);
    const auto final_pools = build_dtype_pools(
        costs, final_indexes, task, protocol, declared.at(task.domain));
    const auto final_constants =
        constant_best(costs, final_indexes, task, protocol, final_pools);
    q::FitConfig final_config;
    if (best_policy->kind == "model") {
      final_config = best_policy->model;
    } else {
      final_config.ridge_lambda = 1.0;
      final_config.knn_k = 0;
      final_config.ridge_weight = 1.0;
      final_config.center_log_cost_per_row = true;
      final_config.feature_names = expected_feature_names();
    }
    final_config.candidate_catalog_sha256 =
        d::sha(Json(declared.at(task.domain)).dump());
    auto core = q::fit(make_training_rows(costs, final_indexes, features, task,
                                          protocol),
                       task.domain, task.objective, final_config);
    std::vector<std::string> train_samples;
    std::set<std::string> train_datasets;
    std::set<std::string> train_sources;
    std::set<std::string> train_legacy;
    for (const auto index : final_indexes) {
      train_samples.push_back(costs[index].dataset_id + ":" +
                              costs[index].sample_id);
      train_datasets.insert(costs[index].dataset_id);
      train_sources.insert(costs[index].source_group);
      train_legacy.insert(costs[index].legacy_group);
      require(!protocol.confirm_legacy_groups.contains(
                  costs[index].legacy_group),
              "final training retained a confirm legacy alias");
    }
    std::sort(train_samples.begin(), train_samples.end());
    Json constants_json = Json::object();
    for (const auto& [dtype, candidate] : final_constants)
      constants_json[dtype] = candidate;
    Json envelope{
        {"schema", "operator-oracle-quality135-model-envelope-v1"},
        {"task_id", task.id},
        {"core", core.to_json()},
        {"driver_metadata",
         {{"selected_policy", policy_json(*best_policy)},
          {"dtype_candidate_pools", pools_json(final_pools)},
          {"constant_best_per_dtype", constants_json},
          {"declared_candidate_ids", declared.at(task.domain)},
          {"algorithm_time_contract",
           is_time_task(task) ? Json(protocol.algorithm_time_contract)
                              : Json(nullptr)},
          {"allowed_time_candidate_ids",
           std::vector<std::string>(protocol.allowed_time_candidate_ids.begin(),
                                    protocol.allowed_time_candidate_ids.end())},
          {"training_sample_ids", train_samples},
          {"training_dataset_ids",
           std::vector<std::string>(train_datasets.begin(),
                                    train_datasets.end())},
          {"training_source_groups",
           std::vector<std::string>(train_sources.begin(),
                                    train_sources.end())},
          {"training_legacy_groups",
           std::vector<std::string>(train_legacy.begin(), train_legacy.end())},
          {"excluded_confirm_legacy_groups",
           std::vector<std::string>(protocol.confirm_legacy_groups.begin(),
                                    protocol.confirm_legacy_groups.end())},
          {"query_costs_used_for_selection", false}}}};
    const auto model_filename = task.id + ".json";
    write_new_json(output_dir / "models" / model_filename, envelope);
    const auto model_sha = d::file_sha(output_dir / "models" / model_filename);
    task_summaries.push_back(
        {{"task_id", task.id},
         {"domain", q::domain_name(task.domain)},
         {"objective", q::objective_name(task.objective)},
         {"selected_policy", policy_json(*best_policy)},
         {"selected_development_cv_dataset_equal_mean_log_regret", best_loss},
         {"policies", std::move(policy_summaries)}});
    index_tasks.push_back(
        {{"task_id", task.id},
         {"domain", q::domain_name(task.domain)},
         {"objective", q::objective_name(task.objective)},
         {"selected_policy", policy_json(*best_policy)},
         {"model_path", "models/" + model_filename},
         {"model_sha256", model_sha},
         {"dtype_candidate_pools", pools_json(final_pools)},
         {"declared_candidate_ids", declared.at(task.domain)},
         {"training_sample_ids", train_samples},
         {"training_dataset_ids",
          std::vector<std::string>(train_datasets.begin(),
                                   train_datasets.end())},
         {"training_source_groups",
          std::vector<std::string>(train_sources.begin(),
                                   train_sources.end())},
         {"training_legacy_groups",
          std::vector<std::string>(train_legacy.begin(), train_legacy.end())}});
  }
  write_new_text(output_dir / "CHOSEN_OOF_FULL_RANKS.jsonl", chosen_ranks);
  const Json training_audit{
      {"schema", "operator-oracle-quality135-training-audit-v1"},
      {"status", "complete"},
      {"fold_assignment", "official_frozen_source_group_folds"},
      {"legacy_alias_purge", true},
      {"confirm_labels_read", false},
      {"tasks", std::move(training_audit_tasks)}};
  write_new_json(output_dir / "TRAINING_AUDIT.json", training_audit);
  Json preparation_inputs = Json::object();
  for (const auto& filename :
       {"CANDIDATES.json", "INPUT_METADATA.json", "COMPLETE.json"}) {
    const fs::path candidate = costs_path.parent_path() / filename;
    if (fs::exists(candidate)) {
      preparation_inputs[filename] =
          {{"path", fs::absolute(candidate).string()},
           {"sha256", d::file_sha(candidate)}};
    } else {
      require(!protocol.require_frozen_current135,
              std::string("formal preparation artifact is missing: ") +
                  filename);
    }
  }
  if (protocol.require_frozen_current135) {
    const auto catalog =
        d::read_json(costs_path.parent_path() / "CANDIDATES.json");
    require(catalog.at("algorithm") ==
                Json(declared.at(q::Domain::Algorithm)) &&
                catalog.at("operator_zstd3") ==
                    Json(declared.at(q::Domain::OperatorZstd3)) &&
                catalog.at("observed_pipeline") ==
                    Json(declared.at(q::Domain::ObservedPipeline)) &&
                catalog.value("NNCP", std::string()) == "OFF" &&
                catalog.value("joint_cross_product_measured", true) == false,
            "prepared candidate catalog does not match development rows");
  }
  Json inputs{{"development_costs",
                     {{"path", fs::absolute(costs_path).string()},
                      {"sha256", d::file_sha(costs_path)}}},
                    {"features",
                     {{"path", fs::absolute(features_path).string()},
                      {"sha256", d::file_sha(features_path)},
                      {"receipt_path",
                       fs::absolute(protocol.features_receipt_path).string()},
                      {"receipt_sha256",
                       d::file_sha(protocol.features_receipt_path)}}},
                    {"protocol",
                     {{"path", fs::absolute(protocol_path).string()},
                      {"sha256", d::file_sha(protocol_path)}}},
                    {"preparation", preparation_inputs}};
  const Json cv_summary{
      {"schema", "operator-oracle-quality135-development-cv-v1"},
      {"status", "complete"},
      {"scope", "development_only_model_selection_not_external_evaluation"},
      {"folds", 5},
      {"loss",
       "dataset-equal mean of per-column log(selected / allowed-domain oracle)"},
      {"unsupported_selection_penalty_ratio", kUnsupportedPenalty},
      {"confirm_used_for_tuning", false},
      {"tasks", task_summaries},
      {"inputs", inputs}};
  write_new_json(output_dir / "CV_SUMMARY.json", cv_summary);
  const Json model_index{
      {"schema", "operator-oracle-quality135-model-index-v1"},
      {"status", "locked_development_only"},
      {"feature_names", expected_feature_names()},
      {"feature_names_sha256", d::sha(Json(expected_feature_names()).dump())},
      {"confirm_used_for_fit_or_selection", false},
      {"NNCP", "OFF"},
      {"excluded_algorithm_methods", Json::array({"PPMD8"})},
      {"excluded_action_ids", Json::array({106, 107, 108, 109, 110})},
      {"joint_cross_product_measured", false},
      {"algorithm_time_contract", protocol.algorithm_time_contract},
      {"inputs", inputs},
      {"training_audit_path", "TRAINING_AUDIT.json"},
      {"training_audit_sha256",
       d::file_sha(output_dir / "TRAINING_AUDIT.json")},
      {"tasks", index_tasks}};
  write_new_json(output_dir / "MODEL_INDEX.json", model_index);
  const Json complete{
      {"schema", "operator-oracle-quality135-tune-complete-v1"},
      {"status", "complete"},
      {"model_index_sha256", d::file_sha(output_dir / "MODEL_INDEX.json")},
      {"cv_summary_sha256", d::file_sha(output_dir / "CV_SUMMARY.json")},
      {"chosen_oof_sha256",
       d::file_sha(output_dir / "CHOSEN_OOF_FULL_RANKS.jsonl")},
      {"training_audit_sha256",
       d::file_sha(output_dir / "TRAINING_AUDIT.json")},
      {"development_rows", costs.size()},
      {"confirm_rows_read", 0}};
  write_new_json(output_dir / "COMPLETE.json", complete);
  std::cout << "[quality] tune complete " << output_dir << std::endl;
}

struct LoadedTask {
  Task task;
  PolicyConfig policy;
  q::QualityModel model;
  DTypePools pools;
  std::map<std::string, std::string> constants;
};

LoadedTask load_task(const fs::path& index_path, const Json& item) {
  LoadedTask result;
  result.task.id = text(item, "task_id", "index task");
  result.task.domain = q::parse_domain(text(item, "domain", "index task"));
  result.task.objective =
      q::parse_objective(text(item, "objective", "index task"));
  require(item.contains("selected_policy"),
          "index task lacks selected_policy");
  result.policy =
      parse_policy(item.at("selected_policy"), "index selected policy");
  const fs::path model_path =
      (index_path.parent_path() / text(item, "model_path", "index task"))
          .lexically_normal();
  require(d::file_sha(model_path) ==
              text(item, "model_sha256", "index task"),
          "model SHA256 mismatch");
  const auto envelope = d::read_json(model_path);
  require(envelope.value("schema", std::string()) ==
              "operator-oracle-quality135-model-envelope-v1" &&
              envelope.at("task_id") == result.task.id,
          "model envelope mismatch");
  require(envelope.contains("core") && envelope.at("core").is_object(),
          "model envelope lacks core object");
  const auto& core = envelope.at("core");
  result.model = q::QualityModel::from_json(core);
  require(result.model.domain() == result.task.domain &&
              result.model.objective() == result.task.objective,
          "model core domain/objective disagree with index task");
  require(envelope.contains("driver_metadata") &&
              envelope.at("driver_metadata").is_object(),
          "model envelope lacks driver_metadata object");
  const auto& metadata = envelope.at("driver_metadata");
  require(metadata.contains("selected_policy"),
          "model driver metadata lacks selected_policy");
  const auto envelope_policy = parse_policy(
      metadata.at("selected_policy"), "model-envelope selected policy");
  require(policy_json(envelope_policy) == policy_json(result.policy),
          "index and model-envelope selected policies disagree");

  const auto index_declared = strings(
      item.at("declared_candidate_ids"), "index declared candidate IDs");
  const auto envelope_declared =
      strings(metadata.at("declared_candidate_ids"),
              "model-envelope declared candidate IDs");
  require(index_declared == envelope_declared,
          "index and model-envelope declared candidates disagree");
  require(result.model.candidate_ids() == index_declared,
          "model core candidates disagree with declared candidates");
  const auto catalog_sha = d::sha(Json(index_declared).dump());
  require(core.contains("candidate_catalog_sha256") &&
              core.at("candidate_catalog_sha256").is_string() &&
              core.at("candidate_catalog_sha256") == catalog_sha,
          "model core candidate catalog SHA256 mismatch");
  if (result.policy.kind == "model") {
    require(core.contains("algorithm") && core.at("algorithm").is_object(),
            "model core lacks algorithm metadata");
    const auto& algorithm = core.at("algorithm");
    const std::string expected_label =
        result.policy.model.center_log_cost_per_row
            ? "per_row_centered_log_cost"
            : "log_cost";
    require(finite_number(algorithm, "ridge_lambda", "model algorithm") ==
                result.policy.model.ridge_lambda &&
                unsigned_integer(algorithm, "knn_k", "model algorithm") ==
                    result.policy.model.knn_k &&
                finite_number(algorithm, "ridge_weight", "model algorithm") ==
                    result.policy.model.ridge_weight &&
                text(algorithm, "label", "model algorithm") ==
                    expected_label,
            "model core algorithm disagrees with selected policy");
    result.policy.model.candidate_catalog_sha256 = catalog_sha;
  }

  auto parse_pools = [&](const Json& value, const std::string& context) {
    require(value.is_object() && !value.empty(),
            context + " must be a non-empty object");
    DTypePools pools;
    const std::set<std::string> declared(index_declared.begin(),
                                         index_declared.end());
    for (const auto& entry : value.items()) {
      auto pool = strings(entry.value(), context + " " + entry.key());
      require(!pool.empty(), context + " contains an empty dtype pool");
      require(std::all_of(pool.begin(), pool.end(), [&](const auto& id) {
                return declared.contains(id);
              }),
              context + " contains an undeclared candidate");
      pools.emplace(entry.key(), std::move(pool));
    }
    return pools;
  };
  result.pools = parse_pools(item.at("dtype_candidate_pools"),
                             "index dtype candidate pools");
  const auto envelope_pools =
      parse_pools(metadata.at("dtype_candidate_pools"),
                  "model-envelope dtype candidate pools");
  require(result.pools == envelope_pools,
          "index and model-envelope dtype pools disagree");
  require(metadata.contains("constant_best_per_dtype") &&
              metadata.at("constant_best_per_dtype").is_object(),
          "constant_best_per_dtype must be an object");
  for (const auto& entry : metadata.at("constant_best_per_dtype").items()) {
    require(entry.value().is_string(), "constant baseline must be a string");
    const auto pool = result.pools.find(entry.key());
    require(pool != result.pools.end(),
            "constant baseline dtype lacks a candidate pool");
    auto candidate = entry.value().get<std::string>();
    require(std::find(pool->second.begin(), pool->second.end(), candidate) !=
                pool->second.end(),
            "constant baseline is outside its dtype candidate pool");
    result.constants[entry.key()] = std::move(candidate);
  }
  require(result.constants.size() == result.pools.size(),
          "constant baselines must cover every dtype pool");
  return result;
}

struct RankingArrays {
  Json ids = Json::array();
  Json scores = Json::array();
};

RankingArrays ranking_arrays(const Selection& selection,
                             const std::vector<std::string>& pool,
                             const std::string& policy_kind) {
  RankingArrays ranking;
  if (policy_kind == "model") {
    for (const auto& item : selection.ranking) {
      ranking.ids.push_back(item.candidate_id);
      ranking.scores.push_back(item.predicted_log_cost);
    }
  } else {
    if (selection.candidate_id) {
      ranking.ids.push_back(*selection.candidate_id);
      ranking.scores.push_back(nullptr);
    }
    for (const auto& candidate : pool)
      if (!selection.candidate_id || candidate != *selection.candidate_id) {
        ranking.ids.push_back(candidate);
        ranking.scores.push_back(nullptr);
      }
  }
  return ranking;
}

void predict(const fs::path& index_path, const fs::path& features_path,
             const fs::path& output_path) {
  const fs::path receipt_path(output_path.string() + ".receipt.json");
  require(!fs::exists(output_path),
          "refusing to overwrite prediction output " + output_path.string());
  require(!fs::exists(receipt_path),
          "refusing to overwrite prediction receipt " +
              receipt_path.string());
  std::cout << "[quality] loading locked models and features\n";
  const auto total_wall_start = d::wall_ns();
  const auto total_cpu_start = d::cpu_ns();
  const auto model_wall_start = d::wall_ns();
  const auto model_cpu_start = d::cpu_ns();
  const auto index = d::read_json(index_path);
  require(index.value("schema", std::string()) ==
              "operator-oracle-quality135-model-index-v1" &&
              index.value("confirm_used_for_fit_or_selection", true) == false,
          "unexpected or unsafe model index");
  require(index.at("feature_names") == Json(expected_feature_names()),
          "model index feature names mismatch");
  std::vector<LoadedTask> tasks;
  for (const auto& item : index.at("tasks"))
    tasks.push_back(load_task(index_path, item));
  require(tasks.size() == kTasks.size(), "model index must contain six tasks");
  const auto model_index_sha256 = d::file_sha(index_path);
  const auto model_cpu_ns = d::cpu_ns() - model_cpu_start;
  const auto model_wall_ns = d::wall_ns() - model_wall_start;
  const auto features_wall_start = d::wall_ns();
  const auto features_cpu_start = d::cpu_ns();
  const auto features = load_features(features_path);
  const auto query_features_sha256 = d::file_sha(features_path);
  const auto features_cpu_ns = d::cpu_ns() - features_cpu_start;
  const auto features_wall_ns = d::wall_ns() - features_wall_start;
  const auto prediction_wall_start = d::wall_ns();
  const auto prediction_cpu_start = d::cpu_ns();
  std::string output;
  for (const auto& [identity, feature] : features) {
    (void)identity;
    for (const auto& task : tasks) {
      const auto pool_it = task.pools.find(feature.dtype);
      require(pool_it != task.pools.end(),
              "prediction dtype absent from locked model pool");
      // One warm call followed by five measured calls. Model setup and feature
      // extraction happen outside this boundary.
      auto selection = select(task.policy, &task.model, task.constants,
                              task.pools, feature);
      std::array<std::uint64_t, 5> timing_samples{};
      for (std::size_t repeat = 0; repeat < timing_samples.size(); ++repeat) {
        const auto start = d::cpu_ns();
        selection = select(task.policy, &task.model, task.constants,
                           task.pools, feature);
        timing_samples[repeat] = d::cpu_ns() - start;
      }
      auto sorted_timings = timing_samples;
      std::sort(sorted_timings.begin(), sorted_timings.end());
      const Json timing_samples_json = timing_samples;
      auto ranking =
          ranking_arrays(selection, pool_it->second, task.policy.kind);
      Json row{{"schema", "operator-oracle-quality135-prediction-v1"},
               {"dataset_id", feature.dataset_id},
               {"sample_id", feature.sample_id},
               {"column_id", feature.sample_id},
               {"dtype", feature.dtype},
               {"task_id", task.task.id},
               {"domain", q::domain_name(task.task.domain)},
               {"objective", q::objective_name(task.task.objective)},
               {"selected_policy", policy_json(task.policy)},
               {"selected_candidate_id",
                selection.candidate_id ? Json(*selection.candidate_id)
                                       : Json(nullptr)},
               {"candidate_pool", pool_it->second},
               {"ranking_candidate_ids", std::move(ranking.ids)},
               {"ranking_predicted_log_costs", std::move(ranking.scores)},
               {"selection_cpu_ns_samples", timing_samples_json},
               {"selection_cpu_ns_median", sorted_timings[2]},
               {"selection_cpu_repeats", 5},
               {"selection_warm_repeats", 1},
               {"timing_excludes_model_setup", true},
               {"timing_excludes_feature_extraction", true},
               {"query_costs_read", false},
               {"query_features_sha256", query_features_sha256},
               {"model_index_sha256", model_index_sha256}};
      output += row.dump();
      output.push_back('\n');
    }
  }
  const auto prediction_cpu_ns = d::cpu_ns() - prediction_cpu_start;
  const auto prediction_wall_ns = d::wall_ns() - prediction_wall_start;
  const auto write_wall_start = d::wall_ns();
  const auto write_cpu_start = d::cpu_ns();
  write_new_text(output_path, output);
  const auto output_write_cpu_ns = d::cpu_ns() - write_cpu_start;
  const auto output_write_wall_ns = d::wall_ns() - write_wall_start;
  const auto total_cpu_ns = d::cpu_ns() - total_cpu_start;
  const auto total_wall_ns = d::wall_ns() - total_wall_start;
  const Json receipt{
      {"schema", "operator-oracle-quality135-prediction-receipt-v1"},
      {"status", "complete"},
      {"output",
       {{"path", fs::absolute(output_path).string()},
        {"sha256", d::file_sha(output_path)},
        {"row_count", features.size() * tasks.size()}}},
      {"inputs",
       {{"model_index",
         {{"path", fs::absolute(index_path).string()},
          {"sha256", model_index_sha256}}},
        {"features",
         {{"path", fs::absolute(features_path).string()},
          {"sha256", query_features_sha256}}}}},
      {"task_count", tasks.size()},
      {"feature_row_count", features.size()},
      {"query_costs_read", false},
      {"timing",
       {{"clock", "CLOCK_THREAD_CPUTIME_ID_and_steady_clock"},
        {"model_load_parse_prepare_cpu_ns", model_cpu_ns},
        {"model_load_parse_prepare_wall_ns", model_wall_ns},
        {"features_load_parse_validate_cpu_ns", features_cpu_ns},
        {"features_load_parse_validate_wall_ns", features_wall_ns},
        {"prediction_build_cpu_ns", prediction_cpu_ns},
        {"prediction_build_wall_ns", prediction_wall_ns},
        {"output_write_cpu_ns", output_write_cpu_ns},
        {"output_write_wall_ns", output_write_wall_ns},
        {"total_through_output_write_cpu_ns", total_cpu_ns},
        {"total_through_output_write_wall_ns", total_wall_ns},
        {"per_query_selection_includes_warmup", false},
        {"per_query_selection_repeats", 5},
        {"per_query_selection_warm_repeats", 1},
        {"codec_runtime_included", false}}}};
  write_new_json(receipt_path, receipt);
  std::cout << "[quality] predict complete rows="
            << features.size() * tasks.size() << " receipt=" << receipt_path
            << std::endl;
}

struct SelftestReloadCase {
  std::string task_id;
  PolicyConfig policy;
  q::QualityModel model;
};

std::vector<q::TrainingRow> selftest_reload_rows() {
  std::vector<q::TrainingRow> rows;
  for (std::size_t i = 0; i < 32; ++i) {
    std::vector<double> features(expected_feature_names().size(), 0.0);
    features[0] = static_cast<double>(i);
    features[1] = static_cast<double>((i * 7) % 11);
    features[2] = static_cast<double>((i * i) % 13);
    const bool local_left = (i < 8) || (i >= 20 && i < 24);
    const double left = local_left ? 1.0 : 9.0;
    const double right = local_left ? 8.0 : 1.25;
    rows.push_back(
        {"reload-sample-" + std::to_string(i),
         "reload-dataset-" + std::to_string(i % 4),
         "reload-source-" + std::to_string(i % 7),
         1.0,
         std::move(features),
         {q::CandidateCosts{"LZ4", true, left, left * 1.7, left * 1.2,
                            left * 1.4},
          q::CandidateCosts{"Zstd", true, right, right * 1.1,
                            right * 1.8, right * 1.35}}});
  }
  return rows;
}

void require_exact_ranking(const q::RankingResult& expected,
                           const q::RankingResult& actual,
                           const std::string& context) {
  require(actual.domain == expected.domain &&
              actual.objective == expected.objective &&
              actual.unscored_candidates == expected.unscored_candidates &&
              actual.ranking.size() == expected.ranking.size(),
          context + " ranking shape/metadata mismatch");
  for (std::size_t i = 0; i < expected.ranking.size(); ++i) {
    const auto& left = expected.ranking[i];
    const auto& right = actual.ranking[i];
    require(right.candidate_id == left.candidate_id &&
                right.predicted_log_cost == left.predicted_log_cost &&
                right.ridge_log_cost == left.ridge_log_cost &&
                right.knn_log_cost == left.knn_log_cost &&
                right.support_rows == left.support_rows &&
                right.neighbor_count == left.neighbor_count,
            context + " ranking score/order mismatch");
  }
}

std::vector<SelftestReloadCase> install_selftest_reload_cases(
    const fs::path& model_index_path) {
  auto index = d::read_json(model_index_path);
  const auto rows = selftest_reload_rows();
  std::vector<SelftestReloadCase> cases;
  auto install = [&](const std::string& task_id, double lambda,
                     std::size_t knn_k, double ridge_weight,
                     bool center_log_cost_per_row) {
    auto item = std::find_if(
        index.at("tasks").begin(), index.at("tasks").end(),
        [&](const Json& candidate) {
          return candidate.at("task_id") == task_id;
        });
    require(item != index.at("tasks").end(),
            "selftest reload task missing from model index");
    PolicyConfig policy;
    policy.id = "reload_" + task_id;
    policy.kind = "model";
    policy.model.ridge_lambda = lambda;
    policy.model.knn_k = knn_k;
    policy.model.ridge_weight = ridge_weight;
    policy.model.center_log_cost_per_row = center_log_cost_per_row;
    policy.model.feature_names = expected_feature_names();
    const auto declared = strings(item->at("declared_candidate_ids"),
                                  "selftest declared candidates");
    policy.model.candidate_catalog_sha256 =
        d::sha(Json(declared).dump());
    const auto task = std::find_if(
        kTasks.begin(), kTasks.end(),
        [&](const Task& candidate) { return candidate.id == task_id; });
    require(task != kTasks.end(), "selftest reload task definition missing");
    auto model = q::fit(rows, task->domain, task->objective, policy.model);
    const fs::path model_path =
        (model_index_path.parent_path() /
         text(*item, "model_path", "selftest index task"))
            .lexically_normal();
    auto envelope = d::read_json(model_path);
    envelope["core"] = model.to_json();
    envelope["driver_metadata"]["selected_policy"] = policy_json(policy);
    d::write_json(model_path, envelope);
    (*item)["selected_policy"] = policy_json(policy);
    (*item)["model_sha256"] = d::file_sha(model_path);
    cases.push_back({task_id, policy, std::move(model)});
  };
  install("algorithm_space", 0.1, 0, 1.0, false);
  install("algorithm_encode", 10.0, 24, 0.5, true);
  d::write_json(model_index_path, index);
  return cases;
}

void verify_selftest_reload_cases(
    const fs::path& model_index_path,
    const std::map<std::string, FeatureRow>& features,
    const std::vector<SelftestReloadCase>& cases) {
  const auto index = d::read_json(model_index_path);
  for (const auto& test_case : cases) {
    const auto item = std::find_if(
        index.at("tasks").begin(), index.at("tasks").end(),
        [&](const Json& candidate) {
          return candidate.at("task_id") == test_case.task_id;
        });
    require(item != index.at("tasks").end(),
            "selftest reload task missing after rewrite");
    const auto loaded = load_task(model_index_path, *item);
    require(policy_json(loaded.policy) == policy_json(test_case.policy),
            "save/load must preserve exact selected-policy metadata");
    for (const auto& [identity, feature] : features) {
      (void)identity;
      const auto pool = loaded.pools.at(feature.dtype);
      const auto expected = test_case.model.rank_with_scoring(
          feature.features, pool, test_case.policy.model.knn_k,
          test_case.policy.model.ridge_weight);
      const auto actual = loaded.model.rank_with_scoring(
          feature.features, pool, loaded.policy.model.knn_k,
          loaded.policy.model.ridge_weight);
      require_exact_ranking(expected, actual,
                            "save/load " + test_case.task_id);
    }
  }
}

void selftest() {
  const fs::path root =
      fs::temp_directory_path() /
      ("operator_oracle_quality135_v1_selftest_" +
       std::to_string(static_cast<unsigned long long>(::getpid())));
  require(!fs::exists(root), "selftest temporary path unexpectedly exists");
  fs::create_directories(root);
  try {
    const auto features_path = root / "FEATURES.jsonl";
    const auto receipt_path = root / "FEATURES_RECEIPT.json";
    const auto costs_path = root / "DEVELOPMENT_COSTS.jsonl";
    const auto protocol_path = root / "PROTOCOL.json";
    const auto output_dir = root / "tuned";
    const auto predictions_path = root / "PREDICTIONS.jsonl";
    const std::string columns_sha(64, 'b');
    const std::string oracle_sha(64, 'c');
    const std::string names_sha = d::sha(Json(expected_feature_names()).dump());
    std::string feature_text;
    std::string cost_text;
    for (std::size_t i = 0; i < kFoldCount; ++i) {
      std::vector<double> vector(expected_feature_names().size(), 0.0);
      vector[0] = static_cast<double>(i);
      const std::string sample(64, static_cast<char>('1' + i));
      Json feature{{"schema", "operator-oracle-quality135-feature-row-v1"},
                   {"dataset_id", "dataset-" + std::to_string(i)},
                   {"column_id", sample},
                   {"input_dtype", "float64"},
                   {"features", vector},
                   {"feature_contract", "selftest-24-v1"},
                   {"feature_names_sha256", names_sha},
                   {"columns_sha256", columns_sha},
                   {"oracle_reference_jsonl_sha256", oracle_sha},
                   {"oracle_labels_read", false}};
      feature_text += feature.dump() + "\n";
      auto measured = [&](const std::string& id, double cost) {
        return Json{{"candidate_id", id},
                    {"eligible", true},
                    {"space_cost", cost},
                    {"encode_cost", cost * 2.0},
                    {"decode_cost", cost * 3.0},
                    {"balanced_cost", cost * 2.0},
                    {"timing_contract", kRequiredTimeContract}};
      };
      const double left = i < 3 ? 1.0 : 2.0;
      const double right = i < 3 ? 2.0 : 1.0;
      Json cost{{"sample_id", sample},
                {"dataset_id", "dataset-" + std::to_string(i)},
                {"dtype", "float64"},
                {"source_group", "source-" + std::to_string(i)},
                {"legacy_group", "legacy-" + std::to_string(i)},
                {"split", "development"},
                {"dev_fold", i},
                {"domains",
                 {{"algorithm",
                   Json::array({measured("LZ4", left),
                                measured("Zstd", right)})},
                  {"operator_zstd3",
                   Json::array({measured("A001", right),
                                measured("A002", left)})},
                  {"observed_pipeline",
                   Json::array({measured("algorithm:LZ4", left),
                                measured("operator_zstd3:A001", right)})}}}};
      cost_text += cost.dump() + "\n";
    }
    d::write_text(features_path, feature_text);
    d::write_json(
        receipt_path,
        {{"schema", "operator-oracle-quality135-features-receipt-v1"},
         {"status", "complete"},
         {"output_path", features_path.string()},
         {"output_sha256", d::file_sha(features_path)},
         {"columns_sha256", columns_sha},
         {"oracle_reference_jsonl_sha256", oracle_sha},
         {"feature_names", expected_feature_names()},
         {"feature_names_sha256", names_sha},
         {"column_count", kFoldCount},
         {"dataset_count", kFoldCount}});
    d::write_text(costs_path, cost_text);
    d::write_json(
        protocol_path,
        {{"schema", "operator-oracle-quality135-protocol-v1"},
         {"dev_fold_count", kFoldCount},
         {"confirm_legacy_groups", Json::array()},
         {"algorithm_time_contract", kRequiredTimeContract},
         {"allowed_time_candidate_ids", Json::array({"LZ4", "Zstd"})},
         {"features_receipt_path", receipt_path.string()},
         {"require_frozen_current135", false}});
    tune(costs_path, features_path, protocol_path, output_dir);
    const auto reload_cases =
        install_selftest_reload_cases(output_dir / "MODEL_INDEX.json");
    const auto selftest_features = load_features(features_path);
    verify_selftest_reload_cases(output_dir / "MODEL_INDEX.json",
                                 selftest_features, reload_cases);
    predict(output_dir / "MODEL_INDEX.json", features_path, predictions_path);
    require(fs::exists(predictions_path.string() + ".receipt.json"),
            "selftest prediction receipt is missing");
    const auto predictions = read_jsonl(predictions_path);
    require(predictions.size() == kFoldCount * kTasks.size(),
            "selftest prediction coverage mismatch");
    for (const auto& prediction : predictions) {
      require(prediction.at("query_costs_read") == false &&
                  prediction.at("selection_cpu_ns_samples").size() == 5 &&
                  prediction.at("ranking_candidate_ids").is_array() &&
                  !prediction.at("ranking_candidate_ids").empty() &&
                  prediction.at("ranking_candidate_ids").size() ==
                      prediction.at("ranking_predicted_log_costs").size(),
              "selftest prediction contract mismatch");
      const auto reload = std::find_if(
          reload_cases.begin(), reload_cases.end(),
          [&](const SelftestReloadCase& candidate) {
            return prediction.at("task_id") == candidate.task_id;
          });
      if (reload != reload_cases.end()) {
        require(prediction.at("selected_policy") ==
                    policy_json(reload->policy),
                "predict must emit the exact serialized selected policy");
        const auto& feature = selftest_features.at(
            key(prediction.at("dataset_id").get<std::string>(),
                prediction.at("sample_id").get<std::string>()));
        const auto expected = reload->model.rank_with_scoring(
            feature.features,
            prediction.at("candidate_pool").get<std::vector<std::string>>(),
            reload->policy.model.knn_k,
            reload->policy.model.ridge_weight);
        Json expected_ids = Json::array();
        Json expected_scores = Json::array();
        for (const auto& ranked : expected.ranking) {
          expected_ids.push_back(ranked.candidate_id);
          expected_scores.push_back(ranked.predicted_log_cost);
        }
        require(prediction.at("ranking_candidate_ids") == expected_ids &&
                    prediction.at("ranking_predicted_log_costs") ==
                        expected_scores,
                "predict save/load ranking must preserve exact IDs/scores");
      }
    }
    require(d::read_json(output_dir / "COMPLETE.json").at("confirm_rows_read") ==
                0,
            "selftest tune read confirm rows");
    const auto prediction_receipt =
        d::read_json(predictions_path.string() + ".receipt.json");
    require(prediction_receipt.at("schema") ==
                    "operator-oracle-quality135-prediction-receipt-v1" &&
                prediction_receipt.at("output").at("sha256") ==
                    d::file_sha(predictions_path) &&
                prediction_receipt.at("query_costs_read") == false,
            "selftest prediction receipt mismatch");
    fs::remove_all(root);
    std::cout << "quality_experiment selftest: PASS\n";
  } catch (...) {
    std::cerr << "quality_experiment selftest artifacts: " << root << '\n';
    throw;
  }
}

void usage() {
  std::cerr
      << "usage:\n"
      << "  quality_experiment tune DEV_COSTS_JSONL FEATURES_JSONL "
         "PROTOCOL_JSON OUTPUT_DIR\n"
      << "  quality_experiment predict MODEL_INDEX_JSON FEATURES_JSONL "
         "OUTPUT_JSONL\n"
      << "  quality_experiment selftest\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 6 && std::string_view(argv[1]) == "tune") {
      tune(argv[2], argv[3], argv[4], argv[5]);
      return 0;
    }
    if (argc == 5 && std::string_view(argv[1]) == "predict") {
      predict(argv[2], argv[3], argv[4]);
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "selftest") {
      selftest();
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "quality_experiment: FAIL: " << error.what() << '\n';
    return 1;
  }
}
