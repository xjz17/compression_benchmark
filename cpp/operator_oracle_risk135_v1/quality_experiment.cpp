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
#include <tuple>
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
    "benchmark_host-current135-valid-value-w1-r3-t1-v1";

const std::vector<std::string>& prefix_feature_names() {
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

const std::vector<std::string>& multi_feature_names() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> result;
    for (const auto* aggregate : {"prefix", "segment_mean", "segment_range",
                                  "segment_tail"}) {
      for (const auto& name : prefix_feature_names())
        result.push_back(std::string(aggregate) + "." + name);
    }
    result.push_back("log1p_column_valid_count");
    result.push_back("log1p_column_row_count");
    return result;
  }();
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

std::vector<double> finite_features(const Json& value, std::size_t expected,
                                    const std::string& context) {
  require(value.is_array(), context + " must be an array");
  auto result = value.get<std::vector<double>>();
  require(result.size() == expected,
          context + " has the wrong feature count");
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

struct FeatureSet {
  std::string variant;
  std::vector<std::string> names;
  fs::path path;
  fs::path receipt_path;
  std::map<std::string, FeatureRow> rows;
};

using FeatureSets = std::map<std::string, FeatureSet>;

enum class WeightMode { DatasetEqual, SqrtBytes, MixedBytes };

std::string_view weight_mode_name(WeightMode mode) {
  switch (mode) {
    case WeightMode::DatasetEqual: return "dataset_equal";
    case WeightMode::SqrtBytes: return "sqrt_bytes";
    case WeightMode::MixedBytes: return "mixed_bytes";
  }
  return "unknown";
}

WeightMode parse_weight_mode(std::string_view name) {
  if (name == "dataset_equal") return WeightMode::DatasetEqual;
  if (name == "sqrt_bytes") return WeightMode::SqrtBytes;
  if (name == "mixed_bytes") return WeightMode::MixedBytes;
  fail("unsupported weight mode " + std::string(name));
}

struct RawCandidate {
  q::CandidateCosts costs;
  std::optional<std::uint64_t> compressed_bytes;
  std::string timing_contract;
};

struct CostRow {
  std::string sample_id;
  std::string dataset_id;
  std::string dtype;
  std::string source_group;
  std::string legacy_group;
  std::size_t dev_fold = 0;
  std::uint64_t original_numeric_bytes = 0;
  std::map<q::Domain, std::vector<RawCandidate>> domains;
};

struct Task {
  std::string id;
  q::Domain domain;
  q::Objective objective;
};

const std::array<Task, 3> kTasks = {
    Task{"algorithm_space", q::Domain::Algorithm, q::Objective::Space},
    Task{"operator_zstd3_space", q::Domain::OperatorZstd3,
         q::Objective::Space},
    Task{"observed_pipeline_space", q::Domain::ObservedPipeline,
         q::Objective::Space}};

bool is_time_task(const Task&) { return false; }

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
  std::string development_costs_sha256;
  std::string oracle_reference_sha256;
  fs::path prefix_receipt_path;
  fs::path multi_receipt_path;
};

Protocol parse_protocol(const fs::path& path) {
  const auto value = d::read_json(path);
  require(value.is_object(), "protocol must be an object");
  require(value.value("schema", std::string()) ==
              "operator-oracle-risk135-protocol-v1",
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
  result.algorithm_time_contract = kRequiredTimeContract;
  if (value.contains("allowed_time_candidate_ids")) {
    for (const auto& candidate :
         strings(value.at("allowed_time_candidate_ids"),
                 "protocol.allowed_time_candidate_ids")) {
      result.allowed_time_candidate_ids.insert(candidate);
    }
  }
  result.require_frozen_current135 =
      value.value("require_frozen_current135", true);
  require(value.contains("source_bindings") &&
              value.at("source_bindings").is_object(),
          "protocol lacks source_bindings");
  result.development_costs_sha256 = text(
      value.at("source_bindings"), "development_costs_sha256",
      "protocol.source_bindings");
  result.oracle_reference_sha256 = text(
      value.at("source_bindings"), "oracle_reference_jsonl_sha256",
      "protocol.source_bindings");
  require(value.contains("features_receipt_paths") &&
              value.at("features_receipt_paths").is_object(),
          "protocol lacks features_receipt_paths");
  auto resolve = [&](const char* name) {
    const fs::path declared(text(value.at("features_receipt_paths"), name,
                                 "protocol.features_receipt_paths"));
    return (declared.is_absolute() ? declared : path.parent_path() / declared)
        .lexically_normal();
  };
  result.prefix_receipt_path = resolve("prefix24");
  result.multi_receipt_path = resolve("multi3");
  return result;
}

std::map<std::string, FeatureRow> load_features(
    const fs::path& path, const std::vector<std::string>& expected_names,
    bool require_frozen = false) {
  const fs::path receipt_path = path.parent_path() / "FEATURES_RECEIPT.json";
  const auto receipt = d::read_json(receipt_path);
  std::map<std::string, FeatureRow> result;
  std::optional<std::string> contract;
  std::optional<std::string> names_sha;
  std::optional<std::string> columns_sha;
  std::optional<std::string> oracle_sha;
  const auto expected_names_sha = d::sha(Json(expected_names).dump());
  for (const auto& value : read_jsonl(path)) {
    require(value.is_object() &&
                value.value("schema", std::string()) ==
                    "operator-oracle-risk135-feature-row-v1",
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
            "feature_names_sha256 does not match canonical feature names");
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
    FeatureRow row{dataset, sample, dtype, finite_features(
        value.at("features"), expected_names.size(), "features")};
    require(result.emplace(key(dataset, sample), std::move(row)).second,
            "duplicate feature identity");
  }
  {
    require(receipt.is_object() &&
                receipt.value("schema", std::string()) ==
                    "operator-oracle-risk135-features-receipt-v1" &&
                receipt.value("status", std::string()) == "complete",
            "unexpected features receipt");
    require(fs::absolute(path).lexically_normal() ==
                fs::path(text(receipt, "output_path", "features receipt"))
                    .lexically_normal(),
            "features receipt output_path mismatch");
    require(d::file_sha(path) ==
                text(receipt, "output_sha256", "features receipt"),
            "features output SHA256 mismatch");
    require(*columns_sha ==
                text(receipt, "columns_sha256", "features receipt") &&
                *oracle_sha == text(receipt,
                                    "oracle_reference_jsonl_sha256",
                                    "features receipt") &&
                *names_sha == text(receipt, "feature_names_sha256",
                                    "features receipt"),
            "features rows disagree with receipt hashes");
    require(receipt.at("feature_names") == Json(expected_names),
            "features receipt names mismatch");
    require(unsigned_integer(receipt, "column_count", "features receipt") ==
                result.size(),
            "features receipt column count mismatch");
    if (require_frozen) {
      require(result.size() == kExpectedDevelopmentRows &&
                  unsigned_integer(receipt, "dataset_count",
                                   "features receipt") == 127,
              "features receipt must contain exactly frozen development rows");
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
  if (value.contains("compressed_bytes") &&
      !value.at("compressed_bytes").is_null()) {
    result.compressed_bytes =
        unsigned_integer(value, "compressed_bytes", context);
    require(*result.compressed_bytes > 0,
            context + " compressed_bytes must be positive");
  }
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
    row.original_numeric_bytes =
        unsigned_integer(value, "original_numeric_bytes", "cost row");
    require(row.original_numeric_bytes > 0,
            "cost row original_numeric_bytes must be positive");
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
    const Protocol& protocol, WeightMode weight_mode) {
  std::map<std::string, std::size_t> dataset_columns;
  for (const auto index : indexes) ++dataset_columns[costs[index].dataset_id];
  const double dataset_count = static_cast<double>(dataset_columns.size());
  double sum_sqrt_bytes = 0.0;
  double sum_bytes = 0.0;
  for (const auto index : indexes) {
    sum_sqrt_bytes += std::sqrt(
        static_cast<double>(costs[index].original_numeric_bytes));
    sum_bytes += static_cast<double>(costs[index].original_numeric_bytes);
  }
  require(dataset_count > 0.0 && sum_sqrt_bytes > 0.0 && sum_bytes > 0.0,
          "invalid training weight denominators");
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
    const double dataset_weight =
        1.0 / static_cast<double>(dataset_columns.at(source.dataset_id));
    if (weight_mode == WeightMode::DatasetEqual) {
      row.weight = dataset_weight;
    } else if (weight_mode == WeightMode::SqrtBytes) {
      row.weight = dataset_count *
                   std::sqrt(static_cast<double>(source.original_numeric_bytes)) /
                   sum_sqrt_bytes;
    } else {
      row.weight = dataset_count *
                   (0.5 / (dataset_count *
                           static_cast<double>(dataset_columns.at(source.dataset_id))) +
                    0.5 * static_cast<double>(source.original_numeric_bytes) /
                        sum_bytes);
    }
    row.features = feature_it->second.features;
    for (const auto& candidate : source.domains.at(task.domain)) {
      auto copy = candidate.costs;
      if (!usable(candidate, task, protocol)) copy.eligible = false;
      row.candidates.push_back(std::move(copy));
    }
    result.push_back(std::move(row));
  }
  double total_weight = 0.0;
  for (const auto& row : result) total_weight += row.weight;
  require(std::abs(total_weight - dataset_count) <=
              1e-10 * std::max(1.0, dataset_count),
          "training weights must sum to the training dataset count");
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

[[maybe_unused]] std::map<std::string, std::string> constant_best(
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
  q::FitConfig model;
  std::string feature_variant;
  WeightMode weight_mode = WeightMode::DatasetEqual;
  bool frozen_prefix_control = false;
};

std::vector<PolicyConfig> policy_grid(const Task& task) {
  std::vector<PolicyConfig> result;
  for (const auto& [variant, names] :
       std::vector<std::pair<std::string, std::vector<std::string>>>{
           {"prefix24", prefix_feature_names()},
           {"multi3", multi_feature_names()}}) {
    for (const auto weight_mode : {WeightMode::DatasetEqual,
                                   WeightMode::SqrtBytes,
                                   WeightMode::MixedBytes}) {
      for (const auto& scorer :
           std::vector<std::tuple<std::string, double, std::size_t, double>>{
               {"ridge_l1", 1.0, 0, 1.0},
               {"blend_l1_k24", 1.0, 24, 0.5},
               {"ridge_l10", 10.0, 0, 1.0}}) {
        q::FitConfig config;
        config.ridge_lambda = std::get<1>(scorer);
        config.knn_k = std::get<2>(scorer);
        config.ridge_weight = std::get<3>(scorer);
        config.center_log_cost_per_row = true;
        config.feature_names = names;
        const std::string id = variant + "__" +
            std::string(weight_mode_name(weight_mode)) + "__" +
            std::get<0>(scorer);
        result.push_back({id, config, variant, weight_mode, false});
      }
    }
  }
  require(result.size() == 18, "risk grid must contain exactly 18 configs");
  q::FitConfig control;
  control.ridge_lambda = 1.0;
  control.knn_k = task.domain == q::Domain::Algorithm ? 8 : 0;
  control.ridge_weight = task.domain == q::Domain::Algorithm ? 0.5 : 1.0;
  control.center_log_cost_per_row = true;
  control.feature_names = prefix_feature_names();
  const std::string control_id = task.domain == q::Domain::Algorithm
      ? "prefix24__dataset_equal__frozen_blend_l1_k8"
      : "prefix24__dataset_equal__ridge_l1";
  auto existing = std::find_if(result.begin(), result.end(),
      [&](const PolicyConfig& policy) { return policy.id == control_id; });
  if (existing == result.end()) {
    result.push_back({control_id, control, "prefix24",
                      WeightMode::DatasetEqual, true});
  } else {
    existing->frozen_prefix_control = true;
  }
  return result;
}

Json policy_json(const PolicyConfig& policy) {
  Json result{{"id", policy.id}, {"kind", "model"},
              {"feature_variant", policy.feature_variant},
              {"weight_mode", weight_mode_name(policy.weight_mode)},
              {"frozen_prefix_control", policy.frozen_prefix_control},
              {"ridge_lambda", policy.model.ridge_lambda},
              {"knn_k", policy.model.knn_k},
              {"ridge_weight", policy.model.ridge_weight},
              {"center_log_cost_per_row",
               policy.model.center_log_cost_per_row},
              {"feature_names", policy.model.feature_names},
              {"feature_names_sha256",
               d::sha(Json(policy.model.feature_names).dump())}};
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
  require(text(value, "kind", context) == "model",
          context + " only model policies are supported");
  result.feature_variant = text(value, "feature_variant", context);
  result.weight_mode = parse_weight_mode(text(value, "weight_mode", context));
  require(value.contains("frozen_prefix_control") &&
              value.at("frozen_prefix_control").is_boolean(),
          context + " lacks frozen_prefix_control");
  result.frozen_prefix_control =
      value.at("frozen_prefix_control").get<bool>();
  result.model.ridge_lambda = finite_number(value, "ridge_lambda", context);
  result.model.knn_k = unsigned_integer(value, "knn_k", context);
  result.model.ridge_weight = finite_number(value, "ridge_weight", context);
  require(value.contains("center_log_cost_per_row") &&
              value.at("center_log_cost_per_row").is_boolean(),
          context + " missing boolean center_log_cost_per_row");
  result.model.center_log_cost_per_row =
      value.at("center_log_cost_per_row").get<bool>();
  result.model.feature_names =
      strings(value.at("feature_names"), context + " feature_names");
  require(value.at("feature_names_sha256") ==
              d::sha(Json(result.model.feature_names).dump()),
          context + " feature names SHA mismatch");
  require((result.feature_variant == "prefix24" &&
           result.model.feature_names == prefix_feature_names()) ||
              (result.feature_variant == "multi3" &&
               result.model.feature_names == multi_feature_names()),
          context + " feature variant/names mismatch");
  require(result.model.ridge_lambda > 0.0,
          context + " ridge_lambda must be positive");
  require(result.model.ridge_weight >= 0.0 &&
              result.model.ridge_weight <= 1.0,
          context + " ridge_weight must be in [0,1]");
  require(result.model.knn_k != 0 || result.model.ridge_weight == 1.0,
          context + " knn_k=0 requires ridge_weight=1");
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
  (void)constants;
  const auto pool_it = pools.find(features.dtype);
  require(pool_it != pools.end(), "query dtype has no training-derived pool");
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

[[maybe_unused]] double oracle_minimum(const CostRow& row, const Task& task,
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

std::uint64_t oracle_minimum_bytes(const CostRow& row, const Task& task,
                                   const Protocol& protocol) {
  std::uint64_t result = std::numeric_limits<std::uint64_t>::max();
  for (const auto& candidate : row.domains.at(task.domain)) {
    if (usable(candidate, task, protocol) && candidate.compressed_bytes)
      result = std::min(result, *candidate.compressed_bytes);
  }
  require(result != std::numeric_limits<std::uint64_t>::max(),
          "validation row has no allowed byte oracle");
  return result;
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
  double dataset_equal_geo_regret = std::numeric_limits<double>::infinity();
  double worst_source_group_byte_regret =
      std::numeric_limits<double>::infinity();
  double cvar95_absolute_excess_bytes =
      std::numeric_limits<double>::infinity();
  std::uint64_t total_selected_bytes = 0;
  std::uint64_t total_allowed_oracle_bytes = 0;
  std::size_t unsupported = 0;
  std::vector<Json> rows;
};

using CVModelCache = std::map<std::string, q::QualityModel>;

std::uint64_t p90_training_bytes(const std::vector<CostRow>& costs,
                                 const std::vector<std::size_t>& indexes) {
  std::vector<std::uint64_t> values;
  values.reserve(indexes.size());
  for (const auto index : indexes)
    values.push_back(costs[index].original_numeric_bytes);
  std::sort(values.begin(), values.end());
  require(!values.empty(), "cannot derive long-column threshold");
  const auto rank = static_cast<std::size_t>(
      std::ceil(0.9 * static_cast<double>(values.size())));
  return values.at(std::max<std::size_t>(1, rank) - 1);
}

void finish_risk_metrics(CVResult& result) {
  require(!result.rows.empty(), "cannot summarize empty OOF rows");
  result.dataset_equal_geo_regret = std::exp(dataset_equal_loss(result.rows));
  std::map<std::string, std::pair<long double, long double>> groups;
  std::vector<double> excess;
  excess.reserve(result.rows.size());
  for (const auto& row : result.rows) {
    if (!row.at("selected_supported_on_validation_row").get<bool>()) continue;
    const auto selected = row.at("selected_compressed_bytes").get<std::uint64_t>();
    const auto oracle = row.at("allowed_oracle_compressed_bytes")
                            .get<std::uint64_t>();
    result.total_selected_bytes += selected;
    result.total_allowed_oracle_bytes += oracle;
    auto& group = groups[row.at("source_group").get<std::string>()];
    group.first += static_cast<long double>(selected);
    group.second += static_cast<long double>(oracle);
    excess.push_back(static_cast<double>(selected - oracle));
  }
  if (result.unsupported != 0) return;
  result.worst_source_group_byte_regret = 0.0;
  for (const auto& [name, aggregate] : groups) {
    (void)name;
    result.worst_source_group_byte_regret = std::max(
        result.worst_source_group_byte_regret,
        static_cast<double>(aggregate.first / aggregate.second));
  }
  std::sort(excess.begin(), excess.end(), std::greater<double>());
  const std::size_t tail = static_cast<std::size_t>(
      std::ceil(0.05 * static_cast<double>(excess.size())));
  require(tail > 0, "CVaR tail is empty");
  result.cvar95_absolute_excess_bytes =
      std::accumulate(excess.begin(), excess.begin() + tail, 0.0) /
      static_cast<double>(tail);
  require(result.cvar95_absolute_excess_bytes >= 0.0,
          "absolute excess-byte CVaR must be nonnegative");
}

CVResult evaluate_policy(
    const std::vector<CostRow>& costs,
    const FeatureSets& feature_sets, const Task& task,
    const Protocol& protocol, const std::vector<std::string>& declared,
    const PolicyConfig& policy, bool keep_full_ranking,
    CVModelCache& model_cache) {
  CVResult result;
  const auto& feature_set = feature_sets.at(policy.feature_variant);
  for (std::size_t fold = 0; fold < kFoldCount; ++fold) {
    const auto train_indexes = training_indexes(costs, fold, protocol);
    const auto valid_indexes = validation_indexes(costs, fold);
    const auto pools =
        build_dtype_pools(costs, train_indexes, task, protocol, declared);
    const std::map<std::string, std::string> no_constants;
    const auto fit_lambda = policy.model.ridge_lambda;
    const std::string cache_key = std::to_string(fold) + "|" +
        policy.feature_variant + "|" +
        std::string(weight_mode_name(policy.weight_mode)) + "|" +
        std::to_string(fit_lambda);
    auto cached = model_cache.find(cache_key);
    if (cached == model_cache.end()) {
      auto fit_config = policy.model;
      fit_config.knn_k = 0;
      fit_config.ridge_weight = 1.0;
      fit_config.candidate_catalog_sha256 = d::sha(Json(declared).dump());
      cached = model_cache.emplace(
          cache_key,
          q::fit(make_training_rows(costs, train_indexes,
                                    feature_set.rows, task, protocol,
                                    policy.weight_mode),
                 task.domain, task.objective, fit_config)).first;
    }
    const auto threshold = p90_training_bytes(costs, train_indexes);
    for (const auto index : valid_indexes) {
      const auto& row = costs[index];
      const auto feature_it =
          feature_set.rows.find(key(row.dataset_id, row.sample_id));
      require(feature_it != feature_set.rows.end(),
              "validation row lacks features");
      auto selection = select(policy, &cached->second, no_constants,
                              pools, feature_it->second);
      const auto oracle_bytes = oracle_minimum_bytes(row, task, protocol);
      bool supported = selection.candidate_id.has_value();
      const RawCandidate* candidate = nullptr;
      if (supported) {
        candidate = find_candidate(row, task.domain, *selection.candidate_id);
        supported = candidate != nullptr && usable(*candidate, task, protocol) &&
                    candidate->compressed_bytes.has_value();
      }
      const double loss = supported
          ? std::log(static_cast<double>(*candidate->compressed_bytes) /
                     static_cast<double>(oracle_bytes))
          : std::log(kUnsupportedPenalty);
      if (!supported) ++result.unsupported;
      Json output{{"schema", "operator-oracle-risk135-oof-row-v1"},
                  {"task_id", task.id},
                  {"policy_id", policy.id},
                  {"fold", fold},
                  {"sample_id", row.sample_id},
                  {"dataset_id", row.dataset_id},
                  {"dtype", row.dtype},
                  {"source_group", row.source_group},
                  {"legacy_group", row.legacy_group},
                  {"original_numeric_bytes", row.original_numeric_bytes},
                  {"long_column_training_p90_bytes", threshold},
                  {"is_long_column", row.original_numeric_bytes >= threshold},
                  {"selected_candidate_id",
                   selection.candidate_id ? Json(*selection.candidate_id)
                                          : Json(nullptr)},
                  {"predicted_log_cost",
                   selection.predicted_log_cost
                       ? Json(*selection.predicted_log_cost)
                       : Json(nullptr)},
                  {"selected_supported_on_validation_row", supported},
                  {"selected_compressed_bytes",
                   supported ? Json(*candidate->compressed_bytes) : Json(nullptr)},
                  {"allowed_oracle_compressed_bytes", oracle_bytes},
                  {"absolute_excess_bytes",
                   supported ? Json(*candidate->compressed_bytes - oracle_bytes)
                             : Json(nullptr)},
                  {"loss_log_regret", loss},
                  {"unsupported_penalty", supported ? Json(nullptr)
                                                      : Json(kUnsupportedPenalty)},
                  {"query_costs_used_for_selection", false}};
      if (keep_full_ranking) {
        Json ranking_ids = Json::array();
        Json ranking_scores = Json::array();
        for (const auto& item : selection.ranking) {
          ranking_ids.push_back(item.candidate_id);
          ranking_scores.push_back(item.predicted_log_cost);
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
  finish_risk_metrics(result);
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

Json compact_metrics(const CVResult& result) {
  Json folds = Json::array();
  for (std::size_t fold = 0; fold < kFoldCount; ++fold) {
    std::uint64_t selected = 0;
    std::uint64_t oracle = 0;
    std::size_t unsupported = 0;
    std::size_t rows = 0;
    std::size_t long_rows = 0;
    std::optional<std::uint64_t> threshold;
    std::map<std::string, std::pair<double, std::size_t>> datasets;
    std::map<std::string, std::pair<long double, long double>> groups;
    std::vector<double> excess;
    for (const auto& row : result.rows) {
      if (row.at("fold").get<std::size_t>() != fold) continue;
      ++rows;
      if (row.at("is_long_column").get<bool>()) ++long_rows;
      threshold = row.at("long_column_training_p90_bytes")
                      .get<std::uint64_t>();
      if (!row.at("selected_supported_on_validation_row").get<bool>()) {
        ++unsupported;
      } else {
        const auto row_selected =
            row.at("selected_compressed_bytes").get<std::uint64_t>();
        const auto row_oracle = row.at("allowed_oracle_compressed_bytes")
                                    .get<std::uint64_t>();
        selected += row_selected;
        oracle += row_oracle;
        auto& dataset = datasets[row.at("dataset_id").get<std::string>()];
        dataset.first += row.at("loss_log_regret").get<double>();
        ++dataset.second;
        auto& group = groups[row.at("source_group").get<std::string>()];
        group.first += static_cast<long double>(row_selected);
        group.second += static_cast<long double>(row_oracle);
        excess.push_back(static_cast<double>(row_selected - row_oracle));
      }
    }
    Json aggregate_regret = nullptr;
    Json dataset_geo = nullptr;
    Json worst_group = nullptr;
    Json cvar = nullptr;
    if (unsupported == 0) {
      aggregate_regret = static_cast<double>(selected) /
                         static_cast<double>(oracle);
      double dataset_mean_log = 0.0;
      for (const auto& [name, value] : datasets) {
        (void)name;
        dataset_mean_log += value.first / static_cast<double>(value.second);
      }
      dataset_geo = std::exp(dataset_mean_log /
                             static_cast<double>(datasets.size()));
      double worst = 0.0;
      for (const auto& [name, value] : groups) {
        (void)name;
        worst = std::max(worst,
                         static_cast<double>(value.first / value.second));
      }
      worst_group = worst;
      std::sort(excess.begin(), excess.end(), std::greater<double>());
      const auto tail = static_cast<std::size_t>(
          std::ceil(0.05 * static_cast<double>(excess.size())));
      cvar = std::accumulate(excess.begin(), excess.begin() + tail, 0.0) /
             static_cast<double>(tail);
    }
    folds.push_back({{"fold", fold}, {"oof_rows", rows},
                     {"unsupported_rows", unsupported},
                     {"selected_bytes", selected},
                     {"allowed_oracle_bytes", oracle},
                     {"aggregate_byte_regret", aggregate_regret},
                     {"dataset_equal_geo_regret", dataset_geo},
                     {"worst_official_source_group_aggregate_byte_regret",
                      worst_group},
                     {"cvar95_absolute_excess_bytes", cvar},
                     {"training_bytes_p90_threshold", threshold},
                     {"long_column_rows", long_rows}});
  }
  return {{"total_selected_bytes", result.total_selected_bytes},
          {"total_allowed_oracle_bytes", result.total_allowed_oracle_bytes},
          {"aggregate_byte_regret",
           result.unsupported == 0
               ? Json(static_cast<double>(result.total_selected_bytes) /
                      static_cast<double>(result.total_allowed_oracle_bytes))
               : Json(nullptr)},
          {"dataset_equal_geo_regret", result.dataset_equal_geo_regret},
          {"worst_official_source_group_aggregate_byte_regret",
           std::isfinite(result.worst_source_group_byte_regret)
               ? Json(result.worst_source_group_byte_regret) : Json(nullptr)},
          {"cvar95_absolute_excess_bytes",
           std::isfinite(result.cvar95_absolute_excess_bytes)
               ? Json(result.cvar95_absolute_excess_bytes) : Json(nullptr)},
          {"unsupported_validation_rows", result.unsupported},
          {"oof_rows", result.rows.size()},
          {"fold_metrics", std::move(folds)}};
}

bool risk_feasible(const CVResult& candidate, const CVResult& control) {
  constexpr double epsilon = 1e-12;
  return candidate.unsupported == 0 &&
         candidate.dataset_equal_geo_regret <=
             control.dataset_equal_geo_regret * 1.05 + epsilon &&
         candidate.worst_source_group_byte_regret <=
             control.worst_source_group_byte_regret + epsilon &&
         candidate.cvar95_absolute_excess_bytes <=
             control.cvar95_absolute_excess_bytes + epsilon;
}

void tune(const fs::path& costs_path, const fs::path& prefix_features_path,
          const fs::path& multi_features_path,
          const fs::path& protocol_path, const fs::path& output_dir) {
  require(!fs::exists(output_dir),
          "output directory already exists: " + output_dir.string());
  require(fs::create_directories(output_dir / "cv_configs") &&
              fs::create_directories(output_dir / "models"),
          "cannot create output directory");
  std::cout << "[risk] loading protocol/features/development costs\n";
  const auto protocol = parse_protocol(protocol_path);
  require(d::file_sha(costs_path) == protocol.development_costs_sha256,
          "development costs SHA256 disagrees with frozen protocol");
  require((prefix_features_path.parent_path() / "FEATURES_RECEIPT.json")
                  .lexically_normal() == protocol.prefix_receipt_path &&
              (multi_features_path.parent_path() / "FEATURES_RECEIPT.json")
                  .lexically_normal() == protocol.multi_receipt_path,
          "feature receipt paths disagree with frozen protocol");
  FeatureSets feature_sets;
  feature_sets.emplace("prefix24", FeatureSet{
      "prefix24", prefix_feature_names(), prefix_features_path,
      prefix_features_path.parent_path() / "FEATURES_RECEIPT.json",
      load_features(prefix_features_path, prefix_feature_names(),
                    protocol.require_frozen_current135)});
  feature_sets.emplace("multi3", FeatureSet{
      "multi3", multi_feature_names(), multi_features_path,
      multi_features_path.parent_path() / "FEATURES_RECEIPT.json",
      load_features(multi_features_path, multi_feature_names(),
                    protocol.require_frozen_current135)});
  for (const auto& [variant, feature_set] : feature_sets) {
    (void)variant;
    const auto receipt = d::read_json(feature_set.receipt_path);
    require(receipt.at("oracle_reference_jsonl_sha256") ==
                protocol.oracle_reference_sha256,
            "feature receipt oracle binding disagrees with protocol");
  }
  require(feature_sets.at("prefix24").rows.size() ==
              feature_sets.at("multi3").rows.size(),
          "feature variants have different row counts");
  for (const auto& [identity, prefix] : feature_sets.at("prefix24").rows) {
    const auto it = feature_sets.at("multi3").rows.find(identity);
    require(it != feature_sets.at("multi3").rows.end() &&
                it->second.dataset_id == prefix.dataset_id &&
                it->second.sample_id == prefix.sample_id &&
                it->second.dtype == prefix.dtype,
            "feature variants have different identities/dtypes");
  }
  std::map<q::Domain, std::vector<std::string>> declared;
  const auto costs = load_costs(costs_path, protocol, declared);
  for (const auto& row : costs) {
    const auto identity = key(row.dataset_id, row.sample_id);
    require(feature_sets.at("prefix24").rows.contains(identity) &&
                feature_sets.at("multi3").rows.contains(identity),
            "development cost row lacks a feature variant");
  }

  Json task_summaries = Json::array();
  Json index_tasks = Json::array();
  Json training_audit_tasks = Json::array();
  std::string selected_ranks;
  std::string control_ranks;
  for (const auto& task : kTasks) {
    std::cout << "[risk] tuning " << task.id << std::endl;
    const auto policies = policy_grid(task);
    training_audit_tasks.push_back(task_training_audit(
        costs, task, protocol, declared.at(task.domain)));
    struct Evaluated { PolicyConfig policy; CVResult result; };
    std::vector<Evaluated> evaluated;
    CVModelCache model_cache;
    for (const auto& policy : policies) {
      std::cout << "[risk]   CV " << policy.id << std::endl;
      auto result = evaluate_policy(costs, feature_sets, task, protocol,
                                    declared.at(task.domain), policy, false,
                                    model_cache);
      evaluated.push_back({policy, std::move(result)});
    }
    const auto control_it = std::find_if(
        evaluated.begin(), evaluated.end(),
        [](const Evaluated& item) { return item.policy.frozen_prefix_control; });
    require(control_it != evaluated.end(), "frozen prefix control is absent");
    require(risk_feasible(control_it->result, control_it->result),
            "frozen prefix control must always be feasible against itself");
    if (protocol.require_frozen_current135) {
      const std::map<std::string, std::uint64_t> expected_control_bytes{
          {"algorithm_space", 2941000129ULL},
          {"operator_zstd3_space", 2662068813ULL},
          {"observed_pipeline_space", 2678889909ULL}};
      require(control_it->result.total_selected_bytes ==
                  expected_control_bytes.at(task.id),
              "frozen prefix control OOF byte total differs from the locked "
              "quality135 v2 baseline");
    }
    const auto unconstrained_it = std::min_element(
        evaluated.begin(), evaluated.end(), [](const auto& left, const auto& right) {
          const auto l = left.result.unsupported == 0
              ? left.result.total_selected_bytes
              : std::numeric_limits<std::uint64_t>::max();
          const auto r = right.result.unsupported == 0
              ? right.result.total_selected_bytes
              : std::numeric_limits<std::uint64_t>::max();
          return l != r ? l < r : left.policy.id < right.policy.id;
        });
    auto selected_it = control_it;
    for (auto it = evaluated.begin(); it != evaluated.end(); ++it) {
      if (!risk_feasible(it->result, control_it->result) ||
          it->result.total_selected_bytes >=
              control_it->result.total_selected_bytes)
        continue;
      if (selected_it == control_it ||
          it->result.total_selected_bytes < selected_it->result.total_selected_bytes ||
          (it->result.total_selected_bytes == selected_it->result.total_selected_bytes &&
           it->policy.id < selected_it->policy.id))
        selected_it = it;
    }
    const std::string selection_reason = selected_it == control_it
        ? "no_feasible_strict_total_byte_improvement_choose_frozen_control"
        : "minimum_total_selected_bytes_among_feasible_strict_improvements";
    Json policy_summaries = Json::array();
    for (const auto& item : evaluated) {
      auto summary = compact_metrics(item.result);
      summary["policy"] = policy_json(item.policy);
      summary["risk_feasible_against_frozen_control"] =
          risk_feasible(item.result, control_it->result);
      summary["strict_total_byte_improvement_over_control"] =
          item.result.unsupported == 0 &&
          item.result.total_selected_bytes < control_it->result.total_selected_bytes;
      policy_summaries.push_back(std::move(summary));
    }
    std::cout << "[risk]   selected " << selected_it->policy.id
              << " bytes=" << selected_it->result.total_selected_bytes
              << std::endl;
    auto chosen = evaluate_policy(costs, feature_sets, task, protocol,
                                  declared.at(task.domain), selected_it->policy, true,
                                  model_cache);
    auto control = evaluate_policy(costs, feature_sets, task, protocol,
                                   declared.at(task.domain), control_it->policy, true,
                                   model_cache);
    selected_ranks += jsonl_text(chosen.rows);
    control_ranks += jsonl_text(control.rows);

    const auto final_indexes = training_indexes(costs, std::nullopt, protocol);
    const auto final_pools = build_dtype_pools(
        costs, final_indexes, task, protocol, declared.at(task.domain));
    q::FitConfig final_config = selected_it->policy.model;
    final_config.candidate_catalog_sha256 =
        d::sha(Json(declared.at(task.domain)).dump());
    const auto& selected_features =
        feature_sets.at(selected_it->policy.feature_variant).rows;
    const auto final_training_rows = make_training_rows(
        costs, final_indexes, selected_features, task, protocol,
        selected_it->policy.weight_mode);
    Json weight_bindings = Json::array();
    for (const auto& row : final_training_rows)
      weight_bindings.push_back({{"sample_id", row.sample_id},
                                 {"weight", row.weight}});
    const Json training_binding{
        {"schema", "operator-oracle-risk135-final-training-binding-v1"},
        {"row_count", final_training_rows.size()},
        {"feature_variant", selected_it->policy.feature_variant},
        {"feature_names_sha256",
         d::sha(Json(final_config.feature_names).dump())},
        {"weight_mode", weight_mode_name(selected_it->policy.weight_mode)},
        {"ordered_sample_weight_json_sha256", d::sha(weight_bindings.dump())}};
    auto core = q::fit(final_training_rows, task.domain, task.objective,
                       final_config);
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
    Json envelope{
        {"schema", "operator-oracle-risk135-model-envelope-v1"},
        {"task_id", task.id},
        {"core", core.to_json()},
        {"driver_metadata",
         {{"selected_policy", policy_json(selected_it->policy)},
          {"frozen_prefix_control_policy", policy_json(control_it->policy)},
          {"final_training_binding", training_binding},
          {"dtype_candidate_pools", pools_json(final_pools)},
          {"declared_candidate_ids", declared.at(task.domain)},
          {"algorithm_time_contract", nullptr},
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
         {"selected_policy", policy_json(selected_it->policy)},
         {"frozen_prefix_control_policy", policy_json(control_it->policy)},
         {"unconstrained_minimum_byte_policy", policy_json(unconstrained_it->policy)},
         {"selection_reason", selection_reason},
         {"final_training_binding", training_binding},
         {"selected_metrics", compact_metrics(chosen)},
         {"control_metrics", compact_metrics(control)},
         {"policies", std::move(policy_summaries)}});
    index_tasks.push_back(
        {{"task_id", task.id},
         {"domain", q::domain_name(task.domain)},
         {"objective", q::objective_name(task.objective)},
         {"selected_policy", policy_json(selected_it->policy)},
         {"frozen_prefix_control_policy", policy_json(control_it->policy)},
         {"unconstrained_minimum_byte_policy", policy_json(unconstrained_it->policy)},
         {"selection_reason", selection_reason},
         {"final_training_binding", training_binding},
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
  write_new_text(output_dir / "SELECTED_OOF_FULL_RANKS.jsonl", selected_ranks);
  write_new_text(output_dir / "CONTROL_OOF_FULL_RANKS.jsonl", control_ranks);
  const Json training_audit{
      {"schema", "operator-oracle-risk135-training-audit-v1"},
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
  Json feature_inputs = Json::object();
  for (const auto& [variant, feature_set] : feature_sets) {
    feature_inputs[variant] =
        {{"path", fs::absolute(feature_set.path).string()},
         {"sha256", d::file_sha(feature_set.path)},
         {"receipt_path", fs::absolute(feature_set.receipt_path).string()},
         {"receipt_sha256", d::file_sha(feature_set.receipt_path)},
         {"feature_names", feature_set.names},
         {"feature_names_sha256", d::sha(Json(feature_set.names).dump())}};
  }
  Json inputs{{"development_costs",
                     {{"path", fs::absolute(costs_path).string()},
                      {"sha256", d::file_sha(costs_path)}}},
                    {"features", feature_inputs},
                    {"protocol",
                     {{"path", fs::absolute(protocol_path).string()},
                      {"sha256", d::file_sha(protocol_path)}}},
                    {"preparation", preparation_inputs}};
  const Json cv_summary{
      {"schema", "operator-oracle-risk135-development-cv-v1"},
      {"status", "complete"},
      {"scope", "development_only_model_selection_not_external_evaluation"},
      {"folds", 5},
      {"primary_selection_objective",
       "minimum total selected compressed bytes over development OOF"},
      {"risk_constraints",
       {{"unsupported_validation_rows", 0},
        {"dataset_equal_geo_regret_max_multiple_of_control", 1.05},
        {"worst_official_source_group_aggregate_byte_regret_max",
         "frozen_prefix_control"},
        {"cvar95_absolute_excess_bytes_max", "frozen_prefix_control"}}},
      {"cvar_contract",
       "mean largest ceil(0.05*N) per-column nonnegative absolute excess bytes over allowed candidate-pool oracle"},
      {"long_column_contract",
       "per-fold training original_numeric_bytes nearest-rank P90; heldout bytes >= threshold"},
      {"confirm_used_for_tuning", false},
      {"tasks", task_summaries},
      {"inputs", inputs}};
  write_new_json(output_dir / "CV_SUMMARY.json", cv_summary);
  const Json model_index{
      {"schema", "operator-oracle-risk135-model-index-v1"},
      {"status", "locked_development_only"},
      {"feature_variants", feature_inputs},
      {"confirm_used_for_fit_or_selection", false},
      {"NNCP", "OFF"},
      {"excluded_algorithm_methods", Json::array({"PPMD8"})},
      {"excluded_action_ids", Json::array({106, 107, 108, 109, 110})},
      {"joint_cross_product_measured", false},
      {"objectives", Json::array({"space"})},
      {"inputs", inputs},
      {"training_audit_path", "TRAINING_AUDIT.json"},
      {"training_audit_sha256",
       d::file_sha(output_dir / "TRAINING_AUDIT.json")},
      {"tasks", index_tasks}};
  write_new_json(output_dir / "MODEL_INDEX.json", model_index);
  const Json complete{
      {"schema", "operator-oracle-risk135-tune-complete-v1"},
      {"status", "complete"},
      {"model_index_sha256", d::file_sha(output_dir / "MODEL_INDEX.json")},
      {"cv_summary_sha256", d::file_sha(output_dir / "CV_SUMMARY.json")},
      {"selected_oof_sha256",
       d::file_sha(output_dir / "SELECTED_OOF_FULL_RANKS.jsonl")},
      {"control_oof_sha256",
       d::file_sha(output_dir / "CONTROL_OOF_FULL_RANKS.jsonl")},
      {"training_audit_sha256",
       d::file_sha(output_dir / "TRAINING_AUDIT.json")},
      {"development_rows", costs.size()},
      {"confirm_rows_read", 0}};
  write_new_json(output_dir / "COMPLETE.json", complete);
  std::cout << "[risk] tune complete " << output_dir << std::endl;
}

struct LoadedTask {
  Task task;
  PolicyConfig policy;
  q::QualityModel model;
  DTypePools pools;
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
              "operator-oracle-risk135-model-envelope-v1" &&
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
  require(item.contains("final_training_binding") &&
              metadata.contains("final_training_binding") &&
              item.at("final_training_binding") ==
                  metadata.at("final_training_binding"),
          "index and model-envelope final training bindings disagree");
  require(result.model.feature_count() == result.policy.model.feature_names.size(),
          "model core feature count disagrees with selected feature variant");

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
  require(core.contains("algorithm") && core.at("algorithm").is_object(),
          "model core lacks algorithm metadata");
  const auto& algorithm = core.at("algorithm");
  const std::string expected_label =
      result.policy.model.center_log_cost_per_row
          ? "per_row_centered_log_cost" : "log_cost";
  require(finite_number(algorithm, "ridge_lambda", "model algorithm") ==
              result.policy.model.ridge_lambda &&
              unsigned_integer(algorithm, "knn_k", "model algorithm") ==
                  result.policy.model.knn_k &&
              finite_number(algorithm, "ridge_weight", "model algorithm") ==
                  result.policy.model.ridge_weight &&
              text(algorithm, "label", "model algorithm") == expected_label,
          "model core algorithm disagrees with selected policy");
  result.policy.model.candidate_catalog_sha256 = catalog_sha;

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
             const fs::path& multi_features_path,
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
              "operator-oracle-risk135-model-index-v1" &&
              index.value("confirm_used_for_fit_or_selection", true) == false,
          "unexpected or unsafe model index");
  std::vector<LoadedTask> tasks;
  for (const auto& item : index.at("tasks"))
    tasks.push_back(load_task(index_path, item));
  require(tasks.size() == kTasks.size(), "model index must contain three tasks");
  const auto model_index_sha256 = d::file_sha(index_path);
  const auto model_cpu_ns = d::cpu_ns() - model_cpu_start;
  const auto model_wall_ns = d::wall_ns() - model_wall_start;
  const auto features_wall_start = d::wall_ns();
  const auto features_cpu_start = d::cpu_ns();
  FeatureSets feature_sets;
  feature_sets.emplace("prefix24", FeatureSet{
      "prefix24", prefix_feature_names(), features_path,
      features_path.parent_path() / "FEATURES_RECEIPT.json",
      load_features(features_path, prefix_feature_names(), true)});
  feature_sets.emplace("multi3", FeatureSet{
      "multi3", multi_feature_names(), multi_features_path,
      multi_features_path.parent_path() / "FEATURES_RECEIPT.json",
      load_features(multi_features_path, multi_feature_names(), true)});
  require(feature_sets.at("prefix24").rows.size() ==
              feature_sets.at("multi3").rows.size(),
          "prediction feature variants differ in size");
  const auto prefix_sha256 = d::file_sha(features_path);
  const auto multi_sha256 = d::file_sha(multi_features_path);
  require(index.at("inputs").at("features").at("prefix24").at("sha256") ==
              prefix_sha256 &&
              index.at("inputs").at("features").at("multi3").at("sha256") ==
                  multi_sha256,
          "prediction features differ from locked training feature variants");
  const auto features_cpu_ns = d::cpu_ns() - features_cpu_start;
  const auto features_wall_ns = d::wall_ns() - features_wall_start;
  const auto prediction_wall_start = d::wall_ns();
  const auto prediction_cpu_start = d::cpu_ns();
  std::string output;
  for (const auto& [identity, prefix_feature] :
       feature_sets.at("prefix24").rows) {
    for (const auto& task : tasks) {
      const auto& feature =
          feature_sets.at(task.policy.feature_variant).rows.at(identity);
      const auto pool_it = task.pools.find(feature.dtype);
      require(pool_it != task.pools.end(),
              "prediction dtype absent from locked model pool");
      // One warm call followed by five measured calls. Model setup and feature
      // extraction happen outside this boundary.
      const std::map<std::string, std::string> no_constants;
      auto selection = select(task.policy, &task.model, no_constants,
                              task.pools, feature);
      std::array<std::uint64_t, 5> timing_samples{};
      for (std::size_t repeat = 0; repeat < timing_samples.size(); ++repeat) {
        const auto start = d::cpu_ns();
        selection = select(task.policy, &task.model, no_constants,
                           task.pools, feature);
        timing_samples[repeat] = d::cpu_ns() - start;
      }
      auto sorted_timings = timing_samples;
      std::sort(sorted_timings.begin(), sorted_timings.end());
      const Json timing_samples_json = timing_samples;
      auto ranking =
          ranking_arrays(selection, pool_it->second, "model");
      Json row{{"schema", "operator-oracle-risk135-prediction-v1"},
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
               {"query_feature_variant", task.policy.feature_variant},
               {"query_features_sha256",
                task.policy.feature_variant == "prefix24"
                    ? prefix_sha256 : multi_sha256},
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
      {"schema", "operator-oracle-risk135-prediction-receipt-v1"},
      {"status", "complete"},
      {"output",
       {{"path", fs::absolute(output_path).string()},
        {"sha256", d::file_sha(output_path)},
        {"row_count", feature_sets.at("prefix24").rows.size() *
                          tasks.size()}}},
      {"inputs",
       {{"model_index",
         {{"path", fs::absolute(index_path).string()},
          {"sha256", model_index_sha256}}},
        {"features",
         {{"prefix24",
           {{"path", fs::absolute(features_path).string()},
            {"sha256", prefix_sha256}}},
          {"multi3",
           {{"path", fs::absolute(multi_features_path).string()},
            {"sha256", multi_sha256}}}}}}},
      {"task_count", tasks.size()},
      {"feature_row_count", feature_sets.at("prefix24").rows.size()},
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
            << feature_sets.at("prefix24").rows.size() * tasks.size()
            << " receipt=" << receipt_path
            << std::endl;
}

#if 0
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
#endif

void selftest() {
  require(prefix_feature_names().size() == 24 &&
              multi_feature_names().size() == 98,
          "canonical feature dimensions changed");
  std::vector<q::TrainingRow> rows;
  for (std::size_t i = 0; i < 12; ++i) {
    std::vector<double> features(multi_feature_names().size(), 0.0);
    features[0] = static_cast<double>(i);
    features[24] = static_cast<double>((i * 3) % 7);
    const double left = i < 6 ? 1.0 : 4.0;
    const double right = i < 6 ? 3.0 : 1.0;
    rows.push_back({"sample-" + std::to_string(i),
                    "dataset-" + std::to_string(i % 3),
                    "source-" + std::to_string(i % 5),
                    1.0,
                    std::move(features),
                    {q::CandidateCosts{"left", true, left, {}, {}, {}},
                     q::CandidateCosts{"right", true, right, {}, {}, {}}}});
  }
  q::FitConfig config;
  config.ridge_lambda = 1.0;
  config.knn_k = 24;
  config.ridge_weight = 0.5;
  config.center_log_cost_per_row = true;
  config.feature_names = multi_feature_names();
  config.candidate_catalog_sha256 =
      d::sha(Json(std::vector<std::string>{"left", "right"}).dump());
  const auto model = q::fit(rows, q::Domain::Algorithm,
                            q::Objective::Space, config);
  const auto loaded = q::QualityModel::from_json(model.to_json());
  const auto expected = model.rank(rows.front().features, {"left", "right"});
  const auto actual = loaded.rank(rows.front().features, {"left", "right"});
  require(expected.ranking.size() == actual.ranking.size(),
          "nondefault multi98 save/load ranking size mismatch");
  for (std::size_t i = 0; i < expected.ranking.size(); ++i) {
    require(expected.ranking[i].candidate_id == actual.ranking[i].candidate_id &&
                expected.ranking[i].predicted_log_cost ==
                    actual.ranking[i].predicted_log_cost,
            "nondefault multi98 save/load full rank mismatch");
  }

  std::vector<CostRow> costs(3);
  for (std::size_t i = 0; i < costs.size(); ++i) {
    costs[i].sample_id = "c" + std::to_string(i);
    costs[i].dataset_id = i < 2 ? "d0" : "d1";
    costs[i].source_group = "s" + std::to_string(i);
    costs[i].dtype = "float64";
    costs[i].original_numeric_bytes = i == 2 ? 10000 : 100;
    RawCandidate candidate;
    candidate.costs = {"left", true, 1.0, {}, {}, {}};
    candidate.compressed_bytes = costs[i].original_numeric_bytes / 2;
    costs[i].domains[q::Domain::Algorithm] = {candidate};
  }
  std::map<std::string, FeatureRow> features;
  for (std::size_t i = 0; i < costs.size(); ++i)
    features.emplace(key(costs[i].dataset_id, costs[i].sample_id),
                     FeatureRow{costs[i].dataset_id, costs[i].sample_id,
                                "float64", rows[i].features});
  Protocol protocol;
  protocol.require_frozen_current135 = false;
  protocol.algorithm_time_contract = kRequiredTimeContract;
  const Task task{"algorithm_space", q::Domain::Algorithm,
                  q::Objective::Space};
  const std::vector<std::size_t> indexes{0, 1, 2};
  const auto dataset_equal = make_training_rows(
      costs, indexes, features, task, protocol, WeightMode::DatasetEqual);
  const auto sqrt_bytes = make_training_rows(
      costs, indexes, features, task, protocol, WeightMode::SqrtBytes);
  const auto mixed = make_training_rows(
      costs, indexes, features, task, protocol, WeightMode::MixedBytes);
  auto sum_weights = [](const auto& weighted) {
    double sum = 0.0;
    for (const auto& row : weighted) sum += row.weight;
    return sum;
  };
  require(std::abs(sum_weights(dataset_equal) - 2.0) < 1e-12 &&
              std::abs(sum_weights(sqrt_bytes) - 2.0) < 1e-12 &&
              std::abs(sum_weights(mixed) - 2.0) < 1e-12,
          "weight modes must sum to dataset count");
  require(sqrt_bytes[2].weight > dataset_equal[2].weight &&
              mixed[2].weight > dataset_equal[2].weight,
          "byte-aware modes must increase long-column influence");

  CVResult control;
  control.unsupported = 0;
  control.total_selected_bytes = 100;
  control.dataset_equal_geo_regret = 1.2;
  control.worst_source_group_byte_regret = 1.5;
  control.cvar95_absolute_excess_bytes = 20.0;
  require(risk_feasible(control, control),
          "frozen control must be feasible against itself");
  auto unsafe = control;
  unsafe.cvar95_absolute_excess_bytes = 21.0;
  require(!risk_feasible(unsafe, control),
          "absolute-byte CVaR increase must violate the constraint");
  unsafe = control;
  unsafe.worst_source_group_byte_regret = 1.6;
  require(!risk_feasible(unsafe, control),
          "source-group risk increase must violate the constraint");
  std::cout << "risk_experiment selftest: PASS\n";
}

void usage() {
  std::cerr
      << "usage:\n"
      << "  risk_experiment tune DEV_COSTS_JSONL PREFIX_FEATURES_JSONL "
         "MULTI_FEATURES_JSONL PROTOCOL_JSON OUTPUT_DIR\n"
      << "  risk_experiment predict MODEL_INDEX_JSON PREFIX_FEATURES_JSONL "
         "MULTI_FEATURES_JSONL OUTPUT_JSONL\n"
      << "  risk_experiment selftest\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 7 && std::string_view(argv[1]) == "tune") {
      tune(argv[2], argv[3], argv[4], argv[5], argv[6]);
      return 0;
    }
    if (argc == 6 && std::string_view(argv[1]) == "predict") {
      predict(argv[2], argv[3], argv[4], argv[5]);
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "selftest") {
      selftest();
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "risk_experiment: FAIL: " << error.what() << '\n';
    return 1;
  }
}
