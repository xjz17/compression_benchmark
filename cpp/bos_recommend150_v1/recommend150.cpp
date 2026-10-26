#include "recommend150.hpp"

#include "../operator_oracle_quality135_v1/quality_model.hpp"
#include "../operator_recommend135_v1/common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <zlib.h>

namespace bos_recommend150_v1 {
namespace {

namespace fs = std::filesystem;
namespace q = operator_oracle_quality135_v1;
namespace d = operator_recommend135_v1::detail;
using Json = nlohmann::json;

constexpr std::size_t kFoldCount = 5;
constexpr std::size_t kFeatureMapCount = 43;
constexpr std::size_t kMissingIndicatorCount = 12;
constexpr std::size_t kDerivedFeatureCount = 6;
constexpr std::string_view kBos = "BOS";
constexpr std::string_view kLz4 = "LZ4";
constexpr std::string_view kZstd3 = "Zstd";

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("bos_recommend150_v1: " + message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::string required_string(const Json& value, const char* key,
                            const std::string& context) {
  require(value.contains(key) && value.at(key).is_string() &&
              !value.at(key).get_ref<const std::string&>().empty(),
          context + " missing non-empty string " + key);
  return value.at(key).get<std::string>();
}

std::string optional_string(const Json& value, const char* key) {
  return value.contains(key) && value.at(key).is_string()
             ? value.at(key).get<std::string>()
             : std::string();
}

std::uint64_t required_u64(const Json& value, const char* key,
                           const std::string& context) {
  require(value.contains(key) &&
              (value.at(key).is_number_unsigned() ||
               value.at(key).is_number_integer()),
          context + " missing integer " + key);
  if (value.at(key).is_number_integer()) {
    const auto result = value.at(key).get<std::int64_t>();
    require(result >= 0, context + " has negative " + key);
    return static_cast<std::uint64_t>(result);
  }
  return value.at(key).get<std::uint64_t>();
}

double required_positive(const Json& value, const char* key,
                         const std::string& context) {
  require(value.contains(key) && value.at(key).is_number(),
          context + " missing numeric " + key);
  const double result = value.at(key).get<double>();
  require(std::isfinite(result) && result > 0.0,
          context + " has non-positive/non-finite " + key);
  return result;
}

std::uint64_t numeric_dtype_width(const std::string& dtype) {
  if (dtype == "int8") return 1;
  if (dtype == "int16") return 2;
  if (dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "float64") return 8;
  fail("unsupported physical dtype " + dtype);
}

void for_each_jsonl(const fs::path& path,
                    const std::function<void(const Json&)>& consume) {
  std::size_t line_number = 0;
  std::size_t rows = 0;
  auto parse = [&](const std::string& line) {
    ++line_number;
    if (line.empty()) return;
    try {
      consume(Json::parse(line));
      ++rows;
    } catch (const std::exception& error) {
      fail(path.string() + ":" + std::to_string(line_number) + ": " +
           error.what());
    }
  };
  if (path.extension() == ".gz") {
    gzFile source = gzopen(path.c_str(), "rb");
    require(source != nullptr, "cannot read gzip JSONL " + path.string());
    std::array<char, 65536> chunk{};
    std::string line;
    while (true) {
      char* result = gzgets(source, chunk.data(), static_cast<int>(chunk.size()));
      if (result == nullptr) break;
      line.append(result);
      if (!line.empty() && line.back() == '\n') {
        line.pop_back();
        parse(line);
        line.clear();
      }
    }
    if (!line.empty()) parse(line);
    int error_number = Z_OK;
    const char* error_text = gzerror(source, &error_number);
    const bool ok = error_number == Z_OK || error_number == Z_STREAM_END;
    const std::string saved_error = error_text == nullptr ? "unknown" : error_text;
    gzclose(source);
    require(ok, "error reading gzip JSONL " + path.string() + ": " + saved_error);
  } else {
    std::ifstream source(path);
    require(bool(source), "cannot read JSONL " + path.string());
    std::string line;
    while (std::getline(source, line)) parse(line);
    require(source.eof(), "error reading JSONL " + path.string());
  }
  require(rows != 0, "empty JSONL " + path.string());
}

void write_new_json(const fs::path& path, const Json& value) {
  require(!fs::exists(path), "refusing to replace " + path.string());
  d::write_json(path, value);
}

std::string key(const std::string& dataset, const std::string& sample) {
  return dataset + "\n" + sample;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool forbidden_candidate(const std::string& id) {
  const auto name = lower(id);
  return name == "ppmd8" || name == "ppmd" || name == "nncp" ||
         name == "dmc" || name == "paq8" || name == "cmix" ||
         name == "lstm";
}

bool valid_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                  (c >= 'A' && c <= 'F');
         });
}

enum class Objective { Space, Encode, Decode };

constexpr std::array<Objective, 3> kObjectives = {
    Objective::Space, Objective::Encode, Objective::Decode};

std::string_view objective_name(Objective objective) {
  switch (objective) {
    case Objective::Space: return "space";
    case Objective::Encode: return "encode";
    case Objective::Decode: return "decode";
  }
  return "unknown";
}

q::Objective quality_objective(Objective objective) {
  switch (objective) {
    case Objective::Space: return q::Objective::Space;
    case Objective::Encode: return q::Objective::Encode;
    case Objective::Decode: return q::Objective::Decode;
  }
  fail("unknown objective");
}

struct SplitAssignment {
  std::string partition;
  std::string source_group;
  std::optional<std::size_t> fold;
};

using SplitMap = std::map<std::string, SplitAssignment>;
using CatalogByDtype = std::map<std::string, std::vector<std::string>>;

struct CohortAssignment {
  std::string cohort;
  std::string source_group;
  std::string legacy_group;
  std::optional<std::size_t> fold;
};
using CohortMap = std::map<std::string, CohortAssignment>;

SplitMap load_split(const fs::path& path) {
  const auto root = d::read_json(path);
  require(root.is_object() && root.contains("dataset_assignments") &&
              root.at("dataset_assignments").is_object(),
          "split lacks dataset_assignments");
  if (root.contains("policy")) {
    require(root.at("policy").value("dev_fold_count", 0U) == kFoldCount,
            "split must use five development folds");
  }
  SplitMap result;
  std::map<std::string, std::size_t> source_fold;
  for (const auto& [dataset, value] :
       root.at("dataset_assignments").items()) {
    require(value.is_object(), "split assignment is not an object");
    SplitAssignment assignment;
    assignment.partition = required_string(value, "split", "split assignment");
    if (value.contains("source_identity_sha256") &&
        value.at("source_identity_sha256").is_string()) {
      assignment.source_group = value.at("source_identity_sha256").get<std::string>();
    } else {
      assignment.source_group = required_string(
          value, "source_identity", "split assignment");
    }
    if (assignment.partition == "development") {
      require(value.contains("dev_fold") && value.at("dev_fold").is_number_integer(),
              "development assignment lacks dev_fold");
      assignment.fold = value.at("dev_fold").get<std::size_t>();
      require(*assignment.fold < kFoldCount, "dev_fold outside [0,4]");
      const auto [it, inserted] =
          source_fold.emplace(assignment.source_group, *assignment.fold);
      require(inserted || it->second == *assignment.fold,
              "source_group crosses development folds");
    } else {
      require(assignment.partition == "confirm",
              "unsupported old135 split partition " + assignment.partition);
      require(!value.contains("dev_fold") || value.at("dev_fold").is_null(),
              "confirm assignment must not have dev_fold");
    }
    require(result.emplace(dataset, std::move(assignment)).second,
            "duplicate dataset in split");
  }
  return result;
}

CohortMap load_cohorts(const fs::path& path, const SplitMap& split) {
  const auto root = d::read_json(path);
  require(root.value("schema", std::string()) ==
              "bos-recommend150-split-cohorts-v1" &&
              root.value("status", std::string()) == "frozen_label_free" &&
              !root.value("compression_results_read", true) &&
              !root.value("confirm_or_external_used_for_tuning", true),
          "cohorts file is not the frozen label-free assignment");
  require(root.contains("dataset_assignments") &&
              root.at("dataset_assignments").is_object(),
          "cohorts lacks dataset_assignments");
  CohortMap result;
  for (const auto& [dataset, value] :
       root.at("dataset_assignments").items()) {
    CohortAssignment item;
    item.cohort = required_string(value, "cohort", "cohort assignment");
    item.source_group = required_string(value, "source_identity_sha256",
                                        "cohort assignment");
    item.legacy_group = required_string(value, "legacy_group",
                                        "cohort assignment");
    require(item.cohort == "development" || item.cohort == "confirm" ||
                item.cohort == "external_final",
            "unknown frozen cohort " + item.cohort);
    if (item.cohort == "development") {
      require(value.contains("dev_fold") &&
                  value.at("dev_fold").is_number_integer(),
              "development cohort lacks dev_fold");
      item.fold = value.at("dev_fold").get<std::size_t>();
      require(*item.fold < kFoldCount, "cohort dev_fold outside [0,4]");
    } else {
      require(!value.contains("dev_fold") || value.at("dev_fold").is_null(),
              "non-development cohort must not have dev_fold");
    }
    const auto old = split.find(dataset);
    if (item.cohort == "external_final") {
      require(old == split.end(), "external cohort contains old135 dataset");
    } else {
      require(old != split.end(), "old135 cohort dataset absent from split");
      require(old->second.partition == item.cohort &&
                  old->second.source_group == item.source_group &&
                  old->second.fold == item.fold,
              "cohorts differs from frozen old135 split for " + dataset);
    }
    require(result.emplace(dataset, std::move(item)).second,
            "duplicate dataset in cohorts");
  }
  require(result.size() == 150 || result.size() < 100,
          "cohorts must describe frozen150 or a small self-test fixture");
  return result;
}

struct CandidateSpec {
  struct BosMeasurementSpec {
    std::string source;
    std::string source_result_sha256;
    std::string input_profile;
    std::string timing_contract;
    std::string paired_acceptance_sha256;
    std::vector<std::string> binary_sha256;
    std::vector<std::string> route_ids;
    std::vector<std::string> profile_versions;
  };
  std::vector<std::string> primary;
  std::vector<std::string> restricted;
  std::vector<std::string> full_reference_exclusions;
  std::string bos_float_route;
  std::string bos_int64_route;
  std::map<std::string, BosMeasurementSpec> bos_measurements;
};

CandidateSpec load_candidates(const fs::path& path, std::size_t expected_methods) {
  const auto root = d::read_json(path);
  require(root.value("schema", std::string()) ==
              "bos-recommend150-candidates-v1",
          "unexpected candidates schema");
  CandidateSpec result;
  result.primary = root.at("primary_selectable_ids")
                       .get<std::vector<std::string>>();
  result.restricted = root.at("restricted_ablation_ids")
                          .get<std::vector<std::string>>();
  result.full_reference_exclusions =
      root.at("full_reference_excluded_from_serving")
          .get<std::vector<std::string>>();
  const auto& replacement = root.at("bos_replacement");
  require(replacement.value("candidate_id", std::string()) == kBos &&
              !replacement.value("duplicate_candidate_created", true),
          "candidates does not specify in-place BOS replacement");
  result.bos_float_route =
      required_string(replacement, "float_route", "BOS replacement");
  result.bos_int64_route =
      required_string(replacement, "int64_route", "BOS replacement");
  const auto& measurements = replacement.at("benchmark_measurement_provenance");
  require(measurements.is_object() && !measurements.empty(),
          "BOS benchmark measurement provenance is absent");
  for (const auto& [name, value] : measurements.items()) {
    require(value.is_object(), "invalid BOS measurement provenance " + name);
    CandidateSpec::BosMeasurementSpec spec;
    spec.source = required_string(value, "source", "BOS measurement " + name);
    spec.source_result_sha256 = required_string(
        value, "source_result_sha256", "BOS measurement " + name);
    spec.input_profile =
        required_string(value, "input_profile", "BOS measurement " + name);
    spec.timing_contract =
        required_string(value, "timing_contract", "BOS measurement " + name);
    spec.paired_acceptance_sha256 =
        optional_string(value, "paired_acceptance_sha256");
    spec.binary_sha256 =
        value.at("allowed_binary_sha256").get<std::vector<std::string>>();
    spec.route_ids =
        value.at("allowed_route_ids").get<std::vector<std::string>>();
    spec.profile_versions =
        value.at("allowed_profile_versions").get<std::vector<std::string>>();
    require(valid_sha256(spec.source_result_sha256) &&
                !spec.binary_sha256.empty() && !spec.route_ids.empty() &&
                !spec.profile_versions.empty(),
            "incomplete BOS measurement provenance " + name);
    for (const auto& hash : spec.binary_sha256)
      require(valid_sha256(hash), "invalid BOS measurement binary SHA256");
    if (!spec.paired_acceptance_sha256.empty())
      require(valid_sha256(spec.paired_acceptance_sha256),
              "invalid BOS paired acceptance SHA256");
    require(result.bos_measurements.emplace(spec.source, std::move(spec)).second,
            "duplicate BOS measurement provenance source");
  }
  if (expected_methods == 85) {
    const std::set<std::string> expected_sources = {
        "bos_optimized_bfe", "latest135_bos_reger", "new15_algorithm85"};
    std::set<std::string> actual_sources;
    for (const auto& [source, unused] : result.bos_measurements) {
      (void)unused;
      actual_sources.insert(source);
    }
    require(actual_sources == expected_sources,
            "production BOS measurement source closure differs");
  }
  std::sort(result.primary.begin(), result.primary.end());
  std::sort(result.restricted.begin(), result.restricted.end());
  require(result.restricted ==
              std::vector<std::string>({"BOS", "LZ4", "Zstd"}),
          "restricted candidate set differs from BOS/LZ4/Zstd");
  if (expected_methods == 85) {
    require(result.primary.size() == 84 &&
                result.full_reference_exclusions ==
                    std::vector<std::string>({"PPMD8"}),
            "production primary84/full85 exclusion contract differs");
  }
  for (const auto& id : result.primary)
    require(!forbidden_candidate(id), "forbidden selectable candidate " + id);
  return result;
}

struct FeatureRow {
  std::string dataset;
  std::string sample;
  std::string dtype;
  std::string column_name;
  std::uint64_t row_count = 0;
  std::uint64_t valid_count = 0;
  std::uint64_t width = 0;
  std::vector<double> features;
};

struct FeatureTable {
  std::vector<std::string> names;
  std::map<std::string, FeatureRow> by_identity;
  std::map<std::string, std::string> aliases;
  std::set<std::string> ambiguous_aliases;
};

FeatureTable load_features(const fs::path& path) {
  FeatureTable table;
  std::vector<std::string> base_names;
  const std::set<std::string> integer_not_applicable = {
      "sample_decimal_min_p_max", "sample_decimal_min_p_median",
      "sample_decimal_min_p_p90", "sample_decimal_reversible_fraction_max_p0",
      "sample_decimal_reversible_fraction_max_p12",
      "sample_decimal_reversible_fraction_max_p18",
      "sample_decimal_reversible_fraction_max_p3",
      "sample_decimal_reversible_fraction_max_p6",
      "sample_decimal_reversible_fraction_max_p9",
      "sample_decimal_reversible_fraction_p0_to_p18",
      "sample_negative_zero_rate", "sample_nonfinite_rate"};
  for_each_jsonl(path, [&](const Json& value) {
    require(value.is_object() &&
                value.value("schema", std::string()) ==
                    "solid-dtype-correct-column-features-v1",
            "unexpected clean150 feature schema");
    const auto dataset = required_string(value, "dataset_id", "feature row");
    // The accepted feature artifact defines column_id as the stable identity.
    // In particular, integer rows often carry a non-empty JSON column_key;
    // descriptive metadata must not take precedence over the stable ID.
    const auto sample = required_string(value, "column_id", "feature row");
    const auto dtype = required_string(value, "dtype", "feature row");
    require(dtype == "float32" || dtype == "float64" || dtype == "int64",
            "unsupported clean150 feature dtype " + dtype);
    const auto width = required_u64(value, "width", "feature row");
    require((dtype == "float32" && width == 4) ||
                ((dtype == "float64" || dtype == "int64") && width == 8),
            "feature dtype/width mismatch");
    const auto rows = required_u64(value, "row_count", "feature row");
    const auto valid = required_u64(value, "valid_count", "feature row");
    require(valid > 0 && valid <= rows, "invalid feature row/valid counts");
    require(value.contains("features") && value.at("features").is_object(),
            "feature row lacks feature map");
    std::vector<std::string> current;
    for (const auto& [name, unused] : value.at("features").items()) {
      (void)unused;
      current.push_back(name);
    }
    std::sort(current.begin(), current.end());
    require(current.size() == kFeatureMapCount,
            "clean150 feature map must have exactly 43 fields");
    if (base_names.empty()) base_names = current;
    require(current == base_names, "clean150 feature maps have different keys");

    FeatureRow row;
    row.dataset = dataset;
    row.sample = sample;
    row.dtype = dtype;
    row.column_name = optional_string(value, "column_name");
    row.row_count = rows;
    row.valid_count = valid;
    row.width = width;
    row.features.reserve(kFeatureMapCount + kMissingIndicatorCount +
                         kDerivedFeatureCount);
    for (const auto& name : base_names) {
      const auto& encoded = value.at("features").at(name);
      if (encoded.is_null()) {
        require(dtype == "int64" && integer_not_applicable.contains(name),
                "unexpected null feature " + name + " for " + dtype);
        row.features.push_back(0.0);
        continue;
      }
      require(encoded.is_number(), "feature " + name + " is not numeric");
      const auto number = encoded.get<double>();
      require(std::isfinite(number), "feature " + name + " is not finite");
      row.features.push_back(number);
    }
    for (const auto& name : integer_not_applicable)
      row.features.push_back(value.at("features").at(name).is_null() ? 1.0
                                                                     : 0.0);
    row.features.push_back(dtype == "float32" ? 1.0 : 0.0);
    row.features.push_back(dtype == "float64" ? 1.0 : 0.0);
    row.features.push_back(dtype == "int64" ? 1.0 : 0.0);
    row.features.push_back(static_cast<double>(width));
    row.features.push_back(std::log1p(static_cast<double>(rows)));
    row.features.push_back(std::log1p(static_cast<double>(valid)));

    const auto canonical = key(dataset, sample);
    require(table.by_identity.emplace(canonical, row).second,
            "duplicate canonical feature identity " + canonical);
    auto alias = [&](const std::string& id) {
      if (id.empty()) return;
      const auto alias_key = key(dataset, id);
      if (table.ambiguous_aliases.contains(alias_key)) return;
      const auto [it, inserted] = table.aliases.emplace(alias_key, canonical);
      if (!inserted && it->second != canonical) {
        table.aliases.erase(it);
        table.ambiguous_aliases.insert(alias_key);
      }
    };
    alias(sample);
    alias(row.column_name);
    if (value.contains("column_key") && value.at("column_key").is_string())
      alias(value.at("column_key").get<std::string>());
  });
  require(table.by_identity.size() == 3644 || table.by_identity.size() < 100,
          "feature input must be frozen3644 or a small self-test fixture");
  for (const auto& name : base_names) table.names.push_back("map:" + name);
  for (const auto& name : integer_not_applicable)
    table.names.push_back("map_missing:" + name);
  table.names.insert(table.names.end(),
                     {"meta:dtype_float32", "meta:dtype_float64",
                      "meta:dtype_int64", "meta:width_bytes",
                      "meta:log1p_row_count", "meta:log1p_valid_count"});
  require(table.names.size() == kFeatureMapCount + kMissingIndicatorCount +
                                    kDerivedFeatureCount,
          "internal feature-name count mismatch");
  return table;
}

struct MethodCost {
  std::string id;
  bool eligible = false;
  double bytes = 0.0;
  double encode_ns = 0.0;
  double decode_ns = 0.0;
  std::string binary_sha256;
  std::string route_id;
  std::string input_profile;
  std::string profile_version;
  std::string timing_contract;
  std::string provenance_source;
  std::string provenance_source_file_sha256;
  std::string provenance_input_profile;
  std::string provenance_paired_acceptance_sha256;
};

struct SampleRow {
  const FeatureRow* feature = nullptr;
  std::string dataset;
  std::string sample;
  std::string source_group;
  std::string legacy_group;
  std::size_t fold = 0;
  std::uint64_t original_bytes = 0;
  std::uint64_t source_native_bytes = 0;
  std::uint64_t valid_values = 0;
  std::map<std::string, MethodCost> methods;
};

using RealizedSelections =
    std::vector<std::pair<const SampleRow*, const MethodCost*>>;
struct RankedRow {
  const SampleRow* row = nullptr;
  std::vector<std::string> ranking;
};
using RankedRows = std::vector<RankedRow>;

const MethodCost& oracle_method(const SampleRow& row,
                                const std::vector<std::string>& catalog,
                                Objective objective);
Json realized_metrics(const RealizedSelections& selected);
Json budget_feedback_evaluation(const RankedRows& rankings,
                                const CatalogByDtype& catalogs,
                                Objective objective);

double cost(const MethodCost& value, Objective objective) {
  switch (objective) {
    case Objective::Space: return value.bytes;
    case Objective::Encode: return value.encode_ns;
    case Objective::Decode: return value.decode_ns;
  }
  fail("unknown objective");
}

struct FeedLoad {
  std::vector<SampleRow> rows;
  std::vector<std::string> full_catalog;
  std::vector<std::string> primary_catalog;
  std::vector<std::string> restricted_catalog;
  Json bos_provenance = Json::array();
};

CatalogByDtype development_full_support(
    const std::vector<SampleRow>& rows,
    const std::vector<std::string>& base_catalog) {
  std::set<std::string> dtypes;
  for (const auto& row : rows) dtypes.insert(row.feature->dtype);
  CatalogByDtype result;
  for (const auto& dtype : dtypes) {
    auto& catalog = result[dtype];
    for (const auto& candidate : base_catalog) {
      const bool supported = std::all_of(
          rows.begin(), rows.end(), [&](const SampleRow& row) {
            return row.feature->dtype != dtype ||
                   row.methods.at(candidate).eligible;
          });
      if (supported) catalog.push_back(candidate);
    }
    require(!catalog.empty(),
            "development has no full-support candidates for dtype " + dtype);
  }
  return result;
}

FeedLoad load_feed(const fs::path& path, const FeatureTable& features,
                   const SplitMap& split, const CohortMap& cohorts,
                   const CandidateSpec& candidates, const std::string& partition,
                   std::size_t expected_samples,
                   std::size_t expected_methods) {
  require(partition == "development" || partition == "confirm" ||
              partition == "external-final",
          "unsupported requested partition " + partition);
  struct Pending {
    std::string dataset;
    std::string raw_sample;
    std::string canonical_feature;
    std::uint64_t original_bytes = 0;
    std::uint64_t source_native_bytes = 0;
    std::uint64_t valid_values = 0;
    std::map<std::string, MethodCost> methods;
  };
  std::map<std::string, Pending> pending;
  std::set<std::string> bos_provenance;
  for_each_jsonl(path, [&](const Json& value) {
    require(value.is_object(), "feed row is not an object");
    if (expected_methods == 85)
      require(value.value("schema", std::string()) ==
                  "clean150-column-method-result-v1",
              "unexpected production clean150 feed schema");
    const auto dataset = required_string(value, "dataset_id", "feed row");
    const auto split_it = split.find(dataset);
    if (partition == "external-final") {
      require(split_it == split.end(),
              "external-final feed contains old135 dataset " + dataset);
    } else {
      require(split_it != split.end(),
              partition + " feed contains dataset outside old135 split: " + dataset);
      require(split_it->second.partition == partition,
              "tune/final partition rejection: dataset " + dataset + " is " +
                  split_it->second.partition + ", not " + partition);
    }

    // The accepted method feed defines stable_id as its column identity. Join
    // it exactly. Names, manifests, and input order are not identity fallbacks.
    const auto raw_sample = required_string(value, "stable_id", "feed row");
    const auto canonical = key(dataset, raw_sample);
    require(features.by_identity.contains(canonical),
            "feed stable_id cannot join clean150 feature column_id: " +
                key(dataset, raw_sample));
    const auto pending_key = key(dataset, raw_sample);
    auto [pending_it, inserted] = pending.emplace(pending_key, Pending{});
    auto& sample = pending_it->second;
    const char* byte_field = value.contains("canonical_input_bytes")
                                 ? "canonical_input_bytes"
                                 : "original_numeric_bytes";
    if (expected_methods == 85)
      require(std::string_view(byte_field) == "canonical_input_bytes",
              "production feed lacks canonical_input_bytes");
    const auto original = required_u64(value, byte_field, "feed row");
    if (expected_methods == 85)
      require(value.contains("source_native_numeric_bytes"),
              "production feed lacks source_native_numeric_bytes");
    const auto source_native = value.contains("source_native_numeric_bytes")
                                   ? required_u64(value,
                                                  "source_native_numeric_bytes",
                                                  "feed row")
                                   : original;
    const auto valid = required_u64(value, "valid_values", "feed row");
    require(original > 0 && source_native > 0 && valid > 0,
            "feed row has empty input");
    if (expected_methods == 85) {
      const auto physical = required_string(value, "physical_dtype", "feed row");
      require(source_native == valid * numeric_dtype_width(physical),
              "source-native bytes disagree with physical dtype/value count");
    }
    if (inserted) {
      sample.dataset = dataset;
      sample.raw_sample = raw_sample;
      sample.canonical_feature = canonical;
      sample.original_bytes = original;
      sample.source_native_bytes = source_native;
      sample.valid_values = valid;
    } else {
      require(sample.canonical_feature == canonical &&
                  sample.original_bytes == original &&
                  sample.source_native_bytes == source_native &&
                  sample.valid_values == valid,
              "feed methods disagree on sample input contract");
    }
    MethodCost measured;
    measured.id = required_string(value, "method", "feed row");
    measured.eligible = value.value("status", std::string()) == "ok" &&
                        value.value("lossless_verified", false) &&
                        value.value("rank_eligible", false);
    if (measured.eligible) {
      measured.bytes = required_positive(value, "compressed_bytes", "feed row");
      if (value.contains("compression_time_ns_per_value")) {
        measured.encode_ns =
            required_positive(value, "compression_time_ns_per_value", "feed row") *
            static_cast<double>(valid);
        measured.decode_ns = required_positive(
                                 value, "decompression_time_ns_per_value",
                                 "feed row") *
                             static_cast<double>(valid);
      } else {
        measured.encode_ns =
            required_positive(value, "compression_ns", "feed row");
        measured.decode_ns =
            required_positive(value, "decompression_ns", "feed row");
      }
    }
    measured.binary_sha256 = optional_string(value, "binary_sha256");
    measured.route_id = optional_string(value, "route_id");
    measured.input_profile = optional_string(value, "input_profile");
    measured.profile_version = optional_string(value, "profile_version");
    measured.timing_contract = optional_string(value, "timing_contract");
    if (value.contains("provenance") && value.at("provenance").is_object()) {
      const auto& provenance = value.at("provenance");
      measured.provenance_source = optional_string(provenance, "source");
      measured.provenance_source_file_sha256 =
          optional_string(provenance, "source_file_sha256");
      measured.provenance_input_profile =
          optional_string(provenance, "input_profile");
      measured.provenance_paired_acceptance_sha256 =
          optional_string(provenance, "paired_acceptance_sha256");
    }
    require(sample.methods.emplace(measured.id, measured).second,
            "duplicate method for sample " + pending_key);
    if (measured.id == kBos) {
      require(valid_sha256(measured.binary_sha256),
              "BOS row must bind a binary SHA256");
      const auto spec_it =
          candidates.bos_measurements.find(measured.provenance_source);
      require(spec_it != candidates.bos_measurements.end(),
              "BOS row provenance source is outside the frozen closure: " +
                  measured.provenance_source);
      const auto& spec = spec_it->second;
      auto contains = [](const std::vector<std::string>& values,
                         const std::string& item) {
        return std::find(values.begin(), values.end(), item) != values.end();
      };
      require(measured.provenance_source_file_sha256 ==
                      spec.source_result_sha256 &&
                  measured.input_profile == spec.input_profile &&
                  measured.provenance_input_profile == spec.input_profile &&
                  measured.timing_contract == spec.timing_contract &&
                  contains(spec.binary_sha256, measured.binary_sha256) &&
                  contains(spec.route_ids, measured.route_id) &&
                  contains(spec.profile_versions, measured.profile_version) &&
                  measured.provenance_paired_acceptance_sha256 ==
                      spec.paired_acceptance_sha256,
              "BOS row differs from frozen benchmark measurement provenance: " +
                  measured.provenance_source);
      bos_provenance.insert(
          Json{{"source", measured.provenance_source},
               {"source_result_sha256", measured.provenance_source_file_sha256},
               {"input_profile", measured.input_profile},
               {"timing_contract", measured.timing_contract},
               {"binary_sha256", measured.binary_sha256},
               {"route_id", measured.route_id.empty() ? Json(nullptr)
                                                       : Json(measured.route_id)},
               {"profile_version",
                measured.profile_version.empty() ? Json(nullptr)
                                                 : Json(measured.profile_version)},
               {"paired_acceptance_sha256",
                measured.provenance_paired_acceptance_sha256.empty()
                    ? Json(nullptr)
                    : Json(measured.provenance_paired_acceptance_sha256)}}
              .dump());
    }
  });

  FeedLoad result;
  std::optional<std::vector<std::string>> catalog;
  std::set<std::string> used_features;
  for (auto& [unused, item] : pending) {
    (void)unused;
    std::vector<std::string> current;
    for (const auto& [method, cost_value] : item.methods) {
      (void)cost_value;
      current.push_back(method);
    }
    require(current.size() == expected_methods,
            "sample " + key(item.dataset, item.raw_sample) + " has " +
                std::to_string(current.size()) + " methods, expected " +
                std::to_string(expected_methods));
    if (!catalog) catalog = current;
    require(*catalog == current, "method catalog changes between samples");
    require(used_features.insert(item.canonical_feature).second,
            "multiple feed samples map to one feature row");
    const auto feature_it = features.by_identity.find(item.canonical_feature);
    require(feature_it != features.by_identity.end(), "internal feature join failure");
    require(feature_it->second.valid_count == item.valid_values,
            "feature/feed valid_count mismatch for " + item.canonical_feature);
    require(feature_it->second.width * item.valid_values == item.original_bytes,
            "feature/feed original byte mismatch for " + item.canonical_feature);
    SampleRow row;
    row.feature = &feature_it->second;
    row.dataset = item.dataset;
    row.sample = item.canonical_feature;
    row.original_bytes = item.original_bytes;
    row.source_native_bytes = item.source_native_bytes;
    row.valid_values = item.valid_values;
    row.methods = std::move(item.methods);
    const auto cohort_it = cohorts.find(row.dataset);
    require(cohort_it != cohorts.end(), "dataset absent from frozen cohorts");
    const std::string expected_cohort =
        partition == "external-final" ? "external_final" : partition;
    require(cohort_it->second.cohort == expected_cohort,
            "feed partition differs from frozen cohorts for " + row.dataset);
    row.source_group = cohort_it->second.source_group;
    row.legacy_group = cohort_it->second.legacy_group;
    if (partition == "development") row.fold = *cohort_it->second.fold;
    result.rows.push_back(std::move(row));
  }
  require(result.rows.size() == expected_samples,
          partition + " feed has " + std::to_string(result.rows.size()) +
              " samples, expected " + std::to_string(expected_samples));
  require(catalog.has_value(), "feed has no method catalog");
  result.full_catalog = *catalog;
  result.primary_catalog = candidates.primary;
  result.restricted_catalog = candidates.restricted;
  for (const auto& id : result.primary_catalog)
    require(std::binary_search(result.full_catalog.begin(),
                               result.full_catalog.end(), id),
            "frozen primary candidate missing from feed: " + id);
  if (expected_methods == 85) {
    require(result.full_catalog.size() == 85,
            "production full reference catalog must have 85 methods");
    require(result.primary_catalog.size() == 84,
            "production selectable catalog must have 84 methods after exclusions");
    require(std::find(result.full_catalog.begin(), result.full_catalog.end(),
                      "PPMD8") != result.full_catalog.end(),
            "production full85 reference must identify PPMD8");
    std::vector<std::string> feed_exclusions;
    std::set_difference(result.full_catalog.begin(), result.full_catalog.end(),
                        result.primary_catalog.begin(), result.primary_catalog.end(),
                        std::back_inserter(feed_exclusions));
    require(feed_exclusions == candidates.full_reference_exclusions,
            "feed full85 minus frozen primary84 differs from reference exclusions");
  }
  for (const auto& item : bos_provenance)
    result.bos_provenance.push_back(Json::parse(item));
  return result;
}

std::vector<q::TrainingRow> training_rows(
    const std::vector<SampleRow>& rows, const CatalogByDtype& catalogs,
    const std::set<std::string>& confirm_legacy_groups,
    std::optional<std::size_t> excluded_fold = std::nullopt,
    Json* audit = nullptr) {
  std::set<std::string> heldout_sources;
  std::set<std::string> heldout_legacy;
  if (excluded_fold) {
    for (const auto& row : rows) {
      if (row.fold != *excluded_fold) continue;
      heldout_sources.insert(row.source_group);
      heldout_legacy.insert(row.legacy_group);
    }
  }
  auto retained = [&](const SampleRow& row) {
    if (confirm_legacy_groups.contains(row.legacy_group)) return false;
    if (!excluded_fold) return true;
    return !heldout_sources.contains(row.source_group) &&
           !heldout_legacy.contains(row.legacy_group);
  };
  std::map<std::string, std::size_t> dataset_counts;
  for (const auto& row : rows)
    if (retained(row)) ++dataset_counts[row.dataset];
  std::vector<q::TrainingRow> result;
  std::set<std::string> retained_datasets;
  std::set<std::string> retained_sources;
  std::set<std::string> retained_legacy;
  for (const auto& row : rows) {
    if (!retained(row)) continue;
    q::TrainingRow output;
    output.sample_id = row.sample;
    output.dataset_id = row.dataset;
    output.source_group = row.source_group;
    output.weight = 1.0 / static_cast<double>(dataset_counts.at(row.dataset));
    output.features = row.feature->features;
    const auto catalog_it = catalogs.find(row.feature->dtype);
    require(catalog_it != catalogs.end(),
            "no locked development catalog for dtype " + row.feature->dtype);
    for (const auto& id : catalog_it->second) {
      const auto& measured = row.methods.at(id);
      q::CandidateCosts costs;
      costs.candidate_id = id;
      costs.eligible = measured.eligible;
      if (measured.eligible) {
        costs.space_cost = measured.bytes;
        costs.encode_cost = measured.encode_ns;
        costs.decode_cost = measured.decode_ns;
      }
      output.candidates.push_back(std::move(costs));
    }
    result.push_back(std::move(output));
    retained_datasets.insert(row.dataset);
    retained_sources.insert(row.source_group);
    retained_legacy.insert(row.legacy_group);
  }
  if (audit != nullptr) {
    *audit = {{"excluded_fold", excluded_fold ? Json(*excluded_fold) : Json(nullptr)},
              {"input_rows", rows.size()},
              {"retained_rows", result.size()},
              {"retained_datasets", retained_datasets.size()},
              {"retained_source_groups", retained_sources.size()},
              {"retained_legacy_groups", retained_legacy.size()},
              {"confirm_legacy_groups_purged", confirm_legacy_groups},
              {"heldout_source_groups_purged", heldout_sources},
              {"heldout_legacy_groups_purged", heldout_legacy}};
  }
  return result;
}

std::vector<std::string> eligible(const SampleRow& row,
                                  const CatalogByDtype& catalogs) {
  const auto it = catalogs.find(row.feature->dtype);
  require(it != catalogs.end() && !it->second.empty(),
          "no locked inference candidates for dtype " + row.feature->dtype);
  // Deliberately do not inspect held-out method status here. Eligibility is a
  // development-only, per-dtype property frozen into MODEL_INDEX.json.
  return it->second;
}

double oracle_cost(const SampleRow& row,
                   const std::vector<std::string>& catalog,
                   Objective objective) {
  double result = std::numeric_limits<double>::infinity();
  for (const auto& id : catalog) {
    const auto& item = row.methods.at(id);
    if (item.eligible) result = std::min(result, cost(item, objective));
  }
  require(std::isfinite(result), "sample has no oracle cost");
  return result;
}

struct SelectionPoint {
  std::string dataset;
  std::string source_group;
  double selected = 0.0;
  double oracle = 0.0;
};

Json risk_metrics(const std::vector<SelectionPoint>& points) {
  require(!points.empty(), "cannot summarize empty selections");
  double selected = 0.0;
  double oracle = 0.0;
  std::map<std::string, std::pair<double, double>> datasets;
  std::map<std::string, std::pair<double, double>> groups;
  std::vector<double> excess;
  for (const auto& point : points) {
    require(point.selected > 0.0 && point.oracle > 0.0,
            "invalid selection point");
    selected += point.selected;
    oracle += point.oracle;
    datasets[point.dataset].first += point.selected;
    datasets[point.dataset].second += point.oracle;
    groups[point.source_group].first += point.selected;
    groups[point.source_group].second += point.oracle;
    excess.push_back(point.selected / point.oracle - 1.0);
  }
  double dataset_logs = 0.0;
  for (const auto& [unused, totals] : datasets) {
    (void)unused;
    dataset_logs += std::log(totals.first / totals.second);
  }
  std::string worst_group;
  double worst_ratio = -1.0;
  for (const auto& [group, totals] : groups) {
    const double ratio = totals.first / totals.second;
    if (ratio > worst_ratio || (ratio == worst_ratio && group < worst_group)) {
      worst_ratio = ratio;
      worst_group = group;
    }
  }
  std::sort(excess.begin(), excess.end(), std::greater<>());
  const auto tail = std::max<std::size_t>(1, (excess.size() + 19) / 20);
  const double cvar =
      std::accumulate(excess.begin(), excess.begin() + tail, 0.0) /
      static_cast<double>(tail);
  return {{"row_count", points.size()},
          {"dataset_count", datasets.size()},
          {"source_group_count", groups.size()},
          {"total_selected_cost", selected},
          {"total_oracle_cost", oracle},
          {"total_cost_ratio", selected / oracle},
          {"total_excess_cost", selected - oracle},
          {"dataset_equal_geometric_regret_ratio",
           std::exp(dataset_logs / static_cast<double>(datasets.size()))},
          {"worst_source_group_regret_ratio", worst_ratio},
          {"worst_source_group", worst_group},
          {"cvar95_relative_excess", cvar},
          {"cvar95_contract",
           "mean of the worst ceil(5%) per-column relative excess costs"}};
}

struct Hyperparameters {
  double lambda = 1.0;
  std::size_t knn_k = 0;
  double ridge_weight = 1.0;
};

std::string hyper_id(const Hyperparameters& value) {
  std::ostringstream out;
  out << "lambda=" << value.lambda << ",k=" << value.knn_k
      << ",ridge=" << value.ridge_weight;
  return out.str();
}

Json hyper_json(const Hyperparameters& value) {
  return {{"ridge_lambda", value.lambda},
          {"knn_k", value.knn_k},
          {"ridge_weight", value.ridge_weight}};
}

std::vector<Hyperparameters> grid() {
  std::vector<Hyperparameters> result;
  for (double lambda : {0.1, 1.0, 10.0}) {
    result.push_back({lambda, 0, 1.0});
    result.push_back({lambda, 8, 0.0});
    result.push_back({lambda, 8, 0.5});
  }
  return result;
}

struct TuningResult {
  Hyperparameters selected;
  Json grid_report;
  Json purge_audit;
  q::QualityModel final_model;
};

TuningResult tune_one(const std::vector<SampleRow>& rows,
                      const CatalogByDtype& catalogs,
                      const std::vector<std::string>& feature_names,
                      Objective objective,
                      const std::string& catalog_sha256,
                      const std::set<std::string>& confirm_legacy_groups) {
  const auto parameters = grid();
  std::map<std::string, std::vector<SelectionPoint>> outcomes;
  std::map<std::string, RealizedSelections> realized;
  std::map<std::string, RankedRows> ranked_rows;
  for (const auto& item : parameters)
    realized.emplace(hyper_id(item), RealizedSelections{});
  for (const auto& item : parameters)
    ranked_rows.emplace(hyper_id(item), RankedRows{});
  Json purge_audit = Json::array();
  for (const auto& item : parameters) outcomes.emplace(hyper_id(item), std::vector<SelectionPoint>{});
  for (std::size_t fold = 0; fold < kFoldCount; ++fold) {
    bool found_validation = false;
    Json fold_audit;
    const auto train = training_rows(rows, catalogs, confirm_legacy_groups,
                                     fold, &fold_audit);
    purge_audit.push_back(std::move(fold_audit));
    require(!train.empty(), "development fold has no training rows");
    for (double lambda : {0.1, 1.0, 10.0}) {
      q::FitConfig config;
      config.ridge_lambda = lambda;
      config.knn_k = 0;
      config.ridge_weight = 1.0;
      config.feature_names = feature_names;
      config.candidate_catalog_sha256 = catalog_sha256;
      const auto model = q::fit(train, q::Domain::Algorithm,
                                quality_objective(objective), config);
      for (const auto& row : rows) {
        if (row.fold != fold) continue;
        found_validation = true;
        const auto ids = eligible(row, catalogs);
        const auto oracle = oracle_cost(row, ids, objective);
        // One k=8 call exposes both ridge and kNN component scores for every
        // candidate. Reuse it for ridge weights 0, 0.5, and 1 instead of
        // repeating the expensive distance ordering three times.
        const auto component_ranking = model.rank_with_scoring(
            row.feature->features, ids, 8, 0.5);
        require(!component_ranking.ranking.empty(), "OOF ranking is empty");
        for (const auto& parameter : parameters) {
          if (parameter.lambda != lambda) continue;
          std::string winner;
          double winner_score = std::numeric_limits<double>::infinity();
          std::vector<std::pair<double, std::string>> scored;
          for (const auto& candidate : component_ranking.ranking) {
            require(candidate.knn_log_cost.has_value(),
                    "full-support candidate lacks kNN score");
            const double score =
                parameter.ridge_weight * candidate.ridge_log_cost +
                (1.0 - parameter.ridge_weight) * *candidate.knn_log_cost;
            scored.emplace_back(score, candidate.candidate_id);
            if (score < winner_score ||
                (score == winner_score && candidate.candidate_id < winner)) {
              winner_score = score;
              winner = candidate.candidate_id;
            }
          }
          require(!winner.empty(), "OOF component ranking has no winner");
          std::sort(scored.begin(), scored.end(), [](const auto& left,
                                                     const auto& right) {
            return left.first < right.first ||
                   (left.first == right.first && left.second < right.second);
          });
          RankedRow ranked{&row, {}};
          const auto keep = std::min<std::size_t>(10, scored.size());
          ranked.ranking.reserve(keep);
          for (std::size_t index = 0; index < keep; ++index)
            ranked.ranking.push_back(scored[index].second);
          ranked_rows.at(hyper_id(parameter)).push_back(std::move(ranked));
          const auto selected = cost(row.methods.at(winner), objective);
          outcomes.at(hyper_id(parameter))
              .push_back({row.dataset, row.source_group, selected, oracle});
          realized.at(hyper_id(parameter))
              .push_back({&row, &row.methods.at(winner)});
        }
      }
    }
    require(found_validation, "development fold has no validation rows");
  }

  const Hyperparameters baseline{1.0, 0, 1.0};
  const auto baseline_metrics = risk_metrics(outcomes.at(hyper_id(baseline)));
  Json report = Json::array();
  std::optional<Hyperparameters> best;
  double best_total = std::numeric_limits<double>::infinity();
  auto no_worse = [&](const Json& metrics) {
    constexpr double tolerance = 1e-12;
    for (const char* field : {"total_cost_ratio",
                              "dataset_equal_geometric_regret_ratio",
                              "worst_source_group_regret_ratio",
                              "cvar95_relative_excess"}) {
      if (metrics.at(field).get<double>() >
          baseline_metrics.at(field).get<double>() + tolerance)
        return false;
    }
    return true;
  };
  for (const auto& parameter : parameters) {
    auto metrics = risk_metrics(outcomes.at(hyper_id(parameter)));
    const bool feasible = objective != Objective::Space || no_worse(metrics);
    report.push_back({{"hyperparameters", hyper_json(parameter)},
                      {"metrics", metrics},
                      {"space_frozen_baseline_feasible", feasible}});
    const double total = metrics.at("total_selected_cost").get<double>();
    if (feasible &&
        (total < best_total ||
         (total == best_total && (!best || hyper_id(parameter) < hyper_id(*best))))) {
      best = parameter;
      best_total = total;
    }
  }
  require(best.has_value(), "hyperparameter grid has no feasible selection");

  const auto& chosen_points = outcomes.at(hyper_id(*best));
  const auto& chosen_realized = realized.at(hyper_id(*best));
  require(chosen_points.size() == rows.size() &&
              chosen_realized.size() == rows.size(),
          "OOF selected policy does not cover every development row exactly once");
  RealizedSelections lz4, zstd, scope_oracle, full_oracle;
  double scope_oracle_cost = 0.0, full_oracle_cost = 0.0;
  for (const auto& row : rows) {
    lz4.push_back({&row, &row.methods.at(std::string(kLz4))});
    zstd.push_back({&row, &row.methods.at(std::string(kZstd3))});
    const auto scope_ids = eligible(row, catalogs);
    const auto& scope_best = oracle_method(row, scope_ids, objective);
    std::vector<std::string> full_ids;
    full_ids.reserve(row.methods.size());
    for (const auto& [id, unused] : row.methods) {
      (void)unused;
      full_ids.push_back(id);
    }
    const auto& full_best = oracle_method(row, full_ids, objective);
    scope_oracle.push_back({&row, &scope_best});
    full_oracle.push_back({&row, &full_best});
    scope_oracle_cost += cost(scope_best, objective);
    full_oracle_cost += cost(full_best, objective);
  }
  const double selected_oof_cost =
      risk_metrics(chosen_points).at("total_selected_cost").get<double>();
  Json oof_evaluation =
      {{"contract",
        "strict five-fold development OOF: every selected candidate comes from the model fitted without its held-out source and legacy groups; all realized metrics use that same candidate"},
       {"selection_objective", objective_name(objective)},
       {"joint_optimum_claimed", false},
       {"budgeted_harness_feedback",
        budget_feedback_evaluation(ranked_rows.at(hyper_id(*best)), catalogs,
                                   objective)},
       {"selector",
        {{"realized_metrics", realized_metrics(chosen_realized)},
         {"risk_vs_scope_oracle", risk_metrics(chosen_points)},
         {"selection_loss_vs_scope_oracle",
          selected_oof_cost / scope_oracle_cost - 1.0},
         {"scope_restriction_vs_experiment_scope_full85_algorithm_oracle",
          scope_oracle_cost / full_oracle_cost - 1.0},
         {"total_gap_to_experiment_scope_full85_algorithm_oracle",
          selected_oof_cost / full_oracle_cost - 1.0}}},
       {"fixed_lz4", {{"realized_metrics", realized_metrics(lz4)}}},
       {"fixed_zstd3", {{"realized_metrics", realized_metrics(zstd)}}},
       {"scope_oracle",
        {{"selection_objective", objective_name(objective)},
         {"independent_objective_oracle", true},
         {"realized_metrics", realized_metrics(scope_oracle)}}},
       {"experiment_scope_full85_algorithm_oracle",
        {{"selection_objective", objective_name(objective)},
         {"independent_objective_oracle", true},
         {"reference_only", true},
         {"includes_forbidden_ppmd8", true},
         {"realized_metrics", realized_metrics(full_oracle)}}}};

  q::FitConfig config;
  config.ridge_lambda = best->lambda;
  config.knn_k = best->knn_k;
  config.ridge_weight = best->ridge_weight;
  config.feature_names = feature_names;
  config.candidate_catalog_sha256 = catalog_sha256;
  Json final_audit;
  auto final_model = q::fit(training_rows(rows, catalogs, confirm_legacy_groups,
                                          std::nullopt, &final_audit),
                            q::Domain::Algorithm,
                            quality_objective(objective), config);
  return {*best,
          {{"objective", objective_name(objective)},
           {"selection_rule",
            objective == Objective::Space
                ? "minimum OOF total bytes among grid points no worse than the frozen ridge-only baseline on all four risk statistics"
                : "minimum OOF total objective cost; objective is tuned independently"},
           {"frozen_baseline", hyper_json(baseline)},
           {"frozen_baseline_metrics", baseline_metrics},
           {"selected", hyper_json(*best)},
           {"selected_oof_evaluation", std::move(oof_evaluation)},
           {"grid", std::move(report)}},
          { {"oof_folds", std::move(purge_audit)},
            {"final_fit", std::move(final_audit)},
            {"contract", "confirm legacy groups are purged from every fit; each OOF fit additionally purges every source and legacy group represented by its held-out fold"}},
          std::move(final_model)};
}

Json candidate_contract(const std::vector<std::string>& full,
                        const std::vector<std::string>& primary,
                        const std::vector<std::string>& restricted,
                        const CatalogByDtype* primary_by_dtype = nullptr,
                        const CatalogByDtype* restricted_by_dtype = nullptr,
                        const CandidateSpec* frozen = nullptr) {
  return {{"experiment_scope_full85_algorithm_reference_only", full},
          {"primary_selectable84", primary},
          {"restricted3_ablation", restricted},
          {"forbidden_selectable_families",
           {"NNCP", "PPMd/PPMD8", "DMC", "PAQ8", "CMIX", "LSTM"}},
          {"development_full_support_by_dtype",
           primary_by_dtype == nullptr
               ? Json(nullptr)
               : Json{{"primary84", *primary_by_dtype},
                      {"restricted3", *restricted_by_dtype}}},
          {"heldout_status_used_for_inference_eligibility", false},
          {"joint_optimum_claimed", false},
          {"bos_contract",
           "BOS is one in-place candidate ID; deployment runtime routes are distinct from benchmark measurement provenance, whose accepted source/result/profile/timing/binary/route fields are validated fail-closed for every feed row"},
          {"bos_runtime_replacement_routes",
           frozen == nullptr
               ? Json(nullptr)
               : Json{{"float", frozen->bos_float_route},
                      {"int64", frozen->bos_int64_route}}},
          {"bos_benchmark_measurement_sources",
           frozen == nullptr
               ? Json(nullptr)
               : [&]() {
                   Json values = Json::array();
                   for (const auto& [source, spec] : frozen->bos_measurements)
                     values.push_back(
                         {{"source", source},
                          {"source_result_sha256", spec.source_result_sha256},
                          {"input_profile", spec.input_profile},
                          {"timing_contract", spec.timing_contract},
                          {"allowed_binary_sha256", spec.binary_sha256},
                          {"allowed_route_ids", spec.route_ids},
                          {"allowed_profile_versions", spec.profile_versions},
                          {"paired_acceptance_sha256",
                           spec.paired_acceptance_sha256.empty()
                               ? Json(nullptr)
                               : Json(spec.paired_acceptance_sha256)}});
                   return values;
                 }()}};
}

std::string model_filename(const std::string& scope, Objective objective) {
  return "MODEL_" + scope + "_" + std::string(objective_name(objective)) + ".json";
}

struct LoadedModel {
  std::string scope;
  Objective objective;
  CatalogByDtype catalogs;
  q::QualityModel model;
};

std::vector<LoadedModel> verify_and_load_models(const fs::path& directory,
                                                Json& lock_out,
                                                Json& index_out) {
  const auto lock_path = directory / "LOCK.json";
  const auto index_path = directory / "MODEL_INDEX.json";
  require(fs::exists(lock_path),
          "final-eval requires LOCK.json before any final labels are read");
  lock_out = d::read_json(lock_path);
  require(lock_out.value("state", std::string()) ==
              "locked_before_confirm_or_external_labels" &&
              !lock_out.value("confirm_labels_read", true) &&
              !lock_out.value("external_final_labels_read", true),
          "model lock does not prove label isolation");
  require(lock_out.at("model_index_sha256").get<std::string>() ==
              d::file_sha(index_path),
          "MODEL_INDEX hash differs from lock");
  index_out = d::read_json(index_path);
  std::vector<LoadedModel> result;
  for (const auto& item : index_out.at("models")) {
    const fs::path path = directory / item.at("path").get<std::string>();
    require(d::file_sha(path) == item.at("sha256").get<std::string>(),
            "model hash differs from lock/index: " + path.string());
    Objective objective;
    const auto objective_text = item.at("objective").get<std::string>();
    if (objective_text == "space") objective = Objective::Space;
    else if (objective_text == "encode") objective = Objective::Encode;
    else if (objective_text == "decode") objective = Objective::Decode;
    else fail("unknown model objective " + objective_text);
    const auto scope = item.at("scope").get<std::string>();
    const auto catalogs = index_out.at("candidate_scopes_by_dtype").at(scope)
                              .get<CatalogByDtype>();
    result.push_back({scope, objective, catalogs,
                      q::QualityModel::from_json(d::read_json(path))});
  }
  require(result.size() == 6, "model index must contain six prepared models");
  return result;
}

const LoadedModel* find_model(const std::vector<LoadedModel>& models,
                              const std::string& scope,
                              Objective objective) {
  for (const auto& model : models)
    if (model.scope == scope && model.objective == objective) return &model;
  fail("model missing for " + scope + "/" + std::string(objective_name(objective)));
}

struct PreparedModel {
  explicit PreparedModel(const LoadedModel& source) : model(source.model) {}
  std::string rank_first(const std::vector<double>& features,
                         const std::vector<std::string>& eligible_ids) const {
    const auto ranking = model.rank(features, eligible_ids);
    require(!ranking.ranking.empty(), "prepared ranking is empty");
    return ranking.ranking.front().candidate_id;
  }
  q::QualityModel model;
};

struct PreparedRegistry {
  explicit PreparedRegistry(CatalogByDtype locked) : catalogs(std::move(locked)) {
    require(!catalogs.empty(), "PreparedRegistry has no dtype catalogs");
    for (const auto& [dtype, ids] : catalogs) {
      require(dtype == "float32" || dtype == "float64" || dtype == "int64",
              "PreparedRegistry contains unknown dtype " + dtype);
      require(!ids.empty() && std::is_sorted(ids.begin(), ids.end()) &&
                  std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
              "PreparedRegistry candidate IDs must be nonempty, sorted, unique");
    }
  }

  const std::vector<std::string>& validate_and_resolve(
      const FeatureRow& row) const {
    require(row.dtype == "float32" || row.dtype == "float64" ||
                row.dtype == "int64",
            "PreparedRegistry input has unsupported dtype");
    require((row.dtype == "float32" && row.width == 4) ||
                ((row.dtype == "float64" || row.dtype == "int64") &&
                 row.width == 8),
            "PreparedRegistry input dtype/width mismatch");
    require(row.row_count > 0 && row.valid_count > 0 &&
                row.valid_count <= row.row_count,
            "PreparedRegistry input row/valid counts are invalid");
    require(row.features.size() ==
                kFeatureMapCount + kMissingIndicatorCount +
                    kDerivedFeatureCount,
            "PreparedRegistry input feature shape differs from frozen 61");
    require(std::all_of(row.features.begin(), row.features.end(),
                        [](double value) { return std::isfinite(value); }),
            "PreparedRegistry input feature vector is non-finite");
    const auto it = catalogs.find(row.dtype);
    require(it != catalogs.end(),
            "PreparedRegistry has no locked candidates for " + row.dtype);
    return it->second;
  }

  // This registry deliberately contains only the development-locked,
  // per-dtype candidate IDs. It does not import the operator97 registry or run
  // any raw-column feature extractor.
  CatalogByDtype catalogs;
};

Json timing_benchmark(const std::vector<LoadedModel>& models,
                      const std::vector<SampleRow>& rows, std::size_t repeats) {
  require(repeats >= 5, "timing_repeats must be at least 5");
  require(!rows.empty(), "timing benchmark has no held-out rows");
  Json output = Json::array();
  volatile std::size_t sink = 0;
  auto summarize = [](std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto total = std::accumulate(values.begin(), values.end(), 0.0);
    return Json{{"min_ns", values.front()},
                {"median_ns", values[values.size() / 2]},
                {"p95_ns", values[(values.size() * 95) / 100]},
                {"mean_ns", total / static_cast<double>(values.size())}};
  };
  for (const auto& source : models) {
    PreparedModel prepared(source);
    PreparedRegistry registry(source.catalogs);
    std::vector<const std::vector<std::string>*> ids;
    ids.reserve(rows.size());
    for (const auto& row : rows)
      ids.push_back(&registry.validate_and_resolve(*row.feature));
    sink = sink +
           prepared.rank_first(rows.front().feature->features, *ids.front()).size();
    sink = sink + registry.validate_and_resolve(*rows.front().feature).size();
    sink = sink + prepared
                      .rank_first(rows.front().feature->features,
                                  registry.validate_and_resolve(*rows.front().feature))
                      .size();
    std::vector<std::uint64_t> registry_wall, registry_cpu, rank_wall, rank_cpu,
        end_to_end_wall, end_to_end_cpu;
    for (auto* values : {&registry_wall, &registry_cpu, &rank_wall, &rank_cpu,
                         &end_to_end_wall, &end_to_end_cpu})
      values->reserve(repeats);
    for (std::size_t i = 0; i < repeats; ++i) {
      const auto& row = rows.at(i % rows.size());
      auto wall_start = d::wall_ns();
      auto cpu_start = d::cpu_ns();
      sink = sink + registry.validate_and_resolve(*row.feature).size();
      registry_cpu.push_back(d::cpu_ns() - cpu_start);
      registry_wall.push_back(d::wall_ns() - wall_start);

      wall_start = d::wall_ns();
      cpu_start = d::cpu_ns();
      sink = sink +
             prepared.rank_first(row.feature->features,
                                 *ids.at(i % rows.size()))
                 .size();
      rank_cpu.push_back(d::cpu_ns() - cpu_start);
      rank_wall.push_back(d::wall_ns() - wall_start);

      wall_start = d::wall_ns();
      cpu_start = d::cpu_ns();
      const auto& resolved = registry.validate_and_resolve(*row.feature);
      sink = sink + prepared.rank_first(row.feature->features, resolved).size();
      end_to_end_cpu.push_back(d::cpu_ns() - cpu_start);
      end_to_end_wall.push_back(d::wall_ns() - wall_start);
    }
    output.push_back({{"scope", source.scope},
                      {"objective", objective_name(source.objective)},
                      {"candidate_count_min", [&] {
                         std::size_t value = std::numeric_limits<std::size_t>::max();
                         for (const auto* item : ids) value = std::min(value, item->size());
                         return value;
                       }()},
                      {"candidate_count_max", [&] {
                         std::size_t value = 0;
                         for (const auto* item : ids) value = std::max(value, item->size());
                         return value;
                       }()},
                      {"heldout_samples_cycled", std::min(repeats, rows.size())},
                      {"warmups", 1},
                      {"repeats", repeats},
                      {"wall", summarize(rank_wall)},
                      {"thread_cpu", summarize(rank_cpu)},
                      {"phases",
                       {{"prepared_registry_plus_feature_row_validation",
                         {{"wall", summarize(registry_wall)},
                          {"thread_cpu", summarize(registry_cpu)}}},
                        {"prepared_model_hot_full_rank",
                         {{"wall", summarize(rank_wall)},
                          {"thread_cpu", summarize(rank_cpu)}}},
                        {"validation_registry_rank_end_to_end",
                         {{"wall", summarize(end_to_end_wall)},
                          {"thread_cpu", summarize(end_to_end_cpu)}}}}}});
  }
  require(sink != 0, "timing sink unexpectedly zero");
  return {{"contract",
           "PreparedModel and PreparedRegistry loaded once; each timed input is an already-materialized frozen FeatureRow; registry+validation, hot full rank, and their end-to-end composition are timed separately while cycling held-out columns"},
          {"raw_column_feature_extraction_included", false},
          {"json_feature_read_or_generation_included", false},
          {"feature_boundary",
           "precomputed 61-value FeatureRow; no raw-column feature extractor is invoked or represented by these timings"},
          {"registry_contract",
           "PreparedRegistry contains only development-locked per-dtype eligible algorithm IDs; it does not use the operator97 registry"},
          {"legacy_rank_only_fields",
           "measurement.wall and measurement.thread_cpu retain the original PreparedModel hot full-rank boundary"},
          {"heldout_sample_count", rows.size()},
          {"measurements", std::move(output)}};
}

Json aggregate_entry(double selected, const std::vector<SampleRow>& rows,
                     Objective objective) {
  std::uint64_t original = 0;
  std::uint64_t values = 0;
  for (const auto& row : rows) {
    original += row.original_bytes;
    values += row.valid_values;
  }
  Json result{{"total_cost", selected}};
  if (objective == Objective::Space) {
    result["compressed_bytes"] = selected;
    result["original_numeric_bytes"] = original;
    result["compression_ratio"] = selected / static_cast<double>(original);
  } else {
    result["total_time_ns"] = selected;
    result["valid_values"] = values;
    result["ns_per_value"] = selected / static_cast<double>(values);
  }
  return result;
}

const MethodCost& oracle_method(const SampleRow& row,
                                const std::vector<std::string>& catalog,
                                Objective objective) {
  const MethodCost* best = nullptr;
  for (const auto& id : catalog) {
    const auto& item = row.methods.at(id);
    if (!item.eligible) continue;
    if (best == nullptr || cost(item, objective) < cost(*best, objective) ||
        (cost(item, objective) == cost(*best, objective) && item.id < best->id))
      best = &item;
  }
  require(best != nullptr, "sample has no oracle method");
  return *best;
}

Json realized_metrics(
    const std::vector<std::pair<const SampleRow*, const MethodCost*>>& selected) {
  require(!selected.empty(), "cannot aggregate an empty realized selection");
  struct Totals {
    double bytes = 0.0;
    double original = 0.0;
    double source_native = 0.0;
    double values = 0.0;
    double encode = 0.0;
    double decode = 0.0;
  } total;
  std::map<std::string, Totals> datasets;
  std::map<std::string, std::size_t> candidates;
  double column_ratio = 0.0, column_source_ratio = 0.0;
  double column_encode = 0.0, column_decode = 0.0;
  std::string audit;
  for (const auto& [row, method] : selected) {
    require(method->eligible, "realized selection is not accepted/lossless");
    Totals item{method->bytes, static_cast<double>(row->original_bytes),
                static_cast<double>(row->source_native_bytes),
                static_cast<double>(row->valid_values), method->encode_ns,
                method->decode_ns};
    total.bytes += item.bytes;
    total.original += item.original;
    total.source_native += item.source_native;
    total.values += item.values;
    total.encode += item.encode;
    total.decode += item.decode;
    auto& dataset = datasets[row->dataset];
    dataset.bytes += item.bytes;
    dataset.original += item.original;
    dataset.source_native += item.source_native;
    dataset.values += item.values;
    dataset.encode += item.encode;
    dataset.decode += item.decode;
    column_ratio += item.bytes / item.original;
    column_source_ratio += item.bytes / item.source_native;
    column_encode += item.encode / item.values;
    column_decode += item.decode / item.values;
    ++candidates[method->id];
    audit.append(row->sample).push_back('\t');
    audit.append(method->id).push_back('\n');
  }
  double dataset_ratio = 0.0, dataset_source_ratio = 0.0;
  double dataset_encode = 0.0, dataset_decode = 0.0;
  for (const auto& [unused, item] : datasets) {
    (void)unused;
    dataset_ratio += item.bytes / item.original;
    dataset_source_ratio += item.bytes / item.source_native;
    dataset_encode += item.encode / item.values;
    dataset_decode += item.decode / item.values;
  }
  return {{"row_count", selected.size()},
          {"dataset_count", datasets.size()},
          {"compressed_bytes", total.bytes},
          {"canonical_input_bytes", total.original},
          {"compression_ratio", total.bytes / total.original},
          {"source_native_input_bytes", total.source_native},
          {"compressed_to_source_native_reference_ratio",
           total.bytes / total.source_native},
          {"source_native_ratio_contract",
           "accounting reference only; int8/int16-to-int64 widening is never counted as compression gain"},
          {"compression_time_ns", total.encode},
          {"decompression_time_ns", total.decode},
          {"valid_values", total.values},
          {"compression_time_ns_per_value", total.encode / total.values},
          {"decompression_time_ns_per_value", total.decode / total.values},
          {"column_equal_arithmetic",
           {{"compression_ratio", column_ratio / selected.size()},
            {"compressed_to_source_native_reference_ratio",
             column_source_ratio / selected.size()},
            {"compression_time_ns_per_value", column_encode / selected.size()},
            {"decompression_time_ns_per_value", column_decode / selected.size()}}},
          {"dataset_equal_arithmetic",
           {{"compression_ratio", dataset_ratio / datasets.size()},
            {"compressed_to_source_native_reference_ratio",
             dataset_source_ratio / datasets.size()},
            {"compression_time_ns_per_value", dataset_encode / datasets.size()},
            {"decompression_time_ns_per_value", dataset_decode / datasets.size()}}},
          {"selected_candidate_counts", candidates},
          {"selection_identity_sha256", d::sha(audit)}};
}

Json budget_feedback_evaluation(const RankedRows& rankings,
                                const CatalogByDtype& catalogs,
                                Objective objective) {
  require(!rankings.empty(), "budget feedback has no rankings");
  Json budgets = Json::array();
  for (const std::size_t requested_budget : {1U, 3U, 5U, 10U}) {
    RealizedSelections chosen;
    std::vector<SelectionPoint> points;
    double scope_oracle_total = 0.0, full_oracle_total = 0.0;
    double probe_encode_ns = 0.0, probe_decode_ns = 0.0;
    std::uint64_t total_values = 0;
    std::size_t probe_executions = 0;
    std::set<std::size_t> effective_budgets;
    for (const auto& ranked : rankings) {
      require(ranked.row != nullptr && !ranked.ranking.empty(),
              "budget feedback ranking is empty");
      const auto& row = *ranked.row;
      const auto effective = std::min(requested_budget, ranked.ranking.size());
      effective_budgets.insert(effective);
      const MethodCost* winner = nullptr;
      for (std::size_t index = 0; index < effective; ++index) {
        const auto& measured = row.methods.at(ranked.ranking[index]);
        require(measured.eligible,
                "budget feedback candidate lacks accepted held-out measurement");
        probe_encode_ns += measured.encode_ns;
        probe_decode_ns += measured.decode_ns;
        ++probe_executions;
        if (winner == nullptr || cost(measured, objective) < cost(*winner, objective) ||
            (cost(measured, objective) == cost(*winner, objective) &&
             measured.id < winner->id))
          winner = &measured;
      }
      require(winner != nullptr, "budget feedback found no measured candidate");
      const auto scope_ids = eligible(row, catalogs);
      const auto& scope_best = oracle_method(row, scope_ids, objective);
      std::vector<std::string> full_ids;
      for (const auto& [id, unused] : row.methods) {
        (void)unused;
        full_ids.push_back(id);
      }
      const auto& full_best = oracle_method(row, full_ids, objective);
      chosen.push_back({&row, winner});
      points.push_back(
          {row.dataset, row.source_group, cost(*winner, objective),
           cost(scope_best, objective)});
      scope_oracle_total += cost(scope_best, objective);
      full_oracle_total += cost(full_best, objective);
      total_values += row.valid_values;
    }
    const auto risk = risk_metrics(points);
    const auto selected_total = risk.at("total_selected_cost").get<double>();
    budgets.push_back(
        {{"requested_budget", requested_budget},
         {"effective_budget_values", effective_budgets},
         {"semantics", requested_budget == 1
                           ? "top-1 recommender selection"
                           : "ranked-prefix candidates measured by the harness; choose the best observed target cost within the frozen budget"},
         {"realized_metrics", realized_metrics(chosen)},
         {"risk_vs_scope_oracle", risk},
         {"selection_loss_vs_scope_oracle",
          selected_total / scope_oracle_total - 1.0},
         {"scope_restriction_vs_experiment_scope_full85_algorithm_oracle",
          scope_oracle_total / full_oracle_total - 1.0},
         {"total_gap_to_experiment_scope_full85_algorithm_oracle",
          selected_total / full_oracle_total - 1.0},
         {"harness_probe_cost",
          {{"candidate_executions", probe_executions},
           {"compression_time_ns", probe_encode_ns},
           {"decompression_time_ns", probe_decode_ns},
           {"compression_time_ns_per_input_value",
            probe_encode_ns / static_cast<double>(total_values)},
           {"decompression_time_ns_per_input_value",
            probe_decode_ns / static_cast<double>(total_values)}}}});
  }
  return {{"budget_set_frozen_on_development", {1, 3, 5, 10}},
          {"feedback_source",
           "accepted lossless per-column measurements; full85 is used only for hindsight gap accounting"},
          {"budgets", std::move(budgets)}};
}

Json evaluate_objective(const std::vector<SampleRow>& rows,
                        const FeedLoad& feed,
                        const std::vector<LoadedModel>& models,
                        Objective objective) {
  const auto* const primary_model = find_model(models, "primary84", objective);
  const auto* const restricted_model =
      find_model(models, "restricted3", objective);
  std::vector<SelectionPoint> primary_points;
  std::vector<SelectionPoint> restricted_points;
  double primary_total = 0.0, restricted_total = 0.0;
  double lz4_total = 0.0, zstd_total = 0.0;
  double pool_oracle_total = 0.0, restricted_oracle_total = 0.0,
         full_oracle_total = 0.0;
  using Selections =
      std::vector<std::pair<const SampleRow*, const MethodCost*>>;
  Selections primary_selected, restricted_selected, lz4_selected, zstd_selected,
      pool_oracle_selected, restricted_oracle_selected, full_oracle_selected;
  RankedRows primary_ranked, restricted_ranked;
  for (const auto& row : rows) {
    const auto primary_ids = eligible(row, primary_model->catalogs);
    const auto restricted_ids = eligible(row, restricted_model->catalogs);
    const auto primary_ranking = primary_model->model.rank(
        row.feature->features, primary_ids);
    const auto restricted_ranking = restricted_model->model.rank(
        row.feature->features, restricted_ids);
    require(!primary_ranking.ranking.empty() && !restricted_ranking.ranking.empty(),
            "final ranking is empty");
    const auto& primary_measurement =
        row.methods.at(primary_ranking.ranking.front().candidate_id);
    const auto& restricted_measurement =
        row.methods.at(restricted_ranking.ranking.front().candidate_id);
    require(primary_measurement.eligible,
            "locked primary candidate selected on held-out row but has no "
            "accepted measurement: " + primary_measurement.id);
    require(restricted_measurement.eligible,
            "locked restricted candidate selected on held-out row but has no "
            "accepted measurement: " + restricted_measurement.id);
    const auto primary = cost(primary_measurement, objective);
    const auto restricted = cost(restricted_measurement, objective);
    const auto pool_oracle = oracle_cost(row, feed.primary_catalog, objective);
    const auto restricted_oracle =
        oracle_cost(row, feed.restricted_catalog, objective);
    const auto full_oracle = oracle_cost(row, feed.full_catalog, objective);
    const auto& pool_oracle_method =
        oracle_method(row, feed.primary_catalog, objective);
    const auto& restricted_oracle_method =
        oracle_method(row, feed.restricted_catalog, objective);
    const auto& full_oracle_method =
        oracle_method(row, feed.full_catalog, objective);
    primary_points.push_back(
        {row.dataset, row.source_group, primary, pool_oracle});
    restricted_points.push_back(
        {row.dataset, row.source_group, restricted, restricted_oracle});
    primary_total += primary;
    restricted_total += restricted;
    lz4_total += cost(row.methods.at(std::string(kLz4)), objective);
    zstd_total += cost(row.methods.at(std::string(kZstd3)), objective);
    pool_oracle_total += pool_oracle;
    restricted_oracle_total += restricted_oracle;
    full_oracle_total += full_oracle;
    primary_selected.push_back({&row, &primary_measurement});
    restricted_selected.push_back({&row, &restricted_measurement});
    lz4_selected.push_back({&row, &row.methods.at(std::string(kLz4))});
    zstd_selected.push_back({&row, &row.methods.at(std::string(kZstd3))});
    pool_oracle_selected.push_back({&row, &pool_oracle_method});
    restricted_oracle_selected.push_back({&row, &restricted_oracle_method});
    full_oracle_selected.push_back({&row, &full_oracle_method});
    RankedRow primary_rank{&row, {}};
    for (const auto& item : primary_ranking.ranking)
      primary_rank.ranking.push_back(item.candidate_id);
    primary_ranked.push_back(std::move(primary_rank));
    RankedRow restricted_rank{&row, {}};
    for (const auto& item : restricted_ranking.ranking)
      restricted_rank.ranking.push_back(item.candidate_id);
    restricted_ranked.push_back(std::move(restricted_rank));
  }
  auto primary_risk = risk_metrics(primary_points);
  auto restricted_risk = risk_metrics(restricted_points);
  return {
      {"unit", objective == Objective::Space ? "bytes" : "nanoseconds"},
      {"selection_objective", objective_name(objective)},
      {"realized_metric_contract",
       "every selector/oracle entry reports bytes, encode time, and decode time for the same per-row chosen candidate; the three objective-specific oracles are independent and are not a joint optimum"},
      {"primary84_selector",
       {{"aggregate", aggregate_entry(primary_total, rows, objective)},
        {"realized_metrics", realized_metrics(primary_selected)},
        {"budgeted_harness_feedback",
         budget_feedback_evaluation(primary_ranked, primary_model->catalogs,
                                    objective)},
        {"risk_vs_pool84_oracle", primary_risk},
        {"selection_loss", primary_total / pool_oracle_total - 1.0},
        {"selection_loss_vs_pool84_oracle",
         primary_total / pool_oracle_total - 1.0},
        {"pool_restriction_gap", pool_oracle_total / full_oracle_total - 1.0},
        {"pool84_restriction_vs_experiment_scope_full85_algorithm_oracle",
         pool_oracle_total / full_oracle_total - 1.0},
        {"total_gap_to_experiment_scope_full85_algorithm_oracle",
         primary_total / full_oracle_total - 1.0}}},
      {"restricted3_selector",
       {{"aggregate", aggregate_entry(restricted_total, rows, objective)},
        {"realized_metrics", realized_metrics(restricted_selected)},
        {"budgeted_harness_feedback",
         budget_feedback_evaluation(restricted_ranked,
                                    restricted_model->catalogs, objective)},
        {"risk_vs_restricted3_oracle", restricted_risk},
        {"selection_loss", restricted_total / restricted_oracle_total - 1.0},
        {"selection_loss_vs_restricted3_oracle",
         restricted_total / restricted_oracle_total - 1.0},
        {"pool_restriction_gap",
         restricted_oracle_total / full_oracle_total - 1.0},
        {"restricted3_restriction_vs_experiment_scope_full85_algorithm_oracle",
         restricted_oracle_total / full_oracle_total - 1.0},
        {"total_gap_to_experiment_scope_full85_algorithm_oracle",
         restricted_total / full_oracle_total - 1.0}}},
      {"fixed_lz4",
       {{"aggregate", aggregate_entry(lz4_total, rows, objective)},
        {"realized_metrics", realized_metrics(lz4_selected)},
        {"total_gap_to_experiment_scope_full85_algorithm_oracle", lz4_total / full_oracle_total - 1.0}}},
      {"fixed_zstd3",
       {{"candidate_id", kZstd3},
        {"aggregate", aggregate_entry(zstd_total, rows, objective)},
        {"realized_metrics", realized_metrics(zstd_selected)},
        {"total_gap_to_experiment_scope_full85_algorithm_oracle", zstd_total / full_oracle_total - 1.0}}},
      {"pool84_oracle",
       {{"aggregate", aggregate_entry(pool_oracle_total, rows, objective)},
        {"selection_objective", objective_name(objective)},
        {"realized_metrics", realized_metrics(pool_oracle_selected)},
        {"independent_objective_oracle", true},
        {"reference_only", false}}},
      {"restricted3_oracle",
       {{"aggregate", aggregate_entry(restricted_oracle_total, rows, objective)},
        {"selection_objective", objective_name(objective)},
        {"realized_metrics", realized_metrics(restricted_oracle_selected)},
        {"independent_objective_oracle", true},
        {"reference_only", false}}},
      {"experiment_scope_full85_algorithm_oracle",
       {{"aggregate", aggregate_entry(full_oracle_total, rows, objective)},
        {"selection_objective", objective_name(objective)},
        {"realized_metrics", realized_metrics(full_oracle_selected)},
        {"independent_objective_oracle", true},
        {"joint_optimum_claimed", false},
        {"reference_only", true},
        {"includes_forbidden_ppmd8", true}}}};
}

}  // namespace

Json tune(const TuneRequest& request) {
  require(request.expected_samples > 0 && request.expected_methods >= 4,
          "invalid tune expected counts");
  require(!fs::exists(request.output_directory) ||
              fs::is_empty(request.output_directory),
          "tune output directory must be absent or empty");
  fs::create_directories(request.output_directory);
  const auto split = load_split(request.split_json);
  const auto cohorts = load_cohorts(request.cohorts_json, split);
  const auto candidates =
      load_candidates(request.candidates_json, request.expected_methods);
  const auto features = load_features(request.features_jsonl);
  // This is the only label read in tune. load_feed rejects a row immediately
  // unless its old135 split assignment is development.
  const auto feed = load_feed(request.development_feed_jsonl, features, split,
                              cohorts, candidates,
                              "development", request.expected_samples,
                              request.expected_methods);
  std::set<std::string> confirm_legacy_groups;
  for (const auto& [unused, item] : cohorts) {
    (void)unused;
    if (item.cohort == "confirm")
      confirm_legacy_groups.insert(item.legacy_group);
  }
  require(!confirm_legacy_groups.empty(),
          "frozen cohorts contain no confirm legacy groups");
  const auto primary_by_dtype =
      development_full_support(feed.rows, feed.primary_catalog);
  const auto restricted_by_dtype =
      development_full_support(feed.rows, feed.restricted_catalog);
  for (const auto& [dtype, catalog] : restricted_by_dtype) {
    require(catalog == feed.restricted_catalog,
            "restricted3 is not full-support on development dtype " + dtype);
  }
  if (request.expected_methods == 85) {
    for (const auto& [dtype, catalog] : primary_by_dtype) {
      require(catalog.size() == 84,
              "primary84 is not full-support on development dtype " + dtype);
    }
  }
  const auto primary_sha = d::sha(Json(primary_by_dtype).dump());
  const auto restricted_sha = d::sha(Json(restricted_by_dtype).dump());
  Json cv;
  cv["schema"] = "bos-recommend150-development-oof-grid-v1";
  cv["label_partition"] = "development-only";
  cv["fold_count"] = kFoldCount;
  cv["risk_statistics"] =
      {"total objective cost (space is total bytes)",
       "dataset-equal geometric regret ratio",
       "worst source-group regret ratio",
       "CVaR95 relative excess"};
  cv["scopes"] = Json::object();
  Json model_entries = Json::array();
  for (const auto& [scope, catalogs, sha] :
       {std::tuple<std::string, const CatalogByDtype*, std::string>(
            "primary84", &primary_by_dtype, primary_sha),
        std::tuple<std::string, const CatalogByDtype*, std::string>(
            "restricted3", &restricted_by_dtype, restricted_sha)}) {
    cv["scopes"][scope] = Json::object();
    for (const auto objective : kObjectives) {
      auto result = tune_one(feed.rows, *catalogs, features.names, objective, sha,
                             confirm_legacy_groups);
      cv["scopes"][scope][std::string(objective_name(objective))] =
          result.grid_report;
      cv["scopes"][scope][std::string(objective_name(objective))]
        ["purge_audit"] = result.purge_audit;
      auto model_json = result.final_model.to_json();
      model_json["recommend150"] =
          {{"schema", "bos-recommend150-model-binding-v1"},
           {"scope", scope},
           {"objective", objective_name(objective)},
           {"selected_hyperparameters", hyper_json(result.selected)},
           {"development_only", true},
           {"confirm_labels_read", false},
           {"external_final_labels_read", false}};
      const auto filename = model_filename(scope, objective);
      const auto path = request.output_directory / filename;
      write_new_json(path, model_json);
      model_entries.push_back(
          {{"scope", scope},
           {"objective", objective_name(objective)},
           {"path", filename},
           {"sha256", d::file_sha(path)},
           {"selected_hyperparameters", hyper_json(result.selected)}});
    }
  }
  const auto cv_path = request.output_directory / "CV_SUMMARY.json";
  write_new_json(cv_path, cv);
  Json index{{"schema", "bos-recommend150-model-index-v1"},
             {"models", model_entries},
             {"feature_names", features.names},
             {"feature_transform",
              {{"source_map_fields", 43},
               {"null_value_imputation", 0.0},
               {"null_missing_indicators", 12},
               {"null_contract",
                "only the 12 frozen decimal/nonfinite fields may be null, and only on int64 rows; each is encoded as value=0 plus an explicit missing indicator"},
               {"derived_physical_features", 6}}},
             {"candidate_scopes_by_dtype",
              {{"primary84", primary_by_dtype},
               {"restricted3", restricted_by_dtype}}},
             {"experiment_scope_full85_algorithm_catalog", feed.full_catalog},
             {"candidate_contract",
              candidate_contract(feed.full_catalog, feed.primary_catalog,
                                 feed.restricted_catalog, &primary_by_dtype,
                                 &restricted_by_dtype, &candidates)},
             {"bos_provenance", feed.bos_provenance},
             {"training_sample_count", feed.rows.size()},
             {"training_partition", "development"}};
  const auto index_path = request.output_directory / "MODEL_INDEX.json";
  write_new_json(index_path, index);
  Json lock{{"schema", "bos-recommend150-label-gate-lock-v1"},
            {"state", "locked_before_confirm_or_external_labels"},
            {"confirm_labels_read", false},
            {"external_final_labels_read", false},
            {"features_sha256", d::file_sha(request.features_jsonl)},
            {"split_sha256", d::file_sha(request.split_json)},
            {"cohorts_sha256", d::file_sha(request.cohorts_json)},
            {"candidates_sha256", d::file_sha(request.candidates_json)},
            {"development_feed_sha256",
             d::file_sha(request.development_feed_jsonl)},
            {"cv_summary_sha256", d::file_sha(cv_path)},
            {"model_index_sha256", d::file_sha(index_path)},
            {"primary84_catalog_sha256", primary_sha},
            {"restricted3_catalog_sha256", restricted_sha},
            {"development_sample_count", feed.rows.size()},
            {"note",
             "This immutable hash binding is created by tune before final-eval is permitted to open confirm or external-final feeds."}};
  const auto lock_path = request.output_directory / "LOCK.json";
  write_new_json(lock_path, lock);
  return {{"status", "locked"},
          {"output_directory", fs::absolute(request.output_directory).string()},
          {"development_samples", feed.rows.size()},
          {"primary_candidate_count", feed.primary_catalog.size()},
          {"full_reference_candidate_count", feed.full_catalog.size()},
          {"model_count", model_entries.size()},
          {"lock_sha256", d::file_sha(lock_path)}};
}

Json final_evaluate(const FinalRequest& request) {
  require(request.partition == "confirm" ||
              request.partition == "external-final",
          "final-eval partition must be confirm or external-final");
  require(request.expected_samples > 0 && request.expected_methods >= 4,
          "invalid final expected counts");
  require(!fs::exists(request.output_json),
          "refusing to replace final report " + request.output_json.string());
  // Verify the development-only lock and all model hashes before opening the
  // supplied final-label feed.
  Json lock, index;
  const auto models = verify_and_load_models(request.model_directory, lock, index);
  require(lock.at("features_sha256").get<std::string>() ==
              d::file_sha(request.features_jsonl),
          "final features differ from development lock");
  require(lock.at("split_sha256").get<std::string>() ==
              d::file_sha(request.split_json),
          "final split differs from development lock");
  require(lock.at("cohorts_sha256").get<std::string>() ==
              d::file_sha(request.cohorts_json),
          "final cohorts differ from development lock");
  require(lock.at("candidates_sha256").get<std::string>() ==
              d::file_sha(request.candidates_json),
          "final candidates differ from development lock");
  const auto split = load_split(request.split_json);
  const auto cohorts = load_cohorts(request.cohorts_json, split);
  const auto candidates =
      load_candidates(request.candidates_json, request.expected_methods);
  const auto features = load_features(request.features_jsonl);
  const auto feed = load_feed(request.feed_jsonl, features, split, cohorts,
                              candidates,
                              request.partition, request.expected_samples,
                              request.expected_methods);
  require(feed.full_catalog ==
              index.at("experiment_scope_full85_algorithm_catalog")
                  .get<std::vector<std::string>>(),
          "final full85 reference catalog differs from development lock");
  const auto primary_by_dtype =
      index.at("candidate_scopes_by_dtype").at("primary84")
          .get<CatalogByDtype>();
  const auto restricted_by_dtype =
      index.at("candidate_scopes_by_dtype").at("restricted3")
          .get<CatalogByDtype>();
  for (const auto& [dtype, catalog] : primary_by_dtype)
    for (const auto& candidate : catalog)
      require(std::binary_search(feed.full_catalog.begin(),
                                 feed.full_catalog.end(), candidate),
              "held-out feed lacks locked candidate " + candidate +
                  " for dtype " + dtype);

  Json objectives = Json::object();
  for (const auto objective : kObjectives)
    objectives[std::string(objective_name(objective))] =
        evaluate_objective(feed.rows, feed, models, objective);
  Json report{{"schema", "bos-recommend150-final-evaluation-v1"},
              {"status", "complete"},
              {"partition", request.partition},
              {"final_labels_opened_after_lock_verification", true},
              {"model_lock_sha256",
               d::file_sha(request.model_directory / "LOCK.json")},
              {"feed_sha256", d::file_sha(request.feed_jsonl)},
              {"sample_count", feed.rows.size()},
              {"dataset_count", [&] {
                 std::set<std::string> datasets;
                 for (const auto& row : feed.rows) datasets.insert(row.dataset);
                 return datasets.size();
               }()},
              {"candidate_contract",
               candidate_contract(feed.full_catalog, feed.primary_catalog,
                                  feed.restricted_catalog, &primary_by_dtype,
                                  &restricted_by_dtype, &candidates)},
              {"bos_provenance", feed.bos_provenance},
              {"objectives", std::move(objectives)},
              {"selection_overhead",
               timing_benchmark(models, feed.rows,
                                request.timing_repeats)}};
  write_new_json(request.output_json, report);
  return {{"status", "complete"},
          {"partition", request.partition},
          {"sample_count", feed.rows.size()},
          {"output", fs::absolute(request.output_json).string()},
          {"sha256", d::file_sha(request.output_json)}};
}

}  // namespace bos_recommend150_v1
