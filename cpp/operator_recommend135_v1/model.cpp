#include "model.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace operator_recommend135_v1 {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kNeighbors = 5;
constexpr std::size_t kMaxFeatureWords = 1024;
constexpr std::size_t kMaxFeatureValueBytes = 8192;
constexpr double kDistanceFloor = 1e-12;
constexpr double kTieTolerance = 1e-12;
constexpr std::array<const char*, 4> kTargets = {"space", "encode", "decode",
                                                 "balanced"};
constexpr std::array<std::size_t, 3> kTopKs = {1, 3, 5};

struct DTypeInfo {
  std::size_t width = 0;
  unsigned bits = 0;
  bool is_float = false;
  bool is_signed = false;
  bool little_endian = true;
};

struct Costs {
  double space = 0;
  double encode = 0;
  double decode = 0;
  double balanced = 0;
  double encode_wall = 0;
  double decode_wall = 0;
  double balanced_wall = 0;
};

struct BaselineChoice {
  std::string candidate;
  std::size_t rows = 0;
  double mean_log_cost = 0;
};

std::uint64_t wall_now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::uint64_t cpu_now_ns() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
  timespec value{};
  if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0) {
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
  }
#endif
  return static_cast<std::uint64_t>(
      (static_cast<long double>(std::clock()) * 1000000000.0L) /
      static_cast<long double>(CLOCKS_PER_SEC));
}

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("operator_recommend135_v1: " + message);
}

std::string required_string(const Json& value, const char* key) {
  if (!value.contains(key) || !value.at(key).is_string() ||
      value.at(key).get_ref<const std::string&>().empty()) {
    fail(std::string("missing non-empty string field '") + key + "'");
  }
  return value.at(key).get<std::string>();
}

double required_finite_number(const Json& value, const char* key) {
  if (!value.contains(key) || !value.at(key).is_number()) {
    fail(std::string("missing numeric field '") + key + "'");
  }
  const double result = value.at(key).get<double>();
  if (!std::isfinite(result)) {
    fail(std::string("non-finite numeric field '") + key + "'");
  }
  return result;
}

double optional_finite_number(const Json& value, const char* key,
                              double fallback = 0.0) {
  if (!value.contains(key)) {
    return fallback;
  }
  return required_finite_number(value, key);
}

std::vector<double> feature_vector(const Json& row) {
  if (!row.contains("features") || !row.at("features").is_array()) {
    fail("row is missing array field 'features'");
  }
  const auto features = row.at("features").get<std::vector<double>>();
  if (features.size() != feature_names().size()) {
    fail("feature vector shape does not match feature_names()");
  }
  for (const double value : features) {
    if (!std::isfinite(value)) {
      fail("feature vector contains a non-finite value");
    }
  }
  return features;
}

std::vector<std::string> string_array(const Json& value,
                                      const std::string& context) {
  if (!value.is_array()) {
    fail(context + " must be an array");
  }
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto& item : value) {
    if (!item.is_string() || item.get_ref<const std::string&>().empty()) {
      fail(context + " must contain non-empty strings");
    }
    auto candidate = item.get<std::string>();
    if (seen.insert(candidate).second) {
      result.push_back(std::move(candidate));
    }
  }
  return result;
}

std::optional<std::string> option_catalog_sha256(const Json& options) {
  const bool has_short = options.contains("catalog_sha256");
  const bool has_long = options.contains("candidate_catalog_sha256");
  if (!has_short && !has_long) {
    return std::nullopt;
  }
  if ((has_short && !options.at("catalog_sha256").is_string()) ||
      (has_long && !options.at("candidate_catalog_sha256").is_string())) {
    fail("catalog SHA-256 must be a string");
  }
  const auto value = has_short
                         ? options.at("catalog_sha256").get<std::string>()
                         : options.at("candidate_catalog_sha256")
                               .get<std::string>();
  if (has_short && has_long &&
      value != options.at("candidate_catalog_sha256").get<std::string>()) {
    fail("catalog_sha256 aliases disagree");
  }
  if (value.size() != 64 ||
      !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
      })) {
    fail("catalog SHA-256 must contain exactly 64 hexadecimal characters");
  }
  std::string canonical = value;
  std::transform(canonical.begin(), canonical.end(), canonical.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return canonical;
}

const Json& rows_array(const Json& input) {
  const Json* result = &input;
  if (input.is_object() && input.contains("rows")) {
    result = &input.at("rows");
  }
  if (!result->is_array()) {
    fail("rows must be an array or an object with an array field 'rows'");
  }
  return *result;
}

std::string canonical_dtype(std::string_view dtype) {
  std::string result;
  result.reserve(dtype.size());
  for (const unsigned char c : dtype) {
    if (!std::isspace(c) && c != '_') {
      result.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  return result;
}

DTypeInfo dtype_info(std::string_view dtype) {
  std::string name = canonical_dtype(dtype);
  bool little_endian = true;
  if (!name.empty() && (name.front() == '<' || name.front() == '>' ||
                        name.front() == '=' || name.front() == '|')) {
    little_endian = name.front() != '>';
    name.erase(name.begin());
  }

  const auto make = [little_endian](std::size_t width, bool is_float,
                                    bool is_signed) {
    return DTypeInfo{width, static_cast<unsigned>(width * 8), is_float,
                     is_signed, little_endian};
  };
  if (name == "i8" || name == "int8" || name == "int8t") {
    return make(1, false, true);
  }
  if (name == "u8" || name == "uint8" || name == "uint8t" ||
      name == "byte") {
    return make(1, false, false);
  }
  if (name == "i16" || name == "int16" || name == "int16t") {
    return make(2, false, true);
  }
  if (name == "u16" || name == "uint16" || name == "uint16t") {
    return make(2, false, false);
  }
  if (name == "i32" || name == "int32" || name == "int32t" ||
      name == "int") {
    return make(4, false, true);
  }
  if (name == "u32" || name == "uint32" || name == "uint32t" ||
      name == "uint") {
    return make(4, false, false);
  }
  if (name == "i64" || name == "int64" || name == "int64t" ||
      name == "long") {
    return make(8, false, true);
  }
  if (name == "u64" || name == "uint64" || name == "uint64t" ||
      name == "ulong") {
    return make(8, false, false);
  }
  if (name == "f32" || name == "float32" || name == "float") {
    return make(4, true, true);
  }
  if (name == "f64" || name == "float64" || name == "double") {
    return make(8, true, true);
  }
  fail("unsupported dtype '" + std::string(dtype) + "'");
}

std::uint64_t read_word(std::span<const std::uint8_t> bytes,
                        std::size_t offset, const DTypeInfo& dtype) {
  std::uint64_t result = 0;
  if (dtype.little_endian) {
    for (std::size_t i = 0; i < dtype.width; ++i) {
      result |= static_cast<std::uint64_t>(bytes[offset + i]) << (8 * i);
    }
  } else {
    for (std::size_t i = 0; i < dtype.width; ++i) {
      result = (result << 8) | bytes[offset + i];
    }
  }
  return result;
}

std::uint64_t width_mask(unsigned bits) {
  return bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                    : ((std::uint64_t{1} << bits) - 1);
}

std::size_t width_bin(std::uint64_t value) {
  if (value == 0) {
    return 0;
  }
  const unsigned width = std::bit_width(value);
  if (width <= 8) {
    return 1;
  }
  if (width <= 16) {
    return 2;
  }
  if (width <= 32) {
    return 3;
  }
  return 4;
}

bool finite_float_bits(std::uint64_t word, const DTypeInfo& dtype) {
  if (!dtype.is_float) {
    return true;
  }
  if (dtype.bits == 32) {
    return (word & 0x7f800000ULL) != 0x7f800000ULL;
  }
  if (dtype.bits == 64) {
    return (word & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
  }
  fail("unsupported floating-point width");
}

std::vector<std::string> candidate_scope(const Json& training_rows,
                                         const Json& options) {
  if (options.contains("candidate_ids")) {
    const auto result = string_array(options.at("candidate_ids"),
                                     "options.candidate_ids");
    if (result.empty()) {
      fail("options.candidate_ids must not be empty");
    }
    return result;
  }

  std::set<std::string> ids;
  for (const auto& row : training_rows) {
    if (!row.contains("metrics") || !row.at("metrics").is_object()) {
      fail("training row is missing object field 'metrics'");
    }
    for (const auto& item : row.at("metrics").items()) {
      ids.insert(item.key());
    }
  }
  if (ids.empty()) {
    fail("training rows do not contain any candidate metrics");
  }
  return {ids.begin(), ids.end()};
}

std::optional<Costs> strict_costs(const Json& row,
                                  const std::string& candidate) {
  if (!row.contains("metrics") || !row.at("metrics").is_object()) {
    fail("row is missing object field 'metrics'");
  }
  const auto& metrics = row.at("metrics");
  if (!metrics.contains(candidate)) {
    return std::nullopt;
  }
  const auto& metric = metrics.at(candidate);
  if (!metric.is_object()) {
    fail("candidate metric must be an object");
  }
  if (!metric.contains("status") || !metric.at("status").is_string()) {
    fail("candidate metric is missing string field 'status'");
  }
  const auto status = metric.at("status").get<std::string>();
  if (status != "strict_verified" && status != "wrapper_verified" &&
      status != "domain_rejected" && status != "failed") {
    fail("unknown candidate metric status '" + status + "'");
  }
  if (status != "strict_verified") {
    return std::nullopt;
  }

  const double input = required_finite_number(row, "input_bytes");
  if (!(input > 0)) {
    fail("input_bytes must be positive");
  }
  const double frame = required_finite_number(metric, "frame_bytes");
  const double encode = required_finite_number(metric, "encode_cpu_ns");
  const double decode = required_finite_number(metric, "decode_cpu_ns");
  const double encode_wall =
      required_finite_number(metric, "encode_wall_ns");
  const double decode_wall =
      required_finite_number(metric, "decode_wall_ns");
  if (!(frame > 0) || encode < 0 || decode < 0 || encode_wall < 0 ||
      decode_wall < 0) {
    fail("strict metric costs must have positive frame bytes and non-negative times");
  }
  Costs result;
  result.space = frame / input;
  result.encode = std::max(encode, 1.0) / input;
  result.decode = std::max(decode, 1.0) / input;
  result.balanced = std::exp((std::log(result.space) +
                              std::log(result.encode) +
                              std::log(result.decode)) /
                             3.0);
  result.encode_wall = std::max(encode_wall, 1.0) / input;
  result.decode_wall = std::max(decode_wall, 1.0) / input;
  result.balanced_wall =
      std::exp((std::log(result.space) + std::log(result.encode_wall) +
                std::log(result.decode_wall)) /
               3.0);
  return result;
}

double target_cost(const Costs& costs, std::string_view target) {
  if (target == "space") {
    return costs.space;
  }
  if (target == "encode") {
    return costs.encode;
  }
  if (target == "decode") {
    return costs.decode;
  }
  if (target == "balanced") {
    return costs.balanced;
  }
  fail("unknown target '" + std::string(target) + "'");
}

double target_wall_cost(const Costs& costs, std::string_view target) {
  if (target == "space") {
    return costs.space;
  }
  if (target == "encode") {
    return costs.encode_wall;
  }
  if (target == "decode") {
    return costs.decode_wall;
  }
  if (target == "balanced") {
    return costs.balanced_wall;
  }
  fail("unknown target '" + std::string(target) + "'");
}

bool in_training_eligibility(const Json& row, const std::string& candidate) {
  if (!row.contains("eligible_candidates")) {
    return true;
  }
  const auto eligible =
      string_array(row.at("eligible_candidates"), "row.eligible_candidates");
  return std::find(eligible.begin(), eligible.end(), candidate) != eligible.end();
}

Json labels_for_row(const Json& row,
                    const std::vector<std::string>& candidates) {
  std::map<std::string, Costs> valid;
  for (const auto& candidate : candidates) {
    if (!in_training_eligibility(row, candidate)) {
      continue;
    }
    if (const auto costs = strict_costs(row, candidate)) {
      valid.emplace(candidate, *costs);
    }
  }

  Json result = Json::object();
  if (valid.empty()) {
    return result;
  }
  std::map<std::string, double> minima;
  for (const char* target : kTargets) {
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& [candidate, costs] : valid) {
      (void)candidate;
      minimum = std::min(minimum, target_cost(costs, target));
    }
    minima[target] = minimum;
  }
  for (const auto& [candidate, costs] : valid) {
    Json label = Json::object();
    for (const char* target : kTargets) {
      label[target] =
          std::log(target_cost(costs, target) / minima.at(target));
    }
    result[candidate] = std::move(label);
  }
  return result;
}

std::vector<double> json_finite_vector(const Json& value,
                                       const std::string& context) {
  if (!value.is_array()) {
    fail(context + " must be an array");
  }
  const auto result = value.get<std::vector<double>>();
  for (double item : result) {
    if (!std::isfinite(item)) {
      fail(context + " contains a non-finite value");
    }
  }
  return result;
}

Json top_candidates(const Json& ranking, std::size_t count) {
  Json result = Json::array();
  const auto limit = std::min(count, ranking.size());
  for (std::size_t i = 0; i < limit; ++i) {
    result.push_back(ranking.at(i).at("candidate_id"));
  }
  return result;
}

std::set<std::string> candidate_set(const Json& array,
                                    const std::string& context) {
  const auto values = string_array(array, context);
  return {values.begin(), values.end()};
}

std::vector<std::string> row_eligible_candidates(const Json& row) {
  if (!row.contains("eligible_candidates")) {
    fail("row is missing explicit array field 'eligible_candidates'");
  }
  return string_array(row.at("eligible_candidates"),
                      "row.eligible_candidates");
}

std::vector<std::string> algorithm_candidates(const Json& training_rows,
                                              const Json& options,
                                              const std::vector<std::string>& scope) {
  std::vector<std::string> requested = scope;
  if (options.contains("fixed_algorithm_candidates")) {
    requested = string_array(options.at("fixed_algorithm_candidates"),
                             "options.fixed_algorithm_candidates");
  } else if (options.contains("algorithm_candidates")) {
    requested = string_array(options.at("algorithm_candidates"),
                             "options.algorithm_candidates");
  }

  const std::set<std::string> allowed(scope.begin(), scope.end());
  std::vector<std::string> result;
  for (const auto& candidate : requested) {
    if (!allowed.contains(candidate)) {
      continue;
    }
    bool observed = false;
    bool exact_algorithm = true;
    for (const auto& row : training_rows) {
      if (!row.contains("metrics") || !row.at("metrics").is_object() ||
          !row.at("metrics").contains(candidate)) {
        continue;
      }
      const auto& metric = row.at("metrics").at(candidate);
      if (!metric.is_object() || !metric.contains("kind") ||
          !metric.at("kind").is_string()) {
        exact_algorithm = false;
        break;
      }
      observed = true;
      std::string kind = metric.at("kind").get<std::string>();
      std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (kind != "algorithm") {
        exact_algorithm = false;
        break;
      }
    }
    if (observed && exact_algorithm) {
      result.push_back(candidate);
    }
  }
  return result;
}

std::optional<BaselineChoice> choose_baseline(
    const Json& training_rows, const std::vector<std::string>& candidates,
    std::string_view target) {
  std::optional<BaselineChoice> best;
  for (const auto& candidate : candidates) {
    double sum = 0;
    std::size_t count = 0;
    bool complete = true;
    for (const auto& row : training_rows) {
      const auto costs = strict_costs(row, candidate);
      if (!costs) {
        complete = false;
        break;
      }
      sum += std::log(target_cost(*costs, target));
      ++count;
    }
    if (!complete || count != training_rows.size()) {
      continue;
    }
    const double mean = sum / static_cast<double>(count);
    if (!best || mean < best->mean_log_cost - kTieTolerance ||
        (std::abs(mean - best->mean_log_cost) <= kTieTolerance &&
         candidate < best->candidate)) {
      best = BaselineChoice{candidate, count, mean};
    }
  }
  return best;
}

Json baseline_json(const std::optional<BaselineChoice>& choice) {
  if (!choice) {
    return nullptr;
  }
  return Json{{"candidate_id", choice->candidate},
              {"training_rows", choice->rows},
              {"training_mean_log_cost", choice->mean_log_cost}};
}

bool tied(double left, double right) {
  return std::abs(left - right) <=
         kTieTolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

Json true_cost_map(const Json& row,
                   const std::vector<std::string>& eligible) {
  Json result = Json::object();
  for (const auto& candidate : eligible) {
    if (const auto costs = strict_costs(row, candidate)) {
      Json candidate_costs = Json::object();
      for (const char* target : kTargets) {
        candidate_costs[target] = target_cost(*costs, target);
        if (std::string_view(target) != "space") {
          candidate_costs[std::string(target) + "_wall"] =
              target_wall_cost(*costs, target);
        }
      }
      result[candidate] = std::move(candidate_costs);
    }
  }
  return result;
}

Json pareto_truth(const Json& costs) {
  std::vector<std::string> ids;
  for (const auto& item : costs.items()) {
    ids.push_back(item.key());
  }
  std::sort(ids.begin(), ids.end());
  std::set<std::string> dominated;
  for (const auto& candidate : ids) {
    const auto& c = costs.at(candidate);
    for (const auto& other : ids) {
      if (candidate == other) {
        continue;
      }
      const auto& o = costs.at(other);
      bool no_worse = true;
      bool strictly_better = false;
      for (const char* target : {"space", "encode", "decode"}) {
        const double ov = o.at(target).get<double>();
        const double cv = c.at(target).get<double>();
        if (ov > cv && !tied(ov, cv)) {
          no_worse = false;
          break;
        }
        if (ov < cv && !tied(ov, cv)) {
          strictly_better = true;
        }
      }
      if (no_worse && strictly_better) {
        dominated.insert(candidate);
        break;
      }
    }
  }
  Json frontier = Json::array();
  Json dominated_json = Json::array();
  for (const auto& id : ids) {
    (dominated.contains(id) ? dominated_json : frontier).push_back(id);
  }
  return Json{{"candidates", std::move(frontier)},
              {"dominated_candidates", std::move(dominated_json)}};
}

Json baseline_score(const Json& row, const Json& true_costs,
                    const Json& baseline, std::string_view target,
                    double oracle, double wall_oracle) {
  if (baseline.is_null()) {
    return nullptr;
  }
  const auto id = baseline.at("candidate_id").get<std::string>();
  if (!true_costs.contains(id)) {
    return Json{{"candidate_id", id}, {"available", false}};
  }
  const double cost = true_costs.at(id).at(std::string(target)).get<double>();
  const auto wall_key = target == "space"
                            ? std::string("space")
                            : std::string(target) + "_wall";
  const double wall_cost = true_costs.at(id).at(wall_key).get<double>();
  Json result{{"candidate_id", id},
              {"available", true},
              {"cost", cost},
              {"multiplicative_regret", cost / oracle},
              {"wall_cost", wall_cost},
              {"wall_multiplicative_regret", wall_cost / wall_oracle}};
  (void)row;
  return result;
}

Json score_target(const Json& row, const Json& target_prediction,
                  const Json& true_costs, const Json& pareto,
                  std::string_view target, const Json& baselines,
                  double online_cpu_ns, double online_wall_ns) {
  Json result = Json::object();
  double oracle = std::numeric_limits<double>::infinity();
  double wall_oracle = std::numeric_limits<double>::infinity();
  const auto wall_key = target == "space"
                            ? std::string("space")
                            : std::string(target) + "_wall";
  for (const auto& item : true_costs.items()) {
    oracle = std::min(
        oracle, item.value().at(std::string(target)).get<double>());
    wall_oracle =
        std::min(wall_oracle, item.value().at(wall_key).get<double>());
  }
  if (!std::isfinite(oracle)) {
    result["oracle"] = nullptr;
    result["topk"] = Json::object();
    result["fixed_algorithm_baseline"] = nullptr;
    result["fixed_candidate_baseline"] = nullptr;
    result["wall_oracle"] = nullptr;
    result["selected_total_encode_cpu_ns"] = nullptr;
    result["selected_total_encode_wall_ns"] = nullptr;
    return result;
  }

  Json oracle_candidates = Json::array();
  std::set<std::string> oracle_set;
  for (const auto& item : true_costs.items()) {
    const double value = item.value().at(std::string(target)).get<double>();
    if (tied(value, oracle)) {
      oracle_candidates.push_back(item.key());
      oracle_set.insert(item.key());
    }
  }
  result["oracle"] = Json{{"cost", oracle},
                          {"candidates", std::move(oracle_candidates)}};
  Json wall_oracle_candidates = Json::array();
  std::set<std::string> wall_oracle_set;
  for (const auto& item : true_costs.items()) {
    const double value = item.value().at(wall_key).get<double>();
    if (tied(value, wall_oracle)) {
      wall_oracle_candidates.push_back(item.key());
      wall_oracle_set.insert(item.key());
    }
  }
  result["wall_oracle"] =
      Json{{"cost", wall_oracle},
           {"candidates", std::move(wall_oracle_candidates)}};

  const auto pareto_set = candidate_set(pareto.at("candidates"), "pareto");
  const auto dominated_set =
      candidate_set(pareto.at("dominated_candidates"), "dominated");
  Json topk = Json::object();
  for (const std::size_t k : kTopKs) {
    const Json predicted = target_prediction.at("top" + std::to_string(k));
    bool hit = false;
    bool wall_hit = false;
    double best_cost = std::numeric_limits<double>::infinity();
    double best_wall_cost = std::numeric_limits<double>::infinity();
    std::size_t dominated_count = 0;
    std::size_t pareto_found = 0;
    for (const auto& candidate_json : predicted) {
      const auto candidate = candidate_json.get<std::string>();
      hit = hit || oracle_set.contains(candidate);
      wall_hit = wall_hit || wall_oracle_set.contains(candidate);
      if (true_costs.contains(candidate)) {
        best_cost = std::min(
            best_cost,
            true_costs.at(candidate).at(std::string(target)).get<double>());
        best_wall_cost =
            std::min(best_wall_cost,
                     true_costs.at(candidate).at(wall_key).get<double>());
      }
      dominated_count += dominated_set.contains(candidate) ? 1 : 0;
      pareto_found += pareto_set.contains(candidate) ? 1 : 0;
    }
    Json score{{"candidates", predicted},
               {"tie_aware_hit", hit},
               {"wall_tie_aware_hit", wall_hit},
               {"dominated_count", dominated_count},
               {"pareto_recall",
                pareto_set.empty()
                    ? 0.0
                    : static_cast<double>(pareto_found) /
                          static_cast<double>(pareto_set.size())}};
    score["oracle_best_within_topk_multiplicative_regret"] =
        std::isfinite(best_cost) ? Json(best_cost / oracle) : Json(nullptr);
    score["wall_oracle_best_within_topk_multiplicative_regret"] =
        std::isfinite(best_wall_cost)
            ? Json(best_wall_cost / wall_oracle)
            : Json(nullptr);
    if (k == 1) {
      score["online_selected_multiplicative_regret"] =
          score.at("oracle_best_within_topk_multiplicative_regret");
      score["wall_online_selected_multiplicative_regret"] =
          score.at("wall_oracle_best_within_topk_multiplicative_regret");
      score["semantics"] = "online_top1_selection_no_probe";
    } else {
      score["semantics"] = "offline_set_quality_no_online_probe";
    }
    topk[std::to_string(k)] = std::move(score);
  }
  result["topk"] = std::move(topk);
  result["fixed_algorithm_baseline"] = baseline_score(
      row, true_costs, baselines.at(std::string(target)).at("fixed_algorithm"),
      target, oracle, wall_oracle);
  result["fixed_candidate_baseline"] = baseline_score(
      row, true_costs, baselines.at(std::string(target)).at("fixed_candidate"),
      target, oracle, wall_oracle);

  const auto& top1 = target_prediction.at("top1");
  if (top1.empty()) {
    result["selected_total_encode_cpu_ns"] = nullptr;
    result["selected_total_encode_wall_ns"] = nullptr;
  } else {
    const auto selected = top1.at(0).get<std::string>();
    const auto costs = strict_costs(row, selected);
    if (!costs) {
      result["selected_total_encode_cpu_ns"] = nullptr;
      result["selected_total_encode_wall_ns"] = nullptr;
    } else {
      const auto& metric = row.at("metrics").at(selected);
      const double encode_cpu = required_finite_number(metric, "encode_cpu_ns");
      const double encode_wall = required_finite_number(metric, "encode_wall_ns");
      result["selected_total_encode_cpu_ns"] = online_cpu_ns + encode_cpu;
      result["selected_total_encode_wall_ns"] = online_wall_ns + encode_wall;
    }
  }
  return result;
}

Json summarize(const std::vector<const Json*>& predictions) {
  Json result = Json::object();
  for (const char* target : kTargets) {
    Json target_summary = Json::object();
    std::size_t scored_rows = 0;
    std::size_t ranked_rows = 0;
    for (const Json* prediction : predictions) {
      if (!prediction->at("scores").at(target).at("oracle").is_null()) {
        ++scored_rows;
      }
      if (!prediction->at("prediction")
               .at("targets")
               .at(target)
               .at("ranking")
               .empty()) {
        ++ranked_rows;
      }
    }
    target_summary["coverage"] =
        Json{{"total_rows", predictions.size()},
             {"scored_rows", scored_rows},
             {"unscored_rows", predictions.size() - scored_rows},
             {"ranked_rows", ranked_rows},
             {"unranked_rows", predictions.size() - ranked_rows}};
    for (const std::size_t k : kTopKs) {
      std::size_t rows = 0;
      std::size_t hits = 0;
      std::size_t regret_rows = 0;
      std::size_t wall_hits = 0;
      std::size_t wall_regret_rows = 0;
      double log_regret_sum = 0;
      double wall_log_regret_sum = 0;
      double pareto_recall_sum = 0;
      std::size_t dominated_count = 0;
      for (const Json* prediction : predictions) {
        const auto& score = prediction->at("scores").at(target);
        if (score.at("oracle").is_null()) {
          continue;
        }
        ++rows;
        const auto& top = score.at("topk").at(std::to_string(k));
        hits += top.at("tie_aware_hit").get<bool>() ? 1 : 0;
        wall_hits += top.at("wall_tie_aware_hit").get<bool>() ? 1 : 0;
        pareto_recall_sum += top.at("pareto_recall").get<double>();
        dominated_count += top.at("dominated_count").get<std::size_t>();
        if (!top.at("oracle_best_within_topk_multiplicative_regret")
                 .is_null()) {
          const double regret =
              top.at("oracle_best_within_topk_multiplicative_regret")
                  .get<double>();
          if (regret > 0 && std::isfinite(regret)) {
            log_regret_sum += std::log(regret);
            ++regret_rows;
          }
        }
        if (!top.at("wall_oracle_best_within_topk_multiplicative_regret")
                 .is_null()) {
          const double regret =
              top.at("wall_oracle_best_within_topk_multiplicative_regret")
                  .get<double>();
          if (regret > 0 && std::isfinite(regret)) {
            wall_log_regret_sum += std::log(regret);
            ++wall_regret_rows;
          }
        }
      }
      target_summary["top" + std::to_string(k)] = Json{
          {"rows_with_oracle", rows},
          {"unscored_rows", predictions.size() - rows},
          {"tie_aware_hits", hits},
          {"tie_aware_hit_rate",
           rows == 0 ? 0.0 : static_cast<double>(hits) / rows},
          {"wall_tie_aware_hits", wall_hits},
          {"wall_tie_aware_hit_rate",
           rows == 0 ? 0.0 : static_cast<double>(wall_hits) / rows},
          {"geometric_mean_oracle_best_within_topk_multiplicative_regret",
           regret_rows == 0
               ? Json(nullptr)
               : Json(std::exp(log_regret_sum /
                               static_cast<double>(regret_rows)))},
          {"wall_geometric_mean_oracle_best_within_topk_multiplicative_regret",
           wall_regret_rows == 0
               ? Json(nullptr)
               : Json(std::exp(wall_log_regret_sum /
                               static_cast<double>(wall_regret_rows)))},
          {"pareto_recall_mean",
           rows == 0 ? 0.0 : pareto_recall_sum / rows},
          {"dominated_count", dominated_count}};
      if (k == 1) {
        target_summary["top1"]
                      ["geometric_mean_online_selected_multiplicative_regret"] =
            target_summary["top1"].at(
                "geometric_mean_oracle_best_within_topk_multiplicative_regret");
        target_summary["top1"]
                      ["wall_geometric_mean_online_selected_multiplicative_regret"] =
            target_summary["top1"].at(
                "wall_geometric_mean_oracle_best_within_topk_multiplicative_regret");
      }
    }
    result[target] = std::move(target_summary);
  }
  return result;
}

}  // namespace

struct PreparedModel::Impl {
  Json candidate_catalog_sha256;
  std::size_t feature_count = 0;
  std::size_t point_count = 0;
  std::vector<double> means;
  std::vector<double> scales;
  std::vector<std::string> candidates;
  std::unordered_map<std::string, std::size_t> candidate_indexes;
  std::vector<double> normalized_features;
  std::vector<double> labels;
  std::vector<std::uint8_t> label_valid;
  std::vector<std::size_t> sample_id_tie_ranks;

  std::size_t label_offset(std::size_t point, std::size_t candidate,
                           std::size_t target) const noexcept {
    return (point * candidates.size() + candidate) * kTargets.size() +
           target;
  }
};

PreparedModel::PreparedModel(const Json& model) {
  if (!model.is_object() ||
      model.value("schema", std::string()) !=
          "operator-recommend135-model-v1") {
    fail("unsupported model schema");
  }
  if (!model.contains("algorithm") || !model.at("algorithm").is_object() ||
      model.at("algorithm").value("k", 0U) != kNeighbors) {
    fail("model does not use the frozen k=5 contract");
  }
  const auto model_names =
      string_array(model.at("feature_names"), "model.feature_names");
  if (model_names != feature_names()) {
    fail("model feature_names do not match this implementation");
  }
  if (!model.contains("normalization") ||
      !model.at("normalization").is_object()) {
    fail("model normalization must be an object");
  }

  auto prepared = std::make_shared<Impl>();
  prepared->feature_count = model_names.size();
  prepared->means = json_finite_vector(
      model.at("normalization").at("mean"), "model.normalization.mean");
  prepared->scales = json_finite_vector(
      model.at("normalization").at("scale"), "model.normalization.scale");
  if (prepared->means.size() != prepared->feature_count ||
      prepared->scales.size() != prepared->feature_count) {
    fail("model normalization has the wrong shape");
  }
  for (const double scale : prepared->scales) {
    if (!(scale > 0)) {
      fail("model normalization scale must be positive");
    }
  }

  prepared->candidates =
      string_array(model.at("candidate_ids"), "model.candidate_ids");
  for (std::size_t i = 0; i < prepared->candidates.size(); ++i) {
    prepared->candidate_indexes.emplace(prepared->candidates[i], i);
  }
  if (!model.contains("candidate_catalog_sha256")) {
    fail("model is missing candidate_catalog_sha256");
  }
  const auto& catalog_sha256 = model.at("candidate_catalog_sha256");
  if (!catalog_sha256.is_null() && !catalog_sha256.is_string()) {
    fail("model candidate_catalog_sha256 must be a string or null");
  }
  prepared->candidate_catalog_sha256 = catalog_sha256;

  if (!model.contains("points") || !model.at("points").is_array()) {
    fail("model points must be an array");
  }
  const auto& points = model.at("points");
  prepared->point_count = points.size();
  prepared->normalized_features.reserve(prepared->point_count *
                                        prepared->feature_count);
  prepared->labels.assign(prepared->point_count *
                              prepared->candidates.size() * kTargets.size(),
                          0.0);
  prepared->label_valid.assign(prepared->labels.size(), 0);

  std::vector<std::string> sample_ids;
  sample_ids.reserve(prepared->point_count);
  for (std::size_t point_index = 0; point_index < points.size();
       ++point_index) {
    const auto& point = points.at(point_index);
    if (!point.is_object()) {
      fail("model point must be an object");
    }
    sample_ids.push_back(required_string(point, "sample_id"));
    const auto point_features = json_finite_vector(
        point.at("normalized_features"), "point.normalized_features");
    if (point_features.size() != prepared->feature_count) {
      fail("model point has the wrong feature shape");
    }
    prepared->normalized_features.insert(
        prepared->normalized_features.end(), point_features.begin(),
        point_features.end());

    if (!point.contains("labels")) {
      continue;
    }
    const auto& point_labels = point.at("labels");
    if (!point_labels.is_object()) {
      fail("point labels must be an object");
    }
    for (const auto& candidate_item : point_labels.items()) {
      const auto candidate_it =
          prepared->candidate_indexes.find(candidate_item.key());
      if (candidate_it == prepared->candidate_indexes.end()) {
        fail("point labels contain a candidate outside model.candidate_ids");
      }
      if (!candidate_item.value().is_object()) {
        fail("candidate labels must be an object");
      }
      for (const auto& target_item : candidate_item.value().items()) {
        const auto target_it =
            std::find_if(kTargets.begin(), kTargets.end(),
                         [&](const char* target) {
                           return target_item.key() == target;
                         });
        if (target_it == kTargets.end()) {
          fail("candidate labels contain an unsupported target");
        }
        if (!target_item.value().is_number()) {
          fail("candidate label must be numeric");
        }
        const double label = target_item.value().get<double>();
        if (!std::isfinite(label)) {
          fail("candidate label must be finite");
        }
        const std::size_t target_index =
            static_cast<std::size_t>(target_it - kTargets.begin());
        const std::size_t offset = prepared->label_offset(
            point_index, candidate_it->second, target_index);
        prepared->labels[offset] = label;
        prepared->label_valid[offset] = 1;
      }
    }
  }

  std::vector<std::string> sorted_sample_ids = sample_ids;
  std::sort(sorted_sample_ids.begin(), sorted_sample_ids.end());
  sorted_sample_ids.erase(
      std::unique(sorted_sample_ids.begin(), sorted_sample_ids.end()),
      sorted_sample_ids.end());
  std::unordered_map<std::string, std::size_t> tie_rank_by_sample;
  tie_rank_by_sample.reserve(sorted_sample_ids.size());
  for (std::size_t i = 0; i < sorted_sample_ids.size(); ++i) {
    tie_rank_by_sample.emplace(sorted_sample_ids[i], i);
  }
  prepared->sample_id_tie_ranks.reserve(sample_ids.size());
  for (const auto& sample_id : sample_ids) {
    prepared->sample_id_tie_ranks.push_back(tie_rank_by_sample.at(sample_id));
  }
  impl_ = std::move(prepared);
}

const Json& PreparedModel::candidate_catalog_sha256() const noexcept {
  return impl_->candidate_catalog_sha256;
}

std::vector<std::string> feature_names() {
  return {"log1p_row_count",
          "log1p_valid_count",
          "scanned_word_count",
          "scanned_value_bytes",
          "dtype_bit_width",
          "dtype_is_float",
          "dtype_is_signed",
          "missing_fraction",
          "byte_entropy",
          "distinct_fraction",
          "run_fraction",
          "value_zero_fraction",
          "finite_fraction",
          "nonfinite_fraction",
          "delta_zero_fraction",
          "delta_bw_1_8_fraction",
          "delta_bw_9_16_fraction",
          "delta_bw_17_32_fraction",
          "delta_bw_33_64_fraction",
          "xor_zero_fraction",
          "xor_bw_1_8_fraction",
          "xor_bw_9_16_fraction",
          "xor_bw_17_32_fraction",
          "xor_bw_33_64_fraction"};
}

std::vector<double> extract_features(std::span<const std::uint8_t> values,
                                     std::string_view dtype_name,
                                     std::span<const std::uint8_t> missing,
                                     std::uint64_t row_count) {
  const DTypeInfo dtype = dtype_info(dtype_name);
  if (values.size() % dtype.width != 0) {
    fail("values byte length is not divisible by dtype width");
  }
  const std::size_t valid_count = values.size() / dtype.width;
  if (row_count < valid_count) {
    fail("row_count is smaller than the packed valid value count");
  }

  std::uint64_t missing_count = 0;
  if (missing.empty()) {
    if (row_count != valid_count) {
      fail("missing bitmap is required when row_count exceeds valid values");
    }
  } else {
    const std::uint64_t expected_bytes = (row_count + 7) / 8;
    if (missing.size() != expected_bytes) {
      fail("missing bitmap shape does not match row_count");
    }
    if (row_count % 8 != 0) {
      const auto used = static_cast<unsigned>(row_count % 8);
      const auto padding_mask = static_cast<std::uint8_t>(0xffU << used);
      if ((missing.back() & padding_mask) != 0) {
        fail("missing bitmap has set padding bits beyond row_count");
      }
    }
    for (const auto byte : missing) {
      missing_count += std::popcount(byte);
    }
    if (missing_count + valid_count != row_count) {
      fail("missing bitmap count does not match packed valid values");
    }
  }

  const std::size_t scanned_words =
      std::min({valid_count, kMaxFeatureWords,
                kMaxFeatureValueBytes / dtype.width});
  const std::size_t scanned_bytes = scanned_words * dtype.width;
  const auto scanned_values = values.first(scanned_bytes);

  std::array<std::uint64_t, 256> byte_counts{};
  for (const auto byte : scanned_values) {
    ++byte_counts[byte];
  }
  double entropy = 0;
  if (!scanned_values.empty()) {
    for (const auto count : byte_counts) {
      if (count == 0) {
        continue;
      }
      const double probability =
          static_cast<double>(count) /
          static_cast<double>(scanned_values.size());
      entropy -= probability * std::log2(probability);
    }
    entropy /= 8.0;
  }

  std::vector<std::uint64_t> words;
  words.reserve(scanned_words);
  std::unordered_set<std::uint64_t> distinct;
  std::uint64_t zero_count = 0;
  std::uint64_t finite_count = 0;
  for (std::size_t i = 0; i < scanned_words; ++i) {
    const auto word = read_word(scanned_values, i * dtype.width, dtype);
    words.push_back(word);
    distinct.insert(word);
    zero_count += word == 0 ? 1 : 0;
    finite_count += finite_float_bits(word, dtype) ? 1 : 0;
  }

  std::uint64_t runs = scanned_words == 0 ? 0 : 1;
  std::array<std::uint64_t, 5> delta_bins{};
  std::array<std::uint64_t, 5> xor_bins{};
  const auto mask = width_mask(dtype.bits);
  for (std::size_t i = 1; i < words.size(); ++i) {
    runs += words[i] != words[i - 1] ? 1 : 0;
    const std::uint64_t delta = (words[i] - words[i - 1]) & mask;
    const std::uint64_t xored = (words[i] ^ words[i - 1]) & mask;
    ++delta_bins[width_bin(delta)];
    ++xor_bins[width_bin(xored)];
  }

  const double valid_denominator =
      scanned_words == 0 ? 1.0 : static_cast<double>(scanned_words);
  const double transition_denominator =
      scanned_words <= 1 ? 1.0 : static_cast<double>(scanned_words - 1);
  std::vector<double> result = {
      std::log1p(static_cast<double>(row_count)),
      std::log1p(static_cast<double>(valid_count)),
      static_cast<double>(scanned_words),
      static_cast<double>(scanned_bytes),
      static_cast<double>(dtype.bits),
      dtype.is_float ? 1.0 : 0.0,
      dtype.is_signed ? 1.0 : 0.0,
      row_count == 0 ? 0.0
                     : static_cast<double>(missing_count) /
                           static_cast<double>(row_count),
      entropy,
      static_cast<double>(distinct.size()) / valid_denominator,
      static_cast<double>(runs) / valid_denominator,
      static_cast<double>(zero_count) / valid_denominator,
      static_cast<double>(finite_count) / valid_denominator,
      static_cast<double>(scanned_words - finite_count) / valid_denominator,
  };
  for (const auto count : delta_bins) {
    result.push_back(scanned_words <= 1
                         ? 0.0
                         : static_cast<double>(count) / transition_denominator);
  }
  for (const auto count : xor_bins) {
    result.push_back(scanned_words <= 1
                         ? 0.0
                         : static_cast<double>(count) / transition_denominator);
  }
  if (result.size() != feature_names().size()) {
    throw std::logic_error("feature_names and feature extraction diverged");
  }
  return result;
}

Json fit_model(const Json& training_input, const Json& options) {
  const auto cpu_start = cpu_now_ns();
  const auto wall_start = wall_now_ns();
  if (!options.is_object()) {
    fail("fit options must be an object");
  }
  if (options.contains("k") &&
      (!options.at("k").is_number_integer() ||
       options.at("k").get<std::size_t>() != kNeighbors)) {
    fail("k is frozen at 5 and cannot be tuned");
  }
  const Json& training_rows = rows_array(training_input);
  if (training_rows.empty()) {
    fail("training rows must not be empty");
  }

  const auto candidates = candidate_scope(training_rows, options);
  const auto catalog_sha256 = option_catalog_sha256(options);
  const std::set<std::string> candidate_id_set(candidates.begin(),
                                               candidates.end());
  const std::size_t dimensions = feature_names().size();
  std::vector<std::vector<double>> features;
  features.reserve(training_rows.size());
  std::set<std::string> group_ids;
  std::set<std::string> dataset_ids;
  for (const auto& row : training_rows) {
    features.push_back(feature_vector(row));
    group_ids.insert(required_string(row, "group_id"));
    dataset_ids.insert(required_string(row, "dataset_id"));
    required_string(row, "sample_id");
    required_string(row, "scale");
    required_string(row, "dtype");
    for (const auto& candidate : row_eligible_candidates(row)) {
      if (!candidate_id_set.contains(candidate)) {
        fail("row eligibility contains a candidate outside the frozen pool");
      }
    }
    if (catalog_sha256) {
      if (!row.contains("catalog_sha256") ||
          !row.at("catalog_sha256").is_string() ||
          canonical_dtype(row.at("catalog_sha256").get<std::string>()) !=
              *catalog_sha256) {
        fail("row catalog_sha256 does not match fit options");
      }
    }
    const double input_bytes = required_finite_number(row, "input_bytes");
    if (!(input_bytes > 0)) {
      fail("input_bytes must be positive");
    }
  }

  std::vector<double> means(dimensions, 0.0);
  std::vector<double> scales(dimensions, 1.0);
  for (const auto& vector : features) {
    for (std::size_t i = 0; i < dimensions; ++i) {
      means[i] += vector[i];
    }
  }
  for (double& mean : means) {
    mean /= static_cast<double>(features.size());
  }
  for (const auto& vector : features) {
    for (std::size_t i = 0; i < dimensions; ++i) {
      const double difference = vector[i] - means[i];
      scales[i] += difference * difference;
    }
  }
  for (std::size_t i = 0; i < dimensions; ++i) {
    double variance = scales[i] - 1.0;
    variance /= static_cast<double>(features.size());
    scales[i] = variance > 0 ? std::sqrt(variance) : 1.0;
  }

  Json points = Json::array();
  for (std::size_t row_index = 0; row_index < training_rows.size(); ++row_index) {
    const auto& row = training_rows.at(row_index);
    std::vector<double> normalized(dimensions);
    for (std::size_t i = 0; i < dimensions; ++i) {
      normalized[i] = (features[row_index][i] - means[i]) / scales[i];
    }
    points.push_back(
        Json{{"sample_id", required_string(row, "sample_id")},
             {"dataset_id", required_string(row, "dataset_id")},
             {"group_id", required_string(row, "group_id")},
             {"scale", required_string(row, "scale")},
             {"normalized_features", std::move(normalized)},
             {"labels", labels_for_row(row, candidates)}});
  }

  Json model{
      {"schema", "operator-recommend135-model-v1"},
      {"algorithm",
       {{"name", "inverse_distance_knn"},
        {"k", kNeighbors},
        {"distance", "zscore_euclidean"},
        {"weight", "inverse_distance"},
        {"exact_match", "uniform_mean"},
        {"label", "per_row_oracle_normalized_log_regret"},
        {"primary_status", "strict_verified"}}},
      {"feature_names", feature_names()},
      {"feature_extraction",
       {{"source", "raw_dtype_bits"},
        {"sampling", "representative_prefix"},
        {"max_words", kMaxFeatureWords},
        {"max_value_bytes", kMaxFeatureValueBytes},
        {"uses_dataset_id", false}}},
      {"candidate_ids", candidates},
      {"candidate_catalog_sha256",
       catalog_sha256 ? Json(*catalog_sha256) : Json(nullptr)},
      {"normalization",
       {{"mean", means},
        {"scale", scales},
        {"scope", "training_rows_only"},
        {"zero_variance_scale", 1.0}}},
      {"training",
       {{"row_count", training_rows.size()},
        {"group_ids", std::vector<std::string>(group_ids.begin(), group_ids.end())},
        {"dataset_ids",
         std::vector<std::string>(dataset_ids.begin(), dataset_ids.end())}}},
      {"points", std::move(points)}};
  const auto cpu_end = cpu_now_ns();
  const auto wall_end = wall_now_ns();
  model["offline_fit_cpu_ns"] = cpu_end - cpu_start;
  model["offline_fit_wall_ns"] = wall_end - wall_start;
  return model;
}

Json predict(const Json& model, const std::vector<double>& features,
             const std::vector<std::string>& eligible_input) {
  const auto cpu_start = cpu_now_ns();
  const auto wall_start = wall_now_ns();
  if (!model.is_object() ||
      model.value("schema", std::string()) !=
          "operator-recommend135-model-v1") {
    fail("unsupported model schema");
  }
  if (!model.contains("algorithm") ||
      model.at("algorithm").value("k", 0U) != kNeighbors) {
    fail("model does not use the frozen k=5 contract");
  }
  const auto model_names = string_array(model.at("feature_names"),
                                        "model.feature_names");
  if (model_names != feature_names()) {
    fail("model feature_names do not match this implementation");
  }
  if (features.size() != model_names.size()) {
    fail("prediction feature vector has the wrong shape");
  }
  for (double value : features) {
    if (!std::isfinite(value)) {
      fail("prediction feature vector contains a non-finite value");
    }
  }

  const auto means = json_finite_vector(model.at("normalization").at("mean"),
                                        "model.normalization.mean");
  const auto scales = json_finite_vector(model.at("normalization").at("scale"),
                                         "model.normalization.scale");
  if (means.size() != features.size() || scales.size() != features.size()) {
    fail("model normalization has the wrong shape");
  }
  std::vector<double> normalized(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (!(scales[i] > 0)) {
      fail("model normalization scale must be positive");
    }
    normalized[i] = (features[i] - means[i]) / scales[i];
  }

  const auto model_candidates =
      string_array(model.at("candidate_ids"), "model.candidate_ids");
  const std::set<std::string> model_candidate_set(model_candidates.begin(),
                                                  model_candidates.end());
  std::vector<std::string> eligible;
  std::vector<std::string> unscored;
  std::set<std::string> seen;
  for (const auto& candidate : eligible_input) {
    if (candidate.empty()) {
      fail("eligible candidate IDs must be non-empty");
    }
    if (!seen.insert(candidate).second) {
      continue;
    }
    (model_candidate_set.contains(candidate) ? eligible : unscored)
        .push_back(candidate);
  }

  if (!model.contains("points") || !model.at("points").is_array()) {
    fail("model points must be an array");
  }
  struct PointDistance {
    const Json* point = nullptr;
    double distance = 0;
    std::string sample_id;
  };
  std::vector<PointDistance> point_distances;
  point_distances.reserve(model.at("points").size());
  for (const auto& point : model.at("points")) {
    const auto point_features = json_finite_vector(
        point.at("normalized_features"), "point.normalized_features");
    if (point_features.size() != normalized.size()) {
      fail("model point has the wrong feature shape");
    }
    double squared = 0;
    for (std::size_t i = 0; i < normalized.size(); ++i) {
      const double difference = normalized[i] - point_features[i];
      squared += difference * difference;
    }
    point_distances.push_back(
        PointDistance{&point, std::sqrt(squared),
                      required_string(point, "sample_id")});
  }
  std::sort(point_distances.begin(), point_distances.end(),
            [](const PointDistance& left, const PointDistance& right) {
              if (left.distance != right.distance) {
                return left.distance < right.distance;
              }
              return left.sample_id < right.sample_id;
            });
  Json targets = Json::object();
  for (const char* target : kTargets) {
    Json ranking = Json::array();
    for (const auto& candidate : eligible) {
      std::array<std::pair<double, double>, kNeighbors> neighbors{};
      std::size_t count = 0;
      for (const auto& point_distance : point_distances) {
        const auto& point = *point_distance.point;
        if (!point.contains("labels") ||
            !point.at("labels").contains(candidate) ||
            !point.at("labels").at(candidate).contains(target)) {
          continue;
        }
        const double label =
            required_finite_number(point.at("labels").at(candidate), target);
        neighbors[count++] = {point_distance.distance, label};
        if (count == kNeighbors) break;
      }
      if (count == 0) {
        unscored.push_back(candidate);
        continue;
      }

      double weighted_sum = 0;
      double weight_sum = 0;
      std::size_t exact_count = 0;
      for (std::size_t i = 0; i < count; ++i) {
        if (neighbors[i].first <= kDistanceFloor) {
          weighted_sum += neighbors[i].second;
          ++exact_count;
        }
      }
      if (exact_count != 0) {
        weighted_sum /= static_cast<double>(exact_count);
      } else {
        weighted_sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
          const double weight = 1.0 / neighbors[i].first;
          weighted_sum += weight * neighbors[i].second;
          weight_sum += weight;
        }
        weighted_sum /= weight_sum;
      }
      ranking.push_back(
          Json{{"candidate_id", candidate},
               {"predicted_log_regret", weighted_sum},
               {"neighbor_count", count},
               {"exact_match_count", exact_count}});
    }
    std::sort(ranking.begin(), ranking.end(), [](const Json& left,
                                                  const Json& right) {
      const double left_score = left.at("predicted_log_regret").get<double>();
      const double right_score = right.at("predicted_log_regret").get<double>();
      if (left_score != right_score) {
        return left_score < right_score;
      }
      return left.at("candidate_id").get<std::string>() <
             right.at("candidate_id").get<std::string>();
    });
    targets[target] = Json{{"ranking", ranking},
                           {"top1", top_candidates(ranking, 1)},
                           {"top3", top_candidates(ranking, 3)},
                           {"top5", top_candidates(ranking, 5)}};
  }

  std::sort(unscored.begin(), unscored.end());
  unscored.erase(std::unique(unscored.begin(), unscored.end()), unscored.end());
  Json serving = Json::object();
  for (const char* target : kTargets) {
    const auto& top1 = targets.at(target).at("top1");
    serving[target] =
        Json{{"selected_candidate_id",
              top1.empty() ? Json(nullptr) : Json(top1.at(0))},
             {"selection", "top1_only"},
             {"probe_used", false},
             {"probe_cost_cpu_ns", 0},
             {"probe_cost_wall_ns", 0}};
  }
  const auto cpu_end = cpu_now_ns();
  const auto wall_end = wall_now_ns();
  return Json{{"schema", "operator-recommend135-prediction-v1"},
              {"candidate_catalog_sha256",
               model.at("candidate_catalog_sha256")},
              {"eligible_candidates", eligible_input},
              {"unscored_candidates", unscored},
              {"targets", std::move(targets)},
              {"serving", std::move(serving)},
              {"top3_top5_semantics", "offline_set_quality_only"},
              {"online_cost",
               {{"model_cpu_ns", cpu_end - cpu_start},
                {"model_wall_ns", wall_end - wall_start},
                {"probe_used", false},
                {"probe_cost_cpu_ns", 0},
                {"probe_cost_wall_ns", 0}}}};
}

Json PreparedModel::predict(
    const std::vector<double>& features,
    const std::vector<std::string>& eligible_input) const {
  const auto cpu_start = cpu_now_ns();
  const auto wall_start = wall_now_ns();
  if (features.size() != impl_->feature_count) {
    fail("prediction feature vector has the wrong shape");
  }
  std::vector<double> normalized(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (!std::isfinite(features[i])) {
      fail("prediction feature vector contains a non-finite value");
    }
    normalized[i] = (features[i] - impl_->means[i]) / impl_->scales[i];
  }

  std::vector<std::string> eligible;
  std::vector<std::size_t> eligible_indexes;
  std::vector<std::string> unscored;
  std::set<std::string> seen;
  eligible.reserve(eligible_input.size());
  eligible_indexes.reserve(eligible_input.size());
  for (const auto& candidate : eligible_input) {
    if (candidate.empty()) {
      fail("eligible candidate IDs must be non-empty");
    }
    if (!seen.insert(candidate).second) {
      continue;
    }
    const auto candidate_it = impl_->candidate_indexes.find(candidate);
    if (candidate_it == impl_->candidate_indexes.end()) {
      unscored.push_back(candidate);
      continue;
    }
    eligible.push_back(candidate);
    eligible_indexes.push_back(candidate_it->second);
  }

  struct PointDistance {
    std::size_t point_index = 0;
    double distance = 0;
    std::size_t sample_id_tie_rank = 0;
  };
  std::vector<PointDistance> point_distances;
  point_distances.reserve(impl_->point_count);
  for (std::size_t point_index = 0; point_index < impl_->point_count;
       ++point_index) {
    double squared = 0;
    const std::size_t feature_offset = point_index * impl_->feature_count;
    for (std::size_t i = 0; i < impl_->feature_count; ++i) {
      const double difference =
          normalized[i] - impl_->normalized_features[feature_offset + i];
      squared += difference * difference;
    }
    point_distances.push_back(PointDistance{
        point_index, std::sqrt(squared),
        impl_->sample_id_tie_ranks[point_index]});
  }
  std::sort(point_distances.begin(), point_distances.end(),
            [](const PointDistance& left, const PointDistance& right) {
              if (left.distance != right.distance) {
                return left.distance < right.distance;
              }
              return left.sample_id_tie_rank < right.sample_id_tie_rank;
            });

  Json targets = Json::object();
  for (std::size_t target_index = 0; target_index < kTargets.size();
       ++target_index) {
    Json ranking = Json::array();
    for (std::size_t eligible_index = 0; eligible_index < eligible.size();
         ++eligible_index) {
      const auto& candidate = eligible[eligible_index];
      const std::size_t candidate_index = eligible_indexes[eligible_index];
      std::array<std::pair<double, double>, kNeighbors> neighbors{};
      std::size_t count = 0;
      for (const auto& point_distance : point_distances) {
        const std::size_t offset = impl_->label_offset(
            point_distance.point_index, candidate_index, target_index);
        if (impl_->label_valid[offset] == 0) {
          continue;
        }
        neighbors[count++] =
            {point_distance.distance, impl_->labels[offset]};
        if (count == kNeighbors) {
          break;
        }
      }
      if (count == 0) {
        unscored.push_back(candidate);
        continue;
      }

      double weighted_sum = 0;
      double weight_sum = 0;
      std::size_t exact_count = 0;
      for (std::size_t i = 0; i < count; ++i) {
        if (neighbors[i].first <= kDistanceFloor) {
          weighted_sum += neighbors[i].second;
          ++exact_count;
        }
      }
      if (exact_count != 0) {
        weighted_sum /= static_cast<double>(exact_count);
      } else {
        weighted_sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
          const double weight = 1.0 / neighbors[i].first;
          weighted_sum += weight * neighbors[i].second;
          weight_sum += weight;
        }
        weighted_sum /= weight_sum;
      }
      ranking.push_back(
          Json{{"candidate_id", candidate},
               {"predicted_log_regret", weighted_sum},
               {"neighbor_count", count},
               {"exact_match_count", exact_count}});
    }
    std::sort(ranking.begin(), ranking.end(), [](const Json& left,
                                                  const Json& right) {
      const double left_score = left.at("predicted_log_regret").get<double>();
      const double right_score =
          right.at("predicted_log_regret").get<double>();
      if (left_score != right_score) {
        return left_score < right_score;
      }
      return left.at("candidate_id").get<std::string>() <
             right.at("candidate_id").get<std::string>();
    });
    const char* target = kTargets[target_index];
    targets[target] = Json{{"ranking", ranking},
                           {"top1", top_candidates(ranking, 1)},
                           {"top3", top_candidates(ranking, 3)},
                           {"top5", top_candidates(ranking, 5)}};
  }

  std::sort(unscored.begin(), unscored.end());
  unscored.erase(std::unique(unscored.begin(), unscored.end()),
                 unscored.end());
  Json serving = Json::object();
  for (const char* target : kTargets) {
    const auto& top1 = targets.at(target).at("top1");
    serving[target] =
        Json{{"selected_candidate_id",
              top1.empty() ? Json(nullptr) : Json(top1.at(0))},
             {"selection", "top1_only"},
             {"probe_used", false},
             {"probe_cost_cpu_ns", 0},
             {"probe_cost_wall_ns", 0}};
  }
  const auto cpu_end = cpu_now_ns();
  const auto wall_end = wall_now_ns();
  return Json{{"schema", "operator-recommend135-prediction-v1"},
              {"candidate_catalog_sha256",
               impl_->candidate_catalog_sha256},
              {"eligible_candidates", eligible_input},
              {"unscored_candidates", unscored},
              {"targets", std::move(targets)},
              {"serving", std::move(serving)},
              {"top3_top5_semantics", "offline_set_quality_only"},
              {"online_cost",
               {{"model_cpu_ns", cpu_end - cpu_start},
                {"model_wall_ns", wall_end - wall_start},
                {"probe_used", false},
                {"probe_cost_cpu_ns", 0},
                {"probe_cost_wall_ns", 0}}}};
}

Json evaluate_leave_group_out(const Json& input, const Json& options) {
  if (!options.is_object()) {
    fail("evaluation options must be an object");
  }
  const Json& all_rows = rows_array(input);
  if (all_rows.empty()) {
    fail("evaluation rows must not be empty");
  }
  if (!options.contains("candidate_ids")) {
    fail("evaluation requires the complete frozen options.candidate_ids pool");
  }
  const auto evaluation_candidates =
      string_array(options.at("candidate_ids"), "options.candidate_ids");
  const std::set<std::string> evaluation_candidate_set(
      evaluation_candidates.begin(), evaluation_candidates.end());
  const auto catalog_sha256 = option_catalog_sha256(options);
  if (!catalog_sha256) {
    fail("evaluation requires options.catalog_sha256");
  }

  std::map<std::string, std::string> dataset_group;
  std::set<std::string> groups;
  for (const auto& row : all_rows) {
    const auto dataset = required_string(row, "dataset_id");
    const auto group = required_string(row, "group_id");
    required_string(row, "sample_id");
    required_string(row, "scale");
    required_string(row, "dtype");
    feature_vector(row);
    for (const auto& candidate : row_eligible_candidates(row)) {
      if (!evaluation_candidate_set.contains(candidate)) {
        fail("row eligibility contains a candidate outside the frozen pool");
      }
    }
    const double feature_cpu = required_finite_number(row, "feature_cpu_ns");
    const double feature_wall = required_finite_number(row, "feature_wall_ns");
    if (feature_cpu < 0 || feature_wall < 0) {
      fail("feature timing must be non-negative");
    }
    if (!row.contains("catalog_sha256") ||
        !row.at("catalog_sha256").is_string() ||
        canonical_dtype(row.at("catalog_sha256").get<std::string>()) !=
            *catalog_sha256) {
      fail("row catalog_sha256 does not match evaluation options");
    }
    const auto [it, inserted] = dataset_group.emplace(dataset, group);
    if (!inserted && it->second != group) {
      fail("dataset '" + dataset + "' is split across source groups");
    }
    groups.insert(group);
  }
  if (groups.size() < 2) {
    fail("leave-group-out evaluation requires at least two groups");
  }

  Json folds = Json::array();
  for (const auto& held_out : groups) {
    Json training_rows = Json::array();
    Json test_rows = Json::array();
    std::set<std::string> training_groups;
    std::set<std::string> training_datasets;
    std::set<std::string> test_datasets;
    for (const auto& row : all_rows) {
      if (row.at("group_id") == held_out) {
        test_rows.push_back(row);
        test_datasets.insert(row.at("dataset_id").get<std::string>());
      } else {
        training_rows.push_back(row);
        training_groups.insert(row.at("group_id").get<std::string>());
        training_datasets.insert(row.at("dataset_id").get<std::string>());
      }
    }
    if (test_rows.empty() || training_rows.empty()) {
      fail("leave-group-out fold has an empty train or test partition");
    }

    Json fold_options = options;
    const Json model = fit_model(training_rows, fold_options);
    const auto model_candidates = string_array(model.at("candidate_ids"),
                                               "model.candidate_ids");
    const auto algorithms =
        algorithm_candidates(training_rows, options, model_candidates);

    Json baselines = Json::object();
    for (const char* target : kTargets) {
      baselines[target] = Json{
          {"fixed_algorithm",
           baseline_json(choose_baseline(training_rows, algorithms, target))},
          {"fixed_candidate",
           baseline_json(
               choose_baseline(training_rows, model_candidates, target))}};
    }

    Json predictions = Json::array();
    for (const auto& test_row : test_rows) {
      const auto features = feature_vector(test_row);
      const auto eligible = row_eligible_candidates(test_row);

      // This call is deliberately completed before any held-out metric is read.
      const Json prediction = predict(model, features, eligible);

      const double feature_cpu =
          required_finite_number(test_row, "feature_cpu_ns");
      const double feature_wall =
          required_finite_number(test_row, "feature_wall_ns");
      if (feature_cpu < 0 || feature_wall < 0) {
        fail("feature timing must be non-negative");
      }
      const double model_cpu =
          prediction.at("online_cost").at("model_cpu_ns").get<double>();
      const double model_wall =
          prediction.at("online_cost").at("model_wall_ns").get<double>();
      const double eligibility_cpu =
          optional_finite_number(test_row, "eligibility_cpu_ns");
      const double eligibility_wall =
          optional_finite_number(test_row, "eligibility_wall_ns");
      if (eligibility_cpu < 0 || eligibility_wall < 0) {
        fail("eligibility timing must be non-negative");
      }
      const std::string eligibility_provenance = test_row.value(
          "eligibility_provenance", std::string("fixture_unspecified"));
      const double online_cpu = eligibility_cpu + feature_cpu + model_cpu;
      const double online_wall = eligibility_wall + feature_wall + model_wall;

      const Json costs = true_cost_map(test_row, eligible);
      const Json pareto = pareto_truth(costs);
      Json scores = Json::object();
      for (const char* target : kTargets) {
        scores[target] = score_target(
            test_row, prediction.at("targets").at(target), costs, pareto,
            target, baselines, online_cpu, online_wall);
      }
      predictions.push_back(
          Json{{"sample_id", required_string(test_row, "sample_id")},
               {"dataset_id", required_string(test_row, "dataset_id")},
               {"group_id", required_string(test_row, "group_id")},
               {"scale", required_string(test_row, "scale")},
               {"dtype", required_string(test_row, "dtype")},
               {"eligible_candidates", eligible},
               {"prediction", prediction},
               {"scores", std::move(scores)},
               {"pareto", pareto},
               {"decision_cost",
                {{"feature_cpu_ns", feature_cpu},
                 {"feature_wall_ns", feature_wall},
                 {"eligibility_cpu_ns", eligibility_cpu},
                 {"eligibility_wall_ns", eligibility_wall},
                 {"eligibility_provenance", eligibility_provenance},
                 {"model_cpu_ns", model_cpu},
                 {"model_wall_ns", model_wall},
                 {"probe_used", false},
                 {"probe_cost_cpu_ns", 0},
                 {"probe_cost_wall_ns", 0},
                 {"online_decision_cpu_ns", online_cpu},
                 {"online_decision_wall_ns", online_wall}}}});
    }

    std::vector<const Json*> fold_prediction_refs;
    fold_prediction_refs.reserve(predictions.size());
    for (const auto& prediction : predictions) {
      fold_prediction_refs.push_back(&prediction);
    }
    folds.push_back(
        Json{{"held_out_group_id", held_out},
             {"training_group_ids",
              std::vector<std::string>(training_groups.begin(),
                                       training_groups.end())},
             {"training_dataset_ids",
              std::vector<std::string>(training_datasets.begin(),
                                       training_datasets.end())},
             {"test_dataset_ids",
              std::vector<std::string>(test_datasets.begin(),
                                       test_datasets.end())},
             {"training_row_count", training_rows.size()},
             {"test_row_count", test_rows.size()},
             {"candidate_ids", model_candidates},
             {"catalog_sha256", *catalog_sha256},
             {"normalization", model.at("normalization")},
             {"offline_fit_cpu_ns", model.at("offline_fit_cpu_ns")},
             {"offline_fit_wall_ns", model.at("offline_fit_wall_ns")},
             {"baselines", std::move(baselines)},
             {"predictions", std::move(predictions)},
             {"summary", summarize(fold_prediction_refs)}});
  }

  std::vector<const Json*> all_prediction_refs;
  for (const auto& fold : folds) {
    for (const auto& prediction : fold.at("predictions")) {
      all_prediction_refs.push_back(&prediction);
    }
  }
  return Json{{"schema", "operator-recommend135-evaluation-v1"},
              {"model_spec",
               {{"algorithm", "inverse_distance_knn"},
                {"k", kNeighbors},
                {"normalization_scope", "training_rows_only"},
                {"primary_status", "strict_verified"},
                {"test_metrics_usage", "post_prediction_scoring_only"},
                {"feature_sampling", "representative_prefix"},
                {"feature_max_words", kMaxFeatureWords},
                {"feature_max_value_bytes", kMaxFeatureValueBytes}}},
              {"candidate_scope",
               {{"candidate_ids", evaluation_candidates},
                {"catalog_sha256", *catalog_sha256}}},
              {"folds", std::move(folds)},
              {"summary", summarize(all_prediction_refs)}};
}

}  // namespace operator_recommend135_v1
