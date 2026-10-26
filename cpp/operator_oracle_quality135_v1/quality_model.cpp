#include "quality_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace operator_oracle_quality135_v1 {
namespace {

using Json = nlohmann::json;

constexpr double kDistanceFloor = 1e-12;
constexpr double kPivotFloor = 1e-12;

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("operator_oracle_quality135_v1: " + message);
}

void require_finite(double value, const std::string& context) {
  if (!std::isfinite(value)) {
    fail(context + " must be finite");
  }
}

std::string required_string(const Json& value, const char* key) {
  if (!value.contains(key) || !value.at(key).is_string() ||
      value.at(key).get_ref<const std::string&>().empty()) {
    fail(std::string("missing non-empty string field '") + key + "'");
  }
  return value.at(key).get<std::string>();
}

double required_number(const Json& value, const char* key) {
  if (!value.contains(key) || !value.at(key).is_number()) {
    fail(std::string("missing numeric field '") + key + "'");
  }
  const double result = value.at(key).get<double>();
  require_finite(result, key);
  return result;
}

std::vector<double> finite_vector(const Json& value,
                                  const std::string& context) {
  if (!value.is_array()) {
    fail(context + " must be an array");
  }
  auto result = value.get<std::vector<double>>();
  for (double item : result) {
    require_finite(item, context);
  }
  return result;
}

std::vector<std::string> string_vector(const Json& value,
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
    auto text = item.get<std::string>();
    if (!seen.insert(text).second) {
      fail(context + " must not contain duplicates");
    }
    result.push_back(std::move(text));
  }
  return result;
}

void validate_domain_objective(Domain domain, Objective objective) {
  if ((domain == Domain::OperatorZstd3 ||
       domain == Domain::ObservedPipeline) &&
      objective != Objective::Space) {
    fail(std::string(domain_name(domain)) +
         " supports only the space objective because operator timing is "
         "not measured");
  }
}

const std::optional<double>& selected_cost(const CandidateCosts& costs,
                                           Objective objective) {
  switch (objective) {
    case Objective::Space:
      return costs.space_cost;
    case Objective::Encode:
      return costs.encode_cost;
    case Objective::Decode:
      return costs.decode_cost;
    case Objective::Balanced:
      return costs.balanced_cost;
  }
  fail("unknown objective");
}

std::vector<double> solve_linear(std::vector<double> matrix,
                                 std::vector<double> rhs,
                                 std::size_t dimensions) {
  if (matrix.size() != dimensions * dimensions ||
      rhs.size() != dimensions) {
    throw std::logic_error("linear-system shape mismatch");
  }
  for (std::size_t column = 0; column < dimensions; ++column) {
    std::size_t pivot = column;
    double pivot_abs = std::abs(matrix[column * dimensions + column]);
    for (std::size_t row = column + 1; row < dimensions; ++row) {
      const double candidate = std::abs(matrix[row * dimensions + column]);
      if (candidate > pivot_abs) {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (!(pivot_abs > kPivotFloor)) {
      fail("ridge normal equation is singular");
    }
    if (pivot != column) {
      for (std::size_t j = column; j < dimensions; ++j) {
        std::swap(matrix[column * dimensions + j],
                  matrix[pivot * dimensions + j]);
      }
      std::swap(rhs[column], rhs[pivot]);
    }
    const double diagonal = matrix[column * dimensions + column];
    for (std::size_t row = column + 1; row < dimensions; ++row) {
      const double factor = matrix[row * dimensions + column] / diagonal;
      matrix[row * dimensions + column] = 0.0;
      for (std::size_t j = column + 1; j < dimensions; ++j) {
        matrix[row * dimensions + j] -=
            factor * matrix[column * dimensions + j];
      }
      rhs[row] -= factor * rhs[column];
    }
  }
  std::vector<double> result(dimensions, 0.0);
  for (std::size_t reverse = 0; reverse < dimensions; ++reverse) {
    const std::size_t row = dimensions - 1 - reverse;
    double value = rhs[row];
    for (std::size_t column = row + 1; column < dimensions; ++column) {
      value -= matrix[row * dimensions + column] * result[column];
    }
    result[row] = value / matrix[row * dimensions + row];
    require_finite(result[row], "ridge coefficient");
  }
  return result;
}

void add_observation(std::vector<double>& xtx, std::vector<double>& xty,
                     const std::vector<double>& design, double target,
                     double weight) {
  const std::size_t dimensions = design.size();
  for (std::size_t i = 0; i < dimensions; ++i) {
    xty[i] += weight * design[i] * target;
    for (std::size_t j = i; j < dimensions; ++j) {
      const double contribution = weight * design[i] * design[j];
      xtx[i * dimensions + j] += contribution;
      if (i != j) xtx[j * dimensions + i] += contribution;
    }
  }
}

std::vector<double> fit_global_ridge(std::vector<double> xtx,
                                     std::vector<double> xty,
                                     double lambda) {
  const std::size_t dimensions = xty.size();
  // The intercept is stabilized only by a tiny diagonal. Slopes use the
  // configured penalty because features are standardized.
  xtx[0] += kPivotFloor;
  for (std::size_t i = 1; i < dimensions; ++i) {
    xtx[i * dimensions + i] += lambda;
  }
  return solve_linear(std::move(xtx), std::move(xty), dimensions);
}

bool valid_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                  (c >= 'A' && c <= 'F');
         });
}

}  // namespace

struct QualityModel::Impl {
  Domain domain = Domain::Algorithm;
  Objective objective = Objective::Space;
  FitConfig config;
  std::size_t feature_count = 0;
  std::size_t row_count = 0;
  std::vector<double> feature_means;
  std::vector<double> feature_scales;
  std::vector<std::string> candidate_ids;
  std::unordered_map<std::string, std::size_t> candidate_indexes;
  std::vector<std::size_t> support_rows;
  std::vector<double> coefficients;
  std::vector<double> normalized_features;
  std::vector<double> labels;
  std::vector<std::uint8_t> label_valid;
  std::vector<std::string> sample_ids;
  std::vector<double> row_weights;
  std::vector<std::string> dataset_ids;
  std::vector<std::string> source_groups;

  std::size_t coefficient_offset(std::size_t candidate) const {
    return candidate * (feature_count + 1);
  }

  std::size_t label_offset(std::size_t row, std::size_t candidate) const {
    return row * candidate_ids.size() + candidate;
  }
};

std::string_view domain_name(Domain domain) noexcept {
  switch (domain) {
    case Domain::Algorithm:
      return "algorithm";
    case Domain::OperatorZstd3:
      return "operator_zstd3";
    case Domain::ObservedPipeline:
      return "observed_pipeline";
  }
  return "unknown";
}

std::string_view objective_name(Objective objective) noexcept {
  switch (objective) {
    case Objective::Space:
      return "space";
    case Objective::Encode:
      return "encode";
    case Objective::Decode:
      return "decode";
    case Objective::Balanced:
      return "balanced";
  }
  return "unknown";
}

Domain parse_domain(std::string_view name) {
  if (name == "algorithm") {
    return Domain::Algorithm;
  }
  if (name == "operator_zstd3") {
    return Domain::OperatorZstd3;
  }
  if (name == "observed_pipeline") {
    return Domain::ObservedPipeline;
  }
  fail("unsupported domain '" + std::string(name) + "'");
}

Objective parse_objective(std::string_view name) {
  if (name == "space") {
    return Objective::Space;
  }
  if (name == "encode") {
    return Objective::Encode;
  }
  if (name == "decode") {
    return Objective::Decode;
  }
  if (name == "balanced") {
    return Objective::Balanced;
  }
  fail("unsupported objective '" + std::string(name) + "'");
}

QualityModel::QualityModel() = default;

QualityModel::QualityModel(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

QualityModel fit(const std::vector<TrainingRow>& rows, Domain domain,
                 Objective objective, const FitConfig& config) {
  validate_domain_objective(domain, objective);
  if (rows.empty()) {
    fail("training rows must not be empty");
  }
  require_finite(config.ridge_lambda, "ridge_lambda");
  require_finite(config.ridge_weight, "ridge_weight");
  if (!(config.ridge_lambda > 0.0)) {
    fail("ridge_lambda must be positive");
  }
  if (config.ridge_weight < 0.0 || config.ridge_weight > 1.0) {
    fail("ridge_weight must be in [0,1]");
  }
  if (config.knn_k == 0 && config.ridge_weight != 1.0) {
    fail("knn_k=0 requires ridge_weight=1");
  }
  if (config.candidate_catalog_sha256 &&
      !valid_sha256(*config.candidate_catalog_sha256)) {
    fail("candidate_catalog_sha256 must contain 64 hexadecimal characters");
  }

  const std::size_t feature_count = rows.front().features.size();
  if (feature_count == 0) {
    fail("feature vectors must not be empty");
  }
  FitConfig canonical_config = config;
  if (canonical_config.feature_names.empty()) {
    for (std::size_t i = 0; i < feature_count; ++i) {
      canonical_config.feature_names.push_back("feature_" +
                                               std::to_string(i));
    }
  }
  if (canonical_config.feature_names.size() != feature_count) {
    fail("feature_names shape does not match training features");
  }
  {
    std::set<std::string> unique;
    for (const auto& name : canonical_config.feature_names) {
      if (name.empty() || !unique.insert(name).second) {
        fail("feature_names must contain unique non-empty strings");
      }
    }
  }

  std::set<std::string> candidate_set;
  std::set<std::string> sample_set;
  std::set<std::string> dataset_set;
  std::set<std::string> group_set;
  for (const auto& row : rows) {
    if (row.sample_id.empty() || row.dataset_id.empty() ||
        row.source_group.empty()) {
      fail("sample_id, dataset_id, and source_group must be non-empty");
    }
    if (!sample_set.insert(row.sample_id).second) {
      fail("sample_id must be unique");
    }
    dataset_set.insert(row.dataset_id);
    group_set.insert(row.source_group);
    require_finite(row.weight, "training row weight");
    if (!(row.weight > 0.0)) {
      fail("training row weight must be positive");
    }
    if (row.features.size() != feature_count) {
      fail("all feature vectors must have the same shape");
    }
    for (double value : row.features) {
      require_finite(value, "training feature");
    }
    std::set<std::string> row_candidates;
    for (const auto& candidate : row.candidates) {
      if (candidate.candidate_id.empty()) {
        fail("candidate_id must be non-empty");
      }
      if (!row_candidates.insert(candidate.candidate_id).second) {
        fail("candidate_id is duplicated within a training row");
      }
      candidate_set.insert(candidate.candidate_id);
      const std::optional<double>* values[] = {
          &candidate.space_cost, &candidate.encode_cost,
          &candidate.decode_cost, &candidate.balanced_cost};
      for (const auto* item : values) {
        if (*item) {
          require_finite(**item, "candidate cost");
          if (!(**item > 0.0)) {
            fail("candidate costs must be positive");
          }
        }
      }
    }
  }
  if (candidate_set.empty()) {
    fail("training rows contain no candidates");
  }

  auto model = std::make_shared<QualityModel::Impl>();
  model->domain = domain;
  model->objective = objective;
  model->config = std::move(canonical_config);
  model->feature_count = feature_count;
  model->row_count = rows.size();
  model->candidate_ids.assign(candidate_set.begin(), candidate_set.end());
  for (std::size_t i = 0; i < model->candidate_ids.size(); ++i) {
    model->candidate_indexes.emplace(model->candidate_ids[i], i);
  }
  model->dataset_ids.assign(dataset_set.begin(), dataset_set.end());
  model->source_groups.assign(group_set.begin(), group_set.end());
  model->sample_ids.reserve(rows.size());
  model->row_weights.reserve(rows.size());

  model->feature_means.assign(feature_count, 0.0);
  model->feature_scales.assign(feature_count, 1.0);
  double total_row_weight = 0.0;
  for (const auto& row : rows) {
    total_row_weight += row.weight;
    for (std::size_t i = 0; i < feature_count; ++i) {
      model->feature_means[i] += row.weight * row.features[i];
    }
  }
  for (double& mean : model->feature_means) {
    mean /= total_row_weight;
  }
  std::fill(model->feature_scales.begin(), model->feature_scales.end(), 0.0);
  for (const auto& row : rows) {
    for (std::size_t i = 0; i < feature_count; ++i) {
      const double difference = row.features[i] - model->feature_means[i];
      model->feature_scales[i] += row.weight * difference * difference;
    }
  }
  for (double& scale : model->feature_scales) {
    scale = std::sqrt(scale / total_row_weight);
    if (!(scale > 0.0)) {
      scale = 1.0;
    }
  }

  model->normalized_features.resize(rows.size() * feature_count);
  for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
    model->sample_ids.push_back(rows[row_index].sample_id);
    model->row_weights.push_back(rows[row_index].weight);
    for (std::size_t i = 0; i < feature_count; ++i) {
      model->normalized_features[row_index * feature_count + i] =
          (rows[row_index].features[i] - model->feature_means[i]) /
          model->feature_scales[i];
    }
  }

  const std::size_t candidate_count = model->candidate_ids.size();
  model->labels.assign(rows.size() * candidate_count, 0.0);
  model->label_valid.assign(rows.size() * candidate_count, 0);
  model->support_rows.assign(candidate_count, 0);
  for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
    std::vector<std::pair<std::size_t, double>> valid;
    for (const auto& candidate : rows[row_index].candidates) {
      if (!candidate.eligible) {
        continue;
      }
      const auto& cost = selected_cost(candidate, objective);
      if (!cost) {
        continue;
      }
      valid.emplace_back(model->candidate_indexes.at(candidate.candidate_id),
                         std::log(*cost));
    }
    if (valid.empty()) {
      fail("every training row must have at least one eligible measured cost "
           "for the selected objective");
    }
    double offset = 0.0;
    if (model->config.center_log_cost_per_row) {
      offset = std::min_element(
                   valid.begin(), valid.end(),
                   [](const auto& left, const auto& right) {
                     return left.second < right.second;
                   })
                   ->second;
    }
    for (const auto& [candidate_index, log_cost] : valid) {
      const std::size_t index = model->label_offset(row_index, candidate_index);
      model->labels[index] = log_cost - offset;
      model->label_valid[index] = 1;
      ++model->support_rows[candidate_index];
    }
  }

  const std::size_t design_count = feature_count + 1;
  std::vector<std::vector<double>> candidate_xtx(
      candidate_count,
      std::vector<double>(design_count * design_count, 0.0));
  std::vector<std::vector<double>> candidate_xty(
      candidate_count, std::vector<double>(design_count, 0.0));
  std::vector<double> design(design_count, 1.0);
  for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
    for (std::size_t i = 0; i < feature_count; ++i) {
      design[i + 1] =
          model->normalized_features[row_index * feature_count + i];
    }
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
      const std::size_t label_index = model->label_offset(row_index, candidate);
      if (model->label_valid[label_index] == 0) {
        continue;
      }
      const double target = model->labels[label_index];
      add_observation(candidate_xtx[candidate], candidate_xty[candidate],
                      design, target, rows[row_index].weight);
    }
  }
  // The pooled system is the exact sum of candidate systems. Build it once
  // after the row pass instead of repeating every outer product twice.
  std::vector<double> global_xtx(design_count * design_count, 0.0);
  std::vector<double> global_xty(design_count, 0.0);
  for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
    for (std::size_t i = 0; i < global_xtx.size(); ++i)
      global_xtx[i] += candidate_xtx[candidate][i];
    for (std::size_t i = 0; i < global_xty.size(); ++i)
      global_xty[i] += candidate_xty[candidate][i];
  }
  const auto global = fit_global_ridge(global_xtx, global_xty,
                                       model->config.ridge_lambda);
  model->coefficients.assign(candidate_count * design_count, 0.0);
  for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
    if (model->support_rows[candidate] == 0) {
      continue;
    }
    auto matrix = std::move(candidate_xtx[candidate]);
    auto rhs = std::move(candidate_xty[candidate]);
    // Candidate coefficients have a Gaussian-style prior centered on the
    // pooled model. This shrinks rare candidates without mixing identities.
    for (std::size_t i = 0; i < design_count; ++i) {
      matrix[i * design_count + i] += model->config.ridge_lambda;
      rhs[i] += model->config.ridge_lambda * global[i];
    }
    const auto coefficients =
        solve_linear(std::move(matrix), std::move(rhs), design_count);
    const std::size_t offset = model->coefficient_offset(candidate);
    std::copy(coefficients.begin(), coefficients.end(),
              model->coefficients.begin() + offset);
  }

  return QualityModel(std::move(model));
}

RankingResult QualityModel::rank(
    const std::vector<double>& features,
    const std::vector<std::string>& eligible_candidate_ids) const {
  if (!impl_) {
    fail("model is not initialized");
  }
  return rank_with_scoring(features, eligible_candidate_ids,
                           impl_->config.knn_k,
                           impl_->config.ridge_weight);
}

RankingResult QualityModel::rank_with_scoring(
    const std::vector<double>& features,
    const std::vector<std::string>& eligible_candidate_ids,
    std::size_t knn_k, double ridge_weight) const {
  if (!impl_) {
    fail("model is not initialized");
  }
  require_finite(ridge_weight, "ridge_weight");
  if (ridge_weight < 0.0 || ridge_weight > 1.0) {
    fail("ridge_weight must be in [0,1]");
  }
  if (knn_k == 0 && ridge_weight != 1.0) {
    fail("knn_k=0 requires ridge_weight=1");
  }
  if (features.size() != impl_->feature_count) {
    fail("prediction feature vector has the wrong shape");
  }
  std::vector<double> normalized(impl_->feature_count);
  for (std::size_t i = 0; i < features.size(); ++i) {
    require_finite(features[i], "prediction feature");
    normalized[i] =
        (features[i] - impl_->feature_means[i]) / impl_->feature_scales[i];
  }

  struct Distance {
    std::size_t row = 0;
    double value = 0.0;
  };
  std::vector<Distance> distances;
  if (knn_k != 0 && ridge_weight < 1.0) {
    distances.reserve(impl_->row_count);
    for (std::size_t row = 0; row < impl_->row_count; ++row) {
      double squared = 0.0;
      for (std::size_t i = 0; i < impl_->feature_count; ++i) {
        const double difference =
            normalized[i] -
            impl_->normalized_features[row * impl_->feature_count + i];
        squared += difference * difference;
      }
      distances.push_back({row, std::sqrt(squared)});
    }
    std::sort(distances.begin(), distances.end(),
              [&](const Distance& left, const Distance& right) {
                if (left.value != right.value) {
                  return left.value < right.value;
                }
                return impl_->sample_ids[left.row] <
                       impl_->sample_ids[right.row];
              });
  }

  RankingResult result;
  result.domain = impl_->domain;
  result.objective = impl_->objective;
  std::unordered_set<std::string> seen;
  for (const auto& candidate_id : eligible_candidate_ids) {
    if (candidate_id.empty()) {
      fail("eligible candidate IDs must be non-empty");
    }
    if (!seen.insert(candidate_id).second) {
      continue;
    }
    const auto candidate_it = impl_->candidate_indexes.find(candidate_id);
    if (candidate_it == impl_->candidate_indexes.end() ||
        impl_->support_rows[candidate_it->second] == 0) {
      result.unscored_candidates.push_back(candidate_id);
      continue;
    }
    const std::size_t candidate = candidate_it->second;
    const std::size_t coefficient_offset =
        impl_->coefficient_offset(candidate);
    double ridge = impl_->coefficients[coefficient_offset];
    for (std::size_t i = 0; i < impl_->feature_count; ++i) {
      ridge += impl_->coefficients[coefficient_offset + i + 1] *
               normalized[i];
    }
    require_finite(ridge, "ridge prediction");

    std::optional<double> knn;
    std::size_t neighbor_count = 0;
    if (!distances.empty()) {
      struct Neighbor {
        double distance = 0.0;
        double label = 0.0;
        double row_weight = 1.0;
      };
      std::vector<Neighbor> neighbors;
      neighbors.reserve(knn_k);
      for (const auto& distance : distances) {
        const std::size_t label_index =
            impl_->label_offset(distance.row, candidate);
        if (impl_->label_valid[label_index] == 0) {
          continue;
        }
        neighbors.push_back({distance.value, impl_->labels[label_index],
                             impl_->row_weights[distance.row]});
        if (neighbors.size() == knn_k) {
          break;
        }
      }
      neighbor_count = neighbors.size();
      if (!neighbors.empty()) {
        double sum = 0.0;
        std::size_t exact = 0;
        double exact_weight = 0.0;
        for (const auto& neighbor : neighbors) {
          if (neighbor.distance <= kDistanceFloor) {
            sum += neighbor.row_weight * neighbor.label;
            exact_weight += neighbor.row_weight;
            ++exact;
          }
        }
        if (exact != 0) {
          knn = sum / exact_weight;
        } else {
          double weights = 0.0;
          sum = 0.0;
          for (const auto& neighbor : neighbors) {
            const double weight = neighbor.row_weight / neighbor.distance;
            sum += weight * neighbor.label;
            weights += weight;
          }
          knn = sum / weights;
        }
      }
    }

    double score = ridge;
    if (ridge_weight < 1.0) {
      if (!knn) {
        throw std::logic_error("supported candidate has no kNN labels");
      }
      score = ridge_weight * ridge + (1.0 - ridge_weight) * *knn;
    }
    require_finite(score, "blended prediction");
    result.ranking.push_back({candidate_id, score, ridge, knn,
                              impl_->support_rows[candidate], neighbor_count});
  }
  std::sort(result.ranking.begin(), result.ranking.end(),
            [](const RankedCandidate& left, const RankedCandidate& right) {
              if (left.predicted_log_cost != right.predicted_log_cost) {
                return left.predicted_log_cost < right.predicted_log_cost;
              }
              return left.candidate_id < right.candidate_id;
            });
  std::sort(result.unscored_candidates.begin(),
            result.unscored_candidates.end());
  return result;
}

Json QualityModel::to_json() const {
  if (!impl_) {
    fail("model is not initialized");
  }
  Json candidates = Json::array();
  for (std::size_t candidate = 0; candidate < impl_->candidate_ids.size();
       ++candidate) {
    const std::size_t offset = impl_->coefficient_offset(candidate);
    candidates.push_back(
        {{"candidate_id", impl_->candidate_ids[candidate]},
         {"support_rows", impl_->support_rows[candidate]},
         {"coefficients",
          std::vector<double>(
              impl_->coefficients.begin() + offset,
              impl_->coefficients.begin() + offset + impl_->feature_count +
                  1)}});
  }
  Json points = Json::array();
  for (std::size_t row = 0; row < impl_->row_count; ++row) {
    Json labels = Json::array();
    for (std::size_t candidate = 0;
         candidate < impl_->candidate_ids.size(); ++candidate) {
      const std::size_t offset = impl_->label_offset(row, candidate);
      labels.push_back(impl_->label_valid[offset] != 0
                           ? Json(impl_->labels[offset])
                           : Json(nullptr));
    }
    points.push_back(
        {{"sample_id", impl_->sample_ids[row]},
         {"row_weight", impl_->row_weights[row]},
         {"normalized_features",
          std::vector<double>(
              impl_->normalized_features.begin() +
                  row * impl_->feature_count,
              impl_->normalized_features.begin() +
                  (row + 1) * impl_->feature_count)},
         {"labels", std::move(labels)}});
  }
  return Json{
      {"schema", "operator-oracle-quality135-model-v1"},
      {"domain", domain_name(impl_->domain)},
      {"objective", objective_name(impl_->objective)},
      {"feature_names", impl_->config.feature_names},
      {"candidate_catalog_sha256",
       impl_->config.candidate_catalog_sha256
           ? Json(*impl_->config.candidate_catalog_sha256)
           : Json(nullptr)},
      {"algorithm",
       {{"name", "candidate_ridge_shrinkage_plus_knn"},
        {"label",
         impl_->config.center_log_cost_per_row
             ? "per_row_centered_log_cost"
             : "log_cost"},
        {"ridge_lambda", impl_->config.ridge_lambda},
        {"knn_k", impl_->config.knn_k},
        {"ridge_weight", impl_->config.ridge_weight},
        {"distance", "zscore_euclidean"},
        {"knn_weight", "inverse_distance"}}},
      {"normalization",
       {{"mean", impl_->feature_means},
        {"scale", impl_->feature_scales},
        {"scope", "development_training_rows_only"}}},
      {"training",
       {{"row_count", impl_->row_count},
        {"dataset_ids", impl_->dataset_ids},
        {"source_groups", impl_->source_groups},
        {"row_weight_contract", "positive_development_only_input"},
        {"identifiers_used_as_features", false}}},
      {"candidates", std::move(candidates)},
      {"points", std::move(points)}};
}

QualityModel QualityModel::from_json(const Json& value) {
  if (!value.is_object() ||
      value.value("schema", std::string()) !=
          "operator-oracle-quality135-model-v1") {
    fail("unsupported model schema");
  }
  auto model = std::make_shared<Impl>();
  model->domain = parse_domain(required_string(value, "domain"));
  model->objective = parse_objective(required_string(value, "objective"));
  validate_domain_objective(model->domain, model->objective);
  model->config.feature_names =
      string_vector(value.at("feature_names"), "feature_names");
  model->feature_count = model->config.feature_names.size();
  if (model->feature_count == 0) {
    fail("feature_names must not be empty");
  }
  if (!value.contains("candidate_catalog_sha256") ||
      !(value.at("candidate_catalog_sha256").is_null() ||
        value.at("candidate_catalog_sha256").is_string())) {
    fail("candidate_catalog_sha256 must be a string or null");
  }
  if (value.at("candidate_catalog_sha256").is_string()) {
    model->config.candidate_catalog_sha256 =
        value.at("candidate_catalog_sha256").get<std::string>();
    if (!valid_sha256(*model->config.candidate_catalog_sha256)) {
      fail("candidate_catalog_sha256 must contain 64 hexadecimal characters");
    }
  }
  if (!value.contains("algorithm") || !value.at("algorithm").is_object()) {
    fail("model is missing algorithm metadata");
  }
  const auto& algorithm = value.at("algorithm");
  if (required_string(algorithm, "name") !=
      "candidate_ridge_shrinkage_plus_knn") {
    fail("unsupported model algorithm");
  }
  model->config.ridge_lambda = required_number(algorithm, "ridge_lambda");
  model->config.ridge_weight = required_number(algorithm, "ridge_weight");
  if (!(model->config.ridge_lambda > 0.0) ||
      model->config.ridge_weight < 0.0 ||
      model->config.ridge_weight > 1.0) {
    fail("invalid serialized ridge configuration");
  }
  if (!algorithm.contains("knn_k") ||
      !algorithm.at("knn_k").is_number_unsigned()) {
    fail("algorithm.knn_k must be an unsigned integer");
  }
  model->config.knn_k = algorithm.at("knn_k").get<std::size_t>();
  if (model->config.knn_k == 0 && model->config.ridge_weight != 1.0) {
    fail("knn_k=0 requires ridge_weight=1");
  }
  const std::string label = required_string(algorithm, "label");
  if (label == "per_row_centered_log_cost") {
    model->config.center_log_cost_per_row = true;
  } else if (label == "log_cost") {
    model->config.center_log_cost_per_row = false;
  } else {
    fail("unsupported label contract");
  }
  if (!value.contains("normalization") ||
      !value.at("normalization").is_object()) {
    fail("model is missing normalization metadata");
  }
  const auto& normalization = value.at("normalization");
  model->feature_means =
      finite_vector(normalization.at("mean"), "normalization.mean");
  model->feature_scales =
      finite_vector(normalization.at("scale"), "normalization.scale");
  if (model->feature_means.size() != model->feature_count ||
      model->feature_scales.size() != model->feature_count) {
    fail("normalization shape does not match feature_names");
  }
  if (std::any_of(model->feature_scales.begin(), model->feature_scales.end(),
                  [](double scale) { return !(scale > 0.0); })) {
    fail("normalization scales must be positive");
  }
  if (!value.contains("training") || !value.at("training").is_object()) {
    fail("model is missing training metadata");
  }
  const auto& training = value.at("training");
  if (!training.contains("row_count") ||
      !training.at("row_count").is_number_unsigned()) {
    fail("training.row_count must be an unsigned integer");
  }
  model->row_count = training.at("row_count").get<std::size_t>();
  model->dataset_ids =
      string_vector(training.at("dataset_ids"), "training.dataset_ids");
  model->source_groups =
      string_vector(training.at("source_groups"), "training.source_groups");
  if (!training.contains("identifiers_used_as_features") ||
      training.at("identifiers_used_as_features") != false) {
    fail("serialized model must state identifiers_used_as_features=false");
  }

  if (!value.contains("candidates") || !value.at("candidates").is_array() ||
      value.at("candidates").empty()) {
    fail("candidates must be a non-empty array");
  }
  const std::size_t design_count = model->feature_count + 1;
  for (const auto& item : value.at("candidates")) {
    if (!item.is_object()) {
      fail("candidate entry must be an object");
    }
    const auto candidate_id = required_string(item, "candidate_id");
    if (model->candidate_indexes.contains(candidate_id)) {
      fail("candidate IDs must be unique");
    }
    if (!item.contains("support_rows") ||
        !item.at("support_rows").is_number_unsigned()) {
      fail("candidate support_rows must be an unsigned integer");
    }
    const auto coefficients =
        finite_vector(item.at("coefficients"), "candidate.coefficients");
    if (coefficients.size() != design_count) {
      fail("candidate coefficient shape mismatch");
    }
    const std::size_t index = model->candidate_ids.size();
    model->candidate_indexes.emplace(candidate_id, index);
    model->candidate_ids.push_back(candidate_id);
    model->support_rows.push_back(
        item.at("support_rows").get<std::size_t>());
    model->coefficients.insert(model->coefficients.end(),
                               coefficients.begin(), coefficients.end());
  }
  if (!std::is_sorted(model->candidate_ids.begin(),
                      model->candidate_ids.end())) {
    fail("serialized candidate IDs must use canonical sorted order");
  }

  if (!value.contains("points") || !value.at("points").is_array() ||
      value.at("points").size() != model->row_count) {
    fail("points shape does not match training.row_count");
  }
  model->normalized_features.reserve(model->row_count * model->feature_count);
  model->labels.assign(model->row_count * model->candidate_ids.size(), 0.0);
  model->label_valid.assign(model->labels.size(), 0);
  std::set<std::string> sample_ids;
  std::vector<std::size_t> observed_support(model->candidate_ids.size(), 0);
  for (std::size_t row = 0; row < model->row_count; ++row) {
    const auto& point = value.at("points").at(row);
    if (!point.is_object()) {
      fail("point entry must be an object");
    }
    const auto sample_id = required_string(point, "sample_id");
    if (!sample_ids.insert(sample_id).second) {
      fail("point sample_id must be unique");
    }
    model->sample_ids.push_back(sample_id);
    const double row_weight = required_number(point, "row_weight");
    if (!(row_weight > 0.0)) {
      fail("point row_weight must be positive");
    }
    model->row_weights.push_back(row_weight);
    const auto features = finite_vector(point.at("normalized_features"),
                                        "point.normalized_features");
    if (features.size() != model->feature_count) {
      fail("point feature shape mismatch");
    }
    model->normalized_features.insert(model->normalized_features.end(),
                                      features.begin(), features.end());
    if (!point.contains("labels") || !point.at("labels").is_array() ||
        point.at("labels").size() != model->candidate_ids.size()) {
      fail("point label shape mismatch");
    }
    for (std::size_t candidate = 0;
         candidate < model->candidate_ids.size(); ++candidate) {
      const auto& label = point.at("labels").at(candidate);
      if (label.is_null()) {
        continue;
      }
      if (!label.is_number()) {
        fail("point labels must be numeric or null");
      }
      const double numeric = label.get<double>();
      require_finite(numeric, "point label");
      const std::size_t offset = model->label_offset(row, candidate);
      model->labels[offset] = numeric;
      model->label_valid[offset] = 1;
      ++observed_support[candidate];
    }
  }
  if (observed_support != model->support_rows) {
    fail("candidate support_rows disagree with serialized labels");
  }
  return QualityModel(std::move(model));
}

Domain QualityModel::domain() const {
  if (!impl_) {
    fail("model is not initialized");
  }
  return impl_->domain;
}

Objective QualityModel::objective() const {
  if (!impl_) {
    fail("model is not initialized");
  }
  return impl_->objective;
}

std::size_t QualityModel::feature_count() const {
  if (!impl_) {
    fail("model is not initialized");
  }
  return impl_->feature_count;
}

const std::vector<std::string>& QualityModel::candidate_ids() const {
  if (!impl_) {
    fail("model is not initialized");
  }
  return impl_->candidate_ids;
}

}  // namespace operator_oracle_quality135_v1
