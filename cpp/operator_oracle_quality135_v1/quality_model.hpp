#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace operator_oracle_quality135_v1 {

enum class Domain {
  Algorithm,
  OperatorZstd3,
  ObservedPipeline,
};

enum class Objective {
  Space,
  Encode,
  Decode,
  Balanced,
};

std::string_view domain_name(Domain domain) noexcept;
std::string_view objective_name(Objective objective) noexcept;
Domain parse_domain(std::string_view name);
Objective parse_objective(std::string_view name);

struct CandidateCosts {
  std::string candidate_id;
  bool eligible = false;
  std::optional<double> space_cost;
  std::optional<double> encode_cost;
  std::optional<double> decode_cost;
  std::optional<double> balanced_cost;
};

struct TrainingRow {
  std::string sample_id;
  std::string dataset_id;
  std::string source_group;
  // Set by the development-only driver, typically inverse columns per dataset.
  double weight = 1.0;
  std::vector<double> features;
  std::vector<CandidateCosts> candidates;
};

struct FitConfig {
  double ridge_lambda = 1.0;
  std::size_t knn_k = 8;
  // Final score = ridge_weight * ridge + (1-ridge_weight) * kNN.
  double ridge_weight = 0.5;
  bool center_log_cost_per_row = true;
  std::vector<std::string> feature_names;
  std::optional<std::string> candidate_catalog_sha256;
};

struct RankedCandidate {
  std::string candidate_id;
  double predicted_log_cost = 0.0;
  double ridge_log_cost = 0.0;
  std::optional<double> knn_log_cost;
  std::size_t support_rows = 0;
  std::size_t neighbor_count = 0;
};

struct RankingResult {
  Domain domain = Domain::Algorithm;
  Objective objective = Objective::Space;
  std::vector<RankedCandidate> ranking;
  std::vector<std::string> unscored_candidates;
};

class QualityModel {
 public:
  QualityModel();

  RankingResult rank(
      const std::vector<double>& features,
      const std::vector<std::string>& eligible_candidate_ids) const;

  // Reuse one fitted ridge/point representation across a bounded scoring
  // grid. These options do not change normalization, labels, or ridge
  // coefficients, so CV can avoid fitting the same fold repeatedly.
  RankingResult rank_with_scoring(
      const std::vector<double>& features,
      const std::vector<std::string>& eligible_candidate_ids,
      std::size_t knn_k, double ridge_weight) const;

  nlohmann::json to_json() const;
  static QualityModel from_json(const nlohmann::json& value);

  Domain domain() const;
  Objective objective() const;
  std::size_t feature_count() const;
  const std::vector<std::string>& candidate_ids() const;

 private:
  struct Impl;
  explicit QualityModel(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;

  friend QualityModel fit(const std::vector<TrainingRow>& rows,
                          Domain domain, Objective objective,
                          const FitConfig& config);
};

QualityModel fit(const std::vector<TrainingRow>& rows, Domain domain,
                 Objective objective, const FitConfig& config);

}  // namespace operator_oracle_quality135_v1
