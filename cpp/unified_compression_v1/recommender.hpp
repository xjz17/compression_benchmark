#pragma once

#include "../operator_recommend135_v1/model.hpp"
#include "selection.hpp"

#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace unified_compression_v1 {

struct RecommendationResult {
  SelectionResult selection;
  nlohmann::json prediction = nlohmann::json::object();
  nlohmann::json provenance = nlohmann::json::object();
  std::uint64_t prediction_wall_time_ns = 0;
  std::uint64_t total_wall_time_ns = 0;
};

// Immutable adapter from the frozen development-only 24-candidate model to
// the measured fixed30 harness scope.  It transports a rank only: every final
// decision still comes from charged, strict harness feedback.
class PreparedRecommender {
 public:
  PreparedRecommender(const nlohmann::json& frozen_model,
                      const nlohmann::json& frozen_split,
                      std::string expected_split_sha256 = {});

  RecommendationResult recommend(const Registry& registry,
                                 const Column& column,
                                 SelectionOptions options = {}) const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

}  // namespace unified_compression_v1
