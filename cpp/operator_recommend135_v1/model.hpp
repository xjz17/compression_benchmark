#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace operator_recommend135_v1 {

class PreparedModel {
 public:
  explicit PreparedModel(const nlohmann::json& model);

  nlohmann::json predict(
      const std::vector<double>& features,
      const std::vector<std::string>& eligible_candidates) const;

  const nlohmann::json& candidate_catalog_sha256() const noexcept;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

std::vector<std::string> feature_names();

std::vector<double> extract_features(std::span<const std::uint8_t> values,
                                     std::string_view dtype,
                                     std::span<const std::uint8_t> missing,
                                     std::uint64_t row_count);

nlohmann::json fit_model(const nlohmann::json& training_rows,
                         const nlohmann::json& options);

nlohmann::json predict(
    const nlohmann::json& model, const std::vector<double>& features,
    const std::vector<std::string>& eligible_candidates);

nlohmann::json evaluate_leave_group_out(const nlohmann::json& rows,
                                        const nlohmann::json& options);

}  // namespace operator_recommend135_v1
