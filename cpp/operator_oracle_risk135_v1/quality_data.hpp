#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace operator_oracle_quality135_v1 {

struct FeatureBuildOptions {
  std::size_t max_valid_values = 32768;
  bool require_frozen_current135 = true;
};

struct FeatureVariantBuildOptions {
  std::size_t max_valid_values_per_window = 32768;
  bool require_frozen_current135 = true;
};

std::vector<std::string> prefix24_feature_names();
std::vector<std::string> multi3_feature_names();

// Builds one canonical JSONL feature row per source column.  This API has no
// oracle-label input: feature extraction is deliberately separated from label
// joining so confirm labels cannot enter this stage.
nlohmann::json build_features(
    const std::filesystem::path& columns_json,
    const std::filesystem::path& output_jsonl,
    const FeatureBuildOptions& options = {});

nlohmann::json build_development_feature_variants(
    const std::filesystem::path& columns_json,
    const std::filesystem::path& input_metadata_json,
    const std::filesystem::path& output_directory,
    const FeatureVariantBuildOptions& options = {});

}  // namespace operator_oracle_quality135_v1
