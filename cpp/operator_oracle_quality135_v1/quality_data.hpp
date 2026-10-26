#pragma once

#include <cstddef>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace operator_oracle_quality135_v1 {

struct FeatureBuildOptions {
  std::size_t max_valid_values = 32768;
  bool require_frozen_current135 = true;
};

// Builds one canonical JSONL feature row per source column.  This API has no
// oracle-label input: feature extraction is deliberately separated from label
// joining so confirm labels cannot enter this stage.
nlohmann::json build_features(
    const std::filesystem::path& columns_json,
    const std::filesystem::path& output_jsonl,
    const FeatureBuildOptions& options = {});

}  // namespace operator_oracle_quality135_v1
