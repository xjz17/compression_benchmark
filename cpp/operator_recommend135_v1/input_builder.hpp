#pragma once

#include <cstddef>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace operator_recommend135_v1 {

nlohmann::json build_inputs(const std::filesystem::path& plan_columns,
                            const std::filesystem::path& output,
                            std::size_t max_values = 32768);

}  // namespace operator_recommend135_v1
