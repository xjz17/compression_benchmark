#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace operator_harness_v1 {

nlohmann::json handle_request(const nlohmann::json& request);
nlohmann::json handle_line(const std::string& line);

}  // namespace operator_harness_v1
