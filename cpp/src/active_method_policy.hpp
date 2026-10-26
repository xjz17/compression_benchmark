#pragma once

#include <string>

namespace active_method_policy {

// Generated from benchmark/reference/methods/active_method_policy_v1.json.
inline constexpr const char* kPolicyId = "paper164-active85-glza-retired-v1";
inline constexpr const char* kPolicySha256 =
    "f48677a529247a165898bce319f3e20ef6ba9c305552676344f3ead39d0bfc5d";
inline constexpr int kCatalogMethodCount = 86;
inline constexpr int kActiveMethodCount = 85;

inline bool is_active_algorithm_key(const std::string& algorithm) {
  return algorithm != "glza";
}

}  // namespace active_method_policy
