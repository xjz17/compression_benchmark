#pragma once

#include <nlohmann/json.hpp>

namespace operator_harness_v2 {

// Validate that a run request is an exact execution of the one eligible
// operator row named by implementation_ids.  The returned object is intended
// to be embedded in the source-schema NGF root.
nlohmann::json validate_catalog_binding(const nlohmann::json& request);

// Validate the immutable binding recovered from an NGF frame against a decode
// request.  A decode request must name the same single implementation.  When
// it supplies catalog_sha256 (or recipe_sha256), that digest must also match.
void validate_decoded_binding(const nlohmann::json& source_schema,
                              const nlohmann::json& request);

}  // namespace operator_harness_v2
