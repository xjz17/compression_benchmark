#pragma once

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace native_baseline86_v1::column_scales {

namespace fs = std::filesystem;
using J = nlohmann::json;

// Returns a canonical JSON string containing only the scientific column
// identity.  Entity and chunk/block coordinates are deliberately excluded.
std::string column_key(const J& manifest, const J& row);

// Reads and validates the frozen typed manifest and its block files before
// timing.  Only resident float32/float64 value bytes are scanned inside the
// measured region.  Warmups are unreported; one nanosecond sample is emitted
// per repeat and per scientific column.
J scan(const fs::path& manifest_path, unsigned warmups, unsigned repeats,
       const std::string& cli_sha256);

// Re-reads the frozen manifest and inputs, recomputes every true column scale,
// and rejects any missing, extra, duplicate, reordered, or modified block,
// scale, byte count, digest, or timing contract.  On success it returns the
// exact receipt unchanged.
J validate(const fs::path& manifest_path, const J& receipt,
           const std::string& cli_sha256);

}  // namespace native_baseline86_v1::column_scales
