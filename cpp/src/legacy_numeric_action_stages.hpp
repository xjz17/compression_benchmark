#pragma once
#include "native_lz_action_stages.hpp"
#include <optional>

// Frozen legacy benchmark adapters, deliberately not canonical ALP/FastPFor.
// LNS1 JSON packets and terminal I/F/ALP2 bytes retain their original identity.
namespace legacy_numeric_action {
using Bytes = native_lz_action::Bytes;
Bytes forward(const std::string& action, const Bytes&, const std::string& kind = "bytes",
              std::optional<int> max_point = std::nullopt);
Bytes inverse(const std::string& action, const Bytes&, std::uint64_t count = 0);
}  // namespace legacy_numeric_action
