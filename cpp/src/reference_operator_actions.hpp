#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Exact port of the existing reference/proxy transforms. These are not
// production codecs merely because the implementation language is C++.
namespace reference_operator_actions {
using Bytes = std::vector<std::uint8_t>;
using Names = std::vector<std::string>;
Bytes encode(std::span<const std::uint8_t> raw_i64, const Names& actions);
Bytes decode(std::span<const std::uint8_t> wire);
Names actions(std::span<const std::uint8_t> wire);
Bytes raw_encode(std::span<const std::uint8_t> raw_i64, const Names& actions);
Bytes serialized(std::span<const std::uint8_t> raw_i64, const Names& actions);
Bytes final_values(std::span<const std::uint8_t> raw_i64, const Names& actions);
Bytes inverse_serialized(std::span<const std::uint8_t> state, const Names& actions);
Bytes compact_encode(std::span<const std::uint8_t> raw_i64);
Bytes compact_decode(std::span<const std::uint8_t> compact_payload);
Bytes inverse_step(std::span<const std::uint8_t> state, const std::string& tag);
// u64 metadata bytes, u64 raw-value bytes, metadata, raw i64, compact values.
Bytes inspect(std::span<const std::uint8_t> raw_i64, const Names& actions);
Bytes legacy_encode(std::span<const std::uint8_t> raw_i64, const Names& actions);
Bytes legacy_decode(std::span<const std::uint8_t> wire);
// Current forward-selectable actions.  The decoder keeps an internal frozen
// 29-entry wire-name table so retired action IDs remain readable.
const Names& all_actions();
}  // namespace reference_operator_actions
