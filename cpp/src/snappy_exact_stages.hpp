#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::snappy {

std::vector<std::uint8_t> original_encode(std::span<const std::uint8_t> input);

// Raw bytes -> semantic literal/copy records.  The inverse expands matches,
// including overlapping copies, without invoking the Snappy decoder.
std::vector<std::uint8_t> lz_match_forward(std::span<const std::uint8_t> input);
// Match decisions of the frozen snap 1.1.1 implementation.
std::vector<std::uint8_t> snap111_match_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> lz_match_inverse(std::span<const std::uint8_t> packet);

// Semantic records -> Snappy tag body while retaining the uncompressed length
// as an integer field in an intermediate packet.
std::vector<std::uint8_t> token_format_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> token_format_inverse(std::span<const std::uint8_t> packet);

// Varint materialization of the uncompressed length.  The output is the exact
// snappy::RawCompress / python-snappy compress byte stream.
std::vector<std::uint8_t> integer_code_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> integer_code_inverse(std::span<const std::uint8_t> wire);

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::snappy
