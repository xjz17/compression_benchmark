#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

// Sprintz retains FIRE/payload semantics; only existing RLE tails use URL1.
// Current RLBE uses URB1 with URL1 width streams and original residual payloads.
// *_legacy and *_ablation retain historical RLBE wire protocols.

namespace tsfile::p04 {

std::vector<std::uint8_t> encode_sprintz_int32(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_sprintz_float(std::span<const float> values);
std::vector<std::uint8_t> encode_sprintz_int32_legacy(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_sprintz_float_legacy(std::span<const float> values);

std::vector<std::uint8_t> encode_rlbe_int32(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_rlbe_float(std::span<const float> values);

std::vector<std::uint8_t> encode_sprintz_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_sprintz_double(std::span<const double> values);

std::vector<std::uint8_t> encode_rlbe_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_rlbe_double(std::span<const double> values);

std::vector<std::uint8_t> encode_sprintz_int64_ablation(
    std::span<const std::int64_t> values, std::string_view omitted_operator);
std::vector<std::uint8_t> encode_sprintz_double_ablation(
    std::span<const double> values, std::string_view omitted_operator);
std::vector<std::uint8_t> encode_rlbe_int64_ablation(
    std::span<const std::int64_t> values, std::string_view omitted_operator);
std::vector<std::uint8_t> encode_rlbe_double_ablation(
    std::span<const double> values, std::string_view omitted_operator);


// Frozen RLBE encoders for historical wire protocols.
std::vector<std::uint8_t> encode_rlbe_int32_legacy(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_rlbe_int64_legacy(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_rlbe_float_legacy(std::span<const float> values);
std::vector<std::uint8_t> encode_rlbe_double_legacy(std::span<const double> values);
std::vector<std::uint8_t> encode_sprintz_int64_reference(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_sprintz_double_reference(std::span<const double> values);

}  // namespace tsfile::p04
