#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

// GORILLA (Gorilla V2) and CHIMP — INT32 and raw float bits, matching Apache TSFile Java encoders.

namespace tsfile::p03 {

std::vector<std::uint8_t> encode_gorilla_int32(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_gorilla_float(std::span<const float> values);

std::vector<std::uint8_t> encode_chimp_int32(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_chimp_float(std::span<const float> values);

std::vector<std::uint8_t> encode_gorilla_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_gorilla_double(std::span<const double> values);
std::vector<std::uint8_t> encode_gorilla_double_ablation(
    std::span<const double> values, std::string_view omitted_operator);

std::vector<std::uint8_t> encode_chimp_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_chimp_double(std::span<const double> values);
std::vector<std::uint8_t> encode_chimp_double_ablation(
    std::span<const double> values, std::string_view omitted_operator);

}  // namespace tsfile::p03
