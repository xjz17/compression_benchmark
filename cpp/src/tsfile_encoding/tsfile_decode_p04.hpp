#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace tsfile::p04 {

bool decode_sprintz_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out);
bool decode_sprintz_float(std::span<const std::uint8_t> encoded, std::vector<float>& out);

bool decode_rlbe_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out);
bool decode_rlbe_float(std::span<const std::uint8_t> encoded, std::vector<float>& out);

bool decode_sprintz_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);
bool decode_sprintz_double(std::span<const std::uint8_t> encoded, std::vector<double>& out);

bool decode_rlbe_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);
bool decode_rlbe_double(std::span<const std::uint8_t> encoded, std::vector<double>& out);

bool decode_sprintz_int64_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<std::int64_t>& out);
bool decode_sprintz_double_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<double>& out);
bool decode_rlbe_int64_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<std::int64_t>& out);
bool decode_rlbe_double_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<double>& out);


bool decode_sprintz_int64_reference(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);
bool decode_sprintz_double_reference(std::span<const std::uint8_t> encoded, std::vector<double>& out);

}  // namespace tsfile::p04
