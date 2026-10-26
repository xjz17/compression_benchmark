#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace tsfile::p03 {

bool decode_gorilla_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out);
bool decode_gorilla_float(std::span<const std::uint8_t> encoded, std::vector<float>& out);

bool decode_chimp_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out);
bool decode_chimp_float(std::span<const std::uint8_t> encoded, std::vector<float>& out);

bool decode_gorilla_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);
bool decode_gorilla_double(std::span<const std::uint8_t> encoded, std::vector<double>& out);
bool decode_gorilla_double_ablation(std::span<const std::uint8_t> encoded,
                                    std::string_view omitted_operator,
                                    std::vector<double>& out);

bool decode_chimp_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);
bool decode_chimp_double(std::span<const std::uint8_t> encoded, std::vector<double>& out);
bool decode_chimp_double_ablation(std::span<const std::uint8_t> encoded,
                                  std::string_view omitted_operator,
                                  std::vector<double>& out);

}  // namespace tsfile::p03
