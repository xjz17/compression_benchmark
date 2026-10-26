#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace pongo_codec {

inline constexpr const char* kRowName = "Pongo";

std::vector<std::uint8_t> encode(std::span<const double> values);
bool decode(std::span<const std::uint8_t> encoded, std::vector<double>& out);

std::vector<std::uint8_t> encode_ablation(std::span<const double> values,
                                          std::string_view omitted_operator);
bool decode_ablation(std::span<const std::uint8_t> encoded,
                     std::string_view omitted_operator,
                     std::vector<double>& out);

}  // namespace pongo_codec
