#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace tsfile::p02 {

bool decode_ts_2diff_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out);
bool decode_rle_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out);

bool decode_ts_2diff_float(std::span<const std::uint8_t> encoded, std::vector<float>& out);
bool decode_rle_float(std::span<const std::uint8_t> encoded, std::vector<float>& out);

bool decode_ts_2diff_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);
bool decode_rle_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out);

bool decode_ts_2diff_double(std::span<const std::uint8_t> encoded, std::vector<double>& out);
bool decode_rle_double(std::span<const std::uint8_t> encoded, std::vector<double>& out);

}  // namespace tsfile::p02
