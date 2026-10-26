#pragma once

#include <cstdint>
#include <span>
#include <vector>

// TS_2DIFF retains its TsFile decimal protocol. Current RLE uses versioned URL1
// on original 32/64-bit symbols; float max_point_number is compatibility-only.
// Explicit *_legacy encoders retain the historical TsFile protocols.

namespace tsfile::p02 {

std::vector<std::uint8_t> encode_ts_2diff_int32(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_rle_int32(std::span<const std::int32_t> values);

std::vector<std::uint8_t> encode_ts_2diff_float(std::span<const float> values, int max_point_number = 0);
std::vector<std::uint8_t> encode_rle_float(std::span<const float> values, int max_point_number = 2);

std::vector<std::uint8_t> encode_ts_2diff_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_rle_int64(std::span<const std::int64_t> values);

std::vector<std::uint8_t> encode_ts_2diff_double(std::span<const double> values, int max_point_number = 0);
std::vector<std::uint8_t> encode_rle_double(std::span<const double> values, int max_point_number = 2);


// Frozen TsFile encoders remain available to historical exact-stage protocols.
std::vector<std::uint8_t> encode_rle_int32_legacy(std::span<const std::int32_t> values);
std::vector<std::uint8_t> encode_rle_int64_legacy(std::span<const std::int64_t> values);
std::vector<std::uint8_t> encode_rle_float_legacy(std::span<const float> values, int max_point_number = 2);
std::vector<std::uint8_t> encode_rle_double_legacy(std::span<const double> values, int max_point_number = 2);

}  // namespace tsfile::p02
