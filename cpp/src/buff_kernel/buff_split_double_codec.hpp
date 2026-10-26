#pragma once

/**
 * Simple split-double column encoder aligned with upstream
 * {@code database/src/compress/split_double.rs} ({@code SplitBDDoubleCompress::encode} first overload)
 * and {@code org.example.buff.BuffSplitDoubleColumnCodec} (Java).
 */

#include <cstdint>
#include <string_view>
#include <vector>

namespace buff_kernel {

int decimal_bits_for_precision(int prec);

/** Packed bytes: header (t, ilen, dlen) then int parts, then decimal parts. */
std::vector<std::uint8_t> encode_doubles(const std::vector<double>& values, int scale);

struct DecodedColumn {
    std::uint32_t t{0};
    int ilen{0};
    int dlen{0};
    std::vector<std::int64_t> int_parts;
    std::vector<std::uint64_t> dec_parts;
    std::vector<double> values;
};

/** Inverse of {@link encode_doubles}; values preserves every IEEE-754 bit. */
DecodedColumn decode_doubles(const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> encode_int64_ablation(
    const std::vector<std::int64_t>& values, std::string_view omitted_operator);
std::vector<std::int64_t> decode_int64_ablation(
    const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> encode_double_ablation(
    const std::vector<double>& values, std::string_view omitted_operator);
std::vector<double> decode_double_ablation(
    const std::vector<std::uint8_t>& bytes);

}  // namespace buff_kernel
