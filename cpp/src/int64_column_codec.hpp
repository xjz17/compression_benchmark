#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace int64_column_codec {

std::int32_t lo32(std::int64_t v);
std::int32_t hi32(std::int64_t v);
std::int64_t reconstruct(std::int32_t lo, std::int32_t hi);

std::vector<std::uint32_t> fast_pfor_compress_limb(const std::int32_t *data, std::size_t n);
std::vector<std::int32_t> fast_pfor_uncompress_limb(const std::uint32_t *words, std::size_t word_count,
                                                    std::size_t n);

/** Wire: LE u32 loWordCount, lo FastPFOR words, hi FastPFOR words. */
std::vector<std::uint8_t> fast_pfor_encode_int64(const std::int64_t *data, std::size_t n);
std::vector<std::int64_t> fast_pfor_decode_int64(const std::uint8_t *wire, std::size_t wire_len,
                                                 std::size_t n);

/** Wire: LE u32 loPayloadLen, Stream VByte delta-1 on lo, then on hi. */
std::vector<std::uint8_t> stream_vbyte_encode_int64(const std::int64_t *data, std::size_t n);
std::vector<std::int64_t> stream_vbyte_decode_int64(const std::uint8_t *wire, std::size_t wire_len,
                                                    std::size_t n);
std::vector<std::uint8_t> stream_vbyte_encode_int64_ablation(
    const std::int64_t *data, std::size_t n, std::string_view omitted_operator);
std::vector<std::int64_t> stream_vbyte_decode_int64_ablation(
    const std::uint8_t *wire, std::size_t wire_len, std::size_t n,
    std::string_view omitted_operator);

/** Simple8b on sequential lo then hi int32 limbs ({@code 2*n} words). */
std::vector<std::uint8_t> simple8b_encode_int64(const std::int64_t *data, std::size_t n);
std::vector<std::int64_t> simple8b_decode_int64(const std::uint8_t *body, std::size_t body_len,
                                                std::size_t n);

bool doubles_match_scaled_column(const std::vector<double> &col, const std::int64_t *data,
                                 std::size_t n, int col_scale);

} // namespace int64_column_codec
