#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace reference_actions::general {

// Brotli-shaped reference chain.  It is independent and reversible, but does
// not claim byte parity with the upstream Brotli encoder.
std::vector<std::uint8_t> dictionary_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> dictionary_inverse(std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> lz_forward(std::span<const std::uint8_t> input,
                                     std::uint32_t window,
                                     std::uint32_t min_match,
                                     std::uint32_t max_match, bool lazy);
std::vector<std::uint8_t> lz_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> token_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> token_inverse(std::span<const std::uint8_t> packet);

// ZPP1 is a validated carrier wrapper; no match-selection policy runs here.
std::vector<std::uint8_t> wrap_lz_semantic_packet(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> unwrap_lz_semantic_packet(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> parse_policy_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> parse_policy_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> huffman_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> huffman_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> rans_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> rans_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> range_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> range_inverse(std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> shuffle_forward(std::span<const std::uint8_t> input,
                                          std::size_t type_size);
std::vector<std::uint8_t> shuffle_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> frame_forward(std::span<const std::uint8_t> input,
                                        bool blosc2);
std::vector<std::uint8_t> frame_inverse(std::span<const std::uint8_t> packet,
                                        bool blosc2);
std::vector<std::uint8_t> gzip_frame_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> gzip_frame_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> zlib_frame_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> zlib_frame_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> xz_frame_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> xz_frame_inverse(std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> bwt_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> bwt_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> mtf_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> mtf_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> rle_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> rle_inverse(std::span<const std::uint8_t> packet);

}  // namespace reference_actions::general
