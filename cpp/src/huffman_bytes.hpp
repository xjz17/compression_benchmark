#pragma once

#include "benchmark_loss.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace huffman_bytes {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> data);
std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded);

}  // namespace huffman_bytes

/** Huffman on raw bytes; CSV row key ``huffman``. See HuffmanBytes.java for wire format. */
void testHuffmanCompression(const std::vector<char> &data, std::vector<long long> &result,
                            LossAccum *loss = nullptr);
