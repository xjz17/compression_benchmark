#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::deflate_container {

// These actions isolate only the RFC 1952 / RFC 1950 wrapper.  The payload
// remains a raw DEFLATE stream and is deliberately not described as an LZ or
// Huffman stage.
std::vector<std::uint8_t> gzip_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> gzip_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> zlib_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> zlib_inverse(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::deflate_container
