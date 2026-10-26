#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::xz_container {

// Isolates the 12-byte XZ stream header and footer.  The packet core still
// contains the block headers, LZMA2 payload, padding, and index.
std::vector<std::uint8_t> forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> inverse(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::xz_container
