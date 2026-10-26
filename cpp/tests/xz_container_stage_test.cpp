#include "xz_container_stage.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  // Empty XZ stream emitted by liblzma/Python FORMAT_XZ.
  const std::vector<std::uint8_t> wire = {
      0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, 0x00, 0x04, 0xe6, 0xd6, 0xb4, 0x46,
      0x00, 0x00, 0x00, 0x00, 0x1c, 0xdf, 0x44, 0x21, 0x1f, 0xb6, 0xf3, 0x7d,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x04, 0x59, 0x5a};
  const auto packet = exact_actions::xz_container::inverse(wire);
  if (exact_actions::xz_container::forward(packet) != wire)
    throw std::runtime_error("XZ container roundtrip failed");
  std::cout << "xz_container_actions=1 status=PASS\n";
}
