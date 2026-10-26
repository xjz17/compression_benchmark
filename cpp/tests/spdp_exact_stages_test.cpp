#include "spdp_exact_stages.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

int main() {
  std::vector<std::uint64_t> values{
      0, 1, 2, 3, 5, 8, 13, 21, 21, 21, 34, 55, 89, 144, 233, 377};
  std::vector<std::uint8_t> raw(values.size() * 8);
  std::memcpy(raw.data(), values.data(), raw.size());
  using namespace exact_actions::spdp;
  const auto delta = delta_forward(raw);
  if (delta_inverse(delta) != raw) throw std::runtime_error("SPDP delta round trip");
  const auto block = block_transform_forward(delta);
  if (block_transform_inverse(block) != delta)
    throw std::runtime_error("SPDP block transform round trip");
  const auto wire = entropy_forward(block);
  if (entropy_inverse(wire, raw.size()) != block)
    throw std::runtime_error("SPDP entropy round trip");
  if (decompose(wire, raw.size()) != raw)
    throw std::runtime_error("SPDP chain round trip");
  if (compose(raw) != wire) throw std::runtime_error("SPDP chain determinism");
  return 0;
}
