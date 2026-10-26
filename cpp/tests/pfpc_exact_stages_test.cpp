#include "pfpc_exact_stages.hpp"
#include "pfpc.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

int main() {
  std::vector<std::uint64_t> values{
      0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};
  std::vector<std::uint8_t> raw(values.size() * 8);
  std::memcpy(raw.data(), values.data(), raw.size());
  using namespace exact_actions::pfpc;
  const auto residuals = predictor_xor_forward(raw);
  if (predictor_xor_inverse(residuals) != raw)
    throw std::runtime_error("pFPC predictor round trip");
  const auto wire = trailing_pack_forward(residuals);
  if (trailing_pack_inverse(wire) != residuals)
    throw std::runtime_error("pFPC pack round trip");
  std::vector<std::uint8_t> reference(raw.size() * 2 + 1024);
  const auto size = pfpc_compress_memory(
      raw.data(), raw.size(), reference.data(), reference.size(), 16, 1,
      static_cast<int>(raw.size()));
  reference.resize(size);
  if (wire != reference) throw std::runtime_error("pFPC public wire mismatch");
  if (decompose(wire) != raw) throw std::runtime_error("pFPC chain round trip");
  return 0;
}
