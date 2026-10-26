#include "pongo_codec.hpp"
#include "pongo_exact_stages.hpp"

#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint8_t> raw(const std::vector<double>& values) {
  std::vector<std::uint8_t> out(values.size() * 8);
  if (!out.empty()) {
    std::memcpy(out.data(), values.data(), out.size());
  }
  return out;
}

void check(const std::vector<std::uint8_t>& source) {
  using namespace exact_actions::pongo;
  const auto eraser = eraser_forward(source);
  if (eraser_inverse(eraser) != source) {
    throw std::runtime_error("Pongo eraser stage roundtrip failed");
  }
  const auto xors = xor_forward(eraser);
  if (xor_inverse(xors) != eraser) {
    throw std::runtime_error("Pongo XOR stage roundtrip failed");
  }
  const auto wire = trailing_pack_forward(xors);
  if (trailing_pack_inverse(wire) != xors) {
    throw std::runtime_error("Pongo trailing-zero stage roundtrip failed");
  }

  std::vector<double> values(source.size() / 8);
  if (!source.empty()) {
    std::memcpy(values.data(), source.data(), source.size());
  }
  if (wire != pongo_codec::encode(values)) {
    throw std::runtime_error("Pongo public wire mismatch");
  }
  if (decompose(compose(source)) != source) {
    throw std::runtime_error("Pongo exact-stage chain roundtrip failed");
  }
}

}  // namespace

int main() {
  check({});
  check(raw({65007.28, 65011.83, 65011.83, 65015.50, 64998.12,
             0.03215, 0.03218, 100.0, 100.5, 100.55, 100.555,
             -42.125, -0.0}));

  std::mt19937_64 random(20260904);
  std::vector<std::uint8_t> random_lanes(4096 * 8);
  for (std::size_t pos = 0; pos < random_lanes.size(); pos += 8) {
    const auto bits = random();
    std::memcpy(random_lanes.data() + pos, &bits, 8);
  }
  check(random_lanes);

  std::cout << "pongo_exact_stage_actions=3 status=PASS\n";
}
