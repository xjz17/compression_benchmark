#include "simple8b_exact_stages.hpp"

#include "int64_column_codec.hpp"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  const std::vector<std::int64_t> integers = {-9, -9, -3, 0, 1, 1, 17, 17, 17};
  const auto integer_wire = exact_actions::simple8b::bitpack_forward_int64(integers);
  const auto core = int64_column_codec::simple8b_encode_int64(
      integers.data(), integers.size());
  if (integer_wire.size() != core.size() + 1 || integer_wire[0] != 'I' ||
      !std::equal(core.begin(), core.end(), integer_wire.begin() + 1)) {
    throw std::runtime_error("Simple8b integer adapter parity");
  }
  if (exact_actions::simple8b::bitpack_inverse_int64(integer_wire) != integers) {
    throw std::runtime_error("Simple8b integer bitpack round trip");
  }

  const std::vector<double> decimals = {-12.34, -1.25, 0.0, 0.01, 7.5, 91.75};
  const auto scaled = exact_actions::simple8b::scaling_forward(decimals, 2);
  const auto staged = exact_actions::simple8b::bitpack_forward_scaled(scaled);
  if (exact_actions::simple8b::bitpack_inverse_scaled(staged) != scaled) {
    throw std::runtime_error("Simple8b bitpack stage packet round trip");
  }
  const auto restored = exact_actions::simple8b::decompose_double(staged);
  if (restored.size() != decimals.size()) throw std::runtime_error("Simple8b decoded size");
  for (std::size_t i = 0; i < decimals.size(); ++i) {
    if (std::bit_cast<std::uint64_t>(restored[i]) !=
        std::bit_cast<std::uint64_t>(decimals[i])) {
      throw std::runtime_error("Simple8b double bitwise round trip");
    }
  }
  std::cout << "simple8b_exact_stage_actions=2 status=PASS\n";
  return 0;
}
