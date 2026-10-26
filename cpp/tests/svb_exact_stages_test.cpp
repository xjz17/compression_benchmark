#include "svb_exact_stages.hpp"

#include "int64_column_codec.hpp"
#include "dbp_exact_stages.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  const std::vector<std::vector<std::int64_t>> cases = {
      {0},
      {-9, -9, -3, 0, 1, 1, 17, 17, 17},
      {0, 1, -1, 0x100000000LL, -0x100000000LL, 0x7fffffffffffffffLL},
  };
  for (const auto& values : cases) {
    const auto delta = exact_actions::svb::delta_forward(values);
    if (exact_actions::svb::delta_inverse(delta) != values) {
      throw std::runtime_error("SVB delta stage round trip");
    }
    const auto staged = exact_actions::svb::integer_code_forward(delta);
    auto direct = int64_column_codec::stream_vbyte_encode_int64(
        values.data(), values.size());
    direct.insert(direct.begin(), static_cast<std::uint8_t>('I'));
    if (staged != direct) throw std::runtime_error("SVB public-wire parity");
    if (exact_actions::svb::integer_code_inverse(staged) != delta) {
      throw std::runtime_error("SVB integer-code stage round trip");
    }
    if (exact_actions::svb::decompose(staged) != values) {
      throw std::runtime_error("SVB full-chain round trip");
    }
  }

  const std::vector<double> doubles{-19.25, -1.0, 0.0, 0.125, 3.5, 17.25};
  const auto scaling = exact_actions::dbp::scaling_forward(doubles, 3);
  const auto staged = exact_actions::svb::compose_scaled(scaling);
  const auto restored_scaling = exact_actions::svb::decompose_scaled(staged);
  if (exact_actions::dbp::scaling_inverse(restored_scaling) != doubles) {
    throw std::runtime_error("SVB scaled adapter round trip");
  }
  std::cout << "svb_exact_stage_actions=2 status=PASS\n";
  return 0;
}
