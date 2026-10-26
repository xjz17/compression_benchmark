#include "operator102_dispatcher.hpp"
#include "reference_operator_actions.hpp"

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using native_graph_v1::Bytes;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Bytes pack(std::initializer_list<std::int64_t> values) {
  Bytes bytes;
  bytes.reserve(values.size() * 8);
  for (const auto value : values) {
    const auto word = std::bit_cast<std::uint64_t>(value);
    for (unsigned shift = 0; shift < 64; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(word >> shift));
    }
  }
  return bytes;
}

void check_reference_roundtrip(const char* action, const Bytes& input) {
  const reference_operator_actions::Names actions{action};
  const auto state = reference_operator_actions::serialized(input, actions);
  check(reference_operator_actions::inverse_serialized(state, actions) == input,
        "reference repair round-trip failed");
  const auto wire = reference_operator_actions::encode(input, actions);
  check(reference_operator_actions::decode(wire) == input,
        "reference repair encoded round-trip failed");
}
}  // namespace

int main() try {
  // This three-value case made the old A015 inverse form an out-of-range
  // signed sum even though forward had already reduced the prediction with
  // numpy_cast and modular subtraction.
  check_reference_roundtrip(
      "op.regression_predictor.reference_v1",
      pack({std::numeric_limits<std::int64_t>::min(),
            -4600000000000000000LL,
            std::numeric_limits<std::int64_t>::max()}));

  // The unbounded A018 Tukey upper fence exceeds INT64_MAX. Saturating the
  // fence is exact on an int64 domain; exceptions retain every clipped value.
  check_reference_roundtrip(
      "op.outlier_separation.reference_v1",
      pack({0, 0, 4602678819172647000LL, 4602678819172647000LL}));

  std::cout << "{\"status\":\"pass\",\"scope\":\"A015_A018_only\","
               "\"a015_roundtrip\":true,\"a018_roundtrip\":true}\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
