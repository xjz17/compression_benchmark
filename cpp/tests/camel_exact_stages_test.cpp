#include "camel_exact_stages.hpp"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::int64_t> bits(const std::vector<double>& values) {
  std::vector<std::int64_t> out(values.size());
  if (!values.empty()) std::memcpy(out.data(), values.data(), values.size() * 8);
  return out;
}

template <class Function>
void require_throws(Function&& function, const char* label) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(label);
}
}  // namespace

int main() {
  using namespace exact_actions::camel;
  for (const std::vector<double> values : {
           std::vector<double>{},
           std::vector<double>{-19.25, -1.0, -0.0, 0.0, 0.125, 3.5, 17.25},
           std::vector<double>{1.0, 1.0, 1.23456789, 10.5, 100.0, 100.5},
           // Decimal candidates with a delta wider than Camel's 16-bit wire
           // field must take the literal escape instead of aborting.
           std::vector<double>{0.0, 65535.0, 65536.0, -65536.0, 1.0e12},
           // Non-finite values are valid float64 bit patterns.  Include two
           // different NaN payloads so this checks bit-exact preservation.
           std::vector<double>{
               std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::quiet_NaN(),
               [] {
                 std::uint64_t payload = 0x7ff8000000000042ULL;
                 double value = 0;
                 std::memcpy(&value, &payload, sizeof(value));
                 return value;
               }()}}) {
    const auto raw = bits(values);
    const auto erased = eraser_forward(raw);
    if (eraser_inverse(erased) != raw)
      throw std::runtime_error("Camel eraser round trip");
    const auto xors = xor_forward(erased);
    if (xor_inverse(xors) != erased)
      throw std::runtime_error("Camel XOR round trip");
    const auto wire = bitstream_pack_forward(xors);
    if (bitstream_pack_inverse(wire) != xors)
      throw std::runtime_error("Camel bitstream round trip");
    if (decompose(wire) != raw)
      throw std::runtime_error("Camel chain round trip");
  }
  const std::vector<std::int64_t> integers{-1000, -1, 0, 1, 17, 10000};
  const auto integer_wire = compose_int64_numeric(integers);
  if (decompose_int64_numeric(integer_wire) != integers)
    throw std::runtime_error("Camel integer conversion round trip");

  // These values exercise the half-open int64 conversion boundary.  In
  // particular, binary64(INT64_MAX) is +2^63 and must never be cast to int64.
  for (const std::vector<std::int64_t> boundary_integers : {
           std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::min()},
           std::vector<std::int64_t>{
               std::numeric_limits<std::int64_t>::max() - 1023}}) {
    const auto wire = compose_int64_numeric(boundary_integers);
    if (decompose_int64_numeric(wire) != boundary_integers)
      throw std::runtime_error("Camel int64 boundary round trip");
  }
  require_throws(
      [] {
        (void)compose_int64_numeric(
            std::vector<std::int64_t>{
                std::numeric_limits<std::int64_t>::max()});
      },
      "Camel INT64_MAX exactness rejection");
  require_throws(
      [] {
        (void)compose_int64_numeric(
            std::vector<std::int64_t>{
                std::numeric_limits<std::int64_t>::max() - 1});
      },
      "Camel rounded int64 exactness rejection");

  // Float64 literals outside int64 remain raw literals; their companion
  // integer history uses Java's defined saturating conversion.
  const std::vector<double> wide_literals{
      0x1p63, -0x1p64, std::nextafter(0x1p63, 0.0)};
  for (double value : wide_literals) {
    const auto raw = bits(std::vector<double>{0.0, value});
    const auto wire = compose(raw);
    if (decompose(wire) != raw)
      throw std::runtime_error("Camel wide literal round trip");
  }
  require_throws(
      [] {
        const auto packet = eraser_forward(bits(std::vector<double>{0x1p63}));
        (void)eraser_inverse_int64_numeric(packet);
      },
      "Camel +2^63 numeric inverse rejection");
  std::cout << "camel_exact_stage_actions=3 status=PASS\n";
}
