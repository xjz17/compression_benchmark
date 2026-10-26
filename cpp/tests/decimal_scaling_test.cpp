#include "decimal_scaling.hpp"
#include <cmath>

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using native_baseline86_v1::decimal_scaling::Bytes;
using native_baseline86_v1::decimal_scaling::Scaled;
using native_baseline86_v1::decimal_scaling::forward;
using native_baseline86_v1::decimal_scaling::inspect;
using native_baseline86_v1::decimal_scaling::inverse;
using native_baseline86_v1::decimal_scaling::required_decimal_places;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Float>
using UInt = std::conditional_t<sizeof(Float) == 4, std::uint32_t,
                                std::uint64_t>;

template <class Float>
Bytes bytes(const std::vector<Float>& values) {
  Bytes output;
  output.reserve(values.size() * sizeof(Float));
  for (const auto value : values) {
    const auto raw = std::bit_cast<UInt<Float>>(value);
    for (std::size_t byte = 0; byte < sizeof(Float); ++byte)
      output.push_back(static_cast<std::uint8_t>(raw >> (byte * 8)));
  }
  return output;
}

template <class Float>
Float value(UInt<Float> raw) {
  return std::bit_cast<Float>(raw);
}

template <class Function>
void throws(Function&& function, const char* message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

std::uint64_t splitmix64(std::uint64_t& state) {
  state += UINT64_C(0x9e3779b97f4a7c15);
  auto result = state;
  result = (result ^ (result >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  result = (result ^ (result >> 27)) * UINT64_C(0x94d049bb133111eb);
  return result ^ (result >> 31);
}

template <class Float>
std::uint64_t random_roundtrips(std::size_t blocks, std::size_t lanes,
                                std::uint64_t seed, std::string_view dtype) {
  std::uint64_t exceptions = 0;
  for (std::size_t block = 0; block < blocks; ++block) {
    std::vector<Float> input;
    input.reserve(lanes);
    for (std::size_t lane = 0; lane < lanes; ++lane)
      input.push_back(value<Float>(static_cast<UInt<Float>>(splitmix64(seed))));
    const auto raw = bytes(input);
    const auto places = required_decimal_places(raw, dtype);
    const auto scaled = forward(raw, dtype, places);
    require(inverse(scaled.values, scaled.metadata, dtype) == raw,
            "random corpus bitwise roundtrip");
    exceptions += scaled.exception_count;
  }
  return exceptions;
}

}  // namespace

int main() {
  std::uint64_t assertions = 0;
  const auto check = [&](bool condition, const char* message) {
    ++assertions;
    require(condition, message);
  };

  const std::vector<double> ordinary{0.1, 1.25, -2.5,
      std::nextafter(1.0, 2.0)};
  const auto ordinary_raw = bytes(ordinary);
  const auto ordinary_places = required_decimal_places(ordinary_raw, "float64");
  check(ordinary_places == 16, "binary64 shortest scale mismatch");
  const auto ordinary_scaled = forward(ordinary_raw, "float64", ordinary_places);
  const auto ordinary_info = inspect(ordinary_scaled.metadata, "float64");
  check(ordinary_info.decimal_places == ordinary_places &&
            ordinary_info.original_count == ordinary.size() &&
            ordinary_info.scaled_count == ordinary.size() &&
            ordinary_info.exception_count == 0,
        "ordinary metadata inspection mismatch");
  check(ordinary_scaled.exception_count == 0 &&
            ordinary_scaled.values.size() == ordinary.size(),
        "ordinary binary64 unexpectedly used exceptions");
  check(inverse(ordinary_scaled.values, ordinary_scaled.metadata, "float64") ==
            ordinary_raw,
        "ordinary binary64 bitwise roundtrip");
  throws([&] { (void)forward(ordinary_raw, "float64", ordinary_places - 1); },
         "forward accepted p below canonical requirement");
  ++assertions;

  const std::vector<float> ordinary32{0.1F, 1.25F,
      std::nextafter(1.0F, 2.0F)};
  const auto ordinary32_raw = bytes(ordinary32);
  const auto ordinary32_places =
      required_decimal_places(ordinary32_raw, "float32");
  check(ordinary32_places == 7, "binary32 shortest scale mismatch");
  const auto ordinary32_scaled =
      forward(ordinary32_raw, "float32", ordinary32_places);
  check(inverse(ordinary32_scaled.values, ordinary32_scaled.metadata, "float32") ==
            ordinary32_raw,
        "ordinary binary32 bitwise roundtrip");

  const std::vector<double> special{
      -0.0,
      value<double>(UINT64_C(0x7ff8000000001234)),
      value<double>(UINT64_C(0x7ff0000000000042)),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  const auto special_raw = bytes(special);
  check(required_decimal_places(special_raw, "float64") == 0,
        "special IEEE values affected decimal places");
  const auto special_scaled = forward(special_raw, "float64", 0);
  check(special_scaled.values.empty() &&
            special_scaled.exception_count == special.size(),
        "special IEEE values were not raw exceptions");
  check(inverse(special_scaled.values, special_scaled.metadata, "float64") ==
            special_raw,
        "special IEEE payload bits changed");

  const std::vector<double> subnormal{
      std::numeric_limits<double>::denorm_min()};
  const auto subnormal_raw = bytes(subnormal);
  check(required_decimal_places(subnormal_raw, "float64") == 324,
        "binary64 minimum subnormal scale mismatch");
  const auto subnormal_scaled = forward(subnormal_raw, "float64", 324);
  check(subnormal_scaled.exception_count == 0 &&
            subnormal_scaled.values == std::vector<std::int64_t>{5},
        "binary64 subnormal was not mapped to 5e-324");
  check(inverse(subnormal_scaled.values, subnormal_scaled.metadata, "float64") ==
            subnormal_raw,
        "binary64 subnormal bitwise roundtrip");

  const std::vector<float> subnormal32{
      std::numeric_limits<float>::denorm_min()};
  const auto subnormal32_raw = bytes(subnormal32);
  check(required_decimal_places(subnormal32_raw, "float32") == 45,
        "binary32 minimum subnormal scale mismatch");
  const auto subnormal32_scaled = forward(subnormal32_raw, "float32", 45);
  check(subnormal32_scaled.exception_count == 0 &&
            subnormal32_scaled.values == std::vector<std::int64_t>{1},
        "binary32 subnormal was not mapped to 1e-45");
  check(inverse(subnormal32_scaled.values, subnormal32_scaled.metadata,
                "float32") == subnormal32_raw,
        "binary32 subnormal bitwise roundtrip");

  const std::vector<double> large{
      std::numeric_limits<double>::max(), 1.0e20, -1.0e20};
  const auto large_raw = bytes(large);
  const auto large_scaled = forward(
      large_raw, "float64", required_decimal_places(large_raw, "float64"));
  check(large_scaled.values.empty() &&
            large_scaled.exception_count == large.size(),
        "large exponents did not take raw exception path");
  check(inverse(large_scaled.values, large_scaled.metadata, "float64") ==
            large_raw,
        "large exponent bitwise roundtrip");

  const std::vector<double> mixed{0.1,
      std::numeric_limits<double>::denorm_min(), -2.5};
  const auto mixed_raw = bytes(mixed);
  const auto mixed_scaled = forward(mixed_raw, "float64", 324);
  check(mixed_scaled.decimal_places == 324 &&
            mixed_scaled.exception_count == 2 &&
            mixed_scaled.values == std::vector<std::int64_t>{5},
        "common p overflow exception accounting");
  check(inverse(mixed_scaled.values, mixed_scaled.metadata, "float64") ==
            mixed_raw,
        "common p overflow bitwise roundtrip");

  // Shortest decimal for the binary64 -2^63 endpoint may lie just outside the
  // signed-int64 decimal range.  It must fail closed to raw bits, never cast
  // through an overflowing signed integer.
  const std::vector<double> endpoints{
      static_cast<double>(std::numeric_limits<std::int64_t>::min()),
      static_cast<double>(std::numeric_limits<std::int64_t>::max())};
  const auto endpoint_raw = bytes(endpoints);
  const auto endpoint_scaled = forward(
      endpoint_raw, "float64", required_decimal_places(endpoint_raw, "float64"));
  check(inverse(endpoint_scaled.values, endpoint_scaled.metadata, "float64") ==
            endpoint_raw,
        "int64 endpoint bitwise roundtrip");

  const auto empty = forward({}, "float64", 0);
  check(empty.values.empty() && empty.exception_count == 0 &&
            inverse(empty.values, empty.metadata, "float64").empty(),
        "empty compact valid stream");

  throws([&] { (void)required_decimal_places(ordinary_raw, "int64"); },
         "required_decimal_places accepted nonfloat dtype");
  throws([&] { (void)forward(ordinary_raw, "float32", 16); },
         "forward accepted unaligned/wrong dtype");
  throws([&] { (void)forward(ordinary_raw, "float64", 325); },
         "forward accepted p above 324");
  assertions += 3;

  throws([&] {
    auto bad = ordinary_scaled.metadata;
    bad[0] ^= 1;
    (void)inverse(ordinary_scaled.values, bad, "float64");
  }, "inverse accepted bad magic");
  throws([&] {
    auto bad = ordinary_scaled.metadata;
    bad[4] = 2;
    (void)inverse(ordinary_scaled.values, bad, "float64");
  }, "inverse accepted bad version");
  throws([&] {
    auto bad = ordinary_scaled.metadata;
    bad[6] = 69;
    bad[7] = 1;  // 325.
    (void)inverse(ordinary_scaled.values, bad, "float64");
  }, "inverse accepted p above 324");
  throws([&] {
    auto bad = ordinary_scaled.metadata;
    bad.push_back(0);
    (void)inverse(ordinary_scaled.values, bad, "float64");
  }, "inverse accepted trailing metadata");
  throws([&] {
    (void)inverse(ordinary_scaled.values, ordinary_scaled.metadata, "float32");
  }, "inverse accepted dtype mismatch");
  throws([&] {
    std::vector<std::int64_t> short_values(ordinary_scaled.values.begin(),
                                           ordinary_scaled.values.end() - 1);
    (void)inverse(short_values, ordinary_scaled.metadata, "float64");
  }, "inverse accepted truncated integer payload");
  assertions += 6;

  throws([&] {
    auto bad = special_scaled.metadata;
    bad.pop_back();
    (void)inverse(special_scaled.values, bad, "float64");
  }, "inverse accepted truncated raw exception payload");
  throws([&] {
    auto bad = special_scaled.metadata;
    bad[48] |= 0x80;  // Five lanes use bits 0..4; bit 7 is padding.
    (void)inverse(special_scaled.values, bad, "float64");
  }, "inverse accepted nonzero bitmap padding");
  throws([&] {
    auto bad = special_scaled.metadata;
    bad[24] = 4;  // Header says four exceptions, bitmap contains five.
    (void)inverse(special_scaled.values, bad, "float64");
  }, "inverse accepted exception count mismatch");
  assertions += 3;

  // Replace the raw bits of a canonical exception with +1.0.  +1.0 is exactly
  // scalable at p=0, so an exception marker for it is noncanonical.
  throws([&] {
    const auto one_exception = forward(bytes(std::vector<double>{1.0e20}),
                                       "float64", 0);
    auto bad = one_exception.metadata;
    const auto one = std::bit_cast<std::uint64_t>(1.0);
    for (std::size_t byte = 0; byte < 8; ++byte)
      bad[49 + byte] = static_cast<std::uint8_t>(one >> (byte * 8));
    (void)inverse(one_exception.values, bad, "float64");
  }, "inverse accepted a noncanonical raw exception");
  ++assertions;

  constexpr std::size_t random_blocks = 2500;
  constexpr std::size_t random_lanes = 16;
  const auto random64_exceptions = random_roundtrips<double>(
      random_blocks, random_lanes, UINT64_C(0x6a09e667f3bcc909), "float64");
  const auto random32_exceptions = random_roundtrips<float>(
      random_blocks, random_lanes, UINT64_C(0xbb67ae8584caa73b), "float32");
  assertions += 2 * random_blocks * random_lanes;

  std::cout
      << "{\"schema\":\"decimal-scaling-module-test-v1\","
      << "\"status\":\"PASS\","
      << "\"assertions\":" << assertions << ','
      << "\"binary64_required_places\":" << ordinary_places << ','
      << "\"binary32_required_places\":" << ordinary32_places << ','
      << "\"binary64_min_subnormal_places\":324,"
      << "\"binary32_min_subnormal_places\":45,"
      << "\"random_binary64_lanes\":" << random_blocks * random_lanes << ','
      << "\"random_binary64_exceptions\":" << random64_exceptions << ','
      << "\"random_binary32_lanes\":" << random_blocks * random_lanes << ','
      << "\"random_binary32_exceptions\":" << random32_exceptions
      << "}\n";
}
