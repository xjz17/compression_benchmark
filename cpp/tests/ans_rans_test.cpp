#include "ans_rans.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void expect_roundtrip(const std::vector<std::uint8_t>& input) {
  const auto encoded = ans_rans::encode(input);
  const auto decoded = ans_rans::decode(encoded);
  if (decoded != input) throw std::runtime_error("ANS roundtrip mismatch");
}

std::uint64_t fnv1a(std::span<const std::uint8_t> bytes) {
  std::uint64_t hash = 14695981039346656037ull;
  for (const auto byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

void expect_python_wire(const std::vector<std::uint8_t>& input,
                        std::size_t expected_size, std::uint64_t expected_hash) {
  const auto encoded = ans_rans::encode(input);
  if (encoded.size() != expected_size || fnv1a(encoded) != expected_hash) {
    throw std::runtime_error("ANS Python/C++ wire mismatch");
  }
}

void expect_rejects(std::vector<std::uint8_t> encoded) {
  try {
    static_cast<void>(ans_rans::decode(encoded));
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("ANS accepted malformed stream");
}

}  // namespace

int main() {
  expect_roundtrip({});
  expect_roundtrip({0});
  expect_roundtrip({255});
  expect_roundtrip(std::vector<std::uint8_t>(65536, 0));

  std::vector<std::uint8_t> all_symbols;
  all_symbols.reserve(256 * 64);
  for (int repeat = 0; repeat < 64; ++repeat) {
    for (int value = 0; value < 256; ++value) {
      all_symbols.push_back(static_cast<std::uint8_t>(value));
    }
  }
  expect_roundtrip(all_symbols);
  expect_python_wire(all_symbols, 16904, 0xdfde02b4fac3c085ull);

  std::vector<std::uint8_t> skewed(131072);
  std::uint32_t state = 0x12345678u;
  for (auto& value : skewed) {
    state = state * 1664525u + 1013904223u;
    value = static_cast<std::uint8_t>((state >> 24) & 0x0fu);
  }
  expect_roundtrip(skewed);
  expect_python_wire(skewed, 66228, 0xa0830c516fad022cull);
  auto truncated = ans_rans::encode(skewed);
  truncated.pop_back();
  expect_rejects(std::move(truncated));
  auto trailing = ans_rans::encode(all_symbols);
  trailing.push_back(0);
  expect_rejects(std::move(trailing));
  return 0;
}
