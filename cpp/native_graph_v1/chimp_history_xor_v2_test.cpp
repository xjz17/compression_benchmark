#include "chimp_history_xor_v2.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace chimp = native_graph_v1::chimp_history_xor_v2;

namespace {

void require(bool condition, const std::string& why) {
  if (!condition) throw std::runtime_error(why);
}

template <class Function>
void rejects(Function&& function, const std::string& why) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, why);
}

std::vector<std::uint8_t> bytes(std::span<const std::uint64_t> words) {
  std::vector<std::uint8_t> result(words.size() * 8);
  if (!result.empty()) std::memcpy(result.data(), words.data(), result.size());
  return result;
}

std::vector<std::uint8_t> oracle_references(
    std::span<const std::uint64_t> words) {
  if (words.empty()) return {};
  std::array<std::uint64_t, 128> stored{};
  std::array<std::size_t, 16384> indices{};
  stored[0] = words.front();
  indices[words.front() & 16383] = 0;
  std::size_t index = 0;
  std::size_t current = 0;
  std::vector<std::uint8_t> references;
  references.reserve(words.size() - 1);
  for (std::size_t i = 1; i < words.size(); ++i) {
    const auto value = words[i];
    const auto key = value & 16383;
    const auto candidate_index = indices[key];
    std::size_t reference = index % 128;
    if (index - candidate_index < 128) {
      const auto slot = candidate_index % 128;
      if (std::countr_zero(value ^ stored[slot]) > 13) reference = slot;
    }
    references.push_back(static_cast<std::uint8_t>(reference));
    current = (current + 1) % 128;
    stored[current] = value;
    ++index;
    indices[key] = index;
  }
  return references;
}

void roundtrip(const std::vector<std::uint64_t>& words,
               chimp::Profile profile) {
  const auto encoded = chimp::encode_words(words, profile);
  require(encoded.reference_slots == oracle_references(words),
          "reference selection differs from the canonical low14/history128 oracle");
  require(chimp::decode_words(encoded) == words, "word round-trip mismatch");

  const auto raw = bytes(words);
  const auto payload = chimp::encode_bytes(raw, profile);
  require(payload.metadata == chimp::serialize_metadata(encoded),
          "stage metadata is not deterministic");
  require(chimp::encode_bytes(raw, profile) == payload,
          "stage payload is not deterministic");
  require(chimp::decode_bytes(payload, profile) == raw,
          "stage byte round-trip mismatch");
  require(chimp::parse_stage_payload(payload.metadata, payload.reference_slots,
                                     payload.xor_residuals_le, profile) == encoded,
          "split stage payload parse mismatch");
}

void test_empty_and_singleton() {
  for (const auto profile : {chimp::Profile::Int64, chimp::Profile::Float64}) {
    roundtrip({}, profile);
    roundtrip({0}, profile);
    roundtrip({std::numeric_limits<std::uint64_t>::max()}, profile);
  }
}

void test_reference_selection_and_history_wrap() {
  const std::vector<std::uint64_t> branch_fixture{
      0x0000000000010000ULL,  // key 0, slot 0
      0x0000000000000001ULL,  // absent key: previous-word fallback slot 0
      0x0000000000000002ULL,  // absent key: previous-word fallback slot 1
      0x0000000000020000ULL,  // key 0: non-adjacent slot 0, ctz > 13
  };
  const auto encoded = chimp::encode_words(branch_fixture, chimp::Profile::Int64);
  require(encoded.reference_slots == std::vector<std::uint8_t>({0, 1, 0}),
          "candidate and fallback branch fixture chose unexpected slots");
  roundtrip(branch_fixture, chimp::Profile::Int64);

  std::vector<std::uint64_t> wrapped;
  wrapped.reserve(400);
  for (std::uint64_t i = 0; i < 400; ++i) {
    // Repeated keys, exact repeats, absent keys, and enough values to evict
    // multiple generations from the 128-entry ring.
    std::uint64_t value = ((i * 0x9e3779b97f4a7c15ULL) & ~std::uint64_t{16383}) |
                          ((i * 37) & 16383);
    if (i > 4 && i % 29 == 0) value = wrapped[i - 4];
    if (i > 140 && i % 31 == 0)
      value = (wrapped[i - 140] & 16383) | (i << 20);
    wrapped.push_back(value);
  }
  roundtrip(wrapped, chimp::Profile::Int64);
}

void test_float_bit_patterns() {
  const std::vector<std::uint64_t> bits{
      0x0000000000000000ULL,  // +0
      0x8000000000000000ULL,  // -0 and the old int64 sentinel
      0x7ff0000000000000ULL,  // +inf
      0xfff0000000000000ULL,  // -inf
      0x7ff8000000000000ULL,  // canonical qNaN and the old float sentinel
      0x7ff8000000000001ULL,  // distinct NaN payload
      0xfff8123456789abcULL,  // signed NaN payload
      0x0000000000000001ULL,  // minimum positive subnormal
      0x7fefffffffffffffULL,  // maximum finite positive
      0xffefffffffffffffULL,  // maximum finite negative
      0x3ff0000000000000ULL,  // 1.0
      0xbff0000000000000ULL,  // -1.0
  };
  roundtrip(bits, chimp::Profile::Float64);
}

void test_malformed_payloads() {
  const std::vector<std::uint64_t> words{1, 2, 3, 0x10001, 4};
  const auto good = chimp::encode_bytes(bytes(words), chimp::Profile::Int64);

  auto malformed = good;
  malformed.metadata.pop_back();
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "truncated metadata accepted");

  malformed = good;
  malformed.metadata[0] ^= 0x80;
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "bad magic/version accepted");

  malformed = good;
  malformed.metadata[6] = 1;
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "non-zero reserved metadata accepted");

  rejects([&] { chimp::decode_bytes(good, chimp::Profile::Float64); },
          "wrong logical profile accepted");

  malformed = good;
  malformed.reference_slots.pop_back();
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "truncated reference stream accepted");

  malformed = good;
  malformed.reference_slots[0] = 128;
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "out-of-range reference slot accepted");

  malformed = good;
  malformed.xor_residuals_le.pop_back();
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "truncated residual stream accepted");

  malformed = good;
  malformed.xor_residuals_le.push_back(0);
  rejects([&] { chimp::decode_bytes(malformed, chimp::Profile::Int64); },
          "trailing residual byte accepted");

  auto noncanonical_empty = chimp::encode_bytes({}, chimp::Profile::Int64);
  noncanonical_empty.metadata[16] = 1;  // first_value must be zero for count=0.
  rejects([&] { chimp::decode_bytes(noncanonical_empty, chimp::Profile::Int64); },
          "non-canonical empty first value accepted");

  const std::array<std::uint8_t, 7> partial{};
  rejects([&] { chimp::encode_bytes(partial, chimp::Profile::Int64); },
          "partial input word accepted");
}

}  // namespace

int main() try {
  test_empty_and_singleton();
  test_reference_selection_and_history_wrap();
  test_float_bit_patterns();
  test_malformed_payloads();
  std::cout << "chimp_history_xor_v2_test: PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "chimp_history_xor_v2_test: FAIL: " << error.what() << '\n';
  return 1;
}
