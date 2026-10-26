#include "optimal_rle.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<std::uint8_t>;
using Values = std::vector<std::uint64_t>;
std::size_t checks = 0, oracle_cases = 0, rejected = 0;
std::size_t differential_cases = 0, differential_bytes = 0;

void check(bool value, const char* detail) {
  ++checks;
  if (!value) throw std::runtime_error(detail);
}

Bytes raw(const Values& values, unsigned width) {
  Bytes out;
  for (auto value : values)
    for (unsigned i = 0; i < width; ++i) out.push_back(value >> (8 * i));
  return out;
}

// Independent oracle: recursive enumeration of every legal segmentation.
// Deliberately no DP table, RMQ, core helpers, or std::bit_width.
unsigned oracle_width(std::uint64_t value) {
  unsigned width = 0;
  while (value) { value /= 2; ++width; }
  return width;
}
unsigned oracle_varint(std::uint64_t value) {
  unsigned bytes = 1;
  while (value > 127) { value /= 128; ++bytes; }
  return bytes;
}
std::size_t brute_segments(const Values& v, std::size_t start) {
  if (start == v.size()) return 0;
  std::size_t answer = std::numeric_limits<std::size_t>::max();
  std::uint64_t aggregate = 0;
  bool same = true;
  for (std::size_t length = 1; start + length <= v.size(); ++length) {
    aggregate |= v[start + length - 1];
    same &= v[start + length - 1] == v[start];
    const auto suffix = brute_segments(v, start + length);
    if (length <= 128) {
      const auto literal = oracle_varint((length - 1) * 2) + 1 +
                           (length * oracle_width(aggregate) + 7) / 8;
      answer = std::min(answer, literal + suffix);
    }
    if (length >= 2 && same) {
      const auto run = oracle_varint((length - 2) * 2 + 1) + 1 +
                       (oracle_width(v[start]) + 7) / 8;
      answer = std::min(answer, run + suffix);
    }
  }
  return answer;
}
std::size_t brute_frame_size(const Values& v, unsigned width) {
  std::size_t body = v.size() * width;
  if (!v.empty()) {
    const bool same = std::all_of(v.begin(), v.end(), [&](auto a) { return a == v[0]; });
    if (same) body = std::min<std::size_t>(body, width);
    std::uint64_t aggregate = 0;
    for (auto a : v) aggregate |= a;
    body = std::min(body, 1 + (v.size() * oracle_width(aggregate) + 7) / 8);
    body = std::min(body, brute_segments(v, 0));
  }
  return 6 + oracle_varint(v.size()) + body;
}

void oracle_case(const Values& values, unsigned width) {
  const auto source = raw(values, width);
  const auto encoded = unified_rle::encode(source, width);
  check(encoded == unified_rle::testing::encode_frozen_reference(source, width),
        "optimized/reference bitstream mismatch");
  check(encoded.size() == brute_frame_size(values, width), "not grammar-byte-optimal");
  check(unified_rle::decode(encoded) == source, "oracle roundtrip");
  check(unified_rle::symbol_width(encoded) == width, "symbol width");
  ++oracle_cases;
}
void enumerate(Values& values, std::size_t target, unsigned width, const Values& alphabet) {
  if (values.size() == target) { oracle_case(values, width); return; }
  for (auto a : alphabet) {
    values.push_back(a);
    enumerate(values, target, width, alphabet);
    values.pop_back();
  }
}

template<class Function> void must_reject(Function operation) {
  bool failed = false;
  try { operation(); } catch (const std::exception&) { failed = true; }
  check(failed, "malformed input was accepted");
  ++rejected;
}
void rejects(const Bytes& frame) { must_reject([&] { unified_rle::decode(frame); }); }

void put_varint(Bytes& out, std::uint64_t value) {
  while (value >= 128) { out.push_back((value % 128) + 128); value /= 128; }
  out.push_back(value);
}
Bytes frame(unsigned width, unsigned mode, std::uint64_t count, Bytes body = {}) {
  Bytes out{'U','R','L','1',static_cast<std::uint8_t>(width),static_cast<std::uint8_t>(mode)};
  put_varint(out, count);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

void exhaustive() {
  Values values;
  for (std::size_t n = 0; n <= 8; ++n) enumerate(values, n, 1, {0,1,255});
  for (unsigned width = 2; width <= 8; ++width) {
    const auto mask = width == 8 ? UINT64_MAX : (UINT64_C(1) << (8 * width)) - 1;
    for (std::size_t n = 0; n <= 6; ++n) enumerate(values, n, width, {0,1,mask});
  }
}

void random_roundtrip() {
  std::mt19937_64 rng(0x55524c31);
  for (unsigned width = 1; width <= 8; ++width) {
    const auto mask = width == 8 ? UINT64_MAX : (UINT64_C(1) << (8 * width)) - 1;
    for (unsigned trial = 0; trial < 100; ++trial) {
      Values values(rng() % 400);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = trial % 3 == 0 ? (rng() & 7) : (i && rng() % 3 == 0 ? values[i-1] : rng() & mask);
      const auto source = raw(values, width);
      const auto encoded = unified_rle::encode(source, width);
      check(encoded == unified_rle::testing::encode_frozen_reference(source, width),
            "random optimized/reference bitstream mismatch");
      check(unified_rle::decode(encoded, source.size()) == source, "random roundtrip");
      check(unified_rle::encode(source, width) == encoded, "nondeterministic encoding");
      check(encoded.size() <= 6 + oracle_varint(values.size()) + source.size(), "raw bound");
    }
  }
}

void differential_case(const Values& values, unsigned width) {
  const auto source = raw(values, width);
  const auto optimized = unified_rle::encode(source, width);
  const auto reference = unified_rle::testing::encode_frozen_reference(source, width);
  check(optimized == reference, "targeted optimized/reference bitstream mismatch");
  check(unified_rle::decode(optimized, source.size()) == source,
        "targeted optimized roundtrip");
  check(unified_rle::decode(reference, source.size()) == source,
        "targeted reference roundtrip");
  ++differential_cases;
  differential_bytes += source.size();
}

void differential() {
  std::mt19937_64 rng(0x55524c3150415249ULL);
  constexpr std::size_t lengths[] = {
      0,1,2,3,7,8,9,63,64,65,66,127,128,129,130,255,256,257,1023,8192};
  for (unsigned width = 1; width <= 8; ++width) {
    const auto mask = width == 8 ? UINT64_MAX
                                 : (UINT64_C(1) << (8 * width)) - 1;
    for (const auto length : lengths) {
      Values values(length);
      for (std::size_t i = 0; i < length; ++i) {
        if (i && i % 11 < 5) values[i] = values[i - 1];
        else if (i % 17 == 0) values[i] = 0;
        else values[i] = rng() & mask;
      }
      differential_case(values, width);
      for (std::size_t i = 0; i < length; ++i)
        values[i] = ((i / 64) ^ (i / 127)) & mask;
      differential_case(values, width);
    }
  }

  // Byte streams exercise every literal-token boundary and bit-width band.
  constexpr std::size_t large = 256 * 1024;
  for (unsigned pattern = 0; pattern < 6; ++pattern) {
    Values values(large);
    for (std::size_t i = 0; i < values.size(); ++i) {
      switch (pattern) {
        case 0: values[i] = rng() & 255; break;
        case 1: values[i] = (i / 5 + i / 131) & 255; break;
        case 2: values[i] = i % 97 ? (rng() & 7) : 255; break;
        case 3: values[i] = (i / 64) & 1 ? 0 : (i & 255); break;
        case 4: values[i] = (i % 257 < 129) ? 1 : (i & 15); break;
        default: values[i] = (i * 73 + (i >> 3)) & 255; break;
      }
    }
    differential_case(values, 1);
  }
}

std::pair<double,double> profile_byte_path() {
  constexpr std::size_t count = 256 * 1024;
  std::mt19937_64 rng(0x55524c3153504545ULL);
  Values values(count);
  for (std::size_t i = 0; i < count; ++i)
    values[i] = i && i % 7 < 3 ? values[i - 1] : rng() & 255;
  const auto source = raw(values, 1);
  const auto old_start = std::chrono::steady_clock::now();
  const auto reference = unified_rle::testing::encode_frozen_reference(source, 1);
  const auto old_elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - old_start).count();
  const auto new_start = std::chrono::steady_clock::now();
  const auto optimized = unified_rle::encode(source, 1);
  const auto new_elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - new_start).count();
  check(optimized == reference, "profile optimized/reference bitstream mismatch");
  check(unified_rle::decode(optimized, source.size()) == source,
        "profile optimized decode mismatch");
  return {old_elapsed, new_elapsed};
}

void boundaries() {
  for (auto n : {1,2,63,64,65,66,127,128,129,8192,8193,8194}) {
    Values same(n, 1);
    auto source = raw(same, 8);
    auto encoded = unified_rle::encode(source, 8);
    check(unified_rle::decode(encoded) == source, "repeat boundary");
    // On a uniform sequence, every split adds another width and value field;
    // one run/literal/global representation dominates every split.
    std::size_t candidate = std::min<std::size_t>(n * 8, 8);
    candidate = std::min(candidate, std::size_t(1 + (n + 7) / 8));
    if (n >= 2) candidate = std::min(candidate, std::size_t(oracle_varint(2ULL*n-3)+2));
    check(encoded.size() == 6 + oracle_varint(n) + candidate, "repeat varint threshold optimum");
    same[0] = UINT64_MAX;
    same[n-1] = UINT64_C(0x100000000);
    source = raw(same, 8);
    encoded = unified_rle::encode(source, 8);
    check(unified_rle::decode(encoded) == source, "mixed boundary");
  }
  const Values ieee{0,UINT64_C(0x8000000000000000),1,UINT64_C(0x000fffffffffffff),
      UINT64_C(0x0010000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
      UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff8000000001234),UINT64_C(0x7ff0000000000001),
      UINT64_C(0x3ff0000000000000),UINT64_MAX};
  auto source = raw(ieee, 8);
  check(unified_rle::decode(unified_rle::encode(source,8)) == source, "IEEE-754 bit pattern preservation");
  source = raw({1,2,3}, 1);
  check(unified_rle::encode(source,1) == frame(1,2,3,{2,57}), "little-endian bitpack layout");
  check(unified_rle::encode(raw(Values(200,0),1),1)[5] == 1, "constant wins equal-size mode tie");
  check(unified_rle::encode({},8) == frame(8,0,0), "canonical empty frame");
}

void malformed() {
  const auto good = unified_rle::encode(raw({1,2,3,3,3,3,4,5,6},8),8);
  for (std::size_t length = 0; length < good.size(); ++length)
    rejects(Bytes(good.begin(),good.begin()+length));
  auto trailing = good; trailing.push_back(0); rejects(trailing);
  rejects(frame(0,0,0)); rejects(frame(9,0,0)); rejects(frame(1,4,0));
  rejects(frame(1,1,0)); rejects(frame(1,2,0)); rejects(frame(1,3,0));
  rejects(frame(8,0,UINT64_MAX));
  rejects({'U','R','L','1',1,0,0x80,0});
  rejects({'U','R','L','1',1,0,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,2});
  rejects(frame(1,2,3,{1,128}));  // nonzero padding
  rejects(frame(1,2,1,{2,1}));    // nonminimal bit width
  rejects(frame(1,2,1,{9,0,0}));  // greater than symbol width
  rejects(frame(1,3,2,{1,1,128})); // repeat nonzero padding
  rejects(frame(1,3,1,{1,0}));     // repeat count exceeds total
  rejects(frame(1,3,129,{128,2,0})); // literal length 129
  rejects(frame(1,3,2,{128,0,0})); // overlong token
  rejects(frame(1,3,2,{1,8,1}));   // nonminimal repeat width
  rejects(frame(1,0,3,{1,2}));
  auto bad_magic = good; bad_magic[0] = 'X'; rejects(bad_magic);
  must_reject([] { unified_rle::encode(Bytes{1},0); });
  must_reject([] { unified_rle::encode(Bytes{1},9); });
  must_reject([] { unified_rle::encode(Bytes{1},2); });
  const auto bounded = frame(8,1,100,{1,0,0,0,0,0,0,0});
  must_reject([&] { unified_rle::decode(bounded,799); });
  check(unified_rle::decode(bounded,800).size() == 800, "explicit decode bound");
  must_reject([] { unified_rle::decode(frame(8,1,unified_rle::kDefaultMaxDecodedBytes/8+1,{0,0,0,0,0,0,0,0})); });
  check(unified_rle::decode(frame(8,0,0),0).empty(), "zero limit permits empty frame");
  check(unified_rle::is_frame(Bytes{'U','R','L','1'}), "magic recognition");
  check(!unified_rle::is_frame(Bytes{'U','R','L'}), "short magic recognition");
}
}

int main() try {
  exhaustive(); random_roundtrip(); boundaries(); malformed(); differential();
  const auto [reference_ms, optimized_ms] = profile_byte_path();
  const auto start = std::chrono::steady_clock::now();
  const auto source = raw(Values(200000,1),8);
  const auto encoded = unified_rle::encode(source,8);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  check(unified_rle::decode(encoded) == source, "long-run roundtrip");
  check(encoded.size() == 14, "long-run size");
  std::cout << "optimal_rle=PASS oracle_cases=" << oracle_cases
            << " checks=" << checks << " malformed_rejected=" << rejected
            << " long_run_symbols=200000 encoded_bytes=" << encoded.size()
            << " encode_seconds=" << elapsed
            << " differential_cases=" << differential_cases
            << " differential_bytes=" << differential_bytes
            << " byte_256k_reference_ms=" << reference_ms
            << " byte_256k_optimized_ms=" << optimized_ms
            << " byte_256k_speedup=" << reference_ms / optimized_ms << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "optimal_rle=FAIL " << error.what() << '\n';
  return 1;
}
