#include "bos_float_exception.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Compile the frozen pre-optimization implementation into a separate
// namespace.  This keeps wire identity executable instead of relying only on
// hand-maintained golden bytes.
#define bos_float_exception bos_float_exception_before
#include "fixtures/bos_float_exception_before_v1/bos_float_exception.cpp"
#undef bos_float_exception

namespace {

using bos_float_exception::Bytes;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void put_u32(Bytes& output, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_i64(Bytes& output, std::int64_t value) {
  const auto word = std::bit_cast<std::uint64_t>(value);
  for (unsigned shift = 0; shift != 64; shift += 8)
    output.push_back(static_cast<std::uint8_t>(word >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& offset) {
  if (input.size() - std::min(input.size(), offset) < 4)
    throw std::invalid_argument("test int codec truncated count");
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift != 32; shift += 8)
    value |= static_cast<std::uint32_t>(input[offset++]) << shift;
  return value;
}

std::int64_t get_i64(std::span<const std::uint8_t> input, std::size_t& offset) {
  if (input.size() - std::min(input.size(), offset) < 8)
    throw std::invalid_argument("test int codec truncated value");
  std::uint64_t word = 0;
  for (unsigned shift = 0; shift != 64; shift += 8)
    word |= static_cast<std::uint64_t>(input[offset++]) << shift;
  return std::bit_cast<std::int64_t>(word);
}

// A deterministic, self-contained test codec.  Constant streams are compact;
// all other streams use explicit little-endian int64 values.
Bytes encode_int(std::span<const std::int64_t> values) {
  Bytes output;
  const bool constant = !values.empty() &&
                        std::all_of(values.begin(), values.end(),
                                    [&](auto value) { return value == values.front(); });
  output.push_back(constant ? 1 : 0);
  put_u32(output, static_cast<std::uint32_t>(values.size()));
  if (constant) {
    put_i64(output, values.front());
  } else {
    for (const auto value : values) put_i64(output, value);
  }
  return output;
}

std::vector<std::int64_t> decode_int(std::span<const std::uint8_t> input) {
  if (input.size() < 5) throw std::invalid_argument("test int codec truncated header");
  std::size_t offset = 0;
  const auto mode = input[offset++];
  const auto count = get_u32(input, offset);
  std::vector<std::int64_t> output;
  output.reserve(count);
  if (mode == 1) {
    const auto value = get_i64(input, offset);
    output.assign(count, value);
  } else if (mode == 0) {
    for (std::uint32_t i = 0; i < count; ++i) output.push_back(get_i64(input, offset));
  } else {
    throw std::invalid_argument("test int codec invalid mode");
  }
  if (offset != input.size()) throw std::invalid_argument("test int codec trailing bytes");
  return output;
}

bos_float_exception_before::DType before_dtype(bos_float_exception::DType dtype) {
  return dtype == bos_float_exception::DType::Float32
             ? bos_float_exception_before::DType::Float32
             : bos_float_exception_before::DType::Float64;
}

bos_float_exception_before::EncodeOptions before_options(
    const bos_float_exception::EncodeOptions& options) {
  bos_float_exception_before::EncodeOptions before;
  before.candidate_precisions = options.candidate_precisions;
  before.pilot_values = options.pilot_values;
  before.fixed_precision = options.fixed_precision;
  return before;
}

void expect_same_wire_as_before(
    const Bytes& input,
    bos_float_exception::DType dtype,
    const bos_float_exception::EncodeOptions& options = {}) {
  bos_float_exception::Stats current_stats;
  const auto current = bos_float_exception::encode(
      input, dtype, encode_int, &current_stats, options);
  bos_float_exception_before::Stats old_stats;
  const auto old = bos_float_exception_before::encode(
      input, before_dtype(dtype), encode_int, &old_stats, before_options(options));
  require(current == old, "optimized BFE1 changed the frozen wire");
  require(current_stats.selected_p == old_stats.selected_p,
          "optimized BFE1 changed selected precision");
  require(current_stats.exceptions == old_stats.exceptions,
          "optimized BFE1 changed exception count");
  require(static_cast<unsigned>(current_stats.mode) ==
              static_cast<unsigned>(old_stats.mode),
          "optimized BFE1 changed mode");
  require(bos_float_exception::decode(old, decode_int) == input,
          "current BFE1 decoder rejected frozen wire");
  require(bos_float_exception_before::decode(current, decode_int) == input,
          "frozen BFE1 decoder rejected optimized wire");
}

template <typename Float, typename Word>
Bytes ieee_bytes(const std::vector<Word>& words) {
  static_assert(sizeof(Float) == sizeof(Word));
  Bytes output;
  output.reserve(words.size() * sizeof(Word));
  for (const auto word : words)
    for (std::size_t byte = 0; byte < sizeof(Word); ++byte)
      output.push_back(static_cast<std::uint8_t>(word >> (byte * 8)));
  return output;
}

void expect_round_trip(const Bytes& input, bos_float_exception::DType dtype) {
  bos_float_exception::Stats encoded_stats;
  const auto wire = bos_float_exception::encode(input, dtype, encode_int, &encoded_stats);
  bos_float_exception::Stats decoded_stats;
  const auto output = bos_float_exception::decode(wire, decode_int, &decoded_stats);
  require(output == input, "BFE1 IEEE bytes changed");
  require(encoded_stats.mode == decoded_stats.mode, "BFE1 stats mode mismatch");
  require(encoded_stats.selected_p == decoded_stats.selected_p,
          "BFE1 stats precision mismatch");
  require(encoded_stats.exceptions == decoded_stats.exceptions,
          "BFE1 stats exception mismatch");
}

void edge_cases() {
  const std::vector<std::uint32_t> float_words{
      0x00000000U, 0x80000000U, 0x00000001U, 0x007fffffU,
      0x00800000U, 0x7f7fffffU, 0xff7fffffU, 0x7f800000U,
      0xff800000U, 0x7fc00001U, 0x7fa12345U, 0xffcabcdeU,
      std::bit_cast<std::uint32_t>(1.25F), std::bit_cast<std::uint32_t>(-91.0625F)};
  expect_round_trip(ieee_bytes<float>(float_words), bos_float_exception::DType::Float32);

  const std::vector<std::uint64_t> double_words{
      0x0000000000000000ULL, 0x8000000000000000ULL, 0x0000000000000001ULL,
      0x000fffffffffffffULL, 0x0010000000000000ULL, 0x7fefffffffffffffULL,
      0xffefffffffffffffULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL,
      0x7ff8000000000001ULL, 0x7ff0123456789abcULL, 0xfff8abcdef012345ULL,
      std::bit_cast<std::uint64_t>(1.25), std::bit_cast<std::uint64_t>(-91.0625)};
  expect_round_trip(ieee_bytes<double>(double_words), bos_float_exception::DType::Float64);
}

void random_bits() {
  std::mt19937_64 random(0x42464531ULL);
  std::vector<std::uint32_t> float_words(10000);
  for (auto& word : float_words) word = static_cast<std::uint32_t>(random());
  expect_round_trip(ieee_bytes<float>(float_words), bos_float_exception::DType::Float32);
  std::vector<std::uint64_t> double_words(10000);
  for (auto& word : double_words) word = random();
  expect_round_trip(ieee_bytes<double>(double_words), bos_float_exception::DType::Float64);
}

void exact_frozen_wire() {
  std::mt19937_64 random(0x4246453157495245ULL);
  for (const std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{17},
                                  std::size_t{256}, std::size_t{257},
                                  std::size_t{10000}}) {
    std::vector<std::uint32_t> float_words(count);
    std::vector<std::uint64_t> double_words(count);
    for (std::size_t index = 0; index < count; ++index) {
      float_words[index] = static_cast<std::uint32_t>(random());
      double_words[index] = random();
    }
    if (count >= 8) {
      const std::array<std::uint32_t, 8> float_special{
          0U, 0x80000000U, 0x7f800000U, 0xff800000U,
          0x7fc12345U, 0x00000001U, 0x007fffffU,
          std::bit_cast<std::uint32_t>(1.25F)};
      const std::array<std::uint64_t, 8> double_special{
          0ULL, 0x8000000000000000ULL, 0x7ff0000000000000ULL,
          0xfff0000000000000ULL, 0x7ff8123456789abcULL,
          0x0000000000000001ULL, 0x000fffffffffffffULL,
          std::bit_cast<std::uint64_t>(1.25)};
      std::copy(float_special.begin(), float_special.end(), float_words.begin());
      std::copy(double_special.begin(), double_special.end(), double_words.begin());
    }
    const auto float_input = ieee_bytes<float>(float_words);
    const auto double_input = ieee_bytes<double>(double_words);
    expect_same_wire_as_before(float_input, bos_float_exception::DType::Float32);
    expect_same_wire_as_before(double_input, bos_float_exception::DType::Float64);

    bos_float_exception::EncodeOptions exhaustive;
    exhaustive.pilot_values = 0;
    expect_same_wire_as_before(float_input, bos_float_exception::DType::Float32,
                               exhaustive);
    expect_same_wire_as_before(double_input, bos_float_exception::DType::Float64,
                               exhaustive);

    for (const int precision : {0, 4, 18}) {
      bos_float_exception::EncodeOptions fixed;
      fixed.fixed_precision = precision;
      expect_same_wire_as_before(float_input, bos_float_exception::DType::Float32,
                                 fixed);
      expect_same_wire_as_before(double_input, bos_float_exception::DType::Float64,
                                 fixed);
    }
  }

  std::vector<std::uint64_t> repeated(10000);
  for (std::size_t index = 0; index < repeated.size(); ++index) {
    if (index % 997 == 0)
      repeated[index] = 0x7ff8123456789abcULL;
    else if (index % 251 == 0)
      repeated[index] = 0x8000000000000000ULL;
    else
      repeated[index] = std::bit_cast<std::uint64_t>(12.375);
  }
  expect_same_wire_as_before(ieee_bytes<double>(repeated),
                             bos_float_exception::DType::Float64);

  // Concentrate exponents in the range that produces nonzero int64
  // quantizations.  Uniform random IEEE words mostly fall outside that range
  // and would under-test decimal prediction equivalence.
  std::vector<std::uint32_t> finite_float_words(4096);
  std::vector<std::uint64_t> finite_double_words(4096);
  for (std::size_t index = 0; index < finite_float_words.size(); ++index) {
    const int float_exponent = static_cast<int>(random() % 121) - 60;
    finite_float_words[index] =
        (static_cast<std::uint32_t>(random() & 1) << 31) |
        (static_cast<std::uint32_t>(float_exponent + 127) << 23) |
        (static_cast<std::uint32_t>(random()) & 0x007fffffU);
    const int double_exponent = static_cast<int>(random() % 121) - 60;
    finite_double_words[index] =
        ((random() & 1ULL) << 63) |
        (static_cast<std::uint64_t>(double_exponent + 1023) << 52) |
        (random() & 0x000fffffffffffffULL);
  }
  bos_float_exception::EncodeOptions exhaustive;
  exhaustive.pilot_values = 0;
  expect_same_wire_as_before(ieee_bytes<float>(finite_float_words),
                             bos_float_exception::DType::Float32, exhaustive);
  expect_same_wire_as_before(ieee_bytes<double>(finite_double_words),
                             bos_float_exception::DType::Float64, exhaustive);
}

void exception_mode_and_candidates() {
  std::vector<std::uint64_t> words(1000, std::bit_cast<std::uint64_t>(1.25));
  const auto input = ieee_bytes<double>(words);
  std::size_t calls = 0;
  auto counting_encoder = [&](std::span<const std::int64_t> values) {
    ++calls;
    return encode_int(values);
  };
  bos_float_exception::Stats stats;
  const auto wire = bos_float_exception::encode(
      input, bos_float_exception::DType::Float64, counting_encoder, &stats);
  require(calls >= 6, "BFE1 did not actually encode every precision candidate");
  require(stats.mode == bos_float_exception::Mode::Exception,
          "compressible decimals did not select exception mode");
  require(stats.exceptions == 0, "exact decimal values unexpectedly need repairs");
  require(bos_float_exception::decode(wire, decode_int) == input,
          "exception-mode BFE1 round trip failed");

  bos_float_exception::EncodeOptions options;
  options.fixed_precision = 2;
  calls = 0;
  const auto fixed = bos_float_exception::encode(
      input, bos_float_exception::DType::Float64, counting_encoder, &stats, options);
  require(calls == 1 && stats.selected_p == 2,
          "fixed BFE1 precision did not constrain candidate search");
  require(bos_float_exception::decode(fixed, decode_int) == input,
          "fixed-precision BFE1 round trip failed");

  // Exercise the sparse repair path itself (not only an exact decimal body).
  words[613] = std::bit_cast<std::uint64_t>(1.2500000000000002);
  const auto repaired_input = ieee_bytes<double>(words);
  const auto repaired = bos_float_exception::encode(
      repaired_input, bos_float_exception::DType::Float64, encode_int, &stats, options);
  require(stats.mode == bos_float_exception::Mode::Exception && stats.exceptions == 1,
          "sparse BFE1 repair was not selected");
  require(bos_float_exception::decode(repaired, decode_int) == repaired_input,
          "sparse BFE1 repair changed IEEE bits");

  // Non-finite values use q=0 and raw XOR repairs, including NaN payload bits.
  std::vector<std::uint64_t> nan_words(1000, 0x7ff8123456789abcULL);
  const auto nan_input = ieee_bytes<double>(nan_words);
  const auto nan_wire = bos_float_exception::encode(
      nan_input, bos_float_exception::DType::Float64, encode_int, &stats);
  require(stats.mode == bos_float_exception::Mode::Exception && stats.exceptions == 1000,
          "NaN BFE1 repairs were not selected");
  require(bos_float_exception::decode(nan_wire, decode_int) == nan_input,
          "BFE1 changed NaN payload bits");
}

void pilot_selection() {
  std::vector<std::uint64_t> words;
  words.reserve(1000);
  for (std::uint64_t value = 0; value < 1000; ++value)
    words.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(value)));
  const auto input = ieee_bytes<double>(words);
  std::vector<std::size_t> call_sizes;
  std::vector<std::pair<std::int64_t, std::int64_t>> endpoints;
  auto recording_encoder = [&](std::span<const std::int64_t> values) {
    call_sizes.push_back(values.size());
    endpoints.emplace_back(values.empty() ? 0 : values.front(),
                           values.empty() ? 0 : values.back());
    return encode_int(values);
  };
  const auto wire = bos_float_exception::encode(
      input, bos_float_exception::DType::Float64, recording_encoder);
  require(call_sizes.size() == 7,
          "BFE1 pilot did not encode six samples and one full candidate");
  const std::array<std::int64_t, 6> expected_last{
      999, 9990, 99900, 999000, 9990000, 999000000};
  for (std::size_t i = 0; i < 6; ++i) {
    require(call_sizes[i] == 256, "BFE1 pilot has the wrong value count");
    require(endpoints[i].first == 0 && endpoints[i].second == expected_last[i],
            "BFE1 pilot does not cover both endpoints");
  }
  require(call_sizes.back() == 1000, "BFE1 did not encode one full candidate");
  require(bos_float_exception::decode(wire, decode_int) == input,
          "pilot-selected BFE1 round trip failed");

  bos_float_exception::EncodeOptions exhaustive;
  exhaustive.pilot_values = 0;
  call_sizes.clear();
  const auto exhaustive_wire = bos_float_exception::encode(
      input, bos_float_exception::DType::Float64, recording_encoder, nullptr, exhaustive);
  require(call_sizes.size() == 6 &&
              std::all_of(call_sizes.begin(), call_sizes.end(),
                          [](auto size) { return size == 1000; }),
          "BFE1 exhaustive mode did not encode every full candidate exactly once");
  require(bos_float_exception::decode(exhaustive_wire, decode_int) == input,
          "exhaustive BFE1 round trip failed");

  std::vector<std::uint64_t> zero_words(64, std::bit_cast<std::uint64_t>(0.0));
  const auto small_input = ieee_bytes<double>(zero_words);
  call_sizes.clear();
  bos_float_exception::Stats stats;
  const auto small_wire = bos_float_exception::encode(
      small_input, bos_float_exception::DType::Float64, recording_encoder, &stats);
  require(call_sizes.size() == 6,
          "small BFE1 block re-encoded its selected full candidate");
  require(stats.mode == bos_float_exception::Mode::Exception && stats.selected_p == 0,
          "small BFE1 block selected an unexpected candidate");
  require(bos_float_exception::decode(small_wire, decode_int) == small_input,
          "small BFE1 block round trip failed");

  bos_float_exception::EncodeOptions fixed;
  fixed.pilot_values = 1;  // Must be ignored by fixed_precision.
  fixed.fixed_precision = 3;
  call_sizes.clear();
  const auto fixed_wire = bos_float_exception::encode(
      input, bos_float_exception::DType::Float64, recording_encoder, nullptr, fixed);
  require(call_sizes.size() == 1 && call_sizes.front() == 1000,
          "fixed BFE1 precision unexpectedly ran a pilot");
  require(bos_float_exception::decode(fixed_wire, decode_int) == input,
          "fixed BFE1 pilot bypass round trip failed");
}

template <typename Function>
void expect_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void malformed_frames() {
  std::vector<std::uint64_t> words(128, std::bit_cast<std::uint64_t>(12.34));
  const auto input = ieee_bytes<double>(words);
  const auto good = bos_float_exception::encode(
      input, bos_float_exception::DType::Float64, encode_int);
  for (std::size_t length = 0; length < good.size(); ++length) {
    expect_invalid([&] { bos_float_exception::decode(
                            std::span<const std::uint8_t>(good.data(), length), decode_int); },
                   "truncated BFE1 frame was accepted");
  }
  auto bad = good;
  bad[0] ^= 1;
  expect_invalid([&] { bos_float_exception::decode(bad, decode_int); },
                 "bad BFE1 magic was accepted");
  bad = good;
  bad[4] = 2;
  expect_invalid([&] { bos_float_exception::decode(bad, decode_int); },
                 "bad BFE1 version was accepted");
  bad = good;
  bad[6] = 9;
  expect_invalid([&] { bos_float_exception::decode(bad, decode_int); },
                 "bad BFE1 mode was accepted");
  bad = good;
  bad[12] = 0xff;
  bad[13] = 0xff;
  bad[14] = 0xff;
  bad[15] = 0x7f;
  expect_invalid([&] { bos_float_exception::decode(bad, decode_int); },
                 "bad BFE1 exception count was accepted");
  bad = good;
  bad[16] ^= 1;
  expect_invalid([&] { bos_float_exception::decode(bad, decode_int); },
                 "bad BFE1 stream length was accepted");
  bad = good;
  bad.push_back(0);
  expect_invalid([&] { bos_float_exception::decode(bad, decode_int); },
                 "trailing BFE1 byte was accepted");
}

void invalid_inputs() {
  expect_invalid(
      [&] {
        bos_float_exception::encode(Bytes{0, 1, 2},
                                    bos_float_exception::DType::Float32, encode_int);
      },
      "partial IEEE input was accepted");
  Bytes too_many(10001 * sizeof(float));
  expect_invalid(
      [&] {
        bos_float_exception::encode(too_many, bos_float_exception::DType::Float32,
                                    encode_int);
      },
      "oversized BFE1 block was accepted");
  bos_float_exception::EncodeOptions options;
  options.candidate_precisions = {2, 2};
  expect_invalid(
      [&] {
        bos_float_exception::encode(Bytes{}, bos_float_exception::DType::Float32,
                                    encode_int, nullptr, options);
      },
      "duplicate BFE1 precision was accepted");
  options.candidate_precisions = {0, 1};
  options.pilot_values = 1;
  std::vector<std::uint64_t> two_words{
      std::bit_cast<std::uint64_t>(1.0), std::bit_cast<std::uint64_t>(2.0)};
  const auto two_values = ieee_bytes<double>(two_words);
  expect_invalid(
      [&] {
        bos_float_exception::encode(two_values, bos_float_exception::DType::Float64,
                                    encode_int, nullptr, options);
      },
      "one-value BFE1 pilot was accepted for a multi-value block");
}

void performance_smoke() {
  using Clock = std::chrono::steady_clock;
  std::vector<std::uint64_t> double_words(10000);
  for (std::size_t index = 0; index < double_words.size(); ++index) {
    const double value =
        static_cast<double>(static_cast<int>(index % 2001) - 1000) * 0.001 +
        static_cast<double>(index / 2001) * 0.0001;
    double_words[index] = std::bit_cast<std::uint64_t>(value);
    if (index % 997 == 0) double_words[index] = 0x7ff8123456789abcULL;
    if (index % 1499 == 0) double_words[index] = 0x8000000000000000ULL;
  }
  const auto input = ieee_bytes<double>(double_words);
  expect_same_wire_as_before(input, bos_float_exception::DType::Float64);

  constexpr std::size_t iterations = 40;
  std::chrono::nanoseconds old_elapsed{};
  std::chrono::nanoseconds new_elapsed{};
  std::size_t sink = 0;
  const auto run_old = [&] {
    const auto begin = Clock::now();
    const auto wire = bos_float_exception_before::encode(
        input, bos_float_exception_before::DType::Float64, encode_int);
    old_elapsed += Clock::now() - begin;
    sink += wire.size();
  };
  const auto run_new = [&] {
    const auto begin = Clock::now();
    const auto wire = bos_float_exception::encode(
        input, bos_float_exception::DType::Float64, encode_int);
    new_elapsed += Clock::now() - begin;
    sink += wire.size();
  };
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    if ((iteration & 1U) == 0) {
      run_old();
      run_new();
    } else {
      run_new();
      run_old();
    }
  }
  const auto old_ns_per_value =
      static_cast<double>(old_elapsed.count()) / (iterations * double_words.size());
  const auto new_ns_per_value =
      static_cast<double>(new_elapsed.count()) / (iterations * double_words.size());
  std::cout << "old_ns_per_value=" << old_ns_per_value
            << " new_ns_per_value=" << new_ns_per_value
            << " speedup=" << old_ns_per_value / new_ns_per_value
            << " sink=" << sink << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
    performance_smoke();
    return 0;
  }
  edge_cases();
  random_bits();
  exact_frozen_wire();
  exception_mode_and_candidates();
  pilot_selection();
  malformed_frames();
  invalid_inputs();
  std::cout << "bos_float_exception=PASS bit_exact=PASS malformed=PASS\n";
}
