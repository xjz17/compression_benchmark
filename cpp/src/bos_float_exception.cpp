#include "bos_float_exception.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace bos_float_exception {
namespace {

constexpr std::size_t kHeaderSize = 32;
constexpr std::uint8_t kVersion = 1;
constexpr std::uint8_t kRawPrecision = 0xff;
constexpr std::uint32_t kMaximumValues = 10000;
constexpr int kMaximumPrecision = 18;
constexpr std::array<std::uint64_t, 20> kPowersOfTen{
    1ULL,
    10ULL,
    100ULL,
    1000ULL,
    10000ULL,
    100000ULL,
    1000000ULL,
    10000000ULL,
    100000000ULL,
    1000000000ULL,
    10000000000ULL,
    100000000000ULL,
    1000000000000ULL,
    10000000000000ULL,
    100000000000000ULL,
    1000000000000000ULL,
    10000000000000000ULL,
    100000000000000000ULL,
    1000000000000000000ULL,
    10000000000000000000ULL,
};

[[noreturn]] void fail(const char* message) { throw std::invalid_argument(message); }

void require(bool condition, const char* message) {
  if (!condition) fail(message);
}

void put_u32(Bytes& output, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& offset) {
  require(offset <= input.size() && input.size() - offset >= 4,
          "truncated BFE1 uint32");
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift != 32; shift += 8)
    value |= static_cast<std::uint32_t>(input[offset++]) << shift;
  return value;
}

template <typename Word>
Word get_le_word(std::span<const std::uint8_t> input, std::size_t offset) {
  static_assert(sizeof(Word) == 4 || sizeof(Word) == 8);
  require(offset <= input.size() && input.size() - offset >= sizeof(Word),
          "truncated IEEE word");
  if constexpr (std::endian::native == std::endian::little) {
    Word value;
    std::memcpy(&value, input.data() + offset, sizeof(value));
    return value;
  } else {
    Word value = 0;
    for (std::size_t byte = 0; byte < sizeof(Word); ++byte)
      value |= static_cast<Word>(input[offset + byte]) << (byte * 8);
    return value;
  }
}

void put_le_word(Bytes& output, std::uint64_t value, std::size_t width) {
  for (std::size_t byte = 0; byte < width; ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

std::uint32_t checked_length(std::size_t size) {
  require(size <= std::numeric_limits<std::uint32_t>::max(),
          "BFE1 stream exceeds uint32 length");
  return static_cast<std::uint32_t>(size);
}

std::size_t dtype_width(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return 4;
    case DType::Float64:
      return 8;
  }
  fail("invalid BFE1 dtype");
}

DType parse_dtype(std::uint8_t value) {
  if (value == static_cast<std::uint8_t>(DType::Float32)) return DType::Float32;
  if (value == static_cast<std::uint8_t>(DType::Float64)) return DType::Float64;
  fail("invalid BFE1 dtype");
}

void validate_precision(int precision) {
  require(precision >= 0 && precision <= kMaximumPrecision,
          "BFE1 precision must be in [0,18]");
}

template <typename Float>
Float decimal_prediction(std::int64_t value, int precision) {
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  validate_precision(precision);
  if (value == 0) return Float{0};

  // Convert the exact rational |value| / 10^precision directly to IEEE-754,
  // using round-to-nearest, ties-to-even.  With int64 numerators and p<=18,
  // the widest intermediate is at most 113 bits, so unsigned __int128 is
  // sufficient.  This produces the same bits as parsing "<value>e-<p>"
  // without formatting and reparsing a decimal string for every value.
  using Wide = unsigned __int128;
  using Word = std::conditional_t<std::is_same_v<Float, float>,
                                  std::uint32_t, std::uint64_t>;
  constexpr int kFractionBits = std::numeric_limits<Float>::digits - 1;
  constexpr int kExponentBits = std::is_same_v<Float, float> ? 8 : 11;
  constexpr int kExponentBias = std::is_same_v<Float, float> ? 127 : 1023;
  constexpr int kWordBits = sizeof(Word) * 8;

  const bool negative = value < 0;
  const std::uint64_t magnitude =
      negative ? static_cast<std::uint64_t>(-(value + 1)) + 1
               : static_cast<std::uint64_t>(value);
  const std::uint64_t denominator = kPowersOfTen[precision];
  int exponent = static_cast<int>(std::bit_width(magnitude)) -
                 static_cast<int>(std::bit_width(denominator));
  const bool at_least_power =
      exponent >= 0
          ? Wide{magnitude} >= (Wide{denominator} << exponent)
          : (Wide{magnitude} << -exponent) >= Wide{denominator};
  if (!at_least_power) --exponent;

  const int shift = kFractionBits - exponent;
  const Wide numerator =
      shift >= 0 ? (Wide{magnitude} << shift) : Wide{magnitude};
  const Wide divisor =
      shift >= 0 ? Wide{denominator} : (Wide{denominator} << -shift);
  Wide significand = numerator / divisor;
  const Wide remainder = numerator % divisor;
  const Wide twice_remainder = remainder << 1;
  if (twice_remainder > divisor ||
      (twice_remainder == divisor && (significand & 1U) != 0))
    ++significand;

  const Wide overflow = Wide{1} << (kFractionBits + 1);
  if (significand == overflow) {
    significand >>= 1;
    ++exponent;
  }
  const int biased_exponent = exponent + kExponentBias;
  require(biased_exponent > 0 &&
              biased_exponent < (1 << kExponentBits) - 1,
          "BFE1 decimal prediction outside normal IEEE range");
  const Word fraction_mask = (Word{1} << kFractionBits) - 1;
  Word word = static_cast<Word>(significand) & fraction_mask;
  word |= static_cast<Word>(biased_exponent) << kFractionBits;
  if (negative) word |= Word{1} << (kWordBits - 1);
  return std::bit_cast<Float>(word);
}

template <typename Word>
struct ParsedDecimal {
  Word original_word = 0;
  std::uint64_t coefficient = 0;
  int decimal_exponent = 0;
  bool negative = false;
  bool quantizable = false;
};

template <typename Word>
constexpr bool non_finite_word(Word word) {
  if constexpr (sizeof(Word) == sizeof(std::uint32_t)) {
    return (word & Word{0x7f800000U}) == Word{0x7f800000U};
  } else {
    return (word & Word{0x7ff0000000000000ULL}) == Word{0x7ff0000000000000ULL};
  }
}

template <typename Word>
constexpr bool negative_zero_word(Word word) {
  if constexpr (sizeof(Word) == sizeof(std::uint32_t)) {
    return word == Word{0x80000000U};
  } else {
    return word == Word{0x8000000000000000ULL};
  }
}

template <typename Float, typename Word>
ParsedDecimal<Word> parse_decimal(Word original_word) {
  static_assert(sizeof(Float) == sizeof(Word));
  ParsedDecimal<Word> parsed;
  parsed.original_word = original_word;
  if (non_finite_word(original_word) || negative_zero_word(original_word)) return parsed;

  const auto value = std::bit_cast<Float>(original_word);
  // Work from the implementation's shortest round-trippable decimal spelling.
  // This makes decimal scaling deterministic and avoids an out-of-range
  // floating-to-integer conversion.  The decimal coefficient is at most
  // max_digits10 digits, so uint64 is ample for parsing and rounding; positive
  // powers are checked against the final signed-int64 limit before multiplying.
  std::array<char, 64> text{};
  const auto formatted =
      std::to_chars(text.data(), text.data() + text.size(), value, std::chars_format::general);
  if (formatted.ec != std::errc{})
    throw std::runtime_error("BFE1 floating formatting failed");
  const char* cursor = text.data();
  const char* end = formatted.ptr;
  bool negative = false;
  if (cursor != end && (*cursor == '-' || *cursor == '+')) {
    negative = *cursor == '-';
    ++cursor;
  }
  require(cursor != end, "empty BFE1 decimal spelling");

  std::uint64_t coefficient = 0;
  int fractional_digits = 0;
  bool after_point = false;
  bool saw_digit = false;
  while (cursor != end && *cursor != 'e' && *cursor != 'E') {
    if (*cursor == '.') {
      require(!after_point, "duplicate BFE1 decimal point");
      after_point = true;
    } else {
      require(*cursor >= '0' && *cursor <= '9', "invalid BFE1 decimal digit");
      require(coefficient <= std::numeric_limits<std::uint64_t>::max() / 10,
              "BFE1 decimal coefficient overflow");
      coefficient = coefficient * 10 + static_cast<unsigned>(*cursor - '0');
      if (after_point) ++fractional_digits;
      saw_digit = true;
    }
    ++cursor;
  }
  require(saw_digit, "BFE1 decimal spelling has no digits");

  int explicit_exponent = 0;
  if (cursor != end) {
    ++cursor;
    require(cursor != end, "empty BFE1 decimal exponent");
    bool exponent_negative = false;
    if (*cursor == '-' || *cursor == '+') {
      exponent_negative = *cursor == '-';
      ++cursor;
    }
    require(cursor != end, "empty BFE1 decimal exponent");
    while (cursor != end) {
      require(*cursor >= '0' && *cursor <= '9', "invalid BFE1 decimal exponent");
      require(explicit_exponent <= 10000, "BFE1 decimal exponent too large");
      explicit_exponent = explicit_exponent * 10 + (*cursor++ - '0');
    }
    if (exponent_negative) explicit_exponent = -explicit_exponent;
  }

  parsed.coefficient = coefficient;
  parsed.decimal_exponent = explicit_exponent - fractional_digits;
  parsed.negative = negative;
  parsed.quantizable = true;
  return parsed;
}

template <typename Word>
std::int64_t quantize(const ParsedDecimal<Word>& value, int precision) {
  if (!value.quantizable) return 0;
  const int shift = value.decimal_exponent + precision;
  std::uint64_t magnitude = value.coefficient;
  if (shift >= 0) {
    // More than 19 decimal digits cannot fit in signed int64.  Check the
    // signed limit during multiplication so the arithmetic itself stays safe.
    if (shift > 19) return 0;
    const std::uint64_t limit = value.negative ? (std::uint64_t{1} << 63)
                                               : std::numeric_limits<std::int64_t>::max();
    const auto multiplier = kPowersOfTen[static_cast<std::size_t>(shift)];
    if (magnitude > limit / multiplier) return 0;
    magnitude *= multiplier;
  } else {
    const unsigned divisor_exponent = static_cast<unsigned>(-shift);
    if (divisor_exponent > 19) {
      magnitude = 0;
    } else {
      const std::uint64_t divisor = kPowersOfTen[divisor_exponent];
      const std::uint64_t quotient = magnitude / divisor;
      const std::uint64_t remainder = magnitude % divisor;
      magnitude = quotient;
      // The shortest coefficient has at most max_digits10 (17) digits while
      // this branch divides by at least ten, so doubling the remainder is safe.
      const std::uint64_t twice_remainder = remainder * 2;
      if (twice_remainder > divisor ||
          (twice_remainder == divisor && (quotient & 1) != 0))
        ++magnitude;
    }
  }

  const std::uint64_t limit = value.negative ? (std::uint64_t{1} << 63)
                                             : std::numeric_limits<std::int64_t>::max();
  if (magnitude > limit) return 0;
  if (value.negative) {
    if (magnitude == (std::uint64_t{1} << 63))
      return std::numeric_limits<std::int64_t>::min();
    return -static_cast<std::int64_t>(magnitude);
  }
  return static_cast<std::int64_t>(magnitude);
}

template <typename Float, typename Word>
std::vector<ParsedDecimal<Word>> parse_values(std::span<const std::uint8_t> input) {
  static_assert(sizeof(Float) == sizeof(Word));
  const std::size_t count = input.size() / sizeof(Float);
  std::vector<ParsedDecimal<Word>> parsed;
  parsed.reserve(count);
  bool have_previous = false;
  Word previous_word = 0;
  ParsedDecimal<Word> previous;
  for (std::size_t index = 0; index < count; ++index) {
    const auto word = get_le_word<Word>(input, index * sizeof(Float));
    if (have_previous && word == previous_word) {
      parsed.push_back(previous);
    } else {
      previous = parse_decimal<Float>(word);
      previous_word = word;
      have_previous = true;
      parsed.push_back(previous);
    }
  }
  return parsed;
}

template <typename Float, typename Word>
struct CandidateStreams {
  std::vector<std::int64_t> quantized;
  std::vector<std::int64_t> gaps;
  std::vector<std::int64_t> masks;
};

template <typename Float, typename Word>
CandidateStreams<Float, Word> make_streams(
    std::span<const ParsedDecimal<Word>> values,
    int precision) {
  static_assert(sizeof(Float) == sizeof(Word));
  CandidateStreams<Float, Word> result;
  result.quantized.reserve(values.size());
  result.gaps.reserve(values.size());
  result.masks.reserve(values.size());
  std::int64_t previous = -1;
  bool have_previous_prediction = false;
  std::int64_t previous_quantized = 0;
  Word previous_prediction_word = 0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    const auto quantized = quantize(values[index], precision);
    Word prediction_word;
    if (have_previous_prediction && quantized == previous_quantized) {
      prediction_word = previous_prediction_word;
    } else {
      prediction_word = quantized == 0
                            ? Word{0}
                            : std::bit_cast<Word>(
                                  decimal_prediction<Float>(quantized, precision));
      previous_quantized = quantized;
      previous_prediction_word = prediction_word;
      have_previous_prediction = true;
    }
    const Word mask = values[index].original_word ^ prediction_word;
    result.quantized.push_back(quantized);
    if (mask != 0) {
      const auto current = static_cast<std::int64_t>(index);
      result.gaps.push_back(current - previous - 1);
      if constexpr (sizeof(Word) == sizeof(std::uint64_t)) {
        result.masks.push_back(std::bit_cast<std::int64_t>(mask));
      } else {
        result.masks.push_back(static_cast<std::int64_t>(mask));
      }
      previous = current;
    }
  }
  return result;
}

Bytes make_header(DType dtype,
                  Mode mode,
                  int precision,
                  std::uint32_t count,
                  std::uint32_t exceptions,
                  std::uint32_t quantized_length,
                  std::uint32_t gap_length,
                  std::uint32_t mask_length,
                  std::uint32_t raw_length) {
  Bytes wire{'B', 'F', 'E', '1', kVersion, static_cast<std::uint8_t>(dtype),
             static_cast<std::uint8_t>(mode),
             mode == Mode::Raw ? kRawPrecision : static_cast<std::uint8_t>(precision)};
  put_u32(wire, count);
  put_u32(wire, exceptions);
  put_u32(wire, quantized_length);
  put_u32(wire, gap_length);
  put_u32(wire, mask_length);
  put_u32(wire, raw_length);
  require(wire.size() == kHeaderSize, "internal BFE1 header size mismatch");
  return wire;
}

Bytes raw_frame(std::span<const std::uint8_t> input, DType dtype, std::uint32_t count) {
  auto wire = make_header(dtype, Mode::Raw, -1, count, 0, 0, 0, 0,
                          checked_length(input.size()));
  wire.insert(wire.end(), input.begin(), input.end());
  return wire;
}

template <typename Word>
std::vector<ParsedDecimal<Word>> uniform_pilot(
    std::span<const ParsedDecimal<Word>> values,
    std::uint32_t pilot_values) {
  const auto count = static_cast<std::uint32_t>(values.size());
  require(pilot_values >= 2, "BFE1 pilot must contain at least two values");
  require(pilot_values < count, "BFE1 pilot must be smaller than the block");
  std::vector<ParsedDecimal<Word>> pilot;
  pilot.reserve(pilot_values);
  for (std::uint32_t sample = 0; sample < pilot_values; ++sample) {
    // count <= 10000, so the product is comfortably inside uint64.  Integer
    // division makes the sample deterministic while endpoints map to 0,N-1.
    const auto index = static_cast<std::uint32_t>(
        (std::uint64_t{sample} * (count - 1)) / (pilot_values - 1));
    pilot.push_back(values[index]);
  }
  return pilot;
}

template <typename Float, typename Word>
std::pair<Bytes, std::uint32_t> exception_frame(
    std::span<const ParsedDecimal<Word>> values,
    DType dtype,
    int precision,
    const EncodeInt& encode_int) {
  const auto streams = make_streams<Float, Word>(values, precision);
  const auto quantized_wire = encode_int(streams.quantized);
  const auto gap_wire = streams.gaps.empty() ? Bytes{} : encode_int(streams.gaps);
  const auto mask_wire = streams.masks.empty() ? Bytes{} : encode_int(streams.masks);
  const auto exceptions = static_cast<std::uint32_t>(streams.gaps.size());
  auto wire = make_header(dtype, Mode::Exception, precision,
                          static_cast<std::uint32_t>(streams.quantized.size()), exceptions,
                          checked_length(quantized_wire.size()),
                          checked_length(gap_wire.size()), checked_length(mask_wire.size()), 0);
  const std::uint64_t total = std::uint64_t{kHeaderSize} + quantized_wire.size() +
                              gap_wire.size() + mask_wire.size();
  require(total <= Bytes{}.max_size(), "BFE1 frame byte count overflow");
  wire.reserve(static_cast<std::size_t>(total));
  wire.insert(wire.end(), quantized_wire.begin(), quantized_wire.end());
  wire.insert(wire.end(), gap_wire.begin(), gap_wire.end());
  wire.insert(wire.end(), mask_wire.begin(), mask_wire.end());
  return {std::move(wire), exceptions};
}

template <typename Float, typename Word>
std::pair<Bytes, Stats> encode_parsed(
    std::span<const std::uint8_t> input,
    DType dtype,
    std::span<const ParsedDecimal<Word>> values,
    const EncodeInt& encode_int,
    std::span<const int> precisions,
    const EncodeOptions& options) {
  const auto count = static_cast<std::uint32_t>(values.size());
  Bytes best = raw_frame(input, dtype, count);
  Stats best_stats{-1, 0, Mode::Raw};
  const auto encode_exception = [&](std::span<const ParsedDecimal<Word>> candidate_values,
                                    int precision) {
    return exception_frame<Float, Word>(candidate_values, dtype, precision, encode_int);
  };

  if (options.fixed_precision.has_value()) {
    auto candidate = encode_exception(values, precisions.front());
    if (candidate.first.size() < best.size()) {
      best = std::move(candidate.first);
      best_stats = Stats{precisions.front(), candidate.second, Mode::Exception};
    }
  } else if (options.pilot_values != 0 && count > options.pilot_values) {
    require(options.pilot_values >= 2,
            "BFE1 pilot_values must be zero or at least two");
    const auto pilot = uniform_pilot(values, options.pilot_values);
    int selected_precision = precisions.front();
    std::size_t selected_size = std::numeric_limits<std::size_t>::max();
    // Raw fallback is deliberately not considered during p selection.  Even
    // an all-raw-looking pilot therefore chooses the least costly exception
    // candidate before the complete input is compared against its raw frame.
    for (const auto precision : precisions) {
      auto candidate = encode_exception(pilot, precision);
      if (candidate.first.size() < selected_size) {
        selected_size = candidate.first.size();
        selected_precision = precision;
      }
    }
    auto candidate = encode_exception(values, selected_precision);
    if (candidate.first.size() < best.size()) {
      best = std::move(candidate.first);
      best_stats = Stats{selected_precision, candidate.second, Mode::Exception};
    }
  } else {
    // Exhaustive mode, and blocks already no larger than the pilot, both
    // evaluate full candidates.  Retain the winning bytes so it is never
    // encoded a second time.
    std::pair<Bytes, std::uint32_t> selected;
    int selected_precision = precisions.front();
    bool have_selected = false;
    for (const auto precision : precisions) {
      auto candidate = encode_exception(values, precision);
      if (!have_selected || candidate.first.size() < selected.first.size()) {
        selected = std::move(candidate);
        selected_precision = precision;
        have_selected = true;
      }
    }
    if (selected.first.size() < best.size()) {
      best = std::move(selected.first);
      best_stats = Stats{selected_precision, selected.second, Mode::Exception};
    }
  }
  return {std::move(best), best_stats};
}

struct Header {
  DType dtype;
  Mode mode;
  int precision;
  std::uint32_t count;
  std::uint32_t exceptions;
  std::array<std::uint32_t, 4> lengths;
};

Header parse_header(std::span<const std::uint8_t> wire) {
  require(wire.size() >= kHeaderSize, "truncated BFE1 header");
  require(std::equal(wire.begin(), wire.begin() + 4,
                     std::array<std::uint8_t, 4>{'B', 'F', 'E', '1'}.begin()),
          "invalid BFE1 magic");
  require(wire[4] == kVersion, "unsupported BFE1 version");
  Header header{parse_dtype(wire[5]), Mode::Raw, -1, 0, 0, {}};
  if (wire[6] == static_cast<std::uint8_t>(Mode::Raw)) {
    header.mode = Mode::Raw;
    require(wire[7] == kRawPrecision, "raw BFE1 frame has a precision");
  } else if (wire[6] == static_cast<std::uint8_t>(Mode::Exception)) {
    header.mode = Mode::Exception;
    require(wire[7] <= kMaximumPrecision, "invalid BFE1 precision");
    header.precision = wire[7];
  } else {
    fail("invalid BFE1 mode");
  }
  std::size_t offset = 8;
  header.count = get_u32(wire, offset);
  header.exceptions = get_u32(wire, offset);
  for (auto& length : header.lengths) length = get_u32(wire, offset);
  require(offset == kHeaderSize, "internal BFE1 header parse mismatch");
  require(header.count <= kMaximumValues, "BFE1 block exceeds 10000 values");
  require(header.exceptions <= header.count, "BFE1 exception count exceeds values");
  std::uint64_t payload = 0;
  for (const auto length : header.lengths) payload += length;
  require(payload == wire.size() - kHeaderSize,
          "BFE1 stream lengths do not match frame size");
  return header;
}

std::span<const std::uint8_t> take_stream(std::span<const std::uint8_t> wire,
                                          std::size_t& offset,
                                          std::uint32_t length) {
  require(offset <= wire.size() && wire.size() - offset >= length,
          "truncated BFE1 stream");
  auto stream = wire.subspan(offset, length);
  offset += length;
  return stream;
}

template <typename Float, typename Word>
Bytes restore_exception(const Header& header,
                        std::span<const std::uint8_t> quantized_wire,
                        std::span<const std::uint8_t> gap_wire,
                        std::span<const std::uint8_t> mask_wire,
                        const DecodeInt& decode_int) {
  static_assert(sizeof(Float) == sizeof(Word));
  const auto quantized = decode_int(quantized_wire);
  const auto gaps = header.exceptions == 0 ? std::vector<std::int64_t>{}
                                            : decode_int(gap_wire);
  const auto masks = header.exceptions == 0 ? std::vector<std::int64_t>{}
                                             : decode_int(mask_wire);
  require(quantized.size() == header.count, "BFE1 quantized count mismatch");
  require(gaps.size() == header.exceptions, "BFE1 gap count mismatch");
  require(masks.size() == header.exceptions, "BFE1 mask count mismatch");

  std::vector<std::size_t> positions;
  positions.reserve(header.exceptions);
  std::int64_t previous = -1;
  for (const auto gap : gaps) {
    require(gap >= 0, "negative BFE1 exception gap");
    require(previous <= std::numeric_limits<std::int64_t>::max() - gap - 1,
            "BFE1 exception position overflow");
    const auto current = previous + gap + 1;
    require(current >= 0 && static_cast<std::uint64_t>(current) < header.count,
            "BFE1 exception position outside block");
    positions.push_back(static_cast<std::size_t>(current));
    previous = current;
  }

  Bytes output;
  output.reserve(static_cast<std::size_t>(header.count) * sizeof(Float));
  std::size_t exception = 0;
  bool have_previous_prediction = false;
  std::int64_t previous_quantized = 0;
  Word previous_prediction_word = 0;
  for (std::size_t index = 0; index < header.count; ++index) {
    Word word;
    if (have_previous_prediction && quantized[index] == previous_quantized) {
      word = previous_prediction_word;
    } else {
      word = quantized[index] == 0
                 ? Word{0}
                 : std::bit_cast<Word>(
                       decimal_prediction<Float>(quantized[index], header.precision));
      previous_quantized = quantized[index];
      previous_prediction_word = word;
      have_previous_prediction = true;
    }
    if (exception < positions.size() && positions[exception] == index) {
      Word mask;
      if constexpr (sizeof(Word) == sizeof(std::uint64_t)) {
        mask = std::bit_cast<std::uint64_t>(masks[exception]);
      } else {
        require(masks[exception] >= 0 &&
                    static_cast<std::uint64_t>(masks[exception]) <=
                        std::numeric_limits<std::uint32_t>::max(),
                "float32 BFE1 mask exceeds 32 bits");
        mask = static_cast<std::uint32_t>(masks[exception]);
      }
      require(mask != 0, "zero BFE1 exception mask");
      word ^= mask;
      ++exception;
    }
    put_le_word(output, word, sizeof(Float));
  }
  require(exception == positions.size(), "unused BFE1 exception position");
  return output;
}

}  // namespace

Bytes encode(std::span<const std::uint8_t> input,
             DType dtype,
             const EncodeInt& encode_int,
             Stats* stats,
             const EncodeOptions& options) {
  require(static_cast<bool>(encode_int), "BFE1 integer encoder is empty");
  const auto width = dtype_width(dtype);
  require(input.size() % width == 0, "BFE1 input is not complete IEEE words");
  const auto count_size = input.size() / width;
  require(count_size <= kMaximumValues, "BFE1 block exceeds 10000 values");
  std::vector<int> precisions;
  if (options.fixed_precision.has_value()) {
    precisions.push_back(*options.fixed_precision);
  } else {
    precisions = options.candidate_precisions;
  }
  require(!precisions.empty(), "BFE1 requires at least one precision candidate");
  std::vector<int> seen;
  for (const auto precision : precisions) {
    validate_precision(precision);
    require(std::find(seen.begin(), seen.end(), precision) == seen.end(),
            "duplicate BFE1 precision candidate");
    seen.push_back(precision);
  }

  std::pair<Bytes, Stats> encoded;
  if (dtype == DType::Float32) {
    const auto parsed = parse_values<float, std::uint32_t>(input);
    encoded = encode_parsed<float, std::uint32_t>(input, dtype, parsed, encode_int,
                                                  precisions, options);
  } else {
    const auto parsed = parse_values<double, std::uint64_t>(input);
    encoded = encode_parsed<double, std::uint64_t>(input, dtype, parsed, encode_int,
                                                   precisions, options);
  }
  if (stats != nullptr) *stats = encoded.second;
  return std::move(encoded.first);
}

Bytes decode(std::span<const std::uint8_t> wire,
             const DecodeInt& decode_int,
             Stats* stats) {
  require(static_cast<bool>(decode_int), "BFE1 integer decoder is empty");
  const auto header = parse_header(wire);
  const auto width = dtype_width(header.dtype);
  std::size_t offset = kHeaderSize;
  const auto quantized_wire = take_stream(wire, offset, header.lengths[0]);
  const auto gap_wire = take_stream(wire, offset, header.lengths[1]);
  const auto mask_wire = take_stream(wire, offset, header.lengths[2]);
  const auto raw_wire = take_stream(wire, offset, header.lengths[3]);
  require(offset == wire.size(), "trailing BFE1 bytes");

  Bytes output;
  Stats result;
  if (header.mode == Mode::Raw) {
    require(header.exceptions == 0, "raw BFE1 frame has exceptions");
    require(header.lengths[0] == 0 && header.lengths[1] == 0 &&
                header.lengths[2] == 0,
            "raw BFE1 frame has integer streams");
    require(header.lengths[3] == std::uint64_t{header.count} * width,
            "raw BFE1 byte count mismatch");
    output.assign(raw_wire.begin(), raw_wire.end());
    result = Stats{-1, 0, Mode::Raw};
  } else {
    require(header.lengths[3] == 0, "exception BFE1 frame has raw payload");
    require(header.count == 0 || header.lengths[0] != 0,
            "nonempty BFE1 block has empty quantized stream");
    require((header.exceptions == 0) ==
                (header.lengths[1] == 0 && header.lengths[2] == 0),
            "BFE1 exception stream presence mismatch");
    if (header.dtype == DType::Float32) {
      output = restore_exception<float, std::uint32_t>(
          header, quantized_wire, gap_wire, mask_wire, decode_int);
    } else {
      output = restore_exception<double, std::uint64_t>(
          header, quantized_wire, gap_wire, mask_wire, decode_int);
    }
    result = Stats{header.precision, header.exceptions, Mode::Exception};
  }
  if (stats != nullptr) *stats = result;
  return output;
}

}  // namespace bos_float_exception
