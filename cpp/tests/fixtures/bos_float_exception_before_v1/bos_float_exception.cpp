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

std::uint64_t get_le_word(std::span<const std::uint8_t> input,
                          std::size_t offset,
                          std::size_t width) {
  require(width == 4 || width == 8, "invalid BFE1 floating width");
  require(offset <= input.size() && input.size() - offset >= width,
          "truncated IEEE word");
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < width; ++byte)
    value |= static_cast<std::uint64_t>(input[offset + byte]) << (byte * 8);
  return value;
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
  std::array<char, 64> text{};
  auto integer = std::to_chars(text.data(), text.data() + text.size(), value);
  if (integer.ec != std::errc{}) throw std::runtime_error("BFE1 integer formatting failed");
  char* cursor = integer.ptr;
  require(text.data() + text.size() - cursor >= 3,
          "BFE1 decimal prediction buffer overflow");
  *cursor++ = 'e';
  *cursor++ = '-';
  auto exponent = std::to_chars(cursor, text.data() + text.size(), precision);
  if (exponent.ec != std::errc{})
    throw std::runtime_error("BFE1 exponent formatting failed");
  Float prediction = 0;
  auto parsed = std::from_chars(text.data(), exponent.ptr, prediction,
                                std::chars_format::scientific);
  if (parsed.ec != std::errc{} || parsed.ptr != exponent.ptr)
    throw std::runtime_error("BFE1 decimal prediction parse failed");
  return prediction;
}

std::uint64_t power_of_ten(unsigned exponent) {
  require(exponent <= 19, "BFE1 decimal divisor exponent too large");
  std::uint64_t value = 1;
  for (unsigned i = 0; i < exponent; ++i) value *= 10;
  return value;
}

template <typename Float>
std::int64_t quantize(Float value, int precision) {
  if (!std::isfinite(value) || (value == Float{0} && std::signbit(value))) return 0;
  validate_precision(precision);

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

  const int shift = explicit_exponent - fractional_digits + precision;
  std::uint64_t magnitude = coefficient;
  if (shift >= 0) {
    // More than 19 decimal digits cannot fit in signed int64.  Check the
    // signed limit during multiplication so the arithmetic itself stays safe.
    if (shift > 19) return 0;
    const std::uint64_t limit = negative ? (std::uint64_t{1} << 63)
                                         : std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < shift; ++i) {
      if (magnitude > limit / 10) return 0;
      magnitude *= 10;
    }
  } else {
    const unsigned divisor_exponent = static_cast<unsigned>(-shift);
    if (divisor_exponent > 19) {
      magnitude = 0;
    } else {
      const std::uint64_t divisor = power_of_ten(divisor_exponent);
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

  const std::uint64_t limit = negative ? (std::uint64_t{1} << 63)
                                       : std::numeric_limits<std::int64_t>::max();
  if (magnitude > limit) return 0;
  if (negative) {
    if (magnitude == (std::uint64_t{1} << 63))
      return std::numeric_limits<std::int64_t>::min();
    return -static_cast<std::int64_t>(magnitude);
  }
  return static_cast<std::int64_t>(magnitude);
}

template <typename Float, typename Word>
struct CandidateStreams {
  std::vector<std::int64_t> quantized;
  std::vector<std::int64_t> gaps;
  std::vector<std::int64_t> masks;
};

template <typename Float, typename Word>
CandidateStreams<Float, Word> make_streams(std::span<const std::uint8_t> input,
                                           int precision) {
  static_assert(sizeof(Float) == sizeof(Word));
  const std::size_t count = input.size() / sizeof(Float);
  CandidateStreams<Float, Word> result;
  result.quantized.reserve(count);
  std::int64_t previous = -1;
  for (std::size_t index = 0; index < count; ++index) {
    const auto original_word =
        static_cast<Word>(get_le_word(input, index * sizeof(Float), sizeof(Float)));
    const auto original = std::bit_cast<Float>(original_word);
    const auto quantized = quantize(original, precision);
    const auto prediction = decimal_prediction<Float>(quantized, precision);
    const auto prediction_word = std::bit_cast<Word>(prediction);
    const Word mask = original_word ^ prediction_word;
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

Bytes uniform_pilot(std::span<const std::uint8_t> input,
                    std::size_t width,
                    std::uint32_t count,
                    std::uint32_t pilot_values) {
  require(pilot_values >= 2, "BFE1 pilot must contain at least two values");
  require(pilot_values < count, "BFE1 pilot must be smaller than the block");
  Bytes pilot;
  pilot.reserve(static_cast<std::size_t>(pilot_values) * width);
  for (std::uint32_t sample = 0; sample < pilot_values; ++sample) {
    // count <= 10000, so the product is comfortably inside uint64.  Integer
    // division makes the sample deterministic while endpoints map to 0,N-1.
    const auto index = static_cast<std::uint32_t>(
        (std::uint64_t{sample} * (count - 1)) / (pilot_values - 1));
    const auto begin = input.begin() + static_cast<std::size_t>(index) * width;
    pilot.insert(pilot.end(), begin, begin + width);
  }
  return pilot;
}

template <typename Float, typename Word>
std::pair<Bytes, std::uint32_t> exception_frame(
    std::span<const std::uint8_t> input,
    DType dtype,
    int precision,
    const EncodeInt& encode_int) {
  const auto streams = make_streams<Float, Word>(input, precision);
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
  for (std::size_t index = 0; index < header.count; ++index) {
    const Float prediction = decimal_prediction<Float>(quantized[index], header.precision);
    Word word = std::bit_cast<Word>(prediction);
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
  const auto count = static_cast<std::uint32_t>(count_size);

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

  Bytes best = raw_frame(input, dtype, count);
  Stats best_stats{-1, 0, Mode::Raw};

  const auto encode_exception = [&](std::span<const std::uint8_t> values, int precision) {
    if (dtype == DType::Float32)
      return exception_frame<float, std::uint32_t>(values, dtype, precision, encode_int);
    return exception_frame<double, std::uint64_t>(values, dtype, precision, encode_int);
  };

  if (options.fixed_precision.has_value()) {
    auto candidate = encode_exception(input, precisions.front());
    if (candidate.first.size() < best.size()) {
      best = std::move(candidate.first);
      best_stats = Stats{precisions.front(), candidate.second, Mode::Exception};
    }
  } else if (options.pilot_values != 0 && count > options.pilot_values) {
    require(options.pilot_values >= 2,
            "BFE1 pilot_values must be zero or at least two");
    const auto pilot = uniform_pilot(input, width, count, options.pilot_values);
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
    auto candidate = encode_exception(input, selected_precision);
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
      auto candidate = encode_exception(input, precision);
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
  if (stats != nullptr) *stats = best_stats;
  return best;
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
