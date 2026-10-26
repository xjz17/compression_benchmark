#include "decimal_scaling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

namespace native_baseline86_v1::decimal_scaling {
namespace {

using U128 = unsigned __int128;
constexpr std::array<std::uint8_t, 4> magic{'D', 'S', 'C', '1'};
constexpr std::uint8_t version = 1;
constexpr std::size_t header_bytes = 48;

struct Type {
  std::uint8_t width;
};

Type type(std::string_view dtype) {
  if (dtype == "float32") return {4};
  if (dtype == "float64") return {8};
  throw std::invalid_argument("decimal scaling requires float32 or float64");
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t checked_size(std::uint64_t value, const char* message) {
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::length_error(message);
  return static_cast<std::size_t>(value);
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char* message) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::length_error(message);
  return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right,
                          const char* message) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::length_error(message);
  return left * right;
}

std::uint64_t bitmap_size(std::uint64_t count) {
  return checked_add(count, 7, "decimal scaling bitmap count overflow") / 8;
}

void put_u16(Bytes& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void put_u64(Bytes& out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint16_t get_u16(std::span<const std::uint8_t> input, std::size_t& pos) {
  require(input.size() - pos >= 2, "truncated DSC1 uint16");
  const auto value = static_cast<std::uint16_t>(input[pos]) |
      static_cast<std::uint16_t>(input[pos + 1]) << 8;
  pos += 2;
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t& pos) {
  require(input.size() - pos >= 8, "truncated DSC1 uint64");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(input[pos++]) << shift;
  return value;
}

template <class UInt>
UInt read_little(std::span<const std::uint8_t> input, std::size_t index) {
  const auto offset = index * sizeof(UInt);
  UInt value = 0;
  for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
    value |= static_cast<UInt>(input[offset + byte]) << (byte * 8);
  return value;
}

template <class UInt>
void append_little(Bytes& output, UInt value) {
  for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

struct Decimal {
  bool negative = false;
  std::uint64_t coefficient = 0;
  int exponent10 = 0;
  std::uint16_t required_places = 0;
};

template <class Float, class UInt, int FractionBits, int ExponentBits,
          int ExponentBias>
std::optional<Decimal> decimal_from_bits(UInt raw) {
  static_assert(std::numeric_limits<Float>::is_iec559);
  constexpr UInt exponent_mask = (UInt{1} << ExponentBits) - 1;
  constexpr UInt sign_mask = UInt{1} << (FractionBits + ExponentBits);
  const UInt encoded_exponent = (raw >> FractionBits) & exponent_mask;
  // Preserve NaN payloads, infinities, and the sign bit of negative zero
  // without first performing floating arithmetic on them.
  if (encoded_exponent == exponent_mask || raw == sign_mask)
    return std::nullopt;

  const Float value = std::bit_cast<Float>(raw);
  std::array<char, 128> buffer{};
  const auto rendered = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                      value, std::chars_format::general);
  require(rendered.ec == std::errc{}, "floating to_chars failed");
  const std::string_view text(buffer.data(),
                              static_cast<std::size_t>(rendered.ptr - buffer.data()));
  std::size_t pos = 0;
  bool negative = false;
  if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
    negative = text[pos] == '-';
    ++pos;
  }
  const auto exponent_pos = text.find_first_of("eE", pos);
  const auto mantissa_end = exponent_pos == std::string_view::npos
      ? text.size() : exponent_pos;
  int explicit_exponent = 0;
  if (exponent_pos != std::string_view::npos) {
    const char* begin = text.data() + exponent_pos + 1;
    if (begin != text.data() + text.size() && *begin == '+') ++begin;
    const auto parsed = std::from_chars(begin, text.data() + text.size(),
                                        explicit_exponent);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
            "invalid exponent emitted by floating to_chars");
  }
  std::uint64_t coefficient = 0;
  bool after_point = false;
  bool saw_digit = false;
  int digits_after_point = 0;
  for (; pos < mantissa_end; ++pos) {
    const char ch = text[pos];
    if (ch == '.') {
      require(!after_point, "multiple points emitted by floating to_chars");
      after_point = true;
      continue;
    }
    require(ch >= '0' && ch <= '9',
            "nondigit emitted by finite floating to_chars");
    saw_digit = true;
    const auto digit = static_cast<unsigned>(ch - '0');
    require(coefficient <=
                (std::numeric_limits<std::uint64_t>::max() - digit) / 10,
            "shortest decimal coefficient exceeds uint64");
    coefficient = coefficient * 10 + digit;
    if (after_point) ++digits_after_point;
  }
  require(saw_digit, "floating to_chars emitted no mantissa digit");
  const int exponent10 = explicit_exponent - digits_after_point;
  const int places = std::max(0, -exponent10);
  require(places <= maximum_decimal_places,
          "shortest binary32/64 decimal requires more than 324 places");
  return Decimal{negative, coefficient, exponent10,
                 static_cast<std::uint16_t>(places)};
}

template <class UInt>
std::optional<Decimal> decimal(UInt raw);

template <>
std::optional<Decimal> decimal(std::uint32_t raw) {
  return decimal_from_bits<float, std::uint32_t, 23, 8, 127>(raw);
}

template <>
std::optional<Decimal> decimal(std::uint64_t raw) {
  return decimal_from_bits<double, std::uint64_t, 52, 11, 1023>(raw);
}

std::optional<std::int64_t> integer_at(const Decimal& value,
                                       std::uint16_t places) {
  if (places < value.required_places) return std::nullopt;
  const int zeros = value.exponent10 + static_cast<int>(places);
  require(zeros >= 0, "decimal scaling places do not yield an integer");
  const U128 limit = value.negative ? (U128{1} << 63)
                                    : static_cast<U128>(
                                          std::numeric_limits<std::int64_t>::max());
  U128 magnitude = value.coefficient;
  for (int i = 0; i < zeros; ++i) {
    if (magnitude > limit / 10) return std::nullopt;
    magnitude *= 10;
  }
  if (magnitude > limit) return std::nullopt;
  if (!value.negative) return static_cast<std::int64_t>(magnitude);
  if (magnitude == (U128{1} << 63))
    return std::numeric_limits<std::int64_t>::min();
  return -static_cast<std::int64_t>(magnitude);
}

template <class Float>
std::optional<Float> decode_integer(std::int64_t value,
                                    std::uint16_t places) {
  if (places > maximum_decimal_places) return std::nullopt;
  std::array<char, 128> buffer{};
  auto out = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (out.ec != std::errc{}) return std::nullopt;
  if (places != 0) {
    if (out.ptr == buffer.data() + buffer.size()) return std::nullopt;
    *out.ptr++ = 'e';
    if (out.ptr == buffer.data() + buffer.size()) return std::nullopt;
    *out.ptr++ = '-';
    const auto exponent = std::to_chars(out.ptr, buffer.data() + buffer.size(),
                                        places);
    if (exponent.ec != std::errc{}) return std::nullopt;
    out.ptr = exponent.ptr;
  }
  Float decoded{};
  const auto parsed = std::from_chars(buffer.data(), out.ptr, decoded,
                                      std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != out.ptr) return std::nullopt;
  return decoded;
}

template <class UInt>
bool scalable(UInt raw, const Decimal& value, std::uint16_t places,
              std::int64_t* encoded) {
  const auto integer = integer_at(value, places);
  if (!integer) return false;
  if constexpr (sizeof(UInt) == 4) {
    const auto restored = decode_integer<float>(*integer, places);
    if (!restored || std::bit_cast<std::uint32_t>(*restored) != raw)
      return false;
  } else {
    const auto restored = decode_integer<double>(*integer, places);
    if (!restored || std::bit_cast<std::uint64_t>(*restored) != raw)
      return false;
  }
  if (encoded) *encoded = *integer;
  return true;
}

template <class UInt>
std::uint16_t required_impl(std::span<const std::uint8_t> input) {
  std::uint16_t required_places = 0;
  for (std::size_t i = 0; i < input.size() / sizeof(UInt); ++i) {
    const auto parsed = decimal(read_little<UInt>(input, i));
    if (parsed)
      required_places = std::max(required_places, parsed->required_places);
  }
  return required_places;
}

void validate_input_size(std::span<const std::uint8_t> input,
                         std::uint8_t width) {
  if (input.size() % width != 0)
    throw std::invalid_argument("decimal scaling input width mismatch");
  const auto count = static_cast<std::uint64_t>(input.size() / width);
  const auto integer_bytes = checked_mul(count, 8,
      "decimal scaling integer payload size overflow");
  const auto worst_metadata = checked_add(
      header_bytes,
      checked_add(bitmap_size(count), checked_mul(count, width,
          "decimal scaling exception size overflow"),
          "decimal scaling metadata size overflow"),
      "decimal scaling metadata size overflow");
  if (input.size() > maximum_buffer_bytes ||
      integer_bytes > maximum_buffer_bytes ||
      worst_metadata > maximum_buffer_bytes)
    throw std::length_error("decimal scaling 256 MiB bound exceeded");
}

template <class UInt>
Scaled forward_impl(std::span<const std::uint8_t> input, std::uint8_t width,
                    std::uint16_t places) {
  const auto count = static_cast<std::uint64_t>(input.size() / width);
  Bytes bitmap(checked_size(bitmap_size(count), "bitmap size_t overflow"), 0);
  Bytes raw;
  std::vector<std::int64_t> integers;
  integers.reserve(checked_size(count, "integer count size_t overflow"));
  std::uint64_t exceptions = 0;
  for (std::size_t i = 0; i < checked_size(count, "input count size_t overflow");
       ++i) {
    const UInt original = read_little<UInt>(input, i);
    const auto parsed = decimal(original);
    std::int64_t integer = 0;
    if (parsed && scalable(original, *parsed, places, &integer)) {
      integers.push_back(integer);
    } else {
      bitmap[i / 8] |= static_cast<std::uint8_t>(1U << (i % 8));
      append_little(raw, original);
      ++exceptions;
    }
  }
  const auto metadata_size = checked_add(
      header_bytes, checked_add(bitmap.size(), raw.size(),
          "decimal scaling metadata size overflow"),
      "decimal scaling metadata size overflow");
  if (metadata_size > maximum_buffer_bytes)
    throw std::length_error("decimal scaling metadata exceeds 256 MiB");
  Bytes metadata;
  metadata.reserve(checked_size(metadata_size, "metadata size_t overflow"));
  metadata.insert(metadata.end(), magic.begin(), magic.end());
  metadata.push_back(version);
  metadata.push_back(width);
  put_u16(metadata, places);
  put_u64(metadata, count);
  put_u64(metadata, integers.size());
  put_u64(metadata, exceptions);
  put_u64(metadata, bitmap.size());
  put_u64(metadata, raw.size());
  require(metadata.size() == header_bytes, "internal DSC1 header size mismatch");
  metadata.insert(metadata.end(), bitmap.begin(), bitmap.end());
  metadata.insert(metadata.end(), raw.begin(), raw.end());
  return {std::move(integers), std::move(metadata), places, exceptions};
}

struct Header {
  std::uint8_t width = 0;
  std::uint16_t places = 0;
  std::uint64_t count = 0;
  std::uint64_t scaled_count = 0;
  std::uint64_t exception_count = 0;
  std::uint64_t bitmap_bytes = 0;
  std::uint64_t raw_bytes = 0;
};

Header parse_header(std::span<const std::uint8_t> metadata) {
  require(metadata.size() >= header_bytes, "truncated DSC1 metadata");
  require(std::equal(magic.begin(), magic.end(), metadata.begin()),
          "invalid DSC1 magic");
  std::size_t pos = 4;
  require(metadata[pos++] == version, "unsupported DSC1 version");
  Header header;
  header.width = metadata[pos++];
  require(header.width == 4 || header.width == 8, "invalid DSC1 dtype width");
  header.places = get_u16(metadata, pos);
  require(header.places <= maximum_decimal_places,
          "DSC1 decimal places exceeds 324");
  header.count = get_u64(metadata, pos);
  header.scaled_count = get_u64(metadata, pos);
  header.exception_count = get_u64(metadata, pos);
  header.bitmap_bytes = get_u64(metadata, pos);
  header.raw_bytes = get_u64(metadata, pos);
  require(pos == header_bytes, "internal DSC1 parser offset mismatch");
  require(header.scaled_count <= header.count &&
              header.exception_count <= header.count &&
              checked_add(header.scaled_count, header.exception_count,
                          "DSC1 count sum overflow") == header.count,
          "DSC1 value counts mismatch");
  require(header.bitmap_bytes == bitmap_size(header.count),
          "DSC1 bitmap length mismatch");
  require(header.raw_bytes == checked_mul(header.exception_count, header.width,
              "DSC1 raw exception length overflow"),
          "DSC1 raw exception length mismatch");
  const auto expected = checked_add(
      header_bytes,
      checked_add(header.bitmap_bytes, header.raw_bytes,
                  "DSC1 total metadata overflow"),
      "DSC1 total metadata overflow");
  require(expected == metadata.size(), "DSC1 trailing or truncated metadata");
  require(expected <= maximum_buffer_bytes,
          "DSC1 metadata exceeds 256 MiB");
  require(checked_mul(header.count, header.width,
                      "DSC1 output size overflow") <= maximum_buffer_bytes,
          "DSC1 decoded output exceeds 256 MiB");
  require(checked_mul(header.scaled_count, 8,
                      "DSC1 integer size overflow") <= maximum_buffer_bytes,
          "DSC1 integer payload exceeds 256 MiB");
  return header;
}

template <class UInt>
void validate_metadata(std::span<const std::uint8_t> metadata,
                       const Header& header) {
  const auto bitmap = metadata.subspan(
      header_bytes,
      checked_size(header.bitmap_bytes, "DSC1 bitmap size_t overflow"));
  if (header.count % 8 != 0 && !bitmap.empty()) {
    const auto used = static_cast<unsigned>(header.count % 8);
    require((bitmap.back() & static_cast<std::uint8_t>(0xffU << used)) == 0,
            "DSC1 bitmap padding is nonzero");
  }
  std::uint64_t ones = 0;
  for (const auto byte : bitmap) ones += std::popcount(byte);
  require(ones == header.exception_count,
          "DSC1 exception bitmap/count mismatch");

  std::size_t raw_pos = checked_size(
      checked_add(header_bytes, header.bitmap_bytes,
                  "DSC1 raw offset overflow"),
      "DSC1 raw offset size_t overflow");
  for (std::uint64_t i = 0; i < header.count; ++i) {
    const bool exception =
        (bitmap[checked_size(i / 8, "DSC1 bitmap index overflow")] >> (i % 8)) & 1U;
    if (!exception) continue;
    require(metadata.size() - raw_pos >= sizeof(UInt),
            "truncated DSC1 raw exception");
    UInt original = 0;
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
      original |= static_cast<UInt>(metadata[raw_pos++]) << (byte * 8);
    const auto parsed = decimal(original);
    std::int64_t integer = 0;
    require(!parsed || !scalable(original, *parsed, header.places, &integer),
            "DSC1 exception lane is noncanonical");
  }
  require(raw_pos == metadata.size(), "extra DSC1 raw exceptions");
}

template <class UInt, class Float>
Bytes inverse_impl(std::span<const std::int64_t> values,
                   std::span<const std::uint8_t> metadata,
                   const Header& header) {
  require(values.size() == header.scaled_count,
          "DSC1 scaled integer count mismatch");
  const auto bitmap_begin = header_bytes;
  std::size_t raw_pos = checked_size(
      checked_add(header_bytes, header.bitmap_bytes,
                  "DSC1 raw offset overflow"),
      "DSC1 raw offset size_t overflow");
  const auto bitmap = metadata.subspan(bitmap_begin,
      checked_size(header.bitmap_bytes, "DSC1 bitmap size_t overflow"));

  Bytes output;
  output.reserve(checked_size(checked_mul(header.count, header.width,
      "DSC1 output size overflow"), "DSC1 output size_t overflow"));
  std::size_t scaled_pos = 0;
  for (std::uint64_t i = 0; i < header.count; ++i) {
    const bool exception =
        (bitmap[checked_size(i / 8, "DSC1 bitmap index overflow")] >> (i % 8)) & 1U;
    if (exception) {
      require(metadata.size() - raw_pos >= sizeof(UInt),
              "truncated DSC1 raw exception");
      UInt original = 0;
      for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
        original |= static_cast<UInt>(metadata[raw_pos++]) << (byte * 8);
      append_little(output, original);
    } else {
      require(scaled_pos < values.size(), "truncated DSC1 scaled values");
      const auto decoded = decode_integer<Float>(values[scaled_pos++],
                                                  header.places);
      require(decoded.has_value(), "DSC1 scaled integer cannot be decoded");
      const UInt original = std::bit_cast<UInt>(*decoded);
      const auto parsed = decimal(original);
      std::int64_t canonical = 0;
      require(parsed && scalable(original, *parsed, header.places, &canonical) &&
                  canonical == values[scaled_pos - 1],
              "DSC1 scaled lane is not canonical");
      append_little(output, original);
    }
  }
  require(scaled_pos == values.size(), "extra DSC1 scaled values");
  require(raw_pos == metadata.size(), "extra DSC1 raw exceptions");
  return output;
}

}  // namespace

std::uint16_t required_decimal_places(std::span<const std::uint8_t> input,
                                      std::string_view dtype) {
  const auto info = type(dtype);
  validate_input_size(input, info.width);
  return info.width == 4 ? required_impl<std::uint32_t>(input)
                         : required_impl<std::uint64_t>(input);
}

Scaled forward(std::span<const std::uint8_t> input, std::string_view dtype,
               std::uint16_t places) {
  const auto info = type(dtype);
  validate_input_size(input, info.width);
  if (places > maximum_decimal_places)
    throw std::invalid_argument("decimal scaling places must be in [0,324]");
  const auto required = info.width == 4
      ? required_impl<std::uint32_t>(input)
      : required_impl<std::uint64_t>(input);
  if (places < required)
    throw std::invalid_argument(
        "decimal scaling places is below canonical block requirement");
  return info.width == 4
      ? forward_impl<std::uint32_t>(input, info.width, places)
      : forward_impl<std::uint64_t>(input, info.width, places);
}

MetadataInfo inspect(std::span<const std::uint8_t> metadata,
                     std::string_view dtype) {
  const auto expected = type(dtype);
  const auto header = parse_header(metadata);
  require(header.width == expected.width, "DSC1 dtype mismatch");
  if (header.width == 4)
    validate_metadata<std::uint32_t>(metadata, header);
  else
    validate_metadata<std::uint64_t>(metadata, header);
  return {header.places, header.count, header.scaled_count,
          header.exception_count};
}

Bytes inverse(std::span<const std::int64_t> values,
              std::span<const std::uint8_t> metadata,
              std::string_view dtype) {
  const auto expected = type(dtype);
  const auto header = parse_header(metadata);
  require(header.width == expected.width, "DSC1 dtype mismatch");
  if (header.width == 4)
    validate_metadata<std::uint32_t>(metadata, header);
  else
    validate_metadata<std::uint64_t>(metadata, header);
  return header.width == 4
      ? inverse_impl<std::uint32_t, float>(values, metadata, header)
      : inverse_impl<std::uint64_t, double>(values, metadata, header);
}

}  // namespace native_baseline86_v1::decimal_scaling
