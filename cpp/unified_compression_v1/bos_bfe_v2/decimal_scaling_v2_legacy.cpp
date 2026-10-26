#include "decimal_scaling_v2.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace native_baseline86_v1::decimal_scaling_v2::legacy {
namespace {

using U128 = unsigned __int128;

constexpr std::array<std::uint8_t, 4> magic{'D', 'S', 'C', '2'};
constexpr std::uint8_t version = 1;
constexpr std::size_t header_bytes = 56;
constexpr std::uint64_t maximum_buffer_bytes = 256ULL << 20;

enum class Class : std::uint8_t {
  common_scaled = 0,
  finite_factored = 1,
  ieee_special = 2,
};

struct Type {
  std::uint8_t width;
};

Type type(std::string_view dtype) {
  if (dtype == "float32") return {4};
  if (dtype == "float64") return {8};
  throw std::invalid_argument("DSC2 decimal scaling requires float32 or float64");
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

std::uint64_t classmap_size(std::uint64_t count) {
  return checked_add(count, 3, "DSC2 class count overflow") / 4;
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
  require(input.size() - pos >= 2, "truncated DSC2 uint16");
  const auto value = static_cast<std::uint16_t>(input[pos]) |
      static_cast<std::uint16_t>(input[pos + 1]) << 8;
  pos += 2;
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t& pos) {
  require(input.size() - pos >= 8, "truncated DSC2 uint64");
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

template <class Float, class UInt, int FractionBits, int ExponentBits>
std::optional<Decimal> decimal_from_bits(UInt raw) {
  static_assert(std::numeric_limits<Float>::is_iec559);
  constexpr UInt exponent_mask = (UInt{1} << ExponentBits) - 1;
  constexpr UInt sign_mask = UInt{1} << (FractionBits + ExponentBits);
  const UInt encoded_exponent = (raw >> FractionBits) & exponent_mask;
  // NaN, infinity, and negative zero retain their exact IEEE bits through
  // algorithm-compressed uint32 words instead of decimal conversion.
  if (encoded_exponent == exponent_mask || raw == sign_mask)
    return std::nullopt;

  const Float value = std::bit_cast<Float>(raw);
  std::array<char, 128> buffer{};
  const auto rendered = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                      value, std::chars_format::general);
  require(rendered.ec == std::errc{}, "floating to_chars failed");
  const std::string_view text(
      buffer.data(), static_cast<std::size_t>(rendered.ptr - buffer.data()));
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
            "nondigit emitted by floating to_chars");
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
  // Binary32/64 shortest round-trip decimal strings use at most max_digits10
  // significant digits (9/17), so their coefficient always fits int64.
  require(coefficient <=
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
          "shortest decimal coefficient exceeds int64");
  return Decimal{negative, coefficient, exponent10,
                 static_cast<std::uint16_t>(places)};
}

template <class UInt>
std::optional<Decimal> decimal(UInt raw);

template <>
std::optional<Decimal> decimal(std::uint32_t raw) {
  return decimal_from_bits<float, std::uint32_t, 23, 8>(raw);
}

template <>
std::optional<Decimal> decimal(std::uint64_t raw) {
  return decimal_from_bits<double, std::uint64_t, 52, 11>(raw);
}

std::int64_t signed_coefficient(const Decimal& value) {
  const auto magnitude = static_cast<std::int64_t>(value.coefficient);
  return value.negative ? -magnitude : magnitude;
}

std::optional<std::int64_t> integer_at(const Decimal& value,
                                       std::uint16_t places) {
  if (places < value.required_places) return std::nullopt;
  const int zeros = value.exponent10 + static_cast<int>(places);
  require(zeros >= 0, "DSC2 decimal places do not yield an integer");
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
std::optional<Float> decode_decimal(std::int64_t coefficient, int exponent10) {
  std::array<char, 128> buffer{};
  auto out = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                           coefficient);
  if (out.ec != std::errc{} || out.ptr == buffer.data() + buffer.size())
    return std::nullopt;
  *out.ptr++ = 'e';
  const auto exponent = std::to_chars(out.ptr, buffer.data() + buffer.size(),
                                      exponent10);
  if (exponent.ec != std::errc{}) return std::nullopt;
  out.ptr = exponent.ptr;
  Float decoded{};
  const auto parsed = std::from_chars(buffer.data(), out.ptr, decoded,
                                      std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != out.ptr) return std::nullopt;
  return decoded;
}

template <class UInt, class Float>
bool common_scaled(UInt raw, const Decimal& value, std::uint16_t places,
                   std::int64_t* encoded) {
  const auto integer = integer_at(value, places);
  if (!integer) return false;
  const auto restored = decode_decimal<Float>(*integer,
                                               -static_cast<int>(places));
  if (!restored || std::bit_cast<UInt>(*restored) != raw) return false;
  if (encoded) *encoded = *integer;
  return true;
}

void set_class(Bytes& classmap, std::size_t index, Class value) {
  classmap[index / 4] |= static_cast<std::uint8_t>(
      static_cast<std::uint8_t>(value) << (2 * (index % 4)));
}

std::uint8_t get_class(std::span<const std::uint8_t> classmap,
                       std::uint64_t index) {
  return static_cast<std::uint8_t>(
      (classmap[checked_size(index / 4, "DSC2 class index overflow")] >>
       (2 * (index % 4))) & 3U);
}

struct Header {
  std::uint8_t width = 0;
  std::uint16_t places = 0;
  std::uint64_t count = 0;
  std::uint64_t word_count = 0;
  std::uint64_t scaled_count = 0;
  std::uint64_t overflow_count = 0;
  std::uint64_t special_count = 0;
  std::uint64_t classmap_bytes = 0;
};

Header parse_header(std::span<const std::uint8_t> metadata) {
  require(metadata.size() >= header_bytes, "truncated DSC2 metadata");
  require(std::equal(magic.begin(), magic.end(), metadata.begin()),
          "invalid DSC2 magic");
  std::size_t pos = 4;
  require(metadata[pos++] == version, "unsupported DSC2 version");
  Header header;
  header.width = metadata[pos++];
  require(header.width == 4 || header.width == 8, "invalid DSC2 dtype width");
  header.places = get_u16(metadata, pos);
  require(header.places <= maximum_decimal_places,
          "DSC2 decimal places exceeds 324");
  header.count = get_u64(metadata, pos);
  header.word_count = get_u64(metadata, pos);
  header.scaled_count = get_u64(metadata, pos);
  header.overflow_count = get_u64(metadata, pos);
  header.special_count = get_u64(metadata, pos);
  header.classmap_bytes = get_u64(metadata, pos);
  require(pos == header_bytes, "internal DSC2 parser offset mismatch");

  const auto classified = checked_add(
      header.scaled_count,
      checked_add(header.overflow_count, header.special_count,
                  "DSC2 classified count overflow"),
      "DSC2 classified count overflow");
  require(classified == header.count, "DSC2 value counts mismatch");
  const auto special_words = checked_mul(
      header.special_count, header.width / 4,
      "DSC2 special word count overflow");
  const auto expected_words = checked_add(
      header.scaled_count,
      checked_add(checked_mul(header.overflow_count, 2,
                              "DSC2 overflow word count overflow"),
                  special_words, "DSC2 word count overflow"),
      "DSC2 word count overflow");
  require(expected_words == header.word_count, "DSC2 integer word count mismatch");
  require(header.classmap_bytes == classmap_size(header.count),
          "DSC2 classmap length mismatch");
  const auto expected_size = checked_add(
      header_bytes, header.classmap_bytes, "DSC2 metadata size overflow");
  require(expected_size == metadata.size(),
          "DSC2 trailing or truncated metadata");
  require(expected_size <= maximum_buffer_bytes,
          "DSC2 metadata exceeds 256 MiB");
  require(checked_mul(header.word_count, 8,
                      "DSC2 word payload size overflow") <= maximum_buffer_bytes,
          "DSC2 integer word payload exceeds 256 MiB");
  require(checked_mul(header.count, header.width,
                      "DSC2 output size overflow") <= maximum_buffer_bytes,
          "DSC2 decoded output exceeds 256 MiB");
  return header;
}

void validate_classmap(std::span<const std::uint8_t> metadata,
                       const Header& header) {
  const auto classmap = metadata.subspan(
      header_bytes,
      checked_size(header.classmap_bytes, "DSC2 classmap size_t overflow"));
  if (header.count % 4 != 0 && !classmap.empty()) {
    const auto used_bits = static_cast<unsigned>((header.count % 4) * 2);
    require((classmap.back() & static_cast<std::uint8_t>(0xffU << used_bits)) == 0,
            "DSC2 classmap padding is nonzero");
  }
  std::uint64_t scaled = 0, overflow = 0, special = 0;
  for (std::uint64_t i = 0; i < header.count; ++i) {
    switch (get_class(classmap, i)) {
      case static_cast<std::uint8_t>(Class::common_scaled): ++scaled; break;
      case static_cast<std::uint8_t>(Class::finite_factored): ++overflow; break;
      case static_cast<std::uint8_t>(Class::ieee_special): ++special; break;
      default: throw std::runtime_error("DSC2 class 3 is invalid");
    }
  }
  require(scaled == header.scaled_count &&
              overflow == header.overflow_count &&
              special == header.special_count,
          "DSC2 classmap/count mismatch");
}

void validate_input_size(std::span<const std::uint8_t> input,
                         std::uint8_t width) {
  if (input.size() % width != 0)
    throw std::invalid_argument("DSC2 input width mismatch");
  const auto count = static_cast<std::uint64_t>(input.size() / width);
  const auto worst_words = checked_mul(count, 2, "DSC2 worst word count overflow");
  if (input.size() > maximum_buffer_bytes ||
      checked_mul(worst_words, 8, "DSC2 worst word bytes overflow") >
          maximum_buffer_bytes ||
      checked_add(header_bytes, classmap_size(count),
                  "DSC2 worst metadata overflow") > maximum_buffer_bytes)
    throw std::length_error("DSC2 256 MiB bound exceeded");
}

template <class UInt, class Float>
Scaled forward_impl(std::span<const std::uint8_t> input, std::uint8_t width,
                    std::uint16_t places) {
  const auto count = static_cast<std::uint64_t>(input.size() / width);
  Bytes classmap(checked_size(classmap_size(count),
                              "DSC2 classmap size_t overflow"), 0);
  std::vector<std::int64_t> words;
  words.reserve(checked_size(checked_mul(count, 2,
      "DSC2 word reserve overflow"), "DSC2 word reserve size_t overflow"));
  std::uint64_t scaled = 0, overflow = 0, special = 0;

  for (std::size_t i = 0; i < checked_size(count, "DSC2 count size_t overflow");
       ++i) {
    const UInt raw = read_little<UInt>(input, i);
    const auto parsed = decimal(raw);
    std::int64_t common = 0;
    if (parsed && common_scaled<UInt, Float>(raw, *parsed, places, &common)) {
      set_class(classmap, i, Class::common_scaled);
      words.push_back(common);
      ++scaled;
      continue;
    }
    if (parsed) {
      require(places >= parsed->required_places,
              "DSC2 finite value exceeds frozen decimal precision");
      const auto k = static_cast<std::int64_t>(places) + parsed->exponent10;
      require(k >= 0, "DSC2 factored decimal exponent is negative");
      const auto coefficient = signed_coefficient(*parsed);
      const auto restored = decode_decimal<Float>(coefficient,
                                                   parsed->exponent10);
      require(restored && std::bit_cast<UInt>(*restored) == raw,
              "DSC2 factored decimal is not bit-exact");
      set_class(classmap, i, Class::finite_factored);
      words.push_back(coefficient);
      words.push_back(k);
      ++overflow;
      continue;
    }

    set_class(classmap, i, Class::ieee_special);
    words.push_back(static_cast<std::int64_t>(
        static_cast<std::uint32_t>(raw & UInt{0xffffffffU})));
    if constexpr (sizeof(UInt) == 8) {
      words.push_back(static_cast<std::int64_t>(
          static_cast<std::uint32_t>(raw >> 32)));
    }
    ++special;
  }

  Bytes metadata;
  metadata.reserve(header_bytes + classmap.size());
  metadata.insert(metadata.end(), magic.begin(), magic.end());
  metadata.push_back(version);
  metadata.push_back(width);
  put_u16(metadata, places);
  put_u64(metadata, count);
  put_u64(metadata, words.size());
  put_u64(metadata, scaled);
  put_u64(metadata, overflow);
  put_u64(metadata, special);
  put_u64(metadata, classmap.size());
  require(metadata.size() == header_bytes, "internal DSC2 header size mismatch");
  metadata.insert(metadata.end(), classmap.begin(), classmap.end());
  return {std::move(words), std::move(metadata), places,
          scaled, overflow, special};
}

template <class UInt, class Float>
Bytes inverse_impl(std::span<const std::int64_t> values,
                   std::span<const std::uint8_t> metadata,
                   const Header& header) {
  require(values.size() == header.word_count,
          "DSC2 decoded integer word count mismatch");
  const auto classmap = metadata.subspan(
      header_bytes,
      checked_size(header.classmap_bytes, "DSC2 classmap size_t overflow"));
  Bytes output;
  output.reserve(checked_size(checked_mul(header.count, header.width,
      "DSC2 output size overflow"), "DSC2 output size_t overflow"));
  std::size_t word_pos = 0;
  for (std::uint64_t i = 0; i < header.count; ++i) {
    const auto cls = get_class(classmap, i);
    if (cls == static_cast<std::uint8_t>(Class::common_scaled)) {
      require(word_pos < values.size(), "truncated DSC2 common word");
      const auto integer = values[word_pos++];
      const auto decoded = decode_decimal<Float>(
          integer, -static_cast<int>(header.places));
      require(decoded.has_value(), "DSC2 common word cannot be decoded");
      const UInt raw = std::bit_cast<UInt>(*decoded);
      const auto parsed = decimal(raw);
      std::int64_t canonical = 0;
      require(parsed && common_scaled<UInt, Float>(
                            raw, *parsed, header.places, &canonical) &&
                  canonical == integer,
              "DSC2 common word is not canonical");
      append_little(output, raw);
      continue;
    }
    if (cls == static_cast<std::uint8_t>(Class::finite_factored)) {
      require(values.size() - word_pos >= 2,
              "truncated DSC2 factored words");
      const auto coefficient = values[word_pos++];
      const auto k = values[word_pos++];
      require(k >= 0 &&
                  k <= static_cast<std::int64_t>(
                      maximum_decimal_places + 400),
              "DSC2 factored exponent adjustment is out of range");
      const auto exponent64 = k - static_cast<std::int64_t>(header.places);
      require(exponent64 >= std::numeric_limits<int>::min() &&
                  exponent64 <= std::numeric_limits<int>::max(),
              "DSC2 factored exponent overflows int");
      const auto decoded = decode_decimal<Float>(
          coefficient, static_cast<int>(exponent64));
      require(decoded.has_value(), "DSC2 factored words cannot be decoded");
      const UInt raw = std::bit_cast<UInt>(*decoded);
      const auto parsed = decimal(raw);
      std::int64_t common = 0;
      require(parsed && parsed->required_places <= header.places &&
                  !common_scaled<UInt, Float>(
                      raw, *parsed, header.places, &common) &&
                  signed_coefficient(*parsed) == coefficient &&
                  static_cast<std::int64_t>(header.places) +
                          parsed->exponent10 == k,
              "DSC2 factored words are not canonical");
      append_little(output, raw);
      continue;
    }
    if (cls == static_cast<std::uint8_t>(Class::ieee_special)) {
      require(word_pos < values.size(), "truncated DSC2 special word");
      const auto low_word = values[word_pos++];
      require(low_word >= 0 &&
                  static_cast<std::uint64_t>(low_word) <= UINT32_MAX,
              "DSC2 special low word is not uint32");
      UInt raw = static_cast<UInt>(static_cast<std::uint32_t>(low_word));
      if constexpr (sizeof(UInt) == 8) {
        require(word_pos < values.size(), "truncated DSC2 special high word");
        const auto high_word = values[word_pos++];
        require(high_word >= 0 &&
                    static_cast<std::uint64_t>(high_word) <= UINT32_MAX,
                "DSC2 special high word is not uint32");
        raw |= static_cast<UInt>(static_cast<std::uint32_t>(high_word)) << 32;
      }
      require(!decimal(raw).has_value(),
              "DSC2 special words encode a nonspecial value");
      append_little(output, raw);
      continue;
    }
    throw std::runtime_error("DSC2 class 3 is invalid");
  }
  require(word_pos == values.size(), "extra DSC2 integer words");
  return output;
}

}  // namespace

std::uint16_t required_decimal_places(std::span<const std::uint8_t> input,
                                      std::string_view dtype) {
  return decimal_scaling::required_decimal_places(input, dtype);
}

Scaled forward(std::span<const std::uint8_t> input, std::string_view dtype,
               std::uint16_t p) {
  const auto info = type(dtype);
  validate_input_size(input, info.width);
  if (p > maximum_decimal_places)
    throw std::invalid_argument("DSC2 decimal places must be in [0,324]");
  const auto required = required_decimal_places(input, dtype);
  if (p < required)
    throw std::invalid_argument(
        "DSC2 decimal places is below canonical block requirement");
  return info.width == 4
      ? forward_impl<std::uint32_t, float>(input, info.width, p)
      : forward_impl<std::uint64_t, double>(input, info.width, p);
}

MetadataInfo inspect(std::span<const std::uint8_t> metadata,
                     std::string_view dtype) {
  const auto expected = type(dtype);
  const auto header = parse_header(metadata);
  require(header.width == expected.width, "DSC2 dtype mismatch");
  validate_classmap(metadata, header);
  return {header.count, header.word_count, header.scaled_count,
          header.overflow_count, header.special_count, header.places};
}

Bytes inverse(std::span<const std::int64_t> values,
              std::span<const std::uint8_t> metadata,
              std::string_view dtype) {
  const auto expected = type(dtype);
  const auto header = parse_header(metadata);
  require(header.width == expected.width, "DSC2 dtype mismatch");
  validate_classmap(metadata, header);
  return header.width == 4
      ? inverse_impl<std::uint32_t, float>(values, metadata, header)
      : inverse_impl<std::uint64_t, double>(values, metadata, header);
}

}  // namespace native_baseline86_v1::decimal_scaling_v2
