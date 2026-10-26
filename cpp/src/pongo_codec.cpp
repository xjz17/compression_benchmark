#include "pongo_codec.hpp"
#include "pongo_exact_stages.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace pongo_codec {
namespace {

constexpr int kDoubleMantissaBits = 52;
constexpr int kDoubleExponentBias = 1023;
constexpr int kMaxDecimalPlaces = 16;

struct BitWriter {
  std::vector<std::uint8_t> bytes;
  int used = 0;

  void write_bit(bool value) {
    if (used == 0) {
      bytes.push_back(0);
    }
    if (value) {
      bytes.back() |= static_cast<std::uint8_t>(1u << (7 - used));
    }
    used = (used + 1) & 7;
  }

  void write_bits(std::uint64_t value, int bit_count) {
    for (int bit = bit_count - 1; bit >= 0; --bit) {
      write_bit(((value >> bit) & 1u) != 0);
    }
  }
};

struct BitReader {
  std::span<const std::uint8_t> bytes;
  std::size_t byte_index = 0;
  int used = 0;
  bool ok = true;

  bool read_bit() {
    if (!ok || byte_index >= bytes.size()) {
      ok = false;
      return false;
    }
    const bool value = ((bytes[byte_index] >> (7 - used)) & 1u) != 0;
    ++used;
    if (used == 8) {
      used = 0;
      ++byte_index;
    }
    return value;
  }

  std::uint64_t read_bits(int bit_count) {
    std::uint64_t value = 0;
    for (int bit = 0; bit < bit_count; ++bit) {
      value = (value << 1) | static_cast<std::uint64_t>(read_bit());
      if (!ok) {
        return 0;
      }
    }
    return value;
  }
};

struct DecimalParts {
  std::string integer;
  std::string fraction;
};

struct ReencodedValue {
  std::uint64_t bits = 0;
  int leading_decimal_zeros = 0;
};

enum class Ablation { kNone, kEraser, kXor, kTrailingZeroPack };

Ablation parse_ablation(std::string_view omitted_operator) {
  if (omitted_operator.empty() || omitted_operator == "none") {
    return Ablation::kNone;
  }
  if (omitted_operator == "eraser") {
    return Ablation::kEraser;
  }
  if (omitted_operator == "xor") {
    return Ablation::kXor;
  }
  if (omitted_operator == "trailing_zero_pack" ||
      omitted_operator == "trailing-zero packing" ||
      omitted_operator == "trailing zero packing") {
    return Ablation::kTrailingZeroPack;
  }
  throw std::invalid_argument("unsupported Pongo ablation operator");
}

void append_count(std::vector<std::uint8_t>& out, std::uint64_t count) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::uint8_t>((count >> (8 * i)) & 0xffu));
  }
}

bool read_count(std::span<const std::uint8_t> encoded, std::uint64_t& count) {
  if (encoded.size() < 8) {
    return false;
  }
  count = 0;
  for (int i = 0; i < 8; ++i) {
    count |= static_cast<std::uint64_t>(encoded[static_cast<std::size_t>(i)]) << (8 * i);
  }
  return true;
}

std::uint64_t low_mask(int bit_count) {
  if (bit_count <= 0) {
    return 0;
  }
  if (bit_count >= 64) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return (std::uint64_t{1} << bit_count) - 1;
}

std::uint64_t reverse_low_bits(std::uint64_t value, int bit_count) {
  std::uint64_t reversed = 0;
  for (int i = 0; i < bit_count; ++i) {
    reversed = (reversed << 1) | ((value >> i) & 1u);
  }
  return reversed;
}

bool shortest_decimal_parts(double value, DecimalParts& out) {
  char buffer[128];
  const auto result = std::to_chars(
      buffer, buffer + sizeof(buffer), std::fabs(value), std::chars_format::general);
  if (result.ec != std::errc{}) {
    return false;
  }
  std::string_view text(buffer, static_cast<std::size_t>(result.ptr - buffer));
  const std::size_t exponent_pos = text.find_first_of("eE");
  const std::string_view mantissa = text.substr(0, exponent_pos);
  int decimal_exponent = 0;
  if (exponent_pos != std::string_view::npos) {
    const std::string_view exponent = text.substr(exponent_pos + 1);
    const auto parsed = std::from_chars(
        exponent.data(), exponent.data() + exponent.size(), decimal_exponent);
    if (parsed.ec != std::errc{} || parsed.ptr != exponent.data() + exponent.size()) {
      return false;
    }
  }

  std::string digits;
  int decimal_index = 0;
  const std::size_t dot = mantissa.find('.');
  if (dot == std::string_view::npos) {
    digits.assign(mantissa);
    decimal_index = static_cast<int>(digits.size());
  } else {
    digits.assign(mantissa.substr(0, dot));
    digits.append(mantissa.substr(dot + 1));
    decimal_index = static_cast<int>(dot);
  }
  decimal_index += decimal_exponent;

  if (decimal_index <= 0) {
    out.integer = "0";
    out.fraction.assign(static_cast<std::size_t>(-decimal_index), '0');
    out.fraction += digits;
  } else if (decimal_index >= static_cast<int>(digits.size())) {
    out.integer = digits;
    out.integer.append(
        static_cast<std::size_t>(decimal_index - static_cast<int>(digits.size())), '0');
    out.fraction.clear();
  } else {
    out.integer = digits.substr(0, static_cast<std::size_t>(decimal_index));
    out.fraction = digits.substr(static_cast<std::size_t>(decimal_index));
  }

  const std::size_t first_integer_digit = out.integer.find_first_not_of('0');
  out.integer = first_integer_digit == std::string::npos
                    ? "0"
                    : out.integer.substr(first_integer_digit);
  while (!out.fraction.empty() && out.fraction.back() == '0') {
    out.fraction.pop_back();
  }
  return true;
}

bool parse_decimal(std::string_view text, double& value) {
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, std::chars_format::general);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool restore_reencoded(std::uint64_t reencoded_bits, int leading_decimal_zeros,
                       std::uint64_t& restored_bits) {
  const int biased_exponent = static_cast<int>((reencoded_bits >> 52) & 0x7ffu);
  if (biased_exponent == 0x7ff) {
    return false;
  }
  const int exponent = std::max(0, biased_exponent - kDoubleExponentBias);
  const int available_bits = kDoubleMantissaBits - exponent;
  if (available_bits <= 0) {
    return false;
  }

  const std::uint64_t field = reencoded_bits & low_mask(available_bits);
  if (field == 0) {
    return false;
  }
  const int trailing_zeros = std::countr_zero(field);
  const int fraction_bit_count = available_bits - trailing_zeros;
  const std::uint64_t reversed_fraction = field >> trailing_zeros;
  const std::uint64_t decimal_fraction =
      reverse_low_bits(reversed_fraction, fraction_bit_count);
  if (decimal_fraction == 0) {
    return false;
  }

  const double reencoded_value = std::bit_cast<double>(reencoded_bits);
  char integer_buffer[128];
  const double integer_value = std::trunc(std::fabs(reencoded_value));
  const auto integer_result = std::to_chars(
      integer_buffer, integer_buffer + sizeof(integer_buffer), integer_value,
      std::chars_format::fixed, 0);
  if (integer_result.ec != std::errc{}) {
    return false;
  }

  std::string decimal;
  if ((reencoded_bits >> 63) != 0) {
    decimal.push_back('-');
  }
  decimal.append(integer_buffer, integer_result.ptr);
  decimal.push_back('.');
  decimal.append(static_cast<std::size_t>(leading_decimal_zeros), '0');
  decimal += std::to_string(decimal_fraction);

  double restored = 0.0;
  if (!parse_decimal(decimal, restored)) {
    return false;
  }
  restored_bits = std::bit_cast<std::uint64_t>(restored);
  return true;
}

bool try_reencode(double value, ReencodedValue& out) {
  if (!std::isfinite(value) || value == 0.0) {
    return false;
  }
  DecimalParts decimal;
  if (!shortest_decimal_parts(value, decimal) || decimal.fraction.empty() ||
      decimal.fraction.size() > kMaxDecimalPlaces) {
    return false;
  }

  const std::size_t first_nonzero = decimal.fraction.find_first_not_of('0');
  if (first_nonzero == std::string::npos || first_nonzero > 15) {
    return false;
  }
  std::uint64_t decimal_fraction = 0;
  for (const char digit : decimal.fraction.substr(first_nonzero)) {
    decimal_fraction = decimal_fraction * 10 + static_cast<unsigned>(digit - '0');
  }
  if (decimal_fraction == 0) {
    return false;
  }

  const std::uint64_t original_bits = std::bit_cast<std::uint64_t>(value);
  const int biased_exponent = static_cast<int>((original_bits >> 52) & 0x7ffu);
  const int exponent = std::max(0, biased_exponent - kDoubleExponentBias);
  const int available_bits = kDoubleMantissaBits - exponent;
  const int fraction_bit_count = std::bit_width(decimal_fraction);
  if (available_bits <= 0 || fraction_bit_count > available_bits) {
    return false;
  }

  const std::uint64_t reversed = reverse_low_bits(decimal_fraction, fraction_bit_count);
  const std::uint64_t field = reversed << (available_bits - fraction_bit_count);
  const std::uint64_t mask = low_mask(available_bits);
  const std::uint64_t reencoded_bits = (original_bits & ~mask) | field;
  std::uint64_t restored_bits = 0;
  if (!restore_reencoded(reencoded_bits, static_cast<int>(first_nonzero), restored_bits) ||
      restored_bits != original_bits) {
    return false;
  }
  out = ReencodedValue{reencoded_bits, static_cast<int>(first_nonzero)};
  return true;
}

void write_xor(BitWriter& writer, std::uint64_t xor_value, int& stored_leading,
               int& stored_trailing, bool pack_zero_regions) {
  if (xor_value == 0) {
    writer.write_bit(false);
    return;
  }
  writer.write_bit(true);
  if (!pack_zero_regions) {
    writer.write_bits(xor_value, 64);
    return;
  }
  const int leading = std::countl_zero(xor_value);
  const int trailing = std::countr_zero(xor_value);
  if (stored_leading >= 0 && leading >= stored_leading && trailing >= stored_trailing) {
    writer.write_bit(false);
    const int significant = 64 - stored_leading - stored_trailing;
    writer.write_bits(xor_value >> stored_trailing, significant);
    return;
  }
  writer.write_bit(true);
  const int significant = 64 - leading - trailing;
  writer.write_bits(static_cast<std::uint64_t>(leading), 6);
  writer.write_bits(static_cast<std::uint64_t>(significant == 64 ? 0 : significant), 6);
  writer.write_bits(xor_value >> trailing, significant);
  stored_leading = leading;
  stored_trailing = trailing;
}

bool read_xor(BitReader& reader, std::uint64_t& stored_value, int& stored_leading,
              int& stored_trailing, bool pack_zero_regions) {
  if (!reader.read_bit()) {
    return reader.ok;
  }
  if (!pack_zero_regions) {
    stored_value ^= reader.read_bits(64);
    return reader.ok;
  }
  const bool new_window = reader.read_bit();
  if (!reader.ok) {
    return false;
  }
  if (new_window) {
    stored_leading = static_cast<int>(reader.read_bits(6));
    int significant = static_cast<int>(reader.read_bits(6));
    if (!reader.ok) {
      return false;
    }
    if (significant == 0) {
      significant = 64;
    }
    stored_trailing = 64 - stored_leading - significant;
    if (stored_trailing < 0) {
      reader.ok = false;
      return false;
    }
  } else if (stored_leading < 0) {
    reader.ok = false;
    return false;
  }
  const int significant = 64 - stored_leading - stored_trailing;
  const std::uint64_t xor_value = reader.read_bits(significant) << stored_trailing;
  if (!reader.ok) {
    return false;
  }
  stored_value ^= xor_value;
  return true;
}

}  // namespace

std::vector<std::uint8_t> encode_ablation(std::span<const double> values,
                                          std::string_view omitted_operator) {
  const Ablation ablation = parse_ablation(omitted_operator);
  BitWriter writer;
  writer.bytes.reserve(8 + values.size() * 3);
  append_count(writer.bytes, static_cast<std::uint64_t>(values.size()));
  if (values.empty()) {
    return writer.bytes;
  }

  std::uint64_t stored_value = std::bit_cast<std::uint64_t>(values.front());
  writer.write_bits(stored_value, 64);
  int stored_leading = -1;
  int stored_trailing = 0;
  for (std::size_t i = 1; i < values.size(); ++i) {
    const std::uint64_t original_bits = std::bit_cast<std::uint64_t>(values[i]);
    ReencodedValue reencoded;
    if (ablation != Ablation::kEraser && try_reencode(values[i], reencoded)) {
      if (reencoded.leading_decimal_zeros == 0) {
        writer.write_bit(true);
      } else {
        writer.write_bits(0, 2);
        writer.write_bits(static_cast<std::uint64_t>(reencoded.leading_decimal_zeros), 4);
      }
      if (ablation == Ablation::kXor) {
        writer.write_bits(reencoded.bits, 64);
      } else {
        write_xor(writer, stored_value ^ reencoded.bits, stored_leading, stored_trailing,
                  ablation != Ablation::kTrailingZeroPack);
      }
      stored_value = reencoded.bits;
    } else {
      writer.write_bits(1, 2);
      if (ablation == Ablation::kXor) {
        writer.write_bits(original_bits, 64);
      } else {
        write_xor(writer, stored_value ^ original_bits, stored_leading, stored_trailing,
                  ablation != Ablation::kTrailingZeroPack);
      }
      stored_value = original_bits;
    }
  }
  return writer.bytes;
}

bool decode_ablation(std::span<const std::uint8_t> encoded,
                     std::string_view omitted_operator,
                     std::vector<double>& out) {
  const Ablation ablation = parse_ablation(omitted_operator);
  out.clear();
  std::uint64_t count = 0;
  if (!read_count(encoded, count) || count > encoded.size() * 8 + 1) {
    return false;
  }
  if (count == 0) {
    return encoded.size() == 8;
  }

  BitReader reader{encoded.subspan(8)};
  std::uint64_t stored_value = reader.read_bits(64);
  if (!reader.ok) {
    return false;
  }
  out.reserve(static_cast<std::size_t>(count));
  out.push_back(std::bit_cast<double>(stored_value));
  int stored_leading = -1;
  int stored_trailing = 0;
  for (std::uint64_t i = 1; i < count; ++i) {
    bool reencoded = false;
    int leading_decimal_zeros = 0;
    if (reader.read_bit()) {
      reencoded = true;
    } else if (reader.read_bit()) {
      reencoded = false;
    } else {
      reencoded = true;
      leading_decimal_zeros = static_cast<int>(reader.read_bits(4));
    }
    if (!reader.ok) {
      return false;
    }
    if (ablation == Ablation::kXor) {
      stored_value = reader.read_bits(64);
    } else if (!read_xor(reader, stored_value, stored_leading, stored_trailing,
                         ablation != Ablation::kTrailingZeroPack)) {
      return false;
    }
    if (!reader.ok) return false;

    std::uint64_t original_bits = stored_value;
    if (reencoded &&
        !restore_reencoded(stored_value, leading_decimal_zeros, original_bits)) {
      return false;
    }
    out.push_back(std::bit_cast<double>(original_bits));
  }
  return reader.ok;
}

std::vector<std::uint8_t> encode(std::span<const double> values) {
  return encode_ablation(values, {});
}

bool decode(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  return decode_ablation(encoded, {}, out);
}

}  // namespace pongo_codec

namespace exact_actions::pongo {
namespace {

constexpr std::array<std::uint8_t, 4> kEraserMagic{'P', 'E', 'R', '1'};
constexpr std::array<std::uint8_t, 4> kXorMagic{'P', 'X', 'R', '1'};
constexpr std::uint8_t kReencoded = 0x80;

struct Record {
  bool reencoded = false;
  std::uint8_t leading_decimal_zeros = 0;
  std::uint64_t value = 0;
};

struct Packet {
  std::uint64_t count = 0;
  std::uint64_t first = 0;
  std::vector<Record> records;
};

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (input.size() - pos < 8) {
    throw std::runtime_error("truncated Pongo stage packet");
  }
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(input[pos++]) << shift;
  }
  return value;
}

std::uint8_t metadata(const Record& record) {
  if (!record.reencoded) {
    return 0;
  }
  if (record.leading_decimal_zeros > 15) {
    throw std::runtime_error("invalid Pongo decimal-zero metadata");
  }
  return static_cast<std::uint8_t>(kReencoded | record.leading_decimal_zeros);
}

Record decode_record(std::uint8_t encoded, std::uint64_t value) {
  if ((encoded & kReencoded) == 0) {
    if (encoded != 0) {
      throw std::runtime_error("invalid Pongo literal metadata");
    }
    return {false, 0, value};
  }
  if ((encoded & 0x70u) != 0) {
    throw std::runtime_error("invalid Pongo re-encoding metadata");
  }
  return {true, static_cast<std::uint8_t>(encoded & 0x0fu), value};
}

std::vector<std::uint8_t> serialize(
    const std::array<std::uint8_t, 4>& magic, const Packet& packet) {
  if (packet.count == 0 && !packet.records.empty()) {
    throw std::runtime_error("empty Pongo packet contains records");
  }
  if (packet.count > 0 && packet.records.size() != packet.count - 1) {
    throw std::runtime_error("Pongo packet record count mismatch");
  }
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  put_u64(out, packet.count);
  if (packet.count == 0) {
    return out;
  }
  put_u64(out, packet.first);
  for (const auto& record : packet.records) {
    out.push_back(metadata(record));
    put_u64(out, record.value);
  }
  return out;
}

Packet parse(const std::array<std::uint8_t, 4>& magic,
             std::span<const std::uint8_t> input) {
  if (input.size() < 12 ||
      !std::equal(magic.begin(), magic.end(), input.begin())) {
    throw std::runtime_error("wrong Pongo stage packet type");
  }
  std::size_t pos = 4;
  Packet packet;
  packet.count = get_u64(input, pos);
  if (packet.count == 0) {
    if (input.size() != pos) {
      throw std::runtime_error("trailing bytes in empty Pongo stage packet");
    }
    return packet;
  }
  if (input.size() < 20) {
    throw std::runtime_error("truncated Pongo first value");
  }
  const std::uint64_t records = packet.count - 1;
  if (records > (std::numeric_limits<std::size_t>::max() - 20) / 9 ||
      input.size() != 20 + static_cast<std::size_t>(records) * 9) {
    throw std::runtime_error("Pongo stage packet size mismatch");
  }
  packet.first = get_u64(input, pos);
  packet.records.reserve(static_cast<std::size_t>(records));
  for (std::uint64_t index = 0; index < records; ++index) {
    const std::uint8_t encoded_metadata = input[pos++];
    packet.records.push_back(
        decode_record(encoded_metadata, get_u64(input, pos)));
  }
  return packet;
}

void require_zero_padding(const pongo_codec::BitReader& reader) {
  const std::size_t bit_pos = reader.byte_index * 8 + reader.used;
  const std::size_t total_bits = reader.bytes.size() * 8;
  if (total_bits - bit_pos >= 8) {
    throw std::runtime_error("trailing bytes in Pongo public wire");
  }
  for (std::size_t pos = bit_pos; pos < total_bits; ++pos) {
    if (((reader.bytes[pos / 8] >> (7 - (pos % 8))) & 1u) != 0) {
      throw std::runtime_error("non-zero Pongo padding bits");
    }
  }
}

}  // namespace

std::vector<std::uint8_t> eraser_forward(
    std::span<const std::uint8_t> raw_lanes) {
  if (raw_lanes.size() % 8 != 0) {
    throw std::runtime_error("Pongo eraser requires 64-bit lanes");
  }
  Packet packet;
  packet.count = raw_lanes.size() / 8;
  if (packet.count == 0) {
    return serialize(kEraserMagic, packet);
  }
  std::size_t pos = 0;
  packet.first = get_u64(raw_lanes, pos);
  packet.records.reserve(static_cast<std::size_t>(packet.count - 1));
  for (std::uint64_t index = 1; index < packet.count; ++index) {
    const std::uint64_t original = get_u64(raw_lanes, pos);
    pongo_codec::ReencodedValue reencoded;
    if (pongo_codec::try_reencode(std::bit_cast<double>(original), reencoded)) {
      packet.records.push_back(
          {true, static_cast<std::uint8_t>(reencoded.leading_decimal_zeros),
           reencoded.bits});
    } else {
      packet.records.push_back({false, 0, original});
    }
  }
  return serialize(kEraserMagic, packet);
}

std::vector<std::uint8_t> eraser_inverse(
    std::span<const std::uint8_t> encoded) {
  const Packet packet = parse(kEraserMagic, encoded);
  std::vector<std::uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(packet.count) * 8);
  if (packet.count == 0) {
    return raw;
  }
  put_u64(raw, packet.first);
  for (const auto& record : packet.records) {
    std::uint64_t original = record.value;
    if (record.reencoded &&
        !pongo_codec::restore_reencoded(
            record.value, record.leading_decimal_zeros, original)) {
      throw std::runtime_error("invalid Pongo decimal restoration");
    }
    put_u64(raw, original);
  }
  return raw;
}

std::vector<std::uint8_t> xor_forward(
    std::span<const std::uint8_t> eraser_packet) {
  Packet packet = parse(kEraserMagic, eraser_packet);
  std::uint64_t previous = packet.first;
  for (auto& record : packet.records) {
    const std::uint64_t target = record.value;
    record.value = previous ^ target;
    previous = target;
  }
  return serialize(kXorMagic, packet);
}

std::vector<std::uint8_t> xor_inverse(
    std::span<const std::uint8_t> xor_packet) {
  Packet packet = parse(kXorMagic, xor_packet);
  std::uint64_t previous = packet.first;
  for (auto& record : packet.records) {
    record.value ^= previous;
    previous = record.value;
  }
  return serialize(kEraserMagic, packet);
}

std::vector<std::uint8_t> trailing_pack_forward(
    std::span<const std::uint8_t> xor_packet) {
  const Packet packet = parse(kXorMagic, xor_packet);
  pongo_codec::BitWriter writer;
  writer.bytes.reserve(8 + static_cast<std::size_t>(packet.count) * 3);
  pongo_codec::append_count(writer.bytes, packet.count);
  if (packet.count == 0) {
    return writer.bytes;
  }
  writer.write_bits(packet.first, 64);
  int stored_leading = -1;
  int stored_trailing = 0;
  for (const auto& record : packet.records) {
    if (!record.reencoded) {
      writer.write_bits(1, 2);
    } else if (record.leading_decimal_zeros == 0) {
      writer.write_bit(true);
    } else {
      writer.write_bits(0, 2);
      writer.write_bits(record.leading_decimal_zeros, 4);
    }
    pongo_codec::write_xor(
        writer, record.value, stored_leading, stored_trailing, true);
  }
  return writer.bytes;
}

std::vector<std::uint8_t> trailing_pack_inverse(
    std::span<const std::uint8_t> public_wire) {
  std::uint64_t count = 0;
  if (!pongo_codec::read_count(public_wire, count) ||
      count > public_wire.size() * 8 + 1) {
    throw std::runtime_error("invalid Pongo public header");
  }
  if (count == 0) {
    if (public_wire.size() != 8) {
      throw std::runtime_error("trailing bytes in empty Pongo public wire");
    }
    return serialize(kXorMagic, Packet{});
  }

  pongo_codec::BitReader reader{public_wire.subspan(8)};
  Packet packet;
  packet.count = count;
  packet.first = reader.read_bits(64);
  if (!reader.ok) {
    throw std::runtime_error("truncated Pongo first value");
  }
  packet.records.reserve(static_cast<std::size_t>(count - 1));
  std::uint64_t stored_value = packet.first;
  int stored_leading = -1;
  int stored_trailing = 0;
  for (std::uint64_t index = 1; index < count; ++index) {
    Record record;
    if (reader.read_bit()) {
      record.reencoded = true;
    } else if (reader.read_bit()) {
      record.reencoded = false;
    } else {
      record.reencoded = true;
      record.leading_decimal_zeros =
          static_cast<std::uint8_t>(reader.read_bits(4));
    }
    if (!reader.ok) {
      throw std::runtime_error("truncated Pongo eraser metadata");
    }
    const std::uint64_t previous = stored_value;
    if (!pongo_codec::read_xor(
            reader, stored_value, stored_leading, stored_trailing, true)) {
      throw std::runtime_error("truncated Pongo XOR payload");
    }
    record.value = previous ^ stored_value;
    packet.records.push_back(record);
  }
  require_zero_padding(reader);
  return serialize(kXorMagic, packet);
}

std::vector<std::uint8_t> compose(
    std::span<const std::uint8_t> raw_lanes) {
  return trailing_pack_forward(xor_forward(eraser_forward(raw_lanes)));
}

std::vector<std::uint8_t> decompose(
    std::span<const std::uint8_t> public_wire) {
  return eraser_inverse(xor_inverse(trailing_pack_inverse(public_wire)));
}

}  // namespace exact_actions::pongo
