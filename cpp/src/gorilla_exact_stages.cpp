#include "gorilla_exact_stages.hpp"
#include "adjacent_residual.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::gorilla {
namespace {

constexpr std::array<std::uint8_t, 4> kIntMagic{'G', 'X', 'I', '1'};
constexpr std::array<std::uint8_t, 4> kFloatMagic{'G', 'X', 'F', '1'};
constexpr std::uint64_t kIntEnd = 0x8000000000000000ull;
constexpr std::uint64_t kFloatEnd = 0x7ff8000000000000ull;

struct Packet { bool floating; std::vector<std::int64_t> values; };

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint64_t read_u64_le(std::span<const std::uint8_t> data, std::size_t& pos) {
  if (data.size() - pos < 8) throw std::runtime_error("truncated Gorilla XOR packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

std::vector<std::uint8_t> serialize(bool floating,
                                    std::span<const std::int64_t> values) {
  const auto& magic = floating ? kFloatMagic : kIntMagic;
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  append_u64_le(out, values.size());
  for (const auto value : values) append_u64_le(out, static_cast<std::uint64_t>(value));
  return out;
}

Packet parse(std::span<const std::uint8_t> packet) {
  if (packet.size() < 12) throw std::runtime_error("truncated Gorilla XOR packet");
  bool floating = false;
  if (std::equal(kIntMagic.begin(), kIntMagic.end(), packet.begin())) floating = false;
  else if (std::equal(kFloatMagic.begin(), kFloatMagic.end(), packet.begin())) floating = true;
  else throw std::runtime_error("wrong Gorilla XOR packet type");
  std::size_t pos = 4;
  const auto count = read_u64_le(packet, pos);
  if (count > (packet.size() - pos) / 8 || count * 8 != packet.size() - pos)
    throw std::runtime_error("invalid Gorilla XOR packet count");
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i)
    values.push_back(static_cast<std::int64_t>(read_u64_le(packet, pos)));
  return {floating, std::move(values)};
}

std::vector<std::uint8_t> xor_forward(bool floating,
                                      std::span<const std::int64_t> values) {
  std::vector<std::int64_t> residuals(values.begin(), values.end());
  for (std::size_t i = values.size(); i-- > 1;)
    residuals[i] = static_cast<std::int64_t>(adjacent_residual::xor_word(
        static_cast<std::uint64_t>(values[i]), static_cast<std::uint64_t>(values[i - 1])));
  return serialize(floating, residuals);
}

struct BitWriter {
  int stored_leading = std::numeric_limits<int>::max();
  int stored_trailing = 0;
  std::uint8_t buffer = 0;
  int bits_left = 8;
  std::vector<std::uint8_t> out;

  void flush_byte() {
    if (bits_left == 0) { out.push_back(buffer); buffer = 0; bits_left = 8; }
  }
  void write_bit(bool value) {
    if (value) buffer |= static_cast<std::uint8_t>(1u << (bits_left - 1));
    --bits_left;
    flush_byte();
  }
  void write_bits(std::uint64_t value, int bits) {
    while (bits > 0) {
      const int take = std::min(bits, bits_left);
      const int source_shift = bits - take;
      const std::uint64_t mask = take == 64 ? ~std::uint64_t{0}
                                             : ((std::uint64_t{1} << take) - 1);
      buffer |= static_cast<std::uint8_t>(((value >> source_shift) & mask)
                                         << (bits_left - take));
      bits -= take;
      bits_left -= take;
      flush_byte();
    }
  }
  void finish() { bits_left = 0; flush_byte(); }

  void write_xor(std::uint64_t value) {
    if (value == 0) { write_bit(false); return; }
    write_bit(true);
    const int leading = __builtin_clzll(value);
    const int trailing = __builtin_ctzll(value);
    if (leading >= stored_leading && trailing >= stored_trailing) {
      write_bit(false);
      write_bits(value >> stored_trailing,
                 64 - stored_leading - stored_trailing);
    } else {
      write_bit(true);
      const int significant = 64 - leading - trailing;
      write_bits(static_cast<std::uint64_t>(leading), 6);
      write_bits(static_cast<std::uint64_t>(significant - 1), 6);
      write_bits(value >> trailing, significant);
      stored_leading = leading;
      stored_trailing = trailing;
    }
  }
};

struct BitReader {
  std::span<const std::uint8_t> wire;
  std::size_t bit_pos{};

  std::uint64_t read_bits(int count) {
    if (count < 0 || count > 64 ||
        static_cast<std::size_t>(count) > wire.size() * 8 - bit_pos)
      throw std::runtime_error("truncated Gorilla bitstream");
    std::uint64_t value = 0;
    for (int i = 0; i < count; ++i) {
      value = (value << 1) |
              ((wire[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1u);
      ++bit_pos;
    }
    return value;
  }

  bool read_bit() { return read_bits(1) != 0; }

  void require_zero_padding() const {
    for (std::size_t pos = bit_pos; pos < wire.size() * 8; ++pos) {
      if (((wire[pos / 8] >> (7 - (pos % 8))) & 1u) != 0)
        throw std::runtime_error("non-zero Gorilla trailing bits");
    }
  }
};

std::vector<std::uint8_t> inverse_wire(std::span<const std::uint8_t> wire,
                                       bool floating) {
  BitReader reader{wire};
  const auto end = floating ? kFloatEnd : kIntEnd;
  const auto first = reader.read_bits(64);
  if (first == end) {
    reader.require_zero_padding();
    return serialize(floating, {});
  }

  std::vector<std::int64_t> residuals{static_cast<std::int64_t>(first)};
  std::uint64_t current = first;
  int stored_leading = std::numeric_limits<int>::max();
  int stored_trailing = 0;
  while (true) {
    std::uint64_t xor_value = 0;
    if (reader.read_bit()) {
      if (reader.read_bit()) {
        stored_leading = static_cast<int>(reader.read_bits(6));
        const int significant = static_cast<int>(reader.read_bits(6)) + 1;
        if (stored_leading + significant > 64)
          throw std::runtime_error("invalid Gorilla significant width");
        stored_trailing = 64 - stored_leading - significant;
        xor_value = reader.read_bits(significant) << stored_trailing;
      } else {
        if (stored_leading == std::numeric_limits<int>::max())
          throw std::runtime_error("Gorilla reused an unset leading-zero window");
        const int significant = 64 - stored_leading - stored_trailing;
        xor_value = reader.read_bits(significant) << stored_trailing;
      }
    }
    const auto next = current ^ xor_value;
    if (next == end) {
      reader.require_zero_padding();
      break;
    }
    residuals.push_back(static_cast<std::int64_t>(xor_value));
    current = next;
  }
  return serialize(floating, residuals);
}

}  // namespace

std::vector<std::uint8_t> xor_forward_int64(std::span<const std::int64_t> values) {
  return xor_forward(false, values);
}

std::vector<std::uint8_t> xor_forward_double_bits(std::span<const std::int64_t> values) {
  return xor_forward(true, values);
}

std::vector<std::int64_t> xor_inverse(std::span<const std::uint8_t> packet) {
  auto parsed = parse(packet);
  for (std::size_t i = 1; i < parsed.values.size(); ++i)
    parsed.values[i] ^= parsed.values[i - 1];
  return parsed.values;
}

std::vector<std::uint8_t> trailing_pack_forward(std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet);
  BitWriter writer;
  const std::uint64_t end = parsed.floating ? kFloatEnd : kIntEnd;
  if (parsed.values.empty()) {
    writer.write_bits(end, 64);
  } else {
    writer.write_bits(static_cast<std::uint64_t>(parsed.values.front()), 64);
    for (std::size_t i = 1; i < parsed.values.size(); ++i)
      writer.write_xor(static_cast<std::uint64_t>(parsed.values[i]));
    std::uint64_t last = static_cast<std::uint64_t>(parsed.values.front());
    for (std::size_t i = 1; i < parsed.values.size(); ++i)
      last ^= static_cast<std::uint64_t>(parsed.values[i]);
    writer.write_xor(last ^ end);
  }
  writer.finish();
  return writer.out;
}

std::vector<std::uint8_t> trailing_pack_inverse_int64(
    std::span<const std::uint8_t> wire) { return inverse_wire(wire, false); }

std::vector<std::uint8_t> trailing_pack_inverse_double(
    std::span<const std::uint8_t> wire) { return inverse_wire(wire, true); }

std::vector<std::uint8_t> compose_int64(std::span<const std::int64_t> values) {
  return trailing_pack_forward(xor_forward_int64(values));
}

std::vector<std::uint8_t> compose_double_bits(std::span<const std::int64_t> values) {
  return trailing_pack_forward(xor_forward_double_bits(values));
}

std::vector<std::int64_t> decompose_int64(std::span<const std::uint8_t> wire) {
  return xor_inverse(trailing_pack_inverse_int64(wire));
}

std::vector<std::int64_t> decompose_double_bits(std::span<const std::uint8_t> wire) {
  return xor_inverse(trailing_pack_inverse_double(wire));
}

}  // namespace exact_actions::gorilla
