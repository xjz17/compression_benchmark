#pragma once

#include "max_value_bitpack.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace exact_actions::exception_separation {

using Bytes = std::vector<std::uint8_t>;
using Words = std::vector<std::uint32_t>;

// Width is supplied by the caller. This stage does not select a FastPFor width.
struct Parts {
  unsigned width = 0;
  Words low;
  Words positions;
  Words high;
};

inline void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

inline std::uint32_t low_mask(unsigned width) {
  require(width <= 32, "exception separation width must be in [0,32]");
  return width == 32 ? UINT32_MAX : (std::uint32_t{1} << width) - 1;
}

inline void validate(const Parts& parts) {
  const auto mask = low_mask(parts.width);
  require(parts.low.size() <= UINT32_MAX, "exception separation count exceeds uint32");
  require(parts.positions.size() == parts.high.size(), "exception stream lengths differ");
  require(parts.positions.size() <= parts.low.size(), "too many exceptions");
  for (const auto value : parts.low)
    require(value <= mask, "low stream value exceeds supplied width");
  const auto maximum_high = parts.width == 32 ? 0 : UINT32_MAX >> parts.width;
  for (std::size_t i = 0; i < parts.positions.size(); ++i) {
    require(parts.positions[i] < parts.low.size(), "exception position outside input");
    require(i == 0 || parts.positions[i - 1] < parts.positions[i],
            "exception positions must be strictly increasing");
    require(parts.high[i] != 0 && parts.high[i] <= maximum_high,
            "exception high value is zero or exceeds remaining width");
  }
}

inline Parts separate(std::span<const std::uint32_t> values, unsigned width) {
  const auto mask = low_mask(width);
  require(values.size() <= UINT32_MAX, "exception separation count exceeds uint32");
  Parts parts{width, {}, {}, {}};
  parts.low.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    parts.low.push_back(values[i] & mask);
    const auto high = width == 32 ? 0 : values[i] >> width;
    if (high != 0) {
      parts.positions.push_back(static_cast<std::uint32_t>(i));
      parts.high.push_back(high);
    }
  }
  return parts;
}

inline Words restore(const Parts& parts) {
  validate(parts);
  auto values = parts.low;
  for (std::size_t i = 0; i < parts.positions.size(); ++i)
    values[parts.positions[i]] |= parts.high[i] << parts.width;
  return values;
}

inline void put_u32(Bytes& out, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}

inline std::uint32_t get_u32(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  require(pos <= bytes.size() && bytes.size() - pos >= 4, "truncated uint32 word");
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t{bytes[pos++]} << (i * 8);
  return value;
}

inline Words words(std::span<const std::uint8_t> raw) {
  require(raw.size() % 4 == 0, "exception separation requires complete uint32 LE words");
  require(raw.size() / 4 <= UINT32_MAX, "exception separation count exceeds uint32");
  Words values;
  values.reserve(raw.size() / 4);
  for (std::size_t pos = 0; pos < raw.size();) values.push_back(get_u32(raw, pos));
  return values;
}

inline Bytes raw_bytes(std::span<const std::uint32_t> values) {
  require(values.size() <= Bytes{}.max_size() / 4, "uint32 byte count overflow");
  Bytes raw;
  raw.reserve(values.size() * 4);
  for (const auto value : values) put_u32(raw, value);
  return raw;
}

// EHS1: magic, b:u8, N:u32, E:u32, low[N], positions[E], high[E], all LE.
inline Bytes serialize(const Parts& parts) {
  validate(parts);
  const std::uint64_t size = 13 + 4 * std::uint64_t{parts.low.size()} +
                             8 * std::uint64_t{parts.positions.size()};
  require(size <= Bytes{}.max_size(), "exception packet byte count overflow");
  Bytes out{'E', 'H', 'S', '1'};
  out.reserve(static_cast<std::size_t>(size));
  out.push_back(static_cast<std::uint8_t>(parts.width));
  put_u32(out, static_cast<std::uint32_t>(parts.low.size()));
  put_u32(out, static_cast<std::uint32_t>(parts.positions.size()));
  for (const auto* stream : {&parts.low, &parts.positions, &parts.high})
    for (const auto value : *stream) put_u32(out, value);
  return out;
}

inline Parts parse(std::span<const std::uint8_t> wire) {
  require(wire.size() >= 13 && std::equal(wire.begin(), wire.begin() + 4, "EHS1"),
          "invalid exception separation packet");
  Parts parts{wire[4], {}, {}, {}};
  low_mask(parts.width);
  std::size_t pos = 5;
  const auto count = get_u32(wire, pos), exceptions = get_u32(wire, pos);
  require(exceptions <= count &&
              std::uint64_t{count} * 4 + std::uint64_t{exceptions} * 8 == wire.size() - pos,
          "exception packet stream lengths do not match header");
  parts.low.reserve(count);
  parts.positions.reserve(exceptions);
  parts.high.reserve(exceptions);
  for (std::uint32_t i = 0; i < count; ++i) parts.low.push_back(get_u32(wire, pos));
  for (std::uint32_t i = 0; i < exceptions; ++i) parts.positions.push_back(get_u32(wire, pos));
  for (std::uint32_t i = 0; i < exceptions; ++i) parts.high.push_back(get_u32(wire, pos));
  validate(parts);
  return parts;
}

inline Bytes forward(std::span<const std::uint8_t> raw, unsigned width) {
  return serialize(separate(words(raw), width));
}

inline Bytes inverse(std::span<const std::uint8_t> wire) {
  return raw_bytes(restore(parse(wire)));
}

// EHP1 is an explicit A111 -> A020 recipe, not the upstream FastPFor wire.
// Its three payloads use A020 MSB byte packing; b is never reselected.
inline Bytes pack(const Parts& parts) {
  validate(parts);
  namespace bp = max_value_bitpack;
  const std::vector<std::uint64_t> lows(parts.low.begin(), parts.low.end());
  const std::vector<std::uint64_t> positions(parts.positions.begin(), parts.positions.end());
  const std::vector<std::uint64_t> highs(parts.high.begin(), parts.high.end());
  const auto low = bp::pack_fixed(lows, parts.width);
  const auto pos = bp::pack(positions);
  const auto high = bp::pack(highs);
  Bytes out{'E', 'H', 'P', '1', static_cast<std::uint8_t>(parts.width)};
  put_u32(out, static_cast<std::uint32_t>(parts.low.size()));
  put_u32(out, static_cast<std::uint32_t>(parts.positions.size()));
  out.push_back(pos.width);
  out.push_back(high.width);
  for (const auto* payload : {&low.payload, &pos.payload, &high.payload})
    out.insert(out.end(), payload->begin(), payload->end());
  return out;
}

inline Parts unpack(std::span<const std::uint8_t> wire) {
  require(wire.size() >= 15 && std::equal(wire.begin(), wire.begin() + 4, "EHP1"),
          "invalid packed exception recipe");
  Parts parts{wire[4], {}, {}, {}};
  low_mask(parts.width);
  std::size_t pos = 5;
  const auto count = get_u32(wire, pos), exceptions = get_u32(wire, pos);
  const unsigned pos_width = wire[pos++], high_width = wire[pos++];
  require(exceptions <= count && pos_width <= 32 && high_width <= 32,
          "invalid packed exception stream shape");
  require(exceptions == 0 || parts.width < 32,
          "full-width low stream cannot have exceptions");
  require(high_width <= 32 - parts.width,
          "packed exception high width exceeds remaining width");
  namespace bp = max_value_bitpack;
  const auto low_size = bp::payload_size(count, parts.width);
  const auto pos_size = bp::payload_size(exceptions, pos_width);
  const auto high_size = bp::payload_size(exceptions, high_width);
  require(std::uint64_t{low_size} + pos_size + high_size == wire.size() - pos,
          "packed exception payload lengths do not match header");
  // Decode only after checking all extents. No current input bytes are cached.
  const auto read_stream = [&](Words& output, std::size_t count_, unsigned width,
                               std::size_t size, bool minimal) {
    const auto decoded = bp::unpack(wire.subspan(pos, size), count_, width);
    pos += size;
    require(!minimal || bp::maximum_width(decoded) == width,
            "noncanonical packed exception width");
    output.assign(decoded.begin(), decoded.end());
  };
  read_stream(parts.low, count, parts.width, low_size, false);
  read_stream(parts.positions, exceptions, pos_width, pos_size, true);
  read_stream(parts.high, exceptions, high_width, high_size, true);
  validate(parts);
  return parts;
}

}  // namespace exact_actions::exception_separation
