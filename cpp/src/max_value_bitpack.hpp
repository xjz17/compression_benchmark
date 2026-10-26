#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

// One unsigned maximum-value-width packing algorithm. Codec headers, block
// boundaries, predictors, FOR bases and exception streams belong to adapters.
namespace exact_actions::max_value_bitpack {

enum class BitOrder { msb_first, lsb_first };

struct Format {
  BitOrder order = BitOrder::msb_first;
  // An I/O choice: tightly packed bytes or zero-padded little-endian words.
  std::size_t alignment_bytes = 1;
};

struct PackedBlock {
  std::uint8_t width = 0;
  std::vector<std::uint8_t> payload;
};

inline std::uint8_t maximum_width(std::span<const std::uint64_t> values) {
  std::uint64_t maximum = 0;
  for (const auto value : values) maximum = std::max(maximum, value);
  return static_cast<std::uint8_t>(std::bit_width(maximum));
}

inline std::size_t bit_count(std::size_t count, unsigned width) {
  if (width > 64) throw std::invalid_argument("bitpack width must be in [0,64]");
  if (width != 0 && count > std::numeric_limits<std::size_t>::max() / width)
    throw std::overflow_error("bitpack bit count overflow");
  return count * width;
}

inline void validate_format(Format format) {
  if ((format.order != BitOrder::msb_first && format.order != BitOrder::lsb_first) ||
      (format.alignment_bytes != 1 && format.alignment_bytes != 8))
    throw std::invalid_argument("invalid bitpack I/O format");
}

inline std::size_t payload_size(std::size_t count, unsigned width,
                                Format format = {}) {
  validate_format(format);
  const auto bits = bit_count(count, width);
  const auto bytes = bits / 8 + (bits % 8 != 0);
  const auto remainder = bytes % format.alignment_bytes;
  const auto padding = remainder == 0 ? 0 : format.alignment_bytes - remainder;
  if (bytes > std::numeric_limits<std::size_t>::max() - padding)
    throw std::overflow_error("bitpack padded byte count overflow");
  return bytes + padding;
}

inline std::uint8_t allowed_width(std::span<const std::uint64_t> values,
                                 unsigned minimum = 0, unsigned wide_cutoff = 64) {
  if (minimum > 64 || wide_cutoff > 64 || minimum > wide_cutoff)
    throw std::invalid_argument("invalid bitpack allowed-width profile");
  const auto width = std::max<unsigned>(minimum, maximum_width(values));
  return static_cast<std::uint8_t>(width > wide_cutoff ? 64 : width);
}

inline PackedBlock pack_fixed(std::span<const std::uint64_t> values,
                              unsigned width, Format format = {}) {
  PackedBlock out;
  if (width > 64 || maximum_width(values) > width)
    throw std::invalid_argument("bitpack value exceeds supplied width");
  out.width = static_cast<std::uint8_t>(width);
  out.payload.assign(payload_size(values.size(), out.width, format), 0);
  if (out.width == 0) return out;
  // Very narrow fields keep the minimal bit loop; byte fragments amortize
  // their bookkeeping for wider fields.  Width eight is a direct byte copy.
  if (width <= 4) {
    std::size_t bit = 0;
    for (const auto value : values) {
      for (unsigned i = 0; i < width; ++i, ++bit) {
        const auto from = format.order == BitOrder::msb_first ? width - 1 - i : i;
        const auto to = format.order == BitOrder::msb_first ? 7 - bit % 8 : bit % 8;
        out.payload[bit / 8] |= static_cast<std::uint8_t>(((value >> from) & 1U) << to);
      }
    }
    return out;
  }
  if (width == 8) {
    std::copy(values.begin(), values.end(), out.payload.begin());
    return out;
  }
  std::size_t position = 0;
  // Emit a byte fragment at a time.  The wire order and zero padding remain
  // identical to the scalar bit loop, including values spanning byte edges.
  if (format.order == BitOrder::msb_first) {
    for (const auto value : values) {
      unsigned remaining = out.width;
      while (remaining) {
        const unsigned free_bits = 8 - static_cast<unsigned>(position % 8);
        const unsigned take = std::min(remaining, free_bits);
        const unsigned shift = remaining - take;
        const auto fragment = static_cast<unsigned>((value >> shift) & ((1U << take) - 1U));
        out.payload[position / 8] |= static_cast<std::uint8_t>(fragment << (free_bits - take));
        position += take;
        remaining -= take;
      }
    }
  } else {
    for (const auto value : values) {
      unsigned consumed = 0;
      while (consumed < out.width) {
        const unsigned offset = static_cast<unsigned>(position % 8);
        const unsigned take = std::min<unsigned>(out.width - consumed, 8 - offset);
        const auto fragment = static_cast<unsigned>((value >> consumed) & ((1U << take) - 1U));
        out.payload[position / 8] |= static_cast<std::uint8_t>(fragment << offset);
        position += take;
        consumed += take;
      }
    }
  }
  return out;
}

inline PackedBlock pack(std::span<const std::uint64_t> values,
                        Format format = {}) {
  return pack_fixed(values, maximum_width(values), format);
}

inline std::vector<std::uint64_t> unpack(std::span<const std::uint8_t> payload,
                                        std::size_t count, unsigned width,
                                        Format format = {}) {
  if (payload.size() != payload_size(count, width, format))
    throw std::invalid_argument("bitpack payload byte count mismatch");
  if (count > std::vector<std::uint64_t>().max_size())
    throw std::overflow_error("bitpack decoded value count overflow");
  const auto bits = bit_count(count, width);
  const auto used_bytes = bits / 8 + (bits % 8 != 0);
  if (bits % 8 != 0) {
    const auto used = bits % 8;
    const auto padding_mask = format.order == BitOrder::msb_first
                                  ? (1U << (8 - used)) - 1U
                                  : (0xffU << used) & 0xffU;
    if ((payload[used_bytes - 1] & padding_mask) != 0)
      throw std::invalid_argument("bitpack nonzero trailing padding bits");
  }
  for (std::size_t i = used_bytes; i < payload.size(); ++i)
    if (payload[i] != 0)
      throw std::invalid_argument("bitpack nonzero alignment padding");
  std::vector<std::uint64_t> values(count, 0);
  if (width == 0) return values;
  if (width <= 4) {
    std::size_t bit = 0;
    for (auto& value : values) {
      for (unsigned i = 0; i < width; ++i, ++bit) {
        const auto from = format.order == BitOrder::msb_first ? 7 - bit % 8 : bit % 8;
        const auto to = format.order == BitOrder::msb_first ? width - 1 - i : i;
        value |= static_cast<std::uint64_t>((payload[bit / 8] >> from) & 1U) << to;
      }
    }
    return values;
  }
  if (width == 8) {
    std::copy(payload.begin(), payload.begin() + count, values.begin());
    return values;
  }
  std::size_t position = 0;
  if (format.order == BitOrder::msb_first) {
    for (auto& value : values) {
      unsigned remaining = width;
      while (remaining) {
        const unsigned free_bits = 8 - static_cast<unsigned>(position % 8);
        const unsigned take = std::min(remaining, free_bits);
        const auto fragment = (payload[position / 8] >> (free_bits - take)) & ((1U << take) - 1U);
        value = (value << take) | fragment;
        position += take;
        remaining -= take;
      }
    }
  } else {
    for (auto& value : values) {
      unsigned consumed = 0;
      while (consumed < width) {
        const unsigned offset = static_cast<unsigned>(position % 8);
        const unsigned take = std::min(width - consumed, 8 - offset);
        const auto fragment = (payload[position / 8] >> offset) & ((1U << take) - 1U);
        value |= static_cast<std::uint64_t>(fragment) << consumed;
        position += take;
        consumed += take;
      }
    }
  }
  return values;
}

// MVW1: magic[4], count:u64 little-endian, width:u8, MSB packed payload.
// No FOR or ZigZag is implicit: the input is exactly uint64 little-endian[].
inline std::vector<std::uint8_t> encode(std::span<const std::uint8_t> raw) {
  if (raw.size() % 8 != 0)
    throw std::invalid_argument("bitpack uint64 input must be 8-byte aligned");
  std::vector<std::uint64_t> values(raw.size() / 8, 0);
  for (std::size_t i = 0; i < raw.size(); ++i)
    values[i / 8] |= static_cast<std::uint64_t>(raw[i]) << ((i % 8) * 8);
  auto packed = pack(values);
  std::vector<std::uint8_t> out{'M', 'V', 'W', '1'};
  const auto count = static_cast<std::uint64_t>(values.size());
  for (unsigned shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(count >> shift));
  out.push_back(packed.width);
  out.insert(out.end(), packed.payload.begin(), packed.payload.end());
  return out;
}

inline std::vector<std::uint8_t> decode(std::span<const std::uint8_t> wire) {
  if (wire.size() < 13 || wire[0] != 'M' || wire[1] != 'V' ||
      wire[2] != 'W' || wire[3] != '1')
    throw std::invalid_argument("invalid or truncated MVW1 header");
  std::uint64_t count = 0;
  for (unsigned i = 0; i < 8; ++i)
    count |= static_cast<std::uint64_t>(wire[4 + i]) << (i * 8);
  if (count > std::numeric_limits<std::size_t>::max() / 8 ||
      count > std::vector<std::uint8_t>().max_size() / 8)
    throw std::overflow_error("MVW1 decoded byte count overflow");
  const auto values = unpack(wire.subspan(13), static_cast<std::size_t>(count), wire[12]);
  if (maximum_width(values) != wire[12])
    throw std::invalid_argument("MVW1 width is not the maximum input value width");
  std::vector<std::uint8_t> raw(values.size() * 8);
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<std::uint8_t>(values[i / 8] >> ((i % 8) * 8));
  return raw;
}

}  // namespace exact_actions::max_value_bitpack
