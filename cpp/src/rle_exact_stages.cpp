#include "rle_exact_stages.hpp"

#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::rle {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'R', 'L', 'S', '1'};

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_uvarint(std::vector<std::uint8_t>& out, std::uint32_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<std::uint8_t>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t read_u32(std::span<const std::uint8_t> data, std::size_t& pos) {
  if (data.size() - pos < 4) throw std::runtime_error("truncated RLE scaling packet");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(data[pos++]) << shift;
  return value;
}

std::uint64_t read_u64(std::span<const std::uint8_t> data, std::size_t& pos) {
  if (data.size() - pos < 8) throw std::runtime_error("truncated RLE scaling packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

std::uint32_t read_uvarint(std::span<const std::uint8_t> data, std::size_t& pos) {
  std::uint32_t value = 0;
  for (int shift = 0; shift <= 28; shift += 7) {
    if (pos == data.size()) throw std::runtime_error("truncated RLE scale varint");
    const auto byte = data[pos++];
    value |= static_cast<std::uint32_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return value;
  }
  throw std::runtime_error("invalid RLE scale varint");
}

std::int64_t convert(double value, double scale) {
  const double product = value * scale;
  const double minimum = static_cast<double>(std::numeric_limits<std::int64_t>::min());
  const double maximum = static_cast<double>(std::numeric_limits<std::int64_t>::max());
  if (product > maximum || product < minimum) {
    if (value > maximum || value < minimum)
      return std::bit_cast<std::int64_t>(value);
    return static_cast<std::int64_t>(std::llround(value));
  }
  if (std::isnan(value)) return std::bit_cast<std::int64_t>(value);
  return static_cast<std::int64_t>(std::llround(product));
}

bool delta_chain_fits(std::span<const std::int64_t> values) {
  for (std::size_t i = 1; i < values.size(); ++i) {
    const __int128 delta = static_cast<__int128>(values[i]) - values[i - 1];
    if (delta < std::numeric_limits<std::int64_t>::min() ||
        delta > std::numeric_limits<std::int64_t>::max())
      return false;
  }
  return true;
}

struct Packet {
  int scale;
  std::vector<std::int64_t> values;
};

std::vector<std::uint8_t> serialize(int scale,
                                    std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> out(kMagic.begin(), kMagic.end());
  append_u32(out, static_cast<std::uint32_t>(scale));
  append_u64(out, values.size());
  for (const auto value : values) append_u64(out, static_cast<std::uint64_t>(value));
  return out;
}

Packet parse(std::span<const std::uint8_t> packet) {
  if (packet.size() < 16 ||
      !std::equal(kMagic.begin(), kMagic.end(), packet.begin()))
    throw std::runtime_error("wrong RLE scaling packet type");
  std::size_t pos = 4;
  const auto scale = static_cast<int>(read_u32(packet, pos));
  const auto count = read_u64(packet, pos);
  if (scale < 0 || scale > 18 || count > (packet.size() - pos) / 8 ||
      count * 8 != packet.size() - pos)
    throw std::runtime_error("invalid RLE scaling packet");
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i)
    values.push_back(static_cast<std::int64_t>(read_u64(packet, pos)));
  return {scale, std::move(values)};
}

}  // namespace

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int requested_max_point) {
  if (requested_max_point < 0 || requested_max_point > 18)
    throw std::invalid_argument("RLE max_point must be in [0,18]");
  if (std::any_of(values.begin(), values.end(),
                  [](double value) { return !std::isfinite(value); }))
    throw std::invalid_argument("RLE scaling requires finite float64 values");

  int scale = requested_max_point;
  std::vector<std::int64_t> scaled;
  while (true) {
    const double power = std::pow(10.0, scale);
    scaled.clear();
    scaled.reserve(values.size());
    for (const auto value : values) scaled.push_back(convert(value, power));
    if (scale == 0 || delta_chain_fits(scaled)) break;
    --scale;
  }

  const double power = std::pow(10.0, scale);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double recovered =
        scale == 0 ? static_cast<double>(scaled[i]) : scaled[i] / power;
    if (std::bit_cast<std::uint64_t>(recovered) !=
        std::bit_cast<std::uint64_t>(values[i]))
      throw std::invalid_argument("RLE decimal scaling is not bitwise reversible");
  }
  return serialize(scale, scaled);
}

std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet);
  const double power = std::pow(10.0, parsed.scale);
  std::vector<double> values;
  values.reserve(parsed.values.size());
  for (const auto value : parsed.values)
    values.push_back(parsed.scale == 0 ? static_cast<double>(value) : value / power);
  return values;
}

std::vector<std::uint8_t> run_length_forward(
    std::span<const std::uint8_t> scaling_packet) {
  const auto parsed = parse(scaling_packet);
  std::vector<std::uint8_t> out;
  append_uvarint(out, static_cast<std::uint32_t>(parsed.scale));
  const auto body = tsfile::p02::encode_rle_int64_legacy(parsed.values);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> run_length_inverse(
    std::span<const std::uint8_t> public_wire) {
  std::size_t pos = 0;
  const auto scale = read_uvarint(public_wire, pos);
  if (scale > 18) throw std::runtime_error("invalid RLE public scale");
  std::vector<std::int64_t> values;
  if (!tsfile::p02::decode_rle_int64(public_wire.subspan(pos), values))
    throw std::runtime_error("invalid TsFile integer RLE body");
  return serialize(static_cast<int>(scale), values);
}

std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int requested_max_point) {
  return run_length_forward(scaling_forward(values, requested_max_point));
}

std::vector<double> decompose_double(std::span<const std::uint8_t> public_wire) {
  return scaling_inverse(run_length_inverse(public_wire));
}

}  // namespace exact_actions::rle
