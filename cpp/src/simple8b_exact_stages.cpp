#include "simple8b_exact_stages.hpp"

#include "int64_column_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace exact_actions::simple8b {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'S', '8', 'S', '1'};

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), raw, raw + 4);
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), raw, raw + 8);
}

std::uint32_t read_u32_le(std::span<const std::uint8_t> bytes,
                          std::size_t& pos) {
  if (bytes.size() - pos < 4) throw std::runtime_error("truncated Simple8b packet");
  std::uint32_t value = 0;
  std::memcpy(&value, bytes.data() + pos, 4);
  pos += 4;
  return value;
}

std::uint64_t read_u64_le(std::span<const std::uint8_t> bytes,
                          std::size_t& pos) {
  if (bytes.size() - pos < 8) throw std::runtime_error("truncated Simple8b packet");
  std::uint64_t value = 0;
  std::memcpy(&value, bytes.data() + pos, 8);
  pos += 8;
  return value;
}

std::vector<std::uint8_t> serialize(int scale,
                                    std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> out(kMagic.begin(), kMagic.end());
  append_u32_le(out, static_cast<std::uint32_t>(scale));
  append_u64_le(out, values.size());
  for (const auto value : values) append_u64_le(out, static_cast<std::uint64_t>(value));
  return out;
}

std::pair<int, std::vector<std::int64_t>> parse(
    std::span<const std::uint8_t> packet) {
  if (packet.size() < 16 ||
      !std::equal(kMagic.begin(), kMagic.end(), packet.begin())) {
    throw std::runtime_error("wrong Simple8b scaling packet type");
  }
  std::size_t pos = 4;
  const int scale = static_cast<int>(read_u32_le(packet, pos));
  const std::uint64_t count = read_u64_le(packet, pos);
  if (scale < 0 || scale > 18 || count > (packet.size() - pos) / 8 ||
      count * 8 != packet.size() - pos) {
    throw std::runtime_error("invalid Simple8b scaling packet");
  }
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    values.push_back(static_cast<std::int64_t>(read_u64_le(packet, pos)));
  }
  return {scale, std::move(values)};
}

std::vector<std::uint8_t> encode_core(std::span<const std::int64_t> values) {
  return int64_column_codec::simple8b_encode_int64(values.data(), values.size());
}

std::vector<std::int64_t> decode_core(
    std::span<const std::uint8_t> body) {
  if (body.size() < 8) throw std::runtime_error("truncated Simple8b public wire");
  std::uint32_t limb_count = 0;
  std::memcpy(&limb_count, body.data(), 4);
  if ((limb_count & 1u) != 0) throw std::runtime_error("odd Simple8b limb count");
  const std::size_t count = limb_count / 2;
  return int64_column_codec::simple8b_decode_int64(
      body.data(), body.size(), count);
}

}  // namespace

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int max_point) {
  if (max_point < 0 || max_point > 18) throw std::invalid_argument("max_point must be in [0,18]");
  const double power = std::pow(10.0, max_point);
  std::vector<std::int64_t> scaled;
  scaled.reserve(values.size());
  for (const double value : values) {
    if (!std::isfinite(value)) throw std::invalid_argument("Simple8b scaling requires finite values");
    const long double product = static_cast<long double>(value) * power;
    // The power-of-two bound is exact even when long double has 53-bit precision.
    // Converting INT64_MAX to that format rounds it up and admits +2^63.
    if (product < -0x1p63L || product >= 0x1p63L) {
      throw std::overflow_error("Simple8b scaled value exceeds int64");
    }
    const auto encoded = static_cast<std::int64_t>(std::llround(product));
    const double recovered = static_cast<double>(encoded) / power;
    if (std::bit_cast<std::uint64_t>(recovered) != std::bit_cast<std::uint64_t>(value)) {
      throw std::invalid_argument("Simple8b decimal scaling is not bitwise reversible");
    }
    scaled.push_back(encoded);
  }
  return serialize(max_point, scaled);
}

std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet) {
  const auto [scale, values] = parse(packet);
  const double power = std::pow(10.0, scale);
  std::vector<double> restored;
  restored.reserve(values.size());
  for (const auto value : values) restored.push_back(static_cast<double>(value) / power);
  return restored;
}

std::vector<std::uint8_t> bitpack_forward_int64(
    std::span<const std::int64_t> values) {
  auto body = encode_core(values);
  body.insert(body.begin(), static_cast<std::uint8_t>('I'));
  return body;
}

std::vector<std::uint8_t> bitpack_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  const auto [scale, values] = parse(scaling_packet);
  std::vector<std::uint8_t> out{'F'};
  append_u32_le(out, static_cast<std::uint32_t>(scale));
  const auto body = encode_core(values);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::int64_t> bitpack_inverse_int64(
    std::span<const std::uint8_t> public_wire) {
  if (public_wire.empty() || public_wire[0] != 'I') {
    throw std::runtime_error("not an integer Simple8b adapter wire");
  }
  return decode_core(public_wire.subspan(1));
}

std::vector<std::uint8_t> bitpack_inverse_scaled(
    std::span<const std::uint8_t> public_wire) {
  if (public_wire.size() < 5 || public_wire[0] != 'F') {
    throw std::runtime_error("not a floating Simple8b adapter wire");
  }
  std::size_t pos = 1;
  const int scale = static_cast<int>(read_u32_le(public_wire, pos));
  const auto values = decode_core(public_wire.subspan(pos));
  return serialize(scale, values);
}

std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int max_point) {
  return bitpack_forward_scaled(scaling_forward(values, max_point));
}

std::vector<double> decompose_double(
    std::span<const std::uint8_t> public_wire) {
  return scaling_inverse(bitpack_inverse_scaled(public_wire));
}

}  // namespace exact_actions::simple8b
