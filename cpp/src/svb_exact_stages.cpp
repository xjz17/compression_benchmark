#include "svb_exact_stages.hpp"
#include "dbp_exact_stages.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
size_t svb_encode(uint8_t* out, const uint32_t* in, uint32_t count, int delta,
                  int type);
size_t svb_decode(uint32_t* out, uint8_t* in, int delta, int type);
}

namespace exact_actions::svb {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'S', 'V', 'D', '1'};

std::uint32_t read_u32_le(std::span<const std::uint8_t> bytes,
                          std::size_t& pos) {
  if (bytes.size() - pos < 4) throw std::runtime_error("truncated SVB packet");
  std::uint32_t value = 0;
  std::memcpy(&value, bytes.data() + pos, 4);
  pos += 4;
  return value;
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), raw, raw + 4);
}

std::vector<std::uint8_t> serialize(int scale,
                                    std::span<const std::uint32_t> lo,
                                    std::span<const std::uint32_t> hi) {
  if (lo.size() != hi.size() || lo.size() > 0xffffffffu) {
    throw std::invalid_argument("invalid SVB limb count");
  }
  std::vector<std::uint8_t> out;
  out.insert(out.end(), kMagic.begin(), kMagic.end());
  append_u32_le(out, static_cast<std::uint32_t>(scale));
  append_u32_le(out, static_cast<std::uint32_t>(lo.size()));
  for (const auto value : lo) append_u32_le(out, value);
  for (const auto value : hi) append_u32_le(out, value);
  return out;
}

struct Packet {
  int scale;
  std::vector<std::uint32_t> lo;
  std::vector<std::uint32_t> hi;
};

Packet parse(
    std::span<const std::uint8_t> packet) {
  if (packet.size() < 12 ||
      !std::equal(kMagic.begin(), kMagic.end(), packet.begin())) {
    throw std::runtime_error("wrong SVB delta packet type");
  }
  std::size_t pos = 4;
  const int scale = static_cast<int>(static_cast<std::int32_t>(read_u32_le(packet, pos)));
  const std::uint32_t count = read_u32_le(packet, pos);
  if (packet.size() - pos != static_cast<std::size_t>(count) * 8) {
    throw std::runtime_error("invalid SVB delta packet count");
  }
  std::vector<std::uint32_t> lo(count), hi(count);
  for (auto& value : lo) value = read_u32_le(packet, pos);
  for (auto& value : hi) value = read_u32_le(packet, pos);
  return {scale, std::move(lo), std::move(hi)};
}

std::vector<std::uint8_t> encode_plain(std::span<const std::uint32_t> values) {
  if (values.empty()) return {};
  std::vector<std::uint8_t> buffer(
      4 + (values.size() + 3) / 4 + values.size() * 5 + 64);
  const std::size_t encoded = svb_encode(
      buffer.data(), values.data(), static_cast<std::uint32_t>(values.size()), 0,
      1);
  if (encoded == 0 || encoded > buffer.size()) {
    throw std::runtime_error("Stream-VByte encode failed");
  }
  buffer.resize(encoded);
  return buffer;
}

std::vector<std::uint32_t> decode_plain(
    std::span<const std::uint8_t> payload) {
  if (payload.empty()) return {};
  if (payload.size() < 4) throw std::runtime_error("truncated Stream-VByte payload");
  std::uint32_t count = 0;
  std::memcpy(&count, payload.data(), 4);
  std::vector<std::uint32_t> values(count);
  std::vector<std::uint8_t> mutable_payload(payload.begin(), payload.end());
  const std::size_t consumed =
      svb_decode(values.data(), mutable_payload.data(), 0, 1);
  if (consumed != payload.size()) {
    throw std::runtime_error("Stream-VByte decode length mismatch");
  }
  return values;
}

}  // namespace

std::vector<std::uint8_t> delta_forward_impl(
    int scale, std::span<const std::int64_t> values) {
  std::vector<std::uint32_t> lo(values.size()), hi(values.size());
  std::uint32_t previous_lo = 0;
  std::uint32_t previous_hi = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto bits = static_cast<std::uint64_t>(values[i]);
    const std::uint32_t current_lo = static_cast<std::uint32_t>(bits);
    const std::uint32_t current_hi = static_cast<std::uint32_t>(bits >> 32);
    lo[i] = current_lo - previous_lo;
    hi[i] = current_hi - previous_hi;
    previous_lo = current_lo;
    previous_hi = current_hi;
  }
  return serialize(scale, lo, hi);
}

std::vector<std::uint8_t> delta_forward(std::span<const std::int64_t> values) {
  return delta_forward_impl(-1, values);
}

std::vector<std::uint8_t> delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  const auto [scale, values] = exact_actions::dbp::scaling_values(scaling_packet);
  return delta_forward_impl(scale, values);
}

std::vector<std::int64_t> delta_inverse(
    std::span<const std::uint8_t> delta_packet) {
  auto parsed = parse(delta_packet);
  if (parsed.scale != -1) throw std::runtime_error("SVB delta packet is scaled");
  auto& lo = parsed.lo;
  auto& hi = parsed.hi;
  std::vector<std::int64_t> values(lo.size());
  std::uint32_t previous_lo = 0;
  std::uint32_t previous_hi = 0;
  for (std::size_t i = 0; i < lo.size(); ++i) {
    previous_lo += lo[i];
    previous_hi += hi[i];
    const std::uint64_t bits =
        (static_cast<std::uint64_t>(previous_hi) << 32) | previous_lo;
    values[i] = static_cast<std::int64_t>(bits);
  }
  return values;
}

std::vector<std::uint8_t> delta_inverse_scaled(
    std::span<const std::uint8_t> delta_packet) {
  auto parsed = parse(delta_packet);
  if (parsed.scale < 0) throw std::runtime_error("SVB delta packet is not scaled");
  std::vector<std::int64_t> values(parsed.lo.size());
  std::uint32_t previous_lo = 0, previous_hi = 0;
  for (std::size_t i = 0; i < parsed.lo.size(); ++i) {
    previous_lo += parsed.lo[i];
    previous_hi += parsed.hi[i];
    values[i] = static_cast<std::int64_t>(
        (static_cast<std::uint64_t>(previous_hi) << 32) | previous_lo);
  }
  return exact_actions::dbp::make_scaling_packet(parsed.scale, values);
}

std::vector<std::uint8_t> integer_code_forward(
    std::span<const std::uint8_t> delta_packet) {
  const auto parsed = parse(delta_packet);
  const auto encoded_lo = encode_plain(parsed.lo);
  const auto encoded_hi = encode_plain(parsed.hi);
  std::vector<std::uint8_t> out;
  if (parsed.scale < 0) {
    out.push_back('I');
  } else {
    out.push_back('F');
    append_u32_le(out, static_cast<std::uint32_t>(parsed.scale));
  }
  append_u32_le(out, static_cast<std::uint32_t>(encoded_lo.size()));
  out.insert(out.end(), encoded_lo.begin(), encoded_lo.end());
  out.insert(out.end(), encoded_hi.begin(), encoded_hi.end());
  return out;
}

std::vector<std::uint8_t> integer_code_inverse(
    std::span<const std::uint8_t> public_wire) {
  if (public_wire.empty() || (public_wire[0] != 'I' && public_wire[0] != 'F'))
    throw std::runtime_error("invalid SVB adapter tag");
  std::size_t pos = 1;
  const int scale = public_wire[0] == 'I'
                        ? -1
                        : static_cast<int>(read_u32_le(public_wire, pos));
  const std::uint32_t lo_size = read_u32_le(public_wire, pos);
  if (lo_size > public_wire.size() - pos) {
    throw std::runtime_error("truncated SVB low-limb payload");
  }
  const auto lo = decode_plain(public_wire.subspan(pos, lo_size));
  pos += lo_size;
  const auto hi = decode_plain(public_wire.subspan(pos));
  if (lo.size() != hi.size()) throw std::runtime_error("SVB limb count mismatch");
  return serialize(scale, lo, hi);
}

std::vector<std::uint8_t> compose(std::span<const std::int64_t> values) {
  return integer_code_forward(delta_forward(values));
}

std::vector<std::uint8_t> compose_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  return integer_code_forward(delta_forward_scaled(scaling_packet));
}

std::vector<std::int64_t> decompose(
    std::span<const std::uint8_t> public_wire) {
  return delta_inverse(integer_code_inverse(public_wire));
}


std::vector<std::uint8_t> decompose_scaled(
    std::span<const std::uint8_t> public_wire) {
  return delta_inverse_scaled(integer_code_inverse(public_wire));
}

}  // namespace exact_actions::svb
