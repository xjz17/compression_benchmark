#include "xz_container_stage.hpp"

#include <cstring>
#include <stdexcept>

namespace exact_actions::xz_container {
namespace {

constexpr std::uint8_t kPacketMagic[] = {'X', 'Z', 'C', '1'};
constexpr std::uint8_t kStreamMagic[] = {0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00};

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 4 > input.size()) throw std::runtime_error("truncated XZC1 length");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(input[pos++]) << shift;
  return value;
}

}  // namespace

std::vector<std::uint8_t> inverse(std::span<const std::uint8_t> wire) {
  if (wire.size() < 24 || std::memcmp(wire.data(), kStreamMagic, 6) != 0 ||
      wire[wire.size() - 2] != 0x59 || wire.back() != 0x5a)
    throw std::runtime_error("invalid or truncated single XZ stream");
  constexpr std::size_t header_size = 12;
  constexpr std::size_t footer_size = 12;
  const std::size_t core_size = wire.size() - header_size - footer_size;
  std::vector<std::uint8_t> out(std::begin(kPacketMagic), std::end(kPacketMagic));
  put_u32(out, static_cast<std::uint32_t>(core_size));
  out.insert(out.end(), wire.begin(), wire.begin() + header_size);
  out.insert(out.end(), wire.begin() + header_size,
             wire.end() - static_cast<std::ptrdiff_t>(footer_size));
  out.insert(out.end(), wire.end() - static_cast<std::ptrdiff_t>(footer_size), wire.end());
  return out;
}

std::vector<std::uint8_t> forward(std::span<const std::uint8_t> packet) {
  if (packet.size() < 8 + 24 || std::memcmp(packet.data(), kPacketMagic, 4) != 0)
    throw std::runtime_error("bad XZC1 packet");
  std::size_t pos = 4;
  const auto core_size = get_u32(packet, pos);
  const std::uint64_t expected = static_cast<std::uint64_t>(pos) + 12 + core_size + 12;
  if (expected != packet.size()) throw std::runtime_error("XZC1 packet size mismatch");
  return {packet.begin() + static_cast<std::ptrdiff_t>(pos), packet.end()};
}

}  // namespace exact_actions::xz_container
