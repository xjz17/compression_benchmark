#include "deflate_container_stages.hpp"

#include <cstring>
#include <stdexcept>

namespace exact_actions::deflate_container {
namespace {

constexpr std::uint8_t kPacketMagic[] = {'D', 'F', 'C', '1'};

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 4 > input.size()) throw std::runtime_error("truncated DFC1 length");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(input[pos++]) << shift;
  return value;
}

std::vector<std::uint8_t> make_packet(
    char kind, std::span<const std::uint8_t> header,
    std::span<const std::uint8_t> core, std::span<const std::uint8_t> trailer) {
  std::vector<std::uint8_t> out(std::begin(kPacketMagic), std::end(kPacketMagic));
  out.push_back(static_cast<std::uint8_t>(kind));
  put_u32(out, static_cast<std::uint32_t>(header.size()));
  put_u32(out, static_cast<std::uint32_t>(core.size()));
  put_u32(out, static_cast<std::uint32_t>(trailer.size()));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), core.begin(), core.end());
  out.insert(out.end(), trailer.begin(), trailer.end());
  return out;
}

struct PacketView {
  char kind = 0;
  std::span<const std::uint8_t> header;
  std::span<const std::uint8_t> core;
  std::span<const std::uint8_t> trailer;
};

PacketView parse_packet(std::span<const std::uint8_t> input, char expected_kind) {
  if (input.size() < 17 || std::memcmp(input.data(), kPacketMagic, 4) != 0)
    throw std::runtime_error("bad DFC1 packet");
  std::size_t pos = 4;
  const char kind = static_cast<char>(input[pos++]);
  if (kind != expected_kind) throw std::runtime_error("wrong DFC1 wrapper kind");
  const auto header_size = get_u32(input, pos);
  const auto core_size = get_u32(input, pos);
  const auto trailer_size = get_u32(input, pos);
  const std::uint64_t total = static_cast<std::uint64_t>(pos) + header_size +
                              core_size + trailer_size;
  if (total != input.size()) throw std::runtime_error("DFC1 packet size mismatch");
  PacketView result;
  result.kind = kind;
  result.header = input.subspan(pos, header_size);
  pos += header_size;
  result.core = input.subspan(pos, core_size);
  pos += core_size;
  result.trailer = input.subspan(pos, trailer_size);
  return result;
}

std::vector<std::uint8_t> assemble(std::span<const std::uint8_t> packet,
                                   char expected_kind) {
  const auto parsed = parse_packet(packet, expected_kind);
  std::vector<std::uint8_t> out;
  out.reserve(parsed.header.size() + parsed.core.size() + parsed.trailer.size());
  out.insert(out.end(), parsed.header.begin(), parsed.header.end());
  out.insert(out.end(), parsed.core.begin(), parsed.core.end());
  out.insert(out.end(), parsed.trailer.begin(), parsed.trailer.end());
  return out;
}

std::size_t zero_terminated_end(std::span<const std::uint8_t> wire,
                                std::size_t pos, std::size_t limit) {
  while (pos < limit && wire[pos] != 0) ++pos;
  if (pos >= limit) throw std::runtime_error("truncated gzip text field");
  return pos + 1;
}

}  // namespace

std::vector<std::uint8_t> gzip_forward(std::span<const std::uint8_t> packet) {
  return assemble(packet, 'G');
}

std::vector<std::uint8_t> gzip_inverse(std::span<const std::uint8_t> wire) {
  if (wire.size() < 18 || wire[0] != 0x1f || wire[1] != 0x8b || wire[2] != 8)
    throw std::runtime_error("invalid gzip member");
  const std::uint8_t flags = wire[3];
  if ((flags & 0xe0) != 0) throw std::runtime_error("reserved gzip flags set");
  const std::size_t trailer_begin = wire.size() - 8;
  std::size_t pos = 10;
  if (flags & 0x04) {
    if (pos + 2 > trailer_begin) throw std::runtime_error("truncated gzip FEXTRA");
    const std::size_t extra = static_cast<std::size_t>(wire[pos]) |
                              (static_cast<std::size_t>(wire[pos + 1]) << 8);
    pos += 2;
    if (pos + extra > trailer_begin) throw std::runtime_error("truncated gzip extra data");
    pos += extra;
  }
  if (flags & 0x08) pos = zero_terminated_end(wire, pos, trailer_begin);
  if (flags & 0x10) pos = zero_terminated_end(wire, pos, trailer_begin);
  if (flags & 0x02) {
    if (pos + 2 > trailer_begin) throw std::runtime_error("truncated gzip FHCRC");
    pos += 2;
  }
  if (pos > trailer_begin) throw std::runtime_error("gzip header overlaps trailer");
  return make_packet('G', wire.first(pos), wire.subspan(pos, trailer_begin - pos),
                     wire.last(8));
}

std::vector<std::uint8_t> zlib_forward(std::span<const std::uint8_t> packet) {
  return assemble(packet, 'Z');
}

std::vector<std::uint8_t> zlib_inverse(std::span<const std::uint8_t> wire) {
  if (wire.size() < 6) throw std::runtime_error("truncated zlib stream");
  const std::uint8_t cmf = wire[0];
  const std::uint8_t flg = wire[1];
  if ((cmf & 0x0f) != 8 || (cmf >> 4) > 7)
    throw std::runtime_error("invalid zlib CMF");
  if (((static_cast<unsigned>(cmf) << 8) | flg) % 31 != 0)
    throw std::runtime_error("invalid zlib FCHECK");
  const std::size_t header_size = (flg & 0x20) ? 6 : 2;
  if (wire.size() < header_size + 4) throw std::runtime_error("truncated zlib dictionary header");
  return make_packet('Z', wire.first(header_size),
                     wire.subspan(header_size, wire.size() - header_size - 4),
                     wire.last(4));
}

}  // namespace exact_actions::deflate_container
