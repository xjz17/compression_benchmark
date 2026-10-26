#include "lz4_exact_stages.hpp"

#include <lz4frame.h>
#define XXH_INLINE_ALL
#include <xxhash.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace exact_actions::lz4 {
namespace {

constexpr std::uint8_t kMatchMagic[] = {'L', 'Z', 'M', '1'};
constexpr std::uint8_t kTokenMagic[] = {'L', 'Z', 'T', '1'};
constexpr std::uint8_t kIntegerMagic[] = {'L', 'Z', 'I', '1'};
constexpr std::size_t kBlockSize = 65536;

void put_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}
void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint16_t get_u16(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 2 > in.size()) throw std::runtime_error("LZ4 stage truncated u16");
  const auto value = static_cast<std::uint16_t>(in[pos]) |
                     static_cast<std::uint16_t>(in[pos + 1] << 8);
  pos += 2;
  return value;
}
std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("LZ4 stage truncated u32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(in[pos++]) << (8 * i);
  return value;
}
std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("LZ4 stage truncated u64");
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(in[pos++]) << (8 * i);
  return value;
}
bool magic(std::span<const std::uint8_t> in, const std::uint8_t (&value)[4]) {
  return in.size() >= 4 && std::memcmp(in.data(), value, 4) == 0;
}

struct Sequence {
  std::vector<std::uint8_t> literals;
  bool has_match = false;
  std::uint16_t offset = 0;
  std::uint32_t match_length = 0;
};
struct Block {
  bool uncompressed = false;
  std::uint32_t raw_size = 0;
  std::vector<Sequence> sequences;
  std::vector<std::uint8_t> raw_body;
};
struct Stream {
  std::uint64_t raw_size = 0;
  std::vector<Block> blocks;
};

std::uint32_t read_extension(std::span<const std::uint8_t> body, std::size_t& pos) {
  std::uint32_t value = 0;
  for (;;) {
    if (pos >= body.size()) throw std::runtime_error("LZ4 truncated length extension");
    const std::uint8_t byte = body[pos++];
    if (value > std::numeric_limits<std::uint32_t>::max() - byte)
      throw std::runtime_error("LZ4 length overflow");
    value += byte;
    if (byte != 255) return value;
  }
}

std::vector<Sequence> parse_block(std::span<const std::uint8_t> body,
                                  std::uint32_t expected_size,
                                  std::size_t prior_output) {
  std::vector<Sequence> out;
  std::size_t pos = 0;
  std::size_t produced = 0;
  while (pos < body.size()) {
    const std::uint8_t token = body[pos++];
    std::uint32_t literal_length = token >> 4;
    if (literal_length == 15) literal_length += read_extension(body, pos);
    if (pos + literal_length > body.size()) throw std::runtime_error("LZ4 truncated literals");
    Sequence sequence;
    sequence.literals.assign(body.begin() + static_cast<std::ptrdiff_t>(pos),
                             body.begin() + static_cast<std::ptrdiff_t>(pos + literal_length));
    pos += literal_length;
    produced += literal_length;
    if (pos == body.size()) {
      out.push_back(std::move(sequence));
      break;
    }
    sequence.has_match = true;
    sequence.offset = get_u16(body, pos);
    if (sequence.offset == 0 || sequence.offset > prior_output + produced)
      throw std::runtime_error("LZ4 invalid match offset");
    std::uint32_t match_code = token & 15;
    if (match_code == 15) match_code += read_extension(body, pos);
    sequence.match_length = match_code + 4;
    produced += sequence.match_length;
    out.push_back(std::move(sequence));
  }
  if (produced != expected_size) throw std::runtime_error("LZ4 block output size mismatch");
  return out;
}

void expand_sequences(const std::vector<Sequence>& sequences,
                      std::vector<std::uint8_t>& output) {
  for (const auto& sequence : sequences) {
    output.insert(output.end(), sequence.literals.begin(), sequence.literals.end());
    if (!sequence.has_match) continue;
    if (sequence.offset == 0 || sequence.offset > output.size())
      throw std::runtime_error("LZ4 semantic match before output start");
    for (std::uint32_t i = 0; i < sequence.match_length; ++i)
      output.push_back(output[output.size() - sequence.offset]);
  }
}

std::vector<std::uint8_t> compress_frame(std::span<const std::uint8_t> input) {
  LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
  prefs.frameInfo.blockSizeID = LZ4F_max64KB;
  prefs.frameInfo.blockMode = LZ4F_blockLinked;
  prefs.frameInfo.contentChecksumFlag = LZ4F_noContentChecksum;
  prefs.frameInfo.blockChecksumFlag = LZ4F_noBlockChecksum;
  prefs.frameInfo.contentSize = input.size();
  prefs.compressionLevel = 0;
  prefs.autoFlush = 0;
  const std::size_t capacity = LZ4F_compressFrameBound(input.size(), &prefs);
  std::vector<std::uint8_t> out(capacity);
  const std::size_t size = LZ4F_compressFrame(
      out.data(), out.size(), input.data(), input.size(), &prefs);
  if (LZ4F_isError(size))
    throw std::runtime_error(std::string("LZ4F compression failed: ") + LZ4F_getErrorName(size));
  out.resize(size);
  return out;
}

Stream parse_frame(std::span<const std::uint8_t> wire) {
  if (wire.size() < 7 || wire[0] != 0x04 || wire[1] != 0x22 ||
      wire[2] != 0x4d || wire[3] != 0x18)
    throw std::runtime_error("unsupported LZ4 frame magic");
  std::size_t pos = 4;
  const std::uint8_t flg = wire[pos++];
  const std::uint8_t bd = wire[pos++];
  if ((flg & 0xc0) != 0x40 || (bd & 0x70) != 0x40 || (flg & 0x10) || (flg & 1))
    throw std::runtime_error("unsupported LZ4 frame options");
  Stream stream;
  if (flg & 0x08) stream.raw_size = get_u64(wire, pos);
  if (flg & 0x01) (void)get_u32(wire, pos);
  if (pos >= wire.size()) throw std::runtime_error("truncated LZ4 descriptor checksum");
  ++pos;
  std::size_t produced = 0;
  while (true) {
    const std::uint32_t header = get_u32(wire, pos);
    if (header == 0) break;
    Block block;
    block.uncompressed = (header & 0x80000000U) != 0;
    const std::uint32_t body_size = header & 0x7fffffffU;
    if (pos + body_size > wire.size()) throw std::runtime_error("truncated LZ4 block");
    const auto body = wire.subspan(pos, body_size);
    pos += body_size;
    block.raw_size = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(kBlockSize, stream.raw_size - produced));
    if (block.uncompressed) {
      if (body_size != block.raw_size) throw std::runtime_error("LZ4 raw block size mismatch");
      block.raw_body.assign(body.begin(), body.end());
      Sequence literal;
      literal.literals = block.raw_body;
      block.sequences.push_back(std::move(literal));
    } else {
      block.sequences = parse_block(body, block.raw_size, produced);
      block.raw_body.assign(body.begin(), body.end());
    }
    produced += block.raw_size;
    stream.blocks.push_back(std::move(block));
  }
  if (flg & 0x04) (void)get_u32(wire, pos);
  if (pos != wire.size() || produced != stream.raw_size)
    throw std::runtime_error("LZ4 frame length mismatch");
  return stream;
}

std::vector<std::uint8_t> write_match_packet(const Stream& stream) {
  std::vector<std::uint8_t> out(std::begin(kMatchMagic), std::end(kMatchMagic));
  put_u64(out, stream.raw_size);
  put_u32(out, static_cast<std::uint32_t>(stream.blocks.size()));
  for (const auto& block : stream.blocks) {
    out.push_back(block.uncompressed ? 1 : 0);
    put_u32(out, block.raw_size);
    put_u32(out, static_cast<std::uint32_t>(block.sequences.size()));
    for (const auto& seq : block.sequences) {
      put_u32(out, static_cast<std::uint32_t>(seq.literals.size()));
      out.insert(out.end(), seq.literals.begin(), seq.literals.end());
      out.push_back(seq.has_match ? 1 : 0);
      if (seq.has_match) {
        put_u16(out, seq.offset);
        put_u32(out, seq.match_length);
      }
    }
  }
  return out;
}

Stream read_match_packet(std::span<const std::uint8_t> packet) {
  if (!magic(packet, kMatchMagic)) throw std::runtime_error("bad LZM1 packet");
  std::size_t pos = 4;
  Stream stream;
  stream.raw_size = get_u64(packet, pos);
  const auto blocks = get_u32(packet, pos);
  for (std::uint32_t b = 0; b < blocks; ++b) {
    if (pos >= packet.size()) throw std::runtime_error("truncated LZM1 block");
    Block block;
    block.uncompressed = packet[pos++] != 0;
    block.raw_size = get_u32(packet, pos);
    const auto count = get_u32(packet, pos);
    for (std::uint32_t i = 0; i < count; ++i) {
      Sequence seq;
      const auto lit = get_u32(packet, pos);
      if (pos + lit > packet.size()) throw std::runtime_error("truncated LZM1 literal");
      seq.literals.assign(packet.begin() + static_cast<std::ptrdiff_t>(pos),
                          packet.begin() + static_cast<std::ptrdiff_t>(pos + lit));
      pos += lit;
      if (pos >= packet.size()) throw std::runtime_error("truncated LZM1 match flag");
      seq.has_match = packet[pos++] != 0;
      if (seq.has_match) {
        seq.offset = get_u16(packet, pos);
        seq.match_length = get_u32(packet, pos);
      }
      block.sequences.push_back(std::move(seq));
    }
    stream.blocks.push_back(std::move(block));
  }
  if (pos != packet.size()) throw std::runtime_error("LZM1 trailing bytes");
  return stream;
}

std::vector<std::uint8_t> write_token_packet(const Stream& stream) {
  std::vector<std::uint8_t> out(std::begin(kTokenMagic), std::end(kTokenMagic));
  put_u64(out, stream.raw_size);
  put_u32(out, static_cast<std::uint32_t>(stream.blocks.size()));
  for (const auto& block : stream.blocks) {
    out.push_back(block.uncompressed ? 1 : 0);
    put_u32(out, block.raw_size);
    put_u32(out, static_cast<std::uint32_t>(block.sequences.size()));
    for (const auto& seq : block.sequences) {
      const auto lit = static_cast<std::uint32_t>(seq.literals.size());
      const auto match_code = seq.has_match ? seq.match_length - 4 : 0;
      const auto token = static_cast<std::uint8_t>((std::min(lit, 15U) << 4) |
                                                   std::min(match_code, 15U));
      out.push_back(token);
      put_u32(out, lit > 15 ? lit - 15 : 0);
      put_u32(out, lit);
      out.insert(out.end(), seq.literals.begin(), seq.literals.end());
      out.push_back(seq.has_match ? 1 : 0);
      if (seq.has_match) {
        put_u16(out, seq.offset);
        put_u32(out, match_code > 15 ? match_code - 15 : 0);
      }
    }
  }
  return out;
}

Stream read_token_packet(std::span<const std::uint8_t> packet) {
  if (!magic(packet, kTokenMagic)) throw std::runtime_error("bad LZT1 packet");
  std::size_t pos = 4;
  Stream stream;
  stream.raw_size = get_u64(packet, pos);
  const auto blocks = get_u32(packet, pos);
  for (std::uint32_t b = 0; b < blocks; ++b) {
    if (pos >= packet.size()) throw std::runtime_error("truncated LZT1 block");
    Block block;
    block.uncompressed = packet[pos++] != 0;
    block.raw_size = get_u32(packet, pos);
    const auto count = get_u32(packet, pos);
    for (std::uint32_t i = 0; i < count; ++i) {
      if (pos >= packet.size()) throw std::runtime_error("truncated LZT1 token");
      const auto token = packet[pos++];
      const auto lit_extra = get_u32(packet, pos);
      const auto lit = get_u32(packet, pos);
      if (lit != static_cast<std::uint32_t>(token >> 4) +
                     ((token >> 4) == 15 ? lit_extra : 0))
        throw std::runtime_error("LZT1 literal length mismatch");
      if (pos + lit > packet.size()) throw std::runtime_error("truncated LZT1 literal");
      Sequence seq;
      seq.literals.assign(packet.begin() + static_cast<std::ptrdiff_t>(pos),
                          packet.begin() + static_cast<std::ptrdiff_t>(pos + lit));
      pos += lit;
      if (pos >= packet.size()) throw std::runtime_error("truncated LZT1 match flag");
      seq.has_match = packet[pos++] != 0;
      if (seq.has_match) {
        seq.offset = get_u16(packet, pos);
        const auto extra = get_u32(packet, pos);
        seq.match_length = 4 + (token & 15) + ((token & 15) == 15 ? extra : 0);
      } else if ((token & 15) != 0) {
        throw std::runtime_error("LZT1 terminal match nibble must be zero");
      }
      block.sequences.push_back(std::move(seq));
    }
    stream.blocks.push_back(std::move(block));
  }
  if (pos != packet.size()) throw std::runtime_error("LZT1 trailing bytes");
  return stream;
}

void emit_extension(std::vector<std::uint8_t>& out, std::uint32_t value) {
  do {
    const auto byte = static_cast<std::uint8_t>(std::min(value, 255U));
    out.push_back(byte);
    value -= byte;
  } while (out.back() == 255);
}

std::vector<std::uint8_t> encode_block(const Block& block) {
  if (block.uncompressed) {
    std::vector<std::uint8_t> raw;
    expand_sequences(block.sequences, raw);
    return raw;
  }
  std::vector<std::uint8_t> out;
  for (const auto& seq : block.sequences) {
    const auto lit = static_cast<std::uint32_t>(seq.literals.size());
    const auto match_code = seq.has_match ? seq.match_length - 4 : 0;
    out.push_back(static_cast<std::uint8_t>((std::min(lit, 15U) << 4) |
                                            std::min(match_code, 15U)));
    if (lit >= 15) emit_extension(out, lit - 15);
    out.insert(out.end(), seq.literals.begin(), seq.literals.end());
    if (seq.has_match) {
      put_u16(out, seq.offset);
      if (match_code >= 15) emit_extension(out, match_code - 15);
    }
  }
  return out;
}

std::vector<std::uint8_t> write_integer_packet(const Stream& stream) {
  std::vector<std::uint8_t> out(std::begin(kIntegerMagic), std::end(kIntegerMagic));
  put_u64(out, stream.raw_size);
  put_u32(out, static_cast<std::uint32_t>(stream.blocks.size()));
  for (const auto& block : stream.blocks) {
    out.push_back(block.uncompressed ? 1 : 0);
    put_u32(out, block.raw_size);
    const auto body = encode_block(block);
    put_u32(out, static_cast<std::uint32_t>(body.size()));
    out.insert(out.end(), body.begin(), body.end());
  }
  return out;
}

Stream read_integer_packet(std::span<const std::uint8_t> packet) {
  if (!magic(packet, kIntegerMagic)) throw std::runtime_error("bad LZI1 packet");
  std::size_t pos = 4;
  Stream stream;
  stream.raw_size = get_u64(packet, pos);
  const auto blocks = get_u32(packet, pos);
  std::size_t prior = 0;
  for (std::uint32_t b = 0; b < blocks; ++b) {
    if (pos >= packet.size()) throw std::runtime_error("truncated LZI1 block");
    Block block;
    block.uncompressed = packet[pos++] != 0;
    block.raw_size = get_u32(packet, pos);
    const auto size = get_u32(packet, pos);
    if (pos + size > packet.size()) throw std::runtime_error("truncated LZI1 body");
    const auto body = packet.subspan(pos, size);
    pos += size;
    block.raw_body.assign(body.begin(), body.end());
    if (block.uncompressed) {
      if (size != block.raw_size) throw std::runtime_error("LZI1 raw size mismatch");
      Sequence seq;
      seq.literals = block.raw_body;
      block.sequences.push_back(std::move(seq));
    } else {
      block.sequences = parse_block(body, block.raw_size, prior);
    }
    prior += block.raw_size;
    stream.blocks.push_back(std::move(block));
  }
  if (pos != packet.size() || prior != stream.raw_size)
    throw std::runtime_error("LZI1 stream size mismatch");
  return stream;
}

std::vector<std::uint8_t> write_frame(const Stream& stream) {
  std::vector<std::uint8_t> out{0x04, 0x22, 0x4d, 0x18};
  std::vector<std::uint8_t> descriptor;
  if (stream.raw_size == 0) {
    descriptor = {0x60, 0x40};
  } else {
    descriptor.push_back(stream.raw_size <= kBlockSize ? 0x68 : 0x48);
    descriptor.push_back(0x40);
    for (int shift = 0; shift < 64; shift += 8)
      descriptor.push_back(static_cast<std::uint8_t>(stream.raw_size >> shift));
  }
  out.insert(out.end(), descriptor.begin(), descriptor.end());
  out.push_back(static_cast<std::uint8_t>((XXH32(descriptor.data(), descriptor.size(), 0) >> 8) & 0xff));
  for (const auto& block : stream.blocks) {
    const auto body = block.raw_body.empty() ? encode_block(block) : block.raw_body;
    put_u32(out, static_cast<std::uint32_t>(body.size()) |
                     (block.uncompressed ? 0x80000000U : 0));
    out.insert(out.end(), body.begin(), body.end());
  }
  put_u32(out, 0);
  return out;
}

}  // namespace

std::vector<std::uint8_t> original_encode(
    std::span<const std::uint8_t> input) {
  return compress_frame(input);
}

std::vector<std::uint8_t> lz_match_forward(std::span<const std::uint8_t> input) {
  return write_match_packet(parse_frame(compress_frame(input)));
}
std::vector<std::uint8_t> lz_match_inverse(std::span<const std::uint8_t> packet) {
  const auto stream = read_match_packet(packet);
  std::vector<std::uint8_t> out;
  out.reserve(stream.raw_size);
  for (const auto& block : stream.blocks) expand_sequences(block.sequences, out);
  if (out.size() != stream.raw_size) throw std::runtime_error("LZM1 expansion size mismatch");
  return out;
}
std::vector<std::uint8_t> token_format_forward(std::span<const std::uint8_t> packet) {
  return write_token_packet(read_match_packet(packet));
}
std::vector<std::uint8_t> token_format_inverse(std::span<const std::uint8_t> packet) {
  return write_match_packet(read_token_packet(packet));
}
std::vector<std::uint8_t> integer_code_forward(std::span<const std::uint8_t> packet) {
  return write_integer_packet(read_token_packet(packet));
}
std::vector<std::uint8_t> integer_code_inverse(std::span<const std::uint8_t> packet) {
  return write_token_packet(read_integer_packet(packet));
}
std::vector<std::uint8_t> container_forward(std::span<const std::uint8_t> packet) {
  return write_frame(read_integer_packet(packet));
}
std::vector<std::uint8_t> container_inverse(std::span<const std::uint8_t> wire) {
  return write_integer_packet(parse_frame(wire));
}
std::vector<std::uint8_t> compose(std::span<const std::uint8_t> input) {
  return container_forward(integer_code_forward(token_format_forward(lz_match_forward(input))));
}
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire) {
  return lz_match_inverse(token_format_inverse(integer_code_inverse(container_inverse(wire))));
}

}  // namespace exact_actions::lz4
