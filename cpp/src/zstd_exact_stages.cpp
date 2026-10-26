#include "zstd_exact_stages.hpp"

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace exact_actions::zstd {
namespace {

constexpr std::uint8_t kMagic[] = {'Z', 'S', 'Q', '1'};

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("Zstd sequence packet truncated u32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(in[pos++]) << (8 * i);
  return value;
}
std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("Zstd sequence packet truncated u64");
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(in[pos++]) << (8 * i);
  return value;
}

struct Parsed {
  std::uint64_t raw_size = 0;
  std::vector<ZSTD_Sequence> sequences;
  std::vector<std::vector<std::uint8_t>> literals;
};

void check(std::size_t code, const char* label) {
  if (ZSTD_isError(code))
    throw std::runtime_error(std::string(label) + ": " + ZSTD_getErrorName(code));
}

void set_level_and_boundaries(ZSTD_CCtx* context) {
  check(ZSTD_CCtx_setParameter(context, ZSTD_c_compressionLevel, 9), "Zstd level");
  check(ZSTD_CCtx_setParameter(context, ZSTD_c_blockDelimiters,
                               ZSTD_sf_explicitBlockDelimiters),
        "Zstd block delimiters");
}

void set_level(ZSTD_CCtx* context) {
  check(ZSTD_CCtx_setParameter(context, ZSTD_c_compressionLevel, 9), "Zstd level");
}

std::vector<std::uint8_t> make_packet(std::span<const std::uint8_t> source,
                                      std::span<const ZSTD_Sequence> sequences) {
  std::vector<std::uint8_t> out(std::begin(kMagic), std::end(kMagic));
  put_u64(out, source.size());
  put_u32(out, static_cast<std::uint32_t>(sequences.size()));
  std::size_t cursor = 0;
  for (const auto& seq : sequences) {
    if (cursor + seq.litLength > source.size())
      throw std::runtime_error("Zstd generated literal range exceeds source");
    put_u32(out, seq.litLength);
    put_u32(out, seq.matchLength);
    put_u32(out, seq.offset);
    put_u32(out, seq.rep);
    out.insert(out.end(), source.begin() + static_cast<std::ptrdiff_t>(cursor),
               source.begin() + static_cast<std::ptrdiff_t>(cursor + seq.litLength));
    cursor += seq.litLength + seq.matchLength;
    if (cursor > source.size()) throw std::runtime_error("Zstd generated sequence exceeds source");
  }
  if (cursor != source.size()) {
    // ZSTD_generateSequences normally emits an explicit final-literals record.
    // Reject rather than silently invent a sequence boundary.
    throw std::runtime_error("Zstd generated sequences omit source tail");
  }
  return out;
}

Parsed parse_packet(std::span<const std::uint8_t> packet) {
  if (packet.size() < 16 || std::memcmp(packet.data(), kMagic, 4) != 0)
    throw std::runtime_error("bad ZSQ1 packet");
  std::size_t pos = 4;
  Parsed parsed;
  parsed.raw_size = get_u64(packet, pos);
  const auto count = get_u32(packet, pos);
  parsed.sequences.reserve(count);
  parsed.literals.reserve(count);
  std::uint64_t total = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    ZSTD_Sequence seq{};
    seq.litLength = get_u32(packet, pos);
    seq.matchLength = get_u32(packet, pos);
    seq.offset = get_u32(packet, pos);
    seq.rep = get_u32(packet, pos);
    if (pos + seq.litLength > packet.size())
      throw std::runtime_error("truncated ZSQ1 literals");
    parsed.literals.emplace_back(
        packet.begin() + static_cast<std::ptrdiff_t>(pos),
        packet.begin() + static_cast<std::ptrdiff_t>(pos + seq.litLength));
    pos += seq.litLength;
    total += static_cast<std::uint64_t>(seq.litLength) + seq.matchLength;
    parsed.sequences.push_back(seq);
  }
  if (pos != packet.size() || total != parsed.raw_size)
    throw std::runtime_error("ZSQ1 size mismatch");
  return parsed;
}

std::vector<std::uint8_t> expand(const Parsed& parsed) {
  std::vector<std::uint8_t> out;
  out.reserve(parsed.raw_size);
  for (std::size_t i = 0; i < parsed.sequences.size(); ++i) {
    const auto& seq = parsed.sequences[i];
    out.insert(out.end(), parsed.literals[i].begin(), parsed.literals[i].end());
    if (seq.matchLength == 0) {
      if (seq.offset != 0) throw std::runtime_error("invalid Zstd delimiter sequence");
      continue;
    }
    if (seq.offset == 0 || seq.offset > out.size())
      throw std::runtime_error("invalid Zstd semantic offset");
    for (std::uint32_t j = 0; j < seq.matchLength; ++j)
      out.push_back(out[out.size() - seq.offset]);
  }
  if (out.size() != parsed.raw_size) throw std::runtime_error("Zstd expansion size mismatch");
  return out;
}

}  // namespace

std::vector<std::uint8_t> sequence_forward(std::span<const std::uint8_t> input) {
  if (input.empty()) {
    ZSTD_Sequence delimiter{};
    return make_packet(input, std::span<const ZSTD_Sequence>(&delimiter, 1));
  }
  ZSTD_CCtx* context = ZSTD_createCCtx();
  if (context == nullptr) throw std::runtime_error("ZSTD_createCCtx failed");
  try {
    // Ask the producer to preserve the same block boundaries that the replay
    // API expects.  Without explicit delimiters, large inputs can be replayed
    // as one differently partitioned block even though the sequences decode
    // to the same bytes.
    set_level_and_boundaries(context);
    std::vector<ZSTD_Sequence> sequences(ZSTD_sequenceBound(input.size()));
    const auto count = ZSTD_generateSequences(
        context, sequences.data(), sequences.size(), input.data(), input.size());
    if (ZSTD_isError(count)) {
      ZSTD_Sequence literals{};
      literals.litLength = static_cast<unsigned>(input.size());
      sequences.assign(1, literals);
    } else {
      sequences.resize(count);
    }
    auto packet = make_packet(input, sequences);
    ZSTD_freeCCtx(context);
    return packet;
  } catch (...) {
    ZSTD_freeCCtx(context);
    throw;
  }
}

std::vector<std::uint8_t> sequence_inverse(std::span<const std::uint8_t> packet) {
  return expand(parse_packet(packet));
}

std::vector<std::uint8_t> entropy_frame_forward(std::span<const std::uint8_t> packet) {
  const Parsed parsed = parse_packet(packet);
  const auto source = expand(parsed);
  ZSTD_CCtx* context = ZSTD_createCCtx();
  if (context == nullptr) throw std::runtime_error("ZSTD_createCCtx failed");
  try {
    set_level_and_boundaries(context);
    std::vector<std::uint8_t> out(ZSTD_compressBound(source.size()));
    const auto size = ZSTD_compressSequences(
        context, out.data(), out.size(), parsed.sequences.data(),
        parsed.sequences.size(), source.data(), source.size());
    check(size, "ZSTD_compressSequences");
    out.resize(size);
    ZSTD_freeCCtx(context);
    return out;
  } catch (...) {
    ZSTD_freeCCtx(context);
    throw;
  }
}

std::vector<std::uint8_t> entropy_frame_inverse(std::span<const std::uint8_t> wire) {
  const auto size = ZSTD_getFrameContentSize(wire.data(), wire.size());
  if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN)
    throw std::runtime_error("Zstd frame has no exact content size");
  std::vector<std::uint8_t> raw(size);
  const auto decoded = ZSTD_decompress(raw.data(), raw.size(), wire.data(), wire.size());
  check(decoded, "ZSTD_decompress");
  if (decoded != raw.size()) throw std::runtime_error("Zstd decoded size mismatch");
  return sequence_forward(raw);
}

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> input) {
  return entropy_frame_forward(sequence_forward(input));
}
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire) {
  return sequence_inverse(entropy_frame_inverse(wire));
}

}  // namespace exact_actions::zstd
