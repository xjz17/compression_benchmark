#include "pfpc_exact_stages.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace exact_actions::pfpc {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'P', 'F', 'R', '1'};
constexpr std::size_t kMaxBlockValues = 1u << 20;
constexpr std::size_t kMaxChunkParameter = 1u << 20;
struct Record { bool dfcm; std::uint64_t residual; };
struct Packet {
  unsigned predictor_lg2;
  std::uint32_t chunk_parameter;
  std::vector<Record> records;
};

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint32_t read_u32_le(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  if (bytes.size() - pos < 4) throw std::runtime_error("truncated pFPC u32");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(bytes[pos++]) << shift;
  return value;
}
std::uint64_t read_u64_le(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  if (bytes.size() - pos < 8) throw std::runtime_error("truncated pFPC u64");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(bytes[pos++]) << shift;
  return value;
}
void patch_u32_le(std::vector<std::uint8_t>& out, std::size_t pos,
                  std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out[pos++] = static_cast<std::uint8_t>(value >> shift);
}

std::vector<std::uint8_t> serialize(const Packet& packet) {
  std::vector<std::uint8_t> out(kMagic.begin(), kMagic.end());
  out.push_back(static_cast<std::uint8_t>(packet.predictor_lg2));
  append_u32_le(out, packet.chunk_parameter);
  append_u64_le(out, packet.records.size());
  for (const auto& record : packet.records) {
    out.push_back(record.dfcm ? 1 : 0);
    append_u64_le(out, record.residual);
  }
  return out;
}
Packet parse(std::span<const std::uint8_t> value) {
  if (value.size() < 17 || !std::equal(kMagic.begin(), kMagic.end(), value.begin()))
    throw std::runtime_error("wrong pFPC packet type");
  Packet packet{value[4], 0, {}};
  if (packet.predictor_lg2 > 24) throw std::runtime_error("bad pFPC predictor size");
  std::size_t pos = 5;
  packet.chunk_parameter = read_u32_le(value, pos);
  const auto count = read_u64_le(value, pos);
  if (packet.chunk_parameter == 0 || count > (value.size() - pos) / 9 ||
      count * 9 != value.size() - pos)
    throw std::runtime_error("invalid pFPC packet");
  packet.records.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto selector = value[pos++];
    if (selector > 1) throw std::runtime_error("invalid pFPC selector");
    packet.records.push_back({selector != 0, read_u64_le(value, pos)});
  }
  return packet;
}

struct Predictors {
  std::uint64_t mask;
  std::vector<std::uint64_t> fcm, dfcm;
  std::uint64_t hash = 0, dhash = 0, last = 0, pred1 = 0, pred2 = 0;
  explicit Predictors(unsigned lg2)
      : mask((std::uint64_t{1} << lg2) - 1),
        fcm(static_cast<std::size_t>(mask + 1), 0),
        dfcm(static_cast<std::size_t>(mask + 1), 0) {}
  Record encode(std::uint64_t value) {
    const auto a = value ^ pred1;
    const auto b = value ^ (last + pred2);
    Record result{a > b, std::min(a, b)};
    update(value);
    return result;
  }
  std::uint64_t decode(const Record& record) {
    const auto value = record.residual ^ (record.dfcm ? last + pred2 : pred1);
    update(value);
    return value;
  }
  void update(std::uint64_t value) {
    fcm[hash] = value;
    hash = ((hash << 6) ^ (value >> 48)) & mask;
    pred1 = fcm[hash];
    const auto stride = value - last;
    last = value;
    dfcm[dhash] = stride;
    dhash = ((dhash << 2) ^ (stride >> 40)) & mask;
    pred2 = dfcm[dhash];
  }
};

int byte_code(std::uint64_t value) {
  if (value == 0) return 0;
  if ((value >> 8) == 0) return 1;
  if ((value >> 16) == 0) return 2;
  if ((value >> 24) == 0) return 3;
  if ((value >> 40) == 0) return 4;
  if ((value >> 48) == 0) return 5;
  if ((value >> 56) == 0) return 6;
  return 7;
}
int byte_count(int code) { return code + (code >> 2); }
void append_residual(std::vector<std::uint8_t>& out, const Record& record) {
  const int count = byte_count(byte_code(record.residual));
  for (int i = 0; i < count; ++i)
    out.push_back(static_cast<std::uint8_t>(record.residual >> (8 * i)));
}
Record read_record(std::span<const std::uint8_t> chunk, std::size_t& pos,
                   int nibble) {
  const int count = byte_count(nibble & 7);
  if (chunk.size() - pos < static_cast<std::size_t>(count))
    throw std::runtime_error("truncated pFPC residual");
  std::uint64_t residual = 0;
  for (int i = 0; i < count; ++i)
    residual |= static_cast<std::uint64_t>(chunk[pos++]) << (8 * i);
  return {(nibble & 8) != 0, residual};
}

}  // namespace

std::vector<std::uint8_t> predictor_xor_forward(
    std::span<const std::uint8_t> raw, unsigned predictor_lg2) {
  if (raw.empty() || raw.size() % 8 != 0 || predictor_lg2 > 24)
    throw std::invalid_argument("pFPC requires non-empty 64-bit-word bytes");
  Packet packet{predictor_lg2,
                static_cast<std::uint32_t>(std::min(raw.size(), kMaxChunkParameter)), {}};
  const std::size_t count = raw.size() / 8;
  packet.records.reserve(count);
  std::size_t byte_pos = 0;
  for (std::size_t block = 0; block < count; block += kMaxBlockValues) {
    Predictors state(predictor_lg2);
    const auto end = std::min(count, block + kMaxBlockValues);
    for (std::size_t i = block; i < end; ++i)
      packet.records.push_back(state.encode(read_u64_le(raw, byte_pos)));
  }
  return serialize(packet);
}

std::vector<std::uint8_t> predictor_xor_inverse(
    std::span<const std::uint8_t> value) {
  const auto packet = parse(value);
  std::vector<std::uint8_t> raw;
  raw.reserve(packet.records.size() * 8);
  for (std::size_t block = 0; block < packet.records.size(); block += kMaxBlockValues) {
    Predictors state(packet.predictor_lg2);
    const auto end = std::min(packet.records.size(), block + kMaxBlockValues);
    for (std::size_t i = block; i < end; ++i)
      append_u64_le(raw, state.decode(packet.records[i]));
  }
  return raw;
}

std::vector<std::uint8_t> trailing_pack_forward(
    std::span<const std::uint8_t> value) {
  const auto packet = parse(value);
  if (packet.records.empty()) throw std::invalid_argument("empty pFPC packet");
  std::vector<std::uint8_t> out{
      static_cast<std::uint8_t>(packet.predictor_lg2), 1};
  append_u32_le(out, packet.chunk_parameter);
  for (std::size_t block = 0; block < packet.records.size(); block += kMaxBlockValues) {
    const std::size_t count = std::min(kMaxBlockValues, packet.records.size() - block);
    append_u32_le(out, static_cast<std::uint32_t>(count));
    const std::size_t chunk_start = out.size();
    append_u32_le(out, 0);
    const std::size_t codes_start = out.size();
    out.resize(out.size() + ((count + 1) >> 1), 0);
    for (std::size_t i = 0; i < count; ++i) {
      const auto& record = packet.records[block + i];
      const int nibble = (record.dfcm ? 8 : 0) | byte_code(record.residual);
      if ((i & 1) == 0) out[codes_start + (i >> 1)] = static_cast<std::uint8_t>(nibble << 4);
      else out[codes_start + (i >> 1)] |= static_cast<std::uint8_t>(nibble);
      append_residual(out, record);
    }
    patch_u32_le(out, chunk_start,
                 static_cast<std::uint32_t>(out.size() - chunk_start));
  }
  return out;
}

std::vector<std::uint8_t> trailing_pack_inverse(
    std::span<const std::uint8_t> wire) {
  if (wire.size() < 14 || wire[0] > 24 || wire[1] != 1)
    throw std::runtime_error("invalid pFPC wire");
  std::size_t pos = 2;
  Packet packet{wire[0], read_u32_le(wire, pos), {}};
  while (pos < wire.size()) {
    const auto count = read_u32_le(wire, pos);
    if (count == 0 || count > kMaxBlockValues)
      throw std::runtime_error("invalid pFPC block count");
    const std::size_t chunk_start = pos;
    const auto chunk_size = read_u32_le(wire, pos);
    if (chunk_size < 4 + ((count + 1) >> 1) || chunk_size > wire.size() - chunk_start)
      throw std::runtime_error("invalid pFPC chunk size");
    const auto chunk = wire.subspan(chunk_start, chunk_size);
    const std::size_t codes_start = 4;
    std::size_t residual_pos = 4 + ((count + 1) >> 1);
    for (std::size_t i = 0; i < count; ++i) {
      const int code_byte = chunk[codes_start + (i >> 1)];
      const int nibble = (i & 1) == 0 ? code_byte >> 4 : code_byte & 0x0f;
      packet.records.push_back(read_record(chunk, residual_pos, nibble));
    }
    if (residual_pos != chunk.size()) throw std::runtime_error("pFPC chunk size mismatch");
    pos = chunk_start + chunk_size;
  }
  return serialize(packet);
}

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw,
                                  unsigned predictor_lg2) {
  return trailing_pack_forward(predictor_xor_forward(raw, predictor_lg2));
}
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire) {
  return predictor_xor_inverse(trailing_pack_inverse(wire));
}

}  // namespace exact_actions::pfpc
