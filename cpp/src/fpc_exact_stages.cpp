#include "fpc_exact_stages.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace exact_actions::fpc {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'F', 'P', 'R', '1'};
constexpr std::size_t kChunkValues = 32768;

struct Record {
  bool dfcm;
  std::uint64_t residual;
};
struct Packet {
  unsigned predictor_lg2;
  std::vector<Record> records;
};

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint64_t read_u64_le(std::span<const std::uint8_t> bytes,
                          std::size_t& pos) {
  if (bytes.size() - pos < 8) throw std::runtime_error("truncated FPC packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(bytes[pos++]) << shift;
  return value;
}
std::uint32_t read_u24_le(std::span<const std::uint8_t> bytes,
                          std::size_t offset) {
  if (bytes.size() - offset < 3) throw std::runtime_error("truncated FPC header");
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16);
}
void write_u24_le(std::vector<std::uint8_t>& bytes, std::size_t offset,
                  std::uint32_t value) {
  bytes[offset] = static_cast<std::uint8_t>(value);
  bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
  bytes[offset + 2] = static_cast<std::uint8_t>(value >> 16);
}

std::vector<std::uint8_t> serialize(const Packet& packet) {
  std::vector<std::uint8_t> out(kMagic.begin(), kMagic.end());
  out.push_back(static_cast<std::uint8_t>(packet.predictor_lg2));
  append_u64_le(out, packet.records.size());
  for (const auto& record : packet.records) {
    out.push_back(record.dfcm ? 1 : 0);
    append_u64_le(out, record.residual);
  }
  return out;
}

Packet parse(std::span<const std::uint8_t> value) {
  if (value.size() < 13 || !std::equal(kMagic.begin(), kMagic.end(), value.begin()))
    throw std::runtime_error("wrong FPC packet type");
  const unsigned lg2 = value[4];
  if (lg2 > 24) throw std::runtime_error("invalid FPC predictor size");
  std::size_t pos = 5;
  const auto count = read_u64_le(value, pos);
  if (count > (value.size() - pos) / 9 || count * 9 != value.size() - pos)
    throw std::runtime_error("invalid FPC record count");
  std::vector<Record> records;
  records.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto selector = value[pos++];
    if (selector > 1) throw std::runtime_error("invalid FPC predictor selector");
    records.push_back({selector != 0, read_u64_le(value, pos)});
  }
  return {lg2, std::move(records)};
}

struct Predictors {
  std::uint64_t mask;
  std::vector<std::uint64_t> fcm;
  std::vector<std::uint64_t> dfcm;
  std::uint64_t hash = 0, dhash = 0, last = 0, pred1 = 0, pred2 = 0;

  explicit Predictors(unsigned lg2)
      : mask((std::uint64_t{1} << lg2) - 1),
        fcm(static_cast<std::size_t>(mask + 1), 0),
        dfcm(static_cast<std::size_t>(mask + 1), 0) {}

  Record encode(std::uint64_t value) {
    const auto xor1 = value ^ pred1;
    const auto xor2 = value ^ (last + pred2);
    const Record record{xor1 > xor2, std::min(xor1, xor2)};
    update(value);
    return record;
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

void append_residual(std::vector<std::uint8_t>& out, std::uint64_t residual,
                     int code) {
  for (int i = 0; i < byte_count(code); ++i)
    out.push_back(static_cast<std::uint8_t>(residual >> (8 * i)));
}

Record read_record(std::span<const std::uint8_t> chunk, std::size_t& residual_pos,
                   int nibble) {
  const int code = nibble & 7;
  const int bytes = byte_count(code);
  if (chunk.size() - residual_pos < static_cast<std::size_t>(bytes))
    throw std::runtime_error("truncated FPC residual");
  std::uint64_t residual = 0;
  for (int i = 0; i < bytes; ++i)
    residual |= static_cast<std::uint64_t>(chunk[residual_pos++]) << (8 * i);
  return {(nibble & 8) != 0, residual};
}

}  // namespace

std::vector<std::uint8_t> predictor_xor_forward(
    std::span<const std::uint8_t> raw, unsigned predictor_lg2) {
  if (raw.size() % 8 != 0 || predictor_lg2 > 24)
    throw std::invalid_argument("FPC requires 64-bit-word bytes and predictor_lg2 <= 24");
  Packet packet{predictor_lg2, {}};
  if (raw.empty()) return serialize(packet);
  Predictors state(predictor_lg2);
  packet.records.reserve(raw.size() / 8);
  for (std::size_t pos = 0; pos < raw.size();) {
    packet.records.push_back(state.encode(read_u64_le(raw, pos)));
  }
  return serialize(packet);
}

std::vector<std::uint8_t> predictor_xor_inverse(
    std::span<const std::uint8_t> value) {
  const auto packet = parse(value);
  if (packet.records.empty()) return {};
  Predictors state(packet.predictor_lg2);
  std::vector<std::uint8_t> raw;
  raw.reserve(packet.records.size() * 8);
  for (const auto& record : packet.records) append_u64_le(raw, state.decode(record));
  return raw;
}

std::vector<std::uint8_t> trailing_pack_forward(
    std::span<const std::uint8_t> value) {
  const auto packet = parse(value);
  // The native encoder writes its predictor-size byte even when there are
  // no values, then emits no chunks. This is the existing empty wire format.
  if (packet.records.empty())
    return {static_cast<std::uint8_t>(packet.predictor_lg2)};
  Predictors state(packet.predictor_lg2);
  std::vector<std::uint8_t> out{static_cast<std::uint8_t>(packet.predictor_lg2)};
  for (std::size_t start = 0; start < packet.records.size(); start += kChunkValues) {
    const std::size_t count = std::min(kChunkValues, packet.records.size() - start);
    std::vector<std::uint8_t> chunk(6 + ((count + 1) >> 1), 0);
    for (std::size_t i = 0; i < count; i += 2) {
      const auto first = packet.records[start + i];
      state.decode(first);
      const int first_code = byte_code(first.residual);
      int code = (first.dfcm ? 0x80 : 0) | (first_code << 4);
      append_residual(chunk, first.residual, first_code);
      if (i + 1 < count) {
        const auto second = packet.records[start + i + 1];
        state.decode(second);
        const int second_code = byte_code(second.residual);
        code |= (second.dfcm ? 0x08 : 0) | second_code;
        append_residual(chunk, second.residual, second_code);
      } else {
        // The reference encoder computes a padded zero record solely for the
        // low nibble, then discards its residual bytes.
        const auto dummy = state.encode(0);
        code |= (dummy.dfcm ? 0x08 : 0) | byte_code(dummy.residual);
      }
      chunk[6 + (i >> 1)] = static_cast<std::uint8_t>(code);
    }
    write_u24_le(chunk, 0, static_cast<std::uint32_t>(count));
    write_u24_le(chunk, 3, static_cast<std::uint32_t>(chunk.size()));
    out.insert(out.end(), chunk.begin(), chunk.end());
  }
  return out;
}

std::vector<std::uint8_t> trailing_pack_inverse(
    std::span<const std::uint8_t> wire) {
  if (wire.empty() || wire[0] > 24) throw std::runtime_error("invalid FPC wire");
  Packet packet{wire[0], {}};
  std::size_t pos = 1;
  while (pos < wire.size()) {
    if (wire.size() - pos < 6) throw std::runtime_error("truncated FPC chunk");
    const auto count = read_u24_le(wire, pos);
    const auto size = read_u24_le(wire, pos + 3);
    if (count == 0 || count > kChunkValues || size < 6 + ((count + 1) >> 1) ||
        size > wire.size() - pos)
      throw std::runtime_error("invalid FPC chunk header");
    const auto chunk = wire.subspan(pos, size);
    std::size_t residual_pos = 6 + ((count + 1) >> 1);
    for (std::size_t i = 0; i < count; i += 2) {
      const int code = chunk[6 + (i >> 1)];
      packet.records.push_back(read_record(chunk, residual_pos, code >> 4));
      if (i + 1 < count)
        packet.records.push_back(read_record(chunk, residual_pos, code & 0x0f));
    }
    if (residual_pos != chunk.size()) throw std::runtime_error("FPC chunk size mismatch");
    pos += size;
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

}  // namespace exact_actions::fpc
