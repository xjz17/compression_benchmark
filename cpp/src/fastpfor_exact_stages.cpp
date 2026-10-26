#include "fastpfor_exact_stages.hpp"

#include "dbp_exact_stages.hpp"
#include "int64_column_codec.hpp"
#include "fastpfor.h"
#include "variablebyte.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::fastpfor {
namespace {

using FastPForLib::FastPFor;
using FastPForLib::VariableByte;

constexpr std::array<std::uint8_t, 4> kStageMagic{'P', 'F', 'S', '1'};
constexpr std::uint8_t kIntKind = 0;
constexpr std::uint8_t kScaledKind = 1;
constexpr std::size_t kBlockSize = FastPFor<8>::BlockSize;

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> data, std::size_t& pos,
                      const char* label) {
  if (data.size() - pos < 4)
    throw std::runtime_error(std::string("truncated FastPFor ") + label);
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(data[pos++]) << shift;
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> data, std::size_t& pos,
                      const char* label) {
  if (data.size() - pos < 8)
    throw std::runtime_error(std::string("truncated FastPFor ") + label);
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

struct LimbStage {
  std::vector<std::uint32_t> pfor_words;
  std::vector<std::uint32_t> raw_tail;
};

struct StagePacket {
  std::uint8_t kind{};
  int scale{};
  std::size_t count{};
  LimbStage low;
  LimbStage high;
};

std::vector<std::uint32_t> split_limb(
    std::span<const std::int64_t> values, bool high) {
  std::vector<std::uint32_t> out;
  out.reserve(values.size());
  for (const auto value : values) {
    const auto bits = static_cast<std::uint64_t>(value);
    out.push_back(static_cast<std::uint32_t>(high ? bits >> 32 : bits));
  }
  return out;
}

std::vector<std::uint32_t> encode_pfor_prefix(
    std::span<const std::uint32_t> limb, std::size_t prefix_count) {
  if (limb.empty()) return {};
  const auto capacity = std::max<std::size_t>(
      4096U, std::max(prefix_count * 2U, prefix_count + 4096U));
  std::vector<std::uint32_t> out(capacity);
  std::size_t words = out.size();
  FastPFor<8> codec;
  codec.encodeArray(limb.data(), prefix_count, out.data(), words);
  out.resize(words);
  return out;
}

std::vector<std::uint32_t> decode_pfor_prefix(
    std::span<const std::uint32_t> words, std::size_t prefix_count) {
  if (words.empty()) {
    if (prefix_count != 0)
      throw std::runtime_error("missing FastPFor prefix words");
    return {};
  }
  std::vector<std::uint32_t> decoded(std::max<std::size_t>(prefix_count, 1U));
  std::size_t count = prefix_count;
  FastPFor<8> codec;
  const auto* end = codec.decodeArray(
      words.data(), words.size(), decoded.data(), count);
  if (count != prefix_count || end != words.data() + words.size())
    throw std::runtime_error("FastPFor prefix shape mismatch");
  decoded.resize(prefix_count);
  return decoded;
}

LimbStage encode_limb(std::span<const std::uint32_t> limb) {
  const auto prefix_count = limb.size() / kBlockSize * kBlockSize;
  LimbStage stage;
  stage.pfor_words = encode_pfor_prefix(limb, prefix_count);
  stage.raw_tail.assign(limb.begin() + static_cast<std::ptrdiff_t>(prefix_count),
                        limb.end());
  return stage;
}

void put_limb(std::vector<std::uint8_t>& out, const LimbStage& limb) {
  put_u64(out, limb.pfor_words.size());
  for (const auto value : limb.pfor_words) put_u32(out, value);
  put_u64(out, limb.raw_tail.size());
  for (const auto value : limb.raw_tail) put_u32(out, value);
}

LimbStage get_limb(std::span<const std::uint8_t> packet, std::size_t& pos,
                   std::size_t expected_tail) {
  LimbStage limb;
  const auto pfor_count = get_u64(packet, pos, "PFor word count");
  if (pfor_count > (packet.size() - pos) / 4)
    throw std::runtime_error("invalid FastPFor word count");
  limb.pfor_words.reserve(static_cast<std::size_t>(pfor_count));
  for (std::uint64_t i = 0; i < pfor_count; ++i)
    limb.pfor_words.push_back(get_u32(packet, pos, "PFor word"));
  const auto tail_count = get_u64(packet, pos, "raw tail count");
  if (tail_count != expected_tail || tail_count > (packet.size() - pos) / 4)
    throw std::runtime_error("invalid FastPFor raw tail count");
  limb.raw_tail.reserve(static_cast<std::size_t>(tail_count));
  for (std::uint64_t i = 0; i < tail_count; ++i)
    limb.raw_tail.push_back(get_u32(packet, pos, "raw tail value"));
  return limb;
}

std::vector<std::uint8_t> serialize_stage(const StagePacket& stage) {
  if (stage.kind > kScaledKind)
    throw std::runtime_error("invalid FastPFor stage kind");
  std::vector<std::uint8_t> out(kStageMagic.begin(), kStageMagic.end());
  out.push_back(stage.kind);
  put_u32(out, static_cast<std::uint32_t>(stage.scale));
  put_u64(out, stage.count);
  put_limb(out, stage.low);
  put_limb(out, stage.high);
  return out;
}

StagePacket parse_stage(std::span<const std::uint8_t> packet) {
  if (packet.size() < kStageMagic.size() ||
      !std::equal(kStageMagic.begin(), kStageMagic.end(), packet.begin()))
    throw std::runtime_error("wrong FastPFor stage packet type");
  std::size_t pos = kStageMagic.size();
  if (pos >= packet.size()) throw std::runtime_error("truncated FastPFor kind");
  StagePacket stage;
  stage.kind = packet[pos++];
  if (stage.kind > kScaledKind)
    throw std::runtime_error("invalid FastPFor stage kind");
  stage.scale = static_cast<int>(static_cast<std::int32_t>(
      get_u32(packet, pos, "scale")));
  const auto count = get_u64(packet, pos, "value count");
  if (count > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("FastPFor stage count exceeds size_t");
  stage.count = static_cast<std::size_t>(count);
  const auto prefix_count = stage.count / kBlockSize * kBlockSize;
  const auto tail_count = stage.count - prefix_count;
  stage.low = get_limb(packet, pos, tail_count);
  stage.high = get_limb(packet, pos, tail_count);
  if (pos != packet.size())
    throw std::runtime_error("FastPFor stage packet trailing bytes");
  if (stage.count == 0 &&
      (!stage.low.pfor_words.empty() || !stage.high.pfor_words.empty()))
    throw std::runtime_error("empty FastPFor stage has encoded prefix words");
  if (stage.count != 0 &&
      (stage.low.pfor_words.empty() || stage.high.pfor_words.empty()))
    throw std::runtime_error("non-empty FastPFor stage lacks prefix marker");
  return stage;
}

StagePacket make_stage(std::span<const std::int64_t> values,
                       std::uint8_t kind, int scale) {
  const auto low = split_limb(values, false);
  const auto high = split_limb(values, true);
  return {kind, scale, values.size(), encode_limb(low), encode_limb(high)};
}

std::vector<std::int64_t> restore_values(const StagePacket& stage) {
  const auto prefix_count = stage.count / kBlockSize * kBlockSize;
  auto low = decode_pfor_prefix(stage.low.pfor_words, prefix_count);
  auto high = decode_pfor_prefix(stage.high.pfor_words, prefix_count);
  low.insert(low.end(), stage.low.raw_tail.begin(), stage.low.raw_tail.end());
  high.insert(high.end(), stage.high.raw_tail.begin(), stage.high.raw_tail.end());
  if (low.size() != stage.count || high.size() != stage.count)
    throw std::runtime_error("FastPFor reconstructed limb length mismatch");
  std::vector<std::int64_t> values(stage.count);
  for (std::size_t i = 0; i < stage.count; ++i) {
    const auto bits = (static_cast<std::uint64_t>(high[i]) << 32) | low[i];
    std::memcpy(&values[i], &bits, sizeof(bits));
  }
  return values;
}

std::vector<std::uint32_t> encode_variable_byte(
    std::span<const std::uint32_t> tail) {
  if (tail.empty()) return {};
  std::vector<std::uint32_t> out(std::max<std::size_t>(16U, tail.size() * 2U));
  std::size_t words = out.size();
  VariableByte codec;
  codec.encodeArray(tail.data(), tail.size(), out.data(), words);
  out.resize(words);
  return out;
}

std::vector<std::uint32_t> limb_public_words(const LimbStage& stage) {
  auto out = stage.pfor_words;
  const auto tail = encode_variable_byte(stage.raw_tail);
  out.insert(out.end(), tail.begin(), tail.end());
  return out;
}

LimbStage split_public_limb(std::span<const std::uint32_t> words,
                            std::size_t count) {
  if (count == 0) {
    if (!words.empty()) throw std::runtime_error("empty FastPFor limb has words");
    return {};
  }
  const auto prefix_count = count / kBlockSize * kBlockSize;
  std::vector<std::uint32_t> decoded(std::max<std::size_t>(prefix_count, 1U));
  std::size_t decoded_count = prefix_count;
  FastPFor<8> pfor;
  const auto* tail_begin = pfor.decodeArray(
      words.data(), words.size(), decoded.data(), decoded_count);
  if (decoded_count != prefix_count || tail_begin < words.data() ||
      tail_begin > words.data() + words.size())
    throw std::runtime_error("invalid FastPFor public prefix");
  const auto pfor_word_count = static_cast<std::size_t>(tail_begin - words.data());
  LimbStage stage;
  stage.pfor_words.assign(words.begin(),
                          words.begin() + static_cast<std::ptrdiff_t>(pfor_word_count));
  const auto tail_word_count = words.size() - pfor_word_count;
  const auto expected_tail = count - prefix_count;
  if (expected_tail == 0) {
    if (tail_word_count != 0)
      throw std::runtime_error("FastPFor limb has unexpected VariableByte tail");
    return stage;
  }
  std::vector<std::uint32_t> tail_values(
      std::max<std::size_t>(expected_tail, tail_word_count * 4U));
  std::size_t tail_count = tail_values.size();
  VariableByte variable_byte;
  variable_byte.decodeArray(tail_begin, tail_word_count,
                            tail_values.data(), tail_count);
  if (tail_count != expected_tail)
    throw std::runtime_error("FastPFor VariableByte tail length mismatch");
  tail_values.resize(tail_count);
  stage.raw_tail = std::move(tail_values);
  return stage;
}

std::vector<std::uint32_t> body_words(std::span<const std::uint8_t> body) {
  if (body.size() % 4 != 0)
    throw std::runtime_error("FastPFor public body is not word aligned");
  std::vector<std::uint32_t> words;
  words.reserve(body.size() / 4);
  std::size_t pos = 0;
  while (pos < body.size()) words.push_back(get_u32(body, pos, "public word"));
  return words;
}

}  // namespace

std::vector<std::uint8_t> pfor_stage_forward_int(
    std::span<const std::int64_t> values) {
  return serialize_stage(make_stage(values, kIntKind, 0));
}

std::vector<std::uint8_t> pfor_stage_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  const auto [scale, values] = exact_actions::dbp::scaling_values(scaling_packet);
  return serialize_stage(make_stage(values, kScaledKind, scale));
}

std::vector<std::int64_t> pfor_stage_inverse_int(
    std::span<const std::uint8_t> packet) {
  const auto stage = parse_stage(packet);
  if (stage.kind != kIntKind)
    throw std::runtime_error("wrong integer FastPFor stage kind");
  return restore_values(stage);
}

std::vector<std::uint8_t> pfor_stage_inverse_scaled(
    std::span<const std::uint8_t> packet) {
  const auto stage = parse_stage(packet);
  if (stage.kind != kScaledKind)
    throw std::runtime_error("wrong scaled FastPFor stage kind");
  const auto values = restore_values(stage);
  return exact_actions::dbp::make_scaling_packet(stage.scale, values);
}

std::vector<std::uint8_t> variable_byte_forward(
    std::span<const std::uint8_t> packet) {
  const auto stage = parse_stage(packet);
  const auto low = limb_public_words(stage.low);
  const auto high = limb_public_words(stage.high);
  if (low.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("FastPFor low limb exceeds public word count");
  std::vector<std::uint8_t> out;
  out.push_back(stage.kind == kIntKind ? 'I' : 'F');
  if (stage.kind == kScaledKind)
    put_u32(out, static_cast<std::uint32_t>(stage.scale));
  put_u32(out, static_cast<std::uint32_t>(low.size()));
  for (const auto value : low) put_u32(out, value);
  for (const auto value : high) put_u32(out, value);
  return out;
}

std::vector<std::uint8_t> variable_byte_inverse(
    std::span<const std::uint8_t> wire, std::size_t expected_count) {
  if (wire.empty() || (wire.front() != 'I' && wire.front() != 'F'))
    throw std::runtime_error("wrong FastPFor public wire type");
  StagePacket stage;
  stage.kind = wire.front() == 'I' ? kIntKind : kScaledKind;
  stage.count = expected_count;
  std::size_t pos = 1;
  if (stage.kind == kScaledKind)
    stage.scale = static_cast<int>(static_cast<std::int32_t>(
        get_u32(wire, pos, "public scale")));
  const auto words = body_words(wire.subspan(pos));
  if (words.empty()) throw std::runtime_error("missing FastPFor limb header");
  const auto low_count = words.front();
  if (low_count > words.size() - 1)
    throw std::runtime_error("invalid FastPFor low-limb word count");
  stage.low = split_public_limb(
      std::span<const std::uint32_t>(words).subspan(1, low_count), expected_count);
  stage.high = split_public_limb(
      std::span<const std::uint32_t>(words).subspan(1 + low_count), expected_count);
  return serialize_stage(stage);
}

std::vector<std::uint8_t> adapter_forward_int(
    std::span<const std::int64_t> values) {
  const auto body = int64_column_codec::fast_pfor_encode_int64(
      values.data(), values.size());
  std::vector<std::uint8_t> out{'I'};
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::int64_t> adapter_inverse_int(
    std::span<const std::uint8_t> wire, std::size_t expected_count) {
  if (wire.empty() || wire.front() != 'I')
    throw std::runtime_error("wrong integer FastPFor adapter wire type");
  return int64_column_codec::fast_pfor_decode_int64(
      wire.data() + 1, wire.size() - 1, expected_count);
}

std::vector<std::uint8_t> adapter_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  const auto [scale, values] = exact_actions::dbp::scaling_values(scaling_packet);
  const auto body = int64_column_codec::fast_pfor_encode_int64(
      values.data(), values.size());
  std::vector<std::uint8_t> out{'F'};
  const auto raw_scale = static_cast<std::uint32_t>(scale);
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(raw_scale >> shift));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> adapter_inverse_scaled(
    std::span<const std::uint8_t> wire, std::size_t expected_count) {
  if (wire.size() < 5 || wire.front() != 'F')
    throw std::runtime_error("wrong floating FastPFor adapter wire type");
  std::uint32_t raw_scale = 0;
  for (int i = 0; i < 4; ++i)
    raw_scale |= static_cast<std::uint32_t>(wire[1 + i]) << (8 * i);
  const int scale = static_cast<int>(static_cast<std::int32_t>(raw_scale));
  const auto values = int64_column_codec::fast_pfor_decode_int64(
      wire.data() + 5, wire.size() - 5, expected_count);
  return exact_actions::dbp::make_scaling_packet(scale, values);
}

}  // namespace exact_actions::fastpfor
