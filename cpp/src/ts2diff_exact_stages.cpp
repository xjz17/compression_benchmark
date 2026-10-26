#include "ts2diff_exact_stages.hpp"
#include "minimum_for.hpp"
#include "max_value_bitpack.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::ts2diff {
namespace {

constexpr std::size_t kBlockSize = 128;
constexpr std::array<std::uint8_t, 4> kScalingMagic{'S', '2', 'S', '1'};
constexpr std::array<std::uint8_t, 4> kDeltaMagic{'S', '2', 'D', '1'};
constexpr std::array<std::uint8_t, 4> kForMagic{'S', '2', 'F', '1'};

struct Metadata {
  bool floating_point{};
  std::uint32_t max_point{};
};

struct DeltaBlock {
  std::int64_t first{};
  std::vector<std::int64_t> deltas;
};

struct ForBlock {
  std::int64_t first{};
  std::int64_t minimum{};
  std::vector<std::uint64_t> normalized;
};

struct DeltaPacket {
  Metadata metadata;
  std::vector<DeltaBlock> blocks;
};

struct ForPacket {
  Metadata metadata;
  std::vector<ForBlock> blocks;
};

class Cursor {
 public:
  explicit Cursor(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  void require(std::size_t n) const {
    if (n > bytes_.size() - pos_) throw std::runtime_error("truncated stage packet");
  }
  std::uint8_t u8() {
    require(1);
    return bytes_[pos_++];
  }
  std::uint32_t u32_le() {
    require(4);
    std::uint32_t value = 0;
    for (int shift = 0; shift <= 24; shift += 8) {
      value |= static_cast<std::uint32_t>(bytes_[pos_++]) << shift;
    }
    return value;
  }
  std::uint64_t u64_le() {
    require(8);
    std::uint64_t value = 0;
    for (int shift = 0; shift <= 56; shift += 8) {
      value |= static_cast<std::uint64_t>(bytes_[pos_++]) << shift;
    }
    return value;
  }
  std::uint32_t u32_be() {
    require(4);
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value = (value << 8) | bytes_[pos_++];
    return value;
  }
  std::uint64_t u64_be() {
    require(8);
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value = (value << 8) | bytes_[pos_++];
    return value;
  }
  std::uint32_t uvarint() {
    std::uint32_t value = 0;
    int shift = 0;
    while (shift <= 28) {
      const std::uint8_t byte = u8();
      value |= static_cast<std::uint32_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return value;
      shift += 7;
    }
    throw std::runtime_error("invalid uvarint");
  }
  std::span<const std::uint8_t> take(std::size_t n) {
    require(n);
    const auto result = bytes_.subspan(pos_, n);
    pos_ += n;
    return result;
  }
  bool empty() const { return pos_ == bytes_.size(); }
  std::size_t remaining() const { return bytes_.size() - pos_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t pos_{};
};

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift <= 24; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift <= 56; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u32_be(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64_be(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_uvarint(std::vector<std::uint8_t>& out, std::uint32_t value) {
  while ((value & 0xffffff80u) != 0) {
    out.push_back(static_cast<std::uint8_t>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(value));
}

template <std::size_t N>
void append_magic(std::vector<std::uint8_t>& out,
                  const std::array<std::uint8_t, N>& magic) {
  out.insert(out.end(), magic.begin(), magic.end());
}

template <std::size_t N>
void expect_magic(Cursor& cursor, const std::array<std::uint8_t, N>& magic) {
  const auto actual = cursor.take(N);
  if (!std::equal(actual.begin(), actual.end(), magic.begin())) {
    throw std::runtime_error("wrong stage packet type");
  }
}

void append_metadata(std::vector<std::uint8_t>& out, Metadata metadata) {
  out.push_back(metadata.floating_point ? 1 : 0);
  append_u32_le(out, metadata.max_point);
}

Metadata read_metadata(Cursor& cursor) {
  const std::uint8_t kind = cursor.u8();
  if (kind > 1) throw std::runtime_error("invalid stage logical type");
  Metadata metadata{kind == 1, cursor.u32_le()};
  if (!metadata.floating_point && metadata.max_point != 0) {
    throw std::runtime_error("integer stage packet has a decimal scale");
  }
  return metadata;
}

std::int64_t checked_sub(std::int64_t left, std::int64_t right) {
  const __int128 value = static_cast<__int128>(left) - right;
  if (value < std::numeric_limits<std::int64_t>::min() ||
      value > std::numeric_limits<std::int64_t>::max()) {
    throw std::overflow_error("TS_2DIFF delta exceeds int64");
  }
  return static_cast<std::int64_t>(value);
}

std::int64_t checked_add(std::int64_t left, std::int64_t right) {
  const __int128 value = static_cast<__int128>(left) + right;
  if (value < std::numeric_limits<std::int64_t>::min() ||
      value > std::numeric_limits<std::int64_t>::max()) {
    throw std::overflow_error("TS_2DIFF reconstruction exceeds int64");
  }
  return static_cast<std::int64_t>(value);
}

std::vector<std::uint8_t> serialize_scaling(Metadata metadata,
                                             std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> out;
  append_magic(out, kScalingMagic);
  append_metadata(out, metadata);
  append_u64_le(out, values.size());
  for (const auto value : values) append_u64_le(out, static_cast<std::uint64_t>(value));
  return out;
}

std::pair<Metadata, std::vector<std::int64_t>> parse_scaling(
    std::span<const std::uint8_t> packet) {
  Cursor cursor(packet);
  expect_magic(cursor, kScalingMagic);
  Metadata metadata = read_metadata(cursor);
  if (!metadata.floating_point) throw std::runtime_error("not a floating scaling packet");
  const std::uint64_t count = cursor.u64_le();
  if (count > cursor.remaining() / 8 || count * 8 != cursor.remaining()) {
    throw std::runtime_error("invalid scaling packet value count");
  }
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    values.push_back(static_cast<std::int64_t>(cursor.u64_le()));
  }
  return {metadata, std::move(values)};
}

std::vector<std::uint8_t> serialize_delta(const DeltaPacket& packet) {
  std::vector<std::uint8_t> out;
  append_magic(out, kDeltaMagic);
  append_metadata(out, packet.metadata);
  append_u32_le(out, static_cast<std::uint32_t>(packet.blocks.size()));
  for (const auto& block : packet.blocks) {
    append_u64_le(out, static_cast<std::uint64_t>(block.first));
    append_u32_le(out, static_cast<std::uint32_t>(block.deltas.size()));
    for (const auto value : block.deltas) append_u64_le(out, static_cast<std::uint64_t>(value));
  }
  return out;
}

DeltaPacket parse_delta(std::span<const std::uint8_t> bytes) {
  Cursor cursor(bytes);
  expect_magic(cursor, kDeltaMagic);
  DeltaPacket packet;
  packet.metadata = read_metadata(cursor);
  const std::uint32_t block_count = cursor.u32_le();
  packet.blocks.reserve(block_count);
  for (std::uint32_t b = 0; b < block_count; ++b) {
    DeltaBlock block;
    block.first = static_cast<std::int64_t>(cursor.u64_le());
    const std::uint32_t count = cursor.u32_le();
    if (count > kBlockSize || static_cast<std::uint64_t>(count) * 8 > cursor.remaining()) {
      throw std::runtime_error("invalid delta block size");
    }
    block.deltas.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      block.deltas.push_back(static_cast<std::int64_t>(cursor.u64_le()));
    }
    packet.blocks.push_back(std::move(block));
  }
  if (!cursor.empty()) throw std::runtime_error("trailing delta packet bytes");
  return packet;
}

std::vector<std::uint8_t> serialize_for(const ForPacket& packet) {
  std::vector<std::uint8_t> out;
  append_magic(out, kForMagic);
  append_metadata(out, packet.metadata);
  append_u32_le(out, static_cast<std::uint32_t>(packet.blocks.size()));
  for (const auto& block : packet.blocks) {
    append_u64_le(out, static_cast<std::uint64_t>(block.first));
    append_u64_le(out, static_cast<std::uint64_t>(block.minimum));
    append_u32_le(out, static_cast<std::uint32_t>(block.normalized.size()));
    for (const auto value : block.normalized) append_u64_le(out, value);
  }
  return out;
}

ForPacket parse_for(std::span<const std::uint8_t> bytes) {
  Cursor cursor(bytes);
  expect_magic(cursor, kForMagic);
  ForPacket packet;
  packet.metadata = read_metadata(cursor);
  const std::uint32_t block_count = cursor.u32_le();
  packet.blocks.reserve(block_count);
  for (std::uint32_t b = 0; b < block_count; ++b) {
    ForBlock block;
    block.first = static_cast<std::int64_t>(cursor.u64_le());
    block.minimum = static_cast<std::int64_t>(cursor.u64_le());
    const std::uint32_t count = cursor.u32_le();
    if (count > kBlockSize || static_cast<std::uint64_t>(count) * 8 > cursor.remaining()) {
      throw std::runtime_error("invalid FOR block size");
    }
    block.normalized.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) block.normalized.push_back(cursor.u64_le());
    packet.blocks.push_back(std::move(block));
  }
  if (!cursor.empty()) throw std::runtime_error("trailing FOR packet bytes");
  return packet;
}

DeltaPacket make_delta(Metadata metadata, std::span<const std::int64_t> values) {
  DeltaPacket packet{metadata, {}};
  for (std::size_t offset = 0; offset < values.size(); offset += kBlockSize + 1) {
    const std::size_t end = std::min(values.size(), offset + kBlockSize + 1);
    DeltaBlock block;
    block.first = values[offset];
    block.deltas.reserve(end - offset - 1);
    std::int64_t previous = block.first;
    for (std::size_t i = offset + 1; i < end; ++i) {
      block.deltas.push_back(checked_sub(values[i], previous));
      previous = values[i];
    }
    packet.blocks.push_back(std::move(block));
  }
  return packet;
}

std::vector<std::int64_t> restore_values(const DeltaPacket& packet) {
  std::vector<std::int64_t> values;
  for (const auto& block : packet.blocks) {
    values.push_back(block.first);
    std::int64_t previous = block.first;
    for (const auto delta : block.deltas) {
      previous = checked_add(previous, delta);
      values.push_back(previous);
    }
  }
  return values;
}

ForPacket make_for(const DeltaPacket& delta) {
  ForPacket packet{delta.metadata, {}};
  packet.blocks.reserve(delta.blocks.size());
  for (const auto& source : delta.blocks) {
    ForBlock block;
    block.first = source.first;
    auto encoded = minimum_for::encode<std::int64_t>(source.deltas);
    // The historical TSFile empty-delta block has a distinct wire sentinel.
    block.minimum = source.deltas.empty() ? std::numeric_limits<std::int64_t>::max()
                                          : encoded.minimum;
    block.normalized = std::move(encoded.offsets);
    packet.blocks.push_back(std::move(block));
  }
  return packet;
}

DeltaPacket restore_delta(const ForPacket& packet) {
  DeltaPacket delta{packet.metadata, {}};
  delta.blocks.reserve(packet.blocks.size());
  for (const auto& source : packet.blocks) {
    DeltaBlock block;
    block.first = source.first;
    block.deltas.reserve(source.normalized.size());
    for (const auto value : source.normalized) {
      const __int128 restored = static_cast<__int128>(source.minimum) + value;
      if (restored < std::numeric_limits<std::int64_t>::min() ||
          restored > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("TS_2DIFF FOR inverse exceeds int64");
      }
      block.deltas.push_back(minimum_for::restore(source.minimum, value));
    }
    delta.blocks.push_back(std::move(block));
  }
  return delta;
}


}  // namespace

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int max_point) {
  if (max_point < 0 || max_point > 18) throw std::invalid_argument("max_point must be in [0,18]");
  const double power = std::pow(10.0, max_point);
  std::vector<std::int64_t> scaled;
  scaled.reserve(values.size());
  for (const double value : values) {
    if (!std::isfinite(value)) throw std::invalid_argument("TS_2DIFF scaling requires finite values");
    const long double product = static_cast<long double>(value) * power;
    // The power-of-two bound is exact even when long double has 53-bit precision.
    // Converting INT64_MAX to that format rounds it up and admits +2^63.
    if (product < -0x1p63L || product >= 0x1p63L) {
      throw std::overflow_error("TS_2DIFF scaled value exceeds int64");
    }
    const auto encoded = static_cast<std::int64_t>(std::llround(product));
    const double recovered = static_cast<double>(encoded) / power;
    if (std::bit_cast<std::uint64_t>(recovered) != std::bit_cast<std::uint64_t>(value)) {
      throw std::invalid_argument("TS_2DIFF decimal scaling is not bitwise reversible");
    }
    scaled.push_back(encoded);
  }
  return serialize_scaling(Metadata{true, static_cast<std::uint32_t>(max_point)}, scaled);
}

std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet) {
  const auto [metadata, values] = parse_scaling(packet);
  const double power = std::pow(10.0, metadata.max_point);
  std::vector<double> restored;
  restored.reserve(values.size());
  for (const auto value : values) restored.push_back(static_cast<double>(value) / power);
  return restored;
}

std::vector<std::uint8_t> delta_forward_int64(std::span<const std::int64_t> values) {
  return serialize_delta(make_delta(Metadata{false, 0}, values));
}

std::vector<std::uint8_t> delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  const auto [metadata, values] = parse_scaling(scaling_packet);
  return serialize_delta(make_delta(metadata, values));
}

std::vector<std::int64_t> delta_inverse_int64(std::span<const std::uint8_t> packet) {
  const auto parsed = parse_delta(packet);
  if (parsed.metadata.floating_point) throw std::runtime_error("delta packet is floating-point");
  return restore_values(parsed);
}

std::vector<std::uint8_t> delta_inverse_scaled(std::span<const std::uint8_t> packet) {
  const auto parsed = parse_delta(packet);
  if (!parsed.metadata.floating_point) throw std::runtime_error("delta packet is integer");
  const auto values = restore_values(parsed);
  return serialize_scaling(parsed.metadata, values);
}

std::vector<std::uint8_t> frame_of_reference_forward(
    std::span<const std::uint8_t> delta_packet) {
  return serialize_for(make_for(parse_delta(delta_packet)));
}

std::vector<std::uint8_t> frame_of_reference_inverse(
    std::span<const std::uint8_t> for_packet) {
  return serialize_delta(restore_delta(parse_for(for_packet)));
}

std::vector<std::uint8_t> bitpack_forward(std::span<const std::uint8_t> for_packet) {
  const auto packet = parse_for(for_packet);
  std::vector<std::uint8_t> out;
  if (packet.metadata.floating_point) append_uvarint(out, packet.metadata.max_point);
  for (const auto& block : packet.blocks) {
    const auto packed = max_value_bitpack::pack(block.normalized);
    const auto width = packed.width;
    append_u32_be(out, static_cast<std::uint32_t>(block.normalized.size()));
    append_u32_be(out, static_cast<std::uint32_t>(width));
    append_u64_be(out, static_cast<std::uint64_t>(block.minimum));
    append_u64_be(out, static_cast<std::uint64_t>(block.first));
    out.insert(out.end(), packed.payload.begin(), packed.payload.end());
  }
  return out;
}

std::vector<std::uint8_t> bitpack_inverse(
    std::span<const std::uint8_t> public_wire, bool floating_point,
    int max_point) {
  if (max_point < 0) throw std::invalid_argument("max_point must be non-negative");
  Cursor cursor(public_wire);
  Metadata metadata{floating_point, static_cast<std::uint32_t>(floating_point ? max_point : 0)};
  if (floating_point) {
    const auto encoded_scale = cursor.uvarint();
    if (encoded_scale != metadata.max_point) throw std::runtime_error("TS_2DIFF scale mismatch");
  }
  ForPacket packet{metadata, {}};
  while (!cursor.empty()) {
    const std::uint32_t count = cursor.u32_be();
    const std::uint32_t width = cursor.u32_be();
    const auto minimum = static_cast<std::int64_t>(cursor.u64_be());
    const auto first = static_cast<std::int64_t>(cursor.u64_be());
    if (count > kBlockSize || width > 64) throw std::runtime_error("invalid TS_2DIFF block header");
    const auto byte_count = max_value_bitpack::payload_size(count, width);
    ForBlock block{first, minimum, max_value_bitpack::unpack(cursor.take(byte_count), count, width)};
    packet.blocks.push_back(std::move(block));
  }
  return serialize_for(packet);
}

std::vector<std::uint8_t> compose_int64(std::span<const std::int64_t> values) {
  const auto delta = delta_forward_int64(values);
  const auto normalized = frame_of_reference_forward(delta);
  return bitpack_forward(normalized);
}

std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int max_point) {
  const auto scaled = scaling_forward(values, max_point);
  const auto delta = delta_forward_scaled(scaled);
  const auto normalized = frame_of_reference_forward(delta);
  return bitpack_forward(normalized);
}

std::vector<std::int64_t> decompose_int64(
    std::span<const std::uint8_t> public_wire) {
  const auto normalized = bitpack_inverse(public_wire, false, 0);
  const auto delta = frame_of_reference_inverse(normalized);
  return delta_inverse_int64(delta);
}

std::vector<double> decompose_double(
    std::span<const std::uint8_t> public_wire, int max_point) {
  const auto normalized = bitpack_inverse(public_wire, true, max_point);
  const auto delta = frame_of_reference_inverse(normalized);
  const auto scaled = delta_inverse_scaled(delta);
  return scaling_inverse(scaled);
}

}  // namespace exact_actions::ts2diff
