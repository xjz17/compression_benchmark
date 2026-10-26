#include "bos2_refined.hpp"
#include "minimum_for.hpp"
#include "max_value_bitpack.hpp"
#include "zigzag.hpp"

#include "dbp_exact_stages.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace unified_compression_v1::bos_bfe_v2::integer {
namespace max_value_bitpack = ::exact_actions::max_value_bitpack;
namespace {

constexpr std::array<std::uint8_t, 4> kDeltaMagic{'B', 'D', 'L', '1'};
constexpr std::array<std::uint8_t, 4> kFor1Magic{'B', 'F', '1', '1'};
constexpr std::array<std::uint8_t, 4> kOutlierMagic{'B', 'O', 'S', '1'};
constexpr std::array<std::uint8_t, 4> kFor2Magic{'B', 'F', '2', '1'};
constexpr std::array<std::uint8_t, 4> kAdapterSignedDeltaMagic{'B', '6', 'S', '1'};
constexpr std::array<std::uint8_t, 4> kAdapterScaledSignedDeltaMagic{'B', '6', 'T', '1'};
constexpr std::array<std::uint8_t, 4> kAdapterDeltaMagic{'B', '6', 'D', '1'};
constexpr std::array<std::uint8_t, 4> kAdapterScaledDeltaMagic{'B', '6', 'F', '1'};
constexpr std::array<std::uint8_t, 4> kPublicMagic{'B', 'O', 'S', 'A'};
constexpr std::array<std::uint8_t, 4> kOptimizedMagic{'B', 'O', 'S', '2'};
constexpr std::uint8_t kPublicFlags = 0x0f;

std::uint64_t to_u64(std::int64_t value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::int64_t to_i64(std::uint64_t value) {
  std::int64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int bit_width(std::uint64_t value) {
  return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

void put_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u32_be(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u64_be(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32_le(std::span<const std::uint8_t> data, std::size_t& pos,
                         const char* label) {
  if (data.size() - pos < 4) throw std::runtime_error(std::string("truncated ") + label);
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(data[pos++]) << shift;
  return value;
}

std::uint64_t get_u64_le(std::span<const std::uint8_t> data, std::size_t& pos,
                         const char* label) {
  if (data.size() - pos < 8) throw std::runtime_error(std::string("truncated ") + label);
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

std::uint32_t get_u32_be(std::span<const std::uint8_t> data, std::size_t& pos,
                         const char* label) {
  if (data.size() - pos < 4) throw std::runtime_error(std::string("truncated ") + label);
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value = (value << 8) | data[pos++];
  return value;
}

std::uint64_t get_u64_be(std::span<const std::uint8_t> data, std::size_t& pos,
                         const char* label) {
  if (data.size() - pos < 8) throw std::runtime_error(std::string("truncated ") + label);
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value = (value << 8) | data[pos++];
  return value;
}

void require_magic(std::span<const std::uint8_t> data,
                   const std::array<std::uint8_t, 4>& magic,
                   const char* label) {
  if (data.size() < magic.size() ||
      !std::equal(magic.begin(), magic.end(), data.begin()))
    throw std::runtime_error(std::string("wrong ") + label + " packet type");
}

struct DeltaPacket {
  std::uint64_t original_count{};
  std::uint64_t first{};
  std::vector<std::int64_t> deltas;
};

struct For1Packet {
  std::uint64_t original_count{};
  std::uint64_t first{};
  std::int64_t source_base{};
  std::vector<std::uint64_t> codes;
};

struct SplitEntry {
  bool outlier{};
  std::uint64_t value{};
};

struct SplitPacket {
  bool normalized{};
  std::uint64_t original_count{};
  std::uint64_t first{};
  std::int64_t source_base{};
  std::uint64_t range_base{};
  int width{};
  std::vector<SplitEntry> entries;
};

std::vector<std::uint8_t> serialize_delta(
    const DeltaPacket& packet,
    const std::array<std::uint8_t, 4>& magic = kDeltaMagic) {
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  put_u64_le(out, packet.original_count);
  put_u64_le(out, packet.first);
  put_u64_le(out, packet.deltas.size());
  for (const auto value : packet.deltas) put_u64_le(out, to_u64(value));
  return out;
}

DeltaPacket parse_delta(std::span<const std::uint8_t> packet,
                        const std::array<std::uint8_t, 4>& magic = kDeltaMagic) {
  require_magic(packet, magic, "BOS delta");
  std::size_t pos = 4;
  DeltaPacket parsed;
  parsed.original_count = get_u64_le(packet, pos, "BOS delta count");
  parsed.first = get_u64_le(packet, pos, "BOS delta first value");
  const auto count = get_u64_le(packet, pos, "BOS delta item count");
  if (count > (packet.size() - pos) / 8 || count * 8 != packet.size() - pos)
    throw std::runtime_error("invalid BOS delta item count");
  parsed.deltas.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i)
    parsed.deltas.push_back(to_i64(get_u64_le(packet, pos, "BOS delta value")));
  const auto expected = parsed.original_count == 0 ? 0 : parsed.original_count - 1;
  if (count != expected) throw std::runtime_error("BOS delta shape mismatch");
  return parsed;
}

std::vector<std::uint8_t> serialize_for1(const For1Packet& packet) {
  std::vector<std::uint8_t> out(kFor1Magic.begin(), kFor1Magic.end());
  put_u64_le(out, packet.original_count);
  put_u64_le(out, packet.first);
  put_u64_le(out, to_u64(packet.source_base));
  put_u64_le(out, packet.codes.size());
  for (const auto value : packet.codes) put_u64_le(out, value);
  return out;
}

For1Packet parse_for1(std::span<const std::uint8_t> packet) {
  require_magic(packet, kFor1Magic, "BOS first-FOR");
  std::size_t pos = 4;
  For1Packet parsed;
  parsed.original_count = get_u64_le(packet, pos, "BOS first-FOR count");
  parsed.first = get_u64_le(packet, pos, "BOS first-FOR first value");
  parsed.source_base = to_i64(get_u64_le(packet, pos, "BOS first-FOR base"));
  const auto count = get_u64_le(packet, pos, "BOS first-FOR item count");
  if (count > (packet.size() - pos) / 8 || count * 8 != packet.size() - pos)
    throw std::runtime_error("invalid BOS first-FOR item count");
  parsed.codes.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i)
    parsed.codes.push_back(get_u64_le(packet, pos, "BOS first-FOR code"));
  return parsed;
}

struct NormalRange {
  std::uint64_t base{};
  int width{};
  std::size_t count{};
};

NormalRange select_normal_range(const std::vector<std::uint64_t>& codes) {
  if (codes.empty()) return {};
  const auto [minimum, maximum] = std::minmax_element(codes.begin(), codes.end());
  NormalRange best{*minimum, bit_width(*maximum - *minimum), codes.size()};
  std::size_t best_bits = best.count * static_cast<std::size_t>(best.width);
  auto sorted = codes;
  std::sort(sorted.begin(), sorted.end());
  for (int width = 0; width < 64; ++width) {
    const std::uint64_t limit = width == 0 ? 0U : (std::uint64_t{1} << width) - 1U;
    std::size_t left = 0;
    for (std::size_t right = 0; right < sorted.size(); ++right) {
      while (sorted[right] - sorted[left] > limit) ++left;
      const auto normal_count = right - left + 1;
      const auto outlier_count = sorted.size() - normal_count;
      const auto bits = normal_count * static_cast<std::size_t>(width) +
                        outlier_count * 96U;
      if (bits < best_bits) {
        best = {sorted[left], width, normal_count};
        best_bits = bits;
      }
    }
  }
  return best;
}

bool in_normal_range(std::uint64_t code, const NormalRange& range) {
  if (code < range.base) return false;
  if (range.width == 64) return true;
  const auto limit = range.width == 0 ? 0U : (std::uint64_t{1} << range.width) - 1U;
  return code - range.base <= limit;
}

std::vector<std::uint8_t> serialize_split(const SplitPacket& packet) {
  const auto& magic = packet.normalized ? kFor2Magic : kOutlierMagic;
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  put_u64_le(out, packet.original_count);
  put_u64_le(out, packet.first);
  put_u64_le(out, to_u64(packet.source_base));
  put_u64_le(out, packet.range_base);
  out.push_back(static_cast<std::uint8_t>(packet.width));
  put_u64_le(out, packet.entries.size());
  for (const auto& entry : packet.entries) {
    out.push_back(entry.outlier ? 1U : 0U);
    put_u64_le(out, entry.value);
  }
  return out;
}

SplitPacket parse_split(std::span<const std::uint8_t> packet, bool normalized) {
  require_magic(packet, normalized ? kFor2Magic : kOutlierMagic,
                normalized ? "BOS second-FOR" : "BOS outlier");
  std::size_t pos = 4;
  SplitPacket parsed;
  parsed.normalized = normalized;
  parsed.original_count = get_u64_le(packet, pos, "BOS split count");
  parsed.first = get_u64_le(packet, pos, "BOS split first value");
  parsed.source_base = to_i64(get_u64_le(packet, pos, "BOS split source base"));
  parsed.range_base = get_u64_le(packet, pos, "BOS split range base");
  if (pos >= packet.size()) throw std::runtime_error("truncated BOS split width");
  parsed.width = packet[pos++];
  if (parsed.width < 0 || parsed.width > 64)
    throw std::runtime_error("invalid BOS split width");
  const auto count = get_u64_le(packet, pos, "BOS split item count");
  if (count > (packet.size() - pos) / 9 || count * 9 != packet.size() - pos)
    throw std::runtime_error("invalid BOS split item count");
  parsed.entries.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto tag = packet[pos++];
    if (tag > 1) throw std::runtime_error("invalid BOS split tag");
    parsed.entries.push_back(
        {tag != 0, get_u64_le(packet, pos, "BOS split value")});
  }
  return parsed;
}

}  // namespace

std::vector<std::uint8_t> delta_forward(std::span<const std::int64_t> values) {
  DeltaPacket packet;
  packet.original_count = values.size();
  packet.first = values.empty() ? 0U : to_u64(values.front());
  packet.deltas.reserve(values.empty() ? 0 : values.size() - 1);
  for (std::size_t i = 1; i < values.size(); ++i)
    packet.deltas.push_back(to_i64(to_u64(values[i]) - to_u64(values[i - 1])));
  return serialize_delta(packet);
}

std::vector<std::int64_t> delta_inverse(std::span<const std::uint8_t> packet) {
  const auto parsed = parse_delta(packet);
  if (parsed.original_count == 0) return {};
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(parsed.original_count));
  values.push_back(to_i64(parsed.first));
  for (const auto delta : parsed.deltas)
    values.push_back(to_i64(to_u64(values.back()) + to_u64(delta)));
  return values;
}

std::vector<std::uint8_t> first_for_forward(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_delta(packet);
  For1Packet out;
  out.original_count = parsed.original_count;
  out.first = parsed.first;
  auto encoded = minimum_for::encode<std::int64_t>(parsed.deltas);
  out.source_base = encoded.minimum;
  out.codes = std::move(encoded.offsets);
  return serialize_for1(out);
}

std::vector<std::uint8_t> first_for_inverse(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_for1(packet);
  DeltaPacket out{parsed.original_count, parsed.first, {}};
  out.deltas.reserve(parsed.codes.size());
  for (const auto code : parsed.codes)
    out.deltas.push_back(minimum_for::restore(parsed.source_base, code));
  return serialize_delta(out);
}

std::vector<std::uint8_t> outlier_forward(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_for1(packet);
  const auto range = select_normal_range(parsed.codes);
  SplitPacket out{false, parsed.original_count, parsed.first,
                  parsed.source_base, range.base, range.width, {}};
  out.entries.reserve(parsed.codes.size());
  for (const auto code : parsed.codes)
    out.entries.push_back({!in_normal_range(code, range), code});
  return serialize_split(out);
}

std::vector<std::uint8_t> outlier_inverse(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_split(packet, false);
  For1Packet out{parsed.original_count, parsed.first, parsed.source_base, {}};
  out.codes.reserve(parsed.entries.size());
  for (const auto& entry : parsed.entries) out.codes.push_back(entry.value);
  return serialize_for1(out);
}

std::vector<std::uint8_t> second_for_forward(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse_split(packet, false);
  parsed.normalized = true;
  std::vector<std::uint64_t> normal;
  for (const auto& entry : parsed.entries) if (!entry.outlier) normal.push_back(entry.value);
  const auto encoded = minimum_for::encode<std::uint64_t>(normal);
  std::size_t index = 0;
  for (auto& entry : parsed.entries) if (!entry.outlier) {
    // select_normal_range chooses an included sorted[left], hence its base is
    // min(normal). Old externally supplied packets may use any base instead;
    // preserve their modulo offsets and their original header byte for byte.
    entry.value = encoded.offsets[index++] + encoded.minimum - parsed.range_base;
  }
  return serialize_split(parsed);
}

std::vector<std::uint8_t> second_for_inverse(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse_split(packet, true);
  parsed.normalized = false;
  for (auto& entry : parsed.entries) {
    if (!entry.outlier) entry.value = minimum_for::restore(parsed.range_base, entry.value);
  }
  return serialize_split(parsed);
}

std::vector<std::uint8_t> bitpack_forward(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_split(packet, true);
  if (parsed.original_count > std::numeric_limits<std::uint32_t>::max() ||
      parsed.entries.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("BOS stage input exceeds public wire limits");
  std::vector<std::pair<std::uint32_t, std::uint64_t>> outliers;
  std::vector<std::uint64_t> normal;
  for (std::size_t i = 0; i < parsed.entries.size(); ++i) {
    if (parsed.entries[i].outlier)
      outliers.emplace_back(static_cast<std::uint32_t>(i), parsed.entries[i].value);
    else
      normal.push_back(parsed.entries[i].value);
  }
  auto body = max_value_bitpack::pack_fixed(normal, parsed.width).payload;

  std::vector<std::uint8_t> out(kPublicMagic.begin(), kPublicMagic.end());
  put_u32_le(out, static_cast<std::uint32_t>(parsed.original_count));
  out.push_back(kPublicFlags);
  put_u64_le(out, parsed.first);
  put_u64_le(out, to_u64(parsed.source_base));
  put_u64_le(out, parsed.range_base);
  out.push_back(static_cast<std::uint8_t>(parsed.width));
  put_u32_le(out, static_cast<std::uint32_t>(outliers.size()));
  put_u32_le(out, static_cast<std::uint32_t>(body.size()));
  for (const auto& [position, value] : outliers) {
    const std::array<std::uint64_t, 1> positions{position}, values{value};
    const max_value_bitpack::Format little{max_value_bitpack::BitOrder::lsb_first, 1};
    const auto packed_position = max_value_bitpack::pack_fixed(positions, 32, little);
    const auto packed_value = max_value_bitpack::pack_fixed(values, 64, little);
    out.insert(out.end(), packed_position.payload.begin(), packed_position.payload.end());
    out.insert(out.end(), packed_value.payload.begin(), packed_value.payload.end());
  }
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  require_magic(wire, kPublicMagic, "BOS public");
  std::size_t pos = 4;
  const auto original_count = get_u32_le(wire, pos, "BOS public count");
  if (pos >= wire.size() || wire[pos++] != kPublicFlags)
    throw std::runtime_error("BOS public wire is not the full five-stage mode");
  const auto first = get_u64_le(wire, pos, "BOS public first value");
  const auto source_base = to_i64(get_u64_le(wire, pos, "BOS public source base"));
  const auto range_base = get_u64_le(wire, pos, "BOS public range base");
  if (pos >= wire.size()) throw std::runtime_error("truncated BOS public width");
  const int width = wire[pos++];
  if (width < 0 || width > 64) throw std::runtime_error("invalid BOS public width");
  const auto outlier_count = get_u32_le(wire, pos, "BOS public outlier count");
  const auto body_size = get_u32_le(wire, pos, "BOS public body size");
  const auto source_count = original_count == 0 ? 0U : original_count - 1U;
  if (outlier_count > source_count ||
      wire.size() - pos < static_cast<std::size_t>(outlier_count) * 12U)
    throw std::runtime_error("invalid BOS public outlier table");

  std::vector<SplitEntry> entries(source_count);
  for (std::uint32_t i = 0; i < outlier_count; ++i) {
    const auto index = get_u32_le(wire, pos, "BOS public outlier position");
    const auto value = get_u64_le(wire, pos, "BOS public outlier value");
    if (index >= entries.size() || entries[index].outlier)
      throw std::runtime_error("invalid BOS public outlier position");
    entries[index] = {true, value};
  }
  if (wire.size() - pos != body_size)
    throw std::runtime_error("invalid BOS public body length");
  const auto normal_count = entries.size() - outlier_count;
  const auto expected_body =
      (normal_count * static_cast<std::size_t>(width) + 7U) / 8U;
  if (body_size != expected_body)
    throw std::runtime_error("invalid BOS public packed body length");
  const auto normal = max_value_bitpack::unpack(wire.subspan(pos), normal_count, width);
  std::size_t normal_index = 0;
  for (auto& entry : entries) {
    if (!entry.outlier) entry.value = normal[normal_index++];
  }
  return serialize_split(
      {true, original_count, first, source_base, range_base, width,
       std::move(entries)});
}

std::vector<std::uint8_t> compose(std::span<const std::int64_t> values) {
  return bitpack_forward(second_for_forward(outlier_forward(
      first_for_forward(delta_forward(values)))));
}

std::vector<std::int64_t> decompose(std::span<const std::uint8_t> wire) {
  return delta_inverse(first_for_inverse(outlier_inverse(
      second_for_inverse(bitpack_inverse(wire)))));
}

std::vector<std::uint8_t> adapter_plain_delta_forward(
    std::span<const std::int64_t> values) {
  DeltaPacket packet;
  packet.original_count = values.size();
  packet.first = values.empty() ? 0U : to_u64(values.front());
  packet.deltas.reserve(values.empty() ? 0 : values.size() - 1);
  for (std::size_t i = 1; i < values.size(); ++i) {
    const auto delta = to_i64(to_u64(values[i]) - to_u64(values[i - 1]));
    packet.deltas.push_back(delta);
  }
  return serialize_delta(packet, kAdapterSignedDeltaMagic);
}

std::vector<std::uint8_t> adapter_plain_delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  const auto [scale, values] = exact_actions::dbp::scaling_values(scaling_packet);
  const auto inner = adapter_plain_delta_forward(values);
  std::vector<std::uint8_t> out(kAdapterScaledSignedDeltaMagic.begin(),
                                kAdapterScaledSignedDeltaMagic.end());
  put_u32_le(out, static_cast<std::uint32_t>(scale));
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

std::vector<std::int64_t> adapter_plain_delta_inverse(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_delta(packet, kAdapterSignedDeltaMagic);
  if (parsed.original_count == 0) return {};
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(parsed.original_count));
  values.push_back(to_i64(parsed.first));
  for (const auto delta : parsed.deltas)
    values.push_back(to_i64(to_u64(values.back()) + to_u64(delta)));
  return values;
}

std::vector<std::uint8_t> adapter_plain_delta_inverse_scaled(
    std::span<const std::uint8_t> packet) {
  require_magic(packet, kAdapterScaledSignedDeltaMagic,
                "BOS64 scaled signed delta");
  std::size_t pos = 4;
  const auto scale = static_cast<int>(static_cast<std::int32_t>(
      get_u32_le(packet, pos, "BOS64 scale")));
  const auto values = adapter_plain_delta_inverse(packet.subspan(pos));
  return exact_actions::dbp::make_scaling_packet(scale, values);
}

std::vector<std::uint8_t> adapter_zigzag_forward(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse_delta(packet, kAdapterSignedDeltaMagic);
  for (auto& delta : parsed.deltas)
    delta = to_i64(compression_operators::zigzag::encode64(delta));
  return serialize_delta(parsed, kAdapterDeltaMagic);
}

std::vector<std::uint8_t> adapter_zigzag_forward_scaled(
    std::span<const std::uint8_t> packet) {
  require_magic(packet, kAdapterScaledSignedDeltaMagic,
                "BOS64 scaled signed delta");
  std::size_t pos = 4;
  const auto scale = get_u32_le(packet, pos, "BOS64 scale");
  const auto inner = adapter_zigzag_forward(packet.subspan(pos));
  std::vector<std::uint8_t> out(kAdapterScaledDeltaMagic.begin(),
                                kAdapterScaledDeltaMagic.end());
  put_u32_le(out, scale);
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

std::vector<std::uint8_t> adapter_zigzag_inverse(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse_delta(packet, kAdapterDeltaMagic);
  for (auto& code : parsed.deltas)
    code = compression_operators::zigzag::decode64(to_u64(code));
  return serialize_delta(parsed, kAdapterSignedDeltaMagic);
}

std::vector<std::uint8_t> adapter_zigzag_inverse_scaled(
    std::span<const std::uint8_t> packet) {
  require_magic(packet, kAdapterScaledDeltaMagic, "BOS64 scaled delta");
  std::size_t pos = 4;
  const auto scale = get_u32_le(packet, pos, "BOS64 scale");
  const auto inner = adapter_zigzag_inverse(packet.subspan(pos));
  std::vector<std::uint8_t> out(kAdapterScaledSignedDeltaMagic.begin(),
                                kAdapterScaledSignedDeltaMagic.end());
  put_u32_le(out, scale);
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

std::vector<std::uint8_t> adapter_delta_forward(
    std::span<const std::int64_t> values) {
  return adapter_zigzag_forward(adapter_plain_delta_forward(values));
}

std::vector<std::uint8_t> adapter_delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet) {
  return adapter_zigzag_forward_scaled(
      adapter_plain_delta_forward_scaled(scaling_packet));
}

std::vector<std::int64_t> adapter_delta_inverse(
    std::span<const std::uint8_t> packet) {
  return adapter_plain_delta_inverse(adapter_zigzag_inverse(packet));
}

std::vector<std::uint8_t> adapter_delta_inverse_scaled(
    std::span<const std::uint8_t> packet) {
  return adapter_plain_delta_inverse_scaled(
      adapter_zigzag_inverse_scaled(packet));
}

std::vector<std::uint8_t> adapter_bitpack_forward(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_delta(packet, kAdapterDeltaMagic);
  if (parsed.original_count == 0) return {};
  if (parsed.original_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::runtime_error("BOS64 stage input exceeds int32 row count");

  std::vector<std::uint64_t> codes;
  codes.reserve(parsed.deltas.size());
  for (const auto stored : parsed.deltas) codes.push_back(to_u64(stored));
  const int packed_width = max_value_bitpack::allowed_width(codes, 1, 56);
  const auto payload_size = packed_width == 64
                                ? parsed.deltas.size() * 8U
                                : (parsed.deltas.size() *
                                       static_cast<std::size_t>(packed_width) +
                                   7U) /
                                      8U;
  if (payload_size >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::runtime_error("BOS64 stage payload exceeds int32 length");

  auto body = max_value_bitpack::pack_fixed(codes, packed_width).payload;
  if (body.size() != payload_size)
    throw std::runtime_error("BOS64 stage internal payload length mismatch");

  const auto count = static_cast<std::uint32_t>(parsed.original_count);
  std::vector<std::uint8_t> out;
  out.reserve(25U + body.size());
  put_u32_be(out, count);
  put_u32_be(out, count);
  put_u32_be(out, count);
  put_u64_be(out, parsed.first);
  out.push_back(static_cast<std::uint8_t>(packed_width));
  put_u32_be(out, static_cast<std::uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> adapter_bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  if (wire.empty()) return serialize_delta({}, kAdapterDeltaMagic);
  std::size_t pos = 0;
  const auto count = get_u32_be(wire, pos, "BOS64 count");
  const auto segment_rows = get_u32_be(wire, pos, "BOS64 segment rows");
  const auto block_len = get_u32_be(wire, pos, "BOS64 block length");
  if (count == 0 || segment_rows != count || block_len != count)
    throw std::runtime_error("BOS64 stage expects one non-empty full-column block");
  const auto first = get_u64_be(wire, pos, "BOS64 first value");
  if (pos >= wire.size()) throw std::runtime_error("truncated BOS64 width");
  const auto encoded_width = wire[pos++];
  if ((encoded_width & 0x80U) != 0)
    throw std::runtime_error("BOS64 stage does not accept direct-FOR wire");
  const int width = encoded_width & 0x7fU;
  if (width <= 0 || width > 64)
    throw std::runtime_error("invalid BOS64 packed width");
  const auto payload_size = get_u32_be(wire, pos, "BOS64 payload length");
  const auto value_count = static_cast<std::size_t>(count - 1U);
  const auto expected_size = width == 64
                                 ? value_count * 8U
                                 : (value_count * static_cast<std::size_t>(width) +
                                    7U) /
                                       8U;
  if (payload_size != expected_size || wire.size() - pos != payload_size)
    throw std::runtime_error("invalid BOS64 packed body length");

  DeltaPacket packet{count, first, {}};
  packet.deltas.reserve(value_count);
  for (const auto code : max_value_bitpack::unpack(wire.subspan(pos), value_count, width))
    packet.deltas.push_back(to_i64(code));
  return serialize_delta(packet, kAdapterDeltaMagic);
}

std::vector<std::uint8_t> adapter_wire_forward_int(
    std::span<const std::uint8_t> delta_packet) {
  auto body = adapter_bitpack_forward(delta_packet);
  if (body.empty()) body = {0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<std::uint8_t> out{'I'};
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> adapter_wire_forward_scaled(
    std::span<const std::uint8_t> delta_packet) {
  require_magic(delta_packet, kAdapterScaledDeltaMagic, "BOS64 scaled delta");
  std::size_t pos = 4;
  const auto scale = static_cast<int>(static_cast<std::int32_t>(
      get_u32_le(delta_packet, pos, "BOS64 scale")));
  auto body = adapter_bitpack_forward(delta_packet.subspan(pos));
  if (body.empty()) body = {0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<std::uint8_t> out{'F'};
  put_u32_le(out, static_cast<std::uint32_t>(scale));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> adapter_wire_inverse_int(
    std::span<const std::uint8_t> wire) {
  if (wire.empty() || wire.front() != 'I')
    throw std::runtime_error("wrong integer BOS64 adapter wire type");
  return adapter_bitpack_inverse(wire.subspan(1));
}

std::vector<std::uint8_t> adapter_wire_inverse_scaled(
    std::span<const std::uint8_t> wire) {
  if (wire.size() < 5 || wire.front() != 'F')
    throw std::runtime_error("wrong floating BOS64 adapter wire type");
  std::size_t pos = 1;
  const auto scale = get_u32_le(wire, pos, "BOS64 wire scale");
  const auto inner = adapter_bitpack_inverse(wire.subspan(pos));
  std::vector<std::uint8_t> out(kAdapterScaledDeltaMagic.begin(),
                                kAdapterScaledDeltaMagic.end());
  put_u32_le(out, scale);
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

std::vector<std::uint8_t> adapter_compose(
    std::span<const std::int64_t> values) {
  return adapter_bitpack_forward(
      adapter_zigzag_forward(adapter_plain_delta_forward(values)));
}

std::vector<std::int64_t> adapter_decompose(
    std::span<const std::uint8_t> wire) {
  return adapter_plain_delta_inverse(
      adapter_zigzag_inverse(adapter_bitpack_inverse(wire)));
}

namespace {

struct OptimizedSplit {
  int threshold_width = 64;
  std::uint64_t normal_base = 0;
  int normal_width = 0;
  std::uint64_t outlier_base = 0;
  int outlier_width = 0;
  std::size_t normal_count = 0;
  std::size_t outlier_count = 0;
  std::size_t encoded_bytes = 0;
};

std::size_t packed_bytes(std::size_t count, int width) {
  if (width < 0 || width > 64 ||
      (width != 0 && count >
           (std::numeric_limits<std::size_t>::max() - 7U) /
               static_cast<std::size_t>(width)))
    throw std::runtime_error("BOS2 packed-size overflow");
  return (count * static_cast<std::size_t>(width) + 7U) / 8U;
}

OptimizedSplit select_optimized_split(
    std::span<const std::int64_t> values) {
  OptimizedSplit best;
  if (values.size() < 2) return best;
  const auto code_count = values.size() - 1U;

  struct Bucket {
    std::size_t count = 0;
    std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximum = 0;
  };
  std::array<Bucket, 65> buckets{};
  for (std::size_t index = 1; index < values.size(); ++index) {
    const auto delta = to_i64(to_u64(values[index]) - to_u64(values[index - 1]));
    const auto code = compression_operators::zigzag::encode64(delta);
    const int width = bit_width(code);
    auto& bucket = buckets[static_cast<std::size_t>(width)];
    ++bucket.count;
    bucket.minimum = std::min(bucket.minimum, code);
    bucket.maximum = std::max(bucket.maximum, code);
  }

  // The original implementation rescanned all 65 buckets for each of the 65
  // candidate thresholds.  Prefix/suffix summaries evaluate the identical
  // candidates and preserve every tie-break, while making model selection a
  // fixed O(65) operation after the one data pass.
  std::array<Bucket, 65> prefixes{};
  std::array<Bucket, 66> suffixes{};
  Bucket prefix{};
  for (std::size_t width = 0; width < buckets.size(); ++width) {
    const auto& bucket = buckets[width];
    if (bucket.count != 0) {
      prefix.count += bucket.count;
      prefix.minimum = std::min(prefix.minimum, bucket.minimum);
      prefix.maximum = std::max(prefix.maximum, bucket.maximum);
    }
    prefixes[width] = prefix;
  }
  Bucket suffix{};
  for (std::size_t width = buckets.size(); width-- > 0;) {
    const auto& bucket = buckets[width];
    if (bucket.count != 0) {
      suffix.count += bucket.count;
      suffix.minimum = std::min(suffix.minimum, bucket.minimum);
      suffix.maximum = std::max(suffix.maximum, bucket.maximum);
    }
    suffixes[width] = suffix;
  }

  // The fixed header is 38 bytes.  A membership bitmap is charged whenever
  // exceptions exist.  Evaluate every magnitude threshold against the exact
  // byte cost; this is data-dependent coding, not dataset-level tuning.
  constexpr std::size_t header_bytes = 38U;
  bool have_best = false;
  for (int threshold = 0; threshold <= 64; ++threshold) {
    const auto& normal_bucket = prefixes[static_cast<std::size_t>(threshold)];
    const auto& outlier_bucket =
        suffixes[static_cast<std::size_t>(threshold) + 1U];
    const auto normal_count = normal_bucket.count;
    const auto outlier_count = outlier_bucket.count;
    if (normal_count == 0) continue;
    const auto normal_min = normal_bucket.minimum;
    const auto normal_max = normal_bucket.maximum;
    const auto outlier_min = outlier_bucket.minimum;
    const auto outlier_max = outlier_bucket.maximum;
    const int normal_width = bit_width(normal_max - normal_min);
    const int outlier_width = outlier_count == 0
                                  ? 0
                                  : bit_width(outlier_max - outlier_min);
    const auto bitmap_bytes =
        outlier_count == 0 ? 0U : (code_count + 7U) / 8U;
    const auto bytes = header_bytes + bitmap_bytes +
                       packed_bytes(normal_count, normal_width) +
                       packed_bytes(outlier_count, outlier_width);
    if (!have_best || bytes < best.encoded_bytes ||
        (bytes == best.encoded_bytes && outlier_count < best.outlier_count) ||
        (bytes == best.encoded_bytes && outlier_count == best.outlier_count &&
         threshold > best.threshold_width)) {
      best = {threshold,
              normal_min,
              normal_width,
              outlier_count == 0 ? 0U : outlier_min,
              outlier_width,
              normal_count,
              outlier_count,
              bytes};
      have_best = true;
    }
  }
  if (!have_best) throw std::runtime_error("BOS2 could not select a split");
  return best;
}

void put_msb_field(std::vector<std::uint8_t>& out, std::size_t byte_offset,
                   std::size_t bit_offset, std::uint64_t value,
                   unsigned width) {
  if (width > 64 || (width == 0 && value != 0) ||
      (width != 0 && width < 64 && value >= (std::uint64_t{1} << width)))
    throw std::runtime_error("BOS2 value exceeds selected width");
  unsigned remaining = width;
  while (remaining != 0) {
    const unsigned free_bits = 8U - static_cast<unsigned>(bit_offset % 8U);
    const unsigned take = std::min(remaining, free_bits);
    const unsigned shift = remaining - take;
    const auto mask = (std::uint64_t{1} << take) - 1U;
    const auto fragment = static_cast<std::uint8_t>((value >> shift) & mask);
    out.at(byte_offset + bit_offset / 8U) |=
        static_cast<std::uint8_t>(fragment << (free_bits - take));
    bit_offset += take;
    remaining -= take;
  }
}

std::uint64_t get_msb_field(std::span<const std::uint8_t> input,
                            std::size_t bit_offset, unsigned width) {
  if (width > 64) throw std::runtime_error("invalid BOS2 field width");
  std::uint64_t value = 0;
  unsigned remaining = width;
  while (remaining != 0) {
    const unsigned free_bits = 8U - static_cast<unsigned>(bit_offset % 8U);
    const unsigned take = std::min(remaining, free_bits);
    const auto mask = static_cast<std::uint8_t>((1U << take) - 1U);
    const auto fragment = static_cast<std::uint8_t>(
        (input[bit_offset / 8U] >> (free_bits - take)) & mask);
    value = (value << take) | fragment;
    bit_offset += take;
    remaining -= take;
  }
  return value;
}

void require_zero_msb_padding(std::span<const std::uint8_t> input,
                              std::size_t count, unsigned width) {
  if (width > 64 || (width != 0 &&
                     count > std::numeric_limits<std::size_t>::max() / width))
    throw std::runtime_error("invalid BOS2 packed bit count");
  const auto bits = count * static_cast<std::size_t>(width);
  if (bits % 8U == 0 || input.empty()) return;
  const auto padding = 8U - static_cast<unsigned>(bits % 8U);
  const auto mask = static_cast<std::uint8_t>((1U << padding) - 1U);
  if ((input.back() & mask) != 0)
    throw std::runtime_error("BOS2 nonzero trailing padding bits");
}

}  // namespace

std::vector<std::uint8_t> optimized_compose(
    std::span<const std::int64_t> values) {
  if (values.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("BOS2 input exceeds uint32 row count");
  std::vector<std::uint8_t> out(kOptimizedMagic.begin(), kOptimizedMagic.end());
  put_u32_le(out, static_cast<std::uint32_t>(values.size()));
  if (values.empty()) return out;
  put_u64_le(out, to_u64(values.front()));
  if (values.size() == 1) return out;

  const auto code_count = values.size() - 1U;
  const auto split = select_optimized_split(values);
  put_u64_le(out, split.normal_base);
  out.push_back(static_cast<std::uint8_t>(split.normal_width));
  put_u64_le(out, split.outlier_base);
  out.push_back(static_cast<std::uint8_t>(split.outlier_width));
  put_u32_le(out, static_cast<std::uint32_t>(split.outlier_count));

  const auto bitmap_size =
      split.outlier_count == 0 ? 0U : (code_count + 7U) / 8U;
  const auto normal_size = packed_bytes(split.normal_count, split.normal_width);
  const auto outlier_size =
      packed_bytes(split.outlier_count, split.outlier_width);
  const auto bitmap_offset = out.size();
  const auto normal_offset = bitmap_offset + bitmap_size;
  const auto outlier_offset = normal_offset + normal_size;
  const auto normal_width = static_cast<unsigned>(split.normal_width);
  const auto outlier_width = static_cast<unsigned>(split.outlier_width);
  if (outlier_offset + outlier_size != split.encoded_bytes)
    throw std::runtime_error("BOS2 split-size mismatch");
  out.resize(split.encoded_bytes, 0);
  std::size_t normal_index = 0;
  std::size_t outlier_index = 0;
  for (std::size_t index = 0; index < code_count; ++index) {
    const auto value_index = index + 1U;
    const auto delta = to_i64(to_u64(values[value_index]) -
                              to_u64(values[value_index - 1U]));
    const auto code = compression_operators::zigzag::encode64(delta);
    const bool is_outlier = bit_width(code) > split.threshold_width;
    if (is_outlier) {
      out[bitmap_offset + index / 8U] |=
          static_cast<std::uint8_t>(1U << (index % 8U));
      put_msb_field(out, outlier_offset,
                    outlier_index * static_cast<std::size_t>(outlier_width),
                    code - split.outlier_base, outlier_width);
      ++outlier_index;
    } else {
      put_msb_field(out, normal_offset,
                    normal_index * static_cast<std::size_t>(normal_width),
                    code - split.normal_base, normal_width);
      ++normal_index;
    }
  }
  if (normal_index != split.normal_count ||
      outlier_index != split.outlier_count)
    throw std::runtime_error("BOS2 split count mismatch");
  if (out.size() != split.encoded_bytes)
    throw std::runtime_error("BOS2 encoded-size mismatch");
  return out;
}

std::vector<std::int64_t> optimized_decompose(
    std::span<const std::uint8_t> wire) {
  require_magic(wire, kOptimizedMagic, "BOS2");
  std::size_t pos = 4;
  const auto count = get_u32_le(wire, pos, "BOS2 count");
  if (count == 0) {
    if (pos != wire.size()) throw std::runtime_error("BOS2 empty trailing data");
    return {};
  }
  const auto first = get_u64_le(wire, pos, "BOS2 first value");
  if (count == 1) {
    if (pos != wire.size()) throw std::runtime_error("BOS2 singleton trailing data");
    return {to_i64(first)};
  }
  const auto normal_base = get_u64_le(wire, pos, "BOS2 normal base");
  if (pos >= wire.size()) throw std::runtime_error("truncated BOS2 normal width");
  const int normal_width = wire[pos++];
  const auto outlier_base = get_u64_le(wire, pos, "BOS2 outlier base");
  if (pos >= wire.size()) throw std::runtime_error("truncated BOS2 outlier width");
  const int outlier_width = wire[pos++];
  const auto outlier_count = get_u32_le(wire, pos, "BOS2 outlier count");
  const std::size_t code_count = static_cast<std::size_t>(count - 1U);
  if (normal_width > 64 || outlier_width > 64 || outlier_count > code_count)
    throw std::runtime_error("invalid BOS2 split header");
  const std::size_t normal_count = code_count - outlier_count;
  if (normal_count == 0)
    throw std::runtime_error("BOS2 requires a non-empty normal partition");
  const std::size_t bitmap_size = outlier_count == 0 ? 0U : (code_count + 7U) / 8U;
  const std::size_t normal_size = packed_bytes(normal_count, normal_width);
  const std::size_t outlier_size = packed_bytes(outlier_count, outlier_width);
  if (wire.size() - pos != bitmap_size + normal_size + outlier_size)
    throw std::runtime_error("invalid BOS2 body length");
  const auto bitmap = wire.subspan(pos, bitmap_size);
  pos += bitmap_size;
  const auto normal = wire.subspan(pos, normal_size);
  const auto normal_width_u = static_cast<unsigned>(normal_width);
  const auto outlier_width_u = static_cast<unsigned>(outlier_width);
  require_zero_msb_padding(normal, normal_count, normal_width_u);
  pos += normal_size;
  const auto outliers = wire.subspan(pos, outlier_size);
  require_zero_msb_padding(outliers, outlier_count, outlier_width_u);
  std::size_t normal_index = 0;
  std::size_t outlier_index = 0;
  std::vector<std::int64_t> values;
  values.reserve(count);
  values.push_back(to_i64(first));
  for (std::size_t index = 0; index < code_count; ++index) {
    const bool is_outlier = outlier_count != 0 &&
                            ((bitmap[index / 8U] >> (index % 8U)) & 1U) != 0;
    const auto code = is_outlier
                          ? outlier_base + get_msb_field(
                              outliers,
                              outlier_index++ *
                                  static_cast<std::size_t>(outlier_width_u),
                              outlier_width_u)
                          : normal_base + get_msb_field(
                              normal,
                              normal_index++ *
                                  static_cast<std::size_t>(normal_width_u),
                              normal_width_u);
    const auto delta = compression_operators::zigzag::decode64(code);
    values.push_back(to_i64(to_u64(values.back()) + to_u64(delta)));
  }
  if (normal_index != normal_count || outlier_index != outlier_count)
    throw std::runtime_error("BOS2 decoded split count mismatch");
  return values;
}

}  // namespace unified_compression_v1::bos_bfe_v2::integer
