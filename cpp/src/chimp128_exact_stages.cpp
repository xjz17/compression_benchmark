#include "chimp128_exact_stages.hpp"
#include "residual_window_pack.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::chimp128 {
namespace {

constexpr std::array<std::uint8_t, 4> kIntMagic{'C', 'X', 'I', '1'};
constexpr std::array<std::uint8_t, 4> kFloatMagic{'C', 'X', 'F', '1'};
constexpr std::uint64_t kIntEnd = 0x8000000000000000ull;
constexpr std::uint64_t kFloatEnd = 0x7ff8000000000000ull;
constexpr int kHistory = 128;
constexpr int kHistoryLog2 = 7;
constexpr int kThreshold = 13;
constexpr int kKeyMask = (1 << 14) - 1;
// Chimp's two-bit case 01 followed by leading class 7 and a zero significant
// width is invalid in the source format: zero means 64 significant bits, which
// cannot coexist with 24 leading zeros. Use that otherwise unreachable token
// to distinguish a source value equal to the terminal marker from the actual
// terminal marker. Ordinary source-compatible wires are unchanged.
constexpr std::uint8_t kMarkerEscapePrevious = 0;
constexpr std::uint8_t kMarkerEscapeLeadingClass = 7;
constexpr int kLeadingRepresentation[] = {0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7};
constexpr int kLeadingRound[] = {0, 0, 0, 0, 0, 0, 0, 0,
    8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24};
constexpr int kDecodeLeading[] = {0, 8, 12, 16, 18, 20, 22, 24};

struct Record { std::uint8_t previous; std::uint8_t trailing; std::uint64_t xor_value; };
struct Packet { bool floating; std::uint64_t count; std::uint64_t first; std::vector<Record> records; };

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint64_t read_u64_le(std::span<const std::uint8_t> data, std::size_t& pos) {
  if (data.size() - pos < 8) throw std::runtime_error("truncated Chimp128 packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

std::vector<std::uint8_t> serialize(const Packet& packet) {
  const auto& magic = packet.floating ? kFloatMagic : kIntMagic;
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  append_u64_le(out, packet.count);
  append_u64_le(out, packet.first);
  for (const auto& record : packet.records) {
    out.push_back(record.previous);
    out.push_back(record.trailing);
    append_u64_le(out, record.xor_value);
  }
  return out;
}

Packet parse(std::span<const std::uint8_t> packet) {
  if (packet.size() < 20) throw std::runtime_error("truncated Chimp128 packet");
  bool floating;
  if (std::equal(kIntMagic.begin(), kIntMagic.end(), packet.begin())) floating = false;
  else if (std::equal(kFloatMagic.begin(), kFloatMagic.end(), packet.begin())) floating = true;
  else throw std::runtime_error("wrong Chimp128 packet type");
  std::size_t pos = 4;
  const auto count = read_u64_le(packet, pos);
  const auto first = read_u64_le(packet, pos);
  if (count > (packet.size() - pos) / 10 || count * 10 != packet.size() - pos)
    throw std::runtime_error("invalid Chimp128 record count");
  std::vector<Record> records;
  records.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto previous = packet[pos++];
    const auto trailing = packet[pos++];
    const auto xor_value = read_u64_le(packet, pos);
    if (previous >= kHistory || trailing > 64)
      throw std::runtime_error("invalid Chimp128 record");
    records.push_back({previous, trailing, xor_value});
  }
  return {floating, count, first, std::move(records)};
}

Packet build(bool floating, std::span<const std::int64_t> input) {
  const std::uint64_t end = floating ? kFloatEnd : kIntEnd;
  Packet packet{floating, input.size(), input.empty() ? end : static_cast<std::uint64_t>(input[0]), {}};
  std::uint64_t stored[kHistory]{};
  int indices[1 << 14]{};
  int index = 0, current = 0;
  stored[0] = packet.first;
  indices[static_cast<int>(packet.first & kKeyMask)] = 0;
  auto add = [&](std::uint64_t value) {
    const int key = static_cast<int>(value & kKeyMask);
    const int candidate_index = indices[key];
    std::uint64_t xor_value;
    int previous_index;
    int trailing = 0;
    if (index - candidate_index < kHistory) {
      const auto candidate_xor = value ^ stored[candidate_index % kHistory];
      trailing = candidate_xor == 0 ? 64 : __builtin_ctzll(candidate_xor);
      if (trailing > kThreshold) {
        previous_index = candidate_index % kHistory;
        xor_value = candidate_xor;
      } else {
        previous_index = index % kHistory;
        xor_value = stored[previous_index] ^ value;
      }
    } else {
      previous_index = index % kHistory;
      xor_value = stored[previous_index] ^ value;
    }
    packet.records.push_back({static_cast<std::uint8_t>(previous_index),
                              static_cast<std::uint8_t>(trailing), xor_value});
    current = (current + 1) % kHistory;
    stored[current] = value;
    ++index;
    indices[key] = index;
  };
  if (!input.empty()) {
    for (std::size_t i = 1; i < input.size(); ++i)
      add(static_cast<std::uint64_t>(input[i]));
    add(end);
  }
  return packet;
}

struct BitWriter {
  int stored_leading = std::numeric_limits<int>::max();
  std::uint8_t buffer = 0;
  int bits_left = 8;
  std::vector<std::uint8_t> out;
  void flip() { if (!bits_left) { out.push_back(buffer); buffer = 0; bits_left = 8; } }
  void bit(bool value) { if (value) buffer |= static_cast<std::uint8_t>(1u << (bits_left - 1)); --bits_left; flip(); }
  void bits(std::uint64_t value, int count) {
    while (count > 0) {
      const int take = std::min(count, bits_left);
      const int shift = count - take;
      const auto mask = take == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << take) - 1);
      buffer |= static_cast<std::uint8_t>(((value >> shift) & mask) << (bits_left - take));
      count -= take; bits_left -= take; flip();
    }
  }
  void finish() { bits_left = 0; flip(); }
  void marker_escape() {
    bits(1, 2);
    bits(kMarkerEscapePrevious, kHistoryLog2);
    bits(kMarkerEscapeLeadingClass, 3);
    bits(0, 6);
    stored_leading = 65;
  }
};

struct BitReader {
  std::span<const std::uint8_t> wire;
  std::size_t bit_pos{};

  std::uint64_t bits(int count) {
    if (count < 0 || count > 64 ||
        static_cast<std::size_t>(count) > wire.size() * 8 - bit_pos)
      throw std::runtime_error("truncated Chimp128 bitstream");
    std::uint64_t value = 0;
    for (int i = 0; i < count; ++i) {
      value = (value << 1) |
              ((wire[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1u);
      ++bit_pos;
    }
    return value;
  }

  void require_zero_padding() const {
    for (std::size_t pos = bit_pos; pos < wire.size() * 8; ++pos) {
      if (((wire[pos / 8] >> (7 - (pos % 8))) & 1u) != 0)
        throw std::runtime_error("non-zero Chimp128 trailing bits");
    }
  }

  bool has_only_zero_padding() const {
    for (std::size_t pos = bit_pos; pos < wire.size() * 8; ++pos) {
      if (((wire[pos / 8] >> (7 - (pos % 8))) & 1u) != 0) return false;
    }
    return true;
  }

  void require_marker_escape() {
    if (bits(2) != 1 || bits(kHistoryLog2) != kMarkerEscapePrevious ||
        bits(3) != kMarkerEscapeLeadingClass || bits(6) != 0)
      throw std::runtime_error("invalid Chimp128 end-marker escape");
  }
};

std::vector<std::uint8_t> inverse_wire(std::span<const std::uint8_t> wire,
                                       bool floating) {
  const std::uint64_t end = floating ? kFloatEnd : kIntEnd;
  BitReader reader{wire};
  const auto first = reader.bits(64);
  if (first == end) {
    if (reader.has_only_zero_padding())
      return serialize(Packet{floating, 0, first, {}});
    reader.require_marker_escape();
  }

  std::vector<std::int64_t> values{static_cast<std::int64_t>(first)};
  std::uint64_t stored[kHistory]{};
  stored[0] = first;
  int current = 0;
  int stored_leading = first == end ? 65 : std::numeric_limits<int>::max();
  while (true) {
    const int control = static_cast<int>(reader.bits(2));
    Record record{};
    bool escaped_marker = false;
    if (control == 0) {
      record.previous = static_cast<std::uint8_t>(reader.bits(kHistoryLog2));
      record.trailing = 64;
      record.xor_value = 0;
      stored_leading = 65;
    } else if (control == 1) {
      record.previous = static_cast<std::uint8_t>(reader.bits(kHistoryLog2));
      const int leading_class = static_cast<int>(reader.bits(3));
      const int leading = kDecodeLeading[leading_class];
      const int significant_code = static_cast<int>(reader.bits(6));
      if (record.previous == kMarkerEscapePrevious &&
          leading_class == kMarkerEscapeLeadingClass &&
          significant_code == 0) {
        escaped_marker = true;
        record.xor_value = 0;
        stored_leading = 65;
      }
      int significant = significant_code == 0 ? 64 : significant_code;
      if (escaped_marker) {
        // The reserved escape carries the source marker literally and has no
        // significant payload bits.
        significant = 0;
      }
      if (leading + significant > 64)
        throw std::runtime_error("invalid Chimp128 significant width");
      if (!escaped_marker) {
        record.trailing = static_cast<std::uint8_t>(64 - leading - significant);
        record.xor_value = reader.bits(significant) << record.trailing;
        stored_leading = 65;
      }
    } else {
      record.previous = static_cast<std::uint8_t>(current);
      int leading = stored_leading;
      if (control == 3) {
        const int leading_class = static_cast<int>(reader.bits(3));
        leading = kDecodeLeading[leading_class];
        stored_leading = leading;
      } else if (leading == std::numeric_limits<int>::max() || leading > 64) {
        throw std::runtime_error("Chimp128 reused an unset leading-zero window");
      }
      record.xor_value = reader.bits(64 - leading);
      record.trailing = static_cast<std::uint8_t>(
          record.xor_value == 0 ? 64 : __builtin_ctzll(record.xor_value));
    }

    const auto value = escaped_marker
                           ? end
                           : stored[record.previous] ^ record.xor_value;
    current = (current + 1) % kHistory;
    stored[current] = value;
    if (value == end && !escaped_marker) {
      reader.require_zero_padding();
      break;
    }
    values.push_back(static_cast<std::int64_t>(value));
  }
  // Some branch-selection metadata is intentionally not carried in the
  // public stream once its control case is known. Rebuild the unique
  // canonical XOR packet from the values decoded by this stage-local parser.
  return serialize(build(floating, values));
}

std::vector<std::uint8_t> forward(bool floating,
                                  std::span<const std::int64_t> values) {
  return serialize(build(floating, values));
}

}  // namespace

std::vector<std::uint8_t> xor_forward_int64(std::span<const std::int64_t> values) { return forward(false, values); }
std::vector<std::uint8_t> xor_forward_double_bits(std::span<const std::int64_t> values) { return forward(true, values); }

std::vector<std::int64_t> xor_inverse(std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet);
  if (parsed.count == 0) return {};
  std::uint64_t stored[kHistory]{};
  int current = 0;
  stored[0] = parsed.first;
  std::vector<std::int64_t> values{static_cast<std::int64_t>(parsed.first)};
  for (std::size_t i = 0; i + 1 < parsed.records.size(); ++i) {
    const auto& record = parsed.records[i];
    const auto value = stored[record.previous] ^ record.xor_value;
    current = (current + 1) % kHistory;
    stored[current] = value;
    values.push_back(static_cast<std::int64_t>(value));
  }
  if (values.size() != parsed.count) throw std::runtime_error("Chimp128 inverse count mismatch");
  return values;
}

std::vector<std::uint8_t> trailing_pack_forward(std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet);
  BitWriter writer;
  writer.bits(parsed.first, 64);
  const auto end = parsed.floating ? kFloatEnd : kIntEnd;
  if (parsed.count != 0 && parsed.first == end) writer.marker_escape();
  std::uint64_t stored[kHistory]{};
  stored[0] = parsed.first;
  int current = 0;
  for (std::size_t index = 0; index < parsed.records.size(); ++index) {
    const auto& record = parsed.records[index];
    const auto xor_value = record.xor_value;
    const auto value = stored[record.previous] ^ xor_value;
    const bool terminal = index + 1 == parsed.records.size();
    if (!terminal && value == end) {
      writer.marker_escape();
      current = (current + 1) % kHistory;
      stored[current] = value;
      continue;
    }
    const int leading = xor_value == 0 ? 0 : kLeadingRound[std::min(__builtin_clzll(xor_value), 63)];
    residual_window_pack::chimp(
        [&](std::uint64_t v, int n) { if (n) writer.bits(v, n); },
        xor_value, record.previous, leading, record.trailing,
        kLeadingRepresentation[leading], writer.stored_leading,
        {64, kHistoryLog2, 6, kThreshold});
    current = (current + 1) % kHistory;
    stored[current] = value;
  }
  writer.finish();
  return writer.out;
}

std::vector<std::uint8_t> trailing_pack_inverse_int64(std::span<const std::uint8_t> wire) { return inverse_wire(wire, false); }
std::vector<std::uint8_t> trailing_pack_inverse_double(std::span<const std::uint8_t> wire) { return inverse_wire(wire, true); }
std::vector<std::uint8_t> compose_int64(std::span<const std::int64_t> values) { return trailing_pack_forward(xor_forward_int64(values)); }
std::vector<std::uint8_t> compose_double_bits(std::span<const std::int64_t> values) { return trailing_pack_forward(xor_forward_double_bits(values)); }
std::vector<std::int64_t> decompose_int64(std::span<const std::uint8_t> wire) { return xor_inverse(trailing_pack_inverse_int64(wire)); }
std::vector<std::int64_t> decompose_double_bits(std::span<const std::uint8_t> wire) { return xor_inverse(trailing_pack_inverse_double(wire)); }

}  // namespace exact_actions::chimp128
