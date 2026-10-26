#include "spdp_exact_stages.hpp"
#include "operator_layout_primitives.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace exact_actions::spdp {
namespace {

constexpr std::array<std::uint8_t, 4> kDeltaMagic{'S', '2', 'D', '1'};
constexpr std::array<std::uint8_t, 4> kBlockMagic{'S', 'B', 'T', '1'};
constexpr std::size_t kMaxBlock = 1u << 23;
constexpr std::size_t kMaxTable = 1u << 18;

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint64_t read_u64_le(std::span<const std::uint8_t> bytes,
                          std::size_t offset) {
  if (bytes.size() - offset < 8) throw std::runtime_error("truncated SPDP packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
  return value;
}

std::uint64_t load_u64_le(std::span<const std::uint8_t> bytes,
                          std::size_t offset) {
  return read_u64_le(bytes, offset);
}

void store_u64_le(std::span<std::uint8_t> bytes, std::size_t offset,
                  std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    bytes[offset++] = static_cast<std::uint8_t>(value >> shift);
}

std::vector<std::uint8_t> packet(
    const std::array<std::uint8_t, 4>& magic,
    std::span<const std::uint8_t> payload) {
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  append_u64_le(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::span<const std::uint8_t> parse(
    std::span<const std::uint8_t> value,
    const std::array<std::uint8_t, 4>& magic) {
  if (value.size() < 12 || !std::equal(magic.begin(), magic.end(), value.begin()))
    throw std::runtime_error("wrong SPDP packet type");
  const auto size = read_u64_le(value, 4);
  if (size != value.size() - 12) throw std::runtime_error("invalid SPDP packet size");
  return value.subspan(12);
}

std::size_t table_mask(unsigned level) {
  if (level > 9) throw std::invalid_argument("SPDP level must be in [0, 9]");
  return std::min<std::size_t>(std::size_t{1} << (level + 9), kMaxTable) - 1;
}

std::vector<std::uint8_t> entropy_encode(std::span<const std::uint8_t> input,
                                         unsigned level) {
  const std::size_t mask = table_mask(level);
  std::vector<std::uint32_t> last(mask + 1, 0);
  std::vector<std::uint8_t> out;
  out.reserve(input.size());
  std::size_t read = 0;
  std::uint32_t history = 0;
  while (read < input.size()) {
    std::uint8_t value = input[read];
    std::size_t previous = last[history];
    if (previous >= 6 &&
        std::equal(input.begin() + static_cast<std::ptrdiff_t>(previous - 6),
                   input.begin() + static_cast<std::ptrdiff_t>(previous),
                   input.begin() + static_cast<std::ptrdiff_t>(read - 6))) {
      std::uint8_t count = 0;
      while (value == input[previous] && count < 255 && read < input.size() - 1) {
        last[history] = static_cast<std::uint32_t>(read);
        history = ((history << 2) ^ value) & static_cast<std::uint32_t>(mask);
        ++read;
        ++previous;
        ++count;
        value = input[read];
      }
      out.push_back(count);
    }
    out.push_back(value);
    last[history] = static_cast<std::uint32_t>(read);
    history = ((history << 2) ^ value) & static_cast<std::uint32_t>(mask);
    ++read;
  }
  return out;
}

std::vector<std::uint8_t> entropy_decode(std::span<const std::uint8_t> input,
                                         std::size_t output_size,
                                         unsigned level) {
  const std::size_t mask = table_mask(level);
  std::vector<std::uint32_t> last(mask + 1, 0);
  std::vector<std::uint8_t> out(output_size);
  std::size_t read = 0, written = 0;
  std::uint32_t history = 0;
  while (read < input.size()) {
    std::size_t previous = last[history];
    const bool repeat_context =
        previous >= 6 && written >= 6 && previous <= written &&
        std::equal(out.begin() + static_cast<std::ptrdiff_t>(previous - 6),
                   out.begin() + static_cast<std::ptrdiff_t>(previous),
                   out.begin() + static_cast<std::ptrdiff_t>(written - 6));
    if (repeat_context) {
      if (read >= input.size()) throw std::runtime_error("truncated SPDP repeat count");
      const auto count = input[read++];
      for (unsigned i = 0; i < count; ++i) {
        if (previous >= written || written >= output_size)
          throw std::runtime_error("SPDP repeat exceeds output");
        const auto value = out[previous++];
        out[written] = value;
        last[history] = static_cast<std::uint32_t>(written++);
        history = ((history << 2) ^ value) & static_cast<std::uint32_t>(mask);
      }
    }
    if (read >= input.size() || written >= output_size)
      throw std::runtime_error("invalid SPDP literal");
    const auto value = input[read++];
    out[written] = value;
    last[history] = static_cast<std::uint32_t>(written++);
    history = ((history << 2) ^ value) & static_cast<std::uint32_t>(mask);
  }
  if (written != output_size) throw std::runtime_error("SPDP decoded size mismatch");
  return out;
}

}  // namespace

std::vector<std::uint8_t> delta_forward(std::span<const std::uint8_t> raw) {
  if (raw.size() > kMaxBlock || raw.size() % 8 != 0)
    throw std::invalid_argument("SPDP action requires at most 8 MiB of 64-bit words");
  std::vector<std::uint8_t> transformed(raw.size());
  std::uint64_t previous2 = 0, previous1 = 0;
  for (std::size_t offset = 0; offset < raw.size(); offset += 8) {
    const auto current = load_u64_le(raw, offset);
    store_u64_le(transformed, offset, current - previous2);
    previous2 = previous1;
    previous1 = current;
  }
  return packet(kDeltaMagic, transformed);
}

std::vector<std::uint8_t> delta_inverse(std::span<const std::uint8_t> value) {
  const auto deltas = parse(value, kDeltaMagic);
  std::vector<std::uint8_t> raw(deltas.size());
  std::uint64_t previous2 = 0, previous1 = 0;
  for (std::size_t offset = 0; offset < deltas.size(); offset += 8) {
    const auto current = load_u64_le(deltas, offset) + previous2;
    store_u64_le(raw, offset, current);
    previous2 = previous1;
    previous1 = current;
  }
  return raw;
}

std::vector<std::uint8_t> block_transform_forward(
    std::span<const std::uint8_t> value) {
  const auto input = parse(value, kDeltaMagic);
  const auto transformed = operator_layout::column_delta(input, 8);
  return packet(kBlockMagic, transformed);
}

std::vector<std::uint8_t> block_transform_inverse(
    std::span<const std::uint8_t> value) {
  const auto input = parse(value, kBlockMagic);
  const auto deltas = operator_layout::inverse_column_delta(input, 8);
  return packet(kDeltaMagic, deltas);
}

std::vector<std::uint8_t> entropy_forward(
    std::span<const std::uint8_t> value, unsigned level) {
  return entropy_encode(parse(value, kBlockMagic), level);
}

std::vector<std::uint8_t> entropy_inverse(
    std::span<const std::uint8_t> wire, std::size_t original_size,
    unsigned level) {
  if (original_size > kMaxBlock || original_size % 8 != 0)
    throw std::invalid_argument("invalid SPDP original size");
  return packet(kBlockMagic, entropy_decode(wire, original_size, level));
}

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw,
                                  unsigned level) {
  return entropy_forward(block_transform_forward(delta_forward(raw)), level);
}

std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire,
                                    std::size_t original_size,
                                    unsigned level) {
  return delta_inverse(block_transform_inverse(
      entropy_inverse(wire, original_size, level)));
}

}  // namespace exact_actions::spdp
