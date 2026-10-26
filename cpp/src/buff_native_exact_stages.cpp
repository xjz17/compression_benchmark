#include "buff_native_exact_stages.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace exact_actions::buff_native {
namespace {

void require_words(std::span<const std::uint8_t> data) {
  if (data.size() % 8 != 0)
    throw std::invalid_argument("BUFF native input requires complete float64 words");
  if (data.size() / 8 > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("BUFF native row count exceeds uint32");
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift != 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> wire, std::size_t offset) {
  std::uint32_t value = 0;
  for (int i = 0; i != 4; ++i)
    value |= static_cast<std::uint32_t>(wire[offset + i]) << (8 * i);
  return value;
}

}  // namespace

std::vector<std::uint8_t> word_order_forward(std::span<const std::uint8_t> data) {
  require_words(data);
  std::vector<std::uint8_t> out(data.begin(), data.end());
  // The original bit writer emits the high uint32 before the low uint32;
  // within each uint32 its least-significant byte is emitted first.
  for (std::size_t i = 0; i < out.size(); i += 8)
    for (std::size_t byte = 0; byte != 4; ++byte)
      std::swap(out[i + byte], out[i + 4 + byte]);
  return out;
}

std::vector<std::uint8_t> word_order_inverse(std::span<const std::uint8_t> words) {
  return word_order_forward(words);
}

std::vector<std::uint8_t> frame_forward(std::span<const std::uint8_t> words) {
  require_words(words);
  if (words.empty()) return {};
  std::vector<std::uint8_t> out;
  out.reserve(12 + words.size());
  put_u32(out, static_cast<std::uint32_t>(words.size() / 8));
  put_u32(out, 64);
  put_u32(out, 0);
  out.insert(out.end(), words.begin(), words.end());
  return out;
}

std::vector<std::uint8_t> frame_inverse(std::span<const std::uint8_t> wire) {
  if (wire.empty()) return {};
  if (wire.size() < 12)
    throw std::invalid_argument("BUFF native truncated frame header");
  const auto count = get_u32(wire, 0);
  if (count == 0 || get_u32(wire, 4) != 64 || get_u32(wire, 8) != 0)
    throw std::invalid_argument("BUFF native frame is not raw float64 storage");
  if (static_cast<std::uint64_t>(count) * 8 != wire.size() - 12)
    throw std::invalid_argument("BUFF native frame length mismatch");
  return {wire.begin() + 12, wire.end()};
}

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> float64_le) {
  return frame_forward(word_order_forward(float64_le));
}

std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire) {
  return word_order_inverse(frame_inverse(wire));
}

}  // namespace exact_actions::buff_native
