#include "lzf_operator_ablation.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

extern "C" {
unsigned int lzf_compress(const void* input, unsigned int input_size, void* output,
                          unsigned int output_size);
unsigned int lzf_compress_very(const void* input, unsigned int input_size, void* output,
                               unsigned int output_size);
unsigned int lzf_compress_very_small_window(const void* input, unsigned int input_size,
                                            void* output, unsigned int output_size);
unsigned int lzf_decompress(const void* input, unsigned int input_size, void* output,
                            unsigned int output_size);
}

namespace lzf_operator_ablation {
namespace {

constexpr std::size_t kMaxLiteral = 32;
constexpr std::size_t kMaxOffset = 8192;
constexpr std::size_t kMaxMatch = 255;
constexpr std::array<std::uint8_t, 4> kByteAlignedMagic = {'L', 'Z', 'A', '1'};

std::vector<std::uint8_t> encode_lzf_literals(std::span<const std::uint8_t> input) {
  std::vector<std::uint8_t> output;
  output.reserve(input.size() + (input.size() + kMaxLiteral - 1) / kMaxLiteral);
  std::size_t position = 0;
  while (position < input.size()) {
    const std::size_t length = std::min(kMaxLiteral, input.size() - position);
    output.push_back(static_cast<std::uint8_t>(length - 1));
    output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(position),
                  input.begin() + static_cast<std::ptrdiff_t>(position + length));
    position += length;
  }
  return output;
}

int key_at(std::span<const std::uint8_t> input, std::size_t position) {
  if (position + 2 >= input.size()) return -1;
  return (static_cast<int>(input[position]) << 16) |
         (static_cast<int>(input[position + 1]) << 8) | input[position + 2];
}

std::pair<std::size_t, std::size_t> find_match(
    std::span<const std::uint8_t> input, std::size_t position,
    const std::unordered_map<int, std::size_t>& latest) {
  const int key = key_at(input, position);
  if (key < 0) return {0, 0};
  const auto it = latest.find(key);
  if (it == latest.end() || position <= it->second) return {0, 0};
  const std::size_t offset = position - it->second;
  if (offset > kMaxOffset) return {0, 0};
  const std::size_t limit = std::min(kMaxMatch, input.size() - position);
  std::size_t length = 0;
  while (length < limit && input[it->second + length] == input[position + length]) ++length;
  return length >= 3 ? std::pair{length, offset} : std::pair<std::size_t, std::size_t>{0, 0};
}

void remember(std::unordered_map<int, std::size_t>& latest,
              std::span<const std::uint8_t> input, std::size_t position) {
  const int key = key_at(input, position);
  if (key >= 0) latest[key] = position;
}

std::vector<std::uint8_t> encode_byte_aligned(std::span<const std::uint8_t> input) {
  std::vector<std::uint8_t> output(kByteAlignedMagic.begin(), kByteAlignedMagic.end());
  std::unordered_map<int, std::size_t> latest;
  std::size_t position = 0;
  while (position < input.size()) {
    const auto [match_length, offset] = find_match(input, position, latest);
    if (match_length >= 3) {
      output.push_back(1);
      output.push_back(static_cast<std::uint8_t>(offset & 0xff));
      output.push_back(static_cast<std::uint8_t>((offset >> 8) & 0xff));
      output.push_back(static_cast<std::uint8_t>(match_length));
      for (std::size_t i = 0; i < match_length; ++i) remember(latest, input, position + i);
      position += match_length;
      continue;
    }
    const std::size_t literal_start = position++;
    remember(latest, input, literal_start);
    while (position < input.size() && position - literal_start < 255) {
      const auto next = find_match(input, position, latest);
      if (next.first >= 3) break;
      remember(latest, input, position++);
    }
    output.push_back(0);
    output.push_back(static_cast<std::uint8_t>(position - literal_start));
    output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(literal_start),
                  input.begin() + static_cast<std::ptrdiff_t>(position));
  }
  return output;
}

std::vector<std::uint8_t> decode_byte_aligned(std::span<const std::uint8_t> encoded,
                                              std::size_t expected_size) {
  if (encoded.size() < kByteAlignedMagic.size() ||
      !std::equal(kByteAlignedMagic.begin(), kByteAlignedMagic.end(), encoded.begin())) {
    throw std::runtime_error("LZF byte-aligned ablation magic mismatch");
  }
  std::vector<std::uint8_t> output;
  output.reserve(expected_size);
  std::size_t position = kByteAlignedMagic.size();
  while (output.size() < expected_size) {
    if (position >= encoded.size()) throw std::runtime_error("LZF byte-aligned stream truncated");
    const std::uint8_t tag = encoded[position++];
    if (tag == 0) {
      if (position >= encoded.size()) throw std::runtime_error("LZF literal length missing");
      const std::size_t length = encoded[position++];
      if (length == 0 || position + length > encoded.size() || output.size() + length > expected_size) {
        throw std::runtime_error("LZF literal token invalid");
      }
      output.insert(output.end(), encoded.begin() + static_cast<std::ptrdiff_t>(position),
                    encoded.begin() + static_cast<std::ptrdiff_t>(position + length));
      position += length;
    } else if (tag == 1) {
      if (position + 3 > encoded.size()) throw std::runtime_error("LZF match token truncated");
      const std::size_t offset = encoded[position] | (static_cast<std::size_t>(encoded[position + 1]) << 8);
      const std::size_t length = encoded[position + 2];
      position += 3;
      if (offset == 0 || offset > output.size() || length < 3 || output.size() + length > expected_size) {
        throw std::runtime_error("LZF match token invalid");
      }
      const std::size_t start = output.size() - offset;
      for (std::size_t i = 0; i < length; ++i) output.push_back(output[start + i]);
    } else {
      throw std::runtime_error("LZF byte-aligned token tag invalid");
    }
  }
  if (position != encoded.size()) throw std::runtime_error("LZF byte-aligned trailing bytes");
  return output;
}

std::vector<std::uint8_t> encode_upstream(std::span<const std::uint8_t> input, int level,
                                          bool small_window) {
  const std::size_t bound = input.size() + input.size() / 8 + 1024;
  std::vector<std::uint8_t> output(bound);
  unsigned int size = 0;
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("LZF input exceeds upstream API limit");
  }
  if (small_window) {
    size = lzf_compress_very_small_window(input.data(), static_cast<unsigned int>(input.size()),
                                          output.data(), static_cast<unsigned int>(output.size()));
  } else if (level == 0) {
    size = lzf_compress(input.data(), static_cast<unsigned int>(input.size()), output.data(),
                        static_cast<unsigned int>(output.size()));
  } else {
    size = lzf_compress_very(input.data(), static_cast<unsigned int>(input.size()), output.data(),
                             static_cast<unsigned int>(output.size()));
  }
  if (input.empty()) return {};
  if (size == 0) throw std::runtime_error("LZF upstream compression failed");
  output.resize(size);
  return output;
}

}  // namespace

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> input, int level,
                                 std::string_view omitted_operator) {
  if (omitted_operator.empty()) return encode_upstream(input, level, false);
  if (omitted_operator == "lz_match") return encode_lzf_literals(input);
  if (omitted_operator == "match_window_model") return encode_upstream(input, level, true);
  if (omitted_operator == "match_token_format") return encode_byte_aligned(input);
  throw std::invalid_argument("unsupported LZF ablation operator");
}

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded,
                                 std::size_t expected_size, int level,
                                 std::string_view omitted_operator) {
  if (omitted_operator == "match_token_format") return decode_byte_aligned(encoded, expected_size);
  if (omitted_operator == "lz_match") {
    std::vector<std::uint8_t> output;
    output.reserve(expected_size);
    std::size_t position = 0;
    while (position < encoded.size()) {
      const std::size_t length = static_cast<std::size_t>(encoded[position++]) + 1;
      if (position + length > encoded.size() || output.size() + length > expected_size) {
        throw std::runtime_error("LZF literal-only stream invalid");
      }
      output.insert(output.end(), encoded.begin() + static_cast<std::ptrdiff_t>(position),
                    encoded.begin() + static_cast<std::ptrdiff_t>(position + length));
      position += length;
    }
    if (output.size() != expected_size) throw std::runtime_error("LZF literal-only size mismatch");
    return output;
  }
  std::vector<std::uint8_t> output(expected_size);
  if (encoded.size() > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) ||
      expected_size > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("LZF output exceeds upstream API limit");
  }
  const unsigned int size = lzf_decompress(encoded.data(), static_cast<unsigned int>(encoded.size()),
                                           output.data(), static_cast<unsigned int>(output.size()));
  if (size != expected_size) throw std::runtime_error("LZF upstream decompression failed");
  return output;
}

}  // namespace lzf_operator_ablation
