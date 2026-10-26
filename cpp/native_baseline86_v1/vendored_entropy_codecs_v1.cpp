#include "vendored_entropy_codecs_v1.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

extern "C" {
#include "fse.h"
#include "huf.h"
}

namespace native_baseline86_v1::vendored_entropy {
namespace {
constexpr std::size_t chunk_size = 128U << 10;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

void put_u32(Bytes& output, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    output.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& pos) {
  require(pos <= input.size() && input.size() - pos >= 4,
          "entropy adapter: truncated u32");
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= static_cast<std::uint32_t>(input[pos++]) << (8 * i);
  return value;
}

void append(Bytes& output, const std::uint8_t* data, std::size_t size) {
  output.insert(output.end(), data, data + size);
}
}  // namespace

Bytes encode(const std::string& label, std::span<const std::uint8_t> input) {
  require(label == "FSE" || label == "Huff0",
          label + ": unknown vendored entropy codec");
  require(input.size() <= std::numeric_limits<std::uint32_t>::max(),
          label + ": input exceeds adapter range");
  const auto count = static_cast<std::uint32_t>(
      (input.size() + chunk_size - 1) / chunk_size);
  Bytes output;
  output.reserve(input.size() + 16 + 9ULL * count);
  put_u32(output, count);
  for (std::size_t offset = 0; offset < input.size(); offset += chunk_size) {
    const auto size = std::min(chunk_size, input.size() - offset);
    const auto* source = input.data() + offset;
    Bytes compressed(label == "FSE" ? FSE_compressBound(size)
                                     : HUF_compressBound(size));
    const auto written = label == "FSE"
                             ? FSE_compress(compressed.data(), compressed.size(),
                                            source, size)
                             : HUF_compress(compressed.data(), compressed.size(),
                                            source, size);
    const bool error = label == "FSE" ? FSE_isError(written)
                                       : HUF_isError(written);
    require(!error, label + ": upstream compression failed");
    std::uint8_t kind = 0;
    const std::uint8_t* body = source;
    std::size_t body_size = size;
    if (label == "FSE" && written == 1) {
      kind = 2;
      body_size = 1;
    } else if (written > 0 && written < size) {
      kind = 1;
      body = compressed.data();
      body_size = written;
    }
    put_u32(output, static_cast<std::uint32_t>(size));
    output.push_back(kind);
    put_u32(output, static_cast<std::uint32_t>(body_size));
    append(output, body, body_size);
  }
  return output;
}

Bytes decode(const std::string& label, std::span<const std::uint8_t> payload,
             std::size_t expected_size) {
  require(label == "FSE" || label == "Huff0",
          label + ": unknown vendored entropy codec");
  std::size_t pos = 0;
  const auto count = get_u32(payload, pos);
  require(count == (expected_size + chunk_size - 1) / chunk_size,
          label + ": chunk count mismatch");
  Bytes output;
  output.reserve(expected_size);
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto raw_size = get_u32(payload, pos);
    require(raw_size > 0 && raw_size <= chunk_size &&
                raw_size <= expected_size - output.size(),
            label + ": invalid chunk raw size");
    require(pos < payload.size(), label + ": missing chunk kind");
    const auto kind = payload[pos++];
    const auto body_size = get_u32(payload, pos);
    require(pos <= payload.size() && body_size <= payload.size() - pos,
            label + ": truncated chunk body");
    const auto* body = payload.data() + pos;
    const auto old_size = output.size();
    output.resize(old_size + raw_size);
    if (kind == 0) {
      require(body_size == raw_size, label + ": raw chunk size mismatch");
      std::memcpy(output.data() + old_size, body, raw_size);
    } else if (kind == 2) {
      require(label == "FSE" && body_size == 1,
              label + ": invalid RLE chunk");
      std::memset(output.data() + old_size, body[0], raw_size);
    } else {
      require(kind == 1, label + ": unknown chunk kind");
      const auto produced = label == "FSE"
                                ? FSE_decompress(output.data() + old_size,
                                                 raw_size, body, body_size)
                                : HUF_decompress(output.data() + old_size,
                                                 raw_size, body, body_size);
      const bool error = label == "FSE" ? FSE_isError(produced)
                                         : HUF_isError(produced);
      require(!error && produced == raw_size,
              label + ": invalid or wrong-sized compressed chunk");
    }
    pos += body_size;
  }
  require(pos == payload.size() && output.size() == expected_size,
          label + ": trailing data or decoded size mismatch");
  return output;
}

}  // namespace native_baseline86_v1::vendored_entropy
