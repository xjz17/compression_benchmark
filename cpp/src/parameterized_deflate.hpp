#pragma once

#include <zlib.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace exact_actions::parameterized_deflate {

using Bytes = std::vector<std::uint8_t>;
enum class Wrapper { zlib = 15, gzip = 31 };

inline int validate_level(int level) {
  if (level < 0 || level > 9)
    throw std::invalid_argument("DEFLATE level must be an integer in [0, 9]");
  return level;
}

inline int parse_level(std::string_view text) {
  int level = -1;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), level);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    throw std::invalid_argument("DEFLATE level must be an integer in [0, 9]");
  return validate_level(level);
}

// Shared by the gzip/zlib prefix adapters and CORAD's CTK2 adapter. The
// wrapper only selects the legacy wire envelope; all use the same DEFLATE
// window, memory level and strategy. Keep one finish path for byte parity.
inline Bytes encode(std::span<const std::uint8_t> raw, int level, Wrapper wrapper) {
  validate_level(level);
  if (raw.size() > std::numeric_limits<uInt>::max())
    throw std::invalid_argument("DEFLATE input exceeds uint32 size");
  z_stream stream{};
  if (deflateInit2(&stream, level, Z_DEFLATED, static_cast<int>(wrapper), 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("zlib encoder initialization failed");
  struct Cleanup {
    z_stream* stream;
    ~Cleanup() { deflateEnd(stream); }
  } cleanup{&stream};
  stream.next_in = const_cast<Bytef*>(raw.data());
  stream.avail_in = static_cast<uInt>(raw.size());
  Bytes output;
  std::array<std::uint8_t, 65536> chunk{};
  int status;
  do {
    stream.next_out = chunk.data();
    stream.avail_out = static_cast<uInt>(chunk.size());
    status = deflate(&stream, Z_FINISH);
    if (status != Z_OK && status != Z_STREAM_END)
      throw std::runtime_error("zlib compression failed");
    output.insert(output.end(), chunk.begin(), chunk.end() - stream.avail_out);
  } while (status != Z_STREAM_END);
  return output;
}

inline Bytes decode(std::span<const std::uint8_t> wire, Wrapper wrapper,
                    std::optional<std::size_t> expected_size = std::nullopt) {
  if (wire.size() > std::numeric_limits<uInt>::max())
    throw std::invalid_argument("DEFLATE wire exceeds uint32 size");
  z_stream stream{};
  if (inflateInit2(&stream, static_cast<int>(wrapper)) != Z_OK)
    throw std::runtime_error("zlib decoder initialization failed");
  struct Cleanup {
    z_stream* stream;
    ~Cleanup() { inflateEnd(stream); }
  } cleanup{&stream};
  stream.next_in = const_cast<Bytef*>(wire.data());
  stream.avail_in = static_cast<uInt>(wire.size());
  Bytes output;
  std::array<std::uint8_t, 65536> chunk{};
  int status;
  do {
    stream.next_out = chunk.data();
    stream.avail_out = static_cast<uInt>(chunk.size());
    status = inflate(&stream, Z_NO_FLUSH);
    if (status != Z_OK && status != Z_STREAM_END)
      throw std::runtime_error("zlib decompression failed");
    const auto produced = chunk.size() - stream.avail_out;
    if (expected_size &&
        (output.size() > *expected_size || produced > *expected_size - output.size()))
      throw std::runtime_error("DEFLATE decoded size exceeds declared length");
    output.insert(output.end(), chunk.begin(), chunk.begin() + produced);
  } while (status != Z_STREAM_END);
  if (stream.avail_in != 0 || (expected_size && output.size() != *expected_size))
    throw std::runtime_error("DEFLATE wire or decoded length mismatch");
  return output;
}

}  // namespace exact_actions::parameterized_deflate
