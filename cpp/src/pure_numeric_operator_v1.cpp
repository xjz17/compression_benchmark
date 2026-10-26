// Independent v1 typed primitives. No legacy OPA1/C206 code is modified.
// zigzag-* operate on raw 64-bit lanes ONLY: no packing, compression or frame.
#include "zigzag.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

using Bytes = std::vector<std::uint8_t>;
using U = std::uint64_t;

static void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
static U get(const Bytes& input, std::size_t offset, unsigned width) {
  require(offset <= input.size() && width <= input.size() - offset, "truncated input");
  U value = 0;
  for (unsigned i = 0; i < width; ++i) value |= U{input[offset + i]} << (8 * i);
  return value;
}
static void put(Bytes& output, U value, unsigned width) {
  for (unsigned i = 0; i < width; ++i) output.push_back(value >> (8 * i));
}
static Bytes zigzag(const Bytes& input, bool inverse) {
  require(input.size() % 8 == 0, "ZigZag requires complete 64-bit lanes");
  Bytes output; output.reserve(input.size());
  for (std::size_t offset = 0; offset < input.size(); offset += 8) {
    const U word = get(input, offset, 8);
    // All arithmetic is unsigned: no signed overflow or right-shift assumption.
    const U result = inverse
        ? std::bit_cast<U>(compression_operators::zigzag::decode64(word))
        : compression_operators::zigzag::encode64(std::bit_cast<std::int64_t>(word));
    put(output, result, 8);
  }
  return output;
}
static Bytes compact_forward(const Bytes& input) {
  require(input.size() % 8 == 0, "compact_uint64 requires complete uint64 lanes");
  U maximum = 0;
  for (std::size_t i = 0; i < input.size(); i += 8) maximum = std::max(maximum, get(input, i, 8));
  const unsigned width = input.empty() ? 0 : maximum <= UINT8_MAX ? 1 :
      maximum <= UINT16_MAX ? 2 : maximum <= UINT32_MAX ? 4 : 8;
  // This packer's own explicit metadata is 13 bytes: magic, count, byte width.
  Bytes output{'U', 'C', 'V', '1'}; put(output, input.size() / 8, 8); put(output, width, 1);
  for (std::size_t i = 0; i < input.size(); i += 8) put(output, get(input, i, 8), width);
  return output;
}
static Bytes compact_inverse(const Bytes& input) {
  require(input.size() >= 13 && std::equal(input.begin(), input.begin() + 4, "UCV1"),
          "invalid compact_uint64 header");
  const U count = get(input, 4, 8), width = get(input, 12, 1);
  require((count == 0 && width == 0) || (count != 0 && (width == 1 || width == 2 || width == 4 || width == 8)),
          "invalid compact_uint64 count/width");
  require(width == 0 ? input.size() == 13 :
          ((input.size() - 13) % width == 0 && count == (input.size() - 13) / width),
          "compact_uint64 payload length mismatch");
  require(count <= std::numeric_limits<std::size_t>::max() / 8, "compact_uint64 size overflow");
  Bytes output; output.reserve(static_cast<std::size_t>(count) * 8);
  for (U i = 0; i < count; ++i) put(output, get(input, 13 + i * width, width), 8);
  return output;
}
static Bytes zlib_forward(const Bytes& input) {
  require(input.size() <= std::numeric_limits<uLong>::max(), "zlib input too large");
  uLongf count = compressBound(input.size()); Bytes output(count);
  const auto* data = input.empty() ? reinterpret_cast<const Bytef*>("") : input.data();
  require(compress2(output.data(), &count, data, input.size(), 6) == Z_OK, "zlib6 failed");
  output.resize(count); return output;
}
static Bytes zlib_inverse(const Bytes& input) {
  require(input.size() <= UINT_MAX, "zlib input too large");
  z_stream stream{}; require(inflateInit(&stream) == Z_OK, "inflate initialization failed");
  stream.next_in = const_cast<Bytef*>(input.data()); stream.avail_in = input.size();
  Bytes output; std::array<std::uint8_t, 32768> block{}; int status = Z_OK;
  while (status == Z_OK) {
    stream.next_out = block.data(); stream.avail_out = block.size();
    status = inflate(&stream, Z_NO_FLUSH);
    output.insert(output.end(), block.begin(), block.begin() + block.size() - stream.avail_out);
    if (output.size() > (U{1} << 30)) { inflateEnd(&stream); throw std::runtime_error("inflate exceeds 1 GiB safety limit"); }
  }
  const auto remaining = stream.avail_in; inflateEnd(&stream);
  require(status == Z_STREAM_END && remaining == 0, "invalid or trailing zlib stream");
  return output;
}
int main(int argc, char** argv) try {
  require(argc == 2, "expected zigzag-forward|zigzag-inverse|compact-u64-forward|compact-u64-inverse|zlib6-forward|zlib6-inverse");
  const std::string command = argv[1];
  Bytes input{std::istreambuf_iterator<char>(std::cin), {}}, output;
  if (command == "zigzag-forward") output = zigzag(input, false);
  else if (command == "zigzag-inverse") output = zigzag(input, true);
  else if (command == "compact-u64-forward") output = compact_forward(input);
  else if (command == "compact-u64-inverse") output = compact_inverse(input);
  else if (command == "zlib6-forward") output = zlib_forward(input);
  else if (command == "zlib6-inverse") output = zlib_inverse(input);
  else throw std::runtime_error("unknown command");
  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  require(bool(std::cout), "failed to write output"); return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n'; return 2;
}
