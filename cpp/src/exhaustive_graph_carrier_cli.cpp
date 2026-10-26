// Native zlib carrier for exhaustive graph semantic ports.
// Encoding reproduces the historical zlib level-6 OPA1 carrier. This links
// against zlib (a C native dependency); no Python compression kernel is used.
#include <zlib.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t kChunk = 1u << 16;

void write_output(const std::array<unsigned char, kChunk>& output, uInt remaining) {
  const auto produced = output.size() - remaining;
  std::cout.write(reinterpret_cast<const char*>(output.data()), produced);
  if (!std::cout) throw std::runtime_error("carrier output write failed");
}

void encode() {
  z_stream stream{};
  if (deflateInit(&stream, 6) != Z_OK)
    throw std::runtime_error("zlib deflate initialization failed");
  struct Cleanup { z_stream* s; ~Cleanup() { deflateEnd(s); } } cleanup{&stream};
  std::array<unsigned char, kChunk> input{}, output{};
  int status = Z_OK;
  do {
    std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
    if (std::cin.bad()) throw std::runtime_error("carrier input read failed");
    stream.next_in = input.data();
    stream.avail_in = static_cast<uInt>(std::cin.gcount());
    const int flush = std::cin.eof() ? Z_FINISH : Z_NO_FLUSH;
    do {
      stream.next_out = output.data();
      stream.avail_out = static_cast<uInt>(output.size());
      status = deflate(&stream, flush);
      if (status != Z_OK && status != Z_STREAM_END)
        throw std::runtime_error("zlib carrier encode failed");
      write_output(output, stream.avail_out);
    } while (stream.avail_out == 0 || (flush == Z_FINISH && status != Z_STREAM_END));
  } while (status != Z_STREAM_END);
}

void decode() {
  z_stream stream{};
  if (inflateInit(&stream) != Z_OK)
    throw std::runtime_error("zlib inflate initialization failed");
  struct Cleanup { z_stream* s; ~Cleanup() { inflateEnd(s); } } cleanup{&stream};
  std::array<unsigned char, kChunk> input{}, output{};
  bool ended = false;
  while (!ended) {
    std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
    if (std::cin.bad()) throw std::runtime_error("carrier input read failed");
    stream.next_in = input.data();
    stream.avail_in = static_cast<uInt>(std::cin.gcount());
    if (!stream.avail_in) throw std::invalid_argument("truncated zlib carrier");
    do {
      stream.next_out = output.data();
      stream.avail_out = static_cast<uInt>(output.size());
      const int status = inflate(&stream, Z_NO_FLUSH);
      if (status != Z_OK && status != Z_STREAM_END)
        throw std::invalid_argument("invalid zlib carrier");
      write_output(output, stream.avail_out);
      if (status == Z_STREAM_END) {
        ended = true;
        if (stream.avail_in != 0 || std::cin.peek() != std::char_traits<char>::eof())
          throw std::invalid_argument("trailing bytes after zlib carrier");
        break;
      }
    } while (stream.avail_in != 0 || stream.avail_out == 0);
  }
}
}  // namespace

int main(int argc, char** argv) try {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  if (argc != 2) throw std::invalid_argument("expected encode, decode or version");
  const std::string command = argv[1];
  if (command == "encode") encode();
  else if (command == "decode") decode();
  else if (command == "version") {
    std::cout << "exhaustive-graph-zlib-carrier-v1 zlib=" << zlibVersion()
              << " compile-zlib=" << ZLIB_VERSION << " level=6\n";
  } else throw std::invalid_argument("unknown carrier command");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
