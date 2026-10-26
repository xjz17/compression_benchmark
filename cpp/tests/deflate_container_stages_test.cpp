#include "deflate_container_stages.hpp"

#include <zlib.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint8_t> compress(const std::vector<std::uint8_t>& input,
                                   int window_bits, int level) {
  z_stream stream{};
  if (deflateInit2(&stream, level, Z_DEFLATED, window_bits, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("deflateInit2 failed");
  stream.next_in = const_cast<Bytef*>(input.data());
  stream.avail_in = static_cast<uInt>(input.size());
  std::vector<std::uint8_t> out(deflateBound(&stream, input.size()));
  stream.next_out = out.data();
  stream.avail_out = static_cast<uInt>(out.size());
  const int status = deflate(&stream, Z_FINISH);
  const std::size_t size = stream.total_out;
  deflateEnd(&stream);
  if (status != Z_STREAM_END) throw std::runtime_error("deflate failed");
  out.resize(size);
  return out;
}

}  // namespace

int main() {
  using namespace exact_actions::deflate_container;
  std::vector<std::vector<std::uint8_t>> probes{{}, {'a'}, std::vector<std::uint8_t>(4096, 7)};
  for (const auto& input : probes) {
    const auto gzip = compress(input, 15 + 16, 6);
    if (gzip_forward(gzip_inverse(gzip)) != gzip)
      throw std::runtime_error("gzip container roundtrip failed");
    const auto zlib = compress(input, 15, 9);
    if (zlib_forward(zlib_inverse(zlib)) != zlib)
      throw std::runtime_error("zlib container roundtrip failed");
  }
  std::cout << "deflate_container_actions=2 status=PASS\n";
}
