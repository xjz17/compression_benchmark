#include "general_codec_reference_stages.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;

void expect(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

}  // namespace

int main() {
  using namespace reference_actions::general;
  const std::string text =
      "the sensor temperature value and compression column; "
      "the sensor temperature value and compression column; "
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const Bytes raw(text.begin(), text.end());

  auto dictionary = dictionary_forward(raw);
  expect(dictionary_inverse(dictionary) == raw, "dictionary round trip");

  for (const auto& parameters : std::vector<std::array<std::size_t, 4>>{
           {65535, 4, 258, 1}, {65535, 3, 258, 1}, {65535, 4, 258, 0}}) {
    auto semantic = lz_forward(raw, parameters[0], parameters[1], parameters[2],
                               parameters[3] != 0);
    expect(lz_inverse(semantic) == raw, "LZ semantic round trip");
    auto packed = token_forward(semantic);
    expect(token_inverse(packed) == semantic, "LZ token round trip");
  }

  auto zstd_lz = lz_forward(raw, 65535, 3, 258, true);
  auto parsed = parse_policy_forward(zstd_lz);
  expect(parse_policy_inverse(parsed) == zstd_lz, "parse-policy round trip");
  auto huffman = huffman_forward(parsed);
  expect(huffman_inverse(huffman) == parsed, "Huffman round trip");
  auto rans = rans_forward(huffman);
  expect(rans_inverse(rans) == huffman, "rANS round trip");
  auto range = range_forward(rans);
  expect(range_inverse(range) == rans, "range round trip");

  for (std::size_t type_size : {std::size_t(1), std::size_t(4), std::size_t(8)}) {
    auto shuffled = shuffle_forward(raw, type_size);
    expect(shuffle_inverse(shuffled) == raw, "shuffle round trip");
  }
  auto blosc_frame = frame_forward(raw, false);
  expect(frame_inverse(blosc_frame, false) == raw, "Blosc frame round trip");
  auto blosc2_frame = frame_forward(raw, true);
  expect(frame_inverse(blosc2_frame, true) == raw, "Blosc2 frame round trip");
  expect(gzip_frame_inverse(gzip_frame_forward(raw)) == raw, "gzip frame round trip");
  expect(zlib_frame_inverse(zlib_frame_forward(raw)) == raw, "zlib frame round trip");
  expect(xz_frame_inverse(xz_frame_forward(raw)) == raw, "XZ frame round trip");

  for (const Bytes probe : {Bytes{}, Bytes{'x'}, raw, Bytes{0, 0, 1, 0, 0, 1}}) {
    auto bwt = bwt_forward(probe);
    expect(bwt_inverse(bwt) == probe, "BWT round trip");
    auto mtf = mtf_forward(bwt);
    expect(mtf_inverse(mtf) == bwt, "MTF round trip");
    auto rle = rle_forward(mtf);
    expect(rle_inverse(rle) == mtf, "RLE round trip");
    auto entropy = huffman_forward(rle);
    expect(huffman_inverse(entropy) == rle, "Bzip2 reference chain round trip");
  }

  std::cout << "general codec reference stages PASS\n";
  return 0;
}
