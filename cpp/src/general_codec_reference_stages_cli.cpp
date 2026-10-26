#include "general_codec_reference_stages.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected one reference-stage command");
    const std::string command = argv[1];
    const std::vector<std::uint8_t> input(
        (std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    using namespace reference_actions::general;
    std::vector<std::uint8_t> output;
    if (command == "dictionary-forward") output = dictionary_forward(input);
    else if (command == "dictionary-inverse") output = dictionary_inverse(input);
    else if (command == "brotli-lz-forward") output = lz_forward(input, 65535, 4, 258, true);
    else if (command == "zstd-lz-forward") output = lz_forward(input, 65535, 3, 258, true);
    else if (command == "blosclz-forward") output = lz_forward(input, 65535, 4, 258, false);
    else if (command == "deflate-lz-forward") output = lz_forward(input, 32768, 3, 258, true);
    else if (command == "xz-lz-forward") output = lz_forward(input, 16777216, 3, 273, true);
    else if (command == "lz-inverse") output = lz_inverse(input);
    else if (command == "token-forward") output = token_forward(input);
    else if (command == "token-inverse") output = token_inverse(input);
    else if (command == "parse-forward") output = parse_policy_forward(input);
    else if (command == "parse-inverse") output = parse_policy_inverse(input);
    else if (command == "huffman-forward") output = huffman_forward(input);
    else if (command == "huffman-inverse") output = huffman_inverse(input);
    else if (command == "rans-forward") output = rans_forward(input);
    else if (command == "rans-inverse") output = rans_inverse(input);
    else if (command == "range-forward") output = range_forward(input);
    else if (command == "range-inverse") output = range_inverse(input);
    else if (command == "shuffle1-forward") output = shuffle_forward(input, 1);
    else if (command == "shuffle4-forward") output = shuffle_forward(input, 4);
    else if (command == "shuffle8-forward") output = shuffle_forward(input, 8);
    else if (command == "shuffle-inverse") output = shuffle_inverse(input);
    else if (command == "blosc-frame-forward") output = frame_forward(input, false);
    else if (command == "blosc-frame-inverse") output = frame_inverse(input, false);
    else if (command == "blosc2-frame-forward") output = frame_forward(input, true);
    else if (command == "blosc2-frame-inverse") output = frame_inverse(input, true);
    else if (command == "gzip-frame-forward") output = gzip_frame_forward(input);
    else if (command == "gzip-frame-inverse") output = gzip_frame_inverse(input);
    else if (command == "zlib-frame-forward") output = zlib_frame_forward(input);
    else if (command == "zlib-frame-inverse") output = zlib_frame_inverse(input);
    else if (command == "xz-frame-forward") output = xz_frame_forward(input);
    else if (command == "xz-frame-inverse") output = xz_frame_inverse(input);
    else if (command == "bwt-forward") output = bwt_forward(input);
    else if (command == "bwt-inverse") output = bwt_inverse(input);
    else if (command == "mtf-forward") output = mtf_forward(input);
    else if (command == "mtf-inverse") output = mtf_inverse(input);
    else if (command == "rle-forward") output = rle_forward(input);
    else if (command == "rle-inverse") output = rle_inverse(input);
    else throw std::invalid_argument("unknown reference-stage command: " + command);
    std::cout.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output.size()));
    return std::cout ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
