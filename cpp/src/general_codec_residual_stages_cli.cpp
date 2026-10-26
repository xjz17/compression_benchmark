#include "general_codec_residual_stages.hpp"
#include "general_codec_reference_stages.hpp"
#include "parameterized_deflate.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::invalid_argument("expected a residual-stage command");
    const std::string command = argv[1];
    const std::vector<std::uint8_t> input(
        (std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    using namespace exact_actions::general_residual;
    std::vector<std::uint8_t> output;
    if (command == "deflate-forward" || command == "deflate-inverse") {
      const bool inverse = command == "deflate-inverse";
      if (argc != 4 && !(inverse && argc == 3))
        throw std::invalid_argument("expected deflate-forward IO_VARIANT LEVEL or deflate-inverse IO_VARIANT [LEVEL]");
      const std::string variant = argv[2];
      if (variant != "gzip_prefix" && variant != "zlib_prefix")
        throw std::invalid_argument("DEFLATE io_variant must be gzip_prefix or zlib_prefix");
      const int level = argc == 4 ? exact_actions::parameterized_deflate::parse_level(argv[3])
                                 : (variant == "gzip_prefix" ? 6 : 9);
      if (variant == "gzip_prefix")
        output = inverse ? gzip_inverse(input) : gzip_forward(input, level);
      else
        output = inverse ? zlib_inverse(input) : zlib_forward(input, level);
    } else if (command == "blosc1-forward" || command == "blosc1-inverse" ||
               command == "blosc2-forward" || command == "blosc2-inverse") {
      if (argc != 3)
        throw std::invalid_argument("expected blosc1/blosc2-forward/inverse TYPESIZE");
      const std::string size_text = argv[2];
      if (size_text != "1" && size_text != "4" && size_text != "8")
        throw std::invalid_argument("Blosc typesize must be 1, 4, or 8");
      const int type_size = size_text[0] - '0';
      const bool inverse = command.ends_with("-inverse");
      const bool blosc1 = command.starts_with("blosc1-");
      if (inverse) {
        if (input.size() < 16 || input[3] != type_size)
          throw std::invalid_argument("Blosc typesize does not match the encoded header");
        output = blosc1 ? blosc1_inverse(input, type_size) : blosc2_inverse(input, type_size);
      } else {
        // The inverse reconstructs this prefix at the selected width. A
        // different width would silently change the upstream SHF1 packet.
        namespace ref = reference_actions::general;
        const auto shuffled = ref::lz_inverse(ref::token_inverse(input));
        const auto raw = ref::shuffle_inverse(shuffled);
        if (shuffled[4] != type_size)
          throw std::invalid_argument("Blosc typesize does not match the SHF1 prefix");
        output = blosc1 ? blosc1_encode(raw, type_size) : blosc2_encode(raw, type_size);
      }
    } else {
      if (argc != 2) throw std::invalid_argument("expected one legacy residual-stage command");
      if (command == "original-brotli-encode") output = brotli_encode(input);
      else if (command == "original-zstd-encode") output = zstd_encode(input);
      else if (command == "original-bzip2-encode") output = bzip2_encode(input);
      else if (command == "original-gzip-encode") output = gzip_encode(input);
      else if (command == "original-zlib-encode") output = zlib_encode(input);
      else if (command == "original-xz-encode") output = xz_encode(input);
      else if (command == "brotli-forward") output = brotli_forward(input);
      else if (command == "brotli-inverse") output = brotli_inverse(input);
      else if (command == "zstd-forward") output = zstd_forward(input);
      else if (command == "zstd-inverse") output = zstd_inverse(input);
      else if (command == "bzip2-forward") output = bzip2_forward(input);
      else if (command == "bzip2-inverse") output = bzip2_inverse(input);
      else if (command == "bzip2-optimal-forward") output = bzip2_optimal_forward(input);
      else if (command == "bzip2-optimal-inverse") output = bzip2_optimal_inverse(input);
      else if (command == "gzip-forward") output = gzip_forward(input);
      else if (command == "gzip-inverse") output = gzip_inverse(input);
      else if (command == "zlib-forward") output = zlib_forward(input);
      else if (command == "zlib-inverse") output = zlib_inverse(input);
      else if (command == "xz-forward") output = xz_forward(input);
      else if (command == "xz-inverse") output = xz_inverse(input);
      else {
        const bool inverse = command.find("-inverse-") != std::string::npos;
        int type_size = 0;
        if (!command.empty() && command.back() >= '1' && command.back() <= '8') {
          type_size = command.back() - '0';
        }
        if (type_size != 1 && type_size != 4 && type_size != 8) {
          throw std::invalid_argument("unknown residual-stage command: " + command);
        }
        if (command.rfind("blosc1-", 0) == 0) {
          output = command.rfind("blosc1-original-", 0) == 0
                       ? blosc1_encode(input, type_size)
                       : inverse ? blosc1_inverse(input, type_size)
                                 : blosc1_forward(input, type_size);
        } else if (command.rfind("blosc2-", 0) == 0) {
          output = command.rfind("blosc2-original-", 0) == 0
                       ? blosc2_encode(input, type_size)
                       : inverse ? blosc2_inverse(input, type_size)
                                 : blosc2_forward(input, type_size);
        } else {
          throw std::invalid_argument("unknown residual-stage command: " + command);
        }
      }
    }
    std::cout.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output.size()));
    return std::cout ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
