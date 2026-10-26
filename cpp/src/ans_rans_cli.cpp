#include "ans_rans.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t hash = 14695981039346656037ull;
  for (const auto byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: ans_rans_cli INPUT.bin [ENCODED.bin]\n";
    return 2;
  }
  try {
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input");
    const std::vector<std::uint8_t> raw{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const auto encoded = ans_rans::encode(raw);
    const auto decoded = ans_rans::decode(encoded);
    if (argc == 3) {
      std::ofstream output(argv[2], std::ios::binary);
      if (!output) throw std::runtime_error("cannot open encoded output");
      output.write(reinterpret_cast<const char*>(encoded.data()),
                   static_cast<std::streamsize>(encoded.size()));
      if (!output) throw std::runtime_error("cannot write encoded output");
    }
    std::cout << raw.size() << ',' << encoded.size() << ',' << fnv1a(encoded)
              << ',' << (decoded == raw ? 1 : 0) << '\n';
    return decoded == raw ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
