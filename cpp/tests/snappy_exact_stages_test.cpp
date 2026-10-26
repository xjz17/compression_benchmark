#include "snappy_exact_stages.hpp"

#include <snappy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

int main() {
  using namespace exact_actions::snappy;
  std::vector<std::vector<std::uint8_t>> probes{
      {},
      {'a'},
      {'a','b','c','a','b','c','a','b','c','a','b','c'},
      std::vector<std::uint8_t>(4096, 7),
  };
  std::vector<std::uint8_t> ramp;
  for (int i = 0; i < 4096; ++i) ramp.push_back(static_cast<std::uint8_t>(i));
  probes.push_back(std::move(ramp));
  for (const auto& input : probes) {
    const auto matches = lz_match_forward(input);
    if (lz_match_inverse(matches) != input) throw std::runtime_error("Snappy LZ roundtrip");
    const auto tokens = token_format_forward(matches);
    if (token_format_inverse(tokens) != matches) throw std::runtime_error("Snappy token roundtrip");
    const auto wire = integer_code_forward(tokens);
    if (integer_code_inverse(wire) != tokens) throw std::runtime_error("Snappy integer roundtrip");
    std::vector<char> expected(snappy::MaxCompressedLength(input.size()));
    std::size_t size = 0;
    snappy::RawCompress(reinterpret_cast<const char*>(input.data()), input.size(),
                        expected.data(), &size);
    expected.resize(size);
    if (wire.size() != expected.size() ||
        std::memcmp(wire.data(), expected.data(), wire.size()) != 0)
      throw std::runtime_error("Snappy public wire parity");
    if (decompose(wire) != input) throw std::runtime_error("Snappy chain roundtrip");
  }
  std::cout << "snappy_exact_stage_actions=3 status=PASS\n";
}
