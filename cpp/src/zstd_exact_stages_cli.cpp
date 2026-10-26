#include "zstd_exact_stages.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_all() {
  return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
}
void write(const std::vector<std::uint8_t>& value) {
  std::cout.write(reinterpret_cast<const char*>(value.data()), value.size());
}
}
int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("missing Zstd stage command");
  const std::string command = argv[1];
  const auto input = read_all();
  using namespace exact_actions::zstd;
  if (command == "sequence-forward") write(sequence_forward(input));
  else if (command == "sequence-inverse") write(sequence_inverse(input));
  else if (command == "entropy-forward") write(entropy_frame_forward(input));
  else if (command == "entropy-inverse") write(entropy_frame_inverse(input));
  else if (command == "compose") write(compose(input));
  else if (command == "decompose") write(decompose(input));
  else throw std::invalid_argument("unknown Zstd command");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
