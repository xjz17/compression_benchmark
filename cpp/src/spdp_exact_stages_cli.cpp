#include "spdp_exact_stages.hpp"

#include <cstdlib>
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
  std::cout.write(reinterpret_cast<const char*>(value.data()),
                  static_cast<std::streamsize>(value.size()));
}
std::size_t size_arg(int argc, char** argv) {
  if (argc != 3) throw std::invalid_argument("inverse command requires original byte count");
  char* end = nullptr;
  const auto value = std::strtoull(argv[2], &end, 10);
  if (end == argv[2] || *end != '\0') throw std::invalid_argument("invalid byte count");
  return static_cast<std::size_t>(value);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) throw std::invalid_argument("missing SPDP stage command");
  const std::string command = argv[1];
  const auto input = read_all();
  using namespace exact_actions::spdp;
  if (command == "delta-forward") write(delta_forward(input));
  else if (command == "delta-inverse") write(delta_inverse(input));
  else if (command == "block-forward") write(block_transform_forward(input));
  else if (command == "block-inverse") write(block_transform_inverse(input));
  else if (command == "entropy-forward") write(entropy_forward(input));
  else if (command == "entropy-inverse") write(entropy_inverse(input, size_arg(argc, argv)));
  else if (command == "compose") write(compose(input));
  else if (command == "decompose") write(decompose(input, size_arg(argc, argv)));
  else throw std::invalid_argument("unknown SPDP command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
