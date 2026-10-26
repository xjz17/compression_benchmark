#include "pongo_exact_stages.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_all() {
  return {std::istreambuf_iterator<char>(std::cin),
          std::istreambuf_iterator<char>()};
}

void write(const std::vector<std::uint8_t>& value) {
  std::cout.write(reinterpret_cast<const char*>(value.data()), value.size());
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) {
    throw std::invalid_argument("missing Pongo stage command");
  }
  const std::string command = argv[1];
  const auto input = read_all();
  using namespace exact_actions::pongo;
  if (command == "eraser-forward") write(eraser_forward(input));
  else if (command == "eraser-inverse") write(eraser_inverse(input));
  else if (command == "xor-forward") write(xor_forward(input));
  else if (command == "xor-inverse") write(xor_inverse(input));
  else if (command == "pack-forward") write(trailing_pack_forward(input));
  else if (command == "pack-inverse") write(trailing_pack_inverse(input));
  else if (command == "compose") write(compose(input));
  else if (command == "decompose") write(decompose(input));
  else throw std::invalid_argument("unknown Pongo stage command");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
