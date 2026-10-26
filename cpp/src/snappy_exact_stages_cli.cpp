#include "snappy_exact_stages.hpp"

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
}

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("missing Snappy stage command");
  const std::string command = argv[1];
  const auto input = read_all();
  using namespace exact_actions::snappy;
  if (command == "original-encode") write(original_encode(input));
  else if (command == "lz-forward") write(lz_match_forward(input));
  else if (command == "snap111-forward") write(snap111_match_forward(input));
  else if (command == "lz-inverse") write(lz_match_inverse(input));
  else if (command == "token-forward") write(token_format_forward(input));
  else if (command == "token-inverse") write(token_format_inverse(input));
  else if (command == "integer-forward") write(integer_code_forward(input));
  else if (command == "integer-inverse") write(integer_code_inverse(input));
  else if (command == "compose") write(compose(input));
  else if (command == "decompose") write(decompose(input));
  else throw std::invalid_argument("unknown Snappy stage command");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
