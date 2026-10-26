#include "deflate_container_stages.hpp"

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
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("missing DEFLATE container command");
  const std::string command = argv[1];
  const auto input = read_all();
  using namespace exact_actions::deflate_container;
  if (command == "gzip-forward") write(gzip_forward(input));
  else if (command == "gzip-inverse") write(gzip_inverse(input));
  else if (command == "zlib-forward") write(zlib_forward(input));
  else if (command == "zlib-inverse") write(zlib_inverse(input));
  else throw std::invalid_argument("unknown DEFLATE container command");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
