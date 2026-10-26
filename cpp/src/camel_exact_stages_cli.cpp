#include "camel_exact_stages.hpp"

#include <cstring>
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
std::vector<std::int64_t> words(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % 8) throw std::runtime_error("float64 input byte count mismatch");
  std::vector<std::int64_t> values(raw.size() / 8);
  if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}
template <class T>
void write(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("missing Camel stage command");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::camel;
  if (command == "eraser-forward") write(eraser_forward(words(raw)));
  else if (command == "eraser-forward-int") write(eraser_forward_int64_numeric(words(raw)));
  else if (command == "eraser-inverse") write(eraser_inverse(raw));
  else if (command == "eraser-inverse-int") write(eraser_inverse_int64_numeric(raw));
  else if (command == "xor-forward") write(xor_forward(raw));
  else if (command == "xor-inverse") write(xor_inverse(raw));
  else if (command == "pack-forward") write(bitstream_pack_forward(raw));
  else if (command == "pack-inverse") write(bitstream_pack_inverse(raw));
  else if (command == "compose") write(compose(words(raw)));
  else if (command == "decompose") write(decompose(raw));
  else if (command == "compose-int") write(compose_int64_numeric(words(raw)));
  else if (command == "decompose-int") write(decompose_int64_numeric(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
