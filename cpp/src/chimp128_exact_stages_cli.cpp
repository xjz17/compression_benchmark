#include "chimp128_exact_stages.hpp"

#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_all() { return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()}; }
std::vector<std::int64_t> words(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % 8) throw std::runtime_error("64-bit input byte count mismatch");
  std::vector<std::int64_t> values(raw.size() / 8);
  if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}
template <class T> void write(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("missing Chimp128 stage command");
  const std::string command = argv[1]; const auto raw = read_all();
  using namespace exact_actions::chimp128;
  if (command == "xor-forward-int") write(xor_forward_int64(words(raw)));
  else if (command == "xor-forward-float") write(xor_forward_double_bits(words(raw)));
  else if (command == "xor-inverse") write(xor_inverse(raw));
  else if (command == "pack-forward") write(trailing_pack_forward(raw));
  else if (command == "pack-inverse-int") write(trailing_pack_inverse_int64(raw));
  else if (command == "pack-inverse-float") write(trailing_pack_inverse_double(raw));
  else if (command == "compose-int") write(compose_int64(words(raw)));
  else if (command == "compose-float") write(compose_double_bits(words(raw)));
  else if (command == "decompose-int") write(decompose_int64(raw));
  else if (command == "decompose-float") write(decompose_double_bits(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout"); return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
