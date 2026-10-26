#include "rle_exact_stages.hpp"

#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_all() {
  return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
}
template <class T> std::vector<T> typed(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % sizeof(T)) throw std::runtime_error("typed input byte count mismatch");
  std::vector<T> values(raw.size() / sizeof(T));
  if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}
template <class T> void write(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2 || argc > 3)
    throw std::invalid_argument("usage: rle_exact_stages_cli COMMAND [max_point]");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::rle;
  if (command == "scale-forward") {
    if (argc != 3) throw std::invalid_argument("scale-forward requires max_point");
    write(scaling_forward(typed<double>(raw), std::stoi(argv[2])));
  } else if (command == "scale-inverse") write(scaling_inverse(raw));
  else if (command == "run-forward") write(run_length_forward(raw));
  else if (command == "run-inverse") write(run_length_inverse(raw));
  else if (command == "compose") {
    if (argc != 3) throw std::invalid_argument("compose requires max_point");
    write(compose_double(typed<double>(raw), std::stoi(argv[2])));
  } else if (command == "decompose") write(decompose_double(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
