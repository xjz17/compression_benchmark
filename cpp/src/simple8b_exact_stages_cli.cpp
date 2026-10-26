#include "simple8b_exact_stages.hpp"

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

template <class T>
std::vector<T> typed(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % sizeof(T) != 0) throw std::runtime_error("typed input byte count mismatch");
  std::vector<T> values(raw.size() / sizeof(T));
  if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}

template <class T>
void write(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}

int scale_arg(int argc, char** argv) {
  if (argc != 3) throw std::invalid_argument("this command requires max_point");
  return std::stoi(argv[2]);
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) throw std::invalid_argument("missing command");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::simple8b;
  if (command == "scale-forward") write(scaling_forward(typed<double>(raw), scale_arg(argc, argv)));
  else if (command == "scale-inverse") write(scaling_inverse(raw));
  else if (command == "bitpack-forward-int") write(bitpack_forward_int64(typed<std::int64_t>(raw)));
  else if (command == "bitpack-forward-scaled") write(bitpack_forward_scaled(raw));
  else if (command == "bitpack-inverse-int") write(bitpack_inverse_int64(raw));
  else if (command == "bitpack-inverse-scaled") write(bitpack_inverse_scaled(raw));
  else if (command == "compose-float") write(compose_double(typed<double>(raw), scale_arg(argc, argv)));
  else if (command == "decompose-float") write(decompose_double(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
