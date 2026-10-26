#include "ts2diff_exact_stages.hpp"

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_all() {
  return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
}

template <class T>
std::span<const T> typed_input(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % sizeof(T) != 0) throw std::runtime_error("typed input byte count mismatch");
  return {reinterpret_cast<const T*>(raw.data()), raw.size() / sizeof(T)};
}

template <class T>
void write_typed(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void write_bytes(const std::vector<std::uint8_t>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size()));
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
  using namespace exact_actions::ts2diff;

  if (command == "scale-forward") write_bytes(scaling_forward(typed_input<double>(raw), scale_arg(argc, argv)));
  else if (command == "scale-inverse") write_typed(scaling_inverse(raw));
  else if (command == "delta-forward-int") write_bytes(delta_forward_int64(typed_input<std::int64_t>(raw)));
  else if (command == "delta-forward-scaled") write_bytes(delta_forward_scaled(raw));
  else if (command == "delta-inverse-int") write_typed(delta_inverse_int64(raw));
  else if (command == "delta-inverse-scaled") write_bytes(delta_inverse_scaled(raw));
  else if (command == "for-forward") write_bytes(frame_of_reference_forward(raw));
  else if (command == "for-inverse") write_bytes(frame_of_reference_inverse(raw));
  else if (command == "bitpack-forward") write_bytes(bitpack_forward(raw));
  else if (command == "bitpack-inverse-int") write_bytes(bitpack_inverse(raw, false, 0));
  else if (command == "bitpack-inverse-float") write_bytes(bitpack_inverse(raw, true, scale_arg(argc, argv)));
  else if (command == "compose-int") write_bytes(compose_int64(typed_input<std::int64_t>(raw)));
  else if (command == "compose-float") write_bytes(compose_double(typed_input<double>(raw), scale_arg(argc, argv)));
  else if (command == "decompose-int") write_typed(decompose_int64(raw));
  else if (command == "decompose-float") write_typed(decompose_double(raw, scale_arg(argc, argv)));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
