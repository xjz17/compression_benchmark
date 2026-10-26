#include "fastpfor_exact_stages.hpp"

#include <charconv>
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
  if (raw.size() % 8) throw std::runtime_error("int64 input byte count mismatch");
  std::vector<std::int64_t> values(raw.size() / 8);
  if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}
std::size_t count_arg(const char* text) {
  std::size_t value = 0;
  const auto* end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, value);
  if (result.ec != std::errc{} || result.ptr != end)
    throw std::invalid_argument("invalid FastPFor expected count");
  return value;
}
template <class T>
void write(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) throw std::invalid_argument("missing FastPFor stage command");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::fastpfor;
  if (command == "pfor-forward-int" && argc == 2) {
    write(pfor_stage_forward_int(words(raw)));
  } else if (command == "pfor-forward-scaled" && argc == 2) {
    write(pfor_stage_forward_scaled(raw));
  } else if (command == "pfor-inverse-int" && argc == 2) {
    write(pfor_stage_inverse_int(raw));
  } else if (command == "pfor-inverse-scaled" && argc == 2) {
    write(pfor_stage_inverse_scaled(raw));
  } else if (command == "variable-byte-forward" && argc == 2) {
    write(variable_byte_forward(raw));
  } else if (command == "variable-byte-inverse" && argc == 3) {
    write(variable_byte_inverse(raw, count_arg(argv[2])));
  } else if (command == "adapter-forward-int" && argc == 2) {
    write(adapter_forward_int(words(raw)));
  } else if (command == "adapter-inverse-int" && argc == 3) {
    write(adapter_inverse_int(raw, count_arg(argv[2])));
  } else if (command == "adapter-forward-scaled" && argc == 2) {
    write(adapter_forward_scaled(raw));
  } else if (command == "adapter-inverse-scaled" && argc == 3) {
    write(adapter_inverse_scaled(raw, count_arg(argv[2])));
  } else {
    throw std::invalid_argument("invalid FastPFor stage command arguments");
  }
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
