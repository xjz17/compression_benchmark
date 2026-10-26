#include "svb_exact_stages.hpp"

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

std::vector<std::int64_t> read_int64(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % sizeof(std::int64_t) != 0) {
    throw std::runtime_error("int64 input byte count mismatch");
  }
  std::vector<std::int64_t> values(raw.size() / sizeof(std::int64_t));
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
  if (argc != 2) throw std::invalid_argument("expected one command");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::svb;
  if (command == "delta-forward") write(delta_forward(read_int64(raw)));
  else if (command == "delta-forward-scaled") write(delta_forward_scaled(raw));
  else if (command == "delta-inverse") write(delta_inverse(raw));
  else if (command == "delta-inverse-scaled") write(delta_inverse_scaled(raw));
  else if (command == "integer-code-forward") write(integer_code_forward(raw));
  else if (command == "integer-code-inverse") write(integer_code_inverse(raw));
  else if (command == "compose") write(compose(read_int64(raw)));
  else if (command == "compose-scaled") write(compose_scaled(raw));
  else if (command == "decompose") write(decompose(raw));
  else if (command == "decompose-scaled") write(decompose_scaled(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
