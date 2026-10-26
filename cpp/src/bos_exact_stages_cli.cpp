#include "bos_exact_stages.hpp"

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
template <class T>
void write(const std::vector<T>& values) {
  std::cout.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("missing BOS stage command");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::bos;
  if (command == "delta-forward") write(delta_forward(words(raw)));
  else if (command == "delta-inverse") write(delta_inverse(raw));
  else if (command == "for1-forward") write(first_for_forward(raw));
  else if (command == "for1-inverse") write(first_for_inverse(raw));
  else if (command == "outlier-forward") write(outlier_forward(raw));
  else if (command == "outlier-inverse") write(outlier_inverse(raw));
  else if (command == "for2-forward") write(second_for_forward(raw));
  else if (command == "for2-inverse") write(second_for_inverse(raw));
  else if (command == "bitpack-forward") write(bitpack_forward(raw));
  else if (command == "bitpack-inverse") write(bitpack_inverse(raw));
  else if (command == "compose") write(compose(words(raw)));
  else if (command == "decompose") write(decompose(raw));
  else if (command == "adapter-plain-delta-forward")
    write(adapter_plain_delta_forward(words(raw)));
  else if (command == "adapter-plain-delta-forward-scaled")
    write(adapter_plain_delta_forward_scaled(raw));
  else if (command == "adapter-plain-delta-inverse") write(adapter_plain_delta_inverse(raw));
  else if (command == "adapter-plain-delta-inverse-scaled")
    write(adapter_plain_delta_inverse_scaled(raw));
  else if (command == "adapter-zigzag-forward") write(adapter_zigzag_forward(raw));
  else if (command == "adapter-zigzag-forward-scaled") write(adapter_zigzag_forward_scaled(raw));
  else if (command == "adapter-zigzag-inverse") write(adapter_zigzag_inverse(raw));
  else if (command == "adapter-zigzag-inverse-scaled") write(adapter_zigzag_inverse_scaled(raw));
  else if (command == "adapter-delta-forward") write(adapter_delta_forward(words(raw)));
  else if (command == "adapter-delta-forward-scaled") write(adapter_delta_forward_scaled(raw));
  else if (command == "adapter-delta-inverse") write(adapter_delta_inverse(raw));
  else if (command == "adapter-delta-inverse-scaled") write(adapter_delta_inverse_scaled(raw));
  else if (command == "adapter-bitpack-forward") write(adapter_bitpack_forward(raw));
  else if (command == "adapter-bitpack-inverse") write(adapter_bitpack_inverse(raw));
  else if (command == "adapter-wire-forward-int") write(adapter_wire_forward_int(raw));
  else if (command == "adapter-wire-forward-scaled") write(adapter_wire_forward_scaled(raw));
  else if (command == "adapter-wire-inverse-int") write(adapter_wire_inverse_int(raw));
  else if (command == "adapter-wire-inverse-scaled") write(adapter_wire_inverse_scaled(raw));
  else if (command == "adapter-compose") write(adapter_compose(words(raw)));
  else if (command == "adapter-decompose") write(adapter_decompose(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
