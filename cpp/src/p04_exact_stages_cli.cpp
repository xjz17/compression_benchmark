#include "p04_exact_stages.hpp"

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

std::vector<std::int64_t> integers(const std::vector<std::uint8_t>& raw) {
  if (raw.size() % 8 != 0) throw std::runtime_error("int64 input byte count mismatch");
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
  if (argc != 2) throw std::invalid_argument("missing p04 stage command");
  const std::string command = argv[1];
  const auto raw = read_all();
  using namespace exact_actions::p04;
  if (command == "rlbe-delta-forward") write(rlbe_delta_forward(integers(raw)));
  else if (command == "rlbe-delta-inverse") write(rlbe_delta_inverse(raw));
  else if (command == "rlbe-run-forward") write(rlbe_run_length_forward(raw));
  else if (command == "rlbe-run-inverse") write(rlbe_run_length_inverse(raw));
  else if (command == "rlbe-compose") write(rlbe_compose(integers(raw)));
  else if (command == "rlbe-decompose") write(rlbe_decompose(raw));
  else if (command == "sprintz-predictor-forward") write(sprintz_predictor_forward(integers(raw)));
  else if (command == "sprintz-predictor-forward-double") write(sprintz_predictor_forward_double_bits(integers(raw)));
  else if (command == "sprintz-predictor-inverse") write(sprintz_predictor_inverse(raw));
  else if (command == "sprintz-zigzag-forward") write(sprintz_zigzag_forward(raw));
  else if (command == "sprintz-zigzag-inverse") write(sprintz_zigzag_inverse(raw));
  else if (command == "sprintz-bitpack-forward") write(sprintz_bitpack_forward(raw));
  else if (command == "sprintz-bitpack-forward-current") write(sprintz_bitpack_forward_current(raw));
  else if (command == "sprintz-bitpack-inverse-current") write(sprintz_bitpack_inverse_current(raw));
  else if (command == "sprintz-bitpack-inverse-double-current") write(sprintz_bitpack_inverse_double_current(raw));
  else if (command == "sprintz-bitpack-inverse") write(sprintz_bitpack_inverse(raw));
  else if (command == "sprintz-bitpack-inverse-double") write(sprintz_bitpack_inverse_double(raw));
  else if (command == "sprintz-compose") write(sprintz_compose(integers(raw)));
  else if (command == "sprintz-compose-double") write(sprintz_compose_double_bits(integers(raw)));
  else if (command == "sprintz-decompose") write(sprintz_decompose(raw));
  else if (command == "sprintz-decompose-double") write(sprintz_decompose_double_bits(raw));
  else throw std::invalid_argument("unknown command: " + command);
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
