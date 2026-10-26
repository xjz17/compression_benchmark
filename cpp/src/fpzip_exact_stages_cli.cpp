#include "fpzip_exact_stages.hpp"

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
  if (argc != 2) throw std::invalid_argument("missing FPZIP stage command");
  const std::string command = argv[1];
  const auto input = read_all();
  using namespace exact_actions::fpzip;
  if (command == "predictor-forward") write(predictor_forward(input));
  else if (command == "predictor-inverse") write(predictor_inverse(input));
  else if (command == "range-forward") write(range_forward(input));
  else if (command == "range-inverse") write(range_inverse(input));
  else if (command == "compose") write(compose(input));
  else if (command == "decompose") write(decompose(input));
  else throw std::invalid_argument("unknown FPZIP stage command");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
