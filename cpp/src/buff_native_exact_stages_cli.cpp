#include "buff_native_exact_stages.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"

#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("expected one BUFF native stage command");
  const std::string command = argv[1];
  const std::vector<std::uint8_t> input(
      (std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  using namespace exact_actions::buff_native;
  std::vector<std::uint8_t> output;
  if (command == "word-order-forward") output = word_order_forward(input);
  else if (command == "word-order-inverse") output = word_order_inverse(input);
  else if (command == "frame-forward") output = frame_forward(input);
  else if (command == "frame-inverse") output = frame_inverse(input);
  else if (command == "compose") output = compose(input);
  else if (command == "decompose") output = decompose(input);
  else if (command == "original-decode") {
    // Validate this native format before invoking the original decoder, which
    // also understands older component formats that do not reconstruct values.
    frame_inverse(input);
    const auto decoded = buff_kernel::decode_doubles(input);
    output.resize(decoded.values.size() * sizeof(double));
    if (!output.empty()) std::memcpy(output.data(), decoded.values.data(), output.size());
  }
  else if (command == "original-encode") {
    if (input.size() % 8 != 0) throw std::invalid_argument("expected float64 input");
    std::vector<double> values(input.size() / 8);
    if (!input.empty()) std::memcpy(values.data(), input.data(), input.size());
    output = buff_kernel::encode_doubles(values, 10000);
  } else throw std::invalid_argument("unknown BUFF native stage command");
  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  return std::cout ? 0 : 2;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
