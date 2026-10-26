#include "max_value_bitpack.hpp"

#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
  try {
    if (argc != 2 || (std::string_view(argv[1]) != "encode" &&
                      std::string_view(argv[1]) != "decode"))
      throw std::invalid_argument("usage: max_value_bitpack_cli encode|decode");
    const std::vector<std::uint8_t> input(
        (std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    const auto output = std::string_view(argv[1]) == "encode"
        ? exact_actions::max_value_bitpack::encode(input)
        : exact_actions::max_value_bitpack::decode(input);
    if (output.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
      throw std::overflow_error("bitpack stdout byte count overflow");
    std::cout.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output.size()));
    return std::cout ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
