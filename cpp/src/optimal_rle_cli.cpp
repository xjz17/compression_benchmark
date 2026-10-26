#include "optimal_rle.hpp"

#include <charconv>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace {
std::size_t number(std::string_view text) {
  std::size_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("invalid unsigned argument");
  return value;
}
}

int main(int argc, char** argv) try {
  if (argc < 2) throw std::invalid_argument("expected encode WIDTH | decode [--max-bytes N] | inspect [--max-bytes N]");
  const std::string_view command = argv[1];
  std::size_t limit = unified_rle::kDefaultMaxDecodedBytes;
  if (command != "encode") {
    if (argc == 4 && std::string_view(argv[2]) == "--max-bytes") limit = number(argv[3]);
    else if (argc != 2) throw std::invalid_argument("invalid decode/inspect arguments");
  }
  const std::vector<std::uint8_t> input{std::istreambuf_iterator<char>(std::cin), {}};
  std::vector<std::uint8_t> output;
  if (command == "encode") {
    if (argc != 3) throw std::invalid_argument("expected encode WIDTH");
    const auto width = number(argv[2]);
    if (width < 1 || width > 8) throw std::invalid_argument("WIDTH must be 1..8");
    output = unified_rle::encode(input, static_cast<std::uint8_t>(width));
  } else if (command == "decode" || command == "inspect") {
    output = unified_rle::decode(input, limit);
    if (command == "inspect") {
      std::cout << "{\"format\":\"URL1\",\"symbol_width\":"
                << unsigned(unified_rle::symbol_width(input))
                << ",\"mode\":" << unsigned(input[5])
                << ",\"decoded_bytes\":" << output.size()
                << ",\"encoded_bytes\":" << input.size() << "}\n";
      return std::cout ? 0 : 2;
    }
  } else throw std::invalid_argument("unknown command");
  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  if (!std::cout) throw std::runtime_error("cannot write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
