#include "exception_separation.hpp"

#include <charconv>
#include <iostream>
#include <iterator>
#include <string_view>

int main(int argc, char** argv) try {
  namespace es = exact_actions::exception_separation;
  es::require(argc >= 2, "expected forward b|inverse|validate|pack|unpack|compose b|decompose");
  const std::string_view command = argv[1];
  const bool takes_width = command == "forward" || command == "compose";
  es::require(argc == (takes_width ? 3 : 2), "unexpected exception separation arguments");
  unsigned width = 0;
  if (takes_width) {
    const std::string_view argument = argv[2];
    const auto result = std::from_chars(argument.data(), argument.data() + argument.size(), width);
    es::require(result.ec == std::errc{} && result.ptr == argument.data() + argument.size(),
                "width must be an integer in [0,32]");
    es::low_mask(width);
  }
  const es::Bytes input{std::istreambuf_iterator<char>(std::cin), {}};
  es::Bytes output;
  if (command == "forward") output = es::forward(input, width);
  else if (command == "inverse") output = es::inverse(input);
  else if (command == "validate") output = es::serialize(es::parse(input));
  else if (command == "pack") output = es::pack(es::parse(input));
  else if (command == "unpack") output = es::serialize(es::unpack(input));
  else if (command == "compose") output = es::pack(es::separate(es::words(input), width));
  else if (command == "decompose") output = es::raw_bytes(es::restore(es::unpack(input)));
  else throw std::invalid_argument("unknown exception separation command");
  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  es::require(bool(std::cout), "failed to write exception separation output");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
