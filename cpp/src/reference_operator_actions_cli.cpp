#include "reference_operator_actions.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
  if (argc < 2) throw std::invalid_argument("expected a reference action command");
  const std::string command = argv[1];
  reference_operator_actions::Names names;
  for (int i = 2; i < argc; ++i) names.emplace_back(argv[i]);
  reference_operator_actions::Bytes input{std::istreambuf_iterator<char>(std::cin), {}};
  reference_operator_actions::Bytes output;
  using namespace reference_operator_actions;
  if (command == "encode") output = encode(input, names);
  else if (command == "decode") output = decode(input);
  else if (command == "state") output = serialized(input, names);
  else if (command == "values") output = final_values(input, names);
  else if (command == "invert-state") output = inverse_serialized(input, names);
  else if (command == "raw-encode") output = raw_encode(input, names);
  else if (command == "compact-encode") output = compact_encode(input);
  else if (command == "compact-decode") output = compact_decode(input);
  else if (command == "inspect") output = inspect(input, names);
  else if (command == "legacy-encode") output = legacy_encode(input, names);
  else if (command == "legacy-decode") output = legacy_decode(input);
  else if (command == "inverse-step") {
    if (names.size() != 1) throw std::invalid_argument("inverse-step requires one metadata tag");
    output = inverse_step(input, names.front());
  }
  else if (command == "all-actions") {
    if (!names.empty()) throw std::invalid_argument("all-actions takes no action names");
    for (const auto& name : all_actions()) std::cout << name << '\n';
    return 0;
  }
  else if (command == "actions") {
    for (const auto& name : actions(input)) std::cout << name << '\n';
    return 0;
  } else throw std::invalid_argument("unknown reference action command");
  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  if (!std::cout) throw std::runtime_error("failed to write reference output");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
