#include "remaining_codec_action_catalog.hpp"

#include <iostream>
#include <string_view>

namespace {

void csv_field(std::string_view value) {
  std::cout << '"';
  for (const char ch : value) {
    if (ch == '"') std::cout << '"';
    std::cout << ch;
  }
  std::cout << '"';
}

}  // namespace

int main() {
  std::cout << "action_name,operator_id,implementation_id,implementation_source,"
               "method,selected_level,evidence,standalone_composable\n";
  for (const auto& action : remaining_codec_actions::concrete_actions()) {
    csv_field(action.name);
    std::cout << ',';
    csv_field(action.operator_id);
    std::cout << ',';
    csv_field(action.implementation_id);
    std::cout << ',';
    csv_field(action.implementation_source);
    std::cout << ',';
    csv_field(action.method);
    std::cout << ',';
    csv_field(action.selected_level);
    std::cout << ',';
    csv_field(remaining_codec_actions::evidence_name(action.evidence));
    std::cout << ',' << (action.standalone_composable ? '1' : '0') << '\n';
  }
  return 0;
}
