#include "protocol.hpp"

#include <iostream>
#include <string>

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      std::cout << operator_harness_v1::handle_line(line).dump() << '\n';
      std::cout.flush();
    } catch (const std::exception& error) {
      // The protocol handler is expected to catch request failures. Keep this
      // last-resort boundary per line so one exceptional request cannot stop
      // later independent result submissions.
      std::cout << nlohmann::json{{"schema", "operator-harness-response-v1"},
                                  {"status", "failed"},
                                  {"command", ""},
                                  {"case_id", ""},
                                  {"error", error.what()}}
                       .dump()
                << '\n';
      std::cout.flush();
    }
  }
  return 0;
}
