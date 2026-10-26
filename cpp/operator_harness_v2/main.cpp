#include "protocol.hpp"

#include <iostream>
#include <string>

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::cout << operator_harness_v2::handle_line(line).dump() << '\n';
    std::cout.flush();
  }
  return 0;
}

