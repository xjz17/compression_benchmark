#include "legacy_numeric_action_stages.hpp"
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
  using namespace legacy_numeric_action;
  try {
    if (argc < 3) throw std::invalid_argument("usage: forward ACTION [kind] [max_point|none]; inverse ACTION [count]");
    const std::string mode = argv[1]; const Bytes raw(std::istreambuf_iterator<char>(std::cin),{}); Bytes out;
    if (mode == "forward" && argc <= 5) {
      std::optional<int> point;
      if (argc == 5 && std::string(argv[4]) != "none") point = std::stoi(argv[4]);
      out = forward(argv[2],raw,argc >= 4 ? argv[3] : "bytes",point);
    } else if (mode == "inverse" && argc <= 4) out = inverse(argv[2],raw,argc == 4 ? std::stoull(argv[3]) : 0);
    else throw std::invalid_argument("invalid legacy numeric CLI arguments");
    std::cout.write(reinterpret_cast<const char*>(out.data()),out.size());
    if (!std::cout) throw std::runtime_error("cannot write legacy stage output");
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
