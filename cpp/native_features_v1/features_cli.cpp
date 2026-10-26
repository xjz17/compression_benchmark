// Text driver for parity tests. File/stdin I/O is not a performance measurement.
#include "features.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace wc::features_v1;
int main(int argc, char** argv) {
  try {
    if (argc != 2 || (std::string(argv[1]) != "task" && std::string(argv[1]) != "state"))
      throw std::invalid_argument("expected task or state");
    std::size_t count;
    if (!(std::cin >> count) || !count || count > 32) throw std::invalid_argument("flow count");
    std::vector<std::vector<std::uint8_t>> data(count);
    std::vector<FlowView> flows;
    for (std::size_t i = 0; i < count; ++i) {
      std::string kind, view, hex; unsigned width;
      if (!(std::cin >> kind >> width >> view >> hex)) throw std::invalid_argument("flow fields");
      if (hex == "-") hex.clear();
      if (hex.size() % 2 || hex.size() > (1u << 28)) throw std::invalid_argument("hex length");
      auto nibble = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        throw std::invalid_argument("invalid hex");
      };
      for (std::size_t j = 0; j < hex.size(); j += 2)
        data[i].push_back(std::uint8_t(nibble(hex[j]) * 16 + nibble(hex[j + 1])));
      flows.push_back({kind, width, data[i], view});
    }
    std::cout << std::setprecision(17);
    if (std::string(argv[1]) == "task") {
      for (double value : task(flows)) std::cout << value << '\n';
    } else {
      State input;
      unsigned mode; double objective; long long pending_index; std::string pending_kind;
      if (!(std::cin >> input.original_numeric_bytes >> input.action_count >> input.node_count
          >> input.policy_edits >> input.replayed_nodes >> input.edit_count >> mode
          >> input.candidate_used >> input.candidate_limit >> pending_kind >> pending_index >> objective))
        throw std::invalid_argument("state fields");
      if (mode > 1 || pending_index < -1 || pending_index > 64 || objective < -1)
        throw std::invalid_argument("state field range");
      input.continuation_edit_mode = mode;
      input.pending_edit_kind = pending_kind == "-" ? "" : pending_kind;
      if (pending_index >= 0) input.pending_edit_index = unsigned(pending_index);
      if (objective != -1) input.current_objective = objective;
      for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t depth; std::int64_t parent;
        if (!(std::cin >> depth >> parent)) throw std::invalid_argument("flow state fields");
        input.flows.push_back({flows[i], depth, parent});
      }
      for (double value : state(input)) std::cout << value << '\n';
    }
    std::string extra;
    if (std::cin >> extra) throw std::invalid_argument("trailing request data");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 2;
  }
}
