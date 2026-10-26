#include "learning_codec_exact_stages.hpp"
#include "ts2diff_exact_stages.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
struct Golden {
  std::string_view codec;
  std::string_view packet;
  std::string_view wire;
};
constexpr Golden kGolden[] = {
#include "max_value_bitpack_golden.inc"
};

std::vector<std::uint8_t> unhex(std::string_view hex) {
  std::vector<std::uint8_t> out;
  const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  for (std::size_t i = 0; i < hex.size(); i += 2)
    out.push_back(static_cast<std::uint8_t>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
  return out;
}
}  // namespace

int main() {
  using namespace exact_actions;
  for (const auto& golden : kGolden) {
    const auto packet = unhex(golden.packet);
    const auto wire = unhex(golden.wire);
    std::vector<std::uint8_t> encoded, decoded;
    if (golden.codec == "ts2diff" || golden.codec == "ts2diff_float") {
      encoded = ts2diff::bitpack_forward(packet);
      decoded = ts2diff::bitpack_inverse(wire, golden.codec == "ts2diff_float",
                                        golden.codec == "ts2diff_float" ? 2 : 0);
    } else if (golden.codec == "leco") {
      encoded = learning::leco_bitpack_forward(packet);
      decoded = learning::leco_bitpack_inverse(wire);
    } else if (golden.codec == "alp") {
      encoded = learning::alp_bitpack_forward(packet);
      decoded = learning::alp_bitpack_inverse(wire);
    } else if (golden.codec == "reger") {
      encoded = learning::reger_bitpack_forward(packet);
      decoded = learning::reger_bitpack_inverse(wire);
    } else {
      throw std::runtime_error("unknown fixture codec");
    }
    if (encoded != wire || decoded != packet)
      throw std::runtime_error("pre-refactor wire parity failed: " + std::string(golden.codec));
  }
  std::cout << "max_value_bitpack_adapters=PASS old_wire_fixtures=" << std::size(kGolden)
            << " codecs=TS_2DIFF/LECO/ALP/REGER metadata=unchanged\n";
}
