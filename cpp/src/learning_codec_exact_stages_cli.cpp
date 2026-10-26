#include "learning_codec_exact_stages.hpp"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Fn = std::vector<std::uint8_t> (*)(std::span<const std::uint8_t>);

Fn resolve(const std::string& command) {
  using namespace exact_actions::learning;
  if (command == "leco-regression-forward") return leco_regression_forward;
  if (command == "leco-regression-inverse") return leco_regression_inverse;
  if (command == "leco-zigzag-forward") return leco_zigzag_forward;
  if (command == "leco-zigzag-inverse") return leco_zigzag_inverse;
  if (command == "leco-autoregression-forward") return leco_autoregression_forward;
  if (command == "alp-original-encode") return alp_original_encode;
  if (command == "reger-original-encode") return reger_original_encode;
  if (command == "leco-autoregression-inverse") return leco_autoregression_inverse;
  if (command == "leco-bitpack-forward") return leco_bitpack_forward;
  if (command == "leco-bitpack-inverse") return leco_bitpack_inverse;
  if (command == "alp-scaling-forward") return alp_scaling_forward;
  if (command == "alp-scaling-inverse") return alp_scaling_inverse;
  if (command == "alp-for-forward") return alp_for_forward;
  if (command == "alp-for-inverse") return alp_for_inverse;
  if (command == "alp-bitpack-forward") return alp_bitpack_forward;
  if (command == "alp-bitpack-inverse") return alp_bitpack_inverse;
  if (command == "alp-dictionary-forward") return alp_dictionary_forward;
  if (command == "alp-dictionary-inverse") return alp_dictionary_inverse;
  if (command == "alp-residual-core-forward") return alp_residual_core_forward;
  if (command == "alp-residual-core-inverse") return alp_residual_core_inverse;
  if (command == "reger-reorder-forward") return reger_reorder_forward;
  if (command == "reger-reorder-inverse") return reger_reorder_inverse;
  if (command == "reger-delta-forward") return reger_delta_forward;
  if (command == "reger-delta-inverse") return reger_delta_inverse;
  if (command == "reger-zigzag-forward") return reger_zigzag_forward;
  if (command == "reger-zigzag-inverse") return reger_zigzag_inverse;
  if (command == "reger-autoregression-forward") return reger_autoregression_forward;
  if (command == "reger-autoregression-inverse") return reger_autoregression_inverse;
  if (command == "reger-for-forward") return reger_for_forward;
  if (command == "reger-for-inverse") return reger_for_inverse;
  if (command == "reger-bitpack-forward") return reger_bitpack_forward;
  if (command == "reger-bitpack-inverse") return reger_bitpack_inverse;
  if (command == "reger-residual-core-forward") return reger_residual_core_forward;
  if (command == "reger-residual-core-inverse") return reger_residual_core_inverse;
  if (command == "buff-scaling-forward") return buff_scaling_forward;
  if (command == "buff-scaling-inverse") return buff_scaling_inverse;
  if (command == "buff-for-forward") return buff_for_forward;
  if (command == "buff-for-inverse") return buff_for_inverse;
  if (command == "buff-byte-plane-inverse") return buff_byte_plane_inverse;
  if (command == "buff-column-split-forward") return buff_column_split_forward;
  if (command == "buff-column-split-inverse") return buff_column_split_inverse;
  if (command == "buff-bitpack-forward") return buff_bitpack_forward;
  if (command == "buff-bitpack-inverse") return buff_bitpack_inverse;
  throw std::invalid_argument("unknown learning-codec stage command: " + command);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::invalid_argument("expected one stage command");
    const std::vector<std::uint8_t> input(
        (std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<std::uint8_t> output;
    const std::string command = argv[1];
    if (command == "buff-byte-plane-forward") {
      if (argc != 3) throw std::invalid_argument("buff-byte-plane-forward requires width0..64");
      const std::string text = argv[2];
      if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("BUFF byte-plane width must be a decimal integer0..64");
      const auto width = std::stoul(text);
      if (width > 64) throw std::invalid_argument("BUFF byte-plane width must be0..64");
      output = exact_actions::learning::buff_byte_plane_forward(input, static_cast<unsigned>(width));
    } else if (command == "buff-byte-plane-width") {
      if (argc != 2) throw std::invalid_argument("expected one stage command");
      output.push_back(static_cast<std::uint8_t>(exact_actions::learning::buff_byte_plane_width(input)));
    } else {
      if (argc != 2) throw std::invalid_argument("expected one stage command");
      output = resolve(command)(input);
    }
    std::cout.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output.size()));
    return std::cout ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << '\n';
    return 1;
  }
}
