#pragma once

#include <cstdint>
#include <span>
#include <vector>

// These stages target buff_kernel::encode_doubles as it exists in this tree:
// lossless raw IEEE-754 storage, not the older precision-bounded BUFF kernel
// and not encode_double_ablation's decimal-scaling/FOR format.
namespace exact_actions::buff_native {

std::vector<std::uint8_t> word_order_forward(std::span<const std::uint8_t> float64_le);
std::vector<std::uint8_t> word_order_inverse(std::span<const std::uint8_t> words);
std::vector<std::uint8_t> frame_forward(std::span<const std::uint8_t> words);
std::vector<std::uint8_t> frame_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose(std::span<const std::uint8_t> float64_le);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::buff_native
