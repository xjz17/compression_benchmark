#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::learning {

std::vector<std::uint8_t> alp_original_encode(
    std::span<const std::uint8_t> float64_le);
std::vector<std::uint8_t> reger_original_encode(
    std::span<const std::uint8_t> int64_le);

// LECO source-level chain: int64 -> signed linear-model residuals -> ZigZag
// codes -> packed residual wire.  The autoregression names are compatibility
// wrappers for the first two explicit stages.
std::vector<std::uint8_t> leco_regression_forward(
    std::span<const std::uint8_t> int64_le);
std::vector<std::uint8_t> leco_regression_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> leco_zigzag_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> leco_zigzag_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> leco_autoregression_forward(
    std::span<const std::uint8_t> int64_le);
std::vector<std::uint8_t> leco_autoregression_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> leco_bitpack_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> leco_bitpack_inverse(
    std::span<const std::uint8_t> wire);

// ALP main path and ALP-RD dictionary alternative.  Dictionary is deliberately
// a sibling action, not inserted into the scaling -> FOR -> bitpack chain.
std::vector<std::uint8_t> alp_scaling_forward(
    std::span<const std::uint8_t> float64_le);
std::vector<std::uint8_t> alp_scaling_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> alp_for_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> alp_for_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> alp_bitpack_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> alp_bitpack_inverse(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> alp_dictionary_forward(
    std::span<const std::uint8_t> float64_le);
std::vector<std::uint8_t> alp_dictionary_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> alp_residual_core_forward(
    std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> alp_residual_core_inverse(
    std::span<const std::uint8_t> wire);

// REGER reference chain for one int64 series.  The production multi-column
// codec keeps its separate source-level row-order search; this action makes
// every transformation explicit and independently reversible.  The old
// autoregression names remain compatibility wrappers for delta -> ZigZag.
std::vector<std::uint8_t> reger_reorder_forward(
    std::span<const std::uint8_t> int64_le);
std::vector<std::uint8_t> reger_reorder_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_delta_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_delta_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_zigzag_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_zigzag_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_autoregression_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_autoregression_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_for_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_for_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_bitpack_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> reger_bitpack_inverse(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> reger_residual_core_forward(
    std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> reger_residual_core_inverse(
    std::span<const std::uint8_t> wire);

// BUFF scalar path: exact decimal scaling/raw-bit fallback -> minimum FOR ->
// byte-plane split with an explicit width -> fixed-width wire adapter.
// The plane kernels transform only unsigned offsets; base, scale and type are
// passed through by the packet adapters and restored only by the FOR inverse.
std::vector<std::uint8_t> buff_scaling_forward(
    std::span<const std::uint8_t> float64_le);
std::vector<std::uint8_t> buff_scaling_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> buff_for_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> buff_for_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> buff_byte_planes_split(
    std::span<const std::uint64_t> offsets, unsigned bit_width);
std::vector<std::uint64_t> buff_byte_planes_join(
    std::span<const std::uint8_t> planes, std::size_t count, unsigned bit_width);
// Invocation configuration helper, deliberately separate from the split kernel.
unsigned buff_byte_plane_max_width(std::span<const std::uint64_t> offsets);
unsigned buff_byte_plane_width(std::span<const std::uint8_t> for_packet);
std::vector<std::uint8_t> buff_byte_plane_forward(
    std::span<const std::uint8_t> for_packet, unsigned bit_width);
std::vector<std::uint8_t> buff_byte_plane_inverse(
    std::span<const std::uint8_t> packet);
// Historical fused wrappers retain the old BSC1 -> BSF1 interface.
std::vector<std::uint8_t> buff_column_split_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> buff_column_split_inverse(
    std::span<const std::uint8_t> packet);
// Framing adapter for planes already produced by FOR + byte-plane split.
std::vector<std::uint8_t> buff_plane_frame_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> buff_plane_frame_inverse(
    std::span<const std::uint8_t> wire);
// Historical names retain the original wire format.
std::vector<std::uint8_t> buff_bitpack_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> buff_bitpack_inverse(
    std::span<const std::uint8_t> wire);

}  // namespace exact_actions::learning
