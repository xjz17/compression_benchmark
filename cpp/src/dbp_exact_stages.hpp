#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace exact_actions::dbp {

inline constexpr int maximum_scaling_point = 18;

// One bit per max_point in [0,18].  This check never allocates and does not
// build a DBP packet.  input_value_visits counts value loads (at most two per
// value), rather than the constant-size scale checks performed per value.
struct ScalingFeasibility {
  std::uint32_t legal_max_points = 0;
  std::uint64_t input_value_visits = 0;

  bool legal(int max_point) const noexcept {
    return max_point >= 0 && max_point <= maximum_scaling_point &&
        (legal_max_points & (std::uint32_t{1} << max_point)) != 0;
  }
};

// Exact predicates shared with scaling_forward().  The byte overload reads
// little-endian binary64 lanes without aliasing or alignment assumptions.
bool can_scale_bitwise_exact(std::span<const double> values,
                             int max_point) noexcept;
ScalingFeasibility scaling_feasibility(
    std::span<const double> values) noexcept;
ScalingFeasibility scaling_feasibility_le64(
    std::span<const std::uint8_t> bytes) noexcept;

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int max_point);
std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet);
std::pair<int, std::vector<std::int64_t>> scaling_values(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> make_scaling_packet(
    int scale, std::span<const std::int64_t> values);

std::vector<std::uint8_t> delta_forward_int64(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> delta_forward_scaled(
    std::span<const std::uint8_t> packet);
std::vector<std::int64_t> delta_inverse_int64(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> delta_inverse_scaled(
    std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> zigzag_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> zigzag_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> varint_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> varint_inverse(std::span<const std::uint8_t> wire);

std::vector<std::uint8_t> compose_int64(std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose_int64(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int max_point);
std::vector<double> decompose_double(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::dbp
