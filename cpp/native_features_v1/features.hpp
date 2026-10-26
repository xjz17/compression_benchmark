#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wc::features_v1 {
inline constexpr const char* task_schema = "weight-conditioned-graph-task-features-v1";
inline constexpr const char* state_schema = "weight-conditioned-graph-state-features-v2";
inline constexpr std::size_t task_width = 18, state_width = 28, observation_width = 49;

// Views borrow immutable, original-width bytes. Feature conversion never changes
// the data later supplied to the compression graph.
struct FlowView {
  std::string kind;
  std::uint32_t width;
  std::span<const std::uint8_t> bytes;
  std::string logical_view;
};
struct ActiveFlow {
  FlowView flow;
  std::uint32_t depth = 0;
  std::int64_t parent_action = -1;
};
struct State {
  std::vector<ActiveFlow> flows;  // Frozen graph insertion order, not map sort order.
  std::uint64_t original_numeric_bytes = 0;
  std::uint32_t node_count = 0, policy_edits = 0, replayed_nodes = 0, edit_count = 0;
  std::uint32_t max_nodes = 64, max_depth = 10, max_active = 32;
  std::uint32_t max_policy_edits = 64, max_recipe_edits = 2;
  std::uint32_t candidate_used = 0, candidate_limit = 64, action_count = 0;
  bool continuation_edit_mode = false;
  std::string pending_edit_kind;  // empty, insert, delete, replace, parameter_change
  std::optional<std::uint32_t> pending_edit_index;
  std::optional<double> current_objective;
};

std::array<double, task_width> task(std::span<const FlowView> flows);
std::array<double, state_width> state(const State& input);
std::array<double, observation_width> observation(
    std::span<const FlowView> roots, const State& input,
    const std::array<double, 3>& weights);
}  // namespace wc::features_v1
