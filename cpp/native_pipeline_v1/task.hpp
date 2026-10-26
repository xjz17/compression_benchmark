#pragma once

#include "catalog.hpp"
#include "features.hpp"
#include "graph.hpp"
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace wc::pipeline_v1 {

struct TaskConfig {
  std::uint32_t max_ordinary_depth = 10;
  std::uint32_t max_graph_nodes = 64;
  std::uint32_t max_active_streams = 32;
  std::uint32_t max_policy_edits = 64;
  std::uint32_t timing_warmups = 1;
  std::uint32_t timing_repetitions = 5;
  bool allow_large_memory = false;
  native_graph_v1::Limits graph_limits;
};

struct TaskRequest {
  std::string task_id;
  std::vector<native_graph_v1::RootInput> roots;
  std::array<double, 3> weights{1.0, 0.0, 0.0};
  double compression_reference_ns_per_byte = 1.0;
  double decompression_reference_ns_per_byte = 1.0;
  std::vector<std::uint32_t> previous_recipe;
  // First-block seed selected from a frozen training-only solution library.
  // The external representation may end in the single STOP action; TaskState
  // stores only replayable graph actions.
  std::vector<std::uint32_t> starting_recipe;
  std::uint32_t block_index = 0;
  bool continuation_edit_mode = false;
  std::uint32_t max_recipe_edits = 2;
  std::optional<std::uint64_t> original_numeric_bytes;
  TaskConfig config;
  bool require_complete_native_catalog = false;
};

struct RecipeEdit {
  std::string kind;
  std::uint32_t position = 0;
  std::optional<std::uint32_t> action_id;
  bool operator==(const RecipeEdit&) const = default;
};

struct TaskState {
  std::vector<std::uint32_t> recipe;
  bool recipe_valid = true;
  std::optional<std::string> recipe_error;
  std::uint32_t policy_edits = 0;
  std::uint32_t replayed_nodes = 0;
  bool continuation_edit_mode = false;
  std::optional<std::string> pending_edit_kind;
  std::optional<std::uint32_t> pending_edit_index;
  std::uint32_t edit_count = 0;
  std::vector<RecipeEdit> edit_history;
};

struct StateInspection {
  std::array<double, features_v1::task_width> task_features{};
  std::array<double, features_v1::state_width> state_features{};
  std::vector<std::uint32_t> legal_action_ids;
  std::uint32_t node_count = 0, active_data_flows = 0, maximum_depth = 0;
  bool terminal_evaluable = false;
};

struct ReplayCacheStats {
  std::uint64_t hits = 0, misses = 0, evictions = 0;
  std::uint64_t retained_bytes = 0, byte_limit = 0;
  std::uint32_t entries = 0;
};

struct Evaluation {
  bool success = true;
  std::string failure_error;
  native_graph_v1::Bytes frame;
  native_graph_v1::Recipe native_recipe;
  std::vector<std::uint64_t> compression_samples_ns, decompression_samples_ns;
  std::uint64_t original_numeric_bytes = 0;
  std::uint64_t max_frame_bytes = 0;
  std::uint64_t compression_time_ns = 0, decompression_time_ns = 0;
  double compression_ratio = 0.0;
  double compression_ns_per_byte = 0.0, decompression_ns_per_byte = 0.0;
  double objective = 0.0;
  std::uint32_t candidate_ordinal = 0, candidate_budget = 0;
};

class GraphTask {
 public:
  GraphTask(const ActionCatalog&, const native_graph_v1::StageDispatcher&, TaskRequest);
  const TaskRequest& request() const noexcept { return request_; }
  // Reconstruct the external request accepted by the constructor. Internally
  // the constructor removes the one terminal STOP from starting_recipe.
  TaskRequest fresh_request() const;
  const ActionSpec& action_record(std::uint32_t id) const { return catalog_.at(id); }
  const TaskState& initial_state() const noexcept { return initial_; }
  TaskState starting_state(std::vector<std::uint32_t>) const;
  std::uint64_t original_numeric_bytes() const noexcept { return original_numeric_bytes_; }
  std::uint32_t candidate_budget() const noexcept { return candidate_budget_; }
  std::uint32_t candidate_used() const noexcept { return candidate_used_; }
  std::uint32_t max_steps() const noexcept;
  bool is_terminal_evaluable(const TaskState&) const;
  // Certified lower bound for every complete NGF1 frame reachable from this
  // state. It uses only mandatory wire bytes and policy nodes that cannot all
  // be deleted within the remaining continuation edit budget. It never uses
  // intermediate payload size or executes a complete candidate.
  std::uint64_t certified_frame_byte_lower_bound(const TaskState&) const;
  StateInspection inspect(const TaskState&) const;
  std::vector<std::uint32_t> legal_actions(const TaskState&) const;
  std::optional<TaskState> transition(const TaskState&, std::uint32_t action) const;
  Evaluation evaluate_terminal(const TaskState&);
  std::uint32_t consume_failed_candidate();
  native_graph_v1::Recipe compile_recipe(const std::vector<std::uint32_t>&) const;
  static std::vector<std::uint32_t> edit_recipe(
      const ActionCatalog&, std::vector<std::uint32_t>, const RecipeEdit&);
  std::optional<std::array<double, 3>> measured_action_cost(std::uint32_t) const;
  std::size_t action_measurement_count(std::uint32_t) const;
  ReplayCacheStats replay_cache_stats() const noexcept;

 private:
  struct Replay;
  struct ReplayCacheEntry {
    std::shared_ptr<const Replay> value;
    std::uint64_t retained_bytes = 0, last_used = 0;
  };
  std::shared_ptr<const Replay> replay(const std::vector<std::uint32_t>&) const;
  std::shared_ptr<const Replay> replay_for_features(const TaskState&) const;
  std::uint64_t replay_retained_bytes(const Replay&) const;
  TaskState recipe_state(std::vector<std::uint32_t>, const TaskState* parent = nullptr) const;
  bool graph_action_legal(const Replay&, const ActionSpec&, bool ignore_policy_limit,
                          std::uint32_t policy_edits) const;
  bool legal(const TaskState&, const ActionSpec&, bool replay_mode = false) const;
  std::vector<std::uint32_t> parameter_change_candidates(std::uint32_t) const;
  bool edit_position_has_completion(const TaskState&, const std::string&,
                                    std::uint32_t) const;
  std::optional<TaskState> edited(const TaskState&, const std::string&, std::uint32_t,
                                  std::optional<std::uint32_t>) const;
  features_v1::State feature_state(const TaskState&, const Replay&) const;
  void verify_state(const TaskState&) const;

  const ActionCatalog& catalog_;
  const native_graph_v1::StageDispatcher& dispatcher_;
  TaskRequest request_;
  TaskState initial_;
  std::uint64_t original_numeric_bytes_ = 0;
  std::uint32_t candidate_budget_ = 64, candidate_used_ = 0;
  mutable std::map<std::vector<std::uint32_t>, double> measured_objectives_;
  std::map<std::uint32_t, std::vector<std::array<double, 3>>> measured_action_costs_;
  static constexpr std::uint64_t replay_cache_byte_limit_ = 64ULL << 20;
  mutable std::map<std::vector<std::uint32_t>, ReplayCacheEntry> replay_cache_;
  mutable std::uint64_t replay_cache_bytes_ = 0, replay_cache_serial_ = 0;
  mutable std::uint64_t replay_cache_hits_ = 0, replay_cache_misses_ = 0,
                        replay_cache_evictions_ = 0;
};

}  // namespace wc::pipeline_v1
