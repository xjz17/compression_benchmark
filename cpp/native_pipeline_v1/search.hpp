#pragma once

#include "policy.hpp"
#include "task.hpp"
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wc::pipeline_v1 {

enum class SearchMode { Policy, Beam30 };
const char* search_mode_name(SearchMode) noexcept;

struct SearchOptions {
  SearchMode mode = SearchMode::Policy;
  std::uint32_t beam_width = 30;
  bool enable_certified_pruning = true;
  bool enable_objective_guided_ordering = true;
};

double certified_objective_lower_bound(const GraphTask&, const TaskState&);
bool certified_should_prune(const GraphTask&, const TaskState&, double incumbent);

struct SearchTimings {
  std::uint64_t task_initialization_ns = 0;
  std::uint64_t feature_ns = 0, policy_ns = 0, action_ordering_ns = 0;
  std::uint64_t transition_ns = 0, certified_pruning_ns = 0, candidate_evaluation_ns = 0;
  std::uint64_t total_prediction_ns = 0, final_encode_ns = 0, total_compression_ns = 0;
};

struct CandidateSummary {
  bool success = true;
  std::string failure_error;
  std::vector<std::uint32_t> recipe, search_actions;
  double objective = 0.0, compression_ratio = 0.0;
  double compression_ns_per_byte = 0.0, decompression_ns_per_byte = 0.0;
  std::uint64_t complete_bytes = 0;
};

struct SearchResult {
  bool success = false;
  std::vector<std::string> failure_errors;
  std::vector<std::uint32_t> best_search_actions, best_recipe;
  Evaluation best_evaluation;
  std::vector<CandidateSummary> candidates;
  std::uint32_t evaluations = 0, evaluation_budget = 0;
  std::uint64_t budget_unexplored_actions = 0, duplicate_states_skipped = 0;
  std::uint64_t certified_pruned_states = 0;
  SearchTimings timing;
  native_graph_v1::Bytes final_frame;
};

class PolicySearch {
 public:
  explicit PolicySearch(const weighted_grpo::native_policy_v1::Policy* policy) : policy_(policy) {}
  SearchResult run(GraphTask&, const native_graph_v1::StageDispatcher&, SearchOptions = {},
                   const std::vector<std::vector<std::uint32_t>>& starting_recipes = {},
                   std::optional<std::chrono::steady_clock::time_point> total_begin = std::nullopt,
                   std::uint64_t task_initialization_ns = 0) const;
 private:
  const weighted_grpo::native_policy_v1::Policy* policy_ = nullptr;
};

}  // namespace wc::pipeline_v1
