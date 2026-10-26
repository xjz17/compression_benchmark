#include "catalog.hpp"
#include "search.hpp"
#include "stage_dispatch.hpp"
#include "task.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;

namespace {
using graph::Bytes;
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
template<class T> Bytes raw(std::initializer_list<T> values) {
  Bytes out;
  for (const auto value : values) {
    const auto bits = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
    out.insert(out.end(), bits.begin(), bits.end());
  }
  return out;
}
ngpipe::TaskRequest append_request() {
  ngpipe::TaskRequest out;
  out.task_id = "bounded-pruning-exhaustive";
  out.roots = {{{graph::Kind::Numeric, 8,
      raw<std::int64_t>({0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047})},
      true, "int64"}};
  out.weights = {1.0, 0.0, 0.0};
  out.config.timing_warmups = 0;
  out.config.timing_repetitions = 1;
  out.config.max_policy_edits = 2;
  return out;
}
ngpipe::TaskRequest nonbinary_ratio_request() {
  auto out = append_request();
  out.task_id = "nonbinary-double-tie";
  out.roots = {{{graph::Kind::Numeric, 1, Bytes{1, 2, 3}}, true, "bytes"}};
  out.weights = {0.4, 0.3, 0.3};
  return out;
}
struct MeasuredState {
  ngpipe::TaskState state;
  std::optional<std::size_t> parent;
  std::uint32_t depth = 0;
  ngpipe::Evaluation evaluation;
};
bool is_descendant(const std::vector<MeasuredState>& states, std::size_t child,
                   std::size_t ancestor) {
  while (child != ancestor && states[child].parent) child = *states[child].parent;
  return child == ancestor;
}
std::size_t best_index(const std::vector<MeasuredState>& states) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < states.size(); ++i)
    if (states[i].evaluation.objective < states[best].evaluation.objective) best = i;
  return best;
}
std::size_t best_with_certified_pruning(const ngpipe::GraphTask& task,
                                        const std::vector<MeasuredState>& states,
                                        std::size_t* pruned) {
  std::vector<bool> blocked(states.size(), false);
  double incumbent = std::numeric_limits<double>::infinity();
  std::size_t best = 0;
  bool have_best = false;
  *pruned = 0;
  for (std::size_t i = 0; i < states.size(); ++i) {
    if (states[i].parent && blocked[*states[i].parent]) {
      blocked[i] = true;
      continue;
    }
    if (ngpipe::certified_should_prune(task, states[i].state, incumbent)) {
      blocked[i] = true;
      ++*pruned;
      continue;
    }
    if (!have_best || states[i].evaluation.objective < incumbent) {
      incumbent = states[i].evaluation.objective;
      best = i;
      have_best = true;
    }
  }
  require(have_best, "certified pruning removed every complete state");
  return best;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: native_pipeline_v1_pruning_contract CATALOG");
    const auto catalog = ngpipe::ActionCatalog::load(std::filesystem::path(argv[1]));
    auto dispatcher = graph::make_default_dispatcher();

    // Exhaust every legal recipe of depth <=2 in a fixed three-action space.
    // This is deliberately small enough to compare every reachable complete
    // frame, while including a scalar stage, a multi-output stage and Split.
    ngpipe::GraphTask task(catalog, *dispatcher, append_request());
    std::vector<MeasuredState> states{{task.initial_state(), std::nullopt, 0, {}}};
    const std::array<std::uint32_t, 3> bounded_actions{5, 91, 194};
    for (std::uint32_t depth = 0; depth < 2; ++depth) {
      const auto level_size = states.size();
      for (std::size_t i = 0; i < level_size; ++i) {
        if (states[i].depth != depth) continue;
        const auto legal = task.legal_actions(states[i].state);
        for (const auto action : bounded_actions) {
          if (std::find(legal.begin(), legal.end(), action) == legal.end()) continue;
          const auto next = task.transition(states[i].state, action);
          require(next.has_value(), "bounded legal action failed transition");
          states.push_back({*next, i, depth + 1, {}});
        }
      }
    }
    require(states.size() > 4 && states.size() < task.candidate_budget(),
            "bounded exhaustive space has an unexpected size");
    for (auto& row : states) {
      row.evaluation = task.evaluate_terminal(row.state);
      require(row.evaluation.success, "bounded complete recipe failed evaluation");
    }
    const auto cache = task.replay_cache_stats();
    require(cache.hits > 0 && cache.entries > 0 &&
                cache.retained_bytes <= cache.byte_limit &&
                cache.byte_limit == (64ULL << 20),
            "bounded exhaustive replay cache did not enforce its audited LRU limit");
    for (std::size_t ancestor = 0; ancestor < states.size(); ++ancestor)
      for (std::size_t child = 0; child < states.size(); ++child)
        if (is_descendant(states, child, ancestor))
          require(task.certified_frame_byte_lower_bound(states[ancestor].state) <=
                      states[child].evaluation.frame.size(),
                  "append lower bound exceeded a reachable descendant frame");
    std::size_t append_pruned = 0;
    const auto exhaustive_best = best_index(states);
    const auto certified_best = best_with_certified_pruning(
        task, states, &append_pruned);
    require(append_pruned > 0, "bounded fixture did not exercise certified pruning");
    require(states[certified_best].state.recipe == states[exhaustive_best].state.recipe &&
            states[certified_best].evaluation.objective ==
                states[exhaustive_best].evaluation.objective,
            "certified pruning changed the exhaustive optimum");

    // A bound equal to the incumbent is a retained tie, never a pruned state.
    const auto tied_lower = ngpipe::certified_objective_lower_bound(
        task, states.back().state);
    require(!ngpipe::certified_should_prune(
                task, states.back().state, static_cast<double>(tied_lower)),
            "certified pruning removed an objective tie");

    // Use a denominator and weight that are not exactly representable in
    // binary. The certified value must be evaluated in the same double domain
    // as J_selection and conservatively rounded down; a pure-ratio incumbent
    // equal to the unadjusted runtime expression remains a tie.
    ngpipe::GraphTask nonbinary(
        catalog, *dispatcher, nonbinary_ratio_request());
    const auto nonbinary_bytes = nonbinary.certified_frame_byte_lower_bound(
        nonbinary.initial_state());
    const auto runtime_tie = 0.4 * (double(nonbinary_bytes) / 3.0);
    const auto certified_tie = ngpipe::certified_objective_lower_bound(
        nonbinary, nonbinary.initial_state());
    require(certified_tie == std::nextafter(runtime_tie, 0.0) &&
                !ngpipe::certified_should_prune(
                    nonbinary, nonbinary.initial_state(), runtime_tie),
            "non-binary ratio rounding incorrectly pruned a double objective tie");

    // Continuation reachability includes both deletion and replacement. Its
    // lower bound deducts every still-available deletion before claiming any
    // policy node is unavoidable.
    auto continuation_request = append_request();
    continuation_request.task_id = "bounded-continuation-pruning";
    continuation_request.block_index = 1;
    continuation_request.continuation_edit_mode = true;
    continuation_request.previous_recipe = {91};
    continuation_request.max_recipe_edits = 1;
    ngpipe::GraphTask continuation(catalog, *dispatcher, continuation_request);
    const auto delete_type = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) { return row.edit_kind == "delete"; })->action_id;
    const auto replace_type = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) { return row.edit_kind == "replace"; })->action_id;
    const auto position_zero = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) {
          return row.category == ngpipe::ActionCategory::EditPosition &&
                 row.parameter_value == 0;
        })->action_id;
    const auto delete_prefix = continuation.transition(
        continuation.initial_state(), delete_type);
    require(delete_prefix.has_value(), "continuation delete token was not legal");
    const auto deleted = continuation.transition(*delete_prefix, position_zero);
    require(deleted.has_value() && deleted->recipe.empty(),
            "continuation delete did not produce the empty recipe");
    auto replace_prefix = continuation.transition(
        continuation.initial_state(), replace_type);
    require(replace_prefix.has_value(), "continuation replace token was not legal");
    replace_prefix = continuation.transition(*replace_prefix, position_zero);
    require(replace_prefix.has_value(), "continuation replacement position was not legal");
    const auto replaced = continuation.transition(*replace_prefix, 194);
    require(replaced.has_value() && replaced->recipe == std::vector<std::uint32_t>{194},
            "continuation replacement action was not legal");
    std::vector<MeasuredState> continuation_states{
        {continuation.initial_state(), std::nullopt, 0, {}},
        {*deleted, 0, 1, {}},
        {*replaced, 0, 1, {}},
    };
    for (auto& row : continuation_states) {
      row.evaluation = continuation.evaluate_terminal(row.state);
      require(row.evaluation.success, "continuation complete recipe failed evaluation");
      require(continuation.certified_frame_byte_lower_bound(
                  continuation.initial_state()) <= row.evaluation.frame.size(),
              "continuation initial bound exceeded delete/replace descendant frame");
      require(continuation.certified_frame_byte_lower_bound(row.state) <=
                  row.evaluation.frame.size(),
              "continuation state bound exceeded its complete frame");
    }
    const auto continuation_best = best_index(continuation_states);
    std::size_t continuation_pruned = 0;
    const auto continuation_certified_best = best_with_certified_pruning(
        continuation, continuation_states, &continuation_pruned);
    require(continuation_states[continuation_best].state.recipe ==
                continuation_states[continuation_certified_best].state.recipe,
            "continuation certified pruning changed the exhaustive optimum");

    // With a=0 the byte lower bound is inapplicable and must never prune.
    auto time_only_request = append_request();
    time_only_request.task_id = "bounded-pruning-a-zero";
    time_only_request.weights = {0.0, 1.0, 0.0};
    ngpipe::GraphTask time_only(catalog, *dispatcher, time_only_request);
    require(ngpipe::certified_objective_lower_bound(
                time_only, time_only.initial_state()) == 0.0 &&
            !ngpipe::certified_should_prune(
                time_only, time_only.initial_state(), 0.0),
            "a=0 incorrectly enabled a byte-only target prune");

    std::cout << "native_pipeline_v1 bounded pruning contract passed; states="
              << states.size() << " append_pruned=" << append_pruned
              << " continuation_pruned=" << continuation_pruned << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
