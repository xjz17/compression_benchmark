#include "search.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace wc::pipeline_v1 {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t elapsed(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}
struct Node { TaskState state; std::vector<std::uint32_t> actions; double rank = 0; std::uint32_t depth = 0; };
std::string key(const TaskState& state) {
  std::ostringstream out; out << (state.continuation_edit_mode ? "c:" : "r:");
  for (auto id : state.recipe) out << id << ',';
  if (state.continuation_edit_mode) {
    out << '|' << state.pending_edit_kind.value_or("") << '|';
    if (state.pending_edit_index) out << *state.pending_edit_index;
    out << '|' << state.edit_count;
    for (const auto& edit : state.edit_history) {
      out << '|' << edit.kind << ':' << edit.position << ':';
      if (edit.action_id) out << *edit.action_id;
    }
  }
  return out.str();
}
bool includes(std::string text, const char* needle) { return text.find(needle) != std::string::npos; }
std::string group(const ActionSpec& action) {
  auto text = action.key + " " + action.runtime_action + " " + category_name(action.category);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
  if (includes(text,"delta") || includes(text,"zigzag") || includes(text,"bitpack") ||
      includes(text,"bitunpack") || includes(text,"frame_of_reference")) return "delta";
  if (includes(text,"dictionary") || includes(text,"run_length") || includes(text,"rle")) return "dictionary";
  if (includes(text,"xor") || includes(text,"window") || includes(text,"scaling") ||
      includes(text,"outlier") || includes(text,"exception")) return "xor";
  return "other";
}
double order_hint(GraphTask& task, const ActionSpec& action,
                  const std::array<double, features_v1::task_width>& features) {
  const auto& w = task.request().weights;
  const auto smooth = std::clamp(features[16], 0.0, 1.0);
  const auto repeated = std::clamp(std::max(features[13], features[15]), 0.0, 1.0);
  const auto floating = double(std::count_if(task.request().roots.begin(), task.request().roots.end(),
      [](const auto& root) { return root.logical_view.starts_with("float"); })) / task.request().roots.size();
  std::map<std::string,double> mass{{"delta",1+2*w[0]*smooth},{"dictionary",1+2*w[0]*repeated},
      {"xor",1+2*w[0]*floating},{"fast_e",1+2*w[1]},{"fast_d",1+2*w[2]},{"other",1}};
  double total=0; for(const auto& [_,v]:mass) total+=v; for(auto& [_,v]:mass)v/=total;
  const auto measured = task.measured_action_cost(action.action_id);
  const auto ratio = measured ? (*measured)[0] : 1.0;
  const auto e = measured ? (*measured)[1] : task.request().compression_reference_ns_per_byte;
  const auto d = measured ? (*measured)[2] : task.request().decompression_reference_ns_per_byte;
  const auto en = e/(e+task.request().compression_reference_ns_per_byte);
  const auto dn = d/(d+task.request().decompression_reference_ns_per_byte);
  const auto bonus=0.10/std::sqrt(task.action_measurement_count(action.action_id)+1.0)+0.05*mass[group(action)];
  return w[0]*ratio+w[1]*en+w[2]*dn-bonus+double(action.action_id)*1e-12;
}
CandidateSummary summary(const Node& node, const Evaluation& e) {
  CandidateSummary out;
  out.success=e.success;out.failure_error=e.failure_error;out.recipe=node.state.recipe;
  out.search_actions=node.actions;
  if(e.success){out.objective=e.objective;out.compression_ratio=e.compression_ratio;
    out.compression_ns_per_byte=e.compression_ns_per_byte;
    out.decompression_ns_per_byte=e.decompression_ns_per_byte;out.complete_bytes=e.frame.size();}
  return out;
}
template<class T> std::vector<T> select_exploration(GraphTask& task, const std::vector<T>& ordered,
                                                     std::size_t limit) {
  if (ordered.size() <= limit || limit <= 1) return {ordered.begin(), ordered.begin()+std::min(limit,ordered.size())};
  const auto explore = std::max<std::size_t>(1,limit/5), exploit=limit-explore;
  std::vector<T> out(ordered.begin(),ordered.begin()+exploit);
  std::vector<T> remaining(ordered.begin()+exploit,ordered.end());
  std::sort(remaining.begin(),remaining.end(),[&](const T& a,const T& b){
    const auto ca=task.action_measurement_count(std::get<2>(a)),cb=task.action_measurement_count(std::get<2>(b));
    return ca!=cb?ca<cb:std::get<2>(a)<std::get<2>(b);
  });
  out.insert(out.end(),remaining.begin(),remaining.begin()+explore); return out;
}
}  // namespace

double certified_objective_lower_bound(const GraphTask& task,
                                       const TaskState& state) {
  const auto a = task.request().weights[0];
  if (!(a > 0.0)) return 0.0;
  const auto bytes = task.certified_frame_byte_lower_bound(state);
  // Match evaluate_terminal's binary64 evaluation order. The byte bound is no
  // larger than any reachable frame and binary64 division/multiplication are
  // monotone for these finite nonnegative operands. Move one representable
  // value toward zero so a rounded runtime tie cannot become a prune proof.
  const auto rounded = a * (double(bytes) / double(task.original_numeric_bytes()));
  return std::nextafter(rounded, 0.0);
}

bool certified_should_prune(const GraphTask& task, const TaskState& state,
                            double incumbent) {
  if (!std::isfinite(incumbent)) return false;
  const auto lower = certified_objective_lower_bound(task, state);
  // Strict comparison retains ties in the same double objective domain used
  // by candidate evaluation. All omitted normalized time terms are nonnegative.
  return lower > incumbent;
}

namespace {
double failed_candidate_rank(const GraphTask& task) {
  const auto& w = task.request().weights;
  return 1.0 + w[0] * double(task.request().config.graph_limits.max_frame_bytes) /
      double(task.original_numeric_bytes()) + w[1] + w[2];
}
}  // namespace

const char* search_mode_name(SearchMode mode) noexcept {
  return mode == SearchMode::Policy ? "policy" : "beam30";
}

SearchResult PolicySearch::run(GraphTask& task,
                               const native_graph_v1::StageDispatcher& dispatcher,
                               SearchOptions options,
                               const std::vector<std::vector<std::uint32_t>>& starting_recipes,
                               std::optional<Clock::time_point> external_total_begin,
                               std::uint64_t task_initialization_ns) const {
  if (options.beam_width < 1)
    throw std::invalid_argument("beam width must be positive");
  if (options.mode == SearchMode::Policy && policy_ == nullptr)
    throw std::invalid_argument("policy search requires a frozen model");
  if (task.candidate_used() != 0)
    throw std::invalid_argument("search requires an unused complete-candidate budget");
  const auto total_begin = external_total_begin.value_or(Clock::now());
  SearchResult result;
  result.timing.task_initialization_ns = task_initialization_ns;
  result.evaluation_budget = task.candidate_budget();
  std::vector<Node> frontier;
  if (starting_recipes.empty()) frontier.push_back({task.initial_state(), {}, 0, 0});
  else {
    if (task.request().block_index != 0 || task.request().continuation_edit_mode)
      throw std::invalid_argument("solution-library frontier is first-block only");
    for (const auto& recipe : starting_recipes)
      frontier.push_back({task.starting_state(recipe), {}, 0, 0});
  }
  std::set<std::string> seen;
  double incumbent = std::numeric_limits<double>::infinity();
  const auto horizon = task.max_steps();

  auto record_candidate = [&](Node& node) -> double {
    const auto started = Clock::now();
    auto evaluated = task.evaluate_terminal(node.state);
    result.timing.candidate_evaluation_ns += elapsed(started, Clock::now());
    ++result.evaluations;
    result.candidates.push_back(summary(node, evaluated));
    if (!evaluated.success) {
      result.failure_errors.push_back(evaluated.failure_error);
      return failed_candidate_rank(task);
    }
    const auto objective = evaluated.objective;
    if (objective < incumbent) {
      incumbent = objective;
      result.best_search_actions = node.actions;
      result.best_recipe = node.state.recipe;
      result.best_evaluation = std::move(evaluated);
    }
    return objective;
  };
  auto should_prune = [&](const TaskState& state) {
    const auto begin = Clock::now();
    const auto answer = options.enable_certified_pruning &&
        certified_should_prune(task, state, incumbent);
    result.timing.certified_pruning_ns += elapsed(begin, Clock::now());
    if (answer) ++result.certified_pruned_states;
    return answer;
  };
  auto charge_unexplored_frontier = [&](std::size_t current) {
    const auto legal = task.legal_actions(frontier[current].state);
    result.budget_unexplored_actions += legal.size() -
        unsigned(task.is_terminal_evaluable(frontier[current].state));
    for (std::size_t j = current + 1; j < frontier.size(); ++j) {
      const auto remaining = task.legal_actions(frontier[j].state);
      result.budget_unexplored_actions += remaining.size() -
          unsigned(task.is_terminal_evaluable(frontier[j].state));
    }
  };

  if (options.mode == SearchMode::Policy) {
    while (!frontier.empty() && result.evaluations < result.evaluation_budget) {
      std::vector<std::tuple<double, Node, std::uint32_t>> proposals;
      bool budget_hit = false;
      for (std::size_t ni = 0; ni < frontier.size(); ++ni) {
        auto& node = frontier[ni];
        if (!seen.insert(key(node.state)).second) {
          ++result.duplicate_states_skipped;
          continue;
        }
        // Every training-library start is an explicit first-block candidate
        // and must be charged/evaluated before policy expansion. Certified
        // pruning applies only after this mandatory library selection layer.
        if (!(node.depth == 0 && !starting_recipes.empty()) &&
            should_prune(node.state)) continue;
        if (task.is_terminal_evaluable(node.state)) {
          (void)record_candidate(node);
          if (result.evaluations >= result.evaluation_budget) {
            charge_unexplored_frontier(ni);
            budget_hit = true;
            break;
          }
        }
        if (node.depth >= horizon - 1) continue;
        const auto feature_begin = Clock::now();
        const auto inspected = task.inspect(node.state);
        std::array<double, features_v1::observation_width> observation{};
        std::copy(inspected.task_features.begin(), inspected.task_features.end(),
                  observation.begin());
        std::copy(inspected.state_features.begin(), inspected.state_features.end(),
                  observation.begin() + features_v1::task_width);
        std::copy(task.request().weights.begin(), task.request().weights.end(),
                  observation.begin() + features_v1::task_width + features_v1::state_width);
        result.timing.feature_ns += elapsed(feature_begin, Clock::now());
        std::vector<std::uint8_t> mask(action_count);
        for (auto id : inspected.legal_action_ids) mask[id] = 1;
        const auto policy_begin = Clock::now();
        const auto prediction = policy_->infer(observation, mask);
        result.timing.policy_ns += elapsed(policy_begin, Clock::now());
        const auto order_begin = Clock::now();
        std::vector<std::tuple<double, double, std::uint32_t>> scored;
        for (auto action : inspected.legal_action_ids) if (action != stop_action) {
          const auto hint = options.enable_objective_guided_ordering
              ? order_hint(task, task.action_record(action), inspected.task_features) : 0.0;
          scored.emplace_back(-prediction.probabilities[action], hint, action);
        }
        std::sort(scored.begin(), scored.end());
        result.timing.action_ordering_ns += elapsed(order_begin, Clock::now());
        for (const auto& [negative_probability, objective_hint, action] : scored) {
          (void)negative_probability;
          proposals.emplace_back(
              node.rank + std::log(std::max(prediction.probabilities[action], 1e-300))
                  - 0.25 * objective_hint,
              node, action);
        }
      }
      if (budget_hit || proposals.empty()) break;
      std::sort(proposals.begin(), proposals.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
        if (std::get<1>(a).actions != std::get<1>(b).actions)
          return std::get<1>(a).actions < std::get<1>(b).actions;
        return std::get<2>(a) < std::get<2>(b);
      });
      const auto limit = std::min<std::size_t>(options.beam_width, proposals.size());
      const auto selected = options.enable_objective_guided_ordering
          ? select_exploration(task, proposals, limit)
          : std::vector<std::tuple<double, Node, std::uint32_t>>(
                proposals.begin(), proposals.begin() + limit);
      result.budget_unexplored_actions += proposals.size() - selected.size();
      std::vector<Node> next;
      for (const auto& [rank, node, action] : selected) {
        const auto begin = Clock::now();
        auto following = task.transition(node.state, action);
        result.timing.transition_ns += elapsed(begin, Clock::now());
        if (!following) throw std::runtime_error("legal action transition failed");
        auto actions = node.actions;
        actions.push_back(action);
        next.push_back({std::move(*following), std::move(actions), rank, node.depth + 1});
      }
      frontier = std::move(next);
    }
  } else {
    // Model-independent Beam30. Exact complete-candidate J ranks parents;
    // objective guidance only orders their legal child actions.
    while (!frontier.empty() && result.evaluations < result.evaluation_budget) {
      std::vector<std::pair<double, Node>> ranked;
      bool budget_hit = false;
      for (std::size_t ni = 0; ni < frontier.size(); ++ni) {
        auto& node = frontier[ni];
        if (!seen.insert(key(node.state)).second) {
          ++result.duplicate_states_skipped;
          continue;
        }
        if (!(node.depth == 0 && !starting_recipes.empty()) &&
            should_prune(node.state)) continue;
        auto parent_rank = node.rank;
        if (task.is_terminal_evaluable(node.state)) {
          parent_rank = record_candidate(node);
          if (result.evaluations >= result.evaluation_budget) {
            charge_unexplored_frontier(ni);
            budget_hit = true;
            break;
          }
        }
        ranked.emplace_back(parent_rank, node);
      }
      if (budget_hit || ranked.empty()) break;
      std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second.actions < b.second.actions;
      });
      std::vector<std::tuple<double, Node, std::uint32_t>> proposals;
      for (std::size_t i = 0; i < std::min<std::size_t>(options.beam_width, ranked.size()); ++i) {
        const auto& [parent_rank, node] = ranked[i];
        if (node.depth >= horizon - 1) continue;
        auto ranked_node = node;
        ranked_node.rank = parent_rank;
        const auto feature_begin = Clock::now();
        const auto inspected = task.inspect(node.state);
        result.timing.feature_ns += elapsed(feature_begin, Clock::now());
        const auto order_begin = Clock::now();
        for (const auto action : inspected.legal_action_ids) if (action != stop_action) {
          const auto score = options.enable_objective_guided_ordering
              ? order_hint(task, task.action_record(action), inspected.task_features)
              : double(action);
          proposals.emplace_back(score, ranked_node, action);
        }
        result.timing.action_ordering_ns += elapsed(order_begin, Clock::now());
      }
      if (proposals.empty()) break;
      std::sort(proposals.begin(), proposals.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        if (std::get<1>(a).actions != std::get<1>(b).actions)
          return std::get<1>(a).actions < std::get<1>(b).actions;
        return std::get<2>(a) < std::get<2>(b);
      });
      const auto remaining = result.evaluation_budget - result.evaluations;
      const auto limit = std::min<std::size_t>(
          {options.beam_width, proposals.size(), std::max<std::uint32_t>(1, remaining)});
      const auto selected = options.enable_objective_guided_ordering
          ? select_exploration(task, proposals, limit)
          : std::vector<std::tuple<double, Node, std::uint32_t>>(
                proposals.begin(), proposals.begin() + limit);
      result.budget_unexplored_actions += proposals.size() - selected.size();
      std::vector<Node> next;
      for (const auto& [score, node, action] : selected) {
        (void)score;
        const auto begin = Clock::now();
        auto following = task.transition(node.state, action);
        result.timing.transition_ns += elapsed(begin, Clock::now());
        if (!following) throw std::runtime_error("legal action transition failed");
        auto actions = node.actions;
        actions.push_back(action);
        next.push_back({std::move(*following), std::move(actions), node.rank, node.depth + 1});
      }
      frontier = std::move(next);
    }
  }

  result.timing.total_prediction_ns = elapsed(total_begin, Clock::now());
  if (!std::isfinite(incumbent)) {
    result.timing.total_compression_ns = elapsed(total_begin, Clock::now());
    return result;
  }
  result.success = true;
  const auto final_begin = Clock::now();
  result.final_frame = native_graph_v1::encode(
      task.request().roots, task.compile_recipe(result.best_recipe), dispatcher,
      task.request().config.graph_limits);
  result.timing.final_encode_ns = elapsed(final_begin, Clock::now());
  result.timing.total_compression_ns = elapsed(total_begin, Clock::now());
  if (result.final_frame != result.best_evaluation.frame)
    throw std::runtime_error("independent final encode changed complete frame bytes");
  return result;
}

}  // namespace wc::pipeline_v1
