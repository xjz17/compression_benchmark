#include "pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace wc::pipeline_v1 {
namespace {
namespace graph = native_graph_v1;
namespace features = wc::features_v1;
using Clock = std::chrono::steady_clock;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}
std::uint64_t elapsed(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
struct Mapping { std::uint32_t slot; bool int64; };
std::optional<Mapping> mapping(std::uint32_t id) {
  // Frozen ACTION_CATALOG rows have 195 configurations per target slot.
  for (std::uint32_t slot = 0; slot < 32; ++slot) {
    if (id == 100 + 195 * slot) return Mapping{slot, false};
    if (id == 101 + 195 * slot) return Mapping{slot, true};
  }
  return std::nullopt;
}
features::FlowView flow_view(const graph::Stream& flow, const std::string& logical) {
  const std::string kind = flow.kind == graph::Kind::Serial ? "serial" :
      flow.kind == graph::Kind::Numeric ? "numeric" : "struct";
  return {kind, flow.width, flow.data, logical};
}
std::vector<features::FlowView> root_views(const Request& request) {
  std::vector<features::FlowView> out;
  for (const auto& root : request.roots) {
    require(!root.logical_view.empty(), "pipeline requires explicit root logical views");
    out.push_back(flow_view(root.stream, root.logical_view));
  }
  return out;
}
features::State state(const graph::GraphSession& session, const Request& request,
                      std::uint32_t edits) {
  features::State out;
  out.original_numeric_bytes = session.original_numeric_bytes();
  out.node_count = session.node_count(); out.policy_edits = edits;
  out.max_nodes = request.limits.max_nodes;
  out.max_depth = request.limits.max_ordinary_depth;
  out.max_active = request.limits.max_active_streams;
  out.max_policy_edits = request.max_policy_steps;
  out.action_count = action_count;
  for (const auto& [id, flow] : session.active_streams()) {
    const auto& info = session.flow_info(id);
    out.flows.push_back({flow_view(flow, info.logical_view), info.ordinary_depth,
                        info.policy_action_id < 0 ? stop_action : info.policy_action_id});
  }
  return out;
}
auto target(const graph::GraphSession& session, std::uint32_t slot) {
  auto found = session.active_streams().begin();
  if (slot >= session.active_streams().size()) return session.active_streams().end();
  std::advance(found, slot); return found;
}
std::vector<std::uint8_t> legal_mask(const Request& request,
                                    const graph::GraphSession& session,
                                    std::uint32_t edits) {
  std::vector<std::uint8_t> out(action_count, 0);
  out[stop_action] = request.enabled_actions[stop_action];
  if (edits >= request.max_policy_steps || session.node_count() >= request.limits.max_nodes)
    return out;
  for (const auto action : supported_action_ids()) {
    if (action == stop_action || !request.enabled_actions[action]) continue;
    const auto spec = *mapping(action);
    const auto found = target(session, spec.slot);
    if (found == session.active_streams().end()) continue;
    const auto& flow = found->second;
    const auto& info = session.flow_info(found->first);
    if (flow.data.empty() || flow.data.size() % 8 != 0 ||
        (flow.kind != graph::Kind::Serial && flow.width != 8) ||
        info.ordinary_depth >= request.limits.max_ordinary_depth) continue;
    if (spec.int64 && info.logical_view != "int64") continue;
    out[action] = 1;
  }
  return out;
}
void verify(const graph::Decoded& decoded, const Request& request, std::uint64_t raw) {
  require(decoded.original_numeric_bytes == raw && decoded.roots.size() == request.roots.size(),
          "pipeline decoded root count or original numeric denominator mismatch");
  for (std::size_t i = 0; i < request.roots.size(); ++i) {
    const auto& expected = request.roots[i]; const auto& actual = decoded.roots[i];
    require(expected.stream.kind == actual.stream.kind && expected.stream.width == actual.stream.width &&
            expected.stream.data == actual.stream.data && expected.logical_view == actual.logical_view &&
            expected.original_numeric_payload == actual.original_numeric_payload,
            "pipeline bitwise/root metadata roundtrip mismatch");
  }
}
std::optional<double> normalized(std::uint64_t n, std::uint64_t d) {
  return d ? std::optional<double>(double(n) / double(d)) : std::nullopt;
}
}  // namespace

policy::RuntimeIdentity runtime_identity() {
  return {registry_fingerprint, policy::feature_schema, action_count, stop_action, 18, 28};
}
std::vector<std::uint32_t> supported_action_ids() {
  std::vector<std::uint32_t> result{stop_action};
  for (std::uint32_t slot = 0; slot < 32; ++slot) {
    result.push_back(100 + 195 * slot); result.push_back(101 + 195 * slot);
  }
  return result;
}
std::optional<double> Result::compression_ratio() const {
  return normalized(frame.size(), original_numeric_bytes);
}
std::optional<double> Result::compression_ns_per_byte() const {
  return normalized(timing.compression_ns, original_numeric_bytes);
}
std::optional<double> Result::decompression_ns_per_byte() const {
  return normalized(timing.decompression_ns, original_numeric_bytes);
}
Pipeline Pipeline::load(const std::filesystem::path& model) {
  return Pipeline(policy::Policy::load(model, runtime_identity()));
}
Result Pipeline::compress(const Request& request, const graph::StageDispatcher& dispatcher) const {
  const auto begin = Clock::now();
  require(request.enabled_actions.size() == action_count, "enabled action mask dimension mismatch");
  for (std::uint32_t id = 0; id < action_count; ++id) {
    require(request.enabled_actions[id] <= 1, "enabled action mask is not binary");
    require(!request.enabled_actions[id] || id == stop_action || mapping(id).has_value(),
            "UNSUPPORTED_NATIVE_POLICY_ACTION: " + std::to_string(id));
  }
  require(request.enabled_actions[stop_action], "explicit STOP permission is required");
  require(request.max_policy_steps > 0 && request.max_policy_steps <= 64,
          "policy step limit must be in [1,64]");
  graph::GraphSession session(request.roots, dispatcher, request.limits, registry_fingerprint);
  require(session.original_numeric_bytes() > 0,
          "NO_ORIGINAL_NUMERIC_PAYLOAD: frozen feature schema is undefined; metrics N/A");
  Result out; out.original_numeric_bytes = session.original_numeric_bytes();
  out.recipe.policy_registry = registry_fingerprint;
  for (std::uint32_t edits = 0;; ++edits) {
    const auto feature_begin = Clock::now();
    const auto views = root_views(request);
    const auto observation = features::observation(views, state(session, request, edits), request.weights);
    if (edits == 0 && request.expected_initial_features) {
      for (std::size_t i = 0; i < observation.size(); ++i) {
        const auto expected = (*request.expected_initial_features)[i];
        require(std::isfinite(expected) && std::abs(expected - observation[i]) <=
                    1e-12 * std::max(1.0, std::abs(observation[i])),
                "supplied initial 49-D features disagree with native extraction");
      }
    }
    const auto feature_end = Clock::now();
    out.timing.feature_ns += elapsed(feature_begin, feature_end);
    const auto mask = legal_mask(request, session, edits);
    const auto policy_begin = Clock::now();
    const auto prediction = policy_.infer(observation, mask);
    const auto policy_end = Clock::now();
    out.timing.policy_ns += elapsed(policy_begin, policy_end);
    const auto action = prediction.ordered_actions.front();
    out.selected_action_ids.push_back(action);
    graph::Step step; step.policy_action_id = action;
    const auto selection_begin = Clock::now();
    if (action == stop_action) {
      step.kind = graph::StepKind::Stop; out.recipe.steps.push_back(step);
      // Candidate completion is part of selection cost, never hidden. The final
      // reported frame is independently re-encoded from original roots below.
      (void)session.stop();
      out.timing.selection_graph_ns += elapsed(selection_begin, Clock::now());
      break;
    }
    const auto spec = mapping(action);
    require(spec.has_value(), "UNSUPPORTED_NATIVE_POLICY_ACTION: " + std::to_string(action));
    const auto found = target(session, spec->slot);
    require(found != session.active_streams().end(), "selected target slot is not active");
    step.inputs = {found->first};
    step.stage = {"op.bitpack_int.max_value_width.cpp_v1", spec->int64 ? "int64" : "bytes", {}};
    // Any dispatcher error aborts this input; no lower-ranked fallback exists.
    session.apply(step); out.recipe.steps.push_back(std::move(step));
    out.timing.selection_graph_ns += elapsed(selection_begin, Clock::now());
  }
  const auto final_begin = Clock::now();
  out.frame = graph::encode(request.roots, out.recipe, dispatcher, request.limits);
  const auto final_end = Clock::now();
  out.timing.final_encode_ns = elapsed(final_begin, final_end);
  out.timing.compression_ns = elapsed(begin, final_end);
  return out;
}
BenchmarkResult Pipeline::benchmark(const Request& request, const graph::StageDispatcher& dispatcher,
                                    std::uint32_t warmups, std::uint32_t repeats) const {
  require(warmups <= 31 && repeats > 0 && repeats <= 1001, "invalid pipeline benchmark repetitions");
  BenchmarkResult result;
  for (std::uint32_t i = 0; i < warmups + repeats; ++i) {
    auto sample = compress(request, dispatcher);
    const auto begin = Clock::now();
    auto restored = graph::decode(sample.frame, dispatcher, request.limits);
    sample.timing.decompression_ns = elapsed(begin, Clock::now());
    verify(restored, request, sample.original_numeric_bytes); // Outside timed regions.
    if (i >= warmups) result.samples.push_back(sample.timing);
    result.last = std::move(sample);
  }
  return result;
}
}  // namespace wc::pipeline_v1
