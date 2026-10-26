#include "online.hpp"
#include "dbp_exact_stages.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace wc::pipeline_v1 {
namespace {
namespace graph = native_graph_v1;
namespace features = wc::features_v1;
namespace policy = weighted_grpo::native_policy_v1;
using Clock = std::chrono::steady_clock;

enum class PacketKind { None, DbpScaling, Simple8bScaling, Ts2diffScaling };

struct Meta {
  std::uint32_t depth = 0;
  std::string view;
  std::uint32_t parent_action = stop_action;
  PacketKind packet_kind = PacketKind::None;
};

void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument(message);
}

std::uint64_t elapsed(Clock::time_point begin, Clock::time_point end) {
  const auto value =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  if (value < 0) throw std::runtime_error("nonmonotonic steady clock");
  return static_cast<std::uint64_t>(value);
}

template <class T>
bool contains(const std::vector<T>& values, const T& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

bool graph_category(ActionCategory value) {
  return value == ActionCategory::Mechanism ||
         value == ActionCategory::SplitSegments ||
         value == ActionCategory::Concat;
}

std::string physical_kind(graph::Kind kind) {
  if (kind == graph::Kind::Serial) return "serial";
  if (kind == graph::Kind::Numeric) return "numeric";
  return "struct";
}

features::FlowView flow_view(const graph::Stream& flow,
                            const std::string& logical_view) {
  return {physical_kind(flow.kind), flow.width, flow.data, logical_view};
}

std::int64_t little_i64(std::span<const std::uint8_t> bytes,
                        std::size_t offset) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= std::uint64_t(bytes[offset + i]) << (8 * i);
  return std::bit_cast<std::int64_t>(value);
}

bool exactly_representable_as_float64(std::int64_t value) {
  const auto magnitude = value < 0
      ? std::uint64_t{0} - static_cast<std::uint64_t>(value)
      : static_cast<std::uint64_t>(value);
  const auto width = std::bit_width(magnitude);
  if (width <= 53) return true;
  const auto discarded = width - 53;
  return (magnitude & ((std::uint64_t{1} << discarded) - 1)) == 0;
}

std::vector<graph::StreamId> active_ids(const graph::GraphSession& session) {
  return session.open_stream_ids();
}

std::vector<std::uint64_t> segment_sizes(const ActionSpec& spec,
                                         const graph::Stream& stream) {
  require(stream.width > 0 && stream.data.size() % stream.width == 0,
          "segmentation input is not element aligned");
  const std::uint64_t count = stream.data.size() / stream.width;
  if (spec.parameter_mode == "half") return {count / 2, count - count / 2};
  require(spec.parameter_mode == "quarters",
          "invalid segmentation parameter mode");
  const auto first = std::max<std::uint64_t>(1, count / 4);
  const auto second = std::max<std::uint64_t>(1, (count - first) / 3);
  const auto third =
      std::max<std::uint64_t>(1, (count - first - second) / 2);
  if (first + second + third >= count)
    return {count / 2, count - count / 2};
  return {first, second, third, count - first - second - third};
}

std::map<std::string, std::string> parameters(const ActionSpec& spec,
                                              const graph::Stream& stream) {
  auto result = spec.parameter_items;
  if (spec.io_variant && *spec.io_variant != spec.runtime_action)
    result["io_variant"] = *spec.io_variant;
  if (spec.max_point) result["max_point"] = std::to_string(*spec.max_point);
  if (spec.runtime_action == "op.scaling.legacy_exact_decimal.cpp_v1" &&
      spec.input_view == "float64")
    result["max_point"] = "18";
  if (spec.parameter_mode == "none") return result;
  if (spec.parameter_mode == "fp32") {
    result["format"] = "fp32";
  } else if (spec.parameter_mode == "top8") {
    result["mode"] = "top8";
    result["width"] = std::to_string(stream.width);
  } else if (spec.parameter_mode == "bits") {
    result["bits"] = std::to_string(*spec.parameter_value);
  } else if (spec.parameter_mode == "width") {
    result["width"] = std::to_string(*spec.parameter_value);
  } else if (spec.parameter_mode == "stream_width") {
    result["mode"] = "transpose";
    result["width"] = std::to_string(stream.width);
  } else if (spec.parameter_mode == "allow_large_memory") {
    result["allow_large_memory"] = "true";
  } else if (spec.parameter_mode != "half" &&
             spec.parameter_mode != "quarters") {
    throw std::invalid_argument("unknown fixed parameter mode: " +
                                spec.parameter_mode);
  }
  return result;
}

std::string stage_profile(const ActionSpec& spec) {
  if (spec.runtime_action == "op.delta.ts_2diff.cpp_v1" &&
      spec.input_view == "bytes")
    return "scaled_packet";
  return spec.input_view == "int64" || spec.input_view == "float64"
      ? spec.input_view
      : "bytes";
}

struct AdapterDecision {
  bool legal = true;
  const InternalAdapter* adapter = nullptr;
};

AdapterDecision adapter_for(const ActionCatalog& catalog, const Meta& meta,
                            const ActionSpec& spec) {
  if (spec.required_parent_runtime_actions.empty()) return {};
  const auto& parent = catalog.at(meta.parent_action).runtime_action;
  if (contains(spec.required_parent_runtime_actions, parent)) return {};
  const auto found = catalog.internal_adapters().find(spec.runtime_action);
  if (found == catalog.internal_adapters().end() ||
      !contains(spec.required_parent_runtime_actions, found->second.action) ||
      (!found->second.required_policy_parents.empty() &&
       !contains(found->second.required_policy_parents, parent)))
    return {false, nullptr};
  return {true, &found->second};
}

const std::unordered_set<std::string>& multiple_output_actions() {
  static const auto actions = [] {
    std::unordered_set<std::string> result;
    for (const auto& capability : graph::native_stage_capabilities())
      if (capability.multiple_outputs) result.insert(capability.action);
    return result;
  }();
  return actions;
}

std::size_t comma_list_size(const std::string& value) {
  if (value.empty()) return 0;
  return 1 + std::count(value.begin(), value.end(), ',');
}

// Exact arity, or a conservative fixed upper bound when the number of ports is
// data dependent (top8 is one or two). Unknown data-dependent reference
// stages remain explicit selected-action failures if their actual arity later
// exceeds the graph bound; they are never trial-executed by the mask.
std::optional<std::size_t> output_arity_bound(const ActionSpec& spec,
                                              const graph::Stream& target) {
  const auto& action = spec.runtime_action;
  if (action == "op.outlier_separation.high_bits.uint32.cpp_v1") return 3;
  if (action == "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1" ||
      action == "op.dictionary.project_words64.cpp_v1" ||
      action == "op.xor.previous_word.semantic_ports.cpp_v1" ||
      action == "op.xor.chimp_history128.cpp_v2" ||
      action == "float_deconstruct")
    return 2;
  if (action == "bitunpack") return 1;
  if (action == "columnize") {
    const auto fields = spec.parameter_items.find("field_widths");
    if (fields != spec.parameter_items.end()) return comma_list_size(fields->second);
    return target.width;
  }
  if (action == "bitsplit") {
    const auto found = spec.parameter_items.find("mode");
    const auto mode = found == spec.parameter_items.end() ? "top8" : found->second;
    if (mode == "top8") return 2;
    if (mode == "fp" || mode == "bf16") return 3;
    if (mode == "widths") {
      const auto widths = spec.parameter_items.find("widths");
      if (widths != spec.parameter_items.end()) return comma_list_size(widths->second);
    }
  }
  return std::nullopt;
}

struct FlowSemanticCache {
  std::optional<bool> nonnegative_i64;
  std::optional<bool> exact_float64_i64;
  std::optional<std::uint32_t> dbp_scaling_legal_max_points;
};

struct MaskContext {
  std::vector<graph::StreamId> ids;
  std::map<graph::StreamId, FlowSemanticCache>* flow_cache = nullptr;
  OnlineCounters* counters = nullptr;
};

bool cached_nonnegative_i64(MaskContext& context, std::size_t slot,
                            const graph::Stream& target) {
  auto& cached = context.flow_cache->at(context.ids.at(slot)).nonnegative_i64;
  if (!cached) {
    bool value = true;
    for (std::size_t i = 0; i < target.data.size(); i += 8)
      if (little_i64(target.data, i) < 0) {
        value = false;
        break;
      }
    cached = value;
  }
  return *cached;
}

bool cached_exact_float64_i64(MaskContext& context, std::size_t slot,
                              const graph::Stream& target) {
  auto& cached =
      context.flow_cache->at(context.ids.at(slot)).exact_float64_i64;
  if (!cached) {
    bool value = true;
    for (std::size_t i = 0; i < target.data.size(); i += 8)
      if (!exactly_representable_as_float64(little_i64(target.data, i))) {
        value = false;
        break;
      }
    cached = value;
  }
  return *cached;
}

bool cached_dbp_scaling_exact(MaskContext& context, std::size_t slot,
                              const graph::Stream& target, int max_point) {
  auto& cached = context.flow_cache->at(context.ids.at(slot))
                     .dbp_scaling_legal_max_points;
  if (!cached) {
    const auto feasibility =
        exact_actions::dbp::scaling_feasibility_le64(target.data);
    cached = feasibility.legal_max_points;
    ++context.counters->semantic_mask_flow_scans;
    context.counters->semantic_mask_value_visits +=
        feasibility.input_value_visits;
  }
  return max_point >= 0 &&
      max_point <= exact_actions::dbp::maximum_scaling_point &&
      (*cached & (std::uint32_t{1} << max_point)) != 0;
}

std::optional<PacketKind> required_scaling_packet(const ActionSpec& spec) {
  if (spec.input_view != "bytes") return std::nullopt;
  if (spec.runtime_action == "op.bitpack_int.fastpfor_pfor.cpp_v1" ||
      spec.runtime_action == "op.delta.stream_vbyte.cpp_v1")
    return PacketKind::DbpScaling;
  if (spec.runtime_action == "op.bitpack_int.simple8b.cpp_v1")
    return PacketKind::Simple8bScaling;
  if (spec.runtime_action == "op.delta.ts_2diff.cpp_v1")
    return PacketKind::Ts2diffScaling;
  return std::nullopt;
}

bool packet_parent_matches(const ActionCatalog& catalog, const Meta& meta,
                           PacketKind expected) {
  if (meta.packet_kind != expected) return false;
  const auto& parent = catalog.at(meta.parent_action);
  if (expected == PacketKind::DbpScaling)
    return parent.runtime_action ==
        "op.scaling.delta_binary_packed.cpp_v1";
  if (parent.runtime_action != "op.scaling.exact_decimal.cpp_v2" ||
      !parent.io_variant)
    return false;
  if (expected == PacketKind::Simple8bScaling)
    return *parent.io_variant == "op.scaling.simple8b.cpp_v1";
  return expected == PacketKind::Ts2diffScaling &&
      *parent.io_variant == "op.scaling.ts_2diff.cpp_v1";
}

bool structurally_legal(const ActionCatalog& catalog,
                        const OnlineRequest& request,
                        const graph::GraphSession& session,
                        const std::map<graph::StreamId, Meta>& meta,
                        const graph::Limits& effective_limits,
                        MaskContext& context,
                        std::uint32_t policy_graph_nodes,
                        const ActionSpec& spec) {
  if (spec.category == ActionCategory::Stop)
    return false;  // Legacy global STOP=0 is disabled in online v2.
  if (!graph_category(spec.category) || !catalog.enabled(spec.action_id) ||
      !catalog.support(spec.action_id).executable)
    return false;
  if (policy_graph_nodes >= request.graph_limits.max_nodes)
    return false;
  if (!spec.target_slot || *spec.target_slot >= context.ids.size()) return false;

  const auto target_slot = *spec.target_slot;
  const auto target_id = context.ids.at(target_slot);
  const auto& target = session.active_streams().at(target_id);
  const auto& target_meta = meta.at(target_id);
  const auto adapter = adapter_for(catalog, target_meta, spec);
  if (!adapter.legal) return false;
  const std::size_t required_nodes = 1 + (adapter.adapter != nullptr);
  if (session.node_count() > effective_limits.max_nodes - required_nodes)
    return false;

  if (spec.input_nonempty && target.data.empty()) return false;
  if (target.data.size() % spec.input_byte_multiple) return false;
  if (spec.required_nonserial_width && target.kind != graph::Kind::Serial &&
      target.width != *spec.required_nonserial_width)
    return false;
  if (!spec.allowed_physical_widths.empty() &&
      !contains(spec.allowed_physical_widths, target.width))
    return false;
  if (!spec.required_logical_views.empty()) {
    if (contains(spec.required_logical_views, std::string("physical_word"))) {
      if (!target_meta.view.starts_with("xor_residual")) return false;
    } else if (!contains(spec.required_logical_views, target_meta.view)) {
      return false;
    }
  }

  if (spec.category == ActionCategory::Concat) {
    if (!spec.other_slot || *spec.other_slot >= context.ids.size() ||
        *spec.other_slot == *spec.target_slot)
      return false;
    const auto& other =
        session.active_streams().at(context.ids.at(*spec.other_slot));
    return target.kind == other.kind && target.width == other.width;
  }

  if (target_meta.depth + unsigned(spec.ordinary_operator) >
      request.graph_limits.max_ordinary_depth)
    return false;
  const bool typed_view = spec.input_view == "int64" ||
      spec.input_view == "float64" || spec.input_view == "float32" ||
      spec.input_view == "uint32";
  if (typed_view &&
      target_meta.view != spec.input_view)
    return false;
  if (const auto expected_packet = required_scaling_packet(spec)) {
    if (!packet_parent_matches(catalog, target_meta, *expected_packet))
      return false;
  }
  if (spec.runtime_action == "op.scaling.delta_binary_packed.cpp_v1") {
    if ((target.kind != graph::Kind::Serial && target.width != 8) ||
        !spec.max_point ||
        !cached_dbp_scaling_exact(context, target_slot, target,
                                  static_cast<int>(*spec.max_point)))
      return false;
  }
  if (spec.runtime_action == "op.bitpack_int.max_value_width.cpp_v1" &&
      target.kind != graph::Kind::Serial && target.width != 8)
    return false;
  if (spec.runtime_action == "op.bitpack_int.max_value_width.cpp_v1" &&
      spec.input_view == "int64" &&
      !cached_nonnegative_i64(context, target_slot, target))
    return false;
  if (spec.runtime_action == "op.eraser.camel.cpp_v1" &&
      spec.input_view == "int64" &&
      !cached_exact_float64_i64(context, target_slot, target))
    return false;
  if (spec.input_view == "physical_word" &&
      !(target.kind == graph::Kind::Numeric &&
        (target.width == 4 || target.width == 8) &&
        target_meta.view ==
            "xor_residual" + std::to_string(target.width * 8)))
    return false;
  if (spec.runtime_action ==
          "op.outlier_separation.high_bits.uint32.cpp_v1" &&
      !(target.kind == graph::Kind::Numeric && target.width == 4))
    return false;
  if (spec.runtime_action == "float_deconstruct" &&
      !(target.kind == graph::Kind::Numeric && target.width == 4 &&
        target_meta.view == "float32"))
    return false;
  if ((spec.runtime_action == "columnize" ||
       spec.runtime_action == "bitsplit") &&
      target.width <= 1)
    return false;
  if (spec.runtime_action == "bitunpack" &&
      target.kind != graph::Kind::Serial)
    return false;
  if (spec.runtime_action == "op.xor.previous_word.cpp_v2" &&
      spec.input_view == "bytes" && target.kind != graph::Kind::Serial)
    return false;

  if (spec.category == ActionCategory::SplitSegments) {
    if (target.data.size() / target.width < 2) return false;
    const auto outputs = segment_sizes(spec, target).size();
    return context.ids.size() - 1 + outputs <=
               request.graph_limits.max_active_streams &&
        session.active_streams().size() - 1 + outputs <=
               effective_limits.max_active_streams;
  }
  if (const auto arity = output_arity_bound(spec, target)) {
    if (context.ids.size() - 1 + *arity >
            request.graph_limits.max_active_streams ||
        session.active_streams().size() - 1 + *arity >
            effective_limits.max_active_streams)
      return false;
  } else if (multiple_output_actions().contains(spec.runtime_action) &&
             (context.ids.size() - 1 + 2 >
                  request.graph_limits.max_active_streams ||
              session.active_streams().size() - 1 + 2 >
                  effective_limits.max_active_streams)) {
    // Unknown data-dependent multi-output arity. When even its minimum two
    // ports cannot fit, reject it without executing a probe. Below capacity,
    // execution remains the authoritative applicability check and failures
    // stay explicit.
    return false;
  }
  return true;
}

std::vector<std::uint32_t> legal_actions(
    const ActionCatalog& catalog, const OnlineRequest& request,
    const graph::GraphSession& session,
    const std::map<graph::StreamId, Meta>& meta,
    const graph::Limits& effective_limits,
    const std::array<std::vector<std::uint32_t>, 32>& actions_by_slot,
    const std::vector<std::uint32_t>& concat_actions,
    std::map<graph::StreamId, FlowSemanticCache>& flow_cache,
    OnlineCounters& counters,
    std::uint32_t policy_graph_nodes) {
  MaskContext context;
  context.ids = active_ids(session);
  context.flow_cache = &flow_cache;
  context.counters = &counters;
  for (const auto id : context.ids) flow_cache.try_emplace(id);
  std::vector<std::uint32_t> out;
  out.reserve(context.ids.size() * 196 + concat_actions.size());
  for (std::size_t slot = 0; slot < context.ids.size(); ++slot) {
    for (const auto action_id : actions_by_slot.at(slot)) {
      const auto& spec = catalog.at(action_id);
      if (structurally_legal(catalog, request, session, meta,
                             effective_limits, context, policy_graph_nodes, spec))
        out.push_back(action_id);
    }
  }
  for (const auto action_id : concat_actions) {
    const auto& spec = catalog.at(action_id);
    if (structurally_legal(catalog, request, session, meta, effective_limits,
                           context, policy_graph_nodes, spec))
      out.push_back(action_id);
  }
  for (std::uint32_t slot = 0; slot < context.ids.size(); ++slot)
    out.push_back(flow_stop_begin + slot);
  require(!out.empty(), "online legality mask produced no action");
  require(std::is_sorted(out.begin(), out.end()) &&
              std::adjacent_find(out.begin(), out.end()) == out.end(),
          "online catalog indexes did not produce ascending unique IDs");
  return out;
}

features::State feature_state(
    const OnlineRequest& request, const graph::GraphSession& session,
    const std::map<graph::StreamId, Meta>& meta,
    std::uint32_t policy_graph_nodes, std::uint32_t policy_calls,
    std::uint64_t original_numeric_bytes) {
  features::State out;
  out.original_numeric_bytes = original_numeric_bytes;
  out.node_count = policy_graph_nodes;
  out.policy_edits = policy_calls;
  out.replayed_nodes = 0;
  out.edit_count = 0;
  out.max_nodes = request.graph_limits.max_nodes;
  out.max_depth = request.graph_limits.max_ordinary_depth;
  out.max_active = request.graph_limits.max_active_streams;
  out.max_policy_edits = request.max_policy_steps;
  out.max_recipe_edits = 2;
  // The v2 feature schema requires a nonzero candidate denominator. Pure
  // online inference has no candidates, so (0,1) is the explicit absent
  // sentinel and current_objective remains nullopt (encoded as -1).
  out.candidate_used = 0;
  out.candidate_limit = 1;
  out.action_count = online_action_count;
  out.current_objective.reset();
  for (const auto id : session.open_stream_ids()) {
    const auto& flow = session.active_streams().at(id);
    const auto& value = meta.at(id);
    out.flows.push_back(
        {flow_view(flow, value.view), value.depth, value.parent_action});
  }
  return out;
}

std::array<double, features::observation_width> make_observation(
    const std::array<double, features::task_width>& task,
    const std::array<double, features::state_width>& state,
    const std::array<double, 3>& weights) {
  std::array<double, features::observation_width> out{};
  std::copy(task.begin(), task.end(), out.begin());
  std::copy(state.begin(), state.end(), out.begin() + features::task_width);
  std::copy(weights.begin(), weights.end(),
            out.begin() + features::task_width + features::state_width);
  return out;
}

std::vector<std::string> output_views(
    const ActionSpec& spec, const std::vector<graph::StreamId>& inputs,
    const std::map<graph::StreamId, Meta>& old_meta, std::size_t count) {
  std::vector<std::string> views;
  if (spec.category == ActionCategory::SplitSegments) {
    views.assign(count, old_meta.at(inputs.front()).view);
  } else if (spec.category == ActionCategory::Concat) {
    std::set<std::string> selected;
    for (const auto id : inputs) selected.insert(old_meta.at(id).view);
    views.assign(count, selected.size() == 1 ? *selected.begin() : "bytes");
  } else if (spec.runtime_action ==
             "op.xor.previous_word.semantic_ports.cpp_v1") {
    views = {"uint64", "xor_residual64"};
  } else if (spec.runtime_action ==
             "op.outlier_separation.high_bits.uint32.cpp_v1") {
    views.assign(count, "uint32");
  } else {
    views.assign(count, "bytes");
  }
  require(views.size() == count,
          "native stage output arity changed from catalog semantics");
  return views;
}

PacketKind output_packet_kind(const ActionSpec& spec) {
  if (spec.category != ActionCategory::Mechanism) return PacketKind::None;
  if (spec.runtime_action == "op.scaling.delta_binary_packed.cpp_v1")
    return PacketKind::DbpScaling;
  if (spec.runtime_action == "op.scaling.exact_decimal.cpp_v2" &&
      spec.io_variant) {
    if (*spec.io_variant == "op.scaling.simple8b.cpp_v1")
      return PacketKind::Simple8bScaling;
    if (*spec.io_variant == "op.scaling.ts_2diff.cpp_v1")
      return PacketKind::Ts2diffScaling;
  }
  return PacketKind::None;
}

graph::Limits effective_limits(const OnlineRequest& request) {
  auto out = request.graph_limits;
  // Internal carrier adapters have no policy action ID and do not consume the
  // policy-visible 64-node/10-depth limits. Reserve one native node/depth for
  // each selected policy node, matching GraphTask's native execution contract.
  out.max_nodes = std::max<std::uint32_t>(
      out.max_nodes, request.graph_limits.max_nodes * 2);
  out.max_ordinary_depth = std::max<std::uint32_t>(
      out.max_ordinary_depth, request.graph_limits.max_ordinary_depth * 2);
  // Closed terminal leaves remain in the complete frame even though they no
  // longer count against the policy-visible open-flow limit. Reserve physical
  // frame capacity for the largest possible retained-leaf growth over the
  // bounded online rollout. Graph/frame validation still enforces this limit.
  const auto retained_capacity =
      std::uint64_t(request.graph_limits.max_active_streams) +
      std::uint64_t(request.graph_limits.max_active_streams - 1) *
          request.max_policy_steps;
  out.max_active_streams = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(1024, retained_capacity));
  return out;
}

policy::RuntimeIdentity runtime_identity() {
  return {online_registry_fingerprint, policy::feature_schema,
          online_action_count, flow_stop_begin, 18, 28};
}

std::string digest_hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    out.push_back(digits[byte >> 4]);
    out.push_back(digits[byte & 15]);
  }
  return out;
}

std::optional<double> normalized(std::uint64_t numerator,
                                 std::uint64_t denominator) {
  if (!denominator) return std::nullopt;
  return double(numerator) / double(denominator);
}

void validate_request(const OnlineRequest& request) {
  require(!request.roots.empty(), "online request requires at least one root");
  require(request.max_policy_steps >= 1 && request.max_policy_steps <= 64,
          "online max_policy_steps must be in [1,64]");
  require(request.graph_limits.max_nodes >= 1 &&
              request.graph_limits.max_nodes <= 64,
          "online policy node limit must be in [1,64]");
  require(request.graph_limits.max_ordinary_depth >= 1 &&
              request.graph_limits.max_ordinary_depth <= 10,
          "online policy ordinary depth must be in [1,10]");
  require(request.graph_limits.max_active_streams >= 1 &&
              request.graph_limits.max_active_streams <= 32,
          "online active data-flow limit must be in [1,32]");
  require(request.roots.size() <= request.graph_limits.max_active_streams,
          "online root count exceeds policy open-flow limit");
  double weight_sum = 0;
  for (const auto weight : request.weights) {
    require(std::isfinite(weight) && weight >= 0,
            "online weights must be finite and nonnegative");
    weight_sum += weight;
  }
  require(std::abs(weight_sum - 1.0) <= 1e-12,
          "online weights must sum to one");
  for (const auto& root : request.roots)
    require(!root.logical_view.empty(),
            "online roots require explicit logical views");
}

std::vector<std::uint8_t> read_small_file(
    const std::filesystem::path& path, std::uint64_t maximum_bytes) {
  std::ifstream stream(path, std::ios::binary);
  require(bool(stream), "cannot open required file: " + path.string());
  stream.seekg(0, std::ios::end);
  const auto end = stream.tellg();
  require(end >= 0 && static_cast<std::uint64_t>(end) <= maximum_bytes,
          "required file is too large");
  stream.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  if (!bytes.empty())
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  require(bool(stream) || stream.eof(),
          "failed to read required file");
  return bytes;
}

OnlineModelAdmission model_admission(
    const policy::Policy& loaded,
    const std::filesystem::path& model_path,
    const ActionCatalog& catalog,
    const std::optional<std::filesystem::path>& credential_path) {
  OnlineModelAdmission admission;
  const auto model_bytes = read_small_file(
      model_path, policy::maximum_model_bytes);
  admission.model_sha256 = digest_hex(policy::sha256(model_bytes));
  admission.source_checkpoint_sha256 =
      loaded.origin().checkpoint_file_sha256;
  if (!credential_path) return admission;
  const auto migrated_identity = [&](const auto& pairs) {
    return std::any_of(pairs.begin(), pairs.end(), [](const auto& row) {
      return row.first == "online_control_migration";
    });
  };
  require(!migrated_identity(loaded.identity().data_fingerprints) &&
              !migrated_identity(loaded.identity().root_fingerprints),
          "diagnostic online-control migration cannot be admitted as matched training");
  const auto semantic_fingerprint_count = static_cast<std::size_t>(
      std::count_if(loaded.identity().data_fingerprints.begin(),
                    loaded.identity().data_fingerprints.end(),
                    [](const auto& row) {
                      return row.first == "semantic_mask_contract" &&
                          row.second == online_semantic_mask_contract_sha256;
                    }));
  require(semantic_fingerprint_count == 1,
          "matched model data_fingerprints must bind semantic_mask_contract");
  const auto bytes = read_small_file(*credential_path, 1U << 20);
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(bytes.begin(), bytes.end());
  } catch (const std::exception& error) {
    throw std::invalid_argument(
        "invalid matched-training credential JSON: " +
        std::string(error.what()));
  }
  const auto exact_string = [&](const char* key, const std::string& expected) {
    require(doc.contains(key) && doc.at(key).is_string() &&
                doc.at(key).get<std::string>() == expected,
            std::string("matched-training credential mismatch: ") + key);
  };
  const auto exact_unsigned = [&](const char* key, std::uint32_t expected) {
    require(doc.contains(key) && doc.at(key).is_number_unsigned() &&
                doc.at(key).get<std::uint32_t>() == expected,
            std::string("matched-training credential mismatch: ") + key);
  };
  const auto exact_bool = [&](const char* key, bool expected) {
    require(doc.contains(key) && doc.at(key).is_boolean() &&
                doc.at(key).get<bool>() == expected,
            std::string("matched-training credential mismatch: ") + key);
  };
  exact_string("schema", online_matched_credential_schema);
  exact_string("native_model_sha256", admission.model_sha256);
  exact_string("source_checkpoint_sha256",
               admission.source_checkpoint_sha256);
  exact_string("runtime_schema", online_runtime_schema);
  exact_string("semantic_mask_contract", online_semantic_mask_contract);
  exact_string("training_session_schema", online_training_session_schema);
  exact_string("training_contract", online_matched_training_contract);
  exact_string("control_schema", online_control_schema);
  exact_string("registry_fingerprint", online_registry_fingerprint);
  exact_string("feature_schema", policy::feature_schema);
  exact_string("catalog_sha256", catalog.source_sha256());
  require(catalog.experiment_scope().has_value(),
          "matched runtime has no experiment profile");
  exact_string("experiment_profile_sha256",
               catalog.experiment_scope()->profile_sha256);
  exact_unsigned("action_count", online_action_count);
  exact_unsigned("flow_stop_begin", flow_stop_begin);
  exact_unsigned("flow_stop_count", flow_stop_count);
  exact_unsigned("max_policy_steps", online_matched_max_policy_steps);
  exact_unsigned("max_nodes", online_matched_max_nodes);
  exact_unsigned("max_active_streams", online_matched_max_active_streams);
  exact_unsigned("max_ordinary_depth",
                 online_matched_max_ordinary_depth);
  exact_bool("starts_from_raw_roots", true);
  exact_bool("per_flow_stop", true);
  exact_bool("online_control_migration", false);
  exact_unsigned("candidate_evaluations", 0);
  exact_unsigned("replayed_prefix_stages", 0);
  const auto valid_digest = [&](const char* key) {
    require(doc.contains(key) && doc.at(key).is_string(),
            std::string("missing matched-training receipt digest: ") + key);
    const auto value = doc.at(key).get<std::string>();
    require(value.size() == 64 &&
                std::all_of(value.begin(), value.end(), [](char c) {
                  return (c >= '0' && c <= '9') ||
                         (c >= 'a' && c <= 'f');
                }),
            std::string("invalid matched-training receipt digest: ") + key);
  };
  valid_digest("training_run_receipt_sha256");
  valid_digest("checkpoint_export_receipt_sha256");
  admission.matched_training = true;
  admission.diagnostic_only = false;
  admission.matched_max_policy_steps = online_matched_max_policy_steps;
  admission.credential_sha256 = digest_hex(policy::sha256(bytes));
  return admission;
}

}  // namespace

std::optional<double> OnlineResult::compression_ratio() const {
  return normalized(frame.size(), original_numeric_bytes);
}

std::optional<double> OnlineResult::compression_ns_per_byte() const {
  if (!complete_online_timing) return std::nullopt;
  return normalized(timing.total_ns, original_numeric_bytes);
}

struct OnlineSession::Impl {
  const ActionCatalog* catalog;
  const std::array<std::vector<std::uint32_t>, 32>* actions_by_slot;
  const std::vector<std::uint32_t>* concat_actions;
  OnlineRequest request;
  const graph::StageDispatcher* dispatcher;
  OnlineTimingMode timing_mode;
  Clock::time_point total_begin;
  graph::Limits limits;
  graph::GraphSession session;
  std::map<graph::StreamId, Meta> meta;
  std::map<graph::StreamId, FlowSemanticCache> semantic_cache;
  std::array<double, features::task_width> task_features{};
  std::uint32_t policy_graph_nodes = 0;
  OnlineResult output;
  std::optional<OnlineObservation> cached_observation;
  bool terminal = false;
  bool result_taken = false;

  Impl(const ActionCatalog& catalog_value,
       const std::array<std::vector<std::uint32_t>, 32>& slot_index,
       const std::vector<std::uint32_t>& concat_index,
       const OnlineRequest& request_value,
       const graph::StageDispatcher& dispatcher_value,
       OnlineTimingMode mode, Clock::time_point begin)
      : catalog(&catalog_value),
        actions_by_slot(&slot_index),
        concat_actions(&concat_index),
        // Keep only the policy controls after construction.  The root payloads
        // are owned by GraphSession, so retaining them here would deep-copy the
        // complete block a second time inside the same online timing window.
        request{{}, request_value.weights, request_value.graph_limits,
                request_value.max_policy_steps},
        dispatcher(&dispatcher_value),
        timing_mode(mode),
        total_begin(begin),
        limits(effective_limits(request)),
        session(request_value.roots, dispatcher_value, limits,
                online_registry_fingerprint) {
    output.effective_graph_limits = limits;
    output.original_numeric_bytes = session.original_numeric_bytes();
    std::uint32_t root_id = 0;
    for (const auto& root : request_value.roots)
      meta.emplace(root_id++,
                   Meta{0, root.logical_view, stop_action, PacketKind::None});
    if (!output.original_numeric_bytes) {
      force_close_open_flows();
      finalize("no_original_numeric_payload_n_a");
      return;
    }
    std::vector<features::FlowView> root_views;
    root_views.reserve(request_value.roots.size());
    for (const auto& root : request_value.roots)
      root_views.push_back(flow_view(root.stream, root.logical_view));
    const auto feature_begin = Clock::now();
    task_features = features::task(root_views);
    output.timing.feature_ns += elapsed(feature_begin, Clock::now());
  }

  void force_close_open_flows() {
    const auto ids = session.open_stream_ids();
    for (std::size_t remaining = ids.size(); remaining > 0; --remaining) {
      const auto slot = static_cast<std::uint32_t>(remaining - 1);
      const auto id = ids[slot];
      output.recipe_action_ids.push_back(flow_stop_begin + slot);
      session.close_flow(id);
      meta.erase(id);
      semantic_cache.erase(id);
      ++output.counters.closed_flow_count;
      ++output.counters.forced_flow_stops;
    }
    cached_observation.reset();
  }

  void finalize(std::string reason) {
    if (terminal) return;
    require(session.open_stream_count() == 0,
            "online frame cannot finish while a data flow remains open");
    graph::Step stop;
    stop.kind = graph::StepKind::Stop;
    stop.policy_action_id = -1;
    output.native_steps.push_back(stop);
    const auto framing_begin = Clock::now();
    if (timing_mode == OnlineTimingMode::ContinuousOuterClock)
      output.timing.recipe_ready_ns = elapsed(total_begin, framing_begin);
    output.frame = session.stop();
    output.timing.framing_ns += elapsed(framing_begin, Clock::now());
    output.termination = std::move(reason);
    require(output.counters.candidate_evaluations == 0 &&
                output.counters.replayed_prefix_stages == 0,
            "online runtime executed forbidden search/replay work");
    if (timing_mode == OnlineTimingMode::ContinuousOuterClock)
      output.timing.total_ns = elapsed(total_begin, Clock::now());
    terminal = true;
    cached_observation.reset();
  }

  const OnlineObservation& observe() {
    require(!terminal, "cannot observe a terminal online session");
    if (cached_observation) return *cached_observation;
    OnlineObservation observation;
    const auto feature_begin = Clock::now();
    const auto state_features = features::state(feature_state(
        request, session, meta, policy_graph_nodes,
        output.counters.policy_calls, output.original_numeric_bytes));
    observation.observation = make_observation(
        task_features, state_features, request.weights);
    output.timing.feature_ns += elapsed(feature_begin, Clock::now());
    const auto mask_begin = Clock::now();
    observation.legal_action_ids = legal_actions(
        *catalog, request, session, meta, limits, *actions_by_slot,
        *concat_actions, semantic_cache, output.counters,
        policy_graph_nodes);
    output.timing.mask_ns += elapsed(mask_begin, Clock::now());
    observation.policy_calls = output.counters.policy_calls;
    observation.policy_graph_nodes = policy_graph_nodes;
    observation.open_flow_count = static_cast<std::uint32_t>(
        session.open_stream_count());
    observation.semantic_mask_flow_scans =
        output.counters.semantic_mask_flow_scans;
    observation.semantic_mask_value_visits =
        output.counters.semantic_mask_value_visits;
    cached_observation = std::move(observation);
    return *cached_observation;
  }

  void step(std::uint32_t action) {
    require(!terminal, "cannot step a terminal online session");
    const auto& current = observe();
    require(std::binary_search(current.legal_action_ids.begin(),
                               current.legal_action_ids.end(), action),
            "forced/policy action is outside the shared online legal set: " +
                std::to_string(action));
    ++output.counters.policy_calls;
    ++output.counters.selected_policy_actions;
    output.recipe_action_ids.push_back(action);
    cached_observation.reset();
    if (action >= flow_stop_begin) {
      require(action < online_action_count,
              "online policy returned an unknown flow-stop action");
      const auto slot = action - flow_stop_begin;
      const auto ids = session.open_stream_ids();
      require(slot < ids.size(),
              "online policy returned an inactive flow-stop slot");
      session.close_flow(ids[slot]);
      meta.erase(ids[slot]);
      semantic_cache.erase(ids[slot]);
      ++output.counters.closed_flow_count;
    } else {
      require(action > stop_action && action < action_count,
              "legacy global STOP or noncatalog action reached online execution");
      const auto& spec = catalog->at(action);
      const auto ids = active_ids(session);
      const auto target_id = ids.at(*spec.target_slot);
      std::vector<graph::StreamId> original_inputs{target_id};
      std::vector<graph::StreamId> step_inputs{target_id};
      if (spec.category == ActionCategory::Concat) {
        original_inputs.push_back(ids.at(*spec.other_slot));
        step_inputs.push_back(ids.at(*spec.other_slot));
      }
      const auto old_meta = meta;
      const auto adapter = adapter_for(*catalog, old_meta.at(target_id), spec);
      require(adapter.legal,
              "selected action lost its internal-adapter precondition");
      const auto stage_begin = Clock::now();
      try {
        if (adapter.adapter) {
          graph::Step adapter_step;
          adapter_step.kind = graph::StepKind::Stage;
          adapter_step.inputs = {target_id};
          adapter_step.stage = {adapter.adapter->action, "bytes", {}};
          adapter_step.policy_action_id = -1;
          ++output.counters.forward_stage_calls;
          const auto outputs = session.apply(adapter_step);
          require(outputs.size() == 1,
                  "internal adapter must produce one data flow");
          output.native_steps.push_back(adapter_step);
          step_inputs = {outputs.front()};
        }
        graph::Step graph_step;
        graph_step.policy_action_id = action;
        graph_step.inputs = step_inputs;
        if (spec.category == ActionCategory::SplitSegments) {
          graph_step.kind = graph::StepKind::SplitSegments;
          graph_step.segment_sizes = segment_sizes(
              spec, session.active_streams().at(step_inputs.front()));
        } else if (spec.category == ActionCategory::Concat) {
          graph_step.kind = graph::StepKind::Concat;
        } else {
          graph_step.kind = graph::StepKind::Stage;
          graph_step.stage.action = spec.runtime_action;
          graph_step.stage.profile = stage_profile(spec);
          graph_step.stage.parameters = parameters(
              spec, session.active_streams().at(step_inputs.front()));
          ++output.counters.forward_stage_calls;
        }
        const auto outputs = session.apply(graph_step);
        require(session.open_stream_count() <=
                    request.graph_limits.max_active_streams,
                "selected action exceeded policy open-flow limit");
        output.native_steps.push_back(graph_step);
        std::map<graph::StreamId, Meta> next;
        for (const auto& [id, value] : old_meta)
          if (!contains(original_inputs, id)) next.emplace(id, value);
        const auto parent_depth = std::accumulate(
            original_inputs.begin(), original_inputs.end(), 0U,
            [&](std::uint32_t value, graph::StreamId id) {
              return std::max(value, old_meta.at(id).depth);
            });
        const auto depth = parent_depth + unsigned(spec.ordinary_operator);
        const auto views = output_views(spec, original_inputs, old_meta,
                                        outputs.size());
        for (const auto id : original_inputs) semantic_cache.erase(id);
        const auto packet_kind = output_packet_kind(spec);
        for (std::size_t i = 0; i < outputs.size(); ++i) {
          require(session.flow_info(outputs[i]).logical_view == views[i],
                  "online policy view disagrees with GraphSession stage view");
          next.emplace(outputs[i],
                       Meta{depth, views[i], action, packet_kind});
        }
        meta = std::move(next);
      } catch (const std::exception& error) {
        throw std::runtime_error(
            "ONLINE_SELECTED_ACTION_FAILED action_id=" +
            std::to_string(action) + " key=" + spec.key + ": " +
            error.what());
      }
      output.timing.stage_ns += elapsed(stage_begin, Clock::now());
      ++policy_graph_nodes;
    }
    if (session.open_stream_count() == 0) {
      finalize("all_flows_closed");
    } else if (output.counters.policy_calls >= request.max_policy_steps) {
      force_close_open_flows();
      finalize("budget_closed");
    }
  }
};

OnlineSession::OnlineSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
OnlineSession::~OnlineSession() = default;
OnlineSession::OnlineSession(OnlineSession&&) noexcept = default;
OnlineSession& OnlineSession::operator=(OnlineSession&&) noexcept = default;

bool OnlineSession::terminal() const noexcept {
  return impl_->terminal;
}

const OnlineObservation& OnlineSession::observe() {
  return impl_->observe();
}

void OnlineSession::step(std::uint32_t action_id) {
  impl_->step(action_id);
}

void OnlineSession::finish(std::string termination) {
  if (impl_->terminal) return;
  impl_->force_close_open_flows();
  impl_->finalize(std::move(termination));
}

const OnlineResult& OnlineSession::result() const {
  require(impl_->terminal, "online result requested before terminal state");
  require(!impl_->result_taken, "online result was already moved");
  return impl_->output;
}

OnlineResult OnlineSession::take_result() {
  require(impl_->terminal, "online result requested before terminal state");
  require(!impl_->result_taken, "online result was already moved");
  impl_->result_taken = true;
  return std::move(impl_->output);
}

void OnlineSession::add_policy_time(std::uint64_t value) {
  require(!impl_->terminal, "cannot record policy time on terminal session");
  impl_->output.timing.policy_ns += value;
}

OnlineRuntime OnlineRuntime::load(
    const std::filesystem::path& catalog_path,
    const std::filesystem::path& experiment_profile) {
  auto catalog = ActionCatalog::load(catalog_path);
  catalog.load_experiment_profile(experiment_profile);
  const std::string_view online_control_descriptor =
      online_control_descriptor_canonical_json;
  const auto descriptor_digest = policy::sha256({
      reinterpret_cast<const std::uint8_t*>(online_control_descriptor.data()),
      online_control_descriptor.size()});
  require(digest_hex(descriptor_digest) == online_registry_fingerprint,
          "online flow-stop descriptor fingerprint mismatch");
  require(catalog.enabled(stop_action),
          "base experiment scope unexpectedly excluded legacy STOP identity");
  for (const auto& spec : catalog.actions()) {
    if (catalog.enabled(spec.action_id) && graph_category(spec.category) &&
        !catalog.support(spec.action_id).executable) {
      throw std::invalid_argument(
          "ONLINE_ENABLED_NATIVE_ACTION_UNSUPPORTED action_id=" +
          std::to_string(spec.action_id) + " key=" + spec.key + " reason=" +
          catalog.support(spec.action_id).reason);
    }
  }
  std::array<std::vector<std::uint32_t>, 32> actions_by_slot;
  std::vector<std::uint32_t> concat_actions;
  std::size_t indexed_graph_actions = 0;
  for (const auto& spec : catalog.actions()) {
    if (!catalog.enabled(spec.action_id) || !graph_category(spec.category))
      continue;
    require(spec.target_slot && *spec.target_slot < actions_by_slot.size(),
            "enabled online graph action has no valid target slot");
    if (spec.category == ActionCategory::Concat) {
      concat_actions.push_back(spec.action_id);
    } else {
      actions_by_slot[*spec.target_slot].push_back(spec.action_id);
    }
    ++indexed_graph_actions;
  }
  const auto expected_graph_actions = static_cast<std::size_t>(std::count_if(
      catalog.actions().begin(), catalog.actions().end(), [&](const auto& spec) {
        return catalog.enabled(spec.action_id) && graph_category(spec.category);
      }));
  require(indexed_graph_actions == expected_graph_actions,
          "online catalog index omitted an enabled graph action");
  for (auto& values : actions_by_slot) std::sort(values.begin(), values.end());
  std::sort(concat_actions.begin(), concat_actions.end());
  return OnlineRuntime(std::move(catalog), std::move(actions_by_slot),
                       std::move(concat_actions));
}

OnlineSession OnlineRuntime::start(
    const OnlineRequest& request, const graph::StageDispatcher& dispatcher,
    OnlineTimingMode timing_mode) const {
  const auto begin = Clock::now();
  validate_request(request);
  return OnlineSession(std::make_unique<OnlineSession::Impl>(
      catalog_, actions_by_slot_, concat_actions_, request, dispatcher,
      timing_mode, begin));
}

OnlineReplayResult OnlineRuntime::replay_forced(
    const OnlineRequest& request,
    std::span<const std::uint32_t> forced_actions,
    const graph::StageDispatcher& dispatcher,
    bool capture_transitions) const {
  auto session = start(request, dispatcher,
                       OnlineTimingMode::ContinuousOuterClock);
  OnlineReplayResult replay;
  for (const auto action : forced_actions) {
    require(!session.terminal(),
            "forced trajectory contains an action after terminal state");
    std::optional<OnlineTransition> transition;
    if (capture_transitions) {
      transition.emplace();
      transition->state = session.observe();
      transition->action_id = action;
    }
    session.step(action);
    if (transition) replay.transitions.push_back(std::move(*transition));
  }
  if (!session.terminal()) session.finish();
  replay.result = session.take_result();
  replay.reward_timing_eligible = false;
  return replay;
}

OnlinePipeline OnlinePipeline::load(
    const std::filesystem::path& model,
    const std::filesystem::path& catalog_path,
    const std::filesystem::path& experiment_profile,
    std::optional<std::filesystem::path> matched_credential) {
  auto runtime = OnlineRuntime::load(catalog_path, experiment_profile);
  auto loaded = policy::Policy::load(model, runtime_identity());
  auto admission = model_admission(
      loaded, model, runtime.catalog(), matched_credential);
  return OnlinePipeline(std::move(loaded), std::move(runtime),
                        std::move(admission));
}

OnlineResult OnlinePipeline::compress(
    const OnlineRequest& request,
    const graph::StageDispatcher& dispatcher) const {
  require(!admission_.matched_training ||
              request.max_policy_steps == admission_.matched_max_policy_steps,
          "matched model max_policy_steps differs from its training credential");
  require(!admission_.matched_training ||
              (request.graph_limits.max_nodes == online_matched_max_nodes &&
               request.graph_limits.max_active_streams ==
                   online_matched_max_active_streams &&
               request.graph_limits.max_ordinary_depth ==
                   online_matched_max_ordinary_depth),
          "matched model graph limits differ from its training credential");
  auto session = runtime_.start(
      request, dispatcher, OnlineTimingMode::ContinuousOuterClock);
  while (!session.terminal()) {
    const auto& state = session.observe();
    const auto policy_begin = Clock::now();
    const auto action = policy_.greedy(
        state.observation, state.legal_action_ids);
    session.add_policy_time(elapsed(policy_begin, Clock::now()));
    session.step(action);
  }
  auto output = session.take_result();
  output.complete_online_timing = true;
  output.matched_training = admission_.matched_training;
  output.diagnostic_only = admission_.diagnostic_only;
  output.matched_max_policy_steps = admission_.matched_max_policy_steps;
  output.model_sha256 = admission_.model_sha256;
  output.source_checkpoint_sha256 = admission_.source_checkpoint_sha256;
  output.matched_credential_sha256 = admission_.credential_sha256;
  return output;
}

OnlineReplayResult OnlinePipeline::replay_forced(
    const OnlineRequest& request,
    std::span<const std::uint32_t> forced_actions,
    const graph::StageDispatcher& dispatcher,
    bool capture_transitions) const {
  require(!admission_.matched_training ||
              request.max_policy_steps == admission_.matched_max_policy_steps,
          "matched model max_policy_steps differs from its training credential");
  require(!admission_.matched_training ||
              (request.graph_limits.max_nodes == online_matched_max_nodes &&
               request.graph_limits.max_active_streams ==
                   online_matched_max_active_streams &&
               request.graph_limits.max_ordinary_depth ==
                   online_matched_max_ordinary_depth),
          "matched model graph limits differ from its training credential");
  auto session = runtime_.start(
      request, dispatcher, OnlineTimingMode::ContinuousOuterClock);
  OnlineReplayResult replay;
  for (const auto action : forced_actions) {
    require(!session.terminal(),
            "forced trajectory contains an action after terminal state");
    const auto& state = session.observe();
    const auto policy_begin = Clock::now();
    const auto greedy_action = policy_.greedy(
        state.observation, state.legal_action_ids);
    session.add_policy_time(elapsed(policy_begin, Clock::now()));
    std::optional<OnlineTransition> transition;
    if (capture_transitions) {
      transition.emplace();
      transition->state = state;
      transition->greedy_action_id = greedy_action;
      transition->action_id = action;
    }
    session.step(action);
    if (transition) replay.transitions.push_back(std::move(*transition));
  }
  const bool external_finish_required = !session.terminal();
  if (external_finish_required) session.finish();
  replay.result = session.take_result();
  replay.result.complete_online_timing =
      !capture_transitions && !external_finish_required;
  replay.result.matched_training = admission_.matched_training;
  replay.result.diagnostic_only = admission_.diagnostic_only;
  replay.result.matched_max_policy_steps =
      admission_.matched_max_policy_steps;
  replay.result.model_sha256 = admission_.model_sha256;
  replay.result.source_checkpoint_sha256 =
      admission_.source_checkpoint_sha256;
  replay.result.matched_credential_sha256 = admission_.credential_sha256;
  replay.reward_timing_eligible =
      !capture_transitions && !external_finish_required;
  return replay;
}

}  // namespace wc::pipeline_v1
