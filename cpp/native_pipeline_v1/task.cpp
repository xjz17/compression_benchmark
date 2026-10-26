#include "task.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>

namespace wc::pipeline_v1 {
namespace {
namespace graph = native_graph_v1;
void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument(message);
}
bool contains(const std::vector<std::string>& values, const std::string& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}
bool contains(const std::vector<std::uint32_t>& values, std::uint32_t value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}
std::uint64_t median(std::vector<std::uint64_t> values) {
  require(!values.empty(), "cannot take median of no timing samples");
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2) return values[middle];
  return values[middle - 1] / 2 + values[middle] / 2 +
         ((values[middle - 1] & 1) && (values[middle] & 1));
}
bool graph_category(ActionCategory value) {
  return value == ActionCategory::Mechanism || value == ActionCategory::SplitSegments ||
         value == ActionCategory::Concat;
}
bool same_parameter_tuple(const ActionSpec& a, const ActionSpec& b) {
  return a.input_view == b.input_view && a.parameter_mode == b.parameter_mode &&
         a.parameter_value == b.parameter_value && a.parameter_items == b.parameter_items &&
         a.max_point == b.max_point && a.io_variant == b.io_variant;
}
std::int64_t little_i64(std::span<const std::uint8_t> bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[offset + i]) << (8 * i);
  return std::bit_cast<std::int64_t>(value);
}
bool exactly_representable_as_float64(std::int64_t value) {
  // IEEE binary64 has 53 significant bits.  Check the discarded low bits in
  // integer space so INT64_MAX never takes an out-of-range double-to-int cast;
  // that cast produced different legality decisions across Clang and GCC.
  const auto magnitude = value < 0
      ? std::uint64_t{0} - static_cast<std::uint64_t>(value)
      : static_cast<std::uint64_t>(value);
  const auto width = std::bit_width(magnitude);
  if (width <= 53) return true;
  const auto discarded = width - 53;
  return (magnitude & ((std::uint64_t{1} << discarded) - 1)) == 0;
}
std::string physical_kind(graph::Kind kind) {
  if (kind == graph::Kind::Serial) return "serial";
  if (kind == graph::Kind::Numeric) return "numeric";
  return "struct";
}
std::map<std::string, std::string> parameters(const ActionSpec& spec, const graph::Stream& stream) {
  auto result = spec.parameter_items;
  if (spec.io_variant && *spec.io_variant != spec.runtime_action)
    result["io_variant"] = *spec.io_variant;
  if (spec.max_point) result["max_point"] = std::to_string(*spec.max_point);
  // A010 predates the fixed max_point action expansion used by A006-A008.
  // Its frozen action has no selector field, so the native execution mapping
  // uses the implementation's full audited decimal bound.  Legality probes
  // the current values and rejects the action unless this exact request is
  // bit-exact and independently reversible.
  if (spec.runtime_action == "op.scaling.legacy_exact_decimal.cpp_v1" &&
      spec.input_view == "float64")
    result["max_point"] = "18";
  if (spec.parameter_mode == "none") return result;
  if (spec.parameter_mode == "fp32") result["format"] = "fp32";
  else if (spec.parameter_mode == "top8") {
    result["mode"] = "top8"; result["width"] = std::to_string(stream.width);
  } else if (spec.parameter_mode == "bits") {
    result["bits"] = std::to_string(*spec.parameter_value);
  } else if (spec.parameter_mode == "width") {
    result["width"] = std::to_string(*spec.parameter_value);
  } else if (spec.parameter_mode == "stream_width") {
    result["mode"] = "transpose"; result["width"] = std::to_string(stream.width);
  } else if (spec.parameter_mode == "allow_large_memory") {
    result["allow_large_memory"] = "true";
  } else if (spec.parameter_mode != "half" && spec.parameter_mode != "quarters") {
    throw std::invalid_argument("unknown fixed parameter mode: " + spec.parameter_mode);
  }
  return result;
}
std::string stage_profile(const ActionSpec& spec) {
  if (spec.runtime_action == "op.delta.ts_2diff.cpp_v1" && spec.input_view == "bytes")
    return "scaled_packet";
  return (spec.input_view == "int64" || spec.input_view == "float64")
             ? spec.input_view : "bytes";
}
bool packet_precondition_requires_probe(const ActionSpec& spec) {
  return (spec.runtime_action == "op.scaling.legacy_exact_decimal.cpp_v1" &&
          spec.input_view == "float64") ||
      ((spec.runtime_action == "op.eraser.elf.cpp_v1" ||
        spec.runtime_action == "op.eraser.camel.cpp_v1" ||
        spec.runtime_action == "op.eraser.pongo.cpp_v1") && spec.input_view == "float64") ||
      (spec.runtime_action == "op.bitpack_int.max_value_width.cpp_v1" &&
       spec.input_view == "bytes") ||
      (spec.input_view == "bytes" &&
      (spec.runtime_action == "op.bitpack_int.fastpfor_pfor.cpp_v1" ||
       spec.runtime_action == "op.bitpack_int.simple8b.cpp_v1" ||
       spec.runtime_action == "op.delta.stream_vbyte.cpp_v1" ||
       spec.runtime_action == "op.delta.ts_2diff.cpp_v1"));
}
std::vector<std::uint64_t> segment_sizes(const ActionSpec& spec, const graph::Stream& stream) {
  require(stream.width > 0 && stream.data.size() % stream.width == 0,
          "segmentation input is not element aligned");
  const std::uint64_t count = stream.data.size() / stream.width;
  if (spec.parameter_mode == "half") return {count / 2, count - count / 2};
  require(spec.parameter_mode == "quarters", "invalid segmentation parameter mode");
  const auto first = std::max<std::uint64_t>(1, count / 4);
  const auto second = std::max<std::uint64_t>(1, (count - first) / 3);
  const auto third = std::max<std::uint64_t>(1, (count - first - second) / 2);
  if (first + second + third >= count) return {count / 2, count - count / 2};
  return {first, second, third, count - first - second - third};
}
void verify_decoded(const graph::Decoded& decoded, const TaskRequest& request) {
  require(decoded.roots.size() == request.roots.size(), "cold decode root count mismatch");
  for (std::size_t i = 0; i < request.roots.size(); ++i)
    require(decoded.roots[i].stream == request.roots[i].stream &&
            decoded.roots[i].logical_view == request.roots[i].logical_view &&
            decoded.roots[i].original_numeric_payload == request.roots[i].original_numeric_payload,
            "cold decode root bytes/type/view mismatch");
}
}  // namespace

struct GraphTask::Replay {
  struct Meta { std::uint32_t depth = 0; std::string view; std::uint32_t parent = stop_action; };
  graph::GraphSession session;
  graph::Recipe native_recipe;
  std::map<graph::StreamId, Meta> meta;
  // Count policy-visible graph nodes separately from native adapter nodes.
  // The frozen policy's 64-node limit is expressed in catalog actions, while
  // GraphSession also records explicit internal carrier adapters.
  std::uint32_t policy_nodes = 0;
  explicit Replay(const TaskRequest& request, const graph::StageDispatcher& dispatcher,
                  graph::Limits limits)
      : session(request.roots, dispatcher, limits, registry_fingerprint) {
    native_recipe.policy_registry = registry_fingerprint;
    std::uint32_t id = 0;
    for (const auto& root : request.roots) meta.emplace(id++, Meta{0, root.logical_view, stop_action});
  }
};

GraphTask::GraphTask(const ActionCatalog& catalog, const graph::StageDispatcher& dispatcher,
                     TaskRequest request)
    : catalog_(catalog), dispatcher_(dispatcher), request_(std::move(request)) {
  require(!request_.task_id.empty(), "task_id cannot be empty");
  require(!request_.roots.empty() && request_.roots.size() <= request_.config.max_active_streams,
          "invalid task root count");
  require(request_.config.max_ordinary_depth >= 1 && request_.config.max_ordinary_depth <= 10,
          "max_ordinary_depth must be in [1,10]");
  require(request_.config.max_graph_nodes >= 1 && request_.config.max_graph_nodes <= 64,
          "max_graph_nodes must be in [1,64]");
  require(request_.config.max_active_streams >= 1 && request_.config.max_active_streams <= 32,
          "max_active_streams must be in [1,32]");
  require(request_.config.max_policy_edits >= 1 &&
          request_.config.max_policy_edits <= request_.config.max_graph_nodes,
          "max_policy_edits must fit graph-node limit");
  require(request_.config.timing_warmups <= 31 && request_.config.timing_repetitions >= 1 &&
          request_.config.timing_repetitions <= 31, "invalid candidate timing repetitions");
  require(request_.max_recipe_edits >= 1 && request_.max_recipe_edits <= 2,
          "continuation recipe edit limit must be in [1,2]");
  const auto sum = request_.weights[0] + request_.weights[1] + request_.weights[2];
  require(std::all_of(request_.weights.begin(), request_.weights.end(),
                      [](double v) { return std::isfinite(v) && v >= 0; }) &&
          std::abs(sum - 1.0) <= 1e-12, "weights must be finite, nonnegative and sum to one");
  require(std::isfinite(request_.compression_reference_ns_per_byte) &&
          request_.compression_reference_ns_per_byte > 0 &&
          std::isfinite(request_.decompression_reference_ns_per_byte) &&
          request_.decompression_reference_ns_per_byte > 0,
          "time references must be finite and positive");
  if (request_.require_complete_native_catalog)
    require(catalog_.complete_native_graph_space(),
            "FORMAL_NATIVE_CATALOG_INCOMPLETE: unsupported catalog actions remain");
  require(!(request_.continuation_edit_mode && !request_.starting_recipe.empty()),
          "continuation task cannot use starting_recipe_action_ids");
  require(!(!request_.continuation_edit_mode && !request_.previous_recipe.empty()),
          "previous_recipe_action_ids is reserved for continuation edit mode");
  if (!request_.starting_recipe.empty()) {
    require(request_.block_index == 0, "starting_recipe_action_ids is first-block only");
    require(request_.starting_recipe.back() == stop_action,
            "starting_recipe_action_ids must include one terminal STOP");
    request_.starting_recipe.pop_back();
    require(std::find(request_.starting_recipe.begin(), request_.starting_recipe.end(), stop_action) ==
            request_.starting_recipe.end(), "STOP may only terminate a starting recipe");
  }
  // Native runtime steps include explicit internal adapters. Give those steps
  // headroom while the policy-visible 64-node/10-depth contract is enforced by
  // this task's independent metadata.
  request_.config.graph_limits.max_nodes = std::max<std::uint32_t>(
      request_.config.graph_limits.max_nodes, request_.config.max_graph_nodes * 2);
  request_.config.graph_limits.max_ordinary_depth = std::max<std::uint32_t>(
      request_.config.graph_limits.max_ordinary_depth, request_.config.max_ordinary_depth * 2);
  graph::GraphSession roots(request_.roots, dispatcher_, request_.config.graph_limits,
                            registry_fingerprint);
  const auto root_numeric = roots.original_numeric_bytes();
  original_numeric_bytes_ = request_.original_numeric_bytes.value_or(root_numeric);
  require(original_numeric_bytes_ > 0 && original_numeric_bytes_ <= root_numeric,
          "original_numeric_bytes must be positive and no larger than numeric root bytes");
  for (const auto action : request_.previous_recipe) {
    const auto& spec = catalog_.at(action);
    require(action != stop_action && graph_category(spec.category),
            "previous recipe contains STOP, edit token, or unknown action");
    require(catalog_.enabled(action),
            "previous recipe contains an action excluded by the experiment scope");
    require(catalog_.support(action).executable,
            "previous recipe contains an unsupported native action");
  }
  for (const auto action : request_.starting_recipe) {
    const auto& spec = catalog_.at(action);
    require(action != stop_action && graph_category(spec.category),
            "starting recipe contains STOP, edit token, or unknown action");
    require(catalog_.enabled(action),
            "starting recipe contains an action excluded by the experiment scope");
    require(catalog_.support(action).executable,
            "starting recipe contains an unsupported native action");
  }
  require(request_.previous_recipe.size() <= request_.config.max_graph_nodes &&
          request_.starting_recipe.size() <= request_.config.max_graph_nodes,
          "starting/previous recipe exceeds graph-node limit");
  initial_.recipe = request_.continuation_edit_mode ? request_.previous_recipe : request_.starting_recipe;
  initial_.continuation_edit_mode = request_.continuation_edit_mode;
  initial_ = recipe_state(std::move(initial_.recipe), &initial_);
  initial_.replayed_nodes = static_cast<std::uint32_t>(initial_.recipe.size());
  candidate_budget_ = request_.block_index == 0 ? 64 : 16;
}

TaskState GraphTask::starting_state(std::vector<std::uint32_t> recipe) const {
  require(!request_.continuation_edit_mode && request_.block_index == 0,
          "solution-library starts are only valid for first-block tasks");
  if (!recipe.empty() && recipe.back() == stop_action) recipe.pop_back();
  require(std::find(recipe.begin(), recipe.end(), stop_action) == recipe.end(),
          "solution-library STOP must be terminal");
  TaskState state = recipe_state(std::move(recipe));
  state.replayed_nodes = static_cast<std::uint32_t>(state.recipe.size());
  return state;
}

TaskRequest GraphTask::fresh_request() const {
  auto result = request_;
  if (!result.starting_recipe.empty()) result.starting_recipe.push_back(stop_action);
  return result;
}

std::uint32_t GraphTask::max_steps() const noexcept {
  return request_.continuation_edit_mode ? request_.max_recipe_edits * 3 + 1
                                         : request_.config.max_policy_edits + 1;
}
bool GraphTask::is_terminal_evaluable(const TaskState& state) const {
  verify_state(state); return !state.pending_edit_kind.has_value();
}
std::uint64_t GraphTask::certified_frame_byte_lower_bound(
    const TaskState& state) const {
  verify_state(state);
  // NGF1 has a 16-byte prefix and a 32-byte checksum. The body necessarily
  // contains dispatcher/registry identities, the denominator, every original
  // root descriptor/digest, both node/leaf counts, and at least one final leaf
  // descriptor. Payload bytes are deliberately lower-bounded by zero.
  std::uint64_t bytes = 48;
  bytes += 4 + dispatcher_.identity().size();
  bytes += 4 + std::string_view(registry_fingerprint).size();
  bytes += 8 + 4;
  for (const auto& root : request_.roots)
    bytes += 13 + 1 + 4 + root.logical_view.size() + 32;
  bytes += 4;       // node count
  bytes += 4 + 17;  // leaf count plus one ID/descriptor, with zero payload

  std::size_t unavoidable_policy_nodes = state.recipe.size();
  if (state.continuation_edit_mode) {
    const auto remaining_deletes = request_.max_recipe_edits - state.edit_count;
    unavoidable_policy_nodes = unavoidable_policy_nodes > remaining_deletes
        ? unavoidable_policy_nodes - remaining_deletes : 0;
  }
  // Every policy graph action emits at least one native node. The smallest
  // possible serialized node is 80 bytes: kind/policy ID, empty stage/control
  // fields, one input descriptor, one output descriptor with a one-byte
  // logical view, and a zero-byte metadata length. Real nodes are larger;
  // internal adapters only increase the frame.
  bytes += 80 * unavoidable_policy_nodes;
  return bytes;
}
void GraphTask::verify_state(const TaskState& state) const {
  require(state.continuation_edit_mode == request_.continuation_edit_mode,
          "task state continuation mode changed");
  require(state.recipe.size() <= request_.config.max_graph_nodes,
          "task state recipe exceeds graph-node limit");
  require(state.recipe_valid == !state.recipe_error.has_value(),
          "task state recipe validity/error mismatch");
  require(state.edit_count <= request_.max_recipe_edits,
          "task state edit count exceeds continuation limit");
}

std::shared_ptr<const GraphTask::Replay> GraphTask::replay(
    const std::vector<std::uint32_t>& recipe) const {
  if (auto found = replay_cache_.find(recipe); found != replay_cache_.end()) {
    ++replay_cache_hits_;
    found->second.last_used = ++replay_cache_serial_;
    return found->second.value;
  }
  ++replay_cache_misses_;
  auto out = std::make_shared<Replay>(
      request_, dispatcher_, request_.config.graph_limits);
  std::uint32_t policy_edits = 0;
  for (const auto action_id : recipe) {
    const auto& spec = catalog_.at(action_id);
    require(graph_action_legal(*out, spec, true, policy_edits),
            "stored recipe became illegal at " + spec.key);
    const std::vector<graph::StreamId> ids = [&] {
      std::vector<graph::StreamId> v; for (const auto& [id, _] : out->session.active_streams()) v.push_back(id);
      return v;
    }();
    auto target_id = ids.at(*spec.target_slot);
    std::vector<graph::StreamId> original_inputs{target_id}, inputs{target_id};
    if (spec.category == ActionCategory::Concat) {
      original_inputs.push_back(ids.at(*spec.other_slot)); inputs.push_back(ids.at(*spec.other_slot));
    }
    const auto old_meta = out->meta;
    const auto parent_action = old_meta.at(target_id).parent;
    const auto parent_runtime = catalog_.at(parent_action).runtime_action;
    if (!spec.required_parent_runtime_actions.empty() &&
        !contains(spec.required_parent_runtime_actions, parent_runtime)) {
      const auto found = catalog_.internal_adapters().find(spec.runtime_action);
      require(found != catalog_.internal_adapters().end(), "required native parent adapter is missing");
      if (!found->second.required_policy_parents.empty())
        require(contains(found->second.required_policy_parents, parent_runtime),
                "internal adapter policy-parent precondition changed");
      graph::Step adapter;
      adapter.kind = graph::StepKind::Stage; adapter.inputs = {target_id};
      adapter.stage = {found->second.action, "bytes", {}}; adapter.policy_action_id = -1;
      const auto adapter_outputs = out->session.apply(adapter);
      require(adapter_outputs.size() == 1, "internal adapter must produce one data flow");
      out->native_recipe.steps.push_back(adapter);
      target_id = adapter_outputs.front(); inputs = {target_id};
    }
    graph::Step step; step.policy_action_id = action_id; step.inputs = inputs;
    if (spec.category == ActionCategory::SplitSegments) {
      step.kind = graph::StepKind::SplitSegments;
      step.segment_sizes = segment_sizes(spec, out->session.active_streams().at(target_id));
    } else if (spec.category == ActionCategory::Concat) {
      step.kind = graph::StepKind::Concat;
    } else {
      step.kind = graph::StepKind::Stage;
      step.stage.action = spec.runtime_action;
      step.stage.profile = stage_profile(spec);
      step.stage.parameters = parameters(spec, out->session.active_streams().at(target_id));
    }
    const auto outputs = out->session.apply(step);
    out->native_recipe.steps.push_back(step);
    std::map<graph::StreamId, Replay::Meta> next;
    for (const auto& [id, meta] : old_meta)
      if (!contains(original_inputs, id)) next.emplace(id, meta);
    const auto parent_depth = std::accumulate(original_inputs.begin(), original_inputs.end(), 0U,
        [&](std::uint32_t value, graph::StreamId id) { return std::max(value, old_meta.at(id).depth); });
    const auto depth = parent_depth + unsigned(spec.ordinary_operator);
    std::vector<std::string> views;
    if (spec.category == ActionCategory::SplitSegments)
      views.assign(outputs.size(), old_meta.at(original_inputs.front()).view);
    else if (spec.category == ActionCategory::Concat) {
      std::set<std::string> selected;
      for (const auto id : original_inputs) selected.insert(old_meta.at(id).view);
      views.assign(outputs.size(), selected.size() == 1 ? *selected.begin() : "bytes");
    } else if (spec.runtime_action == "op.xor.previous_word.semantic_ports.cpp_v1")
      views = {"uint64", "xor_residual64"};
    else if (spec.runtime_action == "op.outlier_separation.high_bits.uint32.cpp_v1")
      views.assign(outputs.size(), "uint32");
    else views.assign(outputs.size(), "bytes");
    require(views.size() == outputs.size(), "native stage output arity changed from catalog semantics");
    for (std::size_t i = 0; i < outputs.size(); ++i)
      next.emplace(outputs[i], Replay::Meta{depth, views[i], action_id});
    out->meta = std::move(next);
    ++policy_edits;
    ++out->policy_nodes;
  }
  const auto retained = replay_retained_bytes(*out);
  std::shared_ptr<const Replay> immutable = std::move(out);
  if (retained <= replay_cache_byte_limit_) {
    while (!replay_cache_.empty() &&
           replay_cache_bytes_ > replay_cache_byte_limit_ - retained) {
      const auto victim = std::min_element(
          replay_cache_.begin(), replay_cache_.end(), [](const auto& a, const auto& b) {
            return a.second.last_used < b.second.last_used;
          });
      replay_cache_bytes_ -= victim->second.retained_bytes;
      replay_cache_.erase(victim);
      ++replay_cache_evictions_;
    }
    replay_cache_.emplace(recipe, ReplayCacheEntry{
        immutable, retained, ++replay_cache_serial_});
    replay_cache_bytes_ += retained;
  }
  return immutable;
}

std::uint64_t GraphTask::replay_retained_bytes(const Replay& replayed) const {
  const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("replay retained-byte estimate overflow");
    total += value;
  };
  const auto checked_multiply = [](std::uint64_t count, std::uint64_t width) {
    if (count != 0 && width > std::numeric_limits<std::uint64_t>::max() / count)
      throw std::overflow_error("replay retained-byte estimate overflow");
    return count * width;
  };
  const auto add_string = [&](std::uint64_t& total, const std::string& value) {
    if (value.capacity() == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("replay retained-byte estimate overflow");
    checked_add(total, static_cast<std::uint64_t>(value.capacity()) + 1);
  };
  const auto map_nodes = [&](std::uint64_t count, std::uint64_t value_bytes) {
    return checked_multiply(count, value_bytes + 4 * sizeof(void*));
  };

  // sizeof(Replay) already contains the GraphSession wrapper; replace that
  // inline size with GraphSession's capacity-based live-owned estimate. Add a
  // conservative make_shared control-block allowance and the cache map/key
  // allocation attributable to this entry. This remains an allocator estimate
  // (not RSS), but it scales with live state instead of max_metadata_bytes.
  std::uint64_t bytes = sizeof(Replay) - sizeof(graph::GraphSession);
  checked_add(bytes, replayed.session.retained_bytes());
  checked_add(bytes, 4 * sizeof(void*));
  using CacheValue = std::pair<const std::vector<std::uint32_t>, ReplayCacheEntry>;
  checked_add(bytes, sizeof(CacheValue) + 4 * sizeof(void*));
  checked_add(bytes, checked_multiply(replayed.policy_nodes, sizeof(std::uint32_t)));

  using MetaValue = std::pair<const graph::StreamId, Replay::Meta>;
  checked_add(bytes, map_nodes(replayed.meta.size(), sizeof(MetaValue)));
  for (const auto& [_, meta] : replayed.meta)
    add_string(bytes, meta.view);

  add_string(bytes, replayed.native_recipe.policy_registry);
  checked_add(bytes, checked_multiply(replayed.native_recipe.steps.capacity(),
                                      sizeof(graph::Step)));
  for (const auto& step : replayed.native_recipe.steps) {
    checked_add(bytes, checked_multiply(step.inputs.capacity(), sizeof(graph::StreamId)));
    checked_add(bytes, checked_multiply(step.segment_sizes.capacity(), sizeof(std::uint64_t)));
    add_string(bytes, step.stage.action);
    add_string(bytes, step.stage.profile);
    using ParameterValue = std::pair<const std::string, std::string>;
    checked_add(bytes, map_nodes(step.stage.parameters.size(), sizeof(ParameterValue)));
    for (const auto& [key, value] : step.stage.parameters) {
      add_string(bytes, key);
      add_string(bytes, value);
    }
  }
  return bytes;
}

TaskState GraphTask::recipe_state(std::vector<std::uint32_t> recipe,
                                  const TaskState* parent) const {
  TaskState out;
  if (parent) out = *parent;
  out.recipe = std::move(recipe);
  out.recipe_error.reset();
  try {
    (void)replay(out.recipe);
    out.recipe_valid = true;
  } catch (const std::exception& error) {
    // Catalog IDs/categories and global limits are validated separately.  A
    // replay rejection here records that this concrete recipe is inapplicable
    // to the current block.  It remains editable in continuation mode.
    out.recipe_valid = false;
    out.recipe_error = error.what();
  }
  return out;
}

std::shared_ptr<const GraphTask::Replay> GraphTask::replay_for_features(
    const TaskState& state) const {
  if (state.recipe_valid) return replay(state.recipe);
  // Use the longest valid prefix as the physical feature view.  Recipe length,
  // edit history and failure status remain represented by TaskState; this
  // prefix is used only to provide the fixed 28-D feature schema while a
  // previously successful recipe is being repaired on a new block.
  for (std::size_t size = state.recipe.size(); size > 0; --size) {
    try {
      return replay({state.recipe.begin(), state.recipe.begin() + size});
    } catch (const std::exception&) {}
  }
  return replay({});
}

bool GraphTask::graph_action_legal(const Replay& state, const ActionSpec& spec,
                                   bool ignore_policy_limit, std::uint32_t policy_edits) const {
  if (spec.category == ActionCategory::Stop) return true;
  if (!graph_category(spec.category) || !catalog_.enabled(spec.action_id) ||
      !catalog_.support(spec.action_id).executable) return false;
  if (state.native_recipe.steps.size() >= request_.config.graph_limits.max_nodes ||
      state.meta.size() > request_.config.max_active_streams ||
      state.policy_nodes >= request_.config.max_graph_nodes ||
      (!ignore_policy_limit && policy_edits >= request_.config.max_policy_edits)) return false;
  if (!spec.target_slot || *spec.target_slot >= state.meta.size()) return false;
  auto stream_it = state.session.active_streams().begin(); std::advance(stream_it, *spec.target_slot);
  const auto target_id = stream_it->first; const auto& target = stream_it->second;
  const auto& meta = state.meta.at(target_id);
  if (spec.input_nonempty && target.data.empty()) return false;
  if (target.data.size() % spec.input_byte_multiple) return false;
  if (spec.required_nonserial_width && target.kind != graph::Kind::Serial &&
      target.width != *spec.required_nonserial_width) return false;
  if (!spec.allowed_physical_widths.empty() && !contains(spec.allowed_physical_widths, target.width)) return false;
  if (!spec.required_logical_views.empty()) {
    if (contains(spec.required_logical_views, "physical_word")) {
      if (!meta.view.starts_with("xor_residual")) return false;
    } else if (!contains(spec.required_logical_views, meta.view)) return false;
  }
  if (spec.category == ActionCategory::Concat) {
    if (!spec.other_slot || *spec.other_slot >= state.meta.size()) return false;
    auto other = state.session.active_streams().begin(); std::advance(other, *spec.other_slot);
    return target.kind == other->second.kind && target.width == other->second.width;
  }
  if (meta.depth + unsigned(spec.ordinary_operator) > request_.config.max_ordinary_depth) return false;
  if (contains(std::vector<std::string>{"int64", "float64", "float32", "uint32"}, spec.input_view) &&
      meta.view != spec.input_view) return false;
  if (spec.runtime_action == "op.bitpack_int.max_value_width.cpp_v1" && spec.input_view == "int64")
    for (std::size_t i = 0; i < target.data.size(); i += 8) if (little_i64(target.data, i) < 0) return false;
  if (spec.runtime_action == "op.eraser.camel.cpp_v1" && spec.input_view == "int64")
    for (std::size_t i = 0; i < target.data.size(); i += 8)
      if (!exactly_representable_as_float64(little_i64(target.data, i))) return false;
  if (spec.input_view == "physical_word" && !(target.kind == graph::Kind::Numeric &&
      (target.width == 4 || target.width == 8) && meta.view == "xor_residual" + std::to_string(target.width * 8)))
    return false;
  const InternalAdapter* inserted_adapter = nullptr;
  if (!spec.required_parent_runtime_actions.empty()) {
    const auto parent = catalog_.at(meta.parent).runtime_action;
    if (!contains(spec.required_parent_runtime_actions, parent)) {
      const auto adapter = catalog_.internal_adapters().find(spec.runtime_action);
      if (adapter == catalog_.internal_adapters().end() ||
          !contains(spec.required_parent_runtime_actions, adapter->second.action) ||
          (!adapter->second.required_policy_parents.empty() &&
           !contains(adapter->second.required_policy_parents, parent))) return false;
      inserted_adapter = &adapter->second;
    }
  }
  if (spec.runtime_action == "op.outlier_separation.high_bits.uint32.cpp_v1" &&
      !(target.kind == graph::Kind::Numeric && target.width == 4)) return false;
  if (spec.runtime_action == "float_deconstruct" &&
      !(target.kind == graph::Kind::Numeric && target.width == 4 && meta.view == "float32")) return false;
  if ((spec.runtime_action == "columnize" || spec.runtime_action == "bitsplit") && target.width <= 1) return false;
  if (spec.runtime_action == "bitunpack" && target.kind != graph::Kind::Serial) return false;
  if (spec.runtime_action == "op.xor.previous_word.cpp_v2" && spec.input_view == "bytes" &&
      target.kind != graph::Kind::Serial) return false;
  if (spec.category == ActionCategory::SplitSegments) {
    if (target.data.size() / target.width < 2) return false;
    const auto outputs = segment_sizes(spec, target).size();
    if (state.meta.size() - 1 + outputs > request_.config.max_active_streams) return false;
  }
  if (spec.runtime_action == "op.prediction_probability.cmix.provider_v1" &&
      !request_.config.allow_large_memory) return false;
  if (spec.runtime_action == "op.prediction_probability.nncp.provider_v1" &&
      target.data.size() > 4096) return false;
  // Python's production mask executes every max_point configuration against
  // the current block before exposing it.  Do the same in the append mask;
  // replay/transition still executes the actual graph and remains the final
  // authority.  Internal replay sets ignore_policy_limit and avoids a second
  // probe before its real stage execution.
  const auto multiple_outputs = std::any_of(
      graph::native_stage_capabilities().begin(), graph::native_stage_capabilities().end(),
      [&](const auto& capability) {
        return capability.action == spec.runtime_action && capability.multiple_outputs;
      });
  if ((spec.max_point || packet_precondition_requires_probe(spec) || multiple_outputs ||
       inserted_adapter) &&
      !ignore_policy_limit) {
    try {
      auto probed_target = target;
      std::size_t adapter_nodes = 0;
      if (inserted_adapter) {
        graph::StageRequest adapter_request;
        adapter_request.action = inserted_adapter->action;
        adapter_request.profile = "bytes";
        const auto adapted = dispatcher_.forward(adapter_request, {target});
        if (adapted.outputs.size() != 1) return false;
        probed_target = adapted.outputs.front();
        adapter_nodes = 1;
      }
      if (state.native_recipe.steps.size() + adapter_nodes >=
          request_.config.graph_limits.max_nodes)
        return false;
      graph::StageRequest probe;
      probe.action = spec.runtime_action;
      probe.profile = stage_profile(spec);
      probe.parameters = parameters(spec, probed_target);
      const auto result = dispatcher_.forward(probe, {probed_target});
      if (result.outputs.empty() ||
          state.meta.size() - 1 + result.outputs.size() > request_.config.max_active_streams)
        return false;
    } catch (const std::exception&) { return false; }
  }
  return true;
}

std::vector<std::uint32_t> GraphTask::parameter_change_candidates(std::uint32_t action) const {
  const auto& old = catalog_.at(action);
  if (!old.implementation_id) return {};
  std::vector<std::uint32_t> out;
  for (const auto& row : catalog_.actions())
    if (row.category == ActionCategory::Mechanism && row.implementation_id == old.implementation_id &&
        row.runtime_action == old.runtime_action && row.target_slot == old.target_slot &&
        !same_parameter_tuple(row, old)) out.push_back(row.action_id);
  return out;
}

bool GraphTask::edit_position_has_completion(const TaskState& state,
                                             const std::string& kind,
                                             std::uint32_t position) const {
  if (kind == "delete")
    return position < state.recipe.size() &&
        edited(state, kind, position, std::nullopt).has_value();
  if (kind != "insert" && kind != "replace" && kind != "parameter_change")
    return false;
  if ((kind == "insert" && position > state.recipe.size()) ||
      (kind != "insert" && position >= state.recipe.size()))
    return false;

  std::shared_ptr<const Replay> prefix;
  try {
    prefix = replay({state.recipe.begin(), state.recipe.begin() + position});
  } catch (const std::exception&) {
    return false;
  }
  const auto parameter_candidates = kind == "parameter_change"
      ? parameter_change_candidates(state.recipe[position])
      : std::vector<std::uint32_t>{};
  for (const auto& candidate : catalog_.actions()) {
    if (!graph_category(candidate.category)) continue;
    if (kind == "parameter_change" &&
        !contains(parameter_candidates, candidate.action_id)) continue;
    if (!graph_action_legal(*prefix, candidate, true, 0)) continue;
    if (edited(state, kind, position, candidate.action_id)) return true;
  }
  return false;
}

std::vector<std::uint32_t> GraphTask::edit_recipe(const ActionCatalog& catalog,
    std::vector<std::uint32_t> recipe, const RecipeEdit& edit) {
  auto valid = [&](std::uint32_t id) {
    const auto& row = catalog.at(id); return id != stop_action && graph_category(row.category);
  };
  if (edit.kind == "insert") {
    require(edit.position <= recipe.size() && edit.action_id && valid(*edit.action_id), "invalid insert edit");
    recipe.insert(recipe.begin() + edit.position, *edit.action_id);
  } else if (edit.kind == "delete") {
    require(edit.position < recipe.size() && !edit.action_id, "invalid delete edit");
    recipe.erase(recipe.begin() + edit.position);
  } else if (edit.kind == "replace" || edit.kind == "parameter_change") {
    require(edit.position < recipe.size() && edit.action_id && valid(*edit.action_id), "invalid replacement edit");
    if (edit.kind == "parameter_change") {
      const auto& old = catalog.at(recipe[edit.position]); const auto& replacement = catalog.at(*edit.action_id);
      require(old.implementation_id == replacement.implementation_id &&
              old.runtime_action == replacement.runtime_action && !same_parameter_tuple(old, replacement),
              "parameter_change must keep mechanism/runtime and change its fixed tuple");
    }
    recipe[edit.position] = *edit.action_id;
  } else throw std::invalid_argument("unknown continuation edit kind: " + edit.kind);
  return recipe;
}

std::optional<TaskState> GraphTask::edited(const TaskState& state, const std::string& kind,
    std::uint32_t position, std::optional<std::uint32_t> action) const {
  try {
    auto recipe = edit_recipe(catalog_, state.recipe, RecipeEdit{kind, position, action});
    if (recipe.size() > request_.config.max_graph_nodes) return std::nullopt;
    auto out = recipe_state(std::move(recipe), &state);
    // A valid state must never expose an edit that creates an invalid recipe.
    // An already-invalid cross-block recipe may remain invalid after its first
    // completed edit so the second permitted local edit can repair it.
    if (state.recipe_valid && !out.recipe_valid) return std::nullopt;
    out.policy_edits = state.policy_edits + 1;
    out.replayed_nodes = static_cast<std::uint32_t>(out.recipe.size());
    out.continuation_edit_mode = true; out.pending_edit_kind.reset();
    out.pending_edit_index.reset(); out.edit_count = state.edit_count + 1;
    out.edit_history = state.edit_history;
    out.edit_history.push_back({kind, position, action});
    return out;
  } catch (const std::exception&) { return std::nullopt; }
}

bool GraphTask::legal(const TaskState& state, const ActionSpec& spec, bool replay_mode) const {
  if (replay_mode || !state.continuation_edit_mode) {
    if (spec.category == ActionCategory::EditType || spec.category == ActionCategory::EditPosition) return false;
    if (!state.recipe_valid) return spec.action_id == stop_action;
    const auto graph = replay(state.recipe);
    return graph_action_legal(*graph, spec, replay_mode, state.policy_edits);
  }
  if (spec.action_id == stop_action) return !state.pending_edit_kind;
  if (state.edit_count >= request_.max_recipe_edits) return false;
  if (spec.category == ActionCategory::EditType) {
    if (state.pending_edit_kind || !spec.edit_kind) return false;
    const auto& kind = *spec.edit_kind;
    if (kind == "insert" && state.recipe.size() >= request_.config.max_graph_nodes)
      return false;
    if ((kind == "delete" || kind == "replace" || kind == "parameter_change") &&
        state.recipe.empty()) return false;
    const auto positions = kind == "insert" ? state.recipe.size() + 1 : state.recipe.size();
    for (std::uint32_t position = 0; position < positions; ++position)
      if (edit_position_has_completion(state, kind, position)) return true;
    return false;
  }
  if (spec.category == ActionCategory::EditPosition) {
    if (!state.pending_edit_kind || state.pending_edit_index || !spec.parameter_value) return false;
    const auto position = static_cast<std::uint32_t>(*spec.parameter_value);
    return edit_position_has_completion(
        state, *state.pending_edit_kind, position);
  }
  if (!graph_category(spec.category) || !state.pending_edit_kind || !state.pending_edit_index) return false;
  if (*state.pending_edit_kind != "insert" && *state.pending_edit_kind != "replace" &&
      *state.pending_edit_kind != "parameter_change") return false;
  if (*state.pending_edit_kind == "parameter_change" &&
      !contains(parameter_change_candidates(state.recipe[*state.pending_edit_index]), spec.action_id)) return false;
  try {
    const std::vector<std::uint32_t> prefix(state.recipe.begin(), state.recipe.begin() + *state.pending_edit_index);
    const auto graph = replay(prefix);
    if (!graph_action_legal(*graph, spec, true, 0)) return false;
  } catch (const std::exception&) { return false; }
  return edited(state, *state.pending_edit_kind, *state.pending_edit_index, spec.action_id).has_value();
}

std::vector<std::uint32_t> GraphTask::legal_actions(const TaskState& state) const {
  verify_state(state);
  if (!state.continuation_edit_mode && state.policy_edits >= request_.config.max_policy_edits)
    return {stop_action};
  if (state.continuation_edit_mode && state.edit_count >= request_.max_recipe_edits &&
      !state.pending_edit_kind) return {stop_action};
  std::vector<std::uint32_t> out;
  if (!state.continuation_edit_mode) {
    if (!state.recipe_valid) return {stop_action};
    // Replaying once is essential: legality is queried for all 6,341 fixed
    // actions at every policy step, and a replay may execute expensive native
    // stages.  The immutable replay is sufficient for every row's static and
    // typed graph checks.
    const auto graph = replay(state.recipe);
    for (const auto& row : catalog_.actions()) {
      if (row.category == ActionCategory::EditType || row.category == ActionCategory::EditPosition)
        continue;
      if (graph_action_legal(*graph, row, false, state.policy_edits)) out.push_back(row.action_id);
    }
  } else {
    // Continuation masks retain the exact edit semantics.  Operator choices
    // after an edit-position token still require a complete edited-recipe
    // replay because a replacement can invalidate a later suffix.
    if (state.pending_edit_kind && state.pending_edit_index &&
        (*state.pending_edit_kind == "insert" || *state.pending_edit_kind == "replace" ||
         *state.pending_edit_kind == "parameter_change")) {
      const auto position = *state.pending_edit_index;
      const std::vector<std::uint32_t> prefix(
          state.recipe.begin(), state.recipe.begin() + position);
      const auto graph = replay(prefix);
      const auto parameter_candidates = *state.pending_edit_kind == "parameter_change"
          ? parameter_change_candidates(state.recipe[position])
          : std::vector<std::uint32_t>{};
      for (const auto& row : catalog_.actions()) {
        if (!graph_category(row.category)) continue;
        if (*state.pending_edit_kind == "parameter_change" &&
            !contains(parameter_candidates, row.action_id)) continue;
        if (!graph_action_legal(*graph, row, true, 0)) continue;
        if (edited(state, *state.pending_edit_kind, position, row.action_id))
          out.push_back(row.action_id);
      }
    } else {
      for (const auto& row : catalog_.actions()) if (legal(state, row)) out.push_back(row.action_id);
    }
  }
  require(!out.empty(), "native legality mask produced no action");
  return out;
}

std::optional<TaskState> GraphTask::transition(const TaskState& state, std::uint32_t action) const {
  verify_state(state); const auto& spec = catalog_.at(action);
  if (action == stop_action || !legal(state, spec)) return std::nullopt;
  if (state.continuation_edit_mode) {
    if (spec.category == ActionCategory::EditType) {
      auto out = state; out.pending_edit_kind = spec.edit_kind; out.pending_edit_index.reset(); return out;
    }
    if (spec.category == ActionCategory::EditPosition) {
      const auto position = static_cast<std::uint32_t>(*spec.parameter_value);
      if (*state.pending_edit_kind == "delete") return edited(state, "delete", position, std::nullopt);
      auto out = state; out.pending_edit_index = position; return out;
    }
    return edited(state, *state.pending_edit_kind, *state.pending_edit_index, action);
  }
  auto out = state; out.recipe.push_back(action); ++out.policy_edits;
  (void)replay(out.recipe); out.recipe_valid = true; out.recipe_error.reset(); return out;
}

features_v1::State GraphTask::feature_state(const TaskState& state, const Replay& replayed) const {
  features_v1::State out;
  out.original_numeric_bytes = original_numeric_bytes_;
  out.node_count = static_cast<std::uint32_t>(state.recipe.size());
  out.policy_edits = state.policy_edits; out.replayed_nodes = state.replayed_nodes;
  out.edit_count = state.edit_count; out.max_nodes = request_.config.max_graph_nodes;
  out.max_depth = request_.config.max_ordinary_depth; out.max_active = request_.config.max_active_streams;
  out.max_policy_edits = request_.config.max_policy_edits; out.max_recipe_edits = request_.max_recipe_edits;
  out.candidate_used = candidate_used_; out.candidate_limit = candidate_budget_;
  out.action_count = action_count; out.continuation_edit_mode = state.continuation_edit_mode;
  out.pending_edit_kind = state.pending_edit_kind.value_or(""); out.pending_edit_index = state.pending_edit_index;
  if (const auto found = measured_objectives_.find(state.recipe); found != measured_objectives_.end())
    out.current_objective = found->second;
  for (const auto& [id, flow] : replayed.session.active_streams()) {
    const auto& meta = replayed.meta.at(id);
    out.flows.push_back({{physical_kind(flow.kind), flow.width, flow.data, meta.view}, meta.depth, meta.parent});
  }
  return out;
}

StateInspection GraphTask::inspect(const TaskState& state) const {
  verify_state(state); const auto graph = replay_for_features(state);
  std::vector<features_v1::FlowView> roots;
  for (const auto& root : request_.roots)
    roots.push_back({physical_kind(root.stream.kind), root.stream.width, root.stream.data, root.logical_view});
  StateInspection out;
  out.task_features = features_v1::task(roots);
  out.state_features = features_v1::state(feature_state(state, *graph));
  out.legal_action_ids = legal_actions(state);
  out.node_count = static_cast<std::uint32_t>(state.recipe.size());
  out.active_data_flows = static_cast<std::uint32_t>(graph->meta.size());
  for (const auto& [_, value] : graph->meta)
    out.maximum_depth = std::max(out.maximum_depth, value.depth);
  out.terminal_evaluable = !state.pending_edit_kind;
  return out;
}

graph::Recipe GraphTask::compile_recipe(const std::vector<std::uint32_t>& actions) const {
  const auto replayed = replay(actions);
  auto recipe = replayed->native_recipe;
  graph::Step stop;
  stop.kind = graph::StepKind::Stop;
  stop.policy_action_id = stop_action;
  recipe.steps.push_back(stop);
  return recipe;
}

Evaluation GraphTask::evaluate_terminal(const TaskState& state) {
  verify_state(state); require(!state.pending_edit_kind, "cannot evaluate an incomplete continuation edit");
  require(candidate_used_ < candidate_budget_, "complete-candidate budget exhausted");
  ++candidate_used_;
  Evaluation out; out.candidate_ordinal = candidate_used_; out.candidate_budget = candidate_budget_;
  out.original_numeric_bytes = original_numeric_bytes_;
  out.max_frame_bytes = request_.config.graph_limits.max_frame_bytes;
  if (!state.recipe_valid) {
    out.success = false;
    out.failure_error = state.recipe_error.value_or("recipe is inapplicable to the current block");
    return out;
  }
  out.native_recipe = compile_recipe(state.recipe);
  graph::BenchmarkResult measured;
  try {
    measured = graph::benchmark(request_.roots, out.native_recipe, dispatcher_,
        request_.config.timing_warmups, request_.config.timing_repetitions,
        request_.config.graph_limits);
  } catch (const std::exception& error) {
    // A data/configuration-specific native benchmark rejection is a charged
    // failed candidate. The frame remains absent. Independent correctness
    // verification below is deliberately outside this catch so corruption or
    // a round-trip invariant violation still aborts as an implementation bug.
    out.success = false;
    out.failure_error = "NATIVE_CANDIDATE_BENCHMARK_FAILED: " +
        std::string(error.what());
    return out;
  }
  out.frame = measured.frame; out.compression_samples_ns = measured.encode_nanoseconds;
  out.decompression_samples_ns = measured.decode_nanoseconds;
  // A second frame-only parse/restore proves that evaluation does not need the
  // policy, task state, or encoder cache. It is correctness-only, outside samples.
  verify_decoded(graph::decode(out.frame, dispatcher_, request_.config.graph_limits), request_);
  out.compression_time_ns = median(out.compression_samples_ns);
  out.decompression_time_ns = median(out.decompression_samples_ns);
  out.compression_ratio = double(out.frame.size()) / double(original_numeric_bytes_);
  out.compression_ns_per_byte = double(out.compression_time_ns) / double(original_numeric_bytes_);
  out.decompression_ns_per_byte = double(out.decompression_time_ns) / double(original_numeric_bytes_);
  const auto en = out.compression_ns_per_byte /
      (out.compression_ns_per_byte + request_.compression_reference_ns_per_byte);
  const auto dn = out.decompression_ns_per_byte /
      (out.decompression_ns_per_byte + request_.decompression_reference_ns_per_byte);
  out.objective = request_.weights[0] * out.compression_ratio + request_.weights[1] * en +
                  request_.weights[2] * dn;
  measured_objectives_[state.recipe] = out.objective;
  for (const auto action : std::set<std::uint32_t>(state.recipe.begin(), state.recipe.end())) {
    auto& rows = measured_action_costs_[action];
    rows.push_back({out.compression_ratio, out.compression_ns_per_byte, out.decompression_ns_per_byte});
    if (rows.size() > 64) rows.erase(rows.begin(), rows.end() - 64);
  }
  return out;
}

std::uint32_t GraphTask::consume_failed_candidate() {
  require(candidate_used_ < candidate_budget_, "complete-candidate budget exhausted");
  return ++candidate_used_;
}

std::optional<std::array<double, 3>> GraphTask::measured_action_cost(std::uint32_t action) const {
  catalog_.at(action); const auto found = measured_action_costs_.find(action);
  if (found == measured_action_costs_.end() || found->second.empty()) return std::nullopt;
  std::array<double, 3> out{};
  for (std::size_t column = 0; column < 3; ++column) {
    std::vector<double> values; for (const auto& row : found->second) values.push_back(row[column]);
    std::sort(values.begin(), values.end()); const auto n = values.size();
    out[column] = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
  }
  return out;
}
std::size_t GraphTask::action_measurement_count(std::uint32_t action) const {
  catalog_.at(action); const auto found = measured_action_costs_.find(action);
  return found == measured_action_costs_.end() ? 0 : found->second.size();
}

ReplayCacheStats GraphTask::replay_cache_stats() const noexcept {
  return {replay_cache_hits_, replay_cache_misses_, replay_cache_evictions_,
          replay_cache_bytes_, replay_cache_byte_limit_,
          static_cast<std::uint32_t>(replay_cache_.size())};
}

}  // namespace wc::pipeline_v1
