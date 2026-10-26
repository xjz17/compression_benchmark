#include "graph_internal.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <utility>

namespace native_graph_v1 {
using detail::require;
namespace {
void match(const Stream& stream, const detail::Spec& expected, const Limits& limits) {
  detail::validate_spec(detail::spec(stream), limits);
  require(detail::spec(stream) == expected, "inverse data flow shape mismatch");
}
std::vector<Stream> split(const Stream& input, const std::vector<std::uint64_t>& sizes) {
  std::uint64_t sum = 0;
  for (const auto n : sizes) sum = detail::checked_add(sum, detail::checked_multiply(n, input.width));
  require(sum == input.data.size(), "continuous Split sizes must exactly cover input");
  std::vector<Stream> out;
  std::size_t position = 0;
  for (const auto n : sizes) {
    const auto bytes = static_cast<std::size_t>(detail::checked_multiply(n, input.width));
    out.push_back({input.kind, input.width,
      Bytes(input.data.begin() + position, input.data.begin() + position + bytes)});
    position += bytes;
  }
  return out;
}
Stream concatenate(const std::vector<Stream>& input, const Limits& limits) {
  require(!input.empty(), "Concat needs input");
  Stream out{input[0].kind, input[0].width, {}};
  std::uint64_t bytes = 0;
  for (const auto& s : input) {
    require(s.kind == out.kind && s.width == out.width, "Concat requires identical kind and width");
    bytes = detail::checked_add(bytes, s.data.size());
  }
  require(bytes <= limits.max_stream_bytes, "Concat output exceeds stream size limit");
  out.data.reserve(static_cast<std::size_t>(bytes));
  for (const auto& s : input) out.data.insert(out.data.end(), s.data.begin(), s.data.end());
  return out;
}
void verify(const Decoded& restored, const std::vector<RootInput>& roots) {
  require(restored.roots.size() == roots.size(), "benchmark root count mismatch");
  for (std::size_t i = 0; i < roots.size(); ++i)
    require(restored.roots[i].stream == roots[i].stream &&
            restored.roots[i].original_numeric_payload == roots[i].original_numeric_payload &&
            restored.roots[i].logical_view == (roots[i].logical_view.empty() ?
              detail::physical_view(roots[i].stream.kind) : roots[i].logical_view),
            "benchmark bitwise restore mismatch");
}
std::uint64_t nanos(std::chrono::steady_clock::time_point started,
                    std::chrono::steady_clock::time_point ended) {
  const auto n = std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count();
  require(n >= 0, "nonmonotonic steady clock");
  return static_cast<std::uint64_t>(n);
}
void retained_add(std::uint64_t& total, std::uint64_t bytes) {
  total = detail::checked_add(total, bytes);
}
template<class T>
void retained_vector(std::uint64_t& total, const std::vector<T>& values) {
  retained_add(total, detail::checked_multiply(values.capacity(), sizeof(T)));
}
void retained_string(std::uint64_t& total, const std::string& value) {
  // capacity()+1 is conservative for a heap string and deliberately
  // overcounts implementations that keep this value in small-string storage.
  retained_add(total, detail::checked_add(value.capacity(), 1));
}
template<class Map>
void retained_map_nodes(std::uint64_t& total, const Map& values) {
  // std::map exposes neither node capacity nor allocator usable size. Four
  // pointers in addition to value_type conservatively cover parent/children,
  // colour/padding and common allocator rounding without charging a fixed
  // maximum unrelated to the number of live entries.
  constexpr auto node_bytes = sizeof(typename Map::value_type) + 4 * sizeof(void*);
  retained_add(total, detail::checked_multiply(values.size(), node_bytes));
}
void retained_step_dynamic(std::uint64_t& total, const Step& step) {
  retained_vector(total, step.inputs);
  retained_vector(total, step.segment_sizes);
  retained_string(total, step.stage.action);
  retained_string(total, step.stage.profile);
  retained_map_nodes(total, step.stage.parameters);
  for (const auto& [key, value] : step.stage.parameters) {
    retained_string(total, key);
    retained_string(total, value);
  }
}
}  // namespace

struct GraphSession::Impl {
  detail::Frame frame;
  const StageDispatcher& dispatcher;
  Limits limits;
  std::map<StreamId, FlowInfo> infos;
  // Closed leaves stay in frame.leaves for complete recovery but can no
  // longer be consumed by apply(). This state is intentionally not serialized:
  // the final frame already identifies every terminal leaf.
  std::set<StreamId> closed;
  StreamId next = 0;
  bool stopped = false;
  Impl(std::vector<RootInput> roots, const StageDispatcher& d, Limits l, std::string registry)
      : dispatcher(d), limits(l) {
    detail::validate_limits(limits);
    require(!roots.empty() && roots.size() <= limits.max_active_streams, "invalid root count");
    frame.dispatcher = dispatcher.identity(); frame.policy_registry = std::move(registry);
    for (auto& root : roots) {
      auto s = detail::spec(root.stream); detail::validate_spec(s, limits);
      const auto view = root.logical_view.empty() ? detail::physical_view(s.kind) : root.logical_view;
      detail::validate_view(view);
      frame.roots.push_back({s, root.original_numeric_payload, detail::sha256(root.stream.data), view});
      if (root.original_numeric_payload)
        frame.numeric_bytes = detail::checked_add(frame.numeric_bytes, s.bytes);
      infos.emplace(next, FlowInfo{0, "stop", view, 0}); frame.leaves.emplace(next++, std::move(root.stream));
    }
    detail::validate_frame(frame, limits);
  }
};
GraphSession::GraphSession(std::vector<RootInput> roots, const StageDispatcher& d, Limits l, std::string registry)
    : impl_(std::make_unique<Impl>(std::move(roots), d, l, std::move(registry))) {}
GraphSession::~GraphSession() = default;
GraphSession::GraphSession(GraphSession&&) noexcept = default;
GraphSession& GraphSession::operator=(GraphSession&&) noexcept = default;
const std::map<StreamId, Stream>& GraphSession::active_streams() const { return impl_->frame.leaves; }
std::vector<StreamId> GraphSession::open_stream_ids() const {
  std::vector<StreamId> ids;
  ids.reserve(impl_->frame.leaves.size() - impl_->closed.size());
  for (const auto& [id, _] : impl_->frame.leaves)
    if (!impl_->closed.contains(id)) ids.push_back(id);
  return ids;
}
std::size_t GraphSession::open_stream_count() const {
  return impl_->frame.leaves.size() - impl_->closed.size();
}
bool GraphSession::is_closed(StreamId id) const {
  return impl_->closed.contains(id);
}
void GraphSession::close_flow(StreamId id) {
  auto& self = *impl_;
  require(!self.stopped, "graph has already stopped");
  require(self.frame.leaves.contains(id), "cannot close a non-leaf data flow");
  require(self.closed.emplace(id).second, "data flow is already closed");
}
std::uint32_t GraphSession::ordinary_depth(StreamId id) const { return impl_->infos.at(id).ordinary_depth; }
const FlowInfo& GraphSession::flow_info(StreamId id) const { return impl_->infos.at(id); }
std::size_t GraphSession::node_count() const { return impl_->frame.nodes.size(); }
std::uint64_t GraphSession::original_numeric_bytes() const { return impl_->frame.numeric_bytes; }
std::uint64_t GraphSession::retained_bytes() const {
  const auto& self = *impl_;
  std::uint64_t bytes = sizeof(GraphSession);
  retained_add(bytes, sizeof(Impl));

  retained_string(bytes, self.frame.dispatcher);
  retained_string(bytes, self.frame.policy_registry);
  retained_vector(bytes, self.frame.roots);
  for (const auto& root : self.frame.roots)
    retained_string(bytes, root.logical_view);

  retained_vector(bytes, self.frame.nodes);
  for (const auto& node : self.frame.nodes) {
    retained_step_dynamic(bytes, node.step);
    retained_vector(bytes, node.input_specs);
    retained_vector(bytes, node.outputs);
    retained_vector(bytes, node.output_specs);
    retained_vector(bytes, node.output_logical_views);
    for (const auto& view : node.output_logical_views)
      retained_string(bytes, view);
    retained_vector(bytes, node.metadata);
  }

  retained_map_nodes(bytes, self.frame.leaves);
  for (const auto& [_, stream] : self.frame.leaves)
    retained_vector(bytes, stream.data);

  retained_map_nodes(bytes, self.infos);
  for (const auto& [_, info] : self.infos) {
    retained_string(bytes, info.parent_action);
    retained_string(bytes, info.logical_view);
  }
  retained_map_nodes(bytes, self.closed);
  return bytes;
}
std::vector<StreamId> GraphSession::apply(const Step& step) {
  auto& self = *impl_;
  require(!self.stopped, "graph has already stopped");
  detail::validate_step(step, self.limits);
  require(step.kind != StepKind::Stop, "call stop() to close the whole graph");
  require(self.frame.nodes.size() < self.limits.max_nodes, "node budget exceeded");
  detail::Node node; node.step = step;
  std::vector<Stream> input;
  std::uint32_t depth = 0;
  for (const auto id : step.inputs) {
    require(self.frame.leaves.contains(id), "node input is not active");
    require(!self.closed.contains(id), "node input data flow is closed");
    const auto& s = self.frame.leaves.at(id);
    node.input_specs.push_back(detail::spec(s)); input.push_back(s);
    depth = std::max(depth, self.infos.at(id).ordinary_depth);
  }
  if (step.kind == StepKind::Stage) ++depth;
  require(depth <= self.limits.max_ordinary_depth, "ordinary operator depth exceeded");
  StageResult result;
  if (step.kind == StepKind::Stage) result = self.dispatcher.forward(step.stage, input);
  else if (step.kind == StepKind::SplitSegments) result.outputs = split(input.at(0), step.segment_sizes);
  else result.outputs.push_back(concatenate(input, self.limits));
  require(!result.outputs.empty() && result.outputs.size() <= self.limits.max_active_streams,
          "invalid number of stage output ports");
  if (step.kind != StepKind::Stage) {
    auto view = self.infos.at(step.inputs[0]).logical_view;
    if (step.kind == StepKind::Concat)
      for (auto id : step.inputs)
        if (self.infos.at(id).logical_view != self.infos.at(step.inputs[0]).logical_view)
          view = "bytes";
    result.output_logical_views.assign(result.outputs.size(), view);
  } else if (result.output_logical_views.empty()) {
    for (const auto& value : result.outputs) result.output_logical_views.push_back(detail::physical_view(value.kind));
  }
  require(result.output_logical_views.size() == result.outputs.size(), "stage logical view count mismatch");
  node.output_logical_views = std::move(result.output_logical_views);
  auto next = self.next;
  for (const auto& s : result.outputs) {
    detail::validate_spec(detail::spec(s), self.limits);
    node.outputs.push_back(next++); node.output_specs.push_back(detail::spec(s));
  }
  node.metadata = std::move(result.metadata);
  // Build a new owned state before commit so exceptions cannot consume inputs.
  detail::Frame candidate{self.frame.dispatcher, self.frame.policy_registry, self.frame.numeric_bytes,
                          self.frame.roots, self.frame.nodes, {}};
  auto infos = self.infos;
  for (const auto& [id, value] : self.frame.leaves)
    if (std::find(step.inputs.begin(), step.inputs.end(), id) == step.inputs.end())
      candidate.leaves.emplace(id, value);
  for (const auto id : step.inputs) infos.erase(id);
  for (std::size_t i = 0; i < result.outputs.size(); ++i) {
    candidate.leaves.emplace(node.outputs[i], std::move(result.outputs[i]));
    const auto parent = step.kind == StepKind::Stage ? step.stage.action :
                        step.kind == StepKind::Concat ? "concat" : "split_segments";
    infos.emplace(node.outputs[i], FlowInfo{depth, parent, node.output_logical_views[i], step.policy_action_id});
  }
  const auto ids = node.outputs;
  candidate.nodes.push_back(std::move(node));
  detail::validate_frame(candidate, self.limits);
  self.frame = std::move(candidate); self.infos = std::move(infos); self.next = next;
  return ids;
}
Bytes GraphSession::stop() {
  require(!impl_->stopped, "graph has already stopped");
  auto frame = detail::serialize_frame(impl_->frame, impl_->limits);
  impl_->stopped = true;
  return frame;
}
Bytes encode(const std::vector<RootInput>& roots, const Recipe& recipe,
             const StageDispatcher& dispatcher, Limits limits) {
  require(!recipe.steps.empty() && recipe.steps.back().kind == StepKind::Stop,
          "recipe requires an explicit terminal STOP");
  require(recipe.steps.size() - 1 <= limits.max_nodes, "recipe exceeds node budget");
  GraphSession graph(roots, dispatcher, limits, recipe.policy_registry);
  for (std::size_t i = 0; i < recipe.steps.size(); ++i) {
    const auto& step = recipe.steps[i];
    if (step.kind == StepKind::Stop) {
      detail::validate_step(step, limits);
      require(i + 1 == recipe.steps.size(), "operation after STOP");
      return graph.stop();
    }
    graph.apply(step);
  }
  throw std::logic_error("unreachable missing STOP");
}
Decoded decode(std::span<const std::uint8_t> bytes, const StageDispatcher& dispatcher, Limits limits) {
  auto frame = detail::parse_frame(bytes, limits);
  require(frame.dispatcher == dispatcher.identity(), "native dispatcher version mismatch");
  auto active = std::move(frame.leaves);
  for (auto it = frame.nodes.rbegin(); it != frame.nodes.rend(); ++it) {
    const auto& node = *it;
    std::vector<Stream> outputs;
    for (std::size_t i = 0; i < node.outputs.size(); ++i) {
      require(active.contains(node.outputs[i]), "inverse node leaf is missing");
      match(active.at(node.outputs[i]), node.output_specs[i], limits);
      outputs.push_back(active.at(node.outputs[i]));
    }
    std::vector<Stream> restored;
    if (node.step.kind == StepKind::Stage)
      restored = dispatcher.inverse(node.step.stage, outputs, node.metadata);
    else if (node.step.kind == StepKind::SplitSegments)
      restored.push_back(concatenate(outputs, limits));
    else {
      std::vector<std::uint64_t> sizes;
      for (const auto& s : node.input_specs) sizes.push_back(s.bytes / s.width);
      restored = split(outputs.at(0), sizes);
    }
    require(restored.size() == node.step.inputs.size(), "inverse input port count mismatch");
    for (std::size_t i = 0; i < restored.size(); ++i) match(restored[i], node.input_specs[i], limits);
    for (const auto id : node.outputs) active.erase(id);
    for (std::size_t i = 0; i < restored.size(); ++i)
      require(active.emplace(node.step.inputs[i], std::move(restored[i])).second,
              "inverse graph duplicates input");
  }
  require(active.size() == frame.roots.size(), "inverse graph root count mismatch");
  Decoded result; result.original_numeric_bytes = frame.numeric_bytes;
  for (std::size_t i = 0; i < frame.roots.size(); ++i) {
    require(active.contains(static_cast<StreamId>(i)), "inverse graph root is missing");
    auto& stream = active.at(static_cast<StreamId>(i));
    match(stream, frame.roots[i].spec, limits);
    require(detail::sha256(stream.data) == frame.roots[i].digest, "restored root SHA-256 mismatch");
    result.roots.push_back({std::move(stream), frame.roots[i].numeric, frame.roots[i].logical_view});
  }
  return result;
}
BenchmarkResult benchmark(const std::vector<RootInput>& roots, const Recipe& recipe,
                          const StageDispatcher& dispatcher, std::uint32_t warmups,
                          std::uint32_t repeats, Limits limits) {
  require(warmups <= 31 && repeats > 0 && repeats <= 1001, "invalid timing repetition counts");
  BenchmarkResult result;
  for (std::uint32_t i = 0; i < warmups + repeats; ++i) {
    const auto begin_encode = std::chrono::steady_clock::now();
    auto frame = encode(roots, recipe, dispatcher, limits);
    const auto end_encode = std::chrono::steady_clock::now();
    auto restored = decode(frame, dispatcher, limits);
    const auto end_decode = std::chrono::steady_clock::now();
    verify(restored, roots); // Bitwise comparison is outside both timed regions.
    if (i >= warmups) {
      result.encode_nanoseconds.push_back(nanos(begin_encode, end_encode));
      result.decode_nanoseconds.push_back(nanos(end_encode, end_decode));
    }
    result.original_numeric_bytes = restored.original_numeric_bytes;
    result.frame = std::move(frame);
  }
  return result;
}
}  // namespace native_graph_v1
