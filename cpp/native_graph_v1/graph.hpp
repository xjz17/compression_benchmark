#pragma once

#include "openzl_layout_stages.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace native_graph_v1 {
using Bytes = openzl_layout::Bytes;
using Stream = openzl_layout::Stream;
using Kind = openzl_layout::Kind;
using StreamId = std::uint32_t;

struct StageRequest {
  std::string action;
  std::string profile;
  std::map<std::string, std::string> parameters;
  bool operator==(const StageRequest&) const = default;
};
struct StageResult {
  std::vector<Stream> outputs;
  // Only inverse metadata, never a cached copy of original input payload.
  Bytes metadata;
  std::vector<std::string> output_logical_views;
};
class StageDispatcher {
 public:
  virtual ~StageDispatcher() = default;
  virtual std::string identity() const = 0;
  virtual StageResult forward(const StageRequest&, const std::vector<Stream>&) const = 0;
  virtual std::vector<Stream> inverse(const StageRequest&, const std::vector<Stream>&,
                                    std::span<const std::uint8_t> metadata) const = 0;
};
struct Limits {
  std::uint32_t max_nodes = 64;
  std::uint32_t max_active_streams = 32;
  std::uint32_t max_ordinary_depth = 10;
  std::uint64_t max_stream_bytes = 256ULL << 20;
  std::uint64_t max_working_bytes = 512ULL << 20;
  std::uint64_t max_frame_bytes = 1ULL << 30;
  std::uint64_t max_metadata_bytes = 16ULL << 20;
};
struct RootInput {
  Stream stream;
  // False only for auxiliary metadata streams (e.g. null masks). All bytes,
  // including auxiliary streams, remain in the complete recoverable frame.
  bool original_numeric_payload = true;
  std::string logical_view;
};
enum class StepKind : std::uint8_t { Stage = 1, SplitSegments = 2, Concat = 3, Stop = 4 };
struct Step {
  StepKind kind = StepKind::Stage;
  std::vector<StreamId> inputs;
  StageRequest stage;
  // Exact continuous segment sizes in elements, not bytes; zero-length
  // segments are allowed and explicitly preserved. Their sum must be exact.
  std::vector<std::uint64_t> segment_sizes;
  std::int64_t policy_action_id = -1;
};
struct Recipe {
  std::string policy_registry;
  // Explicit terminal Stop is required by encode(); GraphSession::stop()
  // provides the equivalent operation to an incremental native policy.
  std::vector<Step> steps;
};

struct FlowInfo {
  std::uint32_t ordinary_depth = 0;
  std::string parent_action;
  std::string logical_view;
  std::int64_t policy_action_id = -1;
};

class GraphSession {
 public:
  GraphSession(std::vector<RootInput>, const StageDispatcher&, Limits = {},
               std::string policy_registry = {});
  ~GraphSession();
  GraphSession(GraphSession&&) noexcept;
  GraphSession& operator=(GraphSession&&) noexcept;
  GraphSession(const GraphSession&) = delete;
  GraphSession& operator=(const GraphSession&) = delete;
  // Backward-compatible view of every current leaf. Per-flow closed leaves
  // remain here because they are part of the complete recoverable frame.
  const std::map<StreamId, Stream>& active_streams() const;
  // Policy-visible leaves that may still be consumed by a graph operation.
  // Closing a flow is scheduling metadata only: it emits no graph node and
  // does not change the complete-frame wire representation.
  std::vector<StreamId> open_stream_ids() const;
  std::size_t open_stream_count() const;
  bool is_closed(StreamId) const;
  void close_flow(StreamId);
  std::uint32_t ordinary_depth(StreamId) const;
  const FlowInfo& flow_info(StreamId) const;
  std::size_t node_count() const;
  std::uint64_t original_numeric_bytes() const;
  // Conservative capacity-based estimate of memory retained by this live
  // session. It includes the wrapper/implementation, active payloads, graph
  // nodes, inverse metadata, roots, logical views and container storage.
  // Standard C++ containers do not expose allocator bookkeeping, so this is
  // an audited cache-accounting estimate rather than allocator-exact RSS; it
  // may overcount small-string storage and tree-node overhead.
  std::uint64_t retained_bytes() const;
  // IDs are assigned monotonically, roots starting at 0; returns new IDs.
  // Strong exception guarantee: a rejected action does not mutate state.
  std::vector<StreamId> apply(const Step&);
  // Closes the whole graph, retaining every active leaf exactly once.
  Bytes stop();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Decoded {
  std::vector<RootInput> roots;
  std::uint64_t original_numeric_bytes = 0;
};
Bytes encode(const std::vector<RootInput>&, const Recipe&, const StageDispatcher&,
             Limits = {});
Decoded decode(std::span<const std::uint8_t>, const StageDispatcher&, Limits = {});

struct BenchmarkResult {
  Bytes frame;
  std::uint64_t original_numeric_bytes = 0;
  std::vector<std::uint64_t> encode_nanoseconds;
  std::vector<std::uint64_t> decode_nanoseconds;
};
// Inputs and recipe are already in memory. Each encode sample includes graph
// scheduling, kernels, metadata, complete framing and checksums; each decode
// sample includes parsing/checksums, inverse scheduling and all inverse kernels.
// Model inference, file I/O and verification comparisons are outside samples.
BenchmarkResult benchmark(const std::vector<RootInput>&, const Recipe&,
                          const StageDispatcher&, std::uint32_t warmups,
                          std::uint32_t repeats, Limits = {});
}  // namespace native_graph_v1
