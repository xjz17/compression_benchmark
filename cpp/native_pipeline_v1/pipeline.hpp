#pragma once

#include "features.hpp"
#include "graph.hpp"
#include "policy.hpp"
#include <array>
#include <filesystem>
#include <optional>

namespace wc::pipeline_v1 {
namespace policy = weighted_grpo::native_policy_v1;
inline constexpr std::uint32_t action_count = 6341, stop_action = 0;
inline constexpr const char* registry_fingerprint =
    "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd";
inline constexpr const char* execution_scope = "native-pipeline-v1-bitpack-append-only-development";

policy::RuntimeIdentity runtime_identity();
// 0 plus A020 bytes/int64 configurations at all 32 active-flow slots. This is
// an explicit development capability list, not the complete production mask.
std::vector<std::uint32_t> supported_action_ids();

struct Request {
  std::vector<native_graph_v1::RootInput> roots;
  std::array<double, 3> weights{1.0, 0.0, 0.0};
  // Exactly 6341 binary entries; every enabled ID must have a native mapping.
  // Unsupported enabled IDs are errors, never silently masked or remapped.
  std::vector<std::uint8_t> enabled_actions;
  // Optional independently constructed first 49-D observation for parity.
  // Native features are always recomputed inside the timed region and used.
  std::optional<std::array<double, 49>> expected_initial_features;
  native_graph_v1::Limits limits;
  std::uint32_t max_policy_steps = 64;
};
struct Timings {
  std::uint64_t compression_ns = 0, decompression_ns = 0;
  std::uint64_t feature_ns = 0, policy_ns = 0;
  std::uint64_t selection_graph_ns = 0, final_encode_ns = 0;
};
struct Result {
  native_graph_v1::Bytes frame;
  std::uint64_t original_numeric_bytes = 0;
  native_graph_v1::Recipe recipe;
  std::vector<std::uint32_t> selected_action_ids;
  Timings timing;
  std::optional<double> compression_ratio() const;
  std::optional<double> compression_ns_per_byte() const;
  std::optional<double> decompression_ns_per_byte() const;
};
struct BenchmarkResult {
  Result last;
  std::vector<Timings> samples;
};

class Pipeline {
 public:
  // Model loading and integrity/identity verification occur once, outside
  // repeated input-to-frame timing. No checkpoint migration is performed.
  static Pipeline load(const std::filesystem::path& model);
  Result compress(const Request&, const native_graph_v1::StageDispatcher&) const;
  BenchmarkResult benchmark(const Request&, const native_graph_v1::StageDispatcher&,
                            std::uint32_t warmups, std::uint32_t repeats) const;
 private:
  explicit Pipeline(policy::Policy value) : policy_(std::move(value)) {}
  policy::Policy policy_;
};
}  // namespace wc::pipeline_v1
