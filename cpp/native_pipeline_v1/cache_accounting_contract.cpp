#include "catalog.hpp"
#include "stage_dispatch.hpp"
#include "task.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;

namespace {
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

class MetadataDispatcher final : public graph::StageDispatcher {
 public:
  explicit MetadataDispatcher(std::size_t metadata_bytes)
      : metadata_bytes_(metadata_bytes) {}
  std::string identity() const override { return "cache-accounting-metadata-v1"; }
  graph::StageResult forward(const graph::StageRequest&,
                             const std::vector<graph::Stream>& inputs) const override {
    require(inputs.size() == 1, "metadata fixture needs one input");
    return {{inputs.front()}, graph::Bytes(metadata_bytes_, 0xa5), {"int64"}};
  }
  std::vector<graph::Stream> inverse(
      const graph::StageRequest&, const std::vector<graph::Stream>& outputs,
      std::span<const std::uint8_t> metadata) const override {
    require(outputs.size() == 1 && metadata.size() == metadata_bytes_,
            "metadata fixture inverse shape changed");
    return outputs;
  }
 private:
  std::size_t metadata_bytes_;
};

graph::RootInput root(std::size_t bytes) {
  require(bytes != 0 && bytes % 8 == 0, "fixture root must contain int64 words");
  graph::Bytes data(bytes);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 0xffU);
  return {{graph::Kind::Numeric, 8, std::move(data)}, true, "int64"};
}

ngpipe::TaskRequest task_request(std::string id, std::size_t bytes) {
  ngpipe::TaskRequest out;
  out.task_id = std::move(id);
  out.roots = {root(bytes)};
  out.weights = {1.0, 0.0, 0.0};
  out.config.timing_warmups = 0;
  out.config.timing_repetitions = 1;
  out.config.max_policy_edits = 2;
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: native_pipeline_v1_cache_accounting_contract CATALOG");
    const auto catalog = ngpipe::ActionCatalog::load(std::filesystem::path(argv[1]));
    auto dispatcher = graph::make_default_dispatcher();

    // A tiny replay must be charged for its live state, not the 16 MiB maximum
    // metadata allowance. This catches the former fixed-per-entry charge.
    ngpipe::GraphTask small(catalog, *dispatcher, task_request("small-cache", 96));
    const auto small_cache = small.replay_cache_stats();
    require(small_cache.entries == 1 && small_cache.retained_bytes > 96 &&
                small_cache.retained_bytes < (1ULL << 20),
            "small replay was not charged by live retained capacity");

    // GraphSession accounting must include inverse metadata actually retained
    // by graph nodes, and the frame must remain independently reversible.
    constexpr std::size_t metadata_bytes = 2ULL << 20;
    MetadataDispatcher metadata_dispatcher(metadata_bytes);
    std::vector<graph::RootInput> metadata_roots{root(4096)};
    graph::GraphSession session(metadata_roots, metadata_dispatcher);
    const auto before = session.retained_bytes();
    graph::Step step;
    step.kind = graph::StepKind::Stage;
    step.inputs = {0};
    step.stage = {"metadata-fixture", "int64", {}};
    step.policy_action_id = 1;
    session.apply(step);
    const auto after = session.retained_bytes();
    require(after >= before + metadata_bytes,
            "live inverse metadata was absent from retained-byte accounting");
    const auto frame = session.stop();
    const auto decoded = graph::decode(frame, metadata_dispatcher);
    require(decoded.roots.size() == metadata_roots.size() &&
                decoded.roots[0].stream == metadata_roots[0].stream &&
                decoded.roots[0].original_numeric_payload ==
                    metadata_roots[0].original_numeric_payload &&
                decoded.roots[0].logical_view == metadata_roots[0].logical_view,
            "cache accounting changed complete-frame roundtrip semantics");

    // Two distinct live 34 MiB split states cannot both fit in the audited
    // 64 MiB cache. The second replay must evict the first while keeping the
    // reported aggregate at or below its cap.
    ngpipe::GraphTask bounded(catalog, *dispatcher,
                              task_request("bounded-cache", 34ULL << 20));
    const auto split = bounded.transition(bounded.initial_state(), 194);
    require(split.has_value(), "large cache fixture Split action was not legal");
    const auto bounded_cache = bounded.replay_cache_stats();
    require(bounded_cache.evictions >= 1 && bounded_cache.entries >= 1 &&
                bounded_cache.retained_bytes <= bounded_cache.byte_limit &&
                bounded_cache.byte_limit == (64ULL << 20),
            "replay cache did not enforce its 64 MiB retained-byte cap");

    std::cout << "native_pipeline_v1 cache accounting contract passed; small="
              << small_cache.retained_bytes << " metadata_delta=" << (after - before)
              << " bounded=" << bounded_cache.retained_bytes << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
