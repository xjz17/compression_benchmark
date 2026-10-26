#pragma once

#include "stage_dispatch.hpp"

namespace native_graph_v1::operator102_measure {

struct ExactRawFallbackInfo {
  bool wrapper = false;
  bool used_raw_fallback = false;
  std::uint64_t original_bytes = 0;
  std::string reason;
};

// Decode the measurement-only, lossless domain wrapper attached to strict
// floating scaling stages.  The wrapper is not an identity substitute: it
// executes the real target when its exact domain admits the block and records
// a charged selector plus the original bytes otherwise.
ExactRawFallbackInfo inspect_exact_raw_fallback_metadata(
    std::span<const std::uint8_t> metadata);

// Adds the already source-separated exact actions whose APIs predate the
// native graph dispatcher. Unknown actions are delegated to the ordinary
// native dispatcher; there is no identity fallback.
class Operator102Dispatcher final : public StageDispatcher {
 public:
  std::string identity() const override;
  StageResult forward(const StageRequest&, const std::vector<Stream>&) const override;
  std::vector<Stream> inverse(const StageRequest&, const std::vector<Stream>&,
                              std::span<const std::uint8_t>) const override;

 private:
  NativeStageDispatcher native_;
};

}  // namespace native_graph_v1::operator102_measure
