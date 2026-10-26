#pragma once
#include "graph.hpp"

namespace native_graph_v1 {
struct StageCapability {
  std::string action;
  std::string source;
  std::string semantics;
  bool multiple_outputs = false;
};
// This inventory means linked in-process entry points.  It is platform
// specific (the real NNCP entry point is available only in its supported
// Linux x86-64 build) and is not by itself formal performance eligibility.
// The frozen catalog gate separately checks every concrete configuration;
// unknown actions and parameters fail closed.
const std::vector<StageCapability>& native_stage_capabilities();
class NativeStageDispatcher final : public StageDispatcher {
 public:
  std::string identity() const override;
  StageResult forward(const StageRequest&, const std::vector<Stream>&) const override;
  std::vector<Stream> inverse(const StageRequest&, const std::vector<Stream>&,
                             std::span<const std::uint8_t>) const override;
};
std::unique_ptr<StageDispatcher> make_default_dispatcher();
}  // namespace native_graph_v1
