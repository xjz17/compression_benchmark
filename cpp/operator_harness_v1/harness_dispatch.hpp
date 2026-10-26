#pragma once

#include "stage_dispatch.hpp"

#include <memory>
#include <string>
#include <vector>

namespace operator_harness_v1 {

inline constexpr const char* kBytePlaneAction = "harness.a048.byte_planes.v1";

struct HarnessCapability {
  native_graph_v1::StageCapability stage;
  bool enabled_by_default = true;
  bool context_heavy = false;
};

// A policy wrapper over the existing native dispatcher. NNCP is always denied.
// PAQ and CMIX are denied unless explicitly enabled for a non-representative run.
class HarnessDispatcher final : public native_graph_v1::StageDispatcher {
 public:
  explicit HarnessDispatcher(bool allow_context_heavy = false);
  std::string identity() const override;
  native_graph_v1::StageResult forward(
      const native_graph_v1::StageRequest&,
      const std::vector<native_graph_v1::Stream>&) const override;
  std::vector<native_graph_v1::Stream> inverse(
      const native_graph_v1::StageRequest&,
      const std::vector<native_graph_v1::Stream>&,
      std::span<const std::uint8_t>) const override;

 private:
  void enforce_policy(const native_graph_v1::StageRequest&) const;
  bool allow_context_heavy_ = false;
  std::unique_ptr<native_graph_v1::StageDispatcher> inner_;
};

std::vector<HarnessCapability> harness_capabilities();

}  // namespace operator_harness_v1
