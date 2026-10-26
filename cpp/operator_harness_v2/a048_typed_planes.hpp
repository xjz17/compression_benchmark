#pragma once

#include "graph.hpp"

#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace operator_harness_v2 {

inline constexpr std::string_view kA048TypedPlaneAction =
    "harness.a048.byte_planes.v1";

class A048InputDomainError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

bool handles_a048_typed_planes(const native_graph_v1::StageRequest& request);

// Pure typed-port exposure of the existing BUFF byte-plane kernel.  The input
// is a numeric uint64 stream.  The outputs are independent serial byte-plane
// ports; inverse metadata contains shape only, never source payload bytes.
native_graph_v1::StageResult forward_a048_typed_planes(
    const native_graph_v1::StageRequest& request,
    const std::vector<native_graph_v1::Stream>& inputs);

std::vector<native_graph_v1::Stream> inverse_a048_typed_planes(
    const native_graph_v1::StageRequest& request,
    const std::vector<native_graph_v1::Stream>& outputs,
    std::span<const std::uint8_t> metadata);

}  // namespace operator_harness_v2
