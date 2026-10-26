#pragma once

#include "graph.hpp"

#include <span>
#include <vector>

namespace operator_harness_v2 {

// True only for source-accurate focal stages implemented by special_stages.cpp.
// Invalid parameters are still handled here and fail closed in forward/inverse;
// unsupported actions return false so the ordinary dispatcher can try them.
bool handles_special(const native_graph_v1::StageRequest& request);

native_graph_v1::StageResult forward_special(
    const native_graph_v1::StageRequest& request,
    const std::vector<native_graph_v1::Stream>& inputs);

std::vector<native_graph_v1::Stream> inverse_special(
    const native_graph_v1::StageRequest& request,
    const std::vector<native_graph_v1::Stream>& outputs,
    std::span<const std::uint8_t> metadata);

}  // namespace operator_harness_v2
