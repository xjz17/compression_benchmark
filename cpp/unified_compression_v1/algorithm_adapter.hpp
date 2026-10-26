#pragma once

#include "api.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace unified_compression_v1::detail {

// Adapts the fixed baseline-86 C++ inventory to the unified API.  The payload
// returned here is the complete inner NGF1 frame; Registry adds the outer UCC1
// method binding and owns public timing.
class AlgorithmAdapter {
 public:
  AlgorithmAdapter();

  const std::vector<MethodInfo>& methods() const noexcept;

  Bytes encode(std::string_view canonical_name, const Column& column) const;

  // expected_name is the canonical outer binding (algorithm:LABEL).  The
  // inner NGF1 action is inspected and checked before any inverse is run.
  Column decode(std::string_view expected_name,
                std::span<const std::uint8_t> inner_frame) const;

 private:
  const MethodInfo& require_method(std::string_view canonical_name) const;
  std::vector<MethodInfo> methods_;
};

}  // namespace unified_compression_v1::detail
