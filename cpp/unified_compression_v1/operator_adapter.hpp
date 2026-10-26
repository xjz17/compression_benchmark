#pragma once

#include "api.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace unified_compression_v1::detail {

// In-memory bridge from the frozen 102-row operator-tree catalog to the
// unified column API.  Encoding is catalog-bound; decoding is frame-only and
// therefore does not require the catalog file that was used by the encoder.
class OperatorAdapter {
 public:
  explicit OperatorAdapter(const std::string& catalog_path);

  const std::vector<MethodInfo>& methods() const noexcept;
  Bytes encode(std::string_view canonical_name, const Column&,
               nlohmann::json* execution_audit = nullptr) const;

  // expected_name is deliberately canonical (operator:Axxx).  The immutable
  // execution binding in the frame must name the same implementation.
  static Column decode(std::string_view expected_name,
                       std::span<const std::uint8_t> frame);

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

}  // namespace unified_compression_v1::detail
