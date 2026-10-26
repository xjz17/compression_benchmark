#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace native_baseline86_v1::upstream_general_b {

using Bytes = std::vector<std::uint8_t>;

struct Descriptor {
  std::string_view label;
  std::string_view implementation;
  std::string_view source;
  std::string_view configuration;
  std::string_view limitations;
};

// These descriptors name same-label upstream C/C++ implementations.  The
// returned payload is a one-byte adapter tag followed by either an upstream
// stream (tag 1) or the original bytes (tag 0).  The verbatim branch is used
// only for empty input or when a successful upstream encoding is not smaller.
std::span<const Descriptor> descriptors();
bool supports(std::string_view label);
Bytes encode(std::string_view label, std::span<const std::uint8_t> input);
Bytes decode(std::string_view label, std::span<const std::uint8_t> payload,
             std::size_t expected_size);

}  // namespace native_baseline86_v1::upstream_general_b
