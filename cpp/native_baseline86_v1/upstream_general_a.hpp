#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace native_baseline86_v1::upstream_general_a {

using Bytes = std::vector<std::uint8_t>;

struct Descriptor {
  std::string label;
  std::string implementation;
  std::string source;
  std::string configuration;
  std::string limitations;
  bool available = false;
};

// Fixed, predeclared profiles.  The returned list is always in the same order
// and contains all ten labels, including architecture/build-disabled entries.
const std::vector<Descriptor>& inventory();
const Descriptor& descriptor(const std::string& label);

// Payload byte 0 is part of the adapter wire: 0 stores the input verbatim and
// 1 stores the named upstream stream.  The verbatim branch is selected only
// when the direct upstream result is not smaller (or for an empty input).
// It is never represented as a successful upstream-compressed stream.
Bytes encode(const std::string& label, std::span<const std::uint8_t> input);
Bytes decode(const std::string& label, std::span<const std::uint8_t> payload,
             std::size_t expected_size);

}  // namespace native_baseline86_v1::upstream_general_a
