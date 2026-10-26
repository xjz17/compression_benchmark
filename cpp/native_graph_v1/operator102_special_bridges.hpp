#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace native_graph_v1::operator102 {

using Bytes = std::vector<std::uint8_t>;

// A bridge exposes the state immediately before and after the focal catalog
// action.  target_after_ports and inverse_metadata are the complete charged
// output state for entropy/Markov measurement.  target_inverse must reproduce
// target_before, while fully_restored must reproduce canonical_raw.
struct SpecialBridgeResult {
  std::string implementation_id;
  Bytes target_before;
  std::string target_before_kind;
  std::uint32_t target_before_width = 1;
  std::vector<Bytes> target_after_ports;
  // Ordered metadata for every prefix stage.  It is part of both compared
  // recoverable states; target metadata is present only in the full state.
  std::vector<Bytes> prefix_inverse_metadata;
  Bytes inverse_metadata;
  std::vector<std::string> output_logical_views;
  std::vector<std::string> output_kinds;
  std::vector<std::uint32_t> output_widths;
  Bytes target_inverse;
  Bytes fully_restored;
  std::string contract;
};

struct SpecialBridgeTiming {
  bool bitexact = false;
  // Kernel-only diagnostic view.
  std::vector<std::uint64_t> encode_ns;
  std::vector<std::uint64_t> decode_ns;
  // Formal ranking view: real bridge kernel plus O2R1 pack/parse.
  std::vector<std::uint64_t> full_encode_ns;
  std::vector<std::uint64_t> full_decode_ns;
};

bool supports_special_bridge(std::string_view implementation_id);

// canonical_raw contains the current135 typed column payload in canonical
// little-endian bytes.  A098 is interpreted as int64 and A099 as float64;
// the remaining bridges operate on the exact bytes named by their catalog
// entry.  The function throws on any semantic, framing, or round-trip error.
SpecialBridgeResult run_special_bridge(
    std::string_view implementation_id,
    std::span<const std::uint8_t> canonical_raw);

// Times the real full reversible recipe used by each special bridge.  Prefix
// stages and the focal stage are included in encode; the actual focal inverse
// and prefix inverses are included in decode.  The two directions are timed
// independently inside each bridge and are never inferred by splitting a
// combined duration.
SpecialBridgeTiming benchmark_special_bridge(
    std::string_view implementation_id,
    std::span<const std::uint8_t> canonical_raw,
    const std::vector<Bytes>& auxiliaries,
    const Bytes* adapter,
    std::size_t warmups,
    std::size_t repeats);

}  // namespace native_graph_v1::operator102
