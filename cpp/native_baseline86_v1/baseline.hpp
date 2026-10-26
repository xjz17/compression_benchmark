#pragma once
#include "graph.hpp"
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace native_baseline86_v1 {
using Bytes=native_graph_v1::Bytes;
inline constexpr const char* profile_version="baseline86-fixed-profiles-20260919-v3";
inline constexpr const char* scaled_profile_version="baseline86-decimal-scaling-int64-exact-v1";
inline constexpr const char* scaling_configuration="typed_shortest_decimal_int64_exceptions_v1";
struct Unsupported : std::runtime_error {using std::runtime_error::runtime_error;};
struct Record {
  std::string label, implementation, source, configuration, limitations;
  std::vector<std::string> dtypes;
  bool linked=false, shared_or_project=false;
  // direct_current: a current same-label implementation or directly linked
  // upstream library; shared_alias: two public labels intentionally share one
  // implementation; project_proxy: reversible project substitute whose output
  // must never be presented as the named upstream algorithm; unsupported: no
  // admitted in-process encode/decode adapter in this build.
  // direct_generic identifies a fixed concrete implementation of a generic
  // algorithm label; direct_exact_subset additionally restricts valid inputs.
  std::string implementation_class="unsupported";
  bool primary_comparison=false;
};
const std::vector<Record>& inventory();  // Exactly 86 immutable catalog labels.
const Record& method(const std::string& label);
// Opt-in float front end. The original inventory and fixed profiles stay intact.
bool supports_decimal_scaling(const std::string& label);
std::vector<Record> scaling_inventory();
struct Block {
  std::string dtype;
  std::uint64_t row_count=0;
  Bytes values;  // Compact valid values, little-endian original numeric width.
  Bytes missing; // ceil(row_count/8); bit=1 means null, LSB first.
  bool operator==(const Block&) const = default;
};
std::uint32_t width(const std::string& dtype);
void validate(const Block&);
// All complete recovery bytes, including bitmap/row count/type/profile/frame,
// are retained. No Python, executable launch, or external payload cache.
Bytes encode(const std::string& label,const Block&);
struct ScalingStats {
  std::uint64_t scaled_value_count=0,exception_value_count=0;
  std::uint64_t scaling_metadata_bytes=0,raw_exception_bytes=0;
};
struct ScalingFrameInfo {
  std::string method,dtype;
  std::uint64_t row_count=0;
  std::uint16_t decimal_places=0;
  ScalingStats stats;
};
ScalingFrameInfo inspect_scaled(std::span<const std::uint8_t> frame);
// Decimal places is a frozen whole-column parameter (0..324), not a choice
// based on a codec's compression outcome. All exception/scale bytes are framed.
Bytes encode_scaled(const std::string& label,const Block&,std::uint16_t decimal_places,
                    ScalingStats* stats=nullptr);
Block decode(std::span<const std::uint8_t> frame);
struct Measurement {
  Bytes frame;
  std::uint64_t original_numeric_bytes=0;
  std::vector<std::uint64_t> compression_ns,decompression_ns;
  std::optional<ScalingStats> scaling_stats;
};
// Timer covers complete encode/decode APIs, including all per-block allocation,
// validation, graph dispatch, kernel and envelope work. Source I/O and correctness
// comparisons are excluded. Samples are raw ns; callers divide by values.size().
Measurement measure(const std::string& label,const Block&,unsigned warmups,unsigned repeats);
Measurement measure_scaled(const std::string& label,const Block&,std::uint16_t decimal_places,
                           unsigned warmups,unsigned repeats);
} // namespace native_baseline86_v1
