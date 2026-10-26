#pragma once

// LEGACY TEST FIXTURE ONLY. Never link this implementation into production.
// Frozen source SHA-256: 0233c7f26f4ea825125a8957669c85486f30c8b168513779a0b3bb1478d33c42
// Original header SHA-256: 5dd9796e9276f06286dbeeaa05740f80e5591b6952244d467dc6ea13c46a8529
// The comment makes this header a distinct #pragma once identity while the
// test remaps its namespace; declarations are otherwise unchanged.

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace bos_float_exception {

using Bytes = std::vector<std::uint8_t>;

enum class DType : std::uint8_t {
  Float32 = 4,
  Float64 = 8,
};

enum class Mode : std::uint8_t {
  Raw = 0,
  Exception = 1,
};

using EncodeInt = std::function<Bytes(std::span<const std::int64_t>)>;
using DecodeInt =
    std::function<std::vector<std::int64_t>(std::span<const std::uint8_t>)>;

struct EncodeOptions {
  // Each listed precision is really encoded on the deterministic pilot before
  // the smallest complete exception frame is selected. Repeated entries are
  // rejected.
  std::vector<int> candidate_precisions{0, 1, 2, 3, 4, 6};
  // For blocks larger than this value, select p on an evenly spaced sample
  // that includes both endpoints, then encode the full block once.  Zero
  // retains exhaustive full-block search.  Blocks no larger than the pilot
  // evaluate all p on the full block and reuse the winning wire.
  std::uint32_t pilot_values = 256;
  // When set, evaluate only this precision.  This is useful when precision is
  // fixed by an outer, already-accounted-for contract; pilot_values is ignored.
  std::optional<int> fixed_precision;
};

struct Stats {
  // Raw mode has no decimal prediction and reports -1.
  int selected_p = -1;
  std::uint32_t exceptions = 0;
  Mode mode = Mode::Raw;
};

// input and the returned bytes from decode are raw little-endian IEEE-754
// words.  A block contains at most 10,000 values.  The callbacks are the only
// dependency on the integer codec and may wrap the latest BOS int64 core.
Bytes encode(std::span<const std::uint8_t> input,
             DType dtype,
             const EncodeInt& encode_int,
             Stats* stats = nullptr,
             const EncodeOptions& options = {});

// Decoding is self-contained apart from the matching integer-codec callback.
// Malformed, non-canonical, truncated, or trailing data is rejected.
Bytes decode(std::span<const std::uint8_t> wire,
             const DecodeInt& decode_int,
             Stats* stats = nullptr);

}  // namespace bos_float_exception
