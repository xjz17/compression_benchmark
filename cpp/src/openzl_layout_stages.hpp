#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Source-informed scalar implementations of OpenZL layout nodes. This module's
// OLS1 transport is NOT the OpenZL wire format. No original payload is kept in
// metadata. All integer/bit-pattern streams use explicit little-endian bytes.
namespace openzl_layout {
using Bytes = std::vector<std::uint8_t>;
enum class Kind : std::uint8_t { Serial = 1, Struct = 2, Numeric = 3 };
enum class Operation : std::uint8_t {
  Raw = 0, Transpose = 1, SplitStruct = 2, FloatDeconstruct = 3,
  BitSplit = 4, Bitunpack = 5, Concat = 6, SplitSegments = 7,
  SplitByRange = 8
};
struct Stream {
  Kind kind = Kind::Serial;
  std::uint32_t width = 1;
  Bytes data;
  bool operator==(const Stream&) const = default;
};
struct SplitResult {
  Operation operation = Operation::Raw;
  Kind input_kind = Kind::Serial;
  std::uint32_t input_width = 1;
  std::uint64_t input_bytes = 0;
  Bytes side;
  std::vector<Stream> streams;
};

Bytes serialize(const SplitResult&);
SplitResult deserialize(std::span<const std::uint8_t>);
SplitResult raw_bundle(std::vector<Stream>);
SplitResult transpose(const Stream&);
SplitResult split_struct(std::span<const std::uint8_t>,
                         std::span<const std::uint32_t> field_widths);
SplitResult float_deconstruct(const Stream&, const std::string& format);
SplitResult bitsplit(const Stream&, std::span<const std::uint8_t> widths);
SplitResult bitsplit_top8(const Stream&);
SplitResult bitsplit_fp(const Stream&);
SplitResult bitsplit_bf16(const Stream&);
SplitResult bitunpack(std::span<const std::uint8_t>, unsigned bits);
SplitResult concat(const std::vector<Stream>&);
SplitResult split_segments(const Stream&, std::span<const std::uint64_t> sizes);
SplitResult split_by_range(const Stream&, std::uint64_t minimum_segment = 0);
std::vector<Stream> inverse(const SplitResult&);
} // namespace openzl_layout
