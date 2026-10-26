#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace unified_rle_numeric {
using Bytes = std::vector<std::uint8_t>;

// URL1 compresses original symbol bits. Width is in bytes, not a numeric cast.
Bytes encode_rle(std::span<const std::uint8_t> raw, unsigned width);
Bytes decode_rle(std::span<const std::uint8_t> frame,
                 std::size_t max_output_bytes = 256u * 1024u * 1024u);
bool is_rle_frame(std::span<const std::uint8_t> frame);
unsigned rle_symbol_width(std::span<const std::uint8_t> frame);

// Complete floating RLE frontend. URF1 records decimal scale vs raw-bit fallback;
// URL1 always encodes the real payload. Scaled mode must restore every input bit.
Bytes encode_rle_scaled(std::span<const std::uint8_t> raw, unsigned width, int requested_scale);
Bytes decode_rle_scaled(std::span<const std::uint8_t> frame,
                       std::size_t max_output_bytes = 256u * 1024u * 1024u);
bool is_scaled_rle_frame(std::span<const std::uint8_t> frame);
unsigned scaled_rle_symbol_width(std::span<const std::uint8_t> frame);

// URB1 preserves RLBE delta / width / residual-payload semantics. Only the
// sequence of residual bit widths uses the shared URL1 run-length encoder.
Bytes encode_rlbe(std::span<const std::uint8_t> raw, unsigned width,
                  bool delta_enabled = true);
Bytes encode_rlbe_float32_arithmetic(std::span<const std::uint8_t> raw);
Bytes decode_rlbe(std::span<const std::uint8_t> frame,
                  std::size_t max_output_bytes = 256u * 1024u * 1024u);
bool is_rlbe_frame(std::span<const std::uint8_t> frame);
unsigned rlbe_symbol_width(std::span<const std::uint8_t> frame);
unsigned rlbe_delta_mode(std::span<const std::uint8_t> frame);

template<class T> Bytes raw_bytes(std::span<const T> values) {
  Bytes raw(values.size_bytes());
  if (!raw.empty()) std::memcpy(raw.data(), values.data(), raw.size());
  if constexpr (std::endian::native == std::endian::big)
    for (std::size_t i=0; i<raw.size(); i+=sizeof(T))
      std::reverse(raw.begin()+i, raw.begin()+i+sizeof(T));
  return raw;
}
template<class T> std::vector<T> typed_values(Bytes raw) {
  if (raw.size()%sizeof(T)) throw std::runtime_error("numeric symbol width mismatch");
  if constexpr (std::endian::native == std::endian::big)
    for (std::size_t i=0; i<raw.size(); i+=sizeof(T))
      std::reverse(raw.begin()+i, raw.begin()+i+sizeof(T));
  std::vector<T> out(raw.size()/sizeof(T));
  if (!raw.empty()) std::memcpy(out.data(),raw.data(),raw.size());
  return out;
}
template<class T> Bytes encode_values(std::span<const T> values) {
  return encode_rle(raw_bytes(values),sizeof(T));
}
template<class T> bool decode_values(std::span<const std::uint8_t> frame,
                                     std::vector<T>& out) {
  out.clear();
  try {
    if (rle_symbol_width(frame)!=sizeof(T)) return false;
    out=typed_values<T>(decode_rle(frame));return true;
  } catch (const std::exception&) {out.clear();return false;}
}
template<class T> bool decode_rlbe_values(std::span<const std::uint8_t> frame,
                                          std::vector<T>& out,
                                          unsigned expected_mode=1) {
  out.clear();
  try {
    if (rlbe_symbol_width(frame)!=sizeof(T) || rlbe_delta_mode(frame)!=expected_mode)
      return false;
    out=typed_values<T>(decode_rlbe(frame));return true;
  } catch (const std::exception&) {out.clear();return false;}
}
}  // namespace unified_rle_numeric
