#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace tsfile::decode {

struct ByteCursor {
  const std::uint8_t* p = nullptr;
  std::size_t len = 0;
  std::size_t pos = 0;
  bool ok = true;

  bool has(std::size_t n) const { return pos + n <= len; }
  std::uint8_t get_u8() {
    if (!has(1)) {
      ok = false;
      return 0;
    }
    return p[pos++];
  }
  std::uint32_t read_uvarint() {
    std::uint32_t v = 0;
    int s = 0;
    for (;;) {
      std::uint8_t b = get_u8();
      if (!ok) return 0;
      v |= (b & 0x7fu) << s;
      if ((b & 0x80u) == 0) return v;
      s += 7;
    }
  }
  std::int32_t read_i32_be() {
    if (!has(4)) {
      ok = false;
      return 0;
    }
    std::uint32_t u = static_cast<std::uint32_t>(p[pos]) << 24 |
                      static_cast<std::uint32_t>(p[pos + 1]) << 16 |
                      static_cast<std::uint32_t>(p[pos + 2]) << 8 |
                      static_cast<std::uint32_t>(p[pos + 3]);
    pos += 4;
    return static_cast<std::int32_t>(u);
  }
  std::int64_t read_i64_be() {
    if (!has(8)) {
      ok = false;
      return 0;
    }
    std::uint64_t u = 0;
    for (int i = 0; i < 8; i++) {
      u = (u << 8) | static_cast<std::uint64_t>(p[pos + static_cast<std::size_t>(i)]);
    }
    pos += 8;
    return static_cast<std::int64_t>(u);
  }
  float read_f32_be() {
    std::int32_t i = read_i32_be();
    float f;
    static_assert(sizeof(float) == sizeof(std::int32_t));
    std::memcpy(&f, &i, sizeof(float));
    return f;
  }
  /** readIntLittleEndianPaddedOnBitWidth(buffer, bitWidth) */
  std::int32_t read_int_le_padded(int bit_width) {
    int n = (bit_width + 7) / 8;
    std::int32_t result = 0;
    int offset = 0;
    while (n-- > 0) {
      int ch = static_cast<int>(get_u8());
      if (!ok) return 0;
      result += ch << offset;
      offset += 8;
    }
    return result;
  }
  std::int64_t read_long_le_padded(int bit_width) {
    int n = (bit_width + 7) / 8;
    std::int64_t result = 0;
    int offset = 0;
    while (n-- > 0) {
      int ch = static_cast<int>(get_u8());
      if (!ok) return 0;
      result += static_cast<std::int64_t>(ch) << offset;
      offset += 8;
    }
    return result;
  }
};

/** org.apache.tsfile.utils.BytesUtils.bytesToInt(result, pos, width) — bit offset pos. */
inline std::int64_t bytes_to_long_at_bits(const std::uint8_t* bytes, int pos_bits, int width) {
  std::int64_t ret = 0;
  int cnt = pos_bits & 7;
  int index = pos_bits >> 3;
  while (width > 0) {
    int m = (width + cnt >= 8) ? (8 - cnt) : width;
    width -= m;
    ret <<= m;
    int y = static_cast<int>(bytes[index] & (0xff >> cnt));
    y = (y & 0xff) >> (8 - cnt - m);
    ret |= static_cast<std::int64_t>(y & 0xff);
    cnt += m;
    if (cnt == 8) {
      cnt = 0;
      index++;
    }
  }
  return ret;
}

inline std::int32_t bytes_to_int_at_bits(const std::uint8_t* bytes, int pos_bits, int width) {
  std::int32_t ret = 0;
  int cnt = pos_bits & 7;
  int index = pos_bits >> 3;
  while (width > 0) {
    int m = (width + cnt >= 8) ? (8 - cnt) : width;
    width -= m;
    ret <<= m;
    int y = static_cast<int>(bytes[index] & (0xff >> cnt));
    y = (y & 0xff) >> (8 - cnt - m);
    ret |= y & 0xff;
    cnt += m;
    if (cnt == 8) {
      cnt = 0;
      index++;
    }
  }
  return ret;
}

inline int ceil_bits_to_bytes(int bits) { return (bits + 7) / 8; }

}  // namespace tsfile::decode
