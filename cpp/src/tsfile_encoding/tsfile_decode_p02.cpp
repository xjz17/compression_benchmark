#include "../unified_rle_numeric.hpp"
#include "../minimum_for.hpp"
#include "tsfile_decode_p02.hpp"

#include "tsfile_decode_common.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace tsfile::p02 {

namespace {

using tsfile::decode::ByteCursor;
using tsfile::decode::bytes_to_int_at_bits;
using tsfile::decode::ceil_bits_to_bytes;

constexpr int kRleMinRepeated = 8;

void unpack8_values(int width, const std::uint8_t* buf, int offset, std::int32_t* values) {
  int byte_idx = offset;
  std::uint64_t buffer = 0;
  int total_bits = 0;
  int value_idx = 0;
  while (value_idx < 8) {
    while (total_bits < width) {
      if (byte_idx >= offset + width) return;
      buffer = (buffer << 8) | static_cast<std::uint64_t>(buf[byte_idx] & 0xff);
      byte_idx++;
      total_bits += 8;
    }
    while (total_bits >= width && value_idx < 8) {
      values[value_idx] = static_cast<std::int32_t>(buffer >> (total_bits - width));
      value_idx++;
      total_bits -= width;
      if (total_bits > 0) {
        buffer &= (1ull << total_bits) - 1ull;
      } else {
        buffer = 0;
      }
    }
  }
}

struct IntRleDecoder {
  ByteCursor outer;
  std::vector<std::uint8_t> pattern_bytes;
  ByteCursor inner{};
  int bit_width = 0;
  int current_count = 0;
  int current_value = 0;
  std::vector<std::int32_t> current_buffer;
  int bit_packing_num = 0;
  bool pkg_header_read = false;
  enum class Mode { Rle, BitPacked } mode = Mode::Rle;

  void reset_outer(std::span<const std::uint8_t> enc) {
    outer = ByteCursor{enc.data(), enc.size(), 0, true};
    pattern_bytes.clear();
    pkg_header_read = false;
    current_count = 0;
  }

  bool read_length_and_width() {
    std::uint32_t length = outer.read_uvarint();
    if (!outer.ok || length == 0) return false;
    if (!outer.has(length)) {
      outer.ok = false;
      return false;
    }
    pattern_bytes.assign(outer.p + outer.pos, outer.p + outer.pos + length);
    outer.pos += length;
    inner = ByteCursor{pattern_bytes.data(), pattern_bytes.size(), 0, true};
    bit_width = static_cast<int>(inner.get_u8());
    if (!inner.ok) return false;
    pkg_header_read = true;
    current_buffer.clear();
    return true;
  }

  bool read_number_rle() {
    current_value = inner.read_int_le_padded(bit_width);
    return inner.ok;
  }

  void unpack_all_bitpacked(int bit_packed_group_count, int last_bit_packed_num) {
    int total_count = (bit_packed_group_count - 1) * kRleMinRepeated + last_bit_packed_num;
    bit_packing_num = total_count;
    current_buffer.resize(static_cast<std::size_t>(total_count));
    int bytes_to_read = bit_packed_group_count * bit_width;
    if (static_cast<std::size_t>(bytes_to_read) > inner.len - inner.pos) {
      inner.ok = false;
      return;
    }
    int k = 0;
    int idx = static_cast<int>(inner.pos);
    while (k < total_count) {
      std::int32_t tv[8];
      unpack8_values(bit_width, pattern_bytes.data(), idx, tv);
      for (int t = 0; t < 8 && k < total_count; t++) current_buffer[static_cast<std::size_t>(k++)] = tv[t];
      idx += bit_width;
    }
    inner.pos = static_cast<std::size_t>(idx);
  }

  bool read_next() {
    std::uint32_t header = inner.read_uvarint();
    if (!inner.ok) return false;
    if ((header & 1u) == 0) {
      mode = Mode::Rle;
      current_count = static_cast<int>(header >> 1);
      return read_number_rle();
    }
    mode = Mode::BitPacked;
    int bit_packed_group_count = static_cast<int>(header >> 1);
    int last_bit_packed_num = static_cast<int>(inner.get_u8());
    if (!inner.ok) return false;
    unpack_all_bitpacked(bit_packed_group_count, last_bit_packed_num);
    current_count = bit_packing_num;
    return inner.ok;
  }

  bool has_next_pkg() { return current_count > 0 || inner.pos < inner.len; }

  std::int32_t read_int() {
    if (!pkg_header_read) {
      if (!read_length_and_width()) {
        outer.ok = false;
        return 0;
      }
    }
    if (current_count == 0) {
      if (!read_next()) {
        outer.ok = false;
        return 0;
      }
    }
    current_count--;
    std::int32_t result = 0;
    if (mode == Mode::Rle) {
      result = current_value;
    } else {
      result = current_buffer[static_cast<std::size_t>(bit_packing_num - current_count - 1)];
    }
    if (!has_next_pkg()) pkg_header_read = false;
    return result;
  }

  bool has_more_outer() const { return outer.pos < outer.len && outer.ok; }

  bool finished() const {
    return outer.ok && outer.pos >= outer.len && !pkg_header_read && current_count == 0;
  }
};

bool decode_rle_int_from_bytes(std::span<const std::uint8_t> enc, std::vector<std::int32_t>& out) {
  out.clear();
  IntRleDecoder dec;
  dec.reset_outer(enc);
  // Encoder prepends one uvarint(length) then one inner byte_cache; outer is fully consumed by
  // read_length_and_width, so every value must be read in an inner loop before leaving this segment.
  while (dec.has_more_outer()) {
    if (!dec.pkg_header_read) {
      if (!dec.read_length_and_width()) return false;
    }
    while (dec.has_next_pkg()) {
      out.push_back(dec.read_int());
      if (!dec.inner.ok) return false;
    }
    dec.pkg_header_read = false;
  }
  return dec.outer.ok && dec.outer.pos == dec.outer.len;
}

static float int_to_float_scaled(std::int32_t v, int max_point) {
  if (max_point <= 0) {
    return static_cast<float>(v);
  }
  const double max_point_value = std::pow(10.0, max_point);
  return static_cast<float>(static_cast<double>(v) / max_point_value);
}

}  // namespace

bool decode_ts_2diff_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  while (c.ok && c.pos < c.len) {
    std::int32_t pack_num = c.read_i32_be();
    if (!c.ok) return false;
    std::int32_t pack_width = c.read_i32_be();
    if (!c.ok) return false;
    std::int32_t min_delta_base = c.read_i32_be();
    if (!c.ok) return false;
    std::int32_t first_value = c.read_i32_be();
    if (!c.ok) return false;
    int enc_len = ceil_bits_to_bytes(static_cast<int>(pack_num) * static_cast<int>(pack_width));
    if (!c.has(static_cast<std::size_t>(enc_len))) {
      c.ok = false;
      return false;
    }
    std::vector<std::uint8_t> delta_buf(static_cast<std::size_t>(enc_len));
    std::memcpy(delta_buf.data(), c.p + c.pos, static_cast<std::size_t>(enc_len));
    c.pos += static_cast<std::size_t>(enc_len);

    out.push_back(first_value);
    std::int32_t previous = first_value;
    for (int i = 0; i < pack_num; i++) {
      std::int32_t v =
          bytes_to_int_at_bits(delta_buf.data(), static_cast<int>(pack_width) * i, static_cast<int>(pack_width));
      std::int32_t cur = minimum_for::restore(previous, static_cast<std::uint32_t>(minimum_for::restore(min_delta_base, static_cast<std::uint32_t>(v))));
      out.push_back(cur);
      previous = cur;
    }
  }
  return c.ok && c.pos == c.len;
}

bool decode_rle_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out) {
  if (unified_rle_numeric::is_rle_frame(encoded))
    return unified_rle_numeric::decode_values(encoded, out);
  return decode_rle_int_from_bytes(encoded, out);
}

bool decode_ts_2diff_float(std::span<const std::uint8_t> encoded, std::vector<float>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  std::uint32_t max_point = c.read_uvarint();
  if (!c.ok) return false;
  const int max_point_i = static_cast<int>(max_point);
  std::vector<std::int32_t> ints;
  std::span<const std::uint8_t> remainder(c.p + c.pos, c.len - c.pos);
  if (!decode_ts_2diff_int32(remainder, ints)) return false;
  c.pos = c.len;
  out.reserve(ints.size());
  for (std::int32_t v : ints) out.push_back(int_to_float_scaled(v, max_point_i));
  return true;
}

bool decode_rle_float(std::span<const std::uint8_t> encoded, std::vector<float>& out) {
  if (unified_rle_numeric::is_scaled_rle_frame(encoded)) {
    out.clear();
    try {
      if(unified_rle_numeric::scaled_rle_symbol_width(encoded)!=4)return false;
      out=unified_rle_numeric::typed_values<float>(unified_rle_numeric::decode_rle_scaled(encoded));return true;
    } catch(const std::exception&) {return false;}
  }
  if (unified_rle_numeric::is_rle_frame(encoded))
    return unified_rle_numeric::decode_values(encoded, out);
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  int max_point = static_cast<int>(c.read_uvarint());
  if (!c.ok) return false;
  std::vector<std::int32_t> ints;
  std::span<const std::uint8_t> remainder(c.p + c.pos, c.len - c.pos);
  if (!decode_rle_int_from_bytes(remainder, ints)) return false;
  out.reserve(ints.size());
  for (std::int32_t v : ints) out.push_back(int_to_float_scaled(v, max_point));
  return true;
}

void unpack8_values_long(int width, const std::uint8_t* buf, int offset, std::int64_t* values) {
  int byte_idx = offset;
  std::uint64_t buffer = 0;
  int total_bits = 0;
  int value_idx = 0;
  while (value_idx < 8) {
    while (total_bits < width) {
      if (byte_idx >= offset + width) return;
      buffer = (buffer << 8) | static_cast<std::uint64_t>(buf[byte_idx] & 0xff);
      byte_idx++;
      total_bits += 8;
    }
    while (total_bits >= width && value_idx < 8) {
      values[value_idx] = static_cast<std::int64_t>(buffer >> (total_bits - width));
      value_idx++;
      total_bits -= width;
      if (total_bits > 0) {
        buffer &= (1ull << total_bits) - 1ull;
      } else {
        buffer = 0;
      }
    }
  }
}

struct LongRleDecoder {
  ByteCursor outer;
  std::vector<std::uint8_t> pattern_bytes;
  ByteCursor inner{};
  int bit_width = 0;
  int current_count = 0;
  std::int64_t current_value = 0;
  std::vector<std::int64_t> current_buffer;
  int bit_packing_num = 0;
  bool pkg_header_read = false;
  enum class Mode { Rle, BitPacked } mode = Mode::Rle;

  void reset_outer(std::span<const std::uint8_t> enc) {
    outer = ByteCursor{enc.data(), enc.size(), 0, true};
    pattern_bytes.clear();
    pkg_header_read = false;
    current_count = 0;
  }

  bool read_length_and_width() {
    std::uint32_t length = outer.read_uvarint();
    if (!outer.ok || length == 0) return false;
    if (!outer.has(length)) {
      outer.ok = false;
      return false;
    }
    pattern_bytes.assign(outer.p + outer.pos, outer.p + outer.pos + length);
    outer.pos += length;
    inner = ByteCursor{pattern_bytes.data(), pattern_bytes.size(), 0, true};
    bit_width = static_cast<int>(inner.get_u8());
    if (!inner.ok) return false;
    pkg_header_read = true;
    current_buffer.clear();
    return true;
  }

  bool read_number_rle() {
    current_value = inner.read_long_le_padded(bit_width);
    return inner.ok;
  }

  void unpack_all_bitpacked(int bit_packed_group_count, int last_bit_packed_num) {
    int total_count = (bit_packed_group_count - 1) * kRleMinRepeated + last_bit_packed_num;
    bit_packing_num = total_count;
    current_buffer.resize(static_cast<std::size_t>(total_count));
    int bytes_to_read = bit_packed_group_count * bit_width;
    if (static_cast<std::size_t>(bytes_to_read) > inner.len - inner.pos) {
      inner.ok = false;
      return;
    }
    int k = 0;
    int idx = static_cast<int>(inner.pos);
    while (k < total_count) {
      std::int64_t tv[8];
      unpack8_values_long(bit_width, pattern_bytes.data(), idx, tv);
      for (int t = 0; t < 8 && k < total_count; t++) {
        current_buffer[static_cast<std::size_t>(k++)] = tv[t];
      }
      idx += bit_width;
    }
    inner.pos = static_cast<std::size_t>(idx);
  }

  bool read_next() {
    std::uint32_t header = inner.read_uvarint();
    if (!inner.ok) return false;
    if ((header & 1u) == 0) {
      mode = Mode::Rle;
      current_count = static_cast<int>(header >> 1);
      return read_number_rle();
    }
    mode = Mode::BitPacked;
    int bit_packed_group_count = static_cast<int>(header >> 1);
    int last_bit_packed_num = static_cast<int>(inner.get_u8());
    if (!inner.ok) return false;
    unpack_all_bitpacked(bit_packed_group_count, last_bit_packed_num);
    current_count = bit_packing_num;
    return inner.ok;
  }

  bool has_next_pkg() { return current_count > 0 || inner.pos < inner.len; }

  std::int64_t read_long() {
    if (!pkg_header_read) {
      if (!read_length_and_width()) {
        outer.ok = false;
        return 0;
      }
    }
    if (current_count == 0) {
      if (!read_next()) {
        outer.ok = false;
        return 0;
      }
    }
    current_count--;
    std::int64_t result = 0;
    if (mode == Mode::Rle) {
      result = current_value;
    } else {
      result = current_buffer[static_cast<std::size_t>(bit_packing_num - current_count - 1)];
    }
    if (!has_next_pkg()) pkg_header_read = false;
    return result;
  }

  bool has_more_outer() const { return outer.pos < outer.len && outer.ok; }
};

bool decode_rle_long_from_bytes(std::span<const std::uint8_t> enc, std::vector<std::int64_t>& out) {
  out.clear();
  LongRleDecoder dec;
  dec.reset_outer(enc);
  while (dec.has_more_outer()) {
    if (!dec.pkg_header_read) {
      if (!dec.read_length_and_width()) return false;
    }
    while (dec.has_next_pkg()) {
      out.push_back(dec.read_long());
      if (!dec.inner.ok) return false;
    }
    dec.pkg_header_read = false;
  }
  return dec.outer.ok && dec.outer.pos == dec.outer.len;
}

bool decode_ts_2diff_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  while (c.ok && c.pos < c.len) {
    std::int32_t pack_num = c.read_i32_be();
    if (!c.ok) return false;
    std::int32_t pack_width = c.read_i32_be();
    if (!c.ok) return false;
    std::int64_t min_delta_base = c.read_i64_be();
    if (!c.ok) return false;
    std::int64_t first_value = c.read_i64_be();
    if (!c.ok) return false;
    int enc_len = ceil_bits_to_bytes(static_cast<int>(pack_num) * static_cast<int>(pack_width));
    if (!c.has(static_cast<std::size_t>(enc_len))) {
      c.ok = false;
      return false;
    }
    std::vector<std::uint8_t> delta_buf(static_cast<std::size_t>(enc_len));
    std::memcpy(delta_buf.data(), c.p + c.pos, static_cast<std::size_t>(enc_len));
    c.pos += static_cast<std::size_t>(enc_len);

    out.push_back(first_value);
    std::int64_t previous = first_value;
    for (int i = 0; i < pack_num; i++) {
      std::int64_t v =
          decode::bytes_to_long_at_bits(delta_buf.data(), pack_width * i, pack_width);
      std::int64_t cur = minimum_for::restore(previous, static_cast<std::uint64_t>(minimum_for::restore(min_delta_base, static_cast<std::uint64_t>(v))));
      out.push_back(cur);
      previous = cur;
    }
  }
  return c.ok && c.pos == c.len;
}

bool decode_rle_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  if (unified_rle_numeric::is_rle_frame(encoded))
    return unified_rle_numeric::decode_values(encoded, out);
  return decode_rle_long_from_bytes(encoded, out);
}

static double long_to_double_scaled(std::int64_t v, int max_point) {
  if (max_point <= 0) {
    return static_cast<double>(v);
  }
  const double max_point_value = std::pow(10.0, max_point);
  return static_cast<double>(v) / max_point_value;
}

bool decode_ts_2diff_double(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  std::uint32_t max_point = c.read_uvarint();
  if (!c.ok) return false;
  const int max_point_i = static_cast<int>(max_point);
  std::vector<std::int64_t> longs;
  std::span<const std::uint8_t> remainder(c.p + c.pos, c.len - c.pos);
  if (!decode_ts_2diff_int64(remainder, longs)) return false;
  c.pos = c.len;
  out.reserve(longs.size());
  for (std::int64_t v : longs) out.push_back(long_to_double_scaled(v, max_point_i));
  return true;
}

bool decode_rle_double(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  if (unified_rle_numeric::is_scaled_rle_frame(encoded)) {
    out.clear();
    try {
      if(unified_rle_numeric::scaled_rle_symbol_width(encoded)!=8)return false;
      out=unified_rle_numeric::typed_values<double>(unified_rle_numeric::decode_rle_scaled(encoded));return true;
    } catch(const std::exception&) {return false;}
  }
  if (unified_rle_numeric::is_rle_frame(encoded))
    return unified_rle_numeric::decode_values(encoded, out);
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  const std::uint32_t max_point_wire = c.read_uvarint();
  if (!c.ok) return false;
  if (max_point_wire == std::numeric_limits<std::uint32_t>::max()) {
    const std::uint32_t value_count = c.read_uvarint();
    if (!c.ok) return false;
    out.reserve(value_count);
    while (out.size() < value_count) {
      const std::uint32_t run_length = c.read_uvarint();
      if (!c.ok || run_length == 0 || !c.has(sizeof(std::int64_t)) ||
          run_length > value_count - out.size()) {
        return false;
      }
      const std::int64_t signed_bits = c.read_i64_be();
      std::uint64_t bits = static_cast<std::uint64_t>(signed_bits);
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      out.insert(out.end(), run_length, value);
    }
    return c.ok && c.pos == c.len;
  }
  std::vector<std::int64_t> longs;
  std::span<const std::uint8_t> remainder(c.p + c.pos, c.len - c.pos);
  if (!decode_rle_long_from_bytes(remainder, longs)) return false;
  out.reserve(longs.size());
  for (std::int64_t v : longs) {
    out.push_back(long_to_double_scaled(v, static_cast<int>(max_point_wire)));
  }
  return true;
}

}  // namespace tsfile::p02
