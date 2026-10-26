#include "../unified_rle_numeric.hpp"
#include "tsfile_decode_p04.hpp"

#include "tsfile_decode_common.hpp"
#include "tsfile_decode_p02.hpp"
#include "tsfile_decode_p03.hpp"

#include <cstring>
#include <span>
#include <vector>

namespace tsfile::p04 {

namespace {

using tsfile::decode::ByteCursor;

void unpack8_values(int width, const std::uint8_t* buf, int offset, std::int32_t* values) {
  int byte_idx = offset;
  std::uint64_t buffer = 0;
  int total_bits = 0;
  int value_idx = 0;
  while (value_idx < 8) {
    while (total_bits < width) {
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

struct IntFire {
  int learn_shift = 2;
  int bit_width = 8;
  int accumulator = 0;
  int delta = 0;
  void reset() {
    accumulator = 0;
    delta = 0;
  }
  int predict(int value) {
    int alpha = accumulator >> learn_shift;
    int diff = (alpha * delta) >> bit_width;
    return value + diff;
  }
  void train(int pre, int val, int err) {
    int gradient = err > 0 ? -delta : delta;
    accumulator -= gradient;
    delta = val - pre;
  }
};

constexpr int kSprintzBlock = 8;

void recalc_int_fire(std::int32_t* buf) {
  for (int i = 1; i <= kSprintzBlock; i++) {
    if ((buf[i] & 1) == 0) buf[i] = -buf[i] / 2;
    else buf[i] = (buf[i] + 1) / 2;
  }
  IntFire fire{};
  int pre = buf[0];
  for (int i = 1; i <= kSprintzBlock; i++) {
    int pred = fire.predict(buf[i - 1]);
    int err = buf[i];
    buf[i] = pred + err;
    fire.train(buf[i - 1], buf[i], err);
  }
}

void recalc_float_fire(float pre_value, std::int32_t* convert_buf, float* out_buf) {
  out_buf[0] = pre_value;
  for (int i = 0; i < kSprintzBlock; i++) {
    if ((convert_buf[i] & 1) == 0) convert_buf[i] = -convert_buf[i] / 2;
    else convert_buf[i] =  (convert_buf[i] + 1) / 2;
  }
  IntFire fire{};
  std::uint32_t pb = 0;
  std::memcpy(&pb, &pre_value, sizeof(float));
  int p = fire.predict(static_cast<int>(pb));
  int e = convert_buf[0];
  convert_buf[0] += p;
  std::memcpy(&out_buf[1], &convert_buf[0], sizeof(float));
  fire.train(static_cast<int>(pb), convert_buf[0], e);
  for (int i = 1; i < kSprintzBlock; i++) {
    int pred = fire.predict(convert_buf[i - 1]);
    int err = convert_buf[i];
    convert_buf[i] += pred;
    std::memcpy(&out_buf[i + 1], &convert_buf[i], sizeof(float));
    fire.train(convert_buf[i - 1], convert_buf[i], err);
  }
}

bool decode_sprintz_int_blocks(ByteCursor& c, std::vector<std::int32_t>& out) {
  while (c.ok && c.pos < c.len) {
    int bit_width_byte = c.read_int_le_padded(1);
    if (!c.ok) return false;
    if ((bit_width_byte & 0x80) != 0) {
      int decode_size = bit_width_byte & ~(1 << 7);
      std::size_t rem = c.len - c.pos;
      std::vector<std::uint8_t> chunk(c.p + c.pos, c.p + c.pos + rem);
      c.pos = c.len;
      std::vector<std::int32_t> tail;
      if (!tsfile::p02::decode_rle_int32(chunk, tail)) return false;
      if (static_cast<int>(tail.size()) != decode_size) return false;
      out.insert(out.end(), tail.begin(), tail.end());
      break;
    }
    std::int32_t pre = static_cast<std::int32_t>(c.read_uvarint());
    if (!c.ok) return false;
    if (!c.has(static_cast<std::size_t>(bit_width_byte))) {
      c.ok = false;
      return false;
    }
    std::vector<std::uint8_t> packc(static_cast<std::size_t>(bit_width_byte));
    for (int i = 0; i < bit_width_byte; i++) packc[static_cast<std::size_t>(i)] = c.get_u8();
    std::int32_t tmp[8]{};
    unpack8_values(bit_width_byte, packc.data(), 0, tmp);
    std::int32_t buf[9];
    buf[0] = pre;
    for (int i = 0; i < 8; i++) buf[i + 1] = tmp[i];
    recalc_int_fire(buf);
    for (int i = 0; i <= kSprintzBlock; i++) out.push_back(buf[i]);
  }
  return c.ok && c.pos == c.len;
}

bool decode_sprintz_float_blocks(ByteCursor& c, std::vector<float>& out) {
  while (c.ok && c.pos < c.len) {
    int bit_width_byte = c.read_int_le_padded(1);
    if (!c.ok) return false;
    if ((bit_width_byte & 0x80) != 0) {
      int decode_size = bit_width_byte & ~(1 << 7);
      std::size_t rem = c.len - c.pos;
      std::vector<std::uint8_t> chunk(c.p + c.pos, c.p + c.pos + rem);
      c.pos = c.len;
      std::vector<float> tail;
      if (!tsfile::p03::decode_gorilla_float(chunk, tail)) return false;
      if (static_cast<int>(tail.size()) != decode_size) return false;
      out.insert(out.end(), tail.begin(), tail.end());
      break;
    }
    float pre = c.read_f32_be();
    if (!c.ok) return false;
    if (!c.has(static_cast<std::size_t>(bit_width_byte))) {
      c.ok = false;
      return false;
    }
    std::vector<std::uint8_t> packc(static_cast<std::size_t>(bit_width_byte));
    for (int i = 0; i < bit_width_byte; i++) packc[static_cast<std::size_t>(i)] = c.get_u8();
    std::int32_t tmp[8]{};
    unpack8_values(bit_width_byte, packc.data(), 0, tmp);
    float buf[9]{};
    std::int32_t convert_buf[8];
    for (int i = 0; i < 8; i++) convert_buf[i] = tmp[i];
    recalc_float_fire(pre, convert_buf, buf);
    for (int i = 0; i <= kSprintzBlock; i++) out.push_back(buf[i]);
  }
  return c.ok && c.pos == c.len;
}

struct RlbBitReader {
  const std::uint8_t* p = nullptr;
  std::size_t len = 0;
  std::size_t pos = 0;
  std::uint8_t byte_buffer = 0;
  int number_left_in_buffer = 0;
  bool ok = true;

  void init(std::span<const std::uint8_t> enc) {
    p = enc.data();
    len = enc.size();
    pos = 0;
    byte_buffer = 0;
    number_left_in_buffer = 0;
    ok = true;
  }

  int readbit() {
    if (number_left_in_buffer == 0) {
      if (pos >= len) {
        ok = false;
        return 0;
      }
      byte_buffer = p[pos++];
      number_left_in_buffer = 8;
    }
    int top = (byte_buffer >> 7) & 1;
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << 1);
    number_left_in_buffer--;
    return top;
  }

  void clear_buffer() {
    while (number_left_in_buffer > 0) readbit();
  }

  int read_block_size_from_stream() {
    int blocksize = 0;
    for (int i = 31; i >= 0; i--) {
      if (readbit()) blocksize |= (1 << i);
    }
    return blocksize;
  }
};

static bool rlbe_read_block_int(std::span<const std::uint8_t> enc, std::vector<std::int32_t>& out_values) {
  RlbBitReader br;
  br.init(enc);
  br.clear_buffer();
  int blocksize = br.read_block_size_from_stream();
  if (!br.ok || blocksize < 1) return false;
  std::vector<std::int32_t> data(static_cast<std::size_t>(blocksize * 2 + 1));
  int fib[4096];
  int writeindex = -1;
  while (writeindex < blocksize - 1 && br.ok) {
    int seglength = 0;
    for (int j = 5; j >= 0; j--) seglength |= br.readbit() << j;
    int now = br.readbit();
    int next = br.readbit();
    int j = 1;
    int runlength = 0;
    fib[0] = 1;
    fib[1] = 1;
    for (;;) {
      if (j > 1) fib[j] = fib[j - 1] + fib[j - 2];
      if (now == 1) runlength += fib[j];
      if (now == 1 && next == 1) break;
      j++;
      now = next;
      next = br.readbit();
    }
    for (int i = 1; i <= runlength && writeindex < blocksize - 1; i++) {
      int readtemp = 0;
      for (int k = seglength - 1; k >= 0; k--) readtemp += br.readbit() << k;
      if (seglength == 32) readtemp -= (1 << 31);
      if (writeindex == -1) {
        data[static_cast<std::size_t>(++writeindex)] = readtemp;
      } else {
        ++writeindex;
        data[static_cast<std::size_t>(writeindex)] =
            data[static_cast<std::size_t>(writeindex - 1)] + readtemp;
      }
    }
  }
  out_values.assign(data.begin(), data.begin() + blocksize);
  return br.ok;
}

static bool rlbe_read_block_float(std::span<const std::uint8_t> enc, std::vector<float>& out_values) {
  RlbBitReader br;
  br.init(enc);
  br.clear_buffer();
  int blocksize = br.read_block_size_from_stream();
  if (!br.ok || blocksize < 1) return false;
  std::vector<float> data(static_cast<std::size_t>(blocksize * 2 + 1));
  int fib[4096];
  int writeindex = -1;
  while (writeindex < blocksize - 1 && br.ok) {
    int seglength = 0;
    for (int j = 5; j >= 0; j--) seglength |= br.readbit() << j;
    int now = br.readbit();
    int next = br.readbit();
    int j = 1;
    int runlength = 0;
    fib[0] = 1;
    fib[1] = 1;
    for (;;) {
      if (j > 1) fib[j] = fib[j - 1] + fib[j - 2];
      if (now == 1) runlength += fib[j];
      if (now == 1 && next == 1) break;
      j++;
      now = next;
      next = br.readbit();
    }
    for (int i = 1; i <= runlength && writeindex < blocksize - 1; i++) {
      int readtemp = 0;
      for (int k = seglength - 1; k >= 0; k--) readtemp += br.readbit() << k;
      if (seglength == 32) readtemp -= (1 << 31);
      float f = 0;
      std::memcpy(&f, &readtemp, sizeof(float));
      if (writeindex == -1) {
        data[static_cast<std::size_t>(++writeindex)] = f;
      } else {
        ++writeindex;
        data[static_cast<std::size_t>(writeindex)] =
            data[static_cast<std::size_t>(writeindex - 1)] + f;
      }
    }
  }
  out_values.assign(data.begin(), data.begin() + blocksize);
  return br.ok;
}

}  // namespace

bool decode_sprintz_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  return decode_sprintz_int_blocks(c, out);
}

bool decode_sprintz_float(std::span<const std::uint8_t> encoded, std::vector<float>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  return decode_sprintz_float_blocks(c, out);
}

bool decode_rlbe_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out) {
  if (unified_rle_numeric::is_rlbe_frame(encoded))
    return unified_rle_numeric::decode_rlbe_values(encoded, out);
  out.clear();
  return rlbe_read_block_int(encoded, out);
}

bool decode_rlbe_float(std::span<const std::uint8_t> encoded, std::vector<float>& out) {
  if (unified_rle_numeric::is_rlbe_frame(encoded)) {
    try {
      const auto mode = unified_rle_numeric::rlbe_delta_mode(encoded);
      if (mode != 0 && mode != 2) return false;
      return unified_rle_numeric::decode_rlbe_values(encoded, out, mode);
    } catch (const std::exception&) {out.clear();return false;}
  }
  out.clear();
  return rlbe_read_block_float(encoded, out);
}

}  // namespace tsfile::p04
