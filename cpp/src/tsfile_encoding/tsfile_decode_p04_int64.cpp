#include "../unified_rle_numeric.hpp"
#include "tsfile_decode_p04.hpp"

#include "tsfile_decode_common.hpp"
#include "tsfile_decode_p02.hpp"
#include "tsfile_decode_p03.hpp"

#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace tsfile::p04 {

namespace {

using tsfile::decode::ByteCursor;

void unpack8_values_long(int width, const std::uint8_t* buf, int offset, std::int64_t* values) {
  int byte_idx = offset;
  unsigned __int128 buffer = 0;
  int total_bits = 0;
  int value_idx = 0;
  while (value_idx < 8) {
    while (total_bits < width) {
      buffer = (buffer << 8) | static_cast<unsigned __int128>(buf[byte_idx] & 0xff);
      byte_idx++;
      total_bits += 8;
    }
    while (total_bits >= width && value_idx < 8) {
      values[value_idx] = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(buffer >> (total_bits - width)));
      value_idx++;
      total_bits -= width;
      if (total_bits > 0) {
        buffer &= (static_cast<unsigned __int128>(1) << total_bits) - 1;
      } else {
        buffer = 0;
      }
    }
  }
}

struct LongFire {
  int learn_shift = 2;
  int bit_width = 16;
  std::int64_t accumulator = 0;
  std::int64_t delta = 0;
  void reset() {
    accumulator = 0;
    delta = 0;
  }
  std::int64_t predict(std::int64_t value) {
    std::int64_t alpha = accumulator >> learn_shift;
    const std::uint64_t product = static_cast<std::uint64_t>(alpha) *
                                  static_cast<std::uint64_t>(delta);
    const std::int64_t diff = static_cast<std::int64_t>(product) >> bit_width;
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(value) +
                                     static_cast<std::uint64_t>(diff));
  }
  void train(std::int64_t pre, std::int64_t val, std::int64_t err) {
    std::int64_t gradient =
        err > 0 ? static_cast<std::int64_t>(std::uint64_t{0} - static_cast<std::uint64_t>(delta))
                : delta;
    accumulator = static_cast<std::int64_t>(static_cast<std::uint64_t>(accumulator) -
                                            static_cast<std::uint64_t>(gradient));
    delta = static_cast<std::int64_t>(static_cast<std::uint64_t>(val) -
                                      static_cast<std::uint64_t>(pre));
  }
};

constexpr int kSprintzBlock = 8;

void recalc_long_fire(std::int64_t* buf, bool omit_predictor, bool omit_zigzag) {
  if (!omit_zigzag) {
    for (int i = 1; i <= kSprintzBlock; i++) {
      const std::uint64_t zz = static_cast<std::uint64_t>(buf[i]);
      buf[i] = static_cast<std::int64_t>((zz >> 1) ^
                                         (std::uint64_t{0} - (zz & 1U)));
    }
  }
  if (omit_predictor) {
    return;
  }
  LongFire fire{};
  std::int64_t pre = buf[0];
  for (int i = 1; i <= kSprintzBlock; i++) {
    std::int64_t pred = fire.predict(buf[i - 1]);
    std::int64_t err = buf[i];
    buf[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(pred) +
                                       static_cast<std::uint64_t>(err));
    fire.train(buf[i - 1], buf[i], err);
  }
}

void recalc_double_fire(double pre_value, std::int64_t* convert_buf, double* out_buf) {
  out_buf[0] = pre_value;
  for (int i = 0; i < kSprintzBlock; i++) {
    const std::uint64_t zz = static_cast<std::uint64_t>(convert_buf[i]);
    convert_buf[i] = static_cast<std::int64_t>((zz >> 1) ^
                                               (std::uint64_t{0} - (zz & 1U)));
  }
  LongFire fire{};
  std::uint64_t pb = 0;
  std::memcpy(&pb, &pre_value, sizeof(double));
  std::int64_t prev = static_cast<std::int64_t>(pb);
  std::int64_t pred = fire.predict(prev);
  std::int64_t e = convert_buf[0];
  convert_buf[0] = static_cast<std::int64_t>(static_cast<std::uint64_t>(convert_buf[0]) +
                                             static_cast<std::uint64_t>(pred));
  std::memcpy(&out_buf[1], &convert_buf[0], sizeof(double));
  fire.train(prev, convert_buf[0], e);
  for (int i = 1; i < kSprintzBlock; i++) {
    pred = fire.predict(convert_buf[i - 1]);
    e = convert_buf[i];
    convert_buf[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(convert_buf[i]) +
                                               static_cast<std::uint64_t>(pred));
    std::memcpy(&out_buf[i + 1], &convert_buf[i], sizeof(double));
    fire.train(convert_buf[i - 1], convert_buf[i], e);
  }
}

bool decode_sprintz_int64_blocks(ByteCursor& c, std::string_view omitted_operator,
                                 std::vector<std::int64_t>& out, bool reference_rle_tail = false) {
  const bool omit_predictor = omitted_operator == "delta";
  const bool omit_zigzag = omitted_operator == "zigzag";
  const bool omit_bitpacking = omitted_operator == "bit-packing";
  while (c.ok && c.pos < c.len) {
    int bit_width_byte = c.read_int_le_padded(1);
    if (!c.ok) return false;
    if ((bit_width_byte & 0x80) != 0) {
      int decode_size = bit_width_byte & ~(1 << 7);
      if (decode_size < 1 || decode_size > 8) return false;
      if (reference_rle_tail) {
        std::vector<std::int64_t> tail;
        if (!tsfile::p02::decode_rle_int64(std::span<const std::uint8_t>(c.p+c.pos,c.len-c.pos),tail) || tail.size()!=static_cast<std::size_t>(decode_size)) return false;
        out.insert(out.end(),tail.begin(),tail.end());c.pos=c.len;break;
      }
      if (!c.has(static_cast<std::size_t>(decode_size) * sizeof(std::int64_t))) return false;
      for (int i = 0; i < decode_size; ++i) {
        out.push_back(c.read_i64_be());
      }
      break;
    }
    std::int64_t pre = c.read_i64_be();
    if (!c.ok) return false;
    const std::size_t payload_size = omit_bitpacking ? 8 * sizeof(std::int64_t)
                                                      : static_cast<std::size_t>(bit_width_byte);
    if (!c.has(payload_size)) {
      c.ok = false;
      return false;
    }
    std::int64_t tmp[8]{};
    if (omit_bitpacking) {
      for (std::int64_t& value : tmp) value = c.read_i64_be();
    } else {
      std::vector<std::uint8_t> packc(static_cast<std::size_t>(bit_width_byte));
      for (int i = 0; i < bit_width_byte; i++) {
        packc[static_cast<std::size_t>(i)] = c.get_u8();
      }
      unpack8_values_long(bit_width_byte, packc.data(), 0, tmp);
    }
    std::int64_t buf[9];
    buf[0] = pre;
    for (int i = 0; i < 8; i++) buf[i + 1] = tmp[i];
    recalc_long_fire(buf, omit_predictor, omit_zigzag);
    for (int i = 0; i <= kSprintzBlock; i++) out.push_back(buf[i]);
  }
  return c.ok && c.pos == c.len;
}

bool decode_sprintz_double_blocks(ByteCursor& c, std::vector<double>& out, bool reference_gorilla_tail = false) {
  while (c.ok && c.pos < c.len) {
    int bit_width_byte = c.read_int_le_padded(1);
    if (!c.ok) return false;
    if ((bit_width_byte & 0x80) != 0) {
      int decode_size = bit_width_byte & ~(1 << 7);
      if (decode_size < 1 || decode_size > 8) return false;
      if (reference_gorilla_tail) {
        std::vector<double> tail;
        if (!tsfile::p03::decode_gorilla_double(std::span<const std::uint8_t>(c.p+c.pos,c.len-c.pos),tail) || tail.size()!=static_cast<std::size_t>(decode_size)) return false;
        out.insert(out.end(),tail.begin(),tail.end());c.pos=c.len;break;
      }
      if (!c.has(static_cast<std::size_t>(decode_size) * sizeof(std::int64_t))) return false;
      for (int i = 0; i < decode_size; ++i) {
        std::int64_t bits = c.read_i64_be();
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(double));
        out.push_back(value);
      }
      break;
    }
    std::int64_t pre_bits = c.read_i64_be();
    if (!c.ok) return false;
    double pre;
    std::memcpy(&pre, &pre_bits, sizeof(double));
    if (!c.has(static_cast<std::size_t>(bit_width_byte))) {
      c.ok = false;
      return false;
    }
    std::vector<std::uint8_t> packc(static_cast<std::size_t>(bit_width_byte));
    for (int i = 0; i < bit_width_byte; i++) packc[static_cast<std::size_t>(i)] = c.get_u8();
    std::int64_t tmp[8]{};
    unpack8_values_long(bit_width_byte, packc.data(), 0, tmp);
    double buf[9]{};
    std::int64_t convert_buf[8];
    for (int i = 0; i < 8; i++) convert_buf[i] = tmp[i];
    recalc_double_fire(pre, convert_buf, buf);
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

static bool rlbe_read_one_block_long(std::span<const std::uint8_t> enc,
                                     std::vector<std::int64_t>& out_values,
                                     std::size_t& consumed,
                                     bool omit_delta) {
  RlbBitReader br;
  br.init(enc);
  br.clear_buffer();
  int blocksize = br.read_block_size_from_stream();
  if (!br.ok || blocksize < 1) return false;
  std::vector<std::int64_t> data(static_cast<std::size_t>(blocksize * 2 + 1));
  std::int64_t fib[4096];
  int writeindex = -1;
  while (writeindex < blocksize - 1 && br.ok) {
    int seglength = 0;
    for (int j = 6; j >= 0; j--) seglength |= br.readbit() << j;
    int now = br.readbit();
    int next = br.readbit();
    int j = 1;
    int runlength = 0;
    fib[0] = 1;
    fib[1] = 1;
    for (;;) {
      if (j > 1) fib[j] = fib[j - 1] + fib[j - 2];
      if (now == 1) runlength += static_cast<int>(fib[j]);
      if (now == 1 && next == 1) break;
      j++;
      now = next;
      next = br.readbit();
    }
    for (int i = 1; i <= runlength && writeindex < blocksize - 1; i++) {
      std::int64_t readtemp = 0;
      for (int k = seglength - 1; k >= 0; k--) {
        readtemp |= static_cast<std::int64_t>(br.readbit()) << k;
      }
      if (writeindex == -1 || omit_delta) {
        data[static_cast<std::size_t>(++writeindex)] = readtemp;
      } else {
        ++writeindex;
        data[static_cast<std::size_t>(writeindex)] = static_cast<std::int64_t>(
            static_cast<std::uint64_t>(data[static_cast<std::size_t>(writeindex - 1)]) +
            static_cast<std::uint64_t>(readtemp));
      }
    }
  }
  out_values.assign(data.begin(), data.begin() + blocksize);
  consumed = br.pos;
  return br.ok;
}

static bool rlbe_read_block_long(std::span<const std::uint8_t> enc,
                                 std::vector<std::int64_t>& out_values,
                                 bool omit_delta) {
  out_values.clear();
  std::size_t offset = 0;
  while (offset < enc.size()) {
    std::vector<std::int64_t> block;
    std::size_t consumed = 0;
    if (!rlbe_read_one_block_long(enc.subspan(offset), block, consumed, omit_delta) ||
        consumed == 0) {
      return false;
    }
    out_values.insert(out_values.end(), block.begin(), block.end());
    offset += consumed;
  }
  return offset == enc.size();
}

static bool rlbe_read_block_double(std::span<const std::uint8_t> enc,
                                   std::vector<double>& out_values,
                                   bool omit_delta) {
  std::vector<std::int64_t> bits;
  if (!rlbe_read_block_long(enc, bits, omit_delta)) return false;
  out_values.resize(bits.size());
  for (std::size_t i = 0; i < bits.size(); i++) {
    std::memcpy(&out_values[i], &bits[i], sizeof(double));
  }
  return true;
}

}  // namespace

bool decode_sprintz_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  return decode_sprintz_int64_ablation(encoded, {}, out);
}

bool decode_sprintz_int64_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<std::int64_t>& out) {
  out.clear();
  ByteCursor c{encoded.data(), encoded.size(), 0, true};
  return decode_sprintz_int64_blocks(c, omitted_operator, out);
}

bool decode_sprintz_double(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  return decode_sprintz_double_ablation(encoded, {}, out);
}

bool decode_sprintz_double_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<double>& out) {
  out.clear();
  std::vector<std::int64_t> bits;
  if (!decode_sprintz_int64_ablation(encoded, omitted_operator, bits)) return false;
  out.resize(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) {
    std::memcpy(&out[i], &bits[i], sizeof(double));
  }
  return true;
}

bool decode_rlbe_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  if (unified_rle_numeric::is_rlbe_frame(encoded))
    return unified_rle_numeric::decode_rlbe_values(encoded, out);
  return decode_rlbe_int64_ablation(encoded, {}, out);
}

bool decode_rlbe_int64_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<std::int64_t>& out) {
  out.clear();
  return rlbe_read_block_long(encoded, out, omitted_operator == "delta");
}

bool decode_rlbe_double(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  if (unified_rle_numeric::is_rlbe_frame(encoded))
    return unified_rle_numeric::decode_rlbe_values(encoded, out);
  return decode_rlbe_double_ablation(encoded, {}, out);
}

bool decode_rlbe_double_ablation(
    std::span<const std::uint8_t> encoded, std::string_view omitted_operator,
    std::vector<double>& out) {
  out.clear();
  return rlbe_read_block_double(encoded, out, omitted_operator == "delta");
}


bool decode_sprintz_int64_reference(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  out.clear();ByteCursor c{encoded.data(),encoded.size(),0,true};
  return decode_sprintz_int64_blocks(c,{},out,true);
}
bool decode_sprintz_double_reference(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  out.clear();ByteCursor c{encoded.data(),encoded.size(),0,true};
  return decode_sprintz_double_blocks(c,out,true);
}

}  // namespace tsfile::p04
