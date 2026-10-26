#include "../unified_rle_numeric.hpp"
#include "tsfile_encode_p04.hpp"

#include <algorithm>

#include "tsfile_encode_p02.hpp"
#include "tsfile_encode_p03.hpp"

#include <cstring>
#include <limits>
#include <vector>

namespace tsfile::p04 {

namespace {

void append_bytes(std::vector<std::uint8_t>& dst, const std::uint8_t* p, std::size_t n) {
  dst.insert(dst.end(), p, p + n);
}

void append_uvarint(std::vector<std::uint8_t>& out, std::uint32_t v) {
  while ((v & 0xffffff80u) != 0) {
    out.push_back(static_cast<std::uint8_t>((v & 0x7f) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(v & 0x7f));
}

void append_int_le_padded(std::vector<std::uint8_t>& out, std::int32_t value, int bit_width) {
  int padded = (bit_width + 7) / 8;
  int offset = 0;
  while (padded > 0) {
    out.push_back(static_cast<std::uint8_t>((static_cast<std::uint32_t>(value) >> offset) & 0xff));
    offset += 8;
    padded--;
  }
}

int clz_u32(std::uint32_t x) {
  if (x == 0) return 32;
  return __builtin_clz(x);
}

int int_value_bit_width(std::int32_t num) {
  return 32 - clz_u32(static_cast<std::uint32_t>(num));
}

int int_max_bit_width_vec(const std::vector<std::int32_t>& list) {
  int max_w = 1;
  for (std::int32_t num : list) {
    max_w = std::max(max_w, int_value_bit_width(num));
  }
  return max_w;
}

void pack8_values(int width, const std::int32_t* values, int offset, std::uint8_t* buf) {
  int buf_idx = 0;
  int value_idx = offset;
  int left_bit = 0;
  while (value_idx < 8 + offset) {
    int buffer = 0;
    int left_size = 32;
    if (left_bit > 0) {
      buffer |= (values[value_idx] << (32 - left_bit));
      left_size -= left_bit;
      left_bit = 0;
      value_idx++;
    }
    while (left_size >= width && value_idx < 8 + offset) {
      buffer |= (values[value_idx] << (left_size - width));
      left_size -= width;
      value_idx++;
    }
    if (left_size > 0 && value_idx < 8 + offset) {
      buffer |= static_cast<int>(static_cast<std::uint32_t>(values[value_idx]) >> (width - left_size));
      left_bit = width - left_size;
    }
    for (int j = 0; j < 4; j++) {
      buf[buf_idx] = static_cast<std::uint8_t>((static_cast<std::uint32_t>(buffer) >> ((3 - j) * 8)) & 0xff);
      buf_idx++;
      if (buf_idx >= width) return;
    }
  }
}

void append_f32_be(std::vector<std::uint8_t>& out, float f) {
  std::uint32_t b = 0;
  std::memcpy(&b, &f, sizeof(float));
  out.push_back(static_cast<std::uint8_t>((b >> 24) & 0xff));
  out.push_back(static_cast<std::uint8_t>((b >> 16) & 0xff));
  out.push_back(static_cast<std::uint8_t>((b >> 8) & 0xff));
  out.push_back(static_cast<std::uint8_t>(b & 0xff));
}

// org.apache.tsfile.encoding.fire.IntFire (learning_rate = 2)
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

static int sprintz_zigzag(int pred) {
  if (pred <= 0) {
    return -2 * pred;
  }
  return 2 * pred - 1;
}

/** org.apache.tsfile.encoding.encoder.IntSprintzEncoder — scheme "fire" only (TSFileConfig default). */
struct IntSprintzEncoder {
  static constexpr int kBlockSize = 8;
  static constexpr int kGroupMax = 16;

  IntFire fire{};
  bool legacy_rle_tail = false;
  std::vector<std::int32_t> values;
  bool is_first_cached = false;
  int group_num = 0;
  std::vector<std::uint8_t> byte_cache;

  int fire_residual(std::int32_t value, std::int32_t pre_value) {
    int pr = fire.predict(pre_value);
    int err = value - pr;
    fire.train(pre_value, value, err);
    return err;
  }

  int predict_stanza(std::int32_t value, std::int32_t pre_value) {
    int raw = fire_residual(value, pre_value);
    return sprintz_zigzag(raw);
  }

  void bit_pack() {
    std::int32_t pre_value = values[0];
    values.erase(values.begin());
    int bw = int_max_bit_width_vec(values);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(bw));
    pack8_values(bw, values.data(), 0, bytes.data());
    append_int_le_padded(byte_cache, bw, 1);
    append_uvarint(byte_cache, static_cast<std::uint32_t>(pre_value));
    append_bytes(byte_cache, bytes.data(), bytes.size());
  }

  void flush_all(std::vector<std::uint8_t>& out) {
    if (!byte_cache.empty()) {
      append_bytes(out, byte_cache.data(), byte_cache.size());
      byte_cache.clear();
    }
    if (!values.empty()) {
      int sz = static_cast<int>(values.size()) | (1 << 7);
      append_int_le_padded(out, sz, 1);
      std::vector<std::uint8_t> rle = legacy_rle_tail
          ? tsfile::p02::encode_rle_int32_legacy(values)
          : tsfile::p02::encode_rle_int32(values);
      append_bytes(out, rle.data(), rle.size());
      values.clear();
    }
    is_first_cached = false;
    group_num = 0;
  }

  void encode(std::int32_t value, std::vector<std::uint8_t>& out) {
    if (!is_first_cached) {
      values.push_back(value);
      is_first_cached = true;
      return;
    }
    values.push_back(value);
    if (static_cast<int>(values.size()) == kBlockSize + 1) {
      std::int32_t pre = values[0];
      fire.reset();
      for (int i = 1; i <= kBlockSize; i++) {
        std::int32_t tmp = values[i];
        values[i] = predict_stanza(values[i], pre);
        pre = tmp;
      }
      bit_pack();
      is_first_cached = false;
      values.clear();
      group_num++;
      if (group_num == kGroupMax) {
        flush_all(out);
      }
    }
  }
};

struct FloatSprintzEncoder {
  static constexpr int kBlockSize = 8;
  static constexpr int kGroupMax = 16;

  IntFire fire{};
  std::vector<float> values;
  std::vector<std::int32_t> convert_buffer;
  bool is_first_cached = false;
  int group_num = 0;
  std::vector<std::uint8_t> byte_cache;

  int delta_residual(float value, float pre_value) {
    std::uint32_t vb = 0, pb = 0;
    std::memcpy(&vb, &value, sizeof(float));
    std::memcpy(&pb, &pre_value, sizeof(float));
    return static_cast<std::int32_t>(vb) - static_cast<std::int32_t>(pb);
  }

  int fire_residual(float value, float pre_value) {
    std::uint32_t pb = 0, vb = 0;
    std::memcpy(&pb, &pre_value, sizeof(float));
    std::memcpy(&vb, &value, sizeof(float));
    int prev = static_cast<int>(pb);
    int val = static_cast<int>(vb);
    int pred = fire.predict(prev);
    int err = val - pred;
    fire.train(prev, val, err);
    return err;
  }

  int predict_stanza(float value, float pre_value) {
    int raw = fire_residual(value, pre_value);
    return sprintz_zigzag(raw);
  }

  void bit_pack() {
    float pre_value = values[0];
    values.erase(values.begin());
    std::vector<std::int32_t> list(convert_buffer.begin(), convert_buffer.end());
    int bw = int_max_bit_width_vec(list);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(bw));
    pack8_values(bw, convert_buffer.data(), 0, bytes.data());
    append_int_le_padded(byte_cache, bw, 1);
    append_f32_be(byte_cache, pre_value);
    append_bytes(byte_cache, bytes.data(), bytes.size());
  }

  void flush_all(std::vector<std::uint8_t>& out) {
    if (!byte_cache.empty()) {
      append_bytes(out, byte_cache.data(), byte_cache.size());
      byte_cache.clear();
    }
    if (!values.empty()) {
      int sz = static_cast<int>(values.size()) | (1 << 7);
      append_int_le_padded(out, sz, 1);
      std::vector<std::uint8_t> gor = tsfile::p03::encode_gorilla_float(values);
      append_bytes(out, gor.data(), gor.size());
      values.clear();
    }
    is_first_cached = false;
    group_num = 0;
  }

  void encode(float value, std::vector<std::uint8_t>& out) {
    if (!is_first_cached) {
      values.push_back(value);
      is_first_cached = true;
      return;
    }
    values.push_back(value);
    if (static_cast<int>(values.size()) == kBlockSize + 1) {
      fire.reset();
      for (int i = 1; i <= kBlockSize; i++) {
        convert_buffer[static_cast<std::size_t>(i - 1)] = predict_stanza(values[i], values[static_cast<std::size_t>(i - 1)]);
      }
      bit_pack();
      is_first_cached = false;
      values.clear();
      group_num++;
      if (group_num == kGroupMax) {
        flush_all(out);
      }
    }
  }

  FloatSprintzEncoder() : convert_buffer(static_cast<std::size_t>(kBlockSize)) {}
};

/** org.apache.tsfile.encoding.encoder.IntRLBE — blockSize 10000 from RLBE parent. */
struct IntRlbEncoder {
  static constexpr int kBlockSize = 10000;

  std::vector<std::uint8_t> block_bits;
  std::int32_t diff_value[kBlockSize + 1]{};
  int length_code[kBlockSize + 1]{};
  int leng_rle[kBlockSize + 1]{};
  int write_index = -1;
  std::int32_t previous_value = 0;
  std::uint8_t byte_buffer = 0;
  int number_left_in_buffer = 0;

  void reset() {
    write_index = -1;
    byte_buffer = 0;
    number_left_in_buffer = 0;
    std::memset(diff_value, 0, sizeof(diff_value));
    std::memset(length_code, 0, sizeof(length_code));
    std::memset(leng_rle, 0, sizeof(leng_rle));
  }

  // Match Java signed-int bit length without C++ undefined (1<<31).
  static int cal_binary_length(std::int32_t val) {
    if (val == 0) return 1;
    std::uint32_t u = static_cast<std::uint32_t>(val);
    return 32 - __builtin_clz(u);
  }

  static int calc_fibonacci(int val) {
    int fib[200];
    fib[0] = 1;
    fib[1] = 1;
    int i = 2;
    for (; fib[i - 1] <= val; i++) {
      fib[i] = fib[i - 1] + fib[i - 2];
    }
    i--;
    int valfib = 0;
    while (val > 0) {
      while (fib[i] > val && i >= 1) {
        i--;
      }
      valfib |= (1 << (i - 1));
      val -= fib[i];
    }
    return valfib;
  }

  void write_bit(bool b) {
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << 1);
    if (b) {
      byte_buffer = static_cast<std::uint8_t>(byte_buffer | 1);
    }
    number_left_in_buffer++;
    if (number_left_in_buffer == 8) {
      clear_buffer();
    }
  }

  void clear_buffer() {
    if (number_left_in_buffer == 0) return;
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << (8 - number_left_in_buffer));
    block_bits.push_back(byte_buffer);
    number_left_in_buffer = 0;
    byte_buffer = 0;
  }

  void rle_on_length_code() {
    int i = 0;
    while (i <= write_index) {
      int j = i;
      int temprlecal = 0;
      while (j <= write_index && length_code[j] == length_code[i]) {
        j++;
        temprlecal++;
      }
      leng_rle[i] = temprlecal;
      i = j;
    }
  }

  void flush_segment(int i) {
    for (int j = 5; j >= 0; j--) {
      write_bit((length_code[i] & (1 << j)) != 0);
    }
    int fib = calc_fibonacci(leng_rle[i]);
    int fiblen = cal_binary_length(fib);
    for (int j = 0; j < fiblen; j++) {
      write_bit((fib & (1 << j)) != 0);
    }
    write_bit(true);
    int j = i;
    do {
      int temp_difflen = cal_binary_length(diff_value[j]);
      std::uint32_t du = static_cast<std::uint32_t>(diff_value[j]);
      for (int k = temp_difflen - 1; k >= 0; k--) {
        write_bit((du & (1u << k)) != 0);
      }
      j++;
    } while (j <= write_index && leng_rle[j] == 0);
  }

  void write_write_index_bits() {
    std::uint32_t n = static_cast<std::uint32_t>(write_index + 1);
    for (int i = 31; i >= 0; i--) {
      write_bit((n & (1u << i)) != 0);
    }
  }

  void flush_block() {
    if (write_index == -1) return;
    write_write_index_bits();
    rle_on_length_code();
    for (int i = 0; i <= write_index; i++) {
      if (leng_rle[i] > 0) {
        flush_segment(i);
      }
    }
    clear_buffer();
    reset();
  }

  void encode_value(std::int32_t value) {
    if (write_index == -1) {
      write_index = 0;
      diff_value[write_index] = value;
      length_code[write_index] = cal_binary_length(value);
      previous_value = value;
      return;
    }
    write_index++;
    diff_value[write_index] = value - previous_value;
    length_code[write_index] = cal_binary_length(diff_value[write_index]);
    previous_value = value;
    if (write_index == kBlockSize - 1) {
      flush_block();
    }
  }
};

/** org.apache.tsfile.encoding.encoder.FloatRLBE (tsfile 2.0.2): float deltas + raw-bit payload. */
struct FloatRlbEncoder {
  static constexpr int kBlockSize = 10000;

  std::vector<std::uint8_t> block_bits;
  float diff_value[kBlockSize + 1]{};
  int length_code[kBlockSize + 1]{};
  int leng_rle[kBlockSize + 1]{};
  int write_index = -1;
  float previous_value = 0.0f;
  std::uint8_t byte_buffer = 0;
  int number_left_in_buffer = 0;

  void reset() {
    write_index = -1;
    byte_buffer = 0;
    number_left_in_buffer = 0;
    std::memset(diff_value, 0, sizeof(diff_value));
    std::memset(length_code, 0, sizeof(length_code));
    std::memset(leng_rle, 0, sizeof(leng_rle));
  }

  static int cal_binary_length_float(float val) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(float));
    if (bits == 0) return 1;
    return 32 - __builtin_clz(bits);
  }

  static int calc_fibonacci(int val) {
    int fib[200];
    fib[0] = 1;
    fib[1] = 1;
    int i = 2;
    for (; fib[i - 1] <= val; i++) {
      fib[i] = fib[i - 1] + fib[i - 2];
    }
    i--;
    int valfib = 0;
    while (val > 0) {
      while (fib[i] > val && i >= 1) {
        i--;
      }
      valfib |= (1 << (i - 1));
      val -= fib[i];
    }
    return valfib;
  }

  void write_bit(bool b) {
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << 1);
    if (b) {
      byte_buffer = static_cast<std::uint8_t>(byte_buffer | 1);
    }
    number_left_in_buffer++;
    if (number_left_in_buffer == 8) {
      clear_buffer();
    }
  }

  void clear_buffer() {
    if (number_left_in_buffer == 0) return;
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << (8 - number_left_in_buffer));
    block_bits.push_back(byte_buffer);
    number_left_in_buffer = 0;
    byte_buffer = 0;
  }

  void rle_on_length_code() {
    int i = 0;
    while (i <= write_index) {
      int j = i;
      int temprlecal = 0;
      while (j <= write_index && length_code[j] == length_code[i]) {
        j++;
        temprlecal++;
      }
      leng_rle[i] = temprlecal;
      i = j;
    }
  }

  void flush_segment(int i) {
    for (int j = 5; j >= 0; j--) {
      write_bit((length_code[i] & (1 << j)) != 0);
    }
    int fib = calc_fibonacci(leng_rle[i]);
    int fiblen = IntRlbEncoder::cal_binary_length(fib);
    for (int j = 0; j < fiblen; j++) {
      write_bit((fib & (1 << j)) != 0);
    }
    write_bit(true);
    int j = i;
    do {
      int temp_difflen = cal_binary_length_float(diff_value[j]);
      std::uint32_t du = 0;
      std::memcpy(&du, &diff_value[j], sizeof(float));
      for (int k = temp_difflen - 1; k >= 0; k--) {
        write_bit((du & (1u << k)) != 0);
      }
      j++;
    } while (j <= write_index && leng_rle[j] == 0);
  }

  void write_write_index_bits() {
    std::uint32_t n = static_cast<std::uint32_t>(write_index + 1);
    for (int i = 31; i >= 0; i--) {
      write_bit((n & (1u << i)) != 0);
    }
  }

  void flush_block() {
    if (write_index == -1) return;
    write_write_index_bits();
    rle_on_length_code();
    for (int i = 0; i <= write_index; i++) {
      if (leng_rle[i] > 0) {
        flush_segment(i);
      }
    }
    clear_buffer();
    reset();
  }

  void encode_value(float value) {
    if (write_index == -1) {
      write_index = 0;
      diff_value[write_index] = value;
      length_code[write_index] = cal_binary_length_float(value);
      previous_value = value;
      return;
    }
    write_index++;
    diff_value[write_index] = value - previous_value;
    length_code[write_index] = cal_binary_length_float(diff_value[write_index]);
    previous_value = value;
    if (write_index == kBlockSize - 1) {
      flush_block();
    }
  }
};

}  // namespace

std::vector<std::uint8_t> encode_sprintz_int32(std::span<const std::int32_t> values) {
  std::vector<std::uint8_t> out;
  IntSprintzEncoder enc;
  for (std::int32_t v : values) {
    enc.encode(v, out);
  }
  enc.flush_all(out);
  return out;
}

std::vector<std::uint8_t> encode_sprintz_int32_legacy(std::span<const std::int32_t> values) {
  std::vector<std::uint8_t> out;
  IntSprintzEncoder enc;
  enc.legacy_rle_tail = true;
  for (std::int32_t v : values) enc.encode(v, out);
  enc.flush_all(out);
  return out;
}

std::vector<std::uint8_t> encode_sprintz_float(std::span<const float> values) {
  std::vector<std::uint8_t> out;
  FloatSprintzEncoder enc;
  for (float f : values) {
    enc.encode(f, out);
  }
  enc.flush_all(out);
  return out;
}

std::vector<std::uint8_t> encode_sprintz_float_legacy(std::span<const float> values) {
  // Float Sprintz has a Gorilla tail, whose format did not change.
  return encode_sprintz_float(values);
}

std::vector<std::uint8_t> encode_rlbe_int32_legacy(std::span<const std::int32_t> values) {
  IntRlbEncoder enc;
  for (std::int32_t v : values) {
    enc.encode_value(v);
  }
  enc.flush_block();
  return std::move(enc.block_bits);
}

std::vector<std::uint8_t> encode_rlbe_float_legacy(std::span<const float> values) {
  FloatRlbEncoder enc;
  for (float f : values) {
    enc.encode_value(f);
  }
  enc.flush_block();
  return std::move(enc.block_bits);
}


std::vector<std::uint8_t> encode_rlbe_int32(std::span<const std::int32_t> values) {
  return unified_rle_numeric::encode_rlbe(unified_rle_numeric::raw_bytes(values),sizeof(std::int32_t),true);
}


std::vector<std::uint8_t> encode_rlbe_float(std::span<const float> values) {
  return unified_rle_numeric::encode_rlbe_float32_arithmetic(unified_rle_numeric::raw_bytes(values));
}

}  // namespace tsfile::p04
