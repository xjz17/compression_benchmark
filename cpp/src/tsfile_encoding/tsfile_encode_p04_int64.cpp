#include "../unified_rle_numeric.hpp"
#include "tsfile_encode_p04.hpp"

#include "tsfile_encode_p02.hpp"
#include "tsfile_encode_p03.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace tsfile::p04 {

namespace {

void append_bytes(std::vector<std::uint8_t>& dst, const std::uint8_t* p, std::size_t n) {
  dst.insert(dst.end(), p, p + n);
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

void append_i64_be(std::vector<std::uint8_t>& out, std::int64_t v) {
  std::uint64_t b = static_cast<std::uint64_t>(v);
  for (int i = 7; i >= 0; i--) {
    out.push_back(static_cast<std::uint8_t>((b >> (i * 8)) & 0xff));
  }
}

void append_f64_be(std::vector<std::uint8_t>& out, double d) {
  std::uint64_t b = 0;
  std::memcpy(&b, &d, sizeof(double));
  for (int i = 7; i >= 0; i--) {
    out.push_back(static_cast<std::uint8_t>((b >> (i * 8)) & 0xff));
  }
}

int clz_u64(std::uint64_t x) {
  if (x == 0) return 64;
  return __builtin_clzll(x);
}

int long_value_bit_width(std::int64_t num) {
  return 64 - clz_u64(static_cast<std::uint64_t>(num));
}

int long_max_bit_width_vec(const std::vector<std::int64_t>& list) {
  int max_w = 1;
  for (std::int64_t num : list) {
    max_w = std::max(max_w, long_value_bit_width(num));
  }
  return max_w;
}

void pack8_values_long(int width, const std::int64_t* values, int offset, std::uint8_t* buf) {
  int buf_idx = 0;
  int value_idx = offset;
  int left_bit = 0;
  while (value_idx < 8 + offset) {
    std::uint64_t buffer = 0;
    int left_size = 64;
    if (left_bit > 0) {
      buffer |= (static_cast<std::uint64_t>(values[value_idx]) << (64 - left_bit));
      left_size -= left_bit;
      left_bit = 0;
      value_idx++;
    }
    while (left_size >= width && value_idx < 8 + offset) {
      buffer |= (static_cast<std::uint64_t>(values[value_idx]) << (left_size - width));
      left_size -= width;
      value_idx++;
    }
    if (left_size > 0 && value_idx < 8 + offset) {
      buffer |= static_cast<std::uint64_t>(static_cast<std::uint64_t>(values[value_idx]) >>
                                           (width - left_size));
      left_bit = width - left_size;
    }
    for (int j = 0; j < 8; j++) {
      buf[buf_idx] = static_cast<std::uint8_t>((buffer >> ((7 - j) * 8)) & 0xff);
      buf_idx++;
      if (buf_idx >= width) return;
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

static std::int64_t sprintz_zigzag(std::int64_t pred) {
  const std::uint64_t sign = pred < 0 ? std::numeric_limits<std::uint64_t>::max() : 0;
  return static_cast<std::int64_t>((static_cast<std::uint64_t>(pred) << 1) ^ sign);
}

struct LongSprintzEncoder {
  static constexpr int kBlockSize = 8;
  static constexpr int kGroupMax = 16;

  LongFire fire{};
  std::vector<std::int64_t> values;
  bool is_first_cached = false;
  int group_num = 0;
  std::vector<std::uint8_t> byte_cache;
  bool omit_predictor = false;
  bool omit_zigzag = false;
  bool omit_bitpacking = false;
  bool reference_rle_tail = false;

  explicit LongSprintzEncoder(std::string_view omitted_operator, bool reference_tail = false) {
    reference_rle_tail = reference_tail;
    omit_predictor = omitted_operator == "delta";
    omit_zigzag = omitted_operator == "zigzag";
    omit_bitpacking = omitted_operator == "bit-packing";
    if (!omitted_operator.empty() && !omit_predictor && !omit_zigzag &&
        !omit_bitpacking) {
      throw std::invalid_argument("unsupported SPRINTZ ablation operator");
    }
  }

  std::int64_t fire_residual(std::int64_t value, std::int64_t pre_value) {
    std::int64_t pr = fire.predict(pre_value);
    std::int64_t err = static_cast<std::int64_t>(static_cast<std::uint64_t>(value) -
                                                 static_cast<std::uint64_t>(pr));
    fire.train(pre_value, value, err);
    return err;
  }

  std::int64_t predict_stanza(std::int64_t value, std::int64_t pre_value) {
    return sprintz_zigzag(fire_residual(value, pre_value));
  }

  void bit_pack() {
    std::int64_t pre_value = values[0];
    values.erase(values.begin());
    int bw = omit_bitpacking ? 64 : long_max_bit_width_vec(values);
    append_int_le_padded(byte_cache, bw, 1);
    append_i64_be(byte_cache, pre_value);
    if (omit_bitpacking) {
      for (std::int64_t value : values) append_i64_be(byte_cache, value);
    } else {
      std::vector<std::uint8_t> bytes(static_cast<std::size_t>(bw));
      pack8_values_long(bw, values.data(), 0, bytes.data());
      append_bytes(byte_cache, bytes.data(), bytes.size());
    }
  }

  void flush_all(std::vector<std::uint8_t>& out) {
    if (!byte_cache.empty()) {
      append_bytes(out, byte_cache.data(), byte_cache.size());
      byte_cache.clear();
    }
    if (!values.empty()) {
      int sz = static_cast<int>(values.size()) | (1 << 7);
      append_int_le_padded(out, sz, 1);
      if (reference_rle_tail) {
        const auto tail = tsfile::p02::encode_rle_int64(values);
        append_bytes(out, tail.data(), tail.size());
      } else {
        for (std::int64_t value : values) append_i64_be(out, value);
      }
      values.clear();
    }
    is_first_cached = false;
    group_num = 0;
  }

  void encode(std::int64_t value, std::vector<std::uint8_t>& out) {
    if (!is_first_cached) {
      values.push_back(value);
      is_first_cached = true;
      return;
    }
    values.push_back(value);
    if (static_cast<int>(values.size()) == kBlockSize + 1) {
      std::int64_t pre = values[0];
      fire.reset();
      for (int i = 1; i <= kBlockSize; i++) {
        std::int64_t tmp = values[i];
        std::int64_t residual = omit_predictor ? values[i] : fire_residual(values[i], pre);
        values[i] = omit_zigzag ? residual : sprintz_zigzag(residual);
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

struct DoubleSprintzEncoder {
  static constexpr int kBlockSize = 8;
  static constexpr int kGroupMax = 16;

  LongFire fire{};
  std::vector<double> values;
  std::vector<std::int64_t> convert_buffer;
  bool is_first_cached = false;
  int group_num = 0;
  std::vector<std::uint8_t> byte_cache;

  std::int64_t fire_residual(double value, double pre_value) {
    std::uint64_t pb = 0, vb = 0;
    std::memcpy(&pb, &pre_value, sizeof(double));
    std::memcpy(&vb, &value, sizeof(double));
    std::int64_t prev = static_cast<std::int64_t>(pb);
    std::int64_t val = static_cast<std::int64_t>(vb);
    std::int64_t pred = fire.predict(prev);
    std::int64_t err = static_cast<std::int64_t>(static_cast<std::uint64_t>(val) -
                                                 static_cast<std::uint64_t>(pred));
    fire.train(prev, val, err);
    return err;
  }

  std::int64_t predict_stanza(double value, double pre_value) {
    return sprintz_zigzag(fire_residual(value, pre_value));
  }

  void bit_pack() {
    double pre_value = values[0];
    values.erase(values.begin());
    int bw = long_max_bit_width_vec(convert_buffer);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(bw));
    pack8_values_long(bw, convert_buffer.data(), 0, bytes.data());
    append_int_le_padded(byte_cache, bw, 1);
    append_f64_be(byte_cache, pre_value);
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
      if (reference_gorilla_tail) {
        const auto tail = tsfile::p03::encode_gorilla_double(values);
        append_bytes(out, tail.data(), tail.size());
      } else {
        for (double value : values) append_f64_be(out, value);
      }
      values.clear();
    }
    is_first_cached = false;
    group_num = 0;
  }

  void encode(double value, std::vector<std::uint8_t>& out) {
    if (!is_first_cached) {
      values.push_back(value);
      is_first_cached = true;
      return;
    }
    values.push_back(value);
    if (static_cast<int>(values.size()) == kBlockSize + 1) {
      fire.reset();
      for (int i = 1; i <= kBlockSize; i++) {
        convert_buffer[static_cast<std::size_t>(i - 1)] =
            predict_stanza(values[i], values[static_cast<std::size_t>(i - 1)]);
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

  bool reference_gorilla_tail = false;
  explicit DoubleSprintzEncoder(bool reference_tail = false)
      : convert_buffer(static_cast<std::size_t>(kBlockSize)), reference_gorilla_tail(reference_tail) {}
};

struct LongRlbEncoder {
  static constexpr int kBlockSize = 10000;

  std::vector<std::uint8_t> block_bits;
  std::int64_t diff_value[kBlockSize + 1]{};
  int length_code[kBlockSize + 1]{};
  int leng_rle[kBlockSize + 1]{};
  int write_index = -1;
  std::int64_t previous_value = 0;
  std::uint8_t byte_buffer = 0;
  int number_left_in_buffer = 0;
  bool omit_delta = false;
  bool omit_run_length = false;

  explicit LongRlbEncoder(std::string_view omitted_operator) {
    omit_delta = omitted_operator == "delta";
    omit_run_length = omitted_operator == "run length";
    if (!omitted_operator.empty() && !omit_delta && !omit_run_length) {
      throw std::invalid_argument("unsupported RLBE ablation operator");
    }
  }

  void reset() {
    write_index = -1;
    byte_buffer = 0;
    number_left_in_buffer = 0;
    std::memset(diff_value, 0, sizeof(diff_value));
    std::memset(length_code, 0, sizeof(length_code));
    std::memset(leng_rle, 0, sizeof(leng_rle));
  }

  static int cal_binary_length(std::int64_t val) {
    if (val == 0) return 1;
    return 64 - __builtin_clzll(static_cast<std::uint64_t>(val));
  }

  static std::int64_t calc_fibonacci(std::int64_t val) {
    std::int64_t fib[200];
    fib[0] = 1;
    fib[1] = 1;
    int i = 2;
    for (; fib[i - 1] <= val; i++) {
      fib[i] = fib[i - 1] + fib[i - 2];
    }
    i--;
    std::int64_t valfib = 0;
    while (val > 0) {
      while (fib[i] > val && i >= 1) {
        i--;
      }
      valfib |= (static_cast<std::int64_t>(1) << (i - 1));
      val -= fib[i];
    }
    return valfib;
  }

  void write_bit(bool b) {
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << 1);
    if (b) byte_buffer = static_cast<std::uint8_t>(byte_buffer | 1);
    number_left_in_buffer++;
    if (number_left_in_buffer == 8) clear_buffer();
  }

  void clear_buffer() {
    if (number_left_in_buffer == 0) return;
    byte_buffer = static_cast<std::uint8_t>(byte_buffer << (8 - number_left_in_buffer));
    block_bits.push_back(byte_buffer);
    number_left_in_buffer = 0;
    byte_buffer = 0;
  }

  void rle_on_length_code() {
    if (omit_run_length) {
      for (int i = 0; i <= write_index; ++i) leng_rle[i] = 1;
      return;
    }
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
    for (int j = 6; j >= 0; j--) {
      write_bit((length_code[i] & (1 << j)) != 0);
    }
    std::int64_t fib = calc_fibonacci(leng_rle[i]);
    int fiblen = cal_binary_length(fib);
    for (int j = 0; j < fiblen; j++) {
      write_bit((fib & (static_cast<std::int64_t>(1) << j)) != 0);
    }
    write_bit(true);
    int j = i;
    do {
      int temp_difflen = cal_binary_length(diff_value[j]);
      std::uint64_t du = static_cast<std::uint64_t>(diff_value[j]);
      for (int k = temp_difflen - 1; k >= 0; k--) {
        write_bit((du & (1ull << k)) != 0);
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
      if (leng_rle[i] > 0) flush_segment(i);
    }
    clear_buffer();
    reset();
  }

  void encode_value(std::int64_t value) {
    if (write_index == -1) {
      write_index = 0;
      diff_value[write_index] = value;
      length_code[write_index] = cal_binary_length(value);
      previous_value = value;
      return;
    }
    write_index++;
    diff_value[write_index] = omit_delta
        ? value
        : static_cast<std::int64_t>(static_cast<std::uint64_t>(value) -
                                    static_cast<std::uint64_t>(previous_value));
    length_code[write_index] = cal_binary_length(diff_value[write_index]);
    previous_value = value;
    if (write_index == kBlockSize - 1) flush_block();
  }
};

struct DoubleRlbEncoder : LongRlbEncoder {
  using LongRlbEncoder::LongRlbEncoder;
  void encode_value_double(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(double));
    encode_value(static_cast<std::int64_t>(bits));
  }
};

}  // namespace

std::vector<std::uint8_t> encode_sprintz_int64(std::span<const std::int64_t> values) {
  return encode_sprintz_int64_ablation(values, {});
}

std::vector<std::uint8_t> encode_sprintz_int64_ablation(
    std::span<const std::int64_t> values, std::string_view omitted_operator) {
  std::vector<std::uint8_t> out;
  LongSprintzEncoder enc(omitted_operator);
  for (std::int64_t v : values) {
    enc.encode(v, out);
  }
  enc.flush_all(out);
  return out;
}

std::vector<std::uint8_t> encode_sprintz_double(std::span<const double> values) {
  return encode_sprintz_double_ablation(values, {});
}

std::vector<std::uint8_t> encode_sprintz_double_ablation(
    std::span<const double> values, std::string_view omitted_operator) {
  std::vector<std::int64_t> bits;
  bits.reserve(values.size());
  for (double d : values) {
    std::int64_t value_bits = 0;
    std::memcpy(&value_bits, &d, sizeof(value_bits));
    bits.push_back(value_bits);
  }
  return encode_sprintz_int64_ablation(bits, omitted_operator);
}

std::vector<std::uint8_t> encode_rlbe_int64_legacy(std::span<const std::int64_t> values) {
  return encode_rlbe_int64_ablation(values, {});
}

std::vector<std::uint8_t> encode_rlbe_int64_ablation(
    std::span<const std::int64_t> values, std::string_view omitted_operator) {
  LongRlbEncoder enc(omitted_operator);
  for (std::int64_t v : values) {
    enc.encode_value(v);
  }
  enc.flush_block();
  return std::move(enc.block_bits);
}

std::vector<std::uint8_t> encode_rlbe_double_legacy(std::span<const double> values) {
  return encode_rlbe_double_ablation(values, {});
}

std::vector<std::uint8_t> encode_rlbe_double_ablation(
    std::span<const double> values, std::string_view omitted_operator) {
  DoubleRlbEncoder enc(omitted_operator);
  for (double d : values) {
    enc.encode_value_double(d);
  }
  enc.flush_block();
  return std::move(enc.block_bits);
}


std::vector<std::uint8_t> encode_rlbe_int64(std::span<const std::int64_t> values) {
  return unified_rle_numeric::encode_rlbe(unified_rle_numeric::raw_bytes(values),sizeof(std::int64_t),true);
}


std::vector<std::uint8_t> encode_rlbe_double(std::span<const double> values) {
  return unified_rle_numeric::encode_rlbe(unified_rle_numeric::raw_bytes(values),sizeof(double),true);
}


std::vector<std::uint8_t> encode_sprintz_int64_reference(std::span<const std::int64_t> values) {
  LongSprintzEncoder enc({}, true);
  std::vector<std::uint8_t> out;
  for (auto value : values) enc.encode(value, out);
  enc.flush_all(out);return out;
}
std::vector<std::uint8_t> encode_sprintz_double_reference(std::span<const double> values) {
  DoubleSprintzEncoder enc(true);
  std::vector<std::uint8_t> out;
  for (auto value : values) enc.encode(value, out);
  enc.flush_all(out);return out;
}

}  // namespace tsfile::p04
