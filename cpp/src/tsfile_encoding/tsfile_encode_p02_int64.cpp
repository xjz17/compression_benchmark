#include "../unified_rle_numeric.hpp"
#include "../minimum_for.hpp"
#include "tsfile_encode_p02.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace tsfile::p02 {

namespace {

constexpr int kBlockDefaultSize = 128;
constexpr int kRleMinRepeated = 8;
constexpr int kRleMaxRepeated = 0x7FFF;
constexpr int kRleMaxBitPackedGroups = 63;

void append_u32_be(std::vector<std::uint8_t>& out, std::int32_t v) {
  out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
  out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
  out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
  out.push_back(static_cast<std::uint8_t>(v & 0xff));
}

void append_i64_be(std::vector<std::uint8_t>& out, std::int64_t v) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(v) >> shift) & 0xff));
  }
}

void append_uvarint(std::vector<std::uint8_t>& out, std::uint32_t v) {
  while ((v & 0xffffff80u) != 0) {
    out.push_back(static_cast<std::uint8_t>((v & 0x7f) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(v & 0x7f));
}

void append_bytes(std::vector<std::uint8_t>& dst, const std::uint8_t* p, std::size_t n) {
  dst.insert(dst.end(), p, p + n);
}

int clz_u64(std::uint64_t x) {
  if (x == 0) return 64;
  return __builtin_clzll(x);
}

void long_to_bytes(std::int64_t src_num, std::vector<std::uint8_t>& result, int pos, int width) {
  int cnt = pos & 7;
  int index = pos >> 3;
  while (width > 0) {
    int m = (width + cnt >= 8) ? (8 - cnt) : width;
    width -= m;
    int mask = 1 << (8 - cnt);
    cnt += m;
    std::uint8_t y = static_cast<std::uint8_t>(static_cast<std::uint64_t>(src_num) >> width);
    y = static_cast<std::uint8_t>(y << (8 - cnt));
    mask = ~(mask - (1 << (8 - cnt)));
    if (index >= static_cast<int>(result.size())) {
      result.resize(static_cast<std::size_t>(index + 1), 0);
    }
    result[static_cast<std::size_t>(index)] =
        static_cast<std::uint8_t>(result[static_cast<std::size_t>(index)] & mask | y);
    src_num &= ~(-1LL << width);
    if (cnt == 8) {
      index++;
      cnt = 0;
    }
  }
}

void pack8_values_long(int width, const std::int64_t* values, int offset, std::uint8_t* buf) {
  int buf_idx = 0;
  int value_idx = offset;
  int left_bit = 0;
  while (value_idx < 8 + offset) {
    unsigned long long buffer = 0;
    int left_size = 64;
    if (left_bit > 0) {
      buffer |= (static_cast<unsigned long long>(values[value_idx]) << (64 - left_bit));
      left_size -= left_bit;
      left_bit = 0;
      value_idx++;
    }
    while (left_size >= width && value_idx < 8 + offset) {
      buffer |= (static_cast<unsigned long long>(values[value_idx]) << (left_size - width));
      left_size -= width;
      value_idx++;
    }
    if (left_size > 0 && value_idx < 8 + offset) {
      buffer |= static_cast<unsigned long long>(values[value_idx]) >>
                static_cast<unsigned>(width - left_size);
      left_bit = width - left_size;
    }
    for (int j = 0; j < 8 && buf_idx < width; j++) {
      buf[buf_idx++] = static_cast<std::uint8_t>((buffer >> ((7 - j) * 8)) & 0xff);
    }
    if (buf_idx >= width) {
      return;
    }
  }
}

void append_long_le_padded(std::vector<std::uint8_t>& out, std::int64_t value, int bit_width) {
  int padded = (bit_width + 7) / 8;
  int offset = 0;
  while (padded > 0) {
    out.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(value) >> offset) & 0xff));
    offset += 8;
    padded--;
  }
}

int long_max_bit_width(const std::vector<std::int64_t>& list) {
  int max_w = 1;
  for (std::int64_t num : list) {
    const std::uint64_t u = static_cast<std::uint64_t>(num);
    max_w = std::max(max_w, 64 - clz_u64(u));
  }
  return max_w;
}

std::vector<std::uint8_t> encode_int64_delta_block(std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> stream;
  std::vector<std::int64_t> delta_block(static_cast<std::size_t>(kBlockDefaultSize));
  std::vector<std::uint8_t> encoding_block;

  int write_index = -1;
  std::int64_t first_value = 0;
  std::int64_t previous_value = 0;
  std::int64_t min_delta_base = std::numeric_limits<std::int64_t>::max();

  auto flush_block = [&]() {
    if (write_index == -1) return;
    const auto normalized = minimum_for::encode<std::int64_t>(
        std::span<const std::int64_t>(delta_block.data(), static_cast<std::size_t>(write_index)));
    // Preserve the historical singleton block's MAX sentinel and original word width.
    if (write_index != 0) min_delta_base = normalized.minimum;
    for (int i = 0; i < write_index; ++i)
      delta_block[static_cast<std::size_t>(i)] = std::bit_cast<std::int64_t>(normalized.offsets[static_cast<std::size_t>(i)]);
    int write_width = 0;
    for (int i = 0; i < write_index; i++) {
      const std::uint64_t u = static_cast<std::uint64_t>(delta_block[static_cast<std::size_t>(i)]);
      write_width = std::max(write_width, 64 - clz_u64(u));
    }
    append_u32_be(stream, write_index);
    append_u32_be(stream, write_width);
    append_i64_be(stream, min_delta_base);
    append_i64_be(stream, first_value);
    if (write_width > 0) {
      int enc_len = (write_index * write_width + 7) / 8;
      encoding_block.assign(static_cast<std::size_t>(enc_len), 0);
      for (int i = 0; i < write_index; i++) {
        long_to_bytes(delta_block[static_cast<std::size_t>(i)], encoding_block, write_width * i,
                      write_width);
      }
      append_bytes(stream, encoding_block.data(), static_cast<std::size_t>(enc_len));
    }
    write_index = -1;
    min_delta_base = std::numeric_limits<std::int64_t>::max();
    first_value = 0;
    previous_value = 0;
  };

  for (std::int64_t value : values) {
    if (write_index == -1) {
      write_index = 0;
      first_value = value;
      previous_value = first_value;
      continue;
    }
    const std::int64_t delta = value - previous_value;
    delta_block[static_cast<std::size_t>(write_index++)] = delta;
    previous_value = value;
    if (write_index == kBlockDefaultSize) {
      flush_block();
    }
  }
  flush_block();
  return stream;
}

class LongRleEncoder {
 public:
  std::vector<std::uint8_t> flush(std::vector<std::int64_t> values) {
    bit_width_ = long_max_bit_width(values);
    byte_cache_.clear();
    num_buffered_ = 0;
    repeat_count_ = 0;
    bit_packed_group_count_ = 0;
    bytes_buffer_.clear();
    is_bit_pack_run_ = false;
    is_bit_width_saved_ = false;
    pre_value_ = 0;

    for (std::int64_t v : values) {
      encode_value(v);
    }

    int last_bit_packed_num = num_buffered_;
    if (repeat_count_ >= kRleMinRepeated) {
      write_rle_run();
    } else if (num_buffered_ > 0) {
      clear_buffer();
      write_or_append_bit_packed_run();
      end_previous_bit_packed_run(last_bit_packed_num);
    } else {
      end_previous_bit_packed_run(kRleMinRepeated);
    }

    std::vector<std::uint8_t> out;
    append_uvarint(out, static_cast<std::uint32_t>(byte_cache_.size()));
    append_bytes(out, byte_cache_.data(), byte_cache_.size());
    return out;
  }

 private:
  void clear_buffer() {
    for (int i = num_buffered_; i < kRleMinRepeated; i++) {
      buffered_[i] = 0;
    }
  }

  void convert_buffer() {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(bit_width_));
    pack8_values_long(bit_width_, buffered_, 0, bytes.data());
    bytes_buffer_.push_back(std::move(bytes));
  }

  void end_previous_bit_packed_run(int last_bit_packed_num) {
    if (!is_bit_pack_run_) return;
    byte_cache_.push_back(static_cast<std::uint8_t>((bit_packed_group_count_ << 1) | 1));
    byte_cache_.push_back(static_cast<std::uint8_t>(last_bit_packed_num));
    for (const auto& b : bytes_buffer_) {
      append_bytes(byte_cache_, b.data(), b.size());
    }
    bytes_buffer_.clear();
    is_bit_pack_run_ = false;
    bit_packed_group_count_ = 0;
  }

  void write_or_append_bit_packed_run() {
    if (bit_packed_group_count_ >= kRleMaxBitPackedGroups) {
      end_previous_bit_packed_run(kRleMinRepeated);
    }
    if (!is_bit_pack_run_) is_bit_pack_run_ = true;
    convert_buffer();
    num_buffered_ = 0;
    repeat_count_ = 0;
    bit_packed_group_count_++;
  }

  void write_rle_run() {
    end_previous_bit_packed_run(kRleMinRepeated);
    append_uvarint(byte_cache_, static_cast<std::uint32_t>(repeat_count_ << 1));
    append_long_le_padded(byte_cache_, pre_value_, bit_width_);
    repeat_count_ = 0;
    num_buffered_ = 0;
  }

  void encode_value(std::int64_t value) {
    if (!is_bit_width_saved_) {
      byte_cache_.push_back(static_cast<std::uint8_t>(bit_width_));
      is_bit_width_saved_ = true;
    }
    if (value == pre_value_) {
      repeat_count_++;
      if (repeat_count_ >= kRleMinRepeated && repeat_count_ <= kRleMaxRepeated) return;
      if (repeat_count_ == kRleMaxRepeated + 1) {
        repeat_count_ = kRleMaxRepeated;
        write_rle_run();
        repeat_count_ = 1;
        pre_value_ = value;
      }
    } else {
      if (repeat_count_ >= kRleMinRepeated) {
        write_rle_run();
      }
      repeat_count_ = 1;
      pre_value_ = value;
    }
    buffered_[num_buffered_++] = value;
    if (num_buffered_ == kRleMinRepeated) {
      write_or_append_bit_packed_run();
    }
  }

  int bit_width_{1};
  int repeat_count_{0};
  int bit_packed_group_count_{0};
  int num_buffered_{0};
  bool is_bit_pack_run_{false};
  bool is_bit_width_saved_{false};
  std::int64_t pre_value_{0};
  std::int64_t buffered_[kRleMinRepeated]{};
  std::vector<std::vector<std::uint8_t>> bytes_buffer_;
  std::vector<std::uint8_t> byte_cache_;
};

std::int64_t convert_double_to_long(double value, int max_point_number, double max_point_value) {
  if (value * max_point_value > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      value * max_point_value < static_cast<double>(std::numeric_limits<std::int64_t>::min())) {
    if (value > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
        value < static_cast<double>(std::numeric_limits<std::int64_t>::min())) {
      std::int64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(double));
      return bits;
    }
    return static_cast<std::int64_t>(std::llround(value));
  }
  if (std::isnan(value)) {
    std::int64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(double));
    return bits;
  }
  return static_cast<std::int64_t>(std::llround(value * max_point_value));
}

}  // namespace

std::vector<std::uint8_t> encode_ts_2diff_int64(std::span<const std::int64_t> values) {
  return encode_int64_delta_block(values);
}

std::vector<std::uint8_t> encode_rle_int64_legacy(std::span<const std::int64_t> values) {
  std::vector<std::int64_t> copy(values.begin(), values.end());
  LongRleEncoder enc;
  return enc.flush(std::move(copy));
}

std::vector<std::uint8_t> encode_ts_2diff_double(std::span<const double> values, int max_point_number) {
  const int max_point = std::max(0, max_point_number);
  const double max_point_value = std::pow(10.0, max_point);
  std::vector<std::int64_t> longs;
  longs.reserve(values.size());
  for (double d : values) {
    longs.push_back(convert_double_to_long(d, max_point, max_point_value));
  }
  std::vector<std::uint8_t> out;
  append_uvarint(out, static_cast<std::uint32_t>(max_point));
  std::vector<std::uint8_t> body = encode_int64_delta_block(longs);
  append_bytes(out, body.data(), body.size());
  return out;
}

std::vector<std::uint8_t> encode_rle_double_legacy(std::span<const double> values, int max_point_number) {
  (void)max_point_number;
  std::vector<std::uint8_t> out;
  append_uvarint(out, std::numeric_limits<std::uint32_t>::max());
  append_uvarint(out, static_cast<std::uint32_t>(values.size()));
  for (std::size_t begin = 0; begin < values.size();) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &values[begin], sizeof(bits));
    std::size_t end = begin + 1;
    while (end < values.size()) {
      std::uint64_t next_bits = 0;
      std::memcpy(&next_bits, &values[end], sizeof(next_bits));
      if (next_bits != bits || end - begin == std::numeric_limits<std::uint32_t>::max()) {
        break;
      }
      ++end;
    }
    append_uvarint(out, static_cast<std::uint32_t>(end - begin));
    append_i64_be(out, static_cast<std::int64_t>(bits));
    begin = end;
  }
  return out;
}


std::vector<std::uint8_t> encode_rle_int64(std::span<const std::int64_t> values) {
  return unified_rle_numeric::encode_values(values);
}


std::vector<std::uint8_t> encode_rle_double(std::span<const double> values, int max_point_number) {
  (void)max_point_number;
  return unified_rle_numeric::encode_values(values);
}

}  // namespace tsfile::p02
