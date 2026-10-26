#include "leco_operator_ablation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace leco_operator_ablation {
namespace {

constexpr std::size_t kBlockSize = 512;

std::uint64_t to_u64(std::int64_t value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::int64_t to_i64(std::uint64_t value) {
  std::int64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t zigzag_encode(std::int64_t value) {
  return (to_u64(value) << 1) ^
         (value < 0 ? std::numeric_limits<std::uint64_t>::max() : 0U);
}

std::int64_t zigzag_decode(std::uint64_t value) {
  return to_i64((value >> 1) ^ (std::uint64_t{0} - (value & 1U)));
}

int bit_width(std::uint64_t value) {
  return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

void put_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xffU));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void put_double(std::vector<std::uint8_t>& out, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  put_u64(out, bits);
}

std::uint16_t get_u16(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 2 > in.size()) throw std::runtime_error("LECO truncated uint16");
  const std::uint16_t value = static_cast<std::uint16_t>(in[pos]) |
                              (static_cast<std::uint16_t>(in[pos + 1]) << 8);
  pos += 2;
  return value;
}

std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("LECO truncated uint32");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(in[pos++]) << shift;
  }
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("LECO truncated uint64");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(in[pos++]) << shift;
  }
  return value;
}

double get_double(std::span<const std::uint8_t> in, std::size_t& pos) {
  const std::uint64_t bits = get_u64(in, pos);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::int64_t predict(double intercept, double slope, std::size_t index) {
  const long double estimate = static_cast<long double>(intercept) +
                               static_cast<long double>(slope) * index;
  if (estimate <= static_cast<long double>(std::numeric_limits<std::int64_t>::min())) {
    return std::numeric_limits<std::int64_t>::min();
  }
  if (estimate >= static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return static_cast<std::int64_t>(std::llround(estimate));
}

std::pair<double, double> fit_line(std::span<const std::int64_t> values) {
  if (values.size() <= 1) {
    return {values.empty() ? 0.0 : static_cast<double>(values.front()), 0.0};
  }
  const long double mean_x = static_cast<long double>(values.size() - 1) / 2.0L;
  long double mean_y = 0.0L;
  for (std::int64_t value : values) mean_y += static_cast<long double>(value);
  mean_y /= static_cast<long double>(values.size());
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const long double dx = static_cast<long double>(i) - mean_x;
    numerator += dx * (static_cast<long double>(values[i]) - mean_y);
    denominator += dx * dx;
  }
  const double slope = static_cast<double>(numerator / denominator);
  const double intercept = static_cast<double>(mean_y - static_cast<long double>(slope) * mean_x);
  return {intercept, slope};
}

class BitWriter {
 public:
  void write(std::uint64_t value, int width) {
    for (int bit = width - 1; bit >= 0; --bit) {
      current_ = static_cast<std::uint8_t>((current_ << 1) | ((value >> bit) & 1U));
      if (++used_ == 8) {
        bytes_.push_back(current_);
        current_ = 0;
        used_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> finish() {
    if (used_ != 0) bytes_.push_back(static_cast<std::uint8_t>(current_ << (8 - used_)));
    return std::move(bytes_);
  }

 private:
  std::vector<std::uint8_t> bytes_;
  std::uint8_t current_ = 0;
  int used_ = 0;
};

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  std::uint64_t read(int width) {
    std::uint64_t value = 0;
    for (int bit = 0; bit < width; ++bit) {
      if (position_ >= bytes_.size() * 8) throw std::runtime_error("LECO packed body truncated");
      value = (value << 1) |
              ((bytes_[position_ / 8] >> (7 - position_ % 8)) & 1U);
      ++position_;
    }
    return value;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t position_ = 0;
};

}  // namespace

std::vector<std::uint8_t> encode(std::span<const std::int64_t> values,
                                 std::string_view omitted_operator) {
  const bool omit_regression = omitted_operator == "regression";
  const bool omit_bitpacking = omitted_operator == "bit-packing";
  if (!omitted_operator.empty() && !omit_regression && !omit_bitpacking) {
    throw std::invalid_argument("unsupported LECO ablation operator");
  }
  if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("LECO column exceeds uint32 row count");
  }

  std::vector<std::uint8_t> out;
  put_u32(out, static_cast<std::uint32_t>(values.size()));
  for (std::size_t offset = 0; offset < values.size(); offset += kBlockSize) {
    const std::size_t length = std::min(kBlockSize, values.size() - offset);
    const auto block = values.subspan(offset, length);
    const auto [intercept, slope] = omit_regression
        ? std::pair<double, double>{0.0, 0.0}
        : fit_line(block);
    std::vector<std::uint64_t> residuals(length);
    std::uint64_t maximum = 0;
    for (std::size_t i = 0; i < length; ++i) {
      const std::int64_t model = predict(intercept, slope, i);
      const std::int64_t residual = to_i64(to_u64(block[i]) - to_u64(model));
      residuals[i] = zigzag_encode(residual);
      maximum = std::max(maximum, residuals[i]);
    }
    const int width = omit_bitpacking ? 64 : bit_width(maximum);
    put_u16(out, static_cast<std::uint16_t>(length));
    put_double(out, intercept);
    put_double(out, slope);
    out.push_back(static_cast<std::uint8_t>(width));
    if (omit_bitpacking) {
      for (std::uint64_t residual : residuals) put_u64(out, residual);
    } else {
      BitWriter writer;
      for (std::uint64_t residual : residuals) writer.write(residual, width);
      auto body = writer.finish();
      out.insert(out.end(), body.begin(), body.end());
    }
  }
  return out;
}

std::vector<std::int64_t> decode(std::span<const std::uint8_t> encoded,
                                 std::string_view omitted_operator) {
  const bool omit_bitpacking = omitted_operator == "bit-packing";
  std::size_t pos = 0;
  const std::uint32_t count = get_u32(encoded, pos);
  std::vector<std::int64_t> values;
  values.reserve(count);
  while (values.size() < count) {
    const std::uint16_t length = get_u16(encoded, pos);
    const double intercept = get_double(encoded, pos);
    const double slope = get_double(encoded, pos);
    if (pos >= encoded.size()) throw std::runtime_error("LECO truncated block width");
    const int width = encoded[pos++];
    if (length == 0 || width < 0 || width > 64 || values.size() + length > count) {
      throw std::runtime_error("LECO invalid block header");
    }
    if (omit_bitpacking) {
      for (std::size_t i = 0; i < length; ++i) {
        const std::int64_t residual = zigzag_decode(get_u64(encoded, pos));
        values.push_back(to_i64(to_u64(predict(intercept, slope, i)) + to_u64(residual)));
      }
    } else {
      const std::size_t body_size = (static_cast<std::size_t>(width) * length + 7) / 8;
      if (pos + body_size > encoded.size()) throw std::runtime_error("LECO block body truncated");
      BitReader reader(encoded.subspan(pos, body_size));
      for (std::size_t i = 0; i < length; ++i) {
        const std::int64_t residual = zigzag_decode(reader.read(width));
        values.push_back(to_i64(to_u64(predict(intercept, slope, i)) + to_u64(residual)));
      }
      pos += body_size;
    }
  }
  if (pos != encoded.size()) throw std::runtime_error("LECO trailing bytes");
  return values;
}

}  // namespace leco_operator_ablation
