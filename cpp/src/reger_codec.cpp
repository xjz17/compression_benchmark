#include "reger_codec.hpp"
#include "minimum_for.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace reger_codec {
namespace {

using clock_ns = std::chrono::high_resolution_clock;

constexpr char kMagic[6] = {'R', 'E', 'G', 'E', 'R', '3'};
constexpr char kDoubleMagic[6] = {'R', 'E', 'G', 'D', 'B', '1'};
constexpr std::uint8_t kDoubleRawBits = 0;
constexpr std::uint8_t kDoubleDecimal = 1;
constexpr int kMaxDoubleDecimalScale = 15;
constexpr std::uint8_t kRawSeries = 0;
constexpr std::uint8_t kConstSeries = 1;
constexpr std::uint8_t kForSeries = 2;
constexpr std::uint8_t kDeltaSeries = 3;
constexpr std::uint8_t kTimeLinearSeries = 4;
constexpr std::uint8_t kPrevLinearSeries = 5;
constexpr std::uint8_t kOrderPermutedFlag = 1;
constexpr std::uint8_t kTimeStreamFlag = 2;
constexpr std::uint8_t kOrderRawFlag = 4;
constexpr std::size_t kSeriesHeaderSize = 5;

struct SeriesPayload {
  std::uint8_t mode = kRawSeries;
  std::vector<std::uint8_t> payload;
};

struct BlockPayload {
  std::uint8_t flags = 0;
  std::vector<std::uint8_t> payload;
};

enum class RegerProfile { kBalanced, kFast };
enum class RegerAblation { kNone, kValueReorder, kRegression, kFor, kBitPacking };

thread_local RegerAblation g_ablation = RegerAblation::kNone;

RegerProfile reger_profile() {
  const char* raw = std::getenv("WEB_COMPRESSION_REGER_PROFILE");
  if (raw == nullptr) return RegerProfile::kBalanced;
  const std::string value(raw);
  return value == "fast" || value == "FAST" ? RegerProfile::kFast
                                               : RegerProfile::kBalanced;
}

bool fast_profile() { return reger_profile() == RegerProfile::kFast; }

bool operator_is_ablated(RegerAblation operator_id) { return g_ablation == operator_id; }

int default_block_size() {
  const char* raw = std::getenv("WEB_COMPRESSION_REGER_BLOCK_SIZE");
  if (raw == nullptr || raw[0] == '\0') return 513;
  try {
    const int n = std::stoi(raw);
    return n > 0 ? n : 513;
  } catch (...) {
    return 513;
  }
}

int default_segment_size() {
  const char* raw = std::getenv("WEB_COMPRESSION_REGER_SEGMENT_SIZE");
  if (raw == nullptr || raw[0] == '\0') return 16;
  try {
    const int n = std::stoi(raw);
    return n > 0 ? n : 16;
  } catch (...) {
    return 16;
  }
}

int default_reorder_iterations(std::size_t column_count) {
  const char* raw = std::getenv("WEB_COMPRESSION_REGER_REORDER_ITERS");
  if (raw == nullptr || raw[0] == '\0') {
    return fast_profile() ? 0 : (column_count == 1 ? 4 : 0);
  }
  try {
    return std::clamp(std::stoi(raw), 0, 20);
  } catch (...) {
    return 4;
  }
}

bool candidate_pruning_enabled() {
  const char* raw = std::getenv("WEB_COMPRESSION_REGER_EXHAUSTIVE");
  if (raw == nullptr) return true;
  const std::string value(raw);
  return !(value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
           value == "YES" || value == "on" || value == "ON");
}

std::size_t high_column_candidate_budget(std::size_t column_count,
                                         std::size_t candidate_count) {
  if (!candidate_pruning_enabled() || column_count < 32) return candidate_count;
  const char* raw = std::getenv("WEB_COMPRESSION_REGER_HIGH_COLUMN_CANDIDATES");
  if (raw == nullptr || raw[0] == '\0') return std::min<std::size_t>(16, candidate_count);
  try {
    const int requested = std::stoi(raw);
    if (requested <= 0) return candidate_count;
    return std::min<std::size_t>(static_cast<std::size_t>(requested), candidate_count);
  } catch (...) {
    return std::min<std::size_t>(16, candidate_count);
  }
}

std::uint64_t to_u64(std::int64_t value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::int64_t to_i64(std::uint64_t value) {
  std::int64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::uint64_t double_to_u64(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

double u64_to_double(std::uint64_t value) {
  double out = 0.0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

void put_u8(std::vector<std::uint8_t>& out, std::uint8_t value) { out.push_back(value); }

void put_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xffu));
  out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
  }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
  }
}

void put_i64(std::vector<std::uint8_t>& out, std::int64_t value) { put_u64(out, to_u64(value)); }

void put_f32(std::vector<std::uint8_t>& out, float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  put_u32(out, bits);
}

std::uint8_t read_u8(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  if (pos >= len) throw std::runtime_error("REGER truncated u8");
  return data[pos++];
}

std::uint16_t read_u16(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  if (pos + 2 > len) throw std::runtime_error("REGER truncated u16");
  const std::uint16_t value = static_cast<std::uint16_t>(data[pos]) |
                              (static_cast<std::uint16_t>(data[pos + 1]) << 8);
  pos += 2;
  return value;
}

std::uint32_t read_u32(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  if (pos + 4 > len) throw std::runtime_error("REGER truncated u32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(data[pos + i]) << (8 * i);
  pos += 4;
  return value;
}

std::uint64_t read_u64(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  if (pos + 8 > len) throw std::runtime_error("REGER truncated u64");
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(data[pos + i]) << (8 * i);
  pos += 8;
  return value;
}

std::int64_t read_i64(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  return to_i64(read_u64(data, len, pos));
}

float read_f32(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  const std::uint32_t bits = read_u32(data, len, pos);
  float value = 0.0F;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

int bit_width_u64(std::uint64_t value) {
  if (value == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
  return 64 - __builtin_clzll(value);
#else
  int width = 0;
  while (value != 0) {
    ++width;
    value >>= 1;
  }
  return width;
#endif
}

int ceil_log2(std::size_t n) {
  if (n <= 1) return 0;
  return bit_width_u64(static_cast<std::uint64_t>(n - 1));
}

class BitWriter {
 public:
  void write(std::uint64_t value, int bits) {
    while (bits > 0) {
      const int take = std::min(8 - used_, bits);
      const int shift = bits - take;
      const std::uint64_t mask = (1ULL << take) - 1ULL;
      cur_ = static_cast<std::uint8_t>((cur_ << take) | ((value >> shift) & mask));
      used_ += take;
      bits -= take;
      if (used_ == 8) {
        out_.push_back(cur_);
        cur_ = 0;
        used_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> finish() {
    if (used_ > 0) out_.push_back(static_cast<std::uint8_t>(cur_ << (8 - used_)));
    used_ = 0;
    cur_ = 0;
    return out_;
  }

 private:
  std::vector<std::uint8_t> out_;
  std::uint8_t cur_ = 0;
  int used_ = 0;
};

class BitReader {
 public:
  BitReader(const std::uint8_t* data, std::size_t len) : data_(data), len_(len) {}

  std::uint64_t read(int bits) {
    std::uint64_t out = 0;
    for (int i = 0; i < bits; ++i) {
      if ((pos_ >> 3) >= len_) throw std::runtime_error("REGER bitread past end");
      const std::uint8_t byte = data_[pos_ >> 3];
      const int shift = 7 - static_cast<int>(pos_ & 7);
      out = (out << 1) | ((byte >> shift) & 1u);
      ++pos_;
    }
    return out;
  }

 private:
  const std::uint8_t* data_;
  std::size_t len_;
  std::size_t pos_ = 0;
};

std::vector<std::uint8_t> pack_values(const std::vector<std::uint64_t>& values, int width) {
  if (values.empty() || width <= 0) return {};
  if (width >= 64) {
    std::vector<std::uint8_t> out;
    out.reserve(values.size() * 8);
    for (std::uint64_t value : values) put_u64(out, value);
    return out;
  }
  BitWriter writer;
  for (std::uint64_t value : values) writer.write(value, width);
  return writer.finish();
}

std::vector<std::uint8_t> pack_values_range(const std::vector<std::uint64_t>& values,
                                            std::size_t start,
                                            std::size_t end,
                                            int width) {
  if (start >= end || width <= 0) return {};
  if (width >= 64) {
    std::vector<std::uint8_t> out;
    out.reserve((end - start) * 8);
    for (std::size_t i = start; i < end; ++i) put_u64(out, values[i]);
    return out;
  }
  BitWriter writer;
  for (std::size_t i = start; i < end; ++i) writer.write(values[i], width);
  return writer.finish();
}

std::vector<std::uint64_t> unpack_values(const std::uint8_t* data, std::size_t len,
                                         std::size_t count, int width) {
  std::vector<std::uint64_t> out(count, 0);
  if (width <= 0) return out;
  if (width >= 64) {
    std::size_t pos = 0;
    for (std::size_t i = 0; i < count; ++i) out[i] = read_u64(data, len, pos);
    return out;
  }
  BitReader reader(data, len);
  for (std::size_t i = 0; i < count; ++i) out[i] = reader.read(width);
  return out;
}


void put_rle_widths(std::vector<std::uint8_t>& out, const std::vector<int>& widths) {
  std::vector<std::pair<int, int>> runs;
  for (int width : widths) {
    if (runs.empty() || runs.back().second != width || runs.back().first == 255) {
      runs.push_back({1, width});
    } else {
      ++runs.back().first;
    }
  }
  put_u16(out, static_cast<std::uint16_t>(runs.size()));
  for (const auto& [count, width] : runs) {
    put_u8(out, static_cast<std::uint8_t>(count));
    put_u8(out, static_cast<std::uint8_t>(width));
  }
}

std::vector<int> read_rle_widths(const std::uint8_t* data, std::size_t len, std::size_t& pos) {
  const std::uint16_t run_count = read_u16(data, len, pos);
  std::vector<int> widths;
  for (std::uint16_t i = 0; i < run_count; ++i) {
    const int count = read_u8(data, len, pos);
    const int width = read_u8(data, len, pos);
    for (int j = 0; j < count; ++j) widths.push_back(width);
  }
  return widths;
}

std::vector<std::uint8_t> encode_segmented_diffs(const std::vector<std::uint64_t>& diffs,
                                                 int segment_size) {
  std::vector<int> widths;
  for (std::size_t start = 0; start < diffs.size(); start += segment_size) {
    const std::size_t end = std::min(diffs.size(), start + static_cast<std::size_t>(segment_size));
    std::uint64_t max_value = 0;
    for (std::size_t i = start; i < end; ++i) {
      max_value = std::max(max_value, diffs[i]);
    }
    widths.push_back(operator_is_ablated(RegerAblation::kBitPacking)
                         ? 64
                         : bit_width_u64(max_value));
  }
  std::vector<std::uint8_t> out;
  put_rle_widths(out, widths);
  std::size_t segment_index = 0;
  for (std::size_t start = 0; start < diffs.size(); start += segment_size, ++segment_index) {
    const std::size_t end = std::min(diffs.size(), start + static_cast<std::size_t>(segment_size));
    auto body = pack_values_range(diffs, start, end, widths[segment_index]);
    out.insert(out.end(), body.begin(), body.end());
  }
  return out;
}

std::vector<std::uint64_t> decode_segmented_diffs(const std::uint8_t* data, std::size_t len,
                                                  std::size_t count, int segment_size,
                                                  std::size_t& pos) {
  const auto widths = read_rle_widths(data, len, pos);
  std::vector<std::uint64_t> out;
  out.reserve(count);
  for (int width : widths) {
    if (out.size() >= count) break;
    const std::size_t n = std::min<std::size_t>(segment_size, count - out.size());
    const std::size_t body_len = width > 0 ? (n * static_cast<std::size_t>(width) + 7) / 8 : 0;
    if (pos + body_len > len) throw std::runtime_error("REGER segmented body truncated");
    auto chunk = unpack_values(data + pos, body_len, n, width);
    pos += body_len;
    out.insert(out.end(), chunk.begin(), chunk.end());
  }
  if (out.size() != count) throw std::runtime_error("REGER segmented length mismatch");
  return out;
}

SeriesPayload raw_series(const std::vector<std::int64_t>& values) {
  std::vector<std::uint8_t> out;
  out.reserve(values.size() * 8);
  for (std::int64_t value : values) put_i64(out, value);
  return {kRawSeries, std::move(out)};
}

SeriesPayload const_series(std::int64_t value) {
  std::vector<std::uint8_t> out;
  put_i64(out, value);
  return {kConstSeries, std::move(out)};
}

SeriesPayload for_series(const std::vector<std::int64_t>& values, int segment_size) {
  auto normalized = minimum_for::encode<std::int64_t>(values);
  const auto minimum = normalized.minimum;
  auto diffs = std::move(normalized.offsets);
  std::vector<std::uint8_t> out;
  put_i64(out, minimum);
  auto body = encode_segmented_diffs(diffs, segment_size);
  out.insert(out.end(), body.begin(), body.end());
  return {kForSeries, std::move(out)};
}

SeriesPayload delta_series(const std::vector<std::int64_t>& values, int segment_size) {
  std::vector<std::int64_t> deltas;
  deltas.reserve(values.size() - 1);
  for (std::size_t i = 1; i < values.size(); ++i) {
    const __int128 delta = static_cast<__int128>(values[i]) - static_cast<__int128>(values[i - 1]);
    if (delta < std::numeric_limits<std::int64_t>::min() ||
        delta > std::numeric_limits<std::int64_t>::max()) {
      throw std::overflow_error("REGER delta out of int64 range");
    }
    deltas.push_back(static_cast<std::int64_t>(delta));
  }
  auto normalized = minimum_for::encode<std::int64_t>(deltas);
  const auto minimum = normalized.minimum;
  auto diffs = std::move(normalized.offsets);
  std::vector<std::uint8_t> out;
  put_i64(out, values.front());
  put_i64(out, minimum);
  auto body = encode_segmented_diffs(diffs, segment_size);
  out.insert(out.end(), body.begin(), body.end());
  return {kDeltaSeries, std::move(out)};
}

std::pair<float, float> fit_linear(const std::vector<long double>& xs,
                                   const std::vector<std::int64_t>& ys) {
  if (xs.size() != ys.size()) throw std::invalid_argument("REGER regression length mismatch");
  if (xs.empty()) return {0.0F, 0.0F};
  long double sx = 0.0L;
  long double sy = 0.0L;
  long double sxx = 0.0L;
  long double sxy = 0.0L;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const long double x = xs[i];
    const long double y = static_cast<long double>(ys[i]);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const long double n = static_cast<long double>(xs.size());
  const long double denominator = n * sxx - sx * sx;
  long double theta0 = 0.0L;
  long double theta1 = 0.0L;
  if (std::abs(denominator) <= 1e-12L) {
    theta0 = sy / n;
  } else {
    theta1 = (n * sxy - sx * sy) / denominator;
    theta0 = (sy - theta1 * sx) / n;
  }
  const float out0 = static_cast<float>(theta0);
  const float out1 = static_cast<float>(theta1);
  if (!std::isfinite(out0) || !std::isfinite(out1)) {
    throw std::overflow_error("REGER regression coefficients are not finite float32 values");
  }
  return {out0, out1};
}

std::int64_t predict(float theta0, float theta1, long double x) {
  const long double raw = static_cast<long double>(theta0) +
                          static_cast<long double>(theta1) * x;
  if (!std::isfinite(raw)) throw std::overflow_error("REGER prediction is not finite");
  if (raw <= static_cast<long double>(std::numeric_limits<std::int64_t>::min())) {
    return std::numeric_limits<std::int64_t>::min();
  }
  if (raw >= static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return static_cast<std::int64_t>(raw);
}

std::int64_t checked_residual(std::int64_t value, std::int64_t prediction) {
  const __int128 residual = static_cast<__int128>(value) - static_cast<__int128>(prediction);
  if (residual < std::numeric_limits<std::int64_t>::min() ||
      residual > std::numeric_limits<std::int64_t>::max()) {
    throw std::overflow_error("REGER regression residual exceeds int64");
  }
  return static_cast<std::int64_t>(residual);
}

SeriesPayload time_linear_series(const std::vector<std::int64_t>& values,
                                 const std::vector<std::int64_t>& times,
                                 int segment_size) {
  if (values.empty() || values.size() != times.size()) {
    throw std::invalid_argument("REGER time-linear input length mismatch");
  }
  const long double t0 = static_cast<long double>(times.front());
  std::vector<long double> xs;
  xs.reserve(times.size());
  for (std::int64_t time : times) xs.push_back(static_cast<long double>(time) - t0);
  const auto [theta0, theta1] = fit_linear(xs, values);
  std::vector<std::int64_t> residuals;
  residuals.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    residuals.push_back(checked_residual(values[i], predict(theta0, theta1, xs[i])));
  }
  auto normalized = minimum_for::encode<std::int64_t>(residuals);
  const auto minimum = normalized.minimum;
  auto diffs = std::move(normalized.offsets);
  std::vector<std::uint8_t> out;
  put_f32(out, theta0);
  put_f32(out, theta1);
  put_i64(out, minimum);
  auto body = encode_segmented_diffs(diffs, segment_size);
  out.insert(out.end(), body.begin(), body.end());
  return {kTimeLinearSeries, std::move(out)};
}

SeriesPayload prev_linear_series(const std::vector<std::int64_t>& values, int segment_size) {
  if (values.size() <= 1) throw std::invalid_argument("REGER previous-linear input too short");
  std::vector<long double> xs;
  std::vector<std::int64_t> ys;
  xs.reserve(values.size() - 1);
  ys.reserve(values.size() - 1);
  for (std::size_t i = 1; i < values.size(); ++i) {
    xs.push_back(static_cast<long double>(values[i - 1]));
    ys.push_back(values[i]);
  }
  const auto [theta0, theta1] = fit_linear(xs, ys);
  std::vector<std::int64_t> residuals;
  residuals.reserve(ys.size());
  for (std::size_t i = 0; i < ys.size(); ++i) {
    residuals.push_back(checked_residual(ys[i], predict(theta0, theta1, xs[i])));
  }
  auto normalized = minimum_for::encode<std::int64_t>(residuals);
  const auto minimum = normalized.minimum;
  auto diffs = std::move(normalized.offsets);
  std::vector<std::uint8_t> out;
  put_i64(out, values.front());
  put_f32(out, theta0);
  put_f32(out, theta1);
  put_i64(out, minimum);
  auto body = encode_segmented_diffs(diffs, segment_size);
  out.insert(out.end(), body.begin(), body.end());
  return {kPrevLinearSeries, std::move(out)};
}

SeriesPayload best_series(const std::vector<std::int64_t>& values, int segment_size,
                          const std::vector<std::int64_t>* times = nullptr) {
  SeriesPayload best;
  bool raw_is_best = true;
  std::size_t best_size = values.size() * 8;
  auto take = [&](const SeriesPayload& candidate) {
    if (candidate.payload.size() < best_size) {
      best = candidate;
      best_size = candidate.payload.size();
      raw_is_best = false;
    }
  };
  if (values.empty()) return raw_series(values);
  bool constant = true;
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] != values[0]) {
      constant = false;
      break;
    }
  }
  if (constant) take(const_series(values[0]));
  const bool fast = fast_profile();
  if (!operator_is_ablated(RegerAblation::kFor)) {
    try {
      take(for_series(values, segment_size));
    } catch (...) {
    }
  }
  if (values.size() > 1) {
    try {
      take(delta_series(values, segment_size));
    } catch (...) {
    }
    if (!fast && !operator_is_ablated(RegerAblation::kRegression)) {
      try {
        take(prev_linear_series(values, segment_size));
      } catch (...) {
      }
    }
  }
  if (times != nullptr && !operator_is_ablated(RegerAblation::kRegression)) {
    try {
      take(time_linear_series(values, *times, segment_size));
    } catch (...) {
    }
  }
  return raw_is_best ? raw_series(values) : best;
}

void write_series(std::vector<std::uint8_t>& out, const SeriesPayload& payload) {
  put_u8(out, payload.mode);
  put_u32(out, static_cast<std::uint32_t>(payload.payload.size()));
  out.insert(out.end(), payload.payload.begin(), payload.payload.end());
}

std::vector<std::int64_t> read_series(const std::uint8_t* data, std::size_t len,
                                      std::size_t count, int segment_size,
                                      std::size_t& pos,
                                      const std::vector<std::int64_t>* times = nullptr) {
  const std::uint8_t mode = read_u8(data, len, pos);
  const std::uint32_t payload_len = read_u32(data, len, pos);
  if (pos + payload_len > len) throw std::runtime_error("REGER series payload truncated");
  const std::uint8_t* payload = data + pos;
  std::size_t p = 0;
  pos += payload_len;
  if (mode == kRawSeries) {
    std::vector<std::int64_t> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) out.push_back(read_i64(payload, payload_len, p));
    return out;
  }
  if (mode == kConstSeries) {
    const std::int64_t value = count > 0 ? read_i64(payload, payload_len, p) : 0;
    return std::vector<std::int64_t>(count, value);
  }
  if (mode == kForSeries) {
    const std::int64_t minimum = read_i64(payload, payload_len, p);
    auto diffs = decode_segmented_diffs(payload, payload_len, count, segment_size, p);
    std::vector<std::int64_t> out;
    out.reserve(count);
    for (std::uint64_t diff : diffs) out.push_back(minimum_for::restore(minimum, diff));
    return out;
  }
  if (mode == kDeltaSeries) {
    if (count == 0) return {};
    const std::int64_t first = read_i64(payload, payload_len, p);
    const std::int64_t minimum = read_i64(payload, payload_len, p);
    auto diffs = decode_segmented_diffs(payload, payload_len, count - 1, segment_size, p);
    std::vector<std::int64_t> out;
    out.reserve(count);
    out.push_back(first);
    for (std::uint64_t diff : diffs) {
      out.push_back(minimum_for::restore(out.back(), to_u64(minimum_for::restore(minimum, diff))));
    }
    return out;
  }
  if (mode == kTimeLinearSeries) {
    if (times == nullptr || times->size() != count) {
      throw std::runtime_error("REGER time-linear stream has no matching time stream");
    }
    const float theta0 = read_f32(payload, payload_len, p);
    const float theta1 = read_f32(payload, payload_len, p);
    const std::int64_t minimum = read_i64(payload, payload_len, p);
    auto diffs = decode_segmented_diffs(payload, payload_len, count, segment_size, p);
    std::vector<std::int64_t> out;
    out.reserve(count);
    const long double t0 = count > 0 ? static_cast<long double>((*times)[0]) : 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
      const long double x = static_cast<long double>((*times)[i]) - t0;
      const __int128 value = static_cast<__int128>(predict(theta0, theta1, x)) +
                             minimum_for::restore_wide(minimum, diffs[i]);
      if (value < std::numeric_limits<std::int64_t>::min() ||
          value > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("REGER decoded time-linear value exceeds int64");
      }
      out.push_back(static_cast<std::int64_t>(value));
    }
    return out;
  }
  if (mode == kPrevLinearSeries) {
    if (count == 0) return {};
    const std::int64_t first = read_i64(payload, payload_len, p);
    const float theta0 = read_f32(payload, payload_len, p);
    const float theta1 = read_f32(payload, payload_len, p);
    const std::int64_t minimum = read_i64(payload, payload_len, p);
    auto diffs = decode_segmented_diffs(payload, payload_len, count - 1, segment_size, p);
    std::vector<std::int64_t> out;
    out.reserve(count);
    out.push_back(first);
    for (std::uint64_t diff : diffs) {
      const __int128 value =
          static_cast<__int128>(predict(theta0, theta1,
                                        static_cast<long double>(out.back()))) +
          minimum_for::restore_wide(minimum, diff);
      if (value < std::numeric_limits<std::int64_t>::min() ||
          value > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("REGER decoded previous-linear value exceeds int64");
      }
      out.push_back(static_cast<std::int64_t>(value));
    }
    return out;
  }
  throw std::runtime_error("invalid REGER series mode");
}

std::size_t common_row_count(const std::vector<std::vector<std::int64_t>>& columns) {
  if (columns.empty()) return 0;
  std::size_t n = columns.front().size();
  for (const auto& col : columns) n = std::min(n, col.size());
  return n;
}

std::vector<std::int64_t> normalized_times(const std::vector<std::int64_t>* times,
                                           std::size_t row_count) {
  if (times == nullptr) {
    std::vector<std::int64_t> out(row_count);
    for (std::size_t i = 0; i < row_count; ++i) out[i] = static_cast<std::int64_t>(i);
    return out;
  }
  if (times->size() < row_count) throw std::runtime_error("REGER time column too short");
  return std::vector<std::int64_t>(times->begin(), times->begin() + static_cast<std::ptrdiff_t>(row_count));
}

void append_order(std::vector<std::vector<std::size_t>>& orders, std::vector<std::size_t> order) {
  for (const auto& existing : orders) {
    if (existing == order) return;
  }
  orders.push_back(std::move(order));
}

bool is_identity_order(const std::vector<std::size_t>& order) {
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] != i) return false;
  }
  return true;
}

std::vector<std::size_t> partition_order(const std::vector<std::int64_t>& values,
                                         const std::vector<std::int64_t>& times,
                                         const std::vector<std::size_t>& base) {
  if (base.size() < 8) return {};
  std::vector<std::int64_t> sorted_values;
  sorted_values.reserve(base.size());
  for (std::size_t index : base) sorted_values.push_back(values[index]);
  std::sort(sorted_values.begin(), sorted_values.end());
  const std::array<std::int64_t, 3> thresholds = {
      sorted_values[sorted_values.size() / 4],
      sorted_values[sorted_values.size() / 2],
      sorted_values[(sorted_values.size() * 3) / 4],
  };
  if (thresholds[0] == thresholds[2]) return {};
  auto order = base;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const auto bucket_a = std::upper_bound(thresholds.begin(), thresholds.end(), values[a]);
    const auto bucket_b = std::upper_bound(thresholds.begin(), thresholds.end(), values[b]);
    if (bucket_a != bucket_b) return bucket_a < bucket_b;
    if (times[a] != times[b]) return times[a] < times[b];
    return a < b;
  });
  return order;
}

std::vector<std::vector<std::size_t>> row_order_candidates(
    const std::vector<std::vector<std::int64_t>>& block_columns,
    const std::vector<std::int64_t>& block_times) {
  const std::size_t n = common_row_count(block_columns);
  std::vector<std::vector<std::size_t>> orders;
  std::vector<std::size_t> base(n);
  for (std::size_t i = 0; i < n; ++i) base[i] = i;
  append_order(orders, base);
  if (operator_is_ablated(RegerAblation::kValueReorder)) return orders;

  auto time_order = base;
  std::sort(time_order.begin(), time_order.end(), [&](std::size_t a, std::size_t b) {
    if (block_times[a] != block_times[b]) return block_times[a] < block_times[b];
    return a < b;
  });
  append_order(orders, std::move(time_order));

  // The fast profile avoids the O(columns * rows * log(rows)) value-order search.
  // Identity and timestamp order cover the common time-series ingestion cases.
  if (fast_profile()) return orders;

  for (const auto& col : block_columns) {
    auto value_order = base;
    std::sort(value_order.begin(), value_order.end(), [&](std::size_t a, std::size_t b) {
      if (col[a] != col[b]) return col[a] < col[b];
      if (block_times[a] != block_times[b]) return block_times[a] < block_times[b];
      return a < b;
    });
    append_order(orders, std::move(value_order));
    auto partition = partition_order(col, block_times, base);
    if (!partition.empty()) append_order(orders, std::move(partition));
  }

  if (block_columns.size() > 1) {
    auto lex_order = base;
    std::sort(lex_order.begin(), lex_order.end(), [&](std::size_t a, std::size_t b) {
      for (const auto& col : block_columns) {
        if (col[a] != col[b]) return col[a] < col[b];
      }
      if (block_times[a] != block_times[b]) return block_times[a] < block_times[b];
      return a < b;
    });
    append_order(orders, std::move(lex_order));
  }
  return orders;
}

std::vector<std::uint8_t> encode_order(const std::vector<std::size_t>& order, bool raw) {
  if (raw) {
    std::vector<std::uint8_t> out;
    out.reserve(order.size() * sizeof(std::uint64_t));
    for (std::size_t value : order) put_u64(out, static_cast<std::uint64_t>(value));
    return out;
  }
  const int width = ceil_log2(order.size());
  std::vector<std::uint64_t> vals;
  vals.reserve(order.size());
  for (std::size_t v : order) vals.push_back(static_cast<std::uint64_t>(v));
  return pack_values(vals, width);
}

std::vector<std::size_t> decode_order(const std::uint8_t* data, std::size_t len, std::size_t n,
                                      std::uint8_t flags, std::size_t& pos) {
  std::vector<std::size_t> order(n);
  if ((flags & kOrderPermutedFlag) == 0) {
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    return order;
  }
  if ((flags & kOrderRawFlag) != 0) {
    for (std::size_t i = 0; i < n; ++i) {
      order[i] = static_cast<std::size_t>(read_u64(data, len, pos));
    }
    return order;
  }
  const int width = ceil_log2(n);
  const std::size_t body_len = width > 0 ? (n * static_cast<std::size_t>(width) + 7) / 8 : 0;
  if (pos + body_len > len) throw std::runtime_error("REGER order body truncated");
  auto vals = unpack_values(data + pos, body_len, n, width);
  pos += body_len;
  for (std::size_t i = 0; i < n; ++i) order[i] = static_cast<std::size_t>(vals[i]);
  return order;
}

class CandidateTooLarge : public std::exception {};

std::vector<std::uint8_t> series_wire(const SeriesPayload& payload) {
  std::vector<std::uint8_t> out;
  out.reserve(kSeriesHeaderSize + payload.payload.size());
  write_series(out, payload);
  return out;
}

BlockPayload encode_block_candidate(
    const std::vector<std::vector<std::int64_t>>& block_columns,
    const std::vector<std::int64_t>& block_times,
    const std::vector<std::size_t>& order,
    int segment_size,
    bool include_time_stream,
    const std::size_t* cutoff,
    const std::vector<std::size_t>* column_priority) {
  BlockPayload out;
  std::vector<std::int64_t> ordered_times;
  if (!is_identity_order(order)) {
    out.flags |= kOrderPermutedFlag;
    const bool raw_order = operator_is_ablated(RegerAblation::kBitPacking);
    if (raw_order) out.flags |= kOrderRawFlag;
    auto order_payload = encode_order(order, raw_order);
    out.payload.insert(out.payload.end(), order_payload.begin(), order_payload.end());
  }
  if (include_time_stream) {
    out.flags |= kTimeStreamFlag;
    ordered_times.reserve(order.size());
    for (std::size_t idx : order) ordered_times.push_back(block_times[idx]);
    write_series(out.payload, best_series(ordered_times, segment_size));
    if (cutoff != nullptr && out.payload.size() >= *cutoff) throw CandidateTooLarge{};
  }

  const bool deferred = cutoff != nullptr && column_priority != nullptr;
  std::vector<std::vector<std::uint8_t>> wires(deferred ? block_columns.size() : 0);
  std::size_t deferred_bytes = 0;
  std::vector<std::size_t> natural_order;
  if (column_priority == nullptr) {
    natural_order.resize(block_columns.size());
    for (std::size_t i = 0; i < natural_order.size(); ++i) natural_order[i] = i;
    column_priority = &natural_order;
  }
  for (std::size_t column_index : *column_priority) {
    const auto& col = block_columns[column_index];
    std::vector<std::int64_t> values;
    values.reserve(order.size());
    for (std::size_t idx : order) values.push_back(col[idx]);
    auto wire = series_wire(best_series(values, segment_size,
                                        include_time_stream ? &ordered_times : nullptr));
    if (deferred) {
      deferred_bytes += wire.size();
      wires[column_index] = std::move(wire);
    } else {
      out.payload.insert(out.payload.end(), wire.begin(), wire.end());
    }
    if (cutoff != nullptr && out.payload.size() + deferred_bytes >= *cutoff) {
      throw CandidateTooLarge{};
    }
  }
  if (deferred) {
    for (const auto& wire : wires) {
      out.payload.insert(out.payload.end(), wire.begin(), wire.end());
    }
  }
  return out;
}

BlockPayload encoded_block_cost(
    const std::vector<std::vector<std::int64_t>>& block_columns,
    const std::vector<std::int64_t>& block_times,
    const std::vector<std::size_t>& order,
    int segment_size,
    const std::size_t* cutoff,
    const std::vector<std::size_t>* column_priority) {
  std::vector<BlockPayload> candidates;
  try {
    candidates.push_back(encode_block_candidate(block_columns, block_times, order, segment_size,
                                                false, cutoff, column_priority));
  } catch (const CandidateTooLarge&) {
  }
  try {
    candidates.push_back(encode_block_candidate(block_columns, block_times, order, segment_size,
                                                true, cutoff, column_priority));
  } catch (const CandidateTooLarge&) {
  } catch (const std::exception&) {
  }
  if (candidates.empty()) {
    if (cutoff != nullptr) throw CandidateTooLarge{};
    return encode_block_candidate(block_columns, block_times, order, segment_size, false, nullptr,
                                  nullptr);
  }
  return *std::min_element(candidates.begin(), candidates.end(),
                           [](const BlockPayload& a, const BlockPayload& b) {
                             return a.payload.size() < b.payload.size();
                           });
}

std::vector<std::size_t> column_cost_priority(const BlockPayload& payload, std::size_t row_count,
                                              std::size_t column_count) {
  std::size_t pos = 0;
  if ((payload.flags & kOrderPermutedFlag) != 0) {
    if ((payload.flags & kOrderRawFlag) != 0) {
      pos += row_count * sizeof(std::uint64_t);
    } else {
      const int width = ceil_log2(row_count);
      pos += width > 0 ? (row_count * static_cast<std::size_t>(width) + 7) / 8 : 0;
    }
  }
  auto skip_series = [&]() -> std::size_t {
    if (pos + kSeriesHeaderSize > payload.payload.size()) {
      throw std::runtime_error("REGER candidate series header truncated");
    }
    ++pos;
    std::size_t size_pos = pos;
    const std::uint32_t length = read_u32(payload.payload.data(), payload.payload.size(), size_pos);
    pos = size_pos + length;
    if (pos > payload.payload.size()) {
      throw std::runtime_error("REGER candidate series body truncated");
    }
    return kSeriesHeaderSize + length;
  };
  if ((payload.flags & kTimeStreamFlag) != 0) (void)skip_series();
  std::vector<std::pair<std::size_t, std::size_t>> costs;
  costs.reserve(column_count);
  for (std::size_t column = 0; column < column_count; ++column) {
    costs.push_back({skip_series(), column});
  }
  std::stable_sort(costs.begin(), costs.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });
  std::vector<std::size_t> priority;
  priority.reserve(costs.size());
  for (const auto& item : costs) priority.push_back(item.second);
  return priority;
}

std::vector<std::size_t> move_order(const std::vector<std::size_t>& order, std::size_t source,
                                    std::size_t destination) {
  auto out = order;
  const std::size_t value = out[source];
  out.erase(out.begin() + static_cast<std::ptrdiff_t>(source));
  if (destination > source) --destination;
  destination = std::min(destination, out.size());
  out.insert(out.begin() + static_cast<std::ptrdiff_t>(destination), value);
  return out;
}

std::vector<std::size_t> outlier_positions(
    const std::vector<std::size_t>& order,
    const std::vector<std::vector<std::int64_t>>& block_columns,
    const std::vector<std::int64_t>& block_times) {
  if (order.size() <= 2) {
    std::vector<std::size_t> all(order.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
    return all;
  }
  std::vector<std::pair<long double, std::size_t>> scores;
  scores.reserve(order.size() - 2);
  for (std::size_t pos = 1; pos + 1 < order.size(); ++pos) {
    const std::size_t previous = order[pos - 1];
    const std::size_t current = order[pos];
    const std::size_t next = order[pos + 1];
    long double score = std::abs(static_cast<long double>(block_times[previous]) -
                                 2.0L * static_cast<long double>(block_times[current]) +
                                 static_cast<long double>(block_times[next]));
    for (const auto& column : block_columns) {
      score += std::abs(static_cast<long double>(column[previous]) -
                        2.0L * static_cast<long double>(column[current]) +
                        static_cast<long double>(column[next]));
    }
    scores.push_back({score, pos});
  }
  std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second > b.second;
  });
  std::vector<std::size_t> positions = {0, order.size() - 1};
  for (std::size_t i = 0; i < std::min<std::size_t>(8, scores.size()); ++i) {
    positions.push_back(scores[i].second);
  }
  std::sort(positions.begin(), positions.end());
  positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
  return positions;
}

BlockPayload improve_order(
    std::vector<std::size_t> best_order,
    const std::vector<std::vector<std::int64_t>>& block_columns,
    const std::vector<std::int64_t>& block_times,
    int segment_size,
    BlockPayload best_payload,
    const std::vector<std::size_t>& column_priority) {
  const int iterations = std::min<int>(default_reorder_iterations(block_columns.size()),
                                       best_order.size() / 2);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    bool improved = false;
    auto trial_order = best_order;
    auto trial_payload = best_payload;
    std::vector<std::size_t> base_targets = {
        0, best_order.size() - 1, best_order.size() / 4, best_order.size() / 2,
        (best_order.size() * 3) / 4,
    };
    for (std::size_t source : outlier_positions(best_order, block_columns, block_times)) {
      auto targets = base_targets;
      targets.push_back(source > 0 ? source - 1 : 0);
      targets.push_back(std::min(best_order.size() - 1, source + 1));
      std::sort(targets.begin(), targets.end());
      targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
      for (std::size_t destination : targets) {
        if (source == destination) continue;
        const auto candidate_order = move_order(best_order, source, destination);
        const std::size_t cutoff_value = trial_payload.payload.size();
        const std::size_t* cutoff = candidate_pruning_enabled() ? &cutoff_value : nullptr;
        try {
          auto candidate = encoded_block_cost(block_columns, block_times, candidate_order,
                                              segment_size, cutoff, &column_priority);
          if (candidate.payload.size() < trial_payload.payload.size()) {
            trial_order = candidate_order;
            trial_payload = std::move(candidate);
            improved = true;
          }
        } catch (const CandidateTooLarge&) {
        }
      }
    }
    if (!improved) break;
    best_order = std::move(trial_order);
    best_payload = std::move(trial_payload);
  }
  return best_payload;
}

BlockPayload best_block_payload(const std::vector<std::vector<std::int64_t>>& block_columns,
                                const std::vector<std::int64_t>& block_times,
                                int segment_size) {
  const auto orders = row_order_candidates(block_columns, block_times);
  BlockPayload best;
  std::vector<std::size_t> best_order;
  std::vector<std::size_t> priority;
  std::size_t best_order_index = std::numeric_limits<std::size_t>::max();
  bool have_best = false;

  auto evaluate = [&](std::size_t order_index) {
    const auto& order = orders[order_index];
    const std::size_t cutoff_value =
        have_best ? best.payload.size() + (order_index < best_order_index ? 1 : 0) : 0;
    const std::size_t* cutoff =
        have_best && candidate_pruning_enabled() ? &cutoff_value : nullptr;
    try {
      auto candidate = encoded_block_cost(block_columns, block_times, order, segment_size, cutoff,
                                          priority.empty() ? nullptr : &priority);
      if (!have_best || candidate.payload.size() < best.payload.size() ||
          (candidate.payload.size() == best.payload.size() && order_index < best_order_index)) {
        best = std::move(candidate);
        best_order = order;
        best_order_index = order_index;
        have_best = true;
      }
      if (priority.empty()) {
        priority = column_cost_priority(best, order.size(), block_columns.size());
      }
    } catch (const CandidateTooLarge&) {
    } catch (const std::exception&) {
    }
  };

  if (!orders.empty()) evaluate(0);
  std::vector<std::size_t> evaluation_order;
  for (std::size_t index = 1; index < orders.size(); ++index) evaluation_order.push_back(index);
  if (candidate_pruning_enabled() && block_columns.size() >= 16 && !priority.empty()) {
    const std::size_t probe_count = std::min<std::size_t>(2, priority.size());
    std::vector<std::vector<std::int64_t>> probe_columns;
    probe_columns.reserve(probe_count);
    for (std::size_t i = 0; i < probe_count; ++i) {
      probe_columns.push_back(block_columns[priority[i]]);
    }
    std::vector<std::pair<std::size_t, std::size_t>> scored;
    scored.reserve(evaluation_order.size());
    for (std::size_t order_index : evaluation_order) {
      std::size_t cost = std::numeric_limits<std::size_t>::max();
      try {
        cost = encoded_block_cost(probe_columns, block_times, orders[order_index], segment_size,
                                  nullptr, nullptr)
                   .payload.size();
      } catch (const std::exception&) {
      }
      scored.push_back({cost, order_index});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });
    evaluation_order.clear();
    for (const auto& item : scored) evaluation_order.push_back(item.second);
    evaluation_order.resize(
        high_column_candidate_budget(block_columns.size(), evaluation_order.size()));
  }
  for (std::size_t order_index : evaluation_order) evaluate(order_index);

  if (!have_best) {
    best_order.resize(block_times.size());
    for (std::size_t i = 0; i < best_order.size(); ++i) best_order[i] = i;
    best = encoded_block_cost(block_columns, block_times, best_order, segment_size, nullptr, nullptr);
    priority = column_cost_priority(best, best_order.size(), block_columns.size());
  }
  if (operator_is_ablated(RegerAblation::kValueReorder)) return best;
  return improve_order(std::move(best_order), block_columns, block_times, segment_size,
                       std::move(best), priority);
}

std::vector<std::uint8_t> encode_int64_columns_impl(
    const std::vector<std::vector<std::int64_t>>& columns,
    const std::vector<std::int64_t>* times) {
  const int block_size = default_block_size();
  const int segment_size = default_segment_size();
  const std::size_t row_count = common_row_count(columns);
  const std::size_t col_count = columns.size();
  const std::uint32_t block_count =
      row_count == 0 ? 0 : static_cast<std::uint32_t>((row_count + block_size - 1) / block_size);
  const auto row_times = normalized_times(times, row_count);

  std::vector<std::uint8_t> out;
  out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
  put_u32(out, static_cast<std::uint32_t>(row_count));
  put_u16(out, static_cast<std::uint16_t>(col_count));
  put_u16(out, static_cast<std::uint16_t>(block_size));
  put_u16(out, static_cast<std::uint16_t>(segment_size));
  put_u32(out, block_count);
  for (std::size_t start = 0; start < row_count; start += static_cast<std::size_t>(block_size)) {
    const std::size_t end = std::min(row_count, start + static_cast<std::size_t>(block_size));
    std::vector<std::vector<std::int64_t>> block_columns;
    block_columns.reserve(col_count);
    for (const auto& col : columns) {
      block_columns.emplace_back(col.begin() + static_cast<std::ptrdiff_t>(start),
                                 col.begin() + static_cast<std::ptrdiff_t>(end));
    }
    std::vector<std::int64_t> block_times(row_times.begin() + static_cast<std::ptrdiff_t>(start),
                                          row_times.begin() + static_cast<std::ptrdiff_t>(end));
    const auto block = best_block_payload(block_columns, block_times, segment_size);
    put_u16(out, static_cast<std::uint16_t>(end - start));
    put_u8(out, block.flags);
    put_u32(out, static_cast<std::uint32_t>(block.payload.size()));
    out.insert(out.end(), block.payload.begin(), block.payload.end());
  }
  return out;
}

bool exact_decimal_column(const std::vector<double>& values, std::size_t row_count,
                          std::uint8_t& scale, std::vector<std::int64_t>& integers) {
  std::vector<std::int64_t> candidate(row_count);
  for (int decimal_scale = 0; decimal_scale <= kMaxDoubleDecimalScale; ++decimal_scale) {
    const double factor = std::pow(10.0, decimal_scale);
    bool exact = true;
    for (std::size_t i = 0; i < row_count; ++i) {
      const double value = values[i];
      if (!std::isfinite(value) || (value == 0.0 && std::signbit(value))) {
        exact = false;
        break;
      }
      const long double scaled =
          static_cast<long double>(value) * static_cast<long double>(factor);
      const long double rounded = std::round(scaled);
      if (!std::isfinite(rounded) ||
          rounded < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
          rounded > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        exact = false;
        break;
      }
      const auto integer = static_cast<std::int64_t>(rounded);
      const double recovered = static_cast<double>(integer) / factor;
      if (double_to_u64(recovered) != double_to_u64(value)) {
        exact = false;
        break;
      }
      candidate[i] = integer;
    }
    if (exact) {
      scale = static_cast<std::uint8_t>(decimal_scale);
      integers = std::move(candidate);
      return true;
    }
  }
  return false;
}

}  // namespace

void setAblatedOperatorForTesting(std::string_view omitted_operator) {
  std::string value(omitted_operator);
  value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                return c == '_' || c == '-' || std::isspace(c);
              }),
              value.end());
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (value.empty()) {
    g_ablation = RegerAblation::kNone;
  } else if (value == "valuereorder") {
    g_ablation = RegerAblation::kValueReorder;
  } else if (value == "regression") {
    g_ablation = RegerAblation::kRegression;
  } else if (value == "for") {
    g_ablation = RegerAblation::kFor;
  } else if (value == "bitpacking") {
    g_ablation = RegerAblation::kBitPacking;
  } else {
    throw std::invalid_argument("unsupported REGER ablation operator");
  }
}

std::vector<std::uint8_t> encodeInt64Column(const std::vector<std::int64_t>& values) {
  return encode_int64_columns_impl({values}, nullptr);
}

std::vector<std::int64_t> decodeInt64Column(const std::vector<std::uint8_t>& payload) {
  const auto cols = decodeInt64Columns(payload);
  return cols.empty() ? std::vector<std::int64_t>{} : cols.front();
}

std::vector<std::uint8_t> encodeDoubleColumn(const std::vector<double>& values) {
  return encodeDoubleColumns({values});
}

std::vector<double> decodeDoubleColumn(const std::vector<std::uint8_t>& payload) {
  const auto columns = decodeDoubleColumns(payload);
  return columns.empty() ? std::vector<double>{} : columns.front();
}

std::vector<std::uint8_t> encodeInt64Columns(
    const std::vector<std::vector<std::int64_t>>& columns) {
  return encode_int64_columns_impl(columns, nullptr);
}

std::vector<std::uint8_t> encodeInt64Columns(
    const std::vector<std::vector<std::int64_t>>& columns,
    const std::vector<std::int64_t>* times) {
  return encode_int64_columns_impl(columns, times);
}

std::vector<std::vector<std::int64_t>> decodeInt64Columns(
    const std::vector<std::uint8_t>& payload) {
  if (payload.size() < 20) throw std::runtime_error("REGER payload too short");
  std::size_t pos = 0;
  for (char expected : kMagic) {
    if (payload[pos++] != static_cast<std::uint8_t>(expected)) {
      throw std::runtime_error("invalid REGER payload");
    }
  }
  const std::uint32_t row_count = read_u32(payload.data(), payload.size(), pos);
  const std::uint16_t col_count = read_u16(payload.data(), payload.size(), pos);
  (void)read_u16(payload.data(), payload.size(), pos);
  const std::uint16_t segment_size = read_u16(payload.data(), payload.size(), pos);
  const std::uint32_t block_count = read_u32(payload.data(), payload.size(), pos);

  std::vector<std::vector<std::int64_t>> columns(col_count,
                                                 std::vector<std::int64_t>(row_count, 0));
  std::size_t block_start = 0;
  for (std::uint32_t block_i = 0; block_i < block_count; ++block_i) {
    const std::uint16_t n = read_u16(payload.data(), payload.size(), pos);
    const std::uint8_t flags = read_u8(payload.data(), payload.size(), pos);
    const std::uint32_t payload_len = read_u32(payload.data(), payload.size(), pos);
    if (pos + payload_len > payload.size()) throw std::runtime_error("REGER block truncated");
    const std::uint8_t* block_data = payload.data() + pos;
    std::size_t block_pos = 0;
    auto order = decode_order(block_data, payload_len, n, flags, block_pos);
    std::vector<std::int64_t> ordered_times;
    if ((flags & kTimeStreamFlag) != 0) {
      ordered_times = read_series(block_data, payload_len, n, segment_size, block_pos);
    }
    std::vector<std::vector<std::int64_t>> decoded_cols;
    decoded_cols.reserve(col_count);
    for (std::uint16_t col = 0; col < col_count; ++col) {
      decoded_cols.push_back(read_series(
          block_data, payload_len, n, segment_size, block_pos,
          (flags & kTimeStreamFlag) != 0 ? &ordered_times : nullptr));
    }
    pos += payload_len;
    for (std::uint16_t ordered = 0; ordered < n; ++ordered) {
      const std::size_t local = order[ordered];
      if (local >= n) throw std::runtime_error("REGER decoded row index out of range");
      const std::size_t row = block_start + local;
      for (std::uint16_t col = 0; col < col_count; ++col) {
        columns[col][row] = decoded_cols[col][ordered];
      }
    }
    block_start += n;
  }
  return columns;
}

std::vector<std::uint8_t> encodeDoubleColumns(const std::vector<std::vector<double>>& columns) {
  return encodeDoubleColumns(columns, nullptr);
}

std::vector<std::uint8_t> encodeDoubleColumns(const std::vector<std::vector<double>>& columns,
                                              const std::vector<std::int64_t>* times) {
  if (columns.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::invalid_argument("REGER double column count exceeds wire format");
  }
  std::size_t row_count = columns.empty() ? 0 : columns.front().size();
  for (const auto& column : columns) row_count = std::min(row_count, column.size());

  std::vector<std::uint8_t> modes;
  std::vector<std::uint8_t> scales;
  std::vector<std::vector<std::int64_t>> integers;
  modes.reserve(columns.size());
  scales.reserve(columns.size());
  integers.reserve(columns.size());
  for (const auto& column : columns) {
    std::uint8_t scale = 0;
    std::vector<std::int64_t> encoded_column;
    if (exact_decimal_column(column, row_count, scale, encoded_column)) {
      modes.push_back(kDoubleDecimal);
      scales.push_back(scale);
    } else {
      modes.push_back(kDoubleRawBits);
      scales.push_back(0);
      encoded_column.reserve(row_count);
      for (std::size_t i = 0; i < row_count; ++i) {
        encoded_column.push_back(to_i64(double_to_u64(column[i])));
      }
    }
    integers.push_back(std::move(encoded_column));
  }

  std::vector<std::uint8_t> out;
  out.insert(out.end(), std::begin(kDoubleMagic), std::end(kDoubleMagic));
  put_u16(out, static_cast<std::uint16_t>(columns.size()));
  for (std::size_t i = 0; i < columns.size(); ++i) {
    put_u8(out, modes[i]);
    put_u8(out, scales[i]);
  }
  auto body = encode_int64_columns_impl(integers, times);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::vector<double>> decodeDoubleColumns(const std::vector<std::uint8_t>& payload) {
  if (payload.size() < 8) throw std::runtime_error("REGER double payload too short");
  std::size_t pos = 0;
  for (char expected : kDoubleMagic) {
    if (payload[pos++] != static_cast<std::uint8_t>(expected)) {
      throw std::runtime_error("invalid REGER double payload");
    }
  }
  const std::uint16_t column_count = read_u16(payload.data(), payload.size(), pos);
  std::vector<std::uint8_t> modes(column_count);
  std::vector<std::uint8_t> scales(column_count);
  for (std::uint16_t i = 0; i < column_count; ++i) {
    modes[i] = read_u8(payload.data(), payload.size(), pos);
    scales[i] = read_u8(payload.data(), payload.size(), pos);
    if (modes[i] > kDoubleDecimal || scales[i] > kMaxDoubleDecimalScale) {
      throw std::runtime_error("invalid REGER double column metadata");
    }
  }
  std::vector<std::uint8_t> body(payload.begin() + static_cast<std::ptrdiff_t>(pos),
                                 payload.end());
  const auto integers = decodeInt64Columns(body);
  if (integers.size() != column_count) {
    throw std::runtime_error("REGER double column metadata mismatch");
  }
  std::vector<std::vector<double>> out;
  out.reserve(integers.size());
  for (std::size_t column_index = 0; column_index < integers.size(); ++column_index) {
    const auto& column = integers[column_index];
    std::vector<double> decoded;
    decoded.reserve(column.size());
    if (modes[column_index] == kDoubleDecimal) {
      const double factor = std::pow(10.0, scales[column_index]);
      for (std::int64_t value : column) decoded.push_back(static_cast<double>(value) / factor);
    } else {
      for (std::int64_t value : column) decoded.push_back(u64_to_double(to_u64(value)));
    }
    out.push_back(std::move(decoded));
  }
  return out;
}

void benchInt64ColumnsReger(const std::vector<std::vector<std::int64_t>>& columns,
                            std::vector<long long>& result,
                            LossAccum* loss) {
  benchInt64ColumnsReger(columns, result, loss, nullptr);
}

void benchInt64ColumnsReger(const std::vector<std::vector<std::int64_t>>& columns,
                            std::vector<long long>& result,
                            LossAccum* loss,
                            const std::vector<std::int64_t>* times) {
  const std::size_t row_count = common_row_count(columns);
  if (row_count == 0 || columns.empty()) {
    result.assign({0, 0, 0, 0});
    return;
  }
  std::vector<std::vector<std::int64_t>> prepared;
  prepared.reserve(columns.size());
  for (const auto& col : columns) {
    prepared.emplace_back(col.begin(), col.begin() + static_cast<std::ptrdiff_t>(row_count));
  }
  std::vector<std::int64_t> prepared_times;
  const std::vector<std::int64_t>* time_ptr = nullptr;
  if (times != nullptr) {
    prepared_times = normalized_times(times, row_count);
    time_ptr = &prepared_times;
  }

  LossAccum local_loss;
  LossAccum& loss_ref = loss != nullptr ? *loss : local_loss;
  const auto t0 = clock_ns::now();
  const auto encoded = encode_int64_columns_impl(prepared, time_ptr);
  const auto t1 = clock_ns::now();
  const auto decoded = decodeInt64Columns(encoded);
  const auto t2 = clock_ns::now();
  for (std::size_t i = 0; i < prepared.size(); ++i) {
    benchmark_loss::accumulate_int64_column(prepared[i], decoded[i], loss_ref);
  }
  result.assign({static_cast<long long>(row_count * prepared.size() * sizeof(std::int64_t)),
                 static_cast<long long>(encoded.size()),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()});
}

void benchDoubleColumnsReger(const std::vector<std::vector<double>>& columns,
                             std::vector<long long>& result,
                             LossAccum* loss) {
  benchDoubleColumnsReger(columns, result, loss, nullptr);
}

void benchDoubleColumnsReger(const std::vector<std::vector<double>>& columns,
                             std::vector<long long>& result,
                             LossAccum* loss,
                             const std::vector<std::int64_t>* times) {
  std::size_t row_count = 0;
  if (!columns.empty()) {
    row_count = columns.front().size();
    for (const auto& col : columns) row_count = std::min(row_count, col.size());
  }
  if (row_count == 0 || columns.empty()) {
    result.assign({0, 0, 0, 0});
    return;
  }
  std::vector<std::vector<double>> prepared;
  prepared.reserve(columns.size());
  for (const auto& col : columns) {
    prepared.emplace_back(col.begin(), col.begin() + static_cast<std::ptrdiff_t>(row_count));
  }
  std::vector<std::int64_t> prepared_times;
  const std::vector<std::int64_t>* time_ptr = nullptr;
  if (times != nullptr) {
    prepared_times = normalized_times(times, row_count);
    time_ptr = &prepared_times;
  }

  LossAccum local_loss;
  LossAccum& loss_ref = loss != nullptr ? *loss : local_loss;
  const auto t0 = clock_ns::now();
  const auto encoded = encodeDoubleColumns(prepared, time_ptr);
  const auto t1 = clock_ns::now();
  const auto decoded = decodeDoubleColumns(encoded);
  const auto t2 = clock_ns::now();
  for (std::size_t i = 0; i < prepared.size(); ++i) {
    benchmark_loss::accumulate_double_column(prepared[i], decoded[i], loss_ref);
  }
  result.assign({static_cast<long long>(row_count * prepared.size() * sizeof(double)),
                 static_cast<long long>(encoded.size()),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()});
}

}  // namespace reger_codec
