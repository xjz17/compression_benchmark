#include "python_wire_compat.hpp"
#include "minimum_for.hpp"
// Integer/block machinery adapted from reger_codec.cpp; Python REGER3 selection
// differs from the production C++ codec in regression arithmetic and order headers.

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
#include <cctype>

namespace reger_legacy_cpp {
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
  std::string value(raw);
  auto first=value.find_first_not_of(" \t\r\n\v\f"),last=value.find_last_not_of(" \t\r\n\v\f");
  value=first==std::string::npos?"":value.substr(first,last-first+1);
  std::transform(value.begin(),value.end(),value.begin(),[](unsigned char c){return std::tolower(c);});
  return value == "fast" ? RegerProfile::kFast
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

std::pair<float,float> fit_linear(const std::vector<long double>& xs,
                                   const std::vector<std::int64_t>& ys) {
  if(xs.size()!=ys.size())throw std::invalid_argument("REGER regression length mismatch");
  if(xs.empty())return {0.0F,0.0F};
  std::vector<double> vx,vy,vxx,vxy;
  for(std::size_t i=0;i<xs.size();++i) {
    double x=static_cast<double>(xs[i]),y=static_cast<double>(ys[i]);
    vx.push_back(x);vy.push_back(y);vxx.push_back(x*x);vxy.push_back(x*y);
  }
  double sx=python_wire::fsum(vx),sy=python_wire::fsum(vy);
  double sxx=python_wire::fsum(vxx),sxy=python_wire::fsum(vxy);
  double n=static_cast<double>(xs.size()),denominator=n*sxx-sx*sx;
  double theta0,theta1=0.0;
  if(std::abs(denominator)<=1e-12)theta0=sy/n;
  else {theta1=(n*sxy-sx*sy)/denominator;theta0=(sy-theta1*sx)/n;}
  float out0=static_cast<float>(theta0),out1=static_cast<float>(theta1);
  if(!std::isfinite(out0)||!std::isfinite(out1))throw std::overflow_error("REGER regression coefficients overflow");
  return {out0,out1};
}

std::int64_t predict(float theta0,float theta1,long double x) {
  double raw=static_cast<double>(theta0)+static_cast<double>(theta1)*static_cast<double>(x);
  if(!std::isfinite(raw))throw std::overflow_error("REGER prediction nonfinite");
  if(raw<=-9223372036854775808.0)return INT64_MIN;
  if(raw>=9223372036854775808.0)return INT64_MAX;
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
  const std::int64_t t0 = times.front();
  std::vector<long double> xs;
  xs.reserve(times.size());
  for (std::int64_t time : times) xs.push_back(static_cast<double>(static_cast<__int128>(time) - t0));
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
    for (std::uint64_t diff : diffs) {
      const __int128 value=minimum_for::restore_wide(minimum, diff);
      if(value<INT64_MIN || value>INT64_MAX)throw std::overflow_error("REGER decoded FOR value exceeds int64");
      out.push_back(static_cast<std::int64_t>(value));
    }
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
      const __int128 value=static_cast<__int128>(out.back())+minimum_for::restore_wide(minimum, diff);
      if(value<INT64_MIN || value>INT64_MAX)throw std::overflow_error("REGER decoded delta value exceeds int64");
      out.push_back(static_cast<std::int64_t>(value));
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
    const std::int64_t t0 = count > 0 ? (*times)[0] : 0;
    for (std::size_t i = 0; i < count; ++i) {
      const double x = static_cast<double>(static_cast<__int128>((*times)[i]) - t0);
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
  if (!include_time_stream && !is_identity_order(order)) {
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
  if (is_identity_order(order)) try {
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
  std::vector<std::pair<__int128, std::size_t>> scores;
  auto abs128=[](__int128 x){return x<0?-x:x;};
  scores.reserve(order.size() - 2);
  for (std::size_t pos = 1; pos + 1 < order.size(); ++pos) {
    const std::size_t previous = order[pos - 1];
    const std::size_t current = order[pos];
    const std::size_t next = order[pos + 1];
    __int128 score = abs128(static_cast<__int128>(block_times[previous]) -
                                 2 * static_cast<__int128>(block_times[current]) +
                                 static_cast<__int128>(block_times[next]));
    for (const auto& column : block_columns) {
      score += abs128(static_cast<__int128>(column[previous]) -
                        2 * static_cast<__int128>(column[current]) +
                        static_cast<__int128>(column[next]));
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
  const auto orders=row_order_candidates(block_columns,block_times);
  BlockPayload best;std::vector<std::size_t> best_order,priority;bool have_best=false;
  for(const auto& order:orders) {
    std::size_t cutoff_value=have_best?best.payload.size():0;
    try {
      auto candidate=encoded_block_cost(block_columns,block_times,order,segment_size,
                                       have_best?&cutoff_value:nullptr,priority.empty()?nullptr:&priority);
      if(priority.empty())priority=column_cost_priority(candidate,order.size(),block_columns.size());
      if(!have_best || candidate.payload.size()<best.payload.size()) {
        best=std::move(candidate);best_order=order;have_best=true;
      }
    } catch(const CandidateTooLarge&) {} catch(const std::exception&) {}
  }
  if(!have_best) {
    best_order.resize(block_times.size());for(std::size_t i=0;i<best_order.size();++i)best_order[i]=i;
    best=encoded_block_cost(block_columns,block_times,best_order,segment_size,nullptr,nullptr);
    priority=column_cost_priority(best,best_order.size(),block_columns.size());
  }
  return improve_order(std::move(best_order),block_columns,block_times,segment_size,std::move(best),priority);
}

}  // anonymous native block machinery namespace
using namespace python_wire;
struct Record {std::uint32_t count;std::uint8_t flags;Bytes body;};
struct Packet {std::uint64_t count;std::uint32_t block_size,segment_size;std::vector<Record> blocks;};
void validate(const Packet& p) {
  require(p.count<=UINT32_MAX && p.block_size && p.block_size<=UINT16_MAX &&
          p.segment_size && p.segment_size<=UINT16_MAX,"REGER3 parameters exceed wire bounds");
  require(p.blocks.size()==(p.count+p.block_size-1)/p.block_size,"REGER3 block count mismatch");
  auto remaining=p.count;
  for(const auto& record:p.blocks) {
    require(record.count==std::min(std::uint64_t(p.block_size),remaining) &&
            !(record.flags&~3) && record.body.size()<=UINT32_MAX,"invalid REGER3 block shape");
    remaining-=record.count;
  }
  require(!remaining,"REGER3 row count mismatch");
}
Bytes pack_packet(const Packet& p) {
  validate(p);Bytes out;magic(out,"RGP1");put(out,p.count,8);put(out,p.block_size,4);
  put(out,p.segment_size,4);put(out,p.blocks.size(),4);
  for(const auto& r:p.blocks){put(out,r.count,4);put(out,r.flags,1);put(out,r.body.size(),8);append(out,r.body);}
  return out;
}
Packet read_packet(const Bytes& raw,bool framed=false) {
  std::size_t pos;
  Packet p;
  std::uint64_t block_count;
  if(framed) {
    require(raw.size()>=20 && std::memcmp(raw.data(),"REGER3",6)==0,"invalid REGER3 header");pos=6;
    p.count=get(raw,pos,4);require(get(raw,pos,2)==1,"expected single-column REGER3");
    p.block_size=get(raw,pos,2);p.segment_size=get(raw,pos,2);block_count=get(raw,pos,4);
  } else {
    require(raw.size()>=24 && is_magic(raw,"RGP1"),"invalid RGP1 header");pos=4;
    p.count=get(raw,pos,8);p.block_size=get(raw,pos,4);p.segment_size=get(raw,pos,4);block_count=get(raw,pos,4);
  }
  require(block_count<=(raw.size()-pos)/(framed?7:13),"truncated REGER records");
  for(std::size_t i=0;i<block_count;++i) {
    Record record;record.count=get(raw,pos,framed?2:4);record.flags=get(raw,pos,1);auto size=get(raw,pos,framed?4:8);
    require(size<=raw.size()-pos,"truncated REGER block");
    record.body=slice(raw,pos,pos+size);pos+=size;p.blocks.push_back(std::move(record));
  }
  require(pos==raw.size(),"REGER trailing bytes");validate(p);return p;
}
Bytes core_forward(const Bytes& raw) {
  require(raw.size()%8==0,"REGER block core requires complete 64-bit lanes");
  auto block_size=default_block_size(),segment_size=default_segment_size();
  require(block_size>0 && block_size<=UINT16_MAX && segment_size>0 && segment_size<=UINT16_MAX,"REGER parameter bounds");
  Packet p{raw.size()/8,static_cast<std::uint32_t>(block_size),static_cast<std::uint32_t>(segment_size),{}};
  require(p.count<=UINT32_MAX,"REGER row count overflow");
  std::vector<std::int64_t> values;std::size_t pos=0;
  while(pos<raw.size())values.push_back(signed_word(get(raw,pos,8)));
  for(std::size_t start=0;start<values.size();start+=block_size) {
    std::size_t end=std::min(values.size(),start+block_size);
    std::vector<std::int64_t> block(values.begin()+start,values.begin()+end),times;
    for(std::size_t i=start;i<end;++i)times.push_back(i);
    auto best=best_block_payload({block},times,segment_size);
    p.blocks.push_back({static_cast<std::uint32_t>(end-start),best.flags,std::move(best.payload)});
  }
  return pack_packet(p);
}
Bytes core_inverse(const Bytes& raw) {
  auto p=read_packet(raw);Bytes out;
  for(const auto& r:p.blocks) {
    std::size_t pos=0;auto order=decode_order(r.body.data(),r.body.size(),r.count,r.flags,pos);
    std::vector<std::int64_t> times;
    if(r.flags&2)times=read_series(r.body.data(),r.body.size(),r.count,p.segment_size,pos);
    auto values=read_series(r.body.data(),r.body.size(),r.count,p.segment_size,pos,(r.flags&2)?&times:nullptr);
    require(pos==r.body.size() && values.size()==r.count,"REGER block body length mismatch");
    std::vector<std::int64_t> restored(r.count);std::vector<bool> seen(r.count,false);
    if((r.flags&2) && !(r.flags&1)) {
      std::vector<std::size_t> positions(r.count);for(std::size_t i=0;i<positions.size();++i)positions[i]=i;
      std::stable_sort(positions.begin(),positions.end(),[&](auto a,auto b){return times[a]<times[b];});
      for(std::size_t local=0;local<r.count;++local)restored[local]=values[positions[local]];
    } else {
      for(std::size_t ordered=0;ordered<r.count;++ordered) {
        auto local=order[ordered];require(local<r.count && !seen[local],"invalid REGER permutation");seen[local]=true;restored[local]=values[ordered];
      }
    }
    for(auto value:restored)put(out,static_cast<std::uint64_t>(value),8);
  }
  require(out.size()==p.count*8,"REGER decoded lane count mismatch");return out;
}
Bytes frame_forward(const Bytes& raw) {
  auto p=read_packet(raw);Bytes out={'R','E','G','E','R','3'};put(out,p.count,4);put(out,1,2);
  put(out,p.block_size,2);put(out,p.segment_size,2);put(out,p.blocks.size(),4);
  for(const auto& r:p.blocks){put(out,r.count,2);put(out,r.flags,1);put(out,r.body.size(),4);append(out,r.body);}
  return out;
}
Bytes frame_inverse(const Bytes& raw){return pack_packet(read_packet(raw,true));}
struct Columns {std::uint64_t rows;std::vector<std::vector<std::int64_t>> values;std::vector<std::int64_t> times;};
Columns read_columns(const Bytes& raw) {
  require(raw.size()>=17 && is_magic(raw,"RGMI"),"invalid RGMI bridge input");std::size_t pos=4;
  Columns columns;columns.rows=get(raw,pos,8);auto count=get(raw,pos,4),has_times=get(raw,pos,1);
  require(columns.rows<=UINT32_MAX && count<=UINT16_MAX && has_times<=1,"RGMI parameter bounds");
  require(columns.rows*(count+has_times)<=(raw.size()-pos)/8 && columns.rows*(count+has_times)*8==raw.size()-pos,"RGMI data length mismatch");
  if(has_times)for(std::size_t i=0;i<columns.rows;++i)columns.times.push_back(signed_word(get(raw,pos,8)));
  else for(std::size_t i=0;i<columns.rows;++i)columns.times.push_back(i);
  for(std::size_t c=0;c<count;++c){columns.values.emplace_back();for(std::size_t i=0;i<columns.rows;++i)columns.values.back().push_back(signed_word(get(raw,pos,8)));}
  return columns;
}
Bytes columns_forward(const Bytes& raw) {
  auto columns=read_columns(raw);auto bs=default_block_size(),ss=default_segment_size();
  require(bs>0 && bs<=UINT16_MAX && ss>0 && ss<=UINT16_MAX,"REGER parameters exceed wire bounds");
  std::size_t blocks=(columns.rows+bs-1)/bs;
  Bytes out={'R','E','G','E','R','3'};put(out,columns.rows,4);put(out,columns.values.size(),2);put(out,bs,2);put(out,ss,2);put(out,blocks,4);
  for(std::size_t start=0;start<columns.rows;start+=bs) {
    auto end=std::min<std::size_t>(columns.rows,start+bs);std::vector<std::vector<std::int64_t>> values;
    for(const auto& column:columns.values)values.emplace_back(column.begin()+start,column.begin()+end);
    std::vector<std::int64_t> times(columns.times.begin()+start,columns.times.begin()+end);
    auto block=best_block_payload(values,times,ss);put(out,end-start,2);put(out,block.flags,1);put(out,block.payload.size(),4);append(out,block.payload);
  }
  return out;
}
Bytes columns_inverse(const Bytes& raw) {
  require(raw.size()>=20 && std::memcmp(raw.data(),"REGER3",6)==0,"invalid REGER3 header");std::size_t pos=6;
  auto rows=get(raw,pos,4),count=get(raw,pos,2),bs=get(raw,pos,2),ss=get(raw,pos,2),blocks=get(raw,pos,4);
  require(bs && ss && blocks==(rows+bs-1)/bs && blocks<=(raw.size()-pos)/7,"invalid REGER3 shape");
  std::vector<std::vector<std::int64_t>> columns(count);std::size_t remaining=rows;
  for(std::size_t block=0;block<blocks;++block) {
    auto n=get(raw,pos,2),flags=get(raw,pos,1),size=get(raw,pos,4);
    require(n==std::min<std::size_t>(bs,remaining) && !(flags&~3) && size<=raw.size()-pos,"invalid REGER3 block");
    auto body=slice(raw,pos,pos+size);pos+=size;std::size_t p=0;
    auto order=decode_order(body.data(),body.size(),n,flags,p);std::vector<std::int64_t> times;
    if(flags&2)times=read_series(body.data(),body.size(),n,ss,p);
    if((flags&2) && !(flags&1)) {
      std::vector<std::size_t> sorted(n);for(std::size_t i=0;i<n;++i)sorted[i]=i;
      std::stable_sort(sorted.begin(),sorted.end(),[&](auto a,auto b){return times[a]<times[b];});
      for(std::size_t i=0;i<n;++i)order[sorted[i]]=i;
    }
    std::vector<bool> seen(n,false);for(auto local:order){require(local<n && !seen[local],"invalid REGER3 order");seen[local]=true;}
    for(auto& column:columns) {
      auto values=read_series(body.data(),body.size(),n,ss,p,(flags&2)?&times:nullptr);
      std::vector<std::int64_t> restored(n);for(std::size_t ordered=0;ordered<n;++ordered)restored[order[ordered]]=values[ordered];
      column.insert(column.end(),restored.begin(),restored.end());
    }
    require(p==body.size(),"REGER3 block trailing bytes");remaining-=n;
  }
  require(pos==raw.size() && !remaining,"REGER3 trailing bytes or row mismatch");
  Bytes out;magic(out,"RGMI");put(out,rows,8);put(out,count,4);put(out,0,1);
  for(const auto& column:columns)for(auto value:column)put(out,static_cast<std::uint64_t>(value),8);
  return out;
}
Bytes orders_forward(const Bytes& raw) {
  auto columns=read_columns(raw);auto orders=row_order_candidates(columns.values,columns.times);Bytes out;put(out,orders.size(),4);put(out,columns.rows,4);
  for(const auto& order:orders)for(auto value:order)put(out,value,4);return out;
}
Bytes run(const std::string& operation,const Bytes& raw,const std::string&) {
  if(operation=="core-forward")return core_forward(raw);if(operation=="core-inverse")return core_inverse(raw);
  if(operation=="frame-forward")return frame_forward(raw);if(operation=="frame-inverse")return frame_inverse(raw);
  if(operation=="columns-forward")return columns_forward(raw);if(operation=="columns-inverse")return columns_inverse(raw);
  if(operation=="orders-forward")return orders_forward(raw);
  throw std::runtime_error("unknown REGER legacy stage");
}
}
int main(int argc,char**argv){return python_wire::cli(argc,argv,reger_legacy_cpp::run);}
