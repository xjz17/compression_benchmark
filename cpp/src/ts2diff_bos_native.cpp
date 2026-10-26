#include "ts2diff_bos_native.hpp"

#include "benchmark_wire.hpp"
#include "leco_double_codec.hpp"
#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace ts2diff_bos_native {
namespace {

using clock_ns = std::chrono::high_resolution_clock;

struct PayloadRow {
  long long original = 0;
  std::vector<std::uint8_t> payload;
  long long enc_ns = 0;
  long long dec_ns = 0;
};

struct WrappedPayload {
  long long size = 0;
  long long enc_ns = 0;
  long long dec_ns = 0;
};

std::uint64_t to_u64(std::int64_t v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

std::int64_t to_i64(std::uint64_t v) {
  std::int64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

std::uint64_t zigzag_encode64(std::int64_t v) {
  return (to_u64(v) << 1) ^ (v < 0 ? std::numeric_limits<std::uint64_t>::max() : 0ULL);
}

std::int64_t zigzag_decode64(std::uint64_t v) {
  const std::uint64_t decoded = (v >> 1) ^ (0ULL - (v & 1ULL));
  return to_i64(decoded);
}

std::int64_t add_zigzag_delta(std::uint64_t &prev_bits, std::uint64_t z) {
  const std::uint64_t decoded = (z >> 1) ^ (0ULL - (z & 1ULL));
  prev_bits += decoded;
  return to_i64(prev_bits);
}

std::uint64_t zigzag_delta_bits(std::uint64_t z) {
  return (z >> 1) ^ (0ULL - (z & 1ULL));
}

std::uint8_t repeated_packed_byte(std::uint64_t z, int width) {
  std::uint8_t out = 0;
  const int values_per_byte = 8 / width;
  for (int i = 0; i < values_per_byte; ++i) {
    out |= static_cast<std::uint8_t>(z << (8 - width * (i + 1)));
  }
  return out;
}

std::uint8_t repeated_tail_byte(std::uint64_t z, int width, int values) {
  std::uint8_t out = 0;
  for (int i = 0; i < values; ++i) {
    out |= static_cast<std::uint8_t>(z << (8 - width * (i + 1)));
  }
  return out;
}

bool decode_constant_delta_if_present(std::int64_t *dst,
                                      int block_len,
                                      int width,
                                      const std::uint8_t *payload,
                                      std::size_t payload_len,
                                      std::int64_t first) {
  const int count = block_len - 1;
  if (count < 1024 || payload_len == 0) return false;

  std::uint64_t z = 0;
  if (width == 1 || width == 2 || width == 4) {
    const int values_per_byte = 8 / width;
    z = payload[0] >> (8 - width);
    const std::uint8_t repeated = repeated_packed_byte(z, width);
    const int full_bytes = count / values_per_byte;
    const int rem = count % values_per_byte;
    const std::size_t expected_len = static_cast<std::size_t>(full_bytes + (rem == 0 ? 0 : 1));
    if (payload_len != expected_len) return false;
    if (full_bytes > 0) {
      if (payload[0] != repeated || payload[full_bytes / 2] != repeated ||
          payload[full_bytes - 1] != repeated) {
        return false;
      }
    }
    if (rem != 0 && payload[full_bytes] != repeated_tail_byte(z, width, rem)) return false;
    for (int i = 0; i < full_bytes; ++i) {
      if (payload[i] != repeated) return false;
    }
  } else if (width == 8) {
    z = payload[0];
    if (payload_len != static_cast<std::size_t>(count)) return false;
    if (payload[0] != z || payload[count / 2] != z || payload[count - 1] != z) return false;
    for (int i = 0; i < count; ++i) {
      if (payload[i] != z) return false;
    }
  } else if (width == 16) {
    if (payload_len != static_cast<std::size_t>(count) * 2) return false;
    z = (static_cast<std::uint64_t>(payload[0]) << 8) | payload[1];
    const std::uint8_t hi = payload[0];
    const std::uint8_t lo = payload[1];
    const auto sample_matches = [&](int value_index) {
      const std::size_t p = static_cast<std::size_t>(value_index) * 2;
      return payload[p] == hi && payload[p + 1] == lo;
    };
    if (!sample_matches(0) || !sample_matches(count / 2) || !sample_matches(count - 1)) {
      return false;
    }
    for (int i = 1; i < count; ++i) {
      const std::size_t p = static_cast<std::size_t>(i) * 2;
      if (payload[p] != hi || payload[p + 1] != lo) return false;
    }
  } else {
    return false;
  }

  const std::int64_t delta = zigzag_decode64(z);
  const __int128 signed_last =
      static_cast<__int128>(first) + static_cast<__int128>(delta) * static_cast<__int128>(count);
  if (signed_last >= static_cast<__int128>(std::numeric_limits<std::int64_t>::min()) &&
      signed_last <= static_cast<__int128>(std::numeric_limits<std::int64_t>::max())) {
    for (int i = 1; i < block_len; ++i) {
      dst[i] = first + delta * static_cast<std::int64_t>(i);
    }
  } else {
    const std::uint64_t first_bits = static_cast<std::uint64_t>(first);
    const std::uint64_t delta_bits = zigzag_delta_bits(z);
    for (int i = 1; i < block_len; ++i) {
      dst[i] = to_i64(first_bits + delta_bits * static_cast<std::uint64_t>(i));
    }
  }
  return true;
}

int bit_width64(std::uint64_t v) {
  return v == 0 ? 1 : 64 - __builtin_clzll(v);
}

std::uint64_t low_bits_mask(int bits) {
  return bits >= 64 ? std::numeric_limits<std::uint64_t>::max() : ((1ULL << bits) - 1ULL);
}

std::uint64_t next_zigzag_delta(const std::vector<std::int64_t> &values,
                                std::size_t index,
                                std::int64_t &prev) {
  const std::int64_t cur = values[index];
  const std::int64_t delta = to_i64(to_u64(cur) - to_u64(prev));
  prev = cur;
  return zigzag_encode64(delta);
}

std::size_t write_packed_deltas(std::uint8_t *out,
                                const std::vector<std::int64_t> &values,
                                int width) {
  std::int64_t prev = values.front();
  std::size_t i = 1;
  std::size_t pos = 0;

  if (width == 1 || width == 2 || width == 4) {
    const int values_per_byte = 8 / width;
    const std::uint64_t mask = low_bits_mask(width);
    while (i < values.size()) {
      std::uint8_t packed = 0;
      for (int slot = 0; slot < values_per_byte && i < values.size(); ++slot, ++i) {
        const int shift = 8 - width * (slot + 1);
        packed |= static_cast<std::uint8_t>((next_zigzag_delta(values, i, prev) & mask)
                                            << shift);
      }
      out[pos++] = packed;
    }
    return pos;
  }

  if (width == 3) {
    while (i + 7 < values.size()) {
      std::uint64_t z[8];
      for (int j = 0; j < 8; ++j, ++i) z[j] = next_zigzag_delta(values, i, prev);
      out[pos++] = static_cast<std::uint8_t>((z[0] << 5) | (z[1] << 2) | (z[2] >> 1));
      out[pos++] = static_cast<std::uint8_t>((z[2] << 7) | (z[3] << 4) | (z[4] << 1) |
                                             (z[5] >> 2));
      out[pos++] = static_cast<std::uint8_t>((z[5] << 6) | (z[6] << 3) | z[7]);
    }
  } else if ((width & 7) == 0) {
    const int bytes_per_value = width / 8;
    if (bytes_per_value == 1) {
      for (; i < values.size(); ++i) {
        out[pos++] = static_cast<std::uint8_t>(next_zigzag_delta(values, i, prev));
      }
      return pos;
    }
    for (; i < values.size(); ++i) {
      const std::uint64_t z = next_zigzag_delta(values, i, prev);
      for (int byte = bytes_per_value - 1; byte >= 0; --byte) {
        out[pos++] = static_cast<std::uint8_t>((z >> (byte * 8)) & 0xffu);
      }
    }
    return pos;
  }

  std::uint64_t buffer = 0;
  int used = 0;
  const std::uint64_t mask = low_bits_mask(width);
  for (; i < values.size(); ++i) {
    buffer = (buffer << width) | (next_zigzag_delta(values, i, prev) & mask);
    used += width;
    while (used >= 8) {
      const int shift = used - 8;
      out[pos++] = static_cast<std::uint8_t>((buffer >> shift) & 0xffu);
      used -= 8;
      buffer = used == 0 ? 0 : (buffer & low_bits_mask(used));
    }
  }
  if (used > 0) {
    out[pos++] = static_cast<std::uint8_t>((buffer << (8 - used)) & 0xffu);
  }
  return pos;
}

void put_i32_be(std::vector<std::uint8_t> &out, std::int32_t v) {
  const auto u = static_cast<std::uint32_t>(v);
  out.push_back(static_cast<std::uint8_t>((u >> 24) & 0xffu));
  out.push_back(static_cast<std::uint8_t>((u >> 16) & 0xffu));
  out.push_back(static_cast<std::uint8_t>((u >> 8) & 0xffu));
  out.push_back(static_cast<std::uint8_t>(u & 0xffu));
}

void put_u64_be(std::vector<std::uint8_t> &out, std::uint64_t v) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xffu));
  }
}

std::int32_t read_i32_be(const std::uint8_t *data, std::size_t len, std::size_t &pos) {
  if (pos + 4 > len) throw std::runtime_error("BOS native truncated int32");
  std::uint32_t u = 0;
  for (int i = 0; i < 4; ++i) u = (u << 8) | data[pos + i];
  pos += 4;
  return static_cast<std::int32_t>(u);
}

std::uint64_t read_u64_be(const std::uint8_t *data, std::size_t len, std::size_t &pos) {
  if (pos + 8 > len) throw std::runtime_error("BOS native truncated int64");
  std::uint64_t u = 0;
  for (int i = 0; i < 8; ++i) u = (u << 8) | data[pos + i];
  pos += 8;
  return u;
}

class BitWriter {
 public:
  explicit BitWriter(std::size_t reserve_bytes = 0) {
    if (reserve_bytes > 0) out_.reserve(reserve_bytes);
  }

  void write(std::uint64_t value, int bits) {
    if (bits <= 0) return;
    value &= low_bits_mask(bits);
    buffer_ = (buffer_ << bits) | value;
    used_ += bits;
    while (used_ >= 8) {
      const int shift = used_ - 8;
      out_.push_back(static_cast<std::uint8_t>((buffer_ >> shift) & 0xffu));
      used_ -= 8;
      buffer_ = used_ == 0 ? 0 : (buffer_ & low_bits_mask(used_));
    }
  }

  std::vector<std::uint8_t> finish() {
    if (used_ > 0) {
      out_.push_back(static_cast<std::uint8_t>((buffer_ << (8 - used_)) & 0xffu));
      buffer_ = 0;
      used_ = 0;
    }
    return out_;
  }

 private:
  std::vector<std::uint8_t> out_;
  std::uint64_t buffer_ = 0;
  int used_ = 0;
};

class BitReader {
 public:
  BitReader(const std::uint8_t *data, std::size_t len) : data_(data), len_(len) {}

  std::uint64_t read(int bits) {
    if (bits <= 0) return 0;
    while (used_ < bits) {
      if (pos_ >= len_) throw std::runtime_error("BOS native bitread past end");
      buffer_ = (buffer_ << 8) | data_[pos_++];
      used_ += 8;
    }
    const int shift = used_ - bits;
    const std::uint64_t out = (buffer_ >> shift) & low_bits_mask(bits);
    used_ -= bits;
    buffer_ = used_ == 0 ? 0 : (buffer_ & low_bits_mask(used_));
    return out;
  }

 private:
  const std::uint8_t *data_;
  std::size_t len_;
  std::size_t pos_ = 0;
  std::uint64_t buffer_ = 0;
  int used_ = 0;
};

class UncheckedBitReader {
 public:
  UncheckedBitReader(const std::uint8_t *data, int width)
      : data_(data), width_(width), mask_(low_bits_mask(width)) {}

  std::uint64_t read() {
    while (used_ < width_) {
      buffer_ = (buffer_ << 8) | data_[pos_++];
      used_ += 8;
    }
    used_ -= width_;
    const std::uint64_t out = (buffer_ >> used_) & mask_;
    return out;
  }

 private:
  const std::uint8_t *data_;
  int width_;
  std::uint64_t mask_;
  std::size_t pos_ = 0;
  std::uint64_t buffer_ = 0;
  int used_ = 0;
};

template <int Width>
void decode_fixed_width_deltas(std::int64_t *dst,
                               int block_len,
                               const std::uint8_t *payload,
                               std::size_t payload_len,
                               std::uint64_t &prev_bits) {
  BitReader br(payload, payload_len);
  int i = 1;
  for (; i + 3 < block_len; i += 4) {
    dst[i] = add_zigzag_delta(prev_bits, br.read(Width));
    dst[i + 1] = add_zigzag_delta(prev_bits, br.read(Width));
    dst[i + 2] = add_zigzag_delta(prev_bits, br.read(Width));
    dst[i + 3] = add_zigzag_delta(prev_bits, br.read(Width));
  }
  for (; i < block_len; ++i) dst[i] = add_zigzag_delta(prev_bits, br.read(Width));
}

std::vector<std::string> split_operators(const char *env_name, const char *fallback) {
  const char *raw = std::getenv(env_name);
  std::string text = (raw != nullptr && raw[0] != '\0') ? raw : fallback;
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, ',')) {
    part.erase(part.begin(), std::find_if(part.begin(), part.end(), [](unsigned char c) {
                 return !std::isspace(c);
               }));
    part.erase(std::find_if(part.rbegin(), part.rend(), [](unsigned char c) {
                 return !std::isspace(c);
               }).base(),
               part.end());
    if (!part.empty()) out.push_back(part);
  }
  if (out.empty()) out.push_back("raw");
  return out;
}

std::vector<std::string> level_operators(int level) {
  const int normalized = std::max(0, std::min(5, level));
  const std::string env_name =
      "WEB_COMPRESSION_BOS_L" + std::to_string(normalized) + "_OPERATORS";
  const char *fallback = "raw";
  if (normalized == 4) {
    fallback = "raw,zstd1,zstd3";
  } else if (normalized >= 5) {
    fallback = "raw,zstd1,zstd3,zstd6,zstd9,zstd12";
  }
  return split_operators(env_name.c_str(), fallback);
}

std::string normalized_operator(std::string op) {
  op.erase(std::remove_if(op.begin(), op.end(), [](unsigned char c) {
             return c == '_' || c == '-' || std::isspace(c);
           }),
           op.end());
  std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return op;
}

std::optional<int> zstd_level_from_operator(const std::string &op) {
  const std::string t = normalized_operator(op);
  if (t == "raw") return std::nullopt;
  if (t.rfind("zstd", 0) != 0) throw std::runtime_error("unsupported BOS level operator");
  if (t.size() == 4) return 3;
  return std::max(1, std::stoi(t.substr(4)));
}

std::optional<WrappedPayload> wrap_payload(const PayloadRow &row, const std::string &op) {
  if (row.original <= 0 || row.payload.empty()) return std::nullopt;
  const auto zlevel = zstd_level_from_operator(op);
  if (!zlevel.has_value()) {
    return WrappedPayload{static_cast<long long>(row.payload.size()), row.enc_ns, row.dec_ns};
  }

  const auto t0 = clock_ns::now();
  const std::size_t bound = ZSTD_compressBound(row.payload.size());
  std::vector<std::uint8_t> comp(bound);
  const std::size_t csize =
      ZSTD_compress(comp.data(), comp.size(), row.payload.data(), row.payload.size(), *zlevel);
  const auto t1 = clock_ns::now();
  if (ZSTD_isError(csize)) return std::nullopt;
  comp.resize(csize);
  std::vector<std::uint8_t> dec(row.payload.size());
  const std::size_t dsize = ZSTD_decompress(dec.data(), dec.size(), comp.data(), comp.size());
  const auto t2 = clock_ns::now();
  if (ZSTD_isError(dsize) || dsize != row.payload.size() || dec != row.payload) {
    return std::nullopt;
  }
  return WrappedPayload{
      static_cast<long long>(comp.size() + 1),
      row.enc_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
      row.dec_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
  };
}

std::string selection_policy() {
  const char *raw = std::getenv("WEB_COMPRESSION_LEVEL_OPERATOR_POLICY");
  std::string value = raw == nullptr ? "" : raw;
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value == "size" ? "size" : "balanced";
}

double time_bytes_per_ns() {
  const char *raw = std::getenv("WEB_COMPRESSION_LEVEL_TIME_BYTES_PER_NS");
  if (raw == nullptr || raw[0] == '\0') return 0.05;
  try {
    return std::max(0.0, std::stod(raw));
  } catch (...) {
    return 0.05;
  }
}

long long level_time_adjustment(int level, long long original_size) {
  const int normalized = std::max(0, std::min(5, level));
  if (normalized == 0) return 0;
  return static_cast<long long>(normalized) *
         std::max<long long>(1000000LL, std::max<long long>(0LL, original_size) * 1000LL);
}

std::array<long long, 4> bench_payload_groups(
    const std::vector<std::vector<std::optional<PayloadRow>>> &groups,
    const std::vector<std::string> &operators,
    bool size_only,
    int level = -1,
    bool report_search_time = false) {
  long long total_o = 0;
  long long total_c = 0;
  long long total_te = 0;
  long long total_td = 0;
  const double penalty = size_only ? 0.0 : time_bytes_per_ns();
  const bool by_size = size_only || selection_policy() == "size";
  for (const auto &group : groups) {
    struct Best {
      double score;
      long long size;
      long long enc;
      long long total_time;
      WrappedPayload wrapped;
      PayloadRow row;
    };
    std::optional<Best> best;
    long long search_enc_ns = 0;
    long long search_dec_ns = 0;
    for (const auto &maybe_row : group) {
      if (!maybe_row.has_value()) continue;
      for (const auto &op : operators) {
        std::optional<WrappedPayload> wrapped;
        try {
          wrapped = wrap_payload(*maybe_row, op);
        } catch (...) {
          wrapped.reset();
        }
        if (!wrapped.has_value()) continue;
        search_enc_ns += wrapped->enc_ns;
        search_dec_ns += wrapped->dec_ns;
        const double score =
            by_size ? static_cast<double>(wrapped->size)
                    : static_cast<double>(wrapped->size) + penalty * wrapped->enc_ns;
        Best cand{score, wrapped->size, wrapped->enc_ns,
                  wrapped->enc_ns + wrapped->dec_ns, *wrapped, *maybe_row};
        if (!best.has_value() ||
            std::tie(cand.score, cand.size, cand.total_time, cand.enc) <
                std::tie(best->score, best->size, best->total_time, best->enc)) {
          best = std::move(cand);
        }
      }
    }
    if (!best.has_value()) continue;
    total_o += best->row.original;
    total_c += best->wrapped.size;
    const long long reported_enc = report_search_time ? search_enc_ns : best->wrapped.enc_ns;
    const long long reported_dec = report_search_time ? search_dec_ns : best->wrapped.dec_ns;
    total_te += reported_enc + (level >= 0 ? level_time_adjustment(level, best->row.original) : 0);
    total_td += reported_dec;
  }
  return {total_o, total_c, total_te, total_td};
}

std::vector<std::uint8_t> encode_raw_int64_payload(const std::vector<std::int64_t> &values) {
  std::vector<std::uint8_t> out;
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  for (std::int64_t v : values) put_u64_be(out, to_u64(v));
  return out;
}

std::vector<std::int64_t> decode_raw_int64_payload(const std::uint8_t *data,
                                                   std::size_t len) {
  std::size_t pos = 0;
  const int n = read_i32_be(data, len, pos);
  if (n < 0) throw std::runtime_error("BOS raw invalid count");
  std::vector<std::int64_t> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) out.push_back(to_i64(read_u64_be(data, len, pos)));
  if (pos != len) throw std::runtime_error("BOS raw trailing bytes");
  return out;
}

std::vector<std::uint8_t> encode_rle_int64_payload(const std::vector<std::int64_t> &values) {
  std::vector<std::uint8_t> out;
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  std::vector<std::pair<std::int64_t, int>> runs;
  for (std::int64_t v : values) {
    if (!runs.empty() && runs.back().first == v &&
        runs.back().second < std::numeric_limits<std::int32_t>::max()) {
      ++runs.back().second;
    } else {
      runs.push_back({v, 1});
    }
  }
  put_i32_be(out, static_cast<std::int32_t>(runs.size()));
  for (const auto &[value, run_len] : runs) {
    put_u64_be(out, to_u64(value));
    put_i32_be(out, static_cast<std::int32_t>(run_len));
  }
  return out;
}

std::vector<std::int64_t> decode_rle_int64_payload(const std::uint8_t *data,
                                                   std::size_t len) {
  std::size_t pos = 0;
  const int n = read_i32_be(data, len, pos);
  const int runs = read_i32_be(data, len, pos);
  if (n < 0 || runs < 0) throw std::runtime_error("BOS RLE invalid header");
  std::vector<std::int64_t> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < runs; ++i) {
    const std::int64_t value = to_i64(read_u64_be(data, len, pos));
    const int run_len = read_i32_be(data, len, pos);
    if (run_len < 0) throw std::runtime_error("BOS RLE invalid run");
    out.insert(out.end(), static_cast<std::size_t>(run_len), value);
  }
  if (out.size() != static_cast<std::size_t>(n) || pos != len) {
    throw std::runtime_error("BOS RLE invalid payload");
  }
  return out;
}

std::vector<std::uint8_t> encode_dod_int64_payload(const std::vector<std::int64_t> &values) {
  std::vector<std::uint8_t> out;
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  if (values.empty()) return out;
  put_u64_be(out, to_u64(values[0]));
  if (values.size() == 1) return out;
  put_u64_be(out, to_u64(values[1]));
  std::vector<std::uint64_t> zz;
  zz.reserve(values.size() - 2);
  std::int64_t prev_delta = to_i64(to_u64(values[1]) - to_u64(values[0]));
  int width = 1;
  for (std::size_t i = 2; i < values.size(); ++i) {
    const std::int64_t delta = to_i64(to_u64(values[i]) - to_u64(values[i - 1]));
    const std::int64_t dod = to_i64(to_u64(delta) - to_u64(prev_delta));
    const std::uint64_t z = zigzag_encode64(dod);
    zz.push_back(z);
    width = std::max(width, bit_width64(z));
    prev_delta = delta;
  }
  const int packed_width = width > 56 ? 64 : width;
  out.push_back(static_cast<std::uint8_t>(packed_width));
  std::vector<std::uint8_t> body;
  if (packed_width >= 64) {
    for (std::uint64_t z : zz) put_u64_be(body, z);
  } else {
    BitWriter bw;
    for (std::uint64_t z : zz) bw.write(z, packed_width);
    body = bw.finish();
  }
  put_i32_be(out, static_cast<std::int32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::int64_t> decode_dod_int64_payload(const std::uint8_t *data,
                                                   std::size_t len) {
  std::size_t pos = 0;
  const int n = read_i32_be(data, len, pos);
  if (n < 0) throw std::runtime_error("BOS DoD invalid count");
  std::vector<std::int64_t> out;
  out.reserve(static_cast<std::size_t>(n));
  if (n == 0) return out;
  out.push_back(to_i64(read_u64_be(data, len, pos)));
  if (n == 1) {
    if (pos != len) throw std::runtime_error("BOS DoD trailing bytes");
    return out;
  }
  out.push_back(to_i64(read_u64_be(data, len, pos)));
  if (pos >= len) throw std::runtime_error("BOS DoD missing width");
  const int width = data[pos++];
  const int payload_len = read_i32_be(data, len, pos);
  if (width <= 0 || width > 64 || payload_len < 0 ||
      pos + static_cast<std::size_t>(payload_len) > len) {
    throw std::runtime_error("BOS DoD invalid payload");
  }
  const std::uint8_t *payload = data + pos;
  pos += static_cast<std::size_t>(payload_len);
  std::int64_t prev_delta = to_i64(to_u64(out[1]) - to_u64(out[0]));
  if (width >= 64) {
    std::size_t p = 0;
    for (int i = 2; i < n; ++i) {
      std::size_t local = p;
      const std::uint64_t z = read_u64_be(payload, payload_len, local);
      p += 8;
      const std::int64_t delta = to_i64(to_u64(prev_delta) + to_u64(zigzag_decode64(z)));
      out.push_back(to_i64(to_u64(out.back()) + to_u64(delta)));
      prev_delta = delta;
    }
  } else {
    BitReader br(payload, static_cast<std::size_t>(payload_len));
    for (int i = 2; i < n; ++i) {
      const std::uint64_t z = br.read(width);
      const std::int64_t delta = to_i64(to_u64(prev_delta) + to_u64(zigzag_decode64(z)));
      out.push_back(to_i64(to_u64(out.back()) + to_u64(delta)));
      prev_delta = delta;
    }
  }
  if (pos != len) throw std::runtime_error("BOS DoD trailing bytes");
  return out;
}

std::optional<PayloadRow> raw_int_row(const std::vector<std::int64_t> &col) {
  if (col.empty()) return std::nullopt;
  try {
    const auto t0 = clock_ns::now();
    auto enc = encode_raw_int64_payload(col);
    const auto t1 = clock_ns::now();
    auto dec = decode_raw_int64_payload(enc.data(), enc.size());
    const auto t2 = clock_ns::now();
    if (dec != col) return std::nullopt;
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), false)),
        std::move(enc),
        0,
        0,
    };
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<PayloadRow> rle_int_row(const std::vector<std::int64_t> &col) {
  if (col.empty()) return std::nullopt;
  try {
    const auto t0 = clock_ns::now();
    auto enc = encode_rle_int64_payload(col);
    const auto t1 = clock_ns::now();
    auto dec = decode_rle_int64_payload(enc.data(), enc.size());
    const auto t2 = clock_ns::now();
    if (dec != col) return std::nullopt;
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), false)),
        std::move(enc),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
    };
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<PayloadRow> dod_int_row(const std::vector<std::int64_t> &col) {
  if (col.empty()) return std::nullopt;
  try {
    const auto t0 = clock_ns::now();
    auto enc = encode_dod_int64_payload(col);
    const auto t1 = clock_ns::now();
    auto dec = decode_dod_int64_payload(enc.data(), enc.size());
    const auto t2 = clock_ns::now();
    if (dec != col) return std::nullopt;
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), false)),
        std::move(enc),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
    };
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<PayloadRow> bos_int_row(const std::vector<std::int64_t> &col) {
  if (col.empty()) return std::nullopt;
  try {
    const auto t0 = clock_ns::now();
    auto enc = encode_int64_column_payload(col);
    const auto t1 = clock_ns::now();
    auto dec = decode_int64_column_payload(enc.data(), enc.size());
    const auto t2 = clock_ns::now();
    if (dec != col) return std::nullopt;
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), false)),
        std::move(enc),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
    };
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<PayloadRow> ts2_int_row(const std::vector<std::int64_t> &col) {
  if (col.empty()) return std::nullopt;
  try {
    const auto t0 = clock_ns::now();
    auto enc = tsfile::p02::encode_ts_2diff_int64(
        std::span<const std::int64_t>(col.data(), col.size()));
    const auto t1 = clock_ns::now();
    std::vector<std::int64_t> dec;
    if (!tsfile::p02::decode_ts_2diff_int64(enc, dec) || dec != col) return std::nullopt;
    const auto t2 = clock_ns::now();
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), false)),
        std::move(enc),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
    };
  } catch (...) {
    return std::nullopt;
  }
}

int clamp_ts2diff_max_point_for_double(const std::vector<double> &col, int requested_max_point) {
  int normalized = std::max(0, requested_max_point);
  double max_abs = 0.0;
  for (double v : col) {
    if (std::isfinite(v)) max_abs = std::max(max_abs, std::abs(v));
  }
  if (!(max_abs > 0.0)) return normalized;
  const double ratio = static_cast<double>(std::numeric_limits<std::int64_t>::max()) / max_abs;
  if (!std::isfinite(ratio) || ratio <= 0.0) return 0;
  int safe_upper = static_cast<int>(std::floor(std::log10(ratio)));
  safe_upper = std::max(0, safe_upper);
  return std::min(normalized, safe_upper);
}

std::optional<PayloadRow> bos_double_row(const std::vector<double> &col,
                                         const int *max_point) {
  if (col.empty()) return std::nullopt;
  int scale = -1;
  std::vector<std::int64_t> ints;
  if (max_point != nullptr && *max_point >= 0) {
    if (!leco_double::quantizeScaledIntsWithMaxPointFallback(col, *max_point, ints, &scale)) {
      return std::nullopt;
    }
  } else {
    scale = leco_double::maxLosslessDecimalScale(col);
    if (scale < 0 || !leco_double::quantizeScaledInts(col, scale, ints)) return std::nullopt;
  }
  auto row = bos_int_row(ints);
  if (!row.has_value()) return std::nullopt;
  row->original = static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
  return row;
}

std::optional<PayloadRow> ts2_double_row(const std::vector<double> &col, const int *max_point) {
  if (col.empty()) return std::nullopt;
  try {
    const int requested = max_point == nullptr ? 0 : *max_point;
    const int effective = clamp_ts2diff_max_point_for_double(col, requested);
    const auto t0 = clock_ns::now();
    auto enc =
        tsfile::p02::encode_ts_2diff_double(std::span<const double>(col.data(), col.size()),
                                            effective);
    const auto t1 = clock_ns::now();
    std::vector<double> dec;
    if (!tsfile::p02::decode_ts_2diff_double(enc, dec) || dec.size() != col.size()) {
      return std::nullopt;
    }
    const auto t2 = clock_ns::now();
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), true)),
        std::move(enc),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
    };
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<PayloadRow> raw_double_row(const std::vector<double> &col) {
  if (col.empty()) return std::nullopt;
  try {
    const auto t0 = clock_ns::now();
    std::vector<std::uint8_t> enc;
    put_i32_be(enc, static_cast<std::int32_t>(col.size()));
    for (double v : col) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &v, sizeof(bits));
      put_u64_be(enc, bits);
    }
    const auto t1 = clock_ns::now();
    std::size_t pos = 0;
    const int n = read_i32_be(enc.data(), enc.size(), pos);
    std::vector<double> dec;
    dec.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const std::uint64_t bits = read_u64_be(enc.data(), enc.size(), pos);
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      dec.push_back(value);
    }
    const auto t2 = clock_ns::now();
    if (dec != col || pos != enc.size()) return std::nullopt;
    return PayloadRow{
        static_cast<long long>(benchmark_wire::column_original_size(col.size(), true)),
        std::move(enc),
        0,
        0,
    };
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<PayloadRow> scaled_double_row(
    const std::vector<double> &col,
    const int *max_point,
    std::optional<PayloadRow> (*builder)(const std::vector<std::int64_t> &)) {
  if (col.empty()) return std::nullopt;
  int scale = -1;
  std::vector<std::int64_t> ints;
  if (max_point != nullptr && *max_point >= 0) {
    if (!leco_double::quantizeScaledIntsWithMaxPointFallback(col, *max_point, ints, &scale)) {
      return std::nullopt;
    }
  } else {
    scale = leco_double::maxLosslessDecimalScale(col);
    if (scale < 0 || !leco_double::quantizeScaledInts(col, scale, ints)) return std::nullopt;
  }
  auto row = builder(ints);
  if (!row.has_value()) return std::nullopt;
  row->original = static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
  return row;
}

std::vector<std::optional<PayloadRow>> int_level_rows(const std::vector<std::int64_t> &col,
                                                      int level) {
  const int normalized = std::max(0, std::min(5, level));
  std::vector<std::optional<PayloadRow>> rows;
  rows.push_back(raw_int_row(col));
  if (normalized >= 1) rows.push_back(ts2_int_row(col));
  if (normalized >= 2) rows.push_back(bos_int_row(col));
  if (normalized >= 3) rows.push_back(dod_int_row(col));
  if (normalized >= 4) rows.push_back(rle_int_row(col));
  return rows;
}

std::vector<std::optional<PayloadRow>> double_level_rows(const std::vector<double> &col,
                                                         const int *max_point,
                                                         int level) {
  const int normalized = std::max(0, std::min(5, level));
  std::vector<std::optional<PayloadRow>> rows;
  rows.push_back(raw_double_row(col));
  if (normalized >= 1) rows.push_back(ts2_double_row(col, max_point));
  if (normalized >= 2) rows.push_back(scaled_double_row(col, max_point, bos_int_row));
  if (normalized >= 3) rows.push_back(scaled_double_row(col, max_point, dod_int_row));
  if (normalized >= 4) rows.push_back(scaled_double_row(col, max_point, rle_int_row));
  return rows;
}

void assign_result(const std::array<long long, 4> &row, std::vector<long long> &result) {
  result.assign({row[0], row[1], row[2], row[3]});
}

} // namespace

std::vector<std::uint8_t> encode_int64_column_payload(const std::vector<std::int64_t> &values) {
  return encode_int64_column_payload_ablation(values, "");
}

std::vector<std::uint8_t> encode_int64_column_payload_ablation(
    const std::vector<std::int64_t> &values, std::string_view omitted_operator) {
  if (values.empty()) return {};
  if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("BOS native column exceeds int32 row count");
  }

  const bool omit_delta = omitted_operator == "delta";
  const bool omit_bitpacking = omitted_operator == "bit-packing";
  int width = 1;
  std::int64_t reference = values.front();
  if (omit_delta) {
    reference = *std::min_element(values.begin(), values.end());
    for (std::int64_t value : values) {
      width = std::max(width, bit_width64(to_u64(value) - to_u64(reference)));
    }
  } else {
    std::int64_t prev = values.front();
    for (std::size_t i = 1; i < values.size(); ++i) {
      const std::int64_t cur = values[i];
      const std::int64_t delta = to_i64(to_u64(cur) - to_u64(prev));
      width = std::max(width, bit_width64(zigzag_encode64(delta)));
      prev = cur;
    }
  }
  const int packed_width = omit_bitpacking || width > 56 ? 64 : width;
  const std::size_t value_count = omit_delta ? values.size() : values.size() - 1;
  const std::size_t payload_len = packed_width >= 64
                                      ? value_count * 8
                                      : (value_count * static_cast<std::size_t>(packed_width) + 7) / 8;
  if (payload_len > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("BOS native payload exceeds int32 length");
  }

  std::vector<std::uint8_t> out;
  out.reserve(25 + payload_len);
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  put_u64_be(out, to_u64(reference));
  out.push_back(static_cast<std::uint8_t>(packed_width | (omit_delta ? 0x80 : 0)));
  put_i32_be(out, static_cast<std::int32_t>(payload_len));
  out.resize(25 + payload_len);
  std::size_t written = 0;
  if (omit_delta) {
    std::vector<std::uint8_t> body;
    if (packed_width >= 64) {
      body.reserve(values.size() * sizeof(std::uint64_t));
      for (std::int64_t value : values) {
        put_u64_be(body, to_u64(value) - to_u64(reference));
      }
    } else {
      BitWriter writer;
      for (std::int64_t value : values) {
        writer.write(to_u64(value) - to_u64(reference), packed_width);
      }
      body = writer.finish();
    }
    std::copy(body.begin(), body.end(), out.begin() + 25);
    written = body.size();
  } else {
    written = write_packed_deltas(out.data() + 25, values, packed_width);
  }
  if (written != payload_len) {
    throw std::runtime_error("BOS native internal payload length mismatch");
  }
  return out;
}

std::vector<std::int64_t> decode_int64_column_payload(const std::uint8_t *data, std::size_t len) {
  if (data == nullptr || len == 0) return {};
  std::size_t pos = 0;
  const int n = read_i32_be(data, len, pos);
  const int segment_rows = read_i32_be(data, len, pos);
  if (n < 0 || segment_rows <= 0) throw std::runtime_error("BOS native invalid header");
  std::vector<std::int64_t> out(static_cast<std::size_t>(n));
  std::size_t out_pos = 0;
  while (out_pos < static_cast<std::size_t>(n)) {
    const int block_len = read_i32_be(data, len, pos);
    if (block_len <= 0 || out_pos + static_cast<std::size_t>(block_len) >
                              static_cast<std::size_t>(n)) {
      throw std::runtime_error("BOS native invalid segment length");
    }
    const std::int64_t first = to_i64(read_u64_be(data, len, pos));
    if (pos >= len) throw std::runtime_error("BOS native truncated width");
    const int encoded_width = data[pos++];
    const bool direct_for = (encoded_width & 0x80) != 0;
    const int width = encoded_width & 0x7f;
    const int payload_len = read_i32_be(data, len, pos);
    if (payload_len < 0 || pos + static_cast<std::size_t>(payload_len) > len) {
      throw std::runtime_error("BOS native invalid payload length");
    }
    const std::uint8_t *payload = data + pos;
    pos += static_cast<std::size_t>(payload_len);
    std::int64_t *dst = out.data() + out_pos;
    if (direct_for) {
      if (width <= 0 || width > 64) throw std::runtime_error("BOS native invalid FOR width");
      const std::size_t required_bytes = width >= 64
                                             ? static_cast<std::size_t>(block_len) * 8
                                             : (static_cast<std::size_t>(block_len) * width + 7) / 8;
      if (static_cast<std::size_t>(payload_len) < required_bytes) {
        throw std::runtime_error("BOS native truncated FOR payload");
      }
      if (width >= 64) {
        std::size_t p = 0;
        for (int i = 0; i < block_len; ++i) {
          dst[i] = to_i64(to_u64(first) + read_u64_be(payload, payload_len, p));
        }
      } else {
        BitReader reader(payload, static_cast<std::size_t>(payload_len));
        for (int i = 0; i < block_len; ++i) {
          dst[i] = to_i64(to_u64(first) + reader.read(width));
        }
      }
    } else {
      dst[0] = first;
    }
    if (!direct_for && block_len > 1) {
      if (width <= 0 || width > 64) throw std::runtime_error("BOS native invalid width");
      std::uint64_t prev_bits = static_cast<std::uint64_t>(first);
      if (decode_constant_delta_if_present(dst, block_len, width, payload,
                                           static_cast<std::size_t>(payload_len), first)) {
        // handled by constant-delta fast path
      } else if (width >= 64) {
        std::size_t p = 0;
        for (int i = 1; i < block_len; ++i) {
          if (p + 8 > static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 64-bit payload");
          }
          const std::uint64_t z = (static_cast<std::uint64_t>(payload[p]) << 56) |
                                  (static_cast<std::uint64_t>(payload[p + 1]) << 48) |
                                  (static_cast<std::uint64_t>(payload[p + 2]) << 40) |
                                  (static_cast<std::uint64_t>(payload[p + 3]) << 32) |
                                  (static_cast<std::uint64_t>(payload[p + 4]) << 24) |
                                  (static_cast<std::uint64_t>(payload[p + 5]) << 16) |
                                  (static_cast<std::uint64_t>(payload[p + 6]) << 8) |
                                  static_cast<std::uint64_t>(payload[p + 7]);
          p += 8;
          dst[i] = add_zigzag_delta(prev_bits, z);
        }
      } else if (width == 1) {
        std::size_t p = 0;
        int i = 1;
        while (i + 7 < block_len) {
          if (p >= static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 1-bit payload");
          }
          const std::uint8_t b = payload[p++];
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 7) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 6) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 5) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 4) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 3) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 2) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 1) & 1u);
          dst[i++] = add_zigzag_delta(prev_bits, b & 1u);
        }
        if (i < block_len) {
          if (p >= static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 1-bit payload");
          }
          const std::uint8_t b = payload[p++];
          for (int shift = 7; shift >= 0 && i < block_len; --shift, ++i) {
            dst[i] = add_zigzag_delta(prev_bits, (b >> shift) & 1u);
          }
        }
      } else if (width == 2) {
        std::size_t p = 0;
        int i = 1;
        while (i + 3 < block_len) {
          if (p >= static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 2-bit payload");
          }
          const std::uint8_t b = payload[p++];
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 6) & 3u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 4) & 3u);
          dst[i++] = add_zigzag_delta(prev_bits, (b >> 2) & 3u);
          dst[i++] = add_zigzag_delta(prev_bits, b & 3u);
        }
        if (i < block_len) {
          if (p >= static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 2-bit payload");
          }
          const std::uint8_t b = payload[p++];
          for (int shift = 6; shift >= 0 && i < block_len; shift -= 2, ++i) {
            dst[i] = add_zigzag_delta(prev_bits, (b >> shift) & 3u);
          }
        }
      } else if (width == 3) {
        std::size_t p = 0;
        int i = 1;
        const int full_end = 1 + ((block_len - 1) / 8) * 8;
        while (i < full_end) {
          if (p + 3 > static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 3-bit payload");
          }
          const std::uint8_t b0 = payload[p];
          const std::uint8_t b1 = payload[p + 1];
          const std::uint8_t b2 = payload[p + 2];
          p += 3;
          dst[i++] = add_zigzag_delta(prev_bits, (b0 >> 5) & 7u);
          dst[i++] = add_zigzag_delta(prev_bits, (b0 >> 2) & 7u);
          dst[i++] = add_zigzag_delta(prev_bits, ((b0 & 3u) << 1) | (b1 >> 7));
          dst[i++] = add_zigzag_delta(prev_bits, (b1 >> 4) & 7u);
          dst[i++] = add_zigzag_delta(prev_bits, (b1 >> 1) & 7u);
          dst[i++] = add_zigzag_delta(prev_bits, ((b1 & 1u) << 2) | (b2 >> 6));
          dst[i++] = add_zigzag_delta(prev_bits, (b2 >> 3) & 7u);
          dst[i++] = add_zigzag_delta(prev_bits, b2 & 7u);
        }
        std::uint64_t buffer = 0;
        int used = 0;
        while (i < block_len) {
          while (used < 3) {
            if (p >= static_cast<std::size_t>(payload_len)) {
              throw std::runtime_error("BOS native truncated 3-bit payload");
            }
            buffer = (buffer << 8) | payload[p++];
            used += 8;
          }
          const std::uint64_t z = (buffer >> (used - 3)) & 7u;
          used -= 3;
          buffer = used == 0 ? 0 : (buffer & low_bits_mask(used));
          dst[i++] = add_zigzag_delta(prev_bits, z);
        }
      } else if (width == 4) {
        std::size_t p = 0;
        int i = 1;
        while (i < block_len) {
          if (p >= static_cast<std::size_t>(payload_len)) {
            throw std::runtime_error("BOS native truncated 4-bit payload");
          }
          const std::uint8_t b = payload[p++];
          dst[i++] = add_zigzag_delta(prev_bits, b >> 4);
          if (i < block_len) {
            dst[i++] = add_zigzag_delta(prev_bits, b & 15u);
          }
        }
      } else if (width == 8) {
        if (static_cast<std::size_t>(payload_len) < static_cast<std::size_t>(block_len - 1)) {
          throw std::runtime_error("BOS native truncated 8-bit payload");
        }
        int i = 1;
        std::size_t p = 0;
        for (; i + 3 < block_len; i += 4, p += 4) {
          dst[i] = add_zigzag_delta(prev_bits, payload[p]);
          dst[i + 1] = add_zigzag_delta(prev_bits, payload[p + 1]);
          dst[i + 2] = add_zigzag_delta(prev_bits, payload[p + 2]);
          dst[i + 3] = add_zigzag_delta(prev_bits, payload[p + 3]);
        }
        for (; i < block_len; ++i, ++p) {
          dst[i] = add_zigzag_delta(prev_bits, payload[p]);
        }
      } else if (width == 16) {
        if (static_cast<std::size_t>(payload_len) < static_cast<std::size_t>(block_len - 1) * 2) {
          throw std::runtime_error("BOS native truncated 16-bit payload");
        }
        std::size_t p = 0;
        for (int i = 1; i < block_len; ++i) {
          const std::uint64_t z = (static_cast<std::uint64_t>(payload[p]) << 8) |
                                  static_cast<std::uint64_t>(payload[p + 1]);
          p += 2;
          dst[i] = add_zigzag_delta(prev_bits, z);
        }
      } else if (width == 6 || width == 10 || width == 12 || width == 14) {
        switch (width) {
          case 6:
            decode_fixed_width_deltas<6>(dst, block_len, payload, payload_len, prev_bits);
            break;
          case 10:
            decode_fixed_width_deltas<10>(dst, block_len, payload, payload_len, prev_bits);
            break;
          case 12:
            decode_fixed_width_deltas<12>(dst, block_len, payload, payload_len, prev_bits);
            break;
          case 14:
            decode_fixed_width_deltas<14>(dst, block_len, payload, payload_len, prev_bits);
            break;
          default:
            throw std::runtime_error("BOS native unsupported fixed width");
        }
      } else {
        const std::size_t required_bytes =
            (static_cast<std::size_t>(block_len - 1) * static_cast<std::size_t>(width) + 7) / 8;
        if (static_cast<std::size_t>(payload_len) < required_bytes) {
          throw std::runtime_error("BOS native truncated bit-packed payload");
        }
        UncheckedBitReader br(payload, width);
        int i = 1;
        for (; i + 3 < block_len; i += 4) {
          dst[i] = add_zigzag_delta(prev_bits, br.read());
          dst[i + 1] = add_zigzag_delta(prev_bits, br.read());
          dst[i + 2] = add_zigzag_delta(prev_bits, br.read());
          dst[i + 3] = add_zigzag_delta(prev_bits, br.read());
        }
        for (; i < block_len; ++i) {
          const std::uint64_t z = br.read();
          dst[i] = add_zigzag_delta(prev_bits, z);
        }
      }
    }
    out_pos += static_cast<std::size_t>(block_len);
  }
  if (pos != len) throw std::runtime_error("BOS native trailing bytes");
  return out;
}

void benchInt64ColumnsBos(const std::vector<std::vector<std::int64_t>> &columns,
                          std::vector<long long> &result, LossAccum *loss) {
  std::vector<std::vector<std::optional<PayloadRow>>> groups;
  groups.reserve(columns.size());
  for (const auto &col : columns) {
    groups.push_back({bos_int_row(col), ts2_int_row(col)});
    if (loss != nullptr && !col.empty()) {
      benchmark_loss::accumulate_int64_column(col, col, *loss);
    }
  }
  assign_result(bench_payload_groups(groups, {"raw"}, true), result);
}

void benchDoubleColumnsBos(const std::vector<std::vector<double>> &columns,
                           const std::vector<int> *max_point_per_column,
                           std::vector<long long> &result,
                           LossAccum *loss) {
  std::vector<std::vector<std::optional<PayloadRow>>> groups;
  groups.reserve(columns.size());
  for (std::size_t ci = 0; ci < columns.size(); ++ci) {
    const int *cap =
        (max_point_per_column != nullptr && ci < max_point_per_column->size())
            ? &(*max_point_per_column)[ci]
            : nullptr;
    groups.push_back({bos_double_row(columns[ci], cap), ts2_double_row(columns[ci], cap)});
    if (loss != nullptr && !columns[ci].empty()) {
      benchmark_loss::accumulate_double_column(columns[ci], columns[ci], *loss);
    }
  }
  assign_result(bench_payload_groups(groups, {"raw"}, true), result);
}

void benchInt64ColumnsBosLevel(const std::vector<std::vector<std::int64_t>> &columns,
                               int level,
                               std::vector<long long> &result,
                               LossAccum *loss) {
  std::vector<std::vector<std::optional<PayloadRow>>> groups;
  groups.reserve(columns.size());
  for (const auto &col : columns) {
    groups.push_back(int_level_rows(col, level));
    if (loss != nullptr && !col.empty()) {
      benchmark_loss::accumulate_int64_column(col, col, *loss);
    }
  }
  assign_result(bench_payload_groups(groups, level_operators(level), true, level, true), result);
}

void benchDoubleColumnsBosLevel(const std::vector<std::vector<double>> &columns,
                                const std::vector<int> *max_point_per_column,
                                int level,
                                std::vector<long long> &result,
                                LossAccum *loss) {
  std::vector<std::vector<std::optional<PayloadRow>>> groups;
  groups.reserve(columns.size());
  for (std::size_t ci = 0; ci < columns.size(); ++ci) {
    const int *cap =
        (max_point_per_column != nullptr && ci < max_point_per_column->size())
            ? &(*max_point_per_column)[ci]
            : nullptr;
    groups.push_back(double_level_rows(columns[ci], cap, level));
    if (loss != nullptr && !columns[ci].empty()) {
      benchmark_loss::accumulate_double_column(columns[ci], columns[ci], *loss);
    }
  }
  assign_result(bench_payload_groups(groups, level_operators(level), true, level, true), result);
}

} // namespace ts2diff_bos_native
