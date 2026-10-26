#include "ts2diff_bos_native.hpp"
#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::int64_t> make_delta_column(std::size_t n, int pattern) {
  std::vector<std::int64_t> values(n);
  std::int64_t cur = 1000;
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = cur;
    switch (pattern) {
      case 1:
        cur += (i + 2 == n) ? 2 : 1;
        break;
      case 2:
        cur += 1;
        break;
      case 3:
        cur += static_cast<std::int64_t>(i % 4) - 1;
        break;
      case 4:
        cur += static_cast<std::int64_t>(i % 8) - 3;
        break;
      case 8:
        cur += static_cast<std::int64_t>(i % 128) - 63;
        break;
      case 10:
        cur += static_cast<std::int64_t>(i % 1024) - 511;
        break;
      case 11:
        cur += static_cast<std::int64_t>(i % 2048) - 1023;
        break;
      case 12:
        cur += static_cast<std::int64_t>(i % 4096) - 2047;
        break;
      case 13:
        cur += static_cast<std::int64_t>(i % 8192) - 4095;
        break;
      case 16:
        cur += static_cast<std::int64_t>((i * 131) % 32768) - 16384;
        break;
      default:
        cur += static_cast<std::int64_t>(i % 16) - 7;
        break;
    }
  }
  return values;
}

std::uint64_t checksum(const std::vector<std::int64_t> &values) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::int64_t v : values) {
    h ^= static_cast<std::uint64_t>(v);
    h *= 1099511628211ULL;
  }
  return h;
}

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

std::uint64_t low_bits_mask(int bits) {
  return bits >= 64 ? std::numeric_limits<std::uint64_t>::max() : ((1ULL << bits) - 1ULL);
}

std::int32_t read_i32_be(const std::uint8_t *data, std::size_t len, std::size_t &pos) {
  if (pos + 4 > len) throw std::runtime_error("baseline truncated int32");
  std::uint32_t u = 0;
  for (int i = 0; i < 4; ++i) u = (u << 8) | data[pos + i];
  pos += 4;
  return static_cast<std::int32_t>(u);
}

std::uint64_t read_u64_be(const std::uint8_t *data, std::size_t len, std::size_t &pos) {
  if (pos + 8 > len) throw std::runtime_error("baseline truncated int64");
  std::uint64_t u = 0;
  for (int i = 0; i < 8; ++i) u = (u << 8) | data[pos + i];
  pos += 8;
  return u;
}

std::int64_t zigzag_decode64(std::uint64_t v) {
  const std::uint64_t decoded = (v >> 1) ^ (0ULL - (v & 1ULL));
  return to_i64(decoded);
}

std::uint64_t zigzag_encode64(std::int64_t v) {
  return (to_u64(v) << 1) ^ (v < 0 ? std::numeric_limits<std::uint64_t>::max() : 0ULL);
}

int bit_width64(std::uint64_t v) {
  return v == 0 ? 1 : 64 - __builtin_clzll(v);
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

std::vector<std::uint8_t> baseline_encode_int64_column_payload(
    const std::vector<std::int64_t> &values) {
  if (values.empty()) return {};
  int width = 1;
  std::int64_t prev = values.front();
  for (std::size_t i = 1; i < values.size(); ++i) {
    const std::int64_t cur = values[i];
    width = std::max(width, bit_width64(zigzag_encode64(
                                to_i64(to_u64(cur) - to_u64(prev)))));
    prev = cur;
  }
  const int packed_width = width > 56 ? 64 : width;
  const std::size_t payload_len = packed_width == 64
                                      ? (values.size() - 1) * 8
                                      : ((values.size() - 1) * packed_width + 7) / 8;
  std::vector<std::uint8_t> out;
  out.reserve(25 + payload_len);
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  put_i32_be(out, static_cast<std::int32_t>(values.size()));
  put_u64_be(out, to_u64(values.front()));
  out.push_back(static_cast<std::uint8_t>(packed_width));
  put_i32_be(out, static_cast<std::int32_t>(payload_len));

  prev = values.front();
  std::uint64_t buffer = 0;
  int used = 0;
  for (std::size_t i = 1; i < values.size(); ++i) {
    const std::int64_t cur = values[i];
    const std::uint64_t z = zigzag_encode64(to_i64(to_u64(cur) - to_u64(prev)));
    prev = cur;
    if (packed_width == 64) {
      put_u64_be(out, z);
      continue;
    }
    buffer = (buffer << packed_width) | (z & low_bits_mask(packed_width));
    used += packed_width;
    while (used >= 8) {
      const int shift = used - 8;
      out.push_back(static_cast<std::uint8_t>((buffer >> shift) & 0xffu));
      used -= 8;
      buffer = used == 0 ? 0 : (buffer & low_bits_mask(used));
    }
  }
  if (used > 0) {
    out.push_back(static_cast<std::uint8_t>((buffer << (8 - used)) & 0xffu));
  }
  return out;
}

std::vector<std::int64_t> baseline_decode_int64_column_payload(const std::uint8_t *data,
                                                               std::size_t len);

void verify_round_trips() {
  const std::size_t lengths[] = {1, 2, 3, 4, 5, 8, 9, 16, 17, 31};
  for (int source_width = 1; source_width <= 64; ++source_width) {
    const std::uint64_t max_z = 1ULL << (source_width - 1);
    const int expected_width = source_width > 56 ? 64 : source_width;
    for (std::size_t n : lengths) {
      std::vector<std::int64_t> values(n);
      values[0] = source_width % 2 == 0 ? std::numeric_limits<std::int64_t>::max() - 7
                                        : std::numeric_limits<std::int64_t>::min() + 7;
      for (std::size_t i = 1; i < n; ++i) {
        const std::uint64_t z = i == 1 ? max_z : ((max_z - 1) & (i * 17));
        values[i] = to_i64(to_u64(values[i - 1]) + to_u64(zigzag_decode64(z)));
      }
      const auto payload = ts2diff_bos_native::encode_int64_column_payload(values);
      const auto baseline_payload = baseline_encode_int64_column_payload(values);
      if (n > 1 && (payload.size() <= 20 || static_cast<int>(payload[20]) != expected_width)) {
        throw std::runtime_error("BOS microbench encoded unexpected bit width");
      }
      if (payload != baseline_payload) {
        throw std::runtime_error("BOS microbench payload compatibility failed");
      }
      const auto decoded =
          ts2diff_bos_native::decode_int64_column_payload(payload.data(), payload.size());
      const auto baseline = baseline_decode_int64_column_payload(payload.data(), payload.size());
      if (decoded != values || baseline != values) {
        throw std::runtime_error("BOS microbench width round-trip failed");
      }
    }
  }

  const std::vector<std::int64_t> extremes = {
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::max(),
      0,
      -1,
      1,
      std::numeric_limits<std::int64_t>::min(),
  };
  const auto payload = ts2diff_bos_native::encode_int64_column_payload(extremes);
  if (payload != baseline_encode_int64_column_payload(extremes)) {
    throw std::runtime_error("BOS microbench extreme payload compatibility failed");
  }
  const auto decoded = ts2diff_bos_native::decode_int64_column_payload(payload.data(), payload.size());
  if (decoded != extremes) throw std::runtime_error("BOS microbench extreme round-trip failed");
}

class BaselineBitReader {
 public:
  BaselineBitReader(const std::uint8_t *data, std::size_t len) : data_(data), len_(len) {}

  std::uint64_t read(int bits) {
    while (used_ < bits) {
      if (pos_ >= len_) throw std::runtime_error("baseline bitread past end");
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

std::vector<std::int64_t> baseline_decode_int64_column_payload(const std::uint8_t *data,
                                                               std::size_t len) {
  if (data == nullptr || len == 0) return {};
  std::size_t pos = 0;
  const int n = read_i32_be(data, len, pos);
  const int segment_rows = read_i32_be(data, len, pos);
  if (n < 0 || segment_rows <= 0) throw std::runtime_error("baseline invalid header");
  std::vector<std::int64_t> out(static_cast<std::size_t>(n));
  std::size_t out_pos = 0;
  while (out_pos < static_cast<std::size_t>(n)) {
    const int block_len = read_i32_be(data, len, pos);
    if (block_len <= 0 || out_pos + static_cast<std::size_t>(block_len) >
                              static_cast<std::size_t>(n)) {
      throw std::runtime_error("baseline invalid segment length");
    }
    const std::int64_t first = to_i64(read_u64_be(data, len, pos));
    if (pos >= len) throw std::runtime_error("baseline truncated width");
    const int width = data[pos++];
    const int payload_len = read_i32_be(data, len, pos);
    if (payload_len < 0 || pos + static_cast<std::size_t>(payload_len) > len) {
      throw std::runtime_error("baseline invalid payload length");
    }
    const std::uint8_t *payload = data + pos;
    pos += static_cast<std::size_t>(payload_len);
    out[out_pos] = first;
    if (block_len > 1) {
      std::int64_t prev = first;
      if (width >= 64) {
        std::size_t p = 0;
        for (int i = 1; i < block_len; ++i) {
          std::size_t local = p;
          const std::uint64_t z = read_u64_be(payload, payload_len, local);
          p += 8;
          prev = to_i64(to_u64(prev) + to_u64(zigzag_decode64(z)));
          out[out_pos + static_cast<std::size_t>(i)] = prev;
        }
      } else {
        BaselineBitReader br(payload, static_cast<std::size_t>(payload_len));
        for (int i = 1; i < block_len; ++i) {
          const std::uint64_t z = br.read(width);
          prev = to_i64(to_u64(prev) + to_u64(zigzag_decode64(z)));
          out[out_pos + static_cast<std::size_t>(i)] = prev;
        }
      }
    }
    out_pos += static_cast<std::size_t>(block_len);
  }
  if (pos != len) throw std::runtime_error("baseline trailing bytes");
  return out;
}

template <typename DecodeFn>
long long time_decode(const std::vector<std::uint8_t> &payload,
                      int rounds,
                      DecodeFn decode_fn,
                      std::uint64_t &guard) {
  const auto t0 = Clock::now();
  for (int i = 0; i < rounds; ++i) {
    const auto out = decode_fn(payload.data(), payload.size());
    guard ^= static_cast<std::uint64_t>(out.size()) + static_cast<std::uint64_t>(i);
    if (!out.empty()) {
      guard ^= static_cast<std::uint64_t>(out.front());
      guard ^= static_cast<std::uint64_t>(out[out.size() / 2]);
      guard ^= static_cast<std::uint64_t>(out.back());
    }
  }
  const auto t1 = Clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

template <typename EncodeFn>
long long time_encode(const std::vector<std::int64_t> &values,
                      int rounds,
                      EncodeFn encode_fn,
                      std::uint64_t &guard) {
  const auto t0 = Clock::now();
  for (int i = 0; i < rounds; ++i) {
    const auto payload = encode_fn(values);
    guard ^= static_cast<std::uint64_t>(payload.size()) + static_cast<std::uint64_t>(i);
    if (!payload.empty()) {
      guard ^= payload.front();
      guard ^= payload[payload.size() / 2];
      guard ^= payload.back();
    }
  }
  const auto t1 = Clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

std::vector<std::int64_t> decode_ts2_payload(const std::uint8_t *data, std::size_t len) {
  std::vector<std::int64_t> out;
  if (!tsfile::p02::decode_ts_2diff_int64(std::span<const std::uint8_t>(data, len), out)) {
    throw std::runtime_error("TS_2DIFF microbench decode failed");
  }
  return out;
}

void run_case(int pattern, std::size_t n, int rounds) {
  const auto values = make_delta_column(n, pattern);
  const auto payload = ts2diff_bos_native::encode_int64_column_payload(values);
  if (payload != baseline_encode_int64_column_payload(values)) {
    throw std::runtime_error("BOS microbench large payload compatibility failed");
  }
  const auto ts2_payload = tsfile::p02::encode_ts_2diff_int64(
      std::span<const std::int64_t>(values.data(), values.size()));
  const auto baseline = baseline_decode_int64_column_payload(payload.data(), payload.size());
  const auto decoded = ts2diff_bos_native::decode_int64_column_payload(payload.data(), payload.size());
  const auto ts2_decoded = decode_ts2_payload(ts2_payload.data(), ts2_payload.size());
  if (baseline != values || decoded != values || ts2_decoded != values) {
    throw std::runtime_error("BOS microbench lossless check failed");
  }

  std::uint64_t baseline_guard = checksum(baseline);
  std::uint64_t optimized_guard = checksum(decoded);
  std::uint64_t ts2_guard = checksum(ts2_decoded);
  std::uint64_t baseline_bos_encode_guard = 0;
  std::uint64_t bos_encode_guard = 0;
  std::uint64_t ts2_encode_guard = 0;
  const auto baseline_bos_encode_ns = time_encode(
      values, rounds, baseline_encode_int64_column_payload, baseline_bos_encode_guard);
  const auto bos_encode_ns = time_encode(
      values, rounds, ts2diff_bos_native::encode_int64_column_payload, bos_encode_guard);
  const auto ts2_encode_ns = time_encode(
      values, rounds,
      [](const std::vector<std::int64_t> &input) {
        return tsfile::p02::encode_ts_2diff_int64(
            std::span<const std::int64_t>(input.data(), input.size()));
      },
      ts2_encode_guard);
  const auto baseline_ns =
      time_decode(payload, rounds, baseline_decode_int64_column_payload, baseline_guard);
  const auto optimized_ns =
      time_decode(payload, rounds, ts2diff_bos_native::decode_int64_column_payload, optimized_guard);
  const auto ts2_decode_ns = time_decode(ts2_payload, rounds, decode_ts2_payload, ts2_guard);
  const double points = static_cast<double>(std::max<std::size_t>(1, n) * rounds);
  const double baseline_bos_encode_ns_per_value =
      static_cast<double>(baseline_bos_encode_ns) / points;
  const double bos_encode_ns_per_value = static_cast<double>(bos_encode_ns) / points;
  const double ts2_encode_ns_per_value = static_cast<double>(ts2_encode_ns) / points;
  const double baseline_ns_per_value =
      static_cast<double>(baseline_ns) / points;
  const double optimized_ns_per_value = static_cast<double>(optimized_ns) / points;
  const double ts2_decode_ns_per_value = static_cast<double>(ts2_decode_ns) / points;
  const double speedup =
      baseline_ns > 0 ? static_cast<double>(baseline_ns) / static_cast<double>(optimized_ns) : 0.0;
  const double reduction = baseline_ns > 0 ? 100.0 * (1.0 - 1.0 / speedup) : 0.0;
  const int packed_width = payload.size() > 20 ? static_cast<int>(payload[20]) : -1;
  std::cout << "pattern,width_hint=" << pattern << ",values=" << n
            << ",rounds=" << rounds << ",compressedSize=" << payload.size()
            << ",ts2CompressedSize=" << ts2_payload.size()
            << ",packedWidth=" << packed_width
            << ",baselineBosEncodeNsPerValue=" << baseline_bos_encode_ns_per_value
            << ",bosEncodeNsPerValue=" << bos_encode_ns_per_value
            << ",ts2EncodeNsPerValue=" << ts2_encode_ns_per_value
            << ",baselineDecodeNs=" << baseline_ns
            << ",optimizedDecodeNs=" << optimized_ns
            << ",baselineNsPerValue=" << baseline_ns_per_value
            << ",optimizedNsPerValue=" << optimized_ns_per_value
            << ",ts2DecodeNsPerValue=" << ts2_decode_ns_per_value
            << ",speedup=" << speedup
            << ",reductionPct=" << reduction
            << ",checksum=" << (baseline_guard ^ optimized_guard ^ ts2_guard ^
                                  baseline_bos_encode_guard ^ bos_encode_guard ^ ts2_encode_guard)
            << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  verify_round_trips();
  const std::size_t n = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 2000000;
  const int rounds = argc > 2 ? std::stoi(argv[2]) : 8;
  for (int pattern : {1, 2, 3, 4, 8, 10, 11, 12, 13, 16}) {
    run_case(pattern, n, rounds);
  }
  return 0;
}
