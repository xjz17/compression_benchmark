#include "tsfile_bench_tsfile.hpp"

#include "algorithm_allow_list.hpp"
#include "benchmark_columns.hpp"
#include "benchmark_loss.hpp"
#include "pongo_codec.hpp"

#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_decode_p03.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <ostream>
#include <span>
#include <vector>

namespace {

using clock_ns = std::chrono::high_resolution_clock;

int clamp_ts2diff_max_point_for_double(const std::vector<double>& col, int requested_max_point) {
  int normalized = std::max(0, requested_max_point);
  if (col.empty()) {
    return normalized;
  }

  double max_abs = 0.0;
  for (double v : col) {
    if (!std::isfinite(v)) {
      continue;
    }
    max_abs = std::max(max_abs, std::abs(v));
  }
  if (!(max_abs > 0.0)) {
    return normalized;
  }

  double ratio = static_cast<double>(std::numeric_limits<std::int64_t>::max()) / max_abs;
  if (!std::isfinite(ratio)) {
    return normalized;
  }
  if (ratio <= 0.0) {
    return 0;
  }

  int safe_upper = static_cast<int>(std::floor(std::log10(ratio)));
  safe_upper = std::max(0, safe_upper);
  return std::min(normalized, safe_upper);
}

static std::span<const std::int32_t> i32span(const std::vector<int>& c) {
  static_assert(sizeof(int) == sizeof(std::int32_t));
  return std::span<const std::int32_t>(reinterpret_cast<const std::int32_t*>(c.data()), c.size());
}

inline bool should_skip(const char* algo_name, const std::unordered_set<std::string>* skip_algorithms) {
  return algorithm_allow_list::should_skip(skip_algorithms, std::string(algo_name));
}

template <class EncFn, class DecFn>
void bench_one(const char* algo_name,
               EncFn encode_fn,
               DecFn decode_fn,
               const std::vector<std::vector<int>>& cols,
               std::ostream& out) {
  long long total_original = 0;
  long long total_encoded = 0;
  long long total_comp = 0;
  long long total_decomp = 0;
  for (const auto& col : cols) {
    total_original += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = encode_fn(col);
    auto t1 = clock_ns::now();
    if (enc.empty()) continue;
    total_encoded += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<std::int32_t> dec;
    if (!decode_fn(enc, dec)) continue;
    auto t3 = clock_ns::now();
    total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    (void)dec;
  }
  out << algo_name << ',' << total_original << ',' << total_encoded << ',' << total_comp << ','
      << total_decomp << '\n';
}

template <class EncFn, class DecFn>
void bench_one_i64(const char* algo_name,
                   EncFn encode_fn,
                   DecFn decode_fn,
                   const std::vector<std::vector<std::int64_t>>& cols,
                   std::ostream& out) {
  long long total_original = 0;
  long long total_encoded = 0;
  long long total_comp = 0;
  long long total_decomp = 0;
  for (const auto& col : cols) {
    if (col.empty()) continue;
    total_original += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = encode_fn(col);
    auto t1 = clock_ns::now();
    if (enc.empty()) continue;
    total_encoded += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<std::int64_t> dec;
    if (!decode_fn(enc, dec)) continue;
    auto t3 = clock_ns::now();
    total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    (void)dec;
  }
  if (total_original > 0) {
    out << algo_name << ',' << total_original << ',' << total_encoded << ',' << total_comp << ','
        << total_decomp << '\n';
  }
}

template <class EncFn, class DecFn>
void bench_one_i64_accumulate(const char* algo_name,
                              EncFn encode_fn,
                              DecFn decode_fn,
                              const std::vector<std::vector<std::int64_t>>& cols,
                              BenchmarkTotals& totals,
                              const std::unordered_set<std::string>* skip_algorithms) {
  if (should_skip(algo_name, skip_algorithms)) {
    return;
  }
  long long total_original = 0;
  long long total_encoded = 0;
  long long total_comp = 0;
  long long total_decomp = 0;
  LossAccum loss;
  for (const auto& col : cols) {
    if (col.empty()) continue;
    total_original += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = encode_fn(col);
    auto t1 = clock_ns::now();
    if (enc.empty()) continue;
    total_encoded += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<std::int64_t> dec;
    if (!decode_fn(enc, dec)) continue;
    auto t3 = clock_ns::now();
    benchmark_loss::accumulate_int64_column(col, dec, loss);
    total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
  }
  totals.merge(algo_name, total_original, total_encoded, total_comp, total_decomp, loss);
}

template <class EncFn, class DecFn>
void bench_one_double(const char* algo_name,
                      EncFn encode_fn,
                      DecFn decode_fn,
                      const std::vector<std::vector<double>>& cols,
                      const std::vector<int>& float_max_point_per_column,
                      std::ostream& out) {
  long long total_original = 0;
  long long total_encoded = 0;
  long long total_comp = 0;
  long long total_decomp = 0;
  for (std::size_t ci = 0; ci < cols.size(); ++ci) {
    const auto& col = cols[ci];
    if (col.empty()) continue;
    int mp = 0;
    if (ci < float_max_point_per_column.size()) {
      mp = std::max(0, float_max_point_per_column[ci]);
    }
    total_original += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = encode_fn(col, mp);
    auto t1 = clock_ns::now();
    if (enc.empty()) continue;
    total_encoded += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<double> dec;
    if (!decode_fn(enc, dec)) continue;
    auto t3 = clock_ns::now();
    total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    (void)dec;
  }
  if (total_original > 0) {
    out << algo_name << ',' << total_original << ',' << total_encoded << ',' << total_comp << ','
        << total_decomp << '\n';
  }
}

template <class EncFn, class DecFn>
void bench_one_double_accumulate(const char* algo_name,
                                 EncFn encode_fn,
                                 DecFn decode_fn,
                                 const std::vector<std::vector<double>>& cols,
                                 const std::vector<int>& float_max_point_per_column,
                                 BenchmarkTotals& totals,
                                 const std::unordered_set<std::string>* skip_algorithms) {
  if (should_skip(algo_name, skip_algorithms)) {
    return;
  }
  long long total_original = 0;
  long long total_encoded = 0;
  long long total_comp = 0;
  long long total_decomp = 0;
  LossAccum loss;
  for (std::size_t ci = 0; ci < cols.size(); ++ci) {
    const auto& col = cols[ci];
    if (col.empty()) continue;
    int mp = 0;
    if (ci < float_max_point_per_column.size()) {
      mp = std::max(0, float_max_point_per_column[ci]);
    }
    total_original += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = encode_fn(col, mp);
    auto t1 = clock_ns::now();
    if (enc.empty()) continue;
    total_encoded += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<double> dec;
    if (!decode_fn(enc, dec)) continue;
    auto t3 = clock_ns::now();
    benchmark_loss::accumulate_double_column(col, dec, loss);
    total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
  }
  totals.merge(algo_name, total_original, total_encoded, total_comp, total_decomp, loss);
}

template <class EncFn, class DecFn>
void bench_one_float(const char* algo_name,
                     EncFn encode_fn,
                     DecFn decode_fn,
                     const std::vector<std::vector<float>>& cols,
                     std::ostream& out) {
  long long total_original = 0;
  long long total_encoded = 0;
  long long total_comp = 0;
  long long total_decomp = 0;
  for (const auto& col : cols) {
    total_original += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = encode_fn(col);
    auto t1 = clock_ns::now();
    if (enc.empty()) continue;
    total_encoded += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<float> dec;
    if (!decode_fn(enc, dec)) continue;
    auto t3 = clock_ns::now();
    total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    (void)dec;
  }
  out << algo_name << ',' << total_original << ',' << total_encoded << ',' << total_comp << ','
      << total_decomp << '\n';
}

}  // namespace

void bench_tsfile_rows_int(const std::vector<std::vector<int>>& columns, std::ostream& out,
                           const std::unordered_set<std::string>* skip_algorithms) {
  if (!should_skip("TS_2DIFF", skip_algorithms)) {
  bench_one(
      "TS_2DIFF",
      [](const std::vector<int>& c) { return tsfile::p02::encode_ts_2diff_int32(i32span(c)); },
      [](const std::vector<std::uint8_t>& e, std::vector<std::int32_t>& d) {
        return tsfile::p02::decode_ts_2diff_int32(e, d);
      },
      columns, out);
  }
  if (!should_skip("RLE", skip_algorithms)) {
  bench_one(
      "RLE",
      [](const std::vector<int>& c) { return tsfile::p02::encode_rle_int32(i32span(c)); },
      [](const std::vector<std::uint8_t>& e, std::vector<std::int32_t>& d) {
        return tsfile::p02::decode_rle_int32(e, d);
      },
      columns, out);
  }
  if (!should_skip("GORILLA", skip_algorithms)) {
  bench_one(
      "GORILLA",
      [](const std::vector<int>& c) { return tsfile::p03::encode_gorilla_int32(i32span(c)); },
      [](const std::vector<std::uint8_t>& e, std::vector<std::int32_t>& d) {
        return tsfile::p03::decode_gorilla_int32(e, d);
      },
      columns, out);
  }
  if (!should_skip("CHIMP", skip_algorithms)) {
  bench_one(
      "CHIMP",
      [](const std::vector<int>& c) { return tsfile::p03::encode_chimp_int32(i32span(c)); },
      [](const std::vector<std::uint8_t>& e, std::vector<std::int32_t>& d) {
        return tsfile::p03::decode_chimp_int32(e, d);
      },
      columns, out);
  }
  if (!should_skip("SPRINTZ", skip_algorithms)) {
  bench_one(
      "SPRINTZ",
      [](const std::vector<int>& c) { return tsfile::p04::encode_sprintz_int32(i32span(c)); },
      [](const std::vector<std::uint8_t>& e, std::vector<std::int32_t>& d) {
        return tsfile::p04::decode_sprintz_int32(e, d);
      },
      columns, out);
  }
  if (!should_skip("RLBE", skip_algorithms)) {
  bench_one(
      "RLBE",
      [](const std::vector<int>& c) { return tsfile::p04::encode_rlbe_int32(i32span(c)); },
      [](const std::vector<std::uint8_t>& e, std::vector<std::int32_t>& d) {
        return tsfile::p04::decode_rlbe_int32(e, d);
      },
      columns, out);
  }
}

void bench_tsfile_rows_float(const std::vector<std::vector<float>>& columns,
                             const std::vector<int>& float_max_point_per_column, std::ostream& out,
                             const std::unordered_set<std::string>* skip_algorithms) {
  auto max_point_for_column = [&](std::size_t col_index) -> int {
    if (col_index < float_max_point_per_column.size()) {
      return std::max(0, float_max_point_per_column[col_index]);
    }
    return 0;
  };

  if (!should_skip("TS_2DIFF", skip_algorithms)) {
    long long total_original = 0;
    long long total_encoded = 0;
    long long total_comp = 0;
    long long total_decomp = 0;
    for (std::size_t ci = 0; ci < columns.size(); ++ci) {
      const std::vector<float>& col = columns[ci];
      if (col.empty()) {
        continue;
      }
      total_original += static_cast<long long>(col.size()) * 8;
      const int mp = max_point_for_column(ci);
      auto t0 = clock_ns::now();
      std::vector<std::uint8_t> enc = tsfile::p02::encode_ts_2diff_float(
          std::span<const float>(col.data(), col.size()), mp);
      auto t1 = clock_ns::now();
      if (enc.empty()) {
        continue;
      }
      total_encoded += static_cast<long long>(enc.size());
      auto t2 = clock_ns::now();
      std::vector<float> dec;
      if (!tsfile::p02::decode_ts_2diff_float(enc, dec)) {
        continue;
      }
      auto t3 = clock_ns::now();
      total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
      (void)dec;
    }
    out << "TS_2DIFF," << total_original << ',' << total_encoded << ',' << total_comp << ',' << total_decomp << '\n';
  }
  if (!should_skip("RLE", skip_algorithms)) {
    long long total_original = 0;
    long long total_encoded = 0;
    long long total_comp = 0;
    long long total_decomp = 0;
    for (std::size_t ci = 0; ci < columns.size(); ++ci) {
      const std::vector<float>& col = columns[ci];
      if (col.empty()) {
        continue;
      }
      total_original += static_cast<long long>(col.size()) * 8;
      const int mp = max_point_for_column(ci);
      auto t0 = clock_ns::now();
      std::vector<std::uint8_t> enc =
          tsfile::p02::encode_rle_float(std::span<const float>(col.data(), col.size()), mp);
      auto t1 = clock_ns::now();
      if (enc.empty()) {
        continue;
      }
      total_encoded += static_cast<long long>(enc.size());
      auto t2 = clock_ns::now();
      std::vector<float> dec;
      if (!tsfile::p02::decode_rle_float(enc, dec)) {
        continue;
      }
      auto t3 = clock_ns::now();
      total_comp += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      total_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
      (void)dec;
    }
    out << "RLE," << total_original << ',' << total_encoded << ',' << total_comp << ',' << total_decomp << '\n';
  }
  if (!should_skip("GORILLA", skip_algorithms)) {
  bench_one_float(
      "GORILLA",
      [](const std::vector<float>& c) {
        return tsfile::p03::encode_gorilla_float(std::span<const float>(c.data(), c.size()));
      },
      [](const std::vector<std::uint8_t>& e, std::vector<float>& d) {
        return tsfile::p03::decode_gorilla_float(e, d);
      },
      columns, out);
  }
  if (!should_skip("CHIMP", skip_algorithms)) {
  bench_one_float(
      "CHIMP",
      [](const std::vector<float>& c) {
        return tsfile::p03::encode_chimp_float(std::span<const float>(c.data(), c.size()));
      },
      [](const std::vector<std::uint8_t>& e, std::vector<float>& d) {
        return tsfile::p03::decode_chimp_float(e, d);
      },
      columns, out);
  }
  if (!should_skip("SPRINTZ", skip_algorithms)) {
  bench_one_float(
      "SPRINTZ",
      [](const std::vector<float>& c) {
        return tsfile::p04::encode_sprintz_float(std::span<const float>(c.data(), c.size()));
      },
      [](const std::vector<std::uint8_t>& e, std::vector<float>& d) {
        return tsfile::p04::decode_sprintz_float(e, d);
      },
      columns, out);
  }
  if (!should_skip("RLBE", skip_algorithms)) {
  bench_one_float(
      "RLBE",
      [](const std::vector<float>& c) {
        return tsfile::p04::encode_rlbe_float(std::span<const float>(c.data(), c.size()));
      },
      [](const std::vector<std::uint8_t>& e, std::vector<float>& d) {
        return tsfile::p04::decode_rlbe_float(e, d);
      },
      columns, out);
  }
}

void bench_tsfile_rows_int64(const std::vector<std::vector<std::int64_t>>& columns, std::ostream& out,
                             const std::unordered_set<std::string>* skip_algorithms) {
  if (!should_skip("TS_2DIFF", skip_algorithms)) {
    bench_one_i64(
        "TS_2DIFF",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p02::encode_ts_2diff_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p02::decode_ts_2diff_int64(e, d);
        },
        columns, out);
  }
  if (!should_skip("CHIMP128", skip_algorithms)) {
    bench_one_i64(
        "CHIMP128",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p03::encode_chimp_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p03::decode_chimp_int64(e, d);
        },
        columns, out);
  }
  if (!should_skip("RLE", skip_algorithms)) {
    bench_one_i64(
        "RLE",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p02::encode_rle_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p02::decode_rle_int64(e, d);
        },
        columns, out);
  }
  if (!should_skip("GORILLA", skip_algorithms)) {
    bench_one_i64(
        "GORILLA",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p03::encode_gorilla_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p03::decode_gorilla_int64(e, d);
        },
        columns, out);
  }
  if (!should_skip("CHIMP", skip_algorithms)) {
    bench_one_i64(
        "CHIMP",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p03::encode_chimp_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p03::decode_chimp_int64(e, d);
        },
        columns, out);
  }
  if (!should_skip("SPRINTZ", skip_algorithms)) {
    bench_one_i64(
        "SPRINTZ",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p04::encode_sprintz_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p04::decode_sprintz_int64(e, d);
        },
        columns, out);
  }
  if (!should_skip("RLBE", skip_algorithms)) {
    bench_one_i64(
        "RLBE",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p04::encode_rlbe_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p04::decode_rlbe_int64(e, d);
        },
        columns, out);
  }
}

void bench_tsfile_rows_double(const std::vector<std::vector<double>>& columns,
                              const std::vector<int>& float_max_point_per_column, std::ostream& out,
                              const std::unordered_set<std::string>* skip_algorithms) {
  if (!should_skip("TS_2DIFF", skip_algorithms)) {
    bench_one_double(
        "TS_2DIFF",
        [](const std::vector<double>& c, int mp) {
          int effective_mp = clamp_ts2diff_max_point_for_double(c, mp);
          return tsfile::p02::encode_ts_2diff_double(std::span<const double>(c.data(), c.size()),
                                                     effective_mp);
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p02::decode_ts_2diff_double(e, d);
        },
        columns, float_max_point_per_column, out);
  }
  if (!should_skip("RLE", skip_algorithms)) {
    bench_one_double(
        "RLE",
        [](const std::vector<double>& c, int mp) {
          return tsfile::p02::encode_rle_double(std::span<const double>(c.data(), c.size()), mp);
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p02::decode_rle_double(e, d);
        },
        columns, float_max_point_per_column, out);
  }
  if (!should_skip("GORILLA", skip_algorithms)) {
    bench_one_double(
        "GORILLA",
        [](const std::vector<double>& c, int) {
          return tsfile::p03::encode_gorilla_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p03::decode_gorilla_double(e, d);
        },
        columns, float_max_point_per_column, out);
  }
  if (!should_skip("CHIMP128", skip_algorithms)) {
    bench_one_double(
        "CHIMP128",
        [](const std::vector<double>& c, int) {
          return tsfile::p03::encode_chimp_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p03::decode_chimp_double(e, d);
        },
        columns, float_max_point_per_column, out);
  }
  if (!should_skip(pongo_codec::kRowName, skip_algorithms)) {
    bench_one_double(
        pongo_codec::kRowName,
        [](const std::vector<double>& c, int) {
          return pongo_codec::encode(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return pongo_codec::decode(e, d);
        },
        columns, float_max_point_per_column, out);
  }
  if (!should_skip("SPRINTZ", skip_algorithms)) {
    bench_one_double(
        "SPRINTZ",
        [](const std::vector<double>& c, int) {
          return tsfile::p04::encode_sprintz_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p04::decode_sprintz_double(e, d);
        },
        columns, float_max_point_per_column, out);
  }
  if (!should_skip("RLBE", skip_algorithms)) {
    bench_one_double(
        "RLBE",
        [](const std::vector<double>& c, int) {
          return tsfile::p04::encode_rlbe_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p04::decode_rlbe_double(e, d);
        },
        columns, float_max_point_per_column, out);
  }
}

void bench_tsfile_rows_int64_accumulate(const std::vector<std::vector<std::int64_t>>& columns,
                                        BenchmarkTotals& totals,
                                        const std::unordered_set<std::string>* skip_algorithms) {
  if (!should_skip("TS_2DIFF", skip_algorithms)) {
    bench_one_i64_accumulate(
        "TS_2DIFF",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p02::encode_ts_2diff_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p02::decode_ts_2diff_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
  if (!should_skip("RLE", skip_algorithms)) {
    bench_one_i64_accumulate(
        "RLE",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p02::encode_rle_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p02::decode_rle_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
  if (!should_skip("GORILLA", skip_algorithms)) {
    bench_one_i64_accumulate(
        "GORILLA",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p03::encode_gorilla_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p03::decode_gorilla_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
  if (!should_skip("CHIMP", skip_algorithms)) {
    bench_one_i64_accumulate(
        "CHIMP",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p03::encode_chimp_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p03::decode_chimp_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
  if (!should_skip("CHIMP128", skip_algorithms)) {
    bench_one_i64_accumulate(
        "CHIMP128",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p03::encode_chimp_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p03::decode_chimp_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
  if (!should_skip("SPRINTZ", skip_algorithms)) {
    bench_one_i64_accumulate(
        "SPRINTZ",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p04::encode_sprintz_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p04::decode_sprintz_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
  if (!should_skip("RLBE", skip_algorithms)) {
    bench_one_i64_accumulate(
        "RLBE",
        [](const std::vector<std::int64_t>& c) {
          return tsfile::p04::encode_rlbe_int64(std::span<const std::int64_t>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<std::int64_t>& d) {
          return tsfile::p04::decode_rlbe_int64(e, d);
        },
        columns, totals, skip_algorithms);
  }
}

void bench_tsfile_rows_double_accumulate(
    const std::vector<std::vector<double>>& columns,
    const std::vector<int>& float_max_point_per_column,
    BenchmarkTotals& totals,
    const std::unordered_set<std::string>* skip_algorithms) {
  if (!should_skip("TS_2DIFF", skip_algorithms)) {
    bench_one_double_accumulate(
        "TS_2DIFF",
        [](const std::vector<double>& c, int mp) {
          int effective_mp = clamp_ts2diff_max_point_for_double(c, mp);
          return tsfile::p02::encode_ts_2diff_double(std::span<const double>(c.data(), c.size()),
                                                     effective_mp);
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p02::decode_ts_2diff_double(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
  if (!should_skip("RLE", skip_algorithms)) {
    bench_one_double_accumulate(
        "RLE",
        [](const std::vector<double>& c, int mp) {
          return tsfile::p02::encode_rle_double(std::span<const double>(c.data(), c.size()), mp);
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p02::decode_rle_double(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
  if (!should_skip("GORILLA", skip_algorithms)) {
    bench_one_double_accumulate(
        "GORILLA",
        [](const std::vector<double>& c, int) {
          return tsfile::p03::encode_gorilla_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p03::decode_gorilla_double(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
  if (!should_skip("CHIMP128", skip_algorithms)) {
    bench_one_double_accumulate(
        "CHIMP128",
        [](const std::vector<double>& c, int) {
          return tsfile::p03::encode_chimp_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p03::decode_chimp_double(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
  if (!should_skip(pongo_codec::kRowName, skip_algorithms)) {
    bench_one_double_accumulate(
        pongo_codec::kRowName,
        [](const std::vector<double>& c, int) {
          return pongo_codec::encode(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return pongo_codec::decode(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
  if (!should_skip("SPRINTZ", skip_algorithms)) {
    bench_one_double_accumulate(
        "SPRINTZ",
        [](const std::vector<double>& c, int) {
          return tsfile::p04::encode_sprintz_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p04::decode_sprintz_double(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
  if (!should_skip("RLBE", skip_algorithms)) {
    bench_one_double_accumulate(
        "RLBE",
        [](const std::vector<double>& c, int) {
          return tsfile::p04::encode_rlbe_double(std::span<const double>(c.data(), c.size()));
        },
        [](const std::vector<std::uint8_t>& e, std::vector<double>& d) {
          return tsfile::p04::decode_rlbe_double(e, d);
        },
        columns, float_max_point_per_column, totals, skip_algorithms);
  }
}
