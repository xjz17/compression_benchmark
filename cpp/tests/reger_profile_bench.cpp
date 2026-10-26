#include "reger_codec.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  constexpr std::size_t kRows = 200'000;
  constexpr std::size_t kColumns = 3;
  std::vector<std::int64_t> times;
  std::vector<std::vector<std::int64_t>> columns(kColumns);
  times.reserve(kRows);
  for (auto& column : columns) column.reserve(kRows);
  for (std::size_t i = 0; i < kRows; ++i) {
    times.push_back(1'700'000'000'000LL + static_cast<std::int64_t>(i) * 1'000LL);
    columns[0].push_back(10'000 + static_cast<std::int64_t>(i) * 3 + i % 5);
    columns[1].push_back(-50'000 + static_cast<std::int64_t>(i) * 7 + i % 3);
    columns[2].push_back(800 + static_cast<std::int64_t>(i / 16));
  }

  for (const char* profile : {"balanced", "fast"}) {
    setenv("COMPRESSION_BENCHMARK_REGER_PROFILE", profile, 1);
    unsetenv("COMPRESSION_BENCHMARK_REGER_REORDER_ITERS");
    auto encoded = reger_codec::encodeInt64Columns(columns, &times);
    assert(reger_codec::decodeInt64Columns(encoded) == columns);

    long long best_ns = std::numeric_limits<long long>::max();
    for (int run = 0; run < 5; ++run) {
      const auto start = std::chrono::steady_clock::now();
      encoded = reger_codec::encodeInt64Columns(columns, &times);
      const auto end = std::chrono::steady_clock::now();
      best_ns = std::min(
          best_ns,
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    const double points = static_cast<double>(kRows * kColumns);
    const double original_bytes = points * sizeof(std::int64_t);
    std::cout << profile << ", ns/point=" << std::fixed << std::setprecision(1)
              << best_ns / points << ", ratio=" << std::setprecision(3)
              << original_bytes / encoded.size() << ", bytes=" << encoded.size() << '\n';
  }
}
