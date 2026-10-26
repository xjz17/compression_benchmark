#pragma once

#include "benchmark_totals.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace wc_lzbench_bench {

bool bench_enabled();

/** Run lzbench on one patch payload; merge rows into ``totals``. */
void accumulatePatchBytes(const std::vector<char> &payload, BenchmarkTotals &totals,
                          const std::unordered_set<std::string> *skip_algorithms = nullptr);

void accumulateByteColumns(const std::vector<std::vector<char>> &column_byte_data,
                           BenchmarkTotals &totals,
                           const std::unordered_set<std::string> *skip_algorithms = nullptr);

}  // namespace wc_lzbench_bench
