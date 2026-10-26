#pragma once

#include "benchmark_totals.hpp"

#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

// Native TSFile six encoders + decoders (matching Java CompressionUtils.testEncoding totals).

void bench_tsfile_rows_int(const std::vector<std::vector<int>>& columns, std::ostream& out,
                           const std::unordered_set<std::string>* skip_algorithms = nullptr);
void bench_tsfile_rows_int64(const std::vector<std::vector<std::int64_t>>& columns, std::ostream& out,
                             const std::unordered_set<std::string>* skip_algorithms = nullptr);
/** Per-column fractional decimal digits for TS_2DIFF(float) and RLE(float) (same as Java MAX_POINT_NUMBER). */
void bench_tsfile_rows_float(const std::vector<std::vector<float>>& columns,
                             const std::vector<int>& float_max_point_per_column, std::ostream& out,
                             const std::unordered_set<std::string>* skip_algorithms = nullptr);
void bench_tsfile_rows_double(const std::vector<std::vector<double>>& columns,
                              const std::vector<int>& float_max_point_per_column, std::ostream& out,
                              const std::unordered_set<std::string>* skip_algorithms = nullptr);

void bench_tsfile_rows_int64_accumulate(const std::vector<std::vector<std::int64_t>>& columns,
                                        BenchmarkTotals& totals,
                                        const std::unordered_set<std::string>* skip_algorithms =
                                            nullptr);
void bench_tsfile_rows_double_accumulate(const std::vector<std::vector<double>>& columns,
                                         const std::vector<int>& float_max_point_per_column,
                                         BenchmarkTotals& totals,
                                         const std::unordered_set<std::string>* skip_algorithms =
                                             nullptr);
