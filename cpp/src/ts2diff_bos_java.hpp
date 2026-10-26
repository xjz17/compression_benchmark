#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ts2diff_bos_java {

/** Aggregated bench like other column codecs; spawns Java when JAR exists (POSIX). */
void benchIntColumnsSubprocess(const std::vector<std::vector<int>> &columns, std::vector<long long> &out);
/** INT64 columns via ``Ts2DiffBosLongBatchMain``; floored by native TS_2DIFF INT64 when smaller. */
void benchInt64ColumnsSubprocess(const std::vector<std::vector<std::int64_t>> &columns,
                                 std::vector<long long> &out);

/** Float columns → scaled int32 in JVM, then TS_2DIFF+BOS; ``Ts2DiffBosFloatBatchMain``. */
void benchFloatColumnsSubprocess(const std::vector<std::vector<float>> &columns,
                                 std::vector<long long> &out);
/** DOUBLE columns via ``Ts2DiffBosDoubleBatchMain``; floored by native TS_2DIFF DOUBLE when smaller. */
void benchDoubleColumnsSubprocess(const std::vector<std::vector<double>> &columns,
                                  const std::vector<int> *max_point_per_column,
                                  std::vector<long long> &out);

} // namespace ts2diff_bos_java
