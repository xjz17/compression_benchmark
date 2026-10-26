#pragma once

/**
 * SIMD column reductions used by the native BUFF kernel path (Intel AVX2 vs Apple Silicon NEON).
 * Full ``buff_simd256_encode`` from upstream Rust is not ported; these ops accelerate
 * validation/stats around the scalar ``encode_doubles`` pipeline.
 */

#include <cstddef>

namespace buff_kernel {

struct ColumnStats {
    double sum{0.0};
    double min_val{0.0};
    double max_val{0.0};
};

/** Horizontal sum + min + max over ``data[0..n)`` (``n`` may be 0). */
ColumnStats simd_column_stats(const double* data, std::size_t n);

/** Count indices where ``data[i] > threshold`` (SIMD where available). */
std::size_t simd_count_gt(const double* data, std::size_t n, double threshold);

}  // namespace buff_kernel
