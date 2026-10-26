#include "buff_simd_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if (defined(__x86_64__) || defined(_M_AMD64)) && (defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)))
#include <immintrin.h>
#define BUFF_KERNEL_HAS_AVX2 1
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define BUFF_KERNEL_HAS_NEON 1
#endif

namespace buff_kernel {

namespace {

ColumnStats scalar_stats(const double* data, std::size_t n) {
    ColumnStats s{};
    if (n == 0) {
        return s;
    }
    s.min_val = data[0];
    s.max_val = data[0];
    for (std::size_t i = 0; i < n; ++i) {
        double v = data[i];
        s.sum += v;
        s.min_val = std::min(s.min_val, v);
        s.max_val = std::max(s.max_val, v);
    }
    return s;
}

static int popcount4(unsigned m) {
#if defined(_MSC_VER)
    return __popcnt(m & 0xF);
#else
    return __builtin_popcount(m & 0xF);
#endif
}

#if defined(BUFF_KERNEL_HAS_AVX2)

ColumnStats avx2_stats(const double* data, std::size_t n) {
    ColumnStats s{};
    if (n == 0) {
        return s;
    }
    if (n < 4) {
        return scalar_stats(data, n);
    }
    __m256d vmin = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d vmax = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    __m256d vsum = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vmin = _mm256_min_pd(vmin, v);
        vmax = _mm256_max_pd(vmax, v);
        vsum = _mm256_add_pd(vsum, v);
    }
    alignas(32) double bmin[4], bmax[4], bsum[4];
    _mm256_storeu_pd(bmin, vmin);
    _mm256_storeu_pd(bmax, vmax);
    _mm256_storeu_pd(bsum, vsum);
    s.min_val = std::min({bmin[0], bmin[1], bmin[2], bmin[3]});
    s.max_val = std::max({bmax[0], bmax[1], bmax[2], bmax[3]});
    s.sum = bsum[0] + bsum[1] + bsum[2] + bsum[3];
    for (; i < n; ++i) {
        double v = data[i];
        s.sum += v;
        s.min_val = std::min(s.min_val, v);
        s.max_val = std::max(s.max_val, v);
    }
    return s;
}

std::size_t avx2_count_gt(const double* data, std::size_t n, double threshold) {
    if (n == 0) {
        return 0;
    }
    __m256d thr = _mm256_set1_pd(threshold);
    std::size_t c = 0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        __m256d gt = _mm256_cmp_pd(v, thr, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(gt);
        c += static_cast<std::size_t>(popcount4(static_cast<unsigned>(mask)));
    }
    for (; i < n; ++i) {
        if (data[i] > threshold) {
            ++c;
        }
    }
    return c;
}

#endif

#if defined(BUFF_KERNEL_HAS_NEON)

ColumnStats neon_stats(const double* data, std::size_t n) {
    ColumnStats s{};
    if (n == 0) {
        return s;
    }
    if (n < 2) {
        s.sum = data[0];
        s.min_val = s.max_val = data[0];
        return s;
    }
    float64x2_t vmin = vdupq_n_f64(std::numeric_limits<double>::infinity());
    float64x2_t vmax = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    float64x2_t vsum = vdupq_n_f64(0.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        vmin = vminq_f64(vmin, v);
        vmax = vmaxq_f64(vmax, v);
        vsum = vaddq_f64(vsum, v);
    }
    double mn[2], mx[2], sm[2];
    vst1q_f64(mn, vmin);
    vst1q_f64(mx, vmax);
    vst1q_f64(sm, vsum);
    s.min_val = std::min(mn[0], mn[1]);
    s.max_val = std::max(mx[0], mx[1]);
    s.sum = sm[0] + sm[1];
    for (; i < n; ++i) {
        double v = data[i];
        s.sum += v;
        s.min_val = std::min(s.min_val, v);
        s.max_val = std::max(s.max_val, v);
    }
    return s;
}

std::size_t neon_count_gt(const double* data, std::size_t n, double threshold) {
    if (n == 0) {
        return 0;
    }
    float64x2_t thr = vdupq_n_f64(threshold);
    std::size_t c = 0;
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t gt = vcgtq_f64(v, thr);
        uint64_t lo = vgetq_lane_u64(gt, 0);
        uint64_t hi = vgetq_lane_u64(gt, 1);
        if (lo != 0) {
            ++c;
        }
        if (hi != 0) {
            ++c;
        }
    }
    for (; i < n; ++i) {
        if (data[i] > threshold) {
            ++c;
        }
    }
    return c;
}

#endif

std::size_t scalar_count_gt(const double* data, std::size_t n, double threshold) {
    std::size_t cc = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] > threshold) {
            ++cc;
        }
    }
    return cc;
}

}  // namespace

ColumnStats simd_column_stats(const double* data, std::size_t n) {
#if defined(BUFF_KERNEL_HAS_AVX2)
    return avx2_stats(data, n);
#elif defined(BUFF_KERNEL_HAS_NEON)
    return neon_stats(data, n);
#else
    return scalar_stats(data, n);
#endif
}

std::size_t simd_count_gt(const double* data, std::size_t n, double threshold) {
#if defined(BUFF_KERNEL_HAS_AVX2)
    return avx2_count_gt(data, n, threshold);
#elif defined(BUFF_KERNEL_HAS_NEON)
    return neon_count_gt(data, n, threshold);
#else
    return scalar_count_gt(data, n, threshold);
#endif
}

}  // namespace buff_kernel
