#include "most_float_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <zstd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace most_float {

static constexpr int kSeg = 128;
/** Relative error bound for lossy reconstruction check (MOST-inspired baseline; not full mostdb). */
static constexpr float kRelError = 1e-3f;
static constexpr double kRelErrorDouble = 1e-3;

static void ols_ab(const float *v, int m, float &a, float &b) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < m; i++) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(v[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double dm = static_cast<double>(m);
    const double den = dm * sxx - sx * sx;
    if (std::fabs(den) < 1e-30) {
        b = 0.f;
        a = m > 0 ? v[0] : 0.f;
        return;
    }
    b = static_cast<float>((dm * sxy - sx * sy) / den);
    a = static_cast<float>((sy - static_cast<double>(b) * sx) / dm);
}

static void ols_ab_double(const double *v, int m, double &a, double &b) {
    long double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < m; i++) {
        const long double x = static_cast<long double>(i);
        const long double y = static_cast<long double>(v[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const long double dm = static_cast<long double>(m);
    const long double den = dm * sxx - sx * sx;
    if (std::fabs(static_cast<double>(den)) < 1e-30) {
        b = 0.0;
        a = m > 0 ? v[0] : 0.0;
        return;
    }
    b = static_cast<double>((dm * sxy - sx * sy) / den);
    a = static_cast<double>((sy - static_cast<long double>(b) * sx) / dm);
}

static void ols_ab_int64(const std::int64_t *v, int m, double &a, double &b) {
    long double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < m; i++) {
        const long double x = static_cast<long double>(i);
        const long double y = static_cast<long double>(v[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const long double dm = static_cast<long double>(m);
    const long double den = dm * sxx - sx * sx;
    if (std::fabs(static_cast<double>(den)) < 1e-30) {
        b = 0.0;
        a = m > 0 ? static_cast<double>(v[0]) : 0.0;
        return;
    }
    b = static_cast<double>((dm * sxy - sx * sy) / den);
    a = static_cast<double>((sy - static_cast<long double>(b) * sx) / dm);
}

static bool packColumnLossy(const std::vector<float> &col, std::vector<std::uint8_t> &raw) {
    const std::size_t n = col.size();
    raw.clear();
    raw.reserve(n * 2 + (n / kSeg + 2) * 32);
    float colMax = 0.f;
    for (float v : col) {
        colMax = std::max(colMax, std::fabs(v));
    }
    if (colMax == 0.f) {
        colMax = 1.f;
    }

    for (std::size_t off = 0; off < n; off += static_cast<std::size_t>(kSeg)) {
        const int m = static_cast<int>(std::min<std::size_t>(kSeg, n - off));
        float a = 0.f, b = 0.f;
        ols_ab(col.data() + off, m, a, b);

        float resMax = 0.f;
        std::vector<float> res(static_cast<std::size_t>(m));
        for (int i = 0; i < m; i++) {
            const float pred = a + b * static_cast<float>(i);
            res[static_cast<std::size_t>(i)] = col[off + static_cast<std::size_t>(i)] - pred;
            resMax = std::max(resMax, std::fabs(res[static_cast<std::size_t>(i)]));
        }
        float scale = resMax > 0.f ? resMax * kRelError : 1.f;
        const float inv = 1.f / scale;

        const std::uint32_t mu = static_cast<std::uint32_t>(m);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&mu),
                   reinterpret_cast<const std::uint8_t *>(&mu) + 4);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&a),
                   reinterpret_cast<const std::uint8_t *>(&a) + 4);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&b),
                   reinterpret_cast<const std::uint8_t *>(&b) + 4);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&scale),
                   reinterpret_cast<const std::uint8_t *>(&scale) + 4);

        for (int i = 0; i < m; i++) {
            double qd = std::llround(static_cast<double>(res[static_cast<std::size_t>(i)]) *
                                     static_cast<double>(inv));
            if (qd > 32767.0) {
                qd = 32767.0;
            }
            if (qd < -32768.0) {
                qd = -32768.0;
            }
            const std::int16_t q = static_cast<std::int16_t>(qd);
            raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&q),
                       reinterpret_cast<const std::uint8_t *>(&q) + 2);
        }
    }
    return true;
}

static bool verifyColumnLossy(const std::vector<float> &col, const std::uint8_t *raw,
                              std::size_t rawLen) {
    const std::size_t n = col.size();
    float colMax = 0.f;
    for (float v : col) {
        colMax = std::max(colMax, std::fabs(v));
    }
    if (colMax == 0.f) {
        colMax = 1.f;
    }

    std::size_t p = 0;
    std::size_t off = 0;
    while (off < n) {
        if (p + 16 > rawLen) {
            return false;
        }
        std::uint32_t m = 0;
        float a = 0.f, b = 0.f, scale = 0.f;
        std::memcpy(&m, raw + p, 4);
        p += 4;
        std::memcpy(&a, raw + p, 4);
        p += 4;
        std::memcpy(&b, raw + p, 4);
        p += 4;
        std::memcpy(&scale, raw + p, 4);
        p += 4;
        if (m > static_cast<std::uint32_t>(kSeg) || off + m > n) {
            return false;
        }
        const std::size_t need = p + static_cast<std::size_t>(m) * 2;
        if (need > rawLen) {
            return false;
        }
        for (std::uint32_t i = 0; i < m; i++) {
            std::int16_t q = 0;
            std::memcpy(&q, raw + p, 2);
            p += 2;
            const float dec =
                a + b * static_cast<float>(i) + static_cast<float>(q) * scale;
            const float err = std::fabs(dec - col[off + i]);
            const float boundAbs =
                kRelError * std::max(std::fabs(col[off + i]), colMax);
            if (err > boundAbs * 1.0001f + 1e-6f) {
                return false;
            }
        }
        off += m;
    }
    return p == rawLen && off == n;
}

static bool packColumnLossyDouble(const std::vector<double> &col, std::vector<std::uint8_t> &raw) {
    const std::size_t n = col.size();
    raw.clear();
    raw.reserve(n * 2 + (n / kSeg + 2) * 48);
    double colMax = 0.0;
    for (double v : col) {
        colMax = std::max(colMax, std::fabs(v));
    }
    if (colMax == 0.0) {
        colMax = 1.0;
    }

    for (std::size_t off = 0; off < n; off += static_cast<std::size_t>(kSeg)) {
        const int m = static_cast<int>(std::min<std::size_t>(kSeg, n - off));
        double a = 0.0, b = 0.0;
        ols_ab_double(col.data() + off, m, a, b);

        double resMax = 0.0;
        std::vector<double> res(static_cast<std::size_t>(m));
        for (int i = 0; i < m; i++) {
            const double pred = a + b * static_cast<double>(i);
            res[static_cast<std::size_t>(i)] = col[off + static_cast<std::size_t>(i)] - pred;
            resMax = std::max(resMax, std::fabs(res[static_cast<std::size_t>(i)]));
        }
        const double scale = resMax > 0.0 ? resMax * kRelErrorDouble : 1.0;
        const double inv = 1.0 / scale;

        const std::uint32_t mu = static_cast<std::uint32_t>(m);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&mu),
                   reinterpret_cast<const std::uint8_t *>(&mu) + 4);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&a),
                   reinterpret_cast<const std::uint8_t *>(&a) + 8);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&b),
                   reinterpret_cast<const std::uint8_t *>(&b) + 8);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&scale),
                   reinterpret_cast<const std::uint8_t *>(&scale) + 8);

        for (int i = 0; i < m; i++) {
            double qd = std::llround(res[static_cast<std::size_t>(i)] * inv);
            if (qd > 32767.0) {
                qd = 32767.0;
            }
            if (qd < -32768.0) {
                qd = -32768.0;
            }
            const std::int16_t q = static_cast<std::int16_t>(qd);
            raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&q),
                       reinterpret_cast<const std::uint8_t *>(&q) + 2);
        }
    }
    return true;
}

static bool verifyColumnLossyDouble(const std::vector<double> &col, const std::uint8_t *raw,
                                    std::size_t rawLen) {
    const std::size_t n = col.size();
    double colMax = 0.0;
    for (double v : col) {
        colMax = std::max(colMax, std::fabs(v));
    }
    if (colMax == 0.0) {
        colMax = 1.0;
    }

    std::size_t p = 0;
    std::size_t off = 0;
    while (off < n) {
        if (p + 28 > rawLen) {
            return false;
        }
        std::uint32_t m = 0;
        double a = 0.0, b = 0.0, scale = 0.0;
        std::memcpy(&m, raw + p, 4);
        p += 4;
        std::memcpy(&a, raw + p, 8);
        p += 8;
        std::memcpy(&b, raw + p, 8);
        p += 8;
        std::memcpy(&scale, raw + p, 8);
        p += 8;
        if (m > static_cast<std::uint32_t>(kSeg) || off + m > n) {
            return false;
        }
        const std::size_t need = p + static_cast<std::size_t>(m) * 2;
        if (need > rawLen) {
            return false;
        }
        for (std::uint32_t i = 0; i < m; i++) {
            std::int16_t q = 0;
            std::memcpy(&q, raw + p, 2);
            p += 2;
            const double dec = a + b * static_cast<double>(i) + static_cast<double>(q) * scale;
            const double err = std::fabs(dec - col[off + i]);
            const double boundAbs = kRelErrorDouble * std::max(std::fabs(col[off + i]), colMax);
            if (err > boundAbs * 1.0001 + 1e-9) {
                return false;
            }
        }
        off += m;
    }
    return p == rawLen && off == n;
}

static void accumulateColumnLossyDouble(const std::vector<double> &col, const std::uint8_t *raw,
                                        std::size_t rawLen, LossAccum &acc) {
    const std::size_t n = col.size();
    std::vector<double> decoded(n);
    std::size_t p = 0;
    std::size_t off = 0;
    while (off < n) {
        if (p + 28 > rawLen) {
            return;
        }
        std::uint32_t m = 0;
        double a = 0.0, b = 0.0, scale = 0.0;
        std::memcpy(&m, raw + p, 4);
        p += 4;
        std::memcpy(&a, raw + p, 8);
        p += 8;
        std::memcpy(&b, raw + p, 8);
        p += 8;
        std::memcpy(&scale, raw + p, 8);
        p += 8;
        if (m > static_cast<std::uint32_t>(kSeg) || off + m > n) {
            return;
        }
        const std::size_t need = p + static_cast<std::size_t>(m) * 2;
        if (need > rawLen) {
            return;
        }
        for (std::uint32_t i = 0; i < m; i++) {
            std::int16_t q = 0;
            std::memcpy(&q, raw + p, 2);
            p += 2;
            decoded[off + i] = a + b * static_cast<double>(i) + static_cast<double>(q) * scale;
        }
        off += m;
    }
    if (p != rawLen || off != n) {
        return;
    }
    benchmark_loss::accumulate_double_column(col, decoded, acc);
}

static void accumulateColumnLossyFloat(const std::vector<float> &col, const std::uint8_t *raw,
                                     std::size_t rawLen, LossAccum &acc) {
    std::vector<double> orig(col.begin(), col.end());
    const std::size_t n = col.size();
    std::vector<double> decoded(n);
    std::size_t p = 0;
    std::size_t off = 0;
    while (off < n) {
        if (p + 16 > rawLen) {
            return;
        }
        std::uint32_t m = 0;
        float a = 0.f, b = 0.f, scale = 0.f;
        std::memcpy(&m, raw + p, 4);
        p += 4;
        std::memcpy(&a, raw + p, 4);
        p += 4;
        std::memcpy(&b, raw + p, 4);
        p += 4;
        std::memcpy(&scale, raw + p, 4);
        p += 4;
        if (m > static_cast<std::uint32_t>(kSeg) || off + m > n) {
            return;
        }
        const std::size_t need = p + static_cast<std::size_t>(m) * 2;
        if (need > rawLen) {
            return;
        }
        for (std::uint32_t i = 0; i < m; i++) {
            std::int16_t q = 0;
            std::memcpy(&q, raw + p, 2);
            p += 2;
            decoded[off + i] =
                static_cast<double>(a + b * static_cast<float>(i) + static_cast<float>(q) * scale);
        }
        off += m;
    }
    if (p != rawLen || off != n) {
        return;
    }
    benchmark_loss::accumulate_double_column(orig, decoded, acc);
}

static bool packColumnLossyInt64(const std::vector<std::int64_t> &col, std::vector<std::uint8_t> &raw) {
    const std::size_t n = col.size();
    raw.clear();
    raw.reserve(n * 2 + (n / kSeg + 2) * 48);
    long double colMax = 0.0;
    for (std::int64_t v : col) {
        colMax = std::max(colMax, std::fabs(static_cast<long double>(v)));
    }
    if (colMax == 0.0) {
        colMax = 1.0;
    }

    for (std::size_t off = 0; off < n; off += static_cast<std::size_t>(kSeg)) {
        const int m = static_cast<int>(std::min<std::size_t>(kSeg, n - off));
        double a = 0.0, b = 0.0;
        ols_ab_int64(col.data() + off, m, a, b);

        long double resMax = 0.0;
        std::vector<long double> res(static_cast<std::size_t>(m));
        for (int i = 0; i < m; i++) {
            const long double pred =
                static_cast<long double>(a) + static_cast<long double>(b) * static_cast<long double>(i);
            res[static_cast<std::size_t>(i)] =
                static_cast<long double>(col[off + static_cast<std::size_t>(i)]) - pred;
            resMax = std::max(resMax, std::fabs(res[static_cast<std::size_t>(i)]));
        }
        const long double scale_ld = resMax > 0.0 ? resMax * static_cast<long double>(kRelErrorDouble) : 1.0;
        const long double inv = 1.0L / scale_ld;
        const double scale = static_cast<double>(scale_ld);

        const std::uint32_t mu = static_cast<std::uint32_t>(m);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&mu),
                   reinterpret_cast<const std::uint8_t *>(&mu) + 4);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&a),
                   reinterpret_cast<const std::uint8_t *>(&a) + 8);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&b),
                   reinterpret_cast<const std::uint8_t *>(&b) + 8);
        raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&scale),
                   reinterpret_cast<const std::uint8_t *>(&scale) + 8);

        for (int i = 0; i < m; i++) {
            long double qd = std::llround(res[static_cast<std::size_t>(i)] * inv);
            if (qd > 32767.0) {
                qd = 32767.0;
            }
            if (qd < -32768.0) {
                qd = -32768.0;
            }
            const std::int16_t q = static_cast<std::int16_t>(qd);
            raw.insert(raw.end(), reinterpret_cast<const std::uint8_t *>(&q),
                       reinterpret_cast<const std::uint8_t *>(&q) + 2);
        }
    }
    return true;
}

static bool verifyColumnLossyInt64(const std::vector<std::int64_t> &col, const std::uint8_t *raw,
                                   std::size_t rawLen) {
    const std::size_t n = col.size();
    long double colMax = 0.0;
    for (std::int64_t v : col) {
        colMax = std::max(colMax, std::fabs(static_cast<long double>(v)));
    }
    if (colMax == 0.0) {
        colMax = 1.0;
    }

    std::size_t p = 0;
    std::size_t off = 0;
    while (off < n) {
        if (p + 28 > rawLen) {
            return false;
        }
        std::uint32_t m = 0;
        double a = 0.0, b = 0.0, scale = 0.0;
        std::memcpy(&m, raw + p, 4);
        p += 4;
        std::memcpy(&a, raw + p, 8);
        p += 8;
        std::memcpy(&b, raw + p, 8);
        p += 8;
        std::memcpy(&scale, raw + p, 8);
        p += 8;
        if (m > static_cast<std::uint32_t>(kSeg) || off + m > n) {
            return false;
        }
        const std::size_t need = p + static_cast<std::size_t>(m) * 2;
        if (need > rawLen) {
            return false;
        }
        for (std::uint32_t i = 0; i < m; i++) {
            std::int16_t q = 0;
            std::memcpy(&q, raw + p, 2);
            p += 2;
            const long double dec = static_cast<long double>(a) + static_cast<long double>(b) * i +
                                    static_cast<long double>(q) * static_cast<long double>(scale);
            const long double orig = static_cast<long double>(col[off + i]);
            const long double err = std::fabs(dec - orig);
            const long double boundAbs = static_cast<long double>(kRelErrorDouble) * std::max(std::fabs(orig), colMax);
            if (err > boundAbs * 1.0001L + 1e-6L) {
                return false;
            }
        }
        off += m;
    }
    return p == rawLen && off == n;
}

void benchFloatColumnsMostMinimal(const std::vector<std::vector<float>> &columns,
                                  std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        std::vector<std::uint8_t> raw;
        if (!packColumnLossy(col, raw)) {
            continue;
        }

        const std::size_t bound = ZSTD_compressBound(raw.size());
        std::vector<std::uint8_t> cbuf(bound);
        auto t0 = std::chrono::high_resolution_clock::now();
        const std::size_t csize =
            ZSTD_compress(cbuf.data(), cbuf.size(), raw.data(), raw.size(), 3);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(csize) != 0) {
            continue;
        }
        cbuf.resize(csize);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> dbuf(raw.size());
        const std::size_t dsize =
            ZSTD_decompress(dbuf.data(), dbuf.size(), cbuf.data(), cbuf.size());
        auto t3 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(dsize) != 0 || dsize != raw.size()) {
            continue;
        }
        if (std::memcmp(dbuf.data(), raw.data(), raw.size()) != 0) {
            continue;
        }
        if (!verifyColumnLossy(col, dbuf.data(), dbuf.size())) {
            continue;
        }
        if (loss != nullptr) {
            accumulateColumnLossyFloat(col, dbuf.data(), dbuf.size(), *loss);
        }

        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        comp += static_cast<long long>(csize);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchDoubleColumnsMostMinimal(const std::vector<std::vector<double>> &columns,
                                   std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        std::vector<std::uint8_t> raw;
        if (!packColumnLossyDouble(col, raw)) {
            continue;
        }

        const std::size_t bound = ZSTD_compressBound(raw.size());
        std::vector<std::uint8_t> cbuf(bound);
        auto t0 = std::chrono::high_resolution_clock::now();
        const std::size_t csize =
            ZSTD_compress(cbuf.data(), cbuf.size(), raw.data(), raw.size(), 3);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(csize) != 0) {
            continue;
        }
        cbuf.resize(csize);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> dbuf(raw.size());
        const std::size_t dsize =
            ZSTD_decompress(dbuf.data(), dbuf.size(), cbuf.data(), cbuf.size());
        auto t3 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(dsize) != 0 || dsize != raw.size()) {
            continue;
        }
        if (std::memcmp(dbuf.data(), raw.data(), raw.size()) != 0) {
            continue;
        }
        if (!verifyColumnLossyDouble(col, dbuf.data(), dbuf.size())) {
            continue;
        }
        if (loss != nullptr) {
            accumulateColumnLossyDouble(col, dbuf.data(), dbuf.size(), *loss);
        }

        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        comp += static_cast<long long>(csize);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchInt64ColumnsMostMinimal(const std::vector<std::vector<std::int64_t>> &columns,
                                  std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        std::vector<std::uint8_t> raw;
        if (!packColumnLossyInt64(col, raw)) {
            continue;
        }

        const std::size_t bound = ZSTD_compressBound(raw.size());
        std::vector<std::uint8_t> cbuf(bound);
        auto t0 = std::chrono::high_resolution_clock::now();
        const std::size_t csize =
            ZSTD_compress(cbuf.data(), cbuf.size(), raw.data(), raw.size(), 3);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(csize) != 0) {
            continue;
        }
        cbuf.resize(csize);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> dbuf(raw.size());
        const std::size_t dsize =
            ZSTD_decompress(dbuf.data(), dbuf.size(), cbuf.data(), cbuf.size());
        auto t3 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(dsize) != 0 || dsize != raw.size()) {
            continue;
        }
        if (std::memcmp(dbuf.data(), raw.data(), raw.size()) != 0) {
            continue;
        }
        if (!verifyColumnLossyInt64(col, dbuf.data(), dbuf.size())) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<double> orig_vals(col.begin(), col.end());
            accumulateColumnLossyDouble(orig_vals, dbuf.data(), dbuf.size(), *loss);
        }

        orig += static_cast<long long>(benchmark_wire::column_original_size(n, false));
        comp += static_cast<long long>(csize);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace most_float
