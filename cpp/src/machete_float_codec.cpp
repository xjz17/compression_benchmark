#include "machete_float_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <zstd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace machete_float {

template <typename T, bool IsFloatInput>
void benchColumnsMacheteRelImpl(const std::vector<std::vector<T>> &columns, double relError,
                                std::vector<long long> &result, LossAccum *loss) {
    if (!(relError > 0.0) || relError >= 1.0) {
        result.assign({0, 0, 0, 0});
        return;
    }
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        double mx = 0.0;
        for (T v : col) {
            mx = std::max(mx, std::fabs(static_cast<double>(v)));
        }
        if (mx == 0.0) {
            mx = 1.0;
        }
        const double scale = mx * relError;
        if (!(scale > 0.0)) {
            continue;
        }
        const double inv = 1.0 / scale;
        std::vector<std::int16_t> q(n);
        for (std::size_t i = 0; i < n; i++) {
            double d = std::llround(static_cast<double>(col[i]) * inv);
            if (d > 32767.0) {
                d = 32767.0;
            }
            if (d < -32768.0) {
                d = -32768.0;
            }
            q[i] = static_cast<std::int16_t>(d);
        }
        const std::size_t rawBytes = 8 + 4 + n * 2;
        std::vector<std::uint8_t> raw(rawBytes);
        std::memcpy(raw.data(), &scale, 8);
        const std::uint32_t nu = static_cast<std::uint32_t>(n);
        std::memcpy(raw.data() + 8, &nu, 4);
        std::memcpy(raw.data() + 12, q.data(), n * 2);

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
        const std::size_t dsize = ZSTD_decompress(dbuf.data(), dbuf.size(), cbuf.data(), cbuf.size());
        auto t3 = std::chrono::high_resolution_clock::now();
        if (ZSTD_isError(dsize) != 0 || dsize != raw.size()) {
            continue;
        }
        double s2 = 0.0;
        std::uint32_t n2 = 0;
        std::memcpy(&s2, dbuf.data(), 8);
        std::memcpy(&n2, dbuf.data() + 8, 4);
        if (n2 != nu || s2 != scale) {
            continue;
        }
        std::vector<std::int16_t> q2(n);
        std::memcpy(q2.data(), dbuf.data() + 12, n * 2);
        bool ok = true;
        for (std::size_t i = 0; i < n; i++) {
            const double dec = static_cast<double>(q2[i]) * s2;
            const double origV = static_cast<double>(col[i]);
            const double err = std::fabs(dec - origV);
            double boundAbs = relError * std::max(std::fabs(origV), mx);
            double eps = IsFloatInput ? 1e-6 : 1.0;
            if (!IsFloatInput) {
                boundAbs = std::max(boundAbs, 1.0);
            }
            if (err > boundAbs * 1.0001 + eps) {
                ok = false;
                break;
            }
            if (loss != nullptr) {
                loss->err += err;
                loss->ref += std::fabs(origV);
            }
        }
        if (!ok) {
            continue;
        }
        orig += static_cast<long long>(benchmark_wire::column_original_size(n, IsFloatInput));
        comp += static_cast<long long>(csize);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchFloatColumnsMacheteRel(const std::vector<std::vector<float>> &columns, double relError,
                                 std::vector<long long> &result, LossAccum *loss) {
    benchColumnsMacheteRelImpl<float, true>(columns, relError, result, loss);
}

void benchDoubleColumnsMacheteRel(const std::vector<std::vector<double>> &columns, double relError,
                                  std::vector<long long> &result, LossAccum *loss) {
    benchColumnsMacheteRelImpl<double, true>(columns, relError, result, loss);
}

void benchInt64ColumnsMacheteRel(const std::vector<std::vector<std::int64_t>> &columns, double relError,
                                 std::vector<long long> &result, LossAccum *loss) {
    benchColumnsMacheteRelImpl<std::int64_t, false>(columns, relError, result, loss);
}

} // namespace machete_float
