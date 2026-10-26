#include "leco_double_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"
#include "int64_column_codec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace leco_double {
namespace {

constexpr int kMaxDecimalScale = 15;

int min_decimal_scale_exact_double(double v) {
    if (!std::isfinite(v)) {
        return -1;
    }
    if (v == 0.0) {
        return 0;
    }
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    for (int s = 0; s <= kMaxDecimalScale; s++) {
        double pow10 = std::pow(10.0, s);
        long long m = static_cast<long long>(std::llround(v * pow10));
        if (m < std::numeric_limits<std::int64_t>::min() ||
            m > std::numeric_limits<std::int64_t>::max()) {
            return -1;
        }
        auto im = static_cast<std::int64_t>(m);
        double r = static_cast<double>(im) / pow10;
        std::uint64_t rb = 0;
        std::memcpy(&rb, &r, 8);
        if (rb == bits) {
            return s;
        }
    }
    return -1;
}

} // namespace

int maxLosslessDecimalScale(const std::vector<double> &col) {
    int col_scale = 0;
    for (double v : col) {
        int s = min_decimal_scale_exact_double(v);
        if (s < 0) {
            return -1;
        }
        if (s > col_scale) {
            col_scale = s;
        }
    }
    return col_scale;
}

bool quantizeScaledInts(const std::vector<double> &col, int p, std::vector<std::int64_t> &out) {
    const int n = static_cast<int>(col.size());
    out.resize(static_cast<std::size_t>(n));
    double pow10 = std::pow(10.0, p);
    for (int i = 0; i < n; i++) {
        double v = col[static_cast<std::size_t>(i)];
        std::uint64_t vb = 0;
        std::memcpy(&vb, &v, 8);
        if (v == 0.0) {
            out[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(vb);
            continue;
        }
        const long double scaled =
            static_cast<long double>(v) * static_cast<long double>(pow10);
        if (!std::isfinite(scaled) ||
            scaled < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
            scaled > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
            return false;
        }
        const std::int64_t im = static_cast<std::int64_t>(std::llround(scaled));
        double r = static_cast<double>(im) / pow10;
        std::uint64_t rb = 0;
        std::memcpy(&rb, &r, 8);
        if (rb != vb) {
            return false;
        }
        out[static_cast<std::size_t>(i)] = im;
    }
    return true;
}

bool quantizeScaledIntsWithMaxPointFallback(const std::vector<double> &col,
                                            int max_point,
                                            std::vector<std::int64_t> &out,
                                            int *used_scale) {
    const int start_scale = std::min(kMaxDecimalScale, std::max(0, max_point));
    for (int p = start_scale; p >= 0; --p) {
        if (quantizeScaledInts(col, p, out)) {
            if (used_scale != nullptr) {
                *used_scale = p;
            }
            return true;
        }
    }
    return false;
}

void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                std::vector<long long> &result,
                                const std::vector<int> *max_point_per_column, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (std::size_t ci = 0; ci < columns.size(); ++ci) {
        const auto &col = columns[ci];
        const int n = static_cast<int>(col.size());
        if (n == 0) {
            continue;
        }
        std::vector<std::int64_t> data;
        data.resize(col.size());
        for (std::size_t i = 0; i < col.size(); ++i) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &col[i], sizeof(bits));
            data[i] = static_cast<std::int64_t>(bits);
        }
        if (data.empty()) {
            continue;
        }
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            const auto payload = int64_column_codec::fast_pfor_encode_int64(data.data(), data.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            auto t2 = std::chrono::high_resolution_clock::now();
            const auto recovered =
                    int64_column_codec::fast_pfor_decode_int64(payload.data(), payload.size(), data.size());
            auto t3 = std::chrono::high_resolution_clock::now();
            if (recovered != data) {
                continue;
            }
            if (loss != nullptr) {
                std::vector<double> dec_d(recovered.size());
                for (std::size_t i = 0; i < recovered.size(); ++i) {
                    const std::uint64_t bits = static_cast<std::uint64_t>(recovered[i]);
                    std::memcpy(&dec_d[i], &bits, sizeof(bits));
                }
                benchmark_loss::accumulate_double_column(col, dec_d, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(payload.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace leco_double
