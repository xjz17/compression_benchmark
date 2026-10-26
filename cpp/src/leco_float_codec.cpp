#include "leco_float_codec.hpp"
#include "benchmark_wire.hpp"

#include "java_fastpfor_codec.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace leco_float {
namespace {

constexpr int kMaxDecimalScale = 15;

int min_decimal_scale_exact_float(float v) {
    if (!std::isfinite(static_cast<double>(v))) {
        return -1;
    }
    if (v == 0.0f) {
        return 0;
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    for (int s = 0; s <= kMaxDecimalScale; s++) {
        double pow10 = std::pow(10.0, s);
        long long m = static_cast<long long>(std::llround(static_cast<double>(v) * pow10));
        if (m < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
            m > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
            return -1;
        }
        auto im = static_cast<std::int32_t>(m);
        double r_d = static_cast<double>(im) / pow10;
        float r = static_cast<float>(r_d);
        std::uint32_t rb = 0;
        std::memcpy(&rb, &r, 4);
        if (rb == bits) {
            return s;
        }
    }
    return -1;
}

} // namespace

int maxLosslessDecimalScale(const std::vector<float> &col) {
    int col_scale = 0;
    for (float v : col) {
        int s = min_decimal_scale_exact_float(v);
        if (s < 0) {
            return -1;
        }
        if (s > col_scale) {
            col_scale = s;
        }
    }
    return col_scale;
}

bool quantizeScaledInts(const std::vector<float> &col, int p, std::vector<std::int32_t> &out) {
    const int n = static_cast<int>(col.size());
    out.resize(static_cast<std::size_t>(n));
    double pow10 = std::pow(10.0, p);
    for (int i = 0; i < n; i++) {
        float v = col[static_cast<std::size_t>(i)];
        std::uint32_t vb = 0;
        std::memcpy(&vb, &v, 4);
        if (v == 0.0f) {
            out[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(vb);
            continue;
        }
        long long m = static_cast<long long>(std::llround(static_cast<double>(v) * pow10));
        if (m < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
            m > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
            return false;
        }
        std::int32_t im = static_cast<std::int32_t>(m);
        float r = static_cast<float>(static_cast<double>(im) / pow10);
        std::uint32_t rb = 0;
        std::memcpy(&rb, &r, 4);
        if (rb != vb) {
            return false;
        }
        out[static_cast<std::size_t>(i)] = im;
    }
    return true;
}

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                               std::vector<long long> &result) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const int n = static_cast<int>(col.size());
        if (n == 0) {
            continue;
        }
        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        int col_scale = maxLosslessDecimalScale(col);
        if (col_scale < 0) {
            continue;
        }
        std::vector<std::int32_t> data;
        if (!quantizeScaledInts(col, col_scale, data)) {
            continue;
        }
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto w = java_fastpfor::compressIntBits(data.data(), data.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::int32_t> recovered(static_cast<std::size_t>(n));
            auto t2 = std::chrono::high_resolution_clock::now();
            java_fastpfor::uncompressIntBits(w.data(), w.size(), recovered.data(), recovered.size());
            auto t3 = std::chrono::high_resolution_clock::now();
            if (std::memcmp(recovered.data(), data.data(), static_cast<std::size_t>(n) * sizeof(std::int32_t)) !=
                0) {
                continue;
            }
            double pow10 = std::pow(10.0, col_scale);
            bool rt_ok = true;
            for (int i = 0; i < n; i++) {
                const float v = col[static_cast<std::size_t>(i)];
                if (v == 0.0f) {
                    std::uint32_t vb = 0;
                    std::memcpy(&vb, &v, 4);
                    if (static_cast<std::uint32_t>(recovered[static_cast<std::size_t>(i)]) != vb) {
                        rt_ok = false;
                        break;
                    }
                    continue;
                }
                float r =
                    static_cast<float>(static_cast<double>(recovered[static_cast<std::size_t>(i)]) / pow10);
                std::uint32_t rb = 0, vb = 0;
                std::memcpy(&rb, &r, 4);
                std::memcpy(&vb, &v, 4);
                if (rb != vb) {
                    rt_ok = false;
                    break;
                }
            }
            if (!rt_ok) {
                continue;
            }
            comp += static_cast<long long>(w.size()) * 4LL + 4LL;
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace leco_float
