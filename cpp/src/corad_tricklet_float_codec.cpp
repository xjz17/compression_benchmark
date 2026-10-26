#include "corad_tricklet_float_codec.hpp"
#include "benchmark_wire.hpp"

#include <zlib.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace corad_tricklet_float {
namespace {

constexpr int kTrick = 40;
constexpr int kAtoms = 4;

std::vector<std::uint8_t> zlib_compress(const std::uint8_t *data, std::size_t len, int level) {
    const uLong src_len = static_cast<uLong>(len);
    const uLong bound = compressBound(src_len);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(bound));
    uLongf dest_len = bound;
    const int rc = compress2(out.data(), &dest_len, data, src_len, level);
    if (rc != Z_OK) {
        return {};
    }
    out.resize(static_cast<std::size_t>(dest_len));
    return out;
}

bool zlib_decompress(const std::uint8_t *compressed, std::size_t comp_len, std::uint8_t *out,
                     std::size_t expected_len) {
    uLongf dest_len = static_cast<uLongf>(expected_len);
    const int rc =
        uncompress(out, &dest_len, compressed, static_cast<uLong>(comp_len));
    return rc == Z_OK && dest_len == expected_len;
}

template <typename T, bool UseFloat32Payload, bool IsFloatInput>
void benchCoradColumnsImpl(const std::vector<std::vector<T>> &columns, std::vector<long long> &result,
                           LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const int n = static_cast<int>(col.size());
        try {
            std::vector<double> x(static_cast<std::size_t>(n));
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                x[static_cast<std::size_t>(i)] = static_cast<double>(col[static_cast<std::size_t>(i)]);
                sum += x[static_cast<std::size_t>(i)];
            }
            const double mu = sum / static_cast<double>(n);
            double var = 0.0;
            for (int i = 0; i < n; i++) {
                const double d = x[static_cast<std::size_t>(i)] - mu;
                var += d * d;
            }
            const double sig = std::sqrt(var / static_cast<double>(n));
            if (sig < 1e-15) {
                // Constant columns are common in the cleaned benchmark data.  The
                // normalized CORAD path has sigma=0, so encode one exact value and
                // its count instead of silently dropping the whole column.
                const std::uint64_t count = static_cast<std::uint64_t>(n);
                std::vector<std::uint8_t> payload(sizeof(count) + sizeof(T));
                std::memcpy(payload.data(), &count, sizeof(count));
                std::memcpy(payload.data() + sizeof(count), col.data(), sizeof(T));
                const auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::uint8_t> compressed =
                    zlib_compress(payload.data(), payload.size(), 6);
                const auto t1 = std::chrono::high_resolution_clock::now();
                if (compressed.empty()) {
                    continue;
                }
                std::vector<std::uint8_t> dec(payload.size());
                const auto t2 = std::chrono::high_resolution_clock::now();
                const bool decoded = zlib_decompress(compressed.data(), compressed.size(),
                                                     dec.data(), dec.size());
                const auto t3 = std::chrono::high_resolution_clock::now();
                if (!decoded || dec != payload) {
                    continue;
                }
                if (loss != nullptr) {
                    if constexpr (IsFloatInput) {
                        std::vector<double> original(col.begin(), col.end());
                        std::vector<double> recovered(original.size(),
                                                      static_cast<double>(col.front()));
                        benchmark_loss::accumulate_double_column(original, recovered, *loss);
                    } else {
                        std::vector<std::int64_t> original(col.begin(), col.end());
                        std::vector<std::int64_t> recovered(original.size(),
                                                            static_cast<std::int64_t>(col.front()));
                        benchmark_loss::accumulate_int64_column(original, recovered, *loss);
                    }
                }
                orig += static_cast<long long>(benchmark_wire::column_original_size(n, IsFloatInput));
                comp += static_cast<long long>(compressed.size());
                ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
                continue;
            }
            if (n < kTrick * 2) {
                continue;
            }
            const int ntrim = (n / kTrick) * kTrick;
            const int nrows = ntrim / kTrick;
            if (nrows < 1) {
                continue;
            }
            const int k = std::min(kAtoms, nrows);
            const std::size_t word = UseFloat32Payload ? 4u : 8u;
            const std::size_t payload_len =
                static_cast<std::size_t>(k * kTrick + nrows * kTrick) * word;
            std::vector<std::uint8_t> payload(payload_len);
            std::size_t off = 0;
            auto append_f32 = [&](float v) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &v, 4);
                payload[off++] = static_cast<std::uint8_t>(bits & 0xFFu);
                payload[off++] = static_cast<std::uint8_t>((bits >> 8) & 0xFFu);
                payload[off++] = static_cast<std::uint8_t>((bits >> 16) & 0xFFu);
                payload[off++] = static_cast<std::uint8_t>((bits >> 24) & 0xFFu);
            };
            auto append_f64 = [&](double v) {
                std::uint64_t bits = 0;
                std::memcpy(&bits, &v, 8);
                for (int b = 0; b < 8; ++b) {
                    payload[off++] = static_cast<std::uint8_t>((bits >> (8 * b)) & 0xFFu);
                }
            };
            for (int r = 0; r < k; r++) {
                for (int c = 0; c < kTrick; c++) {
                    const int idx = r * kTrick + c;
                    const double z = (x[static_cast<std::size_t>(idx)] - mu) / sig;
                    if constexpr (UseFloat32Payload) {
                        append_f32(static_cast<float>(z));
                    } else {
                        append_f64(z);
                    }
                }
            }
            for (int r = 0; r < nrows; r++) {
                for (int c = 0; c < kTrick; c++) {
                    const int idx = r * kTrick + c;
                    const double z = (x[static_cast<std::size_t>(idx)] - mu) / sig;
                    if constexpr (UseFloat32Payload) {
                        append_f32(static_cast<float>(z));
                    } else {
                        append_f64(z);
                    }
                }
            }
            const auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::uint8_t> compressed = zlib_compress(payload.data(), payload.size(), 6);
            const auto t1 = std::chrono::high_resolution_clock::now();
            if (compressed.empty()) {
                continue;
            }
            const auto t2 = std::chrono::high_resolution_clock::now();
            std::vector<std::uint8_t> dec(payload.size());
            if (!zlib_decompress(compressed.data(), compressed.size(), dec.data(), payload.size())) {
                continue;
            }
            const auto t3 = std::chrono::high_resolution_clock::now();
            if (dec != payload) {
                continue;
            }
            (void)loss; // bench verifies zlib round-trip on normalized payload, not value decode
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, IsFloatInput));
            comp += static_cast<long long>(compressed.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void disabled_lossy_benchmark(std::vector<long long>& result) {
    result.assign(4, 0);
    static const bool warned = [] {
        std::cerr << "CORAD C++ legacy benchmark disabled: normalized payload is not "
                     "lossless; use the versioned Python CRD2 baseline.\n";
        return true;
    }();
    (void)warned;
}

} // namespace

void benchFloatColumnsCoradTricklet(const std::vector<std::vector<float>> &columns,
                                    std::vector<long long> &result, LossAccum *loss) {
    (void)columns;
    (void)loss;
    disabled_lossy_benchmark(result);
}

void benchDoubleColumnsCoradTricklet(const std::vector<std::vector<double>> &columns,
                                     std::vector<long long> &result, LossAccum *loss) {
    (void)columns;
    (void)loss;
    disabled_lossy_benchmark(result);
}

void benchInt64ColumnsCoradTricklet(const std::vector<std::vector<std::int64_t>> &columns,
                                    std::vector<long long> &result, LossAccum *loss) {
    (void)columns;
    (void)loss;
    disabled_lossy_benchmark(result);
}

void benchDoubleColumnsCoradTrickletLegacyForensics(
    const std::vector<std::vector<double>>& columns, std::vector<long long>& result,
    LossAccum* loss) {
    benchCoradColumnsImpl<double, false, true>(columns, result, loss);
}

} // namespace corad_tricklet_float
