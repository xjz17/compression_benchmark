#include "java_fastpfor_codec.hpp"

#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"
#include "int64_column_codec.hpp"
#include "compositecodec.h"
#include "fastpfor.h"
#include "variablebyte.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace java_fastpfor {

using FastPForLib::CompositeCodec;
using FastPForLib::FastPFor;
using FastPForLib::VariableByte;

std::vector<std::uint32_t> compressIntBits(const std::int32_t *data, std::size_t n) {
    if (n == 0) {
        return {};
    }
    CompositeCodec<FastPFor<8>, VariableByte> codec;
    const auto *in_u32 = reinterpret_cast<const std::uint32_t *>(data);
    std::vector<std::uint32_t> buf(std::max(n * 2, n + static_cast<std::size_t>(4096)));
    std::size_t cap = buf.size();
    codec.encodeArray(in_u32, n, buf.data(), cap);
    buf.resize(cap);
    return buf;
}

void uncompressIntBits(const std::uint32_t *compressed_words, std::size_t word_count,
                       std::int32_t *out, std::size_t n) {
    if (n == 0) {
        return;
    }
    CompositeCodec<FastPFor<8>, VariableByte> codec;
    std::size_t nvalue = n;
    const std::uint32_t *end = codec.decodeArray(compressed_words, word_count,
                                                 reinterpret_cast<std::uint32_t *>(out), nvalue);
    if (nvalue != n) {
        throw std::runtime_error("java_fastpfor: decoded length mismatch");
    }
    (void)end;
}

void benchIntColumnsLikeJava(const std::vector<std::vector<int>> &columns, std::vector<long long> &result,
                             LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto w = compressIntBits(col.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::int32_t> rec(n);
            auto t2 = std::chrono::high_resolution_clock::now();
            uncompressIntBits(w.data(), w.size(), rec.data(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (std::memcmp(rec.data(), col.data(), n * sizeof(std::int32_t)) != 0) {
                continue;
            }
            if (loss != nullptr) {
                std::vector<std::int64_t> orig64(col.begin(), col.end());
                std::vector<std::int64_t> dec64(rec.begin(), rec.end());
                benchmark_loss::accumulate_int64_column(orig64, dec64, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, false));
            comp += static_cast<long long>(w.size()) * 4;
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                               LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        std::vector<std::int32_t> bits(n);
        for (std::size_t i = 0; i < n; i++) {
            std::uint32_t b;
            std::memcpy(&b, &col[i], sizeof(float));
            bits[i] = static_cast<std::int32_t>(b);
        }
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto w = compressIntBits(bits.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::int32_t> rec(n);
            auto t2 = std::chrono::high_resolution_clock::now();
            uncompressIntBits(w.data(), w.size(), rec.data(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (std::memcmp(rec.data(), bits.data(), n * sizeof(std::int32_t)) != 0) {
                continue;
            }
            if (loss != nullptr) {
                std::vector<double> orig_d(col.begin(), col.end());
                std::vector<double> dec_d(n);
                for (std::size_t i = 0; i < n; i++) {
                    float f = 0;
                    std::memcpy(&f, &rec[i], sizeof(float));
                    dec_d[i] = static_cast<double>(f);
                }
                benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(w.size()) * 4;
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchInt64ColumnsLikeJava(const std::vector<std::vector<std::int64_t>> &columns,
                               std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto payload = int64_column_codec::fast_pfor_encode_int64(col.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto rec = int64_column_codec::fast_pfor_decode_int64(payload.data(), payload.size(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (rec.size() != n || !std::equal(rec.begin(), rec.end(), col.begin())) {
                continue;
            }
            if (loss != nullptr) {
                benchmark_loss::accumulate_int64_column(col, rec, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, false));
            comp += static_cast<long long>(payload.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                                LossAccum *loss) {
    std::vector<std::vector<std::int64_t>> bits_columns;
    bits_columns.reserve(columns.size());
    for (const auto &col : columns) {
        std::vector<std::int64_t> bits;
        bits.reserve(col.size());
        for (double v : col) {
            std::uint64_t u = 0;
            std::memcpy(&u, &v, sizeof(u));
            bits.push_back(static_cast<std::int64_t>(u));
        }
        bits_columns.push_back(std::move(bits));
    }
    benchInt64ColumnsLikeJava(bits_columns, result, loss);
}

} // namespace java_fastpfor
