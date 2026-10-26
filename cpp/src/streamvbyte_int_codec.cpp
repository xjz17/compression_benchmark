#include "streamvbyte_int_codec.hpp"

#include "benchmark_columns.hpp"
#include "benchmark_loss.hpp"
#include "int64_column_codec.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
size_t svb_encode(uint8_t *out, const uint32_t *in, uint32_t count, int delta, int type);
size_t svb_decode(uint32_t *out, uint8_t *in, int delta, int type);
}

namespace streamvbyte_int {

void benchIntColumnsStreamVByteDelta1(const std::vector<std::vector<int>> &columns,
                                      std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0 || n > 0xFFFFFFFFu) {
            continue;
        }
        std::vector<std::uint32_t> u32(n);
        for (std::size_t i = 0; i < n; i++) {
            u32[i] = static_cast<std::uint32_t>(col[i]);
        }
        std::vector<std::uint8_t> buf(4 + ((n + 3) / 4) + n * 5 + 64);
        auto t0 = std::chrono::high_resolution_clock::now();
        const std::size_t enc =
            svb_encode(buf.data(), u32.data(), static_cast<std::uint32_t>(n), 1, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (enc == 0 || enc > buf.size()) {
            continue;
        }
        buf.resize(enc);
        std::vector<std::uint32_t> dec(n);
        auto t2 = std::chrono::high_resolution_clock::now();
        const std::size_t consumed = svb_decode(dec.data(), buf.data(), 1, 1);
        auto t3 = std::chrono::high_resolution_clock::now();
        if (consumed != enc) {
            continue;
        }
        bool ok = true;
        for (std::size_t i = 0; i < n; i++) {
            if (dec[i] != static_cast<std::uint32_t>(col[i])) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<std::int64_t> orig64(col.begin(), col.end());
            std::vector<std::int64_t> dec64(n);
            for (std::size_t i = 0; i < n; i++) {
                dec64[i] = static_cast<std::int64_t>(dec[i]);
            }
            benchmark_loss::accumulate_int64_column(orig64, dec64, *loss);
        }
        orig += static_cast<long long>(n) * 8;
        comp += static_cast<long long>(enc);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchInt64ColumnsStreamVByteDelta1(const std::vector<std::vector<std::int64_t>> &columns,
                                        std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        orig += static_cast<long long>(n) * 8LL;
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            const auto enc = int64_column_codec::stream_vbyte_encode_int64(col.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto t2 = std::chrono::high_resolution_clock::now();
            const auto dec = int64_column_codec::stream_vbyte_decode_int64(enc.data(), enc.size(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (dec != col) {
                continue;
            }
            if (loss != nullptr) {
                benchmark_loss::accumulate_int64_column(col, dec, *loss);
            }
            comp += static_cast<long long>(enc.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace streamvbyte_int
