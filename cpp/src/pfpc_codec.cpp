#include "pfpc_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"
#include "pfpc.h"

#include <cstdlib>
#include <chrono>
#include <cstring>
#include <type_traits>
#include <vector>

namespace pfpc_codec {

unsigned predSizeLg2ToBench() {
    if (const char *env = std::getenv("COMPRESSION_BENCHMARK_PFPC_PRED_SIZE_LG2")) {
        char *end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && v <= 24) {
            return static_cast<unsigned>(v);
        }
    }
    return kDefaultPredSizeLg2;
}

unsigned threadsToBench() {
    if (const char *env = std::getenv("COMPRESSION_BENCHMARK_PFPC_THREADS")) {
        char *end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && v >= 1 && v <= 128) {
            return static_cast<unsigned>(v);
        }
    }
    return kDefaultThreads;
}

int chunkSizeToBench() {
    if (const char *env = std::getenv("COMPRESSION_BENCHMARK_PFPC_CHUNKSIZE")) {
        char *end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && v >= 1 && v <= 65536) {
            return static_cast<int>(v);
        }
    }
    return kDefaultChunkSize;
}

template <typename T>
void benchColumnsPfpcImpl(const std::vector<std::vector<T>> &columns, unsigned pred_size_lg2, unsigned threads,
                          int chunk_size, std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        const std::size_t nbytes = n * sizeof(T);
        if (nbytes % 8 != 0) {
            continue;
        }
        std::vector<unsigned char> raw(nbytes);
        std::memcpy(raw.data(), col.data(), nbytes);

        std::vector<unsigned char> cbuf(nbytes * 2 + 1024);
        auto t0 = std::chrono::high_resolution_clock::now();
        const size_t csize = pfpc_compress_memory(raw.data(), nbytes, cbuf.data(), cbuf.size(),
                                                  static_cast<unsigned char>(pred_size_lg2),
                                                  static_cast<unsigned char>(threads), chunk_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (csize == 0 || csize > cbuf.size()) {
            continue;
        }

        std::vector<unsigned char> dec(nbytes);
        auto t2 = std::chrono::high_resolution_clock::now();
        const size_t consumed = pfpc_decompress_memory(cbuf.data(), csize, dec.data(), dec.size());
        auto t3 = std::chrono::high_resolution_clock::now();
        if (consumed != csize || std::memcmp(dec.data(), raw.data(), nbytes) != 0) {
            continue;
        }

        if (loss != nullptr) {
            if constexpr (std::is_same_v<T, std::int64_t>) {
                std::vector<std::int64_t> dec_vals(n);
                std::memcpy(dec_vals.data(), dec.data(), nbytes);
                benchmark_loss::accumulate_int64_column(col, dec_vals, *loss);
            } else {
                std::vector<double> dec_vals(n);
                std::memcpy(dec_vals.data(), dec.data(), nbytes);
                benchmark_loss::accumulate_double_column(col, dec_vals, *loss);
            }
        }

        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        comp += static_cast<long long>(csize);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchDoubleColumnsPfpc(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                            LossAccum *loss) {
    benchColumnsPfpcImpl(columns, predSizeLg2ToBench(), threadsToBench(), chunkSizeToBench(), result, loss);
}

void benchInt64ColumnsPfpc(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                           LossAccum *loss) {
    benchColumnsPfpcImpl(columns, predSizeLg2ToBench(), threadsToBench(), chunkSizeToBench(), result, loss);
}

} // namespace pfpc_codec
