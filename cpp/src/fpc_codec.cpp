#include "fpc_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <cstdlib>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

extern "C" {
size_t FPC_Compress_Memory(const void *in_stream, size_t in_bytes, void *out_stream, long predsizem1);
size_t FPC_Decompress_Memory(const void *in_stream, size_t in_bytes, void *out_stream);
size_t FPC_Compress_Memory_Ablation(const void *in_stream, size_t in_bytes, void *out_stream,
                                    long predsizem1, int mode);
size_t FPC_Decompress_Memory_Ablation(const void *in_stream, size_t in_bytes, void *out_stream,
                                      int mode);
}

namespace fpc_codec {

unsigned predSizeLg2ToBench() {
    if (const char *env = std::getenv("WEB_COMPRESSION_FPC_PRED_SIZE_LG2")) {
        char *end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && v <= 24) {
            return static_cast<unsigned>(v);
        }
    }
    return kDefaultPredSizeLg2;
}

template <typename T>
void benchColumnsFpcImpl(const std::vector<std::vector<T>> &columns, unsigned pred_size_lg2,
                         std::vector<long long> &result, LossAccum *loss, int mode = 0) {
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
        const size_t csize = FPC_Compress_Memory_Ablation(
            raw.data(), nbytes, cbuf.data(), static_cast<long>(pred_size_lg2), mode);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (csize == 0 || csize > cbuf.size()) {
            continue;
        }

        std::vector<unsigned char> dec(nbytes);
        auto t2 = std::chrono::high_resolution_clock::now();
        const size_t consumed = FPC_Decompress_Memory_Ablation(
            cbuf.data(), csize, dec.data(), mode);
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

void benchDoubleColumnsFpc(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                           LossAccum *loss) {
    benchColumnsFpcImpl(columns, predSizeLg2ToBench(), result, loss);
}

void benchInt64ColumnsFpc(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                          LossAccum *loss) {
    benchColumnsFpcImpl(columns, predSizeLg2ToBench(), result, loss);
}

int ablation_mode(std::string_view omitted_operator) {
    if (omitted_operator.empty()) return 0;
    if (omitted_operator == "xor") return 1;
    if (omitted_operator == "trailing_zero_pack") return 2;
    throw std::invalid_argument("unsupported FPC operator ablation");
}

void benchDoubleColumnsFpcAblation(const std::vector<std::vector<double>> &columns,
                                   std::string_view omitted_operator,
                                   std::vector<long long> &result, LossAccum *loss) {
    benchColumnsFpcImpl(columns, predSizeLg2ToBench(), result, loss,
                        ablation_mode(omitted_operator));
}

void benchInt64ColumnsFpcAblation(const std::vector<std::vector<std::int64_t>> &columns,
                                  std::string_view omitted_operator,
                                  std::vector<long long> &result, LossAccum *loss) {
    benchColumnsFpcImpl(columns, predSizeLg2ToBench(), result, loss,
                        ablation_mode(omitted_operator));
}

} // namespace fpc_codec
