#include "spdp_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <chrono>
#include <cstring>
#include <type_traits>
#include <vector>

extern "C" {
size_t spdp_compress_block(unsigned char level, const unsigned char *in, size_t len, unsigned char *out,
                           size_t out_cap);
int spdp_decompress_block(unsigned char level, const unsigned char *comp, size_t csize, unsigned char *out,
                          size_t out_len);
}

namespace spdp_codec {

static constexpr size_t kMaxBlock = (1u << 23);

std::string rowName(unsigned level) {
    return "spdp_level_" + std::to_string(level);
}

std::vector<unsigned> levelsToBench() {
    std::vector<unsigned> all;
    all.reserve(static_cast<std::size_t>(kMaxLevel - kMinLevel + 1));
    for (unsigned level = kMinLevel; level <= kMaxLevel; ++level) {
        all.push_back(level);
    }
    return all;
}

template <typename T>
void benchColumnsSpdpImpl(const std::vector<std::vector<T>> &columns, unsigned level,
                          std::vector<long long> &result, LossAccum *loss) {
    if (level > 9) {
        level = 5;
    }
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        const std::size_t nbytes = n * sizeof(T);
        std::vector<unsigned char> raw(nbytes);
        std::memcpy(raw.data(), col.data(), nbytes);

        std::vector<std::vector<unsigned char>> chunks;
        chunks.reserve((nbytes + kMaxBlock - 1) / kMaxBlock);
        bool ok = true;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t off = 0; off < nbytes && ok; off += kMaxBlock) {
            const std::size_t chunk = std::min(kMaxBlock, nbytes - off);
            std::vector<unsigned char> cbuf(chunk * 2 + 9);
            const std::size_t csize =
                spdp_compress_block(static_cast<unsigned char>(level), raw.data() + off, chunk,
                                    cbuf.data(), cbuf.size());
            if (csize == 0) {
                ok = false;
                break;
            }
            cbuf.resize(csize);
            chunks.push_back(std::move(cbuf));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!ok) {
            continue;
        }

        std::vector<unsigned char> dec(nbytes);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::size_t off = 0;
        for (std::size_t i = 0; i < chunks.size() && ok; i++) {
            const std::size_t chunk = std::min(kMaxBlock, nbytes - off);
            if (spdp_decompress_block(static_cast<unsigned char>(level), chunks[i].data(), chunks[i].size(),
                                      dec.data() + off, chunk) != 0) {
                ok = false;
                break;
            }
            off += chunk;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (!ok || std::memcmp(dec.data(), raw.data(), nbytes) != 0) {
            continue;
        }
        if (loss != nullptr) {
            if constexpr (std::is_same_v<T, std::int64_t>) {
                std::vector<std::int64_t> dec_vals(n);
                std::memcpy(dec_vals.data(), dec.data(), nbytes);
                benchmark_loss::accumulate_int64_column(col, dec_vals, *loss);
            } else if constexpr (std::is_same_v<T, double>) {
                std::vector<double> dec_vals(n);
                std::memcpy(dec_vals.data(), dec.data(), nbytes);
                benchmark_loss::accumulate_double_column(col, dec_vals, *loss);
            } else {
                std::vector<double> orig_d(col.begin(), col.end());
                std::vector<float> dec_f(n);
                std::memcpy(dec_f.data(), dec.data(), nbytes);
                std::vector<double> dec_d(dec_f.begin(), dec_f.end());
                benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
            }
        }

        std::size_t total_c = 0;
        for (const auto &c : chunks) {
            total_c += c.size();
        }
        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        comp += static_cast<long long>(total_c);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchFloatColumnsSpdp(const std::vector<std::vector<float>> &columns, unsigned level,
                           std::vector<long long> &result, LossAccum *loss) {
    benchColumnsSpdpImpl(columns, level, result, loss);
}

void benchDoubleColumnsSpdp(const std::vector<std::vector<double>> &columns, unsigned level,
                            std::vector<long long> &result, LossAccum *loss) {
    benchColumnsSpdpImpl(columns, level, result, loss);
}

void benchInt64ColumnsSpdp(const std::vector<std::vector<std::int64_t>> &columns, unsigned level,
                           std::vector<long long> &result, LossAccum *loss) {
    benchColumnsSpdpImpl(columns, level, result, loss);
}

} // namespace spdp_codec
