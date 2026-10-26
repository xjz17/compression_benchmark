#include "research_cli_serf_embed.hpp"
#include "benchmark_wire.hpp"

#include <chrono>
#include <cmath>
#include <vector>

#ifndef WEB_COMPRESSION_EMBED_SERF

bool bench_serf_inproc(const std::vector<std::vector<float>> &column_float_data, std::ofstream &out) {
    (void)column_float_data;
    (void)out;
    return false;
}

#else

#include "compressor/serf_xor_compressor.h"
#include "decompressor/serf_xor_decompressor.h"
#include "utils/array.h"

namespace {

const char *kSerfBounds[] = {"1e-1", "1e-2", "1e-3", "1e-4"};
const char *kSerfRows[] = {"serf_ABS_1e-1", "serf_ABS_1e-2", "serf_ABS_1e-3", "serf_ABS_1e-4"};

} // namespace

bool bench_serf_inproc(const std::vector<std::vector<float>> &column_float_data, std::ofstream &out) {
    for (std::size_t pi = 0; pi < sizeof(kSerfBounds) / sizeof(kSerfBounds[0]); ++pi) {
        const char *boundStr = kSerfBounds[pi];
        const double bound = std::stod(boundStr);
        long long orig = 0, comp = 0, ct = 0, dt = 0;

        for (const auto &col : column_float_data) {
            const int n = static_cast<int>(col.size());
            if (n < 1) {
                continue;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            SerfXORCompressor xor_compressor(1000, bound, 0);
            for (float f : col) {
                xor_compressor.AddValue(static_cast<double>(f));
            }
            xor_compressor.Close();
            Array<uint8_t> compressed_bytes = xor_compressor.compressed_bytes_last_block();
            auto t1 = std::chrono::high_resolution_clock::now();

            const long long csize = static_cast<long long>(xor_compressor.compressed_size_last_block());
            if (csize <= 0 || compressed_bytes.length() == 0) {
                continue;
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            SerfXORDecompressor xor_decompressor(0);
            const auto decompressed_values = xor_decompressor.Decompress(compressed_bytes);
            auto t3 = std::chrono::high_resolution_clock::now();

            bool ok = decompressed_values.size() == static_cast<std::size_t>(n);
            for (int i = 0; ok && i < n; ++i) {
                const float got = static_cast<float>(decompressed_values[static_cast<std::size_t>(i)]);
                const float exp = col[static_cast<std::size_t>(i)];
                if (std::fabs(static_cast<double>(got - exp)) > bound * 1.0001 + 1e-6) {
                    ok = false;
                }
            }

            if (ok) {
                orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
                comp += csize;
                ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }
        }
        if (orig > 0) {
            out << kSerfRows[pi] << ',' << orig << ',' << comp << ',' << ct << ',' << dt << '\n';
        }
    }
    return true;
}

#endif
