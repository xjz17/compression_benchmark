#include "zfp_codec.hpp"
#include "benchmark_loss.hpp"

#include <zfp.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace zfp_codec {

namespace {

bool env_bench_on() {
    if (const char *env = std::getenv("WEB_COMPRESSION_ZFP_BENCH")) {
        return env[0] != '0';
    }
    return true;
}

enum class Mode { Reversible, Tolerance };

bool bench_one_column(const std::vector<double> &col, Mode mode, double tolerance, long long &orig,
                      long long &comp, long long &ct, long long &dt, LossAccum *loss) {
    const std::size_t n = col.size();
    if (n == 0 || n > static_cast<std::size_t>(UINT32_MAX)) {
        return false;
    }

    zfp_field *field = zfp_field_1d(const_cast<double *>(col.data()), zfp_type_double,
                                    static_cast<unsigned>(n));
    if (field == nullptr) {
        return false;
    }

    zfp_stream *zfp = zfp_stream_open(nullptr);
    if (zfp == nullptr) {
        zfp_field_free(field);
        return false;
    }

    if (mode == Mode::Reversible) {
        zfp_stream_set_reversible(zfp);
    } else {
        zfp_stream_set_accuracy(zfp, tolerance);
    }

    const size_t cap = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> cbuf(cap);
    bitstream *stream = stream_open(cbuf.data(), cap);
    if (stream == nullptr) {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        return false;
    }
    zfp_stream_set_bit_stream(zfp, stream);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const size_t csize = zfp_compress(zfp, field);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (csize == 0 || csize > cap) {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        return false;
    }

    std::vector<double> dec(n);
    zfp_field *out_field =
        zfp_field_1d(dec.data(), zfp_type_double, static_cast<unsigned>(n));
    if (out_field == nullptr) {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        return false;
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    zfp_stream_rewind(zfp);
    const size_t consumed = zfp_decompress(zfp, out_field);
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (consumed != csize) {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        zfp_field_free(out_field);
        return false;
    }

    if (mode == Mode::Reversible) {
        if (std::memcmp(dec.data(), col.data(), n * sizeof(double)) != 0) {
            stream_close(stream);
            zfp_stream_close(zfp);
            zfp_field_free(field);
            zfp_field_free(out_field);
            return false;
        }
    } else {
        const double verify_atol = std::max(tolerance * 16.0, 1e-6);
        for (std::size_t i = 0; i < n; ++i) {
            if (std::fabs(col[i] - dec[i]) > verify_atol) {
                stream_close(stream);
                zfp_stream_close(zfp);
                zfp_field_free(field);
                zfp_field_free(out_field);
                return false;
            }
        }
    }

    if (loss != nullptr) {
        benchmark_loss::accumulate_double_column(col, dec, *loss);
    }

    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    zfp_field_free(out_field);

    orig = static_cast<long long>(n * sizeof(double));
    comp = static_cast<long long>(csize);
    ct = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    return true;
}

void bench_columns_impl(const std::vector<std::vector<double>> &columns, Mode mode, double tolerance,
                        std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        long long o = 0;
        long long c = 0;
        long long t0 = 0;
        long long t1 = 0;
        if (!bench_one_column(col, mode, tolerance, o, c, t0, t1, loss)) {
            continue;
        }
        orig += o;
        comp += c;
        ct += t0;
        dt += t1;
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace

bool bench_enabled() {
    return env_bench_on();
}

void benchReversibleDoubleColumns(const std::vector<std::vector<double>> &columns,
                                  std::vector<long long> &result, LossAccum *loss) {
    if (!bench_enabled()) {
        result.assign(4, 0);
        return;
    }
    bench_columns_impl(columns, Mode::Reversible, 0.0, result, loss);
}

void benchToleranceDoubleColumns(const std::vector<std::vector<double>> &columns, double tolerance,
                                 const char * /*row_name*/, std::vector<long long> &result,
                                 LossAccum *loss) {
    if (!bench_enabled() || tolerance <= 0.0) {
        result.assign(4, 0);
        return;
    }
    bench_columns_impl(columns, Mode::Tolerance, tolerance, result, loss);
}

} // namespace zfp_codec
