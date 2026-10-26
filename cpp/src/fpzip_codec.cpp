#include "fpzip_codec.hpp"
#include "benchmark_columns.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <fpzip.h>

#include <chrono>
#include <climits>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace fpzip_codec {

namespace {

template <typename T>
bool bench_one_column_typed(const std::vector<T> &col, int fpzip_type, bool is_float_input,
                            long long &orig, long long &comp, long long &ct, long long &dt,
                            LossAccum *loss) {
    const std::size_t n = col.size();
    if (n == 0 || n > static_cast<std::size_t>(INT_MAX)) {
        return false;
    }

    FILE *tmp = std::tmpfile();
    if (tmp == nullptr) {
        return false;
    }

    FPZ *writer = fpzip_write_to_file(tmp);
    if (writer == nullptr) {
        std::fclose(tmp);
        return false;
    }
    writer->type = fpzip_type;
    writer->prec = 0;
    writer->nx = static_cast<int>(n);
    writer->ny = 1;
    writer->nz = 1;
    writer->nf = 1;

    const auto t0 = std::chrono::high_resolution_clock::now();
    if (!fpzip_write_header(writer)) {
        fpzip_write_close(writer);
        std::fclose(tmp);
        return false;
    }
    const std::size_t outbytes = fpzip_write(writer, col.data());
    fpzip_write_close(writer);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (outbytes == 0) {
        std::fclose(tmp);
        return false;
    }

    if (std::fseek(tmp, 0, SEEK_END) != 0) {
        std::fclose(tmp);
        return false;
    }
    const long compPos = std::ftell(tmp);
    if (compPos < 0) {
        std::fclose(tmp);
        return false;
    }
    if (std::fseek(tmp, 0, SEEK_SET) != 0) {
        std::fclose(tmp);
        return false;
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    FPZ *reader = fpzip_read_from_file(tmp);
    if (reader == nullptr) {
        std::fclose(tmp);
        return false;
    }
    if (!fpzip_read_header(reader)) {
        fpzip_read_close(reader);
        std::fclose(tmp);
        return false;
    }
    std::vector<T> decoded(n);
    if (!fpzip_read(reader, decoded.data())) {
        fpzip_read_close(reader);
        std::fclose(tmp);
        return false;
    }
    fpzip_read_close(reader);
    const auto t3 = std::chrono::high_resolution_clock::now();
    std::fclose(tmp);

    if (std::memcmp(decoded.data(), col.data(), n * sizeof(T)) != 0) {
        return false;
    }
    if (loss != nullptr) {
        if constexpr (std::is_same_v<T, double>) {
            benchmark_loss::accumulate_double_column(col, decoded, *loss);
        } else if constexpr (std::is_same_v<T, float>) {
            std::vector<double> orig_d(col.begin(), col.end());
            std::vector<double> dec_d(decoded.begin(), decoded.end());
            benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
        }
    }
    orig = static_cast<long long>(benchmark_wire::column_original_size(static_cast<std::size_t>(n), is_float_input));
    comp = static_cast<long long>(compPos);
    ct = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    return true;
}

} // namespace

void benchFloatColumns(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                       LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_column_typed<float>(col, FPZIP_TYPE_FLOAT, true, o, c, tenc, tdec, loss)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchDoubleColumns(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                        LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_column_typed<double>(col, FPZIP_TYPE_DOUBLE, true, o, c, tenc, tdec, loss)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchInt64Columns(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                       LossAccum *loss) {
    const auto double_cols = integers_as_double_columns(columns);
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : double_cols) {
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_column_typed<double>(col, FPZIP_TYPE_DOUBLE, true, o, c, tenc, tdec, loss)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace fpzip_codec
