#include "research_cli_sz3_embed.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef WEB_COMPRESSION_EMBED_SZ3

bool bench_sz3_family_inproc(const std::vector<std::vector<float>> &column_float_data,
                             const std::string &sz_config_path,
                             std::ofstream &out) {
    (void)column_float_data;
    (void)sz_config_path;
    (void)out;
    return false;
}
bool bench_sz3_family_inproc_double(const std::vector<std::vector<double>> &column_double_data,
                                    const std::string &sz_config_path,
                                    std::ofstream &out) {
    (void)column_double_data;
    (void)sz_config_path;
    (void)out;
    return false;
}

#else

#include "SZ3/api/sz.hpp"

namespace {

struct Sz3Preset {
    const char *row_base;
    const char *err_mode;
    const char *bound_value;
};

const Sz3Preset kSz3Presets[] = {
    {"sz_ABS_1e-1", "ABS", "1E-1"},
    {"sz_ABS_1e-2", "ABS", "1E-2"},
    {"sz_ABS_1e-3", "ABS", "1E-3"},
    {"sz_ABS_1e-4", "ABS", "1E-4"},
    {"sz_REL_1e-2", "REL", "1E-2"},
    {"sz_REL_1e-3", "REL", "1E-3"},
};

} // namespace

bool bench_sz3_family_inproc(const std::vector<std::vector<float>> &column_float_data,
                             const std::string &sz_config_path,
                             std::ofstream &out) {
    using namespace SZ3;
    for (const Sz3Preset &pr : kSz3Presets) {
        long long orig = 0, comp = 0, ct = 0, dt = 0;
        LossAccum loss;
        for (const auto &col : column_float_data) {
            const int n = static_cast<int>(col.size());
            if (n < 1) {
                continue;
            }

            SZ3::Config conf(static_cast<size_t>(n));
            try {
                conf.loadcfg(sz_config_path);
            } catch (const std::exception &) {
                return false;
            }

            if (strcmp(pr.err_mode, "ABS") == 0) {
                conf.errorBoundMode = EB_ABS;
                conf.absErrorBound = std::stod(pr.bound_value);
            } else if (strcmp(pr.err_mode, "REL") == 0) {
                conf.errorBoundMode = EB_REL;
                conf.relErrorBound = std::stod(pr.bound_value);
            } else {
                continue;
            }

            size_t cmp_size = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            char *cmp = nullptr;
            try {
                cmp = SZ_compress(conf, col.data(), cmp_size);
            } catch (const std::exception &) {
                continue;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!cmp || cmp_size == 0) {
                delete[] cmp;
                continue;
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            float *dec = nullptr;
            try {
                dec = SZ_decompress<float>(conf, cmp, cmp_size);
            } catch (const std::exception &) {
                delete[] cmp;
                continue;
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            delete[] cmp;
            if (dec != nullptr) {
                std::vector<double> orig_d(static_cast<std::size_t>(n));
                std::vector<double> dec_d(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    orig_d[static_cast<std::size_t>(i)] =
                        static_cast<double>(col[static_cast<std::size_t>(i)]);
                    dec_d[static_cast<std::size_t>(i)] = static_cast<double>(dec[static_cast<std::size_t>(i)]);
                }
                benchmark_loss::accumulate_double_column(orig_d, dec_d, loss);
                delete[] dec;
            }

            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(cmp_size);
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
        if (orig > 0) {
            std::string name = pr.row_base;
            if (name.rfind("sz_", 0) == 0) {
                name.replace(0, 3, "sz3_");
            }
            out << name << ',' << orig << ',' << comp << ',' << ct << ',' << dt;
            if (loss.err != 0.0 || loss.ref != 0.0) {
                out << ',' << loss.err << ',' << loss.ref;
            }
            out << '\n';
        }
    }
    return true;
}

bool bench_sz3_family_inproc_double(const std::vector<std::vector<double>> &column_double_data,
                                    const std::string &sz_config_path,
                                    std::ofstream &out) {
    using namespace SZ3;
    for (const Sz3Preset &pr : kSz3Presets) {
        long long orig = 0, comp = 0, ct = 0, dt = 0;
        LossAccum loss;
        for (const auto &col : column_double_data) {
            const int n = static_cast<int>(col.size());
            if (n < 1) {
                continue;
            }
            SZ3::Config conf(static_cast<size_t>(n));
            try {
                conf.loadcfg(sz_config_path);
            } catch (const std::exception &) {
                return false;
            }
            if (strcmp(pr.err_mode, "ABS") == 0) {
                conf.errorBoundMode = EB_ABS;
                conf.absErrorBound = std::stod(pr.bound_value);
            } else if (strcmp(pr.err_mode, "REL") == 0) {
                conf.errorBoundMode = EB_REL;
                conf.relErrorBound = std::stod(pr.bound_value);
            } else {
                continue;
            }
            size_t cmp_size = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            char *cmp = nullptr;
            try {
                cmp = SZ_compress(conf, col.data(), cmp_size);
            } catch (const std::exception &) {
                continue;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!cmp || cmp_size == 0) {
                delete[] cmp;
                continue;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double *dec = nullptr;
            try {
                dec = SZ_decompress<double>(conf, cmp, cmp_size);
            } catch (const std::exception &) {
                delete[] cmp;
                continue;
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            delete[] cmp;
            if (dec != nullptr) {
                benchmark_loss::accumulate_double_column(col, std::vector<double>(dec, dec + n), loss);
                delete[] dec;
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(cmp_size);
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
        if (orig > 0) {
            std::string name = pr.row_base;
            if (name.rfind("sz_", 0) == 0) {
                name.replace(0, 3, "sz3_");
            }
            out << name << ',' << orig << ',' << comp << ',' << ct << ',' << dt;
            if (loss.err != 0.0 || loss.ref != 0.0) {
                out << ',' << loss.err << ',' << loss.ref;
            }
            out << '\n';
        }
    }
    return true;
}

#endif
