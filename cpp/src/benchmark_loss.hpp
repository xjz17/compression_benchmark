#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct LossAccum {
    double err = 0.0;
    double ref = 0.0;
};

namespace benchmark_loss {

inline constexpr double kLossRateEps = 1e-15;

inline bool is_known_lossy_algorithm(const std::string &algorithm) {
    std::string key = algorithm;
    for (char &c : key) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    static const char *kPrefixes[] = {
        "sz_", "sz3_", "serf_", "simpiece_", "ext_zfp_tol_", "machete_rel_",
        "machete_float_rel_", "most_minimal", "most_float_minimal",
    };
    for (const char *prefix : kPrefixes) {
        if (key.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

inline void accumulate_bytes(const std::vector<char> &orig, const std::vector<char> &dec,
                             LossAccum &acc) {
    const std::size_t ref = orig.size();
    std::size_t err = 0;
    const std::size_t n = std::min(orig.size(), dec.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (orig[i] != dec[i]) {
            ++err;
        }
    }
    if (orig.size() > dec.size()) {
        err += orig.size() - dec.size();
    } else if (dec.size() > orig.size()) {
        err += dec.size() - orig.size();
    }
    acc.err += static_cast<double>(err);
    acc.ref += static_cast<double>(ref);
}

inline void accumulate_int64_column(const std::vector<std::int64_t> &orig,
                                    const std::vector<std::int64_t> &dec, LossAccum &acc) {
    const std::size_t n = std::min(orig.size(), dec.size());
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (orig[i] != dec[i]) {
            err += 1.0;
        }
    }
    if (orig.size() != dec.size()) {
        err += static_cast<double>(std::max(orig.size(), dec.size()) -
                                   std::min(orig.size(), dec.size()));
    }
    acc.err += err;
    acc.ref += static_cast<double>(std::max(orig.size(), dec.size()));
}

inline void accumulate_double_column(const std::vector<double> &orig,
                                     const std::vector<double> &dec, LossAccum &acc) {
    const std::size_t n = std::min(orig.size(), dec.size());
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t orig_bits = 0;
        std::uint64_t dec_bits = 0;
        std::memcpy(&orig_bits, &orig[i], sizeof(orig_bits));
        std::memcpy(&dec_bits, &dec[i], sizeof(dec_bits));
        if (orig_bits != dec_bits) {
            err += 1.0;
        }
    }
    if (orig.size() != dec.size()) {
        err += static_cast<double>(std::max(orig.size(), dec.size()) -
                                   std::min(orig.size(), dec.size()));
    }
    acc.err += err;
    acc.ref += static_cast<double>(std::max(orig.size(), dec.size()));
}

inline double loss_rate(double sum_err, double sum_ref) {
    if (sum_ref <= 0.0) {
        return sum_err > 0.0 ? 1.0 : 0.0;
    }
    return sum_err / sum_ref;
}

inline int is_lossy_flag(const std::string &algorithm, double sum_err, double sum_ref) {
    if (is_known_lossy_algorithm(algorithm)) {
        return 1;
    }
    return loss_rate(sum_err, sum_ref) > kLossRateEps ? 1 : 0;
}

/** Fixed decimal form for CSV (never scientific notation). */
inline std::string loss_rate_csv(double rate) {
    if (rate == 0.0 || !std::isfinite(rate)) {
        return "0";
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.15f", rate);
    std::string s(buf);
    while (!s.empty() && s.back() == '0') {
        s.pop_back();
    }
    if (!s.empty() && s.back() == '.') {
        s.pop_back();
    }
    return s.empty() ? "0" : s;
}

} // namespace benchmark_loss
