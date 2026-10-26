#include "alp_float_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include "int64_column_codec.hpp"
#include "java_fastpfor_codec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace alp_float {
namespace {

constexpr int kMaxExponent = 10;
constexpr float kMagicNumber = 12582912.0f;
constexpr int kExceptionSize = 32;
constexpr int kExceptionPositionSize = 16;
constexpr double kEncodingUpperLimit = 9223372036854774784.0;
constexpr double kEncodingLowerLimit = -9223372036854774784.0;

alignas(64) static constexpr float kFracArr[] = {
    1.0f,      0.1f,       0.01f,       0.001f,       0.0001f,      0.00001f,     0.000001f,
    0.0000001f, 0.00000001f, 0.000000001f, 0.0000000001f};
alignas(64) static constexpr float kExpArr[] = {1.0f,
                                                10.0f,
                                                100.0f,
                                                1000.0f,
                                                10000.0f,
                                                100000.0f,
                                                1000000.0f,
                                                10000000.0f,
                                                100000000.0f,
                                                1000000000.0f,
                                                10000000000.0f};
alignas(64) static constexpr int kFactArr[] = {1,       10,       100,       1000,       10000,
                                                 100000, 1000000, 10000000, 100000000, 1000000000};
constexpr int kMaxFactorIdx = 9;

bool impossible_to_encode(float n) {
    if (!std::isfinite(static_cast<double>(n))) {
        return true;
    }
    double d = static_cast<double>(n);
    if (d > kEncodingUpperLimit || d < kEncodingLowerLimit) {
        return true;
    }
    return false;
}

int encode_value(float value, int factor_idx, int exponent_idx) {
    float tmp = value * kExpArr[exponent_idx] * kFracArr[factor_idx];
    if (impossible_to_encode(tmp)) {
        return std::numeric_limits<int>::max();
    }
    tmp = tmp + kMagicNumber - kMagicNumber;
    return static_cast<int>(tmp);
}

float decode_value(int encoded, int factor_idx, int exponent_idx) {
    return static_cast<float>(static_cast<double>(encoded) * static_cast<double>(kFactArr[factor_idx]) *
                              static_cast<double>(kFracArr[exponent_idx]));
}

int bit_width_unsigned_span(std::int32_t min_enc, std::int32_t max_enc) {
    auto umin = static_cast<std::uint32_t>(min_enc);
    auto umax = static_cast<std::uint32_t>(max_enc);
    std::uint64_t delta = static_cast<std::uint64_t>(umax) - static_cast<std::uint64_t>(umin);
    if (delta == 0) {
        return 0;
    }
    return 64 - static_cast<int>(__builtin_clzll(delta));
}

/** Returns {fac, exp} or empty if none. */
std::vector<int> find_best_factor_exponent(const float *values, int n) {
    if (n == 1) {
        float v = values[0];
        if (impossible_to_encode(v)) {
            return {};
        }
        std::uint32_t vb = 0;
        std::memcpy(&vb, &v, 4);
        for (int exp = kMaxExponent; exp >= 0; exp--) {
            for (int fac = std::min(exp, kMaxFactorIdx); fac >= 0; fac--) {
                int enc = encode_value(v, fac, exp);
                float dec = decode_value(enc, fac, exp);
                std::uint32_t db = 0;
                std::memcpy(&db, &dec, 4);
                if (db == vb) {
                    return {fac, exp};
                }
            }
        }
        return {};
    }
    long long best_cost = std::numeric_limits<long long>::max();
    int best_fac = 0;
    int best_exp = 0;
    for (int exp = kMaxExponent; exp >= 0; exp--) {
        for (int fac = std::min(exp, kMaxFactorIdx); fac >= 0; fac--) {
            int non_exc = 0;
            int min_e = std::numeric_limits<int>::max();
            int max_e = std::numeric_limits<int>::min();
            bool bad = false;
            for (int i = 0; i < n; i++) {
                float v = values[static_cast<std::size_t>(i)];
                if (impossible_to_encode(v)) {
                    bad = true;
                    break;
                }
                int enc = encode_value(v, fac, exp);
                float dec = decode_value(enc, fac, exp);
                std::uint32_t db = 0, vb = 0;
                std::memcpy(&db, &dec, 4);
                std::memcpy(&vb, &v, 4);
                if (db == vb) {
                    non_exc++;
                    if (enc < min_e) {
                        min_e = enc;
                    }
                    if (enc > max_e) {
                        max_e = enc;
                    }
                }
            }
            if (bad || non_exc < 2) {
                continue;
            }
            int bits = bit_width_unsigned_span(min_e, max_e);
            long long cost = static_cast<long long>(n) * bits +
                               static_cast<long long>(n - non_exc) *
                                   (kExceptionSize + kExceptionPositionSize);
            if (cost < best_cost) {
                best_cost = cost;
                best_fac = fac;
                best_exp = exp;
            }
        }
    }
    if (best_cost == std::numeric_limits<long long>::max()) {
        return {};
    }
    return {best_fac, best_exp};
}

} // namespace

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                               std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const int col_n = static_cast<int>(col.size());
        if (col_n == 0) {
            continue;
        }
        orig += static_cast<long long>(benchmark_wire::column_original_size(col_n, true));
        long long col_comp = 0, col_ct = 0, col_dt = 0;
        bool col_ok = true;
        for (int off = 0; off < col_n && col_ok;) {
            int vec_len = std::min(1024, col_n - off);
            std::vector<float> window(static_cast<std::size_t>(vec_len));
            for (int i = 0; i < vec_len; i++) {
                window[static_cast<std::size_t>(i)] = col[static_cast<std::size_t>(off + i)];
            }
            auto fe = find_best_factor_exponent(window.data(), vec_len);
            if (static_cast<int>(fe.size()) < 2) {
                col_ok = false;
                break;
            }
            int fac = fe[0];
            int exp = fe[1];
            std::vector<int> encoded(static_cast<std::size_t>(vec_len));
            std::vector<int> exc_pos(static_cast<std::size_t>(vec_len));
            std::vector<float> exc_val(static_cast<std::size_t>(vec_len));
            int exc_count = 0;
            int placeholder = 0;
            bool found_ph = false;
            for (int i = 0; i < vec_len; i++) {
                float v = window[static_cast<std::size_t>(i)];
                int enc = encode_value(v, fac, exp);
                float dec = decode_value(enc, fac, exp);
                std::uint32_t db = 0, vb = 0;
                std::memcpy(&db, &dec, 4);
                std::memcpy(&vb, &v, 4);
                if (db == vb) {
                    encoded[static_cast<std::size_t>(i)] = enc;
                    if (!found_ph) {
                        placeholder = enc;
                        found_ph = true;
                    }
                } else {
                    exc_pos[static_cast<std::size_t>(exc_count)] = i;
                    exc_val[static_cast<std::size_t>(exc_count)] = v;
                    exc_count++;
                }
            }
            if (!found_ph) {
                col_ok = false;
                break;
            }
            for (int j = 0; j < exc_count; j++) {
                encoded[static_cast<std::size_t>(exc_pos[static_cast<std::size_t>(j)])] = placeholder;
            }
            try {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::int32_t> enc_i32(vec_len);
                for (int i = 0; i < vec_len; i++) {
                    enc_i32[static_cast<std::size_t>(i)] = encoded[static_cast<std::size_t>(i)];
                }
                auto compressed = java_fastpfor::compressIntBits(enc_i32.data(), enc_i32.size());
                auto t1 = std::chrono::high_resolution_clock::now();
                std::vector<std::int32_t> recovered_enc(static_cast<std::size_t>(vec_len));
                auto t2 = std::chrono::high_resolution_clock::now();
                java_fastpfor::uncompressIntBits(compressed.data(), compressed.size(), recovered_enc.data(),
                                                  recovered_enc.size());
                auto t3 = std::chrono::high_resolution_clock::now();
                if (static_cast<int>(recovered_enc.size()) != vec_len) {
                    col_ok = false;
                    break;
                }
                std::vector<float> round_trip(static_cast<std::size_t>(vec_len));
                for (int i = 0; i < vec_len; i++) {
                    round_trip[static_cast<std::size_t>(i)] =
                        decode_value(recovered_enc[static_cast<std::size_t>(i)], fac, exp);
                }
                for (int j = 0; j < exc_count; j++) {
                    round_trip[static_cast<std::size_t>(exc_pos[static_cast<std::size_t>(j)])] =
                        exc_val[static_cast<std::size_t>(j)];
                }
                for (int i = 0; i < vec_len; i++) {
                    std::uint32_t a = 0, b = 0;
                    std::memcpy(&a, &round_trip[static_cast<std::size_t>(i)], 4);
                    std::memcpy(&b, &window[static_cast<std::size_t>(i)], 4);
                    if (a != b) {
                        col_ok = false;
                        break;
                    }
                }
                if (!col_ok) {
                    break;
                }
                if (loss != nullptr) {
                    std::vector<double> orig_d(window.begin(), window.end());
                    std::vector<double> dec_d(round_trip.begin(), round_trip.end());
                    benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
                }
                col_comp += static_cast<long long>(compressed.size()) * 4LL + 6LL +
                            static_cast<long long>(exc_count) * (2LL + 4LL);
                col_ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                col_dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            } catch (...) {
                col_ok = false;
                break;
            }
            off += vec_len;
        }
        if (!col_ok) {
            continue;
        }
        comp += col_comp;
        ct += col_ct;
        dt += col_dt;
    }
    result.assign({orig, comp, ct, dt});
}

namespace {

void push_u16_le(std::vector<std::uint8_t> &o, std::uint16_t v) {
    o.push_back(static_cast<std::uint8_t>(v & 0xFF));
    o.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
}

void push_u32_le(std::vector<std::uint8_t> &o, std::uint32_t v) {
    o.push_back(static_cast<std::uint8_t>(v & 0xFF));
    o.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
    o.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFF));
    o.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFF));
}

std::uint16_t read_u16_le(const std::uint8_t *p) {
    return static_cast<std::uint16_t>(static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8));
}

std::uint32_t read_u32_le(const std::uint8_t *p) {
    return static_cast<std::uint32_t>(static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8) |
                                      (static_cast<unsigned>(p[2]) << 16) | (static_cast<unsigned>(p[3]) << 24));
}

bool append_one_window(std::vector<std::uint8_t> &out, const float *col, int off, int col_n) {
    int vec_len = std::min(1024, col_n - off);
    std::vector<float> window(static_cast<std::size_t>(vec_len));
    for (int i = 0; i < vec_len; i++) {
        window[static_cast<std::size_t>(i)] = col[off + i];
    }
    auto fe = find_best_factor_exponent(window.data(), vec_len);
    if (static_cast<int>(fe.size()) < 2) {
        return false;
    }
    int fac = fe[0];
    int exp = fe[1];
    std::vector<int> encoded(static_cast<std::size_t>(vec_len));
    std::vector<int> exc_pos(static_cast<std::size_t>(vec_len));
    std::vector<float> exc_val(static_cast<std::size_t>(vec_len));
    int exc_count = 0;
    int placeholder = 0;
    bool found_ph = false;
    for (int i = 0; i < vec_len; i++) {
        float v = window[static_cast<std::size_t>(i)];
        int enc = encode_value(v, fac, exp);
        float dec = decode_value(enc, fac, exp);
        std::uint32_t db = 0, vb = 0;
        std::memcpy(&db, &dec, 4);
        std::memcpy(&vb, &v, 4);
        if (db == vb) {
            encoded[static_cast<std::size_t>(i)] = enc;
            if (!found_ph) {
                placeholder = enc;
                found_ph = true;
            }
        } else {
            exc_pos[static_cast<std::size_t>(exc_count)] = i;
            exc_val[static_cast<std::size_t>(exc_count)] = v;
            exc_count++;
        }
    }
    if (!found_ph) {
        return false;
    }
    for (int j = 0; j < exc_count; j++) {
        encoded[static_cast<std::size_t>(exc_pos[static_cast<std::size_t>(j)])] = placeholder;
    }
    std::vector<std::int32_t> enc_i32(static_cast<std::size_t>(vec_len));
    for (int i = 0; i < vec_len; i++) {
        enc_i32[static_cast<std::size_t>(i)] = encoded[static_cast<std::size_t>(i)];
    }
    std::vector<std::uint32_t> compressed;
    std::vector<std::int32_t> recovered_enc(static_cast<std::size_t>(vec_len));
    try {
        compressed = java_fastpfor::compressIntBits(enc_i32.data(), enc_i32.size());
        java_fastpfor::uncompressIntBits(compressed.data(), compressed.size(), recovered_enc.data(),
                                          recovered_enc.size());
    } catch (...) {
        return false;
    }
    if (static_cast<int>(recovered_enc.size()) != vec_len) {
        return false;
    }
    std::vector<float> round_trip(static_cast<std::size_t>(vec_len));
    for (int i = 0; i < vec_len; i++) {
        round_trip[static_cast<std::size_t>(i)] = decode_value(recovered_enc[static_cast<std::size_t>(i)], fac, exp);
    }
    for (int j = 0; j < exc_count; j++) {
        round_trip[static_cast<std::size_t>(exc_pos[static_cast<std::size_t>(j)])] =
            exc_val[static_cast<std::size_t>(j)];
    }
    for (int i = 0; i < vec_len; i++) {
        std::uint32_t a = 0, b = 0;
        std::memcpy(&a, &round_trip[static_cast<std::size_t>(i)], 4);
        std::memcpy(&b, &window[static_cast<std::size_t>(i)], 4);
        if (a != b) {
            return false;
        }
    }
    out.push_back(static_cast<std::uint8_t>(fac));
    out.push_back(static_cast<std::uint8_t>(exp));
    push_u16_le(out, static_cast<std::uint16_t>(vec_len));
    push_u16_le(out, static_cast<std::uint16_t>(exc_count));
    push_u32_le(out, static_cast<std::uint32_t>(compressed.size()));
    for (std::uint32_t w : compressed) {
        push_u32_le(out, w);
    }
    for (int j = 0; j < exc_count; j++) {
        push_u16_le(out, static_cast<std::uint16_t>(exc_pos[static_cast<std::size_t>(j)]));
        std::uint32_t bits = 0;
        std::memcpy(&bits, &exc_val[static_cast<std::size_t>(j)], 4);
        push_u32_le(out, bits);
    }
    return true;
}

} // namespace

std::vector<std::uint8_t> goldenSerializeColumn(const std::vector<float> &col) {
    std::vector<std::uint8_t> out;
    int col_n = static_cast<int>(col.size());
    for (int off = 0; off < col_n;) {
        if (!append_one_window(out, col.data(), off, col_n)) {
            return {};
        }
        off += std::min(1024, col_n - off);
    }
    return out;
}

bool goldenDeserializeColumn(std::span<const std::uint8_t> bytes, std::vector<float> &out) {
    out.clear();
    std::size_t pos = 0;
    while (pos < bytes.size()) {
        if (pos + 10 > bytes.size()) {
            return false;
        }
        int fac = static_cast<int>(bytes[pos++]);
        int exp = static_cast<int>(bytes[pos++]);
        int vec_len = static_cast<int>(read_u16_le(bytes.data() + pos));
        pos += 2;
        int exc_count = static_cast<int>(read_u16_le(bytes.data() + pos));
        pos += 2;
        std::uint32_t n_words = read_u32_le(bytes.data() + pos);
        pos += 4;
        if (pos + static_cast<std::size_t>(n_words) * 4 > bytes.size()) {
            return false;
        }
        std::vector<std::uint32_t> cw(n_words);
        std::memcpy(cw.data(), bytes.data() + pos, static_cast<std::size_t>(n_words) * 4);
        pos += static_cast<std::size_t>(n_words) * 4;
        if (vec_len <= 0 || exc_count < 0 || static_cast<std::size_t>(exc_count) * 6 > bytes.size() - pos) {
            return false;
        }
        if (pos + static_cast<std::size_t>(exc_count) * 6 > bytes.size()) {
            return false;
        }
        std::vector<std::int32_t> recovered_enc(static_cast<std::size_t>(vec_len));
        try {
            java_fastpfor::uncompressIntBits(cw.data(), cw.size(), recovered_enc.data(),
                                               static_cast<std::size_t>(vec_len));
        } catch (...) {
            return false;
        }
        std::vector<float> window(static_cast<std::size_t>(vec_len));
        for (int i = 0; i < vec_len; i++) {
            window[static_cast<std::size_t>(i)] = decode_value(recovered_enc[static_cast<std::size_t>(i)], fac, exp);
        }
        for (int j = 0; j < exc_count; j++) {
            int idx = static_cast<int>(read_u16_le(bytes.data() + pos));
            pos += 2;
            std::uint32_t bits = read_u32_le(bytes.data() + pos);
            pos += 4;
            if (idx < 0 || idx >= vec_len) {
                return false;
            }
            float fv = 0;
            std::memcpy(&fv, &bits, 4);
            window[static_cast<std::size_t>(idx)] = fv;
        }
        out.insert(out.end(), window.begin(), window.end());
    }
    return pos == bytes.size();
}

namespace {
constexpr int d64MaxExponent = 10;
constexpr double d64MagicNumber = 6755399441055744.0;
constexpr int d64ExceptionPositionSize = 16;
constexpr int d64ExceptionValueSize = 64;
constexpr double d64EncodingUpperLimit = 9223372036854774784.0;
constexpr double d64EncodingLowerLimit = -9223372036854774784.0;
alignas(64) static constexpr double d64FracArr[] = {
    1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001,
};
alignas(64) static constexpr double d64ExpArr[] = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0,
    10000000000.0,
};
alignas(64) static constexpr std::int64_t d64FactArr[] = {
    1LL, 10LL, 100LL, 1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 100000000LL, 1000000000LL,
};
constexpr int d64MaxFactorIdx = static_cast<int>(sizeof(d64FactArr) / sizeof(d64FactArr[0])) - 1;

bool d64_impossible_to_encode(double n) {
    if (!std::isfinite(n)) {
        return true;
    }
    return n > d64EncodingUpperLimit || n < d64EncodingLowerLimit;
}

std::uint64_t d64_raw_bits(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

std::int64_t d64_encode_value(double value, int factor_idx, int exponent_idx) {
    double tmp = value * d64ExpArr[static_cast<std::size_t>(exponent_idx)] *
                 d64FracArr[static_cast<std::size_t>(factor_idx)];
    if (d64_impossible_to_encode(tmp)) {
        return std::numeric_limits<std::int64_t>::max();
    }
    tmp = tmp + d64MagicNumber - d64MagicNumber;
    return static_cast<std::int64_t>(tmp);
}

double d64_decode_value(std::int64_t encoded, int factor_idx, int exponent_idx) {
    return static_cast<double>(encoded) * static_cast<double>(d64FactArr[static_cast<std::size_t>(factor_idx)]) *
           d64FracArr[static_cast<std::size_t>(exponent_idx)];
}

int d64_bit_width_unsigned_span(std::int64_t min_enc, std::int64_t max_enc) {
    const std::uint64_t umin = static_cast<std::uint64_t>(min_enc);
    const std::uint64_t umax = static_cast<std::uint64_t>(max_enc);
    const std::uint64_t delta = umax - umin;
    if (delta == 0) {
        return 0;
    }
    return 64 - static_cast<int>(__builtin_clzll(delta));
}

std::vector<int> d64_find_best_factor_exponent(const double *values, int n) {
    if (n == 1) {
        const double v = values[0];
        if (d64_impossible_to_encode(v)) {
            return {};
        }
        const std::uint64_t vb = d64_raw_bits(v);
        for (int exp = d64MaxExponent; exp >= 0; --exp) {
            for (int fac = std::min(exp, d64MaxFactorIdx); fac >= 0; --fac) {
                const std::int64_t enc = d64_encode_value(v, fac, exp);
                const double dec = d64_decode_value(enc, fac, exp);
                if (d64_raw_bits(dec) == vb) {
                    return {fac, exp};
                }
            }
        }
        return {};
    }
    long long best_cost = std::numeric_limits<long long>::max();
    int best_fac = 0;
    int best_exp = 0;
    for (int exp = d64MaxExponent; exp >= 0; --exp) {
        for (int fac = std::min(exp, d64MaxFactorIdx); fac >= 0; --fac) {
            int non_exc = 0;
            std::int64_t min_e = -1;
            std::int64_t max_e = 0;
            bool bad = false;
            for (int i = 0; i < n; ++i) {
                const double v = values[static_cast<std::size_t>(i)];
                if (d64_impossible_to_encode(v)) {
                    bad = true;
                    break;
                }
                const std::int64_t enc = d64_encode_value(v, fac, exp);
                const double dec = d64_decode_value(enc, fac, exp);
                if (d64_raw_bits(dec) == d64_raw_bits(v)) {
                    ++non_exc;
                    if (static_cast<std::uint64_t>(enc) < static_cast<std::uint64_t>(min_e)) {
                        min_e = enc;
                    }
                    if (static_cast<std::uint64_t>(enc) > static_cast<std::uint64_t>(max_e)) {
                        max_e = enc;
                    }
                }
            }
            if (bad || non_exc < 2) {
                continue;
            }
            const int bits = d64_bit_width_unsigned_span(min_e, max_e);
            const long long cost = static_cast<long long>(n) * bits +
                                   static_cast<long long>(n - non_exc) *
                                       (d64ExceptionValueSize + d64ExceptionPositionSize);
            if (cost < best_cost) {
                best_cost = cost;
                best_fac = fac;
                best_exp = exp;
            }
        }
    }
    if (best_cost == std::numeric_limits<long long>::max()) {
        return {};
    }
    return {best_fac, best_exp};
}

bool d64_encode_raw_bits_block(const std::vector<double> &window, long long &block_comp, long long &block_ct,
                               long long &block_dt) {
    const std::size_t n = window.size();
    if (n == 0) {
        return false;
    }
    std::vector<std::int64_t> raw_bits(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t bits = d64_raw_bits(window[i]);
        raw_bits[i] = static_cast<std::int64_t>(bits);
    }
    try {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto payload = int64_column_codec::fast_pfor_encode_int64(raw_bits.data(), raw_bits.size());
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto t2 = std::chrono::high_resolution_clock::now();
        const auto recovered =
            int64_column_codec::fast_pfor_decode_int64(payload.data(), payload.size(), raw_bits.size());
        const auto t3 = std::chrono::high_resolution_clock::now();
        if (recovered.size() != raw_bits.size()) {
            return false;
        }
        if (!std::equal(recovered.begin(), recovered.end(), raw_bits.begin())) {
            return false;
        }
        block_comp = static_cast<long long>(payload.size());
        block_ct = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        block_dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        return true;
    } catch (...) {
        return false;
    }
}
} // namespace

std::vector<std::uint8_t> encodeDoubleColumnExact(const std::vector<double> &values) {
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("ALP2 input exceeds uint32 count");
    }
    std::vector<std::uint8_t> out{'A', 'L', 'P', '2'};
    push_u32_le(out, static_cast<std::uint32_t>(values.size()));
    for (std::size_t offset = 0; offset < values.size(); offset += 1024) {
        const std::size_t count = std::min<std::size_t>(1024, values.size() - offset);
        const std::vector<double> window(
            values.begin() + static_cast<std::ptrdiff_t>(offset),
            values.begin() + static_cast<std::ptrdiff_t>(offset + count));
        const auto choice = d64_find_best_factor_exponent(
            window.data(), static_cast<int>(count));
        if (choice.size() < 2) {
            out.push_back(0);
            push_u16_le(out, static_cast<std::uint16_t>(count));
            push_u32_le(out, static_cast<std::uint32_t>(count * sizeof(double)));
            const auto* begin = reinterpret_cast<const std::uint8_t*>(window.data());
            out.insert(out.end(), begin, begin + count * sizeof(double));
            continue;
        }
        const int factor = choice[0];
        const int exponent = choice[1];
        std::vector<std::int64_t> encoded(count);
        std::vector<std::pair<std::uint16_t, std::uint64_t>> exceptions;
        std::int64_t placeholder = 0;
        bool found_placeholder = false;
        for (std::size_t i = 0; i < count; ++i) {
            const auto value = d64_encode_value(window[i], factor, exponent);
            if (d64_raw_bits(d64_decode_value(value, factor, exponent)) ==
                d64_raw_bits(window[i])) {
                encoded[i] = value;
                if (!found_placeholder) {
                    placeholder = value;
                    found_placeholder = true;
                }
            } else {
                exceptions.push_back({static_cast<std::uint16_t>(i),
                                      d64_raw_bits(window[i])});
            }
        }
        if (!found_placeholder) {
            out.push_back(0);
            push_u16_le(out, static_cast<std::uint16_t>(count));
            push_u32_le(out, static_cast<std::uint32_t>(count * sizeof(double)));
            const auto* begin = reinterpret_cast<const std::uint8_t*>(window.data());
            out.insert(out.end(), begin, begin + count * sizeof(double));
            continue;
        }
        for (const auto& exception : exceptions) encoded[exception.first] = placeholder;
        const auto body = int64_column_codec::fast_pfor_encode_int64(
            encoded.data(), encoded.size());
        out.push_back(1);
        push_u16_le(out, static_cast<std::uint16_t>(count));
        out.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(factor)));
        out.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(exponent)));
        push_u32_le(out, static_cast<std::uint32_t>(body.size()));
        push_u16_le(out, static_cast<std::uint16_t>(exceptions.size()));
        out.insert(out.end(), body.begin(), body.end());
        for (const auto& [index, bits] : exceptions) {
            push_u16_le(out, index);
            for (int shift = 0; shift < 64; shift += 8) {
                out.push_back(static_cast<std::uint8_t>(bits >> shift));
            }
        }
    }
    return out;
}

std::vector<double> decodeDoubleColumnExact(std::span<const std::uint8_t> wire) {
    if (wire.size() < 8 || std::memcmp(wire.data(), "ALP2", 4) != 0) {
        throw std::invalid_argument("invalid ALP2 wire");
    }
    const std::uint32_t total = read_u32_le(wire.data() + 4);
    std::size_t pos = 8;
    std::vector<double> values;
    values.reserve(total);
    while (values.size() < total) {
        if (pos + 7 > wire.size()) throw std::runtime_error("truncated ALP2 block");
        const std::uint8_t mode = wire[pos++];
        const std::uint16_t count = read_u16_le(wire.data() + pos);
        pos += 2;
        if (count == 0 || values.size() + count > total) {
            throw std::runtime_error("invalid ALP2 block count");
        }
        if (mode == 0) {
            const std::uint32_t size = read_u32_le(wire.data() + pos);
            pos += 4;
            if (size != static_cast<std::uint32_t>(count) * sizeof(double) ||
                pos + size > wire.size()) {
                throw std::runtime_error("invalid ALP2 raw block");
            }
            const std::size_t old_size = values.size();
            values.resize(old_size + count);
            std::memcpy(values.data() + old_size, wire.data() + pos, size);
            pos += size;
            continue;
        }
        if (mode != 1 || pos + 8 > wire.size()) {
            throw std::runtime_error("invalid ALP2 scaled block");
        }
        const int factor = static_cast<std::int8_t>(wire[pos++]);
        const int exponent = static_cast<std::int8_t>(wire[pos++]);
        const std::uint32_t body_size = read_u32_le(wire.data() + pos);
        pos += 4;
        const std::uint16_t exception_count = read_u16_le(wire.data() + pos);
        pos += 2;
        if (factor < 0 || factor > d64MaxFactorIdx || exponent < 0 ||
            exponent > d64MaxExponent || pos + body_size > wire.size()) {
            throw std::runtime_error("invalid ALP2 scaled header");
        }
        const auto decoded = int64_column_codec::fast_pfor_decode_int64(
            wire.data() + pos, body_size, count);
        pos += body_size;
        if (decoded.size() != count) throw std::runtime_error("ALP2 decoded count mismatch");
        std::vector<double> block(count);
        for (std::size_t i = 0; i < count; ++i) {
            block[i] = d64_decode_value(decoded[i], factor, exponent);
        }
        for (std::uint16_t i = 0; i < exception_count; ++i) {
            if (pos + 10 > wire.size()) throw std::runtime_error("truncated ALP2 exception");
            const std::uint16_t index = read_u16_le(wire.data() + pos);
            pos += 2;
            std::uint64_t bits = 0;
            for (int shift = 0; shift < 64; shift += 8) {
                bits |= static_cast<std::uint64_t>(wire[pos++]) << shift;
            }
            if (index >= count) throw std::runtime_error("ALP2 exception index out of range");
            std::memcpy(&block[index], &bits, sizeof(bits));
        }
        values.insert(values.end(), block.begin(), block.end());
    }
    if (pos != wire.size()) throw std::runtime_error("ALP2 trailing bytes");
    return values;
}

void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const int col_n = static_cast<int>(col.size());
        if (col_n == 0) {
            continue;
        }
        long long col_comp = 0, col_ct = 0, col_dt = 0;
        bool col_ok = true;
        for (int off = 0; off < col_n && col_ok;) {
            const int vec_len = std::min(1024, col_n - off);
            std::vector<double> window(static_cast<std::size_t>(vec_len));
            for (int i = 0; i < vec_len; ++i) {
                window[static_cast<std::size_t>(i)] = col[static_cast<std::size_t>(off + i)];
            }
            const auto fe = d64_find_best_factor_exponent(window.data(), vec_len);
            if (static_cast<int>(fe.size()) < 2) {
                long long fb_comp = 0, fb_ct = 0, fb_dt = 0;
                if (!d64_encode_raw_bits_block(window, fb_comp, fb_ct, fb_dt)) {
                    col_ok = false;
                    break;
                }
                col_comp += fb_comp;
                col_ct += fb_ct;
                col_dt += fb_dt;
                off += vec_len;
                continue;
            }
            const int fac = fe[0];
            const int exp = fe[1];
            std::vector<std::int64_t> encoded(static_cast<std::size_t>(vec_len));
            std::vector<int> exc_pos(static_cast<std::size_t>(vec_len));
            std::vector<double> exc_val(static_cast<std::size_t>(vec_len));
            int exc_count = 0;
            std::int64_t placeholder = 0;
            bool found_placeholder = false;
            for (int i = 0; i < vec_len; ++i) {
                const double v = window[static_cast<std::size_t>(i)];
                const std::int64_t enc = d64_encode_value(v, fac, exp);
                const double dec = d64_decode_value(enc, fac, exp);
                if (d64_raw_bits(dec) == d64_raw_bits(v)) {
                    encoded[static_cast<std::size_t>(i)] = enc;
                    if (!found_placeholder) {
                        placeholder = enc;
                        found_placeholder = true;
                    }
                } else {
                    exc_pos[static_cast<std::size_t>(exc_count)] = i;
                    exc_val[static_cast<std::size_t>(exc_count)] = v;
                    ++exc_count;
                }
            }
            if (!found_placeholder) {
                long long fb_comp = 0, fb_ct = 0, fb_dt = 0;
                if (!d64_encode_raw_bits_block(window, fb_comp, fb_ct, fb_dt)) {
                    col_ok = false;
                    break;
                }
                col_comp += fb_comp;
                col_ct += fb_ct;
                col_dt += fb_dt;
                off += vec_len;
                continue;
            }
            for (int j = 0; j < exc_count; ++j) {
                encoded[static_cast<std::size_t>(exc_pos[static_cast<std::size_t>(j)])] = placeholder;
            }
            bool block_ok = true;
            long long block_comp = 0, block_ct = 0, block_dt = 0;
            try {
                const auto t0 = std::chrono::high_resolution_clock::now();
                const auto payload = int64_column_codec::fast_pfor_encode_int64(encoded.data(), encoded.size());
                const auto t1 = std::chrono::high_resolution_clock::now();
                const auto t2 = std::chrono::high_resolution_clock::now();
                const auto recovered_enc =
                    int64_column_codec::fast_pfor_decode_int64(payload.data(), payload.size(), encoded.size());
                const auto t3 = std::chrono::high_resolution_clock::now();
                std::vector<double> round_trip(static_cast<std::size_t>(vec_len));
                if (recovered_enc.size() != encoded.size()) {
                    block_ok = false;
                } else {
                    for (int i = 0; i < vec_len; ++i) {
                        round_trip[static_cast<std::size_t>(i)] =
                            d64_decode_value(recovered_enc[static_cast<std::size_t>(i)], fac, exp);
                    }
                    for (int j = 0; j < exc_count; ++j) {
                        round_trip[static_cast<std::size_t>(exc_pos[static_cast<std::size_t>(j)])] =
                            exc_val[static_cast<std::size_t>(j)];
                    }
                    for (int i = 0; i < vec_len; ++i) {
                        if (d64_raw_bits(round_trip[static_cast<std::size_t>(i)]) !=
                            d64_raw_bits(window[static_cast<std::size_t>(i)])) {
                            block_ok = false;
                            break;
                        }
                    }
                }
                if (block_ok) {
                    if (loss != nullptr) {
                        benchmark_loss::accumulate_double_column(window, round_trip, *loss);
                    }
                    block_comp = static_cast<long long>(payload.size()) + 6LL +
                                 static_cast<long long>(exc_count) * (2LL + 8LL);
                    block_ct = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    block_dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
                }
            } catch (...) {
                block_ok = false;
            }
            if (!block_ok) {
                long long fb_comp = 0, fb_ct = 0, fb_dt = 0;
                if (!d64_encode_raw_bits_block(window, fb_comp, fb_ct, fb_dt)) {
                    col_ok = false;
                    break;
                }
                col_comp += fb_comp;
                col_ct += fb_ct;
                col_dt += fb_dt;
                off += vec_len;
                continue;
            }
            col_comp += block_comp;
            col_ct += block_ct;
            col_dt += block_dt;
            off += vec_len;
        }
        if (!col_ok) {
            continue;
        }
        orig += static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
        comp += col_comp;
        ct += col_ct;
        dt += col_dt;
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
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto payload = int64_column_codec::fast_pfor_encode_int64(col.data(), n);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const auto t2 = std::chrono::high_resolution_clock::now();
            const auto recovered = int64_column_codec::fast_pfor_decode_int64(payload.data(), payload.size(), n);
            const auto t3 = std::chrono::high_resolution_clock::now();
            if (recovered.size() != n || !std::equal(recovered.begin(), recovered.end(), col.begin())) {
                continue;
            }
            if (loss != nullptr) {
                benchmark_loss::accumulate_int64_column(col, recovered, *loss);
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

namespace {

enum class AlpAblationMode { Baseline, NoScaling, NoFor, NoBitPacking };

AlpAblationMode alp_ablation_mode(std::string_view omitted_operator) {
    if (omitted_operator.empty() || omitted_operator == "none") return AlpAblationMode::Baseline;
    if (omitted_operator == "scaling") return AlpAblationMode::NoScaling;
    if (omitted_operator == "for") return AlpAblationMode::NoFor;
    if (omitted_operator == "bit-packing") return AlpAblationMode::NoBitPacking;
    throw std::invalid_argument("unsupported ALP ablation operator");
}

std::uint64_t alp_zigzag(std::int64_t value) {
    const auto bits = static_cast<std::uint64_t>(value);
    return (bits << 1) ^ static_cast<std::uint64_t>(value >> 63);
}

std::int64_t alp_unzigzag(std::uint64_t value) {
    return static_cast<std::int64_t>((value >> 1) ^ (0 - (value & 1)));
}

int alp_width(std::uint64_t value) {
    return value == 0 ? 0 : 64 - static_cast<int>(__builtin_clzll(value));
}

std::vector<std::uint8_t> alp_pack(const std::vector<std::uint64_t> &values, int width) {
    if (values.empty() || width == 0) return {};
    if (width == 64) {
        std::vector<std::uint8_t> out(values.size() * 8);
        std::memcpy(out.data(), values.data(), out.size());
        return out;
    }
    std::vector<std::uint8_t> out((values.size() * static_cast<std::size_t>(width) + 7) / 8, 0);
    std::size_t bit = 0;
    for (std::uint64_t value : values) {
        for (int shift = width - 1; shift >= 0; --shift, ++bit) {
            out[bit / 8] |= static_cast<std::uint8_t>(((value >> shift) & 1u) << (7 - bit % 8));
        }
    }
    return out;
}

std::vector<std::uint64_t> alp_unpack(const std::vector<std::uint8_t> &bytes,
                                      std::size_t count, int width) {
    std::vector<std::uint64_t> values(count, 0);
    if (width == 0) return values;
    const std::size_t required = (count * static_cast<std::size_t>(width) + 7) / 8;
    if (bytes.size() != required) throw std::runtime_error("ALP ablation packed-size mismatch");
    if (width == 64) {
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    }
    std::size_t bit = 0;
    for (auto &value : values) {
        for (int i = 0; i < width; ++i, ++bit) {
            value = (value << 1) | ((bytes[bit / 8] >> (7 - bit % 8)) & 1u);
        }
    }
    return values;
}

struct AlpIntegerTrial {
    std::size_t bytes = 0;
    std::vector<std::int64_t> decoded;
};

AlpIntegerTrial alp_integer_trial(const std::vector<std::int64_t> &values,
                                  AlpAblationMode mode) {
    if (mode == AlpAblationMode::Baseline || mode == AlpAblationMode::NoScaling) {
        const auto payload = int64_column_codec::fast_pfor_encode_int64(values.data(), values.size());
        return {payload.size(), int64_column_codec::fast_pfor_decode_int64(
                                    payload.data(), payload.size(), values.size())};
    }
    if (mode == AlpAblationMode::NoBitPacking) {
        std::vector<std::uint8_t> payload(values.size() * sizeof(std::int64_t));
        if (!payload.empty()) std::memcpy(payload.data(), values.data(), payload.size());
        std::vector<std::int64_t> decoded(values.size());
        if (!payload.empty()) std::memcpy(decoded.data(), payload.data(), payload.size());
        return {payload.size(), std::move(decoded)};
    }
    std::vector<std::uint64_t> encoded;
    encoded.reserve(values.size());
    int width = 0;
    for (std::int64_t value : values) {
        const std::uint64_t code = alp_zigzag(value);
        encoded.push_back(code);
        width = std::max(width, alp_width(code));
    }
    const auto payload = alp_pack(encoded, width);
    const auto recovered = alp_unpack(payload, values.size(), width);
    std::vector<std::int64_t> decoded;
    decoded.reserve(values.size());
    for (std::uint64_t code : recovered) decoded.push_back(alp_unzigzag(code));
    return {payload.size() + 1, std::move(decoded)};
}

} // namespace

void benchInt64ColumnsAblation(const std::vector<std::vector<std::int64_t>> &columns,
                               std::string_view omitted_operator,
                               std::vector<long long> &result, LossAccum *loss) {
    const auto mode = alp_ablation_mode(omitted_operator);
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &column : columns) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto trial = alp_integer_trial(column, mode);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const bool ok = trial.decoded == column;
        const auto t2 = std::chrono::high_resolution_clock::now();
        if (!ok) throw std::runtime_error("ALP integer ablation round-trip mismatch");
        if (loss != nullptr) benchmark_loss::accumulate_int64_column(column, trial.decoded, *loss);
        orig += static_cast<long long>(benchmark_wire::column_original_size(column.size(), false));
        comp += static_cast<long long>(trial.bytes);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({orig, comp, ct, dt});
}

void benchDoubleColumnsAblation(const std::vector<std::vector<double>> &columns,
                                std::string_view omitted_operator,
                                std::vector<long long> &result, LossAccum *loss) {
    const auto mode = alp_ablation_mode(omitted_operator);
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &column : columns) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<double> decoded;
        decoded.reserve(column.size());
        std::size_t column_bytes = 0;
        for (std::size_t offset = 0; offset < column.size(); offset += 1024) {
            const std::size_t count = std::min<std::size_t>(1024, column.size() - offset);
            std::vector<double> window(column.begin() + static_cast<std::ptrdiff_t>(offset),
                                       column.begin() + static_cast<std::ptrdiff_t>(offset + count));
            std::vector<std::int64_t> encoded(count);
            std::vector<std::pair<std::uint16_t, double>> exceptions;
            int factor = 0;
            int exponent = 0;
            bool scaled = mode != AlpAblationMode::NoScaling;
            if (scaled) {
                const auto choice = d64_find_best_factor_exponent(window.data(), static_cast<int>(count));
                if (choice.size() < 2) scaled = false;
                else {
                    factor = choice[0];
                    exponent = choice[1];
                    std::int64_t placeholder = 0;
                    bool found = false;
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto value = d64_encode_value(window[i], factor, exponent);
                        if (d64_raw_bits(d64_decode_value(value, factor, exponent)) ==
                            d64_raw_bits(window[i])) {
                            encoded[i] = value;
                            if (!found) {
                                placeholder = value;
                                found = true;
                            }
                        } else {
                            exceptions.push_back({static_cast<std::uint16_t>(i), window[i]});
                        }
                    }
                    if (!found) scaled = false;
                    else for (const auto &[index, value] : exceptions) {
                        (void)value;
                        encoded[index] = placeholder;
                    }
                }
            }
            if (!scaled) {
                exceptions.clear();
                for (std::size_t i = 0; i < count; ++i) {
                    encoded[i] = static_cast<std::int64_t>(d64_raw_bits(window[i]));
                }
            }
            const auto trial = alp_integer_trial(encoded, mode);
            if (trial.decoded.size() != count) {
                throw std::runtime_error("ALP double ablation decoded count mismatch");
            }
            std::vector<double> recovered(count);
            if (scaled) {
                for (std::size_t i = 0; i < count; ++i) {
                    recovered[i] = d64_decode_value(trial.decoded[i], factor, exponent);
                }
                for (const auto &[index, value] : exceptions) recovered[index] = value;
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const auto bits = static_cast<std::uint64_t>(trial.decoded[i]);
                    std::memcpy(&recovered[i], &bits, sizeof(bits));
                }
            }
            for (std::size_t i = 0; i < count; ++i) {
                if (d64_raw_bits(recovered[i]) != d64_raw_bits(window[i])) {
                    throw std::runtime_error("ALP double ablation round-trip mismatch");
                }
            }
            column_bytes += trial.bytes + 10 + exceptions.size() * 10;
            decoded.insert(decoded.end(), recovered.begin(), recovered.end());
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (decoded.size() != column.size()) {
            throw std::runtime_error("ALP double ablation column-size mismatch");
        }
        for (std::size_t i = 0; i < column.size(); ++i) {
            if (d64_raw_bits(decoded[i]) != d64_raw_bits(column[i])) {
                throw std::runtime_error("ALP double ablation column mismatch");
            }
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        if (loss != nullptr) benchmark_loss::accumulate_double_column(column, decoded, *loss);
        orig += static_cast<long long>(benchmark_wire::column_original_size(column.size(), true));
        comp += static_cast<long long>(column_bytes);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace alp_float
