#include "simple8b_int_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include "int64_column_codec.hpp"
#include "leco_double_codec.hpp"
#include "leco_float_codec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simple8b_int {
namespace {

constexpr int kLogDesc = 4;
constexpr std::int32_t kRawDoubleBitsScale = std::numeric_limits<std::int32_t>::min();
constexpr std::uint64_t kMask64 = 0xffffffffffffffffULL;

struct SelRow {
    int maxNum;
    int log;
    int sel;
};

static const SelRow kRows[] = {
    {240, 0, 0}, {120, 0, 1}, {60, 1, 2},  {30, 2, 3},   {20, 3, 4},   {15, 4, 5},   {12, 5, 6},   {10, 6, 7},
    {8, 7, 8},   {7, 8, 9},   {6, 10, 10}, {5, 12, 11},  {4, 15, 12},  {3, 20, 13},  {2, 30, 14},  {1, 60, 15},
};

static int which_u64(std::uint64_t w) { return static_cast<int>((w >> 60) & 0xfULL); }

static bool try_me(const std::uint32_t *buf, std::size_t base, int rem, int num, int log) {
    const int c = std::min(rem, num);
    if (log >= 32) {
        return true;
    }
    const std::uint32_t lim = 1u << log;
    for (int i = 0; i < c; ++i) {
        if (buf[base + static_cast<std::size_t>(i)] >= lim) {
            return false;
        }
    }
    return true;
}

static std::uint64_t compose_word(int sel, const std::uint32_t *vals, std::size_t from, int take, int log) {
    std::uint64_t w = static_cast<std::uint64_t>(sel & 0xf);
    if (log == 0) {
        if (sel == 0) {
            return 0;
        }
        if (sel == 1) {
            return 1ULL << 60;
        }
        throw std::runtime_error("simple8b: compose log=0");
    }
    std::uint64_t mask = (1ULL << log) - 1ULL;
    if (log >= 32) {
        mask = 0xffffffffULL;
    }
    for (int i = 0; i < take; ++i) {
        const std::uint64_t vv = static_cast<std::uint64_t>(vals[from + static_cast<std::size_t>(i)]) & mask;
        w = (w << log) | vv;
        w &= kMask64;
    }
    const int total_bits = kLogDesc + log * take;
    const int sh = 64 - total_bits;
    if (sh < 0 || sh > 64) {
        throw std::runtime_error("simple8b: compose shift");
    }
    w = (w << sh) & kMask64;
    return w;
}

static std::vector<std::uint64_t> encode_u32(const std::uint32_t *vals, std::size_t n) {
    std::vector<std::uint64_t> out;
    std::size_t pos = 0;
    while (pos < n) {
        const int rem = static_cast<int>(n - pos);
        bool wrote = false;
        for (const SelRow &row : kRows) {
            if (!try_me(vals, pos, rem, row.maxNum, row.log)) {
                continue;
            }
            const int take = std::min(rem, row.maxNum);
            std::uint64_t word;
            if (row.sel <= 1) {
                word = row.sel == 0 ? 0ULL : (1ULL << 60);
            } else {
                word = compose_word(row.sel, vals, pos, take, row.log);
            }
            out.push_back(word);
            pos += static_cast<std::size_t>(take);
            wrote = true;
            break;
        }
        if (!wrote) {
            throw std::runtime_error("simple8b: encode stuck");
        }
    }
    return out;
}

static const SelRow *row_for_sel(int sel) {
    for (const SelRow &r : kRows) {
        if (r.sel == sel) {
            return &r;
        }
    }
    return nullptr;
}

static void decode_u64(const std::uint64_t *words, std::size_t nw, std::size_t n_values, std::uint32_t *out) {
    std::size_t wi = 0;
    std::size_t op = 0;
    while (op < n_values) {
        if (wi >= nw) {
            throw std::runtime_error("simple8b: truncated stream");
        }
        std::uint64_t w = words[wi++] & kMask64;
        const int sel = which_u64(w);
        const int rem = static_cast<int>(n_values - op);
        if (sel == 0) {
            const int take = std::min(rem, 240);
            for (int i = 0; i < take; ++i) {
                out[op++] = 0;
            }
        } else if (sel == 1) {
            const int take = std::min(rem, 120);
            for (int i = 0; i < take; ++i) {
                out[op++] = 0;
            }
        } else {
            const SelRow *row = row_for_sel(sel);
            if (row == nullptr || sel < 2) {
                throw std::runtime_error("simple8b: bad selector");
            }
            const int maxnum = row->maxNum;
            const int log = row->log;
            const int take = std::min(rem, maxnum);
            std::uint64_t mask = (1ULL << log) - 1ULL;
            if (log >= 32) {
                mask = 0xffffffffULL;
            }
            for (int k = 0; k < take; ++k) {
                const int sh = 64 - kLogDesc - log - k * log;
                out[op++] = static_cast<std::uint32_t>((w >> sh) & mask);
            }
        }
    }
    if (op != n_values) {
        throw std::runtime_error("simple8b: length mismatch");
    }
}

static std::vector<std::uint8_t> encode_column_bytes(const std::uint32_t *u32, std::size_t n) {
    std::vector<std::uint64_t> words = encode_u32(u32, n);
    const int nw = static_cast<int>(words.size());
    std::vector<std::uint8_t> out(8 + static_cast<std::size_t>(nw) * 8);
    const std::uint32_t n32 = static_cast<std::uint32_t>(n);
    const std::uint32_t nw32 = static_cast<std::uint32_t>(nw);
    std::memcpy(out.data(), &n32, 4);
    std::memcpy(out.data() + 4, &nw32, 4);
    for (int i = 0; i < nw; ++i) {
        std::uint64_t lw = words[static_cast<std::size_t>(i)];
        std::memcpy(out.data() + 8 + static_cast<std::size_t>(i) * 8, &lw, 8);
    }
    return out;
}

static void decode_column_bytes(const std::uint8_t *comp, std::size_t comp_len, std::uint32_t *out_u32,
                                std::size_t n_expect) {
    if (comp_len < 8) {
        throw std::runtime_error("simple8b: truncated header");
    }
    std::uint32_t n = 0;
    std::uint32_t nw32 = 0;
    std::memcpy(&n, comp, 4);
    std::memcpy(&nw32, comp + 4, 4);
    const int nw = static_cast<int>(nw32);
    const std::size_t need = 8 + static_cast<std::size_t>(nw) * 8;
    if (nw < 0 || comp_len < need) {
        throw std::runtime_error("simple8b: truncated body");
    }
    std::vector<std::uint64_t> words(static_cast<std::size_t>(nw));
    for (int i = 0; i < nw; ++i) {
        std::uint64_t lw = 0;
        std::memcpy(&lw, comp + 8 + static_cast<std::size_t>(i) * 8, 8);
        words[static_cast<std::size_t>(i)] = lw;
    }
    if (static_cast<std::size_t>(n) != n_expect) {
        throw std::runtime_error("simple8b: n mismatch");
    }
    decode_u64(words.data(), words.size(), static_cast<std::size_t>(n), out_u32);
}

} // namespace

void benchFloatColumnsSimple8b(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                               LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    try {
        for (const auto &col : columns) {
            const std::size_t n = col.size();
            if (n == 0) {
                continue;
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, false));
            const int col_scale = leco_float::maxLosslessDecimalScale(col);
            if (col_scale < 0) {
                continue;
            }
            std::vector<std::int32_t> data;
            if (!leco_float::quantizeScaledInts(col, col_scale, data)) {
                continue;
            }
            std::vector<std::uint32_t> u32(n);
            for (std::size_t i = 0; i < n; ++i) {
                u32[i] = static_cast<std::uint32_t>(data[i]);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::uint8_t> body = encode_column_bytes(u32.data(), n);
            std::vector<std::uint8_t> enc(4 + body.size());
            const std::int32_t sc = static_cast<std::int32_t>(col_scale);
            std::memcpy(enc.data(), &sc, 4);
            std::memcpy(enc.data() + 4, body.data(), body.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            std::int32_t sc_dec = 0;
            std::memcpy(&sc_dec, enc.data(), 4);
            if (sc_dec != sc) {
                continue;
            }
            std::vector<std::uint32_t> dec(n);
            auto t2 = std::chrono::high_resolution_clock::now();
            decode_column_bytes(enc.data() + 4, enc.size() - 4, dec.data(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (dec != u32) {
                continue;
            }
            const double pow10 = std::pow(10.0, col_scale);
            bool bits_ok = true;
            for (std::size_t i = 0; i < n; ++i) {
                const float v = col[i];
                if (v == 0.0f) {
                    std::uint32_t vb = 0;
                    std::memcpy(&vb, &v, sizeof(float));
                    if (dec[i] != static_cast<std::uint32_t>(vb)) {
                        bits_ok = false;
                        break;
                    }
                    continue;
                }
                const std::int32_t im = static_cast<std::int32_t>(dec[i]);
                const float r = static_cast<float>(static_cast<double>(im) / pow10);
                std::uint32_t rb = 0;
                std::memcpy(&rb, &r, sizeof(float));
                std::uint32_t vb = 0;
                std::memcpy(&vb, &v, sizeof(float));
                if (rb != vb) {
                    bits_ok = false;
                    break;
                }
            }
            if (!bits_ok) {
                continue;
            }
            if (loss != nullptr) {
                std::vector<double> orig_d(col.begin(), col.end());
                std::vector<double> dec_d(n);
                const double pow10 = std::pow(10.0, col_scale);
                for (std::size_t i = 0; i < n; ++i) {
                    dec_d[i] = static_cast<double>(static_cast<std::int32_t>(dec[i])) / pow10;
                }
                benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
            }
            std::chrono::duration<double, std::nano> enc_ns = t1 - t0;
            std::chrono::duration<double, std::nano> dec_ns = t3 - t2;
            comp += static_cast<long long>(enc.size());
            ct += static_cast<long long>(enc_ns.count());
            dt += static_cast<long long>(dec_ns.count());
        }
    } catch (const std::exception &) {
        result.assign(4, 0);
        return;
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

void benchIntColumnsSimple8b(const std::vector<std::vector<int>> &columns, std::vector<long long> &result,
                             LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    try {
        for (const auto &col : columns) {
            const std::size_t n = col.size();
            if (n == 0) {
                continue;
            }
            std::vector<std::uint32_t> u32(n);
            for (std::size_t i = 0; i < n; ++i) {
                u32[i] = static_cast<std::uint32_t>(col[i]);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::uint8_t> enc = encode_column_bytes(u32.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::uint32_t> dec(n);
            decode_column_bytes(enc.data(), enc.size(), dec.data(), n);
            auto t2 = std::chrono::high_resolution_clock::now();
            if (dec != u32) {
                continue;
            }
            if (loss != nullptr) {
                std::vector<std::int64_t> orig64(col.begin(), col.end());
                std::vector<std::int64_t> dec64(n);
                for (std::size_t i = 0; i < n; ++i) {
                    dec64[i] = static_cast<std::int64_t>(dec[i]);
                }
                benchmark_loss::accumulate_int64_column(orig64, dec64, *loss);
            }
            std::chrono::duration<double, std::nano> enc_ns = t1 - t0;
            std::chrono::duration<double, std::nano> dec_ns = t2 - t1;
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, false));
            comp += static_cast<long long>(enc.size());
            ct += static_cast<long long>(enc_ns.count());
            dt += static_cast<long long>(dec_ns.count());
        }
    } catch (const std::exception &) {
        result.assign(4, 0);
        return;
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

void benchDoubleColumnsSimple8b(const std::vector<std::vector<double>> &columns,
                                std::vector<long long> &result,
                                const std::vector<int> *max_point_per_column, LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    try {
        for (std::size_t ci = 0; ci < columns.size(); ++ci) {
            const auto &col = columns[ci];
            const std::size_t n = col.size();
            if (n == 0) {
                continue;
            }
            int col_scale = -1;
            std::vector<std::int64_t> data;
            bool scaled = false;
            if (max_point_per_column != nullptr && ci < max_point_per_column->size()) {
                const int cap = (*max_point_per_column)[ci];
                if (cap >= 0) {
                    scaled = leco_double::quantizeScaledIntsWithMaxPointFallback(col, cap, data, &col_scale);
                }
            }
            if (!scaled) {
                data.clear();
                col_scale = leco_double::maxLosslessDecimalScale(col);
                if (col_scale >= 0) {
                    scaled = leco_double::quantizeScaledInts(col, col_scale, data);
                }
                if (!scaled) {
                    col_scale = kRawDoubleBitsScale;
                    data.resize(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        std::uint64_t bits = 0;
                        std::memcpy(&bits, &col[i], sizeof(bits));
                        data[i] = static_cast<std::int64_t>(bits);
                    }
                }
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            const auto body = int64_column_codec::simple8b_encode_int64(data.data(), n);
            std::vector<std::uint8_t> enc(4 + body.size());
            const std::int32_t sc = static_cast<std::int32_t>(col_scale);
            std::memcpy(enc.data(), &sc, 4);
            std::memcpy(enc.data() + 4, body.data(), body.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            std::int32_t sc_dec = 0;
            std::memcpy(&sc_dec, enc.data(), 4);
            if (sc_dec != sc) {
                continue;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            const auto recovered = int64_column_codec::simple8b_decode_int64(enc.data() + 4, enc.size() - 4, n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (recovered != data) {
                continue;
            }
            if (loss != nullptr) {
                std::vector<double> decoded(n);
                const double pow10 = col_scale == kRawDoubleBitsScale
                                         ? 1.0
                                         : std::pow(10.0, col_scale);
                for (std::size_t i = 0; i < n; ++i) {
                    if (col_scale == kRawDoubleBitsScale || col[i] == 0.0) {
                        const std::uint64_t bits = static_cast<std::uint64_t>(recovered[i]);
                        std::memcpy(&decoded[i], &bits, sizeof(bits));
                    } else {
                        decoded[i] = static_cast<double>(recovered[i]) / pow10;
                    }
                }
                benchmark_loss::accumulate_double_column(col, decoded, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(enc.size());
            ct += static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            dt += static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
        }
    } catch (const std::exception &) {
        result.assign(4, 0);
        return;
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

void benchInt64ColumnsSimple8b(const std::vector<std::vector<std::int64_t>> &columns,
                               std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    try {
        for (const auto &col : columns) {
            const std::size_t n = col.size();
            if (n == 0) {
                continue;
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            const auto enc = int64_column_codec::simple8b_encode_int64(col.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto t2 = std::chrono::high_resolution_clock::now();
            const auto recovered = int64_column_codec::simple8b_decode_int64(enc.data(), enc.size(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (recovered.size() != n || !std::equal(recovered.begin(), recovered.end(), col.begin())) {
                continue;
            }
            if (loss != nullptr) {
                benchmark_loss::accumulate_int64_column(col, recovered, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, false));
            comp += static_cast<long long>(enc.size());
            ct += static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            dt += static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
        }
    } catch (const std::exception &) {
        result.assign(4, 0);
        return;
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

} // namespace simple8b_int
