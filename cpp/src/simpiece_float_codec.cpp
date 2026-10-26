#include "simpiece_float_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <zstd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace simpiece_float {
namespace {

constexpr double kDblMax = std::numeric_limits<double>::max();

// Mirrors {@code Math.round(double)} in JDK (essentially floor(x+0.5) for normal values).
static long long java_math_round_double(double a) {
    if (std::isnan(a)) {
        return 0;
    }
    if (a == std::numeric_limits<double>::infinity()) {
        return std::numeric_limits<long long>::max();
    }
    if (a == -std::numeric_limits<double>::infinity()) {
        return std::numeric_limits<long long>::min();
    }
    return static_cast<long long>(std::floor(a + 0.5));
}

struct Seg {
    long long init_ts;
    double a_min;
    double a_max;
    double b;

    double a_mid() const { return (a_min + a_max) / 2.0; }
};

static double quantize(double value, double epsilon) {
    double r = value / epsilon;
    long long rq = java_math_round_double(r);
    return static_cast<double>(rq) * epsilon;
}

static void vb_write_u32(std::uint32_t n, std::vector<std::uint8_t> &out) {
    const long long val = static_cast<long long>(n) & 0xFFFFFFFFLL;

    auto ex7 = [&](int i) -> int { return static_cast<int>((val >> (7 * i)) & ((1 << 7) - 1)); };
    auto exm = [&](int i) -> int { return static_cast<int>((val >> (7 * i)) & 0xFF); };

    if (val < (1 << 7)) {
        out.push_back(static_cast<std::uint8_t>((val | (1 << 7)) & 0xFF));
    } else if (val < (1 << 14)) {
        out.push_back(static_cast<std::uint8_t>(ex7(0) & 0xFF));
        out.push_back(static_cast<std::uint8_t>((exm(1) | (1 << 7)) & 0xFF));
    } else if (val < (1 << 21)) {
        out.push_back(static_cast<std::uint8_t>(ex7(0) & 0xFF));
        out.push_back(static_cast<std::uint8_t>(ex7(1) & 0xFF));
        out.push_back(static_cast<std::uint8_t>((exm(2) | (1 << 7)) & 0xFF));
    } else if (val < (1 << 28)) {
        out.push_back(static_cast<std::uint8_t>(ex7(0) & 0xFF));
        out.push_back(static_cast<std::uint8_t>(ex7(1) & 0xFF));
        out.push_back(static_cast<std::uint8_t>(ex7(2) & 0xFF));
        out.push_back(static_cast<std::uint8_t>((exm(3) | (1 << 7)) & 0xFF));
    } else {
        out.push_back(static_cast<std::uint8_t>(ex7(0) & 0xFF));
        out.push_back(static_cast<std::uint8_t>(ex7(1) & 0xFF));
        out.push_back(static_cast<std::uint8_t>(ex7(2) & 0xFF));
        out.push_back(static_cast<std::uint8_t>(ex7(3) & 0xFF));
        out.push_back(static_cast<std::uint8_t>((exm(4) | (1 << 7)) & 0xFF));
    }
}

static int vb_read_u32(const std::uint8_t *buf, std::size_t len, std::size_t &pos) {
    auto rb = [&]() -> int {
        if (pos >= len) {
            throw std::runtime_error("simpiece: vb eof");
        }
        return static_cast<int>(buf[pos++]);
    };
    int inn = rb();
    int number = inn & 0x7F;
    if (inn >= 0x80) {
        return number;
    }
    inn = rb();
    number = ((inn & 0x7F) << 7) | number;
    if (inn >= 0x80) {
        return number;
    }
    inn = rb();
    number = ((inn & 0x7F) << 14) | number;
    if (inn >= 0x80) {
        return number;
    }
    inn = rb();
    number = ((inn & 0x7F) << 21) | number;
    if (inn >= 0x80) {
        return number;
    }
    inn = rb();
    number = (((inn & 0x7F) << 28) | number) & 0xFFFFFFFF;
    return number;
}

static void write_f32_be(float x, std::vector<std::uint8_t> &out) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    out.push_back(static_cast<std::uint8_t>((u >> 24) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((u >> 16) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((u >> 8) & 0xFF));
    out.push_back(static_cast<std::uint8_t>(u & 0xFF));
}

static float read_f32_be(const std::uint8_t *buf, std::size_t len, std::size_t &pos) {
    if (pos + 4 > len) {
        throw std::runtime_error("simpiece: f32 eof");
    }
    std::uint32_t u = (static_cast<std::uint32_t>(buf[pos]) << 24) |
                      (static_cast<std::uint32_t>(buf[pos + 1]) << 16) |
                      (static_cast<std::uint32_t>(buf[pos + 2]) << 8) |
                      static_cast<std::uint32_t>(buf[pos + 3]);
    pos += 4;
    float f = 0;
    std::memcpy(&f, &u, 4);
    return f;
}

static int create_segment(int start_idx, const std::vector<long long> &ts, const std::vector<double> &vals,
                          double epsilon, std::vector<Seg> &segs) {
    const int n = static_cast<int>(vals.size());
    long long init_ts = ts[static_cast<std::size_t>(start_idx)];
    double b = quantize(vals[static_cast<std::size_t>(start_idx)], epsilon);
    if (start_idx + 1 == n) {
        segs.push_back(Seg{init_ts, -kDblMax, kDblMax, b});
        return start_idx + 1;
    }
    double dt = static_cast<double>(ts[static_cast<std::size_t>(start_idx + 1)] - init_ts);
    double a_max = ((vals[static_cast<std::size_t>(start_idx + 1)] + epsilon) - b) / dt;
    double a_min = ((vals[static_cast<std::size_t>(start_idx + 1)] - epsilon) - b) / dt;
    if (start_idx + 2 == n) {
        segs.push_back(Seg{init_ts, a_min, a_max, b});
        return start_idx + 2;
    }
    for (int idx = start_idx + 2; idx < n; idx++) {
        double up_val = vals[static_cast<std::size_t>(idx)] + epsilon;
        double down_val = vals[static_cast<std::size_t>(idx)] - epsilon;
        double dt_i = static_cast<double>(ts[static_cast<std::size_t>(idx)] - init_ts);
        double up_lim = a_max * dt_i + b;
        double down_lim = a_min * dt_i + b;
        if (down_val > up_lim || up_val < down_lim) {
            segs.push_back(Seg{init_ts, a_min, a_max, b});
            return idx;
        }
        if (up_val < up_lim) {
            a_max = std::max((up_val - b) / dt_i, a_min);
        }
        if (down_val > down_lim) {
            a_min = std::min((down_val - b) / dt_i, a_max);
        }
    }
    segs.push_back(Seg{init_ts, a_min, a_max, b});
    return n;
}

static std::vector<Seg> compress_segments(const std::vector<long long> &ts, const std::vector<double> &vals,
                                            double epsilon) {
    std::vector<Seg> segs;
    int cur = 0;
    const int n = static_cast<int>(vals.size());
    while (cur < n) {
        cur = create_segment(cur, ts, vals, epsilon, segs);
    }
    return segs;
}

static std::vector<Seg> merge_per_b(std::vector<Seg> segments) {
    double a_min_temp = -kDblMax;
    double a_max_temp = kDblMax;
    std::optional<double> b_cur;
    std::vector<long long> timestamps;
    std::vector<Seg> merged;

    std::sort(segments.begin(), segments.end(), [](const Seg &x, const Seg &y) {
        if (x.b != y.b) {
            return x.b < y.b;
        }
        return x.a_mid() < y.a_mid();
    });

    for (std::size_t i = 0; i < segments.size(); i++) {
        const Seg &sg = segments[i];
        if (!b_cur.has_value() || *b_cur != sg.b) {
            if (timestamps.size() == 1 && b_cur.has_value()) {
                merged.push_back(Seg{timestamps[0], a_min_temp, a_max_temp, *b_cur});
            } else if (timestamps.size() > 1 && b_cur.has_value()) {
                for (long long t : timestamps) {
                    merged.push_back(Seg{t, a_min_temp, a_max_temp, *b_cur});
                }
            }
            timestamps.clear();
            timestamps.push_back(sg.init_ts);
            a_min_temp = sg.a_min;
            a_max_temp = sg.a_max;
            b_cur = sg.b;
            continue;
        }
        if (sg.a_min <= a_max_temp && sg.a_max >= a_min_temp) {
            timestamps.push_back(sg.init_ts);
            a_min_temp = std::max(a_min_temp, sg.a_min);
            a_max_temp = std::min(a_max_temp, sg.a_max);
        } else {
            if (timestamps.size() == 1) {
                merged.push_back(segments[i - 1]);
            } else {
                for (long long t : timestamps) {
                    merged.push_back(Seg{t, a_min_temp, a_max_temp, *b_cur});
                }
            }
            timestamps.clear();
            timestamps.push_back(sg.init_ts);
            a_min_temp = sg.a_min;
            a_max_temp = sg.a_max;
        }
    }
    if (!timestamps.empty() && b_cur.has_value()) {
        if (timestamps.size() == 1) {
            merged.push_back(Seg{timestamps[0], a_min_temp, a_max_temp, *b_cur});
        } else {
            for (long long t : timestamps) {
                merged.push_back(Seg{t, a_min_temp, a_max_temp, *b_cur});
            }
        }
    }
    return merged;
}

static std::vector<std::pair<long long, double>> to_points(const std::vector<Seg> &segs_in, long long last_ts) {
    std::vector<Seg> segs = segs_in;
    std::sort(segs.begin(), segs.end(), [](const Seg &a, const Seg &b) { return a.init_ts < b.init_ts; });
    if (segs.empty()) {
        return {};
    }
    std::vector<std::pair<long long, double>> out;
    long long cur = segs[0].init_ts;
    for (std::size_t i = 0; i + 1 < segs.size(); i++) {
        const Seg &sg = segs[i];
        long long nxt = segs[i + 1].init_ts;
        while (cur < nxt) {
            out.push_back({cur, sg.a_mid() * static_cast<double>(cur - sg.init_ts) + sg.b});
            cur++;
        }
    }
    const Seg &last = segs.back();
    while (cur <= last_ts) {
        out.push_back({cur, last.a_mid() * static_cast<double>(cur - last.init_ts) + last.b});
        cur++;
    }
    return out;
}

static void to_byte_array_per_b(const std::vector<Seg> &segments, double epsilon, std::vector<std::uint8_t> &buf) {
    std::map<int, std::map<double, std::vector<long long>>> tree;
    for (const Seg &sg : segments) {
        double a = sg.a_mid();
        int b_int = static_cast<int>(java_math_round_double(sg.b / epsilon));
        long long t = sg.init_ts;
        tree[b_int][a].push_back(t);
    }
    vb_write_u32(static_cast<std::uint32_t>(tree.size()), buf);
    if (tree.empty()) {
        return;
    }
    int prev_b = tree.begin()->first;
    vb_write_u32(static_cast<std::uint32_t>(prev_b & 0xFFFFFFFFu), buf);
    for (const auto &be : tree) {
        vb_write_u32(static_cast<std::uint32_t>((be.first - prev_b) & 0xFFFFFFFFu), buf);
        prev_b = be.first;
        const auto &a_map = be.second;
        vb_write_u32(static_cast<std::uint32_t>(a_map.size()), buf);
        for (const auto &ae : a_map) {
            write_f32_be(static_cast<float>(ae.first), buf);
            const auto &ts_list = ae.second;
            std::vector<long long> tsl = ts_list;
            std::sort(tsl.begin(), tsl.end());
            vb_write_u32(static_cast<std::uint32_t>(tsl.size()), buf);
            long long prev_ts = 0;
            for (long long tstamp : tsl) {
                vb_write_u32(static_cast<std::uint32_t>(static_cast<long long>(tstamp - prev_ts) & 0xFFFFFFFFLL),
                             buf);
                prev_ts = tstamp;
            }
        }
    }
}

static std::vector<Seg> read_merged_per_b(const std::uint8_t *raw, std::size_t raw_len, std::size_t &pos,
                                          double epsilon) {
    std::vector<Seg> segments;
    int num_b = vb_read_u32(raw, raw_len, pos);
    if (num_b == 0) {
        return segments;
    }
    int prev_b = vb_read_u32(raw, raw_len, pos);
    for (int i = 0; i < num_b; i++) {
        int b = static_cast<int>(vb_read_u32(raw, raw_len, pos) + static_cast<unsigned>(prev_b));
        prev_b = b;
        int num_a = vb_read_u32(raw, raw_len, pos);
        for (int j = 0; j < num_a; j++) {
            float a = read_f32_be(raw, raw_len, pos);
            int n_ts = vb_read_u32(raw, raw_len, pos);
            long long timestamp = 0;
            for (int k = 0; k < n_ts; k++) {
                timestamp += static_cast<unsigned>(vb_read_u32(raw, raw_len, pos));
                segments.push_back(
                    Seg{timestamp, static_cast<double>(a), static_cast<double>(a), static_cast<double>(b) * epsilon});
            }
        }
    }
    return segments;
}

static std::vector<std::uint8_t> compress_inner(const std::vector<double> &vals, double epsilon) {
    const int n = static_cast<int>(vals.size());
    std::vector<long long> ts(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++) {
        ts[static_cast<std::size_t>(i)] = i;
    }
    auto raw_segs = compress_segments(ts, vals, epsilon);
    auto merged = merge_per_b(std::move(raw_segs));
    std::vector<std::uint8_t> preamble;
    write_f32_be(static_cast<float>(epsilon), preamble);
    to_byte_array_per_b(merged, epsilon, preamble);
    vb_write_u32(static_cast<std::uint32_t>(n - 1), preamble);

    std::size_t bound = ZSTD_compressBound(preamble.size());
    std::vector<std::uint8_t> compressed(bound);
    std::size_t z = ZSTD_compress(compressed.data(), compressed.size(), preamble.data(), preamble.size(), 3);
    if (ZSTD_isError(z)) {
        throw std::runtime_error(std::string("simpiece: zstd compress ") + ZSTD_getErrorName(z));
    }
    compressed.resize(z);
    return compressed;
}

static std::vector<std::pair<long long, double>> decompress_inner(const std::vector<std::uint8_t> &packed) {
    unsigned long long frame = ZSTD_getFrameContentSize(packed.data(), packed.size());
    std::size_t out_cap = (frame != ZSTD_CONTENTSIZE_UNKNOWN && frame != ZSTD_CONTENTSIZE_ERROR)
                              ? static_cast<std::size_t>(frame)
                              : std::max<std::size_t>(packed.size() * 16, 65536);
    std::vector<std::uint8_t> raw(out_cap);
    std::size_t dz = ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size());
    while (ZSTD_isError(dz) && ZSTD_getErrorCode(dz) == ZSTD_error_dstSize_tooSmall) {
        raw.resize(raw.size() * 2);
        dz = ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size());
    }
    if (ZSTD_isError(dz)) {
        throw std::runtime_error(std::string("simpiece: zstd decompress ") + ZSTD_getErrorName(dz));
    }
    raw.resize(dz);
    std::size_t pos = 0;
    float eps_f = read_f32_be(raw.data(), raw.size(), pos);
    double epsilon = static_cast<double>(eps_f);
    auto segs = read_merged_per_b(raw.data(), raw.size(), pos, epsilon);
    int last_ts = vb_read_u32(raw.data(), raw.size(), pos);
    return to_points(segs, last_ts);
}

static double column_epsilon(std::span<const float> col, double rel_range_frac) {
    if (col.empty()) {
        return -1;
    }
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (float f : col) {
        double v = static_cast<double>(f);
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    double range = mx - mn;
    if (range <= 0 || std::isnan(range)) {
        range = 1.0;
    }
    return range * rel_range_frac;
}

static double column_epsilon_double_values(const std::vector<double> &vals, double rel_range_frac) {
    if (vals.empty()) {
        return -1;
    }
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (double v : vals) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    double range = mx - mn;
    if (range <= 0 || std::isnan(range)) {
        range = 1.0;
    }
    return range * rel_range_frac;
}

static bool verifyDecompressToleranceDoubleValues(const std::vector<std::uint8_t> &packed,
                                                  const std::vector<double> &values,
                                                  double rel_range_frac,
                                                  double extra_eps) {
    if (values.empty()) {
        return false;
    }
    const double epsilon = column_epsilon_double_values(values, rel_range_frac);
    if (epsilon <= 0) {
        return false;
    }
    try {
        auto restored = decompress_inner(packed);
        std::vector<double> by_ts(values.size(), std::numeric_limits<double>::quiet_NaN());
        for (const auto &pr : restored) {
            if (pr.first >= 0 && static_cast<std::size_t>(pr.first) < values.size()) {
                by_ts[static_cast<std::size_t>(pr.first)] = pr.second;
            }
        }
        const double tol = epsilon * 1.0001 + extra_eps;
        for (std::size_t i = 0; i < values.size(); i++) {
            double dv = by_ts[i];
            if (std::isnan(dv) || std::abs(dv - values[i]) > tol) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

std::vector<std::uint8_t> compressColumnRelRange(std::span<const float> col, double rel_range_frac) {
    if (col.empty()) {
        throw std::runtime_error("simpiece: empty column");
    }
    double epsilon = column_epsilon(col, rel_range_frac);
    if (epsilon <= 0) {
        throw std::runtime_error("simpiece: epsilon");
    }
    std::vector<double> vals(col.size());
    for (std::size_t i = 0; i < col.size(); i++) {
        vals[i] = static_cast<double>(col[static_cast<std::size_t>(i)]);
    }
    return compress_inner(vals, epsilon);
}

bool verifyDecompressTolerance(std::span<const std::uint8_t> packed, std::span<const float> col,
                               double rel_range_frac) {
    if (col.empty()) {
        return false;
    }
    double epsilon = column_epsilon(col, rel_range_frac);
    if (epsilon <= 0) {
        return false;
    }
    try {
        std::vector<std::uint8_t> pv(packed.begin(), packed.end());
        auto restored = decompress_inner(pv);
        std::vector<double> by_ts(col.size(), std::numeric_limits<double>::quiet_NaN());
        for (const auto &pr : restored) {
            if (pr.first >= 0 && static_cast<std::size_t>(pr.first) < col.size()) {
                by_ts[static_cast<std::size_t>(pr.first)] = pr.second;
            }
        }
        const double tol = epsilon * 1.0001 + 1e-9;
        for (std::size_t i = 0; i < col.size(); i++) {
            double dv = by_ts[i];
            if (std::isnan(dv) || std::abs(dv - static_cast<double>(col[i])) > tol) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

void benchFloatColumnsRelRange(const std::vector<std::vector<float>> &columns, double rel_range_frac,
                               std::vector<long long> &result, LossAccum *loss) {
    using clock = std::chrono::steady_clock;
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    std::size_t col_idx = 0;
    for (const auto &col : columns) {
        ++col_idx;
        std::cerr << "    simpiece: column " << col_idx << "/" << columns.size()
                  << " (rel_range=" << rel_range_frac << ")...\n";
        if (col.empty()) {
            continue;
        }
        double epsilon = column_epsilon(col, rel_range_frac);
        if (epsilon <= 0) {
            continue;
        }
        try {
            auto t0 = clock::now();
            auto packed = compressColumnRelRange(col, rel_range_frac);
            auto t1 = clock::now();
            auto t2 = clock::now();
            if (!verifyDecompressTolerance(packed, col, rel_range_frac)) {
                throw std::runtime_error("simpiece tol");
            }
            if (loss != nullptr) {
                auto restored = decompress_inner(packed);
                std::vector<double> by_ts(col.size(), std::numeric_limits<double>::quiet_NaN());
                for (const auto &pr : restored) {
                    if (pr.first >= 0 && static_cast<std::size_t>(pr.first) < col.size()) {
                        by_ts[static_cast<std::size_t>(pr.first)] = pr.second;
                    }
                }
                std::vector<double> orig_d(col.begin(), col.end());
                benchmark_loss::accumulate_double_column(orig_d, by_ts, *loss);
            }
            auto t3 = clock::now();
            orig += static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
            comp += static_cast<long long>(packed.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

void benchDoubleColumnsRelRange(const std::vector<std::vector<double>> &columns, double rel_range_frac,
                                std::vector<long long> &result, LossAccum *loss) {
    using clock = std::chrono::steady_clock;
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        const double epsilon = column_epsilon_double_values(col, rel_range_frac);
        if (epsilon <= 0) {
            continue;
        }
        try {
            auto t0 = clock::now();
            auto packed = compress_inner(col, epsilon);
            auto t1 = clock::now();
            auto t2 = clock::now();
            if (!verifyDecompressToleranceDoubleValues(packed, col, rel_range_frac, 1e-12)) {
                throw std::runtime_error("simpiece tol double");
            }
            if (loss != nullptr) {
                auto restored = decompress_inner(packed);
                std::vector<double> by_ts(col.size(), std::numeric_limits<double>::quiet_NaN());
                for (const auto &pr : restored) {
                    if (pr.first >= 0 && static_cast<std::size_t>(pr.first) < col.size()) {
                        by_ts[static_cast<std::size_t>(pr.first)] = pr.second;
                    }
                }
                benchmark_loss::accumulate_double_column(col, by_ts, *loss);
            }
            auto t3 = clock::now();
            orig += static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
            comp += static_cast<long long>(packed.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

void benchInt64ColumnsRelRange(const std::vector<std::vector<std::int64_t>> &columns, double rel_range_frac,
                               std::vector<long long> &result, LossAccum *loss) {
    using clock = std::chrono::steady_clock;
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        std::vector<double> values;
        values.reserve(col.size());
        for (std::int64_t v : col) {
            values.push_back(static_cast<double>(v));
        }
        const double epsilon = column_epsilon_double_values(values, rel_range_frac);
        if (epsilon <= 0) {
            continue;
        }
        try {
            auto t0 = clock::now();
            auto packed = compress_inner(values, epsilon);
            auto t1 = clock::now();
            auto t2 = clock::now();
            if (!verifyDecompressToleranceDoubleValues(packed, values, rel_range_frac, 1e-9)) {
                throw std::runtime_error("simpiece tol int64");
            }
            if (loss != nullptr) {
                auto restored = decompress_inner(packed);
                std::vector<double> by_ts(values.size(), std::numeric_limits<double>::quiet_NaN());
                for (const auto &pr : restored) {
                    if (pr.first >= 0 && static_cast<std::size_t>(pr.first) < values.size()) {
                        by_ts[static_cast<std::size_t>(pr.first)] = pr.second;
                    }
                }
                benchmark_loss::accumulate_double_column(values, by_ts, *loss);
            }
            auto t3 = clock::now();
            orig += static_cast<long long>(benchmark_wire::column_original_size(col.size(), false));
            comp += static_cast<long long>(packed.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

} // namespace simpiece_float
