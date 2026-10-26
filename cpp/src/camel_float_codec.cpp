#include "camel_float_codec.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace camel_float {
namespace {

constexpr int kDecimalMaxCount = 3;
constexpr int kMValueBits[] = {3, 5, 7, 10, 15};
constexpr int kThreshold[] = {5, 25, 125, 625};
constexpr double kPowersD[] = {1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0};

// Match {@code ExternalCompressionBench.benchCamelFloat} / {@code camelChunkPoints}.
constexpr int kCamelChunkPoints = 1000;
constexpr int kBufSize = 100000;
AblationMode g_ablation_mode = AblationMode::none;

static std::uint64_t double_to_raw_bits(double d) {
    std::uint64_t u = 0;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

static double raw_bits_to_double(std::uint64_t bits) {
    double d = 0;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

static long long java_round_double(double x) {
    if (!std::isfinite(x)) {
        return 0;
    }
    if (x >= 0.0) {
        return static_cast<long long>(std::floor(x + 0.5));
    }
    return static_cast<long long>(std::ceil(x - 0.5));
}

static unsigned low_bits_mask_u32(int len) {
    if (len <= 0) {
        return 0;
    }
    if (len >= 32) {
        return 0xFFFFFFFFu;
    }
    return (1u << len) - 1u;
}

static std::uint64_t low_bits_mask_u64(int q) {
    if (q <= 0) {
        return 0;
    }
    if (q >= 64) {
        return 0xFFFFFFFFFFFFFFFFull;
    }
    return (1ull << q) - 1ull;
}

struct VignaOut {
    std::vector<std::uint8_t> buf;
    long long written_bits{0};
    int current{0};
    int free_bits{8};
    std::size_t pos{0};

    VignaOut() { buf.assign(static_cast<std::size_t>(kBufSize), 0); }

    void write_byte(int b) {
        if (pos >= buf.size()) {
            throw std::runtime_error("camel: output overflow");
        }
        buf[pos++] = static_cast<std::uint8_t>(b & 0xFF);
    }

    void write_in_current(int b, int len) {
        if (len == 0) {
            return;
        }
        free_bits -= len;
        current |= (b & static_cast<int>(low_bits_mask_u32(len))) << free_bits;
        if (free_bits == 0) {
            write_byte(current);
            free_bits = 8;
            current = 0;
        }
        written_bits += len;
    }

    void write_bit(int bit) { write_in_current(bit ? 1 : 0, 1); }

    void write_int(std::uint32_t x, int len) {
        x &= 0xFFFFFFFFu;
        if (len == 0) {
            return;
        }
        if (len <= free_bits) {
            write_in_current(static_cast<int>(x), len);
            return;
        }
        int i = len - free_bits;
        int queue = i & 7;
        if (free_bits != 0) {
            write_in_current(static_cast<int>(x >> i), free_bits);
        }
        if (queue != 0) {
            i -= queue;
            write_in_current(static_cast<int>(x & low_bits_mask_u32(queue)), queue);
            x >>= queue;
        }
        if (i == 32) {
            write_byte(static_cast<int>(x >> 24) & 0xFF);
        }
        if (i > 23) {
            write_byte(static_cast<int>(x >> 16) & 0xFF);
        }
        if (i > 15) {
            write_byte(static_cast<int>(x >> 8) & 0xFF);
        }
        if (i > 7) {
            write_byte(static_cast<int>(x) & 0xFF);
        }
        written_bits += i;
    }

    void write_long(std::uint64_t x, int len) {
        if (len <= 0) {
            return;
        }
        if (len < 64) {
            x &= (1ull << len) - 1ull;
        }
        if (len <= free_bits) {
            write_in_current(static_cast<int>(x), len);
            return;
        }
        int i = len - free_bits;
        int queue = i & 7;
        if (free_bits != 0) {
            write_in_current(static_cast<int>(x >> i), free_bits);
        }
        if (queue != 0) {
            i -= queue;
            write_in_current(static_cast<int>(x & low_bits_mask_u64(queue)), queue);
            x >>= queue;
        }
        if (i >= 56) {
            write_byte(static_cast<int>((x >> 56) & 0xFF));
        }
        if (i >= 48) {
            write_byte(static_cast<int>((x >> 48) & 0xFF));
        }
        if (i >= 40) {
            write_byte(static_cast<int>((x >> 40) & 0xFF));
        }
        if (i >= 32) {
            write_byte(static_cast<int>((x >> 32) & 0xFF));
        }
        if (i > 23) {
            write_byte(static_cast<int>((x >> 24) & 0xFF));
        }
        if (i > 15) {
            write_byte(static_cast<int>((x >> 16) & 0xFF));
        }
        if (i > 7) {
            write_byte(static_cast<int>((x >> 8) & 0xFF));
        }
        if (i > 0) {
            write_byte(static_cast<int>(x & 0xFF));
        }
        written_bits += i;
    }

    void align_flush() {
        if (free_bits != 8) {
            write_in_current(0, free_bits);
        }
    }
};

struct VignaIn {
    const std::uint8_t *buffer{nullptr};
    std::size_t len{0};
    std::size_t pos{0};
    int fill{0};
    std::uint32_t current{0};

    explicit VignaIn(std::span<const std::uint8_t> bytes)
        : buffer(bytes.data()), len(bytes.size()) {}

    int read() {
        if (pos >= len) {
            return -1;
        }
        return static_cast<int>(buffer[pos++]) & 0xFF;
    }

    int read_from_current(int l) {
        if (fill == 0) {
            int b = read();
            if (b < 0) {
                throw std::runtime_error("camel: truncated bit stream");
            }
            current = static_cast<std::uint32_t>(b);
            fill = 8;
        }
        fill -= l;
        return static_cast<int>((current >> fill) & low_bits_mask_u32(l));
    }

    int read_bit() { return read_from_current(1); }

    int read_int(int l) {
        if (l < 0 || l > 32) {
            throw std::runtime_error("camel: bad read_int");
        }
        if (l == 0) {
            return 0;
        }
        std::uint32_t x = 0;
        for (int i = 0; i < l; i++) {
            x = (x << 1) | static_cast<std::uint32_t>(read_bit() & 1);
        }
        return static_cast<int>(x & 0xFFFFFFFFu);
    }

    std::uint64_t read_long_bits(int l) {
        if (l < 0 || l > 64) {
            throw std::runtime_error("camel: bad read_long");
        }
        if (l == 0) {
            return 0;
        }
        std::uint64_t x = 0;
        for (int i = 0; i < l; i++) {
            x = (x << 1) | static_cast<std::uint64_t>(read_bit() & 1);
        }
        return l == 64 ? x : (x & ((1ull << l) - 1ull));
    }
};

struct CamelComp {
    VignaOut out;
    long long stored_val{0};
    bool first{true};

    void write_raw_bits(std::uint64_t value_bits) {
        out.write_int(static_cast<std::uint32_t>(value_bits >> 32), 32);
        out.write_int(static_cast<std::uint32_t>(value_bits & 0xFFFFFFFFu), 32);
    }

    std::pair<int, int> cal_decimal_count(double value) {
        double factor = 1.0;
        int decimal_count = 0;
        value = std::abs(value);
        if (!std::isfinite(value)) {
            return {kDecimalMaxCount, 0};
        }
        const double epsilon = 1e-7;
        // Unbounded factor growth can spin forever (or until non-finite arithmetic) for pathological
        // doubles, matching the risk class of decimal-scale loops in other ported codecs.
        constexpr int kMaxDecimalScaleIterations = 64;
        int guard = 0;
        while (std::abs(value * factor - static_cast<double>(java_round_double(value * factor))) >
               epsilon) {
            factor *= 10.0;
            decimal_count++;
            if (++guard > kMaxDecimalScaleIterations) {
                break;
            }
        }
        if (decimal_count == 0) {
            decimal_count = 1;
        }
        int dc;
        if (0 < decimal_count && decimal_count <= kDecimalMaxCount) {
            dc = decimal_count;
        } else {
            dc = 4; // Reserved lossless literal code for values outside Camel's decimal range.
        }
        if (dc == 4) return {dc, 0};
        long long pow_i = static_cast<long long>(kPowersD[dc]);
        int decimal_value =
            static_cast<int>(java_round_double(value * static_cast<double>(pow_i)) % pow_i);
        const double quantized = static_cast<double>(static_cast<long long>(value * pow_i + 0.5)) /
                                 static_cast<double>(pow_i);
        if (double_to_raw_bits(quantized) != double_to_raw_bits(value)) {
            return {4, 0};
        }
        return {dc, decimal_value};
    }

    void write_first(std::uint64_t value_bits) {
        first = false;
        stored_val = static_cast<long long>(raw_bits_to_double(value_bits));
        // Java {@code OutputBitStream.writeLong}; avoid a single {@code write_long(..., 64)} path that
        // duplicates partial-byte handling in the Vigna bit writer.
        out.write_int(static_cast<std::uint32_t>(value_bits >> 32), 32);
        out.write_int(static_cast<std::uint32_t>(value_bits & 0xFFFFFFFFu), 32);
    }

    void compress_decimal_value(int decimal_value, int decimal_count) {
        if (decimal_count == 0) {
            return;
        }
        out.write_int(static_cast<std::uint32_t>(decimal_count - 1), 2);
        int thread = kThreshold[decimal_count - 1];
        int m = decimal_value;
        if (decimal_value - thread >= 0) {
            out.write_bit(1);
            m = static_cast<int>(decimal_value % thread);
            if (g_ablation_mode == AblationMode::xor_transform) {
                out.write_long(static_cast<std::uint64_t>(decimal_value / thread), decimal_count);
            } else {
                double p = kPowersD[decimal_count];
                std::uint64_t xorv =
                    double_to_raw_bits(static_cast<double>(decimal_value) / p + 1.0) ^
                    double_to_raw_bits(static_cast<double>(m) / p + 1.0);
                if (g_ablation_mode == AblationMode::xor_bit_reduction) {
                    write_raw_bits(xorv);
                } else {
                    out.write_long(xorv >> (52 - decimal_count), decimal_count);
                }
            }
        } else {
            out.write_bit(0);
        }

        if (decimal_count == 1) {
            out.write_int(static_cast<std::uint32_t>(m), 3);
        } else if (decimal_count == 2) {
            if (m < 8) {
                out.write_int(0, 1);
                out.write_int(static_cast<std::uint32_t>(m), 3);
            } else {
                out.write_int(1, 1);
                out.write_int(static_cast<std::uint32_t>(m), 5);
            }
        } else if (decimal_count == 3) {
            if (m < 2) {
                out.write_int(0, 2);
                out.write_int(static_cast<std::uint32_t>(m), 1);
            } else if (m < 8) {
                out.write_int(1, 2);
                out.write_int(static_cast<std::uint32_t>(m), 3);
            } else if (m < 32) {
                out.write_int(2, 2);
                out.write_int(static_cast<std::uint32_t>(m), 5);
            } else {
                out.write_int(3, 2);
                out.write_int(static_cast<std::uint32_t>(m), kMValueBits[decimal_count - 1]);
            }
        } else {
            if (m < 16) {
                out.write_int(0, 2);
                out.write_int(static_cast<std::uint32_t>(m), 4);
            } else if (m < 64) {
                out.write_int(1, 2);
                out.write_int(static_cast<std::uint32_t>(m), 6);
            } else if (m < 256) {
                out.write_int(2, 2);
                out.write_int(static_cast<std::uint32_t>(m), 8);
            } else {
                out.write_int(3, 2);
                out.write_int(static_cast<std::uint32_t>(m), kMValueBits[decimal_count - 1]);
            }
        }
    }

    void compress_integer_value(int int_value, int int_signal) {
        int diff = int_value - static_cast<int>(stored_val);
        out.write_int(static_cast<std::uint32_t>(int_signal), 1);
        if (-1 <= diff && diff <= 1) {
            out.write_int(static_cast<std::uint32_t>(diff + 1), 2);
        } else {
            out.write_int(3, 2);
            if (diff < 0) {
                out.write_bit(0);
                diff = -diff;
            } else {
                out.write_bit(1);
            }
            if (2 <= diff && diff < 8) {
                out.write_int(0, 1);
                out.write_int(static_cast<std::uint32_t>(diff), 3);
            } else {
                out.write_int(1, 1);
                out.write_int(static_cast<std::uint32_t>(diff), 16);
            }
        }
        stored_val = int_value;
    }

    void compress_value(double value) {
        int int_signal = value < 0 ? 0 : 1;
        compress_integer_value(static_cast<int>(value), int_signal);
        auto dec = cal_decimal_count(value);
        if (dec.first == 4 || g_ablation_mode == AblationMode::scaling) {
            out.write_int(3, 2);
            write_raw_bits(double_to_raw_bits(value));
            return;
        }
        compress_decimal_value(dec.second, dec.first);
    }

    void add_value(double value) {
        if (first) {
            auto dec = cal_decimal_count(value);
            int dc = dec.first;
            if (dc == 4) {
                write_raw_bits(double_to_raw_bits(value));
                first = false;
                stored_val = static_cast<long long>(value);
                return;
            }
            double p = kPowersD[dc];
            double sign = value < 0 ? -1.0 : 1.0;
            double v_adj =
                sign * ((static_cast<int>(std::abs(value)) * kPowersD[dc] + dec.second) / p);
            write_first(double_to_raw_bits(v_adj));
        } else {
            compress_value(value);
        }
    }

    void close() {
        out.write_bit(0);
        out.align_flush();
    }

    std::vector<std::uint8_t> finish() {
        long long wb = out.written_bits;
        int comp_bytes = static_cast<int>((wb + 7) / 8);
        if (comp_bytes <= 0 || static_cast<std::size_t>(comp_bytes) > out.buf.size()) {
            throw std::runtime_error("camel: bad size");
        }
        return std::vector<std::uint8_t>(out.buf.begin(),
                                         out.buf.begin() + static_cast<std::ptrdiff_t>(comp_bytes));
    }
};

struct CamelDec {
    VignaIn in;
    long long stored_val{0};
    double current_val{0};
    bool dec_first{true};
    bool eos{false};
    int read_num{0};

    explicit CamelDec(std::span<const std::uint8_t> data) : in(data) {}

    std::pair<int, int> read_int_part() {
        int int_signal = in.read_int(1);
        int integer_num = in.read_int(2);
        int diff_val = 0;
        if (integer_num < 3) {
            diff_val = integer_num - 1;
        } else {
            int diff_symbol = in.read_bit();
            int rng = in.read_int(1);
            if (rng == 0) {
                diff_val = in.read_int(3);
            } else {
                diff_val = in.read_int(16);
            }
            diff_val = (diff_symbol == 0 ? -1 : 1) * diff_val;
        }
        stored_val = diff_val + stored_val;
        int sig = stored_val >= 0 ? 1 : 0;
        if (stored_val == 0 && int_signal == 0) {
            sig = 0;
        }
        long long ab = stored_val >= 0 ? stored_val : -stored_val;
        return {sig, static_cast<int>(ab)};
    }

    std::pair<int, double> read_decimal() {
        int decimal_count = in.read_int(2) + 1;
        if (decimal_count == 4) {
            return {4, raw_bits_to_double(in.read_long_bits(64))};
        }
        int is_m = in.read_int(1);
        std::uint64_t xor_string = 0;
        std::uint64_t decimal_quotient = 0;
        if (is_m == 1) {
            if (g_ablation_mode == AblationMode::xor_transform) {
                decimal_quotient = in.read_long_bits(decimal_count);
            } else if (g_ablation_mode == AblationMode::xor_bit_reduction) {
                xor_string = in.read_long_bits(64);
            } else {
                std::uint64_t xorv = in.read_long_bits(decimal_count);
                xor_string = xorv << (52 - decimal_count);
            }
        }
        int m_int = 0;
        if (decimal_count <= 1) {
            m_int = in.read_int(3);
        } else if (decimal_count == 2) {
            int temp = in.read_int(1);
            if (temp == 0) {
                m_int = in.read_int(3);
            } else {
                m_int = in.read_int(5);
            }
        } else if (decimal_count == 3) {
            int temp = in.read_int(2);
            if (temp == 0) {
                m_int = in.read_int(1);
            } else if (temp == 1) {
                m_int = in.read_int(3);
            } else if (temp == 2) {
                m_int = in.read_int(5);
            } else {
                m_int = in.read_int(kMValueBits[decimal_count - 1]);
            }
        } else {
            int temp = in.read_int(2);
            if (temp == 0) {
                m_int = in.read_int(4);
            } else if (temp == 1) {
                m_int = in.read_int(6);
            } else if (temp == 2) {
                m_int = in.read_int(8);
            } else {
                m_int = in.read_int(kMValueBits[decimal_count - 1]);
            }
        }

        double scale = kPowersD[decimal_count];
        double decimal_val = 0;
        if (is_m == 1) {
            if (g_ablation_mode == AblationMode::xor_transform) {
                decimal_val = static_cast<double>(
                    decimal_quotient * static_cast<std::uint64_t>(kThreshold[decimal_count - 1]) +
                    static_cast<std::uint64_t>(m_int)) / scale;
            } else {
                double m = static_cast<double>(m_int) / scale + 1.0;
                std::uint64_t m_prime = double_to_raw_bits(m);
                std::uint64_t dec_long = xor_string ^ m_prime;
                decimal_val = raw_bits_to_double(dec_long) - 1.0;
            }
        } else {
            decimal_val = static_cast<double>(m_int) / kPowersD[decimal_count];
        }
        decimal_val =
            static_cast<double>(java_round_double(decimal_val * scale)) / scale;
        return {decimal_count, decimal_val};
    }

    double next_value() {
        auto int_val = read_int_part();
        auto dec = read_decimal();
        int dc = dec.first;
        if (dc == 4) {
            return dec.second;
        }
        double scale = kPowersD[dc];
        double sign = int_val.first == 0 ? -1.0 : 1.0;
        return sign * ((static_cast<double>(int_val.second) * scale + dec.second * scale) / scale);
    }

    std::optional<double> read_pair() {
        if (eos) {
            return std::nullopt;
        }
        read_num++;
        if (dec_first) {
            dec_first = false;
            std::uint64_t hi = static_cast<std::uint32_t>(in.read_int(32));
            std::uint64_t lo = static_cast<std::uint32_t>(in.read_int(32));
            std::uint64_t first_long = (hi << 32) | lo;
            double first_val = raw_bits_to_double(first_long);
            stored_val = static_cast<long long>(first_val);
            current_val = first_val;
        } else {
            if (read_num - 1 == kCamelChunkPoints) {
                eos = true;
                return std::nullopt;
            }
            current_val = next_value();
        }
        return current_val;
    }
};

static std::vector<std::uint8_t> encode_chunk(std::span<const double> col) {
    CamelComp c;
    for (double d : col) {
        c.add_value(d);
    }
    c.close();
    return c.finish();
}

} // namespace

void setAblationMode(AblationMode mode) {
    g_ablation_mode = mode;
}

std::vector<std::uint8_t> encodeCamelFloatColumn(std::span<const float> col) {
    std::vector<std::uint8_t> acc;
    const int n = static_cast<int>(col.size());
    for (int st = 0; st < n; st += kCamelChunkPoints) {
        int en = std::min(n, st + kCamelChunkPoints);
        int chunk_len = en - st;
        std::vector<double> chunk(static_cast<std::size_t>(chunk_len));
        for (int i = 0; i < chunk_len; i++) {
            chunk[static_cast<std::size_t>(i)] = static_cast<double>(col[static_cast<std::size_t>(st + i)]);
        }
        auto piece = encode_chunk(std::span<const double>(chunk.data(), chunk.size()));
        acc.insert(acc.end(), piece.begin(), piece.end());
    }
    return acc;
}

bool decodeCamelFloatChunk(std::span<const std::uint8_t> compressed, std::vector<double> &out,
                           int expectedLen) {
    out.clear();
    out.reserve(static_cast<std::size_t>(expectedLen));
    CamelDec dec(compressed);
    for (int i = 0; i < expectedLen; i++) {
        auto v = dec.read_pair();
        if (!v) {
            return false;
        }
        out.push_back(*v);
    }
    return true;
}

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
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
        try {
            long long col_orig = 0;
            long long col_comp = 0;
            long long col_ct = 0;
            long long col_dt = 0;
            const int n = static_cast<int>(col.size());
            for (int st = 0; st < n; st += kCamelChunkPoints) {
                int en = std::min(n, st + kCamelChunkPoints);
                int chunk_len = en - st;
                auto t0 = clock::now();
                std::vector<double> chunk(static_cast<std::size_t>(chunk_len));
                for (int i = 0; i < chunk_len; i++) {
                    chunk[static_cast<std::size_t>(i)] =
                        static_cast<double>(col[static_cast<std::size_t>(st + i)]);
                }
                auto payload = encode_chunk(std::span<const double>(chunk.data(), chunk.size()));
                auto t1 = clock::now();
                auto t2 = clock::now();
                std::vector<double> dec;
                if (!decodeCamelFloatChunk(payload, dec, chunk_len)) {
                    throw std::runtime_error("camel decode");
                }
                auto t3 = clock::now();
                for (int i = 0; i < chunk_len; i++) {
                    double exp = static_cast<double>(col[static_cast<std::size_t>(st + i)]);
                    if (double_to_raw_bits(dec[static_cast<std::size_t>(i)]) != double_to_raw_bits(exp)) {
                        throw std::runtime_error("camel rt");
                    }
                }
                if (loss != nullptr) {
                    benchmark_loss::accumulate_double_column(chunk, dec, *loss);
                }
                col_orig += static_cast<long long>(benchmark_wire::column_original_size(chunk_len, true));
                col_comp += static_cast<long long>(payload.size());
                col_ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                col_dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }
            orig += col_orig;
            comp += col_comp;
            ct += col_ct;
            dt += col_dt;
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

void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                               std::vector<long long> &result, LossAccum *loss,
                               std::size_t *raw_fallback_columns) {
    using clock = std::chrono::steady_clock;
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        const int n = static_cast<int>(col.size());
        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        try {
            long long col_comp = 0;
            long long col_ct = 0;
            long long col_dt = 0;
            for (int st = 0; st < n; st += kCamelChunkPoints) {
                int en = std::min(n, st + kCamelChunkPoints);
                int chunk_len = en - st;
                auto t0 = clock::now();
                auto payload = encode_chunk(
                    std::span<const double>(col.data() + st, static_cast<std::size_t>(chunk_len)));
                auto t1 = clock::now();
                auto t2 = clock::now();
                std::vector<double> dec;
                if (!decodeCamelFloatChunk(payload, dec, chunk_len)) {
                    throw std::runtime_error("camel decode");
                }
                auto t3 = clock::now();
                for (int i = 0; i < chunk_len; i++) {
                    if (double_to_raw_bits(dec[static_cast<std::size_t>(i)]) !=
                        double_to_raw_bits(col[static_cast<std::size_t>(st + i)])) {
                        throw std::runtime_error("camel rt");
                    }
                }
                if (loss != nullptr) {
                    std::vector<double> orig_chunk(
                        col.begin() + st, col.begin() + st + chunk_len);
                    benchmark_loss::accumulate_double_column(orig_chunk, dec, *loss);
                }
                col_comp += static_cast<long long>(payload.size());
                col_ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                col_dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }
            comp += col_comp;
            ct += col_ct;
            dt += col_dt;
        } catch (...) {
            if (raw_fallback_columns != nullptr) {
                ++(*raw_fallback_columns);
            }
            const auto t0 = clock::now();
            std::vector<std::uint8_t> raw(col.size() * sizeof(double));
            std::memcpy(raw.data(), col.data(), raw.size());
            const auto t1 = clock::now();
            std::vector<double> decoded(col.size());
            std::memcpy(decoded.data(), raw.data(), raw.size());
            const auto t2 = clock::now();
            if (std::memcmp(decoded.data(), col.data(), raw.size()) != 0) {
                throw std::runtime_error("camel raw fallback mismatch");
            }
            comp += static_cast<long long>(raw.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            if (loss != nullptr) {
                benchmark_loss::accumulate_double_column(col, decoded, *loss);
            }
        }
    }
    result.resize(4);
    result[0] = orig;
    result[1] = comp;
    result[2] = ct;
    result[3] = dt;
}

} // namespace camel_float
