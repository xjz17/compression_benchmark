#include "elf_double_codec.hpp"
#include "adjacent_residual.hpp"
#include "residual_window_pack.hpp"

#include "benchmark_wire.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace elf_double {
namespace {

constexpr int kElfChunk = 1600;
constexpr int kChimpChunk = 1000;
constexpr std::size_t kJavaOutputBufBytes = 10000;
constexpr int kMaxBetaStar = 2147483647;
constexpr int kChimpThreshold = 6;
constexpr std::uint64_t kNanBits = 0x7ff8000000000000ull;
constexpr std::uint64_t kEndSign = 0x7ff8000000000000ull;

alignas(64) constexpr std::uint8_t kLeadReprComp[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
alignas(64) constexpr std::uint8_t kLeadRound[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12,
    16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};
alignas(64) constexpr std::uint8_t kLeadDec[8] = {0, 8, 12, 16, 18, 20, 22, 24};

constexpr int kFAlpha[21] = {0,  4,  7,  10, 14, 17, 20, 24, 27, 30, 34,
                             37, 40, 44, 47, 50, 54, 57, 60, 64, 67};
constexpr double kMap10iP[21] = {
    1.0e0,  1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,
    1.0e7,  1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13,
    1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20};
constexpr double kMap10iN[21] = {
    1.0e0,  1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,
    1.0e-7,  1.0e-8,  1.0e-9,  1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13,
    1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19, 1.0e-20};
constexpr long long kSpGt1[10] = {1, 10, 100, 1000, 10000, 100000,
                                  1000000, 10000000, 100000000, 1000000000};
constexpr double kSpLt1[11] = {1.0,       0.1,       0.01,       0.001,
                               0.0001,    0.00001,   0.000001,   0.0000001,
                               0.00000001, 0.000000001, 0.0000000001};

std::uint64_t low_mask64(int len) {
    if (len <= 0) {
        return 0;
    }
    if (len >= 64) {
        return ~0ull;
    }
    return (1ull << len) - 1ull;
}

int nlz64(std::uint64_t x) { return x == 0 ? 64 : __builtin_clzll(x); }
int ntz64(std::uint64_t x) { return x == 0 ? 64 : __builtin_ctzll(x); }

struct BitOut {
    std::vector<std::uint8_t> buf;
    long long written_bits{0};
    int current{0};
    int free_bits{8};
    std::size_t pos{0};

    BitOut() : buf(kJavaOutputBufBytes, 0) {}

    void byte(int b) {
        if (pos >= buf.size()) {
            buf.resize(buf.size() * 2, 0);
        }
        buf[pos++] = static_cast<std::uint8_t>(b & 0xFF);
    }

    void write_current(int b, int len) {
        if (len <= 0) {
            return;
        }
        free_bits -= len;
        current |= (b & static_cast<int>(low_mask64(len))) << free_bits;
        if (free_bits == 0) {
            byte(current);
            current = 0;
            free_bits = 8;
        }
        written_bits += len;
    }

    void bit(int b) { write_current(b ? 1 : 0, 1); }

    void bits(std::uint64_t x, int len) {
        if (len <= 0) {
            return;
        }
        if (len < 64) {
            x &= low_mask64(len);
        }
        if (len <= free_bits) {
            write_current(static_cast<int>(x), len);
            return;
        }
        int i = len - free_bits;
        const int queue = i & 7;
        if (free_bits != 0) {
            write_current(static_cast<int>(x >> i), free_bits);
        }
        if (queue != 0) {
            i -= queue;
            write_current(static_cast<int>(x), queue);
            x >>= queue;
        }
        if (i == 64) byte(static_cast<int>(x >> 56));
        if (i > 55) byte(static_cast<int>(x >> 48));
        if (i > 47) byte(static_cast<int>(x >> 40));
        if (i > 39) byte(static_cast<int>(x >> 32));
        if (i > 31) byte(static_cast<int>(x >> 24));
        if (i > 23) byte(static_cast<int>(x >> 16));
        if (i > 15) byte(static_cast<int>(x >> 8));
        if (i > 7) byte(static_cast<int>(x));
        written_bits += i;
    }

    void flush_align() {
        if (free_bits != 8) {
            write_current(0, free_bits);
        }
    }

    std::vector<std::uint8_t> first_bytes(long long bit_count) const {
        const auto n = static_cast<std::size_t>((bit_count + 7) / 8);
        if (n > buf.size()) {
            throw std::runtime_error("elf double size overflow");
        }
        return {buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n)};
    }
};

struct BitIn {
    std::span<const std::uint8_t> bytes;
    std::size_t pos{0};
    int fill{0};
    std::uint32_t current{0};

    explicit BitIn(std::span<const std::uint8_t> b) : bytes(b) {}

    int read_byte() {
        if (pos >= bytes.size()) {
            throw std::runtime_error("elf double input underflow");
        }
        return static_cast<int>(bytes[pos++]) & 0xFF;
    }

    int read_from_current(int len) {
        if (len <= 0) {
            return 0;
        }
        if (fill == 0) {
            current = static_cast<std::uint32_t>(read_byte());
            fill = 8;
        }
        fill -= len;
        return static_cast<int>((current >> fill) & static_cast<std::uint32_t>(low_mask64(len)));
    }

    std::uint64_t bits(int len) {
        if (len < 0 || len > 64) {
            throw std::runtime_error("elf double bad bit length");
        }
        std::uint64_t x = 0;
        for (int i = 0; i < len; ++i) {
            x = (x << 1) | static_cast<std::uint64_t>(read_from_current(1) & 1);
        }
        return x;
    }
};

std::uint64_t double_bits(double v) {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, 8);
    return u;
}

double bits_double(std::uint64_t u) {
    double v = 0;
    std::memcpy(&v, &u, 8);
    return v;
}

int f_alpha(int alpha) {
    if (alpha < 0) {
        throw std::runtime_error("elf double alpha < 0");
    }
    if (alpha >= 21) {
        static const double log2_10 = std::log(10.0) / std::log(2.0);
        return static_cast<int>(std::ceil(static_cast<double>(alpha) * log2_10));
    }
    return kFAlpha[alpha];
}

double pow10p(int i) {
    if (i < 0) {
        throw std::runtime_error("elf double negative pow10p");
    }
    return i < 21 ? kMap10iP[i] : std::pow(10.0, static_cast<double>(i));
}

double pow10n(int i) {
    if (i < 0) {
        throw std::runtime_error("elf double negative pow10n");
    }
    return i < 21 ? kMap10iN[i] : std::pow(10.0, -static_cast<double>(i));
}

int sp10(double v) {
    if (v >= 1.0) {
        int i = 0;
        while (i < 9) {
            if (v < static_cast<double>(kSpGt1[i + 1])) {
                return i;
            }
            ++i;
        }
    } else {
        int i = 1;
        while (i < 11) {
            if (v >= kSpLt1[i]) {
                return -i;
            }
            ++i;
        }
    }
    if (v == 0.0 || !std::isfinite(v)) {
        return std::numeric_limits<int>::min();
    }
    return static_cast<int>(std::floor(std::log10(v)));
}

void sp_and_flag(double v, int &sp, int &flag) {
    flag = 0;
    if (v >= 1.0) {
        int i = 0;
        while (i < 9) {
            if (v < static_cast<double>(kSpGt1[i + 1])) {
                sp = i;
                return;
            }
            ++i;
        }
    } else {
        int i = 1;
        while (i < 11) {
            if (v >= kSpLt1[i]) {
                sp = -i;
                flag = (v == kSpLt1[i]) ? 1 : 0;
                return;
            }
            ++i;
        }
    }
    const double l = std::log10(v);
    sp = static_cast<int>(std::floor(l));
    flag = (l == static_cast<long long>(l)) ? 1 : 0;
}

int significant_count(double v, int sp, int last_beta) {
    int i = 0;
    if (last_beta != kMaxBetaStar && last_beta != 0) {
        i = std::max(last_beta - sp - 1, 1);
    } else if (last_beta == kMaxBetaStar) {
        i = 17 - sp - 1;
    } else if (sp >= 0) {
        i = 1;
    } else {
        i = -sp;
    }
    constexpr int kMaxI = 340;
    if (i < 0 || i > kMaxI) {
        return 17;
    }
    double temp = v * pow10p(i);
    if (!std::isfinite(temp) || temp >= static_cast<double>(std::numeric_limits<long long>::max()))
        return 17;
    long long temp_long = static_cast<long long>(temp);
    while (static_cast<double>(temp_long) != temp) {
        if (++i > kMaxI) {
            return 17;
        }
        temp = v * pow10p(i);
        if (!std::isfinite(temp) || temp >= static_cast<double>(std::numeric_limits<long long>::max())) {
            return 17;
        }
        temp_long = static_cast<long long>(temp);
    }
    if (temp / pow10p(i) != v) {
        return 17;
    }
    while (i > 0 && temp_long % 10 == 0) {
        --i;
        temp_long /= 10;
    }
    return sp + i + 1;
}

void alpha_beta_star(double v_in, int last_beta, int &alpha, int &beta_star) {
    double v = v_in < 0 ? -v_in : v_in;
    int sp = 0, flag = 0;
    sp_and_flag(v, sp, flag);
    const int beta = significant_count(v, sp, last_beta);
    alpha = beta - sp - 1;
    beta_star = (flag == 1) ? 0 : beta;
}

double round_up(double v, int alpha) {
    const double scale = pow10p(alpha);
    if (v < 0) {
        return std::floor(v * scale) / scale;
    }
    return std::ceil(v * scale) / scale;
}

struct Chimp64Enc {
    BitOut &out;
    int stored_lz{kMaxBetaStar};
    std::uint64_t stored_val{0};
    bool first{true};
    long long size_bits{0};

    explicit Chimp64Enc(BitOut &o) : out(o) {}

    void add_raw(std::uint64_t value) {
        if (first) {
            first = false;
            stored_val = value;
            out.bits(stored_val, 64);
            size_bits += 64;
            return;
        }
        const auto xorv = exact_actions::adjacent_residual::xor_word(value, stored_val);
        const int leading = xorv == 0 ? 0 : kLeadRound[nlz64(xorv)];
        size_bits += exact_actions::residual_window_pack::chimp(
            [&](std::uint64_t v, int n) { if (n) out.bits(v, n); },
            xorv, 0, leading, ntz64(xorv), kLeadReprComp[leading], stored_lz,
            {64, 0, 6, kChimpThreshold});
        stored_val = value;
    }

    void close() {
        add_raw(kEndSign);
        out.bit(0);
        out.flush_align();
    }
};

struct Chimp64Dec {
    BitIn &in;
    int stored_lz{kMaxBetaStar};
    int stored_tz{0};
    std::uint64_t stored_val{0};
    bool first{true};
    bool eos{false};

    explicit Chimp64Dec(BitIn &i) : in(i) {}

    std::optional<std::uint64_t> read_raw() {
        if (eos) {
            return std::nullopt;
        }
        if (first) {
            first = false;
            stored_val = in.bits(64);
            if (stored_val == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            return stored_val;
        }
        const int flag = static_cast<int>(in.bits(2));
        if (flag == 3) {
            stored_lz = kLeadDec[static_cast<std::size_t>(in.bits(3))];
            int sig = 64 - stored_lz;
            if (sig == 0) sig = 64;
            std::uint64_t value = stored_val ^ in.bits(64 - stored_lz);
            if (value == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            stored_val = value;
            return stored_val;
        }
        if (flag == 2) {
            int sig = 64 - stored_lz;
            if (sig == 0) sig = 64;
            std::uint64_t value = stored_val ^ in.bits(64 - stored_lz);
            if (value == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            stored_val = value;
            return stored_val;
        }
        if (flag == 1) {
            stored_lz = kLeadDec[static_cast<std::size_t>(in.bits(3))];
            int sig = static_cast<int>(in.bits(6));
            if (sig == 0) sig = 64;
            stored_tz = 64 - sig - stored_lz;
            std::uint64_t value = in.bits(64 - stored_lz - stored_tz) << stored_tz;
            value ^= stored_val;
            if (value == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            stored_val = value;
            return stored_val;
        }
        return stored_val;
    }
};

struct FullXor64Enc {
    BitOut &out;
    std::uint64_t stored_val{0};
    bool first{true};

    explicit FullXor64Enc(BitOut &o) : out(o) {}

    void add_raw(std::uint64_t value) {
        if (first) {
            first = false;
            stored_val = value;
            out.bits(value, 64);
            return;
        }
        const std::uint64_t xorv = stored_val ^ value;
        if (xorv == 0) {
            out.bit(0);
        } else {
            out.bit(1);
            out.bits(xorv, 64);
        }
        stored_val = value;
    }

    void close() {
        add_raw(kEndSign);
        out.flush_align();
    }
};

struct FullXor64Dec {
    BitIn &in;
    std::uint64_t stored_val{0};
    bool first{true};
    bool eos{false};

    explicit FullXor64Dec(BitIn &i) : in(i) {}

    std::optional<std::uint64_t> read_raw() {
        if (eos) return std::nullopt;
        if (first) {
            first = false;
            stored_val = in.bits(64);
        } else if (in.bits(1) != 0) {
            stored_val ^= in.bits(64);
        }
        if (stored_val == kEndSign) {
            eos = true;
            return std::nullopt;
        }
        return stored_val;
    }
};

struct ElfXor64Enc {
    BitOut &out;
    int sl{kMaxBetaStar};
    int st{kMaxBetaStar};
    std::uint64_t sv{0};
    bool first{true};

    explicit ElfXor64Enc(BitOut &o) : out(o) {}

    void add_raw(std::uint64_t value) {
        if (first) {
            first = false;
            sv = value;
            const int tz = ntz64(value);
            out.bits(static_cast<std::uint64_t>(tz), 7);
            if (tz < 63) {
                out.bits(value >> (tz + 1), 63 - tz);
            }
            return;
        }
        const auto xorv = exact_actions::adjacent_residual::xor_word(value, sv);
        const int lz = xorv == 0 ? 0 : kLeadRound[nlz64(xorv)];
        exact_actions::residual_window_pack::elf_plus(
            [&](std::uint64_t v, int n) { if (n) out.bits(v, n); },
            xorv, lz, ntz64(xorv), kLeadReprComp[lz], sl, st, {64, 4, 6});
        sv = value;
    }

    void close() {
        add_raw(kEndSign);
        out.bit(0);
        out.flush_align();
    }
};

struct ElfXor64Dec {
    BitIn &in;
    std::uint64_t sv{0};
    int sl{kMaxBetaStar};
    int st{kMaxBetaStar};
    bool first{true};
    bool eos{false};

    explicit ElfXor64Dec(BitIn &i) : in(i) {}

    std::optional<std::uint64_t> read_raw() {
        if (eos) {
            return std::nullopt;
        }
        if (first) {
            first = false;
            const int tz = static_cast<int>(in.bits(7));
            sv = tz < 64 ? (((in.bits(63 - tz) << 1) + 1) << tz) : 0;
            if (sv == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            return sv;
        }
        const int flag = static_cast<int>(in.bits(2));
        if (flag == 3) {
            const int lc = static_cast<int>(in.bits(9));
            sl = kLeadDec[lc >> 6];
            int cb = lc & 0x3F;
            if (cb == 0) cb = 64;
            st = 64 - sl - cb;
            std::uint64_t v = (((in.bits(cb - 1) << 1) + 1) << st) ^ sv;
            if (v == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            sv = v;
            return sv;
        }
        if (flag == 2) {
            const int lc = static_cast<int>(in.bits(7));
            sl = kLeadDec[lc >> 4];
            int cb = lc & 0xF;
            if (cb == 0) cb = 16;
            st = 64 - sl - cb;
            std::uint64_t v = (((in.bits(cb - 1) << 1) + 1) << st) ^ sv;
            if (v == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            sv = v;
            return sv;
        }
        if (flag == 1) {
            return sv;
        }
        const int cb = 64 - sl - st;
        std::uint64_t v = (in.bits(cb) << st) ^ sv;
        if (v == kEndSign) {
            eos = true;
            return std::nullopt;
        }
        sv = v;
        return sv;
    }
};

struct Erase64Event {
    std::uint8_t tag;  // 0=reuse beta, 1=literal, 2=new beta
    std::uint8_t beta;
    std::uint64_t value;
};

double restore_erase64(const Erase64Event &event, int &last_beta) {
    if (event.tag == 1) return bits_double(event.value);
    if (event.tag == 2) last_beta = event.beta;
    else if (event.tag != 0 || event.beta != last_beta)
        throw std::runtime_error("Elf64 invalid eraser beta reuse");
    const double prime = bits_double(event.value);
    if (!std::isfinite(prime) || prime == 0.0 || last_beta > 15)
        throw std::runtime_error("Elf64 invalid erased value");
    const int sp = sp10(std::fabs(prime));
    if (last_beta == 0) {
        const double value = pow10n(-sp - 1);
        return prime < 0 ? -value : value;
    }
    return round_up(prime, last_beta - sp - 1);
}

Erase64Event erase_event64(double v, int &last_beta) {
    const std::uint64_t raw = double_bits(v);
    if (std::isnan(v)) {
        throw std::runtime_error("Elf64 NaN conflicts with the codec EOS marker; filter NaNs before encoding");
    }
    // Extreme exponents and large integral magnitudes need no decimal erasure.
    // Avoid undefined float-to-int conversions in the Java-derived search.
    if (v == 0.0 || std::isinf(v) || std::fabs(v) >= 1e18 || std::fabs(v) < 1e-300)
        return {1, 0, raw};
    int alpha = 0, beta = 0;
    alpha_beta_star(v, last_beta, alpha, beta);
    if (beta < 0 || beta > 15 || alpha < 0) return {1, 0, raw};
    const int e = static_cast<int>((raw >> 52) & 0x7FFu);
    const int g_alpha = f_alpha(alpha) + e - 1023;
    const int erase_bits = 52 - g_alpha;
    const int shift = erase_bits & 63;
    const std::uint64_t mask = shift == 0 ? ~0ull : (~0ull << shift);
    const std::uint64_t delta = (~mask) & raw;
    if (delta != 0 && erase_bits > 4) {
        Erase64Event event{static_cast<std::uint8_t>(beta == last_beta ? 0 : 2),
                           static_cast<std::uint8_t>(beta), mask & raw};
        int restored_beta = last_beta;
        try {
            if (double_bits(restore_erase64(event, restored_beta)) == raw) {
                last_beta = restored_beta;
                return event;
            }
        } catch (const std::exception &) {}
    }
    return {1, 0, raw};
}

void write_erase64_prefix(BitOut &out, const Erase64Event &event) {
    if (event.tag == 0) out.bit(0);
    else if (event.tag == 1) out.bits(2, 2);
    else if (event.tag == 2) out.bits(event.beta | 0x30u, 6);
    else throw std::runtime_error("Elf64 invalid eraser tag");
}

std::uint64_t erase_prefix(BitOut &out, double v, int &last_beta) {
    const auto event = erase_event64(v, last_beta);
    write_erase64_prefix(out, event);
    return event.value;
}

template <typename Backend>
std::vector<std::uint8_t> encode_elf_chunk(std::span<const double> chunk,
                                           long long &payload_bits) {
    BitOut out;
    Backend backend(out);
    int last_beta = kMaxBetaStar;
    for (double v : chunk) {
        backend.add_raw(erase_prefix(out, v, last_beta));
    }
    out.bits(2, 2);
    backend.close();
    payload_bits = out.written_bits;
    return out.first_bytes(payload_bits);
}

template <typename Backend>
std::vector<std::uint8_t> encode_elf_without_eraser(std::span<const double> chunk,
                                                    long long &payload_bits) {
    BitOut out;
    Backend backend(out);
    for (double value : chunk) backend.add_raw(double_bits(value));
    backend.close();
    payload_bits = out.written_bits;
    return out.first_bytes(payload_bits);
}

std::vector<std::uint8_t> encode_elf_without_xor(std::span<const double> chunk,
                                                 long long &payload_bits) {
    BitOut out;
    int last_beta = kMaxBetaStar;
    for (double value : chunk) {
        out.bits(erase_prefix(out, value, last_beta), 64);
    }
    out.bits(2, 2);
    out.flush_align();
    payload_bits = out.written_bits;
    return out.first_bytes(payload_bits);
}

std::vector<std::uint8_t> encode_chimp_chunk(std::span<const double> chunk,
                                             long long &payload_bits) {
    BitOut out;
    Chimp64Enc enc(out);
    for (double v : chunk) {
        enc.add_raw(double_bits(v));
    }
    enc.close();
    payload_bits = enc.size_bits;
    return out.first_bytes(payload_bits);
}

std::vector<std::uint8_t> encode_raw_chunk(std::span<const double> chunk,
                                           long long &payload_bits) {
    BitOut out;
    for (double value : chunk) out.bits(double_bits(value), 64);
    out.bits(kEndSign, 64);
    out.flush_align();
    payload_bits = out.written_bits;
    return out.first_bytes(payload_bits);
}

template <typename ReadRaw>
double elf_next(BitIn &in, ReadRaw &&read_raw, int &last_beta) {
    if (in.bits(1) == 0) {
        auto vp = read_raw();
        if (!vp) throw std::runtime_error("elf double early eos");
        const double vprime = bits_double(*vp);
        const int sp = sp10(std::fabs(vprime));
        if (last_beta == 0) {
            double v = pow10n(-sp - 1);
            return vprime < 0 ? -v : v;
        }
        return round_up(vprime, last_beta - sp - 1);
    }
    if (in.bits(1) == 0) {
        auto raw = read_raw();
        if (!raw) throw std::runtime_error("elf double early eos");
        return bits_double(*raw);
    }
    last_beta = static_cast<int>(in.bits(4));
    auto vp = read_raw();
    if (!vp) throw std::runtime_error("elf double early eos");
    const double vprime = bits_double(*vp);
    const int sp = sp10(std::fabs(vprime));
    if (last_beta == 0) {
        double v = pow10n(-sp - 1);
        return vprime < 0 ? -v : v;
    }
    return round_up(vprime, last_beta - sp - 1);
}

template <typename BackendDec>
bool decode_elf_chunk(std::span<const std::uint8_t> bytes, std::vector<double> &out,
                      int expected_n) {
    try {
        BitIn in(bytes);
        BackendDec dec(in);
        int last_beta = kMaxBetaStar;
        for (int i = 0; i < expected_n; ++i) {
            out.push_back(elf_next(in, [&dec]() { return dec.read_raw(); }, last_beta));
        }
        return static_cast<int>(out.size()) == expected_n;
    } catch (...) {
        return false;
    }
}

template <typename BackendDec>
bool decode_elf_without_eraser(std::span<const std::uint8_t> bytes,
                               std::vector<double> &out, int expected_n) {
    try {
        BitIn in(bytes);
        BackendDec dec(in);
        for (int i = 0; i < expected_n; ++i) {
            auto raw = dec.read_raw();
            if (!raw) return false;
            out.push_back(bits_double(*raw));
        }
        return static_cast<int>(out.size()) == expected_n;
    } catch (...) {
        return false;
    }
}

bool decode_elf_without_xor(std::span<const std::uint8_t> bytes,
                            std::vector<double> &out, int expected_n) {
    try {
        BitIn in(bytes);
        int last_beta = kMaxBetaStar;
        for (int i = 0; i < expected_n; ++i) {
            out.push_back(elf_next(in, [&in]() {
                return std::optional<std::uint64_t>(in.bits(64));
            }, last_beta));
        }
        return static_cast<int>(out.size()) == expected_n;
    } catch (...) {
        return false;
    }
}

bool decode_chimp_chunk(std::span<const std::uint8_t> bytes, std::vector<double> &out,
                        int expected_n) {
    try {
        BitIn in(bytes);
        Chimp64Dec dec(in);
        for (int i = 0; i < expected_n; ++i) {
            auto raw = dec.read_raw();
            if (!raw) return false;
            out.push_back(bits_double(*raw));
        }
        if (dec.read_raw()) return false;
        return static_cast<int>(out.size()) == expected_n;
    } catch (...) {
        return false;
    }
}

bool decode_raw_chunk(std::span<const std::uint8_t> bytes, std::vector<double> &out,
                      int expected_n) {
    try {
        BitIn in(bytes);
        for (int i = 0; i < expected_n; ++i) out.push_back(bits_double(in.bits(64)));
        return static_cast<int>(out.size()) == expected_n && in.bits(64) == kEndSign;
    } catch (...) {
        return false;
    }
}

bool same_bits(std::span<const double> a, const std::vector<double> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (double_bits(a[i]) != double_bits(b[i])) {
            return false;
        }
    }
    return true;
}

struct Metrics {
    long long orig{0}, comp{0}, ct{0}, dt{0};
};

template <typename Enc, typename Dec>
bool bench_elf_range(const std::vector<double> &col, int start, int end, Metrics &m,
                     LossAccum *loss) {
    long long bits = 0;
    std::span<const double> chunk(col.data() + start, static_cast<std::size_t>(end - start));
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto cbytes = encode_elf_chunk<Enc>(chunk, bits);
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<double> got;
    const auto t2 = std::chrono::high_resolution_clock::now();
    const bool ok = decode_elf_chunk<Dec>(std::span<const std::uint8_t>(cbytes.data(), cbytes.size()),
                                          got, end - start);
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (!ok || !same_bits(chunk, got)) {
        return false;
    }
    if (loss != nullptr) {
        benchmark_loss::accumulate_double_column(std::vector<double>(chunk.begin(), chunk.end()), got, *loss);
    }
    m.orig += static_cast<long long>(benchmark_wire::column_original_size(chunk.size(), true));
    m.comp += static_cast<long long>((bits + 7) / 8);
    m.ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    m.dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    return true;
}

template <typename Enc, typename Dec>
void bench_elf_adaptive(const std::vector<double> &col, int start, int end, Metrics &m,
                        LossAccum *loss) {
    if (start >= end) {
        return;
    }
    const int try_end = std::min(end, start + kElfChunk);
    try {
        if (bench_elf_range<Enc, Dec>(col, start, try_end, m, loss)) {
            bench_elf_adaptive<Enc, Dec>(col, try_end, end, m, loss);
            return;
        }
    } catch (...) {
    }
    if (try_end - start <= 1) {
        return;
    }
    const int mid = start + (try_end - start) / 2;
    bench_elf_adaptive<Enc, Dec>(col, start, mid, m, loss);
    bench_elf_adaptive<Enc, Dec>(col, mid, end, m, loss);
}

void bench_chimp(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                 LossAccum *loss) {
    Metrics total;
    for (const auto &col : columns) {
        for (int st = 0, n = static_cast<int>(col.size()); st < n; st += kChimpChunk) {
            const int en = std::min(n, st + kChimpChunk);
            try {
                long long bits = 0;
                std::span<const double> chunk(col.data() + st, static_cast<std::size_t>(en - st));
                const auto t0 = std::chrono::high_resolution_clock::now();
                auto cbytes = encode_chimp_chunk(chunk, bits);
                const auto t1 = std::chrono::high_resolution_clock::now();
                std::vector<double> got;
                const auto t2 = std::chrono::high_resolution_clock::now();
                const bool ok = decode_chimp_chunk(
                    std::span<const std::uint8_t>(cbytes.data(), cbytes.size()), got, en - st);
                const auto t3 = std::chrono::high_resolution_clock::now();
                if (!ok || !same_bits(chunk, got)) {
                    continue;
                }
                if (loss != nullptr) {
                    benchmark_loss::accumulate_double_column(std::vector<double>(chunk.begin(), chunk.end()),
                                                             got, *loss);
                }
                total.orig += static_cast<long long>(benchmark_wire::column_original_size(chunk.size(), true));
                total.comp += static_cast<long long>((bits + 7) / 8);
                total.ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                total.dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            } catch (...) {
            }
        }
    }
    result.assign({total.orig, total.comp, total.ct, total.dt});
}

void bench_chimp_ablation(const std::vector<std::vector<double>> &columns,
                          std::string_view omitted_operator,
                          std::vector<long long> &result, LossAccum *loss) {
    if (omitted_operator.empty() || omitted_operator == "none") {
        bench_chimp(columns, result, loss);
        return;
    }
    if (omitted_operator != "xor" && omitted_operator != "trailing_zero_pack") {
        throw std::invalid_argument("unsupported CHIMP ablation operator");
    }
    Metrics total;
    for (const auto &col : columns) {
        for (int start = 0, n = static_cast<int>(col.size()); start < n; start += kChimpChunk) {
            const int end = std::min(n, start + kChimpChunk);
            std::span<const double> chunk(col.data() + start,
                                          static_cast<std::size_t>(end - start));
            long long bits = 0;
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto payload = omitted_operator == "xor"
                                     ? encode_raw_chunk(chunk, bits)
                                     : encode_elf_without_eraser<FullXor64Enc>(chunk, bits);
            const auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<double> decoded;
            const auto t2 = std::chrono::high_resolution_clock::now();
            const bool ok = omitted_operator == "xor"
                                ? decode_raw_chunk(payload, decoded, end - start)
                                : decode_elf_without_eraser<FullXor64Dec>(
                                      payload, decoded, end - start);
            const auto t3 = std::chrono::high_resolution_clock::now();
            if (!ok || !same_bits(chunk, decoded)) {
                throw std::runtime_error("CHIMP ablation round-trip mismatch");
            }
            if (loss != nullptr) {
                benchmark_loss::accumulate_double_column(
                    std::vector<double>(chunk.begin(), chunk.end()), decoded, *loss);
            }
            total.orig += static_cast<long long>(benchmark_wire::column_original_size(chunk.size(), true));
            total.comp += static_cast<long long>((bits + 7) / 8);
            total.ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            total.dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
    }
    result.assign({total.orig, total.comp, total.ct, total.dt});
}

template <typename Enc, typename Dec>
void bench_elf(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
               LossAccum *loss) {
    Metrics total;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        Metrics m;
        bench_elf_adaptive<Enc, Dec>(col, 0, static_cast<int>(col.size()), m, loss);
        total.orig += m.orig;
        total.comp += m.comp;
        total.ct += m.ct;
        total.dt += m.dt;
    }
    result.assign({total.orig, total.comp, total.ct, total.dt});
}

template <typename BaselineEnc, typename BaselineDec>
void bench_elf_ablation(const std::vector<std::vector<double>> &columns,
                        std::string_view omitted_operator,
                        std::vector<long long> &result, LossAccum *loss) {
    if (omitted_operator.empty() || omitted_operator == "none") {
        bench_elf<BaselineEnc, BaselineDec>(columns, result, loss);
        return;
    }
    if (omitted_operator != "eraser" && omitted_operator != "xor" &&
        omitted_operator != "trailing_zero_pack") {
        throw std::invalid_argument("unsupported ELF ablation operator");
    }
    Metrics total;
    for (const auto &col : columns) {
        for (int start = 0, n = static_cast<int>(col.size()); start < n; start += kChimpChunk) {
            const int end = std::min(n, start + kChimpChunk);
            std::span<const double> chunk(col.data() + start,
                                          static_cast<std::size_t>(end - start));
            long long bits = 0;
            const auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::uint8_t> payload;
            if (omitted_operator == "eraser") {
                payload = encode_elf_without_eraser<BaselineEnc>(chunk, bits);
            } else if (omitted_operator == "xor") {
                payload = encode_elf_without_xor(chunk, bits);
            } else {
                payload = encode_elf_chunk<FullXor64Enc>(chunk, bits);
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<double> decoded;
            const auto t2 = std::chrono::high_resolution_clock::now();
            bool ok = false;
            if (omitted_operator == "eraser") {
                ok = decode_elf_without_eraser<BaselineDec>(payload, decoded, end - start);
            } else if (omitted_operator == "xor") {
                ok = decode_elf_without_xor(payload, decoded, end - start);
            } else {
                ok = decode_elf_chunk<FullXor64Dec>(payload, decoded, end - start);
            }
            const auto t3 = std::chrono::high_resolution_clock::now();
            if (!ok || !same_bits(chunk, decoded)) {
                // Some zero-heavy columns are not representable by an exposed
                // bypass.  Preserve a lossless, decodable fallback for that
                // chunk rather than assigning a non-lossless measurement.
                bits = 0;
                payload = encode_raw_chunk(chunk, bits);
                decoded.clear();
                if (!decode_raw_chunk(payload, decoded, end - start) ||
                    !same_bits(chunk, decoded)) {
                    throw std::runtime_error("ELF ablation round-trip mismatch");
                }
            }
            if (loss != nullptr) {
                benchmark_loss::accumulate_double_column(
                    std::vector<double>(chunk.begin(), chunk.end()), decoded, *loss);
            }
            total.orig += static_cast<long long>(benchmark_wire::column_original_size(chunk.size(), true));
            total.comp += static_cast<long long>((bits + 7) / 8);
            total.ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            total.dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
    }
    result.assign({total.orig, total.comp, total.ct, total.dt});
}

} // namespace

void benchChimpDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                     std::vector<long long> &result,
                                     LossAccum *loss) {
    bench_chimp(columns, result, loss);
}

void benchChimpDoubleColumnsAblation(
    const std::vector<std::vector<double>> &columns, std::string_view omitted_operator,
    std::vector<long long> &result, LossAccum *loss) {
    bench_chimp_ablation(columns, omitted_operator, result, loss);
}

void benchElfOnChimpDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                          std::vector<long long> &result,
                                          LossAccum *loss) {
    bench_elf<Chimp64Enc, Chimp64Dec>(columns, result, loss);
}

void benchElfPlusDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                       std::vector<long long> &result,
                                       LossAccum *loss) {
    bench_elf<ElfXor64Enc, ElfXor64Dec>(columns, result, loss);
}

void benchElfOnChimpDoubleColumnsAblation(
    const std::vector<std::vector<double>> &columns, std::string_view omitted_operator,
    std::vector<long long> &result, LossAccum *loss) {
    bench_elf_ablation<Chimp64Enc, Chimp64Dec>(columns, omitted_operator, result, loss);
}

void benchElfPlusDoubleColumnsAblation(
    const std::vector<std::vector<double>> &columns, std::string_view omitted_operator,
    std::vector<long long> &result, LossAccum *loss) {
    bench_elf_ablation<ElfXor64Enc, ElfXor64Dec>(columns, omitted_operator, result, loss);
}

namespace {

void put_le(std::vector<std::uint8_t> &out, std::uint64_t value, int width) {
    for (int i = 0; i < width; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

std::uint64_t get_le(std::span<const std::uint8_t> data, std::size_t &pos, int width) {
    if (pos > data.size() || data.size() - pos < static_cast<std::size_t>(width))
        throw std::runtime_error("truncated Elf64 packet");
    std::uint64_t result = 0;
    for (int i = 0; i < width; ++i) result |= static_cast<std::uint64_t>(data[pos++]) << (8 * i);
    return result;
}

std::vector<double> read_doubles(std::span<const std::uint8_t> raw) {
    if (raw.size() % 8 || raw.size() / 8 > UINT32_MAX)
        throw std::runtime_error("Elf64 input must be complete float64 lanes");
    std::vector<double> values(raw.size() / 8);
    if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
    return values;
}

std::vector<std::uint8_t> event_packet(std::span<const Erase64Event> events) {
    std::vector<std::uint8_t> out{'E', 'E', '6', '4'};
    put_le(out, events.size(), 4);
    for (const auto &event : events) {
        out.push_back(event.tag);
        out.push_back(event.beta);
        put_le(out, event.value, 8);
    }
    return out;
}

std::vector<Erase64Event> read_events(std::span<const std::uint8_t> packet) {
    if (packet.size() < 8 || std::memcmp(packet.data(), "EE64", 4))
        throw std::runtime_error("invalid Elf64 eraser packet");
    std::size_t pos = 4;
    const auto count = get_le(packet, pos, 4);
    if (count != (packet.size() - pos) / 10 || (packet.size() - pos) % 10)
        throw std::runtime_error("Elf64 eraser event count mismatch");
    std::vector<Erase64Event> events;
    int beta = kMaxBetaStar;
    for (std::uint64_t i = 0; i < count; ++i) {
        if (i % kElfChunk == 0) beta = kMaxBetaStar;
        const auto tag = packet[pos++];
        const auto code = packet[pos++];
        const auto value = get_le(packet, pos, 8);
        if (tag > 2 || code > 15 || (tag == 1 && code != 0) || std::isnan(bits_double(value)))
            throw std::runtime_error("invalid Elf64 event fields");
        events.push_back({tag, code, value});
        (void)restore_erase64(events.back(), beta);
    }
    return events;
}

std::vector<std::uint8_t> wire_header(std::size_t count, bool plus) {
    std::vector<std::uint8_t> out{'E', 'L', '6', '4', static_cast<std::uint8_t>(plus)};
    put_le(out, count, 4);
    return out;
}

template <class Backend>
std::vector<std::uint8_t> write_events(std::span<const Erase64Event> events, bool plus) {
    auto out = wire_header(events.size(), plus);
    for (std::size_t start = 0; start < events.size(); start += kElfChunk) {
        const auto count = std::min<std::size_t>(kElfChunk, events.size() - start);
        BitOut bits;
        Backend backend(bits);
        for (const auto &event : events.subspan(start, count)) {
            write_erase64_prefix(bits, event);
            backend.add_raw(event.value);
        }
        bits.bits(2, 2);
        backend.close();
        const auto body = bits.first_bytes(bits.written_bits);
        put_le(out, count, 4);
        put_le(out, body.size(), 4);
        out.insert(out.end(), body.begin(), body.end());
    }
    return out;
}

template <class Decoder>
std::vector<Erase64Event> read_wire(std::span<const std::uint8_t> wire, bool plus) {
    if (wire.size() < 9 || std::memcmp(wire.data(), "EL64", 4) || wire[4] != plus)
        throw std::runtime_error("invalid Elf64 wire or algorithm variant");
    std::size_t pos = 5;
    const auto total = get_le(wire, pos, 4);
    std::vector<Erase64Event> events;
    while (events.size() < total) {
        const auto count = get_le(wire, pos, 4);
        const auto size = get_le(wire, pos, 4);
        if (count != std::min<std::size_t>(kElfChunk, total - events.size()) ||
            size > wire.size() - pos) throw std::runtime_error("Elf64 invalid chunk extent");
        BitIn bits(wire.subspan(pos, size));
        Decoder decoder(bits);
        int last_beta = kMaxBetaStar;
        for (std::size_t i = 0; i < count; ++i) {
            std::uint8_t tag, beta = 0;
            if (bits.bits(1) == 0) {
                if (last_beta > 15) throw std::runtime_error("Elf64 beta reused before initialization");
                tag = 0;
                beta = static_cast<std::uint8_t>(last_beta);
            } else if (bits.bits(1) == 0) {
                tag = 1;
            } else {
                tag = 2;
                beta = static_cast<std::uint8_t>(bits.bits(4));
                last_beta = beta;
            }
            const auto value = decoder.read_raw();
            if (!value) throw std::runtime_error("Elf64 unexpected early EOS");
            events.push_back({tag, beta, *value});
        }
        if (bits.bits(2) != 2 || decoder.read_raw())
            throw std::runtime_error("Elf64 invalid close marker");
        // close() writes one zero bit, then at most seven padding zeroes.
        if (bits.bits(1) != 0 || bits.pos != size ||
            (bits.current & low_mask64(bits.fill)) != 0)
            throw std::runtime_error("Elf64 trailing chunk bits");
        pos += size;
    }
    if (pos != wire.size()) throw std::runtime_error("Elf64 trailing wire bytes");
    return events;
}

template <class Backend>
std::vector<std::uint8_t> original_wire(std::span<const std::uint8_t> raw, bool plus) {
    const auto values = read_doubles(raw);
    auto out = wire_header(values.size(), plus);
    for (std::size_t start = 0; start < values.size(); start += kElfChunk) {
        const auto count = std::min<std::size_t>(kElfChunk, values.size() - start);
        long long bits = 0;
        const auto body = encode_elf_chunk<Backend>(std::span(values).subspan(start, count), bits);
        put_le(out, count, 4);
        put_le(out, body.size(), 4);
        out.insert(out.end(), body.begin(), body.end());
    }
    return out;
}

}  // namespace

std::vector<std::uint8_t> eraser64Forward(std::span<const std::uint8_t> raw) {
    const auto values = read_doubles(raw);
    std::vector<Erase64Event> events;
    int beta = kMaxBetaStar;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i % kElfChunk == 0) beta = kMaxBetaStar;
        events.push_back(erase_event64(values[i], beta));
    }
    return event_packet(events);
}

std::vector<std::uint8_t> eraser64Inverse(std::span<const std::uint8_t> packet) {
    const auto events = read_events(packet);
    std::vector<std::uint8_t> out;
    int beta = kMaxBetaStar;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i % kElfChunk == 0) beta = kMaxBetaStar;
        put_le(out, double_bits(restore_erase64(events[i], beta)), 8);
    }
    return out;
}

std::vector<std::uint8_t> xor64Forward(std::span<const std::uint8_t> packet, bool plus) {
    const auto events = read_events(packet);
    return plus ? write_events<ElfXor64Enc>(events, true) : write_events<Chimp64Enc>(events, false);
}

std::vector<std::uint8_t> xor64Inverse(std::span<const std::uint8_t> wire, bool plus) {
    return event_packet(plus ? read_wire<ElfXor64Dec>(wire, true) : read_wire<Chimp64Dec>(wire, false));
}

std::vector<std::uint8_t> encodeDoubleColumnExact(std::span<const std::uint8_t> raw, bool plus) {
    return plus ? original_wire<ElfXor64Enc>(raw, true) : original_wire<Chimp64Enc>(raw, false);
}

std::vector<std::uint8_t> decodeDoubleColumnExact(std::span<const std::uint8_t> wire, bool plus) {
    return eraser64Inverse(xor64Inverse(wire, plus));
}

} // namespace elf_double
