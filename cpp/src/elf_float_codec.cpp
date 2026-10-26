#include "elf_float_codec.hpp"
#include "adjacent_residual.hpp"
#include "residual_window_pack.hpp"
#include "benchmark_wire.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace elf_float {
namespace detail {

constexpr int kElfChunk = 2048;
/** Matches {@code Chimp32} / {@code ElfXORCompressor32} in start-compress JAR ({@code new byte[5000]}). */
constexpr std::size_t kJavaElfOutputBufBytes = 5000;
constexpr int kChimpThreshold = 5;
constexpr std::uint32_t kNanBits = 0x7fc00000u;
constexpr std::uint32_t kEndSign = 0x7fc00000u;
constexpr int kMaxBetaStar = 2147483647;

alignas(64) static constexpr std::uint8_t kLeadReprComp[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

alignas(64) static constexpr std::uint8_t kLeadRound[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};

alignas(64) static constexpr std::uint8_t kLeadDec[8] = {0, 8, 12, 16, 18, 20, 22, 24};

alignas(64) static constexpr std::uint8_t kXorLr[32] = {
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

alignas(64) static constexpr std::uint8_t kXorRound[32] = {
    0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20};

alignas(64) static constexpr std::uint8_t kXorDecLr[8] = {0, 6, 10, 12, 14, 16, 18, 20};

alignas(64) static constexpr int kFAlpha[21] = {0,  4,  7,  10, 14, 17, 20, 24, 27, 30, 34, 37, 40,
                                                44, 47, 50, 54, 57, 60, 64, 67};

static constexpr float kMap10iP[] = {
    1.0e0f,  1.0e1f,  1.0e2f,  1.0e3f,  1.0e4f,  1.0e5f,  1.0e6f,  1.0e7f,  1.0e8f,
    1.0e9f,  1.0e10f, 1.0e11f, 1.0e12f, 1.0e13f, 1.0e14f, 1.0e15f, 1.0e16f, 1.0e17f,
    1.0e18f, 1.0e19f, 1.0e20f};

static constexpr float kMap10iN[] = {
    1.0e0f,  1.0e-1f, 1.0e-2f, 1.0e-3f, 1.0e-4f, 1.0e-5f, 1.0e-6f, 1.0e-7f, 1.0e-8f,
    1.0e-9f, 1.0e-10f, 1.0e-11f, 1.0e-12f, 1.0e-13f, 1.0e-14f, 1.0e-15f, 1.0e-16f,
    1.0e-17f, 1.0e-18f, 1.0e-19f, 1.0e-20f};

static constexpr int kSpGt1[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

static constexpr float kSpLt1[] = {1.0f,      0.1f,      0.01f,      0.001f,      0.0001f,     0.00001f,
                                   0.000001f, 0.0000001f, 0.00000001f, 0.000000001f, 0.0000000001f};

static double log2_10_cache() {
    static const double v = std::log(10.0) / std::log(2.0);
    return v;
}

static int nlz_u32(std::uint32_t x) {
    if (x == 0) {
        return 32;
    }
    return __builtin_clz(x);
}

static int ntz_u32(std::uint32_t x) {
    if (x == 0) {
        return 32;
    }
    return __builtin_ctz(x);
}

static unsigned low_bits_mask(int len) {
    if (len <= 0) {
        return 0;
    }
    if (len >= 32) {
        return 0xFFFFFFFFu;
    }
    return (1u << len) - 1u;
}

struct VignaOut {
    std::vector<std::uint8_t> buf;
    long long written_bits{0};
    int current{0};
    int free_bits{8};
    std::size_t pos{0};

    VignaOut() { buf.assign(kJavaElfOutputBufBytes, 0); }

    void write_byte(int b) {
        if (pos >= buf.size()) {
            throw std::runtime_error("elf_float: output overflow");
        }
        buf[pos++] = static_cast<std::uint8_t>(b & 0xFF);
    }

    void write_in_current(int b, int len) {
        if (len == 0) {
            return;
        }
        free_bits -= len;
        current |= (b & static_cast<int>(low_bits_mask(len))) << free_bits;
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
            write_in_current(static_cast<int>(x & low_bits_mask(queue)), queue);
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
            write_in_current(static_cast<int>(x & low_bits_mask64(queue)), queue);
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

    static std::uint64_t low_bits_mask64(int q) {
        if (q <= 0) {
            return 0;
        }
        if (q >= 64) {
            return 0xFFFFFFFFFFFFFFFFull;
        }
        return (1ull << q) - 1ull;
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
                throw std::runtime_error("elf_float: truncated bit stream");
            }
            current = static_cast<std::uint32_t>(b);
            fill = 8;
        }
        fill -= l;
        return static_cast<int>((current >> fill) & static_cast<std::uint32_t>(low_bits_mask(l)));
    }

    int read_bit() { return read_from_current(1); }

    int read_int(int l) {
        if (l < 0 || l > 32) {
            throw std::runtime_error("elf_float: bad read_int");
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

    std::uint64_t read_long(int l) {
        if (l < 0 || l > 63) {
            throw std::runtime_error("elf_float: bad read_long");
        }
        std::uint64_t x = 0;
        for (int i = 0; i < l; i++) {
            x = (x << 1) | static_cast<std::uint64_t>(read_bit() & 1);
        }
        return x;
    }
};

static int get_f_alpha(int alpha) {
    if (alpha < 0) {
        throw std::runtime_error("elf_float: alpha");
    }
    if (alpha >= 21) {
        return static_cast<int>(std::ceil(static_cast<double>(alpha) * log2_10_cache()));
    }
    return kFAlpha[static_cast<std::size_t>(alpha)];
}

static float get10i_p(int i) {
    if (i < 0) {
        throw std::runtime_error("elf_float: get10i_p");
    }
    if (i >= 21) {
        return std::pow(10.0f, static_cast<float>(i));
    }
    return kMap10iP[static_cast<std::size_t>(i)];
}

static float get10i_n(int i) {
    if (i < 0) {
        throw std::runtime_error("elf_float: get10i_n");
    }
    if (i >= 21) {
        return std::pow(10.0f, -static_cast<float>(i));
    }
    return kMap10iN[static_cast<std::size_t>(i)];
}

static int get_sp(double v) {
    if (v >= 1.0) {
        int i = 0;
        while (i < 9) {
            if (v < static_cast<double>(kSpGt1[i + 1])) {
                return i;
            }
            i++;
        }
    } else {
        int i = 1;
        while (i < 11) {
            if (v >= static_cast<double>(kSpLt1[static_cast<std::size_t>(i)])) {
                return -i;
            }
            i++;
        }
    }
    return static_cast<int>(std::floor(std::log10(v)));
}

static void get_sp_and_10in_flag(double v, int &sp_out, int &flag_out) {
    if (v >= 1.0) {
        int i = 0;
        while (i < 9) {
            if (v < static_cast<double>(kSpGt1[i + 1])) {
                sp_out = i;
                flag_out = 0;
                return;
            }
            i++;
        }
    } else {
        int i = 1;
        while (i < 11) {
            double boundary = static_cast<double>(kSpLt1[static_cast<std::size_t>(i)]);
            if (v >= boundary) {
                sp_out = -i;
                flag_out = (v == boundary) ? 1 : 0;
                return;
            }
            i++;
        }
    }
    double log10v = std::log10(v);
    sp_out = static_cast<int>(std::floor(log10v));
    flag_out = (log10v == static_cast<long long>(log10v)) ? 1 : 0;
}

static int significant_count(float v_f, int sp, int last_beta_star) {
    // Matches Elf32Utils.getSignificantCount; must terminate: unbounded i++ can spin forever
    // (or until non-finite temp) on pathological floats if the decimal-scale search never lands
    // on an integer-valued `temp`. Java relies on later `temp/get10iP(i)!=v` — still after an
    // infinite loop — so cap i for native bench stability.
    if (!std::isfinite(static_cast<double>(v_f))) {
        return 8;
    }
    constexpr int kMaxDecimalScaleI = 64;
    int i = 0;
    if (last_beta_star != kMaxBetaStar && last_beta_star != 0) {
        i = std::max(last_beta_star - sp - 1, 1);
    } else if (last_beta_star == kMaxBetaStar) {
        i = 8 - sp - 1;
    } else if (sp >= 0) {
        i = 1;
    } else {
        i = -sp;
    }
    // Large finite float32 values can start with a negative decimal scale.
    // beta=sp+1 selects alpha=0, preserving the native literal path.
    if (i < 0) return sp + 1;
    if (i > kMaxDecimalScaleI) {
        return 8;
    }
    float temp = static_cast<float>(static_cast<double>(v_f) * static_cast<double>(get10i_p(i)));
    if (!std::isfinite(temp)) {
        return 8;
    }
    int temp_int = static_cast<int>(temp);
    while (static_cast<float>(temp_int) != temp) {
        i++;
        if (i > kMaxDecimalScaleI) {
            return 8;
        }
        temp = static_cast<float>(static_cast<double>(v_f) * static_cast<double>(get10i_p(i)));
        if (!std::isfinite(temp)) {
            return 8;
        }
        temp_int = static_cast<int>(temp);
    }
    if (static_cast<float>(static_cast<double>(temp) / static_cast<double>(get10i_p(i))) != v_f) {
        return 8;
    }
    while (i > 0 && temp_int % 10 == 0) {
        i--;
        temp_int /= 10;
    }
    return sp + i + 1;
}

static void get_alpha_and_beta_star(float v_in, int last_beta_star, int &alpha_out, int &beta_star_out) {
    float v = v_in;
    if (v < 0) {
        v = -v;
    }
    int sp = 0, flag = 0;
    get_sp_and_10in_flag(static_cast<double>(v), sp, flag);
    int beta = significant_count(v, sp, last_beta_star);
    alpha_out = beta - sp - 1;
    beta_star_out = (flag == 1) ? 0 : beta;
}

static float round_up_f(float v, int alpha) {
    float scale = get10i_p(alpha);
    double vv = static_cast<double>(v);
    if (v < 0) {
        return static_cast<float>(std::floor(vv * static_cast<double>(scale)) / static_cast<double>(scale));
    }
    return static_cast<float>(std::ceil(vv * static_cast<double>(scale)) / static_cast<double>(scale));
}

static std::uint32_t float_to_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

static float bits_to_float(std::uint32_t u) {
    float f = 0;
    std::memcpy(&f, &u, 4);
    return f;
}

static std::uint32_t elf_erase_prefix_write(VignaOut &out, float v, int &last_beta) {
    std::uint32_t v_int = float_to_bits(v);
    std::uint32_t v_prime = v_int;
    if (v == 0.0f || std::isinf(v)) {
        out.write_int(2, 2);
        return v_prime;
    }
    if (std::isnan(v)) {
        out.write_int(2, 2);
        return 0x7fc00000u;
    }
    int ab_alpha = 0, ab_beta = 0;
    get_alpha_and_beta_star(v, last_beta, ab_alpha, ab_beta);
    int e = static_cast<int>((v_int >> 23) & 0xFFu);
    int g_alpha = get_f_alpha(ab_alpha) + e - 127;
    int erase_bits = 23 - g_alpha;
    int es = erase_bits & 31;
    std::uint32_t mask =
        (es == 0) ? 0xFFFFFFFFu : ((0xFFFFFFFFu << es) & 0xFFFFFFFFu);
    std::uint32_t delta = ((~mask) & 0xFFFFFFFFu) & v_int;
    if (delta != 0 && erase_bits > 3) {
        if (ab_beta == last_beta) {
            out.write_bit(0);
        } else {
            out.write_int(static_cast<std::uint32_t>(ab_beta | 0x18), 5);
            last_beta = ab_beta;
        }
        v_prime = mask & v_int;
    } else {
        out.write_int(2, 2);
        v_prime = v_int;
    }
    return v_prime;
}

struct ChimpStreamEnc {
    VignaOut &out;
    int stored_lz{2147483647};
    std::uint32_t stored_val{0};
    bool first{true};

    explicit ChimpStreamEnc(VignaOut &o) : out(o) {}

    void write_first_u32(std::uint32_t value) {
        first = false;
        stored_val = value;
        out.write_int(stored_val, 32);
    }

    void compress_value(std::uint32_t value) {
        const auto xorv = exact_actions::adjacent_residual::xor_word(value, stored_val);
        const int leading = xorv == 0 ? 0 : kLeadRound[nlz_u32(xorv)];
        exact_actions::residual_window_pack::chimp(
            [&](std::uint64_t v, int n) { if (n) out.write_int(static_cast<std::uint32_t>(v), n); },
            xorv, 0, leading, ntz_u32(xorv), kLeadReprComp[leading], stored_lz,
            {32, 0, 5, kChimpThreshold});
        stored_val = value;
    }

    void add_raw_u32(std::uint32_t bits) {
        if (first) {
            write_first_u32(bits);
        } else {
            compress_value(bits);
        }
    }

    void close_chimp() {
        add_raw_u32(kNanBits);
        out.write_bit(0);
        out.align_flush();
    }
};

struct XorStreamEnc {
    VignaOut &out;
    int sl{kMaxBetaStar};
    int st{kMaxBetaStar};
    std::uint32_t sv{0};
    bool first{true};

    explicit XorStreamEnc(VignaOut &o) : out(o) {}

    void add_raw_u32(std::uint32_t v) {
        v &= 0xFFFFFFFFu;
        if (first) {
            first = false;
            sv = v;
            int tz = ntz_u32(v);
            out.write_int(static_cast<std::uint32_t>(tz), 6);
            if (tz < 32) {
                const int mid_len = 31 - tz;
                // Decoder uses read_int (not read_long); write_int matches VignaIn and Python ElfXOR.
                out.write_int(static_cast<std::uint32_t>((static_cast<std::uint64_t>(sv) >> (tz + 1)) &
                                                         low_bits_mask(mid_len)),
                              mid_len);
            }
            return;
        }
        const auto xorv = exact_actions::adjacent_residual::xor_word(v, sv);
        const int lz = xorv == 0 ? 0 : kXorRound[nlz_u32(xorv)];
        exact_actions::residual_window_pack::elf_plus(
            [&](std::uint64_t value, int n) { if (n) out.write_int(static_cast<std::uint32_t>(value), n); },
            xorv, lz, ntz_u32(xorv), kXorLr[lz], sl, st, {32, 3, 5});
        sv = v;
    }

    void close_xor() {
        add_raw_u32(kEndSign);
        out.write_bit(0);
        out.align_flush();
    }
};

struct ElfOnChimpChunkEnc {
    VignaOut out;
    ChimpStreamEnc chimp;
    int last_beta{kMaxBetaStar};

    ElfOnChimpChunkEnc() : chimp(out) {}

    void add_value(float v) {
        std::uint32_t vp = elf_erase_prefix_write(out, v, last_beta);
        chimp.add_raw_u32(vp);
    }

    void close_chunk() {
        out.write_int(2, 2);
        chimp.close_chimp();
    }

    std::vector<std::uint8_t> to_bytes() const {
        const int comp_bytes = static_cast<int>((out.written_bits + 7) / 8);
        if (comp_bytes > static_cast<int>(out.buf.size())) {
            throw std::runtime_error("elf_float: compressed payload larger than bitstream buffer");
        }
        return std::vector<std::uint8_t>(out.buf.begin(), out.buf.begin() + comp_bytes);
    }
};

struct ElfPlusChunkEnc {
    VignaOut out;
    XorStreamEnc xorc;
    int last_beta{kMaxBetaStar};

    ElfPlusChunkEnc() : xorc(out) {}

    void add_value(float v) {
        std::uint32_t vp = elf_erase_prefix_write(out, v, last_beta);
        xorc.add_raw_u32(vp);
    }

    void close_chunk() {
        out.write_int(2, 2);
        xorc.close_xor();
    }

    std::vector<std::uint8_t> to_bytes() const {
        const int comp_bytes = static_cast<int>((out.written_bits + 7) / 8);
        if (comp_bytes > static_cast<int>(out.buf.size())) {
            throw std::runtime_error("elf_float: compressed payload larger than bitstream buffer");
        }
        return std::vector<std::uint8_t>(out.buf.begin(), out.buf.begin() + comp_bytes);
    }
};

std::vector<std::uint8_t> encode_on_chimp_chunk(std::span<const float> col) {
    ElfOnChimpChunkEnc enc;
    for (float v : col) {
        enc.add_value(v);
    }
    enc.close_chunk();
    return enc.to_bytes();
}

std::vector<std::uint8_t> encode_elf_plus_chunk(std::span<const float> col) {
    ElfPlusChunkEnc enc;
    for (float v : col) {
        enc.add_value(v);
    }
    enc.close_chunk();
    return enc.to_bytes();
}

struct ChimpStreamDec {
    VignaIn &in;
    int stored_lz{kMaxBetaStar};
    int stored_tz{0};
    std::uint32_t stored_val{0};
    bool first{true};
    bool eos{false};

    explicit ChimpStreamDec(VignaIn &i) : in(i) {}

    std::optional<std::uint32_t> read_pair() {
        if (eos) {
            return std::nullopt;
        }
        if (first) {
            first = false;
            stored_val = static_cast<std::uint32_t>(in.read_int(32));
            if (stored_val == kNanBits) {
                eos = true;
                return std::nullopt;
            }
            return stored_val;
        }
        int bit = in.read_bit();
        if (bit == 1) {
            if (in.read_bit() == 1) {
                int idx = in.read_int(3);
                stored_lz = static_cast<int>(kLeadDec[static_cast<std::size_t>(idx)]);
            }
            int significant_bits = 32 - stored_lz;
            if (significant_bits == 0) {
                significant_bits = 32;
            }
            std::uint32_t value = static_cast<std::uint32_t>(in.read_int(32 - stored_lz));
            value = stored_val ^ value;
            if (value == kNanBits) {
                eos = true;
                return std::nullopt;
            }
            stored_val = value;
            return stored_val;
        }
        if (in.read_bit() == 1) {
            int idx = in.read_int(3);
            stored_lz = static_cast<int>(kLeadDec[static_cast<std::size_t>(idx)]);
            int significant_bits = in.read_int(5);
            if (significant_bits == 0) {
                significant_bits = 32;
            }
            stored_tz = 32 - significant_bits - stored_lz;
            std::uint32_t value = static_cast<std::uint32_t>(in.read_int(32 - stored_lz - stored_tz));
            value <<= stored_tz;
            value = stored_val ^ value;
            if (value == kNanBits) {
                eos = true;
                return std::nullopt;
            }
            stored_val = value;
            return stored_val;
        }
        return stored_val;
    }
};

struct XorStreamDec {
    VignaIn &in;
    std::uint32_t sv{0};
    int sl{kMaxBetaStar};
    int st{kMaxBetaStar};
    bool first{true};
    bool eos{false};

    explicit XorStreamDec(VignaIn &i) : in(i) {}

    std::optional<std::uint32_t> read_value() {
        if (eos) {
            return std::nullopt;
        }
        if (first) {
            first = false;
            int tz = in.read_int(6);
            if (tz < 32) {
                int mid = in.read_int(31 - tz);
                sv = ((((static_cast<std::uint32_t>(mid) << 1) + 1) << tz) & 0xFFFFFFFFu);
            } else {
                sv = 0;
            }
            if (sv == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            return sv;
        }
        int fl = in.read_int(2);
        if (fl == 3) {
            int lc = in.read_int(8);
            sl = kXorDecLr[lc >> 5];
            int cb = lc & 0x1F;
            if (cb == 0) {
                cb = 32;
            }
            st = 32 - sl - cb;
            int mid = in.read_int(cb - 1);
            std::uint32_t val = (((static_cast<std::uint32_t>(mid) << 1) + 1) << st) & 0xFFFFFFFFu;
            val = (sv ^ val) & 0xFFFFFFFFu;
            if (val == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            sv = val;
            return sv;
        }
        if (fl == 2) {
            int lc = in.read_int(6);
            sl = kXorDecLr[lc >> 3];
            int cb = lc & 7;
            if (cb == 0) {
                cb = 8;
            }
            st = 32 - sl - cb;
            int mid = in.read_int(cb - 1);
            std::uint32_t val = (((static_cast<std::uint32_t>(mid) << 1) + 1) << st) & 0xFFFFFFFFu;
            val = (sv ^ val) & 0xFFFFFFFFu;
            if (val == kEndSign) {
                eos = true;
                return std::nullopt;
            }
            sv = val;
            return sv;
        }
        if (fl == 1) {
            return sv;
        }
        int cb = 32 - sl - st;
        int mid = in.read_int(cb);
        std::uint32_t val = ((static_cast<std::uint32_t>(mid) << st) & 0xFFFFFFFFu) ^ sv;
        val &= 0xFFFFFFFFu;
        if (val == kEndSign) {
            eos = true;
            return std::nullopt;
        }
        sv = val;
        return sv;
    }
};

template <typename XorR>
static float elf_next_decoded(VignaIn &inn, XorR &&xor_read, int &last_beta) {
    int b0 = inn.read_int(1);
    if (b0 == 0) {
        auto vp_opt = xor_read();
        if (!vp_opt) {
            throw std::runtime_error("elf_float: xor early eos");
        }
        float v_prime = bits_to_float(*vp_opt);
        int sp = get_sp(static_cast<double>(std::fabs(static_cast<double>(v_prime))));
        if (last_beta == 0) {
            float vv = get10i_n(-sp - 1);
            if (v_prime < 0) {
                vv = -vv;
            }
            return vv;
        }
        int alpha = last_beta - sp - 1;
        return round_up_f(v_prime, alpha);
    }
    int b1 = inn.read_int(1);
    if (b1 == 0) {
        auto vb = xor_read();
        if (!vb) {
            throw std::runtime_error("elf_float: xor early eos2");
        }
        return bits_to_float(*vb);
    }
    last_beta = inn.read_int(3);
    auto vp_opt = xor_read();
    if (!vp_opt) {
        throw std::runtime_error("elf_float: xor early eos3");
    }
    float v_prime = bits_to_float(*vp_opt);
    int sp = get_sp(static_cast<double>(std::fabs(static_cast<double>(v_prime))));
    if (last_beta == 0) {
        float vv = get10i_n(-sp - 1);
        if (v_prime < 0) {
            vv = -vv;
        }
        return vv;
    }
    int alpha = last_beta - sp - 1;
    return round_up_f(v_prime, alpha);
}

bool decode_on_chimp_chunk(std::span<const std::uint8_t> bytes, std::vector<float> &out, int expected_n) {
    VignaIn inn(bytes);
    ChimpStreamDec ch(inn);
    int last_beta = kMaxBetaStar;
    try {
        for (int i = 0; i < expected_n; i++) {
            float v = elf_next_decoded(
                inn, [&ch]() { return ch.read_pair(); }, last_beta);
            out.push_back(v);
        }
    } catch (...) {
        return false;
    }
    return static_cast<int>(out.size()) == expected_n;
}

bool decode_elf_plus_chunk(std::span<const std::uint8_t> bytes, std::vector<float> &out, int expected_n) {
    VignaIn inn(bytes);
    XorStreamDec xr(inn);
    int last_beta = kMaxBetaStar;
    try {
        for (int i = 0; i < expected_n; i++) {
            float v =
                elf_next_decoded(inn, [&xr]() { return xr.read_value(); }, last_beta);
            out.push_back(v);
        }
    } catch (...) {
        return false;
    }
    return static_cast<int>(out.size()) == expected_n;
}

static void bench_elf_on_chimp(const std::vector<std::vector<float>> &columns, std::vector<long long> &result) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col_raw : columns) {
        if (col_raw.empty()) {
            continue;
        }
        try {
            long long col_o = 0, col_c = 0, col_ct = 0, col_dt = 0;
            const int n = static_cast<int>(col_raw.size());
            for (int st = 0; st < n; st += kElfChunk) {
                int en = std::min(st + kElfChunk, n);
                std::span<const float> chunk(col_raw.data() + st, static_cast<std::size_t>(en - st));
                auto t0 = std::chrono::high_resolution_clock::now();
                auto cbytes = encode_on_chimp_chunk(chunk);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::vector<float> got;
                auto t2 = std::chrono::high_resolution_clock::now();
                if (!decode_on_chimp_chunk(std::span<const std::uint8_t>(cbytes.data(), cbytes.size()), got,
                                           en - st)) {
                    throw std::runtime_error("dec");
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < en - st; i++) {
                    std::uint32_t a = 0, b = 0;
                    std::memcpy(&a, &got[static_cast<std::size_t>(i)], 4);
                    std::memcpy(&b, &col_raw[static_cast<std::size_t>(st + i)], 4);
                    if (a != b) {
                        throw std::runtime_error("rt");
                    }
                }
                col_o += static_cast<long long>(benchmark_wire::column_original_size(static_cast<std::size_t>(en - st), true));
                col_c += static_cast<long long>(cbytes.size());
                col_ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                col_dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }
            orig += col_o;
            comp += col_c;
            ct += col_ct;
            dt += col_dt;
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

static void bench_elf_plus(const std::vector<std::vector<float>> &columns, std::vector<long long> &result) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col_raw : columns) {
        if (col_raw.empty()) {
            continue;
        }
        try {
            long long col_o = 0, col_c = 0, col_ct = 0, col_dt = 0;
            const int n = static_cast<int>(col_raw.size());
            for (int st = 0; st < n; st += kElfChunk) {
                int en = std::min(st + kElfChunk, n);
                std::span<const float> chunk(col_raw.data() + st, static_cast<std::size_t>(en - st));
                auto t0 = std::chrono::high_resolution_clock::now();
                auto cbytes = encode_elf_plus_chunk(chunk);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::vector<float> got;
                auto t2 = std::chrono::high_resolution_clock::now();
                if (!decode_elf_plus_chunk(std::span<const std::uint8_t>(cbytes.data(), cbytes.size()), got,
                                           en - st)) {
                    throw std::runtime_error("dec");
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < en - st; i++) {
                    std::uint32_t a = 0, b = 0;
                    std::memcpy(&a, &got[static_cast<std::size_t>(i)], 4);
                    std::memcpy(&b, &col_raw[static_cast<std::size_t>(st + i)], 4);
                    if (a != b) {
                        throw std::runtime_error("rt");
                    }
                }
                col_o += static_cast<long long>(benchmark_wire::column_original_size(static_cast<std::size_t>(en - st), true));
                col_c += static_cast<long long>(cbytes.size());
                col_ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                col_dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }
            orig += col_o;
            comp += col_c;
            ct += col_ct;
            dt += col_dt;
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace detail

std::vector<std::uint8_t> encodeElfOnChimpFloatColumn(std::span<const float> col) {
    std::vector<std::uint8_t> total;
    const int n = static_cast<int>(col.size());
    for (int st = 0; st < n; st += detail::kElfChunk) {
        int en = std::min(st + detail::kElfChunk, n);
        auto piece = detail::encode_on_chimp_chunk(
            col.subspan(static_cast<std::size_t>(st), static_cast<std::size_t>(en - st)));
        total.insert(total.end(), piece.begin(), piece.end());
    }
    return total;
}

std::vector<std::uint8_t> encodeElfPlusFloatColumn(std::span<const float> col) {
    std::vector<std::uint8_t> total;
    const int n = static_cast<int>(col.size());
    for (int st = 0; st < n; st += detail::kElfChunk) {
        int en = std::min(st + detail::kElfChunk, n);
        auto piece = detail::encode_elf_plus_chunk(
            col.subspan(static_cast<std::size_t>(st), static_cast<std::size_t>(en - st)));
        total.insert(total.end(), piece.begin(), piece.end());
    }
    return total;
}

bool decodeElfOnChimpFloatChunk(std::span<const std::uint8_t> bytes, std::vector<float> &out, int expected_n) {
    out.clear();
    return detail::decode_on_chimp_chunk(bytes, out, expected_n);
}

bool decodeElfPlusFloatChunk(std::span<const std::uint8_t> bytes, std::vector<float> &out, int expected_n) {
    out.clear();
    return detail::decode_elf_plus_chunk(bytes, out, expected_n);
}

void benchElfOnChimpFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                                         std::vector<long long> &result) {
    detail::bench_elf_on_chimp(columns, result);
}

void benchElfPlusFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                                      std::vector<long long> &result) {
    detail::bench_elf_plus(columns, result);
}

} // namespace elf_float

namespace exact_actions::elf {
namespace {

constexpr std::array<std::uint8_t, 4> kEraserMagic{'E', 'E', 'R', '1'};
constexpr std::array<std::uint8_t, 4> kWireMagic{'E', 'L', 'F', '1'};

enum class PrefixTag : std::uint8_t { reuse_beta = 0, literal = 1, new_beta = 2 };

struct EraseEvent {
    PrefixTag tag{};
    std::uint8_t beta{};
    std::uint32_t erased_bits{};
};

void put_u32_le(std::vector<std::uint8_t> &out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void put_u64_le(std::vector<std::uint8_t> &out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

std::uint32_t get_u32_le(std::span<const std::uint8_t> data, std::size_t &pos,
                         const char *label) {
    if (data.size() - pos < 4) {
        throw std::runtime_error(std::string("truncated ") + label);
    }
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        value |= static_cast<std::uint32_t>(data[pos++]) << shift;
    }
    return value;
}

std::uint64_t get_u64_le(std::span<const std::uint8_t> data, std::size_t &pos,
                         const char *label) {
    if (data.size() - pos < 8) {
        throw std::runtime_error(std::string("truncated ") + label);
    }
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        value |= static_cast<std::uint64_t>(data[pos++]) << shift;
    }
    return value;
}

std::uint64_t int64_bits(std::int64_t value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::int64_t signed_bits(std::uint64_t value) {
    std::int64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

EraseEvent erase_one(float value, int &last_beta) {
    const std::uint32_t value_bits = elf_float::detail::float_to_bits(value);
    if (value == 0.0f || std::isinf(value)) {
        return {PrefixTag::literal, 0, value_bits};
    }
    if (std::isnan(value)) {
        return {PrefixTag::literal, 0, elf_float::detail::kNanBits};
    }
    int alpha = 0;
    int beta = 0;
    elf_float::detail::get_alpha_and_beta_star(value, last_beta, alpha, beta);
    const int exponent = static_cast<int>((value_bits >> 23) & 0xffU);
    const int g_alpha = elf_float::detail::get_f_alpha(alpha) + exponent - 127;
    const int erase_bits = 23 - g_alpha;
    const int shift = erase_bits & 31;
    const std::uint32_t mask = shift == 0 ? 0xffffffffU : 0xffffffffU << shift;
    const std::uint32_t delta = (~mask) & value_bits;
    if (delta != 0 && erase_bits > 3) {
        const std::uint32_t erased = mask & value_bits;
        if (beta == last_beta) {
            return {PrefixTag::reuse_beta, static_cast<std::uint8_t>(beta), erased};
        }
        last_beta = beta;
        return {PrefixTag::new_beta, static_cast<std::uint8_t>(beta), erased};
    }
    return {PrefixTag::literal, 0, value_bits};
}

float restore_one(const EraseEvent &event, int &last_beta) {
    if (event.tag == PrefixTag::literal) {
        return elf_float::detail::bits_to_float(event.erased_bits);
    }
    if (event.tag == PrefixTag::new_beta) {
        last_beta = event.beta;
    } else if (event.tag != PrefixTag::reuse_beta) {
        throw std::runtime_error("invalid Elf eraser prefix tag");
    }
    const float value_prime = elf_float::detail::bits_to_float(event.erased_bits);
    const int sp = elf_float::detail::get_sp(
        static_cast<double>(std::fabs(static_cast<double>(value_prime))));
    if (last_beta == 0) {
        float value = elf_float::detail::get10i_n(-sp - 1);
        return value_prime < 0 ? -value : value;
    }
    const int alpha = last_beta - sp - 1;
    return elf_float::detail::round_up_f(value_prime, alpha);
}

std::vector<std::uint8_t> serialize_events(
    std::span<const EraseEvent> events) {
    std::vector<std::uint8_t> out(kEraserMagic.begin(), kEraserMagic.end());
    put_u64_le(out, events.size());
    out.reserve(out.size() + events.size() * 6);
    for (const auto &event : events) {
        out.push_back(static_cast<std::uint8_t>(event.tag));
        out.push_back(event.beta);
        put_u32_le(out, event.erased_bits);
    }
    return out;
}

std::vector<EraseEvent> parse_events(std::span<const std::uint8_t> packet) {
    if (packet.size() < 12 ||
        !std::equal(kEraserMagic.begin(), kEraserMagic.end(), packet.begin())) {
        throw std::runtime_error("wrong Elf eraser packet type");
    }
    std::size_t pos = 4;
    const auto count = get_u64_le(packet, pos, "Elf eraser count");
    if (count > (packet.size() - pos) / 6 || count * 6 != packet.size() - pos) {
        throw std::runtime_error("invalid Elf eraser event count");
    }
    std::vector<EraseEvent> events;
    events.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto raw_tag = packet[pos++];
        const auto beta = packet[pos++];
        if (raw_tag > static_cast<std::uint8_t>(PrefixTag::new_beta) || beta > 7) {
            throw std::runtime_error("invalid Elf eraser event");
        }
        events.push_back({static_cast<PrefixTag>(raw_tag), beta,
                          get_u32_le(packet, pos, "Elf erased bits")});
    }
    return events;
}

void write_prefix(elf_float::detail::VignaOut &out, const EraseEvent &event,
                  int &last_beta) {
    if (event.tag == PrefixTag::reuse_beta) {
        if (event.beta != last_beta) {
            throw std::runtime_error("Elf eraser reused the wrong beta");
        }
        out.write_bit(0);
    } else if (event.tag == PrefixTag::literal) {
        out.write_int(2, 2);
    } else if (event.tag == PrefixTag::new_beta) {
        if (event.beta > 7) throw std::runtime_error("Elf beta exceeds three bits");
        out.write_int(static_cast<std::uint32_t>(event.beta | 0x18U), 5);
        last_beta = event.beta;
    } else {
        throw std::runtime_error("invalid Elf eraser prefix tag");
    }
}

template <typename Encoder>
std::vector<std::uint8_t> encode_chunk(std::span<const EraseEvent> events) {
    elf_float::detail::VignaOut out;
    Encoder encoder(out);
    int last_beta = elf_float::detail::kMaxBetaStar;
    for (const auto &event : events) {
        write_prefix(out, event, last_beta);
        encoder.add_raw_u32(event.erased_bits);
    }
    out.write_int(2, 2);
    if constexpr (std::is_same_v<Encoder, elf_float::detail::ChimpStreamEnc>) {
        encoder.close_chimp();
    } else {
        encoder.close_xor();
    }
    const auto bytes = static_cast<std::size_t>((out.written_bits + 7) / 8);
    if (bytes > out.buf.size()) throw std::runtime_error("Elf stage output overflow");
    return {out.buf.begin(), out.buf.begin() + static_cast<std::ptrdiff_t>(bytes)};
}

template <typename Decoder, typename Read>
std::vector<EraseEvent> decode_chunk(std::span<const std::uint8_t> body,
                                     std::size_t count, Read read_erased) {
    elf_float::detail::VignaIn input(body);
    Decoder decoder(input);
    std::vector<EraseEvent> events;
    events.reserve(count);
    int last_beta = elf_float::detail::kMaxBetaStar;
    for (std::size_t i = 0; i < count; ++i) {
        PrefixTag tag;
        std::uint8_t beta = 0;
        if (input.read_int(1) == 0) {
            tag = PrefixTag::reuse_beta;
            if (last_beta == elf_float::detail::kMaxBetaStar) {
                throw std::runtime_error("Elf stream reused beta before initialization");
            }
            beta = static_cast<std::uint8_t>(last_beta);
        } else if (input.read_int(1) == 0) {
            tag = PrefixTag::literal;
        } else {
            tag = PrefixTag::new_beta;
            beta = static_cast<std::uint8_t>(input.read_int(3));
            last_beta = beta;
        }
        const auto erased = read_erased(decoder);
        if (!erased.has_value()) throw std::runtime_error("Elf XOR ended early");
        events.push_back({tag, beta, *erased});
    }
    // Consume and validate the eraser close marker and XOR sentinel.
    if (input.read_int(2) != 2 || read_erased(decoder).has_value()) {
        throw std::runtime_error("Elf stream terminator mismatch");
    }
    return events;
}

template <typename Encoder>
std::vector<std::uint8_t> xor_forward_impl(
    std::span<const std::uint8_t> eraser_packet) {
    const auto events = parse_events(eraser_packet);
    if (events.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Elf column exceeds uint32 count");
    }
    std::vector<std::uint8_t> out(kWireMagic.begin(), kWireMagic.end());
    put_u32_le(out, static_cast<std::uint32_t>(events.size()));
    for (std::size_t start = 0; start < events.size(); start += elf_float::detail::kElfChunk) {
        const auto count = std::min<std::size_t>(elf_float::detail::kElfChunk,
                                                 events.size() - start);
        const auto body = encode_chunk<Encoder>(
            std::span<const EraseEvent>(events.data() + start, count));
        put_u32_le(out, static_cast<std::uint32_t>(count));
        put_u32_le(out, static_cast<std::uint32_t>(body.size()));
        out.insert(out.end(), body.begin(), body.end());
    }
    return out;
}

template <typename Decoder, typename Read>
std::vector<std::uint8_t> xor_inverse_impl(std::span<const std::uint8_t> wire,
                                           Read read_erased) {
    if (wire.size() < 8 ||
        !std::equal(kWireMagic.begin(), kWireMagic.end(), wire.begin())) {
        throw std::runtime_error("wrong Elf public wire type");
    }
    std::size_t pos = 4;
    const auto total_count = get_u32_le(wire, pos, "Elf public count");
    std::vector<EraseEvent> events;
    events.reserve(total_count);
    while (events.size() < total_count) {
        const auto count = get_u32_le(wire, pos, "Elf chunk count");
        const auto size = get_u32_le(wire, pos, "Elf chunk size");
        if (count == 0 || count > elf_float::detail::kElfChunk ||
            count > total_count - events.size() || wire.size() - pos < size) {
            throw std::runtime_error("invalid Elf chunk header");
        }
        auto chunk = decode_chunk<Decoder>(wire.subspan(pos, size), count, read_erased);
        events.insert(events.end(), chunk.begin(), chunk.end());
        pos += size;
    }
    if (pos != wire.size()) throw std::runtime_error("Elf public wire has trailing bytes");
    return serialize_events(events);
}

}  // namespace

std::vector<std::uint8_t> eraser_forward_double_bits(
    std::span<const std::int64_t> values) {
    std::vector<EraseEvent> events;
    events.reserve(values.size());
    for (std::size_t start = 0; start < values.size(); start += elf_float::detail::kElfChunk) {
        int last_beta = elf_float::detail::kMaxBetaStar;
        const auto end = std::min<std::size_t>(values.size(), start + elf_float::detail::kElfChunk);
        for (std::size_t i = start; i < end; ++i) {
            const auto bits = int64_bits(values[i]);
            double original = 0.0;
            std::memcpy(&original, &bits, sizeof(original));
            if (!std::isfinite(original)) {
                throw std::runtime_error(
                    "Elf32 exact action requires finite float64 values");
            }
            const float narrowed = static_cast<float>(original);
            const double promoted = static_cast<double>(narrowed);
            std::uint64_t promoted_bits = 0;
            std::memcpy(&promoted_bits, &promoted, sizeof(promoted_bits));
            if (promoted_bits != bits) {
                throw std::runtime_error(
                    "Elf32 exact action requires a bitwise lossless float32 cast");
            }
            events.push_back(erase_one(narrowed, last_beta));
        }
    }
    return serialize_events(events);
}

std::vector<std::int64_t> eraser_inverse_double_bits(
    std::span<const std::uint8_t> packet) {
    const auto events = parse_events(packet);
    std::vector<std::int64_t> values;
    values.reserve(events.size());
    for (std::size_t start = 0; start < events.size(); start += elf_float::detail::kElfChunk) {
        int last_beta = elf_float::detail::kMaxBetaStar;
        const auto end = std::min<std::size_t>(events.size(), start + elf_float::detail::kElfChunk);
        for (std::size_t i = start; i < end; ++i) {
            const double value = static_cast<double>(restore_one(events[i], last_beta));
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            values.push_back(signed_bits(bits));
        }
    }
    return values;
}

std::vector<std::uint8_t> elf_xor_forward(
    std::span<const std::uint8_t> eraser_packet) {
    return xor_forward_impl<elf_float::detail::ChimpStreamEnc>(eraser_packet);
}

std::vector<std::uint8_t> elf_xor_inverse(
    std::span<const std::uint8_t> wire) {
    return xor_inverse_impl<elf_float::detail::ChimpStreamDec>(
        wire, [](auto &decoder) { return decoder.read_pair(); });
}

std::vector<std::uint8_t> elf_plus_xor_forward(
    std::span<const std::uint8_t> eraser_packet) {
    return xor_forward_impl<elf_float::detail::XorStreamEnc>(eraser_packet);
}

std::vector<std::uint8_t> elf_plus_xor_inverse(
    std::span<const std::uint8_t> wire) {
    return xor_inverse_impl<elf_float::detail::XorStreamDec>(
        wire, [](auto &decoder) { return decoder.read_value(); });
}

std::vector<std::uint8_t> compose_elf(
    std::span<const std::int64_t> values) {
    return elf_xor_forward(eraser_forward_double_bits(values));
}

std::vector<std::int64_t> decompose_elf(
    std::span<const std::uint8_t> wire) {
    return eraser_inverse_double_bits(elf_xor_inverse(wire));
}

std::vector<std::uint8_t> compose_elf_plus(
    std::span<const std::int64_t> values) {
    return elf_plus_xor_forward(eraser_forward_double_bits(values));
}

std::vector<std::int64_t> decompose_elf_plus(
    std::span<const std::uint8_t> wire) {
    return eraser_inverse_double_bits(elf_plus_xor_inverse(wire));
}

}  // namespace exact_actions::elf
