#include "chimp32_paper_codec.hpp"
#include "benchmark_wire.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <span>

namespace chimp32_paper {
namespace {

constexpr int kThreshold = 5;
constexpr std::uint32_t kNanBits = 0x7fc00000u;

// Chimp32.java leadingRepresentation — indexed by quantized `leadingZeros` (0,8,12,…), not NLZ slot.
alignas(64) static constexpr std::uint8_t kLeadReprComp[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

alignas(64) static constexpr std::uint8_t kLeadRound[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};

alignas(64) static constexpr std::uint8_t kLeadReprDec[8] = {0, 8, 12, 16, 18, 20, 22, 24};

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

/** Vigna-style big-endian bit output over byte buffer (Chimp `OutputBitStream(byte[])`). */
struct OutBits {
    std::vector<std::uint8_t> &buf;
    long long written_bits{0};
    int current{0};
    int free_bits{8};
    std::size_t pos{0};

    explicit OutBits(std::vector<std::uint8_t> &b) : buf(b) {
        if (buf.empty()) {
            throw std::runtime_error("chimp32: output buffer empty");
        }
    }

    void write_byte(int b) {
        if (pos >= buf.size()) {
            throw std::runtime_error("chimp32: buffer overflow");
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

    void align() {
        if (free_bits != 8) {
            write_in_current(0, free_bits);
        }
    }
};

struct InBits {
    const std::uint8_t *buffer{nullptr};
    std::size_t len{0};
    std::size_t pos{0};
    int fill{0};
    std::uint32_t current{0};

    explicit InBits(std::span<const std::uint8_t> bytes)
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
                throw std::runtime_error("chimp32: truncated bit stream");
            }
            current = static_cast<std::uint32_t>(b);
            fill = 8;
        }
        fill -= l;
        return static_cast<int>((current >> fill) & static_cast<std::uint32_t>(low_bits_mask(l)));
    }

    int read_bit() { return read_from_current(1); }

    /** Fixed-width integer, MSB first — matches {@code InputBitStream.readInt} bit order. */
    int read_int(int l) {
        if (l < 0 || l > 32) {
            throw std::runtime_error("chimp32: bad read_int width");
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
};

struct Compressor {
    std::vector<std::uint8_t> buf;
    OutBits out;
    int stored_lz{2147483647};
    std::uint32_t stored_val{0};
    bool first{true};
    /** Mirrors `Chimp32.size` (excludes trailing `writeBit(false)` + flush padding in `close()`). */
    int java_size{0};

    Compressor() : buf(16 * 1024 * 1024, 0), out(buf) {}

    void write_first(std::uint32_t value) {
        first = false;
        stored_val = value;
        out.write_int(stored_val, 32);
        java_size += 32;
    }

    void compress_value(std::uint32_t value) {
        std::uint32_t xorv = stored_val ^ value;
        if (xorv == 0) {
            out.write_bit(0);
            out.write_bit(0);
            java_size += 2;
            stored_lz = 33;
        } else {
            int lz_idx = nlz_u32(xorv);
            int leading_zeros = kLeadRound[lz_idx];
            int trailing_zeros = ntz_u32(xorv);
            if (trailing_zeros > kThreshold) {
                int significant_bits = 32 - leading_zeros - trailing_zeros;
                out.write_bit(0);
                out.write_bit(1);
                out.write_int(kLeadReprComp[static_cast<std::size_t>(leading_zeros)], 3);
                out.write_int(static_cast<std::uint32_t>(significant_bits), 5);
                out.write_int(xorv >> trailing_zeros, significant_bits);
                java_size += 10 + significant_bits;
                stored_lz = 33;
            } else if (leading_zeros == stored_lz) {
                out.write_bit(1);
                out.write_bit(0);
                int significant_bits = 32 - leading_zeros;
                out.write_int(xorv, significant_bits);
                java_size += 2 + significant_bits;
            } else {
                stored_lz = leading_zeros;
                int significant_bits = 32 - leading_zeros;
                out.write_int(24 + kLeadReprComp[static_cast<std::size_t>(leading_zeros)], 5);
                out.write_int(xorv, significant_bits);
                java_size += 5 + significant_bits;
            }
        }
        stored_val = value;
    }

    void add_raw_bits(std::uint32_t bits) {
        if (first) {
            write_first(bits);
        } else {
            compress_value(bits);
        }
    }

    std::vector<std::uint8_t> finish_java_truncation() {
        add_raw_bits(kNanBits);
        out.write_bit(0);
        out.align();
        int comp_bytes = (java_size + 7) / 8;
        return std::vector<std::uint8_t>(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(comp_bytes));
    }
};

struct Decompressor {
    InBits in;
    int stored_lz{2147483647};
    int stored_tz{0};
    std::uint32_t stored_val{0};
    bool first{true};
    bool eos{false};

    explicit Decompressor(std::span<const std::uint8_t> bytes) : in(bytes) {}

    void next() {
        if (first) {
            first = false;
            stored_val = static_cast<std::uint32_t>(in.read_int(32));
            if (stored_val == kNanBits) {
                eos = true;
            }
            return;
        }
        next_value();
    }

    void next_value() {
        int bit = in.read_bit();
        if (bit == 1) {
            if (in.read_bit() == 1) {
                int idx = in.read_int(3);
                stored_lz = kLeadReprDec[idx];
            }
            int significant_bits = 32 - stored_lz;
            if (significant_bits == 0) {
                significant_bits = 32;
            }
            std::uint32_t value =
                static_cast<std::uint32_t>(in.read_int(32 - stored_lz));
            value = stored_val ^ value;
            if (value == kNanBits) {
                eos = true;
                return;
            }
            stored_val = value;
        } else if (in.read_bit() == 1) {
            int idx = in.read_int(3);
            stored_lz = kLeadReprDec[idx];
            int significant_bits = in.read_int(5);
            if (significant_bits == 0) {
                significant_bits = 32;
            }
            stored_tz = 32 - significant_bits - stored_lz;
            std::uint32_t value = static_cast<std::uint32_t>(
                in.read_int(32 - stored_lz - stored_tz));
            value <<= stored_tz;
            value = stored_val ^ value;
            if (value == kNanBits) {
                eos = true;
                return;
            }
            stored_val = value;
        }
    }

    /** Returns next raw int bits or nullopt at EOS marker. */
    bool read_pair(std::uint32_t *out_bits) {
        if (eos) {
            return false;
        }
        next();
        if (eos) {
            return false;
        }
        *out_bits = stored_val;
        return true;
    }
};

std::vector<std::uint8_t> encode_raw_sequence(const std::uint32_t *bits, std::size_t n) {
    if (n == 0) {
        return {};
    }
    Compressor c;
    for (std::size_t i = 0; i < n; i++) {
        c.add_raw_bits(bits[i]);
    }
    return c.finish_java_truncation();
}

void decode_to_raw(std::span<const std::uint8_t> compressed, std::uint32_t *out_bits, std::size_t n) {
    Decompressor dec(compressed);
    for (std::size_t i = 0; i < n; i++) {
        std::uint32_t v = 0;
        if (!dec.read_pair(&v)) {
            throw std::runtime_error("chimp32: early EOS");
        }
        out_bits[i] = v;
    }
    std::uint32_t junk = 0;
    if (dec.read_pair(&junk)) {
        throw std::runtime_error("chimp32: expected stream end after n values");
    }
}

} // namespace

std::vector<std::uint8_t> encodeFloatBitsColumn(const float *values, std::size_t n) {
    if (n == 0) {
        return {};
    }
    std::vector<std::uint32_t> bits(n);
    for (std::size_t i = 0; i < n; i++) {
        std::uint32_t b;
        std::memcpy(&b, &values[i], 4);
        bits[i] = b;
    }
    return encode_raw_sequence(bits.data(), n);
}

std::vector<std::uint8_t> encodeIntColumnAsFloatBits(const std::int32_t *ints, std::size_t n) {
    if (n == 0) {
        return {};
    }
    std::vector<std::uint32_t> bits(n);
    for (std::size_t i = 0; i < n; i++) {
        float f;
        std::uint32_t raw = static_cast<std::uint32_t>(ints[i]);
        std::memcpy(&f, &raw, 4);
        std::memcpy(&bits[i], &f, 4);
    }
    return encode_raw_sequence(bits.data(), n);
}

void decodeFloatBitsColumn(std::span<const std::uint8_t> compressed, float *out, std::size_t n) {
    std::vector<std::uint32_t> raw(n);
    decode_to_raw(compressed, raw.data(), n);
    for (std::size_t i = 0; i < n; i++) {
        std::memcpy(&out[i], &raw[i], 4);
    }
}

void decodeIntColumnAsFloatBits(std::span<const std::uint8_t> compressed, std::int32_t *out,
                                  std::size_t n) {
    std::vector<std::uint32_t> raw(n);
    decode_to_raw(compressed, raw.data(), n);
    for (std::size_t i = 0; i < n; i++) {
        float f;
        std::memcpy(&f, &raw[i], 4);
        std::int32_t ir;
        std::memcpy(&ir, &f, 4);
        out[i] = ir;
    }
}

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns, std::vector<long long> &result) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        try {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto enc = encodeFloatBitsColumn(col.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<float> rec(n);
            auto t2 = std::chrono::high_resolution_clock::now();
            decodeFloatBitsColumn(std::span<const std::uint8_t>(enc.data(), enc.size()), rec.data(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            bool ok = true;
            for (std::size_t i = 0; i < n; i++) {
                std::uint32_t a, b;
                std::memcpy(&a, &col[i], 4);
                std::memcpy(&b, &rec[i], 4);
                if (a != b) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(enc.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchIntColumnsLikeJava(const std::vector<std::vector<int>> &columns, std::vector<long long> &result) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : columns) {
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        try {
            std::vector<std::int32_t> icol(n);
            for (std::size_t i = 0; i < n; i++) {
                icol[i] = static_cast<std::int32_t>(col[i]);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            auto enc = encodeIntColumnAsFloatBits(icol.data(), n);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::int32_t> rec(n);
            auto t2 = std::chrono::high_resolution_clock::now();
            decodeIntColumnAsFloatBits(std::span<const std::uint8_t>(enc.data(), enc.size()), rec.data(), n);
            auto t3 = std::chrono::high_resolution_clock::now();
            bool ok = true;
            for (std::size_t i = 0; i < n; i++) {
                if (rec[i] != icol[i]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(enc.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } catch (...) {
            continue;
        }
    }
    result.assign({orig, comp, ct, dt});
}

} // namespace chimp32_paper
