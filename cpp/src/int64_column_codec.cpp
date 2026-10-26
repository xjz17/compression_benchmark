#include "int64_column_codec.hpp"

#include "java_fastpfor_codec.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
size_t svb_encode(uint8_t *out, const uint32_t *in, uint32_t count, int delta, int type);
size_t svb_decode(uint32_t *out, uint8_t *in, int delta, int type);
}

namespace int64_column_codec {
namespace {

struct SelRow {
    int maxNum;
    int log;
    int sel;
};

static const SelRow kRows[] = {
    {240, 0, 0}, {120, 0, 1}, {60, 1, 2},  {30, 2, 3},   {20, 3, 4},   {15, 4, 5},   {12, 5, 6},   {10, 6, 7},
    {8, 7, 8},   {7, 8, 9},   {6, 10, 10}, {5, 12, 11},  {4, 15, 12},  {3, 20, 13},  {2, 30, 14},  {1, 60, 15},
};

constexpr int kLogDesc = 4;
constexpr std::uint64_t kMask64 = 0xffffffffffffffffULL;

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
            return 0ULL;
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
        const std::uint64_t lw = words[static_cast<std::size_t>(i)];
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

static std::vector<std::uint8_t> svb_encode_u32(const std::uint32_t *u32, std::size_t n,
                                                bool delta = true) {
    if (n == 0 || n > 0xFFFFFFFFu) {
        return {};
    }
    std::vector<std::uint8_t> buf(4 + ((n + 3) / 4) + n * 5 + 64);
    const std::size_t enc = svb_encode(buf.data(), u32, static_cast<uint32_t>(n), delta ? 1 : 0, 1);
    if (enc == 0 || enc > buf.size()) {
        throw std::runtime_error("streamvbyte: encode failed");
    }
    buf.resize(enc);
    return buf;
}

static void svb_decode_u32(const std::uint8_t *buf, std::size_t len, std::uint32_t *out,
                           std::size_t n, bool delta = true) {
    std::vector<std::uint8_t> copy(buf, buf + len);
    const std::size_t consumed = svb_decode(out, copy.data(), delta ? 1 : 0, 1);
    if (consumed != len) {
        throw std::runtime_error("streamvbyte: decode length mismatch");
    }
    (void)n;
}

static void split_lo_hi(const std::int64_t *data, std::size_t n, std::vector<std::int32_t> &lo,
                        std::vector<std::int32_t> &hi) {
    lo.resize(n);
    hi.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        lo[i] = static_cast<std::int32_t>(data[i]);
        hi[i] = static_cast<std::int32_t>(static_cast<std::uint64_t>(data[i]) >> 32);
    }
}

} // namespace

std::int32_t lo32(std::int64_t v) { return static_cast<std::int32_t>(v); }

std::int32_t hi32(std::int64_t v) {
    return static_cast<std::int32_t>(static_cast<std::uint64_t>(v) >> 32);
}

std::int64_t reconstruct(std::int32_t lo, std::int32_t hi) {
    return (static_cast<std::int64_t>(hi) << 32) |
           static_cast<std::int64_t>(static_cast<std::uint32_t>(lo));
}

std::vector<std::uint32_t> fast_pfor_compress_limb(const std::int32_t *data, std::size_t n) {
    return java_fastpfor::compressIntBits(data, n);
}

std::vector<std::int32_t> fast_pfor_uncompress_limb(const std::uint32_t *words, std::size_t word_count,
                                                    std::size_t n) {
    return java_fastpfor::uncompressIntBits(std::vector<std::uint32_t>(words, words + word_count), n);
}

std::vector<std::uint8_t> fast_pfor_encode_int64(const std::int64_t *data, std::size_t n) {
    std::vector<std::int32_t> lo;
    std::vector<std::int32_t> hi;
    split_lo_hi(data, n, lo, hi);
    const auto c_lo = fast_pfor_compress_limb(lo.data(), n);
    const auto c_hi = fast_pfor_compress_limb(hi.data(), n);
    std::vector<std::uint8_t> out(4 + (c_lo.size() + c_hi.size()) * 4);
    const std::uint32_t lo_words = static_cast<std::uint32_t>(c_lo.size());
    std::memcpy(out.data(), &lo_words, 4);
    std::size_t pos = 4;
    for (std::uint32_t w : c_lo) {
        std::memcpy(out.data() + pos, &w, 4);
        pos += 4;
    }
    for (std::uint32_t w : c_hi) {
        std::memcpy(out.data() + pos, &w, 4);
        pos += 4;
    }
    return out;
}

std::vector<std::int64_t> fast_pfor_decode_int64(const std::uint8_t *wire, std::size_t wire_len, std::size_t n) {
    if (wire_len < 4) {
        throw std::runtime_error("FastPFOR int64: truncated header");
    }
    std::uint32_t lo_words = 0;
    std::memcpy(&lo_words, wire, 4);
    if (lo_words > (wire_len - 4) / 4) {
        throw std::runtime_error("FastPFOR int64: truncated body");
    }
    const std::uint32_t *c_lo = reinterpret_cast<const std::uint32_t *>(wire + 4);
    const std::uint32_t *c_hi = c_lo + lo_words;
    const std::size_t hi_words = (wire_len - 4) / 4 - lo_words;
    const auto lo = fast_pfor_uncompress_limb(c_lo, lo_words, n);
    const auto hi = fast_pfor_uncompress_limb(c_hi, hi_words, n);
    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = reconstruct(lo[static_cast<std::size_t>(i)], hi[static_cast<std::size_t>(i)]);
    }
    return out;
}

std::vector<std::uint8_t> stream_vbyte_encode_int64(const std::int64_t *data, std::size_t n) {
    std::vector<std::int32_t> lo;
    std::vector<std::int32_t> hi;
    split_lo_hi(data, n, lo, hi);
    std::vector<std::uint32_t> lo_u(n);
    std::vector<std::uint32_t> hi_u(n);
    for (std::size_t i = 0; i < n; ++i) {
        lo_u[i] = static_cast<std::uint32_t>(lo[i]);
        hi_u[i] = static_cast<std::uint32_t>(hi[i]);
    }
    const auto e_lo = svb_encode_u32(lo_u.data(), n);
    const auto e_hi = svb_encode_u32(hi_u.data(), n);
    std::vector<std::uint8_t> out(4 + e_lo.size() + e_hi.size());
    const std::uint32_t lo_len = static_cast<std::uint32_t>(e_lo.size());
    std::memcpy(out.data(), &lo_len, 4);
    std::memcpy(out.data() + 4, e_lo.data(), e_lo.size());
    std::memcpy(out.data() + 4 + e_lo.size(), e_hi.data(), e_hi.size());
    return out;
}

std::vector<std::int64_t> stream_vbyte_decode_int64(const std::uint8_t *wire, std::size_t wire_len,
                                                    std::size_t n) {
    if (wire_len < 4) {
        throw std::runtime_error("streamvbyte int64: truncated header");
    }
    std::uint32_t lo_len = 0;
    std::memcpy(&lo_len, wire, 4);
    if (lo_len > wire_len - 4) {
        throw std::runtime_error("streamvbyte int64: truncated lo payload");
    }
    const std::uint8_t *e_lo = wire + 4;
    const std::uint8_t *e_hi = wire + 4 + lo_len;
    const std::size_t hi_len = wire_len - 4 - lo_len;
    std::vector<std::uint32_t> lo_u(n);
    std::vector<std::uint32_t> hi_u(n);
    svb_decode_u32(e_lo, lo_len, lo_u.data(), n);
    svb_decode_u32(e_hi, hi_len, hi_u.data(), n);
    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = reconstruct(static_cast<std::int32_t>(lo_u[i]), static_cast<std::int32_t>(hi_u[i]));
    }
    return out;
}

std::vector<std::uint8_t> stream_vbyte_encode_int64_ablation(
    const std::int64_t *data, std::size_t n, std::string_view omitted_operator) {
    if (omitted_operator.empty()) {
        return stream_vbyte_encode_int64(data, n);
    }
    if (omitted_operator == "integer coding") {
        std::vector<std::int32_t> lo;
        std::vector<std::int32_t> hi;
        split_lo_hi(data, n, lo, hi);
        std::vector<std::uint8_t> out(4 + n * sizeof(std::int64_t));
        const std::uint32_t lo_len = static_cast<std::uint32_t>(n * sizeof(std::int32_t));
        std::memcpy(out.data(), &lo_len, sizeof(lo_len));
        if (n != 0) {
            std::memcpy(out.data() + 4, lo.data(), n * sizeof(std::int32_t));
            std::memcpy(out.data() + 4 + lo_len, hi.data(), n * sizeof(std::int32_t));
        }
        return out;
    }
    if (omitted_operator != "delta") {
        throw std::invalid_argument("unsupported Stream VByte ablation");
    }
    std::vector<std::int32_t> lo;
    std::vector<std::int32_t> hi;
    split_lo_hi(data, n, lo, hi);
    std::vector<std::uint32_t> lo_u(n);
    std::vector<std::uint32_t> hi_u(n);
    for (std::size_t i = 0; i < n; ++i) {
        lo_u[i] = static_cast<std::uint32_t>(lo[i]);
        hi_u[i] = static_cast<std::uint32_t>(hi[i]);
    }
    const auto e_lo = svb_encode_u32(lo_u.data(), n, false);
    const auto e_hi = svb_encode_u32(hi_u.data(), n, false);
    std::vector<std::uint8_t> out(4 + e_lo.size() + e_hi.size());
    const std::uint32_t lo_len = static_cast<std::uint32_t>(e_lo.size());
    std::memcpy(out.data(), &lo_len, sizeof(lo_len));
    std::memcpy(out.data() + 4, e_lo.data(), e_lo.size());
    std::memcpy(out.data() + 4 + e_lo.size(), e_hi.data(), e_hi.size());
    return out;
}

std::vector<std::int64_t> stream_vbyte_decode_int64_ablation(
    const std::uint8_t *wire, std::size_t wire_len, std::size_t n,
    std::string_view omitted_operator) {
    if (omitted_operator.empty()) {
        return stream_vbyte_decode_int64(wire, wire_len, n);
    }
    if (wire_len < 4) {
        throw std::runtime_error("streamvbyte ablation: truncated header");
    }
    std::uint32_t lo_len = 0;
    std::memcpy(&lo_len, wire, sizeof(lo_len));
    if (lo_len > wire_len - 4) {
        throw std::runtime_error("streamvbyte ablation: truncated low payload");
    }
    std::vector<std::uint32_t> lo_u(n);
    std::vector<std::uint32_t> hi_u(n);
    if (omitted_operator == "integer coding") {
        if (lo_len != n * sizeof(std::int32_t) || wire_len != 4 + 2 * lo_len) {
            throw std::runtime_error("streamvbyte fixed-width ablation size mismatch");
        }
        if (n != 0) {
            std::memcpy(lo_u.data(), wire + 4, lo_len);
            std::memcpy(hi_u.data(), wire + 4 + lo_len, lo_len);
        }
    } else if (omitted_operator == "delta") {
        svb_decode_u32(wire + 4, lo_len, lo_u.data(), n, false);
        svb_decode_u32(wire + 4 + lo_len, wire_len - 4 - lo_len, hi_u.data(), n, false);
    } else {
        throw std::invalid_argument("unsupported Stream VByte ablation");
    }
    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = reconstruct(static_cast<std::int32_t>(lo_u[i]),
                             static_cast<std::int32_t>(hi_u[i]));
    }
    return out;
}

std::vector<std::uint8_t> simple8b_encode_int64(const std::int64_t *data, std::size_t n) {
    std::vector<std::int32_t> lo;
    std::vector<std::int32_t> hi;
    split_lo_hi(data, n, lo, hi);
    std::vector<std::uint32_t> u32(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        u32[i] = static_cast<std::uint32_t>(lo[i]);
        u32[n + i] = static_cast<std::uint32_t>(hi[i]);
    }
    return encode_column_bytes(u32.data(), 2 * n);
}

std::vector<std::int64_t> simple8b_decode_int64(const std::uint8_t *body, std::size_t body_len, std::size_t n) {
    std::vector<std::uint32_t> u32(2 * n);
    decode_column_bytes(body, body_len, u32.data(), 2 * n);
    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = reconstruct(static_cast<std::int32_t>(u32[i]), static_cast<std::int32_t>(u32[n + i]));
    }
    return out;
}

bool doubles_match_scaled_column(const std::vector<double> &col, const std::int64_t *data, std::size_t n,
                                 int col_scale) {
    const double pow10 = std::pow(10.0, col_scale);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = col[i];
        if (v == 0.0) {
            std::uint64_t vb = 0;
            std::memcpy(&vb, &v, 8);
            if (static_cast<std::uint64_t>(data[i]) != vb) {
                return false;
            }
            continue;
        }
        const double r = static_cast<double>(data[i]) / pow10;
        std::uint64_t rb = 0;
        std::uint64_t vb = 0;
        std::memcpy(&rb, &r, 8);
        std::memcpy(&vb, &v, 8);
        if (rb != vb) {
            return false;
        }
    }
    return true;
}

} // namespace int64_column_codec
