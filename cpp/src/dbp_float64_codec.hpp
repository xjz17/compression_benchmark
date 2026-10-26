#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Complete float64 framing for the project's delta/ZigZag/ULEB128 DBP body.
// See analysis_outputs/dbp_float_e2e_20261009/WIRE_CONTRACT.md.
namespace dbp_float64 {
inline constexpr std::size_t kMaxBlock = 10000;
inline constexpr double kPowers[] = {1., 10., 100., 1000., 10000., 100000.,
    1000000., 10000000., 100000000., 1000000000., 10000000000., 100000000000.,
    1000000000000., 10000000000000., 100000000000000., 1000000000000000.,
    10000000000000000., 100000000000000000., 1000000000000000000.};

inline std::uint64_t bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, 8);
    return u;
}
inline double fromBits(std::uint64_t u) {
    double x;
    std::memcpy(&x, &u, 8);
    return x;
}
inline float fromBits32(std::uint32_t u) {
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}
inline std::uint32_t bits32(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
inline std::int64_t signedWord(std::uint64_t u) {
    std::int64_t x;
    std::memcpy(&x, &u, 8);
    return x;
}
inline void appendWord(std::vector<std::uint8_t>& out, std::uint64_t u) {
    for (int j = 0; j < 8; ++j) out.push_back(static_cast<std::uint8_t>(u >> (8*j)));
}
inline std::uint64_t readWord(const std::uint8_t* p) {
    std::uint64_t u = 0;
    for (int j = 0; j < 8; ++j) u |= std::uint64_t(p[j]) << (8*j);
    return u;
}
inline void appendVarint(std::vector<std::uint8_t>& out, std::uint64_t u) {
    while (u >= 128) {
        out.push_back(static_cast<std::uint8_t>((u & 127) | 128));
        u >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(u));
}
inline std::uint64_t readVarint(const std::vector<std::uint8_t>& in, std::size_t& pos) {
    std::uint64_t u = 0;
    for (int j = 0; j < 10; ++j) {
        if (pos == in.size()) throw std::runtime_error("DBF1 truncated varint");
        const auto b = in[pos++];
        if (j == 9 && b > 1) throw std::runtime_error("DBF1 overflowing varint");
        u |= std::uint64_t(b & 127) << (7*j);
        if (!(b & 128)) {
            if (j && !(b & 127)) throw std::runtime_error("DBF1 noncanonical varint");
            return u;
        }
    }
    throw std::runtime_error("DBF1 overflowing varint");
}
inline double roundEven(double x) {
    // Independent of the process floating-point rounding mode.
    const double low = std::floor(x);
    const double fraction = x - low;
    if (fraction < .5) return low;
    if (fraction > .5) return low + 1.;
    return std::fmod(low, 2.) == 0. ? low : low + 1.;
}
inline std::vector<std::uint8_t> packWords(const std::vector<std::uint64_t>& words,
                                           int scale, char dtypeMagic) {
    std::vector<std::uint8_t> out;
    out.reserve(10 + 10 * words.size());
    out.insert(out.end(), {'D', 'B', 'F', static_cast<std::uint8_t>(dtypeMagic),
        static_cast<std::uint8_t>(scale < 0), static_cast<std::uint8_t>(scale < 0 ? 255 : scale)});
    const auto count = static_cast<std::uint32_t>(words.size());
    for (int j = 0; j < 4; ++j) out.push_back(static_cast<std::uint8_t>(count >> (8*j)));
    if (!words.empty()) {
        appendWord(out, words[0]);
        for (std::size_t i = 1; i < words.size(); ++i) {
            const std::uint64_t delta = words[i] - words[i-1];
            appendVarint(out, (delta << 1) ^ (std::uint64_t(0) - (delta >> 63)));
        }
    }
    return out;
}
inline std::vector<std::uint8_t> encodeFrame(const std::vector<double>& values,
                                              int maxPoint = 18) {
    if (values.size() > kMaxBlock) throw std::runtime_error("DBF1 block exceeds 10000 values");
    std::vector<std::uint64_t> words(values.size());
    int scale = -1;
    const int cap = maxPoint < 0 ? 18 : std::min(maxPoint, 18);
    if (!values.empty()) {
        for (int s = cap; s >= 0; --s) {
            bool exact = true;
            for (std::size_t i = 0; i < values.size(); ++i) {
                const double product = values[i] * kPowers[s];
                if (!std::isfinite(product) || product < -0x1p63 || product >= 0x1p63) {
                    exact = false; break;
                }
                const double rounded = roundEven(product);
                if (rounded < -0x1p63 || rounded >= 0x1p63) { exact = false; break; }
                const auto code = static_cast<std::int64_t>(rounded);
                if (bits(static_cast<double>(code) / kPowers[s]) != bits(values[i])) {
                    exact = false; break;
                }
                words[i] = static_cast<std::uint64_t>(code);
            }
            if (exact) { scale = s; break; }
        }
    }
    if (scale < 0) {
        for (std::size_t i = 0; i < values.size(); ++i) words[i] = bits(values[i]);
    }
    return packWords(words, scale, '1');
}
inline std::vector<std::uint8_t> encodeFrame32(const std::vector<std::uint32_t>& input,
                                               int maxPoint = 18) {
    if (input.size() > kMaxBlock) throw std::runtime_error("DBF2 block exceeds 10000 values");
    std::vector<std::uint64_t> words(input.size());
    int scale = -1;
    const int cap = maxPoint < 0 ? 18 : std::min(maxPoint, 18);
    if (!input.empty()) {
        for (int s = cap; s >= 0; --s) {
            bool exact = true;
            for (std::size_t i = 0; i < input.size(); ++i) {
                const double product = static_cast<double>(fromBits32(input[i])) * kPowers[s];
                if (!std::isfinite(product) || product < -0x1p63 || product >= 0x1p63) {
                    exact = false; break;
                }
                const double rounded = roundEven(product);
                if (rounded < -0x1p63 || rounded >= 0x1p63) { exact = false; break; }
                const auto code = static_cast<std::int64_t>(rounded);
                if (bits32(static_cast<float>(static_cast<double>(code) / kPowers[s])) != input[i]) {
                    exact = false; break;
                }
                words[i] = static_cast<std::uint64_t>(code);
            }
            if (exact) { scale = s; break; }
        }
    }
    if (scale < 0) for (std::size_t i = 0; i < input.size(); ++i) words[i] = input[i];
    return packWords(words, scale, '2');
}
inline std::vector<double> decodeFrame(const std::vector<std::uint8_t>& frame) {
    if (frame.size() < 10 || std::memcmp(frame.data(), "DBF1", 4))
        throw std::runtime_error("DBF1 invalid header");
    const int mode = frame[4], scale = frame[5];
    if (!((mode == 0 && scale <= 18) || (mode == 1 && scale == 255)))
        throw std::runtime_error("DBF1 invalid mode/scale");
    std::uint32_t count = 0;
    for (int j = 0; j < 4; ++j) count |= std::uint32_t(frame[6+j]) << (8*j);
    if (count > kMaxBlock || (count && frame.size() < 18 + std::size_t(count) - 1))
        throw std::runtime_error("DBF1 impossible count");
    if (!count) {
        if (mode != 1 || frame.size() != 10) throw std::runtime_error("DBF1 invalid empty frame");
        return {};
    }
    std::vector<double> values(count);
    auto word = readWord(frame.data() + 10);
    std::size_t pos = 18;
    for (std::size_t i = 0; i < count; ++i) {
        if (i) {
            const auto u = readVarint(frame, pos);
            word += (u >> 1) ^ (std::uint64_t(0) - (u & 1));
        }
        values[i] = mode ? fromBits(word) : static_cast<double>(signedWord(word)) / kPowers[scale];
    }
    if (pos != frame.size()) throw std::runtime_error("DBF1 trailing bytes");
    return values;
}
inline std::vector<std::uint32_t> decodeFrame32(const std::vector<std::uint8_t>& frame) {
    if (frame.size() < 10 || std::memcmp(frame.data(), "DBF2", 4))
        throw std::runtime_error("DBF2 invalid header");
    const int mode = frame[4], scale = frame[5];
    if (!((mode == 0 && scale <= 18) || (mode == 1 && scale == 255)))
        throw std::runtime_error("DBF2 invalid mode/scale");
    std::uint32_t count = 0;
    for (int j = 0; j < 4; ++j) count |= std::uint32_t(frame[6+j]) << (8*j);
    if (count > kMaxBlock || (count && frame.size() < 18 + std::size_t(count) - 1))
        throw std::runtime_error("DBF2 impossible count");
    if (!count) {
        if (mode != 1 || frame.size() != 10) throw std::runtime_error("DBF2 invalid empty frame");
        return {};
    }
    std::vector<std::uint32_t> values(count);
    auto word = readWord(frame.data() + 10);
    std::size_t pos = 18;
    for (std::size_t i = 0; i < count; ++i) {
        if (i) {
            const auto u = readVarint(frame, pos);
            word += (u >> 1) ^ (std::uint64_t(0) - (u & 1));
        }
        if (mode) {
            if (word > UINT32_MAX) throw std::runtime_error("DBF2 invalid raw word");
            values[i] = static_cast<std::uint32_t>(word);
        } else {
            values[i] = bits32(static_cast<float>(static_cast<double>(signedWord(word)) / kPowers[scale]));
        }
    }
    if (pos != frame.size()) throw std::runtime_error("DBF2 trailing bytes");
    return values;
}
inline void verify(const std::vector<double>& input, const std::vector<double>& output) {
    if (input.size() != output.size()) throw std::runtime_error("DBF1 count mismatch");
    for (std::size_t i = 0; i < input.size(); ++i)
        if (bits(input[i]) != bits(output[i])) throw std::runtime_error("DBF1 bitwise mismatch");
}
} // namespace dbp_float64
