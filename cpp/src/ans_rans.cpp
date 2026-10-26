#include "ans_rans.hpp"
#include "benchmark_loss.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kProbBits = 15;
constexpr std::uint32_t kM = 1u << kProbBits;
constexpr std::uint32_t kRansByteL = 1u << 23;

std::vector<int> buildFreq(const std::vector<char> &data) {
    std::vector<int> c(256, 0);
    for (unsigned char uc : data) {
        c[uc]++;
    }
    for (int &v : c) {
        v++;
    }
    long long tot = 0;
    for (int v : c) {
        tot += v;
    }
    std::vector<int> f(256);
    for (int i = 0; i < 256; ++i) {
        f[i] = static_cast<int>(std::max(1LL, (static_cast<long long>(c[i]) * kM) / tot));
    }
    int sum = 0;
    for (int v : f) {
        sum += v;
    }
    while (sum < static_cast<int>(kM)) {
        int best = 0;
        double bestScore = -1e100;
        for (int i = 0; i < 256; ++i) {
            const double score =
                (static_cast<double>(c[i]) * static_cast<double>(kM)) / static_cast<double>(tot) -
                static_cast<double>(f[i]);
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        f[best]++;
        sum++;
    }
    while (sum > static_cast<int>(kM)) {
        int best = -1;
        int bestVal = -1;
        for (int i = 0; i < 256; ++i) {
            if (f[i] > 1 && f[i] > bestVal) {
                bestVal = f[i];
                best = i;
            }
        }
        if (best < 0) {
            throw std::runtime_error("ANS: cannot normalize frequency table");
        }
        f[best]--;
        sum--;
    }
    return f;
}

std::vector<int> uniformFreq() {
    const int base = static_cast<int>(kM) / 256;
    int rem = static_cast<int>(kM) - base * 256;
    std::vector<int> f(256, base);
    f[0] += rem;
    return f;
}

std::vector<int> buildCum(const std::vector<int> &freq) {
    std::vector<int> cum(257, 0);
    for (int i = 0; i < 256; ++i) {
        cum[i + 1] = cum[i] + freq[i];
    }
    if (cum[256] != static_cast<int>(kM)) {
        throw std::runtime_error("ANS cum");
    }
    return cum;
}

int symFromSlot(const std::vector<int> &cum, int slot) {
    int lo = 0;
    int hi = 255;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cum[mid] <= slot) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

void writeU16Le(std::uint8_t* output, std::uint16_t value) {
    output[0] = static_cast<std::uint8_t>(value & 0xffu);
    output[1] = static_cast<std::uint8_t>(value >> 8);
}

void writeU32Le(std::uint8_t* output, std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) {
        output[byte] = static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu);
    }
}

std::uint16_t readU16Le(const std::uint8_t* input) {
    return static_cast<std::uint16_t>(input[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(input[1]) << 8);
}

std::uint32_t readU32Le(const std::uint8_t* input) {
    std::uint32_t value = 0;
    for (int byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(input[byte]) << (byte * 8);
    }
    return value;
}

std::vector<std::uint8_t> ansEncode(const std::vector<char> &dataIn) {
    if (dataIn.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("ANS input exceeds uint32 wire length");
    }
    std::vector<int> freq =
        dataIn.empty() ? uniformFreq() : buildFreq(dataIn);
    std::vector<int> cum = buildCum(freq);
    std::uint64_t state = kRansByteL;
    std::vector<std::uint8_t> emit;
    if (!dataIn.empty()) {
        for (int i = static_cast<int>(dataIn.size()) - 1; i >= 0; --i) {
            const int s =
                static_cast<unsigned char>(dataIn[static_cast<std::size_t>(i)]);
            const int fs = freq[s];
            const int cs = cum[s];
            const std::uint64_t maxv =
                ((static_cast<std::uint64_t>(kRansByteL) >> kProbBits) << 8) *
                static_cast<unsigned>(fs);
            while (state >= maxv) {
                emit.push_back(static_cast<std::uint8_t>(state & 0xFFu));
                state >>= 8;
            }
            state = (state / static_cast<unsigned>(fs)) * kM + (state % static_cast<unsigned>(fs)) +
                    static_cast<unsigned>(cs);
        }
    }
    state &= 0xFFFFFFFFu;
    for (int k = 0; k < 4; ++k) {
        emit.push_back(static_cast<std::uint8_t>(state & 0xFFu));
        state >>= 8;
    }
    std::vector<std::uint8_t> out;
    out.resize(4 + 512 + emit.size());
    const std::uint32_t olen = static_cast<std::uint32_t>(dataIn.size());
    writeU32Le(out.data(), olen);
    for (int i = 0; i < 256; ++i) {
        const std::uint16_t v = static_cast<std::uint16_t>(freq[i] & 0xffff);
        writeU16Le(out.data() + 4 + i * 2, v);
    }
    for (std::size_t j = 0; j < emit.size(); ++j) {
        out[4 + 512 + j] = emit[emit.size() - 1 - j];
    }
    return out;
}

std::vector<char> ansDecode(const std::vector<std::uint8_t> &comp) {
    if (comp.size() < 4 + 512 + 4) {
        throw std::runtime_error("ANS truncated");
    }
    const std::uint32_t origLen = readU32Le(comp.data());
    std::vector<int> freq(256);
    int sumf = 0;
    for (int i = 0; i < 256; ++i) {
        const std::uint16_t v = readU16Le(comp.data() + 4 + i * 2);
        freq[i] = v;
        sumf += freq[i];
    }
    if (sumf != static_cast<int>(kM)) {
        throw std::runtime_error("ANS bad freq sum");
    }
    std::vector<int> cum = buildCum(freq);
    const int bodyStart = 4 + 512;
    const std::size_t n = comp.size();
    // Encoder reverses emit[]: first body bytes are flush as fb3,fb2,fb1,fb0 (big-endian state).
    std::uint32_t state = (static_cast<std::uint32_t>(comp[static_cast<std::size_t>(bodyStart)]) << 24)
        | (static_cast<std::uint32_t>(comp[static_cast<std::size_t>(bodyStart + 1)]) << 16)
        | (static_cast<std::uint32_t>(comp[static_cast<std::size_t>(bodyStart + 2)]) << 8)
        | static_cast<std::uint32_t>(comp[static_cast<std::size_t>(bodyStart + 3)]);
    int ptr = bodyStart + 4;
    if (origLen == 0) {
        if (state != kRansByteL || static_cast<std::size_t>(ptr) != n) {
            throw std::runtime_error("ANS invalid empty stream");
        }
        return {};
    }
    std::vector<char> out(origLen);
    std::size_t op = 0;
    while (op < origLen) {
        while (state < kRansByteL) {
            if (ptr >= static_cast<int>(n)) {
                throw std::runtime_error("ANS truncated body");
            }
            state = (state << 8) | comp[static_cast<std::size_t>(ptr)];
            ptr++;
        }
        const int slot = static_cast<int>(state % kM);
        const int s = symFromSlot(cum, slot);
        out[op++] = static_cast<char>(static_cast<unsigned char>(s));
        const int fs = freq[s];
        const int cs = cum[s];
        state = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(fs) * (state / kM) + (state % kM) - cs) & 0xFFFFFFFFu);
    }
    while (state < kRansByteL) {
        if (ptr >= static_cast<int>(n)) {
            throw std::runtime_error("ANS truncated final state");
        }
        state = (state << 8) | comp[static_cast<std::size_t>(ptr++)];
    }
    if (static_cast<std::size_t>(ptr) != n || state != kRansByteL) {
        throw std::runtime_error("ANS trailing or corrupt body");
    }
    return out;
}

} // namespace

namespace ans_rans {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> data) {
    std::vector<char> input(data.size());
    std::transform(data.begin(), data.end(), input.begin(),
                   [](std::uint8_t value) { return static_cast<char>(value); });
    return ansEncode(input);
}

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded) {
    const std::vector<std::uint8_t> input(encoded.begin(), encoded.end());
    const std::vector<char> restored = ansDecode(input);
    std::vector<std::uint8_t> output(restored.size());
    std::transform(restored.begin(), restored.end(), output.begin(),
                   [](char value) { return static_cast<std::uint8_t>(value); });
    return output;
}

}  // namespace ans_rans

void testAnsEntropyCompression(const std::vector<char> &data,
                               std::vector<long long> &result, LossAccum *loss) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::uint8_t> raw(data.size());
    std::transform(data.begin(), data.end(), raw.begin(),
                   [](char value) { return static_cast<std::uint8_t>(value); });
    std::vector<std::uint8_t> enc = ans_rans::encode(raw);
    auto t1 = std::chrono::high_resolution_clock::now();
    const std::vector<std::uint8_t> decoded = ans_rans::decode(enc);
    std::vector<char> dec(decoded.size());
    std::transform(decoded.begin(), decoded.end(), dec.begin(),
                   [](std::uint8_t value) { return static_cast<char>(value); });
    auto t2 = std::chrono::high_resolution_clock::now();
    if (dec.size() != data.size() || !std::equal(data.begin(), data.end(), dec.begin())) {
        throw std::runtime_error("ANS roundtrip failed");
    }
    std::chrono::duration<double, std::nano> encNs = t1 - t0;
    std::chrono::duration<double, std::nano> decNs = t2 - t1;
    result.resize(4);
    result[0] = static_cast<long long>(data.size());
    result[1] = static_cast<long long>(enc.size());
    result[2] = static_cast<long long>(encNs.count());
    result[3] = static_cast<long long>(decNs.count());
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, dec, *loss);
    }
}
