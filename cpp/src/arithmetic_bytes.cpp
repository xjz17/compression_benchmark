#include "arithmetic_bytes.hpp"
#include "benchmark_loss.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kHeader = 4 + 1024 + 4;
constexpr std::uint32_t kM = 1u << 15;
constexpr std::uint64_t kMaxRange = 0xffffffffULL;
constexpr std::uint64_t kMinRange = 0x10000ULL;
constexpr std::uint64_t kMask = 0xffffffffULL;
constexpr std::uint64_t kMask1 = 0xff000000ULL;
constexpr int kShift = 24;

std::vector<long long> countRaw(const std::vector<char> &data) {
    std::vector<long long> c(256, 0);
    for (unsigned char uc : data) {
        c[uc]++;
    }
    return c;
}

std::vector<int> buildFreqFromRawCounts(const std::vector<long long> &raw) {
    std::vector<int> adj(256);
    long long tot = 0;
    for (int i = 0; i < 256; ++i) {
        adj[i] = static_cast<int>(std::min<long long>(raw[static_cast<std::size_t>(i)], 0xffffffffLL) + 1);
        tot += adj[i];
    }
    std::vector<int> f(256);
    for (int i = 0; i < 256; ++i) {
        f[static_cast<std::size_t>(i)] =
            static_cast<int>(std::max(1LL, (static_cast<long long>(adj[i]) * kM) / tot));
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
                (static_cast<double>(adj[i]) * static_cast<double>(kM)) / static_cast<double>(tot) -
                static_cast<double>(f[static_cast<std::size_t>(i)]);
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        f[static_cast<std::size_t>(best)]++;
        sum++;
    }
    while (sum > static_cast<int>(kM)) {
        int best = -1;
        int bestVal = -1;
        for (int i = 0; i < 256; ++i) {
            if (f[static_cast<std::size_t>(i)] > 1 && f[static_cast<std::size_t>(i)] > bestVal) {
                bestVal = f[static_cast<std::size_t>(i)];
                best = i;
            }
        }
        if (best < 0) {
            throw std::runtime_error("arithmetic: cannot normalize frequency table");
        }
        f[static_cast<std::size_t>(best)]--;
        sum--;
    }
    return f;
}

std::vector<int> buildCum(const std::vector<int> &freq) {
    std::vector<int> cum(257, 0);
    for (int i = 0; i < 256; ++i) {
        cum[i + 1] = cum[i] + freq[static_cast<std::size_t>(i)];
    }
    if (cum[256] != static_cast<int>(kM)) {
        throw std::runtime_error("arithmetic frequency sum");
    }
    return cum;
}

int searchSym(const std::vector<int> &cum, int value) {
    int i = 0;
    int j = 255;
    while (i < j) {
        const int k = (i + j) / 2;
        if (cum[k + 1] <= value) {
            i = k + 1;
        } else {
            j = k;
        }
    }
    return i;
}

void encodeNormalize(std::uint64_t &low, std::uint64_t &range, std::vector<std::uint8_t> &out) {
    low &= kMask;
    while ((low & kMask1) == (((low + range) & kMask) & kMask1)) {
        out.push_back(static_cast<std::uint8_t>((low >> kShift) & 0xff));
        low = (low << 8) & kMask;
        range = (range << 8) & kMask;
    }
    while (range < kMinRange) {
        range = (kMinRange - (low & (kMinRange - 1))) << 8;
        out.push_back(static_cast<std::uint8_t>((low >> kShift) & 0xff));
        low = (low << 8) & kMask;
    }
    range &= kMask;
}

void decodeNormalize(std::uint64_t &low, std::uint64_t &range, std::uint64_t &code,
                     const std::vector<std::uint8_t> &payload, std::size_t &pos) {
    low &= kMask;
    code &= kMask;
    while ((low & kMask1) == (((low + range) & kMask) & kMask1)) {
        if (pos >= payload.size()) {
            throw std::runtime_error("arithmetic: truncated payload (normalize)");
        }
        code = (((code << 8) & kMask) + payload[pos]) & kMask;
        ++pos;
        low = (low << 8) & kMask;
        range = (range << 8) & kMask;
    }
    while (range < kMinRange) {
        range = (kMinRange - (low & (kMinRange - 1))) << 8;
        if (pos >= payload.size()) {
            throw std::runtime_error("arithmetic: truncated payload (normalize 2)");
        }
        code = (((code << 8) & kMask) + payload[pos]) & kMask;
        ++pos;
        low = (low << 8) & kMask;
    }
    range &= kMask;
}

std::vector<std::uint8_t> arithmeticEncode(const std::vector<char> &dataIn) {
    const int origLen = static_cast<int>(dataIn.size());
    std::vector<long long> raw = countRaw(dataIn);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(kHeader), 0);
    std::uint32_t olen = static_cast<std::uint32_t>(origLen);
    std::memcpy(out.data(), &olen, 4);
    for (int i = 0; i < 256; ++i) {
        std::uint32_t v = static_cast<std::uint32_t>(raw[static_cast<std::size_t>(i)] & 0xffffffffLL);
        std::memcpy(out.data() + 4 + i * 4, &v, 4);
    }
    if (origLen == 0) {
        std::uint32_t nb = 0;
        std::memcpy(out.data() + 1028, &nb, 4);
        return out;
    }
    int nz = 0;
    for (long long x : raw) {
        if (x > 0) {
            nz++;
        }
    }
    if (nz == 1) {
        std::uint32_t nb = 0;
        std::memcpy(out.data() + 1028, &nb, 4);
        return out;
    }
    std::vector<int> freq = buildFreqFromRawCounts(raw);
    std::vector<int> cum = buildCum(freq);

    std::vector<std::uint8_t> body;
    std::uint64_t low = 0;
    std::uint64_t range = kMaxRange;
    for (int i = 0; i < origLen; ++i) {
        const int sym = static_cast<unsigned char>(dataIn[static_cast<std::size_t>(i)]);
        std::uint64_t temp = range / kM;
        if (temp == 0) {
            temp = 1;
        }
        low = (low + static_cast<std::uint64_t>(cum[sym]) * temp) & kMask;
        range = static_cast<std::uint64_t>(freq[static_cast<std::size_t>(sym)]) * temp;
        encodeNormalize(low, range, body);
    }
    body.push_back(static_cast<std::uint8_t>((low >> 24) & 0xff));
    body.push_back(static_cast<std::uint8_t>((low >> 16) & 0xff));
    body.push_back(static_cast<std::uint8_t>((low >> 8) & 0xff));
    body.push_back(static_cast<std::uint8_t>(low & 0xff));

    std::uint32_t numBits = static_cast<std::uint32_t>(body.size() * 8);
    std::memcpy(out.data() + 1028, &numBits, 4);
    out.resize(static_cast<std::size_t>(kHeader) + body.size());
    std::memcpy(out.data() + kHeader, body.data(), body.size());
    return out;
}

std::vector<char> arithmeticDecode(const std::vector<std::uint8_t> &comp) {
    if (comp.size() < static_cast<std::size_t>(kHeader)) {
        throw std::runtime_error("arithmetic: truncated header");
    }
    std::uint32_t origLen = 0;
    std::memcpy(&origLen, comp.data(), 4);
    std::vector<long long> raw(256);
    for (int i = 0; i < 256; ++i) {
        std::uint32_t v = 0;
        std::memcpy(&v, comp.data() + 4 + i * 4, 4);
        raw[static_cast<std::size_t>(i)] = v;
    }
    std::uint32_t numBits = 0;
    std::memcpy(&numBits, comp.data() + 1028, 4);
    if (origLen == 0) {
        return {};
    }
    int nz = 0;
    int onlySym = -1;
    for (int s = 0; s < 256; ++s) {
        if (raw[static_cast<std::size_t>(s)] > 0) {
            nz++;
            onlySym = s;
        }
    }
    if (nz == 1) {
        return std::vector<char>(origLen, static_cast<char>(static_cast<unsigned char>(onlySym)));
    }
    const int need = kHeader + (static_cast<int>(numBits) + 7) / 8;
    if (static_cast<int>(comp.size()) < need) {
        throw std::runtime_error("arithmetic: truncated payload");
    }
    const int payloadLen = (static_cast<int>(numBits) + 7) / 8;
    std::vector<std::uint8_t> payload(comp.begin() + kHeader, comp.begin() + kHeader + payloadLen);
    if (payload.size() < 4) {
        throw std::runtime_error("arithmetic: payload too short");
    }
    std::uint64_t code = payload[0];
    code = (code << 8) | payload[1];
    code = (code << 8) | payload[2];
    code = (code << 8) | payload[3];
    code &= kMask;

    std::size_t pos = 4;
    std::uint64_t low = 0;
    std::uint64_t range = kMaxRange;
    std::vector<int> freq = buildFreqFromRawCounts(raw);
    std::vector<int> cum = buildCum(freq);

    std::vector<char> out(origLen);
    for (std::uint32_t op = 0; op < origLen; ++op) {
        std::uint64_t temp = range / kM;
        if (temp == 0) {
            temp = 1;
        }
        long long diff = static_cast<long long>((code - low) / temp);
        int value = static_cast<int>(diff);
        if (value < 0) {
            value = 0;
        }
        if (value >= static_cast<int>(kM)) {
            value = static_cast<int>(kM) - 1;
        }
        const int s = searchSym(cum, value);
        out[op] = static_cast<char>(static_cast<unsigned char>(s));
        low = (low + static_cast<std::uint64_t>(cum[static_cast<std::size_t>(s)]) * temp) & kMask;
        range = static_cast<std::uint64_t>(freq[static_cast<std::size_t>(s)]) * temp;
        decodeNormalize(low, range, code, payload, pos);
    }
    return out;
}

} // namespace

namespace arithmetic_bytes {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> data) {
    std::vector<char> input(data.size());
    std::transform(data.begin(), data.end(), input.begin(),
                   [](std::uint8_t value) { return static_cast<char>(value); });
    return arithmeticEncode(input);
}

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded) {
    const std::vector<std::uint8_t> input(encoded.begin(), encoded.end());
    const std::vector<char> restored = arithmeticDecode(input);
    std::vector<std::uint8_t> output(restored.size());
    std::transform(restored.begin(), restored.end(), output.begin(),
                   [](char value) { return static_cast<std::uint8_t>(value); });
    return output;
}

}  // namespace arithmetic_bytes

void testArithmeticCompression(const std::vector<char> &data, std::vector<long long> &result,
                               LossAccum *loss) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::uint8_t> enc = arithmeticEncode(data);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<char> dec = arithmeticDecode(enc);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (dec.size() != data.size() || !std::equal(data.begin(), data.end(), dec.begin())) {
        throw std::runtime_error("arithmetic roundtrip failed");
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
