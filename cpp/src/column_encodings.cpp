#include "column_encodings.hpp"

#include "benchmark_loss.hpp"
#include "dbp_float64_codec.hpp"

#include <lz4frame.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace colenc {

namespace {

/** Apache Commons `FramedLZ4CompressorOutputStream.Parameters.DEFAULT` frame fields. */
static void apply_apache_framed_lz4_prefs(LZ4F_preferences_t *prefs) {
    *prefs = LZ4F_INIT_PREFERENCES;
    prefs->frameInfo.blockSizeID = LZ4F_max4MB;
    prefs->frameInfo.blockMode = LZ4F_blockIndependent;
    prefs->frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
    prefs->frameInfo.blockChecksumFlag = LZ4F_noBlockChecksum;
    prefs->autoFlush = 1;
}

void writeVarintU32(std::vector<std::uint8_t> &out, std::uint32_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<std::uint8_t>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(v));
}

void writeVarintU64(std::vector<std::uint8_t> &out, std::uint64_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<std::uint8_t>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(v));
}

std::uint64_t readVarintU64(const std::vector<std::uint8_t> &in, std::size_t &pos) {
    std::uint64_t result = 0;
    int shift = 0;
    while (pos < in.size()) {
        std::uint8_t b = in[pos++];
        result |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) {
            return result;
        }
        shift += 7;
        if (shift > 70) {
            throw std::runtime_error("bad varint");
        }
    }
    throw std::runtime_error("truncated varint");
}

int readVarintU32(const std::vector<std::uint8_t> &in, std::size_t &pos) {
    std::uint32_t result = 0;
    int shift = 0;
    while (pos < in.size()) {
        std::uint8_t b = in[pos++];
        result |= static_cast<std::uint32_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) {
            return static_cast<int>(result);
        }
        shift += 7;
        if (shift > 35) {
            throw std::runtime_error("bad varint");
        }
    }
    throw std::runtime_error("truncated varint");
}

std::int32_t zigzag32(std::int32_t d) {
    return (d << 1) ^ (d >> 31);
}

std::int32_t unzigzag32(std::uint32_t u) {
    return static_cast<std::int32_t>((u >> 1) ^
                                     (-static_cast<std::int32_t>(u & 1u)));
}

std::int64_t zigzag64(std::int64_t d) {
    const std::uint64_t bits = static_cast<std::uint64_t>(d);
    return static_cast<std::int64_t>((bits << 1) ^ static_cast<std::uint64_t>(d >> 63));
}

std::int64_t unzigzag64(std::uint64_t u) {
    return static_cast<std::int64_t>((u >> 1) ^ (-static_cast<std::int64_t>(u & 1u)));
}

void writeI64LE(std::vector<std::uint8_t> &out, std::int64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(v) >> (8 * i)) & 0xFF));
    }
}

std::int64_t readI64LE(const std::vector<std::uint8_t> &blob, std::size_t &pos) {
    if (pos + 8 > blob.size()) {
        throw std::runtime_error("truncated i64");
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(blob[pos++]) << (8 * i);
    }
    return static_cast<std::int64_t>(v);
}

std::vector<std::uint8_t> encode_delta_zigzag_varint_impl(const std::vector<int> &values) {
    std::vector<std::uint8_t> out;
    if (values.empty()) {
        return out;
    }
    std::int32_t first = static_cast<std::int32_t>(values[0]);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>((static_cast<std::uint32_t>(first) >> (i * 8)) & 0xFF));
    }
    std::int32_t prev = first;
    for (std::size_t j = 1; j < values.size(); ++j) {
        std::int32_t cur = static_cast<std::int32_t>(values[j]);
        std::int32_t d = cur - prev;
        prev = cur;
        writeVarintU32(out, static_cast<std::uint32_t>(zigzag32(d)));
    }
    return out;
}

std::uint32_t read_u32_le(const std::vector<std::uint8_t> &blob, std::size_t &pos) {
    if (pos + 4 > blob.size()) {
        throw std::runtime_error("truncated u32");
    }
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(blob[pos++]) << (i * 8);
    }
    return v;
}

std::int32_t read_i32_le(const std::vector<std::uint8_t> &blob, std::size_t &pos) {
    return static_cast<std::int32_t>(read_u32_le(blob, pos));
}

std::vector<int> decode_delta_zigzag_varint_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<int> res;
    if (blob.size() < 4) {
        return res;
    }
    std::uint32_t v0 = 0;
    for (int i = 0; i < 4; ++i) {
        v0 |= static_cast<std::uint32_t>(blob[static_cast<std::size_t>(i)]) << (i * 8);
    }
    std::int32_t cur = static_cast<std::int32_t>(v0);
    res.push_back(static_cast<int>(cur));
    std::size_t pos = 4;
    while (pos < blob.size()) {
        std::uint32_t u = static_cast<std::uint32_t>(readVarintU32(blob, pos));
        std::int32_t d = unzigzag32(u);
        cur += d;
        res.push_back(static_cast<int>(cur));
    }
    return res;
}

std::vector<int> decode_dictionary_int_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<int> res;
    if (blob.empty()) {
        return res;
    }
    std::size_t pos = 0;
    std::uint32_t nd = read_u32_le(blob, pos);
    std::vector<int> dict(nd);
    for (std::uint32_t i = 0; i < nd; ++i) {
        dict[i] = read_i32_le(blob, pos);
    }
    std::uint32_t nv = read_u32_le(blob, pos);
    res.resize(static_cast<std::size_t>(nv));
    for (std::uint32_t i = 0; i < nv; ++i) {
        std::uint32_t ix = read_u32_le(blob, pos);
        if (ix >= dict.size()) {
            throw std::runtime_error("dictionary index out of range");
        }
        res[static_cast<std::size_t>(i)] = dict[static_cast<std::size_t>(ix)];
    }
    if (pos != blob.size()) {
        throw std::runtime_error("trailing bytes in dictionary int blob");
    }
    return res;
}

std::vector<float> decode_dictionary_float_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<float> res;
    if (blob.empty()) {
        return res;
    }
    std::size_t pos = 0;
    std::uint32_t nd = read_u32_le(blob, pos);
    std::vector<std::uint32_t> dictBits(nd);
    for (std::uint32_t i = 0; i < nd; ++i) {
        dictBits[i] = read_u32_le(blob, pos);
    }
    std::uint32_t nv = read_u32_le(blob, pos);
    res.resize(static_cast<std::size_t>(nv));
    for (std::uint32_t i = 0; i < nv; ++i) {
        std::uint32_t ix = read_u32_le(blob, pos);
        if (ix >= dictBits.size()) {
            throw std::runtime_error("dictionary float index out of range");
        }
        float f = 0;
        std::uint32_t b = dictBits[static_cast<std::size_t>(ix)];
        std::memcpy(&f, &b, 4);
        res[static_cast<std::size_t>(i)] = f;
    }
    if (pos != blob.size()) {
        throw std::runtime_error("trailing bytes in dictionary float blob");
    }
    return res;
}

std::vector<int> decode_rle_int_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<int> res;
    std::size_t pos = 0;
    while (pos < blob.size()) {
        std::int32_t v = read_i32_le(blob, pos);
        std::uint32_t run = static_cast<std::uint32_t>(readVarintU32(blob, pos));
        if (run == 0) {
            throw std::runtime_error("zero run length in RLE");
        }
        const auto prev = res.size();
        res.resize(prev + static_cast<std::size_t>(run));
        std::fill(res.begin() + static_cast<std::ptrdiff_t>(prev), res.end(), static_cast<int>(v));
    }
    return res;
}

std::vector<std::uint8_t> encode_dictionary_int_impl(const std::vector<int> &values) {
    std::vector<int> order;
    std::unordered_map<int, int> ix;
    ix.reserve(values.size() * 2);
    for (int v : values) {
        if (ix.find(v) == ix.end()) {
            ix[v] = static_cast<int>(order.size());
            order.push_back(v);
        }
    }
    std::vector<std::uint8_t> out;
    auto w32 = [&out](std::uint32_t u) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((u >> (i * 8)) & 0xFF));
        }
    };
    w32(static_cast<std::uint32_t>(order.size()));
    for (int k : order) {
        std::uint32_t x = static_cast<std::uint32_t>(static_cast<std::int32_t>(k));
        w32(x);
    }
    w32(static_cast<std::uint32_t>(values.size()));
    for (int v : values) {
        w32(static_cast<std::uint32_t>(ix[v]));
    }
    return out;
}

std::vector<std::uint8_t> encode_dictionary_float_impl(const std::vector<float> &values) {
    std::vector<std::uint32_t> order;
    std::unordered_map<std::uint32_t, int> ix;
    for (float f : values) {
        std::uint32_t b = 0;
        std::memcpy(&b, &f, 4);
        if (ix.find(b) == ix.end()) {
            ix[b] = static_cast<int>(order.size());
            order.push_back(b);
        }
    }
    std::vector<std::uint8_t> out;
    auto w32 = [&out](std::uint32_t u) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((u >> (i * 8)) & 0xFF));
        }
    };
    w32(static_cast<std::uint32_t>(order.size()));
    for (std::uint32_t k : order) {
        w32(k);
    }
    w32(static_cast<std::uint32_t>(values.size()));
    for (float f : values) {
        std::uint32_t b = 0;
        std::memcpy(&b, &f, 4);
        w32(static_cast<std::uint32_t>(ix[b]));
    }
    return out;
}

std::vector<std::uint8_t> encode_dictionary_int64_impl(const std::vector<std::int64_t> &values) {
    std::vector<std::int64_t> order;
    std::unordered_map<std::int64_t, std::uint32_t> ix;
    ix.reserve(values.size() * 2);
    for (std::int64_t v : values) {
        if (ix.find(v) == ix.end()) {
            ix[v] = static_cast<std::uint32_t>(order.size());
            order.push_back(v);
        }
    }
    std::vector<std::uint8_t> out;
    auto w32 = [&out](std::uint32_t u) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((u >> (i * 8)) & 0xFF));
        }
    };
    w32(static_cast<std::uint32_t>(order.size()));
    for (std::int64_t k : order) {
        writeI64LE(out, k);
    }
    w32(static_cast<std::uint32_t>(values.size()));
    for (std::int64_t v : values) {
        w32(ix[v]);
    }
    return out;
}

std::vector<std::int64_t> decode_dictionary_int64_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<std::int64_t> res;
    if (blob.empty()) {
        return res;
    }
    std::size_t pos = 0;
    const std::uint32_t nd = read_u32_le(blob, pos);
    std::vector<std::int64_t> dict(nd);
    for (std::uint32_t i = 0; i < nd; ++i) {
        dict[static_cast<std::size_t>(i)] = readI64LE(blob, pos);
    }
    const std::uint32_t nv = read_u32_le(blob, pos);
    res.resize(static_cast<std::size_t>(nv));
    for (std::uint32_t i = 0; i < nv; ++i) {
        const std::uint32_t ix = read_u32_le(blob, pos);
        if (ix >= dict.size()) {
            throw std::runtime_error("dictionary int64 index out of range");
        }
        res[static_cast<std::size_t>(i)] = dict[static_cast<std::size_t>(ix)];
    }
    if (pos != blob.size()) {
        throw std::runtime_error("trailing bytes in dictionary int64 blob");
    }
    return res;
}

std::vector<std::uint8_t> encode_dictionary_double64_impl(const std::vector<double> &values) {
    std::vector<std::uint64_t> order;
    std::unordered_map<std::uint64_t, std::uint32_t> ix;
    ix.reserve(values.size() * 2);
    for (double d : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &d, sizeof(bits));
        if (ix.find(bits) == ix.end()) {
            ix[bits] = static_cast<std::uint32_t>(order.size());
            order.push_back(bits);
        }
    }
    std::vector<std::uint8_t> out;
    auto w32 = [&out](std::uint32_t u) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((u >> (i * 8)) & 0xFF));
        }
    };
    w32(static_cast<std::uint32_t>(order.size()));
    for (std::uint64_t bits : order) {
        writeI64LE(out, static_cast<std::int64_t>(bits));
    }
    w32(static_cast<std::uint32_t>(values.size()));
    for (double d : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &d, sizeof(bits));
        w32(ix[bits]);
    }
    return out;
}

std::vector<double> decode_dictionary_double64_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<double> res;
    if (blob.empty()) {
        return res;
    }
    std::size_t pos = 0;
    const std::uint32_t nd = read_u32_le(blob, pos);
    std::vector<std::uint64_t> dict_bits(nd);
    for (std::uint32_t i = 0; i < nd; ++i) {
        dict_bits[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(readI64LE(blob, pos));
    }
    const std::uint32_t nv = read_u32_le(blob, pos);
    res.resize(static_cast<std::size_t>(nv));
    for (std::uint32_t i = 0; i < nv; ++i) {
        const std::uint32_t ix = read_u32_le(blob, pos);
        if (ix >= dict_bits.size()) {
            throw std::runtime_error("dictionary double64 index out of range");
        }
        const std::uint64_t bits = dict_bits[static_cast<std::size_t>(ix)];
        std::memcpy(&res[static_cast<std::size_t>(i)], &bits, sizeof(bits));
    }
    if (pos != blob.size()) {
        throw std::runtime_error("trailing bytes in dictionary double64 blob");
    }
    return res;
}

std::vector<std::uint8_t> encode_delta_zigzag_varint_int64_impl(const std::vector<std::int64_t> &values) {
    std::vector<std::uint8_t> out;
    if (values.empty()) {
        return out;
    }
    writeI64LE(out, values[0]);
    std::uint64_t prev = static_cast<std::uint64_t>(values[0]);
    for (std::size_t j = 1; j < values.size(); ++j) {
        const std::uint64_t cur = static_cast<std::uint64_t>(values[j]);
        const std::int64_t d = static_cast<std::int64_t>(cur - prev);
        prev = cur;
        writeVarintU64(out, static_cast<std::uint64_t>(zigzag64(d)));
    }
    return out;
}

std::vector<std::int64_t> decode_delta_zigzag_varint_int64_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<std::int64_t> res;
    if (blob.size() < 8) {
        return res;
    }
    std::size_t pos = 0;
    std::uint64_t cur = static_cast<std::uint64_t>(readI64LE(blob, pos));
    res.push_back(static_cast<std::int64_t>(cur));
    while (pos < blob.size()) {
        const std::uint64_t u = readVarintU64(blob, pos);
        const std::int64_t d = unzigzag64(u);
        cur += static_cast<std::uint64_t>(d);
        res.push_back(static_cast<std::int64_t>(cur));
    }
    return res;
}

std::vector<std::uint8_t> encode_rle_int64_impl(const std::vector<std::int64_t> &values) {
    std::vector<std::uint8_t> out;
    std::size_t i = 0;
    while (i < values.size()) {
        std::size_t j = i + 1;
        while (j < values.size() && values[j] == values[i]) {
            ++j;
        }
        writeI64LE(out, values[i]);
        writeVarintU64(out, static_cast<std::uint64_t>(j - i));
        i = j;
    }
    return out;
}

std::vector<std::int64_t> decode_rle_int64_impl(const std::vector<std::uint8_t> &blob) {
    std::vector<std::int64_t> out;
    std::size_t pos = 0;
    while (pos < blob.size()) {
        const std::int64_t val = readI64LE(blob, pos);
        const std::uint64_t run = readVarintU64(blob, pos);
        for (std::uint64_t k = 0; k < run; ++k) {
            out.push_back(val);
        }
    }
    return out;
}

std::vector<std::uint8_t> encode_rle_int_impl(const std::vector<int> &values) {
    std::vector<std::uint8_t> out;
    std::size_t i = 0;
    while (i < values.size()) {
        std::size_t j = i + 1;
        while (j < values.size() && values[j] == values[i]) {
            ++j;
        }
        std::int32_t vv = static_cast<std::int32_t>(values[i]);
        auto u = static_cast<std::uint32_t>(vv);
        for (int b = 0; b < 4; ++b) {
            out.push_back(static_cast<std::uint8_t>((u >> (b * 8)) & 0xFF));
        }
        writeVarintU32(out, static_cast<std::uint32_t>(j - i));
        i = j;
    }
    return out;
}

std::vector<std::uint8_t> byteStreamSplitFloat(const std::uint8_t *raw, std::size_t n) {
    std::vector<std::uint8_t> bss(n * 4);
    for (int b = 0; b < 4; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            bss[b * n + i] = raw[i * 4 + static_cast<std::size_t>(b)];
        }
    }
    return bss;
}

std::vector<std::uint8_t> byteStreamSplitDouble(const std::uint8_t *raw, std::size_t n) {
    std::vector<std::uint8_t> bss(n * 8);
    for (int b = 0; b < 8; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            bss[static_cast<std::size_t>(b) * n + i] = raw[i * 8 + static_cast<std::size_t>(b)];
        }
    }
    return bss;
}

std::vector<std::uint8_t> inverseByteStreamSplit(const std::vector<std::uint8_t> &bss,
                                                 std::size_t n) {
    std::vector<std::uint8_t> raw(n * 4);
    for (int b = 0; b < 4; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            raw[i * 4 + static_cast<std::size_t>(b)] = bss[b * n + i];
        }
    }
    return raw;
}

std::vector<std::uint8_t> inverseByteStreamSplitDouble(const std::vector<std::uint8_t> &bss,
                                                       std::size_t n) {
    std::vector<std::uint8_t> raw(n * 8);
    for (int b = 0; b < 8; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            raw[i * 8 + static_cast<std::size_t>(b)] = bss[static_cast<std::size_t>(b) * n + i];
        }
    }
    return raw;
}

std::vector<std::uint8_t> encode_bss_lz4_float_impl(const std::vector<float> &values) {
    std::vector<std::uint8_t> raw(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::memcpy(&raw[i * 4], &values[i], 4);
    }
    std::vector<std::uint8_t> bss = byteStreamSplitFloat(raw.data(), values.size());
    LZ4F_preferences_t prefs{};
    apply_apache_framed_lz4_prefs(&prefs);
    // `LZ4F_compressFrame` applies `LZ4F_optimalBSID`, shrinking block-size in the frame header for
    // small `srcSize` (e.g. 64KB for 256-byte BSS). Java Commons always writes 4MB BD; use streaming.
    std::size_t dstCap = LZ4F_compressFrameBound(bss.size(), &prefs);
    std::vector<std::uint8_t> out(dstCap);

    LZ4F_cctx *cctx = nullptr;
    LZ4F_errorCode_t er = LZ4F_createCompressionContext(&cctx, LZ4F_VERSION);
    if (LZ4F_isError(er)) {
        throw std::runtime_error("LZ4F_createCompressionContext failed");
    }
    std::size_t pos = 0;
    std::size_t n = LZ4F_compressBegin(cctx, out.data(), out.size(), &prefs);
    if (LZ4F_isError(n)) {
        LZ4F_freeCompressionContext(cctx);
        throw std::runtime_error("LZ4F_compressBegin failed");
    }
    pos += n;

    const void *bssPtr = bss.empty() ? static_cast<const void *>(nullptr) : bss.data();
    n = LZ4F_compressUpdate(cctx, out.data() + pos, out.size() - pos, bssPtr, bss.size(), nullptr);
    if (LZ4F_isError(n)) {
        LZ4F_freeCompressionContext(cctx);
        throw std::runtime_error("LZ4F_compressUpdate failed");
    }
    pos += n;

    n = LZ4F_compressEnd(cctx, out.data() + pos, out.size() - pos, nullptr);
    if (LZ4F_isError(n)) {
        LZ4F_freeCompressionContext(cctx);
        throw std::runtime_error("LZ4F_compressEnd failed");
    }
    pos += n;
    LZ4F_freeCompressionContext(cctx);
    out.resize(pos);
    return out;
}

std::vector<std::uint8_t> encode_bss_lz4_double_impl(const std::vector<double> &values) {
    std::vector<std::uint8_t> raw(values.size() * 8);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::memcpy(&raw[i * 8], &values[i], 8);
    }
    std::vector<std::uint8_t> bss = byteStreamSplitDouble(raw.data(), values.size());
    LZ4F_preferences_t prefs{};
    apply_apache_framed_lz4_prefs(&prefs);
    std::size_t dstCap = LZ4F_compressFrameBound(bss.size(), &prefs);
    std::vector<std::uint8_t> out(dstCap);

    LZ4F_cctx *cctx = nullptr;
    LZ4F_errorCode_t er = LZ4F_createCompressionContext(&cctx, LZ4F_VERSION);
    if (LZ4F_isError(er)) {
        throw std::runtime_error("LZ4F_createCompressionContext failed");
    }
    std::size_t pos = 0;
    std::size_t n = LZ4F_compressBegin(cctx, out.data(), out.size(), &prefs);
    if (LZ4F_isError(n)) {
        LZ4F_freeCompressionContext(cctx);
        throw std::runtime_error("LZ4F_compressBegin failed");
    }
    pos += n;

    const void *bssPtr = bss.empty() ? static_cast<const void *>(nullptr) : bss.data();
    n = LZ4F_compressUpdate(cctx, out.data() + pos, out.size() - pos, bssPtr, bss.size(), nullptr);
    if (LZ4F_isError(n)) {
        LZ4F_freeCompressionContext(cctx);
        throw std::runtime_error("LZ4F_compressUpdate failed");
    }
    pos += n;

    n = LZ4F_compressEnd(cctx, out.data() + pos, out.size() - pos, nullptr);
    if (LZ4F_isError(n)) {
        LZ4F_freeCompressionContext(cctx);
        throw std::runtime_error("LZ4F_compressEnd failed");
    }
    pos += n;
    LZ4F_freeCompressionContext(cctx);
    out.resize(pos);
    return out;
}

std::vector<std::uint8_t> lz4_framed_decompress_all(const std::vector<std::uint8_t> &comp) {
    if (comp.empty()) {
        return {};
    }
    LZ4F_dctx *dctx = nullptr;
    LZ4F_errorCode_t er = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
    if (LZ4F_isError(er)) {
        throw std::runtime_error("LZ4F_createDecompressionContext failed");
    }
    std::vector<std::uint8_t> dst((std::max)(comp.size() * 16, std::size_t{1024 * 1024}));
    std::size_t dstPos = 0;
    std::size_t srcPos = 0;
    while (srcPos < comp.size()) {
        std::size_t dstChunk = dst.size() - dstPos;
        std::size_t srcChunk = comp.size() - srcPos;
        std::size_t hint = LZ4F_decompress(dctx, dst.data() + dstPos, &dstChunk,
                                           comp.data() + srcPos, &srcChunk, nullptr);
        if (LZ4F_isError(hint)) {
            LZ4F_freeDecompressionContext(dctx);
            throw std::runtime_error("LZ4F_decompress failed");
        }
        srcPos += srcChunk;
        dstPos += dstChunk;
        if (hint == 0) {
            break;
        }
    }
    LZ4F_freeDecompressionContext(dctx);
    dst.resize(dstPos);
    return dst;
}

std::vector<float> floats_from_planar_bss_impl(const std::vector<std::uint8_t> &bss) {
    if (bss.empty()) {
        return {};
    }
    if (bss.size() % 4 != 0) {
        throw std::runtime_error("planar BSS length not multiple of 4");
    }
    std::size_t n = bss.size() / 4;
    std::vector<std::uint8_t> raw = inverseByteStreamSplit(bss, n);
    std::vector<float> out(n);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

std::vector<float> decode_bss_lz4_float_impl(const std::vector<std::uint8_t> &blob) {
    if (blob.empty()) {
        return {};
    }
    std::vector<std::uint8_t> bss = lz4_framed_decompress_all(blob);
    if (bss.empty()) {
        return {};
    }
    if (bss.size() % 4 != 0) {
        throw std::runtime_error("bad BSS size from LZ4 frame");
    }
    std::size_t nfloat = bss.size() / 4;
    std::vector<std::uint8_t> raw = inverseByteStreamSplit(bss, nfloat);
    std::vector<float> out(nfloat);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

} // namespace

std::vector<std::uint8_t> encodeDeltaZigZagVarint(const std::vector<int> &values) {
    return encode_delta_zigzag_varint_impl(values);
}

std::vector<std::uint8_t> encodeDeltaBinaryPackedInt64Ablation(
    const std::vector<std::int64_t> &values, std::string_view omitted_operator) {
    std::uint8_t mode = 0;
    if (omitted_operator.empty() || omitted_operator == "none" || omitted_operator == "scaling") {
        mode = 0;
    } else if (omitted_operator == "delta") {
        mode = 1;
    } else if (omitted_operator == "zigzag") {
        mode = 2;
    } else if (omitted_operator == "bit-packing") {
        mode = 3;
    } else {
        throw std::invalid_argument("unsupported DELTA_BINARY_PACKED ablation operator");
    }
    std::vector<std::uint8_t> out{mode};
    if (values.empty()) return out;
    if (mode == 0) {
        const auto body = encode_delta_zigzag_varint_int64_impl(values);
        out.insert(out.end(), body.begin(), body.end());
        return out;
    }
    if (mode == 1) {
        for (std::int64_t value : values) {
            writeVarintU64(out, static_cast<std::uint64_t>(zigzag64(value)));
        }
        return out;
    }
    writeI64LE(out, values.front());
    std::uint64_t previous = static_cast<std::uint64_t>(values.front());
    for (std::size_t i = 1; i < values.size(); ++i) {
        const std::uint64_t current = static_cast<std::uint64_t>(values[i]);
        const std::int64_t delta = static_cast<std::int64_t>(current - previous);
        if (mode == 2) {
            writeVarintU64(out, static_cast<std::uint64_t>(delta));
        } else {
            writeI64LE(out, zigzag64(delta));
        }
        previous = current;
    }
    return out;
}

std::vector<std::int64_t> decodeDeltaBinaryPackedInt64Ablation(
    const std::vector<std::uint8_t> &payload, std::size_t expected_count) {
    if (payload.empty()) throw std::runtime_error("empty DELTA_BINARY_PACKED ablation payload");
    const std::uint8_t mode = payload.front();
    std::vector<std::uint8_t> body(payload.begin() + 1, payload.end());
    if (expected_count == 0) {
        if (!body.empty()) throw std::runtime_error("nonempty zero-count DELTA_BINARY_PACKED payload");
        return {};
    }
    if (mode == 0) {
        auto decoded = decode_delta_zigzag_varint_int64_impl(body);
        if (decoded.size() != expected_count) {
            throw std::runtime_error("DELTA_BINARY_PACKED baseline count mismatch");
        }
        return decoded;
    }
    std::size_t pos = 0;
    std::vector<std::int64_t> decoded;
    decoded.reserve(expected_count);
    if (mode == 1) {
        while (pos < body.size() && decoded.size() < expected_count) {
            decoded.push_back(unzigzag64(readVarintU64(body, pos)));
        }
    } else if (mode == 2 || mode == 3) {
        std::uint64_t previous = static_cast<std::uint64_t>(readI64LE(body, pos));
        decoded.push_back(static_cast<std::int64_t>(previous));
        while (decoded.size() < expected_count) {
            std::int64_t delta = 0;
            if (mode == 2) {
                delta = static_cast<std::int64_t>(readVarintU64(body, pos));
            } else {
                delta = unzigzag64(static_cast<std::uint64_t>(readI64LE(body, pos)));
            }
            previous += static_cast<std::uint64_t>(delta);
            decoded.push_back(static_cast<std::int64_t>(previous));
        }
    } else {
        throw std::runtime_error("invalid DELTA_BINARY_PACKED ablation mode");
    }
    if (decoded.size() != expected_count || pos != body.size()) {
        throw std::runtime_error("DELTA_BINARY_PACKED ablation payload mismatch");
    }
    return decoded;
}

std::vector<std::uint8_t> encodeDictionaryFloat(const std::vector<float> &values) {
    return encode_dictionary_float_impl(values);
}

std::vector<std::uint8_t> encodeDictionaryInt(const std::vector<int> &values) {
    return encode_dictionary_int_impl(values);
}

std::vector<std::uint8_t> encodeRleInt(const std::vector<int> &values) {
    return encode_rle_int_impl(values);
}

std::vector<std::uint8_t> encodeBssLz4Float(const std::vector<float> &values) {
    return encode_bss_lz4_float_impl(values);
}

std::vector<int> decodeDeltaZigZagVarint(const std::vector<std::uint8_t> &blob) {
    return decode_delta_zigzag_varint_impl(blob);
}

std::vector<int> decodeDictionaryInt(const std::vector<std::uint8_t> &blob) {
    return decode_dictionary_int_impl(blob);
}

std::vector<float> decodeDictionaryFloat(const std::vector<std::uint8_t> &blob) {
    return decode_dictionary_float_impl(blob);
}

std::vector<int> decodeRleInt(const std::vector<std::uint8_t> &blob) {
    return decode_rle_int_impl(blob);
}

std::vector<std::uint8_t> planarByteStreamSplitFloat(const std::vector<float> &values) {
    if (values.empty()) {
        return {};
    }
    std::vector<std::uint8_t> raw(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::memcpy(&raw[i * 4], &values[i], 4);
    }
    return byteStreamSplitFloat(raw.data(), values.size());
}

std::vector<float> decodeBssLz4Float(const std::vector<std::uint8_t> &blob) {
    return decode_bss_lz4_float_impl(blob);
}

std::vector<float> floatsFromPlanarByteStreamSplit(const std::vector<std::uint8_t> &bss) {
    return floats_from_planar_bss_impl(bss);
}

void benchDeltaZigZagVarint(const std::vector<std::vector<int>> &columns,
                            std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encode_delta_zigzag_varint_impl(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decode_delta_zigzag_varint_impl(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<std::int64_t> orig64(col.begin(), col.end());
            std::vector<std::int64_t> dec64(dec.begin(), dec.end());
            benchmark_loss::accumulate_int64_column(orig64, dec64, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchDictionaryInt(const std::vector<std::vector<int>> &columns,
                        std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encodeDictionaryInt(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decodeDictionaryInt(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<std::int64_t> orig64(col.begin(), col.end());
            std::vector<std::int64_t> dec64(dec.begin(), dec.end());
            benchmark_loss::accumulate_int64_column(orig64, dec64, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchDictionaryFloat(const std::vector<std::vector<float>> &columns,
                          std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encodeDictionaryFloat(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decodeDictionaryFloat(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        bool ok = dec.size() == col.size();
        for (std::size_t i = 0; ok && i < col.size(); ++i) {
            std::uint32_t a = 0, b = 0;
            std::memcpy(&a, &dec[i], sizeof(a));
            std::memcpy(&b, &col[i], sizeof(b));
            if (a != b) {
                ok = false;
            }
        }
        if (!ok) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<double> orig_d(col.begin(), col.end());
            std::vector<double> dec_d(dec.begin(), dec.end());
            benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchDictionaryInt64(const std::vector<std::vector<std::int64_t>> &columns,
                          std::vector<long long> &result, bool double_wire, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encode_dictionary_int64_impl(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decode_dictionary_int64_impl(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) {
            continue;
        }
        (void)double_wire;
        if (loss != nullptr) {
            benchmark_loss::accumulate_int64_column(col, dec, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchDictionaryDouble64(const std::vector<std::vector<double>> &columns,
                             std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encode_dictionary_double64_impl(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decode_dictionary_double64_impl(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        bool ok = dec.size() == col.size();
        for (std::size_t i = 0; ok && i < col.size(); ++i) {
            std::uint64_t a = 0, b = 0;
            std::memcpy(&a, &dec[i], sizeof(a));
            std::memcpy(&b, &col[i], sizeof(b));
            if (a != b) {
                ok = false;
            }
        }
        if (!ok) {
            continue;
        }
        if (loss != nullptr) {
            benchmark_loss::accumulate_double_column(col, dec, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchRleInt(const std::vector<std::vector<int>> &columns,
                 std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encodeRleInt(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decodeRleInt(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<std::int64_t> orig64(col.begin(), col.end());
            std::vector<std::int64_t> dec64(dec.begin(), dec.end());
            benchmark_loss::accumulate_int64_column(orig64, dec64, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchDeltaZigZagVarintInt64(const std::vector<std::vector<std::int64_t>> &columns,
                                 std::vector<long long> &result, bool double_wire, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        to += static_cast<long long>(col.size()) * 8;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encode_delta_zigzag_varint_int64_impl(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decode_delta_zigzag_varint_int64_impl(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) {
            continue;
        }
        (void)double_wire;
        if (loss != nullptr) {
            benchmark_loss::accumulate_int64_column(col, dec, *loss);
        }
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchRleInt64(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                   bool double_wire, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        to += static_cast<long long>(col.size()) * 8;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encode_rle_int64_impl(col);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decode_rle_int64_impl(enc);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) {
            continue;
        }
        (void)double_wire;
        if (loss != nullptr) {
            benchmark_loss::accumulate_int64_column(col, dec, *loss);
        }
        tc += static_cast<long long>(enc.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchBssLz4Float(const std::vector<std::vector<float>> &columns,
                      std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        std::vector<std::uint8_t> raw(col.size() * 4);
        for (std::size_t i = 0; i < col.size(); ++i) {
            std::memcpy(&raw[i * 4], &col[i], 4);
        }
        auto bss = byteStreamSplitFloat(raw.data(), col.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> comp;
        try {
            comp = encode_bss_lz4_float_impl(col);
        } catch (...) {
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        LZ4F_dctx *dctx = nullptr;
        LZ4F_errorCode_t er = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
        if (LZ4F_isError(er)) {
            continue;
        }
        std::vector<std::uint8_t> dec(bss.size());
        std::size_t dstPos = 0;
        std::size_t srcPos = 0;
        bool decomp_ok = true;
        while (srcPos < comp.size()) {
            std::size_t dstChunk = dec.size() - dstPos;
            std::size_t srcChunk = comp.size() - srcPos;
            std::size_t hint = LZ4F_decompress(dctx, dec.data() + dstPos, &dstChunk,
                                               comp.data() + srcPos, &srcChunk, nullptr);
            if (LZ4F_isError(hint)) {
                decomp_ok = false;
                break;
            }
            srcPos += srcChunk;
            dstPos += dstChunk;
            if (hint == 0) {
                break;
            }
        }
        LZ4F_freeDecompressionContext(dctx);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!decomp_ok || dstPos != bss.size()) {
            continue;
        }
        auto raw2 = inverseByteStreamSplit(dec, col.size());
        if (raw2 != raw) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<float> dec_floats(col.size());
            std::memcpy(dec_floats.data(), raw2.data(), raw2.size());
            std::vector<double> orig_d(col.begin(), col.end());
            std::vector<double> dec_d(dec_floats.begin(), dec_floats.end());
            benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(comp.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchBssLz4Double64(const std::vector<std::vector<double>> &columns,
                         std::vector<long long> &result, LossAccum *loss) {
    long long to = 0, tc = 0, te = 0, td = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        std::vector<std::uint8_t> raw(col.size() * 8);
        for (std::size_t i = 0; i < col.size(); ++i) {
            std::memcpy(&raw[i * 8], &col[i], 8);
        }
        auto bss = byteStreamSplitDouble(raw.data(), col.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> comp;
        try {
            comp = encode_bss_lz4_double_impl(col);
        } catch (...) {
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        LZ4F_dctx *dctx = nullptr;
        LZ4F_errorCode_t er = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
        if (LZ4F_isError(er)) {
            continue;
        }
        std::vector<std::uint8_t> dec(bss.size());
        std::size_t dstPos = 0;
        std::size_t srcPos = 0;
        bool decomp_ok = true;
        while (srcPos < comp.size()) {
            std::size_t dstChunk = dec.size() - dstPos;
            std::size_t srcChunk = comp.size() - srcPos;
            std::size_t hint = LZ4F_decompress(dctx, dec.data() + dstPos, &dstChunk,
                                               comp.data() + srcPos, &srcChunk, nullptr);
            if (LZ4F_isError(hint)) {
                decomp_ok = false;
                break;
            }
            srcPos += srcChunk;
            dstPos += dstChunk;
            if (hint == 0) {
                break;
            }
        }
        LZ4F_freeDecompressionContext(dctx);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!decomp_ok || dstPos != bss.size()) {
            continue;
        }
        auto raw2 = inverseByteStreamSplitDouble(dec, col.size());
        if (raw2 != raw) {
            continue;
        }
        if (loss != nullptr) {
            std::vector<double> dec_vals(col.size());
            std::memcpy(dec_vals.data(), raw2.data(), raw2.size());
            benchmark_loss::accumulate_double_column(col, dec_vals, *loss);
        }
        to += static_cast<long long>(col.size()) * 8;
        tc += static_cast<long long>(comp.size());
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        td += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    result.assign({to, tc, te, td});
}

void benchDbpFloat64(const std::vector<std::vector<double>> &columns,
                    std::vector<long long> &result,
                    const std::vector<int> *max_points, LossAccum *loss) {
    long long original = 0, compressed = 0, encode_ns = 0, decode_ns = 0;
    if (loss) *loss = {};
    for (std::size_t c = 0; c < columns.size(); ++c) {
        const int cap = max_points && c < max_points->size() ? (*max_points)[c] : 18;
        const auto &column = columns[c];
        for (std::size_t begin = 0; begin < column.size(); begin += dbp_float64::kMaxBlock) {
            const auto end = std::min(column.size(), begin + dbp_float64::kMaxBlock);
            const std::vector<double> block(column.begin() + begin, column.begin() + end);
            const auto t0 = std::chrono::steady_clock::now();
            const auto frame = dbp_float64::encodeFrame(block, cap);
            const auto t1 = std::chrono::steady_clock::now();
            const auto decoded = dbp_float64::decodeFrame(frame);
            const auto t2 = std::chrono::steady_clock::now();
            dbp_float64::verify(block, decoded);
            original += static_cast<long long>(block.size()) * 8;
            compressed += static_cast<long long>(frame.size());
            encode_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            decode_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            if (loss) loss->ref += block.size();
        }
    }
    result.assign({original, compressed, encode_ns, decode_ns});
}

} // namespace colenc
