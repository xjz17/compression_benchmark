#include "buff_split_double_codec.hpp"

#include "buff_bit_pack.hpp"
#include "buff_precision_bound.hpp"
#include "../leco_double_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace buff_kernel {

namespace {

std::int64_t sign_extend_u32(std::uint32_t v, int bits) {
    if (bits <= 0) {
        return 0;
    }
    v &= (1U << bits) - 1U;
    const std::uint32_t s = v << (32 - bits);
    return static_cast<std::int64_t>(static_cast<std::int32_t>(s) >> (32 - bits));
}

std::uint64_t to_u64(std::int64_t value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::int64_t to_i64(std::uint64_t value) {
    std::int64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int bit_width(std::uint64_t value) {
    return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

std::uint32_t get_u32(const std::vector<std::uint8_t>& in, std::size_t& pos) {
    if (pos + 4 > in.size()) throw std::runtime_error("BUFF truncated uint32");
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        value |= static_cast<std::uint32_t>(in[pos++]) << shift;
    }
    return value;
}

std::uint64_t get_u64(const std::vector<std::uint8_t>& in, std::size_t& pos) {
    if (pos + 8 > in.size()) throw std::runtime_error("BUFF truncated uint64");
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        value |= static_cast<std::uint64_t>(in[pos++]) << shift;
    }
    return value;
}

std::vector<std::uint8_t> encode_fixed_values(
    const std::vector<std::int64_t>& values, bool pack) {
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("BUFF column exceeds uint32 row count");
    }
    const std::int64_t base = values.empty()
        ? 0
        : *std::min_element(values.begin(), values.end());
    std::uint64_t max_offset = 0;
    for (std::int64_t value : values) {
        max_offset = std::max(max_offset, to_u64(value) - to_u64(base));
    }
    const int width = bit_width(max_offset);

    std::vector<std::uint8_t> out;
    out.reserve(14 + values.size() * (pack ? static_cast<std::size_t>((width + 7) / 8) : 8));
    put_u32(out, static_cast<std::uint32_t>(values.size()));
    put_u64(out, to_u64(base));
    out.push_back(static_cast<std::uint8_t>(width));
    out.push_back(pack ? 1U : 0U);

    if (!pack) {
        for (std::int64_t value : values) put_u64(out, to_u64(value) - to_u64(base));
        return out;
    }
    const int byte_slices = (width + 7) / 8;
    const int padding = byte_slices * 8 - width;
    for (int slice = 0; slice < byte_slices; ++slice) {
        const int shift = (byte_slices - 1 - slice) * 8;
        for (std::int64_t value : values) {
            const std::uint64_t offset = to_u64(value) - to_u64(base);
            const std::uint64_t aligned = padding == 0 ? offset : offset << padding;
            out.push_back(static_cast<std::uint8_t>((aligned >> shift) & 0xffU));
        }
    }
    return out;
}

std::vector<std::int64_t> decode_fixed_values(
    const std::vector<std::uint8_t>& bytes, std::size_t& pos) {
    const std::uint32_t count = get_u32(bytes, pos);
    const std::uint64_t base = get_u64(bytes, pos);
    if (pos + 2 > bytes.size()) throw std::runtime_error("BUFF truncated fixed header");
    const int width = bytes[pos++];
    const bool packed = bytes[pos++] != 0;
    if (width < 0 || width > 64) throw std::runtime_error("BUFF invalid bit width");

    std::vector<std::uint64_t> offsets(count, 0);
    if (!packed) {
        for (std::uint32_t i = 0; i < count; ++i) offsets[i] = get_u64(bytes, pos);
    } else {
        const int byte_slices = (width + 7) / 8;
        const int padding = byte_slices * 8 - width;
        const std::size_t body = static_cast<std::size_t>(byte_slices) * count;
        if (pos + body > bytes.size()) throw std::runtime_error("BUFF truncated byte slices");
        for (int slice = 0; slice < byte_slices; ++slice) {
            for (std::uint32_t i = 0; i < count; ++i) {
                offsets[i] = (offsets[i] << 8) | bytes[pos++];
            }
        }
        if (padding != 0) {
            for (std::uint64_t& offset : offsets) offset >>= padding;
        }
    }
    std::vector<std::int64_t> values(count);
    for (std::uint32_t i = 0; i < count; ++i) values[i] = to_i64(base + offsets[i]);
    return values;
}

}  // namespace

int decimal_bits_for_precision(int prec) {
    switch (prec) {
        case 1:
            return 5;
        case 2:
            return 8;
        case 3:
            return 11;
        case 4:
            return 15;
        case 5:
            return 18;
        case 6:
            return 21;
        case 7:
            return 25;
        case 8:
            return 28;
        case 9:
            return 31;
        case 10:
            return 35;
        case 11:
            return 38;
        case 12:
            return 50;
        case 13:
        case 14:
        case 15:
        default:
            return 10;
    }
}

std::vector<std::uint8_t> encode_doubles(const std::vector<double>& values, int scale) {
    if (values.empty()) {
        return {};
    }
    BuffBitPackWriter bp;
    std::uint32_t t = static_cast<std::uint32_t>(values.size());
    bp.write(t, 32);
    bp.write(64, 32);
    bp.write(0, 32);
    for (double value : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        bp.write(static_cast<std::uint32_t>(bits >> 32), 32);
        bp.write(static_cast<std::uint32_t>(bits), 32);
    }
    return bp.to_bytes();
}

DecodedColumn decode_doubles(const std::vector<std::uint8_t>& bytes) {
    DecodedColumn out;
    if (bytes.empty()) {
        return out;
    }
    BuffBitPackReader rd(bytes.data(), bytes.size());
    out.t = rd.read(32);
    out.ilen = static_cast<int>(rd.read(32));
    out.dlen = static_cast<int>(rd.read(32));
    if (out.ilen < 0 || out.ilen > 64 || out.dlen < 0 || out.dlen > 64) {
        return {};
    }
    if (out.ilen == 64 && out.dlen == 0) {
        out.values.reserve(out.t);
        for (std::uint32_t i = 0; i < out.t; ++i) {
            const std::uint64_t bits = (static_cast<std::uint64_t>(rd.read(32)) << 32) |
                                       static_cast<std::uint64_t>(rd.read(32));
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            out.values.push_back(value);
        }
        return out;
    }
    out.int_parts.reserve(out.t);
    out.dec_parts.reserve(out.t);
    for (std::uint32_t i = 0; i < out.t; ++i) {
        std::uint32_t raw = rd.read(out.ilen);
        out.int_parts.push_back(sign_extend_u32(raw, out.ilen));
    }
    for (std::uint32_t i = 0; i < out.t; ++i) {
        std::uint32_t raw = rd.read(out.dlen);
        out.dec_parts.push_back(static_cast<std::uint64_t>(raw));
    }
    return out;
}

std::vector<std::uint8_t> encode_int64_ablation(
    const std::vector<std::int64_t>& values, std::string_view omitted_operator) {
    if (!omitted_operator.empty() && omitted_operator != "bit-packing") {
        throw std::invalid_argument("unsupported BUFF integer ablation");
    }
    return encode_fixed_values(values, omitted_operator != "bit-packing");
}

std::vector<std::int64_t> decode_int64_ablation(
    const std::vector<std::uint8_t>& bytes) {
    std::size_t pos = 0;
    auto values = decode_fixed_values(bytes, pos);
    if (pos != bytes.size()) throw std::runtime_error("BUFF trailing integer bytes");
    return values;
}

std::vector<std::uint8_t> encode_double_ablation(
    const std::vector<double>& values, std::string_view omitted_operator) {
    if (!omitted_operator.empty() && omitted_operator != "scaling" &&
        omitted_operator != "bit-packing") {
        throw std::invalid_argument("unsupported BUFF floating-point ablation");
    }

    bool scaled = omitted_operator != "scaling";
    int scale = 0;
    std::vector<std::int64_t> integers;
    if (scaled) {
        scale = leco_double::maxLosslessDecimalScale(values);
        scaled = scale >= 0 && leco_double::quantizeScaledInts(values, scale, integers);
        if (scaled) {
            const double factor = std::pow(10.0, scale);
            for (std::size_t i = 0; i < values.size(); ++i) {
                const double reconstructed = static_cast<double>(integers[i]) / factor;
                std::uint64_t original_bits = 0;
                std::uint64_t reconstructed_bits = 0;
                std::memcpy(&original_bits, &values[i], sizeof(original_bits));
                std::memcpy(&reconstructed_bits, &reconstructed, sizeof(reconstructed_bits));
                if (original_bits != reconstructed_bits) {
                    scaled = false;
                    break;
                }
            }
        }
    }
    if (!scaled) {
        integers.clear();
        integers.reserve(values.size());
        for (double value : values) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            integers.push_back(to_i64(bits));
        }
    }

    std::vector<std::uint8_t> out;
    out.push_back(scaled ? 1U : 0U);
    out.push_back(static_cast<std::uint8_t>(scale));
    auto body = encode_fixed_values(integers, omitted_operator != "bit-packing");
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

std::vector<double> decode_double_ablation(
    const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 2) throw std::runtime_error("BUFF truncated double header");
    const bool scaled = bytes[0] != 0;
    const int scale = bytes[1];
    std::size_t pos = 2;
    const auto integers = decode_fixed_values(bytes, pos);
    if (pos != bytes.size()) throw std::runtime_error("BUFF trailing double bytes");

    std::vector<double> values;
    values.reserve(integers.size());
    if (!scaled) {
        for (std::int64_t integer : integers) {
            const std::uint64_t bits = to_u64(integer);
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            values.push_back(value);
        }
        return values;
    }
    const double factor = std::pow(10.0, scale);
    for (std::int64_t integer : integers) {
        values.push_back(static_cast<double>(integer) / factor);
    }
    return values;
}

}  // namespace buff_kernel
