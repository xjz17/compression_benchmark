// Independent binary context entropy primitives; no hidden transform or LZ.
// PBV2 header: magic[4], previous-bit context length k[1], reserved[3],
// raw length[8], body length[8], raw FNV64[8], body FNV64[8], little endian.
// Bytes are visited MSB-first. Each of the 2^k contexts starts with [1,1].
// Context is the previous k decoded bits, initially zero; at frequency total
// 4096 each count is halved rounding upward. All state is reproducible.
// Decoding includes deterministic re-encoding to require canonical termination.
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Byte = std::uint8_t;
using Bytes = std::vector<Byte>;
constexpr std::uint64_t kMaxBytes = 256ULL * 1024 * 1024;
constexpr std::size_t kHeader = 40;
constexpr std::uint32_t kHalf = 0x80000000u;
constexpr std::uint32_t kQuarter = 0x40000000u;
constexpr std::uint32_t kThreeQuarter = 0xc0000000u;

std::uint64_t fnv64(const Bytes& data) {
    std::uint64_t value = 14695981039346656037ULL;
    for (Byte byte : data) { value ^= byte; value *= 1099511628211ULL; }
    return value;
}
void put64(Bytes& out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<Byte>(value >> (8 * i)));
}
std::uint64_t get64(const Bytes& in, std::size_t offset) {
    if (offset > in.size() || in.size() - offset < 8) throw std::runtime_error("short header");
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(in[offset + i]) << (8 * i);
    return value;
}
struct BitWriter {
    Bytes data;
    unsigned used = 0;
    void put(unsigned value) {
        if (used == 0) data.push_back(0);
        if (value) data.back() |= static_cast<Byte>(1u << (7 - used));
        used = (used + 1) & 7;
    }
};
struct BitReader {
    const Bytes& data;
    std::size_t bit = 0;
    unsigned get() {
        if (bit >= data.size() * 8 + 32) throw std::runtime_error("arithmetic padding exhausted");
        const std::size_t index = bit++;
        return index < data.size() * 8 ? (data[index / 8] >> (7 - (index & 7))) & 1u : 0;
    }
};
struct Counts {
    std::uint32_t zero = 1, one = 1;
    std::uint32_t total() const { return zero + one; }
    void update(unsigned bit) {
        if (bit) ++one; else ++zero;
        if (total() >= 4096) {
            zero = (zero + 1) >> 1;
            one = (one + 1) >> 1;
        }
    }
};

Bytes encode_body(const Bytes& raw, unsigned context_bits) {
    std::vector<Counts> models(std::size_t{1} << context_bits);
    const unsigned mask = (1u << context_bits) - 1;
    unsigned context = 0;
    std::uint32_t low = 0, high = 0xffffffffu;
    std::uint64_t pending = 0;
    BitWriter writer;
    auto emit = [&](unsigned bit) {
        writer.put(bit);
        while (pending) { writer.put(bit ^ 1u); --pending; }
    };
    for (Byte byte : raw) {
        for (int shift = 7; shift >= 0; --shift) {
            const unsigned bit = (byte >> shift) & 1u;
            Counts& model = models[context];
            const std::uint64_t range = std::uint64_t(high) - low + 1;
            // range <= 2^32 and zero <= 4095: product < 2^44.
            const std::uint32_t split = low + static_cast<std::uint32_t>(range * model.zero / model.total());
            if (bit) low = split; else high = split - 1;
            for (;;) {
                if (high < kHalf) emit(0);
                else if (low >= kHalf) { emit(1); low -= kHalf; high -= kHalf; }
                else if (low >= kQuarter && high < kThreeQuarter) {
                    ++pending; low -= kQuarter; high -= kQuarter;
                } else break;
                low <<= 1;
                high = (high << 1) | 1u;
            }
            model.update(bit);
            context = ((context << 1) | bit) & mask;
        }
    }
    ++pending;
    emit(low >= kQuarter ? 1u : 0u);
    return writer.data;
}

Bytes decode_body(const Bytes& body, unsigned context_bits, std::size_t length) {
    if (body.empty()) throw std::runtime_error("missing arithmetic termination");
    std::vector<Counts> models(std::size_t{1} << context_bits);
    const unsigned mask = (1u << context_bits) - 1;
    unsigned context = 0;
    std::uint32_t low = 0, high = 0xffffffffu, code = 0;
    BitReader reader{body};
    for (unsigned i = 0; i < 32; ++i) code = (code << 1) | reader.get();
    Bytes raw;
    raw.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        Byte byte = 0;
        for (unsigned j = 0; j < 8; ++j) {
            if (code < low || code > high) throw std::runtime_error("arithmetic code outside interval");
            Counts& model = models[context];
            const std::uint64_t range = std::uint64_t(high) - low + 1;
            const std::uint32_t split = low + static_cast<std::uint32_t>(range * model.zero / model.total());
            const unsigned bit = code >= split ? 1u : 0u;
            if (bit) low = split; else high = split - 1;
            for (;;) {
                if (high < kHalf) { }
                else if (low >= kHalf) { low -= kHalf; high -= kHalf; code -= kHalf; }
                else if (low >= kQuarter && high < kThreeQuarter) {
                    low -= kQuarter; high -= kQuarter; code -= kQuarter;
                } else break;
                low <<= 1;
                high = (high << 1) | 1u;
                code = (code << 1) | reader.get();
            }
            byte = static_cast<Byte>((byte << 1) | bit);
            model.update(bit);
            context = ((context << 1) | bit) & mask;
        }
        raw.push_back(byte);
    }
    return raw;
}
Bytes encode(const Bytes& raw, unsigned context_bits) {
    if (raw.size() > kMaxBytes) throw std::runtime_error("raw input exceeds 256 MiB limit");
    const Bytes body = encode_body(raw, context_bits);
    Bytes out{'P', 'B', 'V', '2', static_cast<Byte>(context_bits), 0, 0, 0};
    put64(out, raw.size()); put64(out, body.size()); put64(out, fnv64(raw)); put64(out, fnv64(body));
    out.insert(out.end(), body.begin(), body.end());
    return out;
}
Bytes decode(const Bytes& encoded, unsigned context_bits) {
    if (encoded.size() < kHeader || encoded[0] != 'P' || encoded[1] != 'B' ||
        encoded[2] != 'V' || encoded[3] != '2') throw std::runtime_error("invalid PBV2 header");
    if (encoded[4] != context_bits || encoded[5] || encoded[6] || encoded[7])
        throw std::runtime_error("wrong binary context or reserved flags");
    const std::uint64_t length = get64(encoded, 8), body_length = get64(encoded, 16);
    if (length > kMaxBytes) throw std::runtime_error("decoded output exceeds 256 MiB limit");
    if (body_length != encoded.size() - kHeader) throw std::runtime_error("truncated or trailing payload");
    const Bytes body(encoded.begin() + kHeader, encoded.end());
    if (fnv64(body) != get64(encoded, 32)) throw std::runtime_error("payload checksum mismatch");
    Bytes raw = decode_body(body, context_bits, static_cast<std::size_t>(length));
    if (fnv64(raw) != get64(encoded, 24)) throw std::runtime_error("decoded checksum mismatch");
    // Raw equality/checksum does not validate unused arithmetic termination
    // bits or zero extensions. Compare the complete deterministic encoding.
    if (encode_body(raw, context_bits) != body)
        throw std::runtime_error("noncanonical arithmetic termination or trailing body");
    return raw;
}
Bytes read_stdin() {
    Bytes data;
    std::array<char, 65536> buffer{};
    while (std::cin) {
        std::cin.read(buffer.data(), buffer.size());
        const auto count = std::cin.gcount();
        if (data.size() + static_cast<std::size_t>(count) > kMaxBytes * 2 + kHeader)
            throw std::runtime_error("input exceeds bounded encoded limit");
        data.insert(data.end(), buffer.data(), buffer.data() + count);
    }
    if (!std::cin.eof()) throw std::runtime_error("stdin read failed");
    return data;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: primitive_binary_entropy_v2 bit{4,8,12,16}-{encode,decode}");
        const std::string command = argv[1];
        unsigned context_bits = 0;
        for (unsigned candidate : {4u, 8u, 12u, 16u}) {
            const std::string prefix = "bit" + std::to_string(candidate);
            if (command == prefix + "-encode" || command == prefix + "-decode") context_bits = candidate;
        }
        if (!context_bits) throw std::runtime_error("unknown command");
        const Bytes input = read_stdin();
        const Bytes output = command.ends_with("-encode") ? encode(input, context_bits) : decode(input, context_bits);
        std::cout.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
        if (!std::cout) throw std::runtime_error("stdout write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "primitive_binary_entropy_v2: " << error.what() << '\n';
        return 2;
    }
}
