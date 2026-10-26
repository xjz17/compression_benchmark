// Standalone, adaptive byte entropy primitives. No LZ/BWT/general codec calls.
// PEV1 wire header (little endian, 40 bytes): magic[4], mode[1], reserved[3],
// raw length[8], arithmetic payload length[8], raw FNV-1a[8], payload FNV-1a[8].
// modes 0/2 use one adaptive model; 1/3 use the previous byte (initially 0).
// All models begin with frequency 1 for each byte. At total 32768 frequencies
// are halved rounding upward. The model therefore needs no transmitted table.
// Sparse modes 2/3 instead start with one escape symbol; unseen symbols use
// escape followed by an arithmetic uniform-byte symbol, then join the model.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Byte = std::uint8_t;
using Bytes = std::vector<Byte>;
constexpr std::uint64_t kMaxBytes = 256ULL * 1024 * 1024;
constexpr std::uint32_t kHalf = 0x80000000u;
constexpr std::uint32_t kQuarter = 0x40000000u;
constexpr std::uint32_t kThreeQuarter = 0xc0000000u;
constexpr std::size_t kHeader = 40;

std::uint64_t hash(const Bytes& data) {
    std::uint64_t v = 14695981039346656037ULL;
    for (Byte c : data) { v ^= c; v *= 1099511628211ULL; }
    return v;
}
void put64(Bytes& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<Byte>(value >> (8 * i)));
}
std::uint64_t get64(const Bytes& in, std::size_t offset) {
    if (offset > in.size() || in.size() - offset < 8) throw std::runtime_error("short header");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= std::uint64_t(in[offset + i]) << (8 * i);
    return value;
}
struct BitWriter {
    Bytes data;
    int used = 0;
    void put(bool value) {
        if (used == 0) data.push_back(0);
        if (value) data.back() |= Byte(1u << (7 - used));
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
struct Model {
    std::array<std::uint32_t, 257> freq{};
    std::array<std::uint32_t, 258> tree{};
    std::uint32_t total = 256;
    unsigned seen = 256;
    bool sparse = false;
    explicit Model(bool sparse_model = false) : sparse(sparse_model) {
        if (sparse) { freq.fill(0); freq[256] = 1; seen = 0; }
        else { freq.fill(1); freq[256] = 0; }
        rebuild();
    }
    void rebuild() {
        tree.fill(0);
        total = 0;
        for (unsigned s = 0; s < 257; ++s) {
            total += freq[s];
            for (unsigned j = s + 1; j <= 257; j += j & -j) tree[j] += freq[s];
        }
    }
    std::uint32_t cumulative(unsigned symbol) const {
        std::uint32_t sum = 0;
        for (unsigned j = symbol; j; j -= j & -j) sum += tree[j];
        return sum;
    }
    unsigned lookup(std::uint32_t slot, std::uint32_t& cumulative_out) const {
        unsigned index = 0;
        std::uint32_t prefix = 0;
        for (unsigned step = 256; step; step >>= 1) {
            unsigned next = index + step;
            if (next <= 257 && prefix + tree[next] <= slot) {
                index = next;
                prefix += tree[next];
            }
        }
        if (index >= 257) throw std::runtime_error("invalid arithmetic symbol");
        cumulative_out = prefix;
        return index;
    }
    void update(unsigned symbol) {
        if (sparse && freq[symbol] == 0) ++seen;
        ++freq[symbol]; ++total;
        for (unsigned j = symbol + 1; j <= 257; j += j & -j) ++tree[j];
        if (sparse && seen == 256 && freq[256]) {
            freq[256] = 0; --total; --tree[257];
        }
        if (total >= 32768) {
            for (auto& f : freq) f = (f + 1) >> 1;
            rebuild();
        }
    }
};
Bytes encode_payload(const Bytes& raw, unsigned mode) {
    std::vector<Model> models(mode % 2 == 0 ? 1 : 256, Model(mode >= 2));
    unsigned context = 0;
    std::uint32_t low = 0, high = 0xffffffffu;
    std::uint64_t pending = 0;
    BitWriter writer;
    auto emit = [&](bool bit) {
        writer.put(bit);
        while (pending) { writer.put(!bit); --pending; }
    };
    auto encode_interval = [&](std::uint32_t cumulative, std::uint32_t frequency, std::uint32_t total) {
        const std::uint64_t range = std::uint64_t(high) - low + 1;
        // Products are < 2^47; 64-bit arithmetic cannot overflow here.
        high = low + static_cast<std::uint32_t>((range * (cumulative + frequency)) / total - 1);
        low += static_cast<std::uint32_t>((range * cumulative) / total);
        for (;;) {
            if (high < kHalf) emit(false);
            else if (low >= kHalf) { emit(true); low -= kHalf; high -= kHalf; }
            else if (low >= kQuarter && high < kThreeQuarter) {
                ++pending; low -= kQuarter; high -= kQuarter;
            } else break;
            low <<= 1; high = (high << 1) | 1u;
        }
    };
    for (unsigned symbol : raw) {
        Model& model = models[context];
        if (model.freq[symbol] == 0) {
            encode_interval(model.cumulative(256), model.freq[256], model.total);
            encode_interval(symbol, 1, 256);
        } else encode_interval(model.cumulative(symbol), model.freq[symbol], model.total);
        model.update(symbol);
        context = mode % 2 == 0 ? 0 : symbol;
    }
    ++pending;
    emit(low >= kQuarter);
    return writer.data;
}
Bytes decode_payload(const Bytes& body, unsigned mode, std::size_t raw_len) {
    if (body.empty()) throw std::runtime_error("missing arithmetic termination");
    std::vector<Model> models(mode % 2 == 0 ? 1 : 256, Model(mode >= 2));
    unsigned context = 0;
    std::uint32_t low = 0, high = 0xffffffffu, code = 0;
    BitReader reader{body};
    for (unsigned i = 0; i < 32; ++i) code = (code << 1) | reader.get();
    Bytes raw;
    raw.reserve(raw_len);
    auto decode_slot = [&](std::uint32_t total) {
        if (code < low || code > high) throw std::runtime_error("arithmetic code outside interval");
        const std::uint64_t range = std::uint64_t(high) - low + 1;
        return static_cast<std::uint32_t>(((std::uint64_t(code) - low + 1) * total - 1) / range);
    };
    auto decode_interval = [&](std::uint32_t cumulative, std::uint32_t frequency, std::uint32_t total) {
        const std::uint64_t range = std::uint64_t(high) - low + 1;
        high = low + static_cast<std::uint32_t>((range * (cumulative + frequency)) / total - 1);
        low += static_cast<std::uint32_t>((range * cumulative) / total);
        for (;;) {
            if (high < kHalf) { }
            else if (low >= kHalf) { low -= kHalf; high -= kHalf; code -= kHalf; }
            else if (low >= kQuarter && high < kThreeQuarter) {
                low -= kQuarter; high -= kQuarter; code -= kQuarter;
            } else break;
            low <<= 1; high = (high << 1) | 1u; code = (code << 1) | reader.get();
        }
    };
    for (std::size_t i = 0; i < raw_len; ++i) {
        Model& model = models[context];
        std::uint32_t cumulative = 0;
        unsigned symbol = model.lookup(decode_slot(model.total), cumulative);
        decode_interval(cumulative, model.freq[symbol], model.total);
        if (symbol == 256) {
            if (!model.sparse) throw std::runtime_error("unexpected escape symbol");
            symbol = decode_slot(256);
            decode_interval(symbol, 1, 256);
            if (model.freq[symbol]) throw std::runtime_error("noncanonical escape symbol");
        }
        raw.push_back(static_cast<Byte>(symbol));
        model.update(symbol);
        context = mode % 2 == 0 ? 0 : symbol;
    }
    return raw;
}
Bytes encode(const Bytes& raw, unsigned mode) {
    if (raw.size() > kMaxBytes) throw std::runtime_error("raw input exceeds 256 MiB limit");
    const Bytes body = encode_payload(raw, mode);
    Bytes out{'P', 'E', 'V', '1', static_cast<Byte>(mode), 0, 0, 0};
    put64(out, raw.size()); put64(out, body.size()); put64(out, hash(raw)); put64(out, hash(body));
    out.insert(out.end(), body.begin(), body.end());
    return out;
}
Bytes decode(const Bytes& encoded, unsigned mode) {
    if (encoded.size() < kHeader || encoded[0] != 'P' || encoded[1] != 'E' ||
        encoded[2] != 'V' || encoded[3] != '1') throw std::runtime_error("invalid PEV1 header");
    if (encoded[4] != mode || encoded[5] || encoded[6] || encoded[7])
        throw std::runtime_error("wrong entropy mode or reserved flags");
    const std::uint64_t length = get64(encoded, 8), body_length = get64(encoded, 16);
    if (length > kMaxBytes) throw std::runtime_error("decoded output exceeds 256 MiB limit");
    if (body_length != encoded.size() - kHeader) throw std::runtime_error("truncated or trailing payload");
    Bytes body(encoded.begin() + kHeader, encoded.end());
    if (hash(body) != get64(encoded, 32)) throw std::runtime_error("payload checksum mismatch");
    Bytes raw = decode_payload(body, mode, static_cast<std::size_t>(length));
    if (hash(raw) != get64(encoded, 24)) throw std::runtime_error("decoded checksum mismatch");
    // Arithmetic decoders read a bounded zero extension after the last bit.
    // A matching raw checksum alone therefore cannot prove canonical padding,
    // termination, or absence of a trailing body. Re-encoding is deterministic
    // and validates all of those properties, including an empty source.
    if (encode_payload(raw, mode) != body)
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
        if (argc != 2) throw std::runtime_error("usage: primitive_entropy_v1 {range,sparse}{0,1}-{encode,decode}");
        const std::string command = argv[1];
        unsigned mode;
        if (command == "range0-encode" || command == "range0-decode") mode = 0;
        else if (command == "range1-encode" || command == "range1-decode") mode = 1;
        else if (command == "sparse0-encode" || command == "sparse0-decode") mode = 2;
        else if (command == "sparse1-encode" || command == "sparse1-decode") mode = 3;
        else throw std::runtime_error("unknown command");
        const Bytes input = read_stdin();
        const Bytes output = command.ends_with("-encode") ? encode(input, mode) : decode(input, mode);
        std::cout.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
        if (!std::cout) throw std::runtime_error("stdout write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "primitive_entropy_v1: " << error.what() << '\n';
        return 2;
    }
}
