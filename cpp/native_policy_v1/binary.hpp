#pragma once
#include "policy.hpp"
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace weighted_grpo::native_policy_v1::binary {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline std::vector<std::uint8_t> read(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open input file");
    const auto size = file.tellg();
    require(size >= 0 && std::uint64_t(size) <= limit, "file size exceeds limit");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!bytes.empty()) file.read(reinterpret_cast<char*>(bytes.data()), size);
    require(bool(file), "cannot read complete input file");
    return bytes;
}
struct Reader {
    std::span<const std::uint8_t> bytes;
    std::size_t offset{};
    std::size_t remaining() const { return bytes.size() - offset; }
    std::span<const std::uint8_t> take(std::size_t n) {
        require(n <= remaining(), "truncated binary field");
        auto part = bytes.subspan(offset, n); offset += n; return part;
    }
    std::uint32_t u32() {
        auto b = take(4); std::uint32_t x = 0;
        for (unsigned i = 0; i < 4; ++i) x |= std::uint32_t(b[i]) << (8*i);
        return x;
    }
    std::uint64_t u64() {
        auto b = take(8); std::uint64_t x = 0;
        for (unsigned i = 0; i < 8; ++i) x |= std::uint64_t(b[i]) << (8*i);
        return x;
    }
    double f64() {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
        const auto x = std::bit_cast<double>(u64());
        require(std::isfinite(x), "non-finite binary float64"); return x;
    }
    std::string text() {
        const auto n = u32(); require(n > 0 && n <= 1024*1024, "invalid identity string length");
        auto data = take(n);
        // Strict UTF-8 scalar values; reject overlong encodings, surrogates, NUL.
        for (std::size_t i=0; i<data.size();) {
            const auto c=data[i++];
            if (c<0x80) { require(c!=0, "NUL in identity string"); continue; }
            unsigned more=0; std::uint32_t cp=0, minimum=0;
            if (c>=0xc2 && c<=0xdf) { more=1; cp=c&31; minimum=0x80; }
            else if(c>=0xe0 && c<=0xef) { more=2; cp=c&15; minimum=0x800; }
            else if(c>=0xf0 && c<=0xf4) { more=3; cp=c&7; minimum=0x10000; }
            else require(false, "invalid UTF-8 identity");
            require(more<=data.size()-i, "truncated UTF-8 identity");
            while(more--) { auto d=data[i++]; require((d&0xc0)==0x80,"invalid UTF-8 continuation"); cp=(cp<<6)|(d&63); }
            require(cp>=minimum && cp<=0x10ffff && !(cp>=0xd800 && cp<=0xdfff), "invalid UTF-8 scalar");
        }
        return std::string(reinterpret_cast<const char*>(data.data()), data.size());
    }
    void magic(const char* wanted) {
        auto actual=take(8);
        for (std::size_t i=0;i<8;++i) require(actual[i]==std::uint8_t(wanted[i]), "binary magic/version mismatch");
    }
    void end() const { require(remaining()==0,"trailing binary bytes"); }
};
inline void u32(std::ostream& out, std::uint32_t x) {
    for(unsigned i=0;i<4;++i) out.put(static_cast<char>(x>>(8*i)));
}
inline void f64(std::ostream& out, double value) {
    auto x=std::bit_cast<std::uint64_t>(value);
    for(unsigned i=0;i<8;++i) out.put(static_cast<char>(x>>(8*i)));
}
} // namespace weighted_grpo::native_policy_v1::binary
