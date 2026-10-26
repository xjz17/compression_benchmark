// C++ CompositeCodec<FastPFor<8>, VariableByte> vs Java-exported words (see FastPFORGoldenExport).

#include "java_fastpfor_codec.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(FASTPFOR_GOLDEN_DIR)
#define FASTPFOR_GOLDEN_DIR_STR FASTPFOR_GOLDEN_DIR
#else
#define FASTPFOR_GOLDEN_DIR_STR ""
#endif

static bool read_whole_file(const std::string &path, std::vector<std::uint8_t> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    out.resize(sz);
    if (sz > 0) {
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(sz));
    }
    return true;
}

static std::vector<std::uint32_t> bytes_to_words_le(const std::vector<std::uint8_t> &b) {
    if (b.size() % 4 != 0) {
        throw std::runtime_error("golden length not multiple of 4");
    }
    std::vector<std::uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), b.size());
    return w;
}

static bool memcmp_u32(const std::vector<std::uint32_t> &a, const std::vector<std::uint32_t> &b,
                       const char *label) {
    if (a.size() != b.size()) {
        std::cerr << label << ": word len " << a.size() << " vs " << b.size() << '\n';
        return false;
    }
    if (!a.empty() && std::memcmp(a.data(), b.data(), a.size() * sizeof(std::uint32_t)) != 0) {
        std::cerr << label << ": compressed words differ from Java golden\n";
        return false;
    }
    return true;
}

static std::vector<std::int32_t> default_ints64() {
    std::vector<std::int32_t> v(64);
    for (int i = 0; i < 64; i++) {
        v[static_cast<std::size_t>(i)] = 10000 + i * 3;
    }
    return v;
}

static std::vector<std::int32_t> default_float_bits64() {
    std::vector<std::int32_t> v(64);
    for (int i = 0; i < 64; i++) {
        float f = 1.0f + static_cast<float>(i) * 0.03125f;
        std::uint32_t b;
        std::memcpy(&b, &f, sizeof(float));
        v[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(b);
    }
    return v;
}

static std::vector<std::int32_t> ints512_ramp() {
    std::vector<std::int32_t> v(512);
    for (int i = 0; i < 512; i++) {
        v[static_cast<std::size_t>(i)] = 1000 + i;
    }
    return v;
}

static std::vector<std::int32_t> float_bits512() {
    std::vector<std::int32_t> v(512);
    for (int i = 0; i < 512; i++) {
        float f = 1.25f + static_cast<float>(i) * 0.0625f;
        std::uint32_t b;
        std::memcpy(&b, &f, sizeof(float));
        v[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(b);
    }
    return v;
}

static bool roundtrip_report(const std::vector<std::int32_t> &col, const char *label) {
    try {
        auto w = java_fastpfor::compressIntBits(col.data(), col.size());
        auto dec = java_fastpfor::uncompressIntBits(w, col.size());
        if (dec.size() != col.size()) {
            std::cerr << label << ": roundtrip len\n";
            return false;
        }
        if (!col.empty() &&
            std::memcmp(dec.data(), col.data(), col.size() * sizeof(std::int32_t)) != 0) {
            std::cerr << label << ": roundtrip value mismatch\n";
            return false;
        }
    } catch (const std::exception &e) {
        std::cerr << label << ": " << e.what() << '\n';
        return false;
    }
    return true;
}

int main() {
    const std::string root = FASTPFOR_GOLDEN_DIR_STR;
    if (root.empty()) {
        std::cerr << "FASTPFOR_GOLDEN_DIR not defined\n";
        return 2;
    }

    bool ok = true;

    auto check_file = [&](const char *fn, const std::vector<std::int32_t> &col, const char *label) {
        std::vector<std::uint8_t> bytes;
        if (!read_whole_file(root + '/' + fn, bytes)) {
            std::cerr << "missing golden " << fn << '\n';
            ok = false;
            return;
        }
        try {
            auto golden_words = bytes_to_words_le(bytes);
            auto got = java_fastpfor::compressIntBits(col.data(), col.size());
            ok &= memcmp_u32(got, golden_words, label);
            ok &= roundtrip_report(col, label);
        } catch (const std::exception &e) {
            std::cerr << label << ": " << e.what() << '\n';
            ok = false;
        }
    };

    check_file("fastpfor_int_n64.bin", default_ints64(), "fastpfor_int_n64");
    check_file("fastpfor_float_n64.bin", default_float_bits64(), "fastpfor_float_n64");
    check_file("fastpfor_int_n512.bin", ints512_ramp(), "fastpfor_int_n512");
    check_file("fastpfor_float_n512.bin", float_bits512(), "fastpfor_float_n512");

    return ok ? 0 : 1;
}
