// Native Chimp32 vs Java-exported bytes (see Chimp32GoldenExport).

#include "chimp32_paper_codec.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(CHIMP32_GOLDEN_DIR)
#define CHIMP32_GOLDEN_DIR_STR CHIMP32_GOLDEN_DIR
#else
#define CHIMP32_GOLDEN_DIR_STR ""
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

static bool memcmp_bytes(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b,
                         const char *label) {
    if (a.size() != b.size()) {
        std::cerr << label << ": len " << a.size() << " vs " << b.size() << '\n';
        return false;
    }
    if (!a.empty() && std::memcmp(a.data(), b.data(), a.size()) != 0) {
        std::cerr << label << ": compressed bytes differ from Java golden\n";
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

static std::vector<float> default_floats64() {
    std::vector<float> v(64);
    for (int i = 0; i < 64; i++) {
        v[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i) * 0.03125f;
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

static std::vector<float> floats512() {
    std::vector<float> v(512);
    for (int i = 0; i < 512; i++) {
        v[static_cast<std::size_t>(i)] = 1.25f + static_cast<float>(i) * 0.0625f;
    }
    return v;
}

static bool roundtrip_int(const std::vector<std::int32_t> &col, const char *label) {
    try {
        auto w = chimp32_paper::encodeIntColumnAsFloatBits(col.data(), col.size());
        std::vector<std::int32_t> dec(col.size());
        chimp32_paper::decodeIntColumnAsFloatBits(std::span<const std::uint8_t>(w.data(), w.size()),
                                                    dec.data(), dec.size());
        if (std::memcmp(dec.data(), col.data(), col.size() * sizeof(std::int32_t)) != 0) {
            std::cerr << label << ": int roundtrip mismatch\n";
            return false;
        }
    } catch (const std::exception &e) {
        std::cerr << label << ": " << e.what() << '\n';
        return false;
    }
    return true;
}

static bool roundtrip_float(const std::vector<float> &col, const char *label) {
    try {
        auto w = chimp32_paper::encodeFloatBitsColumn(col.data(), col.size());
        std::vector<float> dec(col.size());
        chimp32_paper::decodeFloatBitsColumn(std::span<const std::uint8_t>(w.data(), w.size()), dec.data(),
                                              dec.size());
        for (std::size_t i = 0; i < col.size(); i++) {
            std::uint32_t a, b;
            std::memcpy(&a, &col[i], 4);
            std::memcpy(&b, &dec[i], 4);
            if (a != b) {
                std::cerr << label << ": float roundtrip mismatch\n";
                return false;
            }
        }
    } catch (const std::exception &e) {
        std::cerr << label << ": " << e.what() << '\n';
        return false;
    }
    return true;
}

int main() {
    const std::string root = CHIMP32_GOLDEN_DIR_STR;
    if (root.empty()) {
        std::cerr << "CHIMP32_GOLDEN_DIR not defined\n";
        return 2;
    }

    bool ok = true;

    auto check_int = [&](const char *fn, const std::vector<std::int32_t> &col, const char *label) {
        std::vector<std::uint8_t> golden;
        if (!read_whole_file(root + '/' + fn, golden)) {
            std::cerr << "missing golden " << fn << '\n';
            ok = false;
            return;
        }
        try {
            auto got = chimp32_paper::encodeIntColumnAsFloatBits(col.data(), col.size());
            ok &= memcmp_bytes(got, golden, label);
            ok &= roundtrip_int(col, label);
        } catch (const std::exception &e) {
            std::cerr << label << ": " << e.what() << '\n';
            ok = false;
        }
    };

    auto check_float = [&](const char *fn, const std::vector<float> &col, const char *label) {
        std::vector<std::uint8_t> golden;
        if (!read_whole_file(root + '/' + fn, golden)) {
            std::cerr << "missing golden " << fn << '\n';
            ok = false;
            return;
        }
        try {
            auto got = chimp32_paper::encodeFloatBitsColumn(col.data(), col.size());
            ok &= memcmp_bytes(got, golden, label);
            ok &= roundtrip_float(col, label);
        } catch (const std::exception &e) {
            std::cerr << label << ": " << e.what() << '\n';
            ok = false;
        }
    };

    check_int("chimp32_int_n64.bin", default_ints64(), "chimp32_int_n64");
    check_float("chimp32_float_n64.bin", default_floats64(), "chimp32_float_n64");
    check_int("chimp32_int_n512.bin", ints512_ramp(), "chimp32_int_n512");
    check_float("chimp32_float_n512.bin", floats512(), "chimp32_float_n512");

    return ok ? 0 : 1;
}
