#include "java_fastpfor_codec.hpp"
#include "leco_float_codec.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(LECO_FLOAT_GOLDEN_DIR)
#define LECO_FLOAT_GOLDEN_DIR_STR LECO_FLOAT_GOLDEN_DIR
#else
#define LECO_FLOAT_GOLDEN_DIR_STR ""
#endif

static bool read_file(const std::string &path, std::vector<std::uint8_t> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    out.resize(sz);
    if (sz) {
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(sz));
    }
    return true;
}

static std::vector<float> floats64() {
    std::vector<float> v(64);
    for (int i = 0; i < 64; i++) {
        v[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i) * 0.03125f;
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

static bool check_one(const std::string &root, const char *fn, const std::vector<float> &col,
                      const char *label) {
    std::vector<std::uint8_t> golden;
    if (!read_file(root + '/' + fn, golden) || golden.size() < 4 || (golden.size() - 4) % 4 != 0) {
        std::cerr << label << ": bad golden file\n";
        return false;
    }
    std::int32_t p_file = 0;
    std::memcpy(&p_file, golden.data(), 4);
    int p = leco_float::maxLosslessDecimalScale(col);
    if (p < 0 || p != p_file) {
        std::cerr << label << ": p mismatch computed=" << p << " file=" << p_file << '\n';
        return false;
    }
    std::vector<std::int32_t> data;
    if (!leco_float::quantizeScaledInts(col, p, data)) {
        std::cerr << label << ": quantize failed\n";
        return false;
    }
    try {
        auto got = java_fastpfor::compressIntBits(data.data(), data.size());
        const auto *gw = reinterpret_cast<const std::uint32_t *>(golden.data() + 4);
        std::size_t nw = (golden.size() - 4) / 4;
        if (got.size() != nw || std::memcmp(got.data(), gw, nw * sizeof(std::uint32_t)) != 0) {
            std::cerr << label << ": compressed words differ\n";
            return false;
        }
        std::vector<std::int32_t> rec(data.size());
        java_fastpfor::uncompressIntBits(got.data(), got.size(), rec.data(), rec.size());
        double pow10 = std::pow(10.0, p);
        for (std::size_t i = 0; i < col.size(); i++) {
            float r = static_cast<float>(static_cast<double>(rec[i]) / pow10);
            std::uint32_t a = 0, b = 0;
            std::memcpy(&a, &r, 4);
            std::memcpy(&b, &col[i], 4);
            if (a != b) {
                std::cerr << label << ": roundtrip float mismatch\n";
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
    const std::string root = LECO_FLOAT_GOLDEN_DIR_STR;
    if (root.empty()) {
        std::cerr << "LECO_FLOAT_GOLDEN_DIR not set\n";
        return 2;
    }
    bool ok = check_one(root, "leco_float_n64.bin", floats64(), "leco64");
    ok = check_one(root, "leco_float_n512.bin", floats512(), "leco512") && ok;
    return ok ? 0 : 1;
}
