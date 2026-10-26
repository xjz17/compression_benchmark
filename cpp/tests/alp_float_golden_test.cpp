#include "alp_float_codec.hpp"

#include <cstring>
#include <span>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(ALP_FLOAT_GOLDEN_DIR)
#define ALP_FLOAT_GOLDEN_DIR_STR ALP_FLOAT_GOLDEN_DIR
#else
#define ALP_FLOAT_GOLDEN_DIR_STR ""
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

static bool check(const std::string &root, const char *fn, const std::vector<float> &expect,
                  const char *label) {
    std::vector<std::uint8_t> golden;
    if (!read_file(root + '/' + fn, golden)) {
        std::cerr << label << ": missing golden\n";
        return false;
    }
    std::vector<float> dec;
    if (!alp_float::goldenDeserializeColumn(std::span<const std::uint8_t>(golden.data(), golden.size()), dec)) {
        std::cerr << label << ": deserialize failed\n";
        return false;
    }
    if (dec.size() != expect.size()) {
        std::cerr << label << ": len " << dec.size() << " vs " << expect.size() << '\n';
        return false;
    }
    for (std::size_t i = 0; i < expect.size(); i++) {
        std::uint32_t a = 0, b = 0;
        std::memcpy(&a, &dec[i], 4);
        std::memcpy(&b, &expect[i], 4);
        if (a != b) {
            std::cerr << label << ": value mismatch @" << i << '\n';
            return false;
        }
    }
    auto regen = alp_float::goldenSerializeColumn(expect);
    if (regen != golden) {
        std::cerr << label << ": serialize != golden bytes\n";
        return false;
    }
    return true;
}

int main() {
    const std::string root = ALP_FLOAT_GOLDEN_DIR_STR;
    if (root.empty()) {
        std::cerr << "ALP_FLOAT_GOLDEN_DIR not set\n";
        return 2;
    }
    bool ok = check(root, "alp_float_n64.bin", floats64(), "alp64");
    ok = check(root, "alp_float_n512.bin", floats512(), "alp512") && ok;
    return ok ? 0 : 1;
}
