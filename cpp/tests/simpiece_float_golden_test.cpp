#include "simpiece_float_codec.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(SIMPIECE_FLOAT_GOLDEN_DIR)
#define SIMPIECE_FLOAT_GLD SIMPIECE_FLOAT_GOLDEN_DIR
#else
#define SIMPIECE_FLOAT_GLD ""
#endif

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) {
        s.pop_back();
    }
    return s;
}

static void split_tab(const std::string &line, std::vector<std::string> &out) {
    out.clear();
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
}

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

static std::vector<float> floats64_stable() {
    std::vector<float> v(64);
    for (int i = 0; i < 64; i++) {
        v[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i);
    }
    return v;
}

static std::vector<float> floats512_stable() {
    std::vector<float> v(512);
    for (int i = 0; i < 512; i++) {
        v[static_cast<std::size_t>(i)] = 100.0f + static_cast<float>(i) * 0.5f;
    }
    return v;
}

static const std::vector<float> *column_for_len(int n) {
    static const std::vector<float> v64 = floats64_stable();
    static const std::vector<float> v512 = floats512_stable();
    if (n == 64) {
        return &v64;
    }
    if (n == 512) {
        return &v512;
    }
    return nullptr;
}

static bool check_manifest_row(const std::string &root, const std::string &fn, int values_len, double rel) {
    const std::vector<float> *col = column_for_len(values_len);
    if (!col || static_cast<int>(col->size()) != values_len) {
        std::cerr << "unknown fixture len " << values_len << '\n';
        return false;
    }
    std::vector<std::uint8_t> golden;
    if (!read_file(root + '/' + fn, golden) || golden.empty()) {
        std::cerr << "missing golden " << fn << '\n';
        return false;
    }
    try {
        auto got = simpiece_float::compressColumnRelRange(*col, rel);
        if (got.size() != golden.size() || std::memcmp(got.data(), golden.data(), golden.size()) != 0) {
            std::cerr << "memcmp " << fn << '\n';
            return false;
        }
        if (!simpiece_float::verifyDecompressTolerance(golden, *col, rel)) {
            std::cerr << "tolerance " << fn << '\n';
            return false;
        }
    } catch (const std::exception &e) {
        std::cerr << fn << ": " << e.what() << '\n';
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    std::string root = SIMPIECE_FLOAT_GLD;
    if (argc > 1 && argv[1][0] != '\0') {
        root = argv[1];
    }
    if (root.empty()) {
        std::cerr << "SIMPIECE_FLOAT_GOLDEN_DIR not set\n";
        return 2;
    }
    std::ifstream mf(root + "/manifest.tsv");
    if (!mf) {
        std::cerr << "no manifest\n";
        return 1;
    }
    std::string line;
    std::getline(mf, line);
    bool ok = true;
    while (std::getline(mf, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> c;
        split_tab(line, c);
        if (c.size() != 4) {
            ok = false;
            continue;
        }
        int vl = std::atoi(c[1].c_str());
        double rel = std::strtod(c[2].c_str(), nullptr);
        ok = check_manifest_row(root, c[0], vl, rel) && ok;
    }
    return ok ? 0 : 1;
}
