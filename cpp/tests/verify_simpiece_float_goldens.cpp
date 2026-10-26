#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(SIMPIECE_FLOAT_GOLDEN_DIR)
#define R SIMPIECE_FLOAT_GOLDEN_DIR
#else
#define R ""
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

static long long pll(const char *s) {
    char *e = nullptr;
    long long v = std::strtoll(s, &e, 10);
    if (e == s || *e) {
        return -1;
    }
    return v;
}

static std::uintmax_t fsz(const std::string &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        return static_cast<std::uintmax_t>(-1);
    }
    return static_cast<std::uintmax_t>(f.tellg());
}

int main(int argc, char **argv) {
    std::string root = R;
    if (argc > 1) {
        root = argv[1];
    }
    if (root.empty()) {
        std::cerr << "Usage: verify_simpiece_float_goldens <dir>\n";
        return 2;
    }
    std::ifstream mf(root + "/manifest.tsv");
    if (!mf) {
        return 1;
    }
    std::string line;
    std::getline(mf, line);
    int n = 0;
    while (std::getline(mf, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> c;
        split_tab(line, c);
        if (c.size() != 4) {
            return 1;
        }
        long long bl = pll(c[3].c_str());
        if (bl < 0) {
            return 1;
        }
        auto z = fsz(root + '/' + c[0]);
        if (z == static_cast<std::uintmax_t>(-1) || static_cast<long long>(z) != bl) {
            return 1;
        }
        n++;
    }
    std::cout << "OK: " << n << " simpiece_float golden(s)\n";
    return 0;
}
