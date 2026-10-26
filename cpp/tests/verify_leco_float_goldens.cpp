// Verifies benchmark/reference/golden/leco_float/manifest.tsv

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(LECO_FLOAT_GOLDEN_DIR)
#define LECO_FLOAT_GOLDEN_DIR_STR LECO_FLOAT_GOLDEN_DIR
#else
#define LECO_FLOAT_GOLDEN_DIR_STR ""
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

static long long parse_ll(const char *s) {
    char *end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        return -1;
    }
    return v;
}

static std::uintmax_t file_size(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return static_cast<std::uintmax_t>(-1);
    }
    return static_cast<std::uintmax_t>(f.tellg());
}

int main(int argc, char **argv) {
    std::string root;
    if (argc >= 2) {
        root = argv[1];
    } else if (const char *e = std::getenv("LECO_FLOAT_GOLDEN_DIR"); e && e[0]) {
        root = e;
    }
    if (root.empty()) {
        root = LECO_FLOAT_GOLDEN_DIR_STR;
    }
    if (root.empty()) {
        std::cerr << "Usage: verify_leco_float_goldens <golden_dir>\n";
        return 2;
    }
    if (root.back() == '/') {
        root.pop_back();
    }
    std::ifstream mf(root + "/manifest.tsv");
    if (!mf) {
        return 1;
    }
    std::string line;
    std::getline(mf, line);
    int n = 0;
    bool ok = true;
    while (std::getline(mf, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> cols;
        split_tab(line, cols);
        if (cols.size() < 3) {
            ok = false;
            continue;
        }
        long long want = parse_ll(cols[2].c_str());
        std::string path = root + "/" + cols[0];
        std::uintmax_t sz = file_size(path);
        if (sz == static_cast<std::uintmax_t>(-1) || static_cast<long long>(sz) != want) {
            ok = false;
        }
        ++n;
    }
    if (!ok || n == 0) {
        return 1;
    }
    std::cout << "OK: " << n << " leco_float golden(s)\n";
    return 0;
}
