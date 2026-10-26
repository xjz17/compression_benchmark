#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(ELF_ON_CHIMP_FLOAT_GOLDEN_DIR)
#define ROOT_DIR ELF_ON_CHIMP_FLOAT_GOLDEN_DIR
#else
#define ROOT_DIR ""
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
    std::string root = ROOT_DIR;
    if (argc > 1) {
        root = argv[1];
    }
    if (root.empty()) {
        std::cerr << "Usage: verify_elf_on_chimp_float_goldens <golden_dir>\n";
        return 2;
    }
    std::ifstream mf(root + "/manifest.tsv");
    if (!mf) {
        std::cerr << "No manifest\n";
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
        std::vector<std::string> cells;
        split_tab(line, cells);
        if (cells.size() != 3) {
            return 1;
        }
        long long blen = parse_ll(cells[2].c_str());
        if (blen < 0) {
            return 1;
        }
        std::string path = root + '/' + cells[0];
        std::uintmax_t sz = file_size(path);
        if (sz == static_cast<std::uintmax_t>(-1) || static_cast<long long>(sz) != blen) {
            return 1;
        }
        n++;
    }
    std::cout << "OK: " << n << " elf_on_chimp_float golden(s)\n";
    return 0;
}
