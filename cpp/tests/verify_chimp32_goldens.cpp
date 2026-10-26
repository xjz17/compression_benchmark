// Verifies `benchmark/reference/golden/chimp32/manifest.tsv` (file presence + bytes_len).

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(CHIMP32_GOLDEN_DIR)
#define CHIMP32_GOLDEN_DIR_STR CHIMP32_GOLDEN_DIR
#else
#define CHIMP32_GOLDEN_DIR_STR ""
#endif

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) {
        s.pop_back();
    }
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
        ++i;
    }
    std::size_t j = s.size();
    while (j > i && (s[j - 1] == ' ' || s[j - 1] == '\t')) {
        --j;
    }
    return s.substr(i, j - i);
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
    std::string golden_root;
    if (argc >= 2) {
        golden_root = argv[1];
    } else if (const char *env = std::getenv("CHIMP32_GOLDEN_DIR"); env && env[0]) {
        golden_root = env;
    }
    if (golden_root.empty()) {
        golden_root = CHIMP32_GOLDEN_DIR_STR;
    }
    if (golden_root.empty()) {
        std::cerr << "Usage: verify_chimp32_goldens <golden_dir>\n";
        return 2;
    }
    if (golden_root.back() == '/') {
        golden_root.pop_back();
    }

    std::ifstream mf(golden_root + "/manifest.tsv");
    if (!mf) {
        std::cerr << "missing manifest: " << golden_root << "/manifest.tsv\n";
        return 1;
    }
    std::string line;
    if (!std::getline(mf, line)) {
        std::cerr << "empty manifest\n";
        return 1;
    }
    line = trim(line);
    if (line.find("filename") == std::string::npos) {
        std::cerr << "unexpected manifest header\n";
        return 1;
    }

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
            std::cerr << "bad line: " << line << '\n';
            ok = false;
            continue;
        }
        const std::string &fn = cols[0];
        long long want_sz = parse_ll(cols[2].c_str());
        if (want_sz < 0) {
            std::cerr << "bad bytes_len: " << line << '\n';
            ok = false;
            continue;
        }
        std::string path = golden_root + '/' + fn;
        std::uintmax_t sz = file_size(path);
        if (sz == static_cast<std::uintmax_t>(-1)) {
            std::cerr << "missing: " << path << '\n';
            ok = false;
            continue;
        }
        if (static_cast<long long>(sz) != want_sz) {
            std::cerr << "size mismatch " << fn << ": got " << sz << " want " << want_sz << '\n';
            ok = false;
            continue;
        }
        ++n;
    }
    if (!ok) {
        return 1;
    }
    if (n == 0) {
        std::cerr << "no manifest rows\n";
        return 1;
    }
    std::cout << "OK: " << n << " chimp32 golden(s) match manifest sizes\n";
    return 0;
}
